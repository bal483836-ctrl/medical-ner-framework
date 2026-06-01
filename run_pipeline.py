"""
医疗 NER 主流程 v4
集成：预学习 skills + Step1 抽取 + Step1.5 嵌套扩展 + Step2 KG 对齐
     + Step2.5 反思校验 + Step2.7 KG 相似度过滤(≥0.80) + Step3 幻觉过滤 + Step4 评估

运行：
  python run_pipeline.py                        # 全量
  python run_pipeline.py --dataset cmeee --split dev
  python run_pipeline.py --preanalysis-only     # 只跑预学习
  python run_pipeline.py --no-reflect           # 跳过反思
  python run_pipeline.py --no-kgfilter          # 跳过 KG 相似度过滤
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import (
    OUTPUT_DIR, DATASET_SPLITS,
    STEP1_PREFIX, STEP1E_PREFIX, STEP2_PREFIX, STEP3_PREFIX,
    HIGH_SIM_THRESHOLD,
)
from src.extract_entities import (
    build_global_few_shot,
    extract_cmeee_split, extract_imcs_split, extract_yidu,
)
from src.cmeee_expand import enrich_cmeee_step1
from src.kg_alignment import align_cmeee_split, align_imcs_split
from src.filter_hallucinations import filter_cmeee, filter_imcs_with_llm
from src.normalize_and_evaluate import (
    evaluate_cmeee, evaluate_imcs, generate_full_report,
)
from src.data_processor import (
    load_cmeee, build_cmeee_entity_vocab, build_imcs_norm_vocab,
    clean_entity_list,
)
from src.preanalysis import run_preanalysis, render_skills_block
from src.reflector import reflect_cmeee, reflect_imcs, reflect_yidu
from src.kg import load_kg
from src.normalize_imcs import normalize_list as imcs_normalize_list
from src.utils import set_global_seed, print_gpu_banner, stage_banner, kv_print


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--quick-test", action="store_true")
    p.add_argument("--dataset", choices=["cmeee", "imcs", "yidu", "all"], default="all")
    p.add_argument("--split", choices=["train", "dev", "test", "all"], default="all")
    p.add_argument("--step", type=int, choices=[1, 2, 3, 4], default=None)
    p.add_argument("--no-step3", action="store_true")
    p.add_argument("--no-reflect", action="store_true")
    p.add_argument("--no-kgfilter", action="store_true")
    p.add_argument("--cmeee-long-min", type=int, default=5)
    p.add_argument("--preanalysis-only", action="store_true")
    return p.parse_args()


def get_path(prefix, ds, split):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, f"{prefix}{ds}_{split}.json")


def load_existing(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def apply_kg_filter(items, kg, field_in, field_out, marker_key="_kg_filtered"):
    """对 items 的 field_in 字符串实体列表做 KG 相似度过滤，输出到 field_out。
    通过 marker_key 跳过已处理项实现断点续传。"""
    from tqdm import tqdm
    pending = [it for it in items if not it.get(marker_key)]
    if not pending:
        print(f"  [Resume] apply_kg_filter 全部已处理，跳过")
        return items
    if len(pending) < len(items):
        print(f"  [Resume] apply_kg_filter 续跑 {len(pending)}/{len(items)}")
    for it in tqdm(pending, desc="KG filter", unit="item"):
        ents = clean_entity_list(it.get(field_in, ""))
        if not ents:
            it[field_out] = ""
            it[marker_key] = True
            continue
        kept = kg.filter_by_similarity(
            ents, threshold=HIGH_SIM_THRESHOLD, skip_normalized=True,
        )
        it[field_out] = ",".join(kept)
        it[marker_key] = True
    return items


def _is_complete(items, key):
    """所有 item 都已写过 key 且非 None → 该阶段已完整。"""
    return bool(items) and all(key in it and it.get(key) is not None for it in items)


def run_cmeee(args, few_shot_str, cmeee_vocab, kg):
    splits_cfg = DATASET_SPLITS["CMeEE_V2"]
    targets = [s for s in splits_cfg if args.split in ("all", s["split"])]
    limit = 20 if args.quick_test else None
    eval_results = []

    for sc in targets:
        split, has_label = sc["split"], sc["has_label"]
        print(f"\n{'='*60}\n  🔵 CMeEE [{split}] has_label={has_label}\n{'='*60}")
        p1  = get_path(STEP1_PREFIX,  "CMeEE_V2", split)
        p1e = get_path(STEP1E_PREFIX, "CMeEE_V2", split)
        p2  = get_path(STEP2_PREFIX,  "CMeEE_V2", split)
        p3  = get_path(STEP3_PREFIX,  "CMeEE_V2", split)

        if args.step is None or args.step == 1:
            items = extract_cmeee_split(split, few_shot_str, p1, limit=limit)
        else:
            items = load_existing(p1)
            if items is None:
                print(f"  ⚠️ 缺 Step1 输出，跳过"); continue

        if args.step is None or args.step == 1:
            cached = load_existing(p1e)
            if cached and _is_complete(cached, "step1_enriched_output"):
                print(f"  [Resume] Step1.5 已完成 → {p1e}")
                items = cached
            else:
                print(f"[Step1.5] 嵌套扩展…")
                items = enrich_cmeee_step1(items, cmeee_vocab, long_min_len=args.cmeee_long_min)
                save_json(items, p1e)
        else:
            tmp = load_existing(p1e)
            if tmp: items = tmp

        # 反思（可选）
        if not args.no_reflect and (args.step is None or args.step == 2):
            print("[Step1.7] DeepSeek 反思校验…")
            items = reflect_cmeee(items, output_path=p1e)
            # 反思结果反哺到 enriched_output（保持后续兼容）
            for it in items:
                if it.get("reflected_output"):
                    it["step1_enriched_output"] = it["reflected_output"]
            save_json(items, p1e)

        if args.step is None or args.step == 2:
            cached = load_existing(p2)
            if cached and _is_complete(cached, "step2_aligned_output"):
                print(f"  [Resume] Step2 已完成 → {p2}")
                items = cached
            else:
                items = align_cmeee_split(items, cmeee_vocab, p2)
        else:
            tmp = load_existing(p2)
            if tmp: items = tmp

        # KG 相似度过滤
        if not args.no_kgfilter and (args.step is None or args.step in (2, 3)):
            print(f"[Step2.5] KG 余弦过滤（≥{HIGH_SIM_THRESHOLD}）…")
            items = apply_kg_filter(items, kg,
                                    field_in="step2_aligned_output",
                                    field_out="step2_aligned_output")
            save_json(items, p2)

        if not args.no_step3 and (args.step is None or args.step == 3):
            items = filter_cmeee(items, p3)
        elif args.no_step3:
            for it in items:
                it.setdefault("step3_final_output", it.get("step2_aligned_output", ""))
            save_json(items, p3)

        if has_label and (args.step is None or args.step == 4):
            eval_results.append(evaluate_cmeee(items, split))
        elif not has_label:
            print(f"  ℹ️ {split} 无标注，跳过评估")
    return eval_results


def run_imcs(args, few_shot_str, norm_vocab, kg):
    splits_cfg = DATASET_SPLITS["IMCS_V2"]
    targets = [s for s in splits_cfg if args.split in ("all", s["split"])]
    limit = 20 if args.quick_test else None
    eval_results = []

    for sc in targets:
        split, has_label = sc["split"], sc["has_label"]
        print(f"\n{'='*60}\n  🟢 IMCS [{split}] has_label={has_label}\n{'='*60}")
        p1 = get_path(STEP1_PREFIX, "IMCS_V2", split)
        p2 = get_path(STEP2_PREFIX, "IMCS_V2", split)
        p3 = get_path(STEP3_PREFIX, "IMCS_V2", split)

        if args.step is None or args.step == 1:
            items = extract_imcs_split(split, few_shot_str, p1, limit=limit)
        else:
            items = load_existing(p1)
            if items is None:
                print(f"  ⚠️ 缺 Step1 输出"); continue

        if not args.no_reflect and (args.step is None or args.step == 2):
            print("[Step1.7] DeepSeek 反思校验…")
            items = reflect_imcs(items, output_path=p1)
            for it in items:
                if it.get("reflected_output"):
                    it["step1_raw_output"] = it["reflected_output"]
            save_json(items, p1)

        if args.step is None or args.step == 2:
            cached = load_existing(p2)
            if cached and _is_complete(cached, "step2_aligned_output") \
                    and all("step2_normalized_map" in it for it in cached):
                print(f"  [Resume] IMCS Step2 + Step2.3 已完成 → {p2}")
                items = cached
            else:
                items = align_imcs_split(items, norm_vocab, p2)
                # 6 级级联兜底：对仍未对齐的原词再走一次（提升归一化召回）
                print("[Step2.3] 6 级归一化级联兜底…")
                nv_set = set(norm_vocab or [])
                for it in items:
                    norm_map = it.get("step2_normalized_map", {}) or {}
                    # 只对还映射到自身（未归一化）的实体走级联
                    unaligned = [e for e, n in norm_map.items() if e == n]
                    if not unaligned:
                        continue
                    res = imcs_normalize_list(unaligned, nv_set)
                    for e, n in res["norm_map"].items():
                        if n and n != e:
                            norm_map[e] = n
                    it["step2_normalized_map"] = norm_map
                    norm_set = list(dict.fromkeys(norm_map.values()))
                    it["step2_norm_output"] = ",".join(norm_set)
                save_json(items, p2)
        else:
            tmp = load_existing(p2)
            if tmp: items = tmp

        # KG 过滤：IMCS 标准词由 kg 模块自动跳过
        if not args.no_kgfilter and (args.step is None or args.step in (2, 3)):
            print(f"[Step2.5] KG 余弦过滤（IMCS 标准词跳过）…")
            items = apply_kg_filter(items, kg,
                                    field_in="step2_aligned_output",
                                    field_out="step2_aligned_output")
            save_json(items, p2)

        if not args.no_step3 and (args.step is None or args.step == 3):
            items = filter_imcs_with_llm(items, p3)
        elif args.no_step3:
            for it in items:
                it.setdefault("step3_final_output", it.get("step2_aligned_output", ""))
            save_json(items, p3)

        if has_label and (args.step is None or args.step == 4):
            eval_results.append(evaluate_imcs(items, split, norm_vocab))
        elif not has_label:
            print(f"  ℹ️ {split} 无标注，跳过评估")
    return eval_results


def run_yidu(args, few_shot_str, kg):
    print(f"\n{'='*60}\n  🟡 yidu_4k [train]\n{'='*60}")
    p1 = get_path(STEP1_PREFIX, "yidu_4k", "train")
    limit = 20 if args.quick_test else None
    if args.step is None or args.step == 1:
        items = extract_yidu(few_shot_str, p1, limit=limit)
    else:
        items = load_existing(p1) or []
    if items and not args.no_reflect:
        print("[Step1.7] DeepSeek 反思校验…")
        items = reflect_yidu(items, output_path=p1)
        for it in items:
            if it.get("reflected_output"):
                it["step1_raw_output"] = it["reflected_output"]
        save_json(items, p1)
    if items and not args.no_kgfilter:
        print(f"[Step2.5] KG 余弦过滤（≥{HIGH_SIM_THRESHOLD}）…")
        items = apply_kg_filter(items, kg,
                                field_in="step1_raw_output",
                                field_out="step1_raw_output")
        save_json(items, p1)


def main():
    args = parse_args()
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*70}\n  🏥 医疗 NER 框架 v4.2  (RTX 5090 优化)\n{'='*70}")
    set_global_seed()
    print_gpu_banner()

    # ----- 阶段 2 前置: 预学习 100 条 -----
    print("\n[Phase] 预学习 (per dataset, 100 samples)…")
    pre = run_preanalysis()
    for ds, rep in pre.items():
        if "error" not in rep:
            print(f"  {ds} skills: {rep.get('skills')}")
    if args.preanalysis_only:
        print("✅ preanalysis-only 已完成"); return

    # ----- 少样本（含 skills 注入）-----
    cmeee_fs, imcs_fs = build_global_few_shot()
    if pre.get("CMeEE_V2", {}).get("skills"):
        cmeee_fs = render_skills_block(pre["CMeEE_V2"]) + "\n\n" + cmeee_fs
    if pre.get("IMCS_V2", {}).get("skills"):
        imcs_fs = render_skills_block(pre["IMCS_V2"]) + "\n\n" + imcs_fs

    # ----- 全局资源 -----
    try:
        cmeee_vocab = build_cmeee_entity_vocab(load_cmeee("train"))
    except Exception as e:
        print(f"  ⚠️ CMeEE 词表失败: {e}"); cmeee_vocab = []
    norm_vocab = build_imcs_norm_vocab()
    kg = load_kg()

    # ----- 分发 -----
    all_eval = []
    if args.dataset in ("cmeee", "all"):
        all_eval.extend(run_cmeee(args, cmeee_fs, cmeee_vocab, kg))
    if args.dataset in ("imcs", "all"):
        all_eval.extend(run_imcs(args, imcs_fs, norm_vocab, kg))
    if args.dataset in ("yidu", "all"):
        run_yidu(args, cmeee_fs, kg)

    if all_eval:
        rep_md   = os.path.join(OUTPUT_DIR, "evaluation_report.md")
        rep_json = os.path.join(OUTPUT_DIR, "evaluation_report.json")
        generate_full_report(all_eval, rep_md)
        with open(rep_json, "w", encoding="utf-8") as f:
            json.dump(all_eval, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 全部完成，耗时 {(time.time()-t0)/60:.1f} 分钟")
    print(f"📁 输出: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

"""
100 条冒烟测试（v4.2）

目的：在真实数据 + 真实模型上，用每数据集 100 条样本快速跑通全部 9 个阶段，
      检查每一步的中间输出是否合理。固定全局种子保证可复现。

跑法：
    # 全量（CMeEE + IMCS + yidu，每个 100 条）
    python run_smoke_test.py

    # 只测某个数据集
    python run_smoke_test.py --dataset cmeee --n 100

    # 跳过断言阶段（只测 NER 1-4）
    python run_smoke_test.py --ner-only

    # 跳过分类器训练（只到阶段 7）
    python run_smoke_test.py --no-train

预期耗时（RTX 5090 + Flash Attention）：
    100 条 CMeEE  ≈ 5 分钟
    100 条 IMCS   ≈ 10 分钟（对话多轮）
    100 条 yidu   ≈ 6 分钟
    断言 5-9      ≈ 15 分钟
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import (
    OUTPUT_DIR, DATASET_SPLITS, ASSERTION_LABELS,
    STEP1_PREFIX, STEP1E_PREFIX, STEP2_PREFIX, STEP3_PREFIX, ASSERT_PREFIX,
)
from src.utils import (
    set_global_seed, print_gpu_banner, stage_banner, kv_print, preview_items,
)


N_DEFAULT = 100


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(items, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


# ==================== NER 阶段（限制 N 条）====================

def run_ner_smoke(dataset_tag: str, n: int):
    """跑 NER 的 1-4 阶段，限制每个 split 前 N 条。"""
    from src.extract_entities import (
        build_global_few_shot,
        extract_cmeee_split, extract_imcs_split, extract_yidu,
    )
    from src.cmeee_expand import enrich_cmeee_step1
    from src.kg_alignment import align_cmeee_split, align_imcs_split
    from src.filter_hallucinations import filter_cmeee, filter_imcs_with_llm
    from src.normalize_and_evaluate import evaluate_cmeee, evaluate_imcs
    from src.data_processor import (
        load_cmeee, build_cmeee_entity_vocab, build_imcs_norm_vocab,
        clean_entity_list,
    )
    from src.preanalysis import run_preanalysis, render_skills_block
    from src.reflector import reflect_cmeee, reflect_imcs, reflect_yidu
    from src.kg import load_kg
    from src.normalize_imcs import normalize_list as imcs_normalize_list

    # ----- 阶段 1+2: 预学习 -----
    stage_banner("阶段 1+2", "数据加载 + 预学习（LLM skills）")
    pre = run_preanalysis(use_llm=True)
    for ds, rep in pre.items():
        if "error" not in rep:
            print(f"  • {ds}: 实体平均长度={rep.get('entity_length_avg')} "
                  f"嵌套对数={rep.get('nested_pair_docs')}")
            skills = rep.get("skills", "")
            if skills:
                print(f"  • skills 前 200 字: {skills[:200]}...")

    cmeee_fs, imcs_fs = build_global_few_shot()
    if pre.get("CMeEE_V2", {}).get("skills"):
        cmeee_fs = render_skills_block(pre["CMeEE_V2"]) + "\n\n" + cmeee_fs
    if pre.get("IMCS_V2", {}).get("skills"):
        imcs_fs = render_skills_block(pre["IMCS_V2"]) + "\n\n" + imcs_fs

    # ----- 全局资源 -----
    cmeee_vocab = build_cmeee_entity_vocab(load_cmeee("train")) if dataset_tag in ("cmeee", "all") else []
    norm_vocab  = build_imcs_norm_vocab()
    kg = load_kg()

    eval_results = []

    # ============================ CMeEE ============================
    if dataset_tag in ("cmeee", "all"):
        for split_cfg in DATASET_SPLITS["CMeEE_V2"]:
            if split_cfg["split"] not in ("train", "dev"):
                continue
            split = split_cfg["split"]
            has_label = split_cfg["has_label"]

            stage_banner(f"阶段 3 [CMeEE/{split}]", f"前 {n} 条")
            p1  = os.path.join(OUTPUT_DIR, f"smoke_{STEP1_PREFIX}CMeEE_V2_{split}.json")
            p1e = os.path.join(OUTPUT_DIR, f"smoke_{STEP1E_PREFIX}CMeEE_V2_{split}.json")
            p2  = os.path.join(OUTPUT_DIR, f"smoke_{STEP2_PREFIX}CMeEE_V2_{split}.json")
            p3  = os.path.join(OUTPUT_DIR, f"smoke_{STEP3_PREFIX}CMeEE_V2_{split}.json")

            print("  ▶ Step1: 大模型抽取")
            items = extract_cmeee_split(split, cmeee_fs, p1, limit=n)
            preview_items(items, n=2, fields=("text", "step1_raw_output", "gold_entities_str"))

            print("\n  ▶ Step1.5: 嵌套扩展")
            items = enrich_cmeee_step1(items, cmeee_vocab, long_min_len=5)
            _save(items, p1e)
            preview_items(items, n=2, fields=("step1_enriched_output", "gold_entities_str"))

            print("\n  ▶ Step1.7: DeepSeek 反思")
            items = reflect_cmeee(items)
            for it in items:
                if it.get("reflected_output"):
                    it["step1_enriched_output"] = it["reflected_output"]
            _save(items, p1e)

            print("\n  ▶ Step2: KG 对齐")
            items = align_cmeee_split(items, cmeee_vocab, p2)

            print("\n  ▶ Step2.5: KG 余弦过滤 ≥ 0.80")
            for it in items:
                ents = clean_entity_list(it.get("step2_aligned_output", ""))
                kept = kg.filter_by_similarity(ents, threshold=0.80)
                it["step2_aligned_output"] = ",".join(kept)
            _save(items, p2)
            preview_items(items, n=2, fields=("step2_aligned_output",))

            print("\n  ▶ Step3: 幻觉过滤")
            items = filter_cmeee(items, p3)
            preview_items(items, n=2, fields=("step3_final_output", "gold_entities_str"))

            if has_label:
                stage_banner(f"阶段 4 [CMeEE/{split}]", "F1 评估")
                result = evaluate_cmeee(items, split)
                kv_print({k: v for k, v in result.items() if not isinstance(v, (list, dict))})
                eval_results.append(result)

    # ============================ IMCS ============================
    if dataset_tag in ("imcs", "all"):
        for split_cfg in DATASET_SPLITS["IMCS_V2"]:
            if split_cfg["split"] not in ("train", "dev"):
                continue
            split = split_cfg["split"]
            has_label = split_cfg["has_label"]

            stage_banner(f"阶段 3 [IMCS/{split}]", f"前 {n} 条对话")
            p1 = os.path.join(OUTPUT_DIR, f"smoke_{STEP1_PREFIX}IMCS_V2_{split}.json")
            p2 = os.path.join(OUTPUT_DIR, f"smoke_{STEP2_PREFIX}IMCS_V2_{split}.json")
            p3 = os.path.join(OUTPUT_DIR, f"smoke_{STEP3_PREFIX}IMCS_V2_{split}.json")

            print("  ▶ Step1: 对话按角色逐轮抽取")
            items = extract_imcs_split(split, imcs_fs, p1, limit=n)
            preview_items(items, n=2, fields=("dialogue_id", "step1_raw_output", "gold_entities_str"))

            print("\n  ▶ Step1.7: DeepSeek 反思")
            items = reflect_imcs(items)
            for it in items:
                if it.get("reflected_output"):
                    it["step1_raw_output"] = it["reflected_output"]
            _save(items, p1)

            print("\n  ▶ Step2: IMCS 归一化对齐")
            items = align_imcs_split(items, norm_vocab, p2)

            print("\n  ▶ Step2.3: 6 级归一化级联兜底")
            nv_set = set(norm_vocab or [])
            n_added = 0
            for it in items:
                nm = it.get("step2_normalized_map", {}) or {}
                unaligned = [e for e, n2 in nm.items() if e == n2]
                if not unaligned: continue
                res = imcs_normalize_list(unaligned, nv_set)
                for e, n2 in res["norm_map"].items():
                    if n2 and n2 != e:
                        nm[e] = n2; n_added += 1
                it["step2_normalized_map"] = nm
                it["step2_norm_output"] = ",".join(dict.fromkeys(nm.values()))
            print(f"    级联补救成功 {n_added} 个原词")

            print("\n  ▶ Step2.5: KG 过滤（IMCS 标准词跳过）")
            for it in items:
                ents = clean_entity_list(it.get("step2_aligned_output", ""))
                kept = kg.filter_by_similarity(ents, threshold=0.80, skip_normalized=True)
                it["step2_aligned_output"] = ",".join(kept)
            _save(items, p2)

            print("\n  ▶ Step3: LLM 幻觉过滤")
            items = filter_imcs_with_llm(items, p3)
            preview_items(items, n=2, fields=("step3_final_output", "step2_norm_output", "gold_entities_str"))

            if has_label:
                stage_banner(f"阶段 4 [IMCS/{split}]", "双 F1 评估（字面 + 归一化）")
                result = evaluate_imcs(items, split, norm_vocab)
                kv_print({k: v for k, v in result.items() if not isinstance(v, (list, dict))})
                eval_results.append(result)

    # ============================ yidu ============================
    if dataset_tag in ("yidu", "all"):
        stage_banner("阶段 3 [yidu/train]", f"前 {n} 条")
        p1 = os.path.join(OUTPUT_DIR, f"smoke_{STEP1_PREFIX}yidu_4k_train.json")
        items = extract_yidu(cmeee_fs, p1, limit=n)
        preview_items(items, n=2, fields=("text", "step1_raw_output", "gold_entities_str"))
        if items:
            print("\n  ▶ KG 过滤")
            for it in items:
                ents = clean_entity_list(it.get("step1_raw_output", ""))
                kept = kg.filter_by_similarity(ents, threshold=0.80)
                it["step1_raw_output"] = ",".join(kept)
            _save(items, p1)

    return eval_results


# ==================== 断言阶段（限制样本）====================

def run_assertion_smoke(dataset_tag: str, n: int,
                         do_train: bool = True):
    """跑断言 5-9 阶段。"""
    from src.kg import load_kg
    from src.context_window import context_from_text, context_from_dialogue
    from src.assertion_annotator import annotate, save as save_anno
    from src.augmentor import augment, label_distribution
    from src.assertion_train import train_multi_seed
    from src.assertion_eval import evaluate
    from src.data_processor import clean_entity_list
    from config.config import CLF_ENSEMBLE_SEEDS

    if dataset_tag == "all":
        # 优先 CMeEE
        dataset_tag = "cmeee"
    ds = "CMeEE_V2" if dataset_tag == "cmeee" else "IMCS_V2"

    kg = load_kg()

    # ----- 阶段 5 + 6: 构建 (实体, 语境, 扩展) 样本 -----
    stage_banner(f"阶段 5+6 [{ds}]", "KG 扩展 + 动态窗口 + 取样本")
    samples_by_split = {}
    for split in ("train", "dev", "test"):
        p3 = os.path.join(OUTPUT_DIR, f"smoke_{STEP3_PREFIX}{ds}_{split}.json")
        items = _load_json(p3)
        if items is None:
            print(f"  ⚠️ 缺 {p3}（先跑 NER 部分）")
            samples_by_split[split] = []
            continue
        out = []
        for it in items[:n]:
            if ds == "CMeEE_V2":
                text = it.get("text", "")
                ents = clean_entity_list(it.get("step3_final_output", ""))
                for ent in ents:
                    exp = kg.expand(ent, topk=5)
                    for c in context_from_text(text, ent):
                        out.append({"doc_id": it.get("id"), "dataset": ds, "split": split,
                                    "entity": ent, "context": c["context"],
                                    "expansion": exp, "source": "text", "label": ""})
            else:
                dlg = it.get("dialogue", []); sr = it.get("self_report", "")
                ents = clean_entity_list(it.get("step3_final_output", ""))
                for ent in ents:
                    exp = kg.expand(ent, topk=5)
                    for c in context_from_dialogue(dlg, ent, self_report=sr):
                        out.append({"dialogue_id": it.get("dialogue_id"), "dataset": ds, "split": split,
                                    "entity": ent, "context": c["context"],
                                    "expansion": exp, "source": "dialogue", "label": ""})
        print(f"  {ds}/{split}: {len(out)} 个 (实体,语境) 样本")
        samples_by_split[split] = out

    # ----- 阶段 6: LLM 断言标注（自洽投票）-----
    stage_banner(f"阶段 6 [{ds}]", "LLM 自洽投票断言标注（vote=3）")
    for split in ("train", "dev", "test"):
        if not samples_by_split[split]: continue
        samples_by_split[split] = annotate(samples_by_split[split], vote_passes=3)
        path = os.path.join(OUTPUT_DIR, f"smoke_{ASSERT_PREFIX}{ds}_{split}.json")
        save_anno(samples_by_split[split], ds, split,
                  out_dir=OUTPUT_DIR)
        cnt = Counter(s.get("label") for s in samples_by_split[split])
        print(f"  {split} 标签分布: {dict(cnt)}")

    # ----- 阶段 7: 分布检测 + 增强 -----
    stage_banner(f"阶段 7 [{ds}]", "分布检测 + 按类目标补足增强")
    merged = samples_by_split["train"] + samples_by_split["dev"]
    print(f"  增强前: {label_distribution(merged)}")
    aug = augment(merged)
    print(f"  增强后: {label_distribution(aug)}")
    n_train = len(samples_by_split["train"])
    samples_by_split["train"] = aug[:n_train] + aug[len(merged):]

    if not do_train:
        print("\n✅ --no-train 已设，跳过阶段 8/9")
        return

    # ----- 阶段 8: 多 seed 训练 -----
    stage_banner(f"阶段 8 [{ds}]", f"RoBERTa 多 seed 训练 (seeds={CLF_ENSEMBLE_SEEDS})")
    model_dirs = train_multi_seed(
        train_samples=samples_by_split["train"],
        dev_samples=samples_by_split["dev"],
        seeds=CLF_ENSEMBLE_SEEDS,
        base_dir=os.path.join(OUTPUT_DIR, "smoke_assertion_clf"),
    )

    # ----- 阶段 9: 评估 -----
    labeled_test = [s for s in samples_by_split["test"] if s.get("label")]
    if labeled_test:
        stage_banner(f"阶段 9 [{ds}]", "test 评估（集成 + dev 阈值搜索）")
        evaluate(model_dirs, labeled_test, val_samples=samples_by_split["dev"])
    else:
        print("\n⚠️ test 集无标注，跳过 macro F1 评估")


# ==================== 主流程 ====================

def main():
    ap = argparse.ArgumentParser(description="100 条冒烟测试")
    ap.add_argument("--dataset", choices=["cmeee", "imcs", "yidu", "all"], default="all")
    ap.add_argument("--n", type=int, default=N_DEFAULT, help="每 split 截取条数")
    ap.add_argument("--ner-only", action="store_true", help="只跑 NER 1-4")
    ap.add_argument("--assertion-only", action="store_true", help="只跑断言 5-9（需已有 step3 输出）")
    ap.add_argument("--no-train", action="store_true", help="跳过阶段 8/9")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "█" * 70)
    print(f"█  🧪 100 条冒烟测试  v4.2".ljust(69) + "█")
    print(f"█  dataset={args.dataset}  n={args.n}  ner_only={args.ner_only}".ljust(69) + "█")
    print(f"█  no_train={args.no_train}  assertion_only={args.assertion_only}".ljust(69) + "█")
    print("█" * 70)

    set_global_seed(args.seed)
    print_gpu_banner()

    # ----- NER 阶段 -----
    if not args.assertion_only:
        eval_results = run_ner_smoke(args.dataset, args.n)
        if eval_results:
            stage_banner("NER 汇总", "")
            for r in eval_results:
                print(f"  {r}")

    # ----- 断言阶段 -----
    if not args.ner_only and args.dataset != "yidu":
        run_assertion_smoke(args.dataset, args.n, do_train=not args.no_train)

    print("\n" + "█" * 70)
    print(f"█  ✅ 完成。总耗时 {(time.time()-t0)/60:.2f} 分钟".ljust(69) + "█")
    print(f"█  📁 输出目录: {OUTPUT_DIR}".ljust(69) + "█")
    print("█" * 70)


if __name__ == "__main__":
    main()

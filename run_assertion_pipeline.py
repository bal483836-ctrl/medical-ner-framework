"""
断言训练总流程（阶段 4-9）
依赖 NER 流程输出的 step3_final_{dataset}_{split}.json

数据流：
  NER 通过的实体（已过滤）
      ↓ 阶段 4: KG 语义扩展
      ↓ 阶段 5: 动态语境窗口截取
      ↓ 阶段 6: LLM 断言标注 → 训练数据
      ↓ 阶段 7: 分布检测 + 增强
      ↓ 阶段 8: RoBERTa 分类器训练 (train+dev)
      ↓ 阶段 9: test 评估 (macro F1 ≥ 0.90)

运行：
  python run_assertion_pipeline.py --dataset cmeee
  python run_assertion_pipeline.py --dataset imcs
  python run_assertion_pipeline.py --skip-annotate  # 已有标注，只重训
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import (
    OUTPUT_DIR, ASSERT_PREFIX, STEP3_PREFIX, DATASET_SPLITS,
)
from src.data_processor import clean_entity_list
from src.kg import load_kg
from src.context_window import context_from_text, context_from_dialogue
from src.assertion_annotator import annotate, save as save_annotations
from src.augmentor import augment, label_distribution
from src.assertion_train import train as train_clf, train_multi_seed
from src.assertion_eval import evaluate
from src.utils import set_global_seed, print_gpu_banner, stage_banner
from config.config import CLF_ENSEMBLE_SEEDS


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_samples_cmeee(items, kg, dataset_tag, split):
    """每个文档每个实体 → 若干语境样本。"""
    out = []
    for it in items:
        text = it.get("text", "")
        ents = clean_entity_list(it.get("step3_final_output",
                                        it.get("step2_aligned_output", "")))
        for ent in ents:
            exp = kg.expand(ent, topk=5)
            for c in context_from_text(text, ent):
                out.append({
                    "doc_id": it.get("id"),
                    "dataset": dataset_tag, "split": split,
                    "entity": ent, "context": c["context"],
                    "expansion": exp, "source": "text",
                    "label": "",
                })
    return out


def build_samples_imcs(items, kg, dataset_tag, split):
    out = []
    for it in items:
        dlg = it.get("dialogue", [])
        sr = it.get("self_report", "")
        ents = clean_entity_list(it.get("step3_final_output",
                                        it.get("step2_aligned_output", "")))
        for ent in ents:
            exp = kg.expand(ent, topk=5)
            for c in context_from_dialogue(dlg, ent, self_report=sr):
                out.append({
                    "dialogue_id": it.get("dialogue_id") or it.get("id"),
                    "dataset": dataset_tag, "split": split,
                    "entity": ent, "context": c["context"],
                    "expansion": exp, "source": "dialogue",
                    "label": "",
                })
    return out


def build_all_samples(dataset: str, kg) -> dict:
    """返回 {"train": [...], "dev": [...], "test": [...]}"""
    if dataset == "cmeee":
        ds = "CMeEE_V2"
        builder = build_samples_cmeee
    elif dataset == "imcs":
        ds = "IMCS_V2"
        builder = build_samples_imcs
    else:
        raise ValueError(dataset)

    out = {}
    for cfg in DATASET_SPLITS[ds]:
        split = cfg["split"]
        path = os.path.join(OUTPUT_DIR, f"{STEP3_PREFIX}{ds}_{split}.json")
        items = _load(path)
        if items is None:
            print(f"  ⚠️  缺少 NER 输出: {path}")
            out[split] = []
            continue
        out[split] = builder(items, kg, ds, split)
        print(f"  [Build] {ds}/{split}: {len(out[split])} 个 (实体,语境) 样本")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cmeee", "imcs"], required=True)
    ap.add_argument("--skip-annotate", action="store_true",
                    help="已有 annotation 文件时跳过 LLM 标注")
    ap.add_argument("--skip-augment", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--single-seed", action="store_true",
                    help="只训单 seed（默认按 CLF_ENSEMBLE_SEEDS 训多 seed）")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_global_seed()
    print_gpu_banner()
    ds_tag = "CMeEE_V2" if args.dataset == "cmeee" else "IMCS_V2"

    print("\n[KG] 加载知识图谱…")
    kg = load_kg()

    # ----- 阶段 4-5: 构建 (实体, 语境, 扩展) 样本 -----
    samples_by_split = build_all_samples(args.dataset, kg)

    # ----- 阶段 6: LLM 断言标注 -----
    for split in ["train", "dev", "test"]:
        path = os.path.join(OUTPUT_DIR, f"{ASSERT_PREFIX}{ds_tag}_{split}.json")
        existing = _load(path)
        if args.skip_annotate and existing is not None:
            samples_by_split[split] = existing
            print(f"  [Skip] 复用已有标注 {path}")
            continue
        if not samples_by_split[split]:
            continue
        # 若有部分历史标注，合并 entity+context 去重
        if existing:
            key = lambda s: (s.get("entity"), s.get("context"))
            seen = {key(s): s for s in existing if s.get("label")}
            for s in samples_by_split[split]:
                k = key(s)
                if k in seen:
                    s["label"] = seen[k]["label"]
        samples_by_split[split] = annotate(samples_by_split[split])
        save_annotations(samples_by_split[split], ds_tag, split)

    # ----- 阶段 7: 分布检测 + 增强（只对 train+dev）-----
    if not args.skip_augment:
        merged_tr = samples_by_split["train"] + samples_by_split["dev"]
        print(f"\n[Augment] 增强前分布: {label_distribution(merged_tr)}")
        augmented = augment(merged_tr)
        print(f"[Augment] 增强后分布: {label_distribution(augmented)}")
        # 把增强后的样本拆回 train（增强样本归到 train）
        n_tr = len(samples_by_split["train"])
        samples_by_split["train"] = augmented[:n_tr] + augmented[len(merged_tr):]
        # dev 保持原样
        save_annotations(samples_by_split["train"], ds_tag, "train_aug")

    # ----- 阶段 8: 小模型训练 (严格不用 test) -----
    if not args.skip_train:
        if args.single_seed or not CLF_ENSEMBLE_SEEDS:
            model_dirs = [train_clf(
                train_samples=samples_by_split["train"],
                dev_samples=samples_by_split["dev"],
            )]
        else:
            model_dirs = train_multi_seed(
                train_samples=samples_by_split["train"],
                dev_samples=samples_by_split["dev"],
                seeds=CLF_ENSEMBLE_SEEDS,
            )
    else:
        # 推断已有 seed 目录
        base = os.path.join(OUTPUT_DIR, "assertion_clf")
        candidates = []
        if CLF_ENSEMBLE_SEEDS:
            candidates = [os.path.join(base, f"seed_{s}", "final")
                          for s in CLF_ENSEMBLE_SEEDS
                          if os.path.exists(os.path.join(base, f"seed_{s}", "final"))]
        model_dirs = candidates or [os.path.join(base, "final")]

    # ----- 阶段 9: test 评估（含验证集阈值搜索 + 集成）-----
    if not args.skip_eval and samples_by_split["test"]:
        labeled_test = [s for s in samples_by_split["test"] if s.get("label")]
        if not labeled_test:
            print("  ⚠️  test 无标注，无法评估 macro F1。")
        else:
            # 把 dev 当作阈值搜索集（test 完全独立）
            val_for_bias = samples_by_split.get("dev", [])
            evaluate(model_dirs, labeled_test, val_samples=val_for_bias)

    print(f"\n✅ 完成。总耗时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()

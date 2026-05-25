"""
统一断言训练流水线（v4.4）
跨数据集训练：CMeEE_V2 + IMCS_V2 + yidu_4k 三个数据集的 train 合并训练，
              dev 合并验证，**test 分别评估**得到三个独立 macro F1。

流程：
  Stage 5-6 (per dataset)：
    NER step3 → 构建 (实体, 语境, KG扩展) 样本 → LLM 自洽投票标注
    每个样本打 dataset 标签 (CMeEE_V2 / IMCS_V2 / yidu_4k)

  Stage 7 (merged)：
    train_all = ∪{CMeEE.train, IMCS.train, yidu.train}
    dev_all   = ∪{CMeEE.dev,   IMCS.dev,   yidu.dev}
    在 train_all 上做按类目标补足增强

  Stage 8 (merged)：
    多 seed 训练（RoBERTa + FocalLoss + FGM + R-Drop）
    train+dev 合并送入，按 doc_id group split 出内部 val

  Stage 9 (per dataset)：
    在 CMeEE.test / IMCS.test / yidu.test 上**分别** evaluate
    每个 test 集报告独立 macro F1 + per-class P/R/F1

数据泄露防护：
  - 任何 test 永不进入训练
  - dev 阈值搜索仅用合并 dev_all（不用任何 test）
  - 增强样本只追加 train_all

运行：
  python run_unified_assertion.py
  python run_unified_assertion.py --skip-annotate   # 已有标注文件，跳过 LLM 标注
  python run_unified_assertion.py --skip-train      # 已有训练好的模型，仅评估
  python run_unified_assertion.py --single-seed     # 训单 seed
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import (
    OUTPUT_DIR, ASSERT_PREFIX, STEP3_PREFIX, STEP1_PREFIX,
    DATASET_SPLITS, ASSERTION_LABELS, CLF_ENSEMBLE_SEEDS,
    F1_TARGET_ASSERTION,
)
from src.data_processor import clean_entity_list
from src.kg import load_kg
from src.context_window import context_from_text, context_from_dialogue
from src.assertion_annotator import annotate, save as save_annotations
from src.augmentor import augment, label_distribution
from src.assertion_train import train as train_clf, train_multi_seed
from src.assertion_eval import evaluate, ensemble_predict
from src.utils import set_global_seed, print_gpu_banner, stage_banner, kv_print


# ==================== 配置 ====================

DATASETS = ["CMeEE_V2", "IMCS_V2", "yidu_4k"]


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _step3_path(ds: str, split: str) -> str:
    """优先用正式 step3 输出；fallback 到冒烟测试输出。"""
    p = os.path.join(OUTPUT_DIR, f"{STEP3_PREFIX}{ds}_{split}.json")
    if os.path.exists(p):
        return p
    smoke = os.path.join(OUTPUT_DIR, f"smoke_{STEP3_PREFIX}{ds}_{split}.json")
    return smoke if os.path.exists(smoke) else p


# ==================== 样本构建（每数据集）====================

def build_samples_one_split(ds: str, split: str, kg) -> List[Dict]:
    """从 NER step3 输出构建 (实体, 语境, KG扩展) 样本。"""
    path = _step3_path(ds, split)
    items = _load(path)
    if items is None:
        print(f"  ⚠️ 缺 NER 输出: {path}")
        return []

    out = []
    for it in items:
        # 取 step3 过滤后的实体
        ents = clean_entity_list(it.get("step3_final_output",
                                        it.get("step2_aligned_output", "")))
        if ds == "IMCS_V2":
            dlg = it.get("dialogue", [])
            sr  = it.get("self_report", "")
            for ent in ents:
                exp = kg.expand(ent, topk=5)
                for c in context_from_dialogue(dlg, ent, self_report=sr):
                    out.append({
                        "dataset": ds, "split": split,
                        "dialogue_id": it.get("dialogue_id") or it.get("id"),
                        "entity": ent, "context": c["context"],
                        "expansion": exp, "source": "dialogue",
                        "label": "",
                    })
        else:
            text = it.get("text", "")
            for ent in ents:
                exp = kg.expand(ent, topk=5)
                for c in context_from_text(text, ent):
                    out.append({
                        "dataset": ds, "split": split,
                        "doc_id": it.get("id"),
                        "entity": ent, "context": c["context"],
                        "expansion": exp, "source": "text",
                        "label": "",
                    })
    return out


# ==================== 主流程 ====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-annotate", action="store_true",
                    help="跳过 LLM 标注（复用已有 assertion_*.json）")
    ap.add_argument("--skip-augment", action="store_true")
    ap.add_argument("--skip-train", action="store_true",
                    help="跳过训练，直接评估已有模型")
    ap.add_argument("--single-seed", action="store_true",
                    help="只训一个 seed（默认按 CLF_ENSEMBLE_SEEDS 训多 seed）")
    ap.add_argument("--vote-passes", type=int, default=3,
                    help="LLM 自洽投票轮数")
    ap.add_argument("--limit-train-per-ds", type=int, default=0,
                    help="每数据集 train 限制条数（0=全部）")
    ap.add_argument("--limit-test-per-ds", type=int, default=0,
                    help="每数据集 test 限制条数（0=全部）")
    args = ap.parse_args()

    t0 = time.time()
    set_global_seed()
    print_gpu_banner()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "█" * 70)
    print("█  🌐 统一断言流水线 v4.4：三集合并训练 + 分别评估".ljust(69) + "█")
    print(f"█  datasets={DATASETS}".ljust(69) + "█")
    print("█" * 70)

    kg = load_kg()

    # ==================== Stage 5-6：每数据集构建 + 标注 ====================
    samples_by_ds_split: Dict[str, Dict[str, List]] = {}

    for ds in DATASETS:
        stage_banner(f"Stage 5-6 [{ds}]", "构建样本 + LLM 自洽投票标注")
        samples_by_ds_split[ds] = {}
        for split in ("train", "dev", "test"):
            samples = build_samples_one_split(ds, split, kg)
            if args.limit_train_per_ds and split == "train":
                samples = samples[: args.limit_train_per_ds]
            if args.limit_test_per_ds and split == "test":
                samples = samples[: args.limit_test_per_ds]

            anno_path = os.path.join(OUTPUT_DIR,
                                     f"{ASSERT_PREFIX}{ds}_{split}.json")
            # 跳过已有标注
            if args.skip_annotate and os.path.exists(anno_path):
                cached = _load(anno_path)
                if cached:
                    print(f"  [Skip] 复用 {anno_path}（{len(cached)} 条）")
                    samples_by_ds_split[ds][split] = cached
                    continue

            print(f"  {ds}/{split}: {len(samples)} 个 (实体,语境) 样本")
            if not samples:
                samples_by_ds_split[ds][split] = []
                continue

            # LLM 标注（带断点续传）
            samples = annotate(samples, vote_passes=args.vote_passes,
                              output_path=anno_path)
            save_annotations(samples, ds, split)
            samples_by_ds_split[ds][split] = samples

            cnt = Counter(s.get("label", "") for s in samples)
            print(f"  {ds}/{split} 标签分布: {dict(cnt)}")

    # ==================== Stage 7：三集合并训练集 + 增强 ====================
    stage_banner("Stage 7", "三集合并训练 + 分布检测 + 增强")
    train_merged: List[Dict] = []
    dev_merged:   List[Dict] = []
    for ds in DATASETS:
        train_merged.extend(samples_by_ds_split[ds].get("train", []))
        dev_merged.extend(samples_by_ds_split[ds].get("dev", []))

    print(f"  合并 train 总数: {len(train_merged)}")
    print(f"    分布: {dict(Counter(s['dataset'] for s in train_merged))}")
    print(f"  合并 dev   总数: {len(dev_merged)}")
    print(f"    分布: {dict(Counter(s['dataset'] for s in dev_merged))}")
    print(f"\n  合并 train 标签分布: {label_distribution(train_merged)}")
    print(f"  合并 dev   标签分布: {label_distribution(dev_merged)}")

    # 保存合并后的训练集（断点用）
    merged_train_path = os.path.join(OUTPUT_DIR, f"{ASSERT_PREFIX}UNIFIED_train.json")
    merged_dev_path   = os.path.join(OUTPUT_DIR, f"{ASSERT_PREFIX}UNIFIED_dev.json")
    with open(merged_train_path, "w", encoding="utf-8") as f:
        json.dump(train_merged, f, ensure_ascii=False, indent=2)
    with open(merged_dev_path, "w", encoding="utf-8") as f:
        json.dump(dev_merged, f, ensure_ascii=False, indent=2)
    print(f"  💾 合并 train → {merged_train_path}")
    print(f"  💾 合并 dev   → {merged_dev_path}")

    # 增强（在合并的 train+dev 上做，但分布检测应只看 train）
    if not args.skip_augment:
        aug_ckpt = os.path.join(OUTPUT_DIR, f"{ASSERT_PREFIX}UNIFIED_aug_ckpt")
        # 只对 train 做增强（dev 保持原始）
        augmented_train = augment(train_merged, checkpoint_path=aug_ckpt)
        # 把新增的增强样本追加（保留 dataset 字段）
        n_added = len(augmented_train) - len(train_merged)
        print(f"  增强新增样本: {n_added}")
        # 给新增样本打 dataset 标签（继承自源样本，augment 已经传递）
        train_merged = augmented_train

    # ==================== Stage 8：统一分类器训练 ====================
    stage_banner("Stage 8", "RoBERTa 统一分类器训练（多 seed 集成）")

    if not args.skip_train:
        if args.single_seed or not CLF_ENSEMBLE_SEEDS:
            model_dir = train_clf(
                train_samples=train_merged,
                dev_samples=dev_merged,
                save_dir=os.path.join(OUTPUT_DIR, "unified_assertion_clf"),
            )
            model_dirs = [model_dir]
        else:
            model_dirs = train_multi_seed(
                train_samples=train_merged,
                dev_samples=dev_merged,
                seeds=CLF_ENSEMBLE_SEEDS,
                base_dir=os.path.join(OUTPUT_DIR, "unified_assertion_clf"),
            )
    else:
        base = os.path.join(OUTPUT_DIR, "unified_assertion_clf")
        candidates = []
        if CLF_ENSEMBLE_SEEDS:
            candidates = [
                os.path.join(base, f"seed_{s}", "final")
                for s in CLF_ENSEMBLE_SEEDS
                if os.path.exists(os.path.join(base, f"seed_{s}", "final"))
            ]
        if not candidates:
            candidates = [os.path.join(base, "final")]
        model_dirs = candidates
        print(f"  跳过训练，复用模型: {model_dirs}")

    # ==================== Stage 9：分别在 3 个 test 集上评估 ====================
    stage_banner("Stage 9", "在 3 个 test 集上分别评估 macro F1")

    summary = {
        "datasets": DATASETS,
        "n_models_ensembled": len(model_dirs),
        "model_dirs": model_dirs,
        "target": F1_TARGET_ASSERTION,
        "per_dataset_results": {},
        "passed_target_per_ds": {},
    }

    for ds in DATASETS:
        test_samples = samples_by_ds_split[ds].get("test", [])
        labeled = [s for s in test_samples
                   if s.get("label") in {l: i for i, l in enumerate(ASSERTION_LABELS)}]

        print(f"\n{'─' * 60}")
        print(f"  📊 评估 [{ds}.test]  样本数: {len(labeled)} / 总 {len(test_samples)}")
        print(f"{'─' * 60}")

        if not labeled:
            print(f"  ⚠️  {ds}.test 无标注样本，跳过")
            summary["per_dataset_results"][ds] = {"error": "no labeled test"}
            continue

        result = evaluate(
            model_dirs, labeled,
            val_samples=dev_merged,        # dev 仅用于阈值搜索
            save_report=False,
        )
        summary["per_dataset_results"][ds] = result
        summary["passed_target_per_ds"][ds] = result.get("passed_target", False)

    # ==================== 汇总报告 ====================
    print("\n" + "═" * 70)
    print("  🏆 三数据集 test 集 macro F1 汇总")
    print("═" * 70)
    for ds in DATASETS:
        r = summary["per_dataset_results"].get(ds, {})
        if "error" in r:
            print(f"  {ds:12s}  →  ❌ {r['error']}")
        else:
            mark = "✅" if r.get("passed_target") else "❌"
            print(f"  {ds:12s}  →  macro F1 = {r['macro_f1']:.4f}  "
                  f"micro F1 = {r['micro_f1']:.4f}  "
                  f"{mark} (target {F1_TARGET_ASSERTION})")

    macros = [r["macro_f1"] for r in summary["per_dataset_results"].values()
              if isinstance(r, dict) and "macro_f1" in r]
    if macros:
        summary["avg_macro_f1"] = round(sum(macros) / len(macros), 4)
        print(f"\n  平均 macro F1 = {summary['avg_macro_f1']:.4f}")

    report_path = os.path.join(OUTPUT_DIR, "unified_assertion_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  📄 汇总报告: {report_path}")
    print(f"  ⏱️  总耗时 {(time.time() - t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()

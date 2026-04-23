"""
独立 F1 评估脚本 v3.2
直接读取 outputs/ 目录中已有的 step3_final 文件，计算 F1，无需重新运行 pipeline。

用法：
  # 评估所有数据集所有 split
  python evaluate_all.py

  # 只评估指定数据集
  python evaluate_all.py --dataset cmeee
  python evaluate_all.py --dataset imcs

  # 只评估指定 split
  python evaluate_all.py --dataset cmeee --split dev
  python evaluate_all.py --dataset imcs  --split train

  # 指定 outputs 目录（默认自动查找）
  python evaluate_all.py --outputs /root/autodl-tmp/MedNER_Project/better/medical_ner_v3/outputs

  # 不加载向量模型（跳过 IMCS 归一化 F1 中的向量补充步骤，速度更快）
  python evaluate_all.py --no-embed
"""
import argparse
import json
import os
import re
import sys
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# ==================== 路径自动查找 ====================

def _find_outputs_dir() -> str:
    """自动查找 outputs 目录：优先找当前目录下的，再找上级目录"""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"),
        "/root/autodl-tmp/MedNER_Project/better/medical_ner_v3/outputs",
        "/root/autodl-tmp/MedNER_Project/medical_ner_v3/outputs",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def _find_symptom_norm_csv() -> Optional[str]:
    """自动查找 symptom_norm.csv"""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "symptom_norm.csv"),
        "/root/autodl-tmp/MedNER_Project/data/symptom_norm.csv",
        "/root/autodl-tmp/MedNER_Project/better/medical_ner_v3/data/symptom_norm.csv",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


# ==================== 工具函数 ====================

def clean_entity_list(raw: str) -> List[str]:
    """将逗号分隔的实体字符串清洗为列表"""
    if not raw:
        return []
    raw = re.sub(r"\s+", "", raw)
    items = re.split(r"[,，、；;]", raw)
    result = []
    for item in items:
        item = item.strip().strip('"\'""''')
        if item and item not in ("无", "null", "None", "none", "NULL"):
            result.append(item)
    return result


def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_symptom_norm_vocab(csv_path: Optional[str]) -> List[str]:
    """加载官方 symptom_norm 词典（331个标准词）"""
    if not csv_path or not os.path.isfile(csv_path):
        print("  [警告] 未找到 symptom_norm.csv，IMCS 归一化将只使用 step2_normalized_map")
        return []
    vocab = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("symptom_norm") and not line.startswith("#"):
                vocab.append(line.split(",")[0].strip())
    print(f"  [词典] 加载 symptom_norm 词典: {len(vocab)} 个标准词")
    return vocab


# ==================== Micro F1 计算 ====================

def compute_micro_f1(
    y_true: List[List[str]],
    y_pred: List[List[str]],
) -> Tuple[float, float, float, int, int, int]:
    """返回 (precision, recall, f1, tp, fp, fn)"""
    tp = fp = fn = 0
    for gold_list, pred_list in zip(y_true, y_pred):
        gold_set = set(gold_list)
        pred_set = set(pred_list)
        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, tp, fp, fn


# ==================== CMeEE 评估 ====================

def evaluate_cmeee_file(path: str, split: str) -> Optional[Dict]:
    """评估单个 CMeEE step3_final 文件"""
    if not os.path.isfile(path):
        print(f"  [跳过] 文件不存在: {path}")
        return None

    items = load_json(path)
    y_true, y_pred = [], []
    no_gold_count = 0

    for item in items:
        gold_raw = item.get("gold_entities_str", "")
        gold = clean_entity_list(gold_raw)
        if not gold:
            no_gold_count += 1

        # 预测来源优先级：step3_final_output > step1_enriched_output > step1_raw_output
        pred_raw = (
            item.get("step3_final_output") or
            item.get("step1_enriched_output") or
            item.get("step1_raw_output") or ""
        )
        pred = clean_entity_list(pred_raw)
        y_true.append(gold)
        y_pred.append(pred)

    if no_gold_count == len(items):
        print(f"  [跳过] CMeEE [{split}] 全部样本无 Gold 标注（test 集），跳过 F1 评估")
        print(f"  [提取] CMeEE [{split}] 共提取 {len(items)} 条，实体数: "
              f"{sum(len(p) for p in y_pred)}")
        return None

    p, r, f1, tp, fp, fn = compute_micro_f1(y_true, y_pred)

    result = {
        "dataset": "CMeEE_V2",
        "split": split,
        "eval_type": "字面匹配",
        "micro_f1": round(f1, 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "total_samples": len(items),
        "no_gold_samples": no_gold_count,
    }
    _print_cmeee_result(result)
    return result


def _print_cmeee_result(r: Dict):
    print(f"\n{'='*60}")
    print(f"  CMeEE_V2 [{r['split']}]  {r['eval_type']}")
    print(f"{'='*60}")
    print(f"  样本数:      {r['total_samples']}  (无Gold: {r['no_gold_samples']})")
    print(f"  TP={r['tp']}  FP={r['fp']}  FN={r['fn']}")
    print(f"  Micro F1:    {r['micro_f1']:.4f}")
    print(f"  Precision:   {r['precision']:.4f}")
    print(f"  Recall:      {r['recall']:.4f}")
    _target_hint(r['micro_f1'])


# ==================== IMCS 评估 ====================

def evaluate_imcs_file(
    path: str,
    split: str,
    norm_vocab: List[str],
    use_embed: bool = True,
) -> Optional[Dict]:
    """评估单个 IMCS step3_final 文件（字面 F1 + 归一化 F1）"""
    if not os.path.isfile(path):
        print(f"  [跳过] 文件不存在: {path}")
        return None

    items = load_json(path)
    y_true, y_pred_literal, y_pred_norm = [], [], []
    no_gold_count = 0

    # 预加载向量模型（如果需要）
    embed_fn = None
    if use_embed and norm_vocab:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from src.embedding_model import find_best_matches
            embed_fn = find_best_matches
            print(f"  [向量模型] 已加载，用于 IMCS 归一化补充")
        except Exception as e:
            print(f"  [向量模型] 加载失败: {e}，将只使用 step2_normalized_map")

    for item in items:
        gold_raw = item.get("gold_entities_str", "")
        gold = clean_entity_list(gold_raw)
        if not gold:
            no_gold_count += 1

        # 字面预测（原词）
        pred_raw = clean_entity_list(
            item.get("step3_final_output", item.get("step1_raw_output", ""))
        )

        # 归一化预测
        norm_map = item.get("step2_normalized_map", {})
        normalized = []
        needs_embed = []
        for ent in pred_raw:
            if ent in norm_map:
                normalized.append(norm_map[ent])
            elif ent in norm_vocab:
                normalized.append(ent)  # 已经是标准词
            else:
                needs_embed.append(ent)

        # 向量模型补充归一化
        if needs_embed and norm_vocab and embed_fn:
            try:
                matches = embed_fn(needs_embed, norm_vocab)
                for ent, (best_match, score, status) in zip(needs_embed, matches):
                    if status in ("exact", "high", "medium"):
                        normalized.append(best_match)
                    else:
                        normalized.append(ent)
            except Exception:
                normalized.extend(needs_embed)
        else:
            normalized.extend(needs_embed)

        y_true.append(gold)
        y_pred_literal.append(pred_raw)
        y_pred_norm.append(list(dict.fromkeys(normalized)))

    if no_gold_count == len(items):
        print(f"  [跳过] IMCS [{split}] 全部样本无 Gold 标注（test 集），跳过 F1 评估")
        return None

    p_l, r_l, f1_l, tp_l, fp_l, fn_l = compute_micro_f1(y_true, y_pred_literal)
    p_n, r_n, f1_n, tp_n, fp_n, fn_n = compute_micro_f1(y_true, y_pred_norm)

    result = {
        "dataset": "IMCS_V2",
        "split": split,
        "literal_micro_f1": round(f1_l, 4),
        "literal_precision": round(p_l, 4),
        "literal_recall": round(r_l, 4),
        "literal_tp": tp_l, "literal_fp": fp_l, "literal_fn": fn_l,
        "normalized_micro_f1": round(f1_n, 4),
        "normalized_precision": round(p_n, 4),
        "normalized_recall": round(r_n, 4),
        "normalized_tp": tp_n, "normalized_fp": fp_n, "normalized_fn": fn_n,
        "total_samples": len(items),
        "no_gold_samples": no_gold_count,
    }
    _print_imcs_result(result)
    return result


def _print_imcs_result(r: Dict):
    print(f"\n{'='*60}")
    print(f"  IMCS_V2 [{r['split']}]")
    print(f"{'='*60}")
    print(f"  样本数:          {r['total_samples']}  (无Gold: {r['no_gold_samples']})")
    print(f"  ── 字面匹配 ──────────────────────────────")
    print(f"  TP={r['literal_tp']}  FP={r['literal_fp']}  FN={r['literal_fn']}")
    print(f"  字面 Micro F1:   {r['literal_micro_f1']:.4f}  "
          f"(P={r['literal_precision']:.4f}, R={r['literal_recall']:.4f})")
    print(f"  ── 归一化匹配 ────────────────────────────")
    print(f"  TP={r['normalized_tp']}  FP={r['normalized_fp']}  FN={r['normalized_fn']}")
    print(f"  归一化 Micro F1: {r['normalized_micro_f1']:.4f}  "
          f"(P={r['normalized_precision']:.4f}, R={r['normalized_recall']:.4f})")
    _target_hint(r['normalized_micro_f1'])


def _target_hint(f1: float, target: float = 0.80):
    if f1 >= target:
        print(f"  ✅ F1={f1:.4f} >= {target}，目标达成！")
    else:
        print(f"  ⚠️  F1={f1:.4f}，距目标 {target} 还差 {target - f1:.4f}")
    print(f"{'='*60}")


# ==================== 汇总报告 ====================

def print_summary(all_results: List[Dict]):
    """打印汇总表格"""
    print(f"\n\n{'#'*60}")
    print(f"  全量评估汇总")
    print(f"{'#'*60}")
    print(f"  {'数据集':<14} {'Split':<8} {'评估类型':<10} {'Micro F1':>9} {'P':>7} {'R':>7}")
    print(f"  {'-'*56}")
    for r in all_results:
        if r is None:
            continue
        if "micro_f1" in r:
            mark = "✅" if r["micro_f1"] >= 0.80 else "  "
            print(f"  {r['dataset']:<14} {r['split']:<8} {'字面匹配':<10} "
                  f"{r['micro_f1']:>8.4f} {r['precision']:>7.4f} {r['recall']:>7.4f}  {mark}")
        elif "literal_micro_f1" in r:
            mark_n = "✅" if r["normalized_micro_f1"] >= 0.80 else "  "
            print(f"  {r['dataset']:<14} {r['split']:<8} {'字面匹配':<10} "
                  f"{r['literal_micro_f1']:>8.4f} {r['literal_precision']:>7.4f} {r['literal_recall']:>7.4f}")
            print(f"  {'':<14} {'':<8} {'归一化匹配':<10} "
                  f"{r['normalized_micro_f1']:>8.4f} {r['normalized_precision']:>7.4f} {r['normalized_recall']:>7.4f}  {mark_n}")
    print(f"{'#'*60}\n")


def save_summary_json(all_results: List[Dict], outputs_dir: str):
    """保存评估结果到 JSON"""
    out_path = os.path.join(outputs_dir, "evaluation_summary.json")
    valid = [r for r in all_results if r is not None]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(valid, f, ensure_ascii=False, indent=2)
    print(f"  评估结果已保存: {out_path}")


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="医疗NER框架独立F1评估脚本")
    parser.add_argument("--outputs", type=str, default=None,
                        help="outputs 目录路径（默认自动查找）")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "cmeee", "imcs"],
                        help="要评估的数据集（默认 all）")
    parser.add_argument("--split", type=str, default="all",
                        choices=["all", "train", "dev", "test"],
                        help="要评估的 split（默认 all）")
    parser.add_argument("--no-embed", action="store_true",
                        help="不加载向量模型（跳过 IMCS 归一化的向量补充步骤）")
    args = parser.parse_args()

    # 确定 outputs 目录
    outputs_dir = args.outputs or _find_outputs_dir()
    print(f"\n[评估] outputs 目录: {outputs_dir}")
    if not os.path.isdir(outputs_dir):
        print(f"[错误] outputs 目录不存在: {outputs_dir}")
        sys.exit(1)

    # 加载 symptom_norm 词典
    norm_csv = _find_symptom_norm_csv()
    norm_vocab = load_symptom_norm_vocab(norm_csv)

    # 确定要评估的数据集和 split
    datasets = ["cmeee", "imcs"] if args.dataset == "all" else [args.dataset]
    splits   = ["train", "dev", "test"] if args.split == "all" else [args.split]

    all_results = []

    # ---- CMeEE 评估 ----
    if "cmeee" in datasets:
        print(f"\n{'='*60}")
        print(f"  开始评估 CMeEE_V2")
        print(f"{'='*60}")
        for split in splits:
            # 优先读 step3_final，其次 step1_enriched
            path_step3    = os.path.join(outputs_dir, f"step3_final_CMeEE_V2_{split}.json")
            path_enriched = os.path.join(outputs_dir, f"step1_enriched_CMeEE_V2_{split}.json")
            path_step1    = os.path.join(outputs_dir, f"step1_raw_CMeEE_V2_{split}.json")

            if os.path.isfile(path_step3):
                print(f"\n  [CMeEE {split}] 使用 step3_final 文件")
                result = evaluate_cmeee_file(path_step3, split)
            elif os.path.isfile(path_enriched):
                print(f"\n  [CMeEE {split}] 使用 step1_enriched 文件（step3 未完成）")
                result = evaluate_cmeee_file(path_enriched, split)
            elif os.path.isfile(path_step1):
                print(f"\n  [CMeEE {split}] 使用 step1_raw 文件（step1 完成，step2/3 未完成）")
                result = evaluate_cmeee_file(path_step1, split)
            else:
                print(f"\n  [CMeEE {split}] 未找到任何输出文件，跳过")
                result = None
            all_results.append(result)

    # ---- IMCS 评估 ----
    if "imcs" in datasets:
        print(f"\n{'='*60}")
        print(f"  开始评估 IMCS_V2")
        print(f"{'='*60}")
        for split in splits:
            path_step3 = os.path.join(outputs_dir, f"step3_final_IMCS_V2_{split}.json")
            path_step1 = os.path.join(outputs_dir, f"step1_raw_IMCS_V2_{split}.json")

            if os.path.isfile(path_step3):
                print(f"\n  [IMCS {split}] 使用 step3_final 文件")
                result = evaluate_imcs_file(
                    path_step3, split, norm_vocab, use_embed=not args.no_embed
                )
            elif os.path.isfile(path_step1):
                print(f"\n  [IMCS {split}] 使用 step1_raw 文件（step2/3 未完成）")
                result = evaluate_imcs_file(
                    path_step1, split, norm_vocab, use_embed=not args.no_embed
                )
            else:
                print(f"\n  [IMCS {split}] 未找到任何输出文件，跳过")
                result = None
            all_results.append(result)

    # ---- 汇总 ----
    valid_results = [r for r in all_results if r is not None]
    if valid_results:
        print_summary(valid_results)
        save_summary_json(valid_results, outputs_dir)
    else:
        print("\n[警告] 没有找到任何有效的评估结果")


if __name__ == "__main__":
    main()

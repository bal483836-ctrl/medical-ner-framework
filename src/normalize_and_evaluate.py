"""
Step 4: 归一化与 F1 评估模块 v3（修复版）
CMeEE：字面 Micro F1（直接比较提取结果与 Gold 标注）
IMCS：
  - 字面 F1（口语化表述直接比较）
  - 归一化 F1（将口语化表述归一化为 symptom_norm 标准词后比较）
yidu：只输出提取结果，不评估 F1

关键修复（v3.1）：
  IMCS 归一化评估现在正确读取 step3_final_output（原词）
  然后通过 step2_normalized_map 映射到标准词，
  对于 step2_normalized_map 中没有的词，再用向量模型归一化。
  Gold 标准也统一使用 symptom_norm 标准词（item["gold_entities_str"]）。
"""
import json
import os
import re
import sys
import warnings
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import OUTPUT_DIR, F1_TARGET, DUAL_F1_EVAL
from src.data_processor import clean_entity_list, build_imcs_norm_vocab
from src.embedding_model import find_best_matches


# ==================== 通用 F1 计算 ====================

def compute_micro_f1(
    y_true: List[List[str]],
    y_pred: List[List[str]],
) -> Tuple[float, float, float]:
    """
    计算 Micro F1（精确率、召回率、F1）
    Args:
        y_true: 每条样本的 Gold 实体列表
        y_pred: 每条样本的预测实体列表
    Returns:
        (precision, recall, f1)
    """
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
    return precision, recall, f1


def compute_macro_f1(
    y_true: List[List[str]],
    y_pred: List[List[str]],
) -> Tuple[float, float, float]:
    """计算 Macro F1"""
    precisions, recalls, f1s = [], [], []
    for gold_list, pred_list in zip(y_true, y_pred):
        gold_set = set(gold_list)
        pred_set = set(pred_list)
        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
    return (
        sum(precisions) / len(precisions) if precisions else 0.0,
        sum(recalls)    / len(recalls)    if recalls    else 0.0,
        sum(f1s)        / len(f1s)        if f1s        else 0.0,
    )


# ==================== CMeEE 评估 ====================

def evaluate_cmeee(
    items: List[Dict],
    split: str,
    dataset_name: str = "CMeEE_V2",
) -> Dict:
    """
    CMeEE 字面 Micro F1 评估
    预测来源：step3_final_output（若不存在则用 step1_enriched_output）
    Gold 来源：gold_entities_str
    """
    y_true, y_pred = [], []
    for item in items:
        gold = clean_entity_list(item.get("gold_entities_str", ""))
        pred_raw = (
            item.get("step3_final_output") or
            item.get("step1_enriched_output") or
            item.get("step1_raw_output") or ""
        )
        pred = clean_entity_list(pred_raw)
        y_true.append(gold)
        y_pred.append(pred)

    mi_p, mi_r, mi_f1 = compute_micro_f1(y_true, y_pred)
    ma_p, ma_r, ma_f1 = compute_macro_f1(y_true, y_pred)

    result = {
        "dataset": dataset_name,
        "split": split,
        "eval_type": "literal",
        "micro_f1": round(mi_f1, 4),
        "micro_precision": round(mi_p, 4),
        "micro_recall": round(mi_r, 4),
        "macro_f1": round(ma_f1, 4),
        "macro_precision": round(ma_p, 4),
        "macro_recall": round(ma_r, 4),
        "total_samples": len(items),
    }

    _print_eval_result(result)
    return result


# ==================== IMCS 评估 ====================

def evaluate_imcs(
    items: List[Dict],
    split: str,
    norm_vocab: Optional[List[str]] = None,
) -> Dict:
    """
    IMCS 双 F1 评估（修复版）：
      1. 字面 F1：直接比较 step3_final_output（原词）与 gold_entities_str（标准词）
      2. 归一化 F1：将 step3_final_output 通过 step2_normalized_map 归一化后，
                   再与 gold_entities_str（标准词）比较

    Gold 标准说明：
      IMCS 的 gold_entities_str 存储的是 symptom_norm 标准词，
      所以归一化 F1 才是主要指标（字面 F1 因口语化差异天然偏低）。
    """
    if norm_vocab is None:
        norm_vocab = build_imcs_norm_vocab()

    # ---- 字面 F1 ----
    y_true_literal, y_pred_literal = [], []
    for item in items:
        gold = clean_entity_list(item.get("gold_entities_str", ""))
        pred = clean_entity_list(item.get("step3_final_output", item.get("step1_raw_output", "")))
        y_true_literal.append(gold)
        y_pred_literal.append(pred)

    mi_p_l, mi_r_l, mi_f1_l = compute_micro_f1(y_true_literal, y_pred_literal)

    # ---- 归一化 F1（修复版）----
    # 从 step3_final_output（原词）→ step2_normalized_map → 标准词
    y_pred_norm = []
    for item in items:
        pred_raw = clean_entity_list(
            item.get("step3_final_output", item.get("step1_raw_output", ""))
        )
        # 优先使用 Step2 已建立的映射表
        norm_map = item.get("step2_normalized_map", {})

        normalized = []
        needs_embed = []
        for ent in pred_raw:
            if ent in norm_map:
                normalized.append(norm_map[ent])
            else:
                needs_embed.append(ent)

        # 对没有 Step2 映射的词，用向量模型归一化
        if needs_embed and norm_vocab:
            matches = find_best_matches(needs_embed, norm_vocab)
            for ent, (best_match, score, status) in zip(needs_embed, matches):
                if status in ("exact", "high"):
                    normalized.append(best_match)
                elif status == "medium":
                    # 中等相似度：也接受归一化（提升召回）
                    normalized.append(best_match)
                else:
                    normalized.append(ent)  # 低相似度保留原词
        else:
            normalized.extend(needs_embed)

        y_pred_norm.append(list(dict.fromkeys(normalized)))

    # Gold 使用标准词（gold_entities_str 已经是 symptom_norm 标准词）
    y_true_norm = y_true_literal  # Gold 已经是标准词，直接复用
    mi_p_n, mi_r_n, mi_f1_n = compute_micro_f1(y_true_norm, y_pred_norm)

    result = {
        "dataset": "IMCS_V2",
        "split": split,
        "literal_micro_f1": round(mi_f1_l, 4),
        "literal_precision": round(mi_p_l, 4),
        "literal_recall": round(mi_r_l, 4),
        "normalized_micro_f1": round(mi_f1_n, 4),
        "normalized_precision": round(mi_p_n, 4),
        "normalized_recall": round(mi_r_n, 4),
        "total_samples": len(items),
    }

    _print_imcs_eval_result(result)
    return result


# ==================== 评估报告输出 ====================

def _print_eval_result(result: Dict):
    ds    = result["dataset"]
    split = result["split"]
    print(f"\n{'='*55}")
    print(f"  {ds} [{split}] 评估报告")
    print(f"{'='*55}")
    print(f"  样本数量:    {result['total_samples']}")
    print(f"  Micro F1:    {result['micro_f1']:.4f}  "
          f"(P={result['micro_precision']:.4f}, R={result['micro_recall']:.4f})")
    print(f"  Macro F1:    {result['macro_f1']:.4f}  "
          f"(P={result['macro_precision']:.4f}, R={result['macro_recall']:.4f})")
    _print_target_status(result['micro_f1'])


def _print_imcs_eval_result(result: Dict):
    split = result["split"]
    print(f"\n{'='*55}")
    print(f"  IMCS_V2 [{split}] 评估报告")
    print(f"{'='*55}")
    print(f"  样本数量:      {result['total_samples']}")
    print(f"  字面 Micro F1:   {result['literal_micro_f1']:.4f}  "
          f"(P={result['literal_precision']:.4f}, R={result['literal_recall']:.4f})")
    print(f"  归一化 Micro F1: {result['normalized_micro_f1']:.4f}  "
          f"(P={result['normalized_precision']:.4f}, R={result['normalized_recall']:.4f})")
    _print_target_status(result['normalized_micro_f1'])


def _print_target_status(f1: float):
    if f1 >= F1_TARGET:
        print(f"  Micro F1 = {f1:.4f} >= {F1_TARGET}，目标达成！")
    else:
        gap = F1_TARGET - f1
        print(f"  距目标 {F1_TARGET} 还差 {gap:.4f}")
    print(f"{'='*55}")


# ==================== 全量评估汇总 ====================

def generate_full_report(all_results: List[Dict], output_path: str):
    """生成全量评估汇总报告"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = ["# 医疗NER框架 全量评估报告\n"]
    lines.append("| 数据集 | Split | 评估类型 | Micro F1 | Precision | Recall |")
    lines.append("|--------|-------|----------|----------|-----------|--------|")

    for r in all_results:
        if "micro_f1" in r:
            lines.append(
                f"| {r['dataset']} | {r['split']} | 字面匹配 | "
                f"**{r['micro_f1']:.4f}** | {r['micro_precision']:.4f} | {r['micro_recall']:.4f} |"
            )
        elif "literal_micro_f1" in r:
            lines.append(
                f"| {r['dataset']} | {r['split']} | 字面匹配 | "
                f"{r['literal_micro_f1']:.4f} | {r['literal_precision']:.4f} | {r['literal_recall']:.4f} |"
            )
            lines.append(
                f"| {r['dataset']} | {r['split']} | 归一化匹配 | "
                f"**{r['normalized_micro_f1']:.4f}** | {r['normalized_precision']:.4f} | {r['normalized_recall']:.4f} |"
            )

    report_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n评估报告已保存至: {output_path}")
    print(report_text)
    return report_text

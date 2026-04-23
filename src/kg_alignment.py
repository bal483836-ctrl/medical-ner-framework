"""
Step 2: 图谱对齐与余弦相似度匹配模块 v3（修复版）
三级匹配策略：
  1. 精确匹配（score=1.0）：直接输出
  2. 高相似度（≥0.82）：接受归一化，扩展图谱值（原文锚定检查）
  3. 低相似度（<0.60）：转交 Step3 大模型验证

关键修复（v3.1）：
  IMCS Step2 之前把口语化词直接替换为标准词，导致 Step3 原文锚定检查失败。
  现在修改为：
    - step2_aligned_output：保留原词（供 Step3 原文锚定使用）
    - step2_normalized_map：记录原词→标准词的映射（供 Step4 归一化评估使用）
    - step2_norm_output：归一化后的标准词集合（供 Step4 直接读取）
  这样 Step3 锚定检查用原词，Step4 归一化评估用标准词，两者不冲突。
"""
import json
import os
import sys
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    OUTPUT_DIR, STEP2_PREFIX,
    HIGH_SIM_THRESHOLD, LOW_SIM_THRESHOLD,
)
from src.data_processor import clean_entity_list
from src.embedding_model import find_best_matches


def align_cmeee_split(
    items: List[Dict],
    vocab: List[str],
    output_path: str,
) -> List[Dict]:
    """
    CMeEE Step2：对 step1_enriched_output 中的实体做图谱对齐
    - 精确匹配：直接保留
    - 高相似度：接受归一化，但扩展词必须通过原文锚定检查
    - 低相似度：保留原词，交给 Step3 大模型验证
    """
    print(f"\n[Step2] CMeEE 图谱对齐，词汇表大小: {len(vocab)}")
    vocab_set = set(vocab)

    for item in items:
        text = item.get("text", "")
        raw_entities = clean_entity_list(
            item.get("step1_enriched_output", item.get("step1_raw_output", ""))
        )
        if not raw_entities:
            item["step2_aligned_output"] = ""
            continue

        # 精确匹配先处理
        exact_matched = []
        needs_embed   = []
        for ent in raw_entities:
            if ent in vocab_set:
                exact_matched.append(ent)
            else:
                needs_embed.append(ent)

        # 向量相似度匹配
        aligned_set = set(exact_matched)
        if needs_embed and vocab:
            matches = find_best_matches(needs_embed, vocab)
            for ent, (best_match, score, status) in zip(needs_embed, matches):
                if status in ("exact", "high"):
                    # 原文锚定检查：扩展词必须在原文中出现
                    if best_match in text:
                        aligned_set.add(best_match)
                    elif ent in text:
                        aligned_set.add(ent)
                    # 两者都不在原文中，丢弃
                else:
                    # 低相似度：保留原词，交给 Step3
                    if ent in text:
                        aligned_set.add(ent)

        item["step2_aligned_output"] = ",".join(list(aligned_set))

    _save_json(items, output_path)
    print(f"  ✅ CMeEE Step2 保存至: {output_path}")
    return items


def align_imcs_split(
    items: List[Dict],
    norm_vocab: List[str],
    output_path: str,
) -> List[Dict]:
    """
    IMCS Step2：将口语化实体对齐到官方 symptom_norm 标准词

    关键设计（修复版）：
      - step2_aligned_output：保留原词（口语化表述），供 Step3 原文锚定使用
      - step2_normalized_map：原词→标准词映射，供 Step4 归一化评估使用
      - step2_norm_output：标准词集合，供 Step4 直接读取

    这样 Step3 可以正确地用原词做原文锚定检查，
    Step4 再通过 step2_normalized_map 将最终结果归一化。
    """
    print(f"\n[Step2] IMCS 图谱对齐，归一化词汇表大小: {len(norm_vocab)}")
    norm_vocab_set = set(norm_vocab)

    for item in items:
        raw_entities = clean_entity_list(item.get("step1_raw_output", ""))
        if not raw_entities:
            item["step2_aligned_output"] = ""
            item["step2_normalized_map"] = {}
            item["step2_norm_output"]    = ""
            continue

        # 构建归一化映射表（原词 → 标准词）
        norm_map: Dict[str, str] = {}

        exact_matched = []
        needs_embed   = []
        for ent in raw_entities:
            if ent in norm_vocab_set:
                exact_matched.append(ent)
                norm_map[ent] = ent  # 精确匹配，标准词=原词
            else:
                needs_embed.append(ent)

        if needs_embed and norm_vocab:
            matches = find_best_matches(needs_embed, norm_vocab)
            for ent, (best_match, score, status) in zip(needs_embed, matches):
                if status in ("exact", "high"):
                    norm_map[ent] = best_match  # 高相似度，映射到标准词
                else:
                    norm_map[ent] = ent  # 低相似度，保留原词（后续 Step4 再尝试归一化）

        # step2_aligned_output：保留原词（供 Step3 原文锚定）
        item["step2_aligned_output"] = ",".join(raw_entities)

        # step2_normalized_map：原词→标准词映射（供 Step4 归一化评估）
        item["step2_normalized_map"] = norm_map

        # step2_norm_output：标准词集合（供 Step4 直接读取）
        norm_set = list(dict.fromkeys(norm_map.values()))
        item["step2_norm_output"] = ",".join(norm_set)

    _save_json(items, output_path)
    print(f"  ✅ IMCS Step2 保存至: {output_path}")
    return items


def _save_json(data: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

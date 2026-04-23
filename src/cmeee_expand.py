"""
Step 1.5: CMeEE 嵌套实体扩展模块 v3
CMeEE 数据集的特殊性：多粒度嵌套标注（"胃酸"和"胃酸分泌增加"都是 Gold 实体）
大模型倾向于只抽取最长词，导致短词漏报。本模块通过词汇表扫描补充漏报的嵌套实体。

策略：
  1. 长词（≥5字）：在原文中精确扫描，直接加入候选集
  2. 短词（<5字）：独立词检查（前后不是汉字/字母才算独立词），防止切割长词
  3. 合并词过滤：过滤含顿号/连字符的合并词
  4. 黑名单过滤：过滤通用词（感染、炎症、细胞等）
"""
import re
import sys
import os
from typing import List, Dict, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 通用词黑名单（这些词在原文中大量出现但通常不是独立实体）
COMMON_WORD_BLACKLIST = {
    "感染", "炎症", "细胞", "组织", "功能", "治疗", "检查", "手术",
    "症状", "疾病", "患者", "病人", "医生", "诊断", "病史", "病情",
    "方法", "结果", "分析", "研究", "报告", "数据", "指标", "水平",
    "正常", "异常", "增高", "降低", "升高", "下降", "明显", "显著",
    "可能", "考虑", "建议", "注意", "观察", "随访", "复查", "复诊",
    "分泌", "代谢", "循环", "系统", "器官", "部位", "区域", "范围",
    "程度", "类型", "形态", "大小", "数量", "比例", "浓度", "含量",
    "活动", "运动", "休息", "睡眠", "饮食", "营养", "发育", "生长",
}

# 合并词过滤正则（含顿号、破折号、连字符的词通常是两个词的合并）
MERGED_WORD_PATTERN = re.compile(r"[、—–-]|[，,]")

# 独立词边界检查（前后不是汉字/字母/数字才算独立）
BOUNDARY_PATTERN = re.compile(r"[\u4e00-\u9fff\w]")


def expand_cmeee_entities(
    text: str,
    step1_entities: List[str],
    vocab: List[str],
    long_min_len: int = 5,
) -> List[str]:
    """
    扩展 CMeEE 实体候选集
    Args:
        text: 原始文本
        step1_entities: Step1 大模型抽取的实体列表
        vocab: CMeEE 词汇表（从 train 集构建）
        long_min_len: 长词阈值（≥此长度直接全文扫描）
    Returns:
        扩展后的实体列表（去重）
    """
    if not text:
        return step1_entities

    candidate_set: Set[str] = set()

    # 先加入 Step1 的所有实体（基础）
    for ent in step1_entities:
        if ent and not _is_merged_word(ent):
            candidate_set.add(ent)

    # 词汇表扫描
    for word in vocab:
        if not word or _is_merged_word(word) or word in COMMON_WORD_BLACKLIST:
            continue

        if word not in text:
            continue

        if len(word) >= long_min_len:
            # 长词：直接加入
            candidate_set.add(word)
        else:
            # 短词：独立词检查
            if _is_independent_word(text, word):
                candidate_set.add(word)

    # 过滤：去除含顿号/合并词
    result = [w for w in candidate_set if not _is_merged_word(w)]

    # 过滤：去除纯英文字母片段（不完整的英文缩写）
    result = [w for w in result if not re.match(r"^[A-Za-z]{1,2}$", w)]

    # 去重并返回
    return list(dict.fromkeys(result))


def _is_merged_word(word: str) -> bool:
    """检查是否为合并词（含顿号/破折号/连字符）"""
    return bool(MERGED_WORD_PATTERN.search(word))


def _is_independent_word(text: str, word: str) -> bool:
    """
    检查词在原文中是否作为独立词出现
    独立词：前后字符不是汉字/字母/数字
    """
    idx = 0
    while True:
        pos = text.find(word, idx)
        if pos == -1:
            return False
        # 检查前边界
        if pos > 0 and BOUNDARY_PATTERN.match(text[pos - 1]):
            idx = pos + 1
            continue
        # 检查后边界
        end = pos + len(word)
        if end < len(text) and BOUNDARY_PATTERN.match(text[end]):
            idx = pos + 1
            continue
        # 前后都是边界，是独立词
        return True


def enrich_cmeee_step1(
    items: List[Dict],
    vocab: List[str],
    long_min_len: int = 5,
) -> List[Dict]:
    """
    批量处理 CMeEE 条目，在 step1_entities 基础上扩展嵌套实体
    结果存入 item["step1_enriched_output"]
    """
    for item in items:
        text = item.get("text", "")
        step1_ents = [e["entity"] for e in item.get("step1_entities", [])]
        # 也从 step1_raw_output 补充
        raw_names = [n.strip() for n in item.get("step1_raw_output", "").split(",") if n.strip()]
        all_step1 = list(dict.fromkeys(step1_ents + raw_names))

        expanded = expand_cmeee_entities(text, all_step1, vocab, long_min_len)
        item["step1_enriched_output"] = ",".join(expanded)

    return items

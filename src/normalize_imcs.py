"""
IMCS 6 级归一化级联（v4.2）—— 吸取 v21 normalize_imcs_step.py

层级（依次降级）：
  1) 已是标准词                            → already_standard
  2) 数据集自带口语映射 (ORAL_TO_NORM)     → oral_map
  3) 学到的口语映射 (build_imcs_dict)      → learned_oral
  4) 实体含某标准词（取最长子串）          → substring_contains
  5) 标准词含实体（多字反向子串）          → substring_contained
  6) 前缀剥离后再走 1-5                    → prefix_stripped

未匹配返回 None。LLM 兜底归一化不在此模块（可选阶段）。
"""
import os
import sys
from typing import Dict, List, Optional, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 复用旧版的口语->标准词手工映射
from src.kg_alignment import ORAL_TO_NORM


# 程度词 / 动词 / 部位前缀（吸取 v21 v20.8 的扩展清单）
_STRIP_PREFIXES = [
    "有点", "有些", "比较", "非常", "轻微", "严重", "持续", "反复",
    "偶尔", "经常", "一直", "总是", "总", "老", "突然", "稍微",
    "出现", "感觉", "感到", "觉得", "好像", "似乎",
    "宝宝", "孩子", "小孩", "他", "她", "我",
]


def _strip_prefix(entity: str) -> str:
    for p in _STRIP_PREFIXES:
        if entity.startswith(p) and len(entity) > len(p):
            return entity[len(p):]
    return entity


def normalize_one(
    entity: str,
    norm_vocab_set: Set[str],
    learned_oral_map: Optional[Dict[str, str]] = None,
) -> Dict:
    """单实体归一化，返回 {normed, method}"""
    ent = (entity or "").strip()
    if not ent:
        return {"normed": None, "method": "fail"}
    learned_oral_map = learned_oral_map or {}

    def _try(e: str) -> Optional[Dict]:
        # 1) 已是标准词
        if e in norm_vocab_set:
            return {"normed": e, "method": "already_standard"}
        # 2) 数据集口语映射
        if e in ORAL_TO_NORM:
            return {"normed": ORAL_TO_NORM[e], "method": "oral_map"}
        # 3) 学到的口语映射
        if e in learned_oral_map:
            return {"normed": learned_oral_map[e], "method": "learned_oral"}
        # 4) 子串：实体含某标准词（最长优先）
        best, best_len = None, 0
        for nv in norm_vocab_set:
            if nv in e and len(nv) > best_len:
                best, best_len = nv, len(nv)
        if best:
            return {"normed": best, "method": "substring_contains"}
        # 5) 反向子串：标准词含实体（实体 ≥2 字，避免误命中）
        if len(e) >= 2:
            for nv in norm_vocab_set:
                if e in nv:
                    return {"normed": nv, "method": "substring_contained"}
        return None

    # 第一遍直接走 1-5
    r = _try(ent)
    if r:
        return r
    # 6) 剥前缀再走一遍
    stripped = _strip_prefix(ent)
    if stripped != ent:
        r = _try(stripped)
        if r:
            r["method"] = "prefix+" + r["method"]
            return r
    return {"normed": None, "method": "fail"}


def normalize_list(
    entities: List[str],
    norm_vocab_set: Set[str],
    learned_oral_map: Optional[Dict[str, str]] = None,
) -> Dict:
    """
    返回：
      norm_map      : 原词 -> 标准词（失败者映射到自身）
      method_stats  : {method: count}
      failed        : 未能归一化的原词
    """
    norm_map = {}
    stats = {}
    failed = []
    for e in entities:
        r = normalize_one(e, norm_vocab_set, learned_oral_map)
        m = r["method"]
        stats[m] = stats.get(m, 0) + 1
        if r["normed"]:
            norm_map[e] = r["normed"]
        else:
            norm_map[e] = e
            failed.append(e)
    return {"norm_map": norm_map, "method_stats": stats, "failed": failed}

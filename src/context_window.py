"""
动态语境窗口（v4.1）—— 吸取 run_extraction.py 的中心锚定 + 双引擎评分

策略：
  1. 中心锚定：以实体最靠近文本几何中心的位置为锚
  2. 初始硬窗口：64 字
  3. 双引擎前瞻评分：BGE 语义相似度 × 10 + spaCy 句法依存 × 5 + 否定关键词 × 3 - 距离惩罚
  4. 贪心扩张：分数 ≥ 0.65 才向那个方向再吃 64 字，否则停
  5. 最大窗口 512 字

对话场景（IMCS）：直接走轮次切片（实体所在轮 ± K）+ 主诉。
spaCy 不可用时退化为纯 BGE 评分。
"""
import os
import re
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import CONTEXT_WINDOW_CHARS, CONTEXT_DIALOGUE_TURNS

# ==================== 双引擎评分参数 ====================
INITIAL_WIN     = 64
LOOKAHEAD_WIN   = 128
STEP_SIZE       = 64
MAX_TOTAL_LEN   = 512
SEMANTIC_THRESHOLD = 0.65

SEMANTIC_WEIGHT  = 10.0
SYNTACTIC_WEIGHT = 5.0
KEYWORD_BONUS    = 3.0
DISTANCE_PENALTY = 0.5

# 断言/否定线索词（句法分析时聚焦这些）
ASSERTION_CUES = [
    "未见", "否认", "排除", "无", "不伴", "未闻及", "没有", "未发现",
    "查体", "诊断", "显示", "提示", "既往", "可能", "疑似", "不排除", "考虑",
]


class _Engines:
    _sem = None
    _syn = None
    _syn_loaded = False

    @classmethod
    def get_sem(cls):
        if cls._sem is None:
            from src.embedding_model import get_embedding_model  # noqa
            cls._sem = True   # 我们用 encode_texts 而非 model 句柄
        return cls._sem

    @classmethod
    def get_syn(cls):
        if cls._syn_loaded:
            return cls._syn
        cls._syn_loaded = True
        try:
            import spacy
            cls._syn = spacy.load("zh_core_web_sm", disable=["ner"])
            print("  [Ctx] spaCy 加载成功，启用句法依存评分")
        except Exception as e:
            print(f"  [Ctx] spaCy 不可用 ({e})，仅用 BGE 评分")
            cls._syn = None
        return cls._syn


def _bge_cosine(a: str, b: str) -> float:
    from src.embedding_model import encode_texts, cosine_similarity_matrix
    v = encode_texts([a, b], is_query=False)
    return float(cosine_similarity_matrix(v[0:1], v[1:2])[0][0])


def _dep_score(nlp, target: str, segment: str) -> float:
    """spaCy 句法依存：实体与断言线索词在依存树上的距离评分。"""
    if nlp is None or not segment.strip():
        return 0.0
    if not any(cue in segment for cue in ASSERTION_CUES):
        return 0.0
    try:
        doc = nlp(f"{target}，{segment}")
    except Exception:
        return 0.0

    tgt_tok, cues = None, []
    for tok in doc:
        if target in tok.text or tok.text in target:
            if tgt_tok is None: tgt_tok = tok
        if tok.text in ASSERTION_CUES:
            cues.append(tok)
    if tgt_tok is None or not cues:
        return 0.0

    best = 0.0
    for cue in cues:
        score = 0.0
        if cue in tgt_tok.ancestors:
            d, t = 0, tgt_tok
            while t != cue and t != t.head and d < 5:
                t = t.head; d += 1
            score = max(0, 1.0 - d * 0.2)
        elif cue.head == tgt_tok.head:
            score = 0.8
        best = max(best, score)
    return best


def _hybrid_score(target: str, segment: str, step: int) -> float:
    if not segment.strip():
        return 0.0
    nlp = _Engines.get_syn()
    sem = _bge_cosine(target, segment)
    syn = _dep_score(nlp, target, segment)
    kw  = KEYWORD_BONUS if any(c in segment for c in ASSERTION_CUES) else 0.0
    if syn >= 0.8:
        return 100.0    # 极强句法关系，强制吃进
    return sem * SEMANTIC_WEIGHT + syn * SYNTACTIC_WEIGHT + kw - step * DISTANCE_PENALTY


# ==================== 文本：动态窗口 ====================

def context_from_text(text: str, entity: str,
                      use_hybrid: bool = True,
                      window: int = CONTEXT_WINDOW_CHARS) -> List[Dict]:
    """普通文本：中心锚定 + 双引擎扩张。窗口太小时不开扩张。"""
    if not entity or entity not in text:
        return []
    positions = [m.start() for m in re.finditer(re.escape(entity), text)]
    if not positions:
        return []
    # 简易版：每个出现位置生成一份语境
    out = []
    for start in positions:
        end = start + len(entity)
        if not use_hybrid or len(text) <= INITIAL_WIN:
            # 简单 ±window
            s = max(0, start - window)
            e = min(len(text), end + window)
            ctx = text[s:e]
        else:
            ctx = _dynamic_window(text, start, end, entity)
        out.append({
            "entity": entity, "context": ctx,
            "source": "text", "loc": {"start": start, "end": end},
        })
    return out


def _dynamic_window(text: str, start: int, end: int, entity: str) -> str:
    """中心锚定 + 贪心扩张。"""
    anchor_c = (start + end) // 2
    half = INITIAL_WIN // 2
    L = max(0, anchor_c - half)
    R = min(len(text), anchor_c + half)
    step = 0
    while (R - L) < MAX_TOTAL_LEN:
        cands = []
        if L > 0:
            seg = text[max(0, L - LOOKAHEAD_WIN): L]
            cands.append(("L", _hybrid_score(entity, seg, step + 1), seg))
        if R < len(text):
            seg = text[R: min(len(text), R + LOOKAHEAD_WIN)]
            cands.append(("R", _hybrid_score(entity, seg, step + 1), seg))
        if not cands: break
        direction, score, _ = max(cands, key=lambda x: x[1])
        if score < SEMANTIC_THRESHOLD: break
        if direction == "L":
            L = max(0, L - STEP_SIZE)
        else:
            R = min(len(text), R + STEP_SIZE)
        step += 1
    return text[L:R].strip().strip("，,。.")


# ==================== 对话：轮次切片 ====================

def context_from_dialogue(dialogue: List[Dict], entity: str,
                          self_report: str = "",
                          turns: int = CONTEXT_DIALOGUE_TURNS) -> List[Dict]:
    out = []
    if self_report and entity in self_report:
        out.append({
            "entity": entity,
            "context": f"主诉: {self_report}",
            "source": "dialogue", "loc": {"turn_index": -1},
        })
    for ti, turn in enumerate(dialogue):
        sent = turn.get("sentence", "")
        if entity not in sent:
            continue
        lo = max(0, ti - turns)
        hi = min(len(dialogue), ti + turns + 1)
        lines = []
        if lo == 0 and self_report:
            lines.append(f"[主诉] {self_report}")
        for tj in range(lo, hi):
            t = dialogue[tj]
            tag = "*" if tj == ti else " "
            lines.append(f"{tag}{t.get('speaker','?')}: {t.get('sentence','')}")
        out.append({
            "entity": entity,
            "context": "\n".join(lines),
            "source": "dialogue",
            "loc": {"turn_index": ti, "speaker": turn.get("speaker", "")},
        })
    return out

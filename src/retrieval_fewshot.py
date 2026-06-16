"""
检索式动态 few-shot（kNN / GPT-NER 思路）— v4.7 新增

动机（来自 smoke200 错误归因）：
  - 原流水线用「全局固定 few-shot」：所有句子共用同一组示例，无法贴合具体语境。
  - CMeEE 召回仅 0.39，IMCS「感冒/病毒感染/支气管炎」等闭集诊断词漏检 50+ 次。
  零样本 LLM-NER 最有效的提升手段是「为每条输入检索最相似的带标注训练样本」
  （Wang et al. GPT-NER 2023；retrieval-augmented ICL）。本模块即实现该机制：
    ① 用 BGE 把 train 样本编码并缓存到磁盘
    ② 对每条待抽取文本，检索 top-K 最相似的 train 样本
    ③ 把这些样本的 gold 标注格式化进 prompt

所有功能均可通过 config 开关关闭，BGE 不可用时自动回退到全局 few-shot。
"""
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    OUTPUT_DIR, FEW_SHOT_COUNT, IMCS_TARGET_SYMPTOM_TYPES,
)
from src.data_processor import (
    load_cmeee, load_imcs,
    extract_cmeee_gold_names,
)

_CACHE_DIR = os.path.join(OUTPUT_DIR, "knowledge", "fewshot_cache")


# ==================== 闭集词表（IMCS symptom_norm）====================

def load_symptom_norm_vocab() -> List[str]:
    """加载 IMCS 的 331 个 symptom_norm 闭集规范词（data/symptom_norm.csv）。

    IMCS-V2 的 gold 全部来自这个闭集，其中「感冒/病毒感染/支气管炎/肺炎/
    中等度热/低热/高热」等诊断/感染/发热分级词也算 symptom_norm —— 这正是
    原 prompt（只抽症状/体征、排除病因诊断）漏检的根因。
    """
    from config.config import IMCS_NORM_DICT_PATH
    path = IMCS_NORM_DICT_PATH
    vocab: List[str] = []
    if not os.path.exists(path):
        return vocab
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip().strip(",").strip()
            if w and w.lower() != "norm":
                vocab.append(w)
    # 长词在前，便于做最长匹配兜底召回
    vocab = sorted(set(vocab), key=lambda x: -len(x))
    return vocab


# 否定 / 排除 / 预防语境前缀：闭集召回时若紧邻这些词则跳过，避免误召
_NEG_PREFIXES = (
    "不", "没", "无", "非", "未", "否认", "排除", "预防", "防止",
    "不是", "不像", "不会", "防", "没有", "查不出", "不考虑",
)


def imcs_vocab_recall(text: str, vocab: List[str]) -> List[str]:
    """闭集兜底召回：扫描整段对话，命中 symptom_norm 词且非否定语境则召回。

    只召回 gold 闭集内的词，因此召回的都是「合法答案候选」，精度风险有限；
    再叠加轻量否定过滤进一步降误召。
    """
    hits: List[str] = []
    for w in vocab:
        start = 0
        while True:
            i = text.find(w, start)
            if i < 0:
                break
            # 否定语境检查：命中词前 3 个字内出现否定前缀则跳过该处
            prefix = text[max(0, i - 3): i]
            if any(neg in prefix for neg in _NEG_PREFIXES):
                start = i + len(w)
                continue
            hits.append(w)
            break  # 该词命中一次即可
        # while end
    # 去重保序
    seen, out = set(), []
    for w in hits:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


# ==================== 检索器 ====================

class RetrievalFewShot:
    """基于 BGE 的 kNN few-shot 检索器（带磁盘缓存）。

    examples: List[(query_text, formatted_block)]
      - query_text  用来编码/检索的文本（CMeEE=原文，IMCS=单轮发言）
      - formatted_block  注入 prompt 的格式化示例块（含 gold 标注）
    """

    def __init__(self, name: str, examples: List[Tuple[str, str]]):
        self.name = name
        self.queries = [q for q, _ in examples]
        self.blocks = [b for _, b in examples]
        self.vecs: Optional[np.ndarray] = None
        # 查询向量缓存：text -> vec。预热后检索全程走 numpy，无需在 LLM 循环里调 BGE。
        self.qcache: Dict[str, np.ndarray] = {}

    # ---- 缓存键：训练样本文本内容哈希 ----
    def _cache_key(self) -> str:
        h = hashlib.md5()
        h.update(("".join(self.queries)).encode("utf-8"))
        return f"{self.name}_{len(self.queries)}_{h.hexdigest()[:10]}"

    def build(self) -> "RetrievalFewShot":
        """编码（带磁盘缓存）。BGE 不可用时抛异常，由调用方回退。"""
        if not self.queries:
            self.vecs = np.zeros((0, 1), dtype=np.float32)
            return self
        os.makedirs(_CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(_CACHE_DIR, self._cache_key() + ".npy")
        if os.path.exists(cache_path):
            try:
                self.vecs = np.load(cache_path)
                print(f"  [Retrieval] {self.name}: 命中向量缓存 {self.vecs.shape}")
                return self
            except Exception:
                pass
        from src.embedding_model import encode_texts
        print(f"  [Retrieval] {self.name}: 编码 {len(self.queries)} 条 train 示例...")
        self.vecs = encode_texts(self.queries, is_query=False)
        try:
            np.save(cache_path, self.vecs)
        except Exception:
            pass
        return self

    def precompute_queries(self, texts: List[str]) -> None:
        """一次性批量编码所有待检索文本（在 LLM 加载前调用），之后检索纯 numpy。

        这是避免 OOM / 加速的关键：原先每条输入在 LLM 循环里单独调 BGE，
        既慢（GPU 串行）又让 BGE 与 vLLM 同时占显存。
        """
        uniq = list(dict.fromkeys(t for t in texts if t))
        uniq = [t for t in uniq if t not in self.qcache]
        if not uniq:
            return
        from src.embedding_model import encode_texts
        print(f"  [Retrieval] {self.name}: 预编码 {len(uniq)} 条待检索文本...")
        vecs = encode_texts(uniq, is_query=True)
        for t, v in zip(uniq, vecs):
            self.qcache[t] = v

    def retrieve_block(self, text: str, k: int = FEW_SHOT_COUNT) -> str:
        """检索 top-k 相似示例，拼成 few-shot 字符串。失败回退空串。"""
        if self.vecs is None or len(self.queries) == 0 or not text:
            return ""
        qv = self.qcache.get(text)
        if qv is None:
            # 未预热的回退路径（慢）：单条编码
            try:
                from src.embedding_model import encode_texts
                qv = encode_texts([text], is_query=True)[0]
            except Exception:
                return ""
        try:
            sims = self.vecs @ qv                       # (N,)
            top = np.argsort(-sims)[:k]
            lines = []
            for n, idx in enumerate(top, 1):
                lines.append(self.blocks[idx].replace("{i}", str(n)))
            return "\n".join(lines)
        except Exception:
            return ""


# ==================== 构建 CMeEE / IMCS 检索器 ====================

def build_cmeee_retriever(max_examples: int = 3000) -> Optional[RetrievalFewShot]:
    """从 CMeEE train 构建检索器：示例 = 原文 → gold 实体名。"""
    from config.config import RETRIEVAL_EXAMPLE_MAXLEN
    try:
        data = load_cmeee("train")
    except Exception as e:
        print(f"  [Retrieval] CMeEE 无法加载 train，跳过检索 few-shot：{e}")
        return None
    examples: List[Tuple[str, str]] = []
    for item in data:
        text = (item.get("text") or "").strip()
        if not text or not (4 <= len(text) <= 200):
            continue
        names = extract_cmeee_gold_names(item)
        if not names:
            continue
        # 截断展示文本，控制 prompt 长度（检索仍用全文编码）
        shown = text if len(text) <= RETRIEVAL_EXAMPLE_MAXLEN else text[:RETRIEVAL_EXAMPLE_MAXLEN] + "…"
        block = f"示例{{i}}：\n文本：{shown}\n实体：{', '.join(names)}\n"
        examples.append((text, block))
        if len(examples) >= max_examples:
            break
    if not examples:
        return None
    try:
        return RetrievalFewShot("cmeee", examples).build()
    except Exception as e:
        print(f"  [Retrieval] CMeEE 编码失败，回退全局 few-shot：{e}")
        return None


def _imcs_turn_norms(turn: Dict) -> List[str]:
    """取单轮 gold symptom_norm（type 1/2）。"""
    norms = []
    for ner_item in turn.get("ner", []) or []:
        if not isinstance(ner_item, dict):
            continue
        stype = str(ner_item.get("symptom_type", ner_item.get("type", -1)))
        norm = (ner_item.get("symptom_norm") or ner_item.get("norm") or "").strip()
        if stype in IMCS_TARGET_SYMPTOM_TYPES and norm and norm.lower() != "null":
            norms.append(norm)
    # 去重保序
    seen, out = set(), []
    for w in norms:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def build_imcs_retriever(max_examples: int = 4000) -> Optional[RetrievalFewShot]:
    """从 IMCS train 构建 turn 级检索器：示例 = 单轮发言 → gold 标准症状词。

    刻意包含医生轮次里出现 gold 诊断/感染词的样本，教会模型抽取
    「感冒/病毒感染/支气管炎」这类闭集症状词。
    """
    try:
        data = load_imcs("train")
    except Exception as e:
        print(f"  [Retrieval] IMCS 无法加载 train，跳过检索 few-shot：{e}")
        return None
    examples: List[Tuple[str, str]] = []
    for item in data:
        for turn in item.get("dialogue", []) or []:
            sent = (turn.get("sentence") or "").strip()
            if not sent or len(sent) > 120:
                continue
            norms = _imcs_turn_norms(turn)
            if not norms:
                continue  # 只收正样本，示范「该抽什么」
            block = f"示例{{i}}：\n发言：{sent}\n症状（标准词）：{', '.join(norms)}\n"
            examples.append((sent, block))
            if len(examples) >= max_examples:
                break
        if len(examples) >= max_examples:
            break
    if not examples:
        return None
    try:
        return RetrievalFewShot("imcs", examples).build()
    except Exception as e:
        print(f"  [Retrieval] IMCS 编码失败，回退全局 few-shot：{e}")
        return None


# ==================== 查询预热（在 LLM 加载前批量编码）====================

def prewarm_cmeee_queries(retriever: Optional[RetrievalFewShot],
                          splits: List[str], limit: Optional[int] = None) -> None:
    if retriever is None:
        return
    texts: List[str] = []
    for sp in splits:
        try:
            data = load_cmeee(sp)
        except Exception:
            continue
        if limit:
            data = data[:limit]
        texts.extend((it.get("text") or "") for it in data)
    retriever.precompute_queries(texts)


def prewarm_imcs_queries(retriever: Optional[RetrievalFewShot],
                         splits: List[str], limit: Optional[int] = None) -> None:
    if retriever is None:
        return
    sents: List[str] = []
    for sp in splits:
        try:
            data = load_imcs(sp)
        except Exception:
            continue
        if limit:
            data = data[:limit]
        for it in data:
            for turn in it.get("dialogue", []) or []:
                # 用与 extract_imcs_split 完全一致的原始 sentence 作为缓存键（勿 strip）
                s = turn.get("sentence", "")
                if s:
                    sents.append(s)
    retriever.precompute_queries(sents)

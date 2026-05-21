"""
知识图谱模块
- 加载外部 KG 文件（用户提供路径，缺失时降级为训练集词典）
- 提供：①向量相似度过滤  ②实体语义扩展（同义词/上位/相关概念）

KG JSON 期望格式（任一即可）：
  A) 平铺列表：["实体1", "实体2", ...]
  B) 字典：{"实体": {"synonyms":[...], "hypernyms":[...], "related":[...]}, ...}

调用接口：
  kg = load_kg()
  ok = kg.filter_by_similarity(entities, threshold=0.80)
  expansions = kg.expand(entity)
"""
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import KG_PATH, HIGH_SIM_THRESHOLD
from src.embedding_model import encode_texts, cosine_similarity_matrix
from src.data_processor import build_imcs_norm_vocab

import numpy as np


class KnowledgeGraph:
    def __init__(self, entries: Dict[str, Dict], source: str):
        self.entries = entries         # name -> dict(synonyms, hypernyms, related)
        self.names = list(entries.keys())
        self.source = source
        self._cache_vecs: Optional[np.ndarray] = None
        # IMCS 标准词集合（不参与相似度过滤）
        self._norm_set = set(build_imcs_norm_vocab() or [])

    # ---------- 1) 相似度过滤 ----------
    def _vectors(self) -> np.ndarray:
        if self._cache_vecs is None:
            self._cache_vecs = encode_texts(self.names, is_query=False)
        return self._cache_vecs

    def filter_by_similarity(self, entities: List[str],
                             threshold: float = HIGH_SIM_THRESHOLD,
                             skip_normalized: bool = True) -> List[str]:
        """保留余弦相似度 >= threshold 的实体；IMCS 标准词直接放行。"""
        if not entities:
            return []
        # 标准词直通
        kept, to_check = [], []
        for e in entities:
            if skip_normalized and e in self._norm_set:
                kept.append(e)
            elif e in self.entries:
                kept.append(e)   # KG 精确命中
            else:
                to_check.append(e)
        if not to_check or not self.names:
            return list(dict.fromkeys(kept))
        qv = encode_texts(to_check, is_query=True)
        sims = cosine_similarity_matrix(qv, self._vectors())
        for i, e in enumerate(to_check):
            if float(sims[i].max()) >= threshold:
                kept.append(e)
        return list(dict.fromkeys(kept))

    # ---------- 2) 语义扩展 ----------
    def expand(self, entity: str, topk: int = 5) -> Dict[str, List[str]]:
        """返回同义、上位、相关概念。若 KG 没有显式字段，则用向量 topk 近邻。"""
        info = self.entries.get(entity, {})
        result = {
            "synonyms":  list(info.get("synonyms", [])),
            "hypernyms": list(info.get("hypernyms", [])),
            "related":   list(info.get("related", [])),
        }
        if not any(result.values()) and self.names:
            qv = encode_texts([entity], is_query=True)
            sims = cosine_similarity_matrix(qv, self._vectors())[0]
            top_idx = np.argsort(-sims)[:topk + 1]
            neighbors = [self.names[i] for i in top_idx
                         if self.names[i] != entity][:topk]
            result["related"] = neighbors
        return result


def _parse_kg_file(path: str) -> Dict[str, Dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return {str(k).strip(): {} for k in raw if str(k).strip()}
    if isinstance(raw, dict):
        # 已是 entity->info 字典
        out = {}
        for k, v in raw.items():
            k = str(k).strip()
            if not k:
                continue
            out[k] = v if isinstance(v, dict) else {}
        return out
    raise ValueError(f"Unsupported KG schema: {type(raw)}")


def _fallback_kg() -> Dict[str, Dict]:
    """缺失外部 KG 时：用 IMCS 标准词 + CMeEE train 词表组合。"""
    print("  [KG] 外部 KG 文件不存在，降级使用 train 词表。")
    entries = {}
    for w in build_imcs_norm_vocab() or []:
        entries[w] = {}
    try:
        from src.data_processor import load_cmeee, build_cmeee_entity_vocab
        for w in build_cmeee_entity_vocab(load_cmeee("train")):
            entries.setdefault(w, {})
    except Exception as e:
        print(f"  [KG] CMeEE 词表降级失败: {e}")
    return entries


_KG_SINGLETON: Optional[KnowledgeGraph] = None


def load_kg(path: str = None) -> KnowledgeGraph:
    global _KG_SINGLETON
    if _KG_SINGLETON is not None:
        return _KG_SINGLETON
    path = path or KG_PATH
    if path and os.path.exists(path):
        entries = _parse_kg_file(path)
        print(f"  [KG] 已加载外部 KG: {path}（{len(entries)} 节点）")
        src = path
    else:
        entries = _fallback_kg()
        src = "fallback"
    _KG_SINGLETON = KnowledgeGraph(entries, source=src)
    return _KG_SINGLETON

"""
知识图谱模块（v4.1）
直接加载 data/entities_dict.txt + data/triples.txt（CMKG 风格的真实医学 KG）

数据来源（用户提供）：
  entities_dict.txt  ~60K 标准医学实体（每行一个词）
  triples.txt        ~354K 三元组 "头,关系,尾"（disease_has_symptom 等）

对外接口：
  kg = load_kg()
  kept = kg.filter_by_similarity(entities, threshold=0.80)
  exp  = kg.expand(entity)   # {synonyms, hypernyms, related, kg_facts}
  fact = kg.kg_knowledge(entity)   # 给分类器用的浓缩字符串
"""
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    KG_PATH, KG_DICT_PATH, KG_TRIPLES_PATH, HIGH_SIM_THRESHOLD,
)
from src.embedding_model import encode_texts, cosine_similarity_matrix
from src.data_processor import build_imcs_norm_vocab


# 关系名 → 中文角色（用于扩展时分类）
_REL_BUCKETS = {
    "synonyms":  ["synonym", "同义词", "alias", "alias_of", "别名"],
    "hypernyms": ["is_a", "subclass_of", "上位", "属于", "category_of"],
    "related":   [],   # 其他都进 related
}


def _classify_rel(rel: str) -> str:
    rl = rel.lower()
    for bucket, keys in _REL_BUCKETS.items():
        if any(k in rl for k in keys):
            return bucket
    return "related"


class KnowledgeGraph:
    def __init__(self, nodes: set, relations: Dict[str, List[tuple]], source: str):
        # nodes: 所有节点
        # relations: head -> [(rel, tail), ...]
        self.nodes = nodes
        self.relations = relations
        self.names = list(nodes)
        self.source = source
        self._cache_vecs: Optional[np.ndarray] = None
        self._norm_set = set(build_imcs_norm_vocab() or [])

    # ---------- 向量缓存 ----------
    def _vectors(self) -> np.ndarray:
        if self._cache_vecs is None:
            print(f"  [KG] 编码 {len(self.names)} 个节点向量…")
            self._cache_vecs = encode_texts(self.names, is_query=False)
        return self._cache_vecs

    # ---------- ① 相似度过滤 ----------
    def filter_by_similarity(self, entities: List[str],
                             threshold: float = HIGH_SIM_THRESHOLD,
                             skip_normalized: bool = True) -> List[str]:
        if not entities:
            return []
        kept, to_check = [], []
        for e in entities:
            if skip_normalized and e in self._norm_set:
                kept.append(e)
            elif e in self.nodes:
                kept.append(e)
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

    # ---------- ② 语义扩展 ----------
    def expand(self, entity: str, topk: int = 5) -> Dict[str, List[str]]:
        """同义/上位/相关 + 直接 KG 三元组事实。"""
        result = {"synonyms": [], "hypernyms": [], "related": [], "kg_facts": []}
        triples = self.relations.get(entity, [])
        for rel, tail in triples[:30]:
            bucket = _classify_rel(rel)
            if tail not in result[bucket]:
                result[bucket].append(tail)
            result["kg_facts"].append(f"{rel}:{tail}")
        # 向量近邻补 related
        if not result["related"] and self.names:
            qv = encode_texts([entity], is_query=True)
            sims = cosine_similarity_matrix(qv, self._vectors())[0]
            idx = np.argsort(-sims)[:topk + 1]
            for i in idx:
                w = self.names[i]
                if w != entity and w not in result["related"]:
                    result["related"].append(w)
                if len(result["related"]) >= topk:
                    break
        # 截断
        for k in ("synonyms", "hypernyms", "related"):
            result[k] = result[k][:topk]
        result["kg_facts"] = result["kg_facts"][:5]
        return result

    def kg_knowledge(self, entity: str) -> str:
        """给分类器/语境用的扁平知识串。"""
        info = self.expand(entity, topk=5)
        parts = []
        if entity in self.nodes:
            parts.append("[KG认证词]")
        for k_zh, k in (("同义", "synonyms"), ("上位", "hypernyms"), ("相关", "related")):
            if info[k]:
                parts.append(f"{k_zh}:{','.join(info[k][:3])}")
        if info["kg_facts"]:
            parts.append(f"事实:{';'.join(info['kg_facts'][:3])}")
        return " | ".join(parts) if parts else "无关联知识"


# ==================== 文件加载 ====================

def _load_dict_txt(path: str) -> set:
    nodes = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                nodes.add(w)
    return nodes


def _load_triples_txt(path: str):
    nodes = set()
    rels = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split(",")
            if len(parts) < 2:
                continue
            head = parts[0].strip()
            rel  = parts[1].strip() if len(parts) >= 3 else "related_to"
            tail = parts[-1].strip()
            if not head or not tail:
                continue
            nodes.add(head); nodes.add(tail)
            rels[head].append((rel, tail))
    return nodes, rels


def _load_json_kg(path: str):
    """兼容 v4.0 的 JSON 格式。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    nodes = set()
    rels = defaultdict(list)
    if isinstance(raw, list):
        for w in raw:
            w = str(w).strip()
            if w: nodes.add(w)
    elif isinstance(raw, dict):
        for k, v in raw.items():
            k = str(k).strip()
            if not k: continue
            nodes.add(k)
            if isinstance(v, dict):
                for tail in v.get("synonyms", []):
                    rels[k].append(("synonym", tail)); nodes.add(tail)
                for tail in v.get("hypernyms", []):
                    rels[k].append(("is_a", tail)); nodes.add(tail)
                for tail in v.get("related", []):
                    rels[k].append(("related_to", tail)); nodes.add(tail)
    return nodes, rels


_SINGLETON: Optional[KnowledgeGraph] = None


def load_kg() -> KnowledgeGraph:
    global _SINGLETON
    if _SINGLETON is not None:
        return _SINGLETON

    nodes, rels = set(), defaultdict(list)
    source_parts = []

    # 优先：entities_dict + triples（真 KG）
    if os.path.exists(KG_DICT_PATH):
        n = _load_dict_txt(KG_DICT_PATH)
        nodes.update(n)
        source_parts.append(f"dict({len(n)})")
        print(f"  [KG] entities_dict: {KG_DICT_PATH} → {len(n)} 节点")

    if os.path.exists(KG_TRIPLES_PATH):
        n2, r2 = _load_triples_txt(KG_TRIPLES_PATH)
        nodes.update(n2)
        for h, ts in r2.items():
            rels[h].extend(ts)
        source_parts.append(f"triples({sum(len(v) for v in r2.values())})")
        print(f"  [KG] triples: {KG_TRIPLES_PATH} → {len(n2)} 节点, {sum(len(v) for v in r2.values())} 关系")

    # 次选：JSON KG
    if not nodes and KG_PATH and os.path.exists(KG_PATH):
        n3, r3 = _load_json_kg(KG_PATH)
        nodes.update(n3)
        for h, ts in r3.items():
            rels[h].extend(ts)
        source_parts.append(f"json({len(n3)})")
        print(f"  [KG] JSON: {KG_PATH}")

    # 兜底：训练集词表
    if not nodes:
        print("  [KG] 外部 KG 不存在，降级训练集词表")
        try:
            for w in (build_imcs_norm_vocab() or []):
                nodes.add(w)
            from src.data_processor import load_cmeee, build_cmeee_entity_vocab
            for w in build_cmeee_entity_vocab(load_cmeee("train")):
                nodes.add(w)
            source_parts.append(f"fallback({len(nodes)})")
        except Exception as e:
            print(f"  [KG] 降级失败: {e}")

    src = " + ".join(source_parts) if source_parts else "empty"
    _SINGLETON = KnowledgeGraph(nodes, rels, source=src)
    print(f"  [KG] 总计 {len(nodes)} 节点 / {sum(len(v) for v in rels.values())} 关系 / source={src}")
    return _SINGLETON

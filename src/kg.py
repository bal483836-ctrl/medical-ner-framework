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
    "synonyms":  ["synonym", "同义词", "alias", "alias_of", "别名", "sameas"],
    "hypernyms": ["is_a", "subclass_of", "上位", "属于", "category_of",
                  "department_belong_department"],
    "related":   [],   # 其他都进 related
}

# 需要建反向索引的"疾病为头"的关系
# 反向后语义：症状 → 可能疾病；检查 → 可能疾病；药物 → 可能疾病；科室 → 该科室疾病
_INVERSE_REL_MAP = {
    "disease_has_symptom":    "symptom_indicates_disease",
    "disease_need_check":     "check_for_disease",
    "disease_recommand_drug": "drug_for_disease",
    "disease_common_drug":    "drug_for_disease",
    "disease_need_treatment": "treatment_for_disease",
    "disease_belong_department": "department_treats_disease",
    "disease_acompany_disease":  "disease_accompanies",   # 对称关系
    "disease_eat_food":       "food_for_disease",
    "disease_recommand_food": "food_for_disease",
    "disease_noteat_food":    "food_avoided_in_disease",
}


def _classify_rel(rel: str) -> str:
    rl = rel.lower()
    for bucket, keys in _REL_BUCKETS.items():
        if any(k in rl for k in keys):
            return bucket
    return "related"


class KnowledgeGraph:
    def __init__(self, nodes: set, relations: Dict[str, List[tuple]],
                 inverse_relations: Dict[str, List[tuple]] = None,
                 source: str = ""):
        # nodes: 所有节点
        # relations: head -> [(rel, tail), ...]
        # inverse_relations: tail -> [(inverse_rel, head), ...]
        #   主要用于"症状 → 可能疾病"等反向查询
        self.nodes = nodes
        self.relations = relations
        self.inverse_relations = inverse_relations or {}
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

    # ---------- ② 语义扩展（含反向索引）----------
    def expand(self, entity: str, topk: int = 5) -> Dict[str, List[str]]:
        """
        正向三元组（疾病→症状/检查/药物等）→ synonyms/hypernyms/related/kg_facts
        反向三元组（症状→可能疾病 等）                → inverse_facts / possible_diseases
        """
        result = {
            "synonyms": [], "hypernyms": [], "related": [],
            "kg_facts": [], "inverse_facts": [], "possible_diseases": [],
        }

        # 正向：entity 作头节点的三元组
        for rel, tail in self.relations.get(entity, [])[:30]:
            bucket = _classify_rel(rel)
            if tail not in result[bucket]:
                result[bucket].append(tail)
            result["kg_facts"].append(f"{rel}:{tail}")

        # 反向：entity 作尾节点的三元组（如 entity 是症状，head 是疾病）
        for inv_rel, head in self.inverse_relations.get(entity, [])[:30]:
            result["inverse_facts"].append(f"{inv_rel}:{head}")
            # 凡是反向后头节点是"疾病"角色的（symptom_indicates_disease 等），
            # 都收进 possible_diseases，给"知识事实"判别用
            if inv_rel in (
                "symptom_indicates_disease",
                "check_for_disease",
                "drug_for_disease",
                "treatment_for_disease",
                "disease_accompanies",
            ):
                if head not in result["possible_diseases"]:
                    result["possible_diseases"].append(head)

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
        result["kg_facts"]          = result["kg_facts"][:5]
        result["inverse_facts"]     = result["inverse_facts"][:8]
        result["possible_diseases"] = result["possible_diseases"][:8]
        return result

    def possible_diseases(self, entity: str, topk: int = 8) -> List[str]:
        """快捷接口：给定症状/检查/药物，返回 KG 中关联的可能疾病列表。"""
        diseases = []
        for inv_rel, head in self.inverse_relations.get(entity, []):
            if inv_rel in (
                "symptom_indicates_disease",
                "check_for_disease",
                "drug_for_disease",
                "treatment_for_disease",
                "disease_accompanies",
            ):
                if head not in diseases:
                    diseases.append(head)
            if len(diseases) >= topk:
                break
        return diseases

    def kg_knowledge(self, entity: str) -> str:
        """给分类器/断言 prompt 用的扁平知识串。"""
        info = self.expand(entity, topk=5)
        parts = []
        if entity in self.nodes:
            parts.append("[KG认证词]")
        for k_zh, k in (("同义", "synonyms"), ("上位", "hypernyms"), ("相关", "related")):
            if info[k]:
                parts.append(f"{k_zh}:{','.join(info[k][:3])}")
        if info["possible_diseases"]:
            # 给"知识事实"类判别提供强信号：该实体是否常被讨论为某些疾病的症状/检查/药物
            parts.append(f"可能关联疾病:{','.join(info['possible_diseases'][:5])}")
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
    """
    返回 (nodes, forward_rels, inverse_rels)
    forward_rels: head -> [(rel, tail), ...]
    inverse_rels: tail -> [(inverse_rel_name, head), ...]
                  仅对 _INVERSE_REL_MAP 中的关系构建（症状→可能疾病等）
    """
    nodes = set()
    rels  = defaultdict(list)
    inv   = defaultdict(list)
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
            # 构建反向索引（仅疾病为头的关系）
            if rel in _INVERSE_REL_MAP:
                inv_rel = _INVERSE_REL_MAP[rel]
                inv[tail].append((inv_rel, head))
    return nodes, rels, inv


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

    inv_rels = defaultdict(list)
    if os.path.exists(KG_TRIPLES_PATH):
        n2, r2, iv2 = _load_triples_txt(KG_TRIPLES_PATH)
        nodes.update(n2)
        for h, ts in r2.items():
            rels[h].extend(ts)
        for t, hs in iv2.items():
            inv_rels[t].extend(hs)
        n_forward = sum(len(v) for v in r2.values())
        n_inverse = sum(len(v) for v in iv2.values())
        source_parts.append(f"triples({n_forward}+{n_inverse}rev)")
        print(f"  [KG] triples: {KG_TRIPLES_PATH} → {len(n2)} 节点, "
              f"{n_forward} 正向关系, {n_inverse} 反向索引")

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
    _SINGLETON = KnowledgeGraph(nodes, rels,
                                inverse_relations=inv_rels, source=src)
    print(f"  [KG] 总计 {len(nodes)} 节点 / {sum(len(v) for v in rels.values())} 正向 / "
          f"{sum(len(v) for v in inv_rels.values())} 反向 / source={src}")
    return _SINGLETON

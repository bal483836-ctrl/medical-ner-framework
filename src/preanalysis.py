"""
预学习分析（阶段 2 前置）
对每个数据集挖掘前 N 条样本，统计：
  - 文本长度分布
  - 实体长度分布
  - 高频前/后缀
  - 嵌套对统计

据此生成"skills"（抽取规则提示）与"skill_references"（典型示例），
作为后续 Step1 的 few-shot 增强材料。
"""
import json
import os
import random
import re
import sys
from collections import Counter
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import PREANALYSIS_SAMPLE_SIZE, FEW_SHOT_SEED, OUTPUT_DIR
from src.data_processor import (
    load_cmeee, load_imcs, load_yidu,
    extract_cmeee_gold_names, extract_imcs_gold, extract_yidu_gold_entities,
)


def _length_buckets(lengths: List[int]) -> Dict[str, int]:
    buckets = {"≤20": 0, "21-50": 0, "51-100": 0, "101-200": 0, ">200": 0}
    for L in lengths:
        if L <= 20: buckets["≤20"] += 1
        elif L <= 50: buckets["21-50"] += 1
        elif L <= 100: buckets["51-100"] += 1
        elif L <= 200: buckets["101-200"] += 1
        else: buckets[">200"] += 1
    return buckets


def _affix_stats(entities: List[str], k: int = 1) -> Dict[str, List]:
    """统计长度 ≥2 的实体的首/尾字符 top-10."""
    pref = Counter(e[:k] for e in entities if len(e) >= 2)
    suff = Counter(e[-k:] for e in entities if len(e) >= 2)
    return {
        "top_prefixes": pref.most_common(10),
        "top_suffixes": suff.most_common(10),
    }


def _nested_pairs(entities_per_doc: List[List[str]]) -> int:
    """统计文档内一对实体 a in b 且 a≠b 的次数。"""
    cnt = 0
    for ents in entities_per_doc:
        uniq = list(set(ents))
        for a in uniq:
            for b in uniq:
                if a != b and a in b:
                    cnt += 1
                    break
    return cnt


def analyze_dataset(name: str) -> Dict:
    random.seed(FEW_SHOT_SEED)
    if name == "CMeEE_V2":
        data = load_cmeee("train")
        sample = random.sample(data, min(PREANALYSIS_SAMPLE_SIZE, len(data)))
        texts = [it.get("text", "") for it in sample]
        ents_per = [extract_cmeee_gold_names(it) for it in sample]
    elif name == "IMCS_V2":
        data = load_imcs("train")
        sample = random.sample(data, min(PREANALYSIS_SAMPLE_SIZE, len(data)))
        # 对话场景：texts 用 self_report + 全部 sentence 拼接
        texts = []
        for it in sample:
            s = it.get("self_report", "")
            s += " " + " ".join(t.get("sentence", "") for t in it.get("dialogue", []))
            texts.append(s)
        ents_per = [extract_imcs_gold(it) for it in sample]
    elif name == "yidu_4k":
        data = load_yidu()
        sample = random.sample(data, min(PREANALYSIS_SAMPLE_SIZE, len(data)))
        texts = [it.get("text", "") for it in sample]
        ents_per = [extract_yidu_gold_entities(it) for it in sample]
    else:
        raise ValueError(f"unknown dataset {name}")

    all_ents = [e for ents in ents_per for e in ents]
    ent_lens = [len(e) for e in all_ents]
    report = {
        "dataset": name,
        "sample_size": len(sample),
        "text_length_buckets": _length_buckets([len(t) for t in texts]),
        "entity_count_per_doc_avg": round(sum(len(e) for e in ents_per) / max(1, len(ents_per)), 2),
        "entity_length_avg":   round(sum(ent_lens) / max(1, len(ent_lens)), 2),
        "entity_length_max":   max(ent_lens) if ent_lens else 0,
        "single_char_entity_ratio": round(
            sum(1 for L in ent_lens if L == 1) / max(1, len(ent_lens)), 3),
        "affixes": _affix_stats(all_ents),
        "nested_pair_docs": _nested_pairs(ents_per),
    }
    report["skills"]    = _derive_skills(name, report)
    report["skill_refs"] = _pick_reference_examples(name, sample, ents_per, n=3)
    return report


def _derive_skills(name: str, rep: Dict) -> List[str]:
    """根据统计自动生成抽取规则提示。"""
    skills = []
    if rep["single_char_entity_ratio"] > 0.05:
        skills.append("单字实体（如脑/肺/肝/胃）必须保留，不要合并到长词中。")
    if rep["nested_pair_docs"] > 0:
        skills.append("当文档存在嵌套实体时，长词与其内部子词都需要单独输出。")
    top_suf = [s for s, _ in rep["affixes"]["top_suffixes"][:5]]
    if top_suf:
        skills.append(f"常见尾字模式：{'/'.join(top_suf)}；遇到这些尾字优先视作实体边界。")
    if rep["entity_length_max"] >= 8:
        skills.append("存在长实体（≥8 字），勿过早切断；先匹配最长，再回扫子词。")
    if name == "IMCS_V2":
        skills.append("对话场景：医生提问中的症状名是询问而非确诊，不抽取；患者陈述里的口语症状原样保留。")
    if name == "CMeEE_V2":
        skills.append("禁止抽取年龄/性别/时间/方位/连词/纯化学基团。")
    return skills


def _pick_reference_examples(name: str, sample, ents_per, n=3) -> List[Dict]:
    """挑 n 条有代表性的样例：实体数量适中，含嵌套或单字。"""
    scored = []
    for it, ents in zip(sample, ents_per):
        if not ents:
            continue
        score = len(ents)
        if any(len(e) == 1 for e in ents):
            score += 5
        if any(a != b and a in b for a in ents for b in ents):
            score += 3
        scored.append((score, it, ents))
    scored.sort(key=lambda x: -x[0])
    refs = []
    for _, it, ents in scored[:n]:
        if name == "IMCS_V2":
            text = it.get("self_report", "") + " | " + " | ".join(
                f"{t.get('speaker','')}:{t.get('sentence','')}"
                for t in it.get("dialogue", [])[:6]
            )
        else:
            text = it.get("text", "")
        refs.append({"text": text[:300], "entities": ents})
    return refs


def render_skills_block(report: Dict) -> str:
    """把 skills + refs 渲染成 prompt 片段。"""
    lines = ["### 数据集 skill 提示"]
    for s in report.get("skills", []):
        lines.append(f"- {s}")
    refs = report.get("skill_refs", [])
    if refs:
        lines.append("\n### 参考示例")
        for i, r in enumerate(refs, 1):
            lines.append(f"例{i} 文本：{r['text']}")
            lines.append(f"例{i} 实体：{', '.join(r['entities'])}")
    return "\n".join(lines)


def run_preanalysis(save_dir: str = None) -> Dict[str, Dict]:
    """对三个数据集做预学习，保存 JSON 报告。"""
    save_dir = save_dir or OUTPUT_DIR
    os.makedirs(save_dir, exist_ok=True)
    reports = {}
    for name in ["CMeEE_V2", "IMCS_V2", "yidu_4k"]:
        try:
            print(f"\n[Pre-analysis] {name} (100 samples)…")
            reports[name] = analyze_dataset(name)
        except Exception as e:
            print(f"  ⚠️  {name} 预学习失败: {e}")
            reports[name] = {"error": str(e)}
    out = os.path.join(save_dir, "preanalysis_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    print(f"  ✅ 预学习报告: {out}")
    return reports


if __name__ == "__main__":
    run_preanalysis()

"""
LLM 断言标注（阶段 6）
输入：NER 通过过滤后的实体 + 语境 + KG 扩展信息
输出：每条 {entity, context, expansion, label} —— label ∈ {确定, 疑似, 无, 知识事实}

标签定义：
  确定 (Certain)：原文明确确认该实体存在 / 阳性
  疑似 (Suspected)：考虑/可能/不排除/疑为
  无 (Negated)：明确否认 / 否定 / 已排除
  知识事实 (Factual)：与该患者无关的通用医学陈述（教科书式描述）
"""
import json
import os
import sys
from typing import Dict, List
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import ASSERTION_LABELS, OUTPUT_DIR, ASSERT_PREFIX
from src.llm_client import batch_generate, clean_llm_output

ANNOT_BATCH = 16


def _format_expansion(exp: Dict) -> str:
    parts = []
    for k, vals in (exp or {}).items():
        if vals:
            parts.append(f"{k}: {','.join(vals[:5])}")
    return " | ".join(parts) if parts else "（无）"


def _prompt(entity: str, context: str, expansion: Dict, source: str) -> str:
    exp = _format_expansion(expansion)
    src_hint = "（对话场景，需区分医生提问与患者陈述）" if source == "dialogue" else ""
    labels = " / ".join(ASSERTION_LABELS)
    return f"""你是医学语义断言专家。基于语境判断对【目标实体】的断言类型。
候选标签：{labels}
- 确定：原文明确确认该实体存在（阳性表述）
- 疑似：考虑/可能/不排除/疑为/印象（不确定的推断）
- 无：明确否认 / 已排除 / 阴性
- 知识事实：通用医学陈述（与具体患者无关，如疾病机制描述）

目标实体：{entity}
KG 扩展信息：{exp}
语境{src_hint}：
{context}

只输出单个标签词，不要解释。
标签："""


def _parse_label(text: str) -> str:
    t = clean_llm_output(text)
    for lab in ASSERTION_LABELS:
        if lab in t:
            return lab
    return "确定"   # 兜底（最常见的类）


def annotate(samples: List[Dict]) -> List[Dict]:
    """
    samples: [{entity, context, expansion, source, ...}]
    返回写入 label 字段后的列表。
    """
    pending_idx = [i for i, s in enumerate(samples) if not s.get("label")]
    print(f"  [Assertion] 待标注 {len(pending_idx)} 条 / 总 {len(samples)}")
    for bs in tqdm(range(0, len(pending_idx), ANNOT_BATCH), desc="annot"):
        idxs = pending_idx[bs: bs + ANNOT_BATCH]
        prompts = [
            _prompt(samples[i]["entity"], samples[i]["context"],
                    samples[i].get("expansion", {}),
                    samples[i].get("source", "text"))
            for i in idxs
        ]
        resps = batch_generate(prompts, max_tokens=16, model_name="main")
        for i, r in zip(idxs, resps):
            samples[i]["label"] = _parse_label(r)
    return samples


def save(samples: List[Dict], dataset: str, split: str,
         out_dir: str = None) -> str:
    out_dir = out_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{ASSERT_PREFIX}{dataset}_{split}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"  ✅ 断言标注结果: {path}")
    return path

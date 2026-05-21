"""
DeepSeek 反思模块（NER Step 2.5）
对 Step1 输出的实体集合做一次质量复核：
  - 删除非实体噪声
  - 补漏明显遗漏
  - 不修改 IMCS 已经归一化的标准词

输入：[{text/sentence, step1_raw_output}], 输出更新 reflected_output 字段
"""
import json
import os
import sys
from typing import List, Dict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.llm_client import batch_generate, clean_llm_output
from src.data_processor import clean_entity_list

REFLECT_BATCH = 8


def _prompt(text: str, entities: List[str], domain_hint: str) -> str:
    ent_str = ", ".join(entities) if entities else "（空）"
    return f"""你是医学实体审核专家。{domain_hint}
原文：{text}
当前抽取实体：{ent_str}

任务：审核上述实体清单。
1) 删除明显不是医学实体或不在原文中的噪声；
2) 补充原文中明显遗漏的医学实体；
3) 保留原文写法，不要做归一化。
直接输出最终实体清单，逗号分隔，不要解释。
最终："""


def reflect_batch(items: List[Dict], text_field: str,
                  in_field: str, out_field: str, domain_hint: str) -> List[Dict]:
    pending = []
    for idx, it in enumerate(items):
        ents = clean_entity_list(it.get(in_field, ""))
        text = it.get(text_field, "")
        if not text:
            it[out_field] = ",".join(ents)
            continue
        pending.append((idx, text, ents))

    for bs in tqdm(range(0, len(pending), REFLECT_BATCH), desc="reflect"):
        batch = pending[bs:bs + REFLECT_BATCH]
        prompts = [_prompt(t, e, domain_hint) for _, t, e in batch]
        resps = batch_generate(prompts, max_tokens=256, model_name="reflect")
        for (idx, text, _), r in zip(batch, resps):
            cleaned = clean_entity_list(clean_llm_output(r))
            # 锚定回原文，丢掉幻觉
            anchored = [e for e in cleaned if e in text]
            items[idx][out_field] = ",".join(dict.fromkeys(anchored))
    return items


def reflect_cmeee(items: List[Dict]) -> List[Dict]:
    return reflect_batch(
        items, text_field="text",
        in_field="step1_enriched_output", out_field="reflected_output",
        domain_hint="数据集为 CMeEE，关注疾病/症状/手术/药物/检查/身体部位等。",
    )


def reflect_imcs(items: List[Dict]) -> List[Dict]:
    """IMCS：按 self_report+全部 sentence 拼成 text。"""
    for it in items:
        it["_full_text"] = it.get("self_report", "") + " " + " ".join(
            t.get("sentence", "") for t in it.get("dialogue", []))
    res = reflect_batch(
        items, text_field="_full_text",
        in_field="step1_raw_output", out_field="reflected_output",
        domain_hint="数据集为 IMCS 儿科对话，关注患者陈述的症状原词，不要做归一化。",
    )
    for it in res:
        it.pop("_full_text", None)
    return res


def reflect_yidu(items: List[Dict]) -> List[Dict]:
    return reflect_batch(
        items, text_field="text",
        in_field="step1_raw_output", out_field="reflected_output",
        domain_hint="数据集为电子病历，关注疾病诊断、检查指标、药物等。",
    )

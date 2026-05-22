"""
DeepSeek CoT 反思（NER Step 1.7）— v4.2.2
增加断点续传：传入 output_path 时，每 5 个 batch 落盘一次。
重启后通过 reflected_output 字段是否存在跳过已完成 item。
"""
import os
import re
import sys
from typing import List, Dict, Optional
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.llm_client import batch_generate
from src.data_processor import clean_entity_list
from src.utils import atomic_save_json, load_checkpoint

REFLECT_BATCH = 8
SAVE_EVERY_BATCHES = 5


def _cot_prompt(text: str, entities: List[str], domain_hint: str) -> str:
    cand = "、".join(entities) if entities else "无"
    return f"""你是医学审稿专家。{domain_hint}逐项审查下列候选实体。

# 原文
{text}

# 候选实体
{cand}

# 审查规则
1. 候选必须字面出现在原文中
2. 否定句中的症状（「不发烧」「未见」）应删除
3. 询问句中的症状应删除
4. 不属于该数据集目标实体的应删除
5. 不要新增原文中没有的实体

# 输出格式（必须严格遵守）
<thinking>
逐项分析每个候选实体...
</thinking>
<answer>
最终实体列表（用顿号"、"分隔，无则输出"无"）
</answer>"""


def _parse_answer(text: str) -> List[str]:
    if not text:
        return []
    text = re.sub(r"```[a-z]*|```", "", text)
    m = re.search(r"<answer>([\s\S]*?)</answer>", text, re.IGNORECASE)
    body = m.group(1) if m else text
    body = re.sub(r"<think(ing)?>[\s\S]*?</think(ing)?>", "", body, flags=re.IGNORECASE)
    return clean_entity_list(body)


def reflect_batch(items: List[Dict], text_field: str,
                  in_field: str, out_field: str, domain_hint: str,
                  output_path: Optional[str] = None) -> List[Dict]:
    """
    支持断点续传：output_path 不为 None 时，
      - 启动先加载已存在的同名文件，把 out_field 已填的 item 标记为完成
      - 每 SAVE_EVERY_BATCHES 个 batch 落盘一次
    """
    # 断点恢复：合并已有结果
    if output_path:
        existing = load_checkpoint(output_path)
        if existing and len(existing) == len(items):
            for src, cached in zip(items, existing):
                if cached.get(out_field) and not src.get(out_field):
                    src[out_field] = cached[out_field]
            done = sum(1 for it in items if it.get(out_field) is not None)
            print(f"  [Resume] reflector 已完成 {done}/{len(items)}")

    pending = []
    for idx, it in enumerate(items):
        if it.get(out_field) is not None:
            continue   # 已经反思过，跳过
        ents = clean_entity_list(it.get(in_field, ""))
        text = it.get(text_field, "")
        if not text:
            it[out_field] = ",".join(ents)
            continue
        pending.append((idx, text, ents))

    batch_count = 0
    for bs in tqdm(range(0, len(pending), REFLECT_BATCH), desc="reflect"):
        batch = pending[bs:bs + REFLECT_BATCH]
        prompts = [_cot_prompt(t, e, domain_hint) for _, t, e in batch]
        resps = batch_generate(prompts, max_tokens=512, model_name="reflect")
        for (idx, text, _), r in zip(batch, resps):
            ents = _parse_answer(r)
            anchored = [e for e in ents if e in text]
            items[idx][out_field] = ",".join(dict.fromkeys(anchored))
        batch_count += 1
        if output_path and batch_count % SAVE_EVERY_BATCHES == 0:
            atomic_save_json(items, output_path)

    if output_path:
        atomic_save_json(items, output_path)
    return items


def reflect_cmeee(items, output_path=None):
    return reflect_batch(
        items, text_field="text",
        in_field="step1_enriched_output", out_field="reflected_output",
        domain_hint="数据集为 CMeEE_V2，9 类实体（dis/sym/pro/equ/dru/ite/bod/mic/dep）。",
        output_path=output_path,
    )


def reflect_imcs(items, output_path=None):
    for it in items:
        it["_full_text"] = (it.get("self_report") or "") + " " + " ".join(
            t.get("sentence", "") for t in it.get("dialogue", []))
    res = reflect_batch(
        items, text_field="_full_text",
        in_field="step1_raw_output", out_field="reflected_output",
        domain_hint="数据集为 IMCS_V2 儿科对话，抽原文症状形态，不归一化。",
        output_path=output_path,
    )
    for it in res:
        it.pop("_full_text", None)
    return res


def reflect_yidu(items, output_path=None):
    return reflect_batch(
        items, text_field="text",
        in_field="step1_raw_output", out_field="reflected_output",
        domain_hint="数据集为电子病历，关注疾病/影像/检验/药物/解剖/手术 6 类。",
        output_path=output_path,
    )

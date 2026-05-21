"""
LLM 断言标注（阶段 6）—— v4.1

关键升级（吸取 run_llm_assertion.py）：
  1. Entity Marker：[E]...[/E] 包裹目标实体，避免同名实体歧义
  2. JSON 输出格式，正则 + json 双重兜底解析
  3. 4 类标签：Present/Possible/Absent/General → 确定/疑似/无/知识事实
  4. 加入 KG 知识进 prompt，提升对"知识事实"类的判定准确度
"""
import json
import os
import re
import sys
from typing import Dict, List
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    ASSERTION_LABELS, ASSERTION_EN2ZH, OUTPUT_DIR, ASSERT_PREFIX,
)
from src.llm_client import batch_generate

ANNOT_BATCH = 8

SYSTEM_PROMPT = "你是一个严谨的临床医学信息提取专家。你的核心任务是对医学文本中的实体进行状态断言分类。"

USER_PROMPT_TEMPLATE = """【任务】分析下文【医学语境】中被特殊标记符 [E] 和 [/E] 包裹的【目标实体】，判断其客观临床存在状态。

【硬约束(Label Set)】严格从以下 4 个标签中选择 1 个，绝不能输出其他词汇：
1. Present (确定/阳性)：语境明确说明该疾病/症状/状态存在，或已被诊断。
2. Absent (无/阴性)：语境明确否认、排除或说明没有该疾病/症状。
3. Possible (疑似)：语境说明疑似、可能、考虑、不排除、待排查。
4. General (知识事实)：语境是医学教材/科普/用药禁忌/泛泛而谈，没有针对具体患者的“有/无”状态。
   例：「阿司匹林可治疗头痛」「高血压会导致晕厥」均属于此类。

【KG 知识参考】{kg_knowledge}

【医学语境】
{marked_text}

【目标实体】
{entity_name}

【输出格式】
仅输出标准 JSON，不要包含任何 Markdown 标记或额外解释：
{{"results": [{{"entity": "{entity_name}", "status": "从4个标签中选1个"}}]}}
"""


def _mark_entity(text: str, entity: str) -> str:
    """在 context 中给实体加 [E]...[/E] 标记。首次出现处标。"""
    if not entity:
        return text
    if entity in text:
        return text.replace(entity, f"[E]{entity}[/E]", 1)
    return f"{text} (目标: [E]{entity}[/E])"


def _parse_status(raw: str) -> str:
    """JSON 优先，正则兜底，返回 4 类标签的英文键。"""
    if not raw:
        return "Present"
    clean = re.sub(r"```json\s*|\s*```", "", raw).strip()
    clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL)
    m = re.search(r"\{.*\}", clean, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            if "results" in data and isinstance(data["results"], list) and data["results"]:
                st = data["results"][0].get("status", "")
                if st: return st.strip()
            if "status" in data:
                return str(data["status"]).strip()
        except json.JSONDecodeError:
            pass
    # 正则硬抓
    if re.search(r"\bPresent\b|阳性|确定", raw, re.IGNORECASE): return "Present"
    if re.search(r"\bAbsent\b|阴性|否认|否定|无\b", raw, re.IGNORECASE): return "Absent"
    if re.search(r"\bPossible\b|可能|疑似", raw, re.IGNORECASE): return "Possible"
    if re.search(r"\bGeneral\b|一般|知识|科普", raw, re.IGNORECASE): return "General"
    return "Present"


def _to_zh_label(status: str) -> str:
    """英文标签 → 中文 4 类。"""
    return ASSERTION_EN2ZH.get(status, "确定")


def _build_prompt(entity: str, context: str, kg_knowledge: str) -> str:
    marked = _mark_entity(context, entity)
    return USER_PROMPT_TEMPLATE.format(
        marked_text=marked,
        entity_name=entity,
        kg_knowledge=kg_knowledge or "无关联知识",
    )


def annotate(samples: List[Dict]) -> List[Dict]:
    pending_idx = [i for i, s in enumerate(samples) if not s.get("label")]
    print(f"  [Assertion] 待标注 {len(pending_idx)} / 总 {len(samples)}")
    for bs in tqdm(range(0, len(pending_idx), ANNOT_BATCH), desc="annot"):
        idxs = pending_idx[bs: bs + ANNOT_BATCH]
        prompts = []
        for i in idxs:
            s = samples[i]
            kg_str = ""
            exp = s.get("expansion") or {}
            if isinstance(exp, dict):
                ps = []
                # possible_diseases 放最前：对"知识事实"判别最有用
                # （若实体能反查到多种关联疾病，则更可能是泛泛医学陈述）
                if exp.get("possible_diseases"):
                    ps.append(f"可能关联疾病:{','.join(exp['possible_diseases'][:5])}")
                if exp.get("kg_facts"):
                    ps.append("事实:" + ";".join(exp["kg_facts"][:3]))
                for k_zh, k in (("同义", "synonyms"), ("上位", "hypernyms"), ("相关", "related")):
                    if exp.get(k):
                        ps.append(f"{k_zh}:{','.join(exp[k][:3])}")
                kg_str = " | ".join(ps) if ps else "无关联知识"
            prompts.append(_build_prompt(s["entity"], s.get("context", ""), kg_str))
        # 加 system prompt
        prompts = [f"<<SYSTEM>>\n{SYSTEM_PROMPT}\n<<END>>\n{p}" for p in prompts]
        resps = batch_generate(prompts, max_tokens=128, model_name="main")
        for i, r in zip(idxs, resps):
            status_en = _parse_status(r)
            samples[i]["label"]       = _to_zh_label(status_en)
            samples[i]["label_en"]    = status_en
            samples[i]["llm_raw"]     = r[:200]
    return samples


def save(samples: List[Dict], dataset: str, split: str,
         out_dir: str = None) -> str:
    out_dir = out_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{ASSERT_PREFIX}{dataset}_{split}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"  ✅ 断言标注: {path}")
    return path

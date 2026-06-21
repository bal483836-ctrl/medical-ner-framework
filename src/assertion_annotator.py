"""
LLM 断言标注（阶段 6）—— v4.2

升级（在 v4.1 基础上）：
  1. Entity Marker：[E]...[/E] 包裹目标实体
  2. JSON 输出 + 正则双重兜底
  3. KG 知识参考（含 possible_diseases 反向索引）
  4. **自洽投票（self-consistency）**：同一样本跑 N 次（带不同提示扰动），
     取多数票。减少 LLM 标注噪声 → 显著提升下游分类器上限。
"""
import json
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Optional
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    ASSERTION_LABELS, ASSERTION_EN2ZH, OUTPUT_DIR, ASSERT_PREFIX,
    ASSERTION_VOTE_PASSES,
)
from src.llm_client import batch_generate
from src.utils import atomic_save_json, load_checkpoint

ANNOT_BATCH = 8
SAVE_EVERY_BATCHES = 10

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


def _build_prompt(entity: str, context: str, kg_knowledge: str,
                  variant: int = 0) -> str:
    """
    variant: 用于自洽投票时的轻微提示扰动。
      0: 标准 prompt
      1: 强调"先识别否定/疑似/科普语境，再确认存在"
      2: 强调"先看 KG 关联疾病数量，再判断个体阳性"
    """
    marked = _mark_entity(context, entity)
    base = USER_PROMPT_TEMPLATE.format(
        marked_text=marked, entity_name=entity,
        kg_knowledge=kg_knowledge or "无关联知识",
    )
    if variant == 1:
        return base + "\n【判定顺序】先排查否定/询问/科普语境，确认非以上三者再选 Present。"
    if variant == 2:
        return base + "\n【判定顺序】先看 KG 是否给出多个关联疾病，若是且无具体患者主诉，倾向 General。"
    return base


def _kg_string(sample: Dict) -> str:
    exp = sample.get("expansion") or {}
    if not isinstance(exp, dict):
        return "无关联知识"
    ps = []
    if exp.get("possible_diseases"):
        ps.append(f"可能关联疾病:{','.join(exp['possible_diseases'][:5])}")
    if exp.get("kg_facts"):
        ps.append("事实:" + ";".join(exp["kg_facts"][:3]))
    for k_zh, k in (("同义", "synonyms"), ("上位", "hypernyms"), ("相关", "related")):
        if exp.get(k):
            ps.append(f"{k_zh}:{','.join(exp[k][:3])}")
    return " | ".join(ps) if ps else "无关联知识"


def annotate(samples: List[Dict],
             vote_passes: int = ASSERTION_VOTE_PASSES,
             output_path: Optional[str] = None) -> List[Dict]:
    """
    自洽投票多数票。
    output_path 不为 None 时启用断点续传：
      - 启动时若 output_path 存在且条数匹配，复用其中已有 label
      - 每 SAVE_EVERY_BATCHES 个 batch 落盘一次
    """
    # 断点恢复
    if output_path:
        existing = load_checkpoint(output_path)
        if existing and len(existing) == len(samples):
            for src, cached in zip(samples, existing):
                if cached.get("label") and not src.get("label"):
                    src["label"]      = cached["label"]
                    src["label_en"]   = cached.get("label_en", "")
                    src["vote_dist"]  = cached.get("vote_dist", {})
                    src["llm_raw"]    = cached.get("llm_raw", "")
            n_done = sum(1 for s in samples if s.get("label"))
            print(f"  [Resume] annotator 已完成 {n_done}/{len(samples)}")

    pending_idx = [i for i, s in enumerate(samples) if not s.get("label")]
    print(f"  [Assertion] 待标注 {len(pending_idx)} / 总 {len(samples)}  "
          f"(vote_passes={vote_passes})")

    batch_count = 0
    for bs in tqdm(range(0, len(pending_idx), ANNOT_BATCH), desc="annot"):
        idxs = pending_idx[bs: bs + ANNOT_BATCH]
        votes: Dict[int, List[str]] = {i: [] for i in idxs}
        raw_first: Dict[int, str] = {}

        for v in range(max(1, vote_passes)):
            prompts = []
            for i in idxs:
                s = samples[i]
                kg_str = _kg_string(s)
                prompts.append(_build_prompt(
                    s["entity"], s.get("context", ""), kg_str, variant=v % 3))
            prompts = [f"<<SYSTEM>>\n{SYSTEM_PROMPT}\n<<END>>\n{p}" for p in prompts]
            resps = batch_generate(prompts, max_tokens=128, model_name="main")
            for i, r in zip(idxs, resps):
                votes[i].append(_parse_status(r))
                if v == 0:
                    raw_first[i] = r[:200]

        # 多数票 + 置信度分层（v2 核心改动）
        for i in idxs:
            cnt = Counter(votes[i])
            status_en, top_n = cnt.most_common(1)[0]
            total_v = sum(cnt.values()) or 1
            agreement = top_n / total_v   # 0.33 / 0.50 / 0.67 / 1.0 ...
            # 置信度分桶（vote_passes=3 时）：
            #   1.00  → strong   (3/3 一致)
            #   0.67  → medium   (2/3 一致)
            #   <0.67 → weak     (分散)
            if agreement >= 1.0 - 1e-6:
                confidence = "strong"
            elif agreement >= 0.5:
                confidence = "medium"
            else:
                confidence = "weak"
            samples[i]["label"]       = _to_zh_label(status_en)
            samples[i]["label_en"]    = status_en
            samples[i]["vote_dist"]   = dict(cnt)
            samples[i]["vote_agreement"] = round(agreement, 3)
            samples[i]["vote_confidence"] = confidence   # strong/medium/weak
            samples[i]["llm_raw"]     = raw_first.get(i, "")

        batch_count += 1
        if output_path and batch_count % SAVE_EVERY_BATCHES == 0:
            atomic_save_json(samples, output_path)

    if output_path:
        atomic_save_json(samples, output_path)
    return samples


def filter_by_confidence(samples: List[Dict],
                          min_confidence: str = "medium") -> List[Dict]:
    """
    按置信度过滤标注样本，用于训练数据清洗。

    min_confidence:
      - "strong" : 只保留 3/3 一致的（最干净，量最少）
      - "medium" : strong + medium (2/3 一致；常用，量适中)
      - "weak"   : 全部（含 1/3 分散；最噪声）

    返回过滤后的样本列表，并打印分桶统计。
    """
    rank = {"strong": 3, "medium": 2, "weak": 1}
    cutoff = rank.get(min_confidence, 2)

    buckets = {"strong": 0, "medium": 0, "weak": 0, "missing": 0}
    kept = []
    for s in samples:
        conf = s.get("vote_confidence")
        if conf is None:
            buckets["missing"] += 1
            # 没有 vote_confidence 字段（旧数据）→ 默认 medium 通过
            if cutoff <= 2:
                kept.append(s)
            continue
        buckets[conf] = buckets.get(conf, 0) + 1
        if rank.get(conf, 0) >= cutoff:
            kept.append(s)

    print(f"  [Confidence] 分布 strong={buckets['strong']}  medium={buckets['medium']}  "
          f"weak={buckets['weak']}  missing={buckets['missing']}")
    print(f"  [Confidence] min_confidence={min_confidence}，保留 {len(kept)}/{len(samples)} 条")
    return kept


def save(samples: List[Dict], dataset: str, split: str,
         out_dir: str = None) -> str:
    out_dir = out_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{ASSERT_PREFIX}{dataset}_{split}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"  ✅ 断言标注: {path}")
    return path

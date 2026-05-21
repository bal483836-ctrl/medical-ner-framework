"""
预学习（阶段 2 前置）— v4.1
  ① 数据统计：长度/前后缀/嵌套对
  ② LLM skill 归纳：给 Qwen 看 30 条样本 → 生成"看到 X 模式做 Y"的可执行规则
  ③ 同时生成 skill_refs（高教学价值参考样本）
  ④ 全部产物缓存到 outputs/knowledge/llm_skills_{ds}.json，避免重跑

吸取 v21 medical_ner_v21_FINAL/src/skill_generator.py 的设计。
"""
import json
import os
import random
import re
import sys
from collections import Counter
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import PREANALYSIS_SAMPLE_SIZE, FEW_SHOT_SEED, OUTPUT_DIR
from src.data_processor import (
    load_cmeee, load_imcs, load_yidu,
    extract_cmeee_gold_names, extract_imcs_gold, extract_yidu_gold_entities,
)

# ==================== 静态后备 skills ====================

_FALLBACK_NER_SKILLS_CMEEE = """【技巧1】单字身体部位召回：原文出现独立的「血/心/肝/肺/肾/尿/胸/腹/脑」等单字时，作为 bod 抽出
【技巧2】整段长描述：5 字以上医学过程描述（如「血常规白细胞计数可增高」），整段作为一个实体
【技巧3】嵌套同抽：长实体含短医学词（"葡萄球菌肺炎"+"葡萄球菌"）→ 两者都抽
【技巧4】英文缩写保留：CT、ERCP、BUN、GFR、MRI、CRP 等英文缩写作为独立实体抽出
【技巧5】前后缀模式：「急性/慢性/左侧/右侧/双侧 X」整体作为 dis；尾字含「炎/症/病/癌/瘤」整体为 dis
【技巧6】指标类边界：心率/血压/血糖/体温/菌落数/肌张力/白细胞计数 → ite，不是 sym
【技巧7】操作类边界：化疗/全身用药/针灸/血液透析 → pro，不是 dru/sym
【技巧8】上位词不抽：单独的"疾病/症状/患者/并发症/情况"不抽
【技巧9】否定/询问不抽：「不/没/否认」+症状词不抽；问句中症状词不抽
【技巧10】边界精确：实体不带尾部"了/的/吗/啊"，不带前缀"治疗/明显的"等修饰"""

_FALLBACK_NER_SKILLS_IMCS = """【技巧1】抽原文形态：「拉肚子」输出「拉肚子」，不要归一化成「腹泻」
【技巧2】数字温度保留：「38.5度」「37.2-38.2度」作为体征抽出
【技巧3】口语症状全抽：拉肚子/流清鼻涕/嗓子哑/没精神/不吃奶都要抽
【技巧4】否定句过滤：「不发烧」「没拉稀」「没什么大碍」不抽
【技巧5】询问句过滤：医生问「有腹泻吗？」「体温多少？」不抽
【技巧6】非症状词过滤：药名/检查项目/治疗手段不抽
【技巧7】完整描述：「绿色的大便」不简化为「大便」，保留完整描述
【技巧8】跨轮锚定：实体在原文（含上文 self_report）中字面出现即可"""

_FALLBACK_NER_SKILLS_YIDU = """【技巧1】英文标记保留：影像/检验项目（CT/MRI/B超/血常规）作为独立实体抽出
【技巧2】疾病诊断完整：「2型糖尿病」「冠状动脉粥样硬化性心脏病」整体抽出
【技巧3】解剖部位单独抽：「肝」「左肾上极」「胃窦」等解剖部位独立抽出
【技巧4】药物抽出：通用名 + 商品名都抽（「阿司匹林」「拜阿司匹林」）
【技巧5】手术操作抽出：动词形态的手术名（「胆囊切除术」）整体抽出
【技巧6】指标值忽略：单纯的数值（"3.2 mmol/L"）不抽，但指标名（"血糖"）要抽"""

_FALLBACK_NORM_SKILLS = """【技巧1】动词类口语映射：「拉 X」→ 腹泻；「流 X」→ 鼻流涕；「咳 X」→ 咳嗽；「吐 X」→ 呕吐
【技巧2】程度修饰词去除：「有点 X」「比较 X」「严重 X」「持续 X」「偶尔 X」→ 去前缀
【技巧3】数字温度推断：体温 ≥39.1°C → 高热；38.1-39.0 → 中等度热；37.3-38.0 → 低热
【技巧4】否定语境过滤：原文中口语词前后有「不/没/未/否认」时该症状不输出
【技巧5】询问句过滤：当前发言以「？」「呢」「吗」结尾且含症状词时不输出
【技巧6】同义合并：「精神不好/萎靡/没精神」→ 精神软；「吃奶少/胃口不好」→ 食欲不振"""


# ==================== 数据统计 ====================

def _length_buckets(lengths):
    b = {"≤20": 0, "21-50": 0, "51-100": 0, "101-200": 0, ">200": 0}
    for L in lengths:
        if L <= 20: b["≤20"] += 1
        elif L <= 50: b["21-50"] += 1
        elif L <= 100: b["51-100"] += 1
        elif L <= 200: b["101-200"] += 1
        else: b[">200"] += 1
    return b


def _affix_stats(entities, k=1):
    pref = Counter(e[:k] for e in entities if len(e) >= 2)
    suff = Counter(e[-k:] for e in entities if len(e) >= 2)
    return {"top_prefixes": pref.most_common(10),
            "top_suffixes": suff.most_common(10)}


def _nested_pair_count(ents_per_doc):
    cnt = 0
    for ents in ents_per_doc:
        uniq = list(set(ents))
        for a in uniq:
            for b in uniq:
                if a != b and a in b:
                    cnt += 1; break
    return cnt


# ==================== 教学样本挑选（含 type 信息）====================

def _select_diverse_demos(samples, dataset_name, n=30):
    scored = []
    for it in samples:
        score = 0
        if dataset_name == "CMeEE_V2":
            ents = it.get("entities", [])
            names = [e.get("entity") or e.get("mention") or "" for e in ents]
            names = [n for n in names if n]
            if not names: continue
            for a in names:
                for b in names:
                    if a != b and a in b: score += 2; break
            if any(len(n) == 1 for n in names): score += 2
            if any(len(n) >= 8 for n in names): score += 3
            score += len({(e.get("type") or e.get("label") or "") for e in ents})
            if 30 <= len(it.get("text", "")) <= 200: score += 1
        elif dataset_name == "IMCS_V2":
            for turn in it.get("dialogue", []):
                ner = turn.get("ner", []) or []
                norms = [n.get("symptom_norm") for n in ner if isinstance(n, dict)]
                norms += list(turn.get("symptom_norm", []))
                norms = [n for n in norms if n]
                score += min(len(norms), 3)
                sent = turn.get("sentence", "")
                for nm in norms:
                    if nm and nm not in sent: score += 2
        else:  # yidu
            ents = extract_yidu_gold_entities(it)
            if ents: score += len(ents) + (3 if any(len(e) >= 6 for e in ents) else 0)
        scored.append((score, it))
    scored.sort(key=lambda x: -x[0])
    return [it for _, it in scored[:n]]


def _format_demos(demos, dataset_name):
    """格式化 demo 给 LLM 看。"""
    lines = []
    if dataset_name == "CMeEE_V2":
        for i, it in enumerate(demos[:20], 1):
            text = (it.get("text", "") or "")[:120]
            ents = it.get("entities", [])
            estr = []
            for e in ents[:8]:
                nm = e.get("entity") or e.get("mention") or ""
                tp = e.get("type") or e.get("label") or ""
                if nm: estr.append(f"{nm}[{tp}]")
            lines.append(f"{i}. 原文: {text}\n   实体: {'; '.join(estr)}")
    elif dataset_name == "IMCS_V2":
        for i, it in enumerate(demos[:15], 1):
            sub = []
            for turn in it.get("dialogue", []):
                ner = turn.get("ner", []) or []
                norms = []
                for ni in ner:
                    if isinstance(ni, dict) and str(ni.get("symptom_type")) in ("1", "2"):
                        norms.append(ni.get("symptom_norm"))
                norms = [n for n in norms if n]
                if norms:
                    sub.append(f"     - {turn.get('sentence','')[:60]} → {'、'.join(norms)}")
            if sub: lines.append(f"{i}. 对话片段:\n" + "\n".join(sub[:3]))
    else:
        for i, it in enumerate(demos[:20], 1):
            text = (it.get("text", "") or "")[:120]
            ents = extract_yidu_gold_entities(it)
            lines.append(f"{i}. 原文: {text}\n   实体: {'、'.join(ents[:10])}")
    return "\n\n".join(lines)


# ==================== LLM Skill 生成 ====================

def _meta_prompt(demos_text: str, dataset_name: str) -> str:
    if dataset_name == "CMeEE_V2":
        ds_desc = "中文医学文献的 9 类实体识别（dis 疾病/sym 症状/pro 操作/equ 设备/dru 药物/ite 指标/bod 部位/mic 微生物/dep 科室）"
    elif dataset_name == "IMCS_V2":
        ds_desc = "儿科医患对话中的症状识别（抽原文形态，不归一化）"
    else:
        ds_desc = "电子病历的 6 类实体识别（疾病/影像/检验/药物/解剖/手术）"

    return f"""你是 NER 专家。仔细观察以下 {ds_desc} 标注样本，归纳出 6-10 条**可立即执行的抽取技巧**。

# 样本
{demos_text}

# 要求
- 每条 1-2 句，告诉 LLM "看到什么模式 → 应该做什么"
- 必须基于上述样本观察，不能凭空写
- 拒绝抽象建议（如"要仔细分析"）

# 输出格式（直接列表，无前缀解释）
【技巧1】..."""


def _parse_skills(raw: str) -> str:
    if not raw: return ""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"```[a-z]*|```", "", raw)
    raw = re.sub(r"^[^【0-9]*?(?=【|\d+[\.\、])", "", raw)
    return raw.strip()


def _llm_generate_skills(dataset_name: str, demos_text: str,
                        cache_path: str, force: bool = False) -> str:
    if not force and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("skills") and cached.get("dataset") == dataset_name:
                print(f"  ✓ 复用 LLM skills 缓存: {cache_path}")
                return cached["skills"]
        except Exception:
            pass

    fallback = {
        "CMeEE_V2": _FALLBACK_NER_SKILLS_CMEEE,
        "IMCS_V2":  _FALLBACK_NER_SKILLS_IMCS,
        "yidu_4k":  _FALLBACK_NER_SKILLS_YIDU,
    }.get(dataset_name, _FALLBACK_NER_SKILLS_CMEEE)

    try:
        from src.llm_client import call_llm
        print(f"  🤖 LLM 生成 {dataset_name} skills（看 demos）…")
        raw = call_llm(_meta_prompt(demos_text, dataset_name),
                       max_tokens=1024, model_name="main")
        skills = _parse_skills(raw)
        if not skills or len(skills) < 60:
            raise ValueError("LLM 输出过短")
    except Exception as e:
        print(f"  ⚠️ LLM skills 生成失败 ({e})，使用后备")
        skills = fallback.strip()

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset_name, "skills": skills},
                  f, ensure_ascii=False, indent=2)
    return skills


# ==================== 分析单数据集 ====================

def analyze_dataset(name: str, use_llm: bool = True,
                    force_regen: bool = False) -> Dict:
    random.seed(FEW_SHOT_SEED)
    if name == "CMeEE_V2":
        data = load_cmeee("train")
        sample = random.sample(data, min(PREANALYSIS_SAMPLE_SIZE, len(data)))
        texts = [it.get("text", "") for it in sample]
        ents_per = [extract_cmeee_gold_names(it) for it in sample]
    elif name == "IMCS_V2":
        data = load_imcs("train")
        sample = random.sample(data, min(PREANALYSIS_SAMPLE_SIZE, len(data)))
        texts = []
        for it in sample:
            s = (it.get("self_report") or "") + " " + " ".join(
                t.get("sentence", "") for t in it.get("dialogue", []))
            texts.append(s)
        ents_per = [extract_imcs_gold(it) for it in sample]
    elif name == "yidu_4k":
        data = load_yidu()
        sample = random.sample(data, min(PREANALYSIS_SAMPLE_SIZE, len(data)))
        texts = [it.get("text", "") for it in sample]
        ents_per = [extract_yidu_gold_entities(it) for it in sample]
    else:
        raise ValueError(name)

    all_ents = [e for ents in ents_per for e in ents]
    ent_lens = [len(e) for e in all_ents]

    report = {
        "dataset": name,
        "sample_size": len(sample),
        "text_length_buckets": _length_buckets([len(t) for t in texts]),
        "entity_count_avg": round(sum(len(e) for e in ents_per) / max(1, len(ents_per)), 2),
        "entity_length_avg": round(sum(ent_lens) / max(1, len(ent_lens)), 2),
        "entity_length_max": max(ent_lens) if ent_lens else 0,
        "single_char_ratio": round(sum(1 for L in ent_lens if L == 1) / max(1, len(ent_lens)), 3),
        "affixes": _affix_stats(all_ents),
        "nested_pair_docs": _nested_pair_count(ents_per),
    }

    # LLM skills（缓存到 outputs/knowledge/）
    cache_path = os.path.join(OUTPUT_DIR, "knowledge", f"llm_skills_{name}.json")
    demos = _select_diverse_demos(sample, name, n=30)
    demos_text = _format_demos(demos, name)
    if use_llm:
        report["skills"] = _llm_generate_skills(name, demos_text, cache_path, force=force_regen)
    else:
        report["skills"] = {
            "CMeEE_V2": _FALLBACK_NER_SKILLS_CMEEE,
            "IMCS_V2":  _FALLBACK_NER_SKILLS_IMCS,
            "yidu_4k":  _FALLBACK_NER_SKILLS_YIDU,
        }.get(name, "")

    # 高教学价值的 3 条作 skill_refs
    report["skill_refs"] = _pick_refs(sample, ents_per, name, n=3)
    return report


def _pick_refs(sample, ents_per, name, n=3):
    scored = []
    for it, ents in zip(sample, ents_per):
        if not ents: continue
        score = len(ents)
        if any(len(e) == 1 for e in ents): score += 5
        if any(a != b and a in b for a in ents for b in ents): score += 3
        scored.append((score, it, ents))
    scored.sort(key=lambda x: -x[0])
    refs = []
    for _, it, ents in scored[:n]:
        if name == "IMCS_V2":
            text = (it.get("self_report") or "") + " | " + " | ".join(
                f"{t.get('speaker','')}:{t.get('sentence','')}"
                for t in it.get("dialogue", [])[:6])
        else:
            text = it.get("text", "")
        refs.append({"text": text[:300], "entities": ents})
    return refs


def render_skills_block(report: Dict) -> str:
    skills = report.get("skills", "")
    refs = report.get("skill_refs", [])
    lines = []
    if skills:
        lines.append("### 抽取技巧（从训练样本归纳）")
        lines.append(skills)
    if refs:
        lines.append("\n### 高质量参考")
        for i, r in enumerate(refs, 1):
            lines.append(f"例{i} 文本：{r['text']}")
            lines.append(f"例{i} 实体：{', '.join(r['entities'])}")
    return "\n".join(lines)


def run_preanalysis(save_dir: str = None, use_llm: bool = True,
                    force_regen: bool = False) -> Dict[str, Dict]:
    save_dir = save_dir or OUTPUT_DIR
    os.makedirs(save_dir, exist_ok=True)
    reports = {}
    for name in ["CMeEE_V2", "IMCS_V2", "yidu_4k"]:
        try:
            print(f"\n[Pre-analysis] {name}（{PREANALYSIS_SAMPLE_SIZE} 条）…")
            reports[name] = analyze_dataset(name, use_llm=use_llm,
                                            force_regen=force_regen)
        except Exception as e:
            print(f"  ⚠️ {name} 失败: {e}")
            reports[name] = {"error": str(e)}
    out = os.path.join(save_dir, "preanalysis_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    print(f"  ✅ 报告: {out}")
    return reports


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run_preanalysis(use_llm=not args.no_llm, force_regen=args.force)

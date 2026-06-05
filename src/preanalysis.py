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

# 给 LLM 看的"易混淆/非医学"反例（与 demos 一起喂入），帮助 skills 写出过滤规则
_NEG_HINTS = {
    "CMeEE_V2": """以下词在原文中常出现但**几乎从不**被 CMeEE 金标标注，属于噪声：
- 性别年龄：男/女/男孩/女孩/患者/患儿/新生儿/婴儿/儿童/成人
- 时间副词：今天/昨天/以前/最近/反复/经常/突然/长期/持续
- 部位修饰：左/右/双/上/下/前/后/内/外/正常/异常
- 数值单位：mg/kg/ml/mmHg/℃/%/次/天/周/月/年
- 治疗/检查的动词形态：治疗/诊断/检查/观察/给予/服用/注射
- 模糊描述：一些/部分/可能/相关/常见/严重/明显
""",
    "IMCS_V2": """以下词在儿科对话原文常出现但**几乎从不**被 IMCS 金标标注为症状：
- 药物/药品：蒙脱石散/益生菌/罗红霉素/补液/口服液/妈咪爱/小儿氨酚黄那敏
- 检查/项目：便常规/血常规/B超/拍片/水电解质
- 病因诱因：受凉/吹风/腹部受冷/吃多了/食积/挑食/纯母乳喂养
- 患者称谓：男孩/女孩/宝宝/孩子/患儿
- 询问语气：有没有/是否/会不会/要不要
- 否定形态：不发烧/没拉稀/没什么大碍/未见
- 模糊描述：精神状态/大便次数多/特别粘/还可以
- 舌象（IMCS 不属症状）：舌苔黄/舌苔厚/舌尖红/舌红
""",
    "yidu_4k": """以下词在电子病历常出现但**几乎从不**被 yidu_4k 金标标注：
- 时间日期：年/月/日/天/小时/分钟/AM/PM
- 数值剂量：mg/g/ml/kg/L/mmHg/g/L
- 医务人员：医生/护士/主任/医师
- 主诉前缀：主诉/现病史/查体/体格检查（这些是文段标题，不是实体）
- 一般动作：服用/给予/检查/复查/观察
- 模糊词：一般/正常/异常/常见/严重
""",
}


def _meta_prompt(demos_text: str, dataset_name: str) -> str:
    if dataset_name == "CMeEE_V2":
        ds_desc = "中文医学文献的 9 类医学实体识别（dis 疾病/sym 症状/pro 操作/equ 设备/dru 药物/ite 指标/bod 部位/mic 微生物/dep 科室）"
        scope = "**必须是医学相关实体**，非医学词（地名/人名/产品名/数值/时间/副词）一律不抽"
    elif dataset_name == "IMCS_V2":
        ds_desc = "儿科医患对话中的症状识别（抽原文形态，不归一化）"
        scope = "**只抽症状/体征**，药物/检查项目/病因诱因/疾病诊断/患者描述 都不属于本任务"
    else:
        ds_desc = "电子病历的 6 类医学实体（疾病/影像/检验/药物/解剖/手术）"
        scope = "**必须是医学实体**，时间/数值/医务人员称谓/段落标题都不抽"

    neg_hints = _NEG_HINTS.get(dataset_name, "")

    return f"""你是 NER 标注规范专家。任务：观察 {ds_desc} 的真实标注样本，归纳出一套**让 LLM 在抽取时既能高召回又能过滤垃圾**的范式化规则。

# 任务范围
{scope}

# 正例（带金标的真实样本）
{demos_text}

# 易混淆反例提示（这些词在原文中常出现但 gold 不标，属于必须过滤的噪声）
{neg_hints}

# 你要产出的 skills（共 10-14 条，分 4 个段落）

【必抽规则】（5-6 条）：每条描述一种"看到 X 模式 → 必须抽出"的范式
  - 给具体模式描述（前缀/后缀/词性/语境）
  - 配一个原样从样本中观察到的实例
  - 重点关注容易被漏抽的类型（如 ite/pro/equ 在 CMeEE，痰/精神软/绿便 在 IMCS）

【过滤规则】（3-4 条）：每条描述一种"看到 X 模式 → 坚决不抽"的范式
  - 来自上方反例提示和你在 demos 中观察到的 gold 没有标的高频词
  - 描述判断逻辑（如"动词形态 + 名词" / "数值 + 单位" / "时间副词"）

【医学性判定】（2 条）：如何判别一个词是医学实体 vs 非医学日常词
  - 列出医学词的特征（专业术语后缀/解剖结构名/疾病分类前缀/...）
  - 列出非医学词的快速排除方法

【边界规则】（2-3 条）：实体的起止位置怎么切
  - 嵌套抽 vs 不抽
  - 前缀修饰语保留 vs 剥离
  - 后缀助词（了/的/吗）必须去掉

# 严格要求
1. 每条规则**1-2 句**，必须可被另一个 LLM 直接执行
2. 必须基于上方样本观察到的真实模式，**禁止凭空写**
3. 禁止抽象建议（如"要仔细分析"、"考虑上下文"）
4. **明确强调医学性**：任何规则不能让模型抽出非医学词
5. **明确强调过滤垃圾**：每条必抽规则配套思考"会不会误抽噪声"

# 输出格式（严格按段，无任何额外解释）
【必抽规则】
1. ...
2. ...
...
【过滤规则】
1. ...
...
【医学性判定】
1. ...
2. ...
【边界规则】
1. ...
..."""


def _parse_skills(raw: str) -> str:
    if not raw: return ""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"```[a-z]*|```", "", raw)
    raw = re.sub(r"^[^【0-9]*?(?=【|\d+[\.\、])", "", raw)
    return raw.strip()


# meta-prompt 改了就 bump 这个值，老缓存自动失效重生
SKILLS_SCHEMA_VERSION = "v2-structured"


def _self_verify_and_refine(skills: str, demos_text: str, dataset_name: str) -> str:
    """
    让 LLM 用刚生成的 skills 在第 1 个 demo 上 dry-run：
    - 模拟抽取 → 自检漏抽/误抽 → 修正 skills
    - 最多 1 轮（避免无限循环）

    输入 demos_text 里第一段就是 demo 1 包含原文+gold，足够 LLM 对照。
    """
    from src.llm_client import call_llm
    verify_prompt = f"""你刚为 {dataset_name} 写了一套 NER 抽取 skills。现在请按这套 skills 对【样本 1】做实战模拟：

# 你的 skills
{skills}

# 训练样本（你能看到 gold 答案）
{demos_text}

# 任务
1. 严格按照你写的 skills，对【样本 1】抽取实体（"模拟输出"）
2. 与 gold 对比，找出：
   - 漏抽：gold 有但模拟输出没有
   - 误抽：模拟输出有但 gold 没有
3. 如果漏抽/误抽涉及某种**模式**（不是个例），修改对应的 skills 条款（加强必抽 / 增加过滤）

# 输出格式（严格按段）
<dryrun>
模拟输出: ...
漏抽: ...
误抽: ...
问题模式: ...（如果有）
</dryrun>

<final_skills>
（修改后的完整 skills，保持原 4 段结构【必抽规则】【过滤规则】【医学性判定】【边界规则】。如果原 skills 已经够好就原样输出）
</final_skills>"""
    try:
        raw = call_llm(verify_prompt, max_tokens=2048, model_name="main")
        # 提取 <final_skills> 块
        m = re.search(r"<final_skills>([\s\S]*?)</final_skills>", raw, re.IGNORECASE)
        if m:
            refined = _parse_skills(m.group(1))
            if refined and len(refined) >= len(skills) * 0.6:
                print(f"  ✓ skills 自校验已细化")
                return refined
        print(f"  ℹ️ skills 自校验未产生有效修订，保留原版")
    except Exception as e:
        print(f"  ⚠️ skills 自校验失败（忽略）: {e}")
    return skills


def _llm_generate_skills(dataset_name: str, demos_text: str,
                        cache_path: str, force: bool = False) -> str:
    if not force and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if (cached.get("skills") and cached.get("dataset") == dataset_name
                    and cached.get("schema_version") == SKILLS_SCHEMA_VERSION):
                print(f"  ✓ 复用 LLM skills 缓存: {cache_path}")
                return cached["skills"]
            if cached.get("schema_version") != SKILLS_SCHEMA_VERSION:
                print(f"  ⚠️ skills schema 升级 ({cached.get('schema_version','无')} → {SKILLS_SCHEMA_VERSION})，重新生成…")
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
                       max_tokens=2048, model_name="main")
        skills = _parse_skills(raw)
        if not skills or len(skills) < 100:
            raise ValueError("LLM 输出过短")

        # === 自校验：让 LLM 用刚生成的 skills 在 1 个 demo 上 dry-run ===
        if os.environ.get("MNER_SKILLS_DRYRUN", "true").lower() == "true":
            print(f"  🔍 skills 自校验（用一个 demo dry-run）…")
            skills = _self_verify_and_refine(skills, demos_text, dataset_name)
    except Exception as e:
        print(f"  ⚠️ LLM skills 生成失败 ({e})，使用后备")
        skills = fallback.strip()

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset_name, "skills": skills,
                   "schema_version": SKILLS_SCHEMA_VERSION},
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

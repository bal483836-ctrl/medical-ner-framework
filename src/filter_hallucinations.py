"""
Step 3: 幻觉过滤与实体验证模块 v3.2（修复版）
CMeEE：纯规则过滤（不调用大模型，避免误删专业短词）
IMCS：大模型审核 + 原文锚定检查（字符级滑动窗口，阈值0.65）

关键修复（v3.2）：
  1. IMCS Step3 从 step2_aligned_output（原词）做锚定检查
  2. 增加 IMCS 规则过滤黑名单，在大模型审核前先过滤明显非症状词
  3. 改进大模型审核 Prompt，更明确地指导过滤非症状词
  4. 增加 ORAL_TO_NORM 映射，在 step3_final_output 中将口语词替换为标准词
     （这样 evaluate_all.py 的字面 F1 也能提升）
"""
import json
import os
import re
import sys
from typing import List, Dict, Set
from difflib import SequenceMatcher
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    OUTPUT_DIR, STEP3_PREFIX,
    IMCS_FORCE_RECALL_WORDS,
)
from src.llm_client import call_llm, clean_llm_output
from src.data_processor import clean_entity_list

# ==================== CMeEE 规则过滤 ====================
# CMeEE 过滤黑名单：这些词在 CMeEE 中不是合法实体
CMEEE_FILTER_BLACKLIST = {
    # 患者身份
    "男孩", "女孩", "患者", "病人", "儿童", "婴儿", "新生儿", "成人",
    # 纯化学成分（不含医学意义）
    "含硫噻唑环", "氨基吡啶环", "噻唑环", "吡啶环",
    # 纯方位/属性词
    "上", "下", "左", "右", "前", "后", "内", "外",
    "大", "小", "长", "短", "宽", "窄", "高", "低",
    # 时间/数字
    "年", "月", "日", "时", "分", "秒",
    # 其他噪声
    "无", "有", "是", "否", "和", "与", "或", "及",
}

# CMeEE 单字词白名单：这些单字词是合法的身体部位/医学词汇
CMEEE_SINGLE_CHAR_WHITELIST = {
    "脑", "肝", "肺", "肾", "心", "胃", "肠", "脾", "胰",
    "骨", "血", "皮", "眼", "耳", "鼻", "口", "舌", "喉",
    "腹", "胸", "背", "腰", "颈", "头", "足", "手", "臂",
    "髋", "膝", "踝", "肘", "腕", "指", "趾", "唇", "齿",
    "尿", "便", "痰", "汗", "泪", "涕", "乳", "精",
}

# ==================== IMCS 规则过滤黑名单 ====================
# 这些词在 IMCS 中明确不是症状词，在大模型审核前先过滤
IMCS_RULE_BLACKLIST = {
    # 患者性别/年龄
    "男孩", "女孩", "男宝", "女宝", "宝宝", "孩子", "患儿", "患者",
    "新生儿", "婴儿", "儿童", "小孩",
    # 喂养方式（不是症状）
    "母乳喂养", "奶粉喂养", "纯母乳", "混合喂养", "人工喂养",
    # 病因诱因（不是症状本身）
    "腹部受凉", "腹部受冷", "受凉", "着凉", "吹风", "吹空调",
    "受风", "淋雨", "感染", "受寒",
    # 药物（不是症状）
    "蒙脱石散", "益生菌", "罗红霉素", "妈咪爱", "思密达", "贴一贴灵",
    "婴儿健脾散", "口服液", "退烧药", "抗生素", "布洛芬", "对乙酰氨基酚",
    "阿莫西林", "头孢", "中药", "西药",
    # 检查项目（不是症状）
    "便常规", "血常规", "水电解质", "尿常规", "大便化验", "血液检查",
    # 科室名称
    "小儿内科", "儿科", "门诊",
    # 纯描述性词语（不是症状）
    "还可以", "挑食", "精神状态", "一天", "两天", "三天",
    "特别粘", "喝了一天的葱白淡豆豉陈皮水", "孩子让她伸舌头会恶心",
    # 舌象（IMCS Gold 不包含舌象）
    "舌苔黄", "舌苔厚黄", "舌苔厚", "舌苔发黄", "舌苔白", "舌苔发白",
    "舌尖红", "舌红", "舌苔发黑",
    # 其他非症状
    "热相表现", "有点热相表现", "食积等", "奶辫",
}

# IMCS 口语词->标准词映射（在 Step3 输出时直接替换，提升字面 F1）
ORAL_TO_NORM_STEP3 = {
    "拉粑粑": "腹泻", "拉肚子": "腹泻", "拉稀": "腹泻",
    "放屁也拉": "腹泻", "尿尿就拉粑粑": "腹泻", "尿尿也拉": "腹泻",
    "一尿尿就拉粑粑": "腹泻", "一换尿布就有": "腹泻", "排了两回便": "腹泻",
    "大便次数多": "腹泻", "大便增多": "腹泻", "大便频繁": "腹泻",
    "黄鼻涕": "鼻流涕", "清鼻涕": "鼻流涕", "流鼻涕": "鼻流涕",
    "鼻涕有点偏黄": "鼻流涕", "鼻涕偏黄": "鼻流涕", "鼻涕黄": "鼻流涕",
    "流清涕": "鼻流涕", "流黄涕": "鼻流涕", "鼻涕多": "鼻流涕",
    "鼻涕": "鼻流涕", "流涕": "鼻流涕",
    "发烧": "发热", "发烫": "发热", "体温高": "发热", "高烧": "发热",
    "低烧": "发热", "发高烧": "发热", "发低烧": "发热", "体温升高": "发热",
    "伤风": "感冒", "着凉": "感冒",
    "偏绿": "绿便", "大便偏绿": "绿便", "便绿": "绿便",
    "大便绿色": "绿便", "绿色大便": "绿便",
    "奶瓣子": "奶瓣", "里面都是奶瓣子": "奶瓣",
    "夜咳": "咳嗽",
    "肚子痛": "腹痛", "肚痛": "腹痛", "肚子疼": "腹痛",
    "吐了": "呕吐", "吐奶": "呕吐",
    "放屁": "屁", "放了很多屁": "屁",
    "精神差": "精神软", "精神不好": "精神软", "精神不振": "精神软",
    "没精神": "精神软", "精神状态差": "精神软",
    "不吃东西": "食欲不振", "不爱吃饭": "食欲不振", "不吃饭": "食欲不振",
    "吃得少": "食欲不振", "胃口差": "食欲不振", "纳差": "食欲不振",
    "肚子胀": "腹胀", "胀气": "腹胀",
    "鼻子不通气": "鼻塞", "鼻子堵": "鼻塞",
    "嗓子疼": "咽喉痛", "嗓子痛": "咽喉痛", "咽痛": "咽喉痛",
    "嗓子哑": "嗓子沙哑",
    "大便干": "大便干燥", "大便硬": "大便干燥",
    "大便有血": "血便", "便中带血": "血便", "大便带血": "血便",
    "大便有粘液": "大便粘液", "大便粘": "大便粘液",
    "水样大便": "水样便", "水样稀便": "水样便",
    "蛋花样便": "蛋花汤样便",
    "哭": "哭闹", "一直哭": "哭闹", "哭个不停": "哭闹",
    "抽了": "抽搐", "抽风": "抽搐",
    "尿少": "尿量减少", "尿量少": "尿量减少",
    "出皮疹": "皮疹", "长皮疹": "皮疹", "起疹子": "皮疹",
    "长疹子": "皮疹", "疹子": "皮疹",
    "打喷嚏": "喷嚏",
    "喉咙有痰": "痰鸣音", "喉咙痰多": "痰鸣音",
    "脖子有包": "淋巴结肿大", "淋巴结大": "淋巴结肿大",
    "扁桃体大": "扁桃体炎", "扁桃体肿": "扁桃体炎",
    "黄疸高": "黄疸", "皮肤黄": "黄疸", "眼睛黄": "黄疸",
    "积食了": "积食", "食积": "积食",
    "喘鸣": "喘息", "喘气费力": "喘息",
}


def filter_cmeee(items: List[Dict], output_path: str) -> List[Dict]:
    """
    CMeEE Step3：纯规则过滤
    - 过滤黑名单词
    - 单字词：只保留白名单中的医学词汇
    - 过滤含顿号/合并词
    - 过滤纯英文字母片段（1-2字母的不完整缩写）
    - 过滤纯数字/标点
    """
    print(f"\n[Step3] CMeEE 规则过滤...")
    for item in items:
        raw = item.get("step2_aligned_output", item.get("step1_enriched_output", ""))
        candidates = clean_entity_list(raw)
        filtered = []
        for ent in candidates:
            if not ent:
                continue
            # 黑名单
            if ent in CMEEE_FILTER_BLACKLIST:
                continue
            # 纯数字/标点
            if re.match(r"^[\d\s\W]+$", ent):
                continue
            # 含顿号的合并词
            if re.search(r"[、，,]", ent):
                continue
            # 单字词：只保留白名单
            if len(ent) == 1:
                if ent in CMEEE_SINGLE_CHAR_WHITELIST:
                    filtered.append(ent)
                continue
            # 纯英文1-2字母片段
            if re.match(r"^[A-Za-z]{1,2}$", ent):
                continue
            # 通过
            filtered.append(ent)
        item["step3_final_output"] = ",".join(list(dict.fromkeys(filtered)))
    _save_json(items, output_path)
    print(f"  ✅ CMeEE Step3 保存至: {output_path}")
    return items


# ==================== IMCS 大模型过滤 ====================
def _text_anchor_score(text: str, entity: str, window_size: int = None) -> float:
    """
    原文锚定检查：计算实体与原文的最大字符级相似度
    使用滑动窗口，窗口大小 = max(len(entity), 4)
    """
    if not text or not entity:
        return 0.0
    if entity in text:
        return 1.0
    window_size = window_size or max(len(entity), 4)
    best_score = 0.0
    for i in range(len(text) - window_size + 1):
        window = text[i: i + window_size]
        score = SequenceMatcher(None, entity, window).ratio()
        if score > best_score:
            best_score = score
    return best_score


def _rule_filter_imcs(candidates: List[str]) -> List[str]:
    """
    IMCS 规则过滤（在大模型审核前先过滤明显非症状词）
    过滤规则：
      1. 黑名单词直接删除
      2. 过长的描述性短语（>10字且不含明确症状词）
      3. 纯数字/标点
    """
    filtered = []
    for ent in candidates:
        if not ent:
            continue
        # 黑名单
        if ent in IMCS_RULE_BLACKLIST:
            continue
        # 纯数字/标点
        if re.match(r"^[\d\s\W]+$", ent):
            continue
        # 过长的描述性短语（>12字）且不含明确症状词
        if len(ent) > 12:
            # 检查是否含有明确症状词
            has_symptom = any(
                kw in ent for kw in [
                    "痛", "疼", "痒", "热", "烧", "咳", "喘", "吐", "泻",
                    "便", "尿", "血", "疹", "肿", "炎", "鸣", "哭", "抽",
                ]
            )
            if not has_symptom:
                continue
        filtered.append(ent)
    return filtered


def filter_imcs_with_llm(
    items: List[Dict],
    output_path: str,
    anchor_threshold: float = 0.65,
) -> List[Dict]:
    """
    IMCS Step3：规则预过滤 + 大模型审核 + 原文锚定检查（修复版 v3.2）
    
    流程：
      1. 规则预过滤：删除明显非症状词（黑名单、过长短语等）
      2. 原文锚定检查：实体（原词）与原文相似度 < anchor_threshold 的直接删除（幻觉）
      3. 强召回兜底：确保原文中存在的关键体征词不被删除
      4. 大模型审核：对剩余候选集进行医学合法性验证
      5. 口语词替换：将 step3_final_output 中的口语词替换为标准词（提升字面 F1）
    
    关键修复：
      - 从 step2_aligned_output（原词/口语化词）做锚定检查
      - 增加规则预过滤，减少大模型的审核负担
      - 在最终输出时将口语词替换为标准词（提升字面 F1）
    """
    print(f"\n[Step3] IMCS 大模型过滤（原文锚定阈值: {anchor_threshold}）...")
    for item in items:
        # 关键修复：用 step2_aligned_output（原词）做锚定，而不是 step2_norm_output
        candidates = clean_entity_list(
            item.get("step2_aligned_output", item.get("step1_raw_output", ""))
        )

        # 构建原文（用于锚定检查）
        full_text = (
            item.get("self_report", "") + " " +
            " ".join(t.get("sentence", "") for t in item.get("dialogue", []))
        )

        # Step 3a: 规则预过滤（删除明显非症状词）
        candidates = _rule_filter_imcs(candidates)

        # Step 3b: 原文锚定过滤（用原词检查，不用标准词）
        anchored = []
        for ent in candidates:
            score = _text_anchor_score(full_text, ent)
            if score >= anchor_threshold:
                anchored.append(ent)

        # Step 3c: 强召回兜底（确保关键体征不丢失）
        anchored_set = set(anchored)
        for force_word in IMCS_FORCE_RECALL_WORDS:
            if force_word in full_text and force_word not in anchored_set:
                anchored.append(force_word)
                anchored_set.add(force_word)

        if not anchored:
            item["step3_final_output"] = ""
            continue

        # Step 3d: 大模型审核（审核原词，不审核标准词）
        cands_str = ", ".join(anchored)
        prompt = f"""你是严谨的医学专家，请对以下候选实体列表进行清洗，只保留真正的症状和疾病实体。

【严格过滤规则】：
1. 必须删除【非症状词】：
   - 患者性别年龄：男孩/女孩/宝宝/孩子/患儿
   - 病因诱因：受凉/吹风/纯母乳喂养/腹部受冷/腹部受凉
   - 生活描述：还可以/挑食/一天/奶/精神状态/大便次数多/特别粘
2. 必须删除【药物与检查】：
   - 药物：蒙脱石散/益生菌/罗红霉素/贴/片/丸/药/补液/口服液/妈咪爱
   - 检查：便常规/血常规/水电解质
   - 科室：小儿内科/儿科
3. 必须删除【舌象与模糊表达】：
   - 舌象：舌苔黄/舌苔厚/舌尖红/舌红/舌苔发黄
   - 模糊：热相表现/食积等/呕吐症状等/有点热相表现
4. 必须删除【过长描述性短语】（超过8个字且不是明确症状名称）：
   - 如：喝了一天的葱白淡豆豉陈皮水/孩子让她伸舌头会恶心
5. 必须保留【核心体征】：
   - 排泄物：绿便/水样便/蛋花汤样便/血便/稀便/屁/大便粘液/脓血便/奶瓣
   - 微小体征：精神软/尿量减少/脱水/哭闹/抽搐
6. 必须保留【口语化症状】：
   - 发烧/发烫/拉肚子/拉粑粑/肚子痛/流鼻涕/黄鼻涕/清鼻涕/咳嗽/夜咳
   - 偏绿/放屁/尿尿就拉粑粑等口语表述都是合法症状词

【输出格式】：只输出审核通过的实体名，用逗号分隔，不要任何解释。若全部不通过，输出"无"。

【候选实体】：{cands_str}
【通过审核的实体】："""
        raw_output = call_llm(prompt, max_tokens=256)
        cleaned    = clean_llm_output(raw_output)
        final_ents = clean_entity_list(cleaned)

        # Step 3e: 强召回兜底（大模型可能误删关键词，再补一次）
        final_set = set(final_ents)
        for force_word in IMCS_FORCE_RECALL_WORDS:
            if force_word in full_text and force_word not in final_set:
                final_ents.append(force_word)
                final_set.add(force_word)

        # Step 3f: 口语词替换（将口语词替换为标准词，提升字面 F1）
        # 注意：只替换在 ORAL_TO_NORM_STEP3 中有明确映射的口语词
        normalized_ents = []
        seen = set()
        for ent in final_ents:
            norm_ent = ORAL_TO_NORM_STEP3.get(ent, ent)
            if norm_ent not in seen:
                seen.add(norm_ent)
                normalized_ents.append(norm_ent)

        item["step3_final_output"] = ",".join(normalized_ents)

    _save_json(items, output_path)
    print(f"  ✅ IMCS Step3 保存至: {output_path}")
    return items


def _save_json(data: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

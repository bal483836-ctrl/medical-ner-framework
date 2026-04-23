"""
Step 3: 幻觉过滤与实体验证模块 v3（修复版）
CMeEE：纯规则过滤（不调用大模型，避免误删专业短词）
IMCS：大模型审核 + 原文锚定检查（字符级滑动窗口，阈值0.65）

关键修复（v3.1）：
  IMCS Step3 现在从 step2_aligned_output（原词）做锚定检查，
  而不是从 step2_norm_output（标准词）做锚定检查。
  这样归一化词（如"腹泻"）不会因为原文中只有"拉肚子"而被误删。
  Step3 输出的 step3_final_output 仍然是原词（口语化），
  Step4 再通过 step2_normalized_map 做最终归一化评估。
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


def filter_imcs_with_llm(
    items: List[Dict],
    output_path: str,
    anchor_threshold: float = 0.65,
) -> List[Dict]:
    """
    IMCS Step3：大模型审核 + 原文锚定检查（修复版）

    关键修复：
      - 从 step2_aligned_output（原词/口语化词）做锚定检查
      - 不再从 step2_norm_output（标准词）做锚定检查
      - 这样"拉肚子"可以通过锚定检查，而不会因为原文没有"腹泻"而被删除
      - step3_final_output 输出原词，Step4 再通过 step2_normalized_map 归一化

    流程：
      1. 原文锚定检查：实体（原词）与原文相似度 < anchor_threshold 的直接删除（幻觉）
      2. 强召回兜底：确保原文中存在的关键体征词不被删除
      3. 大模型审核：对剩余候选集进行医学合法性验证
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

        # Step 3a: 原文锚定过滤（用原词检查，不用标准词）
        anchored = []
        for ent in candidates:
            score = _text_anchor_score(full_text, ent)
            if score >= anchor_threshold:
                anchored.append(ent)

        # Step 3b: 强召回兜底（确保关键体征不丢失）
        anchored_set = set(anchored)
        for force_word in IMCS_FORCE_RECALL_WORDS:
            if force_word in full_text and force_word not in anchored_set:
                anchored.append(force_word)
                anchored_set.add(force_word)

        if not anchored:
            item["step3_final_output"] = ""
            continue

        # Step 3c: 大模型审核（审核原词，不审核标准词）
        cands_str = ", ".join(anchored)
        prompt = f"""你是严谨的医学专家，请对以下候选实体列表进行清洗，只保留真正的症状和疾病实体。

【审核准则】：
1. 剔除【非症状词】：患者性别(男孩/女孩)、病因诱因(受凉/吹风/纯母乳喂养/腹部受冷)、生活废话(还可以/挑食/一天/奶/宝宝/精神状态/大便次数多)。
2. 剔除【药物与检查】：所有药物名(蒙脱石散/益生菌/罗红霉素/贴/片/丸/药/补液/口服液)、检查项目(便常规/水电解质)、科室名称(小儿内科)。
3. 剔除【模糊表达】：剔除"食积等"、"热相表现"、"呕吐症状等"等不明确词汇，剔除单纯舌象(舌苔黄/舌红)。
4. 强制保留【核心体征】：必须保留排泄物性状(绿便/水样便/蛋花汤样便/血便/稀便/屁/大便粘液/脓血便)和微小体征(精神软/尿量减少/脱水/哭闹/抽搐)。
5. 强制保留【口语化症状】：保留发烧/发烫/拉肚子/肚子痛/流鼻涕/咳嗽等口语表述，这些都是合法症状词。
6. 输出格式：只输出审核通过的实体名，用逗号分隔，不要任何解释。若全部不通过，输出"无"。

【候选实体】：{cands_str}
【通过审核的实体】："""

        raw_output = call_llm(prompt, max_tokens=256)
        cleaned    = clean_llm_output(raw_output)
        final_ents = clean_entity_list(cleaned)

        # 强召回兜底（大模型可能误删关键词，再补一次）
        final_set = set(final_ents)
        for force_word in IMCS_FORCE_RECALL_WORDS:
            if force_word in full_text and force_word not in final_set:
                final_ents.append(force_word)
                final_set.add(force_word)

        item["step3_final_output"] = ",".join(list(dict.fromkeys(final_ents)))

    _save_json(items, output_path)
    print(f"  ✅ IMCS Step3 保存至: {output_path}")
    return items


def _save_json(data: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

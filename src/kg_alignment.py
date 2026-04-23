"""
Step 2: 图谱对齐与余弦相似度匹配模块 v3.2（修复版）
三级匹配策略：
  1. 精确匹配（score=1.0）：直接输出
  2. 高相似度（>=HIGH_SIM_THRESHOLD=0.72）：接受归一化
  3. 低相似度（<LOW_SIM_THRESHOLD=0.55）：转交 Step3 大模型验证

关键修复（v3.2）：
  - 增加手工口语化映射词典 ORAL_TO_NORM，覆盖向量模型无法处理的口语->标准词映射
    例如：拉粑粑->腹泻, 黄鼻涕->鼻流涕, 伤风->感冒, 发烧->发热 等
  - 将 HIGH_SIM_THRESHOLD 从 0.82 降低到 0.72，捕获更多口语->标准词的映射
  - IMCS Step2 保留原词（供 Step3 原文锚定使用），同时生成 normalized_map（供 Step4 归一化评估）
"""
import json
import os
import sys
from typing import List, Dict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    OUTPUT_DIR, STEP2_PREFIX,
    HIGH_SIM_THRESHOLD, LOW_SIM_THRESHOLD,
)
from src.data_processor import clean_entity_list
from src.embedding_model import find_best_matches

# ==================== 手工口语词->标准词映射 ====================
# 覆盖 bge-large-zh-v1.5 向量模型无法处理的口语->标准词映射
# 优先级高于向量相似度匹配
ORAL_TO_NORM = {
    # 腹泻类
    "拉粑粑": "腹泻", "拉肚子": "腹泻", "拉稀": "腹泻",
    "放屁也拉": "腹泻", "尿尿就拉粑粑": "腹泻", "尿尿也拉": "腹泻",
    "一尿尿就拉粑粑": "腹泻", "一换尿布就有": "腹泻", "排了两回便": "腹泻",
    "大便次数多": "腹泻", "大便增多": "腹泻", "大便频繁": "腹泻",
    "拉了好几次": "腹泻", "频繁拉肚子": "腹泻", "腹泻症状": "腹泻",
    # 鼻流涕类
    "黄鼻涕": "鼻流涕", "清鼻涕": "鼻流涕", "流鼻涕": "鼻流涕",
    "鼻涕有点偏黄": "鼻流涕", "鼻涕偏黄": "鼻流涕", "鼻涕黄": "鼻流涕",
    "流清涕": "鼻流涕", "流黄涕": "鼻流涕", "鼻涕多": "鼻流涕",
    "鼻涕": "鼻流涕", "流涕": "鼻流涕",
    # 发热类
    "发烧": "发热", "发烫": "发热", "体温高": "发热", "高烧": "发热",
    "低烧": "发热", "烧": "发热", "发高烧": "发热", "发低烧": "发热",
    "体温升高": "发热", "热": "发热",
    # 感冒类
    "伤风": "感冒", "着凉": "感冒", "受凉": "感冒",
    # 绿便类
    "偏绿": "绿便", "大便偏绿": "绿便", "便绿": "绿便",
    "大便绿色": "绿便", "绿色大便": "绿便",
    # 奶瓣类
    "奶瓣子": "奶瓣", "里面都是奶瓣子": "奶瓣",
    # 咳嗽类
    "夜咳": "咳嗽", "干咳嗽": "干咳", "咳嗽有痰": "咳痰",
    "咳嗽咳痰": "咳痰", "有痰咳嗽": "咳痰",
    # 腹痛类
    "肚子痛": "腹痛", "肚痛": "腹痛", "胃痛": "腹痛",
    "肚子不舒服": "腹痛", "肚子疼": "腹痛",
    # 呕吐类
    "吐了": "呕吐", "吐奶": "呕吐", "恶心呕吐": "呕吐",
    # 屁类
    "放屁": "屁", "放了很多屁": "屁",
    # 其他
    "精神差": "精神软", "精神不好": "精神软", "精神不振": "精神软",
    "没精神": "精神软", "精神状态差": "精神软",
    "不吃东西": "食欲不振", "不爱吃饭": "食欲不振", "不吃饭": "食欲不振",
    "吃得少": "食欲不振", "胃口差": "食欲不振", "纳差": "食欲不振",
    "肚子胀": "腹胀", "胀气": "腹胀",
    "喘不上气": "呼吸困难", "呼吸急": "呼吸急促",
    "鼻子不通气": "鼻塞", "鼻子堵": "鼻塞",
    "嗓子疼": "咽喉痛", "嗓子痛": "咽喉痛", "咽痛": "咽喉痛",
    "嗓子不舒服": "咽喉不适", "嗓子哑": "嗓子沙哑",
    "大便干": "大便干燥", "大便硬": "大便干燥", "便秘了": "便秘",
    "不拉大便": "便秘", "几天没大便": "便秘",
    "皮肤痒": "痒", "身上痒": "痒", "皮疹痒": "痒",
    "眼睛红": "结膜炎", "眼红": "结膜炎",
    "耳朵疼": "中耳炎", "耳痛": "中耳炎",
    "口腔有溃疡": "口腔溃疡", "嘴里有溃疡": "口腔溃疡",
    "大便有血": "血便", "便中带血": "血便", "大便带血": "血便",
    "大便有粘液": "大便粘液", "大便粘": "大便粘液",
    "水样大便": "水样便", "水样稀便": "水样便",
    "蛋花样便": "蛋花汤样便",
    "哭": "哭闹", "一直哭": "哭闹", "哭个不停": "哭闹",
    "抽了": "抽搐", "抽风": "抽搐", "惊厥了": "惊厥",
    "尿少": "尿量减少", "尿量少": "尿量减少",
    "出皮疹": "皮疹", "长皮疹": "皮疹", "起疹子": "皮疹",
    "长疹子": "皮疹", "疹子": "皮疹",
    "打喷嚏": "喷嚏", "喷嚏不断": "喷嚏",
    "喉咙有痰": "痰鸣音", "喉咙痰多": "痰鸣音",
    "呼吸有声音": "痰鸣音", "呼噜声": "痰鸣音",
    "脖子有包": "淋巴结肿大", "淋巴结大": "淋巴结肿大",
    "扁桃体大": "扁桃体炎", "扁桃体肿": "扁桃体炎",
    "口腔有白点": "鹅口疮", "嘴里有白点": "鹅口疮",
    "手脚起泡": "手足口病", "口腔起泡": "手足口病",
    "黄疸高": "黄疸", "皮肤黄": "黄疸", "眼睛黄": "黄疸",
    "积食了": "积食", "食积": "积食",
    "肚子有气": "腹胀", "肚子鼓": "腹胀",
    "喘鸣": "喘息", "喘气费力": "喘息",
}

# 降低 IMCS 归一化的相似度阈值（从 0.82 降到 0.72，捕获更多口语->标准词映射）
IMCS_HIGH_SIM_THRESHOLD = 0.72
IMCS_LOW_SIM_THRESHOLD  = 0.55


def align_cmeee_split(
    items: List[Dict],
    vocab: List[str],
    output_path: str,
) -> List[Dict]:
    """
    CMeEE Step2：对 step1_enriched_output 中的实体做图谱对齐
    - 精确匹配：直接保留
    - 高相似度：接受归一化，但扩展词必须通过原文锚定检查
    - 低相似度：保留原词，交给 Step3 大模型验证
    """
    print(f"\n[Step2] CMeEE 图谱对齐，词汇表大小: {len(vocab)}")
    vocab_set = set(vocab)
    for item in items:
        text = item.get("text", "")
        raw_entities = clean_entity_list(
            item.get("step1_enriched_output", item.get("step1_raw_output", ""))
        )
        if not raw_entities:
            item["step2_aligned_output"] = ""
            continue
        # 精确匹配先处理
        exact_matched = []
        needs_embed   = []
        for ent in raw_entities:
            if ent in vocab_set:
                exact_matched.append(ent)
            else:
                needs_embed.append(ent)
        # 向量相似度匹配
        aligned_set = set(exact_matched)
        if needs_embed and vocab:
            matches = find_best_matches(needs_embed, vocab)
            for ent, (best_match, score, status) in zip(needs_embed, matches):
                if status in ("exact", "high"):
                    # 原文锚定检查：扩展词必须在原文中出现
                    if best_match in text:
                        aligned_set.add(best_match)
                    elif ent in text:
                        aligned_set.add(ent)
                    # 两者都不在原文中，丢弃
                else:
                    # 低相似度：保留原词，交给 Step3
                    if ent in text:
                        aligned_set.add(ent)
        item["step2_aligned_output"] = ",".join(list(aligned_set))
    _save_json(items, output_path)
    print(f"  ✅ CMeEE Step2 保存至: {output_path}")
    return items


def align_imcs_split(
    items: List[Dict],
    norm_vocab: List[str],
    output_path: str,
) -> List[Dict]:
    """
    IMCS Step2：将口语化实体对齐到官方 symptom_norm 标准词（修复版 v3.2）
    
    归一化优先级：
      1. 手工口语词映射（ORAL_TO_NORM）：优先级最高，覆盖向量模型无法处理的口语词
      2. 精确匹配：实体直接在标准词表中
      3. 向量高相似度（>=0.72）：接受归一化
      4. 向量中等相似度（>=0.55）：接受归一化（降低阈值，捕获更多口语词）
      5. 低相似度：保留原词，Step4 再次尝试归一化
    
    关键设计（修复版）：
      - step2_aligned_output：保留原词（口语化表述），供 Step3 原文锚定使用
      - step2_normalized_map：原词->标准词映射，供 Step4 归一化评估使用
      - step2_norm_output：标准词集合，供 Step4 直接读取
    """
    print(f"\n[Step2] IMCS 图谱对齐，归一化词汇表大小: {len(norm_vocab)}")
    norm_vocab_set = set(norm_vocab)
    oral_mapped_count = 0
    vector_mapped_count = 0

    for item in items:
        raw_entities = clean_entity_list(item.get("step1_raw_output", ""))
        if not raw_entities:
            item["step2_aligned_output"] = ""
            item["step2_normalized_map"] = {}
            item["step2_norm_output"]    = ""
            continue

        # 构建归一化映射表（原词 -> 标准词）
        norm_map: Dict[str, str] = {}
        needs_embed = []

        for ent in raw_entities:
            # 第一优先级：手工口语词映射
            if ent in ORAL_TO_NORM:
                norm_map[ent] = ORAL_TO_NORM[ent]
                oral_mapped_count += 1
            # 第二优先级：精确匹配标准词表
            elif ent in norm_vocab_set:
                norm_map[ent] = ent
            # 否则需要向量匹配
            else:
                needs_embed.append(ent)

        # 向量相似度匹配（使用降低后的阈值）
        if needs_embed and norm_vocab:
            matches = find_best_matches(needs_embed, norm_vocab)
            for ent, (best_match, score, status) in zip(needs_embed, matches):
                # 使用降低后的阈值（0.72 而非 0.82）
                if score >= IMCS_HIGH_SIM_THRESHOLD:
                    norm_map[ent] = best_match
                    vector_mapped_count += 1
                elif score >= IMCS_LOW_SIM_THRESHOLD:
                    # 中等相似度也接受（0.55-0.72 之间）
                    norm_map[ent] = best_match
                    vector_mapped_count += 1
                else:
                    # 低相似度：保留原词（Step4 再次尝试归一化）
                    norm_map[ent] = ent

        # step2_aligned_output：保留原词（供 Step3 原文锚定）
        item["step2_aligned_output"] = ",".join(raw_entities)
        # step2_normalized_map：原词->标准词映射（供 Step4 归一化评估）
        item["step2_normalized_map"] = norm_map
        # step2_norm_output：标准词集合（供 Step4 直接读取）
        norm_set = list(dict.fromkeys(norm_map.values()))
        item["step2_norm_output"] = ",".join(norm_set)

    print(f"  [统计] 手工口语词映射: {oral_mapped_count} 次，向量映射: {vector_mapped_count} 次")
    _save_json(items, output_path)
    print(f"  ✅ IMCS Step2 保存至: {output_path}")
    return items


def _save_json(data: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

"""
本地验证修复效果：用诊断数据中的样本测试 IMCS 归一化逻辑
"""
import sys
sys.path.insert(0, '/home/ubuntu/medical_ner_v3')

# 模拟诊断数据中的两个 IMCS 样本
test_items = [
    {
        "_record_id": "10312337",
        "gold_entities_str": "屁,绿便,感冒,腹泻,消化不良",
        "step1_raw_output": "拉粑粑,屁,放屁也拉,尿尿就拉粑粑,消化不良,奶辫,母乳喂养,拉肚子,尿尿也拉,感冒,放屁,偏绿,奶粉喂养,腹部受凉,一换尿布就有,尿尿,里面都是奶瓣子,腹部受冷,奶瓣子,一尿尿就拉粑粑",
        "step2_aligned_output": "拉粑粑,屁,放屁也拉,尿尿就拉粑粑,消化不良,奶辫,母乳喂养,拉肚子,尿尿也拉,感冒,放屁,偏绿,奶粉喂养,腹部受凉,一换尿布就有,尿尿,里面都是奶瓣子,腹部受冷,奶瓣子,一尿尿就拉粑粑",
        "step2_normalized_map": {"屁": "屁", "消化不良": "消化不良", "感冒": "感冒", "拉粑粑": "拉粑粑", "放屁也拉": "放屁也拉"},
        "step3_final_output": "拉粑粑,屁,放屁也拉,尿尿就拉粑粑,拉肚子,尿尿也拉,放屁,偏绿,尿尿,奶瓣子,一尿尿就拉粑粑,消化不良",
    },
    {
        "_record_id": "10589886",
        "gold_entities_str": "鼻流涕,咳嗽,感冒,鼻塞",
        "step1_raw_output": "黄鼻涕,舌苔厚,伤风,发烧,鼻塞,发热,舌尖红,清鼻涕,喝了一天的葱白淡豆豉陈皮水,有点热相表现,恶心,排了两回便,罗红霉素,鼻涕有点偏黄,咳嗽,夜咳,孩子让她伸舌头会恶心,舌苔发黄,特别粘",
        "step2_aligned_output": "黄鼻涕,舌苔厚,伤风,发烧,鼻塞,发热,舌尖红,清鼻涕,喝了一天的葱白淡豆豉陈皮水,有点热相表现,恶心,排了两回便,罗红霉素,鼻涕有点偏黄,咳嗽,夜咳,孩子让她伸舌头会恶心,舌苔发黄,特别粘",
        "step2_normalized_map": {"鼻塞": "鼻塞", "发热": "发热", "恶心": "恶心", "咳嗽": "咳嗽"},
        "step3_final_output": "黄鼻涕,发烧,鼻塞,发热,清鼻涕,恶心,咳嗽,夜咳,排了两回便,鼻涕有点偏黄",
    },
]

# 加载 symptom_norm 词典
import csv
norm_vocab = []
with open('/home/ubuntu/medical_ner_v3/data/symptom_norm.csv', 'r') as f:
    for line in f:
        line = line.strip()
        if line and line not in ('norm', 'symptom_norm', '#'):
            norm_vocab.append(line.split(',')[0].strip())
norm_vocab_set = set(norm_vocab)
print(f"词典大小: {len(norm_vocab)}")

# 测试 evaluate_all.py 的 ORAL_TO_NORM 映射
ORAL_TO_NORM = {
    "拉粑粑": "腹泻", "拉肚子": "腹泻", "拉稀": "腹泻",
    "放屁也拉": "腹泻", "尿尿就拉粑粑": "腹泻", "尿尿也拉": "腹泻",
    "一尿尿就拉粑粑": "腹泻", "一换尿布就有": "腹泻", "排了两回便": "腹泻",
    "黄鼻涕": "鼻流涕", "清鼻涕": "鼻流涕", "流鼻涕": "鼻流涕",
    "鼻涕有点偏黄": "鼻流涕", "鼻涕偏黄": "鼻流涕",
    "发烧": "发热", "发烫": "发热",
    "伤风": "感冒",
    "偏绿": "绿便",
    "奶瓣子": "奶瓣",
    "夜咳": "咳嗽",
    "放屁": "屁",
}

import re
def clean_entity_list(raw):
    if not raw:
        return []
    raw = re.sub(r"\s+", "", raw)
    items = re.split(r"[,，、；;]", raw)
    result = []
    for item in items:
        item = item.strip().strip('"\'""\'\'')
        if item and item not in ("无", "null", "None", "none", "NULL"):
            result.append(item)
    return result

def compute_f1(gold, pred):
    gold_set = set(gold)
    pred_set = set(pred)
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1, tp, fp, fn

print("\n" + "="*60)
print("测试 IMCS 归一化效果（使用 evaluate_all.py 的 ORAL_TO_NORM）")
print("="*60)

for i, item in enumerate(test_items):
    gold = clean_entity_list(item["gold_entities_str"])
    pred_raw = clean_entity_list(item["step3_final_output"])
    norm_map = item.get("step2_normalized_map", {})
    
    # 归一化
    normalized = []
    for ent in pred_raw:
        if ent in ORAL_TO_NORM:
            normalized.append(ORAL_TO_NORM[ent])
        elif ent in norm_map:
            mapped = norm_map[ent]
            if mapped in ORAL_TO_NORM:
                normalized.append(ORAL_TO_NORM[mapped])
            else:
                normalized.append(mapped)
        elif ent in norm_vocab_set:
            normalized.append(ent)
        else:
            # 部分匹配
            matched = False
            for norm_word in norm_vocab:
                if norm_word in ent or ent in norm_word:
                    if len(norm_word) >= 2 and len(ent) >= 2:
                        normalized.append(norm_word)
                        matched = True
                        break
            if not matched:
                normalized.append(ent)
    
    normalized = list(dict.fromkeys(normalized))
    
    p_l, r_l, f1_l, tp_l, fp_l, fn_l = compute_f1(gold, pred_raw)
    p_n, r_n, f1_n, tp_n, fp_n, fn_n = compute_f1(gold, normalized)
    
    print(f"\n样本{i} (id={item['_record_id']}):")
    print(f"  gold: {gold}")
    print(f"  pred（字面）: {pred_raw}")
    print(f"  pred（归一化）: {normalized}")
    print(f"  字面 F1: {f1_l:.4f} (P={p_l:.4f}, R={r_l:.4f}, TP={tp_l}, FP={fp_l}, FN={fn_l})")
    print(f"  归一化 F1: {f1_n:.4f} (P={p_n:.4f}, R={r_n:.4f}, TP={tp_n}, FP={fp_n}, FN={fn_n})")
    
    # 显示 TP/FP/FN 的具体词
    gold_set = set(gold)
    norm_set = set(normalized)
    print(f"  TP: {gold_set & norm_set}")
    print(f"  FP: {norm_set - gold_set}")
    print(f"  FN: {gold_set - norm_set}")

print("\n" + "="*60)
print("测试完成")

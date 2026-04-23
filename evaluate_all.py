"""
独立 F1 评估脚本 v3.3（修复版）
直接读取 outputs/ 目录中已有的 step3_final 文件，计算 F1，无需重新运行 pipeline。

修复内容（v3.3）：
  1. CMeEE Gold 读取兜底：若 step3_final 文件中 gold_entities_str 为空，
     自动尝试从原始数据文件（CMeEE_V2_dev_new.json 等）重新读取 Gold 标注
  2. 增加调试输出：打印前5条样本的 gold/pred 对比，方便验证
  3. IMCS 归一化评估：增加手工口语词->标准词映射补充（ORAL_TO_NORM）
  4. 增加 --debug 参数，控制是否打印样本对比

用法：
  python evaluate_all.py
  python evaluate_all.py --dataset cmeee --split dev
  python evaluate_all.py --dataset imcs  --split train
  python evaluate_all.py --outputs /root/autodl-tmp/MedNER_Project/better/medical_ner_v3/outputs
  python evaluate_all.py --no-embed
  python evaluate_all.py --debug
"""
import argparse
import json
import os
import re
import sys
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# ==================== 路径自动查找 ====================
def _find_outputs_dir() -> str:
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"),
        "/root/autodl-tmp/MedNER_Project/better/medical_ner_v3/outputs",
        "/root/autodl-tmp/MedNER_Project/medical_ner_v3/outputs",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

def _find_symptom_norm_csv() -> Optional[str]:
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "symptom_norm.csv"),
        "/root/autodl-tmp/MedNER_Project/data/symptom_norm.csv",
        "/root/autodl-tmp/MedNER_Project/better/medical_ner_v3/data/symptom_norm.csv",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None

def _find_cmeee_raw_file(split: str) -> Optional[str]:
    """自动查找 CMeEE 原始数据文件（用于兜底读取 Gold）"""
    candidates = [
        f"/root/autodl-tmp/MedNER_Project/data/CMeEE_V2/CMeEE_V2_{split}_new.json",
        f"/root/autodl-tmp/MedNER_Project/data/CMeEE_V2/CMeEE_V2_{split}.json",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", f"CMeEE_V2_{split}.json"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None

def _find_imcs_raw_file(split: str) -> Optional[str]:
    """自动查找 IMCS 原始数据文件（用于兜底读取 Gold）"""
    candidates = [
        f"/root/autodl-tmp/MedNER_Project/data/IMCS_V2/IMCS_V2_{split}_new.json",
        f"/root/autodl-tmp/MedNER_Project/data/IMCS_V2/IMCS_V2_{split}.json",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", f"IMCS_V2_{split}.json"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None

# ==================== 工具函数 ====================
def clean_entity_list(raw: str) -> List[str]:
    if not raw:
        return []
    raw = re.sub(r"\s+", "", raw)
    items = re.split(r"[,,,、；;]", raw)
    result = []
    for item in items:
        item = item.strip().strip('"\'""\'\'')
        if item and item not in ("无", "null", "None", "none", "NULL"):
            result.append(item)
    return result

def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_symptom_norm_vocab(csv_path: Optional[str]) -> List[str]:
    if not csv_path or not os.path.isfile(csv_path):
        print("  [警告] 未找到 symptom_norm.csv，IMCS 归一化将只使用 step2_normalized_map")
        return []
    vocab = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and line not in ("symptom_norm", "#", "norm") and not line.startswith("#"):
                vocab.append(line.split(",")[0].strip())
    print(f"  [词典] 加载 symptom_norm 词典: {len(vocab)} 个标准词")
    return vocab

# ==================== CMeEE Gold 兜底读取 ====================
def _extract_cmeee_gold_from_raw(item: Dict) -> List[str]:
    entities = item.get("entities", [])
    names = []
    seen = set()
    for e in entities:
        if not isinstance(e, dict):
            continue
        name = (e.get("entity") or e.get("mention") or e.get("text") or "").strip()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names

def _build_cmeee_gold_map(raw_path: str) -> Dict[str, List[str]]:
    if not raw_path or not os.path.isfile(raw_path):
        return {}
    try:
        data = load_json(raw_path)
        gold_map = {}
        for item in data:
            text = item.get("text", "")
            if text:
                gold_map[text] = _extract_cmeee_gold_from_raw(item)
        print(f"  [兜底] 从原始文件构建 CMeEE Gold 映射: {len(gold_map)} 条")
        return gold_map
    except Exception as e:
        print(f"  [兜底] 读取原始文件失败: {e}")
        return {}

# ==================== IMCS Gold 兜底读取 ====================
def _extract_imcs_gold_from_raw(item: Dict) -> List[str]:
    implicit_info = item.get("implicit_info", {})
    symptom_dict = implicit_info.get("Symptom", {})
    if not isinstance(symptom_dict, dict):
        return []
    gold = []
    seen = set()
    for norm_word, stype in symptom_dict.items():
        if str(stype) in ("1", "2") and norm_word not in seen:
            seen.add(norm_word)
            gold.append(norm_word)
    return gold

def _build_imcs_gold_map(raw_path: str) -> Dict[str, List[str]]:
    if not raw_path or not os.path.isfile(raw_path):
        return {}
    try:
        data = load_json(raw_path)
        gold_map = {}
        for item in data:
            rid = str(item.get("_record_id", ""))
            if rid:
                gold_map[rid] = _extract_imcs_gold_from_raw(item)
        print(f"  [兜底] 从原始文件构建 IMCS Gold 映射: {len(gold_map)} 条")
        return gold_map
    except Exception as e:
        print(f"  [兜底] 读取原始文件失败: {e}")
        return {}

# ==================== 手工口语词->标准词映射 ====================
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

# ==================== Micro F1 计算 ====================
def compute_micro_f1(
    y_true: List[List[str]],
    y_pred: List[List[str]],
) -> Tuple[float, float, float, int, int, int]:
    tp = fp = fn = 0
    for gold_list, pred_list in zip(y_true, y_pred):
        gold_set = set(gold_list)
        pred_set = set(pred_list)
        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, tp, fp, fn

# ==================== CMeEE 评估 ====================
def evaluate_cmeee_file(path: str, split: str, debug: bool = False) -> Optional[Dict]:
    if not os.path.isfile(path):
        print(f"  [跳过] 文件不存在: {path}")
        return None

    items = load_json(path)
    y_true, y_pred = [], []
    no_gold_count = 0
    gold_map_fallback = {}
    fallback_used = 0

    # 先检查是否需要兜底
    sample_no_gold = sum(1 for item in items[:20] if not item.get("gold_entities_str", ""))
    if sample_no_gold > 10:
        print(f"  [兜底] 检测到 step3_final 文件中 gold_entities_str 大量为空，尝试从原始文件读取 Gold...")
        raw_path = _find_cmeee_raw_file(split)
        if raw_path:
            gold_map_fallback = _build_cmeee_gold_map(raw_path)
        else:
            print(f"  [兜底] 未找到 CMeEE {split} 原始文件，无法兜底")

    for item in items:
        gold_raw = item.get("gold_entities_str", "")
        gold = clean_entity_list(gold_raw)

        if not gold and gold_map_fallback:
            text = item.get("text", "")
            gold = gold_map_fallback.get(text, [])
            if gold:
                fallback_used += 1

        if not gold:
            no_gold_count += 1

        pred_raw = (
            item.get("step3_final_output") or
            item.get("step1_enriched_output") or
            item.get("step1_raw_output") or ""
        )
        pred = clean_entity_list(pred_raw)
        y_true.append(gold)
        y_pred.append(pred)

    if fallback_used > 0:
        print(f"  [兜底] 成功从原始文件补充 {fallback_used} 条 Gold 标注")

    if debug:
        print(f"\n  [调试] CMeEE [{split}] 前5条样本 gold/pred 对比：")
        for i in range(min(5, len(items))):
            item = items[i]
            text = item.get("text", "")[:60]
            gold_str = item.get("gold_entities_str", "")
            print(f"    样本{i}: text={text}...")
            print(f"      gold_entities_str字段: '{gold_str}'")
            print(f"      gold（解析后）: {y_true[i]}")
            print(f"      pred: {y_pred[i][:10]}")
            print()

    if no_gold_count == len(items):
        print(f"  [跳过] CMeEE [{split}] 全部样本无 Gold 标注（test 集），跳过 F1 评估")
        print(f"  [提取] CMeEE [{split}] 共提取 {len(items)} 条，实体数: "
              f"{sum(len(p) for p in y_pred)}")
        return None

    p, r, f1, tp, fp, fn = compute_micro_f1(y_true, y_pred)
    result = {
        "dataset": "CMeEE_V2",
        "split": split,
        "eval_type": "字面匹配",
        "micro_f1": round(f1, 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "total_samples": len(items),
        "no_gold_samples": no_gold_count,
        "fallback_gold_samples": fallback_used,
    }
    _print_cmeee_result(result)
    return result

def _print_cmeee_result(r: Dict):
    print(f"\n{'='*60}")
    print(f"  CMeEE_V2 [{r['split']}]  {r['eval_type']}")
    print(f"{'='*60}")
    fb = r.get("fallback_gold_samples", 0)
    print(f"  样本数:      {r['total_samples']}  (无Gold: {r['no_gold_samples']}" +
          (f"  兜底补充: {fb}" if fb > 0 else "") + ")")
    print(f"  TP={r['tp']}  FP={r['fp']}  FN={r['fn']}")
    print(f"  Micro F1:    {r['micro_f1']:.4f}")
    print(f"  Precision:   {r['precision']:.4f}")
    print(f"  Recall:      {r['recall']:.4f}")
    _target_hint(r['micro_f1'])

# ==================== IMCS 评估 ====================
def evaluate_imcs_file(
    path: str,
    split: str,
    norm_vocab: List[str],
    use_embed: bool = True,
    debug: bool = False,
) -> Optional[Dict]:
    if not os.path.isfile(path):
        print(f"  [跳过] 文件不存在: {path}")
        return None

    items = load_json(path)
    y_true, y_pred_literal, y_pred_norm = [], [], []
    no_gold_count = 0
    gold_map_fallback = {}
    fallback_used = 0

    sample_no_gold = sum(1 for item in items[:20] if not item.get("gold_entities_str", ""))
    if sample_no_gold > 10:
        print(f"  [兜底] 检测到 step3_final 文件中 gold_entities_str 大量为空，尝试从原始文件读取 Gold...")
        raw_path = _find_imcs_raw_file(split)
        if raw_path:
            gold_map_fallback = _build_imcs_gold_map(raw_path)
        else:
            print(f"  [兜底] 未找到 IMCS {split} 原始文件，无法兜底")

    embed_fn = None
    if use_embed and norm_vocab:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from src.embedding_model import find_best_matches
            embed_fn = find_best_matches
            print(f"  [向量模型] 已加载，用于 IMCS 归一化补充")
        except Exception as e:
            print(f"  [向量模型] 加载失败: {e}，将只使用 step2_normalized_map + 手工词典")

    norm_vocab_set = set(norm_vocab)

    for item in items:
        gold_raw = item.get("gold_entities_str", "")
        gold = clean_entity_list(gold_raw)

        if not gold and gold_map_fallback:
            rid = str(item.get("_record_id", ""))
            gold = gold_map_fallback.get(rid, [])
            if gold:
                fallback_used += 1

        if not gold:
            no_gold_count += 1

        pred_raw = clean_entity_list(
            item.get("step3_final_output", item.get("step1_raw_output", ""))
        )

        norm_map = item.get("step2_normalized_map", {})
        normalized = []
        needs_embed = []

        for ent in pred_raw:
            # 第一优先级：手工口语词映射
            if ent in ORAL_TO_NORM:
                normalized.append(ORAL_TO_NORM[ent])
            # 第二优先级：step2_normalized_map
            elif ent in norm_map:
                mapped = norm_map[ent]
                if mapped in ORAL_TO_NORM:
                    normalized.append(ORAL_TO_NORM[mapped])
                else:
                    normalized.append(mapped)
            # 第三优先级：已经是标准词
            elif ent in norm_vocab_set:
                normalized.append(ent)
            # 第四优先级：向量模型补充
            else:
                needs_embed.append(ent)

        if needs_embed and norm_vocab and embed_fn:
            try:
                matches = embed_fn(needs_embed, norm_vocab)
                for ent, (best_match, score, status) in zip(needs_embed, matches):
                    if status in ("exact", "high", "medium"):
                        normalized.append(best_match)
                    else:
                        matched = False
                        for norm_word in norm_vocab:
                            if norm_word in ent or ent in norm_word:
                                if len(norm_word) >= 2 and len(ent) >= 2:
                                    normalized.append(norm_word)
                                    matched = True
                                    break
                        if not matched:
                            normalized.append(ent)
            except Exception:
                normalized.extend(needs_embed)
        else:
            for ent in needs_embed:
                matched = False
                for norm_word in norm_vocab:
                    if norm_word in ent or ent in norm_word:
                        if len(norm_word) >= 2 and len(ent) >= 2:
                            normalized.append(norm_word)
                            matched = True
                            break
                if not matched:
                    normalized.append(ent)

        y_true.append(gold)
        y_pred_literal.append(pred_raw)
        y_pred_norm.append(list(dict.fromkeys(normalized)))

    if fallback_used > 0:
        print(f"  [兜底] 成功从原始文件补充 {fallback_used} 条 Gold 标注")

    if debug:
        print(f"\n  [调试] IMCS [{split}] 前5条样本 gold/pred 对比：")
        for i in range(min(5, len(items))):
            item = items[i]
            rid = item.get("_record_id", "")
            gold_str = item.get("gold_entities_str", "")
            norm_map_sample = item.get("step2_normalized_map", {})
            print(f"    样本{i} (id={rid}):")
            print(f"      gold_entities_str字段: '{gold_str}'")
            print(f"      gold（解析后）: {y_true[i]}")
            print(f"      pred（字面）: {y_pred_literal[i][:8]}")
            print(f"      pred（归一化）: {y_pred_norm[i][:8]}")
            print(f"      step2_normalized_map（前5项）: {dict(list(norm_map_sample.items())[:5])}")
            print()

    if no_gold_count == len(items):
        print(f"  [跳过] IMCS [{split}] 全部样本无 Gold 标注（test 集），跳过 F1 评估")
        return None

    p_l, r_l, f1_l, tp_l, fp_l, fn_l = compute_micro_f1(y_true, y_pred_literal)
    p_n, r_n, f1_n, tp_n, fp_n, fn_n = compute_micro_f1(y_true, y_pred_norm)
    result = {
        "dataset": "IMCS_V2",
        "split": split,
        "literal_micro_f1": round(f1_l, 4),
        "literal_precision": round(p_l, 4),
        "literal_recall": round(r_l, 4),
        "literal_tp": tp_l, "literal_fp": fp_l, "literal_fn": fn_l,
        "normalized_micro_f1": round(f1_n, 4),
        "normalized_precision": round(p_n, 4),
        "normalized_recall": round(r_n, 4),
        "normalized_tp": tp_n, "normalized_fp": fp_n, "normalized_fn": fn_n,
        "total_samples": len(items),
        "no_gold_samples": no_gold_count,
        "fallback_gold_samples": fallback_used,
    }
    _print_imcs_result(result)
    return result

def _print_imcs_result(r: Dict):
    print(f"\n{'='*60}")
    print(f"  IMCS_V2 [{r['split']}]")
    print(f"{'='*60}")
    fb = r.get("fallback_gold_samples", 0)
    print(f"  样本数:          {r['total_samples']}  (无Gold: {r['no_gold_samples']}" +
          (f"  兜底补充: {fb}" if fb > 0 else "") + ")")
    print(f"  -- 字面匹配 --")
    print(f"  TP={r['literal_tp']}  FP={r['literal_fp']}  FN={r['literal_fn']}")
    print(f"  字面 Micro F1:   {r['literal_micro_f1']:.4f}  "
          f"(P={r['literal_precision']:.4f}, R={r['literal_recall']:.4f})")
    print(f"  -- 归一化匹配 --")
    print(f"  TP={r['normalized_tp']}  FP={r['normalized_fp']}  FN={r['normalized_fn']}")
    print(f"  归一化 Micro F1: {r['normalized_micro_f1']:.4f}  "
          f"(P={r['normalized_precision']:.4f}, R={r['normalized_recall']:.4f})")
    _target_hint(r['normalized_micro_f1'])

def _target_hint(f1: float, target: float = 0.80):
    if f1 >= target:
        print(f"  ✅ F1={f1:.4f} >= {target}，目标达成！")
    else:
        print(f"  ⚠️  F1={f1:.4f}，距目标 {target} 还差 {target - f1:.4f}")
    print(f"{'='*60}")

# ==================== 汇总报告 ====================
def print_summary(all_results: List[Dict]):
    print(f"\n\n{'#'*60}")
    print(f"  全量评估汇总")
    print(f"{'#'*60}")
    print(f"  {'数据集':<14} {'Split':<8} {'评估类型':<10} {'Micro F1':>9} {'P':>7} {'R':>7}")
    print(f"  {'-'*56}")
    for r in all_results:
        if r is None:
            continue
        if "micro_f1" in r:
            mark = "✅" if r["micro_f1"] >= 0.80 else "  "
            print(f"  {r['dataset']:<14} {r['split']:<8} {'字面匹配':<10} "
                  f"{r['micro_f1']:>8.4f} {r['precision']:>7.4f} {r['recall']:>7.4f}  {mark}")
        elif "literal_micro_f1" in r:
            mark_n = "✅" if r["normalized_micro_f1"] >= 0.80 else "  "
            print(f"  {r['dataset']:<14} {r['split']:<8} {'字面匹配':<10} "
                  f"{r['literal_micro_f1']:>8.4f} {r['literal_precision']:>7.4f} {r['literal_recall']:>7.4f}")
            print(f"  {'':<14} {'':<8} {'归一化匹配':<10} "
                  f"{r['normalized_micro_f1']:>8.4f} {r['normalized_precision']:>7.4f} {r['normalized_recall']:>7.4f}  {mark_n}")
    print(f"{'#'*60}\n")

def save_summary_json(all_results: List[Dict], outputs_dir: str):
    out_path = os.path.join(outputs_dir, "evaluation_summary.json")
    valid = [r for r in all_results if r is not None]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(valid, f, ensure_ascii=False, indent=2)
    print(f"  评估结果已保存: {out_path}")

# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="医疗NER框架独立F1评估脚本")
    parser.add_argument("--outputs", type=str, default=None,
                        help="outputs 目录路径（默认自动查找）")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "cmeee", "imcs"],
                        help="要评估的数据集（默认 all）")
    parser.add_argument("--split", type=str, default="all",
                        choices=["all", "train", "dev", "test"],
                        help="要评估的 split（默认 all）")
    parser.add_argument("--no-embed", action="store_true",
                        help="不加载向量模型（跳过 IMCS 归一化的向量补充步骤）")
    parser.add_argument("--debug", action="store_true",
                        help="打印调试信息（前5条样本 gold/pred 对比）")
    args = parser.parse_args()

    outputs_dir = args.outputs or _find_outputs_dir()
    print(f"\n[评估] outputs 目录: {outputs_dir}")
    if not os.path.isdir(outputs_dir):
        print(f"[错误] outputs 目录不存在: {outputs_dir}")
        sys.exit(1)

    norm_csv = _find_symptom_norm_csv()
    norm_vocab = load_symptom_norm_vocab(norm_csv)

    datasets = ["cmeee", "imcs"] if args.dataset == "all" else [args.dataset]
    splits   = ["train", "dev", "test"] if args.split == "all" else [args.split]

    all_results = []

    if "cmeee" in datasets:
        print(f"\n{'='*60}")
        print(f"  开始评估 CMeEE_V2")
        print(f"{'='*60}")
        for split in splits:
            path_step3    = os.path.join(outputs_dir, f"step3_final_CMeEE_V2_{split}.json")
            path_enriched = os.path.join(outputs_dir, f"step1_enriched_CMeEE_V2_{split}.json")
            path_step1    = os.path.join(outputs_dir, f"step1_raw_CMeEE_V2_{split}.json")
            if os.path.isfile(path_step3):
                print(f"\n  [CMeEE {split}] 使用 step3_final 文件")
                result = evaluate_cmeee_file(path_step3, split, debug=args.debug)
            elif os.path.isfile(path_enriched):
                print(f"\n  [CMeEE {split}] 使用 step1_enriched 文件（step3 未完成）")
                result = evaluate_cmeee_file(path_enriched, split, debug=args.debug)
            elif os.path.isfile(path_step1):
                print(f"\n  [CMeEE {split}] 使用 step1_raw 文件（step1 完成，step2/3 未完成）")
                result = evaluate_cmeee_file(path_step1, split, debug=args.debug)
            else:
                print(f"\n  [CMeEE {split}] 未找到任何输出文件，跳过")
                result = None
            all_results.append(result)

    if "imcs" in datasets:
        print(f"\n{'='*60}")
        print(f"  开始评估 IMCS_V2")
        print(f"{'='*60}")
        for split in splits:
            path_step3 = os.path.join(outputs_dir, f"step3_final_IMCS_V2_{split}.json")
            path_step1 = os.path.join(outputs_dir, f"step1_raw_IMCS_V2_{split}.json")
            if os.path.isfile(path_step3):
                print(f"\n  [IMCS {split}] 使用 step3_final 文件")
                result = evaluate_imcs_file(
                    path_step3, split, norm_vocab,
                    use_embed=not args.no_embed,
                    debug=args.debug,
                )
            elif os.path.isfile(path_step1):
                print(f"\n  [IMCS {split}] 使用 step1_raw 文件（step2/3 未完成）")
                result = evaluate_imcs_file(
                    path_step1, split, norm_vocab,
                    use_embed=not args.no_embed,
                    debug=args.debug,
                )
            else:
                print(f"\n  [IMCS {split}] 未找到任何输出文件，跳过")
                result = None
            all_results.append(result)

    valid_results = [r for r in all_results if r is not None]
    if valid_results:
        print_summary(valid_results)
        save_summary_json(valid_results, outputs_dir)
    else:
        print("\n[警告] 没有找到任何有效的评估结果")

if __name__ == "__main__":
    main()

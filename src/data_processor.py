"""
数据处理模块 v3（修复版）
支持：CMeEE_V2 / IMCS_V2 / yidu_4k (BIO格式)
功能：
  - 全量数据集加载（train/dev/test）
  - 少样本示例构建（只从 train 集提取一次，全局复用）
  - Gold 标准提取（修复 IMCS symptom_norm 字段解析）
  - 实体列表清洗工具
  - 官方 symptom_norm.csv 词典加载
"""
import csv
import json
import os
import re
import random
import sys
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    CMEEE_TRAIN_PATH, CMEEE_DEV_PATH, CMEEE_TEST_PATH,
    IMCS_TRAIN_PATH, IMCS_DEV_PATH, IMCS_TEST_PATH,
    YIDU_TRAIN_PATH,
    IMCS_NORM_DICT_PATH, SYMPTOM_NORM_CSV,
    CMEEE_TARGET_TYPES, CMEEE_TYPE_MAP,
    IMCS_TARGET_SYMPTOM_TYPES,
    FEW_SHOT_COUNT, FEW_SHOT_SEED,
)


# ==================== 通用工具函数 ====================

def clean_entity_list(text: str) -> List[str]:
    """清洗实体字符串，返回去重后的列表"""
    if not text or not isinstance(text, str):
        return []
    # 统一分隔符
    text = re.sub(r"[，、；\n]+", ",", text)
    if text.strip().lower() in ("无", "none", "null", "没有", "无实体", ""):
        return []
    parts = text.split(",")
    result = []
    seen = set()
    for p in parts:
        p = p.strip().strip("\"'「」【】《》（）()[]")
        if not p or p.lower() in ("无", "none", "null", "没有"):
            continue
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


# ==================== CMeEE 数据加载 ====================

def load_cmeee(split: str = "dev") -> List[Dict]:
    """加载 CMeEE 数据集指定 split"""
    path_map = {
        "train": CMEEE_TRAIN_PATH,
        "dev":   CMEEE_DEV_PATH,
        "test":  CMEEE_TEST_PATH,
    }
    path = path_map.get(split)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"CMeEE {split} 文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  [DataLoader] CMeEE {split}: {len(data)} 条")
    return data


def extract_cmeee_gold_names(item: Dict) -> List[str]:
    """
    从 CMeEE 条目中提取 Gold 实体名列表（去重）
    支持格式：
      - {"entities": [{"entity": "xxx", "type": "dis"}, ...]}
      - {"entities": [{"mention": "xxx", "label": "dis"}, ...]}
    """
    entities = item.get("entities", [])
    names = []
    seen = set()
    for e in entities:
        if not isinstance(e, dict):
            continue
        # 实体名字段
        name = (e.get("entity") or e.get("mention") or e.get("text") or "").strip()
        # 实体类型字段
        etype = (e.get("type") or e.get("label") or "").strip()
        # 若 CMEEE_TARGET_TYPES 为空则不过滤类型
        if CMEEE_TARGET_TYPES and etype and etype not in CMEEE_TARGET_TYPES:
            continue
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def build_cmeee_few_shot_examples(train_data: List[Dict], n: int = 8) -> List[Dict]:
    """
    从 CMeEE train 集构建少样本示例
    选取策略：
      1. 实体数量 ≥ 3
      2. 包含单字词实体（如"脑"、"肺"）
      3. 包含嵌套实体（短词是长词的子串）
      4. 文本长度适中（20-200字）
    """
    random.seed(FEW_SHOT_SEED)
    scored = []
    for item in train_data:
        entities = item.get("entities", [])
        text = item.get("text", "")
        if len(entities) < 2 or not (20 <= len(text) <= 200):
            continue
        names = extract_cmeee_gold_names(item)
        if not names:
            continue

        score = len(names)
        # 奖励：含单字词
        if any(len(nm) == 1 for nm in names):
            score += 5
        # 奖励：含嵌套实体
        for a in names:
            for b in names:
                if a != b and a in b:
                    score += 3
                    break
        # 奖励：多种实体类型
        types_in_item = set(e.get("type", e.get("label", "")) for e in entities)
        score += len(types_in_item)

        scored.append((score, item))

    scored.sort(key=lambda x: -x[0])
    selected = [c[1] for c in scored[:n * 3]]
    random.shuffle(selected)
    return selected[:n]


def format_cmeee_few_shot_prompt(examples: List[Dict]) -> str:
    """将 CMeEE 少样本示例格式化为 Prompt 字符串"""
    lines = []
    for i, ex in enumerate(examples):
        text = ex.get("text", "")
        gold_names = extract_cmeee_gold_names(ex)
        if not gold_names:
            continue
        # 截断超长文本
        if len(text) > 200:
            text = text[:200] + "..."
        lines.append(f"示例{i+1}：")
        lines.append(f"文本：{text}")
        lines.append(f"实体：{', '.join(gold_names)}")
        lines.append("")
    return "\n".join(lines)


# ==================== IMCS 数据加载 ====================

def load_imcs(split: str = "dev") -> List[Dict]:
    """
    加载 IMCS 数据集指定 split
    支持两种格式：
      - list 格式：[{dialogue_id, self_report, dialogue, ...}]
      - dict 格式：{dialogue_id: {self_report, dialogue, ...}}
    """
    path_map = {
        "train": IMCS_TRAIN_PATH,
        "dev":   IMCS_DEV_PATH,
        "test":  IMCS_TEST_PATH,
    }
    path = path_map.get(split)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"IMCS {split} 文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 统一转为 list 格式
    if isinstance(data, dict):
        records = []
        for did, record in data.items():
            if isinstance(record, dict):
                record["dialogue_id"] = did
                records.append(record)
        data = records

    print(f"  [DataLoader] IMCS {split}: {len(data)} 条")
    return data


def extract_imcs_gold(item: Dict) -> List[str]:
    """
    从 IMCS 对话条目中提取文档级 Gold symptom_norm 列表
    支持多种字段格式：
      格式1（_new.json）：dialogue[i].ner = [{symptom_norm, symptom_type}, ...]
      格式2（原始格式）：dialogue[i].symptom_norm = [...], dialogue[i].symptom_type = [...]
      格式3：report 字段中的 symptom_norm
    只取 symptom_type 为 "1"（肯定）或 "2"（推断）的标注
    """
    all_norms = set()

    for turn in item.get("dialogue", []):
        # 格式1：ner 字段（_new.json 格式）
        ner_list = turn.get("ner", [])
        if ner_list:
            for ner_item in ner_list:
                if not isinstance(ner_item, dict):
                    continue
                stype = str(ner_item.get("symptom_type", ner_item.get("type", -1)))
                norm  = (ner_item.get("symptom_norm") or ner_item.get("norm") or "").strip()
                if stype in IMCS_TARGET_SYMPTOM_TYPES and norm and norm.lower() != "null":
                    all_norms.add(norm)

        # 格式2：symptom_norm + symptom_type 并列列表
        norms = turn.get("symptom_norm", [])
        types = turn.get("symptom_type", [])
        if isinstance(norms, list) and isinstance(types, list):
            for norm, stype in zip(norms, types):
                stype_str = str(stype)
                norm_str  = str(norm).strip() if norm else ""
                if stype_str in IMCS_TARGET_SYMPTOM_TYPES and norm_str and norm_str.lower() != "null":
                    all_norms.add(norm_str)
        elif isinstance(norms, str) and norms.strip():
            # 单个字符串格式
            stype_str = str(types) if not isinstance(types, list) else ""
            if stype_str in IMCS_TARGET_SYMPTOM_TYPES:
                all_norms.add(norms.strip())

    # 格式3：report 字段
    report = item.get("report", {})
    if isinstance(report, dict):
        for key in ("symptom", "symptoms", "主诉", "现病史", "symptom_norm"):
            val = report.get(key, "")
            if isinstance(val, list):
                for v in val:
                    if isinstance(v, dict):
                        norm = (v.get("symptom_norm") or v.get("norm") or "").strip()
                    else:
                        norm = str(v).strip()
                    if norm and norm.lower() != "null":
                        all_norms.add(norm)
            elif isinstance(val, str) and val.strip() and val.lower() != "null":
                all_norms.add(val.strip())

    # 格式4：implicit_info 字段
    implicit = item.get("implicit_info", {})
    if isinstance(implicit, dict):
        for key in ("主诉", "现病史", "症状", "symptom_norm"):
            val = implicit.get(key, "")
            if isinstance(val, list):
                for v in val:
                    norm = str(v).strip()
                    if norm and norm.lower() != "null":
                        all_norms.add(norm)
            elif isinstance(val, str) and val.strip():
                all_norms.add(val.strip())

    return list(all_norms)


def build_imcs_few_shot_examples(train_data: List[Dict], n: int = 4) -> List[Dict]:
    """
    从 IMCS train 集构建少样本示例
    选取策略：
      1. Gold 症状数量 ≥ 2
      2. 含口语化表述（如"拉肚子"、"发烫"等）
      3. 对话轮次适中（5-20轮）
    """
    COLLOQUIAL_WORDS = {
        "拉肚子", "发烫", "发烧", "肚子痛", "流鼻涕", "咳嗽", "绿便",
        "稀便", "水样便", "蛋花汤", "拉稀", "不舒服", "难受", "哭闹",
    }
    random.seed(FEW_SHOT_SEED)
    scored = []
    for item in train_data:
        gold = extract_imcs_gold(item)
        if len(gold) < 2:
            continue
        dialogue = item.get("dialogue", [])
        if not (5 <= len(dialogue) <= 30):
            continue

        full_text = item.get("self_report", "") + " ".join(
            t.get("sentence", "") for t in dialogue
        )
        score = len(gold)
        if any(w in full_text for w in COLLOQUIAL_WORDS):
            score += 8
        scored.append((score, item))

    scored.sort(key=lambda x: -x[0])
    selected = [c[1] for c in scored[:n * 3]]
    random.shuffle(selected)
    return selected[:n]


def format_imcs_few_shot_prompt(examples: List[Dict]) -> str:
    """
    将 IMCS 少样本示例格式化为 Prompt 字符串
    展示：对话轮次（字面表述）+ Gold 症状标准词
    重点展示口语化词汇如何被提取
    """
    lines = []
    for i, ex in enumerate(examples[:4]):
        self_report = ex.get("self_report", "无")
        gold_norms  = extract_imcs_gold(ex)
        dialogue    = ex.get("dialogue", [])

        # 展示前5轮对话
        dialogue_lines = []
        for turn in dialogue[:5]:
            spk  = turn.get("speaker", "")
            sent = turn.get("sentence", "")
            if sent:
                dialogue_lines.append(f"  {spk}: {sent}")

        dialogue_str = "\n".join(dialogue_lines) if dialogue_lines else "  （无对话）"
        lines.append(f"示例{i+1}：")
        lines.append(f"主诉：{self_report}")
        lines.append(f"对话（前5轮）：\n{dialogue_str}")
        lines.append(f"Gold症状（标准词）：{', '.join(gold_norms)}")
        lines.append("")
    return "\n".join(lines)


def _extract_literal_from_bio(sentence: str, bio_label: str) -> List[str]:
    """从 BIO 标签中提取字面实体词"""
    if not sentence or not bio_label:
        return []
    tokens = list(sentence)
    labels = bio_label.split()
    if len(tokens) != len(labels):
        return []

    entities = []
    current = []
    for token, label in zip(tokens, labels):
        if label.startswith("B-"):
            if current:
                entities.append("".join(current))
            current = [token]
        elif label.startswith("I-") and current:
            current.append(token)
        else:
            if current:
                entities.append("".join(current))
                current = []
    if current:
        entities.append("".join(current))
    return [e for e in entities if e]


# ==================== yidu_4k 数据加载 ====================

def load_yidu(path: str = None) -> List[Dict]:
    """
    加载 yidu_4k 数据集（BIO格式，每行：字 标签）
    空行分隔不同句子
    返回格式：[{"text": str, "tokens": [...], "labels": [...]}]
    """
    path = path or YIDU_TRAIN_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"yidu 文件不存在: {path}")

    samples = []
    tokens, labels = [], []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "":
                if tokens:
                    samples.append({
                        "text": "".join(tokens),
                        "tokens": tokens[:],
                        "labels": labels[:],
                    })
                    tokens, labels = [], []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    tokens.append(parts[0])
                    labels.append(parts[-1])
                elif len(parts) == 1:
                    tokens.append(parts[0])
                    labels.append("O")

    if tokens:
        samples.append({
            "text": "".join(tokens),
            "tokens": tokens[:],
            "labels": labels[:],
        })

    print(f"  [DataLoader] yidu_4k: {len(samples)} 条句子")
    return samples


def extract_yidu_gold_entities(sample: Dict) -> List[str]:
    """从 yidu BIO 标注中提取 Gold 实体（用于验证，实际不做F1评估）"""
    tokens = sample.get("tokens", [])
    labels = sample.get("labels", [])
    entities = []
    current = []
    for token, label in zip(tokens, labels):
        if label.startswith("B-"):
            if current:
                entities.append("".join(current))
            current = [token]
        elif label.startswith("I-") and current:
            current.append(token)
        else:
            if current:
                entities.append("".join(current))
                current = []
    if current:
        entities.append("".join(current))
    return list(dict.fromkeys([e for e in entities if e]))


# ==================== 知识图谱词汇表构建 ====================

def build_cmeee_entity_vocab(train_data: List[Dict]) -> List[str]:
    """
    从 CMeEE train 集构建实体词汇表（用于 Step1.5 嵌套扩展）
    按词长降序排列（长词优先匹配）
    """
    vocab = set()
    for item in train_data:
        for e in item.get("entities", []):
            name = (e.get("entity") or e.get("mention") or "").strip()
            if name:
                vocab.add(name)
    # 按长度降序，确保长词优先
    return sorted(vocab, key=lambda x: -len(x))


def build_imcs_norm_vocab() -> List[str]:
    """
    构建 IMCS 官方归一化词汇表
    优先加载官方 symptom_norm.csv（331个标准词）
    若不存在则从 train 集提取
    """
    # 优先：官方 CSV 词典
    for csv_path in [SYMPTOM_NORM_CSV, IMCS_NORM_DICT_PATH]:
        if csv_path and os.path.exists(csv_path):
            vocab = []
            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row:
                            word = row[0].strip()
                            # 跳过表头
                            if word and word.lower() not in ("symptom_norm", "norm", "word"):
                                vocab.append(word)
                if vocab:
                    print(f"  [KG] 官方 symptom_norm.csv 加载: {len(vocab)} 个标准词")
                    return vocab
            except Exception as e:
                print(f"  [KG] 加载 {csv_path} 失败: {e}")

    # 降级：从 train 集提取
    print("  [KG] 官方词典不存在，从 IMCS train 集提取归一化词汇...")
    try:
        train_data = load_imcs("train")
        vocab = set()
        for item in train_data:
            for turn in item.get("dialogue", []):
                # 格式1：ner 字段
                for ner_item in turn.get("ner", []):
                    if isinstance(ner_item, dict):
                        norm = (ner_item.get("symptom_norm") or ner_item.get("norm") or "").strip()
                        if norm and norm.lower() != "null":
                            vocab.add(norm)
                # 格式2：symptom_norm 列表
                for norm in turn.get("symptom_norm", []):
                    if norm and str(norm).lower() != "null":
                        vocab.add(str(norm).strip())
        vocab_list = sorted(vocab, key=lambda x: -len(x))
        print(f"  [KG] 从 train 集提取 IMCS 归一化词汇: {len(vocab_list)} 个")
        return vocab_list
    except Exception as e:
        print(f"  [KG] 提取失败: {e}")
        return []

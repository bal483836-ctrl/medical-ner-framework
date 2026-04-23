"""
Step 1: 大模型少样本实体抽取模块 v3.2（Prompt压缩+速度优化版）
支持：CMeEE_V2 全量 / IMCS_V2 全量 / yidu_4k（只提取）
关键设计：
  - 少样本示例只从 train 集提取一次，全局复用
  - CMeEE/yidu 使用 batch_generate 批量推理
  - IMCS 对话轮次收集后批量推理
  - Prompt 精简：去掉冗余说明，保留核心规则，减少 ~40% token 数
  - 断点续跑：每 N 条自动保存
"""
import json
import os
import re
import sys
from typing import List, Dict, Optional
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    OUTPUT_DIR, STEP1_PREFIX,
    CMEEE_TYPE_MAP, IMCS_FORCE_RECALL_WORDS,
    IMCS_TARGET_SYMPTOM_TYPES,
    FEW_SHOT_COUNT,
)
from src.llm_client import batch_generate, clean_llm_output
from src.data_processor import (
    load_cmeee, load_imcs, load_yidu,
    extract_cmeee_gold_names, extract_imcs_gold,
    build_cmeee_few_shot_examples, build_imcs_few_shot_examples,
    format_cmeee_few_shot_prompt, format_imcs_few_shot_prompt,
    clean_entity_list,
)

# ==================== batch_size 配置 ====================
# 5090 32GB BF16：CMeEE 短文本 batch=8，IMCS 轮次 batch=16
# 若出现 OOM，llm_client 会自动降级为逐条推理，或在此处改小
CMEEE_BATCH_SIZE     = 8   # CMeEE/yidu 每批处理条数（OOM 时改为 4）
IMCS_TURN_BATCH_SIZE = 16  # IMCS 每批处理的对话轮次数（OOM 时改为 8）


# ==================== Prompt 构建（精简版）====================

def build_cmeee_prompt(text: str, few_shot_str: str) -> str:
    """
    CMeEE / yidu 医学文本实体抽取 Prompt（精简版）
    相比 v3.1 减少约 40% token，核心规则保留，速度更快
    """
    types = "、".join([f"{v}({k})" for k, v in CMEEE_TYPE_MAP.items()])
    return f"""医学NER任务。从文本中提取所有医疗命名实体，原文原样输出，用逗号分隔。

实体类型：{types}

规则：
①原文原样复制，禁止归一化或改写
②单字身体部位必须提取（脑/肝/肺/肾/心/胃/肠/脾/骨/血/皮/眼/耳/鼻/腹/胸等）
③嵌套实体全部提取（"胃酸分泌"同时提取"胃酸"；"脑细胞"同时提取"脑"）
④短词专业术语保留（三凹/鼻扇/环切/缺氧/水肿/坏死/梗死/栓塞等）
⑤禁止提取：性别年龄/时间数字/方位词/连词/纯化学基团

示例：
{few_shot_str}

文本：{text}
实体："""


def build_imcs_turn_prompt(
    sentence: str,
    speaker: str,
    self_report: str,
    context_str: str,
    few_shot_str: str,
) -> str:
    """
    IMCS 单轮对话实体抽取 Prompt（精简版）
    """
    return f"""儿科对话NER任务。从【当前发言】中提取症状/疾病实体，原文原样输出，逗号分隔。

规则：
①原文原样复制，禁止归一化（"拉肚子"→"拉肚子"，不能写"腹泻"）
②保留完整描述（"绿色的大便"不能简化为"大便"）
③提取微小体征（放屁/没精神/尿少/哭闹/抽筋）
④禁止提取：药物/病因/食物/检查项目
⑤无症状则输出"无"

示例：
{few_shot_str}

主诉：{self_report}
背景：{context_str}

当前发言({speaker})：{sentence}
实体："""


# ==================== 后处理工具 ====================

def _postprocess_cmeee(entity_names: List[str], text: str) -> List[Dict]:
    """CMeEE 实体后处理：去噪、原文锚定、去重"""
    entities, seen = [], set()
    for name in entity_names:
        name = re.sub(r"[的等了吗啊呢吧哦嗯]+$", "", name).strip()
        if not name or re.match(r"^[\d\s\W]+$", name) or name in seen:
            continue
        if name not in text:
            continue
        seen.add(name)
        start = text.find(name)
        entities.append({"start_idx": start, "end_idx": start + len(name), "entity": name})
    return entities


def _postprocess_imcs(
    entity_names: List[str], sentence: str, self_report: str
) -> tuple:
    """IMCS 实体后处理：去噪、原文锚定，返回 (turn_entities, doc_predictions)"""
    turn_entities, doc_preds = [], set()
    full_ctx = self_report + " " + sentence
    for name in entity_names:
        name = re.sub(r"[的等了吗啊呢吧哦嗯]+$", "", name).strip()
        if not name:
            continue
        if name in full_ctx:
            start = sentence.find(name)
            if start != -1:
                turn_entities.append({
                    "start_idx": start,
                    "end_idx": start + len(name),
                    "entity": name,
                })
            doc_preds.add(name)
    return turn_entities, doc_preds


# ==================== CMeEE 批量抽取 ====================

def extract_cmeee_split(
    split: str,
    few_shot_str: str,
    output_path: str,
    limit: Optional[int] = None,
) -> List[Dict]:
    """CMeEE 指定 split 批量实体抽取，支持断点续跑"""
    done_ids, results = set(), []
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                results = json.load(f)
                done_ids = {str(item.get("id", i)) for i, item in enumerate(results)}
                print(f"  [Resume] CMeEE {split}: 已完成 {len(results)} 条，继续...")
            except Exception:
                results = []

    data = load_cmeee(split)
    if limit:
        data = data[:limit]

    pending = []
    for idx, item in enumerate(data):
        if str(item.get("id", idx)) not in done_ids:
            item["_idx"] = idx
            pending.append(item)

    print(f"  [CMeEE {split}] 待处理: {len(pending)} 条，batch_size={CMEEE_BATCH_SIZE}")

    for batch_start in tqdm(range(0, len(pending), CMEEE_BATCH_SIZE), desc=f"CMeEE {split}"):
        batch = pending[batch_start: batch_start + CMEEE_BATCH_SIZE]
        prompts = [build_cmeee_prompt(item.get("text", ""), few_shot_str) if item.get("text") else "" for item in batch]

        valid_idx = [i for i, p in enumerate(prompts) if p]
        responses = batch_generate([prompts[i] for i in valid_idx], max_tokens=256) if valid_idx else []
        resp_map = {valid_idx[i]: responses[i] for i in range(len(valid_idx))}

        for i, item in enumerate(batch):
            text = item.get("text", "")
            item["gold_entities_str"] = ",".join(extract_cmeee_gold_names(item))
            raw = resp_map.get(i, "")
            names = clean_entity_list(clean_llm_output(raw))
            ents = _postprocess_cmeee(names, text)
            item["step1_entities"]   = ents
            item["step1_raw_output"] = ",".join(e["entity"] for e in ents)
            results.append(item)

        # 每 200 条保存一次断点
        if batch_start > 0 and (batch_start // CMEEE_BATCH_SIZE) % 25 == 0:
            _save_json(results, output_path)

    _save_json(results, output_path)
    print(f"  ✅ CMeEE {split} Step1 → {output_path}")
    return results


# ==================== IMCS 批量抽取 ====================

def extract_imcs_split(
    split: str,
    few_shot_str: str,
    output_path: str,
    limit: Optional[int] = None,
) -> List[Dict]:
    """
    IMCS 指定 split 批量实体抽取，支持断点续跑
    策略：收集所有轮次 prompt → 一次性批量推理 → 按轮次分配结果
    """
    done_ids, results = set(), []
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                results = json.load(f)
                done_ids = {
                    str(item.get("dialogue_id", item.get("id", i)))
                    for i, item in enumerate(results)
                }
                print(f"  [Resume] IMCS {split}: 已完成 {len(results)} 条，继续...")
            except Exception:
                results = []

    data = load_imcs(split)
    if limit:
        data = data[:limit]

    pending = [
        item for item in data
        if str(item.get("dialogue_id", item.get("id", ""))) not in done_ids
    ]
    print(f"  [IMCS {split}] 待处理: {len(pending)} 条对话，turn_batch={IMCS_TURN_BATCH_SIZE}")

    # ---- 收集所有轮次 prompt ----
    turn_meta = []   # (d_idx, t_idx, sentence, self_report, prompt)
    for d_idx, item in enumerate(pending):
        self_report = item.get("self_report", "无")
        history_texts = []
        for t_idx, turn in enumerate(item.get("dialogue", [])):
            sentence = turn.get("sentence", "")
            speaker  = turn.get("speaker", "未知")
            ctx = "\n".join(history_texts[-3:]) if history_texts else "（无）"
            if sentence:
                prompt = build_imcs_turn_prompt(sentence, speaker, self_report, ctx, few_shot_str)
                turn_meta.append((d_idx, t_idx, sentence, self_report, prompt))
            history_texts.append(f"{speaker}: {sentence}")

    print(f"  [IMCS {split}] 共 {len(turn_meta)} 个对话轮次，开始批量推理...")

    # ---- 批量推理所有轮次 ----
    all_prompts   = [m[4] for m in turn_meta]
    all_responses = []
    for bs in tqdm(range(0, len(all_prompts), IMCS_TURN_BATCH_SIZE), desc=f"IMCS {split} 轮次推理"):
        all_responses.extend(
            batch_generate(all_prompts[bs: bs + IMCS_TURN_BATCH_SIZE], max_tokens=128)
        )

    # ---- 分配结果到各对话轮次 ----
    d_turn_results: Dict[int, Dict[int, dict]] = {d: {} for d in range(len(pending))}
    for ri, (d_idx, t_idx, sentence, self_report, _) in enumerate(turn_meta):
        raw = all_responses[ri] if ri < len(all_responses) else ""
        names = clean_entity_list(clean_llm_output(raw))
        turn_ents, doc_preds = _postprocess_imcs(names, sentence, self_report)
        d_turn_results[d_idx][t_idx] = {"turn_entities": turn_ents, "doc_preds": doc_preds}

    # ---- 汇总文档级预测 ----
    for d_idx, item in enumerate(tqdm(pending, desc=f"IMCS {split} 汇总")):
        self_report = item.get("self_report", "无")
        dialogue    = item.get("dialogue", [])
        item["gold_entities_str"] = ",".join(extract_imcs_gold(item))

        doc_preds = set()
        for t_idx, turn in enumerate(dialogue):
            tr = d_turn_results[d_idx].get(t_idx, {})
            turn["step1_sentence_entities"]   = tr.get("turn_entities", [])
            turn["step1_sentence_raw_output"] = ",".join(
                e["entity"] for e in tr.get("turn_entities", [])
            )
            doc_preds.update(tr.get("doc_preds", set()))

        # 强召回兜底
        full_text = self_report + " " + " ".join(t.get("sentence", "") for t in dialogue)
        for w in IMCS_FORCE_RECALL_WORDS:
            if w in full_text:
                doc_preds.add(w)

        item["step1_raw_output"] = ",".join(list(doc_preds))
        results.append(item)

        if len(results) % 20 == 0:
            _save_json(results, output_path)

    _save_json(results, output_path)
    print(f"  ✅ IMCS {split} Step1 → {output_path}")
    return results


# ==================== yidu_4k 批量抽取 ====================

def extract_yidu(
    few_shot_str: str,
    output_path: str,
    limit: Optional[int] = None,
) -> List[Dict]:
    """yidu_4k 批量实体抽取（只提取，不评估 F1）"""
    done_count, results = 0, []
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                results = json.load(f)
                done_count = len(results)
                print(f"  [Resume] yidu: 已完成 {done_count} 条，继续...")
            except Exception:
                results = []

    data = load_yidu()
    if limit:
        data = data[:limit]
    data = data[done_count:]

    from src.data_processor import extract_yidu_gold_entities
    print(f"  [yidu] 待处理: {len(data)} 条，batch_size={CMEEE_BATCH_SIZE}")

    for bs in tqdm(range(0, len(data), CMEEE_BATCH_SIZE), desc="yidu_4k"):
        batch = data[bs: bs + CMEEE_BATCH_SIZE]
        prompts = [build_cmeee_prompt(item.get("text", ""), few_shot_str) if item.get("text") else "" for item in batch]
        valid_idx = [i for i, p in enumerate(prompts) if p]
        responses = batch_generate([prompts[i] for i in valid_idx], max_tokens=256) if valid_idx else []
        resp_map = {valid_idx[i]: responses[i] for i in range(len(valid_idx))}

        for i, item in enumerate(batch):
            text = item.get("text", "")
            raw  = resp_map.get(i, "")
            names = clean_entity_list(clean_llm_output(raw))
            ents  = _postprocess_cmeee(names, text)
            item["step1_entities"]   = ents
            item["step1_raw_output"] = ",".join(e["entity"] for e in ents)
            item["gold_entities_str"] = ",".join(extract_yidu_gold_entities(item))
            results.append(item)

        if bs > 0 and (bs // CMEEE_BATCH_SIZE) % 50 == 0:
            _save_json(results, output_path)

    _save_json(results, output_path)
    print(f"  ✅ yidu Step1 → {output_path}")
    return results


# ==================== 少样本示例全局构建 ====================

def build_global_few_shot(n: int = FEW_SHOT_COUNT):
    """
    只从 train 集提取一次少样本示例，返回格式化字符串供全局复用
    Returns:
        cmeee_few_shot_str: CMeEE/yidu 通用少样本字符串
        imcs_few_shot_str:  IMCS 专用少样本字符串
    """
    print("\n[FewShot] 从 train 集构建少样本示例（全局复用）...")
    cmeee_few_shot_str = ""
    try:
        cmeee_train = load_cmeee("train")
        examples = build_cmeee_few_shot_examples(cmeee_train, n=n)
        cmeee_few_shot_str = format_cmeee_few_shot_prompt(examples)
        print(f"  [FewShot] CMeEE: {len(examples)} 条示例")
    except Exception as e:
        print(f"  [FewShot] CMeEE train 加载失败: {e}")

    imcs_few_shot_str = ""
    try:
        imcs_train = load_imcs("train")
        examples = build_imcs_few_shot_examples(imcs_train, n=n)
        imcs_few_shot_str = format_imcs_few_shot_prompt(examples)
        print(f"  [FewShot] IMCS: {len(examples)} 条示例")
    except Exception as e:
        print(f"  [FewShot] IMCS train 加载失败: {e}")

    return cmeee_few_shot_str, imcs_few_shot_str


# ==================== 工具函数 ====================

def _save_json(data: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

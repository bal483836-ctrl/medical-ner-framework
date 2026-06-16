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
    FEW_SHOT_COUNT, RETRIEVAL_FEWSHOT_K, IMCS_VOCAB_RECALL,
)
from src.retrieval_fewshot import load_symptom_norm_vocab, imcs_vocab_recall
from src.llm_client import batch_generate, clean_llm_output
from src.data_processor import (
    load_cmeee, load_imcs, load_yidu,
    extract_cmeee_gold_names, extract_imcs_gold,
    build_cmeee_few_shot_examples, build_imcs_few_shot_examples,
    format_cmeee_few_shot_prompt, format_imcs_few_shot_prompt,
    clean_entity_list,
)

# ==================== batch_size 配置 ====================
# 从 config 读 5090 优化默认值；OOM 时 llm_client 会自动降级
from config.config import LLM_BATCH_SIZE_CMEEE, LLM_BATCH_SIZE_IMCS
CMEEE_BATCH_SIZE     = LLM_BATCH_SIZE_CMEEE   # 5090: 12
IMCS_TURN_BATCH_SIZE = LLM_BATCH_SIZE_IMCS    # 5090: 24


# ==================== Prompt 构建（精简版）====================

def build_cmeee_prompt(text: str, few_shot_str: str) -> str:
    """CMeEE prompt (v4.6)：回归 v4.3 简洁版（实测 step1_raw R=0.67）+ 嵌套强调

    重要教训：v4.4/v4.5 加详细禁抽列表导致 LLM 过度保守，step1_raw R 掉到 0.04。
    抽取阶段必须"放手抽"，过滤交给后续 step3 / reflect。
    """
    types = "、".join([f"{v}({k})" for k, v in CMEEE_TYPE_MAP.items()])
    return f"""医学 NER 任务。从文本抽医疗实体，原文逐字符复制，逗号分隔。

实体类型：{types}

⚠️ CMeEE 标注边界规则（必须严格遵守）：
① 单字身体部位独立抽（脑/心/肝/肺/肾/胃/肠/血/尿/胸/腹/头/眼/耳/口/手/脚/骨/皮/喉/腰/鼻/舌/齿）
② 去后缀：「头痛」抽「头」，「瞳孔变化」抽「瞳孔」，「肌张力增高」抽「肌张力」
③ 顿号/括号连接的并列短语保持原样：「水、盐电解质紊乱」整体抽
④ 嵌套都抽：「葡萄球菌肺炎」+「葡萄球菌」；「胃肠功能疾病」+「胃」「肠」；「原因不明的肾功能减退」+「肾功能减退」
⑤ 禁止抽：HTML 标签（<sub>等）、性别年龄、时间数字、带百分号或单位的长描述
⑥ 边界精确：不带前缀「治疗/明显的」，不带尾部「了/的/吗/啊」

示例：
{few_shot_str}

文本：{text}

实体（逗号分隔，**嵌套都抽，宁多勿漏**）："""


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
    return f"""儿科对话NER任务。从【当前发言】抽取**患者的症状/体征及医生给出的疾病判断**，原文原样输出，逗号分隔。
**宁可多抽不可漏抽**。

✅ 抽（症状/体征）：发烧/咳嗽/拉肚子/吐奶/流鼻涕/鼻塞/腹胀/腹痛/绿便/水样便/精神差/没精神/哭闹/放屁/夜咳/嗓子哑/38.5度
✅ 也抽（IMCS 把这些诊断/感染/发热分级也算 symptom_norm，**必须抽**）：
   感冒/病毒感染/细菌感染/支原体感染/呼吸道感染/上呼吸道感染/支气管炎/肺炎/消化不良/
   低热/中等度热/高热（即使由医生说出、即使是诊断结论，也要抽）

🚫 不抽：具体药名/检查项目（血常规、X光）/患者称谓/医务动作/舌象/否定句中的症状

规则：
①原文原样复制，禁止归一化（『拉肚子』保留『拉肚子』，不要写『腹泻』）
②保留完整描述（『绿色的大便』不简化为『大便』）
③微小体征也要抽：放屁/没精神/尿少/哭闹/夜咳
④医生说的疾病/感染/发热分级结论也要抽（如『考虑病毒感染』抽『病毒感染』）
⑤无症状则输出"无"

示例：
{few_shot_str}

主诉：{self_report}
背景：{context_str}

当前发言({speaker})：{sentence}
实体（只抽**症状/体征**）："""


# ==================== 后处理工具 ====================

_BODY_SINGLE_CHARS = set("脑心肝肺肾胃肠脾血尿胸腹头眼耳口舌齿喉腰骨皮足手鼻")
_DROP_SUFFIXES = ["痛", "变化", "增高", "降低", "紊乱", "异常", "障碍", "反应", "状态"]


def _postprocess_cmeee(entity_names: List[str], text: str) -> List[Dict]:
    """CMeEE 后处理 v4.3：去 HTML / 去后缀缩边界 / 单字 force recall / 原文锚定 / 去重"""
    candidates = list(entity_names)
    # 单字身体部位 force recall：原文出现即加
    for ch in _BODY_SINGLE_CHARS:
        if ch in text and ch not in candidates:
            candidates.append(ch)
    entities, seen = [], set()
    for name in candidates:
        if not name:
            continue
        # 去 HTML 标签（<sub> 等）
        name = re.sub(r"<[^>]+>", "", name).strip()
        # 去尾部助词
        name = re.sub(r"[的等了吗啊呢吧哦嗯]+$", "", name).strip()
        # 尝试后缀缩到 CMeEE 标注边界（如「头痛」→「头」）
        for suf in _DROP_SUFFIXES:
            if len(name) > len(suf) and name.endswith(suf):
                short = name[:-len(suf)]
                if short in text and short not in seen:
                    seen.add(short)
                    s = text.find(short)
                    entities.append({"start_idx": s, "end_idx": s + len(short), "entity": short})
                break
        # 噪声过滤
        if not name or re.match(r"^[\d\s\W]+$", name) or name in seen:
            continue
        if "%" in name and len(name) > 6:
            continue
        if "～" in name or " kPa" in name:
            continue
        if name not in text:
            continue
        seen.add(name)
        s = text.find(name)
        entities.append({"start_idx": s, "end_idx": s + len(name), "entity": name})
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
    retriever=None,
) -> List[Dict]:
    """CMeEE 指定 split 批量实体抽取，支持断点续跑。

    retriever 非空时启用检索式动态 few-shot：每条文本检索最相似的 train 样本，
    否则回退到全局固定 few_shot_str。
    """
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
        prompts = []
        for item in batch:
            text = item.get("text", "")
            if not text:
                prompts.append("")
                continue
            fs = few_shot_str
            if retriever is not None:
                rfs = retriever.retrieve_block(text, k=RETRIEVAL_FEWSHOT_K)
                if rfs:
                    fs = rfs
            prompts.append(build_cmeee_prompt(text, fs))

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
    retriever=None,
) -> List[Dict]:
    """
    IMCS 指定 split 批量实体抽取，支持断点续跑
    策略：收集所有轮次 prompt → 一次性批量推理 → 按轮次分配结果
    retriever 非空时对每个发言检索最相似的 train 轮次作为 few-shot。
    """
    # 闭集兜底召回词表（331 个 symptom_norm，含诊断/感染/发热分级）
    _norm_vocab = load_symptom_norm_vocab() if IMCS_VOCAB_RECALL else []
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
                fs = few_shot_str
                if retriever is not None:
                    rfs = retriever.retrieve_block(sentence, k=RETRIEVAL_FEWSHOT_K)
                    if rfs:
                        fs = rfs
                prompt = build_imcs_turn_prompt(sentence, speaker, self_report, ctx, fs)
                turn_meta.append((d_idx, t_idx, sentence, self_report, prompt))
            history_texts.append(f"{speaker}: {sentence}")

    print(f"  [IMCS {split}] 共 {len(turn_meta)} 个对话轮次，开始批量推理...")

    # ---- turn-level 断点缓存：每跑完一条立即 append 到 JSONL，无需等 batch 结束 ----
    #      文件每行一条 {"k": key, "v": response}，append-only，崩了也只丢正在写的那一行
    import hashlib
    turn_cache_path = output_path + ".turncache.jsonl"
    turn_cache: Dict[str, str] = {}
    if os.path.exists(turn_cache_path):
        try:
            with open(turn_cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        turn_cache[rec["k"]] = rec["v"]
                    except Exception:
                        pass
            print(f"  [Resume] IMCS {split} turn cache: {len(turn_cache)} 个轮次已缓存")
        except Exception:
            turn_cache = {}
    # 兼容旧 .turncache.json 格式（如有则一并读入）
    legacy_path = output_path + ".turncache.json"
    if os.path.exists(legacy_path):
        try:
            with open(legacy_path, "r", encoding="utf-8") as f:
                legacy = json.load(f)
            turn_cache.update(legacy)
            print(f"  [Resume] IMCS {split} 合并 legacy cache: +{len(legacy)} 条")
        except Exception:
            pass

    def _turn_key(d_idx: int, t_idx: int, sentence: str) -> str:
        did = pending[d_idx].get("dialogue_id", pending[d_idx].get("id", d_idx))
        h = hashlib.md5(sentence.encode("utf-8")).hexdigest()[:8]
        return f"{did}::{t_idx}::{h}"

    # ---- 批量推理所有未缓存的轮次 ----
    all_responses: List[str] = []
    pending_indices = []
    for ri, (d_idx, t_idx, sentence, _, _) in enumerate(turn_meta):
        k = _turn_key(d_idx, t_idx, sentence)
        if k in turn_cache:
            all_responses.append(turn_cache[k])
        else:
            all_responses.append(None)
            pending_indices.append(ri)

    if pending_indices:
        print(f"  [IMCS {split}] 待推理轮次: {len(pending_indices)}/{len(turn_meta)}")
        # 用 append 模式打开，每条结果一写完就 flush，崩溃只损失 GPU 推理已完成但未 flush 的那一行
        cache_fh = open(turn_cache_path, "a", encoding="utf-8")
        try:
            for bs in tqdm(range(0, len(pending_indices), IMCS_TURN_BATCH_SIZE),
                           desc=f"IMCS {split} 轮次推理"):
                chunk_idx = pending_indices[bs: bs + IMCS_TURN_BATCH_SIZE]
                chunk_prompts = [turn_meta[i][4] for i in chunk_idx]
                chunk_resps = batch_generate(chunk_prompts, max_tokens=128)
                for local_i, ri in enumerate(chunk_idx):
                    resp = chunk_resps[local_i] if local_i < len(chunk_resps) else ""
                    all_responses[ri] = resp
                    d_idx, t_idx, sentence, _, _ = turn_meta[ri]
                    k = _turn_key(d_idx, t_idx, sentence)
                    turn_cache[k] = resp
                    cache_fh.write(json.dumps({"k": k, "v": resp}, ensure_ascii=False) + "\n")
                cache_fh.flush()
        finally:
            cache_fh.close()
    else:
        print(f"  [IMCS {split}] 所有轮次都已在 turn cache 中，跳过推理")

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
        if _norm_vocab:
            # 闭集召回：扫描整段对话命中 331 个 symptom_norm 词（含诊断/感染/发热分级）
            # 并做否定语境过滤，直接补回「感冒/支气管炎/病毒感染」等漏检词
            for w in imcs_vocab_recall(full_text, _norm_vocab):
                doc_preds.add(w)
        else:
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

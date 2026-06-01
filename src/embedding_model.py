"""
向量模型模块 v3
直接本地加载 bge-large-zh-v1.5（1024维）
支持批量编码和余弦相似度计算
"""
import os
import sys
import gc
import numpy as np
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    EMBEDDING_MODEL_PATH, EMBEDDING_BATCH_SIZE, EMBEDDING_DIM,
    HIGH_SIM_THRESHOLD, LOW_SIM_THRESHOLD,
)

# 单例
_embed_model = None
_embed_tokenizer = None


def get_embedding_model():
    # v4.3: 加载 BGE 前释放 LLM 显存
    try:
        from src.llm_client import release_llm
        import torch, gc
        release_llm()
        torch.cuda.empty_cache(); gc.collect()
    except Exception:
        pass
    """懒加载向量模型"""
    global _embed_model, _embed_tokenizer
    if _embed_model is not None:
        return _embed_model, _embed_tokenizer

    print(f"\n[Embed] 正在加载 bge-large-zh-v1.5: {EMBEDDING_MODEL_PATH}")

    # 优先使用 FlagEmbedding（官方推荐）
    try:
        from FlagEmbedding import FlagModel
        _embed_model = FlagModel(
            EMBEDDING_MODEL_PATH,
            query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
            use_fp16=True,
        )
        _embed_tokenizer = None
        print("[Embed] 使用 FlagEmbedding 加载成功")
        return _embed_model, _embed_tokenizer
    except ImportError:
        pass

    # 降级：sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _embed_model = SentenceTransformer(EMBEDDING_MODEL_PATH, device=device)
        # 显存够时启用 fp16，BGE 编码加速 ~2x
        if device == "cuda":
            try:
                _embed_model = _embed_model.half()
            except Exception:
                pass
        _embed_tokenizer = "sentence_transformers"
        print(f"[Embed] 使用 sentence-transformers 加载成功 (device={device}, fp16={device=='cuda'})")
        return _embed_model, _embed_tokenizer
    except ImportError:
        pass

    # 降级：transformers 原生
    import torch
    from transformers import AutoTokenizer, AutoModel
    _embed_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
    _embed_model = AutoModel.from_pretrained(EMBEDDING_MODEL_PATH)
    _embed_model.eval()
    if torch.cuda.is_available():
        _embed_model = _embed_model.cuda()
    print("[Embed] 使用 transformers 原生加载成功")
    return _embed_model, _embed_tokenizer


def encode_texts(texts: List[str], is_query: bool = True) -> np.ndarray:
    """
    批量编码文本为向量
    Args:
        texts: 待编码文本列表
        is_query: True=查询端（加指令前缀），False=文档端
    Returns:
        numpy array, shape (n, dim)
    """
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    model, tokenizer = get_embedding_model()
    # 大批量自动开进度条（避免 Step2 / KG 向量化看起来卡死）
    show_pb = len(texts) >= 512

    # FlagEmbedding
    if tokenizer is None:
        if is_query:
            vecs = model.encode_queries(texts, batch_size=EMBEDDING_BATCH_SIZE)
        else:
            vecs = model.encode_corpus(texts, batch_size=EMBEDDING_BATCH_SIZE)
        return np.array(vecs, dtype=np.float32)

    # sentence-transformers
    if tokenizer == "sentence_transformers":
        # BGE 需要加查询指令
        if is_query:
            texts = ["为这个句子生成表示以用于检索相关文章：" + t for t in texts]
        vecs = model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=show_pb,
            convert_to_numpy=True,
        )
        return np.array(vecs, dtype=np.float32)

    # transformers 原生
    import torch
    from tqdm import tqdm
    all_vecs = []
    rng = range(0, len(texts), EMBEDDING_BATCH_SIZE)
    if show_pb:
        rng = tqdm(rng, desc="[Embed]", unit="batch")
    for i in rng:
        batch = texts[i: i + EMBEDDING_BATCH_SIZE]
        if is_query:
            batch = ["为这个句子生成表示以用于检索相关文章：" + t for t in batch]
        encoded = tokenizer(batch, padding=True, truncation=True,
                            max_length=512, return_tensors="pt")
        device = next(model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            output = model(**encoded)
            # 取 [CLS] 向量
            vecs = output.last_hidden_state[:, 0, :]
            # L2 归一化
            vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
        all_vecs.append(vecs.cpu().numpy())
    return np.vstack(all_vecs).astype(np.float32)


def cosine_similarity_matrix(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """计算查询向量与文档向量的余弦相似度矩阵"""
    # 已归一化，直接点积
    return np.dot(query_vecs, doc_vecs.T)


# vocab 向量缓存：以 candidates 列表的 id + 长度 为键，避免在 align_*_split 的外层循环里
# 对几万个候选词重复跑 BGE 编码（这一步是之前 Step2 卡数小时的根因）
_CAND_VEC_CACHE: dict = {}


def _get_candidate_vecs_cached(candidates: List[str]) -> np.ndarray:
    key = (id(candidates), len(candidates))
    vecs = _CAND_VEC_CACHE.get(key)
    if vecs is None:
        vecs = encode_texts(candidates, is_query=False)
        _CAND_VEC_CACHE[key] = vecs
    return vecs


def clear_candidate_vec_cache():
    _CAND_VEC_CACHE.clear()


def find_best_matches(
    queries: List[str],
    candidates: List[str],
) -> List[Tuple[str, float, str]]:
    """
    为每个 query 在 candidates 中找最佳匹配
    Returns:
        List of (best_match, score, status)
        status: "exact" | "high" | "medium" | "low"
    """
    if not queries or not candidates:
        return [("", 0.0, "low")] * len(queries)

    # 先做精确匹配
    cand_set = set(candidates)
    results = []
    needs_embed_idx = []
    needs_embed_queries = []

    for i, q in enumerate(queries):
        if q in cand_set:
            results.append((q, 1.0, "exact"))
        else:
            results.append(None)
            needs_embed_idx.append(i)
            needs_embed_queries.append(q)

    if not needs_embed_queries:
        return results

    # 向量相似度（candidates 编码缓存：避免被外层循环重复调用时反复编码 vocab）
    query_vecs = encode_texts(needs_embed_queries, is_query=True)
    cand_vecs  = _get_candidate_vecs_cached(candidates)
    sim_matrix = cosine_similarity_matrix(query_vecs, cand_vecs)

    for j, orig_idx in enumerate(needs_embed_idx):
        best_cand_idx = int(np.argmax(sim_matrix[j]))
        score = float(sim_matrix[j][best_cand_idx])
        best_match = candidates[best_cand_idx]

        if score >= HIGH_SIM_THRESHOLD:
            status = "high"
        elif score >= LOW_SIM_THRESHOLD:
            status = "medium"
        else:
            status = "low"

        results[orig_idx] = (best_match, score, status)

    return results


def release_embedding_model():
    """v4.3 加强：把模型搬 CPU 再 del，真正回收显存"""
    global _embed_model, _embed_tokenizer
    if _embed_model is not None:
        import torch
        try:
            if hasattr(_embed_model, "cpu"):
                _embed_model.cpu()
        except Exception:
            pass
        del _embed_model
        _embed_model = None
        _embed_tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print("  [Embed] 已释放显存")

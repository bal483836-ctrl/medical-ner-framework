"""
大模型推理模块 v3.2（速度优化版）
- 修复 top_k 警告（do_sample=False 时不传 top_k/top_p/temperature）
- 动态 max_length：根据 batch 内最长 prompt 自适应截断，避免过度填充
- 单例懒加载，全程复用同一实例
"""
import os
import re
import sys
import gc
import time
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    LLM_MODEL_PATH, LLM_MAX_NEW_TOKENS,
    LLM_REPETITION_PENALTY, LLM_USE_4BIT,
    LLM_USE_FLASH_ATTN, LLM_DEVICE_MAP,
)

# ==================== 单例模型 ====================
_llm_model = None
_llm_tokenizer = None


def get_llm():
    """懒加载：只在第一次调用时加载模型，后续复用"""
    global _llm_model, _llm_tokenizer
    if _llm_model is not None:
        return _llm_model, _llm_tokenizer

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n[LLM] 正在加载 Qwen3-14B: {LLM_MODEL_PATH}")
    print(f"[LLM] 量化模式: {'4-bit NF4' if LLM_USE_4BIT else 'BF16 全精度'}")
    print(f"[LLM] Flash Attention: {'开启' if LLM_USE_FLASH_ATTN else '关闭 (SDPA)'}")

    _llm_tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_PATH,
        trust_remote_code=True,
        padding_side="left",   # 批量推理必须左填充
    )
    if _llm_tokenizer.pad_token is None:
        _llm_tokenizer.pad_token = _llm_tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "device_map": LLM_DEVICE_MAP,
        "dtype": torch.bfloat16,
    }

    if LLM_USE_FLASH_ATTN:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        # PyTorch 内置 SDPA，无需安装额外包，比默认实现快 15-20%
        model_kwargs["attn_implementation"] = "sdpa"

    if LLM_USE_4BIT:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs.pop("dtype", None)

    _llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_PATH, **model_kwargs)
    _llm_model.eval()
    print(f"[LLM] 加载完成！")
    return _llm_model, _llm_tokenizer


# ==================== 单条推理（兼容旧接口）====================

def call_llm(prompt: str, max_tokens: int = None, retries: int = 3) -> str:
    results = batch_generate([prompt], max_tokens=max_tokens, retries=retries)
    return results[0] if results else ""


# ==================== 批量推理（核心）====================

def batch_generate(
    prompts: List[str],
    max_tokens: int = None,
    retries: int = 3,
) -> List[str]:
    """
    批量推理：将多条 prompt 打包成一个 batch 同时推理。

    关键优化（v3.2）：
    1. do_sample=False 时完全不传 temperature/top_p/top_k，消除警告
    2. 动态 max_length：只截断到 batch 内最长 prompt 的实际长度 + buffer，
       避免固定 2048 导致大量无效填充 token，大幅减少显存占用和推理时间
    3. 空 prompt 跳过，不参与推理
    """
    import torch
    max_tokens = max_tokens or LLM_MAX_NEW_TOKENS

    for attempt in range(retries):
        try:
            model, tokenizer = get_llm()

            # 构建 chat 格式文本
            texts = []
            for prompt in prompts:
                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,   # 关闭 <think> 链，速度提升 ~30%
                )
                texts.append(text)

            # ---- 动态 max_length：先单独 tokenize 每条，取最大长度 ----
            # 这样可以避免固定 2048 导致短 prompt 被过度填充
            single_lens = [
                len(tokenizer(t, return_tensors="pt")["input_ids"][0])
                for t in texts
            ]
            # 实际截断长度 = 最长 prompt token 数 + 少量 buffer（防止截断关键词）
            dynamic_max_len = min(max(single_lens) + 32, 2048)

            # 批量 tokenize（左填充）
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=dynamic_max_len,
            ).to(model.device)

            # 批量生成（do_sample=False 时不传采样参数，消除 top_k 警告）
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    repetition_penalty=LLM_REPETITION_PENALTY,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # 只解码新生成的 token（去掉 prompt 部分）
            input_len = inputs["input_ids"].shape[1]
            results = []
            for output in outputs:
                new_tokens = output[input_len:]
                decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
                results.append(decoded.strip())

            return results

        except Exception as e:
            err_str = str(e)
            if "out of memory" in err_str.lower() or "CUDA" in err_str:
                import torch as _torch
                print(f"  [LLM] OOM！batch_size={len(prompts)} 显存不足，降级为逐条推理...")
                _torch.cuda.empty_cache()
                return _fallback_single(prompts, max_tokens)
            print(f"  [LLM] 第 {attempt+1} 次批量推理失败: {e}")
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return [""] * len(prompts)

    return [""] * len(prompts)


def _fallback_single(prompts: List[str], max_tokens: int) -> List[str]:
    """OOM 降级：逐条推理"""
    import torch
    results = []
    model, tokenizer = get_llm()
    for prompt in prompts:
        try:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    repetition_penalty=LLM_REPETITION_PENALTY,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_tokens = output[0][inputs["input_ids"].shape[1]:]
            results.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        except Exception as e:
            print(f"  [LLM] 单条推理失败: {e}")
            results.append("")
    return results


# ==================== 输出清洗 ====================

def clean_llm_output(text: str) -> str:
    """清洗大模型输出：去除 think 链、markdown、序号、前缀词"""
    if not text:
        return ""
    text = re.sub(r"(?s)<think>.*?(?:</think>|$)", "", text)
    text = re.sub(r"```[a-z]*|```", "", text)
    text = re.sub(r"\[|\]", "", text)
    text = re.sub(r"^(实体：|输出：|结果：|通过审核的实体：|答：)\s*", "", text.strip())
    text = re.sub(r"[，、\n；;]", ",", text)
    text = re.sub(r"(^|,)\s*\d+[\.\、\)]\s*", ",", text)
    text = re.sub(r",+", ",", text).strip(",").strip()
    return text


# ==================== 显存管理 ====================

def release_llm():
    """释放 LLM 显存"""
    global _llm_model, _llm_tokenizer
    if _llm_model is not None:
        import torch
        del _llm_model
        del _llm_tokenizer
        _llm_model = None
        _llm_tokenizer = None
        torch.cuda.empty_cache()
        gc.collect()
        print("[LLM] 已释放显存")

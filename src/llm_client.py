"""
LLM 客户端（双后端）
  - main: Qwen3-32B（主抽取与断言）
  - reflect: DeepSeek（反思校验）

单例懒加载，按名取用：get_llm("main") / get_llm("reflect")
"""
import os
import re
import sys
import gc
import time
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    LLM_MODEL_PATH, REFLECT_MODEL_PATH, LLM_MAX_NEW_TOKENS,
    LLM_REPETITION_PENALTY, LLM_USE_4BIT,
    LLM_USE_FLASH_ATTN, LLM_DEVICE_MAP,
)

_INSTANCES = {}   # name -> (model, tokenizer)


def _load(path: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n[LLM] 加载 {path}")
    print(f"[LLM] 4bit={LLM_USE_4BIT}  FlashAttn={LLM_USE_FLASH_ATTN}")

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = {
        "trust_remote_code": True,
        "device_map": LLM_DEVICE_MAP,
        "dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2" if LLM_USE_FLASH_ATTN else "sdpa",
    }
    if LLM_USE_4BIT:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        kwargs.pop("dtype", None)

    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    model.eval()
    print(f"[LLM] 加载完成: {path}")
    return model, tok


def get_llm(name: str = "main"):
    if name in _INSTANCES:
        return _INSTANCES[name]
    path = LLM_MODEL_PATH if name == "main" else REFLECT_MODEL_PATH
    _INSTANCES[name] = _load(path)
    return _INSTANCES[name]


def batch_generate(
    prompts: List[str],
    max_tokens: int = None,
    retries: int = 3,
    model_name: str = "main",
) -> List[str]:
    """批量推理。失败 OOM 时自动降级单条。"""
    import torch
    max_tokens = max_tokens or LLM_MAX_NEW_TOKENS

    for attempt in range(retries):
        try:
            model, tokenizer = get_llm(model_name)
            texts = []
            for p in prompts:
                msgs = [{"role": "user", "content": p}]
                t = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
                texts.append(t)

            single_lens = [len(tokenizer(t, return_tensors="pt")["input_ids"][0]) for t in texts]
            dyn_len = min(max(single_lens) + 32, 4096)

            inputs = tokenizer(
                texts, return_tensors="pt", padding=True,
                truncation=True, max_length=dyn_len,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    repetition_penalty=LLM_REPETITION_PENALTY,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            in_len = inputs["input_ids"].shape[1]
            return [tokenizer.decode(o[in_len:], skip_special_tokens=True).strip()
                    for o in outputs]

        except Exception as e:
            if "out of memory" in str(e).lower():
                import torch as _t
                _t.cuda.empty_cache()
                print(f"  [LLM] OOM 降级单条 batch={len(prompts)}")
                return _fallback_single(prompts, max_tokens, model_name)
            print(f"  [LLM] 第 {attempt+1}/{retries} 次失败: {e}")
            if attempt < retries - 1:
                time.sleep(2)
    return [""] * len(prompts)


def _fallback_single(prompts, max_tokens, model_name):
    import torch
    model, tokenizer = get_llm(model_name)
    results = []
    for p in prompts:
        try:
            msgs = [{"role": "user", "content": p}]
            t = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            ins = tokenizer(t, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **ins, max_new_tokens=max_tokens, do_sample=False,
                    repetition_penalty=LLM_REPETITION_PENALTY,
                    pad_token_id=tokenizer.pad_token_id,
                )
            results.append(tokenizer.decode(
                out[0][ins["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip())
        except Exception as e:
            print(f"  [LLM] 单条失败: {e}")
            results.append("")
    return results


def call_llm(prompt: str, max_tokens: int = None, model_name: str = "main") -> str:
    r = batch_generate([prompt], max_tokens=max_tokens, model_name=model_name)
    return r[0] if r else ""


def clean_llm_output(text: str) -> str:
    """去 think 链 / markdown / 序号 / 前缀。"""
    if not text:
        return ""
    text = re.sub(r"(?s)<think>.*?(?:</think>|$)", "", text)
    text = re.sub(r"```[a-z]*|```", "", text)
    text = re.sub(r"\[|\]", "", text)
    text = re.sub(r"^(实体|输出|结果|答|分类|标签)[：:]\s*", "", text.strip())
    text = re.sub(r"[，、\n；;]", ",", text)
    text = re.sub(r"(^|,)\s*\d+[\.\、\)]\s*", ",", text)
    text = re.sub(r",+", ",", text).strip(",").strip()
    return text


def release_llm(name: str = None):
    import torch
    if name is None:
        keys = list(_INSTANCES.keys())
    else:
        keys = [name] if name in _INSTANCES else []
    for k in keys:
        del _INSTANCES[k]
    torch.cuda.empty_cache()
    gc.collect()

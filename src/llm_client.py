"""
LLM 客户端（双后端：vLLM / transformers）
  - main: Qwen3-32B（主抽取与断言）
  - reflect: DeepSeek（反思校验）

单例懒加载，按名取用：get_llm("main") / get_llm("reflect")
后端通过 MNER_LLM_BACKEND=vllm|hf 控制，默认 vllm（吞吐显著高）。
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
    LLM_BACKEND, LLM_VLLM_GPU_MEMORY_UTIL, LLM_VLLM_MAX_MODEL_LEN, LLM_VLLM_TP_SIZE,
    LLM_VLLM_ENFORCE_EAGER,
)

# name -> {"backend": "vllm"|"hf", ...payload}
_INSTANCES = {}
_LOAD_FAILED = {}


def _validate_local_model_path(path: str):
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"[LLM] 模型目录不存在: {path}\n"
            f"  → 请确认已下载模型到该路径，或通过 MNER_MODEL_ROOT / MNER_LLM_PATH 指定。"
        )
    if not os.path.exists(os.path.join(path, "config.json")):
        raise FileNotFoundError(
            f"[LLM] 目录存在但缺少 config.json: {path}\n"
            f"  → 目录内容: {sorted(os.listdir(path))[:20]}"
        )


def _detect_quantization(path: str) -> str:
    """返回 'awq' / 'gptq' / '' 。"""
    import json
    try:
        with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        qc = cfg.get("quantization_config") or {}
        method = (qc.get("quant_method") or "").lower()
        if method in ("awq", "gptq"):
            return method
    except Exception:
        pass
    base = os.path.basename(path).upper()
    if "AWQ" in base:
        return "awq"
    if "GPTQ" in base:
        return "gptq"
    return ""


# -------------------- vLLM 后端 --------------------

def _load_vllm(path: str):
    """加载 vLLM 引擎。需 pip install vllm。"""
    try:
        from src.embedding_model import release_embedding_model
        import torch
        release_embedding_model()
        torch.cuda.empty_cache(); gc.collect()
    except Exception:
        pass

    from vllm import LLM, SamplingParams  # noqa: F401  (确认安装)

    quant = _detect_quantization(path)
    # 非量化大模型（如 DeepSeek-V2-Lite BF16 ≈ 30GB）在 32GB 卡上需关掉 CUDA Graph
    # 否则 cudagraph 捕获会 OOM。可用 MNER_VLLM_EAGER=true 强制开启。
    eager = LLM_VLLM_ENFORCE_EAGER or (not quant)
    kwargs = dict(
        model=path,
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=LLM_VLLM_GPU_MEMORY_UTIL,
        max_model_len=LLM_VLLM_MAX_MODEL_LEN,
        tensor_parallel_size=LLM_VLLM_TP_SIZE,
        enforce_eager=eager,
        disable_log_stats=True,
    )
    if quant:
        kwargs["quantization"] = quant
    print(f"[LLM/vLLM] 加载 {path}  quant={quant or 'none'}  "
          f"mem_util={LLM_VLLM_GPU_MEMORY_UTIL}  max_len={LLM_VLLM_MAX_MODEL_LEN}  "
          f"eager={eager}")

    llm = LLM(**kwargs)
    tok = llm.get_tokenizer()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"[LLM/vLLM] 加载完成: {path}")
    return {"backend": "vllm", "llm": llm, "tokenizer": tok}


# -------------------- transformers 后端 --------------------

def _load_hf(path: str):
    try:
        from src.embedding_model import release_embedding_model
        import torch
        release_embedding_model()
        torch.cuda.empty_cache(); gc.collect()
    except Exception:
        pass
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n[LLM/HF] 加载 {path}")
    print(f"[LLM/HF] 4bit={LLM_USE_4BIT}  FlashAttn={LLM_USE_FLASH_ATTN}")

    quant = _detect_quantization(path)
    is_prequant = bool(quant)

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = {
        "trust_remote_code": True,
        "device_map": LLM_DEVICE_MAP,
        "attn_implementation": "flash_attention_2" if LLM_USE_FLASH_ATTN else "sdpa",
    }
    if is_prequant:
        print(f"[LLM/HF] 检测到预量化模型（{quant.upper()}），跳过 bnb 4bit、不指定 dtype")
    else:
        kwargs["dtype"] = torch.bfloat16
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
    print(f"[LLM/HF] 加载完成: {path}")
    return {"backend": "hf", "model": model, "tokenizer": tok}


# -------------------- 统一入口 --------------------

def _load(path: str):
    _validate_local_model_path(path)
    backend = LLM_BACKEND
    if backend == "vllm":
        try:
            return _load_vllm(path)
        except ImportError as e:
            print(f"[LLM] vLLM 未安装（{e}），回退 transformers 后端。"
                  f" 建议: pip install vllm")
            return _load_hf(path)
    return _load_hf(path)


def get_llm(name: str = "main"):
    if name in _INSTANCES:
        return _INSTANCES[name]
    if name in _LOAD_FAILED:
        raise _LOAD_FAILED[name]
    # 加载新模型前释放其他实例：NER 任务里 main/reflect 不会同时使用，
    # 而 vLLM 单个模型就占用大量显存（Qwen3-32B-AWQ ~17 GB），不释放会 OOM。
    others = [k for k in _INSTANCES.keys() if k != name]
    for k in others:
        print(f"  [LLM] 加载 {name} 前先释放 {k}")
        release_llm(k)
    path = LLM_MODEL_PATH if name == "main" else REFLECT_MODEL_PATH
    try:
        _INSTANCES[name] = _load(path)
    except Exception as e:
        _LOAD_FAILED[name] = e
        raise
    return _INSTANCES[name]


def _apply_chat(tokenizer, prompt: str) -> str:
    msgs = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )


def batch_generate(
    prompts: List[str],
    max_tokens: int = None,
    retries: int = 3,
    model_name: str = "main",
) -> List[str]:
    """批量推理。vLLM 后端会自动连续批处理；HF 后端 OOM 时降级单条。"""
    max_tokens = max_tokens or LLM_MAX_NEW_TOKENS
    if not prompts:
        return []

    inst = get_llm(model_name)
    backend = inst["backend"]
    tokenizer = inst["tokenizer"]
    texts = [_apply_chat(tokenizer, p) for p in prompts]

    # --------- vLLM 路径 ---------
    if backend == "vllm":
        from vllm import SamplingParams
        sp = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            repetition_penalty=LLM_REPETITION_PENALTY,
        )
        try:
            outs = inst["llm"].generate(texts, sp, use_tqdm=False)
        except Exception as e:
            print(f"  [LLM/vLLM] 推理失败: {e}")
            return [""] * len(prompts)
        # vLLM 按输入顺序返回结果
        return [o.outputs[0].text.strip() if o.outputs else "" for o in outs]

    # --------- HF 路径 ---------
    import torch
    model = inst["model"]
    for attempt in range(retries):
        try:
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
                torch.cuda.empty_cache()
                print(f"  [LLM/HF] OOM 降级单条 batch={len(prompts)}")
                return _fallback_single_hf(prompts, max_tokens, model_name)
            print(f"  [LLM/HF] 第 {attempt+1}/{retries} 次失败: {e}")
            if attempt < retries - 1:
                time.sleep(2)
    return [""] * len(prompts)


def _fallback_single_hf(prompts, max_tokens, model_name):
    import torch
    inst = get_llm(model_name)
    model, tokenizer = inst["model"], inst["tokenizer"]
    results = []
    for p in prompts:
        try:
            t = _apply_chat(tokenizer, p)
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
            print(f"  [LLM/HF] 单条失败: {e}")
            results.append("")
    return results


def call_llm(prompt: str, max_tokens: int = None, model_name: str = "main") -> str:
    r = batch_generate([prompt], max_tokens=max_tokens, model_name=model_name)
    return r[0] if r else ""


def clean_llm_output(text: str) -> str:
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


def _shutdown_vllm(llm) -> None:
    """显式关闭 vLLM 引擎的 EngineCore 子进程，否则显存不会真正释放。"""
    # 新版 vLLM 提供 LLM.shutdown()
    for attr in ("shutdown",):
        fn = getattr(llm, attr, None)
        if callable(fn):
            try:
                fn()
                return
            except Exception:
                pass
    # 兜底：手动停掉 engine_core / executor
    engine = getattr(llm, "llm_engine", None)
    if engine is None:
        return
    core = getattr(engine, "engine_core", None)
    if core is not None:
        for m in ("shutdown", "close"):
            fn = getattr(core, m, None)
            if callable(fn):
                try:
                    fn()
                    break
                except Exception:
                    pass
    executor = getattr(engine, "model_executor", None)
    if executor is not None:
        fn = getattr(executor, "shutdown", None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass


def release_llm(name: str = None):
    """释放显存。vLLM 引擎需显式 shutdown 才能真正回收 GPU 内存。"""
    import torch
    keys = list(_INSTANCES.keys()) if name is None else ([name] if name in _INSTANCES else [])
    for k in keys:
        inst = _INSTANCES.pop(k)
        try:
            if inst["backend"] == "vllm":
                _shutdown_vllm(inst["llm"])
                inst["llm"] = None
            else:
                try:
                    inst["model"].cpu()
                except Exception:
                    pass
                inst["model"] = None
            inst["tokenizer"] = None
        except Exception as e:
            print(f"  [LLM] 释放 {k} 出错（忽略）: {e}")
    # 多轮 GC + empty_cache，给 vLLM 子进程退出时间
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    # vLLM 子进程退出本身需要约 1-2 秒
    time.sleep(2)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if keys:
        print(f"  [LLM] 释放显存：{keys}")

"""
全局工具：随机种子固定 + 5090 显卡环境检测 + 阶段输出格式化
"""
import os
import random
import sys
from typing import Any

GLOBAL_SEED = 42


def set_global_seed(seed: int = GLOBAL_SEED, deterministic: bool = True):
    """
    固定所有随机源：Python / NumPy / PyTorch / CUDA / 环境变量
    deterministic=True 时强制 cudnn 确定性（略慢但 100% 可复现）
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        # CUBLAS 确定性
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(False)  # True 会让某些 kernel 不可用
        except Exception:
            pass
    except ImportError:
        pass
    print(f"🔒 [Seed] 全局种子已固定: {seed}  (deterministic={deterministic})")


def detect_gpu() -> dict:
    """检测 GPU 并提示 5090 优化建议。"""
    info = {"available": False, "name": "", "memory_gb": 0, "is_5090": False}
    try:
        import torch
        if torch.cuda.is_available():
            info["available"] = True
            info["name"] = torch.cuda.get_device_name(0)
            info["memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
            info["is_5090"] = "5090" in info["name"]
    except ImportError:
        pass
    return info


def print_gpu_banner():
    info = detect_gpu()
    if not info["available"]:
        print("⚠️ [GPU] 未检测到 CUDA，将以 CPU 模式跑通（速度会很慢）")
        return
    print(f"🖥️  [GPU] {info['name']}  显存 {info['memory_gb']} GB")
    if info["is_5090"]:
        print("   ✨ 检测到 RTX 5090（Blackwell，32GB）→ 已启用大 batch 推理参数")
        # 提示用户开启 flash attention
        if os.environ.get("MNER_USE_FLASH_ATTN", "false").lower() != "true":
            print("   💡 建议 export MNER_USE_FLASH_ATTN=true 进一步提速 30%")


# ==================== 阶段输出格式化 ====================

def stage_banner(stage: str, desc: str = ""):
    """统一的阶段开头横幅。"""
    print("\n" + "=" * 70)
    print(f"  📍 {stage}  {desc}")
    print("=" * 70)


def sub_banner(name: str):
    print(f"\n  ▶ {name}")


def kv_print(d: dict, prefix: str = "    "):
    """格式化打印 dict。"""
    for k, v in d.items():
        if isinstance(v, (list, tuple)) and len(v) > 8:
            v_str = f"{v[:5]} ... (共 {len(v)} 项)"
        else:
            v_str = v
        print(f"{prefix}{k}: {v_str}")


def preview_items(items: list, n: int = 3, fields: tuple = None):
    """打印列表前 n 条的关键字段。"""
    if not items:
        print("    （空）")
        return
    print(f"    预览前 {min(n, len(items))} 条（共 {len(items)}）：")
    for i, it in enumerate(items[:n]):
        if isinstance(it, dict):
            if fields:
                shown = {k: it.get(k) for k in fields if k in it}
            else:
                shown = {k: it[k] for k in list(it.keys())[:6]}
            print(f"      [{i}] {shown}")
        else:
            print(f"      [{i}] {str(it)[:200]}")

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


# ==================== 通用断点续传 ====================

import json as _json

def atomic_save_json(data, path: str):
    """先写 .tmp 再 rename，避免写一半被中断导致 JSON 损坏。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        _json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_checkpoint(path: str):
    """加载已有断点；损坏自动跳过。"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return _json.load(f)
    except (_json.JSONDecodeError, OSError) as e:
        print(f"  ⚠️ 断点损坏 {path}: {e}，重新开始")
        return None


def resume_done_keys(path: str, key_field: str) -> set:
    """从已有 checkpoint 读出已完成 key 集合。"""
    data = load_checkpoint(path) or []
    return {str(x.get(key_field, i)) for i, x in enumerate(data)}


class PeriodicSaver:
    """
    每处理 every 条就 atomic save 一次。
    用法：
        saver = PeriodicSaver(out_path, every=50)
        for x in items:
            ...
            saver.tick(results)   # results 是当前累积的 list
        saver.finalize(results)
    """
    def __init__(self, path: str, every: int = 50):
        self.path = path
        self.every = max(1, every)
        self.n = 0

    def tick(self, results):
        self.n += 1
        if self.n % self.every == 0:
            atomic_save_json(results, self.path)

    def finalize(self, results):
        atomic_save_json(results, self.path)
        print(f"    💾 已保存 {len(results)} 条 → {self.path}")

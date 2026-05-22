"""
断言数据分布检测 + 增强（阶段 7）— v4.2

策略升级：
  1) 统计 4 类标签分布
  2) 找出"低于目标比例"的所有少数类
  3) **按类目标补足**：每类增强到接近 max_class_count * AUG_TARGET_RATIO
  4) LLM 改写 + 二次断言确认（label 一致才保留）
  5) 限制增强样本最多不超过原样本 N 倍（防过拟合）

吸取 v21 + run_finetune_classifier 的定向增强思路。
"""
import os
import sys
from collections import Counter
from typing import Dict, List
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    ASSERTION_LABELS, AUG_MIN_CLASS_RATIO, AUG_MULTIPLIER, AUG_TARGET_RATIO,
)
from src.llm_client import batch_generate
from src.assertion_annotator import annotate
from src.utils import set_global_seed, atomic_save_json, load_checkpoint

AUG_BATCH = 8


def label_distribution(samples: List[Dict]) -> Dict[str, float]:
    c = Counter(s.get("label", "") for s in samples)
    total = sum(c.values()) or 1
    return {lab: round(c.get(lab, 0) / total, 4) for lab in ASSERTION_LABELS}


def label_counts(samples: List[Dict]) -> Dict[str, int]:
    c = Counter(s.get("label", "") for s in samples)
    return {lab: c.get(lab, 0) for lab in ASSERTION_LABELS}


def _paraphrase_prompt(entity: str, context: str, label: str) -> str:
    """
    定向改写：按目标 label 给出明确语义约束。
    """
    label_hints = {
        "确定": "实体存在/确认/已确诊/已发现/查体有",
        "疑似": "考虑/可能/不排除/疑似/印象/待排查",
        "无":   "否认/未见/不伴/排除/无 + 实体",
        "知识事实": "医学教材/科普/泛指/一般描述，不针对具体患者",
    }
    hint = label_hints.get(label, "")
    return f"""你是医疗文本生成专家。改写下面的临床语境，要求：
1) 实体「{entity}」必须保留原词、字面出现；
2) 改写后语境对【{entity}】的断言类型必须仍是【{label}】（{hint}）；
3) 用不同的句式/同义表达；
4) 直接输出改写后的语境，不要解释，不要带前缀。

原语境：
{context}

改写后："""


def augment(samples: List[Dict],
            target_ratio: float = AUG_TARGET_RATIO,
            max_aug_per_class: int = None,
            checkpoint_path: str = None) -> List[Dict]:
    """
    每个少数类增强到 max_count * target_ratio 条左右 + LLM 二次断言确认。
    checkpoint_path 不为 None 时：
      - 增强生成阶段每 batch 落盘到 {checkpoint_path}.gen.json
      - 二次确认阶段保存到 {checkpoint_path}.verify.json
      - 重启时直接复用
    """
    set_global_seed()
    counts = label_counts(samples)
    max_n = max(counts.values()) if counts else 0
    target_n = int(max_n * target_ratio)
    if max_aug_per_class is None:
        max_aug_per_class = max_n  # 不超过 1:1

    aug_jobs = []
    for lab, c in counts.items():
        deficit = max(0, target_n - c)
        deficit = min(deficit, max_aug_per_class)
        if deficit > 0 and c > 0:
            aug_jobs.append((lab, deficit))
    if not aug_jobs:
        print(f"  [Augment] 已均衡（max={max_n}），跳过")
        return samples

    print(f"  [Augment] 目标每类 ≈ {target_n}（max×{target_ratio}）")
    for lab, n in aug_jobs:
        print(f"    需要增强 {lab}: +{n} 条（当前 {counts[lab]}）")

    by_label = {lab: [s for s in samples if s.get("label") == lab]
                for lab in ASSERTION_LABELS}

    # ---- 改写阶段（断点：复用已生成 generated）----
    gen_ckpt = (checkpoint_path + ".gen.json") if checkpoint_path else None
    generated: List[Dict] = load_checkpoint(gen_ckpt) if gen_ckpt else None
    if generated is None:
        generated = []
    else:
        print(f"  [Resume] 复用增强候选 {len(generated)} 条")

    # 按类已有数量
    have_per_class = Counter(g["expected_label"] for g in generated)

    batch_count = 0
    for lab, need in aug_jobs:
        pool = by_label[lab]
        if not pool:
            continue
        target = need * 2
        already = have_per_class.get(lab, 0)
        if already >= target:
            continue
        cursor = already
        for bs in tqdm(range(already, target, AUG_BATCH), desc=f"aug-{lab}"):
            batch_src = []
            for k in range(AUG_BATCH):
                if cursor >= target:
                    break
                batch_src.append(pool[cursor % len(pool)])
                cursor += 1
            if not batch_src:
                break
            prompts = [_paraphrase_prompt(s["entity"], s["context"], s["label"])
                       for s in batch_src]
            resps = batch_generate(prompts, max_tokens=200, model_name="main")
            for src, ctx in zip(batch_src, resps):
                ctx = (ctx or "").strip()
                if not ctx or src["entity"] not in ctx:
                    continue
                generated.append({
                    **src,
                    "context": ctx,
                    "label": "",
                    "augmented": True,
                    "expected_label": src["label"],
                })
            batch_count += 1
            if gen_ckpt and batch_count % 5 == 0:
                atomic_save_json(generated, gen_ckpt)
    if gen_ckpt:
        atomic_save_json(generated, gen_ckpt)

    # ---- 二次确认（断点由 annotator 内部接管）----
    print(f"  [Augment] 候选 {len(generated)} 条，做二次断言确认 ...")
    verify_ckpt = (checkpoint_path + ".verify.json") if checkpoint_path else None
    confirmed = annotate(generated, vote_passes=1, output_path=verify_ckpt)
    kept = [g for g in confirmed if g.get("label") == g.get("expected_label")]
    drop = len(confirmed) - len(kept)
    print(f"  [Augment] 二次确认通过 {len(kept)} / 丢弃 {drop}")

    # 按类截断到 need
    final = []
    used_per_lab = Counter()
    needs_dict = {lab: n for lab, n in aug_jobs}
    for g in kept:
        lab = g.get("label")
        if used_per_lab[lab] < needs_dict.get(lab, 0):
            final.append(g)
            used_per_lab[lab] += 1

    print(f"  [Augment] 最终采用 {len(final)} 条；增强后分布: "
          f"{label_distribution(samples + final)}")
    return samples + final


# 兼容旧调用
def find_minority(samples, min_ratio=AUG_MIN_CLASS_RATIO):
    dist = label_distribution(samples)
    return [lab for lab, r in dist.items() if r < min_ratio]

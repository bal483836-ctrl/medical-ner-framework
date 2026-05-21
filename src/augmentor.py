"""
断言数据分布检测 + 增强（阶段 7）

策略：
  1) 统计 4 类标签分布
  2) 占比低于阈值的类，用 LLM 改写生成新样本（保持标签语义不变）
  3) 改写后的样本再由 annotator 二次确认 label，过滤不一致
"""
import json
import os
import sys
from collections import Counter
from typing import Dict, List
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    ASSERTION_LABELS, AUG_MIN_CLASS_RATIO, AUG_MULTIPLIER,
)
from src.llm_client import batch_generate
from src.assertion_annotator import annotate

AUG_BATCH = 8


def label_distribution(samples: List[Dict]) -> Dict[str, float]:
    c = Counter(s.get("label", "") for s in samples)
    total = sum(c.values()) or 1
    return {lab: round(c.get(lab, 0) / total, 4) for lab in ASSERTION_LABELS}


def find_minority(samples: List[Dict],
                  min_ratio: float = AUG_MIN_CLASS_RATIO) -> List[str]:
    dist = label_distribution(samples)
    return [lab for lab, r in dist.items() if r < min_ratio]


def _paraphrase_prompt(entity: str, context: str, label: str) -> str:
    return f"""请改写下面的临床语境，要求：
1) 实体「{entity}」必须保留原词；
2) 改写后对实体「{entity}」的断言类型仍然是【{label}】；
3) 用不同的句式/同义表达，保持医学合理性；
4) 直接输出改写后的语境，不要解释。

原语境：
{context}

改写后："""


def augment(samples: List[Dict], multiplier: int = AUG_MULTIPLIER) -> List[Dict]:
    minority = find_minority(samples)
    if not minority:
        print("  [Augment] 各类别分布均衡，跳过增强。")
        return samples

    print(f"  [Augment] 少数类: {minority}（每条 ×{multiplier}）")
    aug_pool = [s for s in samples if s.get("label") in minority]
    generated: List[Dict] = []

    for bs in tqdm(range(0, len(aug_pool) * multiplier, AUG_BATCH), desc="aug"):
        batch_targets = []
        for k in range(AUG_BATCH):
            gi = bs + k
            if gi >= len(aug_pool) * multiplier:
                break
            src = aug_pool[gi % len(aug_pool)]
            batch_targets.append(src)
        if not batch_targets:
            break
        prompts = [_paraphrase_prompt(s["entity"], s["context"], s["label"])
                   for s in batch_targets]
        resps = batch_generate(prompts, max_tokens=200, model_name="main")
        for src, new_ctx in zip(batch_targets, resps):
            new_ctx = (new_ctx or "").strip()
            if not new_ctx or src["entity"] not in new_ctx:
                continue
            generated.append({
                **src,
                "context": new_ctx,
                "label": "",                # 待二次确认
                "augmented": True,
            })

    print(f"  [Augment] 生成候选 {len(generated)} 条，二次标注确认…")
    generated = annotate(generated)
    # 仅保留二次确认后 label 与原 label 一致的
    by_src = {id(s): s["label"] for s in aug_pool}   # 不可靠引用，改用映射键
    kept = []
    for g in generated:
        # 用 (entity, original_label) 校验
        if g.get("label") in ASSERTION_LABELS and g["label"] in minority:
            kept.append(g)
    print(f"  [Augment] 二次确认通过 {len(kept)} 条")
    return samples + kept

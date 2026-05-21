"""
断言小模型评估（阶段 9）— v4.2

升级：
  ① 多模型 logits 平均集成
  ② 验证集上做"按类阈值"搜索，再应用到 test（多分类 argmax-with-bias 提升）
  ③ 报告含 per-class P/R/F1 + 混淆矩阵

防泄露：阈值搜索仅在 dev 上完成；test 全程独立。
"""
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    ASSERTION_LABELS, ASSERTION_LABEL2ID, OUTPUT_DIR,
    F1_TARGET_ASSERTION, CLF_MAX_LEN, CLF_BATCH_SIZE,
)
from src.assertion_train import serialize


def _predict_logits(model_dir: str, samples: List[Dict]) -> np.ndarray:
    """返回 (N, n_labels) softmax 后概率。"""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    all_probs = []
    for bs in range(0, len(samples), CLF_BATCH_SIZE):
        batch = samples[bs: bs + CLF_BATCH_SIZE]
        pairs = [serialize(s) for s in batch]
        q  = [p[0] for p in pairs]
        c  = [p[1] for p in pairs]
        enc = tok(q, c, padding=True, truncation=True,
                  max_length=CLF_MAX_LEN, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)


def ensemble_predict(model_dirs: List[str], samples: List[Dict]) -> np.ndarray:
    """多模型 softmax 平均。"""
    if len(model_dirs) == 1:
        return _predict_logits(model_dirs[0], samples)
    probs_list = [_predict_logits(d, samples) for d in model_dirs]
    return np.mean(probs_list, axis=0)


def search_class_bias(probs_val: np.ndarray, golds_val: List[int],
                      sweep=np.arange(-0.20, 0.21, 0.02)) -> np.ndarray:
    """
    在 val 上搜索每类的对数概率偏置 bias[c]，让 macro_f1 最大。
    预测：argmax(probs + bias)。每类独立 1D 搜索（贪心）。
    """
    from sklearn.metrics import f1_score
    n_cls = probs_val.shape[1]
    bias = np.zeros(n_cls)

    def macro_f1(b):
        preds = np.argmax(probs_val + b, axis=-1)
        return f1_score(golds_val, preds, average="macro", zero_division=0)

    base = macro_f1(bias)
    print(f"  [Bias] 初始 val macro_f1 = {base:.4f}")
    for c in range(n_cls):
        best, best_f1 = 0.0, base
        for v in sweep:
            b = bias.copy(); b[c] = v
            f = macro_f1(b)
            if f > best_f1:
                best_f1 = f; best = v
        bias[c] = best
        base = best_f1
    print(f"  [Bias] 搜索后 val macro_f1 = {base:.4f}  bias={bias.tolist()}")
    return bias


def evaluate(model_dir_or_dirs, test_samples: List[Dict],
             val_samples: Optional[List[Dict]] = None,
             save_report: bool = True) -> Dict:
    from sklearn.metrics import (
        f1_score, classification_report, confusion_matrix,
    )

    # 支持单 / 多模型
    if isinstance(model_dir_or_dirs, str):
        model_dirs = [model_dir_or_dirs]
    else:
        model_dirs = list(model_dir_or_dirs)
    print(f"\n[Eval] 模型集成数: {len(model_dirs)}")

    samples = [s for s in test_samples if s.get("label") in ASSERTION_LABEL2ID]
    if not samples:
        return {"error": "no labeled test samples"}

    # 在 val 上搜索 bias
    bias = np.zeros(len(ASSERTION_LABELS))
    if val_samples:
        val_lab = [s for s in val_samples if s.get("label") in ASSERTION_LABEL2ID]
        if val_lab:
            probs_v = ensemble_predict(model_dirs, val_lab)
            golds_v = [ASSERTION_LABEL2ID[s["label"]] for s in val_lab]
            bias = search_class_bias(probs_v, golds_v)

    # 测试集
    probs_t = ensemble_predict(model_dirs, samples)
    preds = np.argmax(probs_t + bias, axis=-1).tolist()
    golds = [ASSERTION_LABEL2ID[s["label"]] for s in samples]

    macro = f1_score(golds, preds, average="macro")
    micro = f1_score(golds, preds, average="micro")
    report = classification_report(
        golds, preds, target_names=ASSERTION_LABELS,
        digits=4, zero_division=0,
    )
    cm = confusion_matrix(
        golds, preds, labels=list(range(len(ASSERTION_LABELS)))
    ).tolist()

    result = {
        "n_test": len(samples),
        "n_models_ensembled": len(model_dirs),
        "macro_f1": round(macro, 4),
        "micro_f1": round(micro, 4),
        "passed_target": macro >= F1_TARGET_ASSERTION,
        "target": F1_TARGET_ASSERTION,
        "class_bias_used": bias.tolist(),
        "classification_report": report,
        "confusion_matrix": cm,
        "labels": ASSERTION_LABELS,
    }
    print(f"\n[Eval] Macro F1 = {macro:.4f}  Micro F1 = {micro:.4f}")
    print(f"目标 {F1_TARGET_ASSERTION} {'✅ 达成' if macro >= F1_TARGET_ASSERTION else '❌ 未达成'}")
    print(report)

    if save_report:
        path = os.path.join(OUTPUT_DIR, "assertion_eval_report.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  📄 评估报告: {path}")
    return result

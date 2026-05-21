"""
断言小模型评估（阶段 9）— v4.1
目标：macro F1 ≥ 0.90
test split 严格独立，未参与训练。
"""
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    ASSERTION_LABELS, ASSERTION_LABEL2ID, OUTPUT_DIR,
    F1_TARGET_ASSERTION, CLF_MAX_LEN, CLF_BATCH_SIZE,
)
from src.assertion_train import serialize


def evaluate(model_dir: str, test_samples: List[Dict],
             save_report: bool = True) -> Dict:
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.metrics import (
        f1_score, classification_report, confusion_matrix,
    )

    samples = [s for s in test_samples if s.get("label") in ASSERTION_LABEL2ID]
    if not samples:
        return {"error": "no labeled test samples"}

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    preds, golds = [], []
    for bs in range(0, len(samples), CLF_BATCH_SIZE):
        batch = samples[bs: bs + CLF_BATCH_SIZE]
        pairs = [serialize(s) for s in batch]
        queries  = [p[0] for p in pairs]
        contexts = [p[1] for p in pairs]
        enc = tok(queries, contexts, padding=True, truncation=True,
                  max_length=CLF_MAX_LEN, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        preds.extend(logits.argmax(-1).cpu().tolist())
        golds.extend(ASSERTION_LABEL2ID[s["label"]] for s in batch)

    macro = f1_score(golds, preds, average="macro")
    micro = f1_score(golds, preds, average="micro")
    report = classification_report(
        golds, preds, target_names=ASSERTION_LABELS,
        digits=4, zero_division=0,
    )
    cm = confusion_matrix(golds, preds, labels=list(range(len(ASSERTION_LABELS)))).tolist()

    result = {
        "n_test": len(samples),
        "macro_f1": round(macro, 4),
        "micro_f1": round(micro, 4),
        "passed_target": macro >= F1_TARGET_ASSERTION,
        "target": F1_TARGET_ASSERTION,
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

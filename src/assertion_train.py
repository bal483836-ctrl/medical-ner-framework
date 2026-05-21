"""
断言小模型训练（阶段 8）

基座：chinese-roberta-wwm-ext
输入文本：实体 [SEP] 语境 [SEP] KG 扩展（截断到 CLF_MAX_LEN）
训练集：train + dev 的 LLM 断言结果（严格隔离 test 防泄露）

防数据泄露：
  - test split 的样本完全不参与训练
  - 训练前按 (dialogue_id / doc_id) 维度 group split 验证集（不会让同文档实体跨集合）
"""
import json
import os
import random
import sys
from collections import Counter
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    ASSERTION_LABELS, ASSERTION_LABEL2ID,
    CLASSIFIER_BASE_PATH, CLF_MAX_LEN, CLF_BATCH_SIZE,
    CLF_LEARNING_RATE, CLF_EPOCHS, CLF_WARMUP_RATIO,
    CLF_WEIGHT_DECAY, CLF_SEED, OUTPUT_DIR,
)


def _set_seed(seed: int = CLF_SEED):
    import numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _serialize(sample: Dict) -> str:
    ent = sample.get("entity", "")
    ctx = sample.get("context", "")
    exp = sample.get("expansion", {}) or {}
    exp_txt = " ".join(
        f"{k}:{','.join(v[:3])}" for k, v in exp.items() if v
    )
    return f"实体: {ent} [SEP] 语境: {ctx} [SEP] 相关: {exp_txt}"


def _doc_key(sample: Dict) -> str:
    return str(sample.get("doc_id") or sample.get("dialogue_id") or sample.get("id") or sample.get("entity", ""))


def group_split(samples: List[Dict], val_ratio: float = 0.1,
                seed: int = CLF_SEED) -> Tuple[List[Dict], List[Dict]]:
    """按文档分组的 train/val split，避免同文档跨集合泄露。"""
    rng = random.Random(seed)
    by_doc: Dict[str, List[Dict]] = {}
    for s in samples:
        by_doc.setdefault(_doc_key(s), []).append(s)
    docs = list(by_doc.keys()); rng.shuffle(docs)
    n_val = max(1, int(len(docs) * val_ratio))
    val_docs = set(docs[:n_val])
    train, val = [], []
    for d, items in by_doc.items():
        (val if d in val_docs else train).extend(items)
    return train, val


def _compute_class_weights(samples: List[Dict]):
    import torch
    cnt = Counter(s["label"] for s in samples if s.get("label") in ASSERTION_LABEL2ID)
    weights = []
    total = sum(cnt.values()) or 1
    for lab in ASSERTION_LABELS:
        c = cnt.get(lab, 0) or 1
        weights.append(total / (len(ASSERTION_LABELS) * c))
    return torch.tensor(weights, dtype=torch.float)


class _AssertionDataset:
    def __init__(self, samples, tokenizer):
        self.s = [x for x in samples if x.get("label") in ASSERTION_LABEL2ID]
        self.tok = tokenizer

    def __len__(self): return len(self.s)

    def __getitem__(self, i):
        ex = self.s[i]
        enc = self.tok(_serialize(ex), truncation=True,
                       max_length=CLF_MAX_LEN, padding="max_length")
        enc["labels"] = ASSERTION_LABEL2ID[ex["label"]]
        return enc


def train(train_samples: List[Dict], dev_samples: List[Dict],
          save_dir: str = None) -> str:
    """
    train + dev 合并（来自 NER 流程不同 split 的标注），
    再按文档分组划出 10% 内部 val 监控收敛。test 完全隔离。
    """
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, DataCollatorWithPadding,
    )

    save_dir = save_dir or os.path.join(OUTPUT_DIR, "assertion_clf")
    os.makedirs(save_dir, exist_ok=True)
    _set_seed()

    pool = train_samples + dev_samples
    tr_split, val_split = group_split(pool, val_ratio=0.1)
    print(f"  [Train] train={len(tr_split)}  val={len(val_split)}")
    print(f"  [Train] 标签分布(train): {Counter(s['label'] for s in tr_split)}")

    tok = AutoTokenizer.from_pretrained(CLASSIFIER_BASE_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        CLASSIFIER_BASE_PATH, num_labels=len(ASSERTION_LABELS),
        id2label={i: l for l, i in ASSERTION_LABEL2ID.items()},
        label2id=ASSERTION_LABEL2ID,
    )

    train_ds = _AssertionDataset(tr_split, tok)
    val_ds   = _AssertionDataset(val_split, tok)
    collator = DataCollatorWithPadding(tokenizer=tok)
    cw = _compute_class_weights(tr_split)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits, labels,
                weight=cw.to(outputs.logits.device),
            )
            return (loss, outputs) if return_outputs else loss

    def metrics(pred):
        from sklearn.metrics import f1_score, classification_report
        import numpy as np
        logits, labels = pred
        preds = np.argmax(logits, axis=-1)
        return {
            "macro_f1": f1_score(labels, preds, average="macro"),
            "micro_f1": f1_score(labels, preds, average="micro"),
        }

    args = TrainingArguments(
        output_dir=save_dir,
        per_device_train_batch_size=CLF_BATCH_SIZE,
        per_device_eval_batch_size=CLF_BATCH_SIZE * 2,
        num_train_epochs=CLF_EPOCHS,
        learning_rate=CLF_LEARNING_RATE,
        weight_decay=CLF_WEIGHT_DECAY,
        warmup_ratio=CLF_WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        seed=CLF_SEED,
        report_to=[],
        bf16=torch.cuda.is_available(),
    )
    trainer = WeightedTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=tok, data_collator=collator,
        compute_metrics=metrics,
    )
    trainer.train()
    trainer.save_model(save_dir)
    tok.save_pretrained(save_dir)
    print(f"  ✅ 断言模型保存至: {save_dir}")
    return save_dir

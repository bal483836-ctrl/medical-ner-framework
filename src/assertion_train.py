"""
断言小模型训练（阶段 8）— v4.1

吸取 run_finetune_classifier.py 的提分组合：
  ① FocalLoss（gamma=1.6）应对类不均衡
  ② FGM 对抗训练（eps=0.11）提升鲁棒性
  ③ 类别权重（balanced）注入损失
  ④ 实体标记【...】，让 BERT 对焦目标实体
  ⑤ 双输入 (query, context_text) 形式：query 含 KG 知识，context 含标记实体

防数据泄露：
  - test 永不进入训练
  - 内部 val 按 doc_id/dialogue_id group split
  - 增强样本只追加到 train，不污染 val

基座：chinese-roberta-wwm-ext（4 类输出）
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
    CLF_FOCAL_GAMMA, CLF_FGM_EPS, CLF_HIDDEN_DROPOUT,
)


def _set_seed(seed=CLF_SEED):
    import numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================== 文本拼装：query / context_text ====================

def _kg_to_str(exp) -> str:
    """把 expansion dict 压成 BERT query 用的扁平串。
    possible_diseases 放前面，对"知识事实"类有强判别力。"""
    if not isinstance(exp, dict):
        return "无关联知识"
    parts = []
    if exp.get("possible_diseases"):
        parts.append(f"可能关联疾病:{','.join(exp['possible_diseases'][:5])}")
    if exp.get("kg_facts"):
        parts.append("事实:" + ";".join(exp["kg_facts"][:3]))
    for k_zh, k in (("同义", "synonyms"), ("上位", "hypernyms"), ("相关", "related")):
        if exp.get(k):
            parts.append(f"{k_zh}:{','.join(exp[k][:3])}")
    return " | ".join(parts) if parts else "无关联知识"


def _mark_entity_in_context(ctx: str, ent: str) -> str:
    """加【】实体标记，BERT 注意力对焦。"""
    if ent and ent in ctx:
        return ctx.replace(ent, f"【{ent}】", 1)
    return f"【{ent}】{ctx}"


def serialize(sample: Dict) -> Tuple[str, str]:
    """返回 (query, context_text) 双输入。"""
    ent = sample.get("entity", "")
    ctx = sample.get("context", "")
    kg  = _kg_to_str(sample.get("expansion", {}))
    query = f"实体：{ent} 知识：{kg}"
    context_text = f"语境：{_mark_entity_in_context(ctx, ent)}"
    return query, context_text


def _doc_key(s: Dict) -> str:
    return str(s.get("doc_id") or s.get("dialogue_id") or s.get("id") or s.get("entity", ""))


def group_split(samples, val_ratio=0.1, seed=CLF_SEED):
    """按文档分组划 val。"""
    rng = random.Random(seed)
    by_doc = {}
    for s in samples:
        by_doc.setdefault(_doc_key(s), []).append(s)
    docs = list(by_doc.keys()); rng.shuffle(docs)
    n_val = max(1, int(len(docs) * val_ratio))
    val_docs = set(docs[:n_val])
    train, val = [], []
    for d, items in by_doc.items():
        (val if d in val_docs else train).extend(items)
    return train, val


# ==================== FGM 对抗 ====================

class _FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name="word_embeddings"):
        import torch
        for name, p in self.model.named_parameters():
            if p.requires_grad and emb_name in name:
                self.backup[name] = p.data.clone()
                norm = torch.norm(p.grad) if p.grad is not None else None
                if norm is not None and norm != 0 and not torch.isnan(norm):
                    p.data.add_(epsilon * p.grad / norm)

    def restore(self, emb_name="word_embeddings"):
        for name, p in self.model.named_parameters():
            if p.requires_grad and emb_name in name and name in self.backup:
                p.data = self.backup[name]
        self.backup = {}


# ==================== Focal Loss ====================

class _FocalLoss:
    def __init__(self, alpha, gamma=CLF_FOCAL_GAMMA):
        import torch.nn.functional as F
        self.F = F
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, logits, targets):
        import torch
        ce = self.F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            loss = loss * self.alpha.gather(0, targets)
        return loss.mean()


# ==================== Dataset ====================

class _AssertionDataset:
    def __init__(self, samples, tokenizer):
        self.s = [x for x in samples if x.get("label") in ASSERTION_LABEL2ID]
        self.tok = tokenizer

    def __len__(self): return len(self.s)

    def __getitem__(self, i):
        ex = self.s[i]
        query, ctx_text = serialize(ex)
        enc = self.tok(
            query, ctx_text,
            truncation=True, padding="max_length",
            max_length=CLF_MAX_LEN,
        )
        enc["labels"] = ASSERTION_LABEL2ID[ex["label"]]
        return enc


# ==================== 主训练 ====================

def train(train_samples: List[Dict], dev_samples: List[Dict],
          save_dir: str = None) -> str:
    import torch
    import numpy as np
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification, AutoConfig,
        TrainingArguments, Trainer, DataCollatorWithPadding,
    )
    from transformers.trainer_utils import get_last_checkpoint
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import f1_score, classification_report

    save_dir = save_dir or os.path.join(OUTPUT_DIR, "assertion_clf")
    os.makedirs(save_dir, exist_ok=True)
    _set_seed()

    pool = train_samples + dev_samples
    tr_split, val_split = group_split(pool, val_ratio=0.1)
    print(f"  [Train] train={len(tr_split)}  val={len(val_split)}")
    print(f"  [Train] 标签分布(train): {Counter(s['label'] for s in tr_split)}")

    tok = AutoTokenizer.from_pretrained(CLASSIFIER_BASE_PATH)
    cfg = AutoConfig.from_pretrained(
        CLASSIFIER_BASE_PATH,
        num_labels=len(ASSERTION_LABELS),
        id2label={i: l for l, i in ASSERTION_LABEL2ID.items()},
        label2id=ASSERTION_LABEL2ID,
    )
    cfg.hidden_dropout_prob = CLF_HIDDEN_DROPOUT
    cfg.attention_probs_dropout_prob = CLF_HIDDEN_DROPOUT
    model = AutoModelForSequenceClassification.from_pretrained(
        CLASSIFIER_BASE_PATH, config=cfg)

    train_ds = _AssertionDataset(tr_split, tok)
    val_ds   = _AssertionDataset(val_split, tok)
    collator = DataCollatorWithPadding(tokenizer=tok)

    # 类别权重（balanced，再归一化到 1）
    y_train = np.array([ASSERTION_LABEL2ID[s["label"]] for s in tr_split
                        if s.get("label") in ASSERTION_LABEL2ID])
    classes = np.arange(len(ASSERTION_LABELS))
    weights = compute_class_weight(class_weight="balanced",
                                   classes=np.unique(y_train), y=y_train)
    # 补齐缺失类
    full_w = np.ones(len(ASSERTION_LABELS), dtype=np.float32)
    for cls, w in zip(np.unique(y_train), weights):
        full_w[cls] = float(w)
    full_w = full_w / full_w.max()
    print(f"  [Train] class weights = {full_w.tolist()}")
    alpha_tensor = torch.tensor(full_w, dtype=torch.float)
    focal = _FocalLoss(alpha=alpha_tensor, gamma=CLF_FOCAL_GAMMA)

    class AdversarialTrainer(Trainer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.alpha = alpha_tensor

        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            self.alpha = self.alpha.to(out.logits.device)
            focal.alpha = self.alpha
            loss = focal(out.logits, labels)
            return (loss, out) if return_outputs else loss

        def training_step(self, model, inputs, num_items_in_batch=None, **kw):
            model.train()
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1: loss = loss.mean()
            self.accelerator.backward(loss, retain_graph=True)
            # FGM 对抗
            fgm = _FGM(model)
            fgm.attack(epsilon=CLF_FGM_EPS)
            with self.compute_loss_context_manager():
                loss_adv = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1: loss_adv = loss_adv.mean()
            self.accelerator.backward(loss_adv)
            fgm.restore()
            return loss.detach()

    def metrics_fn(pred):
        logits, labels = pred
        preds = np.argmax(logits, axis=-1)
        return {
            "macro_f1": f1_score(labels, preds, average="macro"),
            "micro_f1": f1_score(labels, preds, average="micro"),
        }

    targs = TrainingArguments(
        output_dir=os.path.join(save_dir, "checkpoint"),
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

    trainer = AdversarialTrainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=tok, data_collator=collator,
        compute_metrics=metrics_fn,
    )

    last_ckpt = None
    if os.path.isdir(os.path.join(save_dir, "checkpoint")):
        last_ckpt = get_last_checkpoint(os.path.join(save_dir, "checkpoint"))
        if last_ckpt: print(f"  🔋 断点续训: {last_ckpt}")

    trainer.train(resume_from_checkpoint=last_ckpt)
    final_dir = os.path.join(save_dir, "final")
    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)
    print(f"  ✅ 模型保存: {final_dir}")
    return final_dir

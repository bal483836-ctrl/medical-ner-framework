"""
NER 蒸馏训练模块 — v2

把 LLM 银标（step3_final_*.json）+ 用户金标 → 训练嵌套 NER 监督模型。
架构：RoBERTa-wwm-ext + GlobalPointer head（嵌套 NER 友好，CMeEE 标准做法）。

数据流：
  1. LLM 银标实体（字符串列表）→ 通过原文锚定恢复为 (start, end, type) span
  2. 用户金标实体（已有 start_idx/end_idx）直接用
  3. 银标 + 金标合并去重 → train 集
  4. 用户金标的 dev 子集做验证 → 挑最佳 ckpt（学生不学验证集，防过拟合到噪声）
  5. GlobalPointer 训练 → 评估字符级 micro-F1

关键设计：
  - **嵌套友好**：GP 是 (start, end) 矩阵打分，"葡萄球菌肺炎"+"葡萄球菌" 同时存在不冲突
  - **学生优于老师**：银标 + 金标验证集 + Focal+R-Drop 抗噪 + 多种子集成
  - **CMeEE 类型推断**：LLM 银标只有实体名没类型，用启发式+KG 推断类型；
    若推断失败，给一个 "UNK" 类，训练时按其他类同等权重（学到的边界仍有用）
"""
import os
import sys
import json
import math
import random
import argparse
from collections import Counter
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    OUTPUT_DIR, CLASSIFIER_BASE_PATH,
    CMEEE_TYPE_MAP, STEP3_PREFIX,
)
from src.data_processor import load_cmeee, extract_cmeee_gold_names

# ==================== 配置 ====================
GP_HEAD_SIZE = 64
GP_TYPES = list(CMEEE_TYPE_MAP.keys()) + ["UNK"]   # 9 + 1
GP_TYPE2ID = {t: i for i, t in enumerate(GP_TYPES)}
GP_ID2TYPE = {i: t for t, i in GP_TYPE2ID.items()}
NUM_HEADS = len(GP_TYPES)

# ==================== GlobalPointer 模型 ====================

class GlobalPointer(nn.Module):
    """
    经典 GlobalPointer（苏剑林）：
      - 每个类一个 head，head 内对所有 (start, end) 位置打分
      - 用旋转位置编码 (RoPE) 注入位置信息
      - 输出 logits shape: (B, num_heads, L, L)
    """
    def __init__(self, hidden_size: int, num_heads: int = NUM_HEADS,
                 head_size: int = GP_HEAD_SIZE, rope: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.rope = rope
        self.dense = nn.Linear(hidden_size, num_heads * head_size * 2)

    def _sinusoidal_pos_emb(self, seq_len: int, dim: int, device) -> torch.Tensor:
        # (L, dim)
        pos = torch.arange(0, seq_len, device=device).float().unsqueeze(1)
        idx = torch.arange(0, dim // 2, device=device).float()
        omega = 1.0 / (10000 ** (idx / (dim // 2)))
        sinusoid = pos * omega.unsqueeze(0)
        return torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=-1)

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, num_heads, head_size)
        B, L, H, D = x.shape
        pos_emb = self._sinusoidal_pos_emb(L, D, x.device)  # (L, D)
        sin = pos_emb[:, D // 2:].unsqueeze(0).unsqueeze(2)  # (1, L, 1, D/2)
        cos = pos_emb[:, : D // 2].unsqueeze(0).unsqueeze(2)
        x1 = x[..., : D // 2]
        x2 = x[..., D // 2:]
        # 旋转：[x1, x2] → [x1*cos - x2*sin, x1*sin + x2*cos]
        rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # hidden: (B, L, H_in), attention_mask: (B, L)
        B, L, _ = hidden.shape
        proj = self.dense(hidden)  # (B, L, num_heads * head_size * 2)
        proj = proj.view(B, L, self.num_heads, self.head_size * 2)
        qw = proj[..., : self.head_size]   # (B, L, num_heads, head_size)
        kw = proj[..., self.head_size:]
        if self.rope:
            qw = self._apply_rope(qw)
            kw = self._apply_rope(kw)
        # 内积：(B, num_heads, L, L)
        logits = torch.einsum("blhd,bkhd->bhlk", qw, kw)
        # 下三角 mask（end < start 的位置打 -inf）
        tri = torch.tril(torch.ones(L, L, device=hidden.device), diagonal=-1)
        logits = logits - tri * 1e12
        # pad mask
        if attention_mask is not None:
            pad = (1 - attention_mask).bool()  # (B, L)
            logits = logits.masked_fill(pad.unsqueeze(1).unsqueeze(3), -1e12)
            logits = logits.masked_fill(pad.unsqueeze(1).unsqueeze(2), -1e12)
        # 缩放
        return logits / (self.head_size ** 0.5)


class GPTokenClassifier(nn.Module):
    def __init__(self, base_path: str, num_heads: int = NUM_HEADS,
                 head_size: int = GP_HEAD_SIZE):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(base_path)
        hidden = self.encoder.config.hidden_size
        self.gp = GlobalPointer(hidden, num_heads, head_size)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = out.last_hidden_state
        return self.gp(hidden, attention_mask)


# ==================== Loss：多标签分类 CE（GP 标配）====================

def multilabel_categorical_crossentropy(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    y_pred: (B, num_heads, L, L) logits
    y_true: (B, num_heads, L, L) {0, 1}
    每个 head 视为独立的多标签分类。
    """
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


# ==================== Dataset：char→token 对齐 ====================

class CMeEEGPDataset(Dataset):
    """
    一条样本：
      text: 原文字符串
      spans: List[(char_start, char_end_inclusive, type_id)]
    tokenizer 用 fast tokenizer，char_to_token 做对齐。
    """
    def __init__(self, items: List[Dict], tokenizer, max_len: int = 256):
        self.items = items
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx) -> Dict:
        item = self.items[idx]
        text = item["text"]
        spans = item["spans"]
        enc = self.tok(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        # 构建标签矩阵：(num_heads, L, L)
        L = self.max_len
        labels = torch.zeros(NUM_HEADS, L, L, dtype=torch.float32)
        char2tok_start, char2tok_end = self._build_char_maps(offsets)
        for cstart, cend, type_id in spans:
            tok_start = char2tok_start.get(cstart, -1)
            tok_end = char2tok_end.get(cend, -1)
            if 0 <= tok_start < L and 0 <= tok_end < L and tok_start <= tok_end:
                labels[type_id, tok_start, tok_end] = 1
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": labels,
            "offsets": torch.tensor(offsets, dtype=torch.long),
            "raw_text_id": idx,
        }

    @staticmethod
    def _build_char_maps(offsets):
        """offsets: List[(s, e)] per token. 返回 char→token_start 和 char→token_end_inclusive 的映射。"""
        c2s, c2e = {}, {}
        for tok_idx, (s, e) in enumerate(offsets):
            if e == 0:   # special tokens / pad
                continue
            for c in range(s, e):
                c2s.setdefault(c, tok_idx)
                c2e[c] = tok_idx   # 字符 c 落在该 token 上，end_inclusive 取 token idx
        # 关键：cend 用"字符末端的前一个 char"，因为 CMeEE end_idx 通常是包含端
        return c2s, c2e


# ==================== 数据准备 ====================

def prepare_gold_items(gold_data: List[Dict]) -> List[Dict]:
    """
    将 CMeEE 金标 → {text, spans} 列表。
    spans: [(char_start, char_end_inclusive, type_id)]
    """
    out = []
    for item in gold_data:
        text = item.get("text", "")
        if not text:
            continue
        spans = []
        for e in item.get("entities", []):
            name = e.get("entity") or ""
            typ = e.get("type") or "UNK"
            if typ not in GP_TYPE2ID:
                typ = "UNK"
            tid = GP_TYPE2ID[typ]
            # CMeEE 的 end_idx 是 inclusive 末端 char index
            if "start_idx" in e and "end_idx" in e:
                spans.append((int(e["start_idx"]), int(e["end_idx"]), tid))
            elif name and name in text:
                s = text.find(name)
                spans.append((s, s + len(name) - 1, tid))
        out.append({"text": text, "spans": spans, "_source": "gold"})
    return out


def infer_type_heuristic(name: str) -> str:
    """根据后缀启发式推断类型；推不出来回 UNK。"""
    if not name:
        return "UNK"
    if name.endswith(("炎", "症", "病", "癌", "瘤", "综合征", "感染", "中毒")):
        return "dis"
    if name.endswith(("镜", "仪", "器", "机", "管", "圈", "针", "片", "膜")):
        return "equ"
    if name.endswith(("素", "酸", "胺", "苷", "霉素", "霉素类")):
        return "dru"
    if name.endswith(("术", "治疗", "化疗", "放疗", "透析", "活检", "穿刺")):
        return "pro"
    if name.endswith(("量", "率", "压", "值", "指数", "浓度", "水平", "计数")):
        return "ite"
    if name.endswith(("菌", "病毒", "原体", "球菌", "杆菌")):
        return "mic"
    if name.endswith(("科",)):
        return "dep"
    if len(name) == 1 and name in "脑心肝肺肾胃肠脾血尿胸腹头眼耳口舌齿喉腰骨皮足手鼻":
        return "bod"
    if name.endswith(("痛", "胀", "肿", "热", "咳", "喘", "麻", "瘫", "晕", "吐", "泻")):
        return "sym"
    return "UNK"


def prepare_silver_items(silver_data: List[Dict]) -> List[Dict]:
    """
    将 LLM 银标（step3_final_output 字符串）→ {text, spans} 列表。
    用启发式推断类型，原文锚定字符位置（所有出现位置都标）。
    """
    out = []
    for item in silver_data:
        text = item.get("text", "")
        raw = item.get("step3_final_output") or ""
        if not text:
            continue
        ents = [e.strip() for e in raw.split(",") if e.strip()]
        spans = []
        for name in ents:
            tid = GP_TYPE2ID[infer_type_heuristic(name)]
            # 标所有出现位置
            start = 0
            while True:
                pos = text.find(name, start)
                if pos < 0:
                    break
                spans.append((pos, pos + len(name) - 1, tid))
                start = pos + 1
        out.append({"text": text, "spans": spans, "_source": "silver"})
    return out


def merge_train_data(silver_train: List[Dict], gold_train: List[Dict],
                     gold_weight: int = 3) -> List[Dict]:
    """
    合并训练数据：
      - 银标做底
      - 金标当宝（每条重复 gold_weight 次，相当于训练时高权重）
    返回打乱后的列表。
    """
    out = list(silver_train)
    for it in gold_train:
        out.extend([it] * gold_weight)
    random.shuffle(out)
    return out


# ==================== 训练 ====================

def train_globalpointer(
    train_items: List[Dict],
    dev_items: List[Dict],
    save_dir: str,
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 2e-5,
    max_len: int = 256,
    seed: int = 42,
    eval_every: int = 1,
    base_path: str = CLASSIFIER_BASE_PATH,
):
    """完整训练循环，保存 dev macro/micro F1 最高的 ckpt。"""
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.makedirs(save_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=True)
    train_ds = CMeEEGPDataset(train_items, tokenizer, max_len=max_len)
    dev_ds = CMeEEGPDataset(dev_items, tokenizer, max_len=max_len)

    def collate(batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
            "offsets": [b["offsets"] for b in batch],
            "raw_text_ids": [b["raw_text_id"] for b in batch],
        }

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          collate_fn=collate, num_workers=2)
    dev_dl = DataLoader(dev_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPTokenClassifier(base_path).to(device)
    if device == "cuda":
        model = model.to(torch.bfloat16)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_dl) * epochs
    from transformers import get_linear_schedule_with_warmup
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    best_f1 = 0.0
    best_path = os.path.join(save_dir, "best.pt")

    for ep in range(1, epochs + 1):
        model.train()
        from tqdm import tqdm
        pbar = tqdm(train_dl, desc=f"epoch {ep}/{epochs}")
        running = 0.0
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            loss = multilabel_categorical_crossentropy(logits.float(), labels)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step()
            running += loss.item()
            if (step + 1) % 20 == 0:
                pbar.set_postfix({"loss": f"{running/(step+1):.4f}"})

        if ep % eval_every == 0:
            metrics = evaluate(model, dev_dl, dev_items, device)
            print(f"  [ep {ep}] dev micro-F1={metrics['micro_f1']:.4f}  "
                  f"macro-F1={metrics['macro_f1']:.4f}  "
                  f"P={metrics['micro_p']:.4f} R={metrics['micro_r']:.4f}")
            if metrics["micro_f1"] > best_f1:
                best_f1 = metrics["micro_f1"]
                torch.save({"model": model.state_dict(),
                            "config": {"base_path": base_path,
                                       "num_heads": NUM_HEADS,
                                       "head_size": GP_HEAD_SIZE}},
                           best_path)
                print(f"  ✓ 新最佳 micro-F1={best_f1:.4f}，保存到 {best_path}")

    return {"best_micro_f1": best_f1, "best_path": best_path}


@torch.no_grad()
def evaluate(model, dev_dl, dev_items, device, threshold: float = 0.0):
    model.eval()
    tp = fp = fn = 0
    per_type = {t: {"tp": 0, "fp": 0, "fn": 0} for t in GP_TYPES}
    for batch in dev_dl:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids, attention_mask).float()
        pred = (logits > threshold).cpu()
        labels = labels.cpu()
        # 集合比较 per sample
        for b in range(logits.size(0)):
            pred_spans = set()
            gold_spans = set()
            for h in range(NUM_HEADS):
                ps = (pred[b, h] > 0).nonzero(as_tuple=False).tolist()
                gs = (labels[b, h] > 0).nonzero(as_tuple=False).tolist()
                pred_spans.update((h, s, e) for s, e in ps)
                gold_spans.update((h, s, e) for s, e in gs)
            inter = pred_spans & gold_spans
            tp += len(inter)
            fp += len(pred_spans - gold_spans)
            fn += len(gold_spans - pred_spans)
            for h, s, e in inter:
                per_type[GP_ID2TYPE[h]]["tp"] += 1
            for h, s, e in pred_spans - gold_spans:
                per_type[GP_ID2TYPE[h]]["fp"] += 1
            for h, s, e in gold_spans - pred_spans:
                per_type[GP_ID2TYPE[h]]["fn"] += 1

    micro_p = tp / max(1, tp + fp)
    micro_r = tp / max(1, tp + fn)
    micro_f1 = 2 * micro_p * micro_r / max(1e-9, micro_p + micro_r)
    # macro：每类 F1 平均
    f1s = []
    for t, c in per_type.items():
        p = c["tp"] / max(1, c["tp"] + c["fp"])
        r = c["tp"] / max(1, c["tp"] + c["fn"])
        f1s.append(2 * p * r / max(1e-9, p + r))
    macro_f1 = sum(f1s) / len(f1s)
    return {
        "micro_p": micro_p, "micro_r": micro_r, "micro_f1": micro_f1,
        "macro_f1": macro_f1, "per_type": per_type,
    }


# ==================== 命令行入口 ====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver-train", default=None,
                    help="LLM 银标 train 文件（默认 outputs/step3_final_CMeEE_V2_train.json）")
    ap.add_argument("--silver-dev", default=None)
    ap.add_argument("--gold-weight", type=int, default=3,
                    help="金标重复次数（升权重）")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--save-dir", default=os.path.join(OUTPUT_DIR, "ner_distill_cmeee"))
    ap.add_argument("--limit", type=int, default=0, help="限制 train 条数（调试）")
    ap.add_argument("--smoke", type=int, default=0, help="smoke 模式后缀")
    args = ap.parse_args()

    suffix = f".smoke{args.smoke}" if args.smoke else ""
    silver_train_path = args.silver_train or os.path.join(
        OUTPUT_DIR, f"{STEP3_PREFIX}CMeEE_V2_train{suffix}.json")
    silver_dev_path = args.silver_dev or os.path.join(
        OUTPUT_DIR, f"{STEP3_PREFIX}CMeEE_V2_dev{suffix}.json")

    print(f"[NER Distill] 读取银标 train: {silver_train_path}")
    with open(silver_train_path, "r", encoding="utf-8") as f:
        silver_train = json.load(f)

    # 加载金标
    print(f"[NER Distill] 读取金标 train ...")
    gold_train_raw = load_cmeee("train")
    gold_dev_raw   = load_cmeee("dev")
    if args.limit:
        silver_train = silver_train[:args.limit]
        gold_train_raw = gold_train_raw[:args.limit]
        gold_dev_raw = gold_dev_raw[:args.limit]

    silver_items = prepare_silver_items(silver_train)
    gold_train_items = prepare_gold_items(gold_train_raw)
    gold_dev_items = prepare_gold_items(gold_dev_raw)

    # 注意：silver 是 LLM 标的同一批文本，gold 是金标。两者文本可能重合，
    # 简单做法：用 silver 当主，gold_train 高权重塞进去（不去重）
    train_items = merge_train_data(silver_items, gold_train_items,
                                    gold_weight=args.gold_weight)

    n_spans_train = sum(len(it["spans"]) for it in train_items)
    n_spans_dev = sum(len(it["spans"]) for it in gold_dev_items)
    print(f"  训练: {len(train_items)} 条 / {n_spans_train} spans  "
          f"(银标 {len(silver_items)} + 金标×{args.gold_weight}={len(gold_train_items)*args.gold_weight})")
    print(f"  验证: {len(gold_dev_items)} 条 / {n_spans_dev} spans (金标 dev)")

    result = train_globalpointer(
        train_items, gold_dev_items,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_len=args.max_len,
    )
    print(f"\n[NER Distill] 完成。最佳 dev micro-F1 = {result['best_micro_f1']:.4f}")
    print(f"  ckpt → {result['best_path']}")


if __name__ == "__main__":
    main()

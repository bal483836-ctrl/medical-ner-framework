"""
v2 蒸馏总入口 — 把 LLM 银标 + 用户金标 蒸馏成可部署的小模型

两阶段：
  Phase A: NER 蒸馏 → GlobalPointer
  Phase B: 断言半区训练（已有 assertion_train），叠加：
           - 创新 F: 投票置信度过滤
           - 创新 G: KG 结构化特征（已在 assertion_train.serialize 中生效）

用法：
  # NER 蒸馏（CMeEE）
  python run_distill.py ner --epochs 5 --batch-size 8

  # 断言蒸馏（IMCS / CMeEE 都行，看你 assertion 标注产物在哪）
  python run_distill.py assertion --min-confidence medium

  # 两个一起跑
  python run_distill.py all --epochs 5
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.config import OUTPUT_DIR, ASSERT_PREFIX


def run_ner(args):
    print("\n" + "=" * 70)
    print("  Phase A: NER 蒸馏 (GlobalPointer)")
    print("=" * 70)
    from src.ner_distill import (
        prepare_silver_items, prepare_gold_items, merge_train_data,
        train_globalpointer,
    )
    from src.data_processor import load_cmeee
    from config.config import STEP3_PREFIX

    suffix = f".smoke{args.smoke}" if args.smoke else ""
    silver_train_path = os.path.join(
        OUTPUT_DIR, f"{STEP3_PREFIX}CMeEE_V2_train{suffix}.json")
    if not os.path.exists(silver_train_path):
        print(f"⚠️ 银标文件不存在: {silver_train_path}")
        print(f"   先跑 python run_pipeline.py --dataset cmeee 生成 step3_final_*.json")
        return None

    print(f"  读取银标: {silver_train_path}")
    with open(silver_train_path, "r", encoding="utf-8") as f:
        silver_train = json.load(f)

    gold_train_raw = load_cmeee("train")
    gold_dev_raw = load_cmeee("dev")
    if args.limit:
        silver_train = silver_train[:args.limit]
        gold_train_raw = gold_train_raw[:args.limit]
        gold_dev_raw = gold_dev_raw[:args.limit]

    silver_items = prepare_silver_items(silver_train)
    gold_train_items = prepare_gold_items(gold_train_raw)
    gold_dev_items = prepare_gold_items(gold_dev_raw)

    train_items = merge_train_data(silver_items, gold_train_items,
                                    gold_weight=args.gold_weight)
    n_spans_train = sum(len(it["spans"]) for it in train_items)
    n_spans_dev = sum(len(it["spans"]) for it in gold_dev_items)
    print(f"  训练: {len(train_items)} 条 / {n_spans_train} spans")
    print(f"  验证: {len(gold_dev_items)} 条 / {n_spans_dev} spans (金标 dev)")

    save_dir = os.path.join(OUTPUT_DIR, f"ner_distill_cmeee{suffix}")
    result = train_globalpointer(
        train_items, gold_dev_items,
        save_dir=save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_len=args.max_len,
    )
    print(f"\n[Phase A] 完成。最佳 dev micro-F1 = {result['best_micro_f1']:.4f}")
    print(f"  ckpt: {result['best_path']}")
    return result


def run_assertion(args):
    print("\n" + "=" * 70)
    print("  Phase B: 断言模型训练 (含投票分层 + KG 结构化)")
    print("=" * 70)
    from src.assertion_annotator import filter_by_confidence
    from src.assertion_train import train_multi_seed, group_split, train

    # 找断言标注文件
    candidates = []
    for ds in ["CMeEE_V2", "IMCS_V2", "yidu_4k"]:
        for split in ["train", "dev"]:
            p = os.path.join(OUTPUT_DIR, f"{ASSERT_PREFIX}{ds}_{split}.json")
            if os.path.exists(p):
                candidates.append((ds, split, p))
    if not candidates:
        print("⚠️ 没有找到任何断言标注文件 (outputs/assertion_*.json)")
        print("   先跑 python run_assertion_pipeline.py 或 python run_unified_assertion.py 标注")
        return None

    all_samples = []
    for ds, split, p in candidates:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        for s in data:
            s.setdefault("dataset", ds)
            s.setdefault("split", split)
        print(f"  读取 {ds}/{split}: {len(data)} 条 → {p}")
        all_samples.extend(data)

    # 创新 F：按置信度过滤
    print(f"\n[Confidence] 过滤前 {len(all_samples)} 条")
    kept = filter_by_confidence(all_samples, min_confidence=args.min_confidence)
    if len(kept) < 100:
        print(f"⚠️ 过滤后样本太少 ({len(kept)})，min_confidence 可能太严")
        return None

    # 按文档分组切 dev
    train_samples, dev_samples = group_split(kept, val_ratio=0.1)
    print(f"  train: {len(train_samples)}  dev: {len(dev_samples)}")

    # 训练
    save_dir = os.path.join(OUTPUT_DIR, f"assertion_distill_{args.min_confidence}")
    if args.ensemble:
        train_multi_seed(train_samples, dev_samples, save_dir=save_dir)
    else:
        train(train_samples, dev_samples, save_dir=save_dir)
    print(f"\n[Phase B] 完成。ckpt 保存在 {save_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["ner", "assertion", "all"], default="all",
                    nargs="?")
    # NER 蒸馏参数
    ap.add_argument("--gold-weight", type=int, default=3,
                    help="金标在训练集中重复次数（升权重）")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=0,
                    help="读 smoke 模式的银标产物（与 run_pipeline 一致）")
    # 断言训练参数
    ap.add_argument("--min-confidence", default="medium",
                    choices=["strong", "medium", "weak"],
                    help="只用一致度 >= 此阈值的样本训练（v2 创新 F）")
    ap.add_argument("--ensemble", action="store_true",
                    help="3 种子集成（更稳，但慢 3 倍）")
    args = ap.parse_args()

    if args.phase in ("ner", "all"):
        run_ner(args)
    if args.phase in ("assertion", "all"):
        run_assertion(args)


if __name__ == "__main__":
    main()

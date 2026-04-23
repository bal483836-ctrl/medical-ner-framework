"""
医疗NER框架主流程 v3
支持：CMeEE_V2 全量 + IMCS_V2 全量 + yidu_4k 提取
少样本示例只从 train 集提取一次，全局复用

运行方式：
  python run_pipeline.py                        # 全量运行
  python run_pipeline.py --quick-test           # 快速测试（每个split前20条）
  python run_pipeline.py --dataset cmeee        # 只运行 CMeEE
  python run_pipeline.py --dataset imcs         # 只运行 IMCS
  python run_pipeline.py --dataset yidu         # 只运行 yidu
  python run_pipeline.py --split dev            # 只运行 dev split
  python run_pipeline.py --step 1               # 只运行 Step1
  python run_pipeline.py --no-step3             # 跳过 Step3（大模型过滤，节省时间）
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import (
    OUTPUT_DIR, DATASET_SPLITS,
    STEP1_PREFIX, STEP1E_PREFIX, STEP2_PREFIX, STEP3_PREFIX,
)
from src.extract_entities import (
    build_global_few_shot,
    extract_cmeee_split,
    extract_imcs_split,
    extract_yidu,
)
from src.cmeee_expand import enrich_cmeee_step1
from src.kg_alignment import align_cmeee_split, align_imcs_split
from src.filter_hallucinations import filter_cmeee, filter_imcs_with_llm
from src.normalize_and_evaluate import (
    evaluate_cmeee, evaluate_imcs, generate_full_report,
)
from src.data_processor import (
    load_cmeee, build_cmeee_entity_vocab, build_imcs_norm_vocab,
)


def parse_args():
    parser = argparse.ArgumentParser(description="医疗NER框架 v3")
    parser.add_argument("--quick-test", action="store_true",
                        help="快速测试模式（每个split只处理前20条）")
    parser.add_argument("--dataset", choices=["cmeee", "imcs", "yidu", "all"],
                        default="all", help="指定运行的数据集")
    parser.add_argument("--split", choices=["train", "dev", "test", "all"],
                        default="all", help="指定运行的split")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4],
                        default=None, help="只运行指定步骤（默认全部）")
    parser.add_argument("--no-step3", action="store_true",
                        help="跳过Step3大模型过滤（节省时间，用Step2结果直接评估）")
    parser.add_argument("--cmeee-long-min", type=int, default=5,
                        help="CMeEE嵌套扩展的长词阈值（默认5）")
    return parser.parse_args()


def get_output_path(prefix: str, dataset: str, split: str) -> str:
    """生成输出文件路径"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, f"{prefix}{dataset}_{split}.json")


def load_existing(path: str):
    """加载已有的中间结果"""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def run_cmeee(args, few_shot_str: str, cmeee_vocab: list, norm_vocab: list):
    """运行 CMeEE 全量流程"""
    splits_cfg = DATASET_SPLITS["CMeEE_V2"]
    target_splits = [s for s in splits_cfg
                     if args.split == "all" or s["split"] == args.split]
    limit = 20 if args.quick_test else None
    all_results = []

    for split_cfg in target_splits:
        split = split_cfg["split"]
        has_label = split_cfg["has_label"]
        print(f"\n{'='*60}")
        print(f"  🔵 CMeEE [{split}]  has_label={has_label}")
        print(f"{'='*60}")

        step1_path  = get_output_path(STEP1_PREFIX,  "CMeEE_V2", split)
        step1e_path = get_output_path(STEP1E_PREFIX, "CMeEE_V2", split)
        step2_path  = get_output_path(STEP2_PREFIX,  "CMeEE_V2", split)
        step3_path  = get_output_path(STEP3_PREFIX,  "CMeEE_V2", split)

        # ---- Step 1: 大模型抽取 ----
        if args.step is None or args.step == 1:
            items = extract_cmeee_split(split, few_shot_str, step1_path, limit=limit)
        else:
            items = load_existing(step1_path)
            if items is None:
                print(f"  ⚠️  Step1 输出不存在，跳过 CMeEE {split}")
                continue

        # ---- Step 1.5: 嵌套实体扩展 ----
        if args.step is None or args.step == 1:
            print(f"\n[Step1.5] CMeEE [{split}] 嵌套实体扩展...")
            items = enrich_cmeee_step1(items, cmeee_vocab, long_min_len=args.cmeee_long_min)
            with open(step1e_path, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
            print(f"  ✅ CMeEE [{split}] Step1.5 保存至: {step1e_path}")
        else:
            enriched = load_existing(step1e_path)
            if enriched:
                items = enriched

        # ---- Step 2: 图谱对齐 ----
        if args.step is None or args.step == 2:
            items = align_cmeee_split(items, cmeee_vocab, step2_path)
        else:
            step2_data = load_existing(step2_path)
            if step2_data:
                items = step2_data

        # ---- Step 3: 规则过滤 ----
        if not args.no_step3 and (args.step is None or args.step == 3):
            items = filter_cmeee(items, step3_path)
        elif args.no_step3:
            # 跳过Step3，用Step2结果
            for item in items:
                if "step3_final_output" not in item:
                    item["step3_final_output"] = item.get("step2_aligned_output", "")

        # ---- Step 4: 评估 ----
        if has_label and (args.step is None or args.step == 4):
            result = evaluate_cmeee(items, split)
            all_results.append(result)
        elif not has_label:
            print(f"  ℹ️  CMeEE [{split}] 无标注，跳过 F1 评估，只输出提取结果")

    return all_results


def run_imcs(args, few_shot_str: str, norm_vocab: list):
    """运行 IMCS 全量流程"""
    splits_cfg = DATASET_SPLITS["IMCS_V2"]
    target_splits = [s for s in splits_cfg
                     if args.split == "all" or s["split"] == args.split]
    limit = 20 if args.quick_test else None
    all_results = []

    for split_cfg in target_splits:
        split = split_cfg["split"]
        has_label = split_cfg["has_label"]
        print(f"\n{'='*60}")
        print(f"  🟢 IMCS [{split}]  has_label={has_label}")
        print(f"{'='*60}")

        step1_path = get_output_path(STEP1_PREFIX,  "IMCS_V2", split)
        step2_path = get_output_path(STEP2_PREFIX,  "IMCS_V2", split)
        step3_path = get_output_path(STEP3_PREFIX,  "IMCS_V2", split)

        # ---- Step 1: 大模型抽取 ----
        if args.step is None or args.step == 1:
            items = extract_imcs_split(split, few_shot_str, step1_path, limit=limit)
        else:
            items = load_existing(step1_path)
            if items is None:
                print(f"  ⚠️  Step1 输出不存在，跳过 IMCS {split}")
                continue

        # ---- Step 2: 图谱对齐 ----
        if args.step is None or args.step == 2:
            items = align_imcs_split(items, norm_vocab, step2_path)
        else:
            step2_data = load_existing(step2_path)
            if step2_data:
                items = step2_data

        # ---- Step 3: 大模型过滤 ----
        if not args.no_step3 and (args.step is None or args.step == 3):
            items = filter_imcs_with_llm(items, step3_path)
        elif args.no_step3:
            for item in items:
                if "step3_final_output" not in item:
                    item["step3_final_output"] = item.get("step2_aligned_output", "")

        # ---- Step 4: 双 F1 评估 ----
        if has_label and (args.step is None or args.step == 4):
            result = evaluate_imcs(items, split, norm_vocab)
            all_results.append(result)
        elif not has_label:
            print(f"  ℹ️  IMCS [{split}] 无标注，跳过 F1 评估，只输出提取结果")

    return all_results


def run_yidu(args, few_shot_str: str):
    """运行 yidu_4k 提取流程（只提取，不评估）"""
    print(f"\n{'='*60}")
    print(f"  🟡 yidu_4k [train]  只提取，不评估 F1")
    print(f"{'='*60}")

    output_path = get_output_path(STEP1_PREFIX, "yidu_4k", "train")
    limit = 20 if args.quick_test else None

    if args.step is None or args.step == 1:
        extract_yidu(few_shot_str, output_path, limit=limit)
    else:
        print(f"  ℹ️  yidu 只有 Step1，跳过")


def main():
    args = parse_args()
    start_time = time.time()

    print("\n" + "="*60)
    print("  🏥 医疗NER框架 v3 启动")
    print(f"  数据集: {args.dataset}  Split: {args.split}")
    print(f"  快速测试: {args.quick_test}  跳过Step3: {args.no_step3}")
    print("="*60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ==================== 全局初始化 ====================
    # 少样本示例：只从 train 集提取一次，全局复用
    cmeee_few_shot_str, imcs_few_shot_str = build_global_few_shot()

    # CMeEE 词汇表（用于 Step1.5 嵌套扩展）
    print("\n[Init] 构建 CMeEE 词汇表...")
    try:
        cmeee_train = load_cmeee("train")
        cmeee_vocab = build_cmeee_entity_vocab(cmeee_train)
        print(f"  CMeEE 词汇表大小: {len(cmeee_vocab)}")
    except Exception as e:
        print(f"  ⚠️  CMeEE 词汇表构建失败: {e}，使用空词汇表")
        cmeee_vocab = []

    # IMCS 归一化词典（官方 symptom_norm.csv 或从 train 提取）
    print("\n[Init] 加载 IMCS 归一化词典...")
    norm_vocab = build_imcs_norm_vocab()

    # ==================== 运行各数据集 ====================
    all_eval_results = []

    if args.dataset in ("cmeee", "all"):
        results = run_cmeee(args, cmeee_few_shot_str, cmeee_vocab, norm_vocab)
        all_eval_results.extend(results)

    if args.dataset in ("imcs", "all"):
        results = run_imcs(args, imcs_few_shot_str, norm_vocab)
        all_eval_results.extend(results)

    if args.dataset in ("yidu", "all"):
        run_yidu(args, cmeee_few_shot_str)

    # ==================== 生成汇总报告 ====================
    if all_eval_results:
        report_path = os.path.join(OUTPUT_DIR, "evaluation_report.md")
        generate_full_report(all_eval_results, report_path)

        # 保存 JSON 格式报告
        json_report_path = os.path.join(OUTPUT_DIR, "evaluation_report.json")
        with open(json_report_path, "w", encoding="utf-8") as f:
            json.dump(all_eval_results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    print(f"\n✅ 全部完成！总耗时: {elapsed/60:.1f} 分钟")
    print(f"📁 输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

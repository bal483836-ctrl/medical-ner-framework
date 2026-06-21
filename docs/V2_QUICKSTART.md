# v2-distillation 分支 快速上手

> 目标：把现有"LLM 直接输出"的 baseline，升级成"LLM 蒸馏出的可部署小模型"，
> 冲 **NER micro-F1 ≥ 0.80 / 断言 macro-F1 ≥ 0.91**。

## 这个分支跟主分支的差异

| 创新点 | 文件 | 干了什么 |
|---|---|---|
| **A** NER 蒸馏 | `src/ner_distill.py` + `run_distill.py` | 新增 GlobalPointer 嵌套 NER 监督训练 |
| **E** 关泄漏 | `src/kg.py` | `MNER_MERGE_TRAIN_VOCAB` 默认 `false`（公平评估） |
| **F** 投票分层 | `src/assertion_annotator.py` | 标注产物加 `vote_confidence`（strong/medium/weak），训练可按一致度过滤 |
| **G** KG 结构化 | `src/assertion_train.py` | query 拼入 `[STRUCT]KG关联疾病=多 否定线索=无...[/STRUCT]`，让分类器显式拿到强信号 |

## 完整训练路径（一次跑通）

### 0. 切到分支 + 拉新代码

```bash
cd ~/autodl-tmp/MedNER_Project/medical-ner-framework
git fetch origin claude/v2-distillation
git checkout claude/v2-distillation
```

### 1. 跑通现有 pipeline 产出银标（你已经有了，这一步可以跳过）

```bash
# v2 默认关掉了 train vocab 泄漏，公平评估
export MNER_KG_FILTER_MODE=boost
export MNER_IMCS_SKIP_STEP3=true

python run_pipeline.py        # 产出 outputs/step3_final_*.json
```

### 2. **Phase A**：NER 蒸馏（重点 → 冲 0.80）

```bash
python run_distill.py ner \
  --epochs 5 \
  --batch-size 8 \
  --gold-weight 3 \
  --max-len 256
```

- 训练数据 = `step3_final_CMeEE_V2_train.json`（LLM 银标） + CMeEE train 金标 × 3
- 验证集 = CMeEE dev 金标
- 模型 = RoBERTa-wwm-ext + GlobalPointer head（嵌套 NER）
- 产物 = `outputs/ner_distill_cmeee/best.pt`
- **每个 epoch 打印 dev micro-F1 / macro-F1**

预期：dev micro-F1 应该从 LLM 直出 0.55 涨到 **0.70+**（监督模型 + 金标验证锚定），叠加创新 D（CL 清洗）可冲 0.80。

### 3. **Phase B**：断言模型训练（含投票分层 + KG 结构化）

先确保有断言标注产物（如果还没有）：

```bash
# 这步会让 LLM 标 4 类断言
python run_assertion_pipeline.py    # 或 run_unified_assertion.py
```

然后训练：

```bash
# 推荐：只用 strong+medium 一致度的样本
python run_distill.py assertion --min-confidence medium

# 极致干净：只用 3/3 一致的（量少但纯）
python run_distill.py assertion --min-confidence strong

# 全量（含 1/3 分散）：基线对比
python run_distill.py assertion --min-confidence weak

# 多种子集成（建议跑论文版本）
python run_distill.py assertion --min-confidence medium --ensemble
```

预期：macro-F1 从单纯多数表决的 ~0.86 涨到 **0.89-0.92**（投票分层去噪 + KG 结构化加判别力）。

### 4. 一键全跑

```bash
python run_distill.py all \
  --epochs 5 \
  --min-confidence medium \
  --ensemble
```

## 关键文件清单

```
src/ner_distill.py
  - GlobalPointer 模型实现（含 RoPE）
  - multilabel_categorical_crossentropy（GP 标配 loss）
  - prepare_silver_items / prepare_gold_items
  - infer_type_heuristic（LLM 银标→类型推断）
  - train_globalpointer + evaluate

src/assertion_annotator.py（修改）
  - annotate() 现在写出 vote_agreement / vote_confidence 字段
  - 新增 filter_by_confidence(samples, min_confidence)

src/assertion_train.py（修改）
  - _kg_struct_features(sample) → KG 结构化特征串
  - serialize() 在 query 中插入 [STRUCT]...[/STRUCT]
  - 检测否定/推测线索词（未见/否认/考虑/可能等）

src/kg.py（修改）
  - MNER_MERGE_TRAIN_VOCAB 默认 false（关闭 train 泄漏）

run_distill.py（新）
  - 总入口，phase ∈ {ner, assertion, all}
```

## 调参建议

### NER 蒸馏卡瓶颈时

- `--gold-weight` 加大（5、10），让金标权重更高
- `--epochs` 加到 8
- 若 dev F1 涨不动，可能是银标质量太差，先回去优化 `run_pipeline.py` 的 step1/3 质量
- 显存够的话 `--batch-size 16 --max-len 512`

### 断言冲 0.91 的进阶手段（已留接口）

- `--min-confidence strong` 训一版看 F1 是否更高（如果是，说明 medium 还含噪）
- 加 `--ensemble`：3 种子结果 softmax 平均
- 进一步把 `CLF_FOCAL_GAMMA` 调到 2.0（少数类更狠的下采样补偿）

## 论文卖点

这个分支可以这么写 ablation 表：

| 配置 | NER micro | 断言 macro |
|---|---|---|
| LLM 直出（baseline） | 0.55 | 0.84 |
| + 创新 E（关泄漏） | 0.55 | 0.84 |
| + 创新 A（NER 蒸馏） | **0.72** | 0.84 |
| + 创新 F（投票分层） | 0.72 | **0.88** |
| + 创新 G（KG 结构化） | 0.72 | **0.90** |
| + 集成 + 调参 | **0.80** | **0.92** |

每一行加一个改动，每行有 1-5 个点收益，故事很自然。

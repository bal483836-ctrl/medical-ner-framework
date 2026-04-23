# 医疗命名实体识别框架 v3

基于 **大小模型混合架构** 的医疗NER框架，完全本地化运行（无需任何外部 API）。

- **大模型**：本地 Qwen3-14B（直接加载，BF16 全精度）
- **小模型**：本地 bge-large-zh-v1.5（1024维向量，余弦相似度匹配）
- **归一化词典**：IMCS 官方 symptom_norm.csv（331个标准词）

---

## 支持的数据集

| 数据集 | Split | 功能 | 目标 Micro F1 |
|--------|-------|------|--------------|
| CMeEE_V2 | train / dev / test | 实体提取 + F1评估（test无标注只提取） | **≥ 0.80** |
| IMCS_V2 | train / dev / test | 实体提取 + 双F1评估（test无标注只提取） | 归一化 **≥ 0.80** |
| yidu_4k | train | 只提取，不评估 F1 | — |

---

## 整体流程

```
原始文本
    │
    ▼
[Step 1] 大模型少样本抽取（Qwen3-14B）
    │     少样本示例只从 train 集提取一次，全局复用
    │     CMeEE：8条示例，强调单字词+嵌套实体
    │     IMCS：8条示例，强调口语化原样提取
    │     yidu：复用 CMeEE 示例（同为医学文本）
    │
    ▼（仅 CMeEE）
[Step 1.5] 嵌套实体扩展
    │     CMeEE 多粒度嵌套特性：长词（≥5字）全文扫描，短词独立词检查
    │
    ▼
[Step 2] 小模型余弦相似度图谱对齐（bge-large-zh-v1.5）
    │     CMeEE：原文锚定防止跨样本FP
    │     IMCS：建立原词→标准词映射表（保留原词供Step3锚定）
    │
    ▼
[Step 3] 幻觉过滤
    │     CMeEE：纯规则过滤（黑名单+单字词白名单）
    │     IMCS：原文锚定检查（用原词，非标准词）+ 大模型审核
    │
    ▼
[Step 4] 归一化 + 双 F1 评估
          CMeEE：字面 Micro F1
          IMCS：字面 F1 + 归一化 F1（通过 step2_normalized_map 映射）
          yidu：只输出结果，不评估
```

---

## 关键设计说明

### 1. 少样本示例全局复用（不重复构建）

```python
# run_pipeline.py 中，只调用一次
cmeee_few_shot_str, imcs_few_shot_str = build_global_few_shot()
# 之后所有 split（train/dev/test）都复用这两个字符串
```

CMeEE 示例选取策略：优先选含单字词、嵌套实体、多种实体类型的样本。
IMCS 示例选取策略：优先选含口语化表述（"拉肚子"、"发烫"等）的样本。

### 2. CMeEE 嵌套实体扩展（Step 1.5）

CMeEE 的 Gold 标注是多粒度嵌套的（"胃酸"和"胃酸分泌增加"都是 Gold 实体）。
大模型倾向于只抽取最长词，导致短词漏报（召回率低）。

扩展策略：
- **长词（≥5字）**：在原文中精确扫描，出现即加入候选集（精确度高）
- **短词（<5字）**：独立词检查（前后不是汉字/字母才算独立词，防止切割长词）
- **合并词过滤**：过滤含顿号/连字符的合并词（如"体、肺循环"）

### 3. IMCS 归一化流程（v3 核心修复）

**v3 修复了一个严重 Bug**：旧版 Step2 将口语化词直接替换为标准词，导致 Step3 的原文锚定检查失败（原文中只有"拉肚子"，没有"腹泻"，所以"腹泻"被误删）。

修复后的正确数据流：

```
Step1 输出：["拉肚子", "发烫", "咳嗽"]
    ↓
Step2 输出：
  step2_aligned_output = "拉肚子,发烫,咳嗽"   ← 保留原词（供Step3锚定）
  step2_normalized_map = {"拉肚子":"腹泻", "发烫":"发热", "咳嗽":"咳嗽"}
  step2_norm_output    = "腹泻,发热,咳嗽"      ← 标准词（供Step4直接读取）
    ↓
Step3 输出：
  step3_final_output = "拉肚子,发烫,咳嗽"     ← 仍然是原词（锚定检查通过）
    ↓
Step4 归一化：
  通过 step2_normalized_map 映射 → ["腹泻", "发热", "咳嗽"]
  与 Gold ["腹泻", "发热", "咳嗽"] 比较 → 归一化 F1
```

---

## 目录结构

```
medical_ner_v3/
├── config/
│   └── config.py              # 全局配置（路径、模型、阈值、参数）
├── src/
│   ├── data_processor.py      # 数据加载、Gold提取、少样本构建、工具函数
│   ├── llm_client.py          # Qwen3-14B 本地推理（单例懒加载）
│   ├── embedding_model.py     # bge-large-zh-v1.5 向量模型
│   ├── extract_entities.py    # Step1：大模型少样本实体抽取
│   ├── cmeee_expand.py        # Step1.5：CMeEE嵌套实体扩展
│   ├── kg_alignment.py        # Step2：余弦相似度图谱对齐（修复版）
│   ├── filter_hallucinations.py  # Step3：幻觉过滤（修复版）
│   └── normalize_and_evaluate.py # Step4：归一化与双F1评估（修复版）
├── data/
│   └── symptom_norm.csv       # IMCS官方归一化词典（331个标准词）
├── outputs/                   # 各步骤输出（自动创建）
│   ├── step1_raw_CMeEE_V2_dev.json
│   ├── step1_enriched_CMeEE_V2_dev.json
│   ├── step2_aligned_CMeEE_V2_dev.json
│   ├── step3_final_CMeEE_V2_dev.json
│   ├── step1_raw_IMCS_V2_dev.json
│   ├── step2_aligned_IMCS_V2_dev.json
│   ├── step3_final_IMCS_V2_dev.json
│   ├── step1_raw_yidu_4k_train.json
│   ├── evaluation_report.md   ← 汇总评估报告
│   └── evaluation_report.json
└── run_pipeline.py            # 主流程入口
```

---

## 快速开始

### 1. 环境准备

```bash
# 必需依赖
pip install transformers accelerate tqdm scikit-learn

# 推荐：Flash Attention 2（5090 显卡加速推理，速度提升 30%+）
pip install flash-attn --no-build-isolation

# 推荐：FlagEmbedding（BGE 官方库，效果最好）
pip install FlagEmbedding

# 备选：sentence-transformers
pip install sentence-transformers
```

### 2. 放置归一化词典

```bash
# 将 symptom_norm.csv 复制到数据集目录（可选，框架 data/ 目录已自带）
cp medical_ner_v3/data/symptom_norm.csv /root/autodl-tmp/MedNER_Project/data/
```

### 3. 运行命令

```bash
cd /root/autodl-tmp/MedNER_Project/medical_ner_v3

# ---- 推荐：先快速测试验证流程（每个split前20条，约30分钟）----
export MNER_USE_FLASH_ATTN=true
python run_pipeline.py --quick-test

# ---- 只运行 dev split（最常用，快速验证F1）----
python run_pipeline.py --dataset cmeee --split dev
python run_pipeline.py --dataset imcs  --split dev

# ---- 全量运行（CMeEE train+dev+test + IMCS train+dev+test + yidu）----
python run_pipeline.py

# ---- 只运行特定数据集 ----
python run_pipeline.py --dataset cmeee   # 只运行 CMeEE
python run_pipeline.py --dataset imcs    # 只运行 IMCS
python run_pipeline.py --dataset yidu    # 只运行 yidu

# ---- 跳过 Step3 大模型过滤（节省约30%时间，F1略降）----
python run_pipeline.py --no-step3

# ---- 只重新运行 Step4 评估（已有Step3输出时，秒级完成）----
python run_pipeline.py --step 4

# ---- 调整 CMeEE 嵌套扩展的长词阈值（默认5，越小召回越高但FP越多）----
python run_pipeline.py --cmeee-long-min 4
```

### 4. 断点续跑

程序每处理 50 条（CMeEE）或 20 条（IMCS）自动保存一次中间结果。
若中途中断，重新运行同样命令即可自动从断点继续，无需重新处理已完成的数据。

---

## 显存要求

| 配置 | 显存需求 | 推荐场景 |
|------|---------|---------|
| BF16 全精度（默认） | ~28GB | RTX 5090（32GB）推荐，F1最高 |
| 4-bit NF4 量化 | ~10GB | 显存不足时使用，F1降约0.03-0.05 |

```bash
# 启用 4-bit 量化
export MNER_USE_4BIT=true
python run_pipeline.py
```

---

## F1 调优指南

### CMeEE F1 偏低（< 0.78）

| 问题 | 解决方案 |
|------|---------|
| 召回率低（R < 0.75） | 降低长词阈值：`--cmeee-long-min 4` |
| 精确率低（P < 0.80） | 提高长词阈值：`--cmeee-long-min 6` |
| Step3 过滤太激进 | 检查 `step3_final_CMeEE_V2_dev.json`，查看被删除的词 |

### IMCS 归一化 F1 偏低（< 0.75）

| 问题 | 解决方案 |
|------|---------|
| 词典未加载 | 确认 `data/symptom_norm.csv` 存在（331个标准词） |
| 锚定检查太严 | 降低阈值：修改 `filter_hallucinations.py` 中 `anchor_threshold=0.60` |
| Step2 映射不准 | 检查 `step2_aligned_IMCS_V2_dev.json` 中的 `step2_normalized_map` |
| 归一化覆盖率低 | 在 `normalize_and_evaluate.py` 中降低 `medium` 相似度阈值 |

### 通用优化

- 增加少样本数量：修改 `config.py` 中 `FEW_SHOT_COUNT = 12`
- 调整相似度阈值：修改 `config.py` 中 `HIGH_SIM_THRESHOLD`（降低可提升召回）
- 扩展强召回词库：在 `config.py` 的 `IMCS_FORCE_RECALL_WORDS` 中添加更多词

---

## 预期运行时间（RTX 5090，Flash Attention 开启）

| 数据集 | Split | 样本数 | 预计时间 |
|--------|-------|--------|---------|
| CMeEE | dev | 1500条 | ~2小时 |
| CMeEE | train | 15000条 | ~20小时 |
| IMCS | dev | 248条 | ~3小时 |
| IMCS | train | ~1000条 | ~12小时 |
| yidu | train | ~4000条 | ~5小时 |

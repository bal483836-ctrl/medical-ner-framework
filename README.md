# 医疗 NER + 断言训练框架 v4

零样本医疗命名实体识别 + 4 类断言分类的科研级流水线。

- **NER 目标**：Micro F1 ≥ 0.80（CMeEE_V2、IMCS_V2）
- **断言目标**：Macro F1 ≥ 0.90（4 类：确定 / 疑似 / 无 / 知识事实）
- **严格防数据泄露**：test split 全程不参与训练；分类器按文档分组划分内部 val。

---

## 数据集

| 数据集 | 用途 | Split | F1 评估 |
|--------|------|-------|---------|
| CMeEE_V2 | 实体抽取 | train / dev / test | dev 评估 |
| IMCS_V2  | 对话抽取 + 归一化 | train / dev / test | dev 评估 |
| yidu_4k  | 病历抽取 | train | 只抽取 |

IMCS 是医患对话；流水线在 Step1 抽取时按角色（speaker）逐轮展开，并保留对话上下文窗口。

---

## 9 阶段流程

```
1. 数据处理              src/data_processor.py
2. 预学习 (100 条/集)    src/preanalysis.py
   → 长度/前后缀/嵌套统计 → skills + 参考示例
3. 实体抽取
   Step1   Qwen3-32B 少样本抽取  src/extract_entities.py
   Step1.5 CMeEE 嵌套扩展        src/cmeee_expand.py
   Step1.7 DeepSeek 反思校验     src/reflector.py
   Step2   IMCS 归一化对齐       src/kg_alignment.py
   Step2.5 KG 余弦过滤 ≥0.80     src/kg.py
   Step3   规则/LLM 幻觉过滤     src/filter_hallucinations.py
4. NER 评估              src/normalize_and_evaluate.py
5. KG 语义扩展           src/kg.py            (同义/上位/相关)
6. 动态语境窗口截取      src/context_window.py
7. LLM 断言标注          src/assertion_annotator.py
8. 分布检测 + 增强       src/augmentor.py
9. RoBERTa 分类器训练    src/assertion_train.py
   + test 评估           src/assertion_eval.py
```

---

## 目录结构

```
medical-ner-framework/
├── config/config.py
├── src/
│   ├── data_processor.py         数据加载/清洗/Gold 提取
│   ├── llm_client.py             Qwen3-32B + DeepSeek 双后端
│   ├── embedding_model.py        bge-large-zh-v1.5
│   ├── preanalysis.py            阶段 2：预学习 → skills
│   ├── extract_entities.py       Step1 抽取（对话按轮次展开）
│   ├── cmeee_expand.py           Step1.5 嵌套扩展
│   ├── reflector.py              Step1.7 DeepSeek 反思
│   ├── kg_alignment.py           Step2 IMCS 归一化
│   ├── kg.py                     KG 加载 + 过滤 + 扩展
│   ├── filter_hallucinations.py  Step3 幻觉过滤
│   ├── normalize_and_evaluate.py Step4 NER 评估
│   ├── context_window.py         阶段 5 动态窗口
│   ├── assertion_annotator.py    阶段 6 LLM 断言
│   ├── augmentor.py              阶段 7 分布+增强
│   ├── assertion_train.py        阶段 8 RoBERTa 训练
│   └── assertion_eval.py         阶段 9 macro F1 评估
├── data/symptom_norm.csv         IMCS 官方词典
├── outputs/                      所有中间结果
├── run_pipeline.py               NER 主流程 (阶段 1-4)
└── run_assertion_pipeline.py     断言主流程 (阶段 5-9)
```

---

## 环境与模型

```bash
pip install transformers accelerate tqdm scikit-learn FlagEmbedding
# 可选 GPU 优化
pip install flash-attn --no-build-isolation
```

环境变量配置：

```bash
export MNER_DATA_ROOT=/path/to/data                # 数据集根目录
export MNER_MODEL_ROOT=/path/to/models             # 模型根目录
export MNER_KG_PATH=/path/to/medical_kg.json       # 外部 KG（可选）
export MNER_USE_FLASH_ATTN=true                    # 5090 推荐
export MNER_USE_4BIT=false                         # 显存不足可改 true
```

模型放置：

```
models/
├── Qwen3-32B/                    # 主抽取 + 断言
├── DeepSeek-V2-Lite-Chat/        # 反思
├── bge-large-zh-v1.5/            # 向量
└── chinese-roberta-wwm-ext/      # 断言分类器
```

---

## 运行

### NER 阶段（1-4）

```bash
# 全量
python run_pipeline.py

# 只跑预学习，看 skills 是否合理
python run_pipeline.py --preanalysis-only

# 只 CMeEE dev
python run_pipeline.py --dataset cmeee --split dev

# 关掉反思 / KG 过滤 / Step3（消融用）
python run_pipeline.py --no-reflect
python run_pipeline.py --no-kgfilter
python run_pipeline.py --no-step3

# 快速冒烟（每 split 20 条）
python run_pipeline.py --quick-test
```

产物（关键）：
- `outputs/preanalysis_report.json` 预学习报告
- `outputs/step3_final_{ds}_{split}.json` 过滤后实体
- `outputs/evaluation_report.{md,json}` NER F1 报告

### 断言阶段（5-9）

```bash
# 必须先跑完 NER 的 step3，再执行
python run_assertion_pipeline.py --dataset cmeee
python run_assertion_pipeline.py --dataset imcs

# 已有标注，只重训分类器
python run_assertion_pipeline.py --dataset cmeee --skip-annotate
```

产物：
- `outputs/assertion_{ds}_{split}.json` LLM 标注结果（train/dev/test）
- `outputs/assertion_{ds}_train_aug.json` 增强后训练集
- `outputs/assertion_clf/` 训练好的 RoBERTa 模型
- `outputs/assertion_eval_report.json` macro F1 报告

---

## KG 文件格式

`MNER_KG_PATH` 指向 JSON，支持两种 schema：

```jsonc
// A) 平铺列表
["腹泻", "发热", "咳嗽", ...]

// B) 带语义信息（推荐）
{
  "腹泻": {
    "synonyms":  ["拉肚子", "稀便"],
    "hypernyms": ["消化系统症状"],
    "related":   ["脱水", "腹痛"]
  },
  ...
}
```

KG 缺失时自动降级使用训练集词表（已锚定语料分布，不会引入分布外信息）。

---

## 防数据泄露设计

| 风险 | 对策 |
|------|------|
| test 数据进入训练 | 主流程严格按 split 隔离；分类器 train+dev 才喂入 |
| 同文档实体跨集合 | `group_split` 按 `doc_id / dialogue_id` 分组划 val |
| 预学习偷看 dev/test | `preanalysis.py` 强制只读 `train` |
| Few-shot 偷看 dev/test | `build_global_few_shot` 强制只读 `train` |
| KG 外部信息泄露 | KG 在断言前接入；过滤阈值固定 0.80 |
| 增强样本污染 dev | 增强样本只附加到 `train`，dev 保持原始分布 |

---

## 调参建议

| 现象 | 旋钮 |
|------|------|
| NER 召回低 | `--cmeee-long-min 4`；降低 `HIGH_SIM_THRESHOLD` |
| NER 精度低 | 开启 `--no-step3` 关闭看效果；提高 `HIGH_SIM_THRESHOLD` |
| IMCS 归一化差 | 检查 `symptom_norm.csv` 是否完整；调 `kg_alignment.py` 中阈值 |
| 断言 macro F1 < 0.9 | 检查少数类是否触发增强；调大 `CLF_EPOCHS`；扩 `CONTEXT_WINDOW_CHARS` |
| 显存 OOM | 设 `MNER_USE_4BIT=true`；调小 `extract_entities.py` 中 batch 常量 |

---

## 实施限制说明

本仓库是代码框架。要复现 F1 指标，需准备：

1. 三个数据集（CMeEE_V2 / IMCS_V2 / yidu_4k）
2. 本地模型权重（Qwen3-32B、DeepSeek、bge、roberta）
3. 单卡 ≥ 32GB 显存（BF16）或 ≥ 16GB（4bit）
4. 可选：外部医学 KG JSON

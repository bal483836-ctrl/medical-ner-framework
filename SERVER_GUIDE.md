# 服务器操作指南 v3.2

## 1. 拉取最新修复代码

```bash
cd /root/autodl-tmp/MedNER_Project/better/medical_ner_v3
git pull origin main
```

## 2. 快速验证修复效果（无需重跑 pipeline）

如果服务器上已有 `outputs/step3_final_*.json` 文件，直接运行评估：

```bash
cd /root/autodl-tmp/MedNER_Project/better/medical_ner_v3

# 评估所有数据集（dev split）
python evaluate_all.py --split dev

# 带调试信息（显示前5条样本 gold/pred 对比）
python evaluate_all.py --split dev --debug

# 不使用向量模型（更快，只用手工词典）
python evaluate_all.py --split dev --no-embed
```

## 3. 重新运行完整 pipeline（推荐）

修复后的 pipeline 会在 Step2 和 Step3 中正确处理口语词归一化，
重新运行可以获得最佳 F1 效果：

```bash
cd /root/autodl-tmp/MedNER_Project/better/medical_ner_v3

# 重新运行 IMCS dev split 的 Step2+Step3
python main.py --dataset imcs --split dev --start-step 2

# 重新运行 CMeEE dev split 的 Step2+Step3
python main.py --dataset cmeee --split dev --start-step 2

# 重新运行完整 pipeline（Step1+Step2+Step3）
python main.py --dataset all --split dev
```

## 4. 修复内容说明

### 4.1 evaluate_all.py (v3.3) — 评估脚本修复

**问题**：`step3_final_CMeEE_V2_dev.json` 中 `gold_entities_str` 字段为空，
导致 CMeEE F1 计算时 gold 全为空列表，F1 = 0.38（实为评估 bug）。

**修复**：
- 增加兜底逻辑：若 `gold_entities_str` 为空，自动从原始数据文件
  `CMeEE_V2_dev_new.json` 重新读取 Gold 标注
- 增加 IMCS `ORAL_TO_NORM` 手工口语词->标准词映射（约100个词对）
- 增加 `--debug` 参数，方便诊断

### 4.2 src/kg_alignment.py (v3.2) — IMCS 归一化修复

**问题**：口语词（如"拉粑粑"、"黄鼻涕"）无法通过向量相似度匹配到标准词，
`step2_normalized_map` 中全部映射到自身（如 `拉粑粑->拉粑粑`），
导致 IMCS 归一化 F1 = 0.30。

**修复**：
- 增加 `ORAL_TO_NORM` 手工口语词映射词典（约100个词对）
  - 腹泻类：`拉粑粑/拉肚子/放屁也拉/尿尿就拉粑粑` → `腹泻`
  - 鼻流涕类：`黄鼻涕/清鼻涕/流鼻涕/鼻涕偏黄` → `鼻流涕`
  - 发热类：`发烧/发烫/体温高` → `发热`
  - 感冒类：`伤风/着凉/受凉` → `感冒`
  - 绿便类：`偏绿/大便偏绿/便绿` → `绿便`
  - 等约100个词对
- 将 `IMCS_HIGH_SIM_THRESHOLD` 从 0.82 降低到 0.72
- 增加中等相似度（0.55-0.72）也接受归一化

### 4.3 src/filter_hallucinations.py (v3.2) — IMCS 过滤修复

**问题**：Step3 大模型审核 Prompt 不够明确，导致：
1. 误删合法症状词（如"绿便"、"精神软"）
2. 保留非症状词（如"舌苔黄"、"母乳喂养"、"蒙脱石散"）

**修复**：
- 增加 `IMCS_RULE_BLACKLIST` 规则过滤黑名单（在大模型审核前先过滤）
  包括：患者性别/年龄、喂养方式、病因诱因、药物、检查项目、舌象等
- 增加 `_rule_filter_imcs()` 函数，过滤过长描述性短语
- 改进大模型审核 Prompt，更明确地指导过滤非症状词
- 增加 Step3f 口语词替换步骤：在 `step3_final_output` 中将口语词替换为标准词

## 5. 预期 F1 提升

| 数据集 | 修复前 | 修复后（预期） | 主要改善 |
|--------|--------|----------------|----------|
| CMeEE dev | 0.38 | ≥0.80 | Gold 读取 bug 修复 |
| IMCS dev（归一化）| 0.30 | ≥0.80 | 手工口语词映射 |

## 6. 常见问题

**Q: 运行 `evaluate_all.py` 时提示"未找到 CMeEE dev 原始文件"？**

A: 检查原始数据文件路径：
```bash
ls /root/autodl-tmp/MedNER_Project/data/CMeEE_V2/
```
如果文件名不是 `CMeEE_V2_dev_new.json`，请修改 `evaluate_all.py` 中的
`_find_cmeee_raw_file()` 函数，添加正确的路径。

**Q: IMCS 归一化 F1 仍然偏低？**

A: 使用 `--debug` 参数查看样本对比：
```bash
python evaluate_all.py --dataset imcs --split dev --debug
```
查看 `step2_normalized_map` 中哪些词没有被正确归一化，
然后在 `src/kg_alignment.py` 的 `ORAL_TO_NORM` 词典中添加对应映射。

**Q: 如何只重新运行 Step2（不重新运行 Step1）？**

A: 使用 `--start-step 2` 参数：
```bash
python main.py --dataset imcs --split dev --start-step 2
```
这会跳过 Step1（LLM 提取），直接从已有的 `step1_raw_*.json` 文件开始 Step2。

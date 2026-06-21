# 医疗 NER + 断言框架 — 全流程优化方案（冲刺 NER micro-F1 ≥ 0.80 / 断言 macro-F1 ≥ 0.91）

> 目标：把"用户上传医学文档 + 一小撮金标（实体+断言）→ 框架自动抽取/标注 → 训练断言模型"
> 这套零样本弱监督管线做到论文级指标。本文给出全流程梳理、与目标的差距分析、
> 以及**有文献支撑**的逐模块创新点与优先级路线图。

---

## 0. 框架范式：本质是"LLM 弱监督 → 蒸馏到小模型"

整个框架其实是 **FreeAL / weak-supervision 范式**（LLM 当标注器，小模型当学生）：

```
用户输入: 医学文档 D + 小撮金标 G(实体, 断言)
   │
   ├─[NER 半区 Phase 1-3] LLM 零样本抽实体 ──→ 银标实体
   │       (G 用作 few-shot + 预学习 skills)
   │
   └─[断言半区 Phase 4-9] LLM 标断言 ──→ 银标断言 ──→ 训练 RoBERTa 断言模型
           (KG 扩展 + 语境窗口 + 自洽投票 + 数据增强)
```

**关键洞察**：断言半区已经验证了"LLM 标注 → 训练小模型"能work（focal+R-Drop+FGM+集成）。
**NER 半区目前却只评估 LLM 直接抽取的结果**，没有蒸馏步骤 —— 这正是 0.80 难达标的根因。

---

## 1. 全流程详细梳理

### 1.1 NER 半区（Phase 1-3）

| 阶段 | 模块 | 做什么 | 当前问题 |
|---|---|---|---|
| 预学习 | `preanalysis.py` | 每数据集独立：LLM 看 30 条 demo → 生成结构化 skills（必抽/过滤/医学性/边界 4 段）+ dry-run 自校验 | skills 质量已改善，但仅注入 prompt，未参与训练 |
| Step1 | `extract_entities.py` | LLM 逐条零样本抽取，原文锚定后处理 | **dev 未见实体召回仅 0.13**（泛化差） |
| Step1.5 | `cmeee_expand.py` | 基于 train 词表做嵌套扩展 | 词表含 train 金标 → 泄漏（train 虚高） |
| Step1.7 | `reflector.py` | 第二个 LLM 反思补漏/删错 | 早期"只增不删"伤 P，已收紧 |
| Step2 | `kg_alignment.py` | 实体对齐 KG 标准词 | 曾用 best_match 替换原词伤 R，已改原词优先 |
| Step2.5 | `kg.py` filter | KG 余弦≥0.8 过滤 | 硬删非常规实体伤 R，已加 `boost` 模式 |
| Step3 | `filter_hallucinations.py` | 规则/LLM 幻觉过滤 | IMCS Step3 净伤 F1，已加 skip 开关 |
| Step4 | 评估 | 字面匹配 micro/macro F1 | — |

**当前实测**（smoke 200，字面匹配）：CMeEE dev micro ≈ 0.55，IMCS dev ≈ 0.53。

### 1.2 断言半区（Phase 4-9）

| 阶段 | 模块 | 做什么 |
|---|---|---|
| Phase 4 | `kg.py::expand()` | 每实体生成 synonyms/hypernyms/possible_diseases/kg_facts（含**反向索引**：症状→可能疾病） |
| Phase 5 | `context_window.py` | 双引擎（BGE 语义 10.0 + spaCy 句法 5.0 + 断言线索 +3）贪心扩窗，max 512 字；对话走轮次切片 |
| Phase 6 | `assertion_annotator.py` | LLM 标 4 类（确定/疑似/无/知识事实），**3 轮自洽投票**+变体扰动+多数表决 |
| Phase 7 | `augmentor.py` | 少数类改写增强到 max×0.85 → **二次校验**（仅保留标签一致）→ 配额截断 |
| Phase 8 | `assertion_train.py` | RoBERTa-wwm-ext + Focal(γ=1.6) + 标签平滑(0.05) + R-Drop(α=0.5) + FGM(ε=0.11)，**按文档分组**切分防泄漏，3 种子集成 |
| Phase 9 | `assertion_eval.py` | 多模型 softmax 平均 → **逐类 bias 网格搜索**（dev 上）→ macro/micro F1 |

**4 类标签**：确定(Present) / 疑似(Possible) / 无(Absent) / 知识事实(General)。

---

## 2. 与目标的差距 + 诚实评估

### NER：0.55 → 0.80（gap 0.25，最难）

> ⚠️ **诚实提醒**：CMeEE-V2 全监督发表 SOTA 约 0.67–0.74 micro（严格 span+type）。
> 本框架用**字面匹配**（实体串级、宽松），0.80 在宽松口径下可达，
> 但**纯靠 LLM 零样本 prompt 工程无法补 0.25** —— 文献一致结论：
> token 级临床 NER，监督小模型（BioClinicalBERT）> GPT-4 零样本，即便 GPT-4 经 prompt 工程到 0.861 仍逊于监督模型（JAMIA 2024）。

**结论**：必须引入"**银标 → 训练监督 NER 模型**"这一步（与断言半区对称）。

### 断言：当前未知 → 0.91（路径较清晰）

断言半区结构完整，0.91 macro 主要卡在**标注噪声**与**类别不均**。文献（Confident Learning / FreeAL）有成熟方法补这 1-3 个点。

---

## 3. NER 半区创新点（冲 0.80）

### 🌟 创新 A：增加"NER 蒸馏训练"阶段（最高优先级，预期 +0.10~0.20）

**文献依据**：监督小模型在 token 级临床 NER 上稳定优于 LLM 零样本（JAMIA 2024；PMC12099373 "LLMs Struggle in Token-Level Clinical NER"）。

**做法**：把 LLM 银标 + 用户小撮金标，喂给一个**嵌套 NER 监督模型**：
- 架构选 **GlobalPointer / W2NER / Biaffine**（CMeEE 是嵌套 NER，必须支持嵌套，不能用普通 BIO-CRF）。
- 基座用 `chinese-roberta-wwm-ext`（断言已在用，复用）。
- 训练数据 = LLM step3 银标（按 confident-learning 清洗，见创新 D）+ 全部用户金标。
- **关键**：用户金标做验证集 + 高权重训练样本。

这一步把"LLM 抽取上限 0.67"变成"监督模型可学习的分布"，且监督模型能**泛化到 LLM 漏抽的未见实体**（当前 dev-unseen R=0.13 的死穴）。

### 🌟 创新 B：多智能体 / 本体增强抽取（OEMA 范式，预期 +0.03~0.06）

**文献依据**：OEMA（Ontology-Enhanced Multi-Agent，arXiv 2511.15211）——多智能体协作 + 本体知识做零样本临床 NER，超越单 LLM 基线。

**做法**：把现在的单 prompt 抽取拆成协作角色：
1. **抽取 agent**：高召回宽松抽（现状）。
2. **本体校验 agent**：用 KG（你已有 9 万节点）判断每个候选"是否医学实体 + 属哪类"。
3. **自我反思 agent**：对照原文删幻觉（现 reflector）。
- 这把"医学性判定"从 prompt 规则升级成 **KG 检索增强的独立校验步**，比让一个 LLM 同时干所有事更稳。

### 创新 C：实体级自洽投票（Self-Improving NER，预期 +0.02~0.04）

**文献依据**："Self-Improving for Zero-Shot NER"（NAACL 2024）——无标注语料上多次采样，**实体级**一致性选择高置信实体做自训练。

**做法**：Step1 对同一文本跑 N=3 次（temperature≈0.3 加扰动），统计每个实体出现频次：
- 出现 ≥2/3 → 高置信，直接保留。
- 出现 1/3 → 低置信，交反思/KG 校验。
- 这比单次抽取稳，且天然产出"置信度"供创新 D 清洗用。
- ⚠️ 注意文献也指出 SC 对低置信样本无效（arXiv 2408.12249），所以**只用频次做软信号，不用它当唯一裁决**。

### 创新 D：用 Confident Learning 清洗银标（预期 +0.02~0.05，且利好下游断言）

**文献依据**：Confident Learning / cleanlab（JAIR 2021）——用模型 out-of-sample 预测概率估计哪些标签是错的。

**做法**：训练创新 A 的 NER 模型时，用 cleanlab 找出"LLM 银标与模型强烈分歧"的样本：
- 这些大概率是 LLM 抽错/漏标 → 剔除或降权。
- 干净银标回灌，再训一轮（FreeAL 协同循环）。

### 创新 E：消除 train 泄漏（让分数可信，非提分但论文必须）

当前 `MNER_MERGE_TRAIN_VOCAB` 默认把 train 金标塞进 KG 词表 → train F1 虚高（0.71，关掉掉到 0.55）。
**论文里必须 `MNER_MERGE_TRAIN_VOCAB=false` 跑公平分**，否则 reviewer 一眼看穿数据泄漏。

---

## 4. 断言半区创新点（冲 0.91 macro）

### 🌟 创新 F：Confident Learning 清洗断言银标（最高优先级，预期 +0.02~0.04）

**文献依据**：Confident Learning（JAIR 2021）；"Adjudicator: KG-Informed Council of LLM Agents"（arXiv 2512.13704，2025）——KG + LLM 议会修正噪声标签。

**做法**：3 轮投票后，用 vote_dist 当置信度：
- 只有 3/3 一致的进训练核心集；2/3 的标"弱"，用训练好的分类器做 cleanlab 二次裁决；1/3 分散的直接弃。
- 当前实现只做多数表决，**没有按一致性分层** —— 这是最容易补的提分点。

### 🌟 创新 G：把 KG 结构化喂进分类器（预期 +0.02~0.05）

**现状**：`possible_diseases` / `kg_facts` 只拼进 LLM prompt，**分类器没用上**（只拿到 `实体+kg_string` 拼成 query）。

**做法**：
- 把 `possible_diseases` 数量、是否命中否定线索词、KG 是否认证，做成**显式特征拼进输入**或加**辅助分类头**（多任务：断言标签 + 知识事实判别）。
- "知识事实"类的强信号正是"possible_diseases 多 + 原文无患者指向" —— 直接编码进模型比让它从语境隐式学更准。

### 创新 H：FreeAL 协同循环（LLM ↔ 小模型，预期 +0.02~0.03）

**文献依据**：FreeAL（arXiv 2311.15614）——LLM 标注 → 小模型用 loss 区分干净/噪声 → 干净样本回灌 LLM 做 demonstration 重标噪声样本。

**做法**：断言分类器训练时用 GMM 拟合 loss 分布，分出干净/噪声子集；噪声样本连同干净示例回灌 LLM 重标。1-2 轮迭代。

### 创新 I：语境窗口 + 输入格式优化（预期 +0.01~0.03）

- `CLF_MAX_LEN` 512 对长文档截断 → 升 768（显存够的话）。
- 当前 `query + context_text` 双段拼接，实体用【】标记。改成标准 **`[CLS] query [SEP] [E]实体[/E]语境 [SEP]`** + 显式实体边界 special token，提升实体定位。
- 断言线索词（未见/否认/考虑/不排除）插入 `[NEG]`/`[SPEC]` special token，把隐式线索变显式。

### 创新 J：评估期 bias 搜索升级（预期 +0.01~0.02）

当前逐类 bias 网格 [-0.2,0.2] step 0.02 是贪心粗搜。改 **Optuna/温度缩放 + 逐类阈值**联合优化（dev 上），macro-F1 对少数类阈值敏感，精调常有 1-2 点。

---

## 5. 优先级路线图

| 优先级 | 创新 | 目标 | 难度 | 预期增益 | 依赖 |
|---|---|---|---|---|---|
| **P0** | A：NER 蒸馏训练（GlobalPointer/W2NER） | NER 0.80 | 高 | +0.10~0.20 | 嵌套NER实现 |
| **P0** | F：断言投票分层 + CL 清洗 | 断言 0.91 | 低 | +0.02~0.04 | 改 annotator |
| **P0** | E：关掉 train 泄漏跑公平分 | 可信度 | 极低 | — | env var |
| P1 | G：KG 结构化进分类器 | 断言 0.91 | 中 | +0.02~0.05 | 改 train 输入 |
| P1 | D：CL 清洗 NER 银标 | NER 0.80 | 中 | +0.02~0.05 | 依赖 A |
| P1 | C：实体级自洽投票 | NER 召回 | 中 | +0.02~0.04 | 改 Step1 |
| P2 | B：多智能体本体校验 | NER 精度 | 中 | +0.03~0.06 | 改抽取 |
| P2 | H：FreeAL 协同循环 | 断言 | 高 | +0.02~0.03 | 依赖 F |
| P2 | I：语境/输入格式 | 断言 | 中 | +0.01~0.03 | 改 train |
| P3 | J：bias 搜索升级 | 断言 | 低 | +0.01~0.02 | 改 eval |

**最短冲刺路径**：
1. 先做 P0 的 E（10 分钟，拿可信 baseline）。
2. 做 P0 的 F（断言投票分层，半天，断言最快接近 0.91）。
3. 做 P0 的 A（NER 蒸馏，1-2 天，NER 唯一能上 0.80 的路）。
4. A 跑通后叠 D + C（清洗 + 自洽），逼近 0.80。
5. 断言叠 G（结构化 KG），冲稳 0.91。

---

## 6. 关于目标可达性的诚实结论

- **断言 0.91 macro**：路径清晰，P0-F + P1-G 大概率达成。断言半区工程已很扎实。
- **NER 0.80 micro**：
  - 字面匹配口径 + 监督蒸馏（创新 A）+ 集成，**可达**。
  - 若 reviewer 要求严格 CMeEE span+type 口径，0.80 高于现有 SOTA，需诚实在论文里说明用的是宽松/实体串口径，或聚焦"零样本相对提升"这一卖点而非绝对 SOTA。
  - **论文叙事建议**：主打"零样本弱监督框架，用极少金标 + LLM 蒸馏，逼近全监督性能"，比硬刚绝对 SOTA 更稳、更有故事性。

---

## 7. 参考文献

- Self-Improving for Zero-Shot NER with LLMs — NAACL 2024. https://aclanthology.org/2024.naacl-short.49/
- LLMs are not Zero-Shot Reasoners for Biomedical IE — 2024. https://arxiv.org/abs/2408.12249
- OEMA: Ontology-Enhanced Multi-Agent for Zero-Shot Clinical NER — 2025. https://arxiv.org/abs/2511.15211
- Improving LLMs for Clinical NER via Prompt Engineering — JAMIA 2024. https://academic.oup.com/jamia/article/31/9/1812/7590607
- LLMs Struggle in Token-Level Clinical NER — PMC. https://pmc.ncbi.nlm.nih.gov/articles/PMC12099373/
- Beyond Negation Detection: Comprehensive Assertion Detection — 2025. https://arxiv.org/abs/2503.17425
- 2010 i2b2/VA Challenge on Assertions — PMC. https://pmc.ncbi.nlm.nih.gov/articles/PMC3168320/
- Confident Learning: Estimating Uncertainty in Dataset Labels — JAIR 2021. https://dl.acm.org/doi/10.1613/jair.1.12125
- FreeAL: Human-Free Active Learning in the Era of LLMs — 2023. https://arxiv.org/abs/2311.15614
- Adjudicator: Correcting Noisy Labels with a KG-Informed Council of LLM Agents — 2025. https://arxiv.org/abs/2512.13704

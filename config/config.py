"""
医疗 NER + 断言训练框架配置（v4）

阶段 1-3：实体抽取与评估（NER micro F1 ≥ 0.80）
阶段 4-9：知识图谱扩展 → 语境截取 → 断言标注 → 数据增强 → 小模型训练（macro F1 ≥ 0.90）

后端：Qwen3-32B + DeepSeek-V2 均本地加载；分类器使用 chinese-roberta-wwm-ext。
"""
import os

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_ROOT = os.environ.get("MNER_DATA_ROOT", "/root/autodl-tmp/MedNER_Project/data")

CMEEE_TRAIN_PATH = os.path.join(DATA_ROOT, "CMeEE_V2", "CMeEE_V2_train_new.json")
CMEEE_DEV_PATH   = os.path.join(DATA_ROOT, "CMeEE_V2", "CMeEE_V2_dev_new.json")
CMEEE_TEST_PATH  = os.path.join(DATA_ROOT, "CMeEE_V2", "CMeEE_V2_test_new.json")

IMCS_TRAIN_PATH  = os.path.join(DATA_ROOT, "IMCS_V2", "IMCS_V2_train_new.json")
IMCS_DEV_PATH    = os.path.join(DATA_ROOT, "IMCS_V2", "IMCS_V2_dev_new.json")
IMCS_TEST_PATH   = os.path.join(DATA_ROOT, "IMCS_V2", "IMCS_V2_test_new.json")

# yidu_4k：优先新版 JSON 格式（new_yidu_4k_*.json，带标注），缺失时回退 BIO
_YIDU_NEW_TRAIN = os.path.join(DATA_ROOT, "yidu", "new_yidu_4k_train.json")
_YIDU_NEW_DEV   = os.path.join(DATA_ROOT, "yidu", "new_yidu_4k_dev.json")
_YIDU_NEW_TEST  = os.path.join(DATA_ROOT, "yidu", "new_yidu_4k_test.json")
_YIDU_OLD_BIO   = os.path.join(DATA_ROOT, "yidu_4k", "subtask1_training.txt")
YIDU_TRAIN_PATH = _YIDU_NEW_TRAIN if os.path.exists(_YIDU_NEW_TRAIN) else _YIDU_OLD_BIO
YIDU_DEV_PATH   = _YIDU_NEW_DEV   if os.path.exists(_YIDU_NEW_DEV)   else None
YIDU_TEST_PATH  = _YIDU_NEW_TEST  if os.path.exists(_YIDU_NEW_TEST)  else None

SYMPTOM_NORM_CSV    = os.path.join(BASE_DIR, "data", "symptom_norm.csv")
IMCS_NORM_DICT_PATH = SYMPTOM_NORM_CSV if os.path.exists(SYMPTOM_NORM_CSV) \
                      else os.path.join(DATA_ROOT, "symptom_norm.csv")

# 外部知识图谱文件
# 优先用 entities_dict.txt + triples.txt（CMKG 风格，已随仓库附带）
KG_DICT_PATH    = os.environ.get("MNER_KG_DICT",    os.path.join(BASE_DIR, "data", "entities_dict.txt"))
KG_TRIPLES_PATH = os.environ.get("MNER_KG_TRIPLES", os.path.join(BASE_DIR, "data", "triples.txt"))
# 兼容 JSON 格式 KG（若提供）
KG_PATH         = os.environ.get("MNER_KG_PATH",    os.path.join(DATA_ROOT, "kg", "medical_kg.json"))

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ==================== 本地模型路径 ====================
MODEL_ROOT = os.environ.get("MNER_MODEL_ROOT", "/root/autodl-tmp/MedNER_Project/models")

# 主抽取与断言模型：Qwen3-32B
LLM_MODEL_PATH      = os.path.join(MODEL_ROOT, "Qwen3-32B")
# 反思模型：DeepSeek（用户可改路径）
REFLECT_MODEL_PATH  = os.path.join(MODEL_ROOT, "DeepSeek-V2-Lite-Chat")
# 向量模型
EMBEDDING_MODEL_PATH = os.path.join(MODEL_ROOT, "bge-large-zh-v1.5")
# 断言分类器基座
CLASSIFIER_BASE_PATH = os.path.join(MODEL_ROOT, "chinese-roberta-wwm-ext")

# ==================== LLM 推理参数 ====================
LLM_MAX_NEW_TOKENS     = 512
LLM_REPETITION_PENALTY = 1.05
LLM_USE_4BIT           = os.environ.get("MNER_USE_4BIT", "false").lower() == "true"
LLM_USE_FLASH_ATTN     = os.environ.get("MNER_USE_FLASH_ATTN", "false").lower() == "true"
LLM_DEVICE_MAP         = "auto"

# ==================== RTX 5090 优化（32GB 显存 / Blackwell sm_120）====================
# 默认按 5090 跑：BF16 全精度，大 batch，启用 SDPA
# 显存不足时设 MNER_USE_4BIT=true
LLM_BATCH_SIZE_CMEEE = int(os.environ.get("MNER_BATCH_CMEEE", "12"))   # 5090: 12; 5090-4bit: 16; 24GB 卡: 6
LLM_BATCH_SIZE_IMCS  = int(os.environ.get("MNER_BATCH_IMCS",  "24"))   # 短文本，可以更大
LLM_BATCH_SIZE_ASSERT = int(os.environ.get("MNER_BATCH_ASSERT", "16"))
# 全局随机种子（所有阶段共用）
GLOBAL_SEED = 42

# ==================== 向量模型 ====================
EMBEDDING_BATCH_SIZE = 256
EMBEDDING_DIM        = 1024

# ==================== 相似度阈值（KG 过滤）====================
EXACT_MATCH_THRESHOLD = 1.0
HIGH_SIM_THRESHOLD    = 0.80   # 论文要求：余弦相似度 ≥ 0.80 才保留
LOW_SIM_THRESHOLD     = 0.60

# ==================== 少样本 / 预学习配置 ====================
FEW_SHOT_COUNT          = 8
FEW_SHOT_SEED           = 42
PREANALYSIS_SAMPLE_SIZE = 100   # 阶段 2 预学习挖掘的样本数

# ==================== CMeEE 实体类型 ====================
CMEEE_TYPE_MAP = {
    "dis": "疾病", "sym": "症状", "pro": "手术操作", "equ": "医疗设备",
    "dru": "药物", "ite": "检查项目", "bod": "身体部位",
    "mic": "微生物类", "dep": "科室与部门",
}
CMEEE_TARGET_TYPES = list(CMEEE_TYPE_MAP.keys())

# ==================== IMCS 配置 ====================
# symptom_type: "1"=肯定症状, "2"=推断症状, "0"=否定/询问
IMCS_TARGET_SYMPTOM_TYPES = ["1", "2"]

# ==================== 断言标签（4 类）====================
# 确定 (Present)   ：明确确认实体存在 / 阳性
# 疑似 (Possible)  ：推断、可能、考虑、不排除
# 无 (Absent)      ：明确否认 / 阴性 / 排除
# 知识事实 (General)：通用医学事实陈述，非针对具体患者
ASSERTION_LABELS = ["确定", "疑似", "无", "知识事实"]
ASSERTION_LABEL2ID = {l: i for i, l in enumerate(ASSERTION_LABELS)}
# LLM 英文输出 → 中文标签
ASSERTION_EN2ZH = {
    "Present": "确定", "Positive": "确定", "阳性": "确定",
    "Possible": "疑似", "Suspected": "疑似", "可能": "疑似", "疑似": "疑似",
    "Absent": "无", "Negative": "无", "否定": "无", "阴性": "无",
    "General": "知识事实", "Factual": "知识事实", "一般性描述": "知识事实",
}

# ==================== 动态语境窗口 ====================
# 普通文本：实体前后字符数
CONTEXT_WINDOW_CHARS  = 80
# 对话场景：实体所在轮次前后保留的轮次数
CONTEXT_DIALOGUE_TURNS = 2

# ==================== 评估目标 ====================
F1_TARGET_NER       = 0.80
F1_TARGET_ASSERTION = 0.90
F1_TARGET           = F1_TARGET_NER   # 向后兼容旧模块
DUAL_F1_EVAL        = True

# ==================== 分类器训练超参（吸取 Optuna 黄金参数）====================
CLF_MAX_LEN       = 512
CLF_BATCH_SIZE    = 16
CLF_LEARNING_RATE = 3.38e-5       # Optuna 搜索值
CLF_EPOCHS        = 7
CLF_WARMUP_RATIO  = 0.13
CLF_WEIGHT_DECAY  = 0.01
CLF_SEED          = 42
# Focal Loss + FGM 对抗训练（关键提分点）
CLF_FOCAL_GAMMA   = 1.6
CLF_FGM_EPS       = 0.11
CLF_HIDDEN_DROPOUT = 0.15
# v4.2 新增
CLF_LABEL_SMOOTHING = 0.05          # 标签平滑
CLF_RDROP_ALPHA     = 0.5           # R-Drop KL 权重；设 0 即关闭
CLF_ENSEMBLE_SEEDS  = (42, 2024, 7) # 多种子集成（None 即单 seed）

# ==================== 数据增强 ====================
AUG_MIN_CLASS_RATIO = 0.10   # 触发阈值
AUG_MULTIPLIER      = 3      # 旧接口兼容
# v4.2：按类目标补足，每类增强到 max_class * AUG_TARGET_RATIO
AUG_TARGET_RATIO    = 0.85

# ==================== 自洽投票 ====================
# 同一样本多轮 LLM 标注取多数票，降低标注噪声（直接拉高分类器上限）
ASSERTION_VOTE_PASSES = 3

# ==================== 文件前缀 ====================
STEP1_PREFIX   = "step1_raw_"
STEP1E_PREFIX  = "step1_enriched_"
STEP2_PREFIX   = "step2_aligned_"
STEP3_PREFIX   = "step3_final_"
STEP4_PREFIX   = "step4_normalized_"
ASSERT_PREFIX  = "assertion_"   # 断言阶段输出

DATASET_SPLITS = {
    "CMeEE_V2": [
        {"split": "train", "path": CMEEE_TRAIN_PATH, "has_label": True},
        {"split": "dev",   "path": CMEEE_DEV_PATH,   "has_label": True},
        {"split": "test",  "path": CMEEE_TEST_PATH,  "has_label": False},
    ],
    "IMCS_V2": [
        {"split": "train", "path": IMCS_TRAIN_PATH, "has_label": True},
        {"split": "dev",   "path": IMCS_DEV_PATH,   "has_label": True},
        {"split": "test",  "path": IMCS_TEST_PATH,  "has_label": False},
    ],
    "yidu_4k": [
        {"split": "train", "path": YIDU_TRAIN_PATH, "has_label": False},
    ],
}

# 为保持向后兼容（旧版本字段）
IMCS_FORCE_RECALL_WORDS = [
    "血便", "绿便", "稀便", "水样便", "蛋花汤样便", "脓血便",
    "黑便", "血尿", "尿频", "尿急", "尿痛", "屁", "放屁",
    "抽搐", "脱水", "精神软", "哭闹", "呕吐", "腹泻", "发热",
    "咳嗽", "咳痰", "鼻塞", "流涕", "喘息", "气促",
    "腹痛", "腹胀", "纳差", "食欲不振", "恶心",
    "皮疹", "荨麻疹", "湿疹",
]

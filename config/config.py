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

YIDU_TRAIN_PATH  = os.path.join(DATA_ROOT, "yidu_4k", "subtask1_training.txt")

SYMPTOM_NORM_CSV    = os.path.join(BASE_DIR, "data", "symptom_norm.csv")
IMCS_NORM_DICT_PATH = SYMPTOM_NORM_CSV if os.path.exists(SYMPTOM_NORM_CSV) \
                      else os.path.join(DATA_ROOT, "symptom_norm.csv")

# 外部知识图谱文件（用户后续提供路径；不存在时降级为训练集词典）
KG_PATH = os.environ.get("MNER_KG_PATH", os.path.join(DATA_ROOT, "kg", "medical_kg.json"))

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
# 确定：明确确认实体存在 / 阳性
# 疑似：推断、可能、考虑、不排除
# 无：明确否认 / 阴性 / 排除
# 知识事实：通用医学事实陈述，非针对具体患者
ASSERTION_LABELS = ["确定", "疑似", "无", "知识事实"]
ASSERTION_LABEL2ID = {l: i for i, l in enumerate(ASSERTION_LABELS)}

# ==================== 动态语境窗口 ====================
# 普通文本：实体前后字符数
CONTEXT_WINDOW_CHARS  = 80
# 对话场景：实体所在轮次前后保留的轮次数
CONTEXT_DIALOGUE_TURNS = 2

# ==================== 评估目标 ====================
F1_TARGET_NER       = 0.80
F1_TARGET_ASSERTION = 0.90
DUAL_F1_EVAL        = True

# ==================== 分类器训练超参 ====================
CLF_MAX_LEN       = 256
CLF_BATCH_SIZE    = 32
CLF_LEARNING_RATE = 2e-5
CLF_EPOCHS        = 5
CLF_WARMUP_RATIO  = 0.1
CLF_WEIGHT_DECAY  = 0.01
CLF_SEED          = 42

# ==================== 数据增强 ====================
# 当某断言类别样本占比低于该阈值，触发增强
AUG_MIN_CLASS_RATIO = 0.10
# 增强倍数（每条少数类样本生成 N 条变体）
AUG_MULTIPLIER      = 3

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

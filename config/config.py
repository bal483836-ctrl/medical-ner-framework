"""
医疗NER框架全局配置文件 v3
支持：CMeEE_V2 全量 + IMCS_V2 全量 + yidu_4k 提取
本地化：直接加载 Qwen3-14B + bge-large-zh-v1.5，无需任何外部 API
"""
import os

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据集根目录（可通过环境变量覆盖）
DATA_ROOT = os.environ.get(
    "MNER_DATA_ROOT",
    "/root/autodl-tmp/MedNER_Project/data"
)

# ---- CMeEE_V2 数据集路径 ----
CMEEE_TRAIN_PATH = os.path.join(DATA_ROOT, "CMeEE_V2", "CMeEE_V2_train_new.json")
CMEEE_DEV_PATH   = os.path.join(DATA_ROOT, "CMeEE_V2", "CMeEE_V2_dev_new.json")
CMEEE_TEST_PATH  = os.path.join(DATA_ROOT, "CMeEE_V2", "CMeEE_V2_test_new.json")

# ---- IMCS_V2 数据集路径 ----
IMCS_TRAIN_PATH  = os.path.join(DATA_ROOT, "IMCS_V2", "IMCS_V2_train_new.json")
IMCS_DEV_PATH    = os.path.join(DATA_ROOT, "IMCS_V2", "IMCS_V2_dev_new.json")
IMCS_TEST_PATH   = os.path.join(DATA_ROOT, "IMCS_V2", "IMCS_V2_test_new.json")

# ---- yidu_4k 数据集路径（BIO格式，只提取不评估）----
YIDU_TRAIN_PATH  = os.path.join(DATA_ROOT, "yidu_4k", "subtask1_training.txt")

# ---- IMCS 官方归一化词典 ----
# 优先使用框架 data/ 目录下的官方词典，不存在时使用数据集目录下的备用路径
SYMPTOM_NORM_CSV = os.path.join(BASE_DIR, "data", "symptom_norm.csv")
IMCS_NORM_DICT_PATH = SYMPTOM_NORM_CSV if os.path.exists(SYMPTOM_NORM_CSV) else os.path.join(DATA_ROOT, "symptom_norm.csv")

# ---- 输出目录 ----
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ==================== 本地模型路径 ====================
MODEL_ROOT = os.environ.get(
    "MNER_MODEL_ROOT",
    "/root/autodl-tmp/MedNER_Project/models"
)

# 大语言模型（Qwen3-14B，直接本地加载）
LLM_MODEL_PATH = os.path.join(MODEL_ROOT, "Qwen3-14B")

# 向量模型（bge-large-zh-v1.5，直接本地加载）
EMBEDDING_MODEL_PATH = os.path.join(MODEL_ROOT, "bge-large-zh-v1.5")

# ==================== LLM 推理参数 ====================
LLM_MAX_NEW_TOKENS     = 512
LLM_TEMPERATURE        = 0.1   # 低温度保证输出稳定
LLM_TOP_P              = 0.9
LLM_REPETITION_PENALTY = 1.05

# 是否启用 4-bit 量化
# 5090 有 32GB 显存，推荐关闭（全精度 BF16，约 28GB，效果更好）
# 若显存不足可设 MNER_USE_4BIT=true（约 10GB）
LLM_USE_4BIT = os.environ.get("MNER_USE_4BIT", "false").lower() == "true"

# 是否启用 Flash Attention 2（需 pip install flash-attn）
LLM_USE_FLASH_ATTN = os.environ.get("MNER_USE_FLASH_ATTN", "false").lower() == "true"

LLM_DEVICE_MAP = "auto"

# ==================== 向量模型参数 ====================
EMBEDDING_BATCH_SIZE = 256
EMBEDDING_DIM        = 1024   # bge-large-zh-v1.5 输出维度

# ==================== 相似度阈值 ====================
EXACT_MATCH_THRESHOLD = 1.0
HIGH_SIM_THRESHOLD    = 0.82  # 高相似度：直接接受归一化
LOW_SIM_THRESHOLD     = 0.60  # 低相似度：转交大模型验证

# ==================== 少样本配置 ====================
# 少样本示例只从 train 集提取一次，全局复用
FEW_SHOT_COUNT = 8   # 增加到8条，让模型更好理解数据集风格
FEW_SHOT_SEED  = 42

# ==================== CMeEE 实体类型映射 ====================
CMEEE_TYPE_MAP = {
    "dis": "疾病",
    "sym": "症状",
    "pro": "手术操作",
    "equ": "医疗设备",
    "dru": "药物",
    "ite": "检查项目",
    "bod": "身体部位",
    "mic": "微生物类",
    "dep": "科室与部门",
}
CMEEE_TARGET_TYPES = ["dis", "sym", "pro", "bod", "ite", "dru", "mic", "equ", "dep"]

# ==================== IMCS 配置 ====================
# symptom_type: "1"=肯定症状, "2"=推断症状, "0"=否定/询问（不提取）
IMCS_TARGET_SYMPTOM_TYPES = ["1", "2"]

# IMCS 强召回兜底词库（防止大模型遗漏微小体征）
IMCS_FORCE_RECALL_WORDS = [
    "血便", "绿便", "稀便", "水样便", "蛋花汤样便", "大便粘液", "脓血便",
    "黑便", "白便", "黄便", "便血", "血尿", "尿频", "尿急", "尿痛",
    "屁", "放屁", "尿量减少", "尿量增多", "抽搐", "脱水", "精神软",
    "哭闹", "呕吐", "腹泻", "呃逆", "肠鸣音亢进", "发热", "发烧",
    "咳嗽", "咳痰", "鼻塞", "流涕", "鼻流涕", "喘息", "气促",
    "腹痛", "腹胀", "纳差", "食欲不振", "消化不良", "恶心",
    "啰音", "湿啰音", "干啰音", "哮鸣音", "嗓子沙哑", "声音嘶哑",
    "皮疹", "荨麻疹", "湿疹", "红疹", "疹子",
]

# ==================== 评估配置 ====================
DUAL_F1_EVAL = True   # IMCS 同时评估字面 F1 和归一化 F1
F1_TARGET    = 0.80

# ==================== 输出文件前缀 ====================
STEP1_PREFIX  = "step1_raw_"
STEP1E_PREFIX = "step1_enriched_"   # CMeEE 嵌套扩展后
STEP2_PREFIX  = "step2_aligned_"
STEP3_PREFIX  = "step3_final_"
STEP4_PREFIX  = "step4_normalized_"

# ==================== 数据集 split 定义 ====================
# has_label: 是否有 Gold 标注（用于决定是否评估 F1）
DATASET_SPLITS = {
    "CMeEE_V2": [
        {"split": "train", "path": CMEEE_TRAIN_PATH, "has_label": True},
        {"split": "dev",   "path": CMEEE_DEV_PATH,   "has_label": True},
        {"split": "test",  "path": CMEEE_TEST_PATH,   "has_label": False},  # test无标注
    ],
    "IMCS_V2": [
        {"split": "train", "path": IMCS_TRAIN_PATH, "has_label": True},
        {"split": "dev",   "path": IMCS_DEV_PATH,   "has_label": True},
        {"split": "test",  "path": IMCS_TEST_PATH,   "has_label": False},   # test无标注
    ],
    "yidu_4k": [
        {"split": "train", "path": YIDU_TRAIN_PATH, "has_label": False},    # 只提取
    ],
}

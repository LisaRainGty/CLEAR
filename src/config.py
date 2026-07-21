"""CLAIMARC 数据流水线全局配置。

所有路径都基于环境变量 CLAIMARC_ROOT（缺省为本仓库根目录），避免硬编码 /mnt。
LLM/VLM 走 matpool 网关（OpenAI 兼容）。模型与超参可用环境变量覆盖。
"""
from __future__ import annotations

import os
from pathlib import Path


# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------
def _root() -> Path:
    env = os.environ.get("CLAIMARC_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # 缺省：本文件位于 <root>/src/config.py
    return Path(__file__).resolve().parents[1]


ROOT = _root()
DATA = ROOT / "data"
RAW = DATA / "raw"
INDEX = DATA / "index"
PROCESSED = DATA / "processed"
FINAL = DATA / "final"
CACHE = DATA / "cache"  # LLM/VLM 调用磁盘缓存

# 原始数据
PRODUCT_INDEX = INDEX / "product_index.json"
RAW_COMMENT = RAW / "comment"
RAW_SRT = RAW / "srt_cut"
RAW_IMAGES = RAW / "product_images"

# 阶段产物
STAGE_A = PROCESSED / "stageA"
STAGE_B = PROCESSED / "stageB"
STAGE_C = PROCESSED / "stageC"
LABELS_PATH = PROCESSED / "labels.jsonl"
DATASET_PATH = FINAL / "dataset.jsonl"  # join_split 流水线中间产物

# 论文最终数据集（data/ 根目录，见 data/README.md）
DATASET_SUPERVISED = DATA / "dataset_duallabel_FULLPOOL_PLUS_OBJNEG_supervised_20260615.jsonl"
DEFAULT_TRAIN_DATASET = DATASET_SUPERVISED

def ensure_runtime_directories() -> None:
    """Create writable pipeline directories only when a caller explicitly asks.

    Importing configuration must remain read-only so a compact reviewer checkout
    can keep the archived ``raw`` and ``processed`` layers unexpanded.
    """
    for path in (PROCESSED, FINAL, CACHE, STAGE_A, STAGE_B, STAGE_C):
        path.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# LLM / VLM（matpool 网关，OpenAI 兼容）
# --------------------------------------------------------------------------
MATPOOL_BASE_URL = os.environ.get("MATPOOL_BASE_URL", "https://token.matpool.com/v1")
MATPOOL_API_KEY = os.environ.get("MATPOOL_API_KEY", "")

TEXT_MODEL = os.environ.get("CLAIMARC_TEXT_MODEL", "Qwen-Flash")
VISION_MODEL = os.environ.get("CLAIMARC_VISION_MODEL", "Qwen3-VL-Plus")
EMBED_MODEL = os.environ.get("CLAIMARC_EMBED_MODEL", "BAAI/bge-large-zh-v1.5")
# 带 torch 的 Python（用于 BGE 嵌入）；与主流水线 Python 可不同。空=用当前解释器。
EMBED_PYTHON = os.environ.get("CLAIMARC_EMBED_PYTHON", "")

# 调用控制
LLM_MAX_RETRIES = 5
LLM_TIMEOUT = 120          # 秒
LLM_CONCURRENCY = int(os.environ.get("CLAIMARC_CONCURRENCY", "12"))   # 并发线程数（matpool 网关）
VISION_MAX_SIDE = 896      # C1/C4 送图前最长边缩放，控成本


# --------------------------------------------------------------------------
# Stage A — 属性 schema 与抽取
# --------------------------------------------------------------------------
A0_CLUSTER_DISTANCE = float(os.environ.get("CLAIMARC_A0_DISTANCE", "0.20"))  # 参数 key 层次聚类距离阈值（保守，宁可少合）
A2_CLUSTER_DISTANCE = float(os.environ.get("CLAIMARC_A2_DISTANCE", "0.30"))  # FREE aspect 聚合阈值（食品 pilot 阈值敏感性扫描定档：0.30 为覆盖/粒度拐点）
A2_JACCARD_DEDUP = 0.80        # FREE 粗去重 Jaccard 阈值

# attribute_id 的品类前缀（一级品类 -> 前缀）。未列出的用大写首段兜底。
CATEGORY_PREFIX = {
    "apparel_and_underwear": "APPAREL",
    "baby_kids_and_pets": "BABY",
    "beauty_and_personal_care": "BEAUTY",
    "digital_and_electronics": "DIGITAL",
    "food_and_beverages": "FOOD",
    "general": "GEN",
    "jewelry_and_collectibles": "JEWEL",
    "shoes_and_bags": "SHOEBAG",
    "smart_home": "HOME",
    "sports_and_outdoor": "SPORT",
}


def category_prefix(category: str) -> str:
    return CATEGORY_PREFIX.get(category, (category.split("_")[0] or "CAT").upper())


# --------------------------------------------------------------------------
# Stage B — claim 抽取与对齐
# --------------------------------------------------------------------------
B3_CLAIM_JACCARD_DEDUP = 0.9   # 相邻 claim 去重（防主播复读）
B3_PASSAGE_MAX_TOKENS = 600    # passage 超过则沿时间轴下采样
B3_PASSAGE_TARGET_TOKENS = 500
CLIP_BREAK = "\n===CLIP_BREAK|{srt_file}===\n"


# --------------------------------------------------------------------------
# Stage C — 商品事实证据
# --------------------------------------------------------------------------
IMAGE_CATEGORIES = [
    "spec_table", "certificate", "size_chart",      # -> C3 OCR
    "material_closeup", "product_photo", "scene_demo", "packaging", "comparison", "other",  # -> C4 VLM
]
OCR_CATEGORIES = {"spec_table", "certificate", "size_chart"}
OCR_MODEL = os.environ.get("CLAIMARC_OCR_MODEL", "mobile")  # server=高召回(慢) | mobile=快

# 评价泄漏硬过滤：这些是笼统主观评价/购买意愿，无具体客观属性指向，按文档应归 personal，
# 从 A_cmt 中剔除（不进 pair）。注意保留"价格"等客观属性（不在此列）。
EVAL_LEAKAGE_KEYWORDS = [
    "性价比", "满意度", "满意程度", "回购", "复购", "购买动机", "购买理由", "购买原因",
    "推荐度", "推荐意愿", "推荐程度", "总体感受", "总体评价", "整体评价", "整体感受",
    "喜欢程度", "好评度", "值得购买", "值得入手", "划算", "产品品质", "品质好坏", "购买体验",
]
OCR_ALL_IMAGES = os.environ.get("CLAIMARC_OCR_ALL", "1") == "1"  # True=对全部详情图做OCR（配料/产地/日期文字遍布各图）
OCR_WORKERS = int(os.environ.get("CLAIMARC_OCR_WORKERS", "8"))   # OCR 进程池并发数（CPU）
OCR_TEXT_CAP = int(os.environ.get("CLAIMARC_OCR_CAP", "24000"))  # 单商品 OCR 文本送 LLM 的字符上限
C3_PER_IMAGE = os.environ.get("CLAIMARC_C3_PER_IMAGE", "1") == "1"  # True=逐图定向抽取（抗稀释，全图OCR时更准）
C1_REP_IMAGE_MAX = 8           # C4 代表图上限
CONFIDENCE_BY_COVERAGE = {3: "high", 2: "medium", 1: "low", 0: "absent"}


# --------------------------------------------------------------------------
# §2 标签与样本权重（默认值，见 Methodology_Data §2.6）
# --------------------------------------------------------------------------
GAMMA = 2.0          # explicit_fact_hit 加成
K_SAT = 3.0          # 证据饱和速率
LAMBDA_POS = 0.3     # 正向证据不对称折扣（仅 y=1）
RHO_FAKE = 0.4       # 刷单嫌疑惩罚（仅 y=0）
PHI_BONUS = 1.2      # 强 neg 证据加成
BETA_COV = 1.0       # coverage 软化指数（可选）
STRENGTH_MULT = {"strong": 1.2, "weak": 0.7}
C_FLOOR = 0.05       # 权重下限
FAKE_LOWN = 10       # suspected-fake 规则 (a): N_total < 10 且全 pos
FAKE_JACCARD_DIV = 0.3   # 规则 (b): aligned 评论 bigram Jaccard 多样性阈值
FAKE_WINDOW_DAYS = 3     # 规则 (c): aligned 评论集中在 <=3 天


# --------------------------------------------------------------------------
# 最终划分
# --------------------------------------------------------------------------
SPLIT_RATIO = {"train": 0.70, "val": 0.10, "test": 0.20}
SPLIT_GROUP_KEY = "room_id"     # 按直播间分组
SPLIT_SEED = 42


# --------------------------------------------------------------------------
# Token 流（dual-stream）特殊 token —— 下游模型用，这里登记便于 dataset 校验
# --------------------------------------------------------------------------
SPECIAL_TOKENS = ["[ATTR]", "[CLM]", "[CLM_NULL]", "[EVD]", "[PARAM]", "[OCR]", "[VLM]", "[SEP_C]", "[SEP_E]"]
L_CLAIM_MAX = 384
L_EVIDENCE_MAX = 384


def summary() -> str:
    return (
        f"ROOT={ROOT}\n"
        f"TEXT_MODEL={TEXT_MODEL}  VISION_MODEL={VISION_MODEL}  EMBED_MODEL={EMBED_MODEL}\n"
        f"BASE_URL={MATPOOL_BASE_URL}  KEY_SET={'yes' if MATPOOL_API_KEY else 'NO'}"
    )


if __name__ == "__main__":
    print(summary())

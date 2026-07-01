import os
import torch

# Image settings
IMAGE_SIZE = 256
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)

# Training hyperparameters
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 5e-4
NUM_CLASSES = 10

# Data paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
DATA_ROOT_ENV = os.environ.get("DATA_ROOT")
if DATA_ROOT_ENV:
    DATA_ROOT = os.path.abspath(os.path.expanduser(DATA_ROOT_ENV))
else:
    DATA_ROOT = DEFAULT_DATA_ROOT

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
TEST_DIR = os.path.join(DATA_ROOT, "test")
SPECIAL_TRAIN_DIR = os.path.join(TRAIN_DIR, "special")

EVAL_VARIANTS = ("white_black", "rainbow_bw", "bw_rainbow")
DATASET_VARIANTS = list(EVAL_VARIANTS)

SPECIAL_FILENAME_VARIANTS = ("bw", "rbw", "bwr")
DATASET_FILENAME_VARIANTS = {
    "white_black": ("bw",),
    "rainbow_bw": ("rbw",),
    "bw_rainbow": ("bwr",),
    "special": SPECIAL_FILENAME_VARIANTS,
}

# Output directories
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
OUTPUT_DIR = "outputs"

# DataLoader settings
NUM_WORKERS = 4
PREFETCH_FACTOR = 2  # 僅在 NUM_WORKERS > 0 時使用

# Best-model scoring domains:
# train.py 會使用 TRAIN_DOMAINS ∩ EVAL_VARIANTS 來計算 best_model 指標
# 規則：先比平均 val acc，再以平均 val loss 當平手判定
TRAIN_DOMAINS = ["white_black", "rainbow_bw"]

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

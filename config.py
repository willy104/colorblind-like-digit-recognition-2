import os
import torch

# Image settings
IMAGE_SIZE = 256  # 圖片大小 256*256
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)
'''
此變換通常應用於輸入圖像，然後再將它們輸入到神經網路中。
說明：
transforms.ToTensor()：此變換將 PIL 影像或 NumPy ndarray 轉換為 PyTorch 張量。
並將像素值從圖像的典型範圍 [0,255] 縮放到浮點張量的 [0.0,1.0]。
transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))：此變換將影像張量的像素值標準化。
兩個參數：mean(平均)和std（標準差），在這裡我們設定mean=(0.5,0.5,0.5)和std=(0.5,0.5,0.5)。
所應用的標準化公式為 (x - 平均值) / 標準差。
設定mean=(0.5,0.5,0.5)和std=(0.5,0.5,0.5)的原因：
標準化會將像素值從[0.0,1.0]範圍轉換為[-1.0,1.0]，
讓範圍介於負值和正值之間，是因為後續的激活函數選擇為 LeakyReLU ，負值和正值會有明顯差異對比，
如此一來在神經網路架構的訓練能提升穩定性和效能。
'''

# Training hyperparameters
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 5e-4
NUM_CLASSES = 10  # 最後輸出為 0~9 十種

# Data paths (relative to data root)
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
PREFETCH_FACTOR = 2  # 每個 worker 預先載入的 batch 數量
NUM_WORKERS = 4
'''
NUM_WORKERS 代表有幾個子程序負責資料載入，每個 worker 會預先載入 prefetch_factor 個 batch，
有助於 GPU 訓練時不用「等待資料」。
總共會預先準備 NUM_WORKERS * prefetch_factor 個 batch。
'''

# Best-model scoring domains:
# train.py 會使用 TRAIN_DOMAINS ∩ EVAL_VARIANTS 來計算 best_model 指標
# 規則：先比平均 val acc，再以平均 val loss 當平手判定
TRAIN_DOMAINS = ["white_black", "rainbow_bw"]

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # DEVICE 使用 GPU

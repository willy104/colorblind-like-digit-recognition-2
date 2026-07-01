# Colorblind-like Digit Recognition

本專案將原本的 Google Colab + Drive 架構重構成可在 **本機 Windows/Linux** 執行的完整 PyTorch 專案。  
目標是辨識不同色彩風格的手寫數字圖片，輸出類別 `0~9`。

---

## 專案資料夾結構

> `data/` 以下為**架構示意**；實際資料可不放進版控。  
> `data/example` 用於展示圖片風格長相，方便理解訓練圖。

```text
project/
│
├── data/
│   ├── train/          # 訓練圖片（<variant>_digit_X_NNNNNN.png）
│   │   ├── white_black/
│   │   ├── rainbow_bw/
│   │   ├── bw_rainbow/
│   │   └── special/    # 特殊訓練集（可混合 bw/rbw/bwr）
│   ├── val/            # 驗證圖片
│   │   ├── white_black/
│   │   ├── rainbow_bw/
│   │   └── bw_rainbow/
│   ├── test/           # 測試圖片
│   │   ├── white_black/
│   │   ├── rainbow_bw/
│   │   └── bw_rainbow/
│   └── example/        # 圖片風格示意（展示用途）
│
├── checkpoints/        # 每個 epoch checkpoint 與最佳模型
├── logs/               # 訓練 / 測試 log
├── outputs/            # 指標輸出（Excel）
│
├── config.py           # 超參數、路徑、domain 設定
├── dataset.py          # 自訂 Dataset 與資料前處理
├── model.py            # CNN 模型架構
├── train.py            # 訓練流程（含 cross-domain validation）
├── val.py              # 驗證流程工具
├── test.py             # 測試流程（交叉測試）
├── infer.py            # 單圖推論 CLI
├── utils.py            # 工具函式（checkpoint 存取、Excel 指標輸出）
├── requirements.txt    # Python 套件需求清單
└── README.md           # 本說明檔
```

---

## 快速開始

### 1) 安裝套件

```bash
pip install -r requirements.txt
```

### 2) 準備資料

檔名格式必須為：

```text
<variant>_digit_X_NNNNNN.png
```

- `X` 是 0~9 label
- `variant` 對應（由 `config.py` 控制）：

| 資料夾 | 檔名前綴 |
|---|---|
| `white_black` | `bw` |
| `rainbow_bw` | `rbw` |
| `bw_rainbow` | `bwr` |
| `special` | `bw` / `rbw` / `bwr` |

```
data/train/white_black/   ← 訓練集 (約 52000 張)
data/train/rainbow_bw/    ← 訓練集 (約 52000 張)
data/train/bw_rainbow/    ← 訓練集 (約 52000 張)
data/train/special/       ← 專有訓練集（<variant>_digit_X_NNNNNN.png）
data/val/white_black/     ← 驗證集 (約 8000 張)
data/val/rainbow_bw/      ← 驗證集 (約 8000 張)
data/val/bw_rainbow/      ← 驗證集 (約 8000 張)
data/test/white_black/    ← 測試集 (約 10000 張)
data/test/rainbow_bw/     ← 測試集 (約 10000 張)
data/test/bw_rainbow/     ← 測試集 (約 10000 張)
```

例如：

```text
data/train/white_black/bw_digit_3_000123.png
data/val/rainbow_bw/rbw_digit_8_000456.png
data/test/bw_rainbow/bwr_digit_1_000789.png
data/train/special/bw_digit_3_000123.png
data/train/special/rbw_digit_8_000456.png
data/train/special/bwr_digit_1_000789.png
```

Example:  
white_black（bw）  
![image](https://github.com/willy104/colorblind-like-digit-recognition-2/blob/main/data/example/digit_0_000095.png)  
bw_rainbow（bwr）  
![image](https://github.com/willy104/colorblind-like-digit-recognition-2/blob/main/data/example/digit_2_058563.png)  
rainbow_bw（rbw）  
![image](https://github.com/willy104/colorblind-like-digit-recognition-2/blob/main/data/example/digit_1_003878.png)

訓練時的交叉驗證與測試仍固定使用 `white_black`、`rainbow_bw`、`bw_rainbow` 三種資料集。

若資料不放在專案內的 `data/`，可設定環境變數 `DATA_ROOT` 指向本機資料根目錄（其下仍需 `train/val/test` 與三種分類資料夾）

```bash
# macOS/Linux
DATA_ROOT=/path/to/local/data python train.py --dataset white_black

# Windows PowerShell
$env:DATA_ROOT="C:\your\data\path"; python train.py --dataset white_black
```

若資料原在 Google Drive，可使用 `rclone` 或直接複製同步到本機。
---

## 從 checkpoint 續訓

```bash
# 一般訓練（使用指定 variant 訓練）
python train.py --dataset white_black

# 使用 special 訓練集（模型名可自訂）
python train.py --dataset my_special

# 續訓
python train.py --dataset white_black --resume checkpoints/white_black/checkpoint_epoch10.pth
```

訓練期間：

- 每個 epoch 都會儲存 checkpoint 到 `checkpoints/<dataset>/`
- 每個 epoch 都會在 **三種驗證集**（`EVAL_VARIANTS`）上計算指標並記錄
- 每個 epoch 的 train + val 指標會輸出到 `outputs/<dataset>/epoch_metrics.xlsx`
- log 會寫到 `logs/train_<dataset>.log`

### Best model 選擇規則（重要）

`best_model.pth` 的比較分數只使用 `TRAIN_DOMAINS`（定義於 `config.py`）：

1. **主條件：`TRAIN_DOMAINS` 平均 validation accuracy 較高者優先**
2. **平手時：`TRAIN_DOMAINS` 平均 validation loss 較低者優先**

> 注意：即使 best model 只看 `TRAIN_DOMAINS`，每個 epoch 仍會對全部 `EVAL_VARIANTS` 做驗證與記錄。

---

## 測試

```bash
python test.py --dataset white_black [--checkpoint checkpoints/white_black/best_model.pth]
```

輸出：

- 三種測試資料集 accuracy
- `logs/test_<dataset>.log`

---

## 單圖推論

```bash
python infer.py --image path/to/example.png --dataset white_black \
                [--checkpoint checkpoints/white_black/best_model.pth]
```

輸出範例：

```text
預測類別 (0-9): 3
```

---

## 模型架構

目前 `model.py` 實作為：

- 5 個 ConvBlock（Conv + LeakyReLU；前 4 個含 MaxPool）
- `AdaptiveAvgPool2d(1,1)` 壓縮空間維度
- 2 層全連接輸出 10 類

---

## 主要設定（config.py）

| 參數 | 說明 |
|---|---|
| `IMAGE_SIZE` | 輸入影像尺寸 |
| `BATCH_SIZE` | 每批次樣本數 |
| `EPOCHS` | 訓練 epoch 數 |
| `LEARNING_RATE` | Adam 學習率 |
| `NUM_WORKERS` | DataLoader worker 數 |
| `PREFETCH_FACTOR` | 每個 worker 預抓 batch 數（`NUM_WORKERS > 0` 時使用） |
| `EVAL_VARIANTS` | 每 epoch 固定驗證的 domain 清單 |
| `TRAIN_DOMAINS` | best model 計分用 domain 清單 |

---

CNN 包含：
- 5 個 `ConvBlock`（卷積 + LeakyReLU），前 4 個附 MaxPool2d
- `AdaptiveAvgPool2d(1,1)` 全域平均池化（取代龐大的 Flatten+Linear）
- 兩層 FC 最終輸出 10 類

---
  
## GPU 支援

程式會自動偵測 CUDA；若無 GPU 則使用 CPU（`config.py` 的 `DEVICE`）。

# Colorblind-like Digit Recognition

本專案將原本的 Google Colab + Drive 架構重構成可在 **本機 Windows/Linux** 執行的完整 PyTorch 專案。  
目標是辨識不同色彩風格（domain variant）的手寫數字圖片，輸出類別 `0~9`。

---

## 專案資料夾結構

> `data/` 以下為**架構示意**；實際資料可不放進版控。  
> `data/example` 用於展示圖片風格長相，方便他人理解資料分佈。

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
├── test.py             # 測試流程
├── infer.py            # 單圖推論 CLI
├── utils.py            # checkpoint / 指標輸出工具
├── requirements.txt
└── README.md
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

例如：

```text
data/train/white_black/bw_digit_3_000123.png
data/val/rainbow_bw/rbw_digit_8_000456.png
data/test/bw_rainbow/bwr_digit_1_000789.png
data/train/special/bw_digit_3_000123.png
```

若資料不在專案內建 `data/`，可用環境變數 `DATA_ROOT` 指向資料根目錄（其下仍需 `train/val/test`）。

```bash
# macOS/Linux
DATA_ROOT=/path/to/local/data python train.py --dataset white_black

# Windows PowerShell
$env:DATA_ROOT="C:\your\data\path"; python train.py --dataset white_black
```

---

## 訓練

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

## GPU 支援

程式會自動偵測 CUDA；若無 GPU 則使用 CPU（`config.py` 的 `DEVICE`）。

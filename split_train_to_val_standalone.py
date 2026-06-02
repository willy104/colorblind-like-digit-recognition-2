"""
獨立腳本：從 train 資料夾移動圖片到 val 資料夾
保持原始 MNIST 數字比例

使用方式：
    python split_train_to_val_standalone.py --train_root data/train --val_root data/val
"""
import os
import shutil
import argparse
from collections import defaultdict
from sklearn.model_selection import train_test_split


def split_variant(train_variant_dir, val_variant_dir, variant_name="dataset"):
    """
    分割單個 variant 的資料集
    
    Args:
        train_variant_dir: 訓練集資料夾路徑 (包含 60000 張圖片)
        val_variant_dir: 驗證集資料夾路徑 (會移動 8000 張到這裡)
        variant_name: variant 名稱 (用於日誌)
    """
    # 確保 val 資料夾存在
    os.makedirs(val_variant_dir, exist_ok=True)
    
    # 檢查 train 資料夾是否存在
    if not os.path.isdir(train_variant_dir):
        raise FileNotFoundError(f"Train 資料夾不存在: {train_variant_dir}")
    
    # 按數字分類收集檔案
    files_by_digit = defaultdict(list)
    for filename in os.listdir(train_variant_dir):
        if filename.endswith('.png'):
            # 格式: digit_X_NNNNNN.png → 取出 X (0-9)
            parts = filename.split('_')
            if len(parts) >= 2:
                digit = parts[1]
                files_by_digit[digit].append(filename)
    
    if not files_by_digit:
        raise ValueError(f"沒有找到 PNG 圖片在: {train_variant_dir}")
    
    total_files = sum(len(v) for v in files_by_digit.values())
    print(f"\n{'='*70}")
    print(f"處理 variant: {variant_name}")
    print(f"{'='*70}")
    print(f"總圖片數: {total_files}")
    print(f"\n按數字分類:")
    
    total_train = 0
    total_val = 0
    
    # 分層分割每個數字
    for digit in sorted(files_by_digit.keys()):
        files = files_by_digit[digit]
        count = len(files)
        
        # 按 13.33% 的比例分割 (8000/60000)
        train_files, val_files = train_test_split(
            files,
            test_size=8000/60000,
            random_state=42
        )
        
        # 移動 val 檔案
        for filename in val_files:
            src = os.path.join(train_variant_dir, filename)
            dst = os.path.join(val_variant_dir, filename)
            shutil.move(src, dst)
        
        total_train += len(train_files)
        total_val += len(val_files)
        
        print(f"  數字 {digit}: {count:5d} 張 → train {len(train_files):5d}, val {len(val_files):5d}")
    
    print(f"\n小計:")
    print(f"  Train 保留: {total_train:6d} 張")
    print(f"  Val 移動:   {total_val:6d} 張")
    
    return total_train, total_val


def main():
    parser = argparse.ArgumentParser(
        description="分割 train/val 資料集，保持 MNIST 數字比例"
    )
    parser.add_argument(
        "--train_root",
        type=str,
        default="data/train",
        help="Train 資料夾根目錄 (預設: data/train)",
    )
    parser.add_argument(
        "--val_root",
        type=str,
        default="data/val",
        help="Val 資料夾根目錄 (預設: data/val)",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=["white_black", "rainbow_bw", "bw_rainbow"],
        help="資料集 variant 名稱 (預設: white_black rainbow_bw bw_rainbow)",
    )
    args = parser.parse_args()
    
    print("開始分割 train/val 資料集...")
    print(f"Train 根目錄: {args.train_root}")
    print(f"Val 根目錄:   {args.val_root}")
    
    grand_total_train = 0
    grand_total_val = 0
    
    for variant in args.variants:
        train_dir = os.path.join(args.train_root, variant)
        val_dir = os.path.join(args.val_root, variant)
        
        try:
            train_count, val_count = split_variant(train_dir, val_dir, variant)
            grand_total_train += train_count
            grand_total_val += val_count
        except (FileNotFoundError, ValueError) as e:
            print(f"✗ 錯誤 ({variant}): {e}")
            continue
    
    print(f"\n{'='*70}")
    print(f"最終結果:")
    print(f"{'='*70}")
    print(f"  Train 總數: {grand_total_train:6d} 張")
    print(f"  Val 總數:   {grand_total_val:6d} 張")
    print(f"\n✓ 分割完成！\n")


if __name__ == "__main__":
    main()

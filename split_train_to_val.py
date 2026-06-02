"""
從 data/train/<variant>/ 移動 8000 張圖片到 data/val/<variant>/
保持原始 MNIST 數字比例
"""
import os
import shutil
from collections import defaultdict
from sklearn.model_selection import train_test_split

import config as cfg

def split_variant(variant):
    """分割單個 variant"""
    train_dir = os.path.join(cfg.TRAIN_DIR, variant)
    val_dir = os.path.join(cfg.VAL_DIR, variant)
    
    # 建立 val 資料夾
    os.makedirs(val_dir, exist_ok=True)
    
    # 按數字分類收集檔案
    files_by_digit = defaultdict(list)
    for filename in os.listdir(train_dir):
        if filename.endswith('.png'):
            digit = filename.split('_')[1]
            files_by_digit[digit].append(filename)
    
    print(f"\n{'='*60}")
    print(f"處理 variant: {variant}")
    print(f"{'='*60}")
    
    total_train = 0
    total_val = 0
    
    # 分層分割每個數字
    for digit in sorted(files_by_digit.keys()):
        files = files_by_digit[digit]
        count = len(files)
        
        # 按 13.33% 的比例分割
        train_files, val_files = train_test_split(
            files,
            test_size=8000/60000,  # 13.33%
            random_state=42
        )
        
        # 移動 val 檔案
        for filename in val_files:
            src = os.path.join(train_dir, filename)
            dst = os.path.join(val_dir, filename)
            shutil.move(src, dst)
        
        total_train += len(train_files)
        total_val += len(val_files)
        
        print(f"  數字 {digit}: {count:5d} → train {len(train_files):5d}, val {len(val_files):5d}")
    
    print(f"\n  小計: train {total_train:6d}, val {total_val:6d}")
    return total_train, total_val


def main():
    print("開始分割 train/val 資料集...")
    print(f"Train 資料夾: {cfg.TRAIN_DIR}")
    print(f"Val 資料夾: {cfg.VAL_DIR}")
    
    grand_total_train = 0
    grand_total_val = 0
    
    for variant in cfg.DATASET_VARIANTS:
        train_count, val_count = split_variant(variant)
        grand_total_train += train_count
        grand_total_val += val_count
    
    print(f"\n{'='*60}")
    print(f"最終結果:")
    print(f"{'='*60}")
    print(f"  Train 總數: {grand_total_train:6d}")
    print(f"  Val 總數:   {grand_total_val:6d}")
    print(f"\n✓ 分割完成！")


if __name__ == "__main__":
    main()

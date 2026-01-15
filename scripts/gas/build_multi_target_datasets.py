import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 配置
NORMAL_DATA_DIR = 'data/normal'
OUTPUT_ROOT = 'dataset/gas_multi_dim'
SEQ_LEN = 256
MIN_LEN = SEQ_LEN + 10 

# 目标维度列表
TARGET_COLS = [
    'CHX00F003FT0101', # 乌审旗流量
    'CHX00F002FT0101'  # 鄂托克旗流量
]

def build_dataset_for_col(col_name):
    print(f"\n--- 正在为 {col_name} 构建数据集 (尾部填充模式) ---")
    col_out_dir = os.path.join(OUTPUT_ROOT, col_name)
    os.makedirs(col_out_dir, exist_ok=True)
    
    # 1. 提取训练集
    train_dfs = []
    if not os.path.exists(NORMAL_DATA_DIR):
        print(f"错误: {NORMAL_DATA_DIR} 不存在")
        return
        
    normal_files = sorted([f for f in os.listdir(NORMAL_DATA_DIR) if f.endswith('.csv')])
    valid_segments = 0
    for f in normal_files:
        try:
            df = pd.read_csv(os.path.join(NORMAL_DATA_DIR, f))
            if '日期' in df.columns: df.rename(columns={'日期': 'date'}, inplace=True)
            if col_name in df.columns:
                clean_df = df[['date', col_name]].dropna()
                if len(clean_df) >= MIN_LEN:
                    train_dfs.append(clean_df)
                    valid_segments += 1
        except Exception as e:
            print(f"跳过文件 {f}: {e}")
    
    if train_dfs:
        # 实现尾部填充
        final_dfs = []
        for i, df in enumerate(train_dfs):
            final_dfs.append(df)
            if i < len(train_dfs) - 1:
                # 提取当前片段的最后一行并复制 SEQ_LEN 次
                last_row = df.iloc[[-1]].copy()
                padding = pd.concat([last_row] * SEQ_LEN, ignore_index=True)
                final_dfs.append(padding)
        
        train_full = pd.concat(final_dfs, axis=0)
        train_full.to_csv(os.path.join(col_out_dir, 'train.csv'), index=False)
        train_full.to_csv(os.path.join(col_out_dir, 'test.csv'), index=False)
        print(f"训练集已保存: {len(train_full)} 行 (含尾部填充缓冲带)")

        # 可视化校验
        plt.figure(figsize=(15, 5))
        plt.plot(train_full[col_name].values, color='blue', alpha=0.7, linewidth=0.5)
        plt.title(f'Training Data Check (Tail Padding) - {col_name}')
        plt.grid(True, alpha=0.2)
        plt.savefig(os.path.join(col_out_dir, 'train_check.png'), dpi=150)
        plt.close()
    else:
        print(f"❌ 警告: 没有找到有效长片段！")

def main():
    for col in TARGET_COLS:
        build_dataset_for_col(col)
    print(f"\n✅ 尾部填充构建完成。")

if __name__ == "__main__":
    main()

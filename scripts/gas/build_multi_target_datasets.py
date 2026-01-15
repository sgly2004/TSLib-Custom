import os
import pandas as pd
import matplotlib.pyplot as plt

# 配置
NORMAL_DATA_DIR = 'data/normal'
OUTPUT_ROOT = 'dataset/gas_multi_dim'
SEQ_LEN = 256
MIN_LEN = SEQ_LEN + 10 # 只有长度超过 seq_len 的数据才能提取出至少一个窗口

# 目标维度列表
TARGET_COLS = [
    'CHX00F003FT0101', # 乌审旗流量
    'CHX00F002FT0101'  # 鄂托克旗流量
]

def build_dataset_for_col(col_name):
    print(f"\n--- 正在为 {col_name} 构建数据集 ---")
    col_out_dir = os.path.join(OUTPUT_ROOT, col_name)
    os.makedirs(col_out_dir, exist_ok=True)
    
    # 1. 提取训练集 (只保留长度足够长的片段)
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
                # 关键：只有长度大于 SEQ_LEN 的片段才对训练有贡献
                if len(clean_df) >= MIN_LEN:
                    train_dfs.append(clean_df)
                    valid_segments += 1
        except Exception as e:
            print(f"跳过文件 {f}: {e}")
    
    if train_dfs:
        train_full = pd.concat(train_dfs, axis=0)
        train_full.to_csv(os.path.join(col_out_dir, 'train.csv'), index=False)
        # 为了让 DataLoader 不报错，我们需要一个 test.csv，哪怕它是 train 的副本或一个空文件
        train_full.to_csv(os.path.join(col_out_dir, 'test.csv'), index=False)
        print(f"训练集已保存: {len(train_full)} 行 (包含 {valid_segments} 个长片段)")

        # 可视化校验
        plt.figure(figsize=(15, 5))
        plt.plot(train_full[col_name].values, color='blue', alpha=0.7, linewidth=0.5)
        
        # 标出拼接点
        curr_pos = 0
        for i in range(len(train_dfs) - 1):
            curr_pos += len(train_dfs[i])
            plt.axvline(x=curr_pos, color='red', linestyle='--', alpha=0.2, linewidth=0.8)
            
        plt.title(f'Training Data Check - {col_name} (Red lines: join points)')
        plt.grid(True, alpha=0.2)
        plt.savefig(os.path.join(col_out_dir, 'train_check.png'), dpi=150)
        plt.close()
    else:
        print(f"❌ 警告: 没有找到长度大于 {MIN_LEN} 的正常数据片段！")

def main():
    for col in TARGET_COLS:
        build_dataset_for_col(col)
    print(f"\n✅ 数据构建完成。结果已放入 {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()

import os
import glob
import pandas as pd
import numpy as np

def verify_test_alignment(raw_dir='data/csv_data', single_test_csv='dataset/gas_anomaly_single/test.csv', target_col='CHX00L006PT0101'):
    print(f"--- 开始验证对齐逻辑 ---")
    
    # 1. 模拟拼接逻辑
    raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    all_parts = []
    for fp in raw_files:
        df_raw = pd.read_csv(fp)
        cols = list(df_raw.columns)
        df_raw.columns = ['date'] + cols[1:]
        # 严格执行构造时的过滤逻辑
        df_clean = df_raw.dropna(subset=['date', target_col])
        if not df_clean.empty:
            all_parts.append(df_clean[[target_col]].reset_index(drop=True))
    
    df_reconstructed = pd.concat(all_parts, axis=0, ignore_index=True)
    
    # 2. 加载已有的单维度测试集
    if not os.path.exists(single_test_csv):
        print(f"❌ 错误: 找不到文件 {single_test_csv}")
        return
    df_existing = pd.read_csv(single_test_csv)
    
    # 3. 对比
    len_rec = len(df_reconstructed)
    len_exi = len(df_existing)
    print(f"重建数据长度: {len_rec}")
    print(f"已有数据长度: {len_exi}")
    
    if len_rec != len_exi:
        print(f"❌ 长度不匹配！相差 {len_rec - len_exi} 行。")
    else:
        print(f"✅ 长度完全一致。")
        
    # 逐行数值对比 (取前 50000 行和最后 50000 行)
    diff = np.abs(df_reconstructed[target_col].values[:min(len_rec, len_exi)] - df_existing[target_col].values[:min(len_rec, len_exi)])
    max_diff = np.max(diff)
    
    if max_diff < 1e-6:
        print(f"✅ 数值完全对齐 (最大误差: {max_diff:.2e})")
    else:
        print(f"❌ 数值未对齐！最大误差: {max_diff:.2e}")
        # 找到第一个不一致的地方
        first_mismatch = np.where(diff > 1e-6)[0][0]
        print(f"   第一次不一致发生在第 {first_mismatch} 行")

if __name__ == "__main__":
    verify_test_alignment()

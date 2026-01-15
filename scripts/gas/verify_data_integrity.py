#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证可视化中的压力数据是否与原始CSV一致
"""

import pandas as pd
import numpy as np

def verify_data_integrity(csv_file, target_col='CHX00L006PT0101'):
    # 加载原始数据
    df_raw = pd.read_csv(csv_file)
    df_raw.columns = ['date'] + list(df_raw.columns)[1:]
    df_clean = df_raw.dropna(subset=['date', target_col])
    
    print(f"文件: {csv_file}")
    print(f"数据点数: {len(df_clean)}")
    print(f"\n{target_col} 统计信息:")
    print(f"  最小值: {df_clean[target_col].min():.6f}")
    print(f"  最大值: {df_clean[target_col].max():.6f}")
    print(f"  平均值: {df_clean[target_col].mean():.6f}")
    print(f"  标准差: {df_clean[target_col].std():.6f}")
    print(f"  变化范围: {df_clean[target_col].max() - df_clean[target_col].min():.6f}")
    
    # 计算相邻点之间的变化
    diff = np.abs(np.diff(df_clean[target_col].values))
    print(f"\n相邻点变化:")
    print(f"  平均变化: {np.mean(diff):.6f}")
    print(f"  最大变化: {np.max(diff):.6f}")
    
    # 显示前10个数据点
    print(f"\n前10个数据点:")
    print(df_clean[[target_col]].head(10))

if __name__ == "__main__":
    import sys
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "data/csv_data/1001.csv"
    verify_data_integrity(csv_file)

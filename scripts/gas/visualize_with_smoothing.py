import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt

# 配置
DATA_DIR = 'data/csv_data'
TARGET_COL = 'CHX00L006PT0101'
ORIGIN_DIR = 'vis_results/origin'
SMOOTHED_DIR = 'vis_results/smoothed'
WINDOW_SIZE = 7  # 稍微调大一点，滤波效果更明显

def smooth_series(series, window=7):
    """使用中值滤波去除突变噪声"""
    # 处理含NaN的情况
    if series.isnull().any():
        series = series.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    
    values = series.values
    # 中值滤波对突变点（Spikes）非常有效
    # 注意：kernel_size 必须是奇数
    smoothed = medfilt(values, kernel_size=window)
    return pd.Series(smoothed, index=series.index)

def plot_file(file_path, output_dir, is_smooth=False):
    filename = os.path.basename(file_path)
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"读取文件 {filename} 失败: {e}")
        return
    
    # 确保有日期列
    if 'date' not in df.columns:
        df.columns = ['date'] + list(df.columns)[1:]
    
    df['date'] = pd.to_datetime(df['date'])
    
    plt.figure(figsize=(15, 8))
    
    # 1. 先画出所有其他维度（淡灰色，作为背景）
    other_cols = [c for c in df.columns if c not in ['date', TARGET_COL]]
    for col in other_cols:
        plt.plot(df['date'], df[col], color='gray', alpha=0.1, linewidth=0.5)
    
    # 2. 突出显示目标压力曲线
    if TARGET_COL in df.columns:
        # 如果是平滑模式，对目标列进行处理
        if is_smooth:
            original = df[TARGET_COL].copy()
            df[TARGET_COL] = smooth_series(df[TARGET_COL], window=WINDOW_SIZE)
            # 在平滑图中，用极淡的红线画出原始数据对比
            plt.plot(df['date'], original, color='red', alpha=0.2, linewidth=0.8, label='Original (Noisy)')
        
        # 蓝线画出目标数据
        plt.plot(df['date'], df[TARGET_COL], color='#0047AB', linewidth=2.5, 
                 label=f'Target: {TARGET_COL}', zorder=10)
    
    plt.title(f'Pressure Analysis - {filename} {"(Smoothed)" if is_smooth else "(Original)"}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # 保存结果
    save_name = filename.replace('.csv', '_vis.png')
    plt.savefig(os.path.join(output_dir, save_name), dpi=150, bbox_inches='tight')
    plt.close()

def main():
    # 创建目录
    os.makedirs(ORIGIN_DIR, exist_ok=True)
    os.makedirs(SMOOTHED_DIR, exist_ok=True)
    
    if not os.path.exists(DATA_DIR):
        print(f"错误: 目录 {DATA_DIR} 不存在")
        return

    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
    print(f"找到 {len(files)} 个文件，开始处理...")
    
    for f in files:
        path = os.path.join(DATA_DIR, f)
        print(f"正在处理: {f}")
        # 1. 原始可视化
        plot_file(path, ORIGIN_DIR, is_smooth=False)
        # 2. 平滑可视化
        plot_file(path, SMOOTHED_DIR, is_smooth=True)
    
    print(f"\n✅ 处理完成！")
    print(f"原始结果在: {ORIGIN_DIR}")
    print(f"平滑结果在: {SMOOTHED_DIR}")

if __name__ == "__main__":
    main()

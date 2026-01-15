#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化单文件独立检测的结果
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


def reconstruct_point_energy(test_energy_windows, total_len, seq_len=256, step=8):
    """将窗口能量重构为点能量（通过重叠平均）"""
    n_win = test_energy_windows.shape[0]
    point_energy = np.zeros(total_len)
    count = np.zeros(total_len)
    
    for i in range(n_win):
        start = i * step
        end = start + seq_len
        if start < total_len:
            actual_end = min(end, total_len)
            point_energy[start:actual_end] += test_energy_windows[i, :actual_end-start]
            count[start:actual_end] += 1
    
    point_energy = point_energy / np.maximum(count, 1)
    return point_energy


def visualize_single_file(result_file, raw_csv, target_col='CHX00L006PT0101', 
                         seq_len=256, step=8, threshold=None, output_dir='vis_results/individual'):
    """可视化单个文件的检测结果"""
    
    # 1. 加载结果
    data = np.load(result_file)
    test_energy_windows = data['test_energy']  # [n_windows, seq_len]
    file_name = data['file_name']
    
    # 2. 加载原始数据
    df_raw = pd.read_csv(raw_csv)
    df_raw.columns = ['date'] + list(df_raw.columns)[1:]
    df_clean = df_raw.dropna(subset=['date', target_col])
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    total_len = len(df_clean)
    
    # 3. 重构点能量
    point_energy = reconstruct_point_energy(test_energy_windows, total_len, seq_len, step)
    
    # 4. 确定阈值
    if threshold is None:
        # 自动阈值：能量的 99% 分位数
        threshold = np.percentile(point_energy, 99)
    
    point_pred = (point_energy > threshold).astype(int)
    anomaly_count = np.sum(point_pred)
    
    print(f"文件: {file_name}")
    print(f"  数据点数: {total_len}")
    print(f"  异常点数: {anomaly_count} ({100*anomaly_count/total_len:.2f}%)")
    print(f"  能量范围: [{np.min(point_energy):.2e}, {np.max(point_energy):.2e}]")
    print(f"  阈值: {threshold:.2e}")
    
    # 5. 绘图
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    
    # 子图1：流量背景
    flow_cols = [c for c in df_clean.columns if 'FT' in c]
    for fc in flow_cols:
        axes[0].plot(df_clean['date'], df_clean[fc], alpha=0.6, linewidth=1.2)
    axes[0].set_ylabel('Flow Rate')
    axes[0].set_title(f'Flow Fields Context - {file_name}')
    
    # 子图2：压力 + 异常标记
    axes[1].plot(df_clean['date'], df_clean[target_col], color='blue', linewidth=1.5, label=target_col)
    axes[1].set_ylabel('Pressure')
    axes[1].legend(loc='upper right')
    axes[1].set_title(f'Pressure Detection - {target_col}')
    
    # 子图3：能量曲线
    axes[2].plot(df_clean['date'], point_energy, color='purple', linewidth=1, label='Reconstruction Energy')
    axes[2].axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold={threshold:.2e}')
    axes[2].set_yscale('log')
    axes[2].set_ylabel('Energy (Log Scale)')
    axes[2].set_xlabel('Date')
    axes[2].legend(loc='upper right')
    axes[2].set_title('Reconstruction Energy (No Boundary Effect)')
    
    # 在所有子图上绘制异常背景
    for ax in axes:
        # 找到连续的异常区间
        diff = np.diff(np.concatenate([[0], point_pred, [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            ax.axvspan(df_clean['date'].iloc[s], df_clean['date'].iloc[min(e, total_len-1)],
                      alpha=0.2, color='red', zorder=0)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{file_name}_clean_vis.png")
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"  ✓ 图片保存至: {output_path}\n")


def main():
    parser = argparse.ArgumentParser(description="可视化单文件异常检测结果")
    parser.add_argument("--result_dir", type=str, default="test_results/individual_file_results",
                       help="单文件检测结果目录")
    parser.add_argument("--raw_dir", type=str, default="data/csv_data",
                       help="原始CSV文件目录")
    parser.add_argument("--target", type=str, default="CHX00L006PT0101")
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=None,
                       help="手动指定阈值，不指定则自动计算")
    parser.add_argument("--output_dir", type=str, default="vis_results/individual_clean")
    parser.add_argument("--max_files", type=int, default=10,
                       help="最多可视化前N个文件（0表示全部）")
    
    args = parser.parse_args()
    
    result_files = sorted(glob.glob(os.path.join(args.result_dir, "*_result.npz")))
    
    if not result_files:
        print(f"❌ 错误: 在 {args.result_dir} 下未找到结果文件")
        print(f"请先运行: python scripts/gas/test_individual_files.py")
        return
    
    print(f"找到 {len(result_files)} 个结果文件\n")
    
    if args.max_files > 0:
        result_files = result_files[:args.max_files]
        print(f"将可视化前 {len(result_files)} 个文件\n")
    
    for result_file in result_files:
        file_name = os.path.splitext(os.path.basename(result_file))[0].replace('_result', '')
        raw_csv = os.path.join(args.raw_dir, f"{file_name}.csv")
        
        if not os.path.exists(raw_csv):
            print(f"⚠️ 跳过 {file_name}: 找不到原始文件")
            continue
        
        visualize_single_file(result_file, raw_csv, args.target, args.seq_len, 
                            args.step, args.threshold, args.output_dir)
    
    print(f"✅ 全部完成！可视化结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 必须在 import pyplot 之前设置，用于无图形界面环境
import matplotlib.pyplot as plt

# 配置
DATA_DIR = 'data/csv_data'
RESULT_DIR = 'test_results/individual_file_results'
OUTPUT_DIR = 'vis_results/multi_dim_v2'

# 标签配置 (维度: [颜色, 中文名]) - 仅包含已训练的流量模型
TAGS = {
    'CHX00F002FT0101': ['green', 'Etoke Flow'],
    'CHX00F003FT0101': ['blue', 'Wushen Flow']
}

def reconstruct_point_energy(test_energy_windows, total_len, seq_len=256, step=8):
    """将窗口能量重构为点能量"""
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
    return point_energy / np.maximum(count, 1)

def plot_multi_diagnosis(file_name):
    print(f"  开始处理文件: {file_name}")
    csv_path = os.path.join(DATA_DIR, f"{file_name}.csv")
    if not os.path.exists(csv_path):
        print(f"  ⚠️ 跳过：找不到CSV文件 {csv_path}")
        return

    # 1. 加载原始数据
    try:
        df = pd.read_csv(csv_path)
        if 'date' not in df.columns: df.columns = ['date'] + list(df.columns)[1:]
        df['date'] = pd.to_datetime(df['date'])
        total_len = len(df)
        print(f"    数据行数: {total_len}")
    except Exception as e:
        print(f"  ⚠️ 读取CSV失败: {e}")
        return

    # 2. 准备绘图 (3个子图)
    fig, axes = plt.subplots(3, 1, figsize=(20, 18), sharex=True)
    
    # --- 子图 1: 32维背景图 ---
    all_cols = [c for c in df.columns if c not in ['date'] and ('FT' in c or 'PT' in c)]
    print(f"    找到 {len(all_cols)} 个维度用于背景绘制")
    
    # 绘制背景线（非目标维度）
    background_count = 0
    for col in all_cols:
        if col not in TAGS:
            axes[0].plot(df['date'], df[col], color='gray', alpha=0.15, linewidth=0.6)
            background_count += 1
    print(f"    绘制了 {background_count} 条背景线")
    
    # --- 循环处理 3 个关键维度 ---
    for tag, (color, name) in TAGS.items():
        if tag not in df.columns: continue
        
        # 加载检测结果
        result_path = os.path.join(RESULT_DIR, f"{file_name}_{tag}_result.npz")
        if not os.path.exists(result_path):
            print(f"  ⚠️ 缺少 {tag} 的结果，仅绘制原始曲线")
            axes[0].plot(df['date'], df[tag], color=color, alpha=0.8, linewidth=1.5, label=name)
            axes[1].plot(df['date'], df[tag], color=color, alpha=0.9, linewidth=1.5, label=name)
            continue
            
        res = np.load(result_path)
        point_energy = reconstruct_point_energy(res['test_energy'], total_len)
        threshold = np.percentile(point_energy, 99.5) # 动态阈值建议
        preds = (point_energy > threshold).astype(int)

        # 绘制子图 1 中的高亮线
        axes[0].plot(df['date'], df[tag], color=color, alpha=0.8, linewidth=1.8, label=name)
        
        # 绘制子图 2 (关键维度对比)
        axes[1].plot(df['date'], df[tag], color=color, alpha=0.9, linewidth=1.8, label=f"{name}")
        
        # 绘制子图 3 (能量对比)
        axes[2].plot(df['date'], point_energy, color=color, alpha=0.7, linewidth=1, label=f"{name} Energy")
        axes[2].axhline(y=threshold, color=color, linestyle='--', alpha=0.5, linewidth=1)

        # 标注异常区域 (不同维度的异常用不同颜色的背景)
        diff = np.diff(np.concatenate([[0], preds, [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            # 在子图 2 和 3 标注背景
            for ax_idx in [1, 2]:
                axes[ax_idx].axvspan(df['date'].iloc[s], df['date'].iloc[min(e, total_len-1)],
                                   alpha=0.15, color=color, label='_nolegend_')

    # 图表装饰
    axes[0].set_title(f'Full Context (32 Dims) - {file_name}', fontsize=14)
    axes[0].legend(loc='upper right')
    
    axes[1].set_title('Target Flow Dimensions & Synchronized Anomalies', fontsize=14)
    axes[1].set_ylabel('Flow Rate (m³/h)')
    axes[1].legend(loc='upper right')
    
    axes[2].set_title('Reconstruction Energy & Thresholds (Log Scale)', fontsize=14)
    axes[2].set_yscale('log')
    axes[2].set_ylabel('Energy')
    axes[2].legend(loc='upper right')

    for ax in axes: ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f"{file_name}_full_diagnosis.png")
    
    try:
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"  ✓ 诊断图已保存: {save_path}")
    except Exception as e:
        print(f"  ⚠️ 保存图片失败: {e}")
        plt.close()

def main():
    print("="*60)
    print("多维度异常诊断可视化")
    print("="*60)
    
    # 自动获取已检测的文件列表
    result_files = glob.glob(os.path.join(RESULT_DIR, "*_result.npz"))
    if not result_files:
        print(f"❌ 错误: 在 {RESULT_DIR} 未找到结果文件。请先运行检测脚本。")
        return
    
    print(f"找到 {len(result_files)} 个检测结果文件")
    
    # 智能提取文件基础名：尝试从 "1001_CHX00F002FT0101_result.npz" 提取 "1001"
    # 如果没有 _CHX，则从 "1001_result.npz" 提取 "1001"
    unique_files = set()
    for f in result_files:
        basename = os.path.basename(f)
        if '_CHX' in basename:
            # 新格式: 1001_CHX00F002FT0101_result.npz
            file_id = basename.split('_CHX')[0]
        else:
            # 旧格式: 1001_result.npz
            file_id = basename.replace('_result.npz', '')
        unique_files.add(file_id)
    
    unique_files = sorted(list(unique_files))
    print(f"对应 {len(unique_files)} 个唯一文件，开始生成深度诊断图...\n")
    
    success_count = 0
    for f_name in unique_files: 
        try:
            plot_multi_diagnosis(f_name)
            success_count += 1
        except Exception as e:
            print(f"  ❌ 处理 {f_name} 时出错: {e}")
    
    print(f"\n{'='*60}")
    print(f"✅ 完成！成功生成 {success_count}/{len(unique_files)} 个诊断图")
    print(f"图片保存位置: {OUTPUT_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

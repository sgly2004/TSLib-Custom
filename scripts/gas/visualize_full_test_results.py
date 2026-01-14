import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.visualize_gas_anomaly import _load_pred_vector, _draw_anomaly_regions

def _reconstruct_point_energy(test_energy, total_len, win_size, step):
    """将窗口级别的能量还原为点级别的能量"""
    if len(test_energy) % win_size == 0:
        n_win = len(test_energy) // win_size
        energy_matrix = test_energy.reshape(n_win, win_size)
        print(f"✅ 检测到完整点序列: {n_win} 个窗口 × {win_size} 点")
    else:
        n_win = len(test_energy)
        energy_matrix = test_energy.reshape(n_win, 1)
        win_size = 1
        print(f"✅ 检测到窗口均值模式: {n_win} 个窗口")

    point_energy = np.zeros(total_len)
    count = np.zeros(total_len)
    expected_n_win = (total_len - (win_size if win_size > 1 else 0)) // step + (1 if win_size > 1 else 0)
    
    for i in range(min(n_win, expected_n_win)):
        start = i * step
        end = start + win_size
        if start < total_len:
            actual_end = min(end, total_len)
            slice_len = actual_end - start
            point_energy[start:actual_end] += energy_matrix[i, :slice_len]
            count[start:actual_end] += 1
            
    point_energy = point_energy / np.maximum(count, 1)
    return point_energy

def visualize_full_test_diagnostic(result_file, raw_dir, target_col='CHX00L006PT0101', seq_len=256, threshold=None, step=8):
    # 1. 严格加载数据
    raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    all_parts = []
    file_info = []
    
    print(f"正在加载 {len(raw_files)} 个文件...")
    for fp in raw_files:
        df_raw = pd.read_csv(fp)
        df_raw.columns = ['date'] + list(df_raw.columns)[1:]
        df_clean = df_raw.dropna(subset=['date', target_col]).copy()
        if not df_clean.empty:
            all_parts.append(df_clean)
            file_info.append({'name': os.path.splitext(os.path.basename(fp))[0], 'len': len(df_clean)})
    
    df_full = pd.concat(all_parts, axis=0, ignore_index=True)
    df_full['date'] = pd.to_datetime(df_full['date'])
    total_len = len(df_full)
    print(f"✅ 总数据长度: {total_len}")

    # 2. 重构能量并判定异常
    data = np.load(result_file)
    test_energy_raw = data['test_energy'] if 'test_energy' in data else data.get('energy')
    point_energy = _reconstruct_point_energy(test_energy_raw, total_len, seq_len, step)
    
    final_threshold = threshold if threshold is not None else data.get('threshold', 0.1)
    point_pred = (point_energy > final_threshold).astype(int)
    
    print(f"--- 检测统计 ---")
    print(f"异常点总数: {np.sum(point_pred)} / {total_len} ({100*np.sum(point_pred)/total_len:.2f}%)")
    print(f"当前阈值: {final_threshold:.2e}")

    # 3. 生成全量诊断图
    result_dir = os.path.dirname(result_file)
    print("生成全量诊断图...")
    fig, axes = plt.subplots(2, 1, figsize=(25, 12), sharex=True)
    
    axes[0].plot(df_full['date'], df_full[target_col], color='blue', linewidth=1, label=target_col)
    _draw_anomaly_regions(axes[0], df_full['date'], point_pred)
    axes[0].legend()
    axes[0].set_title('Full Test Set - Aligned Detection')
    
    axes[1].plot(df_full['date'], point_energy, color='purple', alpha=0.7, label='Point Energy')
    axes[1].axhline(y=final_threshold, color='red', linestyle='--', label=f'Threshold={final_threshold:.2e}')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].set_title('Reconstruction Energy (Log Scale)')
    
    plt.savefig(os.path.join(result_dir, "diagnostic_full_vis.png"), dpi=150)
    plt.close()
    print(f"  ✓ 已保存: diagnostic_full_vis.png")

    # 4. 生成前 10 个文件的子图（确保对齐）
    sub_dir = os.path.join(result_dir, "individual_files")
    os.makedirs(sub_dir, exist_ok=True)
    
    print(f"正在生成前 10 个文件的详细子图...")
    current_idx = 0
    
    for i, info in enumerate(file_info[:10]):
        f_len = info['len']
        f_name = info['name']
        
        # 关键：严格按照索引切分
        f_df = df_full.iloc[current_idx : current_idx + f_len].copy()
        f_pred = point_pred[current_idx : current_idx + f_len].copy()
        f_energy = point_energy[current_idx : current_idx + f_len].copy()
        
        # 诊断信息
        anomaly_count = np.sum(f_pred)
        anomaly_ratio = 100 * anomaly_count / f_len if f_len > 0 else 0
        
        # 绘图
        fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
        
        # 子图1：流量背景
        flow_cols = [c for c in f_df.columns if 'FT' in c]
        for fc in flow_cols:
            axes[0].plot(f_df['date'], f_df[fc], alpha=0.15, linewidth=0.8)
        _draw_anomaly_regions(axes[0], f_df['date'], f_pred)
        axes[0].set_title(f"Flow Fields - File: {f_name}")
        
        # 子图2：压力（高亮异常点）
        axes[1].plot(f_df['date'], f_df[target_col], color='blue', linewidth=1.5, label=target_col)
        if anomaly_count > 0:
            anomaly_mask = (f_pred == 1)
            axes[1].scatter(f_df['date'][anomaly_mask], f_df[target_col][anomaly_mask], 
                           color='red', s=15, label=f'Anomaly ({anomaly_count} pts)', zorder=5)
        _draw_anomaly_regions(axes[1], f_df['date'], f_pred)
        axes[1].legend()
        axes[1].set_title(f"Pressure - {target_col} (Anomaly Ratio: {anomaly_ratio:.2f}%)")
        
        # 子图3：能量曲线
        axes[2].plot(f_df['date'], f_energy, color='purple', linewidth=1, label='Energy')
        axes[2].axhline(y=final_threshold, color='red', linestyle='--', label='Threshold')
        axes[2].set_yscale('log')
        axes[2].legend()
        axes[2].set_title('Reconstruction Energy')
        
        plt.tight_layout()
        plt.savefig(os.path.join(sub_dir, f"{f_name}_detailed_vis.png"), dpi=100)
        plt.close()
        
        print(f"  ✓ {i+1}/10 - {f_name}: {anomaly_count} 异常点 ({anomaly_ratio:.2f}%)")
        current_idx += f_len

    print(f"\n✅ 可视化完成！结果保存在: {result_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", type=str, required=True)
    parser.add_argument("--raw_dir", type=str, default="data/csv_data")
    parser.add_argument("--target", type=str, default="CHX00L006PT0101")
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--step", type=int, default=8)
    args = parser.parse_args()
    visualize_full_test_diagnostic(args.result_file, args.raw_dir, args.target, args.seq_len, args.threshold, args.step)

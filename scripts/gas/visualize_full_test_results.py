import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 将项目根目录添加到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.visualize_gas_anomaly import _load_pred_vector, _expand_window_pred_to_points, _draw_anomaly_regions

def _load_full_dimensional_test_data(raw_dir, target_column):
    raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    all_parts = []
    print(f"正在从 {raw_dir} 加载并拼接全维度数据...")
    for fp in raw_files:
        df = pd.read_csv(fp)
        if df.shape[1] < 2: continue
        cols = list(df.columns)
        df.columns = ['date'] + cols[1:]
        df = df.dropna(subset=['date', target_column])
        if not df.empty:
            all_parts.append(df)
    full_df = pd.concat(all_parts, axis=0, ignore_index=True)
    full_df['date'] = pd.to_datetime(full_df['date'])
    return full_df

def visualize_individual_files(df_full, point_pred, result_dir, target_col):
    save_path_root = os.path.join(result_dir, "individual_files")
    os.makedirs(save_path_root, exist_ok=True)
    print(f"正在生成单个文件的精细可视化图至: {save_path_root}")

    raw_dir = 'data/csv_data'
    raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    
    current_idx = 0
    # 我们只取前10个文件进行可视化，避免生成太多文件
    for i, fp in enumerate(raw_files):
        if i >= 10: 
            print("... 仅可视化前 10 个文件，如需更多请修改脚本 ...")
            break
            
        fname = os.path.splitext(os.path.basename(fp))[0]
        df_raw = pd.read_csv(fp)
        df_raw.columns = ['date'] + list(df_raw.columns)[1:]
        df_raw = df_raw.dropna(subset=['date', target_col])
        if df_raw.empty: continue
            
        file_len = len(df_raw)
        file_df = df_full.iloc[current_idx : current_idx + file_len].copy()
        file_pred = point_pred[current_idx : current_idx + file_len]
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        # 流量子图
        flow_fields = [f for f in file_df.columns if 'FT' in f]
        for f in flow_fields:
            is_target = (f == target_col)
            axes[0].plot(file_df['date'], file_df[f], alpha=1.0 if is_target else 0.3, linewidth=2 if is_target else 0.8)
        axes[0].set_ylabel('Flow')
        _draw_anomaly_regions(axes[0], file_df['date'], file_pred)
        
        # 压力子图
        pressure_fields = [f for f in file_df.columns if 'PT' in f]
        for f in pressure_fields:
            is_target = (f == target_col)
            axes[1].plot(file_df['date'], file_df[f], alpha=1.0 if is_target else 0.3, linewidth=2 if is_target else 0.8, label=f if is_target else None)
        axes[1].set_ylabel('Pressure')
        if any(f == target_col for f in pressure_fields): axes[1].legend(loc='upper right')
        _draw_anomaly_regions(axes[1], file_df['date'], file_pred)
        
        anomaly_ratio = (np.sum(file_pred) / file_len) * 100
        plt.suptitle(f"File: {fname} | Anomaly Ratio: {anomaly_ratio:.2f}%", fontsize=14)
        plt.savefig(os.path.join(save_path_root, f"{fname}_focused.png"), dpi=100, bbox_inches='tight')
        plt.close()
        
        current_idx += file_len
        print(f"  ✓ {fname} 完成")

def visualize_full_test_multi_dim(result_file, raw_dir='data/csv_data', target_col='CHX00L006PT0101', seq_len=128, threshold=None):
    df = _load_full_dimensional_test_data(raw_dir, target_col)
    data = np.load(result_file)
    test_energy = data['test_energy'] if 'test_energy' in data else data.get('energy')
    final_threshold = threshold if threshold is not None else data.get('threshold', 0.1)
    
    pred_win = (test_energy > final_threshold).astype(int)
    total_len = len(df)
    point_pred = _expand_window_pred_to_points(pred_win, total_len, win_size=seq_len)
    
    result_dir = os.path.dirname(result_file)
    
    # 1. 生成全量大图
    print("生成全量趋势大图...")
    fig, axes = plt.subplots(2, 1, figsize=(25, 12), sharex=True)
    for field in [f for f in df.columns if 'FT' in f]:
        axes[0].plot(df['date'], df[field], alpha=0.3, linewidth=0.5)
    _draw_anomaly_regions(axes[0], df['date'], point_pred)
    for field in [f for f in df.columns if 'PT' in f]:
        is_target = (field == target_col)
        axes[1].plot(df['date'], df[field], alpha=1.0 if is_target else 0.3, linewidth=2 if is_target else 0.5, label=field if is_target else None)
    if target_col in df.columns: axes[1].legend(loc='upper right')
    _draw_anomaly_regions(axes[1], df['date'], point_pred)
    plt.savefig(os.path.join(result_dir, "full_test_multi_dim_vis.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. 生成单个文件的子图
    visualize_individual_files(df, point_pred, result_dir, target_col)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", type=str, required=True)
    parser.add_argument("--target", type=str, default="CHX00L006PT0101")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    visualize_full_test_multi_dim(args.result_file, target_col=args.target, seq_len=args.seq_len, threshold=args.threshold)

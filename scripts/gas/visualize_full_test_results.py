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
    """
    按照与数据构造脚本完全一致的逻辑，重新拼接全维度的测试数据
    """
    raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    all_parts = []
    
    print(f"正在从 {raw_dir} 加载并拼接全维度数据...")
    for fp in raw_files:
        df = pd.read_csv(fp)
        if df.shape[1] < 2:
            continue
            
        # 统一日期列名
        cols = list(df.columns)
        df.columns = ['date'] + cols[1:]
        
        # 必须使用与单维度构造时完全一致的 dropna 逻辑，确保行数对齐
        df = df.dropna(subset=['date', target_column])
        
        if not df.empty:
            all_parts.append(df)
            
    full_df = pd.concat(all_parts, axis=0, ignore_index=True)
    full_df['date'] = pd.to_datetime(full_df['date'])
    return full_df

def visualize_full_test_multi_dim(result_file, raw_dir='data/csv_data', target_col='CHX00L006PT0101', seq_len=128, threshold=None):
    # 1. 还原全维度数据
    df = _load_full_dimensional_test_data(raw_dir, target_col)
    
    print(f"正在加载检测结果: {result_file}")
    data = np.load(result_file)
    
    # 获取测试集的重构能量
    test_energy = data['test_energy'] if 'test_energy' in data else data.get('energy')
    if test_energy is None:
        print("❌ 错误: 未找到能量数据")
        return

    # 确定阈值
    final_threshold = threshold if threshold is not None else data.get('threshold', 0.1)
    print(f"--- 能量分布统计 ---")
    print(f"最大能量: {np.max(test_energy):.6f}")
    print(f"平均能量: {np.mean(test_energy):.6f}")
    print(f"当前使用阈值: {final_threshold:.6f}")

    # 2. 对齐预测结果
    pred_win = (test_energy > final_threshold).astype(int)
    total_len = len(df)
    point_pred = _expand_window_pred_to_points(pred_win, total_len, win_size=seq_len)
    
    # 3. 分类测点：流量(FT)和压力(PT)
    tag_columns = [col for col in df.columns if col != 'date']
    flow_fields = [f for f in tag_columns if 'FT' in f]
    pressure_fields = [f for f in tag_columns if 'PT' in f]

    print(f"生成全维度趋势图 (26万行数据，绘图可能需要几秒钟)...")
    fig, axes = plt.subplots(2, 1, figsize=(25, 12), sharex=True)
    
    # 绘制流量子图
    for field in flow_fields:
        is_target = (field == target_col)
        alpha = 1.0 if is_target else 0.3
        linewidth = 2 if is_target else 0.5
        axes[0].plot(df['date'], df[field], label=field if is_target else None, alpha=alpha, linewidth=linewidth)
    axes[0].set_ylabel('Flow (m³/h)')
    axes[0].set_title('Full Test Set Context - Flow Fields')
    axes[0].grid(True, alpha=0.1)
    _draw_anomaly_regions(axes[0], df['date'], point_pred)

    # 绘制压力子图
    for field in pressure_fields:
        is_target = (field == target_col)
        alpha = 1.0 if is_target else 0.3
        linewidth = 2 if is_target else 0.5
        axes[1].plot(df['date'], df[field], label=field if is_target else None, alpha=alpha, linewidth=linewidth)
    axes[1].set_ylabel('Pressure (MPa)')
    axes[1].set_title(f'Full Test Set Context - Pressure Fields (Target: {target_col})')
    axes[1].set_xlabel('Time')
    axes[1].grid(True, alpha=0.1)
    if any(f == target_col for f in pressure_fields):
        axes[1].legend(loc='upper right')
    _draw_anomaly_regions(axes[1], df['date'], point_pred)

    # 整体标题
    anomaly_count = np.sum(point_pred)
    anomaly_ratio = (anomaly_count / total_len) * 100
    res_name = os.path.basename(os.path.dirname(result_file))
    plt.suptitle(f"Full Context Visualization (Single-Dimension Model: {target_col})\nResult: {res_name}\nAnomaly Ratio: {anomaly_ratio:.2f}%", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 保存
    save_path = os.path.join(os.path.dirname(result_file), "full_test_multi_dim_vis.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 全维度可视化已保存至: {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, default="dataset/gas_anomaly_single/test.csv")
    parser.add_argument("--result_file", type=str, required=True)
    parser.add_argument("--target", type=str, default="CHX00L006PT0101")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=None)
    
    args = parser.parse_args()
    visualize_full_test_multi_dim(args.result_file, target_col=args.target, seq_len=args.seq_len, threshold=args.threshold)

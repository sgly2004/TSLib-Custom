import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 将项目根目录添加到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.visualize_gas_anomaly import _load_pred_vector, _expand_window_pred_to_points, _draw_anomaly_regions

def visualize_full_test(test_csv, result_file, target_col='CHX00L006PT0101', seq_len=128, manual_threshold=None):
    print(f"正在加载测试集数据: {test_csv}")
    df = pd.read_csv(test_csv)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"正在加载检测结果: {result_file}")
    data = np.load(result_file)
    
    # 获取测试集的重构能量
    if 'test_energy' in data:
        test_energy = data['test_energy']
    else:
        # 兼容处理
        test_energy = data['energy'] if 'energy' in data else None
        
    if test_energy is None:
        print("❌ 错误: 在结果文件中未找到 energy 数据")
        return

    # 打印能量分布情况，帮你分析
    print(f"--- 能量分布统计 ---")
    print(f"最大重构能量: {np.max(test_energy):.6f}")
    print(f"最小重构能量: {np.min(test_energy):.6f}")
    print(f"平均重构能量: {np.mean(test_energy):.6f}")
    
    # 确定阈值
    if manual_threshold is not None:
        threshold = manual_threshold
        print(f"🔔 使用手动指定的阈值: {threshold}")
    else:
        threshold = data['threshold'] if 'threshold' in data else 0.1
        print(f"使用保存的阈值: {threshold}")

    # 根据阈值计算判定结果
    pred_win = (test_energy > threshold).astype(int)
    
    # 将窗口结果还原到每一个数据点
    total_len = len(df)
    point_pred = _expand_window_pred_to_points(pred_win, total_len, win_size=seq_len)
    
    print("生成全量趋势图...")
    plt.figure(figsize=(25, 10))
    
    # 1. 绘制压力曲线
    plt.plot(df['date'], df[target_col], label=f'Pressure ({target_col})', color='blue', linewidth=1, alpha=0.8)
    
    # 2. 绘制重构能量（作为参考线，画在副坐标轴或下方）
    # 这里为了不干扰主图，我们只画异常背景
    _draw_anomaly_regions(plt.gca(), df['date'], point_pred)
    
    # 统计异常
    anomaly_count = np.sum(point_pred)
    anomaly_ratio = (anomaly_count / total_len) * 100
    
    plt.title(f'Full Test Set Anomaly Detection Result\nThreshold: {threshold:.6f}, Anomaly Points: {anomaly_count} ({anomaly_ratio:.2f}%)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Pressure (MPa)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    result_dir = os.path.dirname(result_file)
    out_name = "full_test_vis_manual.png" if manual_threshold else "full_test_vis_auto.png"
    save_path = os.path.join(result_dir, out_name)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 全量可视化完成！结果已保存至: {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, default="dataset/gas_anomaly_single/test.csv")
    parser.add_argument("--result_file", type=str, required=True)
    parser.add_argument("--target", type=str, default="CHX00L006PT0101")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=None, help="手动指定阈值")
    
    args = parser.parse_args()
    visualize_full_test(args.test_csv, args.result_file, args.target, args.seq_len, args.threshold)

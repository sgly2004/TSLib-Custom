import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt

# 配置
DATA_DIR = 'data/csv_data'
# 定义需要高亮的标签及其对应的颜色
HIGHLIGHT_TAGS = {
    'CHX00L006FT0101': 'red',    # 呼和浩特末站流量 -> 红色
    'CHX00F002FT0101': 'green',  # 鄂托克旗热泵站流量 -> 绿色
    'CHX00F003FT0101': 'blue'    # 乌审旗热泵站流量 -> 蓝色
}
ORIGIN_DIR = 'vis_results/origin'
SMOOTHED_DIR = 'vis_results/smoothed'
WINDOW_SIZE = 7  

def smooth_series(series, window=7):
    """使用中值滤波去除突变噪声"""
    if series.isnull().any():
        series = series.interpolate(method='linear').ffill().bfill()
    values = series.values
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

    # 如果是平滑模式，对所有高亮标签进行处理
    if is_smooth:
        for tag in HIGHLIGHT_TAGS:
            if tag in df.columns:
                df[tag] = smooth_series(df[tag], window=WINDOW_SIZE)

    # 提取流量和压力字段
    flow_cols = [c for c in df.columns if 'FT' in c]
    pressure_cols = [c for c in df.columns if 'PT' in c]

    # 创建 2x1 子图
    fig, (ax_flow, ax_pres) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    # --- 1. 流量子图 (Flow) ---
    has_flow_legend = False
    for col in flow_cols:
        color_target = HIGHLIGHT_TAGS.get(col)
        if color_target:
            ax_flow.plot(df['date'], df[col], color=color_target, alpha=0.9, linewidth=2.0, 
                         label=col, zorder=10)
            has_flow_legend = True
        else:
            ax_flow.plot(df['date'], df[col], color='gray', alpha=0.3, linewidth=0.8, zorder=1)
    
    ax_flow.set_title(f'Flow Analysis (m³/h) - {filename} {"(Smoothed)" if is_smooth else ""}')
    ax_flow.set_ylabel('Flow')
    ax_flow.grid(True, alpha=0.3)
    if has_flow_legend: ax_flow.legend(loc='upper right')

    # --- 2. 压力子图 (Pressure) ---
    has_pres_legend = False
    for col in pressure_cols:
        color_target = HIGHLIGHT_TAGS.get(col)
        if color_target:
            ax_pres.plot(df['date'], df[col], color=color_target, alpha=0.9, linewidth=2.5, 
                         label=col, zorder=10)
            has_pres_legend = True
        else:
            ax_pres.plot(df['date'], df[col], color='gray', alpha=0.3, linewidth=0.8, zorder=1)
    
    ax_pres.set_title(f'Pressure Analysis (MPa) - {filename} {"(Smoothed)" if is_smooth else ""}')
    ax_pres.set_ylabel('Pressure')
    ax_pres.set_xlabel('Time')
    ax_pres.grid(True, alpha=0.3)
    if has_pres_legend: ax_pres.legend(loc='upper right')

    plt.tight_layout()
    save_name = filename.replace('.csv', '_vis.png')
    plt.savefig(os.path.join(output_dir, save_name), dpi=150, bbox_inches='tight')
    plt.close()

def main():
    os.makedirs(ORIGIN_DIR, exist_ok=True)
    os.makedirs(SMOOTHED_DIR, exist_ok=True)
    
    if not os.path.exists(DATA_DIR):
        print(f"错误: 目录 {DATA_DIR} 不存在")
        return

    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
    print(f"开始处理 {len(files)} 个文件...")
    
    for f in files:
        path = os.path.join(DATA_DIR, f)
        print(f"正在绘制: {f}")
        plot_file(path, ORIGIN_DIR, is_smooth=False)
        plot_file(path, SMOOTHED_DIR, is_smooth=True)
    
    print("\n✅ 处理完成！请检查 vis_results 文件夹。")

if __name__ == "__main__":
    main()

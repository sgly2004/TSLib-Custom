import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 配置
DATA_DIR = 'data/csv_data'
RESULT_DIR = 'test_results/individual_file_results'
OUTPUT_DIR = 'vis_results/multi_dim_diagnosis'

# 标签配置 (维度: [颜色, 中文名])
TAGS = {
    'CHX00L006FT0101': ['red', 'Hohhot Flow'],
    'CHX00F002FT0101': ['green', 'Etoke Flow'],
    'CHX00F003FT0101': ['blue', 'Wushen Flow']
}

def plot_diagnosis(file_name):
    csv_path = os.path.join(DATA_DIR, f"{file_name}.csv")
    if not os.path.exists(csv_path): return

    df = pd.read_csv(csv_path)
    if 'date' not in df.columns: df.columns = ['date'] + list(df.columns)[1:]
    df['date'] = pd.to_datetime(df['date'])

    fig, (ax_main, ax_energy) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    
    for tag, (color, name) in TAGS.items():
        if tag not in df.columns: continue
        
        # 1. 绘制流量主图
        ax_main.plot(df['date'], df[tag], color=color, alpha=0.5, linewidth=1, label=name)
        
        # 2. 尝试加载该维度的异常检测结果 (.npz)
        # 注意：这里需要你运行 test_multi_flow_individual.py 后生成的结果
        result_path = os.path.join(RESULT_DIR, f"{file_name}_{tag}_result.npz")
        if os.path.exists(result_path):
            res = np.load(result_path)
            # 简化逻辑：这里假设能量超过阈值则标红，由于是演示，我们暂用 99% 分位数作为演示阈值
            energy = res['test_energy']
            # 将 window 能量映射回 point (取各 window 重叠部分的均值或最大值)
            # 为了可视化清晰，我们直接在图上标注能量突变点
            ax_energy.plot(df['date'][:len(energy)], energy.mean(axis=1), color=color, alpha=0.7, label=f'{name} Energy')

    ax_main.set_title(f'Multi-Dimension Anomaly Diagnosis - {file_name}')
    ax_main.set_ylabel('Flow Rate (m³/h)')
    ax_main.legend(loc='upper right')
    ax_main.grid(True, alpha=0.2)

    ax_energy.set_title('Reconstruction Energy (Anomaly Score)')
    ax_energy.set_ylabel('Energy')
    ax_energy.legend(loc='upper right')
    ax_energy.grid(True, alpha=0.2)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{file_name}_diagnosis.png"), dpi=150)
    plt.close()

def main():
    files = [f.replace('.csv', '') for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    print(f"开始生成多维度诊断图，共 {len(files)} 个文件...")
    for f in sorted(files)[:10]: # 先处理前10个
        plot_diagnosis(f)
    print(f"✅ 完成！结果在: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

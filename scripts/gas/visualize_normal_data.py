import pandas as pd
import matplotlib.pyplot as plt
import os

def visualize_normal_data(file_path):
    print(f"正在读取数据: {file_path}")
    # 使用 GBK 编码读取
    try:
        df = pd.read_csv(file_path, encoding='gbk')
    except:
        df = pd.read_csv(file_path)
    
    # 根据之前的分析，第4列是呼和浩特末站压力检测
    target_col = df.columns[3]
    print(f"识别到目标列: {target_col}")
    
    # 设置绘图风格
    plt.figure(figsize=(15, 7))
    
    # 绘制原始数据
    plt.plot(df.index, df[target_col], label='Original Data', alpha=0.5, color='green')
    
    # 绘制移动平均线 (窗口大小设为 300)
    df['rolling_mean'] = df[target_col].rolling(window=300, center=True).mean()
    plt.plot(df.index, df['rolling_mean'], label='Trend (Rolling Mean 300)', color='red', linewidth=2)
    
    plt.title(f'Normal Data Trend: {target_col} (CHX00L006PT0101)', fontsize=15)
    plt.xlabel('Index', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存结果
    save_path = 'vis_results/gas/normal_data_trend.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"可视化结果已保存至: {save_path}")

if __name__ == "__main__":
    file_path = 'data/normal_data.csv'
    if os.path.exists(file_path):
        visualize_normal_data(file_path)
    else:
        print(f"错误: 找不到文件 {file_path}")

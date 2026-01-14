import pandas as pd
import matplotlib.pyplot as plt
import os

def visualize_train_data(file_path, target_col='CHX00L006PT0101'):
    print(f"正在读取数据: {file_path}")
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # 设置绘图风格
    plt.figure(figsize=(15, 7))
    
    # 绘制原始数据
    plt.plot(df['date'], df[target_col], label='Original Data', alpha=0.5, color='blue')
    
    # 绘制移动平均线以显示更清晰的趋势 (窗口大小设为 300，约 10 分钟的数据，假设 2s 一个点)
    df['rolling_mean'] = df[target_col].rolling(window=300, center=True).mean()
    plt.plot(df['date'], df['rolling_mean'], label='Trend (Rolling Mean 300)', color='red', linewidth=2)
    
    plt.title(f'Training Data Trend: {target_col}', fontsize=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Pressure', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存结果
    save_path = 'vis_results/gas/train_data_trend.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"可视化结果已保存至: {save_path}")

if __name__ == "__main__":
    train_csv = 'dataset/gas_anomaly_single/train.csv'
    if os.path.exists(train_csv):
        visualize_train_data(train_csv)
    else:
        print(f"错误: 找不到文件 {train_csv}")

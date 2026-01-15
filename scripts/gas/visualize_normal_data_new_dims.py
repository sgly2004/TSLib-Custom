import pandas as pd
import matplotlib.pyplot as plt
import os

# 字段列表
columns = [
    'CHX00E005PT0101','CHX00E005PT0102','CHX00L006FT0101','CHX00L006PT0101',
    'CHX00A001FT0102','CHX00A001PT0102','CHX00A001PT0411','CHX00A001PT0412',
    'CHX00E009PT0101','CHX00E009PT0102','CHX00F005FT0101','CHX00F005PT0101',
    'CHX00F005PT0102','CHX00G004FT0101','CHX00G004PT0101','CHX00G004PT0102',
    'CHX00F002FT0101','CHX00F002PT0101','CHX00F002PT0102','CHX00F002PT0409',
    'CHX00F002PT0410','CHX00E017PT0101','CHX00E017PT0102','CHX00E013PT0101',
    'CHX00E013PT0102','CHX00E003PT0101','CHX00E003PT0102','CHX00F003FT0101',
    'CHX00F003PT0101','CHX00F003PT0102','CHX00F003PT0411','CHX00F003PT0412'
]

def main():
    csv_path = 'data/normal_data.csv'
    # 由于原始文件有乱码表头，我们跳过第一行并手动指定列名
    try:
        df = pd.read_csv(csv_path, skiprows=1, names=columns, encoding='gbk')
    except:
        df = pd.read_csv(csv_path, skiprows=1, names=columns, encoding='utf-8-sig')
    
    # 目标字段
    targets = ['CHX00F003FT0101', 'CHX00F002FT0101']
    
    plt.figure(figsize=(15, 8))
    
    for col in targets:
        plt.plot(df[col], label=col, alpha=0.8, linewidth=1)
        
    plt.title('Visualization of Normal Data (Specified Dimensions)')
    plt.xlabel('Sample Index')
    plt.ylabel('Flow Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = 'vis_results/normal_data_flow_dims.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"可视化结果已保存至: {output_path}")

if __name__ == "__main__":
    main()

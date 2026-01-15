import os
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from data_provider.data_loader import GasSegLoader

class Args:
    """模拟 run.py 的参数对象"""
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)

def inference_single_file(model_path, csv_path, target_col, seq_len=128):
    # 1. 准备配置 (必须与训练时一致)
    # 从模型路径中推断一些基本配置
    # 示例路径: checkpoints/gas_single_dimension_quick/anomaly_detection_.../checkpoint.pth
    setting = os.path.basename(os.path.dirname(model_path))
    
    # 解析 setting 字符串获取超参数 (这是一个 hack 方法，但有效)
    # 或者直接手动指定
    config = {
        'task_name': 'anomaly_detection',
        'model': 'TimesNet',
        'data': 'GAS',
        'features': 'S',
        'target': target_col,
        'enc_in': 1,
        'c_out': 1,
        'd_model': 64 if 'dm64' in setting else 128,
        'd_ff': 256 if 'df256' in setting else 512,
        'e_layers': 1 if 'el1' in setting else 2,
        'top_k': 3,
        'num_kernels': 4,
        'seq_len': seq_len,
        'pred_len': 0,
        'anomaly_ratio': 1.0,
        'batch_size': 1,
        'use_gpu': torch.cuda.is_available(),
        'gpu': 0,
        'use_multi_gpu': False,
        'output_attention': False,
        'checkpoints': os.path.dirname(os.path.dirname(model_path)),
        'embed': 'timeF',
        'freq': 's',
        'des': 'inference',
        'patience': 3,
        'learning_rate': 0.0001,
        'lradj': 'type1',
        'use_amp': False
    }
    args = Args(config)
    
    print(f"--- 开始单文件推理 ---")
    print(f"数据文件: {csv_path}")
    print(f"目标维度: {target_col}")
    print(f"加载模型: {model_path}")

    # 2. 加载实验类
    exp = Exp_Anomaly_Detection(args)
    
    # 3. 加载权重
    print("正在加载权重...")
    exp.model.load_state_dict(torch.load(model_path, map_location='cpu'))
    if args.use_gpu:
        exp.model.cuda()
    exp.model.eval()

    # 4. 准备数据 (GasSegLoader 期望目录下有 train.csv 和 test.csv)
    # 我们临时创建一个目录，把 csv_path 软链接为 test.csv
    tmp_dir = "./dataset/tmp_inference"
    os.makedirs(tmp_dir, exist_ok=True)
    
    # 为了让 Scaler 正常工作，我们需要原始的 train.csv
    # 假设它在 dataset/gas_anomaly_single/train.csv
    train_src = "./dataset/gas_anomaly_single/train.csv"
    if not os.path.exists(train_src):
        raise FileNotFoundError(f"找不到训练集用于标准化: {train_src}")
    
    # 读取数据
    df_test = pd.read_csv(csv_path)
    df_train = pd.read_csv(train_src)
    
    # 手动处理标准化 (模拟 GasSegLoader 内部逻辑)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(df_train[[target_col]].values)
    
    test_data = scaler.transform(df_test[[target_col]].values)
    
    # 5. 推理 (滑动窗口重构)
    print("正在执行重构检测...")
    attens_energy = []
    
    # 构造 batch 数据
    test_data_t = torch.from_numpy(test_data).float().unsqueeze(0) # [1, L, 1]
    if args.use_gpu:
        test_data_t = test_data_t.cuda()
    
    # 简单的滑动窗口推理
    # 注意：这里为了简化直接用了简易逻辑，实际应该用 DataLoader 保证与训练完全一致
    with torch.no_grad():
        L = test_data.shape[0]
        # 结果向量，长度与原数据一致
        energy = np.zeros(L)
        count = np.zeros(L)
        
        for i in range(0, L - seq_len + 1):
            batch_x = test_data_t[:, i:i+seq_len, :]
            # TimesNet 推理
            output = exp.model(batch_x, None, None, None)
            
            # 计算 MSE
            loss = torch.mean((output - batch_x) ** 2, dim=-1).cpu().numpy() # [1, seq_len]
            
            energy[i:i+seq_len] += loss[0]
            count[i:i+seq_len] += 1
            
        # 平均能耗
        energy = energy / np.maximum(count, 1)

    # 6. 判定异常 (使用训练集计算的阈值)
    # 这里的阈值计算比较复杂，我们简单使用分位数
    # 实际项目中应该从训练结果 energy_and_pred.npz 中读取 threshold
    # 我们假设阈值已经由之前的训练过程确定
    
    print("推理完成。")
    
    # 保存结果
    out_dir = os.path.join("test_results", "single_inference")
    os.makedirs(out_dir, exist_ok=True)
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
    out_path = os.path.join(out_dir, f"{csv_name}_energy.npz")
    
    # 寻找训练时保存的阈值
    # 这里我们只存能量，让可视化脚本通过阈值判定
    np.savez(out_path, energy=energy, seq_len=seq_len)
    print(f"重构能量已保存至: {out_path}")
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--target", type=str, default="CHX00L006PT0101")
    parser.add_argument("--seq_len", type=int, default=128)
    
    args_p = parser.parse_args()
    inference_single_file(args_p.model_path, args_p.csv_path, args_p.target, args_p.seq_len)

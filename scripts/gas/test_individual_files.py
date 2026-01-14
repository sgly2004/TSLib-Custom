#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对 data/csv_data 中的每个文件单独进行异常检测，避免拼接时的边界效应。
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from data_provider.data_factory import data_provider
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from utils.tools import dotdict


def load_trained_model(checkpoint_path, model_config):
    """加载已训练的模型"""
    exp = Exp_Anomaly_Detection(model_config)
    exp.model.load_state_dict(torch.load(checkpoint_path, map_location=exp.device))
    exp.model.eval()
    return exp


def test_single_file(exp, file_path, target_col, scaler, seq_len=256):
    """
    对单个文件进行异常检测
    
    返回:
        test_energy: 每个窗口的重构误差（shape: [n_windows, seq_len]）
        true_data: 原始数据
        pred_data: 重构数据
    """
    # 1. 加载并预处理单个文件
    df_raw = pd.read_csv(file_path)
    df_raw.columns = ['date'] + list(df_raw.columns)[1:]
    df_clean = df_raw.dropna(subset=['date', target_col])[[target_col]].values
    
    # 标准化（使用训练集的 scaler）
    test_data = scaler.transform(df_clean)
    
    # 2. 滑动窗口切分
    n_points = len(test_data)
    step = 8  # 与训练时保持一致
    windows = []
    for i in range(0, n_points - seq_len + 1, step):
        windows.append(test_data[i:i+seq_len])
    
    if len(windows) == 0:
        return None, None, None
    
    windows = np.array(windows)  # [n_windows, seq_len, 1]
    
    # 3. 模型推理
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(windows)),
        batch_size=128, shuffle=False
    )
    
    test_energy_list = []
    outputs_list = []
    
    with torch.no_grad():
        for batch_x, in test_loader:
            batch_x = batch_x.float().to(exp.device)
            outputs = exp.model(batch_x, None, None, None)
            # 计算每个时间点的 MSE
            score = torch.mean((batch_x - outputs) ** 2, dim=-1)  # [batch, seq_len]
            test_energy_list.append(score.cpu().numpy())
            outputs_list.append(outputs.cpu().numpy())
    
    test_energy = np.concatenate(test_energy_list, axis=0)  # [n_windows, seq_len]
    
    return test_energy, test_data, None


def main():
    # 配置（与训练脚本保持完全一致）
    config = dotdict({
        'task_name': 'anomaly_detection',
        'model': 'TimesNet',
        'data': 'GAS',
        'root_path': './dataset/gas_anomaly_single',
        'features': 'S',
        'target': 'CHX00L006PT0101',
        'seq_len': 256,
        'pred_len': 0,
        'enc_in': 1,
        'c_out': 1,
        'd_model': 128,
        'd_ff': 512,
        'num_kernels': 6,
        'top_k': 5,
        'e_layers': 2,
        'd_layers': 1,
        'factor': 1,
        'des': 'individual_test',
        'itr': 1,
        'batch_size': 128,
        'learning_rate': 0.0001,
        'train_epochs': 20,
        'patience': 5,
        'use_amp': False,
        # 其他必需参数
        'embed': 'timeF',
        'freq': 'h',
        'dropout': 0.1,
        'n_heads': 8,
        'activation': 'gelu',
        'd_conv': 4,
        'output_attention': False,
        'checkpoints': './checkpoints/gas_single_dimension',
        'use_gpu': True,
        'gpu': 0,
        'use_multi_gpu': False,
        'devices': '0',
        # DataLoader 必需参数
        'num_workers': 0,
        'lradj': 'type1',
        'label_len': 48,
        'data_path': 'test.csv',
    })
    
    # 1. 加载训练集的 scaler
    train_data, _ = data_provider(config, 'train')
    scaler = train_data.scaler
    
    # 2. 加载训练好的模型
    model_id = 'gas_single_CHX00L006PT0101'
    setting = f"anomaly_detection_{model_id}_TimesNet_GAS_ftS_sl256_ll48_pl0_dm128_nh8_el2_dl1_df512_expand2_dc4_fc1_ebtimeF_dtTrue_CHX00L006PT0101_pressure_0"
    checkpoint_path = os.path.join(config.checkpoints, setting, 'checkpoint.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 错误: 找不到模型文件 {checkpoint_path}")
        return
    
    print(f"正在加载模型: {checkpoint_path}")
    exp = load_trained_model(checkpoint_path, config)
    print("✅ 模型加载成功")
    
    # 3. 对每个文件单独检测
    test_dir = 'data/csv_data'
    output_dir = 'test_results/individual_file_results'
    os.makedirs(output_dir, exist_ok=True)
    
    test_files = sorted(glob.glob(os.path.join(test_dir, '*.csv')))
    print(f"\n开始对 {len(test_files)} 个文件进行独立检测...\n")
    
    for i, file_path in enumerate(test_files, 1):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"[{i}/{len(test_files)}] 正在检测: {file_name}")
        
        test_energy, true_data, _ = test_single_file(
            exp, file_path, config.target, scaler, config.seq_len
        )
        
        if test_energy is None:
            print(f"  ⚠️ 跳过（数据不足）")
            continue
        
        # 保存结果
        output_path = os.path.join(output_dir, f"{file_name}_result.npz")
        np.savez(
            output_path,
            test_energy=test_energy,
            true_data=true_data,
            file_name=file_name
        )
        print(f"  ✓ 保存结果到: {output_path}")
    
    print(f"\n✅ 全部完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    main()

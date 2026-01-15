import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from data_provider.data_factory import data_provider
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from utils.tools import dotdict

# --- 配置区 ---
# 目标维度及其对应的模型路径（请确保训练完后导回的模型文件夹名称与此对应）
TARGET_CONFIGS = {
    'CHX00L006PT0101': {
        'model_id': 'gas_single_CHX00L006PT0101',
        'checkpoint_dir': 'anomaly_detection_gas_single_CHX00L006PT0101_TimesNet_GAS_ftS_sl256_ll48_pl0_dm128_nh8_el2_dl1_df512_expand2_dc4_fc1_ebtimeF_dtTrue_CHX00L006PT0101_pressure_0',
        'root_path': './dataset/gas_anomaly_single'
    },
    'CHX00F002FT0101': {
        'model_id': 'gas_single_F002',
        'checkpoint_dir': 'anomaly_detection_gas_single_F002_TimesNet_GAS_ftS_sl256_ll48_pl0_dm128_nh8_el2_dl1_df512_expand2_dc4_fc1_ebtimeF_dtTrue_CHX00F002FT0101_0',
        'root_path': './dataset/gas_multi_dim/CHX00F002FT0101'
    },
    'CHX00F003FT0101': {
        'model_id': 'gas_single_F003',
        'checkpoint_dir': 'anomaly_detection_gas_single_F003_TimesNet_GAS_ftS_sl256_ll48_pl0_dm128_nh8_el2_dl1_df512_expand2_dc4_fc1_ebtimeF_dtTrue_CHX00F003FT0101_0',
        'root_path': './dataset/gas_multi_dim/CHX00F003FT0101'
    }
}

# 公共模型参数（需与训练时保持一致）
BASE_CONFIG = {
    'task_name': 'anomaly_detection',
    'model': 'TimesNet',
    'data': 'GAS',
    'features': 'S',
    'seq_len': 256,
    'label_len': 48,
    'pred_len': 0,
    'enc_in': 1,
    'c_out': 1,
    'd_model': 128,
    'd_ff': 512,
    'num_kernels': 4,
    'top_k': 3,
    'e_layers': 2,
    'd_layers': 1,
    'factor': 1,
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
    'num_workers': 0,
    'pin_memory': False,
    'drop_last': False,
    'batch_size': 128
}

def load_exp_for_tag(tag):
    cfg_tag = TARGET_CONFIGS[tag]
    config = dotdict(BASE_CONFIG.copy())
    config.target = tag
    config.model_id = cfg_tag['model_id']
    config.root_path = cfg_tag['root_path']
    config.data_path = 'train.csv' # 用于加载 scaler
    
    # 1. 加载 scaler
    train_data, _ = data_provider(config, 'train')
    scaler = train_data.scaler
    
    # 2. 加载模型
    exp = Exp_Anomaly_Detection(config)
    cp_path = os.path.join(config.checkpoints, cfg_tag['checkpoint_dir'], 'checkpoint.pth')
    if not os.path.exists(cp_path):
        print(f"⚠️ 警告: 找不到 {tag} 的模型文件: {cp_path}")
        return None, None
    
    exp.model.load_state_dict(torch.load(cp_path, map_location=exp.device))
    exp.model.eval()
    print(f"✅ 成功加载 {tag} 的模型和 Scaler")
    return exp, scaler

def test_file_for_tag(exp, scaler, file_path, tag):
    df_raw = pd.read_csv(file_path)
    df_raw.columns = ['date'] + list(df_raw.columns)[1:]
    if tag not in df_raw.columns: return None
    
    data = df_raw[[tag]].dropna().values
    if len(data) < BASE_CONFIG['seq_len']: return None
    
    # 标准化
    data_scaled = scaler.transform(data)
    
    # 切窗
    windows = []
    step = 8
    for i in range(0, len(data_scaled) - BASE_CONFIG['seq_len'] + 1, step):
        windows.append(data_scaled[i:i + BASE_CONFIG['seq_len']])
    
    windows = np.array(windows)
    loader = DataLoader(TensorDataset(torch.FloatTensor(windows)), batch_size=64, shuffle=False)
    
    energy_list = []
    with torch.no_grad():
        for batch_x, in loader:
            batch_x = batch_x.to(exp.device)
            outputs = exp.model(batch_x, None, None, None)
            score = torch.mean((batch_x - outputs) ** 2, dim=-1) # [batch, seq_len]
            energy_list.append(score.cpu().numpy())
    
    return np.concatenate(energy_list, axis=0)

def main():
    exps = {}
    scalers = {}
    
    # 1. 预加载所有模型
    for tag in TARGET_CONFIGS:
        exp, scaler = load_exp_for_tag(tag)
        if exp:
            exps[tag] = exp
            scalers[tag] = scaler

    # 2. 遍历文件检测
    csv_files = sorted(glob.glob('data/csv_data/*.csv'))
    output_dir = 'test_results/individual_file_results'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n开始检测 {len(csv_files)} 个文件...")
    for i, f_path in enumerate(csv_files, 1):
        f_name = os.path.basename(f_path).replace('.csv', '')
        print(f"[{i}/{len(csv_files)}] 正在处理: {f_name}")
        
        for tag, exp in exps.items():
            energy = test_file_for_tag(exp, scalers[tag], f_path, tag)
            if energy is not None:
                out_path = os.path.join(output_dir, f"{f_name}_{tag}_result.npz")
                np.savez(out_path, test_energy=energy, tag=tag)
    
    print(f"\n✅ 检测完成！结果已保存在: {output_dir}")

if __name__ == "__main__":
    main()

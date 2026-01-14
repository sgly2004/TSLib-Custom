#!/usr/bin/env python3
"""
提取单维度数据（CHX00L006PT0101 - 呼和浩特末站压力）用于异常检测
从 data/normal 目录提取正常数据，从 data/csv_data 目录提取测试数据
"""

import os
import glob
import pandas as pd
import numpy as np


def _load_and_clean_csv(path: str, target_column: str, min_len: int = 1) -> pd.DataFrame:
    """
    读取单个 CSV，提取目标列，按日期排序并去除缺失。
    
    Args:
        path: CSV 文件路径
        target_column: 要提取的目标列名（如 'CHX00L006PT0101'）
        min_len: 最小长度要求
        
    Returns:
        包含 date 和 target_column 两列的 DataFrame
    """
    try:
        df = pd.read_csv(path)
        if df.shape[1] < 2:
            return pd.DataFrame()

        # 检查目标列是否存在
        if target_column not in df.columns:
            print(f"Warning: {target_column} not found in {path}")
            return pd.DataFrame()

        # 统一日期列名
        cols = list(df.columns)
        first_col = cols[0]
        
        # 只保留日期和目标列
        df = df[[first_col, target_column]].copy()
        df.columns = ['date', target_column]

        # 丢弃完全空行，按日期排序
        df = df.dropna(how='all')
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            df = df.sort_values('date')

        # 去除缺失行
        df = df.dropna()

        if len(df) < min_len:
            return pd.DataFrame()

        return df
        
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return pd.DataFrame()


def build_single_dimension_dataset(
    project_root: str = ".",
    normal_dir: str = "data/normal",
    raw_dir: str = "data/csv_data",
    out_dir: str = "dataset/gas_anomaly_single",
    target_column: str = "CHX00L006PT0101",
    min_normal_len: int = 150,
) -> None:
    """
    从 normal 目录与原始 csv_data 目录构建单维度异常检测所需的 train.csv 与 test.csv。
    
    Args:
        project_root: 项目根目录
        normal_dir: 正常数据目录
        raw_dir: 原始测试数据目录
        out_dir: 输出目录
        target_column: 要提取的目标列名
        min_normal_len: 正常片段最小长度
    """
    normal_path = os.path.join(project_root, normal_dir)
    raw_path = os.path.join(project_root, raw_dir)
    out_path = os.path.join(project_root, out_dir)
    os.makedirs(out_path, exist_ok=True)

    print(f"目标维度: {target_column} (呼和浩特末站压力)")
    print(f"正常数据目录: {normal_path}")
    print(f"测试数据目录: {raw_path}")
    print(f"输出目录: {out_path}")
    print("-" * 80)

    # 1) 构建训练集：只使用 normal 片段
    train_parts = []
    normal_files = sorted(glob.glob(os.path.join(normal_path, "*.csv")))
    
    print(f"\n正在处理 {len(normal_files)} 个正常数据文件...")
    valid_count = 0
    total_length = 0
    
    for fp in normal_files:
        df = _load_and_clean_csv(fp, target_column, min_len=min_normal_len)
        if not df.empty:
            train_parts.append(df)
            valid_count += 1
            total_length += len(df)
            print(f"✓ {os.path.basename(fp)}: {len(df)} 条记录")
        else:
            print(f"✗ {os.path.basename(fp)}: 跳过（长度不足或数据缺失）")

    if train_parts:
        train_df = pd.concat(train_parts, axis=0, ignore_index=True)
        train_out = os.path.join(out_path, "train.csv")
        train_df.to_csv(train_out, index=False)
        print(f"\n✅ 训练数据已保存: {train_out}")
        print(f"   - 有效文件数: {valid_count}/{len(normal_files)}")
        print(f"   - 总记录数: {len(train_df)}")
        print(f"   - 数据列: {list(train_df.columns)}")
    else:
        print("\n❌ 没有找到有效的正常数据片段（可能都不满足最小长度要求）")

    # 2) 构建测试集：使用完整样本 csv_data
    test_parts = []
    raw_files = sorted(glob.glob(os.path.join(raw_path, "*.csv")))
    
    print(f"\n正在处理 {len(raw_files)} 个测试数据文件...")
    valid_test_count = 0
    
    for fp in raw_files:
        df = _load_and_clean_csv(fp, target_column, min_len=1)
        if not df.empty:
            test_parts.append(df)
            valid_test_count += 1
            print(f"✓ {os.path.basename(fp)}: {len(df)} 条记录")
        else:
            print(f"✗ {os.path.basename(fp)}: 跳过（数据缺失）")

    if test_parts:
        test_df = pd.concat(test_parts, axis=0, ignore_index=True)
        test_out = os.path.join(out_path, "test.csv")
        test_df.to_csv(test_out, index=False)
        print(f"\n✅ 测试数据已保存: {test_out}")
        print(f"   - 有效文件数: {valid_test_count}/{len(raw_files)}")
        print(f"   - 总记录数: {len(test_df)}")
        print(f"   - 数据列: {list(test_df.columns)}")
    else:
        print("\n❌ 没有找到有效的测试数据文件")

    # 3) 数据统计
    if train_parts and test_parts:
        print("\n" + "=" * 80)
        print("数据统计摘要")
        print("=" * 80)
        print(f"训练集统计:")
        print(f"  - 数据点数: {len(train_df)}")
        print(f"  - {target_column} 范围: [{train_df[target_column].min():.4f}, {train_df[target_column].max():.4f}]")
        print(f"  - {target_column} 均值: {train_df[target_column].mean():.4f}")
        print(f"  - {target_column} 标准差: {train_df[target_column].std():.4f}")
        
        print(f"\n测试集统计:")
        print(f"  - 数据点数: {len(test_df)}")
        print(f"  - {target_column} 范围: [{test_df[target_column].min():.4f}, {test_df[target_column].max():.4f}]")
        print(f"  - {target_column} 均值: {test_df[target_column].mean():.4f}")
        print(f"  - {target_column} 标准差: {test_df[target_column].std():.4f}")
        print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='构建单维度天然气异常检测数据集')
    parser.add_argument('--project_root', type=str, default='.',
                        help='项目根目录（默认为当前目录）')
    parser.add_argument('--normal_dir', type=str, default='data/normal',
                        help='正常数据目录（相对于项目根目录）')
    parser.add_argument('--raw_dir', type=str, default='data/csv_data',
                        help='原始测试数据目录（相对于项目根目录）')
    parser.add_argument('--out_dir', type=str, default='dataset/gas_anomaly_single',
                        help='输出目录（相对于项目根目录）')
    parser.add_argument('--target_column', type=str, default='CHX00L006PT0101',
                        help='要提取的目标列名')
    parser.add_argument('--min_normal_len', type=int, default=150,
                        help='正常片段最小长度')
    
    args = parser.parse_args()
    
    build_single_dimension_dataset(
        project_root=args.project_root,
        normal_dir=args.normal_dir,
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        target_column=args.target_column,
        min_normal_len=args.min_normal_len,
    )

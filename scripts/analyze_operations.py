#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
管道操作工况数据特征分析脚本
分析不同操作类型的数据特征，观察节点及上下游节点的数据变化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 字段映射
FIELD_MAPPING = {
    'CHX00E005PT0101': '5#阀室进站压力',
    'CHX00E005PT0102': '5#阀室出站压力',
    'CHX00L006FT0101': '呼和浩特末站流量',
    'CHX00L006PT0101': '呼和浩特末站压力',
    'CHX00A001FT0102': '油房庄首站流量',
    'CHX00A001PT0102': '油房庄首站压力',
    'CHX00A001PT0411': '油房庄首站泵入口压力',
    'CHX00A001PT0412': '油房庄首站泵出口压力',
    'CHX00E009PT0101': '9#阀室进站压力',
    'CHX00E009PT0102': '9#阀室出站压力',
    'CHX00F005FT0101': '土默特右旗热泵站流量',
    'CHX00F005PT0101': '土默特右旗热泵站进站压力',
    'CHX00F005PT0102': '土默特右旗热泵站出站压力',
    'CHX00G004FT0101': '达拉特旗热站流量',
    'CHX00G004PT0101': '达拉特旗热站进站压力',
    'CHX00G004PT0102': '达拉特旗热站出站压力',
    'CHX00F002FT0101': '鄂托克旗热泵站流量',
    'CHX00F002PT0101': '鄂托克旗热泵站进站压力',
    'CHX00F002PT0102': '鄂托克旗热泵站出站压力',
    'CHX00F002PT0409': '鄂托克旗热泵站泵入口压力',
    'CHX00F002PT0410': '鄂托克旗热泵站泵出口压力',
    'CHX00E017PT0101': '17#阀室进站压力',
    'CHX00E017PT0102': '17#阀室出站压力',
    'CHX00E013PT0101': '13#阀室进站压力',
    'CHX00E013PT0102': '13#阀室出站压力',
    'CHX00E003PT0101': '3#阀室进站压力',
    'CHX00E003PT0102': '3#阀室出站压力',
    'CHX00F003FT0101': '乌审旗热泵站流量',
    'CHX00F003PT0101': '乌审旗热泵站进站压力',
    'CHX00F003PT0102': '乌审旗热泵站出站压力',
    'CHX00F003PT0411': '乌审旗热泵站泵入口压力',
    'CHX00F003PT0412': '乌审旗热泵站站出口压力',
}

# 节点分组（按管道流向）
NODE_GROUPS = {
    '油房庄': ['CHX00A001FT0102', 'CHX00A001PT0102', 'CHX00A001PT0411', 'CHX00A001PT0412'],
    '3#阀室': ['CHX00E003PT0101', 'CHX00E003PT0102'],
    '乌审旗': ['CHX00F003FT0101', 'CHX00F003PT0101', 'CHX00F003PT0102', 'CHX00F003PT0411', 'CHX00F003PT0412'],
    '5#阀室': ['CHX00E005PT0101', 'CHX00E005PT0102'],
    '9#阀室': ['CHX00E009PT0101', 'CHX00E009PT0102'],
    '达拉特': ['CHX00G004FT0101', 'CHX00G004PT0101', 'CHX00G004PT0102'],
    '13#阀室': ['CHX00E013PT0101', 'CHX00E013PT0102'],
    '鄂托克': ['CHX00F002FT0101', 'CHX00F002PT0101', 'CHX00F002PT0102', 'CHX00F002PT0409', 'CHX00F002PT0410'],
    '17#阀室': ['CHX00E017PT0101', 'CHX00E017PT0102'],
    '土默特': ['CHX00F005FT0101', 'CHX00F005PT0101', 'CHX00F005PT0102'],
    '呼和浩特': ['CHX00L006FT0101', 'CHX00L006PT0101'],
}

# 操作类型
OPERATION_TYPES = {
    1: '增量',
    2: '甩泵',
    3: '切泵',
    4: '启停泵',
    5: '降量',
    6: '紧急启停输',
    7: '计划启停输',
    8: '下载燃料油',
}

class OperationAnalyzer:
    def __init__(self, data_dir='csv_data', metadata_file='case_metadata.csv'):
        self.data_dir = Path(data_dir)
        self.metadata_file = Path(metadata_file)
        self.metadata = None
        self.load_metadata()

    def load_metadata(self):
        """加载元数据"""
        self.metadata = pd.read_csv(self.metadata_file, encoding='utf-8-sig')
        print(f"加载了 {len(self.metadata)} 条案例元数据")
        print(f"操作类型分布:\n{self.metadata['操作'].value_counts()}")

    def load_case_data(self, data_id):
        """加载单个案例数据"""
        file_path = self.data_dir / f"{data_id}.csv"
        if not file_path.exists():
            print(f"警告: 文件 {file_path} 不存在")
            return None

        # 读取数据，跳过可能存在的BOM
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        df['日期'] = pd.to_datetime(df['日期'])
        return df

    def calculate_statistics(self, df, fields):
        """计算指定字段的统计特征"""
        stats = {}
        for field in fields:
            if field not in df.columns:
                continue

            series = df[field].dropna()
            if len(series) < 2:
                continue

            # 基本统计
            stats[field] = {
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'range': series.max() - series.min(),
                # 变化率统计
                'diff_mean': series.diff().mean(),
                'diff_std': series.diff().std(),
                'diff_max': series.diff().abs().max(),
                # 趋势
                'trend': 'up' if series.iloc[-1] > series.iloc[0] else 'down' if series.iloc[-1] < series.iloc[0] else 'stable',
                'change': series.iloc[-1] - series.iloc[0],
                'change_pct': (series.iloc[-1] - series.iloc[0]) / series.iloc[0] * 100 if series.iloc[0] != 0 else 0,
            }

        return stats

    def analyze_operation(self, operation_code, max_samples=5):
        """分析特定操作类型"""
        print(f"\n{'='*80}")
        print(f"分析操作: {OPERATION_TYPES[operation_code]}")
        print(f"{'='*80}\n")

        # 筛选该操作的案例
        cases = self.metadata[self.metadata['操作编码'] == operation_code]
        print(f"找到 {len(cases)} 个案例")

        # 按节点分组
        for node_name in cases['节点'].unique():
            node_cases = cases[cases['节点'] == node_name]
            print(f"\n{'-'*60}")
            print(f"节点: {node_name}, 案例数: {len(node_cases)}")
            print(f"{'-'*60}")

            # 分析前几个案例
            for idx, (_, case) in enumerate(node_cases.head(max_samples).iterrows()):
                data_id = case['data_id']
                print(f"\n案例 {idx+1}: ID={data_id}")

                df = self.load_case_data(data_id)
                if df is None:
                    continue

                # 分析目标节点
                print(f"\n  目标节点: {node_name}")
                if node_name in NODE_GROUPS:
                    node_fields = NODE_GROUPS[node_name]
                    stats = self.calculate_statistics(df, node_fields)
                    self.print_field_changes(stats, node_fields[:3])  # 只打印前3个字段

                # 分析上游和下游节点
                self.analyze_adjacent_nodes(df, node_name)

    def analyze_adjacent_nodes(self, df, target_node):
        """分析上下游节点"""
        # 管道流向顺序
        flow_sequence = ['油房庄', '3#阀室', '乌审旗', '5#阀室', '9#阀室', '达拉特',
                        '13#阀室', '鄂托克', '17#阀室', '土默特', '呼和浩特']

        if target_node not in flow_sequence:
            return

        target_idx = flow_sequence.index(target_node)

        # 上游节点
        if target_idx > 0:
            upstream_node = flow_sequence[target_idx - 1]
            print(f"\n  上游节点: {upstream_node}")
            if upstream_node in NODE_GROUPS:
                stats = self.calculate_statistics(df, NODE_GROUPS[upstream_node])
                self.print_field_changes(stats, NODE_GROUPS[upstream_node][:2])

        # 下游节点
        if target_idx < len(flow_sequence) - 1:
            downstream_node = flow_sequence[target_idx + 1]
            print(f"\n  下游节点: {downstream_node}")
            if downstream_node in NODE_GROUPS:
                stats = self.calculate_statistics(df, NODE_GROUPS[downstream_node])
                self.print_field_changes(stats, NODE_GROUPS[downstream_node][:2])

    def print_field_changes(self, stats, fields):
        """打印字段变化"""
        for field in fields:
            if field not in stats:
                continue

            s = stats[field]
            field_name = FIELD_MAPPING.get(field, field)

            # 判断是否有显著变化
            significant = abs(s['change_pct']) > 1.0  # 变化超过1%认为显著
            marker = "***" if significant else ""

            print(f"    {field_name}:")
            print(f"      初始值: {s['min']:.3f}, 最终值: {s['max']:.3f}")
            print(f"      变化: {s['change']:+.3f} ({s['change_pct']:+.2f}%) {marker}")
            print(f"      趋势: {s['trend']}, 波动std: {s['diff_std']:.4f}")

    def plot_case(self, data_id, save_dir='plots'):
        """绘制单个案例的时序图"""
        # 获取案例信息
        case_info = self.metadata[self.metadata['data_id'] == data_id].iloc[0]
        node = case_info['节点']
        operation = case_info['操作']

        # 加载数据
        df = self.load_case_data(data_id)
        if df is None:
            return

        # 创建保存目录
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        # 绘制主要字段
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(f'案例 {data_id}: {node} - {operation}', fontsize=16, fontweight='bold')

        # 流量
        flow_fields = [f for f in df.columns if 'FT' in f]
        if flow_fields:
            for field in flow_fields:
                axes[0].plot(df['日期'], df[field], label=FIELD_MAPPING.get(field, field), linewidth=1.5)
            axes[0].set_ylabel('流量 (m³/h)', fontsize=12)
            axes[0].legend(loc='best', fontsize=9)
            axes[0].grid(True, alpha=0.3)

        # 泵站压力
        pump_pressure_fields = [f for f in df.columns if ('PT0411' in f or 'PT0412' in f or 'PT0409' in f or 'PT0410' in f)]
        if pump_pressure_fields:
            for field in pump_pressure_fields:
                axes[1].plot(df['日期'], df[field], label=FIELD_MAPPING.get(field, field), linewidth=1.5)
            axes[1].set_ylabel('泵站压力 (MPa)', fontsize=12)
            axes[1].legend(loc='best', fontsize=9)
            axes[1].grid(True, alpha=0.3)

        # 管线压力
        line_pressure_fields = [f for f in df.columns if 'PT' in f and f not in pump_pressure_fields]
        if line_pressure_fields:
            for field in line_pressure_fields[:8]:  # 只画前8个，避免太拥挤
                axes[2].plot(df['日期'], df[field], label=FIELD_MAPPING.get(field, field), linewidth=1, alpha=0.7)
            axes[2].set_ylabel('管线压力 (MPa)', fontsize=12)
            axes[2].set_xlabel('时间', fontsize=12)
            axes[2].legend(loc='best', fontsize=8, ncol=2)
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图片
        output_file = save_path / f"case_{data_id}_{node}_{operation}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {output_file}")
        plt.close()

    def analyze_all_operations(self):
        """分析所有操作类型"""
        for op_code in sorted(OPERATION_TYPES.keys()):
            self.analyze_operation(op_code, max_samples=3)

    def plot_representative_cases(self):
        """绘制每种操作的代表性案例"""
        for op_code in sorted(OPERATION_TYPES.keys()):
            cases = self.metadata[self.metadata['操作编码'] == op_code]
            if len(cases) > 0:
                # 每种操作绘制前3个案例
                for data_id in cases['data_id'].head(3):
                    self.plot_case(data_id)


if __name__ == '__main__':
    # 创建分析器
    analyzer = OperationAnalyzer()

    # 分析所有操作
    analyzer.analyze_all_operations()

    # 绘制代表性案例
    print("\n\n开始绘制案例图表...")
    analyzer.plot_representative_cases()

    print("\n分析完成!")

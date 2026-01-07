#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Git 提交脚本
"""
import subprocess
import os
import sys

def run_command(cmd, cwd=None):
    """执行命令"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            cwd=cwd, 
            capture_output=True, 
            text=True,
            check=False
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(result.stderr, file=sys.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error executing command: {e}", file=sys.stderr)
        return False

def main():
    # 切换到 TSLib-Custom 目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"Working directory: {os.getcwd()}")
    
    # 初始化 git（如果还没有）
    if not os.path.exists('.git'):
        print("Initializing git repository...")
        run_command('git init')
    
    # 创建并切换到新分支
    print("\nCreating/checking out branch: feature/anomaly-detection-boundary-step")
    run_command('git checkout -b feature/anomaly-detection-boundary-step 2>/dev/null || git checkout feature/anomaly-detection-boundary-step')
    
    # 第一次提交：增加 step 参数传递
    print("\n=== First commit: Adding step parameter ===")
    run_command('git add data_provider/data_factory.py')
    
    commit_msg_1 = """feat: 增加 step 参数传递，支持控制滑窗采样步长

功能说明：
- 在异常检测任务中，增加 step 参数的传递
- step 参数用于控制滑窗采样的步长，可以有效地减少样本数量
- 通过 getattr(args, 'step', 1) 获取参数，默认值为 1
- 修改位置：data_provider/data_factory.py 的 anomaly_detection 分支

提示词：
目前 timesnet 在处理异常检测任务时，会尝试用滑窗的方式对于数据采样，但是目前数据集样本点过多，可以通过修改步长来减少采样数量。需要确认修改步长，是否能够带来采样的减少，即步长这个参数是否传递了下去。"""
    
    run_command(f'git commit -m {repr(commit_msg_1)}')
    
    # 第二次提交：增加边界记录功能
    print("\n=== Second commit: Adding boundary recording ===")
    run_command('git add data_provider/data_loader.py data_provider/data_factory.py')
    
    commit_msg_2 = """feat: 增加边界记录功能，避免跨边界采样

功能说明：
- 创建 CustomAnomalySegLoader 类，支持记录 segment 边界
- 避免滑窗采样时产生跨边界的窗口（断层窗口）
- 支持从 dataset_info.txt 自动读取 segment_length，或通过参数传递
- 在 data_factory.py 中添加对自定义数据集的支持
- 修改位置：
  - data_provider/data_loader.py: 新增 CustomAnomalySegLoader 类
  - data_provider/data_factory.py: 添加 CustomAnomalySegLoader 导入和特殊处理

提示词：
目前数据集样本点过多，另外，拼接的连接部分通过滑窗采样，就会是一个断层的采样，所以如何规避断层窗口？需要实现记录边界，避免跨边界采样。"""
    
    run_command(f'git commit -m {repr(commit_msg_2)}')
    
    # 显示提交历史
    print("\n=== Commit history ===")
    run_command('git log --oneline -2')
    
    print("\n=== Current branch ===")
    run_command('git branch --show-current')
    
    print("\nGit commits completed!")

if __name__ == '__main__':
    main()


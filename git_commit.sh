#!/bin/bash
# Git 提交脚本

cd "$(dirname "$0")"

# 初始化 git（如果还没有）
if [ ! -d .git ]; then
    git init
fi

# 创建并切换到新分支
git checkout -b feature/anomaly-detection-boundary-step 2>/dev/null || git checkout feature/anomaly-detection-boundary-step

# 第一次提交：增加 step 参数传递
git add data_provider/data_factory.py
git commit -m "feat: 增加 step 参数传递，支持控制滑窗采样步长

功能说明：
- 在异常检测任务中，增加 step 参数的传递
- step 参数用于控制滑窗采样的步长，可以有效地减少样本数量
- 通过 getattr(args, 'step', 1) 获取参数，默认值为 1
- 修改位置：data_provider/data_factory.py 的 anomaly_detection 分支

提示词：
目前 timesnet 在处理异常检测任务时，会尝试用滑窗的方式对于数据采样，但是目前数据集样本点过多，可以通过修改步长来减少采样数量。需要确认修改步长，是否能够带来采样的减少，即步长这个参数是否传递了下去。"

# 第二次提交：增加边界记录功能
git add data_provider/data_loader.py data_provider/data_factory.py
git commit -m "feat: 增加边界记录功能，避免跨边界采样

功能说明：
- 创建 CustomAnomalySegLoader 类，支持记录 segment 边界
- 避免滑窗采样时产生跨边界的窗口（断层窗口）
- 支持从 dataset_info.txt 自动读取 segment_length，或通过参数传递
- 在 data_factory.py 中添加对自定义数据集的支持
- 修改位置：
  - data_provider/data_loader.py: 新增 CustomAnomalySegLoader 类
  - data_provider/data_factory.py: 添加 CustomAnomalySegLoader 导入和特殊处理

提示词：
目前数据集样本点过多，另外，拼接的连接部分通过滑窗采样，就会是一个断层的采样，所以如何规避断层窗口？需要实现记录边界，避免跨边界采样。"

echo "Git 提交完成！"
echo "当前分支: $(git branch --show-current)"
echo "提交历史:"
git log --oneline -2


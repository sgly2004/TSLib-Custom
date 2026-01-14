# 单维度异常检测完整流程

本文档说明如何对单个维度（CHX00L006PT0101 - 呼和浩特末站压力）进行异常检测。

## 📋 目录

1. [准备工作](#1-准备工作)
2. [数据提取](#2-数据提取)
3. [模型训练](#3-模型训练)
4. [结果可视化](#4-结果可视化)
5. [参数说明](#5-参数说明)

---

## 1. 准备工作

### 1.1 数据准备

确保以下目录中有数据：

- `data/normal/`: 正常运行的数据片段（用于训练）
- `data/csv_data/`: 完整的测试数据（包含正常和异常）

### 1.2 环境要求

确保已安装所需依赖：

```bash
pip install -r requirements.txt
```

---

## 2. 数据提取

### 2.1 提取单维度数据

在项目根目录执行：

```bash
python scripts/gas/build_single_dimension_dataset.py
```

**输出：**
- `dataset/gas_anomaly_single/train.csv`: 训练数据（正常数据）
- `dataset/gas_anomaly_single/test.csv`: 测试数据（全局数据）

**数据格式：**
```
date,CHX00L006PT0101
2021-01-31 08:00:00,0.156
2021-01-31 08:00:02,0.156
...
```

### 2.2 自定义提取参数

如果需要提取其他维度或调整参数：

```bash
python scripts/gas/build_single_dimension_dataset.py \
  --target_column CHX00L006PT0101 \
  --min_normal_len 150 \
  --normal_dir data/normal \
  --raw_dir data/csv_data \
  --out_dir dataset/gas_anomaly_single
```

**参数说明：**
- `--target_column`: 要提取的列名（默认：CHX00L006PT0101）
- `--min_normal_len`: 正常片段最小长度（默认：150）
- `--normal_dir`: 正常数据目录
- `--raw_dir`: 原始测试数据目录
- `--out_dir`: 输出目录

### 2.3 验证数据

提取完成后，脚本会显示数据统计信息：

```
数据统计摘要
================================================================================
训练集统计:
  - 数据点数: XXXXX
  - CHX00L006PT0101 范围: [min, max]
  - CHX00L006PT0101 均值: X.XXXX
  - CHX00L006PT0101 标准差: X.XXXX

测试集统计:
  - 数据点数: XXXXX
  - CHX00L006PT0101 范围: [min, max]
  - CHX00L006PT0101 均值: X.XXXX
  - CHX00L006PT0101 标准差: X.XXXX
```

---

## 3. 模型训练

### 3.1 方案一：标准训练（推荐）

使用完整的训练配置，获得最佳性能：

```bash
bash scripts/gas/train_single_dimension.sh
```

**训练配置：**
- 序列长度：256
- 批大小：128
- 模型维度：128
- 前馈网络维度：512
- 编码器层数：2
- 训练轮数：20
- 学习率：0.0001

**预计训练时间：** 根据数据量和硬件配置，约 10-30 分钟

### 3.2 方案二：快速测试

用于快速验证流程，使用较小的模型和较少的训练轮数：

```bash
bash scripts/gas/train_single_dimension_quick.sh
```

**快速训练配置：**
- 序列长度：128
- 批大小：256
- 模型维度：64
- 前馈网络维度：256
- 编码器层数：1
- 训练轮数：5
- 学习率：0.001

**预计训练时间：** 约 2-5 分钟

### 3.3 手动执行训练命令

如果需要自定义参数，可以直接运行：

```bash
python run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --model_id gas_single_CHX00L006PT0101 \
  --model TimesNet \
  --data GAS \
  --root_path ./dataset/gas_anomaly_single \
  --features S \
  --target CHX00L006PT0101 \
  --enc_in 1 \
  --c_out 1 \
  --seq_len 256 \
  --pred_len 0 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --d_model 128 \
  --d_ff 512 \
  --e_layers 2 \
  --top_k 3 \
  --num_kernels 4 \
  --patience 5 \
  --train_epochs 20 \
  --learning_rate 0.0001 \
  --checkpoints ./checkpoints/gas_single_dimension \
  --des CHX00L006PT0101_pressure \
  --use_amp
```

### 3.4 训练输出

训练完成后，会生成以下文件：

**模型文件：**
```
checkpoints/gas_single_dimension/
  └── anomaly_detection_gas_single_CHX00L006PT0101_TimesNet_GAS_ftS_sl256_ll48_pl0_.../
      └── checkpoint.pth
```

**测试结果：**
```
test_results/
  └── anomaly_detection_gas_single_CHX00L006PT0101_TimesNet_GAS_ftS_sl256_ll48_pl0_.../
      └── energy_and_pred.npz
```

`energy_and_pred.npz` 包含：
- `train_energy`: 训练集重构误差
- `test_energy`: 测试集重构误差
- `threshold`: 异常检测阈值
- `pred`: 异常预测结果（0=正常，1=异常）
- `seq_len`: 序列长度

---

## 4. 结果可视化

### 4.1 可视化单个样本

选择一个测试文件进行可视化，例如 `1004.csv`：

```bash
python utils/visualize_gas_anomaly.py \
  --raw_csv data/csv_data/1004.csv \
  --result_file test_results/anomaly_detection_gas_single_CHX00L006PT0101_TimesNet_GAS_ftS_sl256_ll48_pl0_.../energy_and_pred.npz \
  --pressure_tag CHX00L006PT0101 \
  --seq_len 256
```

**注意：** 将 `result_file` 路径替换为实际生成的结果文件路径。

### 4.2 输出文件

可视化结果保存在：

```
vis_results/gas/1004_energy_and_pred_pressure-CHX00L006PT0101.png
```

图表显示：
- 压力曲线
- 异常区间（红色背景高亮）

### 4.3 批量可视化

如需可视化多个样本，可以创建循环脚本：

```bash
# 创建批量可视化脚本
cat > visualize_all_samples.sh << 'EOF'
#!/bin/bash

RESULT_FILE="test_results/anomaly_detection_gas_single_CHX00L006PT0101_TimesNet_GAS_ftS_sl256_ll48_pl0_.../energy_and_pred.npz"

for csv_file in data/csv_data/*.csv; do
    echo "Processing $csv_file..."
    python utils/visualize_gas_anomaly.py \
      --raw_csv "$csv_file" \
      --result_file "$RESULT_FILE" \
      --pressure_tag CHX00L006PT0101 \
      --seq_len 256
done

echo "所有样本可视化完成！"
EOF

chmod +x visualize_all_samples.sh
bash visualize_all_samples.sh
```

---

## 5. 参数说明

### 5.1 数据相关参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--features` | 特征类型：S=单变量，M=多变量 | S |
| `--target` | 目标列名 | CHX00L006PT0101 |
| `--enc_in` | 输入维度 | 1 |
| `--c_out` | 输出维度 | 1 |
| `--seq_len` | 序列长度 | 128-256 |
| `--root_path` | 数据集路径 | ./dataset/gas_anomaly_single |

### 5.2 模型相关参数

| 参数 | 说明 | 快速测试 | 标准训练 |
|------|------|----------|----------|
| `--d_model` | 模型维度 | 64 | 128-256 |
| `--d_ff` | 前馈网络维度 | 256 | 512-1024 |
| `--e_layers` | 编码器层数 | 1 | 2-3 |
| `--top_k` | Top-K 频率数 | 3 | 3-5 |
| `--num_kernels` | 卷积核数量 | 4 | 4-6 |

### 5.3 训练相关参数

| 参数 | 说明 | 快速测试 | 标准训练 |
|------|------|----------|----------|
| `--batch_size` | 批大小 | 256 | 128-256 |
| `--train_epochs` | 训练轮数 | 5 | 20-50 |
| `--learning_rate` | 学习率 | 0.001 | 0.0001 |
| `--patience` | 早停耐心值 | 3 | 5-10 |

### 5.4 其他参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--anomaly_ratio` | 异常比例（用于阈值计算） | 1.0 |
| `--use_amp` | 使用混合精度训练（加速） | 建议开启 |
| `--checkpoints` | 模型保存路径 | ./checkpoints/... |
| `--des` | 实验描述（用于区分不同实验） | 自定义 |

---

## 6. 常见问题

### Q1: 训练时显示 CUDA out of memory

**解决方案：**
- 减小 `batch_size`（如 128 → 64 → 32）
- 减小 `seq_len`（如 256 → 128）
- 减小 `d_model` 和 `d_ff`

### Q2: 训练效果不好

**可尝试：**
1. 增加训练轮数（`train_epochs`）
2. 调整学习率（`learning_rate`）
3. 增加模型容量（`d_model`, `d_ff`, `e_layers`）
4. 增加序列长度（`seq_len`）
5. 检查数据质量和分布

### Q3: 如何选择合适的序列长度？

**建议：**
- 查看数据的周期性特征
- 常见选择：64, 96, 128, 192, 256
- 压力数据建议：128-256

### Q4: 异常检测阈值如何确定？

**说明：**
- 模型自动基于训练集重构误差的分位数确定阈值
- `anomaly_ratio` 参数控制阈值的严格程度
- 值越小，阈值越严格（更少误报，可能漏报）

---

## 7. 下一步建议

### 7.1 对比实验

1. **不同序列长度：**
   ```bash
   # seq_len = 128
   python run.py ... --seq_len 128 --des seq128
   
   # seq_len = 256
   python run.py ... --seq_len 256 --des seq256
   ```

2. **不同模型：**
   - TimesNet（当前）
   - PatchTST
   - DLinear
   - Autoformer

### 7.2 多维度对比

如果需要对比多个维度的异常检测效果，可以为每个维度重复上述流程：

```bash
# 提取其他维度
python scripts/gas/build_single_dimension_dataset.py \
  --target_column CHX00E005PT0101 \
  --out_dir dataset/gas_anomaly_CHX00E005PT0101

# 训练其他维度
python run.py ... --target CHX00E005PT0101 ...
```

### 7.3 集成方法

可以训练多个单维度模型，然后进行结果融合（OR/AND/投票等）。

---

## 8. 参考文档

- [多维度异常检测流程](gas_anomaly_pipeline.md)
- [TimesNet 教程](../tutorial/TimesNet_tutorial.ipynb)
- [数据加载器说明](../data_provider/data_loader.py)

---

## 📝 总结

**完整流程：**

```bash
# 1. 提取数据
python scripts/gas/build_single_dimension_dataset.py

# 2. 训练模型
bash scripts/gas/train_single_dimension.sh

# 3. 可视化结果
python utils/visualize_gas_anomaly.py \
  --raw_csv data/csv_data/1004.csv \
  --result_file test_results/.../energy_and_pred.npz \
  --pressure_tag CHX00L006PT0101 \
  --seq_len 256
```

**时间估算：**
- 数据提取：< 1 分钟
- 快速训练：2-5 分钟
- 标准训练：10-30 分钟
- 可视化：< 1 分钟/样本

祝训练顺利！ 🚀

# 🚀 单维度异常检测 - 命令速查表

> 快速查找常用命令

---

## 📍 项目路径

```bash
cd /Users/liuqiyuan/Documents/项目/operating-condition-time-series/operating-condition-time-series/TSLib-Custom
```

---

## ⚡ 快速开始

### 一键执行全流程（推荐）

```bash
bash run_single_dimension_full_pipeline.sh
```

---

## 📦 数据提取

### 基本用法

```bash
python scripts/gas/build_single_dimension_dataset.py
```

### 自定义维度

```bash
python scripts/gas/build_single_dimension_dataset.py \
  --target_column CHX00L006PT0101 \
  --min_normal_len 150 \
  --normal_dir data/normal \
  --raw_dir data/csv_data \
  --out_dir dataset/gas_anomaly_single
```

### 提取其他维度

```bash
# 示例：提取 CHX00E005PT0101
python scripts/gas/build_single_dimension_dataset.py \
  --target_column CHX00E005PT0101 \
  --out_dir dataset/gas_anomaly_CHX00E005PT0101
```

---

## 🎯 模型训练

### 快速测试（2-5分钟）

```bash
bash scripts/gas/train_single_dimension_quick.sh
```

### 标准训练（10-30分钟）

```bash
bash scripts/gas/train_single_dimension.sh
```

### 自定义训练

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
  --enc_in 1 --c_out 1 \
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

---

## 📊 可视化

### 单个样本

```bash
python utils/visualize_gas_anomaly.py \
  --raw_csv data/csv_data/1004.csv \
  --result_file test_results/anomaly_detection_gas_single_CHX00L006PT0101_*/energy_and_pred.npz \
  --pressure_tag CHX00L006PT0101 \
  --seq_len 256
```

### 批量可视化

```bash
RESULT_FILE="test_results/anomaly_detection_gas_single_CHX00L006PT0101_TimesNet_GAS_ftS_sl256_ll48_pl0_dm128_nh8_el2_dl1_df512_expand2_dc4_fc1_ebtimeF_dtTrue_CHX00L006PT0101_pressure_0/energy_and_pred.npz"

for csv_file in data/csv_data/*.csv; do
    echo "Processing $csv_file..."
    python utils/visualize_gas_anomaly.py \
      --raw_csv "$csv_file" \
      --result_file "$RESULT_FILE" \
      --pressure_tag CHX00L006PT0101 \
      --seq_len 256
done
```

### 批量可视化（前10个样本）

```bash
RESULT_FILE="test_results/anomaly_detection_gas_single_CHX00L006PT0101_*/energy_and_pred.npz"

for csv_file in data/csv_data/*.csv; do
    count=$((count + 1))
    if [ $count -gt 10 ]; then break; fi
    python utils/visualize_gas_anomaly.py \
      --raw_csv "$csv_file" \
      --result_file "$RESULT_FILE" \
      --pressure_tag CHX00L006PT0101 \
      --seq_len 256
done
```

---

## 🔍 查看结果

### 列出生成的数据

```bash
ls -lh dataset/gas_anomaly_single/
wc -l dataset/gas_anomaly_single/*.csv
```

### 列出训练结果

```bash
ls -lh test_results/anomaly_detection_gas_single_*/
```

### 列出模型文件

```bash
ls -lh checkpoints/gas_single_dimension*/
```

### 列出可视化结果

```bash
ls -lh vis_results/gas/*CHX00L006PT0101*.png
```

### 查看最新的可视化结果

```bash
open vis_results/gas/$(ls -t vis_results/gas/*CHX00L006PT0101*.png | head -1)
```

---

## 🛠️ 调试命令

### 检查 GPU

```bash
nvidia-smi
```

### 检查 Python 环境

```bash
python --version
pip list | grep torch
pip list | grep pandas
```

### 查看训练日志

```bash
# 查看最新的训练日志
cat checkpoints/gas_single_dimension*/*/train.log
```

### 检查数据统计

```bash
# 训练集
python -c "import pandas as pd; df=pd.read_csv('dataset/gas_anomaly_single/train.csv'); print(f'Train: {len(df)} rows'); print(df.describe())"

# 测试集
python -c "import pandas as pd; df=pd.read_csv('dataset/gas_anomaly_single/test.csv'); print(f'Test: {len(df)} rows'); print(df.describe())"
```

### 快速查看数据前几行

```bash
head -20 dataset/gas_anomaly_single/train.csv
head -20 dataset/gas_anomaly_single/test.csv
```

---

## 🔧 参数快速调整

### 减小内存占用

```bash
# 在训练脚本中修改这些参数
--batch_size 64          # 减小批大小
--seq_len 128            # 减小序列长度
--d_model 64             # 减小模型维度
--d_ff 256               # 减小前馈网络维度
```

### 加快训练速度

```bash
--batch_size 256         # 增大批大小（需要更多内存）
--train_epochs 5         # 减少训练轮数
--use_amp                # 使用混合精度（已包含）
```

### 提升模型效果

```bash
--train_epochs 30        # 增加训练轮数
--seq_len 512            # 增加序列长度
--d_model 256            # 增加模型维度
--d_ff 1024              # 增加前馈网络维度
--e_layers 3             # 增加编码器层数
--learning_rate 0.00005  # 降低学习率
```

---

## 📂 文件位置速查

| 类型 | 位置 |
|------|------|
| **原始数据** | |
| 正常数据 | `data/normal/*.csv` |
| 测试数据 | `data/csv_data/*.csv` |
| **生成数据** | |
| 训练集 | `dataset/gas_anomaly_single/train.csv` |
| 测试集 | `dataset/gas_anomaly_single/test.csv` |
| **模型输出** | |
| 模型权重 | `checkpoints/gas_single_dimension*/*/checkpoint.pth` |
| 测试结果 | `test_results/anomaly_detection_gas_single_*/energy_and_pred.npz` |
| 可视化图 | `vis_results/gas/*.png` |
| **脚本** | |
| 数据提取 | `scripts/gas/build_single_dimension_dataset.py` |
| 快速训练 | `scripts/gas/train_single_dimension_quick.sh` |
| 标准训练 | `scripts/gas/train_single_dimension.sh` |
| 一键执行 | `run_single_dimension_full_pipeline.sh` |

---

## 🎓 常用参数速查

### 数据相关

```bash
--data GAS                                  # 数据集类型
--root_path ./dataset/gas_anomaly_single    # 数据根目录
--features S                                # S=单变量，M=多变量
--target CHX00L006PT0101                    # 目标列名
--enc_in 1                                  # 输入维度
--c_out 1                                   # 输出维度
```

### 模型架构

```bash
--model TimesNet         # 模型类型（可选：PatchTST, Autoformer, DLinear）
--seq_len 256            # 序列长度（常用：64, 96, 128, 192, 256, 512）
--d_model 128            # 模型维度（常用：64, 128, 256, 512）
--d_ff 512               # 前馈网络维度（常用：d_model * 2 或 4）
--e_layers 2             # 编码器层数（常用：1, 2, 3）
--top_k 3                # Top-K 频率（TimesNet 特有）
--num_kernels 4          # 卷积核数量（TimesNet 特有）
```

### 训练配置

```bash
--train_epochs 20        # 训练轮数（快速测试：5，标准：20，完整：50）
--batch_size 128         # 批大小（根据内存调整：32, 64, 128, 256）
--learning_rate 0.0001   # 学习率（常用：0.00001, 0.0001, 0.001）
--patience 5             # 早停耐心值（轮数）
--use_amp                # 使用混合精度加速
```

### 异常检测

```bash
--task_name anomaly_detection   # 任务类型
--anomaly_ratio 1               # 异常比例（用于阈值计算）
--pred_len 0                    # 异常检测不需要预测长度
```

---

## 🔄 工作流程速查

### 完整流程

```bash
# 1. 提取数据
python scripts/gas/build_single_dimension_dataset.py

# 2. 训练模型
bash scripts/gas/train_single_dimension.sh

# 3. 可视化结果
python utils/visualize_gas_anomaly.py \
  --raw_csv data/csv_data/1004.csv \
  --result_file test_results/anomaly_detection_gas_single_*/energy_and_pred.npz \
  --pressure_tag CHX00L006PT0101 \
  --seq_len 256
```

### 对比实验

```bash
# 实验1：基线（小模型）
python run.py ... --d_model 64 --d_ff 256 --e_layers 1 --des baseline

# 实验2：中等模型
python run.py ... --d_model 128 --d_ff 512 --e_layers 2 --des medium

# 实验3：大模型
python run.py ... --d_model 256 --d_ff 1024 --e_layers 3 --des large
```

### 快速验证

```bash
# 只用前1000行数据快速测试
head -n 1001 dataset/gas_anomaly_single/train.csv > dataset/gas_anomaly_single/train_debug.csv
head -n 1001 dataset/gas_anomaly_single/test.csv > dataset/gas_anomaly_single/test_debug.csv

# 快速训练
python run.py ... --root_path ./dataset/gas_anomaly_single --train_epochs 2 --des debug
```

---

## 💡 实用技巧

### 后台训练

```bash
# 后台运行训练（长时间训练推荐）
nohup bash scripts/gas/train_single_dimension.sh > train.log 2>&1 &

# 查看进度
tail -f train.log

# 查看进程
ps aux | grep python
```

### 定时提醒

```bash
# 训练完成后发出提醒音
bash scripts/gas/train_single_dimension.sh && say "Training completed"
```

### 自动备份

```bash
# 训练完成后自动备份结果
bash scripts/gas/train_single_dimension.sh && \
  cp -r test_results checkpoints ~/backup/$(date +%Y%m%d_%H%M%S)/
```

---

## 📖 相关文档

| 文档 | 命令 |
|------|------|
| 快速开始 | `cat QUICKSTART_SINGLE_DIMENSION.md` |
| 完整指南 | `cat README_SINGLE_DIMENSION.md` |
| 详细流程 | `cat single_dimension_pipeline.md` |
| 方案总结 | `cat SINGLE_DIMENSION_SUMMARY.md` |
| 从这开始 | `cat START_HERE.md` |

---

## 🆘 遇到问题？

```bash
# 1. 检查 GPU
nvidia-smi

# 2. 检查数据
ls -lh dataset/gas_anomaly_single/

# 3. 检查 Python 环境
pip list | grep torch

# 4. 重新提取数据
rm -rf dataset/gas_anomaly_single/
python scripts/gas/build_single_dimension_dataset.py

# 5. 清理旧模型重新训练
rm -rf checkpoints/gas_single_dimension*/
bash scripts/gas/train_single_dimension_quick.sh
```

---

**保存此文档以便快速查找命令！** 📋

*最后更新：2026-01-14*

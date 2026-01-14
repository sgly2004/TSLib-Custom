# 🎯 单维度异常检测方案总结

## ✅ 已完成的工作

### 1. 数据提取脚本
**文件：** `scripts/gas/build_single_dimension_dataset.py`

**功能：**
- 从 `data/normal/` 提取正常数据（CHX00L006PT0101 维度）
- 从 `data/csv_data/` 提取测试数据（同一维度）
- 生成训练集和测试集

**运行结果：**
```
✅ 训练集：34,826 条记录（52个有效文件）
✅ 测试集：266,533 条记录（69个有效文件）
✅ 数据范围：训练集 [-0.066, 3.778]，测试集 [-0.066, 4.291]
```

**输出位置：**
- `dataset/gas_anomaly_single/train.csv`
- `dataset/gas_anomaly_single/test.csv`

---

### 2. 训练脚本

#### 方案 A：标准训练脚本
**文件：** `scripts/gas/train_single_dimension.sh`

**配置：**
- 序列长度：256
- 批大小：128
- 模型维度：128 / 前馈维度：512
- 编码器层数：2
- 训练轮数：20
- 学习率：0.0001

**适用场景：** 追求最佳性能的实际应用

#### 方案 B：快速测试脚本
**文件：** `scripts/gas/train_single_dimension_quick.sh`

**配置：**
- 序列长度：128
- 批大小：256
- 模型维度：64 / 前馈维度：256
- 编码器层数：1
- 训练轮数：5
- 学习率：0.001

**适用场景：** 快速验证流程，2-5分钟完成

---

### 3. 文档

#### 快速开始指南
**文件：** `QUICKSTART_SINGLE_DIMENSION.md`
- 三步快速开始
- 故障排除
- 预期结果示例

#### 完整流程文档
**文件：** `single_dimension_pipeline.md`
- 详细的步骤说明
- 参数调优指南
- 常见问题解答
- 进阶使用方法

---

## 📝 完整使用流程

### 第一步：数据准备 ✅ 已完成

```bash
cd /Users/liuqiyuan/Documents/项目/operating-condition-time-series/operating-condition-time-series/TSLib-Custom

python scripts/gas/build_single_dimension_dataset.py
```

**结果：**
- ✅ 训练数据：34,826 条正常压力数据
- ✅ 测试数据：266,533 条全局压力数据
- ✅ 数据已保存到 `dataset/gas_anomaly_single/`

---

### 第二步：模型训练

**选择训练方案：**

#### 🚀 快速测试（推荐先运行这个）

```bash
bash scripts/gas/train_single_dimension_quick.sh
```

⏱️ **预计时间：** 2-5 分钟  
💡 **目的：** 快速验证流程是否正常

#### 🎯 完整训练（获得最佳效果）

```bash
bash scripts/gas/train_single_dimension.sh
```

⏱️ **预计时间：** 10-30 分钟  
💡 **目的：** 获得最佳的异常检测性能

---

### 第三步：结果可视化

训练完成后，会显示结果文件路径，例如：

```
test_results/anomaly_detection_gas_single_CHX00L006PT0101_TimesNet_GAS_ftS_sl256_ll48_pl0_.../energy_and_pred.npz
```

**可视化单个样本：**

```bash
python utils/visualize_gas_anomaly.py \
  --raw_csv data/csv_data/1004.csv \
  --result_file test_results/anomaly_detection_gas_single_CHX00L006PT0101_*/energy_and_pred.npz \
  --pressure_tag CHX00L006PT0101 \
  --seq_len 256
```

**可视化所有样本：**

```bash
# 设置结果文件路径
RESULT_FILE="test_results/anomaly_detection_gas_single_CHX00L006PT0101_TimesNet_GAS_ftS_sl256_ll48_pl0_.../energy_and_pred.npz"

# 批量处理
for csv_file in data/csv_data/*.csv; do
    python utils/visualize_gas_anomaly.py \
      --raw_csv "$csv_file" \
      --result_file "$RESULT_FILE" \
      --pressure_tag CHX00L006PT0101 \
      --seq_len 256
done
```

**结果保存位置：**
```
vis_results/gas/1004_energy_and_pred_pressure-CHX00L006PT0101.png
```

---

## 🎯 核心优势

### 相比 32 维方案的优势

1. **更简单的数据处理**
   - 只需关注单个维度
   - 数据缺失问题更容易处理

2. **更快的训练速度**
   - 模型参数更少（enc_in=1 vs enc_in=32）
   - 快速测试：2-5 分钟 vs 原方案 30+ 分钟

3. **更容易调试**
   - 单维度更容易理解模型学到的模式
   - 异常更容易定位和解释

4. **更灵活的部署**
   - 可以独立监控重要维度
   - 可以根据需要组合多个单维度模型

---

## 📊 数据统计

### 训练集（正常数据）
| 指标 | 数值 |
|------|------|
| 数据点数 | 34,826 |
| 有效文件数 | 52/69 |
| 数据范围 | [-0.066, 3.778] |
| 均值 | 0.2475 |
| 标准差 | 0.4623 |

### 测试集（全局数据）
| 指标 | 数值 |
|------|------|
| 数据点数 | 266,533 |
| 有效文件数 | 69/69 |
| 数据范围 | [-0.066, 4.291] |
| 均值 | 0.3745 |
| 标准差 | 0.7070 |

### 观察
- ⚠️ 测试集最大值 (4.291) > 训练集最大值 (3.778)
- ⚠️ 测试集标准差更大，说明可能包含更多异常情况
- ✅ 这是合理的，因为测试集包含异常数据

---

## 🔧 参数配置对比

| 参数 | 快速测试 | 标准训练 | 说明 |
|------|----------|----------|------|
| seq_len | 128 | 256 | 序列长度 |
| batch_size | 256 | 128 | 批大小 |
| d_model | 64 | 128 | 模型维度 |
| d_ff | 256 | 512 | 前馈网络维度 |
| e_layers | 1 | 2 | 编码器层数 |
| train_epochs | 5 | 20 | 训练轮数 |
| learning_rate | 0.001 | 0.0001 | 学习率 |

---

## 📂 项目文件结构

```
TSLib-Custom/
├── data/
│   ├── normal/              # 正常数据（训练用）
│   └── csv_data/            # 全局数据（测试用）
├── dataset/
│   └── gas_anomaly_single/  # ✅ 新生成的单维度数据
│       ├── train.csv        # 训练集（34,826条）
│       └── test.csv         # 测试集（266,533条）
├── scripts/gas/
│   ├── build_single_dimension_dataset.py  # ✅ 数据提取脚本
│   ├── train_single_dimension.sh          # ✅ 标准训练脚本
│   └── train_single_dimension_quick.sh    # ✅ 快速训练脚本
├── checkpoints/             # 模型保存目录（训练后生成）
├── test_results/            # 测试结果（训练后生成）
├── vis_results/gas/         # 可视化结果（可视化后生成）
├── QUICKSTART_SINGLE_DIMENSION.md    # ✅ 快速开始指南
├── single_dimension_pipeline.md       # ✅ 完整流程文档
└── SINGLE_DIMENSION_SUMMARY.md        # ✅ 本文档
```

---

## 🚀 下一步建议

### 1. 立即开始

```bash
# 快速测试（建议首先运行）
bash scripts/gas/train_single_dimension_quick.sh

# 等待 2-5 分钟后查看结果
```

### 2. 优化实验

如果快速测试效果不理想，可以尝试：

1. **增加训练轮数**
   ```bash
   # 修改脚本中的 --train_epochs 参数
   --train_epochs 30  # 或 50
   ```

2. **调整序列长度**
   ```bash
   # 尝试不同的时间窗口
   --seq_len 128   # 更短的上下文
   --seq_len 512   # 更长的上下文
   ```

3. **增加模型容量**
   ```bash
   --d_model 256 --d_ff 1024 --e_layers 3
   ```

### 3. 对比实验

训练多个配置，对比效果：

```bash
# 配置 1：小模型短序列
python run.py ... --seq_len 128 --d_model 64 --des config1

# 配置 2：大模型长序列
python run.py ... --seq_len 256 --d_model 128 --des config2

# 配置 3：超大模型
python run.py ... --seq_len 512 --d_model 256 --des config3
```

### 4. 多维度扩展

如果单维度效果好，可以为其他重要维度重复此流程：

```bash
# 提取其他维度
python scripts/gas/build_single_dimension_dataset.py \
  --target_column CHX00E005PT0101 \
  --out_dir dataset/gas_anomaly_CHX00E005PT0101

# 训练对应模型
python run.py ... --target CHX00E005PT0101 ...
```

---

## 💡 技巧和建议

1. **GPU vs CPU**
   - GPU 训练快 10-50 倍
   - 检查 GPU：`nvidia-smi`
   - 如果没有 GPU，建议先用快速测试脚本

2. **内存优化**
   - 遇到 OOM 错误：减小 batch_size
   - 128 → 64 → 32

3. **可视化技巧**
   - 先可视化几个有代表性的样本
   - 观察异常检测阈值是否合理
   - 根据可视化结果调整 anomaly_ratio

4. **结果评估**
   - 关注训练集和测试集 loss 的差异
   - 如果测试集 loss >> 训练集 loss，可能过拟合
   - 可以增加正则化或减小模型容量

---

## 📞 获取帮助

- **快速开始：** 查看 `QUICKSTART_SINGLE_DIMENSION.md`
- **详细文档：** 查看 `single_dimension_pipeline.md`
- **多维度方案：** 查看 `gas_anomaly_pipeline.md`

---

## ✨ 总结

你现在拥有一套完整的单维度异常检测方案：

✅ **数据已准备好**（34,826 训练 + 266,533 测试）  
✅ **脚本已就绪**（快速测试 + 标准训练）  
✅ **文档已完善**（快速指南 + 详细文档）

**现在可以开始训练了！**

```bash
# 在项目根目录执行
bash scripts/gas/train_single_dimension_quick.sh
```

祝训练顺利！🎉

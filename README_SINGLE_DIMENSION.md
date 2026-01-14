# 🎯 单维度异常检测 - 使用指南

针对 **CHX00L006PT0101（呼和浩特末站压力）** 的单维度时间序列异常检测方案。

---

## 🚀 快速开始（推荐）

### 方式 1：一键执行完整流程

```bash
cd /Users/liuqiyuan/Documents/项目/operating-condition-time-series/operating-condition-time-series/TSLib-Custom

bash run_single_dimension_full_pipeline.sh
```

这个脚本会自动执行：
1. ✅ 数据提取（train + test）
2. ✅ 模型训练（可选快速/完整）
3. ✅ 结果可视化（自动找到结果文件）

---

### 方式 2：分步执行

#### 步骤 1：数据提取

```bash
python scripts/gas/build_single_dimension_dataset.py
```

**输出：**
- `dataset/gas_anomaly_single/train.csv`（34,826条）
- `dataset/gas_anomaly_single/test.csv`（266,533条）

#### 步骤 2：模型训练

**快速测试（2-5分钟）：**
```bash
bash scripts/gas/train_single_dimension_quick.sh
```

**完整训练（10-30分钟）：**
```bash
bash scripts/gas/train_single_dimension.sh
```

#### 步骤 3：结果可视化

```bash
python utils/visualize_gas_anomaly.py \
  --raw_csv data/csv_data/1004.csv \
  --result_file test_results/anomaly_detection_gas_single_CHX00L006PT0101_*/energy_and_pred.npz \
  --pressure_tag CHX00L006PT0101 \
  --seq_len 256
```

---

## 📁 项目文件说明

### 核心脚本

| 文件 | 说明 | 用途 |
|------|------|------|
| `run_single_dimension_full_pipeline.sh` | 🌟 一键执行脚本 | 完整流程自动化 |
| `scripts/gas/build_single_dimension_dataset.py` | 数据提取脚本 | 生成单维度数据集 |
| `scripts/gas/train_single_dimension.sh` | 标准训练脚本 | 完整训练（20轮） |
| `scripts/gas/train_single_dimension_quick.sh` | 快速训练脚本 | 快速验证（5轮） |

### 文档

| 文件 | 说明 | 适合对象 |
|------|------|----------|
| `README_SINGLE_DIMENSION.md` | 📖 本文档 | 所有用户 |
| `QUICKSTART_SINGLE_DIMENSION.md` | 🚀 快速开始 | 新手用户 |
| `single_dimension_pipeline.md` | 📚 完整流程 | 进阶用户 |
| `SINGLE_DIMENSION_SUMMARY.md` | 📊 方案总结 | 需要全貌理解 |

---

## 🎯 数据统计

### ✅ 已提取的数据

| 数据集 | 数据点数 | 有效文件 | 数值范围 | 均值 | 标准差 |
|--------|----------|----------|----------|------|--------|
| 训练集 | 34,826 | 52/69 | [-0.066, 3.778] | 0.248 | 0.462 |
| 测试集 | 266,533 | 69/69 | [-0.066, 4.291] | 0.375 | 0.707 |

**数据来源：**
- 训练集：`data/normal/` 中的正常运行数据
- 测试集：`data/csv_data/` 中的完整运行数据（包含异常）

---

## ⚙️ 训练配置对比

| 参数 | 快速测试 | 标准训练 | 说明 |
|------|----------|----------|------|
| **训练时间** | 2-5分钟 | 10-30分钟 | 取决于硬件 |
| **序列长度** | 128 | 256 | 时间窗口大小 |
| **批大小** | 256 | 128 | 每批样本数 |
| **模型维度** | 64 | 128 | 模型容量 |
| **前馈维度** | 256 | 512 | FFN 维度 |
| **编码器层** | 1 | 2 | 网络深度 |
| **训练轮数** | 5 | 20 | epoch 数 |
| **学习率** | 0.001 | 0.0001 | 优化器学习率 |

**推荐：**
- 新手/验证流程：使用快速测试
- 实际应用：使用标准训练

---

## 📊 预期输出

### 1. 数据提取输出

```
目标维度: CHX00L006PT0101 (呼和浩特末站压力)
正在处理 69 个正常数据文件...
✓ 1001_normal.csv: 345 条记录
...

✅ 训练数据已保存: ./dataset/gas_anomaly_single/train.csv
   - 有效文件数: 52/69
   - 总记录数: 34826

✅ 测试数据已保存: ./dataset/gas_anomaly_single/test.csv
   - 有效文件数: 69/69
   - 总记录数: 266533
```

### 2. 训练输出

```
Epoch: 1 cost time: 12.34s
  train_loss: 0.8234
  test_loss: 0.7891
...
Epoch: 20 cost time: 11.87s
  train_loss: 0.1245
  test_loss: 0.1567

Best model saved!
Test results saved to: test_results/.../energy_and_pred.npz
```

### 3. 可视化输出

生成图片：`vis_results/gas/1004_energy_and_pred_pressure-CHX00L006PT0101.png`

图片内容：
- 压力曲线随时间变化
- 异常区间用红色背景标注

---

## 🔧 常见问题

### Q1: 如何选择训练方案？

**A:** 
- **第一次使用**：选择快速测试，验证流程
- **效果验证**：快速测试效果好 → 再跑完整训练
- **实际应用**：直接用标准训练

### Q2: CUDA 内存不足怎么办？

**A:** 编辑训练脚本，减小以下参数：
```bash
--batch_size 64        # 减小批大小（原128）
--seq_len 128          # 减小序列长度（原256）
--d_model 64           # 减小模型维度（原128）
```

### Q3: 训练效果不好怎么办？

**A:** 尝试以下方法：
1. 增加训练轮数：`--train_epochs 30`
2. 调整序列长度：试试 `128/192/256/512`
3. 增加模型容量：`--d_model 256 --d_ff 1024`
4. 检查数据质量和分布

### Q4: 如何批量可视化所有样本？

**A:** 使用循环：
```bash
RESULT_FILE="test_results/.../energy_and_pred.npz"

for csv in data/csv_data/*.csv; do
    python utils/visualize_gas_anomaly.py \
      --raw_csv "$csv" \
      --result_file "$RESULT_FILE" \
      --pressure_tag CHX00L006PT0101 \
      --seq_len 256
done
```

### Q5: 如何提取其他维度？

**A:** 使用 `--target_column` 参数：
```bash
python scripts/gas/build_single_dimension_dataset.py \
  --target_column CHX00E005PT0101 \
  --out_dir dataset/gas_anomaly_CHX00E005PT0101
```

---

## 🎓 进阶使用

### 1. 自定义训练参数

```bash
python run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --model_id gas_custom \
  --model TimesNet \
  --data GAS \
  --root_path ./dataset/gas_anomaly_single \
  --features S \
  --target CHX00L006PT0101 \
  --enc_in 1 --c_out 1 \
  --seq_len 256 \
  --batch_size 128 \
  --d_model 128 \
  --d_ff 512 \
  --e_layers 2 \
  --train_epochs 20 \
  --learning_rate 0.0001 \
  --checkpoints ./checkpoints/gas_custom \
  --des my_experiment \
  --use_amp
```

### 2. 尝试不同模型

编辑训练脚本，修改 `--model` 参数：

```bash
--model TimesNet     # 当前（推荐）
--model PatchTST     # Transformer变体
--model Autoformer   # 自相关机制
--model DLinear      # 简单线性（快速基线）
```

### 3. 对比实验

```bash
# 实验1：短序列
python run.py ... --seq_len 128 --des exp_seq128

# 实验2：中序列
python run.py ... --seq_len 256 --des exp_seq256

# 实验3：长序列
python run.py ... --seq_len 512 --des exp_seq512
```

### 4. 多维度训练

为每个重要维度训练独立模型：

```bash
# 维度1：CHX00L006PT0101
python scripts/gas/build_single_dimension_dataset.py \
  --target_column CHX00L006PT0101 \
  --out_dir dataset/gas_anomaly_CHX00L006PT0101

# 维度2：CHX00E005PT0101
python scripts/gas/build_single_dimension_dataset.py \
  --target_column CHX00E005PT0101 \
  --out_dir dataset/gas_anomaly_CHX00E005PT0101

# 然后分别训练...
```

---

## 📈 性能优化建议

### 训练速度优化

1. **使用 GPU**
   ```bash
   # 检查 GPU
   nvidia-smi
   
   # 确保使用 --use_amp（混合精度）
   ```

2. **增大批大小**（如果内存允许）
   ```bash
   --batch_size 256  # 或更大
   ```

3. **减少验证频率**
   - 如果数据量大，可以降低验证频率

### 模型效果优化

1. **数据预处理**
   - 检查数据质量
   - 去除异常值/噪声

2. **超参数调优**
   - 学习率：`[0.0001, 0.001, 0.01]`
   - 序列长度：`[64, 96, 128, 192, 256, 512]`
   - 模型维度：`[64, 128, 256, 512]`

3. **集成方法**
   - 训练多个模型，结果投票/平均

---

## 📚 相关文档

- **本项目其他文档：**
  - [多维度异常检测](gas_anomaly_pipeline.md)
  - [问题追踪](gas_anomaly_issues.md)
  - [TimesNet教程](tutorial/TimesNet_tutorial.ipynb)

- **代码位置：**
  - 数据加载器：`data_provider/data_loader.py`（GasSegLoader 类）
  - 异常检测实验：`exp/exp_anomaly_detection.py`
  - 模型定义：`models/TimesNet.py`

---

## 📞 帮助与支持

### 查看日志

```bash
# 训练日志
cat checkpoints/gas_single_dimension*/*/train.log

# 测试结果
ls -lh test_results/anomaly_detection_gas_single_*/
```

### 调试模式

```bash
# 减少数据量快速测试
head -n 1000 dataset/gas_anomaly_single/train.csv > dataset/gas_anomaly_single/train_debug.csv

# 使用调试数据训练
python run.py ... --root_path dataset/gas_anomaly_single_debug
```

---

## ✅ 检查清单

使用前确认：

- [ ] 数据文件存在（`data/normal/` 和 `data/csv_data/`）
- [ ] Python 环境已配置（`pip install -r requirements.txt`）
- [ ] GPU 可用（可选，但强烈推荐）
- [ ] 磁盘空间充足（至少 5GB）

使用后验证：

- [ ] 数据提取成功（检查 `dataset/gas_anomaly_single/`）
- [ ] 训练完成（检查 `checkpoints/` 和 `test_results/`）
- [ ] 可视化结果合理（检查 `vis_results/gas/`）

---

## 🎉 总结

你现在拥有：

✅ **完整的单维度异常检测方案**
- 数据提取 → 模型训练 → 结果可视化

✅ **三种使用方式**
- 一键执行脚本（推荐）
- 分步执行
- 自定义配置

✅ **两种训练配置**
- 快速测试（2-5分钟）
- 标准训练（10-30分钟）

✅ **完善的文档**
- 快速开始指南
- 完整流程文档
- 故障排除手册

---

## 🚀 现在开始

```bash
# 方式 1：一键执行（最简单）
bash run_single_dimension_full_pipeline.sh

# 方式 2：分步执行
python scripts/gas/build_single_dimension_dataset.py
bash scripts/gas/train_single_dimension_quick.sh
python utils/visualize_gas_anomaly.py ...
```

**祝训练顺利！** 🎯

---

*最后更新：2026-01-14*

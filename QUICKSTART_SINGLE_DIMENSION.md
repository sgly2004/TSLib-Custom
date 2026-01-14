# 🚀 单维度异常检测 - 快速开始

> 针对 **CHX00L006PT0101（呼和浩特末站压力）** 进行单维度异常检测

## ⚡ 三步快速开始

### 第一步：提取单维度数据

在项目根目录执行：

```bash
cd /Users/liuqiyuan/Documents/项目/operating-condition-time-series/operating-condition-time-series/TSLib-Custom

python scripts/gas/build_single_dimension_dataset.py
```

✅ **预期输出：**
- `dataset/gas_anomaly_single/train.csv` - 训练数据
- `dataset/gas_anomaly_single/test.csv` - 测试数据

---

### 第二步：训练模型

**方案 A - 快速测试（推荐新手）：**

```bash
bash scripts/gas/train_single_dimension_quick.sh
```

⏱️ 预计时间：2-5 分钟

**方案 B - 完整训练（推荐实际使用）：**

```bash
bash scripts/gas/train_single_dimension.sh
```

⏱️ 预计时间：10-30 分钟

✅ **训练完成后会生成：**
- 模型文件：`checkpoints/gas_single_dimension*/.../ checkpoint.pth`
- 测试结果：`test_results/.../energy_and_pred.npz`

---

### 第三步：可视化结果

找到生成的结果文件路径（在训练输出的最后会显示），然后执行：

```bash
# 示例：可视化 1004.csv 样本的异常检测结果
python utils/visualize_gas_anomaly.py \
  --raw_csv data/csv_data/1004.csv \
  --result_file test_results/anomaly_detection_gas_single_CHX00L006PT0101_*/energy_and_pred.npz \
  --pressure_tag CHX00L006PT0101 \
  --seq_len 256
```

**注意：** 需要将 `result_file` 路径替换为实际生成的路径（使用 tab 自动补全）

✅ **可视化结果保存在：**
- `vis_results/gas/1004_energy_and_pred_pressure-CHX00L006PT0101.png`

---

## 📊 预期结果

### 数据提取输出示例：

```
目标维度: CHX00L006PT0101 (呼和浩特末站压力)
正在处理 70 个正常数据文件...
✓ 1001_normal.csv: 346 条记录
✓ 1002_normal.csv: 306 条记录
...

✅ 训练数据已保存: dataset/gas_anomaly_single/train.csv
   - 有效文件数: 65/70
   - 总记录数: 34825

✅ 测试数据已保存: dataset/gas_anomaly_single/test.csv
   - 有效文件数: 70/70
   - 总记录数: 265205
```

### 训练输出示例：

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

---

## 🔧 故障排除

### 问题 1：找不到数据文件

**错误信息：**
```
FileNotFoundError: No such file or directory: 'data/normal'
```

**解决方案：**
确保在正确的目录运行命令，并且数据文件存在：

```bash
# 检查当前目录
pwd

# 列出数据文件
ls data/normal/
ls data/csv_data/
```

---

### 问题 2：CUDA 内存不足

**错误信息：**
```
RuntimeError: CUDA out of memory
```

**解决方案：**
编辑训练脚本，减小批大小：

```bash
# 修改 batch_size 参数
--batch_size 64   # 原来是 128 或 256
```

或使用更小的模型：

```bash
--d_model 64      # 原来是 128
--d_ff 256        # 原来是 512
```

---

### 问题 3：训练太慢

**解决方案：**
1. 使用快速训练脚本（5轮训练）
2. 确保 GPU 可用：`nvidia-smi`
3. 确保使用了 `--use_amp` 参数（混合精度加速）

---

### 问题 4：如何查看详细的训练日志？

训练日志会保存在 `checkpoints` 目录下。可以查看：

```bash
# 找到最新的训练目录
ls -lt checkpoints/gas_single_dimension*/

# 查看训练日志
cat checkpoints/gas_single_dimension*/.../train.log
```

---

## 📈 下一步

### 1. 调整模型参数

编辑 `scripts/gas/train_single_dimension.sh`，尝试不同配置：

- **增加训练轮数：** `--train_epochs 50`
- **调整序列长度：** `--seq_len 128` 或 `--seq_len 512`
- **增加模型容量：** `--d_model 256 --d_ff 1024`

### 2. 尝试其他维度

提取并训练其他维度：

```bash
# 提取其他维度数据
python scripts/gas/build_single_dimension_dataset.py \
  --target_column CHX00E005PT0101 \
  --out_dir dataset/gas_anomaly_CHX00E005PT0101

# 修改训练脚本中的相关参数后训练
```

### 3. 对比多个模型

尝试不同的模型架构：

```bash
# 修改 --model 参数
--model TimesNet    # 当前
--model PatchTST    # 替代方案 1
--model Autoformer  # 替代方案 2
--model DLinear     # 简单基线
```

### 4. 批量可视化

可视化所有测试样本：

```bash
RESULT_FILE="test_results/.../energy_and_pred.npz"

for csv_file in data/csv_data/*.csv; do
    echo "Processing $csv_file..."
    python utils/visualize_gas_anomaly.py \
      --raw_csv "$csv_file" \
      --result_file "$RESULT_FILE" \
      --pressure_tag CHX00L006PT0101 \
      --seq_len 256
done
```

---

## 📚 更多信息

- **完整文档：** [single_dimension_pipeline.md](single_dimension_pipeline.md)
- **多维度方案：** [gas_anomaly_pipeline.md](gas_anomaly_pipeline.md)
- **问题反馈：** [gas_anomaly_issues.md](gas_anomaly_issues.md)

---

## 💡 提示

1. **第一次训练建议使用快速测试脚本**，确保流程正确
2. **查看数据统计信息**，确保数据提取正确
3. **保存好训练结果路径**，便于后续可视化
4. **GPU 训练比 CPU 快 10-50 倍**

---

祝训练顺利！有问题随时查看完整文档 📖

# 🎯 单维度异常检测 - 从这里开始

> **目标：** 对 CHX00L006PT0101（呼和浩特末站压力）进行单维度异常检测

---

## 🚀 最快开始方式

```bash
cd /Users/liuqiyuan/Documents/项目/operating-condition-time-series/operating-condition-time-series/TSLib-Custom

bash run_single_dimension_full_pipeline.sh
```

这个命令会自动完成：数据提取 → 模型训练 → 结果可视化

**预计总时间：** 5-30 分钟（取决于你选择快速/完整训练）

---

## 📚 文档导航

### 🌟 新手推荐阅读顺序

1. **本文档** - `START_HERE.md`（你正在看）
2. **快速开始** - `QUICKSTART_SINGLE_DIMENSION.md`
3. **完整流程** - `single_dimension_pipeline.md`（需要深入了解时）

### 📖 所有文档说明

| 文档 | 内容 | 适合人群 | 阅读时间 |
|------|------|----------|----------|
| `START_HERE.md` | 👈 本文档 | 所有人 | 2 分钟 |
| `README_SINGLE_DIMENSION.md` | 完整使用指南 | 所有人 | 5 分钟 |
| `QUICKSTART_SINGLE_DIMENSION.md` | 三步快速开始 | 新手 | 3 分钟 |
| `single_dimension_pipeline.md` | 详细流程和参数说明 | 进阶用户 | 10 分钟 |
| `SINGLE_DIMENSION_SUMMARY.md` | 方案总结和数据统计 | 需要全面了解 | 5 分钟 |

---

## 🛠️ 核心脚本

| 脚本 | 功能 | 使用场景 |
|------|------|----------|
| `run_single_dimension_full_pipeline.sh` | 🌟 一键执行全流程 | 最快开始 |
| `scripts/gas/build_single_dimension_dataset.py` | 提取单维度数据 | 单独数据准备 |
| `scripts/gas/train_single_dimension.sh` | 标准训练（20轮） | 追求最佳效果 |
| `scripts/gas/train_single_dimension_quick.sh` | 快速训练（5轮） | 快速验证 |

---

## ✅ 已完成的工作

### 1. 数据已准备好 ✓

```
✅ 训练集：34,826 条正常压力数据
✅ 测试集：266,533 条全局压力数据
✅ 位置：dataset/gas_anomaly_single/
```

### 2. 脚本已就绪 ✓

- ✅ 数据提取脚本
- ✅ 快速训练脚本（2-5分钟）
- ✅ 标准训练脚本（10-30分钟）
- ✅ 一键执行脚本

### 3. 文档已完善 ✓

- ✅ 快速开始指南
- ✅ 完整流程文档
- ✅ 参数说明和调优指南
- ✅ 常见问题解答

---

## 🎯 三种使用方式

### 方式 1：一键执行（最简单） ⭐

```bash
bash run_single_dimension_full_pipeline.sh
```

**优点：** 全自动，交互式选择

---

### 方式 2：快速测试（推荐新手）

```bash
# 1. 数据提取
python scripts/gas/build_single_dimension_dataset.py

# 2. 快速训练
bash scripts/gas/train_single_dimension_quick.sh

# 3. 可视化
python utils/visualize_gas_anomaly.py \
  --raw_csv data/csv_data/1004.csv \
  --result_file test_results/anomaly_detection_gas_single_CHX00L006PT0101_*/energy_and_pred.npz \
  --pressure_tag CHX00L006PT0101 \
  --seq_len 128
```

**优点：** 快速验证（5-10分钟总时间）

---

### 方式 3：完整训练（推荐实际使用）

```bash
# 1. 数据提取
python scripts/gas/build_single_dimension_dataset.py

# 2. 完整训练
bash scripts/gas/train_single_dimension.sh

# 3. 可视化
python utils/visualize_gas_anomaly.py \
  --raw_csv data/csv_data/1004.csv \
  --result_file test_results/anomaly_detection_gas_single_CHX00L006PT0101_*/energy_and_pred.npz \
  --pressure_tag CHX00L006PT0101 \
  --seq_len 256
```

**优点：** 最佳效果

---

## 📊 预期结果

### 数据提取

```
✅ 训练数据已保存: dataset/gas_anomaly_single/train.csv
   - 有效文件数: 52/69
   - 总记录数: 34,826

✅ 测试数据已保存: dataset/gas_anomaly_single/test.csv
   - 有效文件数: 69/69
   - 总记录数: 266,533
```

### 模型训练

```
Epoch: 20 cost time: 11.87s
  train_loss: 0.1245
  test_loss: 0.1567

Best model saved!
Test results saved to: test_results/.../energy_and_pred.npz
```

### 可视化

生成图片：`vis_results/gas/1004_energy_and_pred_pressure-CHX00L006PT0101.png`

---

## ❓ 常见问题

### Q: 我应该先看哪个文档？

**A:** 
1. 先看本文档（START_HERE.md）
2. 然后看 QUICKSTART_SINGLE_DIMENSION.md
3. 需要详细了解时看 single_dimension_pipeline.md

### Q: 我应该用哪个训练脚本？

**A:**
- 第一次使用：`train_single_dimension_quick.sh`（快速验证）
- 实际应用：`train_single_dimension.sh`（最佳效果）

### Q: 训练需要多长时间？

**A:**
- 快速训练：2-5 分钟（GPU）/ 10-20 分钟（CPU）
- 完整训练：10-30 分钟（GPU）/ 1-2 小时（CPU）

### Q: 需要 GPU 吗？

**A:**
- 不是必须的，但强烈推荐
- GPU 比 CPU 快 10-50 倍
- 检查 GPU：`nvidia-smi`

### Q: 内存不够怎么办？

**A:**
编辑训练脚本，减小 `--batch_size` 参数：
```bash
--batch_size 64   # 原来是 128 或 256
```

---

## 🎓 下一步

### 1. 立即开始

```bash
bash run_single_dimension_full_pipeline.sh
```

### 2. 查看结果

训练完成后，查看可视化结果：

```bash
open vis_results/gas/1004_energy_and_pred_pressure-CHX00L006PT0101.png
```

### 3. 调优实验

效果不满意？尝试：

- 增加训练轮数
- 调整序列长度
- 增加模型容量

详见：`single_dimension_pipeline.md` 第5节 参数说明

### 4. 多维度扩展

效果好？为其他维度重复此流程：

```bash
python scripts/gas/build_single_dimension_dataset.py \
  --target_column CHX00E005PT0101 \
  --out_dir dataset/gas_anomaly_CHX00E005PT0101
```

---

## 📂 项目结构

```
TSLib-Custom/
├── 📖 START_HERE.md                          ← 你在这里
├── 📖 README_SINGLE_DIMENSION.md             ← 完整指南
├── 📖 QUICKSTART_SINGLE_DIMENSION.md         ← 快速开始
├── 📖 single_dimension_pipeline.md           ← 详细流程
├── 📖 SINGLE_DIMENSION_SUMMARY.md            ← 方案总结
│
├── 🚀 run_single_dimension_full_pipeline.sh  ← 一键执行
│
├── scripts/gas/
│   ├── 📝 build_single_dimension_dataset.py  ← 数据提取
│   ├── 🎯 train_single_dimension.sh          ← 标准训练
│   └── ⚡ train_single_dimension_quick.sh    ← 快速训练
│
├── dataset/gas_anomaly_single/               ← ✅ 已生成
│   ├── train.csv                             ← 34,826 条
│   └── test.csv                              ← 266,533 条
│
├── data/
│   ├── normal/                               ← 原始正常数据
│   └── csv_data/                             ← 原始测试数据
│
├── checkpoints/                              ← 训练后生成模型
├── test_results/                             ← 训练后生成结果
└── vis_results/gas/                          ← 可视化后生成图片
```

---

## ✨ 方案亮点

### vs 32维方案

| 对比项 | 32维方案 | 单维度方案 | 优势 |
|--------|----------|------------|------|
| 训练时间 | 30+ 分钟 | 2-5 分钟（快速） | ⚡ 快 6-15x |
| 模型复杂度 | 高（enc_in=32） | 低（enc_in=1） | ✅ 简单 |
| 可解释性 | 难 | 易 | ✅ 清晰 |
| 调试难度 | 高 | 低 | ✅ 容易 |
| 灵活性 | 固定 | 可独立部署 | ✅ 灵活 |

---

## 🎯 总结

**你拥有一套完整的解决方案：**

✅ 数据已准备（34,826 训练 + 266,533 测试）  
✅ 脚本已就绪（一键执行 + 分步执行）  
✅ 文档已完善（从快速开始到详细调优）  

**现在就开始吧！**

```bash
bash run_single_dimension_full_pipeline.sh
```

---

## 📞 需要帮助？

1. **快速问题：** 查看 QUICKSTART_SINGLE_DIMENSION.md 的"故障排除"部分
2. **参数调优：** 查看 single_dimension_pipeline.md 的"参数说明"部分
3. **深入理解：** 查看 README_SINGLE_DIMENSION.md 的"进阶使用"部分

---

**祝训练顺利！** 🚀🎉

*最后更新：2026-01-14*

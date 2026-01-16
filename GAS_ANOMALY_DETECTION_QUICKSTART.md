# 气体管道异常检测 - 快速上手指南

## 1. 项目背景与思路

### 1.1 为什么从32维改为单维度检测？

**原32维方案的问题**：
- 使用32个维度（所有传感器）同时训练一个模型
- 一个传感器的坏点可能被其他31个传感器"稀释"掉
- 导致模型变迟钝，对异常不敏感

**单维度检测方案**：
- 为关键维度分别训练独立的异常检测模型
- **逻辑**：只要有一个维度检测出异常，就意味着出现了异常操作，可以进行告警
- 后续分类可以通过大模型进行

### 1.2 两套方案对比

| 方案 | 维度 | 类型 | 优点 | 缺点 | 状态 |
|------|------|------|------|------|------|
| 方案1 | CHX00L006PT0101 | 压力（呼和浩特末站） | 最后一站，理论上对所有异常都有响应 | 变化不够敏感 | 已实现 |
| 方案2 | CHX00F003FT0101<br>CHX00F002FT0101 | 流量（乌审旗、鄂托克） | 对异常更敏感，变化明显 | - | **推荐使用** |

### 1.3 核心原理：基于重构的异常检测

1. **训练阶段**：在正常数据上训练 TimesNet 模型，学习"正常"模式
2. **推理阶段**：
   - 模型尝试重构输入数据
   - 计算 MSE = (真实值 - 重构值)²
   - MSE 高 → 模型无法很好重构 → 数据异常
3. **异常判定**：MSE 超过阈值（基于训练集 MSE 的 99.5% 分位数）

---

## 2. 环境准备

### 2.1 依赖安装

```bash
pip install -r requirements.txt
```

主要依赖：
- PyTorch
- pandas, numpy
- matplotlib
- scikit-learn

### 2.2 数据准备

**必需的数据目录结构**：

```
TSLib-Custom/
├── data/
│   ├── normal/          # 正常数据片段（用于训练）
│   │   ├── 1001_normal.csv
│   │   ├── 1002_normal.csv
│   │   └── ...
│   └── csv_data/        # 完整测试数据（包含异常）
│       ├── 1001.csv
│       ├── 1002.csv
│       └── ...
└── dataset/             # 构建后的训练/测试集（自动生成）
```

**数据格式**：
- CSV 文件，第一列为日期，其余为 32 个传感器维度
- 列顺序：`date, CHX00E005PT0101, CHX00E005PT0102, ...`

---

## 3. 方案1: 压力维度 (CHX00L006PT0101)

### 3.1 快速运行（一键脚本）

```bash
bash run_single_dimension_full_pipeline.sh
```

该脚本会自动完成：
1. 检查数据集是否存在
2. 提示选择训练类型（标准/快速）
3. 执行训练
4. 进行推理
5. 生成可视化结果

### 3.2 分步运行

#### 步骤1：数据构建

```bash
python scripts/gas/build_single_dimension_dataset.py
```

**输出**：
- `dataset/gas_anomaly_single/train.csv` - 训练集（拼接所有正常片段）
- `dataset/gas_anomaly_single/test.csv` - 测试集（拼接所有测试数据）

#### 步骤2：训练模型

**标准训练**（推荐）：
```bash
bash scripts/gas/train_single_dimension.sh
```
- 训练 20 epochs
- seq_len=256, d_model=128, d_ff=512, e_layers=2

**快速训练**（调试用）：
```bash
bash scripts/gas/train_single_dimension_quick.sh
```
- 训练 5 epochs
- seq_len=128, d_model=64, d_ff=256, e_layers=1

**输出**：
- `checkpoints/anomaly_detection_gas_single_CHX00L006PT0101_TimesNet_GAS_*/checkpoint.pth`

#### 步骤3：推理测试

```bash
python scripts/gas/test_individual_files.py
```

对 `data/csv_data/` 中每个文件单独进行异常检测。

**输出**：
- `test_results/individual_file_results/1001_result.npz` - 每个文件的能量结果

#### 步骤4：可视化

```bash
python scripts/gas/visualize_individual_results.py
```

**输出**：
- `vis_results/individual_clean/1001_clean_vis.png` - 每个文件的诊断图

### 3.3 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `seq_len` | 256 | 时间窗口长度（约32小时） |
| `step` | 8 | 滑动窗口步长 |
| `num_kernels` | 4 | TimesNet 卷积核数量 |
| `top_k` | 3 | TimesNet 参数 |
| `e_layers` | 2 | 编码器层数 |
| `d_model` | 128 | 模型维度 |
| `threshold` | 99.5% | 异常阈值（训练集MSE分位数） |

---

## 4. 方案2: 流量维度 (CHX00F003FT0101 & CHX00F002FT0101) ⭐ 推荐

### 4.1 数据构建

```bash
python scripts/gas/build_multi_target_datasets.py
```

**特点**：
- 为每个目标维度创建独立的数据集
- 使用"尾部填充"策略避免边界效应
  - 每个正常片段后追加 256 个该片段最后一个值的副本
  - 确保模型训练时不会跨越不连续的片段边界

**输出**：
```
dataset/gas_multi_dim/
├── CHX00F003FT0101/
│   ├── train.csv          # 乌审旗流量训练集
│   ├── test.csv           # 测试集
│   └── train_check.png    # 训练数据可视化（检查是否平稳）
└── CHX00F002FT0101/
    ├── train.csv          # 鄂托克流量训练集
    ├── test.csv
    └── train_check.png
```

### 4.2 训练模型

为每个维度训练独立模型：

**乌审旗流量 (CHX00F003FT0101)**：
```bash
python run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --model_id gas_single_F003 \
  --model TimesNet \
  --data GAS \
  --root_path ./dataset/gas_multi_dim/CHX00F003FT0101 \
  --features S \
  --target CHX00F003FT0101 \
  --enc_in 1 \
  --c_out 1 \
  --seq_len 256 \
  --pred_len 0 \
  --d_model 128 \
  --d_ff 512 \
  --num_kernels 6 \
  --top_k 3 \
  --e_layers 2 \
  --train_epochs 10 \
  --batch_size 32 \
  --des test
```

**鄂托克流量 (CHX00F002FT0101)**：
```bash
python run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --model_id gas_single_F002 \
  --model TimesNet \
  --data GAS \
  --root_path ./dataset/gas_multi_dim/CHX00F002FT0101 \
  --features S \
  --target CHX00F002FT0101 \
  --enc_in 1 \
  --c_out 1 \
  --seq_len 256 \
  --pred_len 0 \
  --d_model 128 \
  --d_ff 512 \
  --num_kernels 6 \
  --top_k 3 \
  --e_layers 2 \
  --train_epochs 10 \
  --batch_size 32 \
  --des test
```

**输出**：
- `checkpoints/anomaly_detection_gas_single_F003_TimesNet_GAS_*/checkpoint.pth`
- `checkpoints/anomaly_detection_gas_single_F002_TimesNet_GAS_*/checkpoint.pth`

### 4.3 推理

```bash
python scripts/gas/test_multi_flow_individual.py
```

**功能**：
- 加载两个训练好的模型
- 对 `data/csv_data/` 中每个文件单独进行异常检测
- 为每个文件生成两个维度的能量结果

**输出**：
```
test_results/individual_file_results/
├── 1001_CHX00F003FT0101_result.npz
├── 1001_CHX00F002FT0101_result.npz
├── 1002_CHX00F003FT0101_result.npz
├── 1002_CHX00F002FT0101_result.npz
└── ...
```

**注意**：推理参数必须与训练参数完全一致（seq_len, num_kernels, top_k 等）！

### 4.4 可视化

```bash
python scripts/gas/visualize_multi_anomaly_v2.py
```

**可选参数**：
```bash
# 使用自定义阈值百分位数
python scripts/gas/visualize_multi_anomaly_v2.py --percentile 99.0   # 更多异常
python scripts/gas/visualize_multi_anomaly_v2.py --percentile 99.9   # 更少异常
```

**输出**：
- `vis_results/multi_dim_v2/1001_full_diagnosis.png`
- `vis_results/multi_dim_v2/1002_full_diagnosis.png`
- ...

### 4.5 可视化输出解读

每个诊断图包含 **4 个子图**：

1. **流量维度全景** (顶部)
   - 灰色线：所有流量通道（完全可见）
   - 绿色/蓝色粗线：两个目标流量（CHX00F002FT0101, CHX00F003FT0101）
   - 彩色背景：异常区域

2. **压力维度全景** (第二)
   - 灰色线：所有压力通道
   - 用于理解上下文（无异常标注）

3. **目标流量对比** (第三)
   - 绿色线：鄂托克流量
   - 蓝色线：乌审旗流量
   - 彩色背景：对应维度的异常区域

4. **重构能量曲线** (底部，对数尺度)
   - 绿色线：鄂托克流量的 MSE
   - 蓝色线：乌审旗流量的 MSE
   - 虚线：各自的阈值
   - 彩色背景：异常区域

**异常判定规则**：
- 绿色背景 → 鄂托克流量异常
- 蓝色背景 → 乌审旗流量异常
- 重叠背景 → 两个维度同时异常

---

## 5. 核心概念解释

### 5.1 边界效应问题

**问题描述**：
最初将所有测试文件拼接成一个大的 `test.csv`，导致拼接处的 MSE 异常高。

**原因**：
```
文件1末尾: ... 100.5, 100.3, 100.1
文件2开头: 50.2, 50.5, 50.8, ...
```
模型根据文件1的数据预测下一个值应该接近 100，但实际对比的是文件2的 50，导致 MSE 暴增。

**解决方案**：
1. **单文件独立测试**：对每个 CSV 文件单独进行异常检测
2. **训练集尾部填充**：每个正常片段后追加 256 个该片段最后值的副本
   - 示例：`[..., 100.1] + [100.1]*256 + [下一个片段开始]`
   - 模型学会在片段边界处预测"平稳状态"

### 5.2 阈值设定

**为什么只用训练集 MSE？**

**错误做法**：合并训练集和测试集的 MSE 计算分位数
```python
combined_energy = np.concatenate([train_energy, test_energy])
threshold = np.percentile(combined_energy, 99.5)  # ❌ 错误
```

**问题**：
- 如果测试集有 99% 的时间都在正常运行
- 且测试集的波动比训练集还小（"超正常"）
- 分位数会被这些数据拉低
- 导致真正的异常点也无法越过阈值

**正确做法**：只用训练集 MSE
```python
threshold = np.percentile(train_energy, 99.5)  # ✅ 正确
```

**逻辑**：
- 训练集代表"已知的正常状态"
- 阈值基于"正常状态的上限"
- 测试集中超过这个上限的就是异常

### 5.3 MSE 行为模式

**问题**：MSE 是会在整个异常区间都很高，还是只有异常出现的瞬间会很高？

**答案**：取决于异常类型

#### 类型1：点异常 (Point Anomalies)
```
正常 → 正常 → 异常! → 正常 → 正常
MSE:  低    低    高     低    低
```
- 例如：传感器故障、瞬时测量错误
- MSE 只在异常点附近高

#### 类型2：持续异常 (Sustained Anomalies)
```
正常 → 正常 → [异常开始 → 异常持续 → 异常持续] → 恢复
MSE:  低    低      高        高        高         低
```
- 例如：设备进入异常运行状态、操作模式改变
- 整个异常区间 MSE 都高

#### 类型3：窗口扩散效应
由于使用滑动窗口 (seq_len=256, step=8)：
```
异常发生在 t=1000
- 窗口 [744:1000] 包含异常 → MSE 高
- 窗口 [752:1008] 包含异常 → MSE 高
- 窗口 [760:1016] 包含异常 → MSE 高
...
- 窗口 [1000:1256] 包含异常 → MSE 高

结果：一个点异常会"扩散"成约 256 个时间点的异常带
```

**在管道异常检测中**：
- 操作性异常（泵启停、阀门调节）通常持续较长时间
- 整个操作期间 MSE 都会偏高

---

## 6. 常见问题

### 6.1 参数不匹配错误

**症状**：
```
RuntimeError: Error(s) in loading state_dict for Model:
Missing key(s) in state_dict: "model.0.conv.0.kernels.4.weight", ...
```

**原因**：推理时的模型参数与训练时不一致。

**解决**：确保以下参数完全一致：
- `seq_len`
- `num_kernels`
- `top_k`
- `e_layers`
- `d_model`
- `d_ff`

### 6.2 步长错位问题

**症状**：可视化时异常区域与实际数据不对应。

**原因**：
- NPZ 文件存储的是窗口级别的能量
- 需要正确的 `step` 参数将其映射回时间点

**解决**：确保 `step=8` 与训练时一致。

### 6.3 GPU 加速

**检查 GPU 可用性**：
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

**启用 GPU**：
在 `test_multi_flow_individual.py` 中：
```python
'use_gpu': True,
'batch_size': 256,  # GPU 可以用更大的 batch size
```

### 6.4 无异常区域显示

**原因1**：阈值过高
- 解决：降低 `--percentile` 参数（如 99.0, 98.0）

**原因2**：测试数据确实没有异常
- 检查训练集和测试集的数据分布

---

## 7. 文件结构

```
TSLib-Custom/
├── data/                          # 原始数据（需手动上传，不在 Git）
│   ├── normal/                    # 正常数据片段
│   │   ├── 1001_normal.csv
│   │   └── ...
│   └── csv_data/                  # 完整测试数据
│       ├── 1001.csv
│       └── ...
│
├── dataset/                       # 构建后的数据集（自动生成，不在 Git）
│   ├── gas_anomaly_single/        # 方案1数据集
│   │   ├── train.csv
│   │   └── test.csv
│   └── gas_multi_dim/             # 方案2数据集
│       ├── CHX00F003FT0101/
│       │   ├── train.csv
│       │   ├── test.csv
│       │   └── train_check.png
│       └── CHX00F002FT0101/
│           ├── train.csv
│           ├── test.csv
│           └── train_check.png
│
├── checkpoints/                   # 训练好的模型（不在 Git）
│   ├── anomaly_detection_gas_single_CHX00L006PT0101_*/
│   │   └── checkpoint.pth
│   ├── anomaly_detection_gas_single_F003_*/
│   │   └── checkpoint.pth
│   └── anomaly_detection_gas_single_F002_*/
│       └── checkpoint.pth
│
├── test_results/                  # 推理结果（不在 Git）
│   └── individual_file_results/
│       ├── 1001_CHX00F003FT0101_result.npz
│       ├── 1001_CHX00F002FT0101_result.npz
│       └── ...
│
├── vis_results/                   # 可视化图片（不在 Git）
│   ├── individual_clean/          # 方案1可视化
│   │   └── 1001_clean_vis.png
│   └── multi_dim_v2/              # 方案2可视化
│       └── 1001_full_diagnosis.png
│
├── scripts/gas/                   # 核心脚本
│   ├── build_single_dimension_dataset.py
│   ├── build_multi_target_datasets.py
│   ├── train_single_dimension.sh
│   ├── test_individual_files.py
│   ├── test_multi_flow_individual.py
│   ├── visualize_individual_results.py
│   └── visualize_multi_anomaly_v2.py
│
├── run.py                         # Time-Series-Library 主入口
├── run_single_dimension_full_pipeline.sh  # 方案1一键脚本
├── requirements.txt
└── README.md
```

---

## 8. 服务器操作流程

### 8.1 首次部署

**步骤1：上传代码**
```bash
# 在本地
git push origin your-branch

# 在服务器
cd /path/to/project
git clone https://github.com/your-repo/TSLib-Custom.git
cd TSLib-Custom
git checkout your-branch
```

**步骤2：上传数据**
```bash
# 从本地上传到服务器
scp -r data/ user@server:/path/to/TSLib-Custom/
scp -r dataset/ user@server:/path/to/TSLib-Custom/  # 如果已构建
```

**步骤3：安装依赖**
```bash
# 在服务器
pip install -r requirements.txt
```

### 8.2 训练模型

```bash
# 在服务器
cd /path/to/TSLib-Custom

# 方案1
bash scripts/gas/train_single_dimension.sh

# 方案2
python run.py --task_name anomaly_detection --is_training 1 \
  --model_id gas_single_F003 --target CHX00F003FT0101 \
  --root_path ./dataset/gas_multi_dim/CHX00F003FT0101 \
  ... (完整参数见第4.2节)
```

### 8.3 推理

```bash
# 方案1
python scripts/gas/test_individual_files.py

# 方案2
python scripts/gas/test_multi_flow_individual.py
```

### 8.4 下载结果到本地

```bash
# 在本地
scp -r user@server:/path/to/TSLib-Custom/test_results/ ./
scp -r user@server:/path/to/TSLib-Custom/checkpoints/ ./  # 如果需要
```

### 8.5 本地可视化

```bash
# 在本地
# 方案1
python scripts/gas/visualize_individual_results.py

# 方案2
python scripts/gas/visualize_multi_anomaly_v2.py --percentile 99.5
```

### 8.6 查看结果

```bash
# 查看可视化图片
open vis_results/multi_dim_v2/1001_full_diagnosis.png
# 或在文件浏览器中打开 vis_results/ 目录
```

---

## 9. 快速参考

### 9.1 方案1（压力）快速命令

```bash
# 一键运行
bash run_single_dimension_full_pipeline.sh

# 或分步运行
python scripts/gas/build_single_dimension_dataset.py
bash scripts/gas/train_single_dimension.sh
python scripts/gas/test_individual_files.py
python scripts/gas/visualize_individual_results.py
```

### 9.2 方案2（流量）快速命令

```bash
# 数据构建
python scripts/gas/build_multi_target_datasets.py

# 训练（需要为每个维度运行）
python run.py --task_name anomaly_detection --is_training 1 \
  --model_id gas_single_F003 --model TimesNet --data GAS \
  --root_path ./dataset/gas_multi_dim/CHX00F003FT0101 \
  --features S --target CHX00F003FT0101 --enc_in 1 --c_out 1 \
  --seq_len 256 --pred_len 0 --d_model 128 --d_ff 512 \
  --num_kernels 6 --top_k 3 --e_layers 2 --train_epochs 10 \
  --batch_size 32 --des test

# 推理与可视化
python scripts/gas/test_multi_flow_individual.py
python scripts/gas/visualize_multi_anomaly_v2.py
```

---

## 10. 进一步阅读

- **详细文档**：查看 `SCRIPTS_INVENTORY.md` 了解所有脚本的详细说明
- **单维度方案详情**：`README_SINGLE_DIMENSION.md`
- **历史方案记录**：`gas_anomaly_pipeline.md`（32维方案，已废弃）
- **Time-Series-Library**：原始项目的 `README.md`

---

## 11. 联系与支持

如有问题，请参考：
1. 本文档的"常见问题"部分（第6节）
2. `SCRIPTS_INVENTORY.md` 中的脚本说明
3. 代码中的注释和文档字符串

**预计上手时间**：30 分钟
- 5分钟：理解项目背景
- 10分钟：准备环境和数据
- 15分钟：运行第一个完整流程

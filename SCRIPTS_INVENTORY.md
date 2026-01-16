# 脚本分类清单

本文档列出了 `scripts/gas/` 目录下所有脚本及其用途分类。

---

## 📋 目录

- [方案1核心脚本](#方案1核心脚本-chx00l006pt0101-压力维度)
- [方案2核心脚本](#方案2核心脚本-chx00f003ft0101--chx00f002ft0101-流量维度)
- [辅助工具脚本](#辅助工具脚本)
- [历史/已废弃脚本](#历史已废弃脚本)
- [根目录文档](#根目录文档)
- [根目录脚本](#根目录脚本)

---

## 方案1核心脚本 (CHX00L006PT0101 压力维度)

### 数据构建

#### `scripts/gas/build_single_dimension_dataset.py`
**用途**: 从 `data/normal` 和 `data/csv_data` 提取 CHX00L006PT0101 维度数据，构建训练集和测试集。

**输入**:
- `data/normal/*.csv` - 正常数据片段
- `data/csv_data/*.csv` - 完整测试数据

**输出**:
- `dataset/gas_anomaly_single/train.csv` - 训练集（拼接所有正常片段）
- `dataset/gas_anomaly_single/test.csv` - 测试集（拼接所有测试数据）

**使用方法**:
```bash
python scripts/gas/build_single_dimension_dataset.py
```

---

### 训练脚本

#### `scripts/gas/train_single_dimension.sh`
**用途**: 标准训练脚本，训练 CHX00L006PT0101 的 TimesNet 异常检测模型。

**关键参数**:
- `seq_len=256`
- `train_epochs=20`
- `d_model=128`, `d_ff=512`
- `e_layers=2`
- `num_kernels=4`, `top_k=3`

**输出**:
- `checkpoints/anomaly_detection_gas_single_CHX00L006PT0101_*/checkpoint.pth`

**使用方法**:
```bash
bash scripts/gas/train_single_dimension.sh
```

#### `scripts/gas/train_single_dimension_quick.sh`
**用途**: 快速训练脚本，用于调试和快速验证。

**与标准版区别**:
- `seq_len=128` (vs 256)
- `train_epochs=5` (vs 20)
- `d_model=64`, `d_ff=256` (vs 128/512)
- `e_layers=1` (vs 2)

**使用方法**:
```bash
bash scripts/gas/train_single_dimension_quick.sh
```

---

### 推理脚本

#### `scripts/gas/test_individual_files.py`
**用途**: 对 `data/csv_data/` 中每个文件单独进行异常检测推理。

**功能**:
- 加载训练好的模型
- 对每个 CSV 文件独立计算重构误差（MSE）
- 避免文件拼接边界效应

**输出**:
- `test_results/individual_file_results/1001_result.npz`
- `test_results/individual_file_results/1002_result.npz`
- ...

**使用方法**:
```bash
python scripts/gas/test_individual_files.py
```

**注意**: 推理参数必须与训练参数一致！

---

### 可视化脚本

#### `scripts/gas/visualize_individual_results.py`
**用途**: 可视化方案1的异常检测结果，生成诊断图。

**功能**:
- 读取 NPZ 结果文件
- 绘制流量曲线 + 异常区域标注
- 绘制压力曲线 + 异常区域标注
- 绘制重构能量曲线 + 阈值线

**输出**:
- `vis_results/individual_clean/1001_clean_vis.png`
- ...

**使用方法**:
```bash
python scripts/gas/visualize_individual_results.py
```

#### `scripts/gas/visualize_full_test_results.py`
**用途**: 可视化完整测试集的异常检测结果（用于调试）。

**功能**:
- 重建完整的测试集数据对齐
- 生成全景诊断图
- 生成前10个文件的详细子图

**输出**:
- `vis_results/*/diagnostic_reconstructed_vis.png`
- `vis_results/*/*_detailed_vis.png`

**使用方法**:
```bash
python scripts/gas/visualize_full_test_results.py \
  --npz_path test_results/.../energy_and_pred.npz \
  --percentile 99.5
```

---

## 方案2核心脚本 (CHX00F003FT0101 & CHX00F002FT0101 流量维度)

### 数据构建

#### `scripts/gas/build_multi_target_datasets.py`
**用途**: 为多个流量维度构建独立的训练集和测试集。

**目标维度**:
- CHX00F003FT0101 (乌审旗热泵站流量)
- CHX00F002FT0101 (鄂托克旗热泵站流量)

**特点**:
- 使用"尾部填充"策略：每个正常片段后追加 256 个该片段最后值的副本
- 避免训练时跨越不连续片段边界
- 自动过滤长度不足的片段

**输出**:
```
dataset/gas_multi_dim/
├── CHX00F003FT0101/
│   ├── train.csv
│   ├── test.csv
│   └── train_check.png  # 训练数据可视化
└── CHX00F002FT0101/
    ├── train.csv
    ├── test.csv
    └── train_check.png
```

**使用方法**:
```bash
python scripts/gas/build_multi_target_datasets.py
```

---

### 训练脚本

**说明**: 方案2使用与方案1相同的训练脚本 `train_single_dimension.sh`，但需要手动修改参数。

**训练 CHX00F003FT0101**:
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

**关键差异**:
- `num_kernels=6` (vs 方案1的4)
- `root_path` 和 `target` 需要分别设置

---

### 推理脚本

#### `scripts/gas/test_multi_flow_individual.py`
**用途**: 对多个流量维度同时进行异常检测推理。

**功能**:
- 加载两个独立训练的模型（CHX00F003FT0101, CHX00F002FT0101）
- 对每个文件生成多个维度的能量结果
- 支持 GPU 加速

**配置**:
```python
TARGET_CONFIGS = {
    'CHX00F002FT0101': {
        'model_id': 'gas_single_F002',
        'checkpoint_dir': 'anomaly_detection_gas_single_F002_...',
        'root_path': './dataset/gas_multi_dim/CHX00F002FT0101'
    },
    'CHX00F003FT0101': {
        'model_id': 'gas_single_F003',
        'checkpoint_dir': 'anomaly_detection_gas_single_F003_...',
        'root_path': './dataset/gas_multi_dim/CHX00F003FT0101'
    }
}
```

**输出**:
```
test_results/individual_file_results/
├── 1001_CHX00F003FT0101_result.npz
├── 1001_CHX00F002FT0101_result.npz
├── 1002_CHX00F003FT0101_result.npz
├── 1002_CHX00F002FT0101_result.npz
└── ...
```

**使用方法**:
```bash
python scripts/gas/test_multi_flow_individual.py
```

---

### 可视化脚本

#### `scripts/gas/visualize_multi_anomaly_v2.py` ⭐ 最新版
**用途**: 可视化方案2的多维度异常检测结果，生成综合诊断图。

**功能**:
- **4个子图布局**:
  1. 流量维度全景（所有流量通道 + 目标高亮 + 异常区域）
  2. 压力维度全景（所有压力通道）
  3. 目标流量对比（两个目标维度 + 异常区域）
  4. 重构能量曲线（对数尺度 + 阈值 + 异常区域）
- 可配置阈值百分位数
- 自动智能提取文件名

**输出**:
- `vis_results/multi_dim_v2/1001_full_diagnosis.png`
- ...

**使用方法**:
```bash
# 使用默认阈值（99.5%）
python scripts/gas/visualize_multi_anomaly_v2.py

# 使用自定义阈值
python scripts/gas/visualize_multi_anomaly_v2.py --percentile 99.0
python scripts/gas/visualize_multi_anomaly_v2.py --percentile 99.9
```

---

## 辅助工具脚本

### 数据分析与验证

#### `scripts/gas/analyze_operations.py`
**用途**: 分析操作工况数据特征。

**功能**:
- 分析不同操作类型的数据特征
- 观察节点及上下游节点的数据变化
- 生成操作案例的时序图

**使用场景**: 理解数据特征、选择关键维度

**使用方法**:
```bash
python scripts/gas/analyze_operations.py
```

#### `scripts/gas/verify_alignment.py`
**用途**: 验证重建的全维度数据是否与单维度测试集对齐。

**功能**:
- 从 `data/csv_data` 重建完整的 CHX00L006PT0101 数据
- 与 `dataset/gas_anomaly_single/test.csv` 进行对比
- 检查数据长度和内容是否一致

**使用场景**: 调试数据对齐问题

**使用方法**:
```bash
python scripts/gas/verify_alignment.py
```

#### `scripts/gas/verify_data_integrity.py`
**用途**: 验证原始数据的完整性和统计特征。

**功能**:
- 读取指定 CSV 文件的目标列
- 打印统计信息（min, max, mean, std）
- 检查数据是否未被处理

**使用场景**: 确认数据未被意外修改

**使用方法**:
```bash
python scripts/gas/verify_data_integrity.py
```

---

### 可视化辅助

#### `scripts/gas/visualize_normal_data.py`
**用途**: 可视化 `data/normal_data.csv` 中的 CHX00L006PT0101 维度。

**使用场景**: 检查正常数据的分布和趋势

**输出**:
- `vis_results/gas/normal_data_trend.png`

**使用方法**:
```bash
python scripts/gas/visualize_normal_data.py
```

#### `scripts/gas/visualize_normal_data_new_dims.py`
**用途**: 可视化 `data/normal_data.csv` 中的流量维度（CHX00F003FT0101, CHX00F002FT0101）。

**使用场景**: 检查流量维度的正常数据分布

**输出**:
- `vis_results/gas/normal_data_flow_dims.png`

**使用方法**:
```bash
python scripts/gas/visualize_normal_data_new_dims.py
```

#### `scripts/gas/visualize_train_trend.py`
**用途**: 可视化 `dataset/gas_anomaly_single/train.csv` 的趋势。

**功能**:
- 绘制原始训练数据
- 绘制300点移动平均

**使用场景**: 检查训练数据是否平稳，是否有异常波动

**输出**:
- `vis_results/gas/train_data_trend.png`

**使用方法**:
```bash
python scripts/gas/visualize_train_trend.py
```

#### `scripts/gas/visualize_with_smoothing.py`
**用途**: 可视化 `data/csv_data` 的所有维度，支持中值滤波平滑。

**功能**:
- 绘制所有32个维度
- 高亮特定目标维度（CHX00F003FT0101, CHX00F002FT0101）
- 分离流量和压力到不同子图
- 可选中值滤波平滑

**输出**:
- `vis_results/origin/*.png` - 原始数据可视化
- `vis_results/smoothed/*.png` - 平滑后数据可视化

**使用方法**:
```bash
python scripts/gas/visualize_with_smoothing.py
```

---

## 历史/已废弃脚本

以下脚本保留用于参考，但不再用于当前的主要工作流程。

### 32维异常检测方案（已废弃）

#### `scripts/gas/build_gas_anomaly_dataset.py`
**用途**: 构建32维异常检测数据集。

**废弃原因**: 32维方案效果不好，传感器坏点被稀释。

**替代方案**: 方案1/方案2的单维度检测

---

#### `scripts/gas/run_univariate_timesnet.py`
**用途**: 批量运行32个单变量 TimesNet 异常检测实验。

**废弃原因**: 计算成本过高，且不如直接选择关键维度。

---

#### `scripts/gas/fuse_univariate_results.py`
**用途**: 融合32个单变量结果（逻辑OR）。

**废弃原因**: 与 `run_univariate_timesnet.py` 配套使用，已废弃。

---

### 旧版可视化

#### `scripts/gas/visualize_multi_anomaly.py`
**用途**: 旧版多维度异常可视化。

**废弃原因**: 功能被 `visualize_multi_anomaly_v2.py` 替代，新版支持：
- 流量/压力分离
- 可配置阈值
- 更清晰的4子图布局

---

### 旧版推理脚本

#### `scripts/gas/inference_single_file.py`
**用途**: 单文件推理（旧版）。

**废弃原因**: 功能被 `test_individual_files.py` 和 `test_multi_flow_individual.py` 替代。

---

#### `scripts/gas/run_individual_detection.sh`
**用途**: 旧版个体检测脚本。

**废弃原因**: 流程已整合到新的脚本中。

---

#### `scripts/gas/run_test_only.sh`
**用途**: 仅运行测试（快速版本）。

**废弃原因**: 功能已整合到主流程中。

---

#### `scripts/gas/run_test_only_standard.sh`
**用途**: 仅运行测试（标准版本）。

**废弃原因**: 功能已整合到主流程中。

---

## 根目录文档

### 当前方案文档

#### `GAS_ANOMALY_DETECTION_QUICKSTART.md` ⭐ 最新
**用途**: 快速上手交接文档。

**内容**:
- 项目背景与思路
- 两套方案的详细使用方法
- 核心概念解释
- 常见问题
- 服务器操作流程

**适用人群**: 新接手项目的开发者

---

#### `SCRIPTS_INVENTORY.md` (本文档)
**用途**: 脚本分类清单，列出所有脚本及其用途。

---

#### `START_HERE.md`
**用途**: 项目总体入口文档。

**建议**: 指向 `GAS_ANOMALY_DETECTION_QUICKSTART.md`

---

### 方案1（压力）文档

#### `QUICKSTART_SINGLE_DIMENSION.md`
**用途**: 方案1快速开始指南。

---

#### `README_SINGLE_DIMENSION.md`
**用途**: 方案1详细说明。

---

#### `single_dimension_pipeline.md`
**用途**: 方案1流程说明。

---

#### `SINGLE_DIMENSION_SUMMARY.md`
**用途**: 方案1总结。

---

#### `COMMAND_CHEATSHEET.md`
**用途**: 方案1命令速查表。

---

### 历史文档（已废弃）

#### `gas_anomaly_pipeline.md`
**用途**: 32维异常检测流程说明。

**废弃原因**: 32维方案已废弃。

**保留价值**: 记录演进过程，理解为什么改用单维度。

---

#### `gas_anomaly_issues.md`
**用途**: 32维方案遇到的问题记录。

**保留价值**: 记录问题和解决思路，避免重复踩坑。

---

### Time-Series-Library 原始文档

#### `README.md`
**用途**: Time-Series-Library 原始项目的主 README。

**内容**: 各种时序任务和模型的使用方法。

---

#### `README_zh.md`
**用途**: Time-Series-Library 中文 README。

---

#### `CONTRIBUTING.md`
**用途**: Time-Series-Library 贡献指南。

---

## 根目录脚本

### `run_single_dimension_full_pipeline.sh`
**用途**: 方案1的一键运行脚本。

**功能**:
1. 检查数据集是否存在
2. 提示选择训练类型（标准/快速）
3. 执行训练
4. 进行推理
5. 生成可视化结果

**使用方法**:
```bash
bash run_single_dimension_full_pipeline.sh
```

**交互示例**:
```
请选择训练类型:
1) 标准训练 (20 epochs, seq_len=256)
2) 快速训练 (5 epochs, seq_len=128)
请输入选择 (1 或 2):
```

---

### `run.py`
**用途**: Time-Series-Library 主入口脚本。

**功能**: 根据参数调用不同的实验类（训练、测试、预测等）。

**使用方法**: 通过命令行参数指定任务和模型：
```bash
python run.py --task_name anomaly_detection --is_training 1 \
  --model TimesNet --data GAS --target CHX00F003FT0101 ...
```

---

## 快速查找索引

### 按用途查找

| 用途 | 脚本 |
|------|------|
| **构建方案1数据集** | `build_single_dimension_dataset.py` |
| **构建方案2数据集** | `build_multi_target_datasets.py` |
| **训练方案1模型** | `train_single_dimension.sh` |
| **训练方案2模型** | 使用 `run.py` + 完整参数 |
| **方案1推理** | `test_individual_files.py` |
| **方案2推理** | `test_multi_flow_individual.py` |
| **方案1可视化** | `visualize_individual_results.py` |
| **方案2可视化** | `visualize_multi_anomaly_v2.py` |
| **方案1一键运行** | `run_single_dimension_full_pipeline.sh` |
| **检查训练数据** | `visualize_train_trend.py` |
| **检查正常数据** | `visualize_normal_data.py`, `visualize_normal_data_new_dims.py` |
| **数据对齐验证** | `verify_alignment.py` |
| **数据完整性验证** | `verify_data_integrity.py` |

---

### 按方案查找

#### 方案1 (CHX00L006PT0101 压力)

1. 数据构建: `build_single_dimension_dataset.py`
2. 训练: `train_single_dimension.sh` (或 `train_single_dimension_quick.sh`)
3. 推理: `test_individual_files.py`
4. 可视化: `visualize_individual_results.py`
5. 一键运行: `run_single_dimension_full_pipeline.sh`

#### 方案2 (CHX00F003FT0101 & CHX00F002FT0101 流量) ⭐ 推荐

1. 数据构建: `build_multi_target_datasets.py`
2. 训练: `run.py` (手动指定参数)
3. 推理: `test_multi_flow_individual.py`
4. 可视化: `visualize_multi_anomaly_v2.py`

---

## 更新日志

### 2024-01 最新更新
- ✅ 创建 `GAS_ANOMALY_DETECTION_QUICKSTART.md` - 快速上手交接文档
- ✅ 创建 `SCRIPTS_INVENTORY.md` - 本脚本清单文档
- ✅ 更新 `visualize_multi_anomaly_v2.py` - 4子图布局，流量/压力分离
- ✅ 更新 `test_multi_flow_individual.py` - 移除 CHX00L006PT0101，只保留流量维度
- ✅ 更新 `.gitignore` - 忽略数据目录

### 2024-01 方案2实现
- ✅ 实现 `build_multi_target_datasets.py` - 多目标数据构建（尾部填充策略）
- ✅ 实现 `test_multi_flow_individual.py` - 多维度推理
- ✅ 实现 `visualize_multi_anomaly_v2.py` - 多维度可视化

### 2023-12 方案1实现
- ✅ 实现 `build_single_dimension_dataset.py` - 单维度数据构建
- ✅ 实现 `train_single_dimension.sh` - 训练脚本
- ✅ 实现 `test_individual_files.py` - 单文件独立推理
- ✅ 实现 `visualize_individual_results.py` - 可视化

---

## 结语

本文档将持续更新，以反映最新的脚本和工作流程。

**推荐使用**: 方案2（流量维度），更敏感，效果更好。

**快速开始**: 查看 `GAS_ANOMALY_DETECTION_QUICKSTART.md`

**问题反馈**: 请在代码中添加注释或更新相关文档。

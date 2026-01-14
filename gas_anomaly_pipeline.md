## 天然气 TimesNet 异常检测全流程（从数据到可视化）

### 1. 本地数据准备

假设已经把原始 CSV 放到：

- 正常片段：`data/normal/*.csv`（格式：`日期 + 32 列测点`）
- 完整样本：`data/csv_data/*.csv`（同样格式）

在项目根目录执行：

```bash
python scripts/gas/build_gas_anomaly_dataset.py
```

输出：

- `dataset/gas_anomaly/train.csv`：拼接后的正常数据（过滤长度 <150 的片段）
- `dataset/gas_anomaly/test.csv`：拼接后的完整运行数据

### 2. 上传到服务器

把整个项目（或至少这几个目录）同步到服务器某个路径，例如 `/data/TSLib-Custom`：

- `dataset/gas_anomaly/`
- `scripts/gas/`
- `exp/`, `models/`, `data_provider/`, `utils/`, `run.py`, `requirements.txt` 等

### 2.1 服务器上切换到新分支并拉取代码

在服务器上（首次使用该仓库时先 clone，此处省略），进入项目目录，并从远端创建本地分支：

```bash
cd /data/TSLib-Custom
git fetch origin
git checkout -b test-gas-anomaly-detection origin/test-gas-anomaly-detection
```

（具体用 `scp/rsync` 或公司内部工具，这里不赘述。）

### 3. 在服务器上运行 TimesNet

#### 3.1 多变量方案（32 维一起）

在服务器项目根目录：

```bash
python run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --model_id gas_timesnet_32d \
  --model TimesNet \
  --data GAS \
  --root_path ./dataset/gas_anomaly \
  --features M \
  --enc_in 32 --c_out 32 \
  --seq_len 256 \
  --pred_len 0 \
  --anomaly_ratio 1 \
  --batch_size 32 \
  --d_model 256 \
  --d_ff 1024 \
  --e_layers 1 \
  --top_k 3 \
  --num_kernels 4 \
  --patience 3 \
  --train_epochs 10 \
  --checkpoints ./checkpoints/gas_timesnet \
  --des gas_32d_small \
  --use_amp
```

输出：

- 模型：`checkpoints/gas_timesnet/<setting>/checkpoint.pth`
- 结果：`test_results/<setting>/energy_and_pred.npz`
  - 内含 `train_energy, test_energy, threshold, pred, seq_len`

#### 3.2 单变量方案（32 个模型，每个节点一个）

在服务器项目根目录：

```bash
python scripts/gas/run_univariate_timesnet.py
```

脚本会自动为 `GAS_FIELDS` 里的 32 个字段依次执行：

- `features='S'`
- `enc_in=1, c_out=1`
- `target=<字段名>`

输出：

- 各字段模型：`checkpoints/gas_timesnet/` 下对应 `gas_univariate_<字段名>` 的 setting 目录
- 各字段结果：`test_results/anomaly_detection_gas_univariate_<字段名>_TimesNet_GAS.../energy_and_pred.npz`

#### 3.3 单变量结果 OR 融合

```bash
python scripts/gas/fuse_univariate_results.py
```

输出：

- `test_results/gas_univariate_fusion/global_fusion.npz`
  - `global_pred`: 每个时间点 OR 后的全局异常标签（0/1）
  - `num_active`: 每个时间点被多少个节点判为异常

### 4. 从服务器下载到本机

从服务器把以下目录下载到本地同样的项目路径：

- `checkpoints/gas_timesnet/**`
- `test_results/**`

保持相对路径一致，方便可视化脚本直接使用。

### 5. 本机可视化（流量/压力上下子图，标注异常区间）

选择一个完整样本文件（例如 `data/csv_data/1004.csv`）：

#### 5.1 使用多变量 TimesNet 的输出可视化

假设多变量 setting 为 `anomaly_detection_gas_timesnet_32d_TimesNet_GAS_ftM_...`，结果文件为：

- `test_results/anomaly_detection_gas_timesnet_32d_TimesNet_GAS_ftM_.../energy_and_pred.npz`

执行：

```bash
python utils/visualize_gas_anomaly.py \
  --raw_csv data/csv_data/1004.csv \
  --result_file test_results/anomaly_detection_gas_timesnet_32d_TimesNet_GAS_ftM_.../energy_and_pred.npz \
  --flow_tag CHX00L006FT0101 \
  --pressure_tag CHX00L006PT0101 \
  --seq_len 256
```

输出图片：

- `vis_results/gas/1004_energy_and_pred_flow-CHX00L006FT0101_pressure-CHX00L006PT0101.png`
- 上图：呼和浩特末站流量，下图：呼和浩特末站压力，异常区间为红色背景。

#### 5.2 使用 32 个单变量 OR 融合结果可视化

```bash
python utils/visualize_gas_anomaly.py \
  --raw_csv data/csv_data/1004.csv \
  --result_file test_results/gas_univariate_fusion/global_fusion.npz \
  --flow_tag CHX00L006FT0101 \
  --pressure_tag CHX00L006PT0101 \
  --seq_len 256
```

同样会在 `vis_results/gas/` 下生成带异常区间高亮的双子图。



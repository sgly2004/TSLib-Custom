## 天然气 TimesNet 异常检测问题记录

### 1. TimesBlock.forward 时 reshape 尺寸不匹配

**报错信息简要：**

- 核心异常：
  - `RuntimeError: shape '[32, 2, 256, 512]' is invalid for input of size 6815744`
  - 发生位置：`models/TimesNet.py` → `TimesBlock.forward` 中的：
    - `out = out.reshape(B, length // period, period, N)`

**原因分析：**

- 异常检测任务下，DataLoader 提供的输入序列长度为 `T = seq_len`（例如 256），没有预测长度。
- 但 `TimesBlock` 内部使用的是 `self.seq_len + self.pred_len` 来计算 `length` 和 padding：
  - 当命令行没有显式设置 `--pred_len` 时，`run.py` 的默认值是 `pred_len=96`。
  - 于是内部逻辑假设序列长度是 `seq_len + pred_len = 256 + 96 = 352`。
- 实际输入 `x` 的时间长度只有 256，与 352 不一致，padding 后的 `out` 的元素总数和希望 reshape 成的形状不对齐，导致 reshape 抛错。

**解决办法：**

1. 在异常检测任务下，把 `pred_len` 显式设为 0，使内部长度计算与真实输入一致：

   ```bash
   python run.py \
     --task_name anomaly_detection \
     ... \
     --seq_len 256 \
     --pred_len 0 \
     ...
   ```

2. （可选的代码级修正思路）将 `TimesBlock.forward` 内部所有使用 `self.seq_len + self.pred_len` 的地方，替换为当前输入的实际长度 `T`：

   ```python
   def forward(self, x):
       B, T, N = x.size()
       period_list, period_weight = FFT_for_Period(x, self.k)
       ...
       if T % period != 0:
           length = (T // period + 1) * period
           padding = torch.zeros([B, length - T, N]).to(x.device)
           out = torch.cat([x, padding], dim=1)
       else:
           length = T
           out = x
       out = out.reshape(B, length // period, period, N)
   ```

目前采用方案 1（命令行设置 `--pred_len 0`）即可避免该错误。

---

### 2. CUDA Out Of Memory（显存不足）

**报错信息简要：**

- 核心异常：
  - `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 484.00 MiB. ...`
  - 发生位置：`exp/exp_anomaly_detection.py` → 训练阶段 `model_optim.step()` 时，Adam 优化器在做多张量更新。
- 日志显示：
  - GPU 总显存约 24GB，仅剩约 388MB 空闲；
  - 当前进程已占用约 22GB（其中 21.9GB 为 PyTorch 分配）。

**当时的模型与训练配置（大致）：**

- `features='M'`（32 维输入）
- `seq_len=256`
- `enc_in=32, c_out=32`
- `d_model=512`
- `d_ff=2048`
- `e_layers=2`
- `batch_size=32`
- 未开启 AMP（`--use_amp`）

在上述配置下，TimesNet 本身结构较重，再加上较大的 batch 和通道数，显存开销很高；同时，GPU 上还有其他进程占用显存，剩余空间不足以支撑 Adam 再申请新的 buffer，导致 OOM。

**解决办法：**

1. 调小 batch size（最直接、最有效）：

   ```bash
   --batch_size 8
   ```

2. 适当减小模型规模：

   ```bash
   --d_model 256 \
   --d_ff 1024 \
   --e_layers 1 \
   --top_k 3 \
   --num_kernels 4
   ```

3. 若环境支持，开启混合精度训练以进一步节省显存：

   ```bash
   --use_amp
   ```

**最终采用的稳定配置示例：**

文档 `gas_anomaly_pipeline.md` 中的推荐命令最初调整为（更保守的显存配置）：

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
  --batch_size 8 \
  --d_model 256 \
  --d_ff 1024 \
  --e_layers 1 \
  --top_k 3 \
  --num_kernels 4 \
  --train_epochs 10 \
  --checkpoints ./checkpoints/gas_timesnet \
  --des gas_32d_small \
  --use_amp
```

这套参数在 24GB 显存的 GPU 上，结合当前数据规模，训练 TimesNet 异常检测能显著降低 OOM 风险。

**后续为加速训练的调优：**

- 在确认显存余量足够的前提下，将多变量方案的 `batch_size` 提升到 `64`，进一步提升吞吐量；
- 同时在 `GasSegLoader` 中将滑动窗口步长 `step` 的默认值从 `1` 调整为 `4`：

  ```python
  class GasSegLoader(Dataset):
      def __init__(self, args, root_path, win_size, step=4, flag="train"):
          ...
  ```

- 理论上：
  - `step` 从 `1` 改为 `4`，窗口数量约为原来的 `1/4`，单个 epoch 的 batch 数和耗时也大约降到 `1/4`；
  - 在显存允许的条件下，将 `batch_size` 从 `8` 提升到 `64`，可以进一步减少每个 epoch 的 batch 数，加快训练。
- 代价是：
  - 训练看到的窗口更稀疏（重叠减少），时间覆盖仍然完整但细粒度程度降低；
  - `batch_size` 增大需要注意显存占用情况，必要时可在 OOM 时退回较小的 batch 或增大步长。




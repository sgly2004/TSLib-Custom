import numpy as np
import pandas as pd
import torch
from torch import nn
import argparse
import matplotlib.pyplot as plt

from exp.exp_anomaly_detection import Exp_Anomaly_Detection


def load_new_csv(csv_path):
    """
    读取一个新的 CSV:
    - 第 1 列为时间戳/索引（丢弃）
    - 后面列为特征
    返回: data (T, D)
    """
    df = pd.read_csv(csv_path, header=None)
    # 丢弃第一列时间戳
    data = df.iloc[:, 1:].values.astype(np.float32)
    return data


def slide_windows(data, win_size, step):
    """
    对 (T, D) 的数据做滑窗:
    返回:
    - windows: (num_windows, win_size, D)
    - starts:  每个窗口在原始序列中的起点索引列表
    """
    T, D = data.shape
    starts = list(range(0, max(T - win_size + 1, 0), step))
    windows = []
    for s in starts:
        e = s + win_size
        if e > T:
            break
        windows.append(data[s:e])
    if not windows:
        return np.zeros((0, win_size, D), dtype=np.float32), []
    windows = np.stack(windows, axis=0)  # (num_windows, win_size, D)
    return windows, starts


def compute_window_energy(model, device, windows, batch_size=64):
    """
    使用已训练模型，计算每个窗口的重构误差 (窗口级分数)
    windows: (num_windows, win_size, D)
    返回: energy (num_windows,)
    """
    model.eval()
    criterion = nn.MSELoss(reduce=False)

    num_windows, win_size, D = windows.shape
    all_scores = []

    with torch.no_grad():
        for i in range(0, num_windows, batch_size):
            batch = windows[i:i + batch_size]                 # (B, win_size, D)
            batch_x = torch.from_numpy(batch).float().to(device)
            outputs = model(batch_x, None, None, None)        # (B, win_size, D) 近似

            loss_ts = criterion(batch_x, outputs)             # (B, win_size, D)
            loss_ts = loss_ts.mean(dim=-1)                    # (B, win_size) 先在特征维平均
            loss_win = loss_ts.mean(dim=-1)                   # (B,) 再在时间维平均：窗口级分数

            all_scores.append(loss_win.cpu().numpy())

    energy = np.concatenate(all_scores, axis=0)               # (num_windows,)
    return energy


def window_to_pointwise(energy, starts, win_size, T, threshold):
    """
    将窗口级分数映射回时间轴，得到每个时间点的分数+标签。
    energy : (num_windows,)
    starts : 每个窗口起点索引
    win_size: 窗口长度
    T      : 原始序列长度
    threshold: 异常阈值（窗口级与时间步都用这个比较）

    返回:
    - score_t: (T,) 每个时间点的分数（平均后的）
    - pred_t : (T,) 每个时间点的标签 0/1
    """
    score_sum = np.zeros(T, dtype=float)
    score_cnt = np.zeros(T, dtype=float)

    for e, s in zip(energy, starts):
        e_idx = min(s + win_size, T)
        score_sum[s:e_idx] += e
        score_cnt[s:e_idx] += 1.0

    valid = score_cnt > 0
    score_t = np.zeros(T, dtype=float)
    score_t[valid] = score_sum[valid] / score_cnt[valid]

    pred_t = (score_t > threshold).astype(int)
    return score_t, pred_t


def find_anomaly_segments(pred_t, min_len=1):
    """
    从逐时间点标签中找出连续异常段 [start, end)
    min_len: 最小异常段长度（小于这个长度的段可以认为是噪声，过滤掉）
    """
    T = len(pred_t)
    segments = []
    in_seg = False
    start = 0
    for t in range(T):
        if pred_t[t] == 1 and not in_seg:
            in_seg = True
            start = t
        elif pred_t[t] == 0 and in_seg:
            in_seg = False
            end = t
            if end - start >= min_len:
                segments.append((start, end))
    if in_seg:
        end = T
        if end - start >= min_len:
            segments.append((start, end))
    return segments


def plot_anomaly_result(
    data,
    score_t,
    pred_t,
    segments,
    save_path="new_series_anomaly_plot.png",
):
    """
    可视化推理结果：
    - 子图1：原始某个（或若干个）特征随时间的变化
    - 子图2：异常分数 score_t + 异常标签 pred_t，并用背景色区分正常/操作段
    """
    T, D = data.shape
    x = np.arange(T)

    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle("TimesNet 异常检测结果", fontsize=16, fontweight="bold")

    # 1) 原始特征（这里只画前 3 个维度，避免太拥挤）
    max_feat_to_plot = min(3, D)
    for i in range(max_feat_to_plot):
        axes[0].plot(x, data[:, i], label=f"feature_{i}", linewidth=1.0)
    axes[0].set_ylabel("特征值", fontsize=12)
    axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # 用背景色标出异常（操作）段
    for s, e in segments:
        axes[0].axvspan(s, e, color="red", alpha=0.15)

    # 2) 分数 + 标签
    axes[1].plot(x, score_t, label="anomaly_score", color="tab:blue", linewidth=1.0)
    axes[1].set_ylabel("异常分数", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # pred_t=1 的点高亮（操作段）
    axes[1].scatter(
        x[pred_t == 1],
        score_t[pred_t == 1],
        color="red",
        s=6,
        label="operation (pred=1)",
        alpha=0.6,
    )
    axes[1].set_xlabel("时间步 t", fontsize=12)
    axes[1].legend(loc="best", fontsize=9)

    # 同样用背景色区分异常段
    for s, e in segments:
        axes[1].axvspan(s, e, color="red", alpha=0.1)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved anomaly visualization to {save_path}")


def main():
    # ===== 通过命令行参数选择输入 CSV 和模型 checkpoint =====
    parser = argparse.ArgumentParser(description="Use trained TimesNet to detect anomalies on a new CSV series")
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="路径：待检测的 CSV 文件，例如 ./dataset/raw_from_csv_data/1001.csv",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="路径：训练好的模型 checkpoint，例如 ./checkpoints/<setting>/checkpoint.pth",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=300,
        help="窗口长度，需与训练时的 seq_len 一致（默认 300）",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="滑窗步长，默认 1",
    )
    parser.add_argument(
        "--enc_in",
        type=int,
        default=32,
        help="输入特征维度，需与训练时 enc_in 一致（默认 32）",
    )
    parser.add_argument(
        "--c_out",
        type=int,
        default=32,
        help="输出特征维度，通常与 enc_in 一致（默认 32）",
    )
    parser.add_argument(
        "--anomaly_ratio",
        type=float,
        default=1.0,
        help="用于确定阈值的百分位（与训练阶段保持一致，默认 1.0）",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="前向推理时的 batch_size（默认 64）",
    )
    args_cli = parser.parse_args()

    csv_path = args_cli.csv_path
    checkpoint_path = args_cli.checkpoint_path

    seq_len = args_cli.seq_len      # 训练时的 seq_len / win_size
    step = args_cli.step            # 滑窗步长
    enc_in = args_cli.enc_in        # 输入维度（特征数）
    c_out = args_cli.c_out          # 输出维度（一般与 enc_in 一致）

    # 与训练时相同的模型配置
    class Args:
        pass

    args = Args()
    args.model = "TimesNet"
    args.task_name = "anomaly_detection"
    args.data = "anomaly-detection-normal-300"
    args.features = "M"
    args.seq_len = seq_len
    args.pred_len = 0
    args.enc_in = enc_in
    args.c_out = c_out
    args.d_model = 32
    args.d_ff = 64
    args.e_layers = 2
    args.d_layers = 1
    args.top_k = 3
    args.anomaly_ratio = args_cli.anomaly_ratio  # 这里仅用于阈值的百分位

    # 其他 Exp_Anomaly_Detection 里需要但此处无关紧要的字段，给个合理默认值即可
    args.batch_size = args_cli.batch_size
    args.num_workers = 4
    args.use_gpu = torch.cuda.is_available()
    args.gpu = 0
    args.gpu_type = "cuda"
    args.checkpoints = "./checkpoints"   # 虽然不再训练，但类里可能会用到
    args.embed = "timeF"
    args.distil = True
    args.freq = "h"
    args.label_len = 48
    args.target = "OT"
    args.seasonal_patterns = "Yearly"
    args.augmentation_ratio = 0
    # 设备相关补全：避免 AttributeError
    args.use_multi_gpu = False
    args.device_ids = [args.gpu]

    device = torch.device("cuda:0" if args.use_gpu and torch.cuda.is_available() else "cpu")

    # ===== 1. 读取新 CSV =====
    data = load_new_csv(csv_path)   # (T, D)
    T, D = data.shape
    print(f"Loaded new series: T={T}, D={D}")

    if D != enc_in:
        raise ValueError(f"CSV 特征维度 D={D} 与 enc_in={enc_in} 不一致，请检查。")

    # ===== 2. 归一化 =====
    # 理想情况下，这里应该加载你训练时保存的 scaler（mean/std）
    # 这里简单示意：假设新序列绝大部分是正常的，用它本身估计 mean/std
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True) + 1e-6
    data_norm = (data - mean) / std

    # ===== 3. 滑窗切分 =====
    windows, starts = slide_windows(data_norm, win_size=seq_len, step=step)
    print(f"Num windows: {len(windows)}")

    if len(windows) == 0:
        print("序列太短，无法形成一个完整窗口。")
        return

    # ===== 4. 加载模型 =====
    exp = Exp_Anomaly_Detection(args)
    exp.model.to(device)
    state = torch.load(checkpoint_path, map_location=device)
    exp.model.load_state_dict(state)
    exp.model.eval()
    model = exp.model

    # ===== 5. 计算窗口级 energy =====
    energy = compute_window_energy(model, device, windows, batch_size=args.batch_size)
    print("Window energy shape:", energy.shape)

    # ===== 6. 阈值（这里用当前这条序列的百分位数；更严格的是用训练集分布） =====
    threshold = np.percentile(energy, 100 - args.anomaly_ratio)
    print("Threshold:", threshold)

    # ===== 7. 窗口级 → 时间步级 分数+标签 =====
    score_t, pred_t = window_to_pointwise(energy, starts, seq_len, T, threshold)
    print("score_t shape:", score_t.shape)
    print("pred_t  shape:", pred_t.shape)
    print("异常比例:", pred_t.mean())

    # ===== 8. 找出连续的异常段 =====
    segments = find_anomaly_segments(pred_t, min_len=1)
    print("Detected anomaly segments (start, end):")
    for s, e in segments:
        print(s, e)

    # 你也可以把 score_t / pred_t 存成 CSV 方便后续分析
    out_df = pd.DataFrame({
        "score": score_t,
        "pred": pred_t,
    })
    out_df.to_csv("new_series_anomaly_result.csv", index_label="t")
    print("Saved result to new_series_anomaly_result.csv")

    # ===== 9. 可视化推理结果 =====
    plot_anomaly_result(
        data=data,
        score_t=score_t,
        pred_t=pred_t,
        segments=segments,
        save_path="new_series_anomaly_plot.png",
    )


if __name__ == "__main__":
    main()
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FIELD_MAPPING = {
    'CHX00E005PT0101': '5#阀室进站压力',
    'CHX00E005PT0102': '5#阀室出站压力',
    'CHX00L006FT0101': '呼和浩特末站流量',
    'CHX00L006PT0101': '呼和浩特末站压力',
    'CHX00A001FT0102': '油房庄首站流量',
    'CHX00A001PT0102': '油房庄首站压力',
    'CHX00A001PT0411': '油房庄首站泵入口压力',
    'CHX00A001PT0412': '油房庄首站泵出口压力',
    'CHX00E009PT0101': '9#阀室进站压力',
    'CHX00E009PT0102': '9#阀室出站压力',
    'CHX00F005FT0101': '土默特右旗热泵站流量',
    'CHX00F005PT0101': '土默特右旗热泵站进站压力',
    'CHX00F005PT0102': '土默特右旗热泵站出站压力',
    'CHX00G004FT0101': '达拉特旗热站流量',
    'CHX00G004PT0101': '达拉特旗热站进站压力',
    'CHX00G004PT0102': '达拉特旗热站出站压力',
    'CHX00F002FT0101': '鄂托克旗热泵站流量',
    'CHX00F002PT0101': '鄂托克旗热泵站进站压力',
    'CHX00F002PT0102': '鄂托克旗热泵站出站压力',
    'CHX00F002PT0409': '鄂托克旗热泵站泵入口压力',
    'CHX00F002PT0410': '鄂托克旗热泵站泵出口压力',
    'CHX00E017PT0101': '17#阀室进站压力',
    'CHX00E017PT0102': '17#阀室出站压力',
    'CHX00E013PT0101': '13#阀室进站压力',
    'CHX00E013PT0102': '13#阀室出站压力',
    'CHX00E003PT0101': '3#阀室进站压力',
    'CHX00E003PT0102': '3#阀室出站压力',
    'CHX00F003FT0101': '乌审旗热泵站流量',
    'CHX00F003PT0101': '乌审旗热泵站进站压力',
    'CHX00F003PT0102': '乌审旗热泵站出站压力',
    'CHX00F003PT0411': '乌审旗热泵站泵入口压力',
    'CHX00F003PT0412': '乌审旗热泵站站出口压力',
}


def _load_pred_vector(result_file: str, seq_len: int | None = None) -> np.ndarray:
    """
    从 energy_and_pred.npz 或 global_fusion.npz 中读取一维 0/1 序列。
    目前：
    - 多变量 / 单变量原始模型：使用 energy_and_pred.npz 中的 pred（一维，对窗口进行预测）
    - 单变量 OR 融合：使用 global_fusion.npz 中的 global_pred
    为了简化，对齐策略采用“窗口最后一个点”，并在可视化时将窗口预测展开到对应的时间点。
    """
    data = np.load(result_file)
    if "global_pred" in data:
        pred = data["global_pred"].astype(int)
    elif "pred" in data:
        pred = data["pred"].astype(int)
    else:
        raise ValueError(f"Unknown prediction keys in {result_file}, expected 'pred' or 'global_pred'.")

    pred = pred.reshape(-1)
    if seq_len is not None and "seq_len" in data:
        # 当前实现中 seq_len 主要用于后续扩展，这里暂不做进一步重采样
        pass
    return pred


def _expand_window_pred_to_points(pred_win: np.ndarray, total_len: int, win_size: int) -> np.ndarray:
    """
    将按滑动窗口得到的预测（一维，长度为 N_win）展开为按时间点的预测（长度为 total_len）。
    简单规则：
    - 假定窗口步长为 1
    - 对每个窗口，预测标签作用于窗口的最后一个时间点
    - 对前 win_size-1 个点，用第一个窗口的标签填充
    """
    n_win = pred_win.shape[0]
    if total_len != n_win + win_size - 1:
        # 回退：长度不匹配时简单重复/截断到 total_len
        print(f"[WARN] total_len {total_len} != n_win+win_size-1 ({n_win + win_size - 1}), "
              f"fallback to simple repeat/crop.")
        tiled = np.repeat(pred_win, max(1, win_size))
        return tiled[:total_len]

    point_pred = np.zeros((total_len,), dtype=int)

    # 前 win_size-1 个点使用第一个窗口标签
    point_pred[: win_size - 1] = pred_win[0]

    # 从第 win_size-1 个点开始，每个点对应一个窗口的最后位置
    for i in range(n_win):
        t = i + win_size - 1
        if t < total_len:
            point_pred[t] = pred_win[i]

    return point_pred


def visualize(
    raw_csv: str,
    result_file: str,
    flow_tag: str = "CHX00L006FT0101",
    pressure_tag: str = "CHX00L006PT0101",
    seq_len: int = 256,
    save_dir: str = "./vis_results/gas",
    title: str | None = None,
):
    df = pd.read_csv(raw_csv)
    if df.shape[1] < 2:
        raise ValueError(f"raw_csv {raw_csv} has too few columns.")

    cols = list(df.columns)
    # 与构建脚本保持一致：第一列为日期
    cols[0] = "date"
    df.columns = cols

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")
    df = df.reset_index(drop=True)

    if flow_tag not in df.columns or pressure_tag not in df.columns:
        raise ValueError(f"flow_tag or pressure_tag not in columns, got {flow_tag}, {pressure_tag}")

    time_index = df["date"]
    flow = df[flow_tag].values
    pressure = df[pressure_tag].values

    pred_win = _load_pred_vector(result_file, seq_len=seq_len)
    total_len = len(df)
    point_pred = _expand_window_pred_to_points(pred_win, total_len, win_size=seq_len)

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(15, 6))
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)

    # 绘制流量与压力
    ax1.plot(time_index, flow, color="tab:blue", linewidth=1.0, label="Flow")
    ax2.plot(time_index, pressure, color="tab:orange", linewidth=1.0, label="Pressure")

    # 区间着色：根据 point_pred 中的连续段
    in_anom = False
    start_t = None
    for i, flag in enumerate(point_pred):
        if not in_anom and flag == 1:
            in_anom = True
            start_t = time_index.iloc[i]
        elif in_anom and flag == 0:
            end_t = time_index.iloc[i]
            ax1.axvspan(start_t, end_t, color="red", alpha=0.15)
            ax2.axvspan(start_t, end_t, color="red", alpha=0.15)
            in_anom = False
            start_t = None
    # 尾段
    if in_anom and start_t is not None:
        end_t = time_index.iloc[-1]
        ax1.axvspan(start_t, end_t, color="red", alpha=0.15)
        ax2.axvspan(start_t, end_t, color="red", alpha=0.15)

    # 轴标签与图例
    ax1.set_ylabel(FIELD_MAPPING.get(flow_tag, flow_tag))
    ax2.set_ylabel(FIELD_MAPPING.get(pressure_tag, pressure_tag))
    ax2.set_xlabel("Time")

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")

    if title is None:
        base = os.path.basename(raw_csv)
        title = f"Gas anomaly visualization - {base}"
    ax1.set_title(title)

    plt.tight_layout()

    out_name = os.path.splitext(os.path.basename(raw_csv))[0]
    result_base = os.path.splitext(os.path.basename(result_file))[0]
    out_path = os.path.join(
        save_dir,
        f"{out_name}_{result_base}_flow-{flow_tag}_pressure-{pressure_tag}.png",
    )
    plt.savefig(out_path, dpi=150)
    print(f"Saved visualization to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize gas anomaly detection results.")
    parser.add_argument("--raw_csv", type=str, required=True, help="Path to raw csv file.")
    parser.add_argument("--result_file", type=str, required=True,
                        help="Path to npz file (energy_and_pred.npz or global_fusion.npz).")
    parser.add_argument("--flow_tag", type=str, default="CHX00L006FT0101", help="Column name for flow.")
    parser.add_argument("--pressure_tag", type=str, default="CHX00L006PT0101", help="Column name for pressure.")
    parser.add_argument("--seq_len", type=int, default=256, help="Seq_len used during model training.")
    parser.add_argument("--save_dir", type=str, default="./vis_results/gas", help="Output directory.")
    args = parser.parse_args()

    visualize(
        raw_csv=args.raw_csv,
        result_file=args.result_file,
        flow_tag=args.flow_tag,
        pressure_tag=args.pressure_tag,
        seq_len=args.seq_len,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()



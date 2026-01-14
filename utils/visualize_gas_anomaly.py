import os
import argparse
import glob

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
    Load 1D 0/1 sequence from energy_and_pred.npz or global_fusion.npz.
    Currently:
    - Multivariate / univariate original models: use pred from energy_and_pred.npz (1D, window-level prediction)
    - Univariate OR fusion: use global_pred from global_fusion.npz
    For simplicity, alignment strategy uses "last point of window" and expands window predictions to time points during visualization.
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
        # seq_len is mainly for future extension, no further resampling here
        pass
    return pred


def _get_seq_len_from_result(result_file: str) -> int:
    """Read seq_len from result file, return default 256 if not exists"""
    data = np.load(result_file)
    if "seq_len" in data:
        return int(data["seq_len"])
    return 256


def _expand_window_pred_to_points(pred_win: np.ndarray, total_len: int, win_size: int) -> np.ndarray:
    """
    Expand window-based predictions (1D, length N_win) to point-based predictions (length total_len).
    Simple rules:
    - Assume window stride is 1
    - For each window, prediction label applies to the last time point of the window
    - For the first win_size-1 points, use the label from the first window
    """
    n_win = pred_win.shape[0]
    if total_len != n_win + win_size - 1:
        # Fallback: simple repeat/crop to total_len when length mismatch
        print(f"[WARN] total_len {total_len} != n_win+win_size-1 ({n_win + win_size - 1}), "
              f"fallback to simple repeat/crop.")
        tiled = np.repeat(pred_win, max(1, win_size))
        return tiled[:total_len]

    point_pred = np.zeros((total_len,), dtype=int)

    # First win_size-1 points use label from first window
    point_pred[: win_size - 1] = pred_win[0]

    # From win_size-1 point onwards, each point corresponds to the last position of a window
    for i in range(n_win):
        t = i + win_size - 1
        if t < total_len:
            point_pred[t] = pred_win[i]

    return point_pred


def _check_anomaly_status(point_pred: np.ndarray) -> tuple[str, bool]:
    """
    Check anomaly status: no anomaly, all anomaly, or normal (partial anomaly)
    Returns (status_text, is_special_case)
    """
    total = len(point_pred)
    anomaly_count = np.sum(point_pred == 1)
    
    if anomaly_count == 0:
        return "No Anomaly", True
    elif anomaly_count == total:
        return "All Anomaly", True
    else:
        return f"Anomaly: {anomaly_count}/{total} ({anomaly_count/total*100:.1f}%)", False


def _draw_anomaly_regions(ax, time_index: pd.Series, point_pred: np.ndarray):
    """Draw normal regions (green) and anomaly regions (red) on subplot"""
    if len(point_pred) == 0:
        return
    
    # Find all continuous segments
    current_state = point_pred[0]
    start_idx = 0
    
    for i in range(1, len(point_pred)):
        if point_pred[i] != current_state:
            # Segment ended, draw it
            start_t = time_index.iloc[start_idx]
            end_t = time_index.iloc[i]
            color = "red" if current_state == 1 else "green"
            ax.axvspan(start_t, end_t, color=color, alpha=0.15)
            
            # Start new segment
            current_state = point_pred[i]
            start_idx = i
    
    # Draw the last segment
    start_t = time_index.iloc[start_idx]
    end_t = time_index.iloc[-1]
    color = "red" if current_state == 1 else "green"
    ax.axvspan(start_t, end_t, color=color, alpha=0.15)


def visualize_all_tags(
    raw_csv: str,
    result_file: str,
    seq_len: int = 256,
    save_dir: str = "./vis_results/gas",
    target_pressure: str = None,
    target_flow: str = None,
):
    """
    Visualize tags from a CSV file. 
    If target_pressure/target_flow are provided, only plot those.
    Otherwise plot all FT/PT fields.
    """
    df = pd.read_csv(raw_csv)
    if df.shape[1] < 2:
        raise ValueError(f"raw_csv {raw_csv} has too few columns.")

    cols = list(df.columns)
    # First column is date
    cols[0] = "date"
    df.columns = cols

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")
    df = df.reset_index(drop=True)

    time_index = df["date"]

    # Load prediction results
    pred_win = _load_pred_vector(result_file, seq_len=seq_len)
    total_len = len(df)
    point_pred = _expand_window_pred_to_points(pred_win, total_len, win_size=seq_len)

    # Check anomaly status
    status_text, is_special_case = _check_anomaly_status(point_pred)

    # Filter fields
    if target_pressure:
        pressure_fields = [target_pressure] if target_pressure in df.columns else []
    else:
        pressure_fields = [f for f in df.columns if 'PT' in f]

    if target_flow:
        flow_fields = [target_flow] if target_flow in df.columns else []
    else:
        flow_fields = [f for f in df.columns if 'FT' in f]

    # Create subplots
    num_plots = (1 if flow_fields else 0) + (1 if pressure_fields else 0)
    if num_plots == 0:
        print("[WARN] No fields to plot.")
        return

    fig, axes = plt.subplots(num_plots, 1, figsize=(15, 6 * num_plots))
    if num_plots == 1:
        axes = [axes]
    
    ax_idx = 0

    # Plot flow
    if flow_fields:
        ax = axes[ax_idx]
        for field in flow_fields:
            label = FIELD_MAPPING.get(field, field)
            ax.plot(time_index, df[field], label=label, linewidth=1.5)
        ax.set_ylabel('Flow (m³/h)', fontsize=12)
        ax.set_title('Flow Data', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        _draw_anomaly_regions(ax, time_index, point_pred)
        ax_idx += 1

    # Plot pressure
    if pressure_fields:
        ax = axes[ax_idx]
        for field in pressure_fields:
            label = FIELD_MAPPING.get(field, field)
            ax.plot(time_index, df[field], label=label, linewidth=1.5)
        ax.set_ylabel('Pressure (MPa)', fontsize=12)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_title('Pressure Data', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        _draw_anomaly_regions(ax, time_index, point_pred)

    # Set overall title
    csv_base = os.path.splitext(os.path.basename(raw_csv))[0]
    result_name = os.path.basename(os.path.dirname(result_file))
    title = f"Anomaly Visualization - {csv_base}\nModel: {result_name}"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    out_name = f"{csv_base}_focused.png" if (target_pressure or target_flow) else f"{csv_base}_all.png"
    out_path = os.path.join(save_dir, out_name)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize gas anomaly detection results.")
    parser.add_argument("--result_file", type=str, required=True,
                        help="Path to npz file.")
    parser.add_argument("--raw_csv", type=str, default=None,
                        help="Specific raw CSV file.")
    parser.add_argument("--pressure_tag", type=str, default=None,
                        help="Filter: only show this pressure tag.")
    parser.add_argument("--flow_tag", type=str, default=None,
                        help="Filter: only show this flow tag.")
    parser.add_argument("--seq_len", type=int, default=None,
                        help="Override seq_len.")
    args = parser.parse_args()

    # ... (前后的保存路径逻辑保持不变) ...
    # 只需要在调用 visualize_all_tags 时传入参数即可
    # (为了简洁，我这里假设你直接应用更新到 main 函数的调用部分)
    
    # 实际修改处：
    # visualize_all_tags(
    #     raw_csv=csv_file,
    #     result_file=args.result_file,
    #     seq_len=seq_len,
    #     save_dir=save_dir,
    #     target_pressure=args.pressure_tag,
    #     target_flow=args.flow_tag,
    # )


if __name__ == "__main__":
    main()

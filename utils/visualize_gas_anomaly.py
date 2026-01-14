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
):
    """
    Visualize all tags from a CSV file in 2 subplots: flow and pressure.
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

    # Get all tag columns (exclude date column)
    tag_columns = [col for col in df.columns if col != "date"]
    if len(tag_columns) != 32:
        print(f"[WARN] Expected 32 tags, got {len(tag_columns)} in {raw_csv}")

    time_index = df["date"]

    # Load prediction results
    pred_win = _load_pred_vector(result_file, seq_len=seq_len)
    total_len = len(df)
    point_pred = _expand_window_pred_to_points(pred_win, total_len, win_size=seq_len)

    # Check anomaly status
    status_text, is_special_case = _check_anomaly_status(point_pred)

    # Separate flow and pressure fields
    flow_fields = [f for f in tag_columns if 'FT' in f]
    pressure_fields = [f for f in tag_columns if 'PT' in f]

    # Create 2 subplots: flow and pressure
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))

    # Plot flow fields
    if flow_fields:
        for field in flow_fields:
            axes[0].plot(time_index, df[field], label=field, linewidth=1.5, alpha=0.8)
        axes[0].set_ylabel('Flow (m³/h)', fontsize=12)
        axes[0].set_title('Flow Fields', fontsize=14, fontweight='bold')
        axes[0].legend(loc='best', fontsize=9, ncol=2)
        axes[0].grid(True, alpha=0.3)
        # Draw normal (green) and anomaly (red) regions
        _draw_anomaly_regions(axes[0], time_index, point_pred)
        # Add status text if special case
        if is_special_case:
            axes[0].text(0.5, 0.95, status_text, transform=axes[0].transAxes,
                       fontsize=14, fontweight='bold', color='red',
                       ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Plot pressure fields
    if pressure_fields:
        for field in pressure_fields:
            axes[1].plot(time_index, df[field], label=field, linewidth=1, alpha=0.7)
        axes[1].set_ylabel('Pressure (MPa)', fontsize=12)
        axes[1].set_xlabel('Time', fontsize=12)
        axes[1].set_title('Pressure Fields', fontsize=14, fontweight='bold')
        axes[1].legend(loc='best', fontsize=8, ncol=3)
        axes[1].grid(True, alpha=0.3)
        # Draw normal (green) and anomaly (red) regions
        _draw_anomaly_regions(axes[1], time_index, point_pred)
        # Add status text if special case
        if is_special_case:
            axes[1].text(0.5, 0.95, status_text, transform=axes[1].transAxes,
                       fontsize=14, fontweight='bold', color='red',
                       ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Set overall title
    csv_base = os.path.splitext(os.path.basename(raw_csv))[0]
    result_dir = os.path.basename(os.path.dirname(result_file))
    title = f"Gas Anomaly Visualization - {csv_base}\nResult: {result_dir}\n{status_text}"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save image
    out_name = os.path.splitext(os.path.basename(raw_csv))[0]
    out_path = os.path.join(save_dir, f"{out_name}_all_tags.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {out_path} ({status_text})")


def main():
    parser = argparse.ArgumentParser(description="Visualize gas anomaly detection results.")
    parser.add_argument("--result_file", type=str, required=True,
                        help="Path to npz file (energy_and_pred.npz or global_fusion.npz).")
    parser.add_argument("--raw_csv", type=str, default=None,
                        help="Path to a specific raw CSV file to visualize. If not provided, visualizes all in data/csv_data.")
    parser.add_argument("--pressure_tag", type=str, default=None,
                        help="Specific pressure tag to highlight (optional).")
    parser.add_argument("--flow_tag", type=str, default=None,
                        help="Specific flow tag to highlight (optional).")
    parser.add_argument("--seq_len", type=int, default=None,
                        help="Override sequence length (optional).")
    args = parser.parse_args()

    # Infer save directory from result_file path
    result_file_abs = os.path.abspath(args.result_file)
    result_dir_name = os.path.basename(os.path.dirname(result_file_abs))
    
    # Create corresponding folder under vis_results
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(project_root, "vis_results", result_dir_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Output directory: {save_dir}")

    # Automatically read seq_len from result_file if not provided
    seq_len = args.seq_len
    if seq_len is None:
        seq_len = _get_seq_len_from_result(args.result_file)
        print(f"Using seq_len={seq_len} from result file")
    else:
        print(f"Using manual seq_len={seq_len}")

    if args.raw_csv:
        # Visualize specific CSV
        csv_files = [args.raw_csv]
    else:
        # Find all CSV files (hardcoded to ./data/csv_data)
        csv_data_dir = os.path.join(project_root, "data", "csv_data")
        csv_files = sorted(glob.glob(os.path.join(csv_data_dir, "*.csv")))
    
    if not csv_files:
        print(f"[ERROR] No CSV files found.")
        return

    print(f"Found {len(csv_files)} CSV files to visualize")

    # Generate visualization for each CSV file
    for csv_file in csv_files:
        try:
            visualize_all_tags(
                raw_csv=csv_file,
                result_file=args.result_file,
                seq_len=seq_len,
                save_dir=save_dir,
            )
        except Exception as e:
            print(f"[ERROR] Failed to visualize {csv_file}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nVisualization complete! All results saved to: {save_dir}")


if __name__ == "__main__":
    main()

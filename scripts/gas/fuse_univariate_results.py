import os
import numpy as np


GAS_FIELDS = [
    "CHX00E005PT0101",
    "CHX00E005PT0102",
    "CHX00L006FT0101",
    "CHX00L006PT0101",
    "CHX00A001FT0102",
    "CHX00A001PT0102",
    "CHX00A001PT0411",
    "CHX00A001PT0412",
    "CHX00E009PT0101",
    "CHX00E009PT0102",
    "CHX00F005FT0101",
    "CHX00F005PT0101",
    "CHX00F005PT0102",
    "CHX00G004FT0101",
    "CHX00G004PT0101",
    "CHX00G004PT0102",
    "CHX00F002FT0101",
    "CHX00F002PT0101",
    "CHX00F002PT0102",
    "CHX00F002PT0409",
    "CHX00F002PT0410",
    "CHX00E017PT0101",
    "CHX00E017PT0102",
    "CHX00E013PT0101",
    "CHX00E013PT0102",
    "CHX00E003PT0101",
    "CHX00E003PT0102",
    "CHX00F003FT0101",
    "CHX00F003PT0101",
    "CHX00F003PT0102",
    "CHX00F003PT0411",
    "CHX00F003PT0412",
]


def fuse_univariate(
    test_results_root: str = "./test_results",
    checkpoints_root: str = "./checkpoints/gas_timesnet",
):
    """
    将 32 个单变量 TimesNet 的预测结果进行 OR 融合：
    - 遍历每个字段对应的 setting 子目录
    - 读取 energy_and_pred.npz，取出 pred（一维 0/1）
    - 逐元素做逻辑 OR，得到全局的 global_pred
    - 保存到一个新的 npz 文件中，位于 test_results/gas_univariate_fusion 下
    """
    fused_dir = os.path.join(test_results_root, "gas_univariate_fusion")
    os.makedirs(fused_dir, exist_ok=True)

    all_preds = []
    base_len = None

    for field in GAS_FIELDS:
        # 与 run_univariate_timesnet.py 中保持一致的命名
        setting_prefix = f"anomaly_detection_gas_univariate_{field}_TimesNet_GAS"
        # 在 test_results 下查找对应 setting 目录（可能带有超参信息后缀）
        candidates = [
            d for d in os.listdir(test_results_root)
            if d.startswith(setting_prefix)
        ]
        if not candidates:
            print(f"[WARN] No test_results directory found for field {field}, prefix={setting_prefix}")
            continue

        setting_dir = os.path.join(test_results_root, candidates[0])
        npz_path = os.path.join(setting_dir, "energy_and_pred.npz")
        if not os.path.exists(npz_path):
            print(f"[WARN] energy_and_pred.npz not found for field {field} in {setting_dir}")
            continue

        data = np.load(npz_path)
        pred = data["pred"].astype(int)

        if base_len is None:
            base_len = pred.shape[0]
        elif pred.shape[0] != base_len:
            print(f"[WARN] pred length mismatch for field {field}: {pred.shape[0]} vs {base_len}, skipping")
            continue

        all_preds.append(pred)
        print(f"[INFO] loaded pred for field {field}, length={pred.shape[0]}")

    if not all_preds:
        print("No valid univariate predictions found to fuse.")
        return

    # (num_fields, L)
    stack = np.stack(all_preds, axis=0)
    # 按时间维做 OR
    global_pred = (stack.sum(axis=0) > 0).astype(int)
    num_active = stack.sum(axis=0).astype(int)

    out_path = os.path.join(fused_dir, "global_fusion.npz")
    np.savez(
        out_path,
        global_pred=global_pred,
        num_active=num_active,
    )
    print(f"[INFO] saved fused result to {out_path}, length={global_pred.shape[0]}")


if __name__ == "__main__":
    fuse_univariate()



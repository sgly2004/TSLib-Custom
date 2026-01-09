import os
import subprocess


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


def main():
    """
    在服务器上批量运行 32 个单变量 TimesNet 异常检测实验。

    用法（在项目根目录下）：
        python scripts/gas/run_univariate_timesnet.py
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    run_py = os.path.join(project_root, "run.py")

    for field in GAS_FIELDS:
        setting_tag = f"gas_univariate_{field}"
        cmd = [
            "python",
            run_py,
            "--task_name", "anomaly_detection",
            "--is_training", "1",
            "--model_id", setting_tag,
            "--model", "TimesNet",
            "--data", "GAS",
            "--root_path", "./dataset/gas_anomaly",
            "--features", "S",
            "--target", field,
            "--enc_in", "1",
            "--c_out", "1",
            "--seq_len", "256",
            "--anomaly_ratio", "1",
            "--batch_size", "32",
            "--train_epochs", "10",
            "--checkpoints", "./checkpoints/gas_timesnet",
            "--des", "gas_univariate",
        ]

        print("=" * 80)
        print(f"Running univariate TimesNet for field: {field}")
        print("Command:", " ".join(cmd))
        print("=" * 80)

        subprocess.run(cmd, cwd=project_root, check=False)


if __name__ == "__main__":
    main()



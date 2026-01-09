import os
import glob
import pandas as pd


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


def _load_and_clean_csv(path: str, min_len: int = 1) -> pd.DataFrame:
    """
    读取单个 CSV，统一字段名，按日期排序并去除缺失。
    期望第一列为 '日期'，后面为 32 个节点字段。
    """
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        return pd.DataFrame()

    # 统一日期列名为 date，便于与库中其它数据保持一致
    cols = list(df.columns)
    cols[0] = 'date'
    df.columns = cols

    # 丢弃完全空行，按日期排序
    df = df.dropna(how='all')
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df = df.sort_values('date')

    # 只保留 date + 32 个测点列（按现有列顺序）
    expected_cols = ['date'] + cols[1:33]
    df = df[expected_cols]

    # 去除中间的缺失行
    df = df.dropna()

    if len(df) < min_len:
        return pd.DataFrame()

    return df


def build_gas_anomaly_dataset(
    project_root: str = ".",
    normal_dir: str = "data/normal",
    raw_dir: str = "data/csv_data",
    out_dir: str = "dataset/gas_anomaly",
    min_normal_len: int = 150,
) -> None:
    """
    从 normal 目录与原始 csv_data 目录构建异常检测所需的 train.csv 与 test.csv。
    - train.csv：由所有长度 >= min_normal_len 的 normal 片段拼接而成
    - test.csv：由所有原始 csv_data 文件按时间拼接而成
    """
    normal_path = os.path.join(project_root, normal_dir)
    raw_path = os.path.join(project_root, raw_dir)
    out_path = os.path.join(project_root, out_dir)
    os.makedirs(out_path, exist_ok=True)

    # 1) 构建训练集：只使用 normal 片段
    train_parts = []
    normal_files = sorted(glob.glob(os.path.join(normal_path, "*.csv")))
    for fp in normal_files:
        df = _load_and_clean_csv(fp, min_len=min_normal_len)
        if not df.empty:
            train_parts.append(df)

    if train_parts:
        train_df = pd.concat(train_parts, axis=0, ignore_index=True)
        train_out = os.path.join(out_path, "train.csv")
        train_df.to_csv(train_out, index=False)
        print(f"Saved train data: {train_out}, length={len(train_df)}")
    else:
        print("No valid normal segments found for training (after length filtering).")

    # 2) 构建测试集：使用完整样本 csv_data
    test_parts = []
    raw_files = sorted(glob.glob(os.path.join(raw_path, "*.csv")))
    for fp in raw_files:
        df = _load_and_clean_csv(fp, min_len=1)
        if not df.empty:
            test_parts.append(df)

    if test_parts:
        test_df = pd.concat(test_parts, axis=0, ignore_index=True)
        test_out = os.path.join(out_path, "test.csv")
        test_df.to_csv(test_out, index=False)
        print(f"Saved test data: {test_out}, length={len(test_df)}")
    else:
        print("No valid raw csv_data files found for testing.")


if __name__ == "__main__":
    # 默认在项目根目录下执行：python scripts/gas/build_gas_anomaly_dataset.py
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, "..", ".."))
    build_gas_anomaly_dataset(project_root=project_root)



from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, CustomAnomalySegLoader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    # 自定义异常检测数据集，支持边界记录
    'anomaly-detection-normal-300': CustomAnomalySegLoader,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        # 获取 step 参数，如果 args 中有则使用，否则使用默认值
        # 这样可以控制滑窗采样的步长，减少样本数量
        step = getattr(args, 'step', 1)  # 默认值为 1
        
        # 如果是自定义 Loader（支持边界记录），传递 segment_length 参数
        # segment_length 用于标识每个 segment 的长度，避免跨边界采样
        if args.data in ['anomaly-detection-normal-300']:
            segment_length = getattr(args, 'segment_length', 300)  # 默认 300，可从 dataset_info.txt 读取
            data_set = Data(
                args=args,
                root_path=args.root_path,
                win_size=args.seq_len,
                step=step,  # 传递 step 参数，控制滑窗采样步长
                flag=flag,
                segment_length=segment_length,  # 传递 segment_length，用于边界记录
            )
        else:
            # 原有的 Loader（PSM, MSL, SMAP, SMD, SWAT）
            data_set = Data(
                args=args,
                root_path=args.root_path,
                win_size=args.seq_len,
                step=step,  # 传递 step 参数，控制滑窗采样步长
                flag=flag,
            )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader

#!/bin/bash

# 仅运行测试模式 - 针对标准训练模型重新计算阈值和预测结果
# 必须确保参数与 train_single_dimension.sh 一致！

python run.py \
  --task_name anomaly_detection \
  --is_training 0 \
  --model_id gas_single_CHX00L006PT0101 \
  --model TimesNet \
  --data GAS \
  --root_path ./dataset/gas_anomaly_single \
  --features S \
  --target CHX00L006PT0101 \
  --enc_in 1 \
  --c_out 1 \
  --seq_len 256 \
  --pred_len 0 \
  --anomaly_ratio 1.0 \
  --batch_size 128 \
  --d_model 128 \
  --d_ff 512 \
  --e_layers 2 \
  --top_k 3 \
  --num_kernels 4 \
  --checkpoints ./checkpoints/gas_single_dimension \
  --des CHX00L006PT0101_pressure \
  --use_amp

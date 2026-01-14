#!/bin/bash

# 仅运行测试模式 - 针对已有的 quick 模型重新计算阈值和预测结果
# 必须确保参数与训练时完全一致！

python run.py \
  --task_name anomaly_detection \
  --is_training 0 \
  --model_id gas_single_CHX00L006PT0101_quick \
  --model TimesNet \
  --data GAS \
  --root_path ./dataset/gas_anomaly_single \
  --features S \
  --target CHX00L006PT0101 \
  --enc_in 1 \
  --c_out 1 \
  --seq_len 128 \
  --pred_len 0 \
  --anomaly_ratio 1.0 \
  --batch_size 256 \
  --d_model 64 \
  --d_ff 256 \
  --e_layers 1 \
  --top_k 3 \
  --num_kernels 4 \
  --checkpoints ./checkpoints/gas_single_dimension_quick \
  --des CHX00L006PT0101_quick_test \
  --use_amp

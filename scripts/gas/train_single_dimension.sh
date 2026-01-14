#!/bin/bash

# 单维度异常检测训练脚本 - CHX00L006PT0101（呼和浩特末站压力）
# 使用 TimesNet 模型进行异常检测

echo "=========================================="
echo "单维度异常检测训练 - CHX00L006PT0101"
echo "呼和浩特末站压力"
echo "=========================================="

python run.py \
  --task_name anomaly_detection \
  --is_training 1 \
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
  --anomaly_ratio 1 \
  --batch_size 128 \
  --d_model 128 \
  --d_ff 512 \
  --e_layers 2 \
  --top_k 3 \
  --num_kernels 4 \
  --patience 5 \
  --train_epochs 20 \
  --learning_rate 0.0001 \
  --checkpoints ./checkpoints/gas_single_dimension \
  --des CHX00L006PT0101_pressure \
  --use_amp

echo ""
echo "训练完成！"
echo "模型保存位置: ./checkpoints/gas_single_dimension/"
echo "结果保存位置: ./test_results/"

#!/bin/bash

# 单维度异常检测快速训练脚本（用于快速测试）
# 更小的模型配置，更少的训练轮数

echo "=========================================="
echo "单维度异常检测快速训练 - CHX00L006PT0101"
echo "呼和浩特末站压力（快速测试配置）"
echo "=========================================="

python run.py \
  --task_name anomaly_detection \
  --is_training 1 \
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
  --anomaly_ratio 1 \
  --batch_size 256 \
  --d_model 64 \
  --d_ff 256 \
  --e_layers 1 \
  --top_k 3 \
  --num_kernels 4 \
  --patience 3 \
  --train_epochs 5 \
  --learning_rate 0.001 \
  --checkpoints ./checkpoints/gas_single_dimension_quick \
  --des CHX00L006PT0101_quick_test \
  --use_amp

echo ""
echo "快速训练完成！"
echo "模型保存位置: ./checkpoints/gas_single_dimension_quick/"
echo "结果保存位置: ./test_results/"

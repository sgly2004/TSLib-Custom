#!/bin/bash

# 单文件独立异常检测 - 避免边界效应
# 使用方法: bash scripts/gas/run_individual_detection.sh

echo "=============================================="
echo "单文件独立异常检测"
echo "=============================================="
echo ""

# 步骤 1: 对每个文件单独进行检测
echo "【步骤 1/2】正在对每个文件单独进行异常检测..."
echo "----------------------------------------------"
python scripts/gas/test_individual_files.py

if [ $? -ne 0 ]; then
    echo "❌ 检测失败，请检查错误信息"
    exit 1
fi

echo ""
echo "【步骤 2/2】正在生成可视化结果..."
echo "----------------------------------------------"

# 步骤 2: 可视化前 10 个文件的结果
python scripts/gas/visualize_individual_results.py \
    --result_dir test_results/individual_file_results \
    --raw_dir data/csv_data \
    --target CHX00L006PT0101 \
    --seq_len 256 \
    --step 8 \
    --max_files 10 \
    --output_dir vis_results/individual_clean

if [ $? -ne 0 ]; then
    echo "❌ 可视化失败，请检查错误信息"
    exit 1
fi

echo ""
echo "=============================================="
echo "✅ 全部完成！"
echo "检测结果: test_results/individual_file_results/"
echo "可视化结果: vis_results/individual_clean/"
echo "=============================================="

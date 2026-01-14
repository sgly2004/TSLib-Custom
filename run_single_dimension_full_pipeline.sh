#!/bin/bash

# ========================================
# 单维度异常检测完整流程一键执行脚本
# ========================================

set -e  # 遇到错误立即退出

echo "========================================"
echo "单维度异常检测完整流程"
echo "目标维度: CHX00L006PT0101（呼和浩特末站压力）"
echo "========================================"
echo ""

# 步骤 1: 数据提取
echo "【步骤 1/3】数据提取"
echo "----------------------------------------"
python scripts/gas/build_single_dimension_dataset.py

if [ ! -f "dataset/gas_anomaly_single/train.csv" ] || [ ! -f "dataset/gas_anomaly_single/test.csv" ]; then
    echo "❌ 错误: 数据提取失败，请检查 data/normal 和 data/csv_data 目录"
    exit 1
fi

echo ""
echo "✅ 数据提取完成！"
echo ""
read -p "按 Enter 继续训练，或 Ctrl+C 退出..."
echo ""

# 步骤 2: 模型训练
echo "【步骤 2/3】模型训练"
echo "----------------------------------------"
echo "请选择训练方案："
echo "  1) 快速测试（推荐，2-5分钟）"
echo "  2) 完整训练（10-30分钟）"
echo ""
read -p "请输入选择 [1/2]: " choice

case $choice in
    1)
        echo "执行快速测试训练..."
        bash scripts/gas/train_single_dimension_quick.sh
        RESULT_PATTERN="anomaly_detection_gas_single_CHX00L006PT0101_quick_*"
        ;;
    2)
        echo "执行完整训练..."
        bash scripts/gas/train_single_dimension.sh
        RESULT_PATTERN="anomaly_detection_gas_single_CHX00L006PT0101_*"
        ;;
    *)
        echo "❌ 无效选择，退出"
        exit 1
        ;;
esac

echo ""
echo "✅ 训练完成！"
echo ""

# 查找生成的结果文件
echo "【步骤 3/3】结果可视化"
echo "----------------------------------------"

RESULT_DIR=$(find test_results -type d -name "$RESULT_PATTERN" | head -n 1)

if [ -z "$RESULT_DIR" ]; then
    echo "⚠️  警告: 未找到结果目录"
    echo "请手动查找结果文件并运行可视化命令："
    echo ""
    echo "python utils/visualize_gas_anomaly.py \\"
    echo "  --raw_csv data/csv_data/1004.csv \\"
    echo "  --result_file test_results/.../energy_and_pred.npz \\"
    echo "  --pressure_tag CHX00L006PT0101 \\"
    echo "  --seq_len 256"
    exit 0
fi

RESULT_FILE="$RESULT_DIR/energy_and_pred.npz"

if [ ! -f "$RESULT_FILE" ]; then
    echo "⚠️  警告: 未找到结果文件 $RESULT_FILE"
    exit 1
fi

echo "找到结果文件: $RESULT_FILE"
echo ""
echo "是否要可视化样本 1004.csv？"
read -p "按 Enter 继续，或 Ctrl+C 退出..."

# 提取 seq_len
SEQ_LEN=$(echo "$RESULT_DIR" | grep -oP 'sl\K[0-9]+' || echo "256")

python utils/visualize_gas_anomaly.py \
  --raw_csv data/csv_data/1004.csv \
  --result_file "$RESULT_FILE" \
  --pressure_tag CHX00L006PT0101 \
  --seq_len "$SEQ_LEN"

echo ""
echo "✅ 可视化完成！"
echo ""
echo "========================================"
echo "完整流程执行完毕！"
echo "========================================"
echo ""
echo "📂 生成的文件："
echo "  - 训练数据: dataset/gas_anomaly_single/train.csv"
echo "  - 测试数据: dataset/gas_anomaly_single/test.csv"
echo "  - 模型文件: checkpoints/gas_single_dimension*/"
echo "  - 测试结果: $RESULT_FILE"
echo "  - 可视化图: vis_results/gas/1004_energy_and_pred_pressure-CHX00L006PT0101.png"
echo ""
echo "🎯 下一步建议："
echo "  1. 查看可视化结果: open vis_results/gas/1004_*.png"
echo "  2. 可视化其他样本: python utils/visualize_gas_anomaly.py ..."
echo "  3. 调整参数重新训练"
echo "  4. 查看完整文档: cat single_dimension_pipeline.md"
echo ""

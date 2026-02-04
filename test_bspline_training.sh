#!/bin/bash
# 测试B样条训练脚本

echo "========================================="
echo "测试 B-spline 训练脚本"
echo "========================================="

# 第一阶段训练（轨迹重建）
echo "
[测试] Stage 1: 轨迹重建
- 模型: BSplineDiffusionTransformer
- 控制点数: 15
- 损失: 轨迹重建 + 控制点平滑 + 起终点约束
"

python3 train_bspline_dit.py \
  --batchSize=20 \
  --dataFolder=data/sim_dataset \
  --fileDir=data/bspline \
  --stage=1 \
  --epochs=1 \
  --learningRate=5e-4

echo "
Stage 1 测试完成！

如果通过，应该看到：
- 损失正常下降（无 NaN）
- 打印的模型参数显示 15 个控制点
- 损失字典包含: main, smoothness, endpoint, total

接下来可以测试 Stage 2 (cost优化):
python3 train_bspline_dit.py \
  --batchSize=20 \
  --dataFolder=data/sim_dataset \
  --fileDir=data/bspline \
  --stage=2 \
  --epochs=10
"

"""
分析轨迹复杂度
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from dataLoader_dit import UnevenPathDataLoader
from torch.utils.data import DataLoader

# 加载数据
dataset = UnevenPathDataLoader(
    env_list=['env000008'],
    dataFolder='data/sim_dataset/train',
    compute_stability_map=False
)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

print("="*70)
print("轨迹复杂度分析")
print("="*70)

curvatures_all = []
angle_changes_all = []

for i, batch in enumerate(dataloader):
    if i >= 20:
        break
    
    trajectory = batch['trajectory'][0]  # (22, 3)
    positions = trajectory[:, :2]  # (22, 2)
    angles = trajectory[:, 2]  # (22,)
    
    # 计算曲率（角度变化率）
    angle_diff = angles[1:] - angles[:-1]
    # Wrap到[-pi, pi]
    angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
    
    # 计算弧长
    distances = torch.norm(positions[1:] - positions[:-1], dim=1)
    arc_length = distances.sum().item()
    
    # 曲率 = 角度变化 / 弧长变化
    curvature = angle_diff / (distances + 1e-8)
    
    curvatures_all.extend(curvature.abs().numpy())
    angle_changes_all.extend(torch.abs(angle_diff).numpy())
    
    if i < 3:
        print(f"\nSample {i+1}:")
        print(f"  Total arc length: {arc_length:.3f}m")
        print(f"  Max angle change: {torch.abs(angle_diff).max().item()*180/np.pi:.2f}°")
        print(f"  Mean curvature: {curvature.abs().mean().item():.4f} rad/m")
        print(f"  Max curvature: {curvature.abs().max().item():.4f} rad/m")

# 统计
print("\n" + "="*70)
print("总体统计:")
print(f"  平均曲率: {np.mean(curvatures_all):.4f} rad/m")
print(f"  最大曲率: {np.max(curvatures_all):.4f} rad/m")
print(f"  平均角度变化: {np.degrees(np.mean(angle_changes_all)):.2f}°")
print(f"  最大角度变化: {np.degrees(np.max(angle_changes_all)):.2f}°")

# 判断复杂度
max_curv = np.max(curvatures_all)
mean_angle_change = np.degrees(np.mean(angle_changes_all))

print("\n诊断:")
if max_curv > 1.0:  # 曲率半径 < 1m
    print("✗ 轨迹包含极高曲率（急转弯）")
    print(f"  最小转弯半径: ~{1/max_curv:.2f}m")
elif max_curv > 0.5:
    print("⚠ 轨迹包含较高曲率")
    print(f"  最小转弯半径: ~{1/max_curv:.2f}m")
else:
    print("✓ 轨迹曲率适中")

if mean_angle_change > 30:
    print(f"✗ 平均角度变化大 ({mean_angle_change:.1f}°)")
    print("  多项式难以拟合这种曲折轨迹")
elif mean_angle_change > 15:
    print(f"⚠ 平均角度变化较大 ({mean_angle_change:.1f}°)")
else:
    print(f"✓ 平均角度变化适中 ({mean_angle_change:.1f}°)")

print("\n建议:")
print("1. 如果轨迹确实很复杂，考虑直接预测轨迹点（不用参数化）")
print("2. 使用更灵活的表示：B样条、三次样条")
print("3. 使用学习的表示：VAE latent、Transformer直接输出点")
print("="*70)

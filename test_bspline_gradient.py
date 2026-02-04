"""
测试B样条的梯度传递和拟合质量
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from dit.BSplineModels import BSplineInterpolator

print("="*70)
print("测试1: B样条梯度传递")
print("="*70)

# 创建需要梯度的控制点
control_points = torch.randn(2, 10, 1, requires_grad=True)  # (B=2, n_control=10, dim=1)
t_values = torch.linspace(0, 1, 50)

# B样条插值
curve = BSplineInterpolator.evaluate_bspline(control_points, t_values, degree=3)  # (2, 50, 1)

# 定义一个损失（例如：让曲线接近某个目标）
target = torch.zeros_like(curve)
loss = ((curve - target) ** 2).mean()

# 反向传播
loss.backward()

# 检查梯度
if control_points.grad is not None:
    print("✓ 梯度成功传递!")
    print(f"  控制点梯度范数: {control_points.grad.norm().item():.6f}")
    print(f"  梯度形状: {control_points.grad.shape}")
else:
    print("✗ 梯度传递失败!")

print("\n" + "="*70)
print("测试2: B样条轨迹生成的梯度传递")
print("="*70)

# 测试完整的轨迹生成流程
angle_control = torch.randn(2, 10, requires_grad=True)  # (B=2, n_control=10) 角度控制点
start_pose = torch.tensor([[0., 0., 0.], [1., 1., 0.5]])  # (B=2, 3)
total_length = 30.0

# 生成轨迹
trajectory = BSplineInterpolator.integrate_kinematics_from_angle_control_points(
    angle_control, start_pose, total_length, n_path_points=21
)  # (2, 20, 3)

# 定义损失
target_traj = torch.zeros_like(trajectory)
loss = ((trajectory - target_traj) ** 2).mean()

# 反向传播
loss.backward()

if angle_control.grad is not None:
    print("✓ 轨迹生成梯度成功传递!")
    print(f"  角度控制点梯度范数: {angle_control.grad.norm().item():.6f}")
    print(f"  梯度形状: {angle_control.grad.shape}")
else:
    print("✗ 轨迹生成梯度传递失败!")

print("\n" + "="*70)
print("测试3: B样条拟合质量")
print("="*70)

from dataLoader_dit import UnevenPathDataLoader
from torch.utils.data import DataLoader

# 加载数据
dataset = UnevenPathDataLoader(
    env_list=['env000008'],
    dataFolder='data/sim_dataset/train',
    compute_stability_map=False
)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 测试不同控制点数量
n_controls = [8, 10, 12, 15, 20]

print(f"\n{'控制点数':<10} {'平均误差(m)':<15} {'最大误差(m)':<15} {'角度误差(°)':<15}")
print("-"*70)

results = []

for n_control in n_controls:
    pos_errors = []
    angle_errors = []
    max_errors = []
    
    # 测试10个样本
    for i, batch in enumerate(dataloader):
        if i >= 10:
            break
        
        trajectory = batch['trajectory'][0]  # (22, 3)
        start_pose = batch['start_pose'][0]  # (3,)
        
        # 提取角度
        angles = trajectory[:21, 2]  # (21,) 使用前21个点的角度
        
        # 简单拟合：均匀采样控制点
        indices = torch.linspace(0, 20, n_control).long()
        angle_control = angles[indices]  # (n_control,) 直接采样作为控制点
        
        # 计算弧长
        positions = trajectory[:, :2]
        distances = torch.norm(positions[1:] - positions[:-1], dim=1)
        arc_length = distances.sum().item()
        
        # 通过B样条重建
        reconstructed = BSplineInterpolator.integrate_kinematics_from_angle_control_points(
            angle_control.unsqueeze(0),  # (1, n_control)
            start_pose.unsqueeze(0),
            arc_length,
            n_path_points=21
        )[0]  # (20, 3)
        
        # 比较
        gt_middle = trajectory[1:21]  # (20, 3)
        pos_error = torch.norm(reconstructed[:, :2] - gt_middle[:, :2], dim=1)
        angle_error = torch.abs(reconstructed[:, 2] - gt_middle[:, 2])
        angle_error = torch.min(angle_error, 2*np.pi - angle_error)
        
        pos_errors.append(pos_error.mean().item())
        max_errors.append(pos_error.max().item())
        angle_errors.append(angle_error.mean().item())
    
    mean_pos = np.mean(pos_errors)
    mean_max = np.mean(max_errors)
    mean_angle = np.degrees(np.mean(angle_errors))
    
    print(f"{n_control:<10} {mean_pos:<15.4f} {mean_max:<15.4f} {mean_angle:<15.2f}")
    
    results.append({
        'n_control': n_control,
        'mean_pos': mean_pos,
        'mean_max': mean_max,
        'mean_angle': mean_angle
    })

# 找最佳配置
best = min(results, key=lambda x: x['mean_pos'])
print("-"*70)
print(f"最佳控制点数: {best['n_control']}")
print(f"  平均误差: {best['mean_pos']:.4f}m")
print(f"  最大误差: {best['mean_max']:.4f}m")
print(f"  角度误差: {best['mean_angle']:.2f}°")

if best['mean_pos'] < 0.5:
    print("\n✓ B样条拟合质量良好!")
elif best['mean_pos'] < 2.0:
    print("\n⚠ B样条拟合质量尚可，可以使用")
else:
    print("\n✗ B样条拟合质量仍然较差")

print("="*70)

print("\n说明:")
print("1. 梯度传递测试验证了B样条的可微性")
print("2. 拟合质量测试使用简单采样（非优化拟合）")
print("3. 实际训练时，网络会学习最优控制点")
print("4. B样条的局部控制特性比多项式更适合急转弯")

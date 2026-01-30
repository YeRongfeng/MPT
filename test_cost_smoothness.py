"""
测试cost函数的平滑性和数值稳定性
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from grad_optimizer import TrajectoryOptimizerSE2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 创建一个简单的测试轨迹
N = 22  # 总点数（起点+20中间点+终点）

# 基准轨迹：从(0,0,0)到(10,10,0)的直线
base_traj = torch.zeros(N, 3, device=device)
base_traj[:, 0] = torch.linspace(0, 10, N)  # x
base_traj[:, 1] = torch.linspace(0, 10, N)  # y
base_traj[:, 2] = 0.0  # yaw

# 创建虚拟的cost map（全零）
map_size = (36, 100, 100)  # (yaw_bins, height, width)
cost_map = torch.zeros(*map_size, device=device).permute(2, 0, 1)  # (D, H, W)

map_info = {
    'resolution': 0.4,
    'origin': (-20.0, -20.0, -np.pi),
    'size': (100, 100, 36)
}

# 创建优化器
optimizer = TrajectoryOptimizerSE2(
    base_traj.detach(),
    cost_map,
    map_info,
    device=device
)

print("\n=== 测试1: 检查cost对轨迹扰动的敏感度 ===")
# 测试在基准轨迹附近的cost变化
perturbations = torch.linspace(-0.5, 0.5, 21)  # -0.5m到+0.5m
costs = []
gradients = []

for delta in perturbations:
    # 创建扰动轨迹（只扰动中间点的y坐标）
    perturbed_traj = base_traj.clone()
    perturbed_traj[1:-1, 1] += delta  # 只改变中间20个点的y坐标
    perturbed_traj.requires_grad_(True)
    
    # 计算cost
    cost = optimizer.cost_on_poses(perturbed_traj)
    costs.append(cost.item())
    
    # 计算梯度
    cost.backward()
    grad_norm = torch.norm(perturbed_traj.grad).item()
    gradients.append(grad_norm)
    
    print(f"delta={delta:+.2f}m: cost={cost.item():.4f}, grad_norm={grad_norm:.4f}")

# 分析cost的平滑性
costs = np.array(costs)
gradients = np.array(gradients)

print(f"\nCost统计:")
print(f"  最小值: {costs.min():.4f}")
print(f"  最大值: {costs.max():.4f}")
print(f"  范围: {costs.max() - costs.min():.4f}")
print(f"  均值: {costs.mean():.4f}")
print(f"  标准差: {costs.std():.4f}")

print(f"\n梯度统计:")
print(f"  最小值: {gradients.min():.4f}")
print(f"  最大值: {gradients.max():.4f}")
print(f"  范围: {gradients.max() - gradients.min():.4f}")
print(f"  均值: {gradients.mean():.4f}")
print(f"  标准差: {gradients.std():.4f}")

# 计算cost的数值导数
numerical_grad = np.gradient(costs, perturbations.numpy())
print(f"\n数值导数统计:")
print(f"  最小值: {numerical_grad.min():.4f}")
print(f"  最大值: {numerical_grad.max():.4f}")
print(f"  均值: {numerical_grad.mean():.4f}")

# 计算二阶导数（检查平滑性）
second_deriv = np.gradient(numerical_grad, perturbations.numpy())
print(f"\n二阶导数统计（平滑性指标）:")
print(f"  最小值: {second_deriv.min():.4f}")
print(f"  最大值: {second_deriv.max():.4f}")
print(f"  绝对值均值: {np.abs(second_deriv).mean():.4f}")
print(f"  标准差: {second_deriv.std():.4f}")

if np.abs(second_deriv).max() > 100:
    print("\n⚠️  警告：二阶导数很大，cost函数可能不平滑！")
    print("这会导致优化困难和训练不稳定。")

print("\n=== 测试2: 检查梯度的一致性 ===")
# 测试在同一点重复计算梯度是否一致
test_traj = base_traj.clone()
test_traj[10, 1] += 0.1  # 轻微扰动

grads_repeated = []
for i in range(5):
    test_traj_copy = test_traj.clone().requires_grad_(True)
    cost = optimizer.cost_on_poses(test_traj_copy)
    cost.backward()
    grads_repeated.append(test_traj_copy.grad[10, 1].item())

grads_repeated = np.array(grads_repeated)
print(f"重复计算梯度的一致性:")
print(f"  值: {grads_repeated}")
print(f"  均值: {grads_repeated.mean():.6f}")
print(f"  标准差: {grads_repeated.std():.6e}")

if grads_repeated.std() < 1e-6:
    print("✅ 梯度计算是确定性的")
else:
    print("⚠️  警告：梯度计算有随机性！")

print("\n=== 测试3: 检查不同成本项的贡献 ===")
# 检查各个成本项的量级
test_traj = base_traj.clone().requires_grad_(True)

# 直接访问optimizer的内部方法
Sx, Sy, Syaw, Sx_dot, Sy_dot, Syaw_dot, Sx_ddot, Sy_ddot, Syaw_ddot = optimizer._evaluate_spline_se2(test_traj, optimizer.t_dense)

print("\n插值结果统计:")
print(f"  位置范围: x=[{Sx.min().item():.2f}, {Sx.max().item():.2f}], y=[{Sy.min().item():.2f}, {Sy.max().item():.2f}]")
print(f"  速度范围: vx=[{Sx_dot.min().item():.2f}, {Sx_dot.max().item():.2f}], vy=[{Sy_dot.min().item():.2f}, {Sy_dot.max().item():.2f}]")
print(f"  加速度范围: ax=[{Sx_ddot.min().item():.2f}, {Sx_ddot.max().item():.2f}], ay=[{Sy_ddot.min().item():.2f}, {Sy_ddot.max().item():.2f}]")

# 计算各项成本
eps = 1e-6
speed = torch.sqrt(Sx_dot**2 + Sy_dot**2 + eps)
geom_curvature = torch.abs(Sx_dot * Sy_ddot - Sy_dot * Sx_ddot) / (speed**3 + 1e-6)

print(f"\n几何特征:")
print(f"  速度: mean={speed.mean().item():.4f}, max={speed.max().item():.4f}")
print(f"  曲率: mean={geom_curvature.mean().item():.4f}, max={geom_curvature.max().item():.4f}")

# 计算总cost
total_cost = optimizer.cost_on_poses(test_traj)
print(f"\n总cost: {total_cost.item():.4f}")

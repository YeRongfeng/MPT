"""
测试B样条控制点拟合方法

问题：从GT轨迹(x,y,θ)反推角度控制点，使得通过B样条+运动学积分能重建GT轨迹
"""
import torch
import torch.optim as optim
from dit.BSplineModels import BSplineInterpolator
import numpy as np

def fit_control_points_by_optimization(trajectory, start_pose, n_control_points=15, 
                                       n_iterations=1000, lr=0.1):
    """
    通过优化拟合控制点
    
    Args:
        trajectory: (B, N, 3) GT轨迹 [x, y, theta]
        start_pose: (B, 3) 起点
        n_control_points: 控制点数量
        n_iterations: 优化迭代次数
        lr: 学习率
        
    Returns:
        control_points: (B, n_control_points) 优化后的角度控制点
    """
    B, N, _ = trajectory.shape
    device = trajectory.device
    
    # 计算弧长
    positions = trajectory[:, :, :2]
    distances = torch.norm(positions[:, 1:] - positions[:, :-1], dim=2)
    arc_lengths = distances.sum(dim=1)
    
    # 初始化控制点：均匀采样GT轨迹的角度
    indices = torch.linspace(0, N-1, n_control_points, device=device).long()
    control_points = trajectory[:, indices, 2].clone().detach()  # (B, n_control_points)
    control_points.requires_grad = True
    
    # 优化器
    optimizer = optim.Adam([control_points], lr=lr)
    
    # 目标：中间20点
    gt_middle = trajectory[:, 1:-1, :]  # (B, 20, 3)
    
    best_loss = float('inf')
    best_control_points = control_points.clone()
    
    for iteration in range(n_iterations):
        optimizer.zero_grad()
        
        # 从控制点重建轨迹
        reconstructed_list = []
        for b in range(B):
            recon = BSplineInterpolator.integrate_kinematics_from_angle_control_points(
                control_points[b:b+1],
                start_pose[b:b+1],
                arc_lengths[b].item(),
                n_path_points=21
            )  # (1, 20, 3)
            reconstructed_list.append(recon)
        reconstructed = torch.cat(reconstructed_list, dim=0)  # (B, 20, 3)
        
        # 损失：位置 + 角度
        pos_loss = torch.nn.functional.mse_loss(reconstructed[:, :, :2], gt_middle[:, :, :2])
        
        # 角度损失（使用sin/cos避免周期性问题）
        angle_loss = (
            torch.nn.functional.mse_loss(torch.sin(reconstructed[:, :, 2]), torch.sin(gt_middle[:, :, 2])) +
            torch.nn.functional.mse_loss(torch.cos(reconstructed[:, :, 2]), torch.cos(gt_middle[:, :, 2]))
        )
        
        # 平滑性正则化
        smooth_loss = torch.mean((control_points[:, 1:] - control_points[:, :-1])**2)
        
        total_loss = pos_loss + 0.5 * angle_loss + 0.01 * smooth_loss
        
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_control_points = control_points.detach().clone()
        
        total_loss.backward()
        optimizer.step()
        
        if iteration % 100 == 0:
            pos_error = torch.norm(reconstructed[:, :, :2] - gt_middle[:, :, :2], dim=2).mean()
            print(f"Iter {iteration}: loss={total_loss.item():.6f}, pos_error={pos_error.item():.4f}")
    
    print(f"Final best loss: {best_loss:.6f}")
    return best_control_points.detach()


def fit_control_points_simple(trajectory, n_control_points=15):
    """简单拟合：均匀采样角度（当前方法）"""
    B, N, _ = trajectory.shape
    device = trajectory.device
    indices = torch.linspace(0, N-1, n_control_points, device=device).long()
    control_angles = []
    for b in range(B):
        sampled_angles = trajectory[b, indices, 2]
        control_angles.append(sampled_angles)
    return torch.stack(control_angles, dim=0)


if __name__ == "__main__":
    # 加载真实数据测试
    from dataLoader_dit import UnevenPathDataLoader, PaddedSequence
    from torch.utils.data import DataLoader
    
    dataset = UnevenPathDataLoader(
        env_list=['env000008'],
        dataFolder='data/sim_dataset/train',
        compute_stability_map=False
    )
    
    loader = DataLoader(dataset, batch_size=1, collate_fn=PaddedSequence, shuffle=False)
    batch = next(iter(loader))
    
    trajectory = batch['trajectory']
    start_pose = batch['start_pose']
    
    print("="*60)
    print("对比两种拟合方法")
    print("="*60)
    
    # 方法1：简单采样
    print("\n【方法1：简单采样】")
    control_simple = fit_control_points_simple(trajectory, n_control_points=15)
    
    positions = trajectory[:, :, :2]
    distances = torch.norm(positions[:, 1:] - positions[:, :-1], dim=2)
    arc_length = distances.sum(dim=1)
    
    recon_simple = BSplineInterpolator.integrate_kinematics_from_angle_control_points(
        control_simple,
        start_pose,
        arc_length[0].item(),
        n_path_points=21
    )
    
    gt_middle = trajectory[:, 1:-1, :]
    pos_error_simple = torch.norm(recon_simple[:, :, :2] - gt_middle[:, :, :2], dim=2)
    print(f"位置误差: 均值={pos_error_simple.mean():.4f}, 最大={pos_error_simple.max():.4f}")
    
    # 方法2：优化
    print("\n【方法2：梯度优化】")
    control_optimized = fit_control_points_by_optimization(
        trajectory, start_pose, n_control_points=15, 
        n_iterations=500, lr=0.05
    )
    
    recon_optimized = BSplineInterpolator.integrate_kinematics_from_angle_control_points(
        control_optimized,
        start_pose,
        arc_length[0].item(),
        n_path_points=21
    )
    
    pos_error_optimized = torch.norm(recon_optimized[:, :, :2] - gt_middle[:, :, :2], dim=2)
    print(f"位置误差: 均值={pos_error_optimized.mean():.4f}, 最大={pos_error_optimized.max():.4f}")
    
    print(f"\n改进: {(pos_error_simple.mean() - pos_error_optimized.mean()) / pos_error_simple.mean() * 100:.1f}%")

"""
重写版B样条拟合测试
使用最小二乘法直接求解控制点，确保数量可控且拟合精度高
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
import sys


# ==========================================
# 核心：基于矩阵的B样条拟合（最小二乘）
# ==========================================

def create_bspline_basis_matrix(num_traj_points, num_control_points, degree=3):
    """
    创建B样条基函数矩阵
    Args:
        num_traj_points: 轨迹点数量
        num_control_points: 控制点数量
        degree: B样条阶数
    Returns:
        basis_matrix: (num_traj_points, num_control_points) - 基函数矩阵
        knot_vector: 节点向量
    """
    # 创建clamped均匀节点向量
    k = degree
    n = num_control_points
    kv = np.zeros(n + k + 1)
    num_internal = n + k + 1 - 2 * (k + 1)
    if num_internal > 0:
        kv[k+1:n] = np.linspace(0, 1, num_internal + 2)[1:-1]
    kv[n:] = 1.0
    
    # 均匀采样参数u
    u_vec = np.linspace(0, 1, num_traj_points)
    
    # 计算基函数矩阵
    basis_matrix = np.zeros((num_traj_points, num_control_points))
    for i in range(num_control_points):
        c = np.zeros(num_control_points)
        c[i] = 1.0
        spl = BSpline(kv, c, k)
        basis_matrix[:, i] = spl(u_vec)
    
    return basis_matrix, kv


def fit_bspline_least_squares(trajectory, num_control_points=20, degree=3):
    """
    使用最小二乘法拟合B样条控制点
    
    核心思想：
        P = M @ C  (轨迹点 = 基函数矩阵 @ 控制点)
        最小化 ||P - M @ C||^2
        解: C = (M^T @ M)^-1 @ M^T @ P  (伪逆)
    
    Args:
        trajectory: (N, 2) 轨迹点
        num_control_points: 控制点数量
        degree: B样条阶数
    
    Returns:
        control_points: (num_control_points, 2) 控制点
        basis_matrix: 基函数矩阵（用于重建）
        knot_vector: 节点向量
    """
    N, D = trajectory.shape
    
    # 1. 创建基函数矩阵
    M, kv = create_bspline_basis_matrix(N, num_control_points, degree)
    
    # 2. 最小二乘求解: C = (M^T M)^-1 M^T P
    # 使用numpy的lstsq更稳定
    control_points = np.linalg.lstsq(M, trajectory, rcond=None)[0]
    
    return control_points, M, kv


def reconstruct_from_basis_matrix(control_points, basis_matrix):
    """
    从控制点和基函数矩阵重建轨迹
    
    Args:
        control_points: (num_control_points, 2)
        basis_matrix: (num_traj_points, num_control_points)
    
    Returns:
        trajectory: (num_traj_points, 2)
    """
    return basis_matrix @ control_points


# ==========================================
# PyTorch可微版本
# ==========================================

class DifferentiableBSpline(nn.Module):
    """可微的B样条层"""
    def __init__(self, num_control_points=20, num_output_points=100, degree=3):
        super().__init__()
        self.num_cp = num_control_points
        self.num_out = num_output_points
        self.degree = degree
        
        # 预计算基函数矩阵
        basis_matrix, _ = create_bspline_basis_matrix(num_output_points, num_control_points, degree)
        self.register_buffer('basis_matrix', torch.tensor(basis_matrix, dtype=torch.float32))

    def forward(self, control_points):
        """
        Args:
            control_points: (Batch, Num_CP, 2)
        Returns:
            trajectory: (Batch, Num_Out, 2)
        """
        # P = M @ C
        traj = torch.einsum('ij,bjk->bik', self.basis_matrix, control_points)
        return traj


class ArcLengthResampler(nn.Module):
    """可微的弧长重采样层"""
    def __init__(self, num_output_points=100):
        super().__init__()
        self.num_out = num_output_points

    def forward(self, raw_traj):
        """
        Args:
            raw_traj: (Batch, N_dense, 2)
        Returns:
            resampled_traj: (Batch, num_output_points, 2)
        """
        B, N, D = raw_traj.shape
        
        # 计算累积弧长
        diff = raw_traj[:, 1:] - raw_traj[:, :-1]
        seg_lengths = torch.norm(diff, dim=-1) + 1e-8
        
        cum_lengths = torch.zeros(B, N, device=raw_traj.device, dtype=raw_traj.dtype)
        cum_lengths[:, 1:] = torch.cumsum(seg_lengths, dim=1)
        
        # 归一化
        total_length = cum_lengths[:, -1:] + 1e-8
        normalized_cum = cum_lengths / total_length
        
        # 均匀采样目标
        target_u = torch.linspace(0, 1, self.num_out, device=raw_traj.device).expand(B, -1)
        
        # 可微插值
        indices = torch.searchsorted(normalized_cum, target_u)
        indices = torch.clamp(indices, 1, N - 1)
        
        idx_lower = indices - 1
        idx_upper = indices
        
        val_lower = torch.gather(normalized_cum, 1, idx_lower)
        val_upper = torch.gather(normalized_cum, 1, idx_upper)
        
        d_val = val_upper - val_lower + 1e-8
        alpha = (target_u - val_lower) / d_val
        
        idx_lower_ex = idx_lower.unsqueeze(-1).expand(-1, -1, 2)
        idx_upper_ex = idx_upper.unsqueeze(-1).expand(-1, -1, 2)
        
        p_lower = torch.gather(raw_traj, 1, idx_lower_ex)
        p_upper = torch.gather(raw_traj, 1, idx_upper_ex)
        
        resampled = p_lower + alpha.unsqueeze(-1) * (p_upper - p_lower)
        
        return resampled


# ==========================================
# 评估函数
# ==========================================

def compute_trajectory_angle(trajectory):
    """计算轨迹角度"""
    dx = np.zeros(len(trajectory))
    dy = np.zeros(len(trajectory))
    
    dx[1:-1] = (trajectory[2:, 0] - trajectory[:-2, 0]) / 2
    dy[1:-1] = (trajectory[2:, 1] - trajectory[:-2, 1]) / 2
    
    dx[0] = trajectory[1, 0] - trajectory[0, 0]
    dy[0] = trajectory[1, 1] - trajectory[0, 1]
    dx[-1] = trajectory[-1, 0] - trajectory[-2, 0]
    dy[-1] = trajectory[-1, 1] - trajectory[-2, 1]
    
    angles = np.arctan2(dy, dx)
    return angles


def compute_reconstruction_errors(original_traj, reconstructed_traj):
    """计算重建误差"""
    pos_error = np.linalg.norm(original_traj - reconstructed_traj, axis=1)
    
    angles_orig = compute_trajectory_angle(original_traj)
    angles_recon = compute_trajectory_angle(reconstructed_traj)
    angle_diff = np.arctan2(np.sin(angles_recon - angles_orig), np.cos(angles_recon - angles_orig))
    
    return {
        'pos_error_mean': np.mean(pos_error),
        'pos_error_max': np.max(pos_error),
        'pos_error_std': np.std(pos_error),
        'pos_errors': pos_error,
        'angle_error_mean': np.degrees(np.mean(np.abs(angle_diff))),
        'angle_error_max': np.degrees(np.max(np.abs(angle_diff))),
        'angle_diffs': np.degrees(angle_diff)
    }


def generate_test_trajectory(traj_type='sine', num_points=100):
    """生成测试轨迹"""
    t = np.linspace(0, 1, num_points)
    
    if traj_type == 'sine':
        x = t * 10
        y = 2 * np.sin(2 * np.pi * t * 3)
    elif traj_type == 'spiral':
        theta = t * 4 * np.pi
        r = t * 5
        x = r * np.cos(theta)
        y = r * np.sin(theta)
    elif traj_type == 's_curve':
        x = t * 10
        y = 5 / (1 + np.exp(-10 * (t - 0.5)))
    elif traj_type == 'complex':
        x = t * 10 + np.sin(4 * np.pi * t)
        y = 2 * np.sin(2 * np.pi * t) + 0.5 * np.cos(8 * np.pi * t)
    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")
    
    trajectory = np.stack([x, y], axis=1)
    return trajectory


# ==========================================
# 测试函数
# ==========================================

def test_least_squares_fitting():
    """测试最小二乘拟合方法"""
    print("\n" + "="*80)
    print("Test 1: Least Squares B-Spline Fitting (Direct Matrix Method)")
    print("="*80)
    
    test_types = ['sine', 'spiral', 's_curve', 'complex']
    control_point_nums = [20, 30, 40]
    
    results = {}
    
    for num_cp in control_point_nums:
        print(f"\n{'='*80}")
        print(f"Testing with {num_cp} control points")
        print(f"{'='*80}")
        
        results[num_cp] = {}
        
        for traj_type in test_types:
            print(f"\n--- {traj_type} ---")
            
            # 生成轨迹
            traj_gt = generate_test_trajectory(traj_type, 100)
            
            # 最小二乘拟合
            control_points, basis_matrix, _ = fit_bspline_least_squares(
                traj_gt, num_control_points=num_cp, degree=3
            )
            
            print(f"  Fitted {len(control_points)} control points (expected {num_cp})")
            
            # 重建
            traj_recon = reconstruct_from_basis_matrix(control_points, basis_matrix)
            
            # 评估
            errors = compute_reconstruction_errors(traj_gt, traj_recon)
            results[num_cp][traj_type] = errors
            
            print(f"  Position Error: {errors['pos_error_mean']:.6f} (max: {errors['pos_error_max']:.6f})")
            print(f"  Angle Error:    {errors['angle_error_mean']:.4f}° (max: {errors['angle_error_max']:.4f}°)")
    
    # 汇总
    print(f"\n{'='*80}")
    print("Summary: Least Squares Fitting Results")
    print(f"{'='*80}")
    print(f"{'Num CP':<10} {'Trajectory':<15} {'Pos Mean':<12} {'Angle Mean':<12} {'Angle Max':<12}")
    print("-"*80)
    for num_cp in control_point_nums:
        for traj_type in test_types:
            e = results[num_cp][traj_type]
            print(f"{num_cp:<10} {traj_type:<15} {e['pos_error_mean']:<12.6f} "
                  f"{e['angle_error_mean']:<12.4f} {e['angle_error_max']:<12.4f}")
    
    return results


def test_pytorch_optimization():
    """测试PyTorch优化方法"""
    print("\n" + "="*80)
    print("Test 2: PyTorch Optimization with Arc-Length Resampling")
    print("="*80)
    
    test_types = ['spiral']  # 只测试螺旋线
    num_cp_list = [20, 30, 40]
    
    for num_cp in num_cp_list:
        print(f"\n{'='*80}")
        print(f"Testing with {num_cp} control points")
        print(f"{'='*80}")
        
        for traj_type in test_types:
            print(f"\n--- {traj_type} ---")
            
            # 生成轨迹
            traj_gt = generate_test_trajectory(traj_type, 100)
            traj_tensor = torch.tensor(traj_gt, dtype=torch.float32).unsqueeze(0)
            
            # 定义模型
            bspline_layer = DifferentiableBSpline(num_control_points=num_cp, num_output_points=500, degree=3)
            resampler = ArcLengthResampler(num_output_points=100)
            
            # 用最小二乘结果初始化
            cp_init, _, _ = fit_bspline_least_squares(traj_gt, num_control_points=num_cp, degree=3)
            control_points = torch.nn.Parameter(
                torch.tensor(cp_init, dtype=torch.float32).unsqueeze(0)
            )
            
            optimizer = torch.optim.Adam([control_points], lr=0.01)
            
            print(f"  Optimizing (initialized with least squares)...")
            for i in range(2000):
                optimizer.zero_grad()
                
                dense_traj = bspline_layer(control_points)
                final_traj = resampler(dense_traj)
                
                loss_mse = torch.nn.functional.mse_loss(final_traj, traj_tensor)
                
                # 轻微的平滑正则
                diff = control_points[:, 1:] - control_points[:, :-1]
                loss_smooth = 0.0001 * torch.mean(diff ** 2)
                
                loss = loss_mse + loss_smooth
                loss.backward()
                optimizer.step()
                
                if i % 500 == 0:
                    print(f"    Iter {i:4d}: MSE={loss_mse.item():.6f}")
            
            # 评估
            with torch.no_grad():
                dense = bspline_layer(control_points)
                recon = resampler(dense)
                recon_np = recon.squeeze().numpy()
                
                errors = compute_reconstruction_errors(traj_gt, recon_np)
                
                print(f"\n  Final Results:")
                print(f"    Position Error: {errors['pos_error_mean']:.6f}")
                print(f"    Angle Error:    {errors['angle_error_mean']:.4f}°")
                print(f"    Gradient flow:  {'✓' if control_points.grad is not None else '✗'}")


def visualize_comparison(traj_gt, traj_recon, control_points, title, filename):
    """可视化对比"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：轨迹对比
    ax = axes[0]
    ax.plot(traj_gt[:, 0], traj_gt[:, 1], 'b-', linewidth=2, label='Ground Truth', alpha=0.7)
    ax.plot(traj_recon[:, 0], traj_recon[:, 1], 'r--', linewidth=2, label='Reconstructed', alpha=0.7)
    ax.scatter(control_points[:, 0], control_points[:, 1], c='green', s=100, marker='x', 
              linewidths=3, label=f'Control Points ({len(control_points)})', zorder=5)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 右图：误差曲线
    ax = axes[1]
    errors = compute_reconstruction_errors(traj_gt, traj_recon)
    ax.plot(errors['pos_errors'], 'b-', linewidth=2, label='Position Error')
    ax.axhline(y=errors['pos_error_mean'], color='r', linestyle='--', 
              label=f"Mean: {errors['pos_error_mean']:.4f}")
    ax.legend()
    ax.set_title('Position Error')
    ax.set_xlabel('Point Index')
    ax.set_ylabel('Error')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close(fig)


def test_with_visualization():
    """带可视化的测试"""
    print("\n" + "="*80)
    print("Test 3: Visualization Test")
    print("="*80)
    
    traj_type = 'spiral'
    traj_gt = generate_test_trajectory(traj_type, 100)
    
    for num_cp in [20, 30, 40]:
        print(f"\n--- Testing {num_cp} control points on {traj_type} ---")
        
        # 拟合
        control_points, basis_matrix, _ = fit_bspline_least_squares(
            traj_gt, num_control_points=num_cp, degree=3
        )
        
        # 重建
        traj_recon = reconstruct_from_basis_matrix(control_points, basis_matrix)
        
        # 评估
        errors = compute_reconstruction_errors(traj_gt, traj_recon)
        print(f"  Pos Error: {errors['pos_error_mean']:.6f}, Angle Error: {errors['angle_error_mean']:.4f}°")
        
        # 可视化
        visualize_comparison(
            traj_gt, traj_recon, control_points,
            f"Spiral - {num_cp} Control Points (Least Squares)",
            f"bspline_spiral_{num_cp}cp.png"
        )


if __name__ == '__main__':
    print("\n" + "="*80)
    print("B-Spline Fitting Test Suite (Least Squares Method)")
    print("="*80)
    
    # Test 1: 最小二乘拟合测试
    results_ls = test_least_squares_fitting()
    
    # Test 2: PyTorch优化测试
    test_pytorch_optimization()
    
    # Test 3: 可视化测试
    test_with_visualization()
    
    print("\n" + "="*80)
    print("✓ All tests completed!")
    print("="*80)

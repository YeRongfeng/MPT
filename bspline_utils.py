"""
B样条拟合和重建工具类
提供基于最小二乘法的B样条拟合和重建功能
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import BSpline


# ==========================================
# 核心：基于矩阵的B样条拟合（最小二乘）
# ==========================================

def create_bspline_basis_matrix(num_traj_points, num_control_points, degree=3):
    """
    创建B样条基函数矩阵
    
    Args:
        num_traj_points: 轨迹点数量
        num_control_points: 控制点数量
        degree: B样条阶数（默认3次）
        
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


def create_bspline_geometry_matrices(
    num_traj_points,
    num_control_points,
    degree=3,
):
    """创建 B-spline 的位置、一阶导数和二阶导数基矩阵。"""
    if degree < 2:
        raise ValueError("analytic curvature requires a B-spline degree >= 2")

    basis_matrix, knot_vector = create_bspline_basis_matrix(
        num_traj_points,
        num_control_points,
        degree,
    )
    parameters = np.linspace(0.0, 1.0, num_traj_points)
    first_matrix = np.empty_like(basis_matrix)
    second_matrix = np.empty_like(basis_matrix)

    for index in range(num_control_points):
        coefficients = np.zeros(num_control_points)
        coefficients[index] = 1.0
        spline = BSpline(knot_vector, coefficients, degree)
        first_matrix[:, index] = spline.derivative(1)(parameters)
        second_matrix[:, index] = spline.derivative(2)(parameters)

    return basis_matrix, first_matrix, second_matrix, knot_vector


def fit_bspline_least_squares(trajectory, num_control_points=20, degree=3):
    """
    使用最小二乘法拟合B样条控制点
    
    核心思想：
        P = M @ C  (轨迹点 = 基函数矩阵 @ 控制点)
        最小化 ||P - M @ C||^2
        解: C = (M^T @ M)^-1 @ M^T @ P  (伪逆)
    
    Args:
        trajectory: (N, 2) 轨迹点数组
        num_control_points: 控制点数量
        degree: B样条阶数（默认3次）
    
    Returns:
        control_points: (num_control_points, 2) 控制点数组
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


def reconstruct_from_control_points(control_points, num_output_points, degree=3):
    """
    从控制点重建密集轨迹
    
    Args:
        control_points: (num_control_points, 2) 控制点数组
        num_output_points: 输出轨迹点数量
        degree: B样条阶数（默认3次）
    
    Returns:
        trajectory: (num_output_points, 2) 重建的轨迹点
    """
    num_cp = len(control_points)
    
    # 创建基函数矩阵
    M, _ = create_bspline_basis_matrix(num_output_points, num_cp, degree)
    
    # 重建轨迹: P = M @ C
    trajectory = M @ control_points
    
    return trajectory


def reconstruct_from_basis_matrix(control_points, basis_matrix):
    """
    从控制点和预计算的基函数矩阵重建轨迹（更高效）
    
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
    """可微的B样条层（支持梯度回传）"""
    
    def __init__(self, num_control_points=20, num_output_points=100, degree=3):
        """
        Args:
            num_control_points: 控制点数量
            num_output_points: 输出轨迹点数量
            degree: B样条阶数
        """
        super().__init__()
        self.num_cp = num_control_points
        self.num_out = num_output_points
        self.degree = degree
        
        # 预计算基函数矩阵
        (
            basis_matrix,
            first_basis_matrix,
            second_basis_matrix,
            _,
        ) = create_bspline_geometry_matrices(
            num_output_points,
            num_control_points,
            degree,
        )
        self.register_buffer('basis_matrix', torch.tensor(basis_matrix, dtype=torch.float32))
        # Derivative matrices are deterministic helpers, not learned state.
        # persistent=False keeps existing checkpoints that contain this layer
        # backward-compatible.
        self.register_buffer(
            'first_basis_matrix',
            torch.tensor(first_basis_matrix, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            'second_basis_matrix',
            torch.tensor(second_basis_matrix, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, control_points):
        """
        前向传播
        
        Args:
            control_points: (Batch, Num_CP, 2) 控制点张量
            
        Returns:
            trajectory: (Batch, Num_Out, 2) 轨迹张量
        """
        # P = M @ C
        traj = torch.einsum('ij,bjk->bik', self.basis_matrix, control_points)
        return traj

    def evaluate_geometry(self, control_points):
        """返回位置及解析 B-spline 导数、切向 yaw 和绝对曲率。"""
        position = self(control_points)
        first = torch.einsum(
            'ij,bjk->bik',
            self.first_basis_matrix,
            control_points,
        )
        second = torch.einsum(
            'ij,bjk->bik',
            self.second_basis_matrix,
            control_points,
        )
        speed = torch.linalg.vector_norm(first, dim=-1)
        yaw = torch.atan2(first[..., 1], first[..., 0])
        cross = (
            first[..., 0] * second[..., 1]
            - first[..., 1] * second[..., 0]
        )
        curvature = torch.abs(cross) / speed.clamp_min(1e-10).pow(3)
        return {
            'position': position,
            'first_derivative': first,
            'second_derivative': second,
            'speed': speed,
            'yaw': yaw,
            'curvature': curvature,
        }


class ArcLengthResampler(nn.Module):
    """可微的弧长重采样层"""
    
    def __init__(self, num_output_points=100):
        """
        Args:
            num_output_points: 重采样后的点数量
        """
        super().__init__()
        self.num_out = num_output_points

    def forward(self, raw_traj):
        """
        弧长重采样
        
        Args:
            raw_traj: (Batch, N_dense, 2) 原始密集轨迹
            
        Returns:
            resampled_traj: (Batch, num_output_points, 2) 重采样后的轨迹
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
# 评估工具
# ==========================================

def compute_trajectory_angle(trajectory):
    """
    计算轨迹的切向角度
    
    Args:
        trajectory: (N, 2) 轨迹点数组
        
    Returns:
        angles: (N,) 每个点的切向角度（弧度）
    """
    dx = np.zeros(len(trajectory))
    dy = np.zeros(len(trajectory))
    
    # 中心差分
    dx[1:-1] = (trajectory[2:, 0] - trajectory[:-2, 0]) / 2
    dy[1:-1] = (trajectory[2:, 1] - trajectory[:-2, 1]) / 2
    
    # 边界点用前向/后向差分
    dx[0] = trajectory[1, 0] - trajectory[0, 0]
    dy[0] = trajectory[1, 1] - trajectory[0, 1]
    dx[-1] = trajectory[-1, 0] - trajectory[-2, 0]
    dy[-1] = trajectory[-1, 1] - trajectory[-2, 1]
    
    angles = np.arctan2(dy, dx)
    return angles


def compute_reconstruction_errors(original_traj, reconstructed_traj):
    """
    计算重建误差（位置误差和角度误差）
    
    Args:
        original_traj: (N, 2) 原始轨迹
        reconstructed_traj: (N, 2) 重建轨迹
        
    Returns:
        errors: dict包含各种误差指标
            - pos_error_mean: 平均位置误差
            - pos_error_max: 最大位置误差
            - pos_error_std: 位置误差标准差
            - pos_errors: 每个点的位置误差数组
            - angle_error_mean: 平均角度误差（度）
            - angle_error_max: 最大角度误差（度）
            - angle_diffs: 每个点的角度差异数组（度）
    """
    # 位置误差
    pos_error = np.linalg.norm(original_traj - reconstructed_traj, axis=1)
    
    # 角度误差
    angles_orig = compute_trajectory_angle(original_traj)
    angles_recon = compute_trajectory_angle(reconstructed_traj)
    angle_diff = np.arctan2(np.sin(angles_recon - angles_orig), 
                           np.cos(angles_recon - angles_orig))
    
    return {
        'pos_error_mean': np.mean(pos_error),
        'pos_error_max': np.max(pos_error),
        'pos_error_std': np.std(pos_error),
        'pos_errors': pos_error,
        'angle_error_mean': np.degrees(np.mean(np.abs(angle_diff))),
        'angle_error_max': np.degrees(np.max(np.abs(angle_diff))),
        'angle_diffs': np.degrees(angle_diff)
    }


# ==========================================
# 轨迹生成工具
# ==========================================

def generate_test_trajectory(traj_type='sine', num_points=100):
    """
    生成各种类型的测试轨迹
    
    Args:
        traj_type: 轨迹类型
            - 'sine': 正弦波
            - 'spiral': 螺旋线
            - 's_curve': S型曲线
            - 'complex': 复杂曲线
            - 'circle': 圆形
            - 'zigzag': 之字形
        num_points: 轨迹点数量
        
    Returns:
        trajectory: (num_points, 2) 轨迹数组
    """
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
    elif traj_type == 'circle':
        theta = t * 2 * np.pi
        x = 5 * np.cos(theta)
        y = 5 * np.sin(theta)
    elif traj_type == 'zigzag':
        x = t * 10
        y = 2 * np.abs(np.sin(8 * np.pi * t))
    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")
    
    trajectory = np.stack([x, y], axis=1)
    return trajectory

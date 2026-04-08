"""
An Efficient Trajectory Planner for Car-like Robots on Uneven Terrain
Implementation of: MINCO + Discrete Time Sampling + PHR-ALM + L-BFGS

This module implements the trajectory optimization backend with:
- Terrain pose mapping via trilinear interpolation
- MINCO trajectory parameterization
- PHR-ALM (Penalty-free Augmented Lagrangian Method) for constrained optimization
- L-BFGS inner loop for unconstrained minimization
- Automatic differentiation via PyTorch autograd
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# ==========================================
# Standalone time-mapping utilities
# (aligned with C++ expC2/logC2/getTtoTauGrad)
# ==========================================

def exp_c2(tau: torch.Tensor) -> torch.Tensor:
    """C2-smooth mapping from τ to T>0."""
    return MincoTrajectory.exp_c2(tau)


def log_c2(T: torch.Tensor) -> torch.Tensor:
    """Inverse C2-smooth mapping from T to τ."""
    return MincoTrajectory.log_c2(T)


def get_t_to_tau_grad(tau: torch.Tensor) -> torch.Tensor:
    """Gradient dT/dτ for C2-smooth mapping."""
    return MincoTrajectory.get_t_to_tau_grad(tau)


# ==========================================
# Data Structures & Configuration
# ==========================================

@dataclass
class OptimizationConfig:
    """Hyperparameters for trajectory optimization.

    调参建议（先后顺序）：
    1) 先保可行：优先看约束残差（v/a/kappa/attitude/sigma/non_holo）是否下降；
    2) 再控形状：调平滑项（rho_jerk、rho_theta_smooth）抑制抖动；
    3) 最后提效率：再调 rho_time 与 LBFGS 运行参数。

    经验做法：每次只改 1~3 个参数，幅度先用 1.5x~2x，避免同时大改导致难以定位原因。
    """
    # Sampling and discretization
    n_samples: int = 30  # 离散采样点数（align_cpp_sampling=False 时生效）；太小会漏约束，太大更慢
    int_K: int = 16  # 每段积分采样数（C++风格）；约束检出不稳可升到 20~24
    align_cpp_sampling: bool = True  # True: 用 piece_xy*(int_K+1)；更贴近 C++，通常更稳但更耗时
    n_control_points: int = 5  # 控制点数；增大更灵活（更易避障/贴地形），但优化更慢且更易震荡
    
    # Cost function weights (Eq. 28)
    rho_jerk: float = 1.0  # 轨迹平滑（XY jerk）；轨迹抖动/折线感强就增大，过大易“拉直”
    rho_terrain: float = 10.0  # 地形/风险代价权重；想更保守避高风险区域就增大
    rho_time: float = 100000.0  # 时间代价；增大=更快（更短总时长）但更易触发速度/加速度超限
    rho_heading: float = 50.0  # 航向与速度方向一致性；角度乱跳可增大，过大可能压制几何自由度
    rho_ref_xy: float = 200.0  # 对初始 XY 的锚定；优化发散/跑偏增大，想更自由探索就减小
    rho_ref_theta: float = 20.0  # 对初始 theta 的锚定；theta 反复翻转可增大
    rho_dyn_soft: float = 2000.0  # 动力学不等式软惩罚；约束长期不满足时先升这个（常用 1.5x~3x）
    rho_nonholo_soft: float = 2000.0  # 非完整约束软惩罚；侧滑残差大时增大
    rho_theta_smooth: float = 200.0  # theta_dot/theta_ddot 平滑；抑制航向锯齿，过大转向会变钝
    gravity: float = 9.81  # 重力项（地形耦合加速度）；通常保持 9.81，除非单位制不同
    dynamic_rescale_each_iter: bool = False  # 每轮ALM是否做时间缩放保可行；开了更稳但会改变收敛轨迹
    
    # Dynamic constraints
    v_max: float = 0.5  # 最大速度；若频繁 v 超限可增大总时间（降 rho_time）或适当放宽该值
    a_lon_max: float = 5.0  # 纵向加速度上限；过小会难收敛，过大可能轨迹“冲”
    a_lat_max: float = 10.0  # 横向加速度上限；转弯跟踪差可适当放宽，侧倾风险大则收紧
    max_kap: float = 2.1  # 曲率上限（优先项）；转弯不够可略增，转向振荡可略降
    delta_max: float = np.pi / 6  # 当 max_kap<=0 时用转角+轴距换算曲率上限
    L_w: float = 0.5  # 轴距；仅用于 delta_max->曲率换算
    
    # Physical constraints
    min_cxi: float = 0.8  # 姿态安全下界（cos xi）；越大越保守（允许坡度更小）
    c_min: float = 0.8  # min_cxi<=0 时回退值；一般与 min_cxi 保持一致
    sigma_max: float = 0.05  # 地表粗糙/风险上限；若 sigma 违约长期主导，可先适当放宽再逐步收紧
    
    # ALM parameters (aligned with reference/alm_traj_opt.h)
    rho: float = 1.0  # ALM 初始惩罚；约束压不下可提高到 2~10
    beta: float = 1000.0  # rho 上限；太小可能压不住约束，太大数值易僵硬
    gamma: float = 1.0  # rho 增长率；大=更激进压约束，小=更平滑更稳
    epsilon_con: float = 0.001  # 收敛阈值（残差）；越小要求越严格，耗时更长
    max_iter: int = 10  # ALM 外层轮数；约束未收敛时优先增加到 20~30
    
    # L-BFGS parameters (aligned with reference/lbfgs.hpp)
    mem_size: int = 256  # L-BFGS 历史大小；大一些常更稳但更占内存
    past: int = 3  # 与原实现对齐的历史窗口参数（当前主要保留兼容）
    g_epsilon: float = 1e-3  # 梯度停止阈值（保留参数）
    min_step: float = 1e-32  # 最小步长（保留参数）
    delta: float = 1e-4  # 收敛检测相关阈值（保留参数）
    inner_max_iter: int = 10000  # 内层理论上限；实际受 inner_max_iter_runtime 限制
    lbfgs_line_search: str = "strong_wolfe"  # 建议保持 strong_wolfe，禁用后常更快但更易发散
    lbfgs_use_line_search_runtime: bool = True  # 运行时是否启用线搜索；稳定优先建议 True
    # Practical runtime controls for PyTorch LBFGS (to avoid long apparent hangs)
    inner_max_iter_runtime: int = 300  # 每轮ALM内层实际步数；慢可降到 50~150，质量不足再升
    closure_log_interval: int = 20  # 内层日志间隔；调试时可设小，常规运行可设大
    
    # Robustness controls
    tau_clip: float = 12.0  # 时间参数 τ 裁剪；防止极端时间尺度导致数值炸掉
    dual_clip: float = 1e6  # 乘子裁剪；出现乘子爆炸/NaN 时可适当调小
    constraint_clip: float = 1e4  # 约束值裁剪；防止异常样本主导梯度
    
    # Scaling mechanism
    use_scaling: bool = True  # 是否启用缩放机制；一般建议开
    
    # Numerical
    eps: float = 1e-8  # 数值稳定项，防止除零
    delta_v: float = 0.01  # 速度范数下界平滑项；太小可能不稳，太大可能影响低速曲率估计
    safe_max_value: float = 1e12  # 仅用于 NaN/Inf 保护，不是硬截断优化目标


# ==========================================
# 1. Terrain & Pose Mapping Module
# ==========================================

class TerrainPoseMapper:
    """
    Maps 2D planar position (x, y) and heading theta to 3D terrain information.
    Uses trilinear interpolation to provide continuous, differentiable queries.
    Corresponds to Equations 7-8 in the paper (Terrain Pose Mapping).
    """
    
    def __init__(self, grid_map: Dict[str, torch.Tensor], device: torch.device = torch.device("cpu")):
        """
        Args:
            grid_map: Dictionary containing:
                - 'z': Height field [Nx, Ny, Ntheta]
                - 'zb': Terrain normal (as [a, b, c]) [Nx, Ny, Ntheta, 3]
                - 'sigma': Surface roughness/curvature [Nx, Ny, Ntheta]
                - 'x_range': [x_min, x_max]
                - 'y_range': [y_min, y_max]
                - 'theta_range': [theta_min, theta_max]
            device: torch device
        """
        self.device = device
        self.z = grid_map['z'].to(device)
        self.zb = grid_map['zb'].to(device)
        self.sigma = grid_map['sigma'].to(device)
        
        # Grid bounds
        self.x_min, self.x_max = grid_map['x_range']
        self.y_min, self.y_max = grid_map['y_range']
        self.theta_min, self.theta_max = grid_map['theta_range']
        
        # Grid dimensions
        self.nx, self.ny, self.ntheta = self.z.shape[:3]
        
    def _normalize_coords(self, x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize physical coordinates to grid indices [0, 1]"""
        x_norm = (x - self.x_min) / (self.x_max - self.x_min)
        y_norm = (y - self.y_min) / (self.y_max - self.y_min)
        theta_norm = (theta - self.theta_min) / (self.theta_max - self.theta_min)
        return x_norm, y_norm, theta_norm
    
    def _trilinear_interp(self, grid: torch.Tensor, x_norm: torch.Tensor, y_norm: torch.Tensor, theta_norm: torch.Tensor) -> torch.Tensor:
        """
        Trilinear interpolation for smooth, differentiable grid sampling.
        """
        # Clamp to valid range
        x_norm = torch.clamp(x_norm, 0, 1)
        y_norm = torch.clamp(y_norm, 0, 1)
        theta_norm = torch.clamp(theta_norm, 0, 1)
        
        # Convert to grid indices
        x_idx = x_norm * (self.nx - 1)
        y_idx = y_norm * (self.ny - 1)
        theta_idx = theta_norm * (self.ntheta - 1)
        
        # Extract integer and fractional parts
        x_i = torch.floor(x_idx).long()
        y_i = torch.floor(y_idx).long()
        theta_i = torch.floor(theta_idx).long()
        
        dx = x_idx - x_i.float()
        dy = y_idx - y_i.float()
        dtheta = theta_idx - theta_i.float()
        
        # Clamp indices
        x_i = torch.clamp(x_i, 0, self.nx - 2)
        y_i = torch.clamp(y_i, 0, self.ny - 2)
        theta_i = torch.clamp(theta_i, 0, self.ntheta - 2)
        
        # Trilinear interpolation: sample 8 corners
        v000 = grid[x_i, y_i, theta_i]
        v100 = grid[x_i + 1, y_i, theta_i]
        v010 = grid[x_i, y_i + 1, theta_i]
        v110 = grid[x_i + 1, y_i + 1, theta_i]
        v001 = grid[x_i, y_i, theta_i + 1]
        v101 = grid[x_i + 1, y_i, theta_i + 1]
        v011 = grid[x_i, y_i + 1, theta_i + 1]
        v111 = grid[x_i + 1, y_i + 1, theta_i + 1]
        
        # Interpolate
        v00 = v000 * (1 - dx) + v100 * dx
        v10 = v010 * (1 - dx) + v110 * dx
        v01 = v001 * (1 - dx) + v101 * dx
        v11 = v011 * (1 - dx) + v111 * dx
        
        v0 = v00 * (1 - dy) + v10 * dy
        v1 = v01 * (1 - dy) + v11 * dy
        
        result = v0 * (1 - dtheta) + v1 * dtheta
        return result
    
    def query(self, x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Query terrain at position (x, y, theta).
        
        Returns:
            z: Height [batch_size]
            zb: Normal vector [batch_size, 3] with components (a, b, c)
            sigma: Surface roughness/curvature [batch_size]
        """
        x_norm, y_norm, theta_norm = self._normalize_coords(x, y, theta)
        
        # Sample from grids
        z = self._trilinear_interp(self.z, x_norm, y_norm, theta_norm)
        sigma = self._trilinear_interp(self.sigma, x_norm, y_norm, theta_norm)
        
        # For vector field zb, interpolate each component
        zb_a = self._trilinear_interp(self.zb[..., 0], x_norm, y_norm, theta_norm)
        zb_b = self._trilinear_interp(self.zb[..., 1], x_norm, y_norm, theta_norm)
        zb_c = self._trilinear_interp(self.zb[..., 2], x_norm, y_norm, theta_norm)
        
        zb = torch.stack([zb_a, zb_b, zb_c], dim=-1)
        return z, zb, sigma

    def query_zb_sigma(self, x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query only normal `zb` and roughness `sigma`.
        This is used in optimization where height `z` is not needed.
        """
        x_norm, y_norm, theta_norm = self._normalize_coords(x, y, theta)

        sigma = self._trilinear_interp(self.sigma, x_norm, y_norm, theta_norm)
        zb_a = self._trilinear_interp(self.zb[..., 0], x_norm, y_norm, theta_norm)
        zb_b = self._trilinear_interp(self.zb[..., 1], x_norm, y_norm, theta_norm)
        zb_c = self._trilinear_interp(self.zb[..., 2], x_norm, y_norm, theta_norm)
        zb = torch.stack([zb_a, zb_b, zb_c], dim=-1)
        return zb, sigma


# ==========================================
# 2. MINCO Trajectory Parameterization
# ==========================================

class MincoTrajectory:
    """
    MINCO (Minimum Control): Represents trajectory using polynomial segments.
    Corresponds to Section III.B of the paper (Trajectory Parameterization).
    
    The trajectory is parameterized by:
    - Control points q (positions)
    - Time durations tau (via exponential mapping for T > 0)
    
    Uses quintic (5th-order) polynomials for smooth, differentiable trajectories.
    """
    
    def __init__(self, config: OptimizationConfig, device: torch.device = torch.device("cpu")):
        self.config = config
        self.device = device
        self.n_samples = config.n_samples
        self.n_ctrl_pts = config.n_control_points
    
    @staticmethod
    def exp_c2(tau: torch.Tensor) -> torch.Tensor:
        """
        Exponential mapping: tau -> T with T > 0
        Uses C2-smooth approximation from C++ implementation for numerical stability.
        T = (0.5*tau + 1)*tau + 1  when tau > 0
        T = 1 / ((0.5*tau - 1)*tau + 1)  when tau <= 0
        
        Args:
            tau: Log-time parameter
        
        Returns:
            T: Positive time duration
        """
        result = torch.zeros_like(tau)
        positive_mask = tau > 0
        negative_mask = ~positive_mask
        
        # For positive tau
        result[positive_mask] = (0.5 * tau[positive_mask] + 1.0) * tau[positive_mask] + 1.0
        
        # For non-positive tau
        denom = (0.5 * tau[negative_mask] - 1.0) * tau[negative_mask] + 1.0
        result[negative_mask] = 1.0 / (denom + 1e-8)
        
        return result
    
    @staticmethod
    def log_c2(T: torch.Tensor) -> torch.Tensor:
        """
        Inverse exponential mapping: T -> tau
        tau = sqrt(2*T - 1) - 1  when T > 1
        tau = 1 - sqrt(2/T - 1)  when T <= 1
        
        Args:
            T: Time duration
        
        Returns:
            tau: Log-time parameter
        """
        result = torch.zeros_like(T)
        large_mask = T > 1.0
        small_mask = ~large_mask
        
        # For T > 1
        result[large_mask] = torch.sqrt(2.0 * T[large_mask] - 1.0) - 1.0
        
        # For T <= 1
        result[small_mask] = 1.0 - torch.sqrt(2.0 / (T[small_mask] + 1e-8) - 1.0)
        
        return result
    
    @staticmethod
    def get_t_to_tau_grad(tau: torch.Tensor) -> torch.Tensor:
        """
        Get dT/dtau gradient for chain rule.
        dT/dtau = tau + 1  when tau > 0
        dT/dtau = (1 - tau) / ((0.5*tau - 1)*tau + 1)^2  when tau <= 0
        """
        result = torch.zeros_like(tau)
        positive_mask = tau > 0
        negative_mask = ~positive_mask
        
        result[positive_mask] = tau[positive_mask] + 1.0
        
        denom = (0.5 * tau[negative_mask] - 1.0) * tau[negative_mask] + 1.0
        result[negative_mask] = (1.0 - tau[negative_mask]) / (denom ** 2 + 1e-8)
        
        return result
    
    def cal_t_from_tau(self, tau_scalar: torch.Tensor, n_piece: int) -> torch.Tensor:
        """
        C++ `calTfromTau` equivalent:
        T.setConstant(expC2(tau) / T.size())
        """
        total_t = self.exp_c2(tau_scalar.reshape(-1)[0])
        return torch.ones((n_piece,), device=self.device, dtype=tau_scalar.dtype) * (total_t / max(n_piece, 1))

    def _exp_map_time(
        self,
        tau: torch.Tensor,
        n_piece_xy: int,
        n_piece_theta: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Time mapping.
        - If `tau` is scalar: use C++ style uniform time allocation for all pieces.
        - If `tau` is vector: keep backward compatibility with per-segment durations.
        """
        tau_flat = tau.reshape(-1)
        if tau_flat.numel() == 1:
            T_xy = self.cal_t_from_tau(tau_flat[0], n_piece_xy)
            T_theta = self.cal_t_from_tau(tau_flat[0], n_piece_theta)
            return T_xy, T_theta

        # backward compatible branch
        if tau_flat.numel() == (n_piece_xy + n_piece_theta):
            T_xy = self.exp_c2(tau_flat[:n_piece_xy])
            T_theta = self.exp_c2(tau_flat[n_piece_xy:])
            return T_xy, T_theta

        # fallback: use first tau as scalar if shape is unexpected
        T_xy = self.cal_t_from_tau(tau_flat[0], n_piece_xy)
        T_theta = self.cal_t_from_tau(tau_flat[0], n_piece_theta)
        return T_xy, T_theta
    
    def _build_minco_matrices(self, T: torch.Tensor, boundary_start: torch.Tensor, boundary_end: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build quintic polynomial matrices for MINCO.
        Solves for polynomial coefficients given boundary conditions and segment times.
        
        For a single segment [0, T]:
        p(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5
        
        Boundary conditions (pos, vel, acc at start and end):
        p(0) = p0, p'(0) = v0, p''(0) = a0
        p(T) = pf, p'(T) = vf, p''(T) = af
        """
        # This would build the constraint matrix for each segment
        # For simplicity, we use a factorized representation:
        # The polynomial is constructed from boundary values and can be queried at any time t
        
        return boundary_start, boundary_end
    
    def _eval_poly_at_time(self, 
                          segment_idx: int,
                          time_in_segment: torch.Tensor,
                          T: torch.Tensor,
                          boundary_start: torch.Tensor,
                          boundary_end: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate quintic polynomial and its derivatives at time t in segment.
        Returns: [pos, vel, acc, jerk] or higher order derivatives as needed.
        
        Uses Hermite basis polynomials for stable interpolation.
        """
        # Normalized time s = t / T ∈ [0, 1]
        T_seg = T[segment_idx]
        s = torch.clamp(time_in_segment / (T_seg + self.config.eps), 0, 1)
        
        # Extract boundary values
        p0, v0, a0 = boundary_start
        pf, vf, af = boundary_end
        
        # Quintic Hermite basis polynomials and derivatives
        # h00(s) = 1 - 10s^3 + 15s^4 - 6s^5
        s2 = s * s
        s3 = s2 * s
        s4 = s3 * s
        s5 = s4 * s
        
        h00 = 1 - 10*s3 + 15*s4 - 6*s5
        h10 = s - 6*s3 + 8*s4 - 3*s5
        h01 = 10*s3 - 15*s4 + 6*s5
        h11 = -4*s3 + 7*s4 - 3*s5
        
        # First derivative basis (w.r.t. s)
        dh00_ds = -30*s2 + 60*s3 - 30*s4
        dh10_ds = 1 - 18*s2 + 32*s3 - 15*s4
        dh01_ds = 30*s2 - 60*s3 + 30*s4
        dh11_ds = -12*s2 + 28*s3 - 15*s4
        
        # Second derivative basis (w.r.t. s)
        d2h00_ds2 = -60*s + 180*s2 - 120*s3
        d2h10_ds2 = -36*s + 96*s2 - 60*s3
        d2h01_ds2 = 60*s - 180*s2 + 120*s3
        d2h11_ds2 = -24*s + 84*s2 - 60*s3
        
        # Third derivative basis (w.r.t. s)
        d3h00_ds3 = -60 + 360*s - 360*s2
        d3h10_ds3 = -36 + 192*s - 180*s2
        d3h01_ds3 = 60 - 360*s + 360*s2
        d3h11_ds3 = -24 + 168*s - 180*s2
        
        # Position interpolation
        pos = h00 * p0 + h10 * T_seg * v0 + h01 * pf + h11 * T_seg * vf
        
        # Velocity: dp/dt = (dp/ds) / T
        vel = (dh00_ds * p0 + dh10_ds * T_seg * v0 + dh01_ds * pf + dh11_ds * T_seg * vf) / (T_seg + self.config.eps)
        
        # Acceleration: d²p/dt² = (d²p/ds²) / T²
        acc = (d2h00_ds2 * p0 + d2h10_ds2 * T_seg * v0 + d2h01_ds2 * pf + d2h11_ds2 * T_seg * vf) / ((T_seg + self.config.eps) ** 2)
        
        # Jerk: d³p/dt³ = (d³p/ds³) / T³
        jerk = (d3h00_ds3 * p0 + d3h10_ds3 * T_seg * v0 + d3h01_ds3 * pf + d3h11_ds3 * T_seg * vf) / ((T_seg + self.config.eps) ** 3)
        
        return pos, vel, acc, jerk
    
    def evaluate(self, q_xy: torch.Tensor, q_theta: torch.Tensor, tau: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate trajectory at discrete time points.
        
        Args:
            q_xy: Control points for x, y [n_ctrl_pts, 2]
            q_theta: Control points for theta [n_ctrl_pts]
            tau: scalar τ (preferred, C++-style) or per-segment τ vector
        
        Returns:
            states: [n_samples, 3] with columns [x, y, theta]
            derivatives: [n_samples, 8] with [x_dot, y_dot, theta_dot, x_ddot, y_ddot, theta_ddot, x_dddot, y_dddot]
        """
        n_piece_xy = max(q_xy.shape[0] - 1, 1)
        n_piece_theta = max(q_theta.shape[0] - 1, 1)
        T_xy, T_theta = self._exp_map_time(tau, n_piece_xy, n_piece_theta)
        
        # Cumulative time for each sample
        total_time_xy = torch.sum(T_xy)
        total_time_theta = torch.sum(T_theta)
        if self.config.align_cpp_sampling:
            sample_count = max(2, n_piece_xy * (self.config.int_K + 1))
        else:
            sample_count = max(2, self.n_samples)
        sample_times = torch.linspace(
            0,
            max(total_time_xy, total_time_theta).item(),
            sample_count,
            device=self.device,
        )
        
        # =============================
        # Vectorized XY evaluation
        # =============================
        cumsum_xy = torch.cumsum(T_xy, dim=0)
        seg_idx_xy = torch.searchsorted(cumsum_xy, sample_times, right=False)
        seg_idx_xy = torch.clamp(seg_idx_xy, 0, T_xy.shape[0] - 1)
        start_xy = torch.zeros_like(sample_times)
        valid_xy = seg_idx_xy > 0
        start_xy[valid_xy] = cumsum_xy[seg_idx_xy[valid_xy] - 1]
        t_xy = sample_times - start_xy

        Tseg_xy = T_xy[seg_idx_xy]
        s_xy = torch.clamp(t_xy / (Tseg_xy + self.config.eps), 0, 1)
        s2 = s_xy * s_xy
        s3 = s2 * s_xy
        s4 = s3 * s_xy
        s5 = s4 * s_xy

        h00 = 1 - 10 * s3 + 15 * s4 - 6 * s5
        h10 = s_xy - 6 * s3 + 8 * s4 - 3 * s5
        h01 = 10 * s3 - 15 * s4 + 6 * s5
        h11 = -4 * s3 + 7 * s4 - 3 * s5
        dh00 = -30 * s2 + 60 * s3 - 30 * s4
        dh10 = 1 - 18 * s2 + 32 * s3 - 15 * s4
        dh01 = 30 * s2 - 60 * s3 + 30 * s4
        dh11 = -12 * s2 + 28 * s3 - 15 * s4
        d2h00 = -60 * s_xy + 180 * s2 - 120 * s3
        d2h10 = -36 * s_xy + 96 * s2 - 60 * s3
        d2h01 = 60 * s_xy - 180 * s2 + 120 * s3
        d2h11 = -24 * s_xy + 84 * s2 - 60 * s3
        d3h00 = -60 + 360 * s_xy - 360 * s2
        d3h10 = -36 + 192 * s_xy - 180 * s2
        d3h01 = 60 - 360 * s_xy + 360 * s2
        d3h11 = -24 + 168 * s_xy - 180 * s2

        # Estimate knot velocities from neighboring control points (Catmull-Rom-like)
        v_ctrl_xy = torch.zeros_like(q_xy)
        if q_xy.shape[0] >= 2:
            v_ctrl_xy[0] = (q_xy[1] - q_xy[0]) / (T_xy[0] + self.config.eps)
            v_ctrl_xy[-1] = (q_xy[-1] - q_xy[-2]) / (T_xy[-1] + self.config.eps)
        if q_xy.shape[0] > 2:
            denom_mid = (T_xy[:-1] + T_xy[1:]).unsqueeze(-1) + self.config.eps
            v_ctrl_xy[1:-1] = (q_xy[2:] - q_xy[:-2]) / denom_mid

        p0_xy = q_xy[seg_idx_xy]
        pf_xy = q_xy[seg_idx_xy + 1]
        v0_xy = v_ctrl_xy[seg_idx_xy]
        vf_xy = v_ctrl_xy[seg_idx_xy + 1]

        pos_xy = (
            h00.unsqueeze(-1) * p0_xy
            + h10.unsqueeze(-1) * Tseg_xy.unsqueeze(-1) * v0_xy
            + h01.unsqueeze(-1) * pf_xy
            + h11.unsqueeze(-1) * Tseg_xy.unsqueeze(-1) * vf_xy
        )
        vel_xy = (
            dh00.unsqueeze(-1) * p0_xy
            + dh10.unsqueeze(-1) * Tseg_xy.unsqueeze(-1) * v0_xy
            + dh01.unsqueeze(-1) * pf_xy
            + dh11.unsqueeze(-1) * Tseg_xy.unsqueeze(-1) * vf_xy
        ) / (Tseg_xy.unsqueeze(-1) + self.config.eps)
        acc_xy = (
            d2h00.unsqueeze(-1) * p0_xy
            + d2h10.unsqueeze(-1) * Tseg_xy.unsqueeze(-1) * v0_xy
            + d2h01.unsqueeze(-1) * pf_xy
            + d2h11.unsqueeze(-1) * Tseg_xy.unsqueeze(-1) * vf_xy
        ) / ((Tseg_xy.unsqueeze(-1) + self.config.eps) ** 2)
        jerk_xy = (
            d3h00.unsqueeze(-1) * p0_xy
            + d3h10.unsqueeze(-1) * Tseg_xy.unsqueeze(-1) * v0_xy
            + d3h01.unsqueeze(-1) * pf_xy
            + d3h11.unsqueeze(-1) * Tseg_xy.unsqueeze(-1) * vf_xy
        ) / ((Tseg_xy.unsqueeze(-1) + self.config.eps) ** 3)

        # =============================
        # Vectorized Theta evaluation
        # =============================
        cumsum_theta = torch.cumsum(T_theta, dim=0)
        seg_idx_theta = torch.searchsorted(cumsum_theta, sample_times, right=False)
        seg_idx_theta = torch.clamp(seg_idx_theta, 0, T_theta.shape[0] - 1)
        start_theta = torch.zeros_like(sample_times)
        valid_theta = seg_idx_theta > 0
        start_theta[valid_theta] = cumsum_theta[seg_idx_theta[valid_theta] - 1]
        t_theta = sample_times - start_theta

        Tseg_theta = T_theta[seg_idx_theta]
        s_th = torch.clamp(t_theta / (Tseg_theta + self.config.eps), 0, 1)
        s2t = s_th * s_th
        s3t = s2t * s_th
        s4t = s3t * s_th
        s5t = s4t * s_th

        h00t = 1 - 10 * s3t + 15 * s4t - 6 * s5t
        h10t = s_th - 6 * s3t + 8 * s4t - 3 * s5t
        h01t = 10 * s3t - 15 * s4t + 6 * s5t
        h11t = -4 * s3t + 7 * s4t - 3 * s5t
        dh00t = -30 * s2t + 60 * s3t - 30 * s4t
        dh10t = 1 - 18 * s2t + 32 * s3t - 15 * s4t
        dh01t = 30 * s2t - 60 * s3t + 30 * s4t
        dh11t = -12 * s2t + 28 * s3t - 15 * s4t
        d2h00t = -60 * s_th + 180 * s2t - 120 * s3t
        d2h10t = -36 * s_th + 96 * s2t - 60 * s3t
        d2h01t = 60 * s_th - 180 * s2t + 120 * s3t
        d2h11t = -24 * s_th + 84 * s2t - 60 * s3t
        d3h00t = -60 + 360 * s_th - 360 * s2t
        d3h10t = -36 + 192 * s_th - 180 * s2t
        d3h01t = 60 - 360 * s_th + 360 * s2t
        d3h11t = -24 + 168 * s_th - 180 * s2t

        v_ctrl_th = torch.zeros_like(q_theta)
        if q_theta.shape[0] >= 2:
            v_ctrl_th[0] = (q_theta[1] - q_theta[0]) / (T_theta[0] + self.config.eps)
            v_ctrl_th[-1] = (q_theta[-1] - q_theta[-2]) / (T_theta[-1] + self.config.eps)
        if q_theta.shape[0] > 2:
            denom_mid_th = (T_theta[:-1] + T_theta[1:]) + self.config.eps
            v_ctrl_th[1:-1] = (q_theta[2:] - q_theta[:-2]) / denom_mid_th

        p0_th = q_theta[seg_idx_theta]
        pf_th = q_theta[seg_idx_theta + 1]
        v0_th = v_ctrl_th[seg_idx_theta]
        vf_th = v_ctrl_th[seg_idx_theta + 1]

        pos_theta = h00t * p0_th + h10t * Tseg_theta * v0_th + h01t * pf_th + h11t * Tseg_theta * vf_th
        vel_theta = (dh00t * p0_th + dh10t * Tseg_theta * v0_th + dh01t * pf_th + dh11t * Tseg_theta * vf_th) / (Tseg_theta + self.config.eps)
        acc_theta = (d2h00t * p0_th + d2h10t * Tseg_theta * v0_th + d2h01t * pf_th + d2h11t * Tseg_theta * vf_th) / ((Tseg_theta + self.config.eps) ** 2)
        jerk_theta = (d3h00t * p0_th + d3h10t * Tseg_theta * v0_th + d3h01t * pf_th + d3h11t * Tseg_theta * vf_th) / ((Tseg_theta + self.config.eps) ** 3)

        # states: [N, 3], derivatives: [N, 9]
        states = torch.stack([pos_xy[:, 0], pos_xy[:, 1], pos_theta], dim=-1)
        derivatives = torch.stack(
            [
                vel_xy[:, 0], vel_xy[:, 1], vel_theta,
                acc_xy[:, 0], acc_xy[:, 1], acc_theta,
                jerk_xy[:, 0], jerk_xy[:, 1], jerk_theta,
            ],
            dim=-1,
        )
        
        return states, derivatives, T_xy, T_theta


# ==========================================
# 3. Loss & Constraint Computation
# ==========================================

class TrajectoryEvaluator:
    """
    Computes objective function (cost) and constraints for the optimization.
    Corresponds to Equations 28-37 in the paper.
    """
    
    def __init__(self, config: OptimizationConfig, mapper: TerrainPoseMapper, device: torch.device = torch.device("cpu"),
                 yaw_stability_map: torch.Tensor = None):
        self.config = config
        self.mapper = mapper
        self.device = device
        self.yaw_stability_map = yaw_stability_map  # [H, W, D] binary or soft stability map
    
    def _query_stability_from_yaw_stability(self, x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Query stability values from yaw_stability binary map using trilinear interpolation.
        Returns soft safety values (0=unsafe, 1=safe) for constraint checking.
        
        Args:
            x, y, theta: Position and heading tensors [n_samples]
        
        Returns:
            safety_values: [n_samples] soft safety values in [0, 1]
        """
        if self.yaw_stability_map is None:
            # No safety map provided, assume all points are safe
            return torch.ones_like(x)
        
        n_samples = x.shape[0]
        device = x.device
        
        # Get grid dimensions and parameters from mapper
        H, W, D = self.yaw_stability_map.shape
        x_min, x_max = self.mapper.x_min, self.mapper.x_max
        y_min, y_max = self.mapper.y_min, self.mapper.y_max
        theta_min, theta_max = self.mapper.theta_min, self.mapper.theta_max
        
        # Normalize coordinates to [0, 1]
        x_norm = torch.clamp((x - x_min) / (x_max - x_min), 0, 1)
        y_norm = torch.clamp((y - y_min) / (y_max - y_min), 0, 1)
        theta_norm = torch.clamp(((theta - theta_min) % (2 * np.pi)) / (theta_max - theta_min), 0, 1)
        
        # Convert to grid indices
        x_idx = x_norm * (W - 1)
        y_idx = y_norm * (H - 1)
        theta_idx = theta_norm * (D - 1)
        
        # Trilinear interpolation to get soft safety values
        safety_values = torch.zeros(n_samples, device=device, dtype=torch.float32)
        
        for i in range(n_samples):
            x_i = x_idx[i]
            y_i = y_idx[i]
            t_i = theta_idx[i]
            
            # Get integer and fractional parts
            x_i0 = int(torch.clamp(torch.floor(x_i), 0, W - 2).item())
            y_i0 = int(torch.clamp(torch.floor(y_i), 0, H - 2).item())
            t_i0 = int(torch.clamp(torch.floor(t_i), 0, D - 2).item())
            
            dx = torch.clamp(x_i - torch.floor(x_i), 0, 1)
            dy = torch.clamp(y_i - torch.floor(y_i), 0, 1)
            dt = torch.clamp(t_i - torch.floor(t_i), 0, 1)
            
            # Sample 8 corners
            corners = []
            for ti in [t_i0, t_i0 + 1]:
                for yi in [y_i0, y_i0 + 1]:
                    for xi in [x_i0, x_i0 + 1]:
                        if 0 <= ti < D and 0 <= yi < H and 0 <= xi < W:
                            val = self.yaw_stability_map[yi, xi, ti].float()
                        else:
                            val = torch.tensor(0.0, device=device)  # Out of bounds = unsafe
                        corners.append(val)
            
            if len(corners) == 8:
                # Trilinear interpolation
                v00 = corners[0] * (1 - dx) + corners[1] * dx
                v10 = corners[2] * (1 - dx) + corners[3] * dx
                v01 = corners[4] * (1 - dx) + corners[5] * dx
                v11 = corners[6] * (1 - dx) + corners[7] * dx
                
                v0 = v00 * (1 - dy) + v10 * dy
                v1 = v01 * (1 - dy) + v11 * dy
                
                safety_values[i] = v0 * (1 - dt) + v1 * dt
            else:
                safety_values[i] = torch.tensor(0.0, device=device)
        
        return safety_values
    
    def compute_loss_and_constraints(self,
                                   states: torch.Tensor,
                                   derivatives: torch.Tensor,
                                   T_xy: torch.Tensor,
                                   T_theta: torch.Tensor,
                                   lambda_mults: Dict,
                                   mu_penalty: float,
                                   scale_fx: float = 1.0,
                                   return_terms: bool = False):
        """
        Compute total loss (objective + ALM penalty) and constraint violations.
        Aligned with C++ alm_traj_opt.cpp calConstrainCostGrad implementation.
        
        Args:
            states: [n_samples, 3] trajectory states [x, y, theta]
            derivatives: [n_samples, 8] trajectory derivatives
            T_xy, T_theta: Time segments
            lambda_mults: Lagrange multipliers
            mu_penalty: Penalty weight
            scale_fx: Gradient scaling factor
        
        Returns:
            total_loss: Scalar loss for optimization
            constraints: Dictionary of constraint violations
            non_holonomic: Non-holonomic constraint violation
        """
        n_samples = states.shape[0]
        
        x = states[:, 0]
        y = states[:, 1]
        theta = states[:, 2]
        
        x_dot = derivatives[:, 0]
        y_dot = derivatives[:, 1]
        theta_dot = derivatives[:, 2]
        
        x_ddot = derivatives[:, 3]
        y_ddot = derivatives[:, 4]
        theta_ddot = derivatives[:, 5]
        
        x_dddot = derivatives[:, 6]
        y_dddot = derivatives[:, 7]
        
        # --- A. Objective Function (Eq. 28) ---
        
        # 1. Smoothness cost (Jerk): integral of jerk^2
        jerk_xy = torch.sqrt(x_dddot**2 + y_dddot**2 + self.config.eps)
        jerk_cost = self.config.rho_jerk * torch.sum(jerk_xy) * scale_fx
        
        # 2. Safety cost: use cost_map (sigma from mapper) directly
        zb, sigma = self.mapper.query_zb_sigma(x, y, theta)
        sigma = torch.nan_to_num(sigma, nan=0.0, posinf=self.config.constraint_clip, neginf=0.0)
        safety_cost = self.config.rho_terrain * torch.sum(torch.relu(sigma)) * scale_fx
        
        # 3. Time cost
        time_cost = self.config.rho_time * (torch.sum(T_xy) + torch.sum(T_theta)) * scale_fx

        # 4. Heading consistency cost: theta should align with planar velocity direction
        v_planar_obj = torch.sqrt(x_dot**2 + y_dot**2 + self.config.eps)
        v_hat_x = x_dot / (v_planar_obj + self.config.eps)
        v_hat_y = y_dot / (v_planar_obj + self.config.eps)
        heading_x = torch.cos(theta)
        heading_y = torch.sin(theta)
        cos_align = torch.clamp(heading_x * v_hat_x + heading_y * v_hat_y, -1.0, 1.0)
        heading_misalign = 1.0 - cos_align
        heading_cost = self.config.rho_heading * torch.sum(heading_misalign) * scale_fx

        # 5. Heading smoothness to suppress local heading flips/oscillation
        theta_smooth_cost = self.config.rho_theta_smooth * torch.sum(
            theta_dot**2 + 0.1 * theta_ddot**2
        ) * scale_fx
        
        objective_cost = jerk_cost + safety_cost + time_cost + heading_cost + theta_smooth_cost
        
        # --- B. Physical & Dynamic Constraints (Eq. 29, 32-37) ---
        
        # Terrain normal: zb = [a, b, c]
        # cos_xi relates to pitch angle
        cos_xi = torch.clamp(zb[:, 2], 0, 1)
        
        # Planar velocity
        v_planar = torch.sqrt(x_dot**2 + y_dot**2 + self.config.eps)
        
        # Compute terrain-coupled factors (C++ style)
        # Approximate phix/phiy from local terrain representation used in mapper
        phix = torch.atan(zb[:, 0])
        phiy = torch.atan(zb[:, 1])
        inv_cos_vphix = 1.0 / (torch.clamp(torch.abs(torch.cos(phix)), self.config.eps, 1.0))
        inv_cos_vphiy = 1.0 / (torch.clamp(torch.abs(torch.cos(phiy)), self.config.eps, 1.0))
        sin_phix = torch.sin(phix)
        sin_phiy = torch.sin(phiy)
        
        # 3D velocities
        vx_3d = v_planar * inv_cos_vphix
        
        # Longitudinal / lateral acceleration in body frame + explicit gravity terms
        lon_acc = x_ddot * torch.cos(theta) + y_ddot * torch.sin(theta)
        lat_acc = -x_ddot * torch.sin(theta) + y_ddot * torch.cos(theta)
        ax_3d = lon_acc * inv_cos_vphix + self.config.gravity * sin_phix
        ay_3d = lat_acc * inv_cos_vphiy + self.config.gravity * sin_phiy
        
        # Yaw rate and curvature
        inv_cos_xi = 1.0 / torch.clamp(cos_xi, self.config.eps, 1.0)
        omega_z = theta_dot * inv_cos_xi
        # C++-style curvature with terrain-coupled yaw rate and vx
        kappa = omega_z / (torch.sqrt(vx_3d**2 + self.config.delta_v) + self.config.eps)
        # Prefer C++-aligned direct parameter; fallback to wheelbase/steering conversion
        kappa_max = self.config.max_kap if self.config.max_kap > 0 else (np.tan(self.config.delta_max) / self.config.L_w)
        min_cxi = self.config.min_cxi if self.config.min_cxi > 0 else self.config.c_min
        
        # Collect constraint violations (c(x) <= 0)
        constraints = {}
        constraints['v_max'] = vx_3d**2 - self.config.v_max**2
        constraints['a_lon'] = ax_3d**2 - self.config.a_lon_max**2
        constraints['a_lat'] = ay_3d**2 - self.config.a_lat_max**2
        constraints['kappa'] = torch.abs(kappa) - kappa_max
        constraints['attitude'] = min_cxi - cos_xi  # Rollover constraint
        
        # C++一致：surface variation constraint，直接使用 sigma - max_sig
        constraints['sigma'] = sigma - self.config.sigma_max

        # Feasibility time lower-bound: T_total should be long enough for v_max
        dxy = states[1:, :2] - states[:-1, :2]
        path_len = torch.sum(torch.sqrt(torch.sum(dxy * dxy, dim=1) + self.config.eps))
        t_total = torch.sum(T_xy)
        t_min_feasible = path_len / (0.9 * self.config.v_max + self.config.eps)
        constraints['time_min'] = t_min_feasible - t_total

        # Numerical guard on constraints (prevent dual update explosion)
        for name in list(constraints.keys()):
            constraints[name] = torch.nan_to_num(
                constraints[name],
                nan=self.config.constraint_clip,
                posinf=self.config.constraint_clip,
                neginf=-self.config.constraint_clip,
            )
            constraints[name] = torch.clamp(
                constraints[name],
                min=-self.config.constraint_clip,
                max=self.config.constraint_clip,
            )
        
        # --- C. Non-holonomic Constraint (lateral slip = 0) ---
        # Constraint: x_dot * sin(theta) - y_dot * cos(theta) = 0
        non_holonomic = x_dot * torch.sin(theta) - y_dot * torch.cos(theta)

        # Soft feasibility penalties (help inner solver satisfy dynamics earlier)
        dyn_soft_penalty = 0.0
        for c_vals in constraints.values():
            dyn_soft_penalty += torch.sum(torch.relu(c_vals) ** 2)
        dyn_soft_penalty = self.config.rho_dyn_soft * dyn_soft_penalty * scale_fx

        nonholo_soft_penalty = self.config.rho_nonholo_soft * torch.sum(non_holonomic ** 2) * scale_fx
        
        # --- D. PHR-ALM Penalty Computation ---
        alm_penalty = 0.0
        
        # Inequality constraints penalty
        for name, c_vals in constraints.items():
            if name not in lambda_mults['ineq']:
                lambda_mults['ineq'][name] = torch.zeros_like(c_vals, device=self.mapper.device)
            
            lam = lambda_mults['ineq'][name]
            if lam.shape != c_vals.shape:
                lam = torch.zeros_like(c_vals, device=self.mapper.device)
                lambda_mults['ineq'][name] = lam
            
            # PHR penalty for inequality.
            # NOTE: 经典形式里有常数项 -lambda^2/(2*rho)，该项与优化变量无关，
            # 仅会把 loss 整体平移成很大的负数，影响日志可读性但不影响梯度。
            # 这里去掉常数项，保持同梯度、更稳定的监控数值。
            active_c = torch.relu(lam + mu_penalty * c_vals)
            active_c = torch.clamp(active_c, 0.0, self.config.constraint_clip)
            phr_pen = (active_c**2) / (2.0 * mu_penalty + self.config.eps)
            alm_penalty += torch.sum(phr_pen)
        
        # Equality constraint penalty
        if 'non_holo' not in lambda_mults['eq']:
            lambda_mults['eq']['non_holo'] = torch.zeros_like(non_holonomic, device=self.mapper.device)
        
        lam_eq = lambda_mults['eq']['non_holo']
        if lam_eq.shape != non_holonomic.shape:
            lam_eq = torch.zeros_like(non_holonomic, device=self.mapper.device)
            lambda_mults['eq']['non_holo'] = lam_eq
        
        # Standard ALM for equality: lambda*h + (mu/2)*h^2
        non_holonomic_safe = torch.nan_to_num(
            non_holonomic,
            nan=self.config.constraint_clip,
            posinf=self.config.constraint_clip,
            neginf=-self.config.constraint_clip,
        )
        non_holonomic_safe = torch.clamp(
            non_holonomic_safe,
            min=-self.config.constraint_clip,
            max=self.config.constraint_clip,
        )
        alm_penalty += torch.sum(lam_eq * non_holonomic_safe + (mu_penalty / 2.0) * non_holonomic_safe**2)
        
        # IMPORTANT:
        # Do NOT hard-clamp loss to a fixed constant (e.g. 1e6), otherwise
        # optimizer logs become constant and gradients may vanish at saturation.
        # We only guard NaN/Inf here.
        objective_cost = torch.nan_to_num(
            objective_cost,
            nan=self.config.safe_max_value,
            posinf=self.config.safe_max_value,
            neginf=-self.config.safe_max_value,
        )
        alm_penalty = torch.nan_to_num(
            alm_penalty,
            nan=self.config.safe_max_value,
            posinf=self.config.safe_max_value,
            neginf=-self.config.safe_max_value,
        )

        total_loss = objective_cost + alm_penalty + dyn_soft_penalty + nonholo_soft_penalty
        total_loss = torch.nan_to_num(
            total_loss,
            nan=self.config.safe_max_value,
            posinf=self.config.safe_max_value,
            neginf=-self.config.safe_max_value,
        )
        
        if return_terms:
            terms = {
                'objective': objective_cost.detach(),
                'alm_penalty': alm_penalty.detach(),
                'dyn_soft_penalty': torch.as_tensor(dyn_soft_penalty).detach(),
                'nonholo_soft_penalty': torch.as_tensor(nonholo_soft_penalty).detach(),
                'jerk_cost': jerk_cost.detach(),
                'safety_cost': safety_cost.detach(),
                'time_cost': time_cost.detach(),
                'heading_cost': heading_cost.detach(),
                'theta_smooth_cost': theta_smooth_cost.detach(),
            }
            return total_loss, constraints, non_holonomic, terms
        return total_loss, constraints, non_holonomic


# ==========================================
# 4. Main Trajectory Optimizer
# ==========================================

class TrajectoryOptimizer:
    """
    Main optimization loop: Outer ALM + Inner L-BFGS.
    
    Implements the complete algorithm from the paper:
    1. MINCO trajectory parameterization
    2. Discrete time sampling
    3. PHR-ALM for constraint handling
    4. L-BFGS for inner unconstrained optimization
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None, device: Optional[torch.device] = None):
        self.config = config or OptimizationConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        
    def _get_max_violation(self, constraints: Dict[str, torch.Tensor], non_holonomic: torch.Tensor) -> float:
        """
        Compute maximum constraint violation across all constraints.
        """
        max_ineq_violation = 0.0
        for c_vals in constraints.values():
            # Constraint is c(x) <= 0, violation is max(0, c(x))
            violation = torch.max(torch.relu(c_vals)).item()
            max_ineq_violation = max(max_ineq_violation, violation)
        
        max_eq_violation = torch.max(torch.abs(non_holonomic)).item()
        
        return max(max_ineq_violation, max_eq_violation)

    def _judge_convergence(
        self,
        eq_residual: torch.Tensor,
        ineq_residual: Dict[str, torch.Tensor],
        mu_ineq: Dict[str, torch.Tensor],
        rho: float,
    ) -> float:
        """
        C++ equivalent of:
        max( ||hx||_inf, || gx.cwiseMax(-mu/rho) ||_inf )
        """
        h_inf = torch.max(torch.abs(eq_residual)).item()
        g_terms = []
        for name, g in ineq_residual.items():
            mu = mu_ineq.get(name, torch.zeros_like(g, device=self.device))
            g_terms.append(torch.max(torch.maximum(g, -mu / max(rho, self.config.eps))).item())
        g_inf = max(g_terms) if g_terms else 0.0
        return max(h_inf, g_inf)

    def _apply_dynamic_time_rescale(
        self,
        tau: torch.Tensor,
        current_ineq: Dict[str, torch.Tensor],
    ) -> float:
        """
        If dynamic constraints are violated, increase total time to recover feasibility.
        Scaling time does not change path geometry, and reduces |v| by 1/s, |a| by 1/s^2.
        Returns applied scale factor.
        """
        c_v = torch.max(torch.relu(current_ineq.get('v_max', torch.zeros(1, device=self.device)))).item()
        c_ax = torch.max(torch.relu(current_ineq.get('a_lon', torch.zeros(1, device=self.device)))).item()
        c_ay = torch.max(torch.relu(current_ineq.get('a_lat', torch.zeros(1, device=self.device)))).item()

        scale_v = np.sqrt((c_v + self.config.v_max**2) / (self.config.v_max**2 + self.config.eps))
        scale_ax = np.sqrt((c_ax + self.config.a_lon_max**2) / (self.config.a_lon_max**2 + self.config.eps))
        scale_ay = np.sqrt((c_ay + self.config.a_lat_max**2) / (self.config.a_lat_max**2 + self.config.eps))
        scale = max(1.0, scale_v, scale_ax, scale_ay)

        if scale > 1.02:
            with torch.no_grad():
                tau_flat = tau.data.reshape(-1)
                if tau_flat.numel() == 1:
                    t_old = MincoTrajectory.exp_c2(tau_flat[0])
                    t_new = t_old * float(scale) * 1.05
                    tau_flat[0] = MincoTrajectory.log_c2(t_new)
                else:
                    T_old = MincoTrajectory.exp_c2(tau_flat)
                    T_new = T_old * float(scale) * 1.05
                    tau_flat.copy_(MincoTrajectory.log_c2(T_new))
                tau.data.clamp_(-self.config.tau_clip, self.config.tau_clip)
        return float(scale)
    
    def optimize(self,
                init_q_xy: np.ndarray,
                init_q_theta: np.ndarray,
                init_tau: np.ndarray,
                grid_map: Dict[str, torch.Tensor],
                yaw_stability_map: torch.Tensor = None,
                fix_endpoints: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Optimize trajectory using nested ALM + L-BFGS.
        
        Args:
            init_q_xy: Initial control points for x, y [n_ctrl_pts, 2]
            init_q_theta: Initial heading control points [n_ctrl_pts]
            init_tau: Initial log-time τ (scalar preferred; vector also supported)
            grid_map: Terrain grid map
            yaw_stability_map: Optional [H, W, D] binary safety map for stability constraints
            fix_endpoints: If True, fix first and last control points (start/goal)
        
        Returns:
            opt_q_xy: Optimized x, y control points
            opt_q_theta: Optimized theta control points
            opt_tau: Optimized log-time durations
        """
        # Store fixed endpoints if requested
        self.fix_endpoints = fix_endpoints
        if fix_endpoints:
            self.fixed_q_xy_start = torch.tensor(init_q_xy[0], dtype=torch.float32, device=self.device)
            self.fixed_q_xy_end = torch.tensor(init_q_xy[-1], dtype=torch.float32, device=self.device)
            self.fixed_q_theta_start = torch.tensor(init_q_theta[0], dtype=torch.float32, device=self.device)
            self.fixed_q_theta_end = torch.tensor(init_q_theta[-1], dtype=torch.float32, device=self.device)

        # Reference trajectory controls (used as regularization anchors for interior points)
        self.ref_q_xy = torch.tensor(init_q_xy, dtype=torch.float32, device=self.device)
        self.ref_q_theta = torch.tensor(init_q_theta, dtype=torch.float32, device=self.device)
        
        # Initialize tensor variables with gradients enabled
        q_xy = torch.tensor(init_q_xy, dtype=torch.float32, device=self.device, requires_grad=True)
        q_theta = torch.tensor(init_q_theta, dtype=torch.float32, device=self.device, requires_grad=True)
        tau = torch.tensor(init_tau, dtype=torch.float32, device=self.device, requires_grad=True)
        
        # Initialize mapper and trajectory evaluator
        mapper = TerrainPoseMapper(grid_map, device=self.device)
        trajectory = MincoTrajectory(self.config, device=self.device)
        evaluator = TrajectoryEvaluator(self.config, mapper, device=self.device, 
                                       yaw_stability_map=yaw_stability_map)
        
        # Initialize ALM multipliers
        lambda_mults = {
            'ineq': {},  # Will be populated during first evaluation
            'eq': {'non_holo': torch.zeros(self.config.n_samples, device=self.device)}
        }
        
        rho = float(self.config.rho)
        scale_fx = 1.0
        
        self.logger.info(f"Starting trajectory optimization on device: {self.device}")
        self.logger.info(f"ALM max iterations: {self.config.max_iter}")
        if fix_endpoints:
            self.logger.info(f"Endpoint constraint: FIXED (start and goal are fixed)")
        
        # Outer loop: PHR-ALM iterations
        for alm_iter in range(self.config.max_iter):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"ALM Iteration {alm_iter + 1}/{self.config.max_iter}")
            self.logger.info(f"Penalty weight rho = {rho:.6f}")
            self.logger.info(f"{'='*60}")
            
            # Inner loop: L-BFGS optimization
            runtime_inner_iter = min(self.config.inner_max_iter, self.config.inner_max_iter_runtime)
            self.logger.info(
                f"Inner L-BFGS max_iter(runtime): {runtime_inner_iter} "
                f"(configured={self.config.inner_max_iter})"
            )
            runtime_line_search = self.config.lbfgs_line_search if self.config.lbfgs_use_line_search_runtime else None
            self.logger.info(f"Inner L-BFGS line_search(runtime): {runtime_line_search}")
            optimizer = optim.LBFGS(
                [q_xy, q_theta, tau],
                max_iter=runtime_inner_iter,
                max_eval=runtime_inner_iter,
                history_size=self.config.mem_size,
                line_search_fn=runtime_line_search
            )
            
            closure_loss_history = []
            closure_call_count = 0
            
            def closure():
                """Closure function for L-BFGS (compute loss and gradients)"""
                nonlocal closure_call_count
                closure_call_count += 1
                optimizer.zero_grad()
                
                # Sanitize optimization variables to prevent NaN/Inf explosion
                q_xy_eval = torch.nan_to_num(q_xy, nan=0.0, posinf=100.0, neginf=-100.0)
                q_theta_eval = torch.nan_to_num(q_theta, nan=0.0, posinf=np.pi, neginf=-np.pi)
                tau_eval = torch.nan_to_num(tau, nan=0.0, posinf=self.config.tau_clip, neginf=-self.config.tau_clip)
                tau_eval = torch.clamp(tau_eval, -self.config.tau_clip, self.config.tau_clip)

                # Hard endpoint constraints: always evaluate with fixed start/goal
                if self.fix_endpoints:
                    q_xy_eval = q_xy_eval.clone()
                    q_theta_eval = q_theta_eval.clone()
                    q_xy_eval[0] = self.fixed_q_xy_start
                    q_xy_eval[-1] = self.fixed_q_xy_end
                    q_theta_eval[0] = self.fixed_q_theta_start
                    q_theta_eval[-1] = self.fixed_q_theta_end
                
                # Evaluate trajectory
                states, derivatives, T_xy, T_theta = trajectory.evaluate(q_xy_eval, q_theta_eval, tau_eval)
                
                # Compute loss and constraints
                loss, _, _ = evaluator.compute_loss_and_constraints(
                    states, derivatives, T_xy, T_theta, lambda_mults, rho, scale_fx
                )

                # Reference-shape regularization on interior control points
                if q_xy_eval.shape[0] > 2:
                    ref_xy_loss = torch.sum((q_xy_eval[1:-1] - self.ref_q_xy[1:-1]) ** 2)
                    ref_theta_loss = torch.sum((q_theta_eval[1:-1] - self.ref_q_theta[1:-1]) ** 2)
                    loss = loss + self.config.rho_ref_xy * ref_xy_loss + self.config.rho_ref_theta * ref_theta_loss
                
                # Backward pass (automatic differentiation)
                loss.backward()
                
                closure_loss_history.append(loss.item())
                if self.config.closure_log_interval > 0 and closure_call_count % self.config.closure_log_interval == 0:
                    self.logger.info(
                        f"  [inner] closure {closure_call_count}, loss={loss.item():.6e}"
                    )
                return loss
            
            # Run L-BFGS optimization step
            optimizer.step(closure)

            # Project variables back to finite, bounded range (prevents NaN propagation across ALM iterations)
            with torch.no_grad():
                q_xy.data = torch.nan_to_num(q_xy.data, nan=0.0, posinf=100.0, neginf=-100.0)
                q_theta.data = torch.nan_to_num(q_theta.data, nan=0.0, posinf=np.pi, neginf=-np.pi)
                tau.data = torch.nan_to_num(
                    tau.data,
                    nan=0.0,
                    posinf=self.config.tau_clip,
                    neginf=-self.config.tau_clip,
                )
                tau.data.clamp_(-self.config.tau_clip, self.config.tau_clip)
                if self.fix_endpoints:
                    q_xy.data[0] = self.fixed_q_xy_start
                    q_xy.data[-1] = self.fixed_q_xy_end
                    q_theta.data[0] = self.fixed_q_theta_start
                    q_theta.data[-1] = self.fixed_q_theta_end
            
            # Evaluate final state after L-BFGS iteration
            with torch.no_grad():
                q_xy_eval = torch.nan_to_num(q_xy, nan=0.0, posinf=100.0, neginf=-100.0)
                q_theta_eval = torch.nan_to_num(q_theta, nan=0.0, posinf=np.pi, neginf=-np.pi)
                tau_eval = torch.nan_to_num(tau, nan=0.0, posinf=self.config.tau_clip, neginf=-self.config.tau_clip)
                tau_eval = torch.clamp(tau_eval, -self.config.tau_clip, self.config.tau_clip)
                if self.fix_endpoints:
                    q_xy_eval = q_xy_eval.clone()
                    q_theta_eval = q_theta_eval.clone()
                    q_xy_eval[0] = self.fixed_q_xy_start
                    q_xy_eval[-1] = self.fixed_q_xy_end
                    q_theta_eval[0] = self.fixed_q_theta_start
                    q_theta_eval[-1] = self.fixed_q_theta_end
                states, derivatives, T_xy, T_theta = trajectory.evaluate(q_xy_eval, q_theta_eval, tau_eval)
                _, current_ineq, current_eq, terms = evaluator.compute_loss_and_constraints(
                    states, derivatives, T_xy, T_theta, lambda_mults, rho, scale_fx, return_terms=True
                )
                
                max_violation = self._get_max_violation(current_ineq, current_eq)
                conv_res = self._judge_convergence(current_eq, current_ineq, lambda_mults['ineq'], rho)
                final_loss = closure_loss_history[-1] if closure_loss_history else 0
                
                self.logger.info(f"L-BFGS final loss: {final_loss:.6f}")
                self.logger.info(f"Max constraint violation: {max_violation:.6e}")
                self.logger.info(f"Convergence residual: {conv_res:.6e}")
                self.logger.info(
                    "  terms: obj=%.3e, alm=%.3e, dyn_soft=%.3e, nonholo_soft=%.3e, "
                    "jerk=%.3e, safety=%.3e, time=%.3e, heading=%.3e, theta_smooth=%.3e",
                    terms['objective'].item(),
                    terms['alm_penalty'].item(),
                    terms['dyn_soft_penalty'].item(),
                    terms['nonholo_soft_penalty'].item(),
                    terms['jerk_cost'].item(),
                    terms['safety_cost'].item(),
                    terms['time_cost'].item(),
                    terms['heading_cost'].item(),
                    terms['theta_smooth_cost'].item(),
                )
                for cname, cvals in current_ineq.items():
                    cmax = torch.max(torch.relu(cvals)).item()
                    self.logger.info(f"  ineq[{cname}] max+: {cmax:.6e}")
                self.logger.info(f"  eq[non_holo] max|.|: {torch.max(torch.abs(current_eq)).item():.6e}")

                # Optional feasibility projection in time domain for dynamics
                if self.config.dynamic_rescale_each_iter:
                    dyn_scale = self._apply_dynamic_time_rescale(tau, current_ineq)
                    if dyn_scale > 1.02:
                        self.logger.info(f"  Applied dynamic time rescale: x{dyn_scale:.3f}")
            
            # Check convergence
            if conv_res < self.config.epsilon_con:
                self.logger.info(f"\n✓ Optimization converged at ALM iteration {alm_iter + 1}")
                self.logger.info(f"Residual: {conv_res:.6e} < {self.config.epsilon_con:.6e}")
                break
            
            # Update Lagrange multipliers and penalty weight
            with torch.no_grad():
                # Inequality constraints: lambda_new = max(0, lambda + mu * c)
                for name, c_vals in current_ineq.items():
                    if name not in lambda_mults['ineq']:
                        lambda_mults['ineq'][name] = torch.zeros_like(c_vals, device=self.device)
                    c_vals_safe = torch.nan_to_num(
                        c_vals,
                        nan=self.config.constraint_clip,
                        posinf=self.config.constraint_clip,
                        neginf=-self.config.constraint_clip,
                    )
                    c_vals_safe = torch.clamp(
                        c_vals_safe,
                        min=-self.config.constraint_clip,
                        max=self.config.constraint_clip,
                    )
                    
                    lambda_mults['ineq'][name] = torch.relu(
                        lambda_mults['ineq'][name] + rho * c_vals_safe
                    )
                    lambda_mults['ineq'][name] = torch.clamp(
                        torch.nan_to_num(
                            lambda_mults['ineq'][name],
                            nan=0.0,
                            posinf=self.config.dual_clip,
                            neginf=0.0,
                        ),
                        0.0,
                        self.config.dual_clip,
                    )
                
                # Equality constraints: lambda_new = lambda + mu * h
                current_eq_safe = torch.nan_to_num(
                    current_eq,
                    nan=self.config.constraint_clip,
                    posinf=self.config.constraint_clip,
                    neginf=-self.config.constraint_clip,
                )
                current_eq_safe = torch.clamp(
                    current_eq_safe,
                    min=-self.config.constraint_clip,
                    max=self.config.constraint_clip,
                )
                lambda_mults['eq']['non_holo'] += rho * current_eq_safe
                lambda_mults['eq']['non_holo'] = torch.clamp(
                    torch.nan_to_num(
                        lambda_mults['eq']['non_holo'],
                        nan=0.0,
                        posinf=self.config.dual_clip,
                        neginf=-self.config.dual_clip,
                    ),
                    -self.config.dual_clip,
                    self.config.dual_clip,
                )
            
            # C++ style rho update: rho = min((1 + gamma) * rho, beta)
            rho = min((1.0 + self.config.gamma) * rho, self.config.beta)
            
            self.logger.info(f"Updated {len(current_ineq)} inequality and 1 equality multiplier(s)")
        
        with torch.no_grad():
            q_xy = torch.nan_to_num(q_xy, nan=0.0, posinf=100.0, neginf=-100.0)
            q_theta = torch.nan_to_num(q_theta, nan=0.0, posinf=np.pi, neginf=-np.pi)
            tau = torch.nan_to_num(tau, nan=0.0, posinf=self.config.tau_clip, neginf=-self.config.tau_clip)
            tau = torch.clamp(tau, -self.config.tau_clip, self.config.tau_clip)
            if self.fix_endpoints:
                q_xy[0] = self.fixed_q_xy_start
                q_xy[-1] = self.fixed_q_xy_end
                q_theta[0] = self.fixed_q_theta_start
                q_theta[-1] = self.fixed_q_theta_end

            # Final dynamics-feasibility time scaling (geometry unchanged)
            for _ in range(5):
                states_f, derivatives_f, _, _ = trajectory.evaluate(q_xy, q_theta, tau)
                x_dot = derivatives_f[:, 0]
                y_dot = derivatives_f[:, 1]
                x_ddot = derivatives_f[:, 3]
                y_ddot = derivatives_f[:, 4]

                theta_f = states_f[:, 2]
                zb_f, _ = mapper.query_zb_sigma(states_f[:, 0], states_f[:, 1], theta_f)
                v_planar_f = torch.sqrt(x_dot**2 + y_dot**2 + self.config.eps)
                inv_cos_vphix_f = 1.0 / (
                    torch.clamp(torch.abs(torch.cos(torch.atan(zb_f[:, 0]))), self.config.eps, 1.0)
                )
                inv_cos_vphiy_f = 1.0 / (
                    torch.clamp(torch.abs(torch.cos(torch.atan(zb_f[:, 1]))), self.config.eps, 1.0)
                )

                vx_3d_f = v_planar_f * inv_cos_vphix_f
                a_total_f = torch.sqrt(x_ddot**2 + y_ddot**2 + self.config.eps)
                ax_3d_f = a_total_f * inv_cos_vphix_f
                ay_3d_f = a_total_f * inv_cos_vphiy_f

                max_v = torch.max(vx_3d_f).item()
                max_ax = torch.max(ax_3d_f).item()
                max_ay = torch.max(ay_3d_f).item()

                s_v = max_v / max(self.config.v_max, self.config.eps)
                s_ax = np.sqrt(max_ax / max(self.config.a_lon_max, self.config.eps))
                s_ay = np.sqrt(max_ay / max(self.config.a_lat_max, self.config.eps))
                s_need = max(1.0, s_v, s_ax, s_ay)

                if s_need <= 1.01:
                    break

                tau_flat = tau.reshape(-1)
                if tau_flat.numel() == 1:
                    t_old = MincoTrajectory.exp_c2(tau_flat[0])
                    t_new = t_old * float(s_need) * 1.02
                    tau_flat[0] = MincoTrajectory.log_c2(t_new)
                else:
                    T_old = MincoTrajectory.exp_c2(tau_flat)
                    T_new = T_old * float(s_need) * 1.02
                    tau_flat.copy_(MincoTrajectory.log_c2(T_new))
                tau.clamp_(-self.config.tau_clip, self.config.tau_clip)

        return q_xy.detach(), q_theta.detach(), tau.detach()


# ==========================================
# Example Usage & Testing
# ==========================================

def create_dummy_grid_map(nx=20, ny=20, ntheta=8, device=torch.device("cpu")) -> Dict[str, torch.Tensor]:
    """Create a dummy terrain grid map for testing."""
    # Create smooth terrain data
    x_grid = torch.linspace(0, 10, nx, device=device)
    y_grid = torch.linspace(0, 10, ny, device=device)
    theta_grid = torch.linspace(0, 2 * np.pi, ntheta, device=device)
    
    # Create smooth height field
    z = torch.zeros(nx, ny, ntheta, device=device)
    for i in range(nx):
        for j in range(ny):
            # Simple smooth terrain: height based on position
            h = 0.5 * torch.sin(x_grid[i] / 5.0) * torch.cos(y_grid[j] / 5.0)
            z[i, j, :] = h
    
    # Create smooth normal vectors (zb)
    zb = torch.zeros(nx, ny, ntheta, 3, device=device)
    zb[..., 0] = 0.1  # a component
    zb[..., 1] = 0.1  # b component
    zb[..., 2] = 0.99  # c component (mostly vertical, close to 1)
    zb = zb / (torch.norm(zb, dim=-1, keepdim=True) + 1e-8)
    
    # Create smooth roughness
    sigma = 0.05 * torch.ones(nx, ny, ntheta, device=device)
    
    grid_map = {
        'z': z,
        'zb': zb,
        'sigma': sigma,
        'x_range': [0.0, 10.0],
        'y_range': [0.0, 10.0],
        'theta_range': [0.0, 2 * np.pi],
    }
    return grid_map


def initialize_trajectory_with_improved_mapping(
    config: OptimizationConfig,
    grid_map: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[TrajectoryOptimizer, MincoTrajectory, TerrainPoseMapper, TrajectoryEvaluator]:
    """Initialize all major components in one place (single-file workflow)."""
    optimizer = TrajectoryOptimizer(config=config, device=device)
    trajectory = MincoTrajectory(config=config, device=device)
    mapper = TerrainPoseMapper(grid_map=grid_map, device=device)
    evaluator = TrajectoryEvaluator(config=config, mapper=mapper, device=device)
    return optimizer, trajectory, mapper, evaluator


def demonstrate_improved_features() -> None:
    """Print concise summary of integrated C++-aligned improvements."""
    print("\n" + "=" * 64)
    print("TRAJ OPTIMIZER (single-file) improvements")
    print("=" * 64)
    print("1) C2 time mapping: exp_c2/log_c2/get_t_to_tau_grad")
    print("2) Terrain-aware constraints: v/a/curvature/attitude/sigma")
    print("3) PHR-ALM + L-BFGS nested optimization")
    print("4) Autograd-based gradient flow (no hand-derived Jacobian)")


def load_terrain_and_initial_trajectory(
    env_name: str = "env000010",
    data_folder: str = "/home/yrf/MPT/data/sim_dataset/val",
    path_index: int = 42,
    device: torch.device = torch.device("cpu"),
    compute_yaw_stability_if_missing: bool = False,
) -> Tuple[Dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load terrain representation and initial trajectory from dataset.
    Following grad_optimizer.py approach.
    
    Args:
        env_name: Environment name
        data_folder: Path to dataset folder
        path_index: Index of path to load
        device: torch device
    
    Returns:
        terrain_data: Dictionary containing terrain normals and elevation
        initial_trajectory: [N, 3] initial trajectory from dataset
        stability_cost_map: [D, H, W] stability cost map
        yaw_stability: [H, W, D] yaw stability binary map
        map_info: Dictionary with resolution, origin, size
    """
    try:
        from dataLoader_dit import UnevenPathDataLoader
    except ImportError:
        logger.warning("Could not import UnevenPathDataLoader, using dummy data instead")
        return None, None, None, None, None
    
    # Load dataset
    logger.info(f"Loading dataset from {data_folder}")
    dataset = UnevenPathDataLoader([env_name], data_folder, compute_stability_map=True)
    
    if path_index >= len(dataset):
        logger.warning(f"Path index {path_index} exceeds dataset size {len(dataset)}")
        path_index = 0
    
    sample = dataset[path_index]
    if sample is None:
        logger.error(f"Failed to load sample at index {path_index}")
        return None, None, None, None, None
    
    logger.info(f"Loaded sample index {path_index}")
    
    # Extract terrain representation
    nx = sample['map'][0, :, :].to(device)  # [H, W]
    ny = sample['map'][1, :, :].to(device)  # [H, W]
    nz = sample['map'][2, :, :].to(device)  # [H, W]
    nz = torch.abs(nz)  # Use absolute value
    
    terrain_data = {
        'nx': nx,
        'ny': ny,
        'nz': nz,
        'elevation': sample['elevation'].to(device) if 'elevation' in sample else nz,
    }
    
    # Extract initial trajectory
    initial_trajectory = sample['trajectory'].to(device)  # [N, 3] (x, y, yaw)
    
    # Extract stability cost map (prefer dataset-provided, do not recompute)
    if 'cost_map' not in sample:
        raise KeyError("sample 中缺少 cost_map，当前流程不再回退重算。")
    stability_cost_map = sample['cost_map'].to(device)  # expected [D, H, W]

    # Extract yaw_stability (prefer dataset-provided; optional fallback)
    yaw_stability = sample.get('yaw_stability', None)
    if yaw_stability is not None:
        yaw_stability = yaw_stability.to(device)
    elif compute_yaw_stability_if_missing:
        from dataLoader_uneven import compute_map_yaw_bins
        yaw_stability = compute_map_yaw_bins(nx, ny, nz, yaw_bins=36).to(device)  # [H, W, 36]
    else:
        logger.warning("sample 不含 yaw_stability，且已禁用重算；将跳过基于 yaw_stability 的统计。")
    
    # Define map info
    map_size = (100, 100, 36)  # W, H, D for (x, y, yaw)
    resolution = 0.4
    origin = (-20.0, -20.0, -np.pi)
    
    map_info = {
        'resolution': resolution,
        'origin': origin,
        'size': map_size
    }
    
    return terrain_data, initial_trajectory, stability_cost_map, yaw_stability, map_info


def evaluate_trajectory_stability(
    trajectory: np.ndarray,
    yaw_stability: torch.Tensor,
    map_info: Dict,
) -> Tuple[np.ndarray, float]:
    """
    Check trajectory stability (reachability) using yaw_stability map.
    Similar to grad_optimizer.py approach.
    
    Args:
        trajectory: [N, 3] trajectory with [x, y, yaw]
        yaw_stability: [H, W, D] binary stability map
        map_info: Dictionary with resolution, origin, size
    
    Returns:
        stability_mask: [N] boolean array, True if unreachable
        unreachable_ratio: Fraction of unreachable points
    """
    origin = map_info['origin']
    resolution = map_info['resolution']
    
    stability_mask = []
    
    for i in range(len(trajectory)):
        x, y, yaw = trajectory[i, 0], trajectory[i, 1], trajectory[i, 2]

        # Invalid point -> directly mark unreachable (avoid int(NaN) crash)
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(yaw)):
            stability_mask.append(True)
            continue
        
        # Convert to grid indices (matching grad_optimizer.py method)
        x_idx = int((x - origin[0]) / resolution)
        y_idx = int((y - origin[1]) / resolution)
        
        # Normalize yaw to [-π, π] and convert to index
        yaw_norm = ((yaw - origin[2]) % (2 * np.pi))
        yaw_idx = int((yaw_norm / (2 * np.pi)) * 36) % 36
        
        # Check bounds and stability
        H, W, D = yaw_stability.shape
        if 0 <= y_idx < H and 0 <= x_idx < W:
            stability_val = yaw_stability[y_idx, x_idx, yaw_idx]
            if torch.is_tensor(stability_val):
                is_unreachable = bool((stability_val == 0).cpu().numpy())
            else:
                is_unreachable = bool(stability_val == 0)
        else:
            is_unreachable = True  # Out of bounds
        
        stability_mask.append(is_unreachable)
    
    stability_mask = np.array(stability_mask, dtype=bool)
    unreachable_ratio = np.sum(stability_mask) / len(stability_mask) if len(stability_mask) > 0 else 0.0
    
    return stability_mask, unreachable_ratio


def print_terrain_and_trajectory_info(
    terrain_data: Dict,
    initial_trajectory: torch.Tensor,
    yaw_stability: torch.Tensor,
    map_info: Dict,
) -> None:
    """
    Print information about loaded terrain and initial trajectory.
    """
    if terrain_data is None or initial_trajectory is None:
        logger.warning("Terrain or trajectory data is None")
        return
    
    print("\n" + "=" * 70)
    print("TERRAIN AND TRAJECTORY INFORMATION")
    print("=" * 70)
    
    # Terrain info
    nx = terrain_data['nx'].cpu().numpy() if torch.is_tensor(terrain_data['nx']) else terrain_data['nx']
    ny = terrain_data['ny'].cpu().numpy() if torch.is_tensor(terrain_data['ny']) else terrain_data['ny']
    nz = terrain_data['nz'].cpu().numpy() if torch.is_tensor(terrain_data['nz']) else terrain_data['nz']
    
    print("\n[Terrain Representation]")
    print(f"  Normal X (nx) shape: {nx.shape}, range: [{np.min(nx):.4f}, {np.max(nx):.4f}]")
    print(f"  Normal Y (ny) shape: {ny.shape}, range: [{np.min(ny):.4f}, {np.max(ny):.4f}]")
    print(f"  Normal Z (nz) shape: {nz.shape}, range: [{np.min(nz):.4f}, {np.max(nz):.4f}]")
    
    # Trajectory info
    traj_np = initial_trajectory.cpu().numpy() if torch.is_tensor(initial_trajectory) else initial_trajectory
    print(f"\n[Initial Trajectory (from dataset)]")
    print(f"  Shape: {traj_np.shape}")
    print(f"  X range: [{np.min(traj_np[:, 0]):.4f}, {np.max(traj_np[:, 0]):.4f}]")
    print(f"  Y range: [{np.min(traj_np[:, 1]):.4f}, {np.max(traj_np[:, 1]):.4f}]")
    print(f"  Yaw range: [{np.min(traj_np[:, 2]):.4f}, {np.max(traj_np[:, 2]):.4f}]")
    
    # Stability info (optional)
    print(f"\n[Yaw Stability Map (binary, H×W×D)]")
    if yaw_stability is None:
        print("  Not available in current sample.")
        print("  Skip yaw-stability-based statistics.")
    else:
        yaw_stab_np = yaw_stability.cpu().numpy() if torch.is_tensor(yaw_stability) else yaw_stability
        print(f"  Shape: {yaw_stab_np.shape}")
        print(f"  Reachable points ratio: {np.sum(yaw_stab_np > 0) / yaw_stab_np.size * 100:.2f}%")
        print(f"  Unreachable points ratio: {np.sum(yaw_stab_np == 0) / yaw_stab_np.size * 100:.2f}%")

        # Evaluate initial trajectory stability
        stability_mask, unreachable_ratio = evaluate_trajectory_stability(
            traj_np, yaw_stability, map_info
        )

        print(f"\n[Initial Trajectory Stability Check]")
        print(f"  Total points: {len(stability_mask)}")
        print(f"  Unreachable points: {np.sum(stability_mask)}")
        print(f"  Unreachable ratio: {unreachable_ratio * 100:.2f}%")

        if np.sum(stability_mask) > 0:
            unreachable_indices = np.where(stability_mask)[0]
            print(f"  Unreachable point indices (first 5): {unreachable_indices[:5]}")
    
    # Map info
    print(f"\n[Map Information]")
    print(f"  Resolution: {map_info['resolution']} m")
    print(f"  Origin: {map_info['origin']}")
    print(f"  Size: {map_info['size']} (W×H×D)")
    
    print("=" * 70 + "\n")


def extract_dense_trajectory_data(
    trajectory_obj: MincoTrajectory,
    opt_q_xy: torch.Tensor,
    opt_q_theta: torch.Tensor,
    opt_tau: torch.Tensor,
    device: torch.device = torch.device("cpu")
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract dense trajectory samples from optimized control points.
    Similar to grad_optimizer.py approach.
    
    Args:
        trajectory_obj: MincoTrajectory instance
        opt_q_xy: Optimized control points [n_ctrl_pts, 2]
        opt_q_theta: Optimized heading [n_ctrl_pts]
        opt_tau: Optimized time parameter
        device: torch device
    
    Returns:
        dense_trajectory: [n_samples, 3] with [x, y, theta]
        control_points: [n_ctrl_pts, 2] with [x, y]
        control_yaws: [n_ctrl_pts] with theta values
        derivatives: [n_samples, 8] with velocity and acceleration data
    """
    # Convert to numpy for storage
    control_points_np = opt_q_xy.detach().cpu().numpy()
    control_yaws_np = opt_q_theta.detach().cpu().numpy()
    
    # Evaluate dense trajectory
    with torch.no_grad():
        states, derivatives, _, _ = trajectory_obj.evaluate(opt_q_xy, opt_q_theta, opt_tau)
    
    dense_trajectory_np = states.detach().cpu().numpy()
    derivatives_np = derivatives.detach().cpu().numpy()

    # Robustness: prevent downstream integer indexing crash from NaN/Inf
    dense_trajectory_np = np.nan_to_num(dense_trajectory_np, nan=0.0, posinf=1e3, neginf=-1e3)
    derivatives_np = np.nan_to_num(derivatives_np, nan=0.0, posinf=1e3, neginf=-1e3)
    
    return dense_trajectory_np, control_points_np, control_yaws_np, derivatives_np


def visualize_optimized_trajectory(
    dense_trajectory: np.ndarray,
    control_points: np.ndarray,
    control_yaws: np.ndarray,
    grid_map: Dict[str, torch.Tensor],
    map_info: Dict = None,
    save_path: str = "/tmp/trajectory_optimization.png",
) -> None:
    """
    Visualize trajectory on terrain using matplotlib.
    Similar to grad_optimizer.py visualize_terrain_trajectory.
    
    Args:
        dense_trajectory: [n_samples, 3] trajectory [x, y, theta]
        control_points: [n_ctrl_pts, 2] control points
        control_yaws: [n_ctrl_pts] control point headings
        grid_map: Terrain grid
        map_info: Optional map info for proper coordinate extent
        save_path: Path to save visualization
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Extract trajectory components
    traj_xy = dense_trajectory[:, :2]
    traj_theta = dense_trajectory[:, 2]
    
    # Create figure with 2D terrain view
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Get terrain height field
    z_field = grid_map['z']
    if isinstance(z_field, torch.Tensor):
        z_field = z_field.cpu().numpy()
    
    # Use first theta plane
    height_map = z_field[:, :, 0]

    # Terrain extent must match map coordinates (avoid stretched/shifted background)
    if map_info is not None:
        x_min = map_info['origin'][0]
        x_max = map_info['origin'][0] + map_info['size'][0] * map_info['resolution']
        y_min = map_info['origin'][1]
        y_max = map_info['origin'][1] + map_info['size'][1] * map_info['resolution']
    else:
        x_min, x_max = grid_map['x_range']
        y_min, y_max = grid_map['y_range']
    extent = [x_min, x_max, y_min, y_max]

    # Display terrain as background with true map extent
    im = ax.imshow(height_map, cmap='terrain', extent=extent, origin='lower', alpha=0.6)
    plt.colorbar(im, ax=ax, label='Terrain Height')
    
    # Plot dense trajectory
    ax.plot(traj_xy[:, 0], traj_xy[:, 1], 
            color='#0044cc', linestyle='-', linewidth=3, 
            label='Optimized Trajectory', zorder=20)
    
    # Plot control points
    ax.scatter(control_points[:, 0], control_points[:, 1], 
              c='red', s=80, alpha=0.7, label='Control Points', 
              edgecolors='darkred', linewidths=2, zorder=15)
    ax.plot(control_points[:, 0], control_points[:, 1], 
           'r--', alpha=0.5, linewidth=1, zorder=14)
    
    # Plot heading arrows on dense trajectory
    arrow_gap = max(1, len(traj_xy) // 12)
    for i in range(0, len(traj_xy), arrow_gap):
        arrow_length = 0.3
        dx = arrow_length * np.cos(traj_theta[i])
        dy = arrow_length * np.sin(traj_theta[i])
        ax.arrow(traj_xy[i, 0], traj_xy[i, 1], dx, dy,
                head_width=0.15, head_length=0.15, 
                fc='#00e0e0', ec='#00e0e0', alpha=0.8, zorder=25)
    
    ax.set_title('Trajectory Optimization Result', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    # 视角覆盖轨迹并保持与地图坐标一致
    x_range = [np.min(traj_xy[:, 0]) - 1.0, np.max(traj_xy[:, 0]) + 1.0]
    y_range = [np.min(traj_xy[:, 1]) - 1.0, np.max(traj_xy[:, 1]) + 1.0]
    ax.set_xlim(max(x_min, x_range[0]), min(x_max, x_range[1]))
    ax.set_ylim(max(y_min, y_range[0]), min(y_max, y_range[1]))
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Trajectory visualization saved to {save_path}")
    plt.show()


def visualize_before_after_comparison(
    init_trajectory: np.ndarray,
    opt_trajectory: np.ndarray,
    init_control_points: np.ndarray,
    opt_control_points: np.ndarray,
    grid_map: Dict[str, torch.Tensor],
    map_info: Dict = None,
    save_path: str = "/tmp/trajectory_comparison.png",
) -> None:
    """
    Visualize before/after comparison of trajectory optimization.
    
    Args:
        init_trajectory: [n_samples, 3] initial trajectory
        opt_trajectory: [n_samples, 3] optimized trajectory
        init_control_points: [n_ctrl_pts, 2] initial control points
        opt_control_points: [n_ctrl_pts, 2] optimized control points
        grid_map: Terrain grid
        map_info: Optional map info for proper extent
        save_path: Path to save comparison image
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Get terrain height field and extent
    z_field = grid_map['z']
    if isinstance(z_field, torch.Tensor):
        z_field = z_field.cpu().numpy()
    
    # Get map extent from grid_map or map_info
    if map_info is not None:
        x_min, x_max = map_info['origin'][0], map_info['origin'][0] + map_info['size'][0] * map_info['resolution']
        y_min, y_max = map_info['origin'][1], map_info['origin'][1] + map_info['size'][1] * map_info['resolution']
        extent = [x_min, x_max, y_min, y_max]
    else:
        x_min, x_max = grid_map['x_range']
        y_min, y_max = grid_map['y_range']
        extent = [x_min, x_max, y_min, y_max]
    
    # Also compute extent from trajectory data to handle both properly
    init_xy = init_trajectory[:, :2]
    opt_xy = opt_trajectory[:, :2]
    all_xy = np.vstack([init_xy, opt_xy])
    
    x_range = [np.min(all_xy[:, 0]) - 1.0, np.max(all_xy[:, 0]) + 1.0]
    y_range = [np.min(all_xy[:, 1]) - 1.0, np.max(all_xy[:, 1]) + 1.0]
    
    height_map = z_field[:, :, 0]
    
    trajectories = [init_trajectory, opt_trajectory]
    control_pts = [init_control_points, opt_control_points]
    titles = ['Initial Trajectory', 'Optimized Trajectory']
    colors = ['#ff6600', '#0044cc']
    
    for idx, (ax, traj, cpts, title, color) in enumerate(zip(
        axes, trajectories, control_pts, titles, colors
    )):
        # Display terrain with proper extent
        im = ax.imshow(height_map, cmap='terrain', extent=extent, 
                      origin='lower', alpha=0.6)
        
        # Plot trajectory
        traj_xy = traj[:, :2]
        traj_theta = traj[:, 2]
        ax.plot(traj_xy[:, 0], traj_xy[:, 1], 
               color=color, linestyle='-', linewidth=3, 
               label='Trajectory', zorder=20)
        
        # Plot control points
        ax.scatter(cpts[:, 0], cpts[:, 1], 
                  c='red', s=80, alpha=0.7, label='Control Points',
                  edgecolors='darkred', linewidths=2, zorder=15)
        ax.plot(cpts[:, 0], cpts[:, 1], 'r--', alpha=0.5, linewidth=1, zorder=14)
        
        # Plot arrows
        arrow_gap = max(1, len(traj_xy) // 10)
        for i in range(0, len(traj_xy), arrow_gap):
            arrow_length = 0.5
            dx = arrow_length * np.cos(traj_theta[i])
            dy = arrow_length * np.sin(traj_theta[i])
            ax.arrow(traj_xy[i, 0], traj_xy[i, 1], dx, dy,
                    head_width=0.2, head_length=0.2,
                    fc='#00e0e0', ec='#00e0e0', alpha=0.8, zorder=25)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        # Use dynamic range from actual trajectory data
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Comparison visualization saved to {save_path}")
    plt.show()


def print_trajectory_statistics(
    dense_trajectory: np.ndarray,
    derivatives_data: np.ndarray = None,
) -> None:
    """
    Print trajectory statistics (similar to grad_optimizer.py logging).
    
    Args:
        dense_trajectory: [n_samples, 3] trajectory
        derivatives_data: Optional [n_samples, 8] derivatives
    """
    print("\n" + "=" * 70)
    print("TRAJECTORY OPTIMIZATION RESULTS")
    print("=" * 70)
    
    traj_xy = dense_trajectory[:, :2]
    traj_theta = dense_trajectory[:, 2]
    
    # Compute trajectory properties
    diffs = np.diff(traj_xy, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    total_length = np.sum(segment_lengths)
    
    print(f"\n[Geometry]")
    print(f"  Number of samples:      {len(dense_trajectory)}")
    print(f"  Total trajectory length: {total_length:.4f} m")
    print(f"  Mean segment length:     {np.mean(segment_lengths):.6f} m")
    print(f"  Max segment length:      {np.max(segment_lengths):.6f} m")
    
    print(f"\n[Heading]")
    print(f"  Initial theta:          {traj_theta[0]:.4f} rad ({np.degrees(traj_theta[0]):.2f}°)")
    print(f"  Final theta:            {traj_theta[-1]:.4f} rad ({np.degrees(traj_theta[-1]):.2f}°)")
    print(f"  Theta range:            [{np.min(traj_theta):.4f}, {np.max(traj_theta):.4f}]")
    
    print(f"\n[Position Range]")
    print(f"  X range:                [{np.min(traj_xy[:, 0]):.4f}, {np.max(traj_xy[:, 0]):.4f}]")
    print(f"  Y range:                [{np.min(traj_xy[:, 1]):.4f}, {np.max(traj_xy[:, 1]):.4f}]")
    
    if derivatives_data is not None:
        v_x = derivatives_data[:, 0]
        v_y = derivatives_data[:, 1]
        v_planar = np.sqrt(v_x**2 + v_y**2 + 1e-8)
        
        a_x = derivatives_data[:, 3]
        a_y = derivatives_data[:, 4]
        a_planar = np.sqrt(a_x**2 + a_y**2 + 1e-8)
        
        print(f"\n[Velocity]")
        print(f"  Mean velocity:          {np.mean(v_planar):.4f} m/s")
        print(f"  Max velocity:           {np.max(v_planar):.4f} m/s")
        print(f"  Min velocity:           {np.min(v_planar):.6f} m/s")
        
        print(f"\n[Acceleration]")
        print(f"  Mean acceleration:      {np.mean(a_planar):.6f} m/s²")
        print(f"  Max acceleration:       {np.max(a_planar):.6f} m/s²")
        print(f"  Min acceleration:       {np.min(a_planar):.6f} m/s²")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    # ========================================
    # 1. Load terrain and initial trajectory from dataset
    # ========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("=" * 80)
    logger.info("PHASE 1: Loading Real Dataset Terrain and Trajectory")
    logger.info("=" * 80)
    
    terrain_data, initial_trajectory, stability_cost_map, yaw_stability, map_info = load_terrain_and_initial_trajectory(
        env_name="env000010",
        data_folder="/home/yrf/MPT/data/sim_dataset/val",
        path_index=42,
        device=device
    )
    
    if terrain_data is not None and initial_trajectory is not None:
        # Print information about loaded data
        print_terrain_and_trajectory_info(terrain_data, initial_trajectory, yaw_stability, map_info)
        
        # Extract numpy versions for visualization
        initial_traj_np = initial_trajectory.cpu().numpy() if torch.is_tensor(initial_trajectory) else initial_trajectory
        
        # Evaluate stability of initial trajectory (optional)
        if yaw_stability is not None:
            logger.info("\nEvaluating initial trajectory stability...")
            stability_mask_init, unreachable_ratio_init = evaluate_trajectory_stability(
                initial_traj_np, yaw_stability, map_info
            )
            logger.info(f"Initial trajectory unreachable points: {np.sum(stability_mask_init)}/{len(stability_mask_init)} "
                       f"({unreachable_ratio_init*100:.2f}%)")
        else:
            stability_mask_init = None
            unreachable_ratio_init = None
            logger.info("\nSkip initial yaw-stability evaluation (yaw_stability unavailable).")
        
        # Create grid map from terrain for optimization
        logger.info("Converting terrain data to optimization grid map...")
        # Get dimensions
        H, W = terrain_data['nz'].shape
        
        # Create 3D grids [nx, ny, ntheta]
        # sigma 使用数据集 cost_map，并统一转换到 (W, H, D)
        # observed dataset shape is (H, W, D), but we also兼容 (D, H, W)
        if stability_cost_map.ndim != 3:
            raise ValueError(f"cost_map 维度错误: {stability_cost_map.shape}")

        if stability_cost_map.shape[0] == H and stability_cost_map.shape[1] == W:
            # (H, W, D) -> (W, H, D)
            sigma_3d = stability_cost_map.permute(1, 0, 2).contiguous().to(device)
            ntheta = int(stability_cost_map.shape[2])
        elif stability_cost_map.shape[1] == H and stability_cost_map.shape[2] == W:
            # (D, H, W) -> (W, H, D)
            sigma_3d = stability_cost_map.permute(2, 1, 0).contiguous().to(device)
            ntheta = int(stability_cost_map.shape[0])
        else:
            raise ValueError(
                f"无法识别 cost_map 轴顺序: {tuple(stability_cost_map.shape)}, 期望 (H,W,D) 或 (D,H,W)"
            )

        z_3d = torch.zeros(W, H, ntheta, device=device)
        zb_3d = torch.zeros(W, H, ntheta, 3, device=device)
        
        # Tile terrain data across theta dimension
        for t_idx in range(ntheta):
            z_3d[:, :, t_idx] = terrain_data['nz'].T  # Transpose to match nx, ny order
            zb_3d[:, :, t_idx, 0] = terrain_data['nx'].T
            zb_3d[:, :, t_idx, 1] = terrain_data['ny'].T
            zb_3d[:, :, t_idx, 2] = terrain_data['nz'].T
        
        grid_map = {
            'z': z_3d,
            'zb': zb_3d,
            'sigma': sigma_3d,
            'x_range': [map_info['origin'][0], map_info['origin'][0] + map_info['size'][0] * map_info['resolution']],
            'y_range': [map_info['origin'][1], map_info['origin'][1] + map_info['size'][1] * map_info['resolution']],
            'theta_range': [map_info['origin'][2], map_info['origin'][2] + 2*np.pi],
        }
    else:
        # Fallback: use dummy grid map if loading fails
        logger.warning("Failed to load real dataset, using dummy grid map")
        initial_traj_np = None
        stability_mask_init = None
        unreachable_ratio_init = None
        grid_map = create_dummy_grid_map(device=device)
    
    # ========================================
    # 2. Set up optimization
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: Trajectory Optimization with traj_optimizer")
    logger.info("=" * 80)
    
    config = OptimizationConfig(
        n_samples=100,
        n_control_points=15,
        rho_time=500.0,
        rho_heading=200.0,
        rho_terrain=10.0,
        rho_nonholo_soft=8000.0,
        v_max=0.5,
        a_lon_max=5.0,
        a_lat_max=10.0,
        max_kap=2.1,
        min_cxi=0.8,
        c_min=0.8,
        sigma_max=0.05,
        use_scaling=True,
        rho=1.0,
        beta=5000.0,
        gamma=1.0,
        epsilon_con=0.001,
        max_iter=24,
        g_epsilon=1e-3,
        min_step=1e-32,
        inner_max_iter=10000,
        inner_max_iter_runtime=80,
        delta=1e-4,
        mem_size=256,
        past=3,
        int_K=16,
        align_cpp_sampling=False,
        closure_log_interval=20,
    )
    
    optimizer = TrajectoryOptimizer(config=config, device=device)
    trajectory_obj = MincoTrajectory(config=config, device=device)
    
    # Use initial trajectory control points if available, else use default
    if initial_traj_np is not None and len(initial_traj_np) >= 5:
        # Sample 5 control points from the initial trajectory
        indices = np.linspace(0, len(initial_traj_np) - 1, config.n_control_points, dtype=int)
        init_q_xy = initial_traj_np[indices, :2]
        init_q_theta = initial_traj_np[indices, 2]
    else:
        # Default initialization
        init_q_xy = np.array([[0, 0], [2, 2], [4, 4], [6, 6], [8, 8]])
        init_q_theta = np.array([0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2])
    
    # Initialize tau from path length to avoid infeasible ultra-short total time
    if initial_traj_np is not None and len(initial_traj_np) >= 2:
        diffs = np.diff(initial_traj_np[:, :2], axis=0)
        path_len = float(np.sum(np.linalg.norm(diffs, axis=1)))
    else:
        diffs = np.diff(init_q_xy, axis=0)
        path_len = float(np.sum(np.linalg.norm(diffs, axis=1)))

    v_ref = max(0.1, 0.8 * config.v_max)
    desired_total_time = max(1.0, path_len / v_ref)
    tau_init_t = MincoTrajectory.log_c2(torch.tensor([desired_total_time], dtype=torch.float32))
    tau_init_t = torch.clamp(tau_init_t, -config.tau_clip, config.tau_clip)
    init_tau = tau_init_t.cpu().numpy().astype(np.float32)
    logger.info(
        f"Init timing from path length: L={path_len:.3f}m, T_des={desired_total_time:.3f}s, tau={init_tau[0]:.3f}"
    )
    
    # Evaluate initial trajectory before optimization
    logger.info("Preparing initial trajectory for comparison...")
    # Use the raw initial trajectory from dataset for visualization (if available)
    try:
        init_dense_traj = initial_traj_np
        logger.info(f"Using raw initial trajectory from dataset: shape {init_dense_traj.shape}")
    except NameError:
        init_dense_traj = None
        logger.info("No initial trajectory available from dataset")
    
    logger.info("Running trajectory optimization...")
    opt_q_xy, opt_q_theta, opt_tau = optimizer.optimize(
        init_q_xy, init_q_theta, init_tau, grid_map,
        yaw_stability_map=yaw_stability if 'yaw_stability' in locals() else None,
        fix_endpoints=True
    )
    
    logger.info("\nOptimization complete!")
    logger.info(f"Optimized q_xy shape: {opt_q_xy.shape}")
    logger.info(f"Optimized q_theta shape: {opt_q_theta.shape}")
    logger.info(f"Optimized tau shape: {opt_tau.shape}")
    
    # ========================================
    # 3. Data Extraction & Visualization
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3: Results Extraction and Visualization")
    logger.info("=" * 80)
    
    logger.info("Extracting dense trajectory data...")
    dense_trajectory, control_points, control_yaws, derivatives_data = extract_dense_trajectory_data(
        trajectory_obj=trajectory_obj,
        opt_q_xy=opt_q_xy,
        opt_q_theta=opt_q_theta,
        opt_tau=opt_tau,
        device=device
    )
    
    logger.info(f"Dense trajectory shape: {dense_trajectory.shape}")
    logger.info(f"Control points shape: {control_points.shape}")
    logger.info(f"Derivatives shape: {derivatives_data.shape}")
    
    # Evaluate optimized trajectory stability
    if yaw_stability is not None:
        logger.info("Evaluating optimized trajectory stability...")
        stability_mask_opt, unreachable_ratio_opt = evaluate_trajectory_stability(
            dense_trajectory, yaw_stability, map_info
        )
        logger.info(f"Optimized trajectory unreachable points: {np.sum(stability_mask_opt)}/{len(stability_mask_opt)} "
                   f"({unreachable_ratio_opt*100:.2f}%)")
        
        if unreachable_ratio_init is not None:
            improvement = (unreachable_ratio_init - unreachable_ratio_opt) / (unreachable_ratio_init + 1e-8) * 100
            logger.info(f"Stability improvement: {improvement:.2f}%")
    
    # Print trajectory statistics
    print_trajectory_statistics(dense_trajectory, derivatives_data)
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    visualize_optimized_trajectory(
        dense_trajectory=dense_trajectory,
        control_points=control_points,
        control_yaws=control_yaws,
        grid_map=grid_map,
        map_info=map_info if 'map_info' in locals() else None,
        save_path="/tmp/traj_optimizer_result.png"
    )
    
    if initial_traj_np is not None:
        visualize_before_after_comparison(
            init_trajectory=init_dense_traj,
            opt_trajectory=dense_trajectory,
            init_control_points=init_q_xy,
            opt_control_points=control_points,
            grid_map=grid_map,
            map_info=map_info,
            save_path="/tmp/traj_optimizer_comparison.png"
        )
    
    logger.info("\n" + "=" * 80)
    logger.info("All tasks completed successfully!")
    logger.info("=" * 80)

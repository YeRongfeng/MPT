"""train_langevin.py - 两阶段训练不平坦地面路径预测模型。"""

import numpy as np
import pickle
from contextlib import nullcontext

import torch
import torch.optim as optim

import json
import argparse

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from os import path as osp

from transformer import Optim
from dataLoader_dit import UnevenPathDataLoader, PaddedSequence

from torch.utils.tensorboard import SummaryWriter
from timm.utils import ModelEmaV2

from dit.Models import PathDiffusionTransformer

# B样条工具
from bspline_utils import (
    fit_bspline_least_squares,
    DifferentiableBSpline
)
from torch.func import functional_call, jvp

# =================== B样条控制点转换 ===================
def trajectory_to_control_points(trajectory, num_middle_points=24):
    """
    将轨迹转换为B样条控制点（仅返回中间控制点，不包含起终点）
    
    【语义说明】
    - 完整B样条需要26个控制点 = 起点(1) + 中间点(24) + 终点(1)
    - 起点和终点是固定的，由start_pose和goal_pose决定
    - 本函数只负责提取24个中间点；训练时会再拼回真实首尾，组成26点监督目标
    
    Args:
        trajectory: (B, N, 3) 轨迹张量 [x, y, theta]
        num_middle_points: 中间控制点数量（默认24，不包含起终点）
    
    Returns:
        middle_control_points: (B, num_middle_points, 2) 中间控制点张量（只包含x,y，不包含起终点）
    """
    B, N, _ = trajectory.shape
    device = trajectory.device
    
    # 将张量转换为numpy进行B样条拟合
    trajectory_np = trajectory.cpu().numpy()
    
    middle_control_points_list = []
    for i in range(B):
        traj = trajectory_np[i]  # (N, 3)
        
        # 拟合完整的26个控制点（包含起终点）
        full_cp, _, _ = fit_bspline_least_squares(
            traj[:, :2],  # 只使用x,y
            num_control_points=num_middle_points + 2,  # 26个控制点
            degree=3
        )
        
        # 只保留中间24个控制点，排除起终点
        middle_cp = full_cp[1:-1]  # 去掉第一个和最后一个
        middle_control_points_list.append(middle_cp)
    
    # 转换回张量
    middle_control_points = torch.tensor(np.stack(middle_control_points_list, axis=0), dtype=torch.float32, device=device)
    
    return middle_control_points  # (B, num_middle_points, 2)


def compute_tangent_loss(pred_middle_cps, start_pose, goal_pose):
    """
    基于投影距离的切线约束 (比 Cosine Similarity 更强力且稳定)
    
    原理:
    理想情况下，向量 V = P1 - P0 应该与方向向量 D = (cos, sin) 平行。
    这意味着 V 和 D 的 2D 叉乘 (Cross Product) 应该为 0。
    Cross = V_x * D_y - V_y * D_x
    Loss = Cross^2 (本质上是 P1 到理想射线的垂直距离的平方)
    
    Args:
        pred_middle_cps: (B, N, 2) 预测的中间控制点
        start_pose: (B, 4) -> (x, y, cos, sin)
        goal_pose:  (B, 4) -> (x, y, cos, sin)
    """
    # 1. 提取向量 P0 -> P1
    p0 = start_pose[:, :2]
    p1 = pred_middle_cps[:, 0, :]
    vec_start = p1 - p0  # (B, 2)
    
    # 提取方向 D_start
    d_start = start_pose[:, 2:] # (B, 2) -> (cos, sin)
    
    # 2. 提取向量 P_last -> P_goal
    p_goal = goal_pose[:, :2]
    p_last = pred_middle_cps[:, -1, :]
    vec_end = p_goal - p_last # (B, 2)
    
    # 提取方向 D_goal
    d_goal = goal_pose[:, 2:] # (B, 2)
    
    # ==========================================
    # 核心修正 1: 计算 2D 叉乘 (Cross Product)
    # Cross = |V|*|D|*sin(theta). 当平行时为0。
    # 这等价于点到直线的垂直距离 (如果 D 是单位向量)
    # ==========================================
    
    # Start 部分
    # vec_start = (vx, vy), d_start = (dx, dy)
    # cross = vx * dy - vy * dx
    cross_start = vec_start[:, 0] * d_start[:, 1] - vec_start[:, 1] * d_start[:, 0]
    
    # End 部分 (注意方向是 P_last -> P_goal，应与 Goal 朝向一致)
    cross_end = vec_end[:, 0] * d_goal[:, 1] - vec_end[:, 1] * d_goal[:, 0]
    
    # Loss 1: 垂直距离平方
    dist_loss = torch.mean(cross_start ** 2) + torch.mean(cross_end ** 2)
    
    # ==========================================
    # 核心修正 2: 防止反向 (Dot Product Constraint)
    # 我们希望 P1 在 P0 前方，即 Dot > 0
    # 如果 Dot < 0，说明倒车了，给予惩罚
    # ==========================================
    
    # Start 部分 Dot
    dot_start = vec_start[:, 0] * d_start[:, 0] + vec_start[:, 1] * d_start[:, 1]
    # End 部分 Dot
    dot_end = vec_end[:, 0] * d_goal[:, 0] + vec_end[:, 1] * d_goal[:, 1]
    
    # ReLU(-dot): 只有当 dot < 0 (反向) 时才有 loss
    dir_loss = torch.mean(F.relu(-dot_start)) + torch.mean(F.relu(-dot_end))
    
    # 总 Loss
    return dist_loss + dir_loss


class LangevinReplayBuffer:
    """CPU replay buffer for detached stage-2 Langevin targets."""

    def __init__(self, max_size=4096):
        self.max_size = int(max_size)
        self.storage = []
        self.next_idx = 0

    def __len__(self):
        return len(self.storage)

    def add_batch(self, map_input, start_pose, goal_pose, target_middle_cp):
        map_cpu = map_input.detach().to('cpu', dtype=torch.float16)
        start_cpu = start_pose.detach().to('cpu', dtype=torch.float32)
        goal_cpu = goal_pose.detach().to('cpu', dtype=torch.float32)
        target_cpu = target_middle_cp.detach().to('cpu', dtype=torch.float32)

        n = target_cpu.shape[0]
        for i in range(n):
            item = {
                'map': map_cpu[i],
                'start_pose': start_cpu[i],
                'goal_pose': goal_cpu[i],
                'target_middle_cp': target_cpu[i],
            }
            if len(self.storage) < self.max_size:
                self.storage.append(item)
            else:
                self.storage[self.next_idx] = item
                self.next_idx = (self.next_idx + 1) % self.max_size

    def sample(self, batch_size, device):
        if len(self.storage) == 0:
            raise RuntimeError("Cannot sample from an empty Langevin replay buffer.")

        indices = torch.randint(0, len(self.storage), (int(batch_size),)).tolist()
        items = [self.storage[i] for i in indices]
        return {
            'map': torch.stack([item['map'] for item in items], dim=0).to(device=device, dtype=torch.float32),
            'start_pose': torch.stack([item['start_pose'] for item in items], dim=0).to(device=device),
            'goal_pose': torch.stack([item['goal_pose'] for item in items], dim=0).to(device=device),
            'target_middle_cp': torch.stack([item['target_middle_cp'] for item in items], dim=0).to(device=device),
        }


def stage2_optimize_control_points(
    model, map_input, start_normalized, goal_normalized, start_pose, goal_pose,
    stability_cost_map, map_info, device,
    num_iterations=10, num_particles=32, temperature_kappa=1.0,
    rho_grad=0.05, rho_noise=0.02, rho_max=0.1,
    temperature_min=1e-6, eta_min=1e-5, eta_max=10.0,
    noise_scale=0.3, clamp_abs=20.0, stats_ema_alpha=0.05,
    diagnostics=True
):
    """
    【第二阶段 Langevin 分布迁移】从当前模型采样出控制点，并用
    q_T(x|c) ∝ exp[-C(x,c) / T] 对样本做 Langevin 更新。
    
    与直接最小化 cost 不同，Langevin 更新保留噪声项：
        x <- x - eta * grad_x C(x,c) + sqrt(2*T*eta) * noise
    这样第二阶段目标是一个有温度的低 cost 分布，而不是单点最优轨迹。
    
    Args:
        model: PathDiffusionTransformer 模型
        map_input: (B, C, H, W) 输入地图
        start_normalized: (B, 4) 起点（已归一化）
        goal_normalized: (B, 4) 终点（已归一化）
        start_pose: (B, 3) 起点原始坐标
        goal_pose: (B, 3) 终点原始坐标
        stability_cost_map: (D, H, W) 稳定性代价地图
        map_info: dict 地图信息
        device: 设备
        num_iterations: Langevin 内循环步数
        num_particles: 每个条件本次刷新的小粒子数 k_small
        temperature_kappa: T = (P90(C)-P10(C)) / kappa
        rho_grad: 梯度步长占粒子典型距离 D 的比例
        rho_noise: 噪声步长占粒子典型距离 D 的比例
        rho_max: 单步最大梯度位移占 D 的比例，用于 g_max = rho_max * D / eta
        noise_scale: Langevin 噪声倍率；设为0会退化为确定性梯度下降
        clamp_abs: 控制点绝对坐标裁剪范围
    
    Returns:
        optimized_middle_cp: (B*k_small, 24, 2) 优化后的中间控制点
        metrics: Langevin 参数与迁移质量监控指标
    """
    from grad_optimizer import cost_on_dense_trajectory
    
    # 从 start_pose 推导 batch size
    B = start_pose.shape[0]
    K = max(1, int(num_particles))
    
    # B样条层
    bspline_layer = DifferentiableBSpline(
        num_control_points=26,
        num_output_points=100,
        degree=3
    ).to(device)
    
    with torch.enable_grad():
        # 从当前模型分布初始化，而不是从专家轨迹初始化。
        with torch.no_grad():
            sample_model = model.module if hasattr(model, 'module') else model
            prev_training = sample_model.training
            sample_model.eval()
            try:
                sampled_control_points = sample_model.sample(
                    map_input,  # (B, C, H, W)
                    start_normalized,  # (B, 4)
                    goal_normalized,  # (B, 4)
                    num_samples=K,
                    num_steps=3,
                    solver='pmf_refined',
                    reconstruct_trajectory=False,
                    num_traj_points=100
            )  # (B*k_small, 26, 2)
            finally:
                sample_model.train(prev_training)
        
        # 提取中间控制点（去掉起终点）
        x_langevin = sampled_control_points[:, 1:-1, :].clone().detach()  # (B*k_small, 24, 2)
        
        # 起点和终点按粒子数展开，顺序与 model.sample 的 repeat_interleave 保持一致
        start_pose_particles = start_pose.repeat_interleave(K, dim=0)
        goal_pose_particles = goal_pose.repeat_interleave(K, dim=0)
        start_cp = start_pose_particles[:, :2].unsqueeze(1)  # (B*k_small, 1, 2)
        goal_cp = goal_pose_particles[:, :2].unsqueeze(1)    # (B*k_small, 1, 2)

        def build_dense_trajectory(middle_control_points):
            full_control_points = torch.cat([
                start_cp,
                middle_control_points,
                goal_cp
            ], dim=1)  # (B*k_small, 26, 2)
            return bspline_layer(full_control_points)

        def cost_values_for(middle_control_points):
            return cost_on_dense_trajectory(
                build_dense_trajectory(middle_control_points),
                start_pose_particles,
                goal_pose_particles,
                stability_cost_map,
                map_info,
                device,
                return_per_sample=True
            )

        def median_pairwise_distance(middle_control_points):
            flat = middle_control_points.detach().reshape(middle_control_points.shape[0], -1)
            if flat.shape[0] <= 1:
                return torch.tensor(float(flat.shape[1]) ** 0.5, device=device)
            return torch.pdist(flat, p=2).median().clamp_min(1e-6)

        def covariance_trace(middle_control_points):
            flat = middle_control_points.detach().reshape(middle_control_points.shape[0], -1)
            centered = flat - flat.mean(dim=0, keepdim=True)
            return centered.pow(2).sum(dim=1).mean()

        # 先用当前粒子估计 cost 尺度、粒子距离和梯度尺度。
        x_stats = x_langevin.detach().requires_grad_(True)
        cost_before_values = cost_values_for(x_stats)
        grad_stats = torch.autograd.grad(
            cost_before_values.sum(),
            x_stats,
            retain_graph=False,
            create_graph=False
        )[0]
        grad_stats = torch.nan_to_num(grad_stats, nan=0.0, posinf=0.0, neginf=0.0)

        cost_before_flat = cost_before_values.detach().flatten()
        p90 = torch.quantile(cost_before_flat, 0.90)
        p10 = torch.quantile(cost_before_flat, 0.10)
        delta_cost_batch = (p90 - p10).clamp_min(temperature_min)

        D_batch = median_pairwise_distance(x_langevin).clamp_min(1e-6)
        grad_norm = torch.norm(grad_stats.detach().flatten(start_dim=1), dim=1)
        G_batch = grad_norm.median().clamp_min(1e-8)

        ema_alpha = float(stats_ema_alpha)
        ema_alpha = max(0.0, min(1.0, ema_alpha))
        if not hasattr(stage2_optimize_control_points, '_stat_ema'):
            stage2_optimize_control_points._stat_ema = {
                'delta_cost': delta_cost_batch.detach(),
                'D': D_batch.detach(),
                'G': G_batch.detach(),
            }
        else:
            stat_ema = stage2_optimize_control_points._stat_ema
            stat_ema['delta_cost'] = (1.0 - ema_alpha) * stat_ema['delta_cost'].to(device) + ema_alpha * delta_cost_batch.detach()
            stat_ema['D'] = (1.0 - ema_alpha) * stat_ema['D'].to(device) + ema_alpha * D_batch.detach()
            stat_ema['G'] = (1.0 - ema_alpha) * stat_ema['G'].to(device) + ema_alpha * G_batch.detach()

        stat_ema = stage2_optimize_control_points._stat_ema
        delta_cost = stat_ema['delta_cost'].to(device).clamp_min(temperature_min)
        D = stat_ema['D'].to(device).clamp_min(1e-6)
        G = stat_ema['G'].to(device).clamp_min(1e-8)
        temperature = (delta_cost / max(float(temperature_kappa), 1e-8)).clamp_min(temperature_min)

        d = x_langevin[0].numel()
        eta_grad = float(rho_grad) * D / (G + 1e-8)
        eta_noise = (float(rho_noise) * D) ** 2 / (2.0 * temperature * d + 1e-8)
        eta = torch.minimum(eta_grad, eta_noise)
        eta = torch.clamp(eta, min=float(eta_min), max=float(eta_max))
        grad_clip_norm = (float(rho_max) * D / (eta + 1e-8)).clamp_min(1e-8)

        grad_stats_norm_particles = torch.norm(grad_stats.detach().flatten(start_dim=1), dim=1).view(-1, 1, 1)
        grad_stats_scale = torch.clamp(grad_clip_norm.view(1, 1, 1) / (grad_stats_norm_particles + 1e-8), max=1.0)
        grad_stats_clipped = grad_stats.detach() * grad_stats_scale
        grad_step_norm = (
            eta * torch.norm(grad_stats_clipped.flatten(start_dim=1), dim=1).median()
        ).clamp_min(0.0)
        noise_step_norm = (
            torch.sqrt(2.0 * temperature * eta) * float(noise_scale) * (float(d) ** 0.5)
        ).clamp_min(0.0)
        step_snr = grad_step_norm / (noise_step_norm + 1e-8)

        temperature_particles = temperature.view(1, 1, 1)
        eta_particles = eta.view(1, 1, 1)
        grad_clip_particles = grad_clip_norm.view(1, 1, 1)

        def clipped_cost_grad(current_x):
            current_x = current_x.detach().requires_grad_(True)
            current_cost_values = cost_values_for(current_x)
            current_grad = torch.autograd.grad(
                current_cost_values.sum(),
                current_x,
                retain_graph=False,
                create_graph=False
            )[0]
            current_grad = torch.nan_to_num(current_grad, nan=0.0, posinf=0.0, neginf=0.0)

            current_grad_norm = torch.norm(current_grad.flatten(start_dim=1), dim=1).view(-1, 1, 1)
            current_grad_scale = torch.clamp(grad_clip_particles / (current_grad_norm + 1e-8), max=1.0)
            return current_x, current_cost_values, current_grad * current_grad_scale

        cost_grad_only_values = None
        if diagnostics:
            x_grad_only = x_langevin.detach()
            for _ in range(num_iterations):
                x_grad_only, _, grad_only = clipped_cost_grad(x_grad_only)
                with torch.no_grad():
                    x_grad_only = x_grad_only - eta_particles * grad_only
                    x_grad_only = torch.clamp(x_grad_only, -clamp_abs, clamp_abs)
            with torch.no_grad():
                cost_grad_only_values = cost_values_for(x_grad_only.detach())
        
        # batch 级 Langevin 内循环。这里不更新模型参数，只移动样本。
        for _ in range(num_iterations):
            x_langevin, _, grad = clipped_cost_grad(x_langevin)

            noise = torch.randn_like(x_langevin) if noise_scale > 0 else torch.zeros_like(x_langevin)
            with torch.no_grad():
                x_langevin = x_langevin \
                    - eta_particles * grad \
                    + torch.sqrt(2.0 * temperature_particles * eta_particles) * noise_scale * noise
                x_langevin = torch.clamp(x_langevin, -clamp_abs, clamp_abs)
        
        with torch.no_grad():
            cost_after_values = cost_values_for(x_langevin.detach())
            if cost_grad_only_values is None:
                cost_grad_only_values = torch.zeros_like(cost_after_values)
            D_after = median_pairwise_distance(x_langevin)
            cov_before = covariance_trace(x_stats.detach())
            cov_after = covariance_trace(x_langevin)
            diversity_ratio = D_after / (D + 1e-8)
            metrics = {
                'langevin_T': temperature.mean().item(),
                'langevin_delta_cost': delta_cost.mean().item(),
                'langevin_G': G.mean().item(),
                'langevin_eta_grad': eta_grad.mean().item(),
                'langevin_eta_noise': eta_noise.mean().item(),
                'langevin_eta': eta.mean().item(),
                'langevin_eta_at_max': float(eta.mean().item() >= float(eta_max) * 0.999),
                'langevin_gmax': grad_clip_norm.mean().item(),
                'langevin_grad_step_norm': grad_step_norm.mean().item(),
                'langevin_noise_step_norm': noise_step_norm.mean().item(),
                'langevin_step_snr': step_snr.mean().item(),
                'langevin_D_before': D.mean().item(),
                'langevin_D_after': D_after.mean().item(),
                'langevin_diversity_ratio': diversity_ratio.mean().item(),
                'langevin_cov_trace_before': cov_before.mean().item(),
                'langevin_cov_trace_after': cov_after.mean().item(),
                'langevin_cov_trace_ratio': (cov_after / (cov_before + 1e-8)).mean().item(),
                'langevin_cost_before': cost_before_values.detach().mean().item(),
                'langevin_cost_after': cost_after_values.detach().mean().item(),
                'langevin_cost_grad_only': cost_grad_only_values.detach().mean().item(),
                'langevin_cost_delta': (cost_after_values.detach() - cost_before_values.detach()).mean().item(),
                'langevin_cost_delta_grad_only': (
                    cost_grad_only_values.detach() - cost_before_values.detach()
                ).mean().item(),
            }

        # 返回优化后的控制点
        optimized_middle_cp = x_langevin.detach()  # (B*k_small, 24, 2)
    
    return optimized_middle_cp, metrics



def diffusion_loss(model, batch, device, loss_weights=None, current_stage=1, prediction_type='x0'):
    """
    混合损失函数（绝对坐标版本）
    包括：
    - 主损失: epsilon/x0/v预测损失
    - 平滑性损失: 轨迹的二阶导数（改进版：只惩罚过度弯曲）
    - 曲率约束: 限制最大曲率
    - 角度一致性损失: 角度变化的平滑性
    - 均匀性损失: 点间距离的方差
    - 倾覆监督损失: 基于稳定性代价地图的损失
    
    Args:
        loss_weights: dict with keys such as ['main', 'tangent', 'langevin_steps']
        current_stage: 当前训练阶段（1或2），只在阶段2执行高级损失计算
        prediction_type: str - 预测类型 ('epsilon', 'x0', 'v')
    """
    # 默认权重
    if loss_weights is None:
        loss_weights = {
            'main': 1.0,
            'tangent': 0.0,
        }
    map_input = batch['map'].float().to(device)
    trajectory = batch.get('trajectory', None)
    if trajectory is not None:
        trajectory = trajectory.to(device)  # (B, 100, 3)
    start_pose = batch['start_pose'].to(device)  # (B, 3)
    goal_pose = batch['goal_pose'].to(device)  # (B, 3)
    
    B = map_input.shape[0]
    condition_count = B
    
    # =================== B样条控制点转换 ===================
    # 【语义说明】
    # - 完整B样条：26个控制点 = 起点(1) + 中间点(24) + 终点(1)
    # - 起点/终点：从start_pose/goal_pose提取并加入训练目标
    # - 网络对完整26点统一加噪、统一预测；推理结束后才覆盖真实首尾
    # 
    # 【两阶段目标】
    # - 第一阶段（current_stage==1）：使用数据轨迹做模仿学习。
    # - 第二阶段（current_stage==2）：只使用 Langevin 迁移后的粒子做目标，
    #   不再混合专家控制点。
    
    middle_cp = None

    # 第二阶段：只使用 Langevin dynamics 得到的新分布目标样本。
    # q_T(x|c) ∝ exp[-C(x,c)/T]，不再混合模仿学习控制点。
    middle_cp_optimized = None
    langevin_stats = {
        'langevin_T': 0.0,
        'langevin_delta_cost': 0.0,
        'langevin_G': 0.0,
        'langevin_eta_grad': 0.0,
        'langevin_eta_noise': 0.0,
        'langevin_eta': 0.0,
        'langevin_eta_at_max': 0.0,
        'langevin_gmax': 0.0,
        'langevin_grad_step_norm': 0.0,
        'langevin_noise_step_norm': 0.0,
        'langevin_step_snr': 0.0,
        'langevin_D_before': 0.0,
        'langevin_D_after': 0.0,
        'langevin_diversity_ratio': 0.0,
        'langevin_cov_trace_before': 0.0,
        'langevin_cov_trace_after': 0.0,
        'langevin_cov_trace_ratio': 0.0,
        'langevin_cost_before': 0.0,
        'langevin_cost_after': 0.0,
        'langevin_cost_grad_only': 0.0,
        'langevin_cost_delta': 0.0,
        'langevin_cost_delta_grad_only': 0.0,
        'langevin_buffer_size': 0.0,
    }
    if current_stage == 1:
        if trajectory is None:
            raise ValueError("Stage 1 requires batch['trajectory'] for imitation training.")
        middle_cp = trajectory_to_control_points(trajectory, num_middle_points=24)  # (B, 24, 2)
    elif current_stage == 2:
        if 'target_middle_cp' in batch:
            middle_cp = batch['target_middle_cp'].to(device)
            if middle_cp.shape[0] != B:
                raise ValueError(
                    f"target_middle_cp batch mismatch: target={middle_cp.shape[0]}, conditions={B}"
                )
        else:
            if 'cost_map' not in batch:
                raise ValueError("Stage 2 requires replay targets or batch['cost_map']; imitation fallback is disabled.")

            start_normalized_temp = torch.zeros(B, 4, device=device)
            start_normalized_temp[:, :2] = start_pose[:, :2] / 20.0
            start_normalized_temp[:, 2] = torch.cos(start_pose[:, 2])
            start_normalized_temp[:, 3] = torch.sin(start_pose[:, 2])
            start_normalized_temp[:, :2] = torch.clamp(start_normalized_temp[:, :2], -1.0, 1.0)
            
            goal_normalized_temp = torch.zeros(B, 4, device=device)
            goal_normalized_temp[:, :2] = goal_pose[:, :2] / 20.0
            goal_normalized_temp[:, 2] = torch.cos(goal_pose[:, 2])
            goal_normalized_temp[:, 3] = torch.sin(goal_pose[:, 2])
            goal_normalized_temp[:, :2] = torch.clamp(goal_normalized_temp[:, :2], -1.0, 1.0)
            
            middle_cp_optimized, langevin_stats = stage2_optimize_control_points(
                model=model,
                map_input=map_input,
                start_normalized=start_normalized_temp,
                goal_normalized=goal_normalized_temp,
                start_pose=start_pose,
                goal_pose=goal_pose,
                stability_cost_map=batch['cost_map'].to(device),
                map_info={
                    'resolution': 0.4,
                    'origin': (-20.0, -20.0, -np.pi),
                    'size': (100, 100, 36)
                },
                device=device,
                num_iterations=int(loss_weights.get('langevin_steps', 10)),
                num_particles=int(loss_weights.get('langevin_gen_particles', 1)),
                temperature_kappa=float(loss_weights.get('langevin_kappa', 1.0)),
                rho_grad=float(loss_weights.get('langevin_rho_grad', 0.05)),
                rho_noise=float(loss_weights.get('langevin_rho_noise', 0.02)),
                rho_max=float(loss_weights.get('langevin_rho_max', 0.1)),
                temperature_min=float(loss_weights.get('langevin_temperature_min', 1e-6)),
                eta_min=float(loss_weights.get('langevin_eta_min', 1e-5)),
                eta_max=float(loss_weights.get('langevin_eta_max', 10.0)),
                noise_scale=float(loss_weights.get('langevin_noise_scale', 0.3)),
                stats_ema_alpha=float(loss_weights.get('langevin_stats_ema_alpha', 0.05)),
                diagnostics=bool(loss_weights.get('langevin_diagnostics', True)),
            )  # (B*k_small, 24, 2)

            optimized_count = middle_cp_optimized.shape[0]
            particles_per_condition = max(1, optimized_count // B)

            # 只有评估或无 buffer fallback 会走这里；训练时优先从 replay buffer 取普通 batch。
            middle_cp = middle_cp_optimized
            map_input = map_input.repeat_interleave(particles_per_condition, dim=0)
            start_pose = start_pose.repeat_interleave(particles_per_condition, dim=0)
            goal_pose = goal_pose.repeat_interleave(particles_per_condition, dim=0)
            if trajectory is not None:
                trajectory = trajectory.repeat_interleave(particles_per_condition, dim=0)
            
            B = middle_cp.shape[0]  # 更新batch size
    
    # 归一化控制点：坐标范围通常在 [-20, 20]，归一化到 [-1, 1]
    middle_cp_normalized = middle_cp / 20.0  # stage1: data batch, stage2: replay/train batch
    middle_cp_normalized = torch.clamp(middle_cp_normalized, -1.0, 1.0)
    
    # 归一化起点终点坐标（转换为4维：x, y, cos(θ), sin(θ)） - 用于条件
    start_normalized = torch.zeros(B, 4, device=device)
    start_normalized[:, :2] = start_pose[:, :2] / 20.0
    start_normalized[:, 2] = torch.cos(start_pose[:, 2])  # cos(θ)
    start_normalized[:, 3] = torch.sin(start_pose[:, 2])  # sin(θ)
    start_normalized[:, :2] = torch.clamp(start_normalized[:, :2], -1.0, 1.0)
    
    goal_normalized = torch.zeros(B, 4, device=device)
    goal_normalized[:, :2] = goal_pose[:, :2] / 20.0
    goal_normalized[:, 2] = torch.cos(goal_pose[:, 2])  # cos(θ)
    goal_normalized[:, 3] = torch.sin(goal_pose[:, 2])  # sin(θ)
    goal_normalized[:, :2] = torch.clamp(goal_normalized[:, :2], -1.0, 1.0)

    # pMF状态包含完整26个控制点；首尾也参与加噪和主损失。
    control_cp_normalized = torch.cat([
        start_normalized[:, None, :2],
        middle_cp_normalized,
        goal_normalized[:, None, :2],
    ], dim=1)
    
    # ===== pixel Mean Flow: 连续时间采样 =====
    # 采样 t 和 r (0 <= r <= t <= 1)
    if hasattr(model, 'module'):
        t = model.module.sample_timesteps(B, device=device)
    else:
        t = model.sample_timesteps(B, device=device)
    t = torch.clamp(t, min=1e-4, max=1.0 - 1e-4) # 避免极端值
    r = torch.rand_like(t) * t
    noise_cp = torch.randn_like(control_cp_normalized)

    # 准备 JVP 需要的 functional 环境
    if hasattr(model, 'module'):
        params = dict(model.module.named_parameters())
        buffers = dict(model.module.named_buffers())
    else:
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())

    # 1. 定义一个 u_fn，它必须能处理 functional 传参
    def u_fn(z_arg, t_arg, r_arg):
        # 保存当前模型状态并设为 eval
        prev_training = model.training
        model.eval()
        
        try:
            # Forward-AD(JVP) 与 CUDA 高效 SDPA(尤其是 efficient/flash kernel)在部分版本不兼容。
            # 这里在 JVP 路径中强制回退到 math kernel，避免：
            # NotImplementedError: forward AD with _scaled_dot_product_efficient_attention
            sdp_ctx = (
                torch.backends.cuda.sdp_kernel(
                    enable_flash=False,
                    enable_mem_efficient=False,
                    enable_math=True
                )
                if z_arg.is_cuda else nullcontext()
            )

            # 使用 functional_call 代替直接 model()
            # 这里的参数顺序必须对应你 forward 的定义
            with sdp_ctx:
                model_output = functional_call(
                    model, 
                    {**params, **buffers}, 
                    (map_input, z_arg, t_arg, r_arg, start_normalized, goal_normalized)
                )
        finally:
            # 恢复之前的状态
            model.train(prev_training)
        
        t_v = t_arg.view(-1, 1, 1)
        # 根据你的 prediction_type 将输出转为 pred_x0
        if prediction_type == 'epsilon':
            p_x0 = (z_arg - t_v * model_output) / (1.0 - t_v + 1e-5)
        elif prediction_type == 'x0':
            p_x0 = model_output
        elif prediction_type == 'v':
            p_x0 = z_arg - t_v * model_output
        
        # 返回平均速度 u = (z_t - x_0) / t
        return (z_arg - p_x0) / (t_v + 1e-5)

    # 2. 准备 JVP 的输入 (Primals) 和 变化率 (Tangents)
    target_v = noise_cp - control_cp_normalized  # 完整26点的 dz/dt 真值

    # noisy_cp 必须在 requires_grad 环境下生成，确保导数链条完整
    with torch.enable_grad():
        t.requires_grad_(True)
        t_v = t.view(-1, 1, 1)
        # 显式重算 noisy_cp 确保它是 t 的函数
        z_t = (1.0 - t_v) * control_cp_normalized + t_v * noise_cp
        
        # Primals: 当前点
        primals = (z_t, t, r)
        # Tangents: 方向。当 t 变化 1 时，z 变化 target_v，r 不变
        tangents = (target_v, torch.ones_like(t), torch.zeros_like(r))
        
        # 3. 计算 JVP (全导数)
        # u_out: 算出的 u
        # du_dt_full: 算出的全导数 du/dt
        u_out, du_dt_full = jvp(u_fn, primals, tangents)
        
        # 对修正项进行幅度裁剪，增加稳定性
        du_dt_full = torch.clamp(du_dt_full, -5.0, 5.0)

    # 4. 构建 V_theta 并计算 Loss
    # V = u + (t - r) * stopgrad(du/dt)
    
    V_theta = u_out + (t.view(-1, 1, 1) - r.view(-1, 1, 1)).detach() * du_dt_full.detach()

    if loss_type == 'v':
        main_loss_all = F.mse_loss(V_theta, target_v, reduction='none')  # (B, 26, 2)
    elif loss_type == 'x0':
        # x0_rec = z_t - t * V_theta
        pred_x0_corrected = z_t - t.view(-1, 1, 1) * V_theta
        main_loss_all = F.mse_loss(pred_x0_corrected, control_cp_normalized, reduction='none')  # (B, 26, 2)
    else:
        # epsilon 空间的 pMF 修正写法：pred_eps_corrected = V_theta + pred_x0_corrected
        # 但推荐统一使用 v-loss 以符合论文实现
        main_loss_all = F.mse_loss(V_theta, target_v, reduction='none')  # (B, 26, 2)
    
    # 阶段1的 target 是 imitation 控制点；阶段2的 target 是 Langevin 粒子。
    main_loss = main_loss_all.mean()
    dummy_loss = main_loss.sum() * 0.0
    tangent_loss = dummy_loss.clone()
    
    if loss_weights.get('tangent', 0.0) > 0.0:
        tangent_loss = compute_tangent_loss(middle_cp_normalized, start_normalized, goal_normalized)
    
    # main_loss 数值稳定性保护（尤其在 stage2 main=0 时避免无关分支污染）
    if isinstance(main_loss, torch.Tensor) and (not torch.isfinite(main_loss)):
        print("⚠ Warning: main_loss is NaN/Inf, fallback to 0 for this batch")
        main_loss = dummy_loss.clone()

    total_loss = loss_weights.get('main', 1.0) * main_loss \
                 + loss_weights.get('tangent', 0.0) * tangent_loss
    
    # 检查损失异常：不中断训练，回退为零损失并跳过本 batch 更新
    if not torch.isfinite(total_loss):
        total_loss_val = total_loss.item() if isinstance(total_loss, torch.Tensor) else float('nan')
        main_loss_val = main_loss.item() if isinstance(main_loss, torch.Tensor) else float('nan')
        print(f"⚠ Warning: total_loss is NaN/Inf ({total_loss_val}), main_loss={main_loss_val}. Skip this batch.")
        total_loss = dummy_loss.clone()
    
    # 返回各项损失用于记录
    loss_dict = {
        'main': main_loss.item(),
        'tangent': tangent_loss.item(),
        **langevin_stats,
    }
    
    # stage2 的训练 batch 来自 replay buffer，统计样本数就是本次 pMF batch size。
    n_samples = condition_count
    
    return total_loss, 0, n_samples, loss_dict

def train_epoch(model, trainingData, optimizer, device, stage_epoch=0, loss_weights=None, current_stage=1, ema_models=None):
    """
    单轮训练函数
    
    Args:
        ema_models: EMA模型列表，用于更新EMA参数（可选）
    """
    model.train()
    total_loss = 0
    total_samples = 0

    if current_stage == 2:
        buffer_size = int(loss_weights.get('langevin_buffer_size', 4096))
        if (
            not hasattr(train_epoch, '_langevin_buffer') or
            train_epoch._langevin_buffer.max_size != buffer_size
        ):
            train_epoch._langevin_buffer = LangevinReplayBuffer(max_size=buffer_size)
            train_epoch._langevin_step = 0
            train_epoch._langevin_last_stats = {}
        langevin_buffer = train_epoch._langevin_buffer
    else:
        langevin_buffer = None
    
    # 累积各项损失
    loss_accumulator = {
        'main': 0, 
        'tangent': 0, 
        'langevin_T': 0,
        'langevin_delta_cost': 0,
        'langevin_G': 0,
        'langevin_eta_grad': 0,
        'langevin_eta_noise': 0,
        'langevin_eta': 0,
        'langevin_eta_at_max': 0,
        'langevin_gmax': 0,
        'langevin_grad_step_norm': 0,
        'langevin_noise_step_norm': 0,
        'langevin_step_snr': 0,
        'langevin_D_before': 0,
        'langevin_D_after': 0,
        'langevin_diversity_ratio': 0,
        'langevin_cov_trace_before': 0,
        'langevin_cov_trace_after': 0,
        'langevin_cov_trace_ratio': 0,
        'langevin_cost_before': 0,
        'langevin_cost_after': 0,
        'langevin_cost_grad_only': 0,
        'langevin_cost_delta': 0,
        'langevin_cost_delta_grad_only': 0,
        'langevin_buffer_size': 0,
    }
    
    pbar = tqdm(trainingData, mininterval=2, desc=f"Training Epoch {stage_epoch} (Stage {current_stage})")
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()

        generation_stats = None
        if current_stage == 2:
            refresh_interval = max(1, int(loss_weights.get('langevin_refresh_interval', 5)))
            train_batch_size = int(loss_weights.get('langevin_train_batch_size', 0))
            if train_batch_size <= 0:
                train_batch_size = int(batch['map'].shape[0])

            need_refresh = (
                len(langevin_buffer) < train_batch_size or
                (train_epoch._langevin_step % refresh_interval == 0)
            )

            if need_refresh:
                map_input_gen = batch['map'].float().to(device)
                start_pose_gen = batch['start_pose'].to(device)
                goal_pose_gen = batch['goal_pose'].to(device)

                start_normalized_gen = torch.zeros(start_pose_gen.shape[0], 4, device=device)
                start_normalized_gen[:, :2] = torch.clamp(start_pose_gen[:, :2] / 20.0, -1.0, 1.0)
                start_normalized_gen[:, 2] = torch.cos(start_pose_gen[:, 2])
                start_normalized_gen[:, 3] = torch.sin(start_pose_gen[:, 2])

                goal_normalized_gen = torch.zeros(goal_pose_gen.shape[0], 4, device=device)
                goal_normalized_gen[:, :2] = torch.clamp(goal_pose_gen[:, :2] / 20.0, -1.0, 1.0)
                goal_normalized_gen[:, 2] = torch.cos(goal_pose_gen[:, 2])
                goal_normalized_gen[:, 3] = torch.sin(goal_pose_gen[:, 2])

                target_middle_cp, generation_stats = stage2_optimize_control_points(
                    model=model,
                    map_input=map_input_gen,
                    start_normalized=start_normalized_gen,
                    goal_normalized=goal_normalized_gen,
                    start_pose=start_pose_gen,
                    goal_pose=goal_pose_gen,
                    stability_cost_map=batch['cost_map'].to(device),
                    map_info={
                        'resolution': 0.4,
                        'origin': (-20.0, -20.0, -np.pi),
                        'size': (100, 100, 36)
                    },
                    device=device,
                    num_iterations=int(loss_weights.get('langevin_steps', 10)),
                    num_particles=int(loss_weights.get('langevin_gen_particles', 1)),
                    temperature_kappa=float(loss_weights.get('langevin_kappa', 1.0)),
                    rho_grad=float(loss_weights.get('langevin_rho_grad', 0.05)),
                    rho_noise=float(loss_weights.get('langevin_rho_noise', 0.02)),
                    rho_max=float(loss_weights.get('langevin_rho_max', 0.1)),
                    temperature_min=float(loss_weights.get('langevin_temperature_min', 1e-6)),
                    eta_min=float(loss_weights.get('langevin_eta_min', 1e-5)),
                    eta_max=float(loss_weights.get('langevin_eta_max', 10.0)),
                    noise_scale=float(loss_weights.get('langevin_noise_scale', 0.3)),
                    stats_ema_alpha=float(loss_weights.get('langevin_stats_ema_alpha', 0.05)),
                    diagnostics=bool(loss_weights.get('langevin_diagnostics', True)),
                )

                gen_particles = max(1, target_middle_cp.shape[0] // batch['map'].shape[0])
                langevin_buffer.add_batch(
                    batch['map'].float().repeat_interleave(gen_particles, dim=0),
                    batch['start_pose'].repeat_interleave(gen_particles, dim=0),
                    batch['goal_pose'].repeat_interleave(gen_particles, dim=0),
                    target_middle_cp
                )
                if torch.device(device).type == 'cuda' and loss_weights.get('langevin_empty_cache_after_refresh', True):
                    torch.cuda.empty_cache()

            train_epoch._langevin_step += 1
            train_batch = langevin_buffer.sample(train_batch_size, device=device)
            loss, _, n_samples, loss_dict = diffusion_loss(
                model, train_batch, device, loss_weights,
                current_stage=current_stage,
                prediction_type=prediction_type
            )
            if generation_stats is not None:
                train_epoch._langevin_last_stats = generation_stats
            if getattr(train_epoch, '_langevin_last_stats', None):
                loss_dict.update(train_epoch._langevin_last_stats)
            loss_dict['langevin_buffer_size'] = float(len(langevin_buffer))
        else:
            loss, _, n_samples, loss_dict = diffusion_loss(
                model, batch, device, loss_weights, 
                current_stage=current_stage,  # 传递当前阶段
                prediction_type=prediction_type
            )

        # 先检查 loss 再统计，避免把 NaN 累积进 epoch 指标
        if not torch.isfinite(loss):
            print("Warning: NaN/Inf loss, skipping batch")
            continue

        total_loss += loss.item()
        total_samples += n_samples
        
        for key in loss_accumulator.keys():
            loss_accumulator[key] += loss_dict[key]
        
        loss.backward()
        
        # 梯度裁剪 - 第二阶段使用适中裁剪（采样链梯度累积大）
        if current_stage == 1:
            clip_value = 1.0  # 阶段1：正常裁剪
        else:
            clip_value = 1.0  # 阶段2：中等裁剪（采样链累积多步梯度）
        
        original_grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=clip_value, norm_type=2
        )

        # 梯度异常保护：跳过本次参数更新，防止污染模型
        if not torch.isfinite(original_grad_norm):
            print("Warning: Non-finite gradient norm, skipping optimizer step for this batch")
            optimizer.zero_grad()
            continue

        was_clipped = original_grad_norm > clip_value
        
        optimizer.step_and_update_lr()
        
        # 更新EMA（如果启用）
        if ema_models is not None:
            for ema_m in ema_models:
                ema_m.update(model)
        
        # 更新进度条
        grad_info = f'{original_grad_norm:.2f}→{clip_value}' if was_clipped else f'{original_grad_norm:.2f}'
        
        # 根据当前阶段显示不同的信息
        if current_stage == 1:
            # 阶段1：显示主损失、平滑性
            pbar.set_postfix({
                'Total': f'{loss.item():.5f}',
                'Main': f'{loss_dict["main"]:.5f}',
                'Tangent': f'{loss_dict["tangent"]:.4f}',
                'GradNorm': grad_info
            })
        else:
            # 阶段2：显示主损失与物理约束损失
            pbar.set_postfix({
                'Total': f'{loss.item():.5f}',
                'Main': f'{loss_dict["main"]:.5f}',
                'T': f'{loss_dict["langevin_T"]:.2e}',
                'Eta': f'{loss_dict["langevin_eta"]:.2e}',
                'SNR': f'{loss_dict["langevin_step_snr"]:.2f}',
                'dC': f'{loss_dict["langevin_cost_delta"]:.1e}',
                'Div': f'{loss_dict["langevin_diversity_ratio"]:.2f}',
                'GradNorm': grad_info
            })
    
    avg_loss = total_loss / len(trainingData) if len(trainingData) > 0 else 0
    
    # 计算各项平均损失
    avg_loss_dict = {k: v / len(trainingData) for k, v in loss_accumulator.items()}
    
    return avg_loss, 0, total_samples, avg_loss_dict


def eval_epoch(model, validationData, device, loss_weights=None, current_stage=1):
    """
    单轮评估函数
    
    """
    model.eval()
    total_loss = 0
    total_samples = 0
    
    # 累积各项损失
    loss_accumulator = {
        'main': 0, 
        'tangent': 0, 
        'langevin_T': 0,
        'langevin_delta_cost': 0,
        'langevin_G': 0,
        'langevin_eta_grad': 0,
        'langevin_eta_noise': 0,
        'langevin_eta': 0,
        'langevin_eta_at_max': 0,
        'langevin_gmax': 0,
        'langevin_grad_step_norm': 0,
        'langevin_noise_step_norm': 0,
        'langevin_step_snr': 0,
        'langevin_D_before': 0,
        'langevin_D_after': 0,
        'langevin_diversity_ratio': 0,
        'langevin_cov_trace_before': 0,
        'langevin_cov_trace_after': 0,
        'langevin_cov_trace_ratio': 0,
        'langevin_cost_before': 0,
        'langevin_cost_after': 0,
        'langevin_cost_grad_only': 0,
        'langevin_cost_delta': 0,
        'langevin_cost_delta_grad_only': 0,
    }

    def evaluate_stage2_cost(batch):
        from grad_optimizer import cost_on_dense_trajectory

        if 'cost_map' not in batch:
            raise ValueError("Stage 2 validation requires batch['cost_map'] to compute cost.")

        effective_loss_weights = loss_weights or {}
        map_input = batch['map'].float().to(device)
        start_pose = batch['start_pose'].to(device)
        goal_pose = batch['goal_pose'].to(device)
        B = map_input.shape[0]

        start_normalized = torch.zeros(B, 4, device=device)
        start_normalized[:, :2] = torch.clamp(start_pose[:, :2] / 20.0, -1.0, 1.0)
        start_normalized[:, 2] = torch.cos(start_pose[:, 2])
        start_normalized[:, 3] = torch.sin(start_pose[:, 2])

        goal_normalized = torch.zeros(B, 4, device=device)
        goal_normalized[:, :2] = torch.clamp(goal_pose[:, :2] / 20.0, -1.0, 1.0)
        goal_normalized[:, 2] = torch.cos(goal_pose[:, 2])
        goal_normalized[:, 3] = torch.sin(goal_pose[:, 2])

        val_samples = max(1, int(effective_loss_weights.get('stage2_val_samples', 1)))
        sample_steps = max(1, int(effective_loss_weights.get('stage2_val_sample_steps', 3)))
        sample_model = model.module if hasattr(model, 'module') else model
        sampled_traj = sample_model.sample(
            map_input,
            start_normalized,
            goal_normalized,
            num_samples=val_samples,
            num_steps=sample_steps,
            solver='pmf_refined',
            reconstruct_trajectory=True,
            num_traj_points=100
        )

        start_pose_expanded = start_pose.repeat_interleave(val_samples, dim=0)
        goal_pose_expanded = goal_pose.repeat_interleave(val_samples, dim=0)
        map_info = {
            'resolution': 0.4,
            'origin': (-20.0, -20.0, -np.pi),
            'size': (100, 100, 36)
        }
        cost = cost_on_dense_trajectory(
            sampled_traj,
            start_pose_expanded,
            goal_pose_expanded,
            batch['cost_map'].to(device),
            map_info,
            device
        )

        loss_dict = {key: 0.0 for key in loss_accumulator}
        loss_dict['main'] = cost.item()
        loss_dict['langevin_cost_before'] = cost.item()
        loss_dict['langevin_cost_after'] = cost.item()
        return cost, 0, B, loss_dict
    
    with torch.no_grad():
        for batch in tqdm(validationData, mininterval=2, desc="Validation"):
            if current_stage == 2:
                loss, _, n_samples, loss_dict = evaluate_stage2_cost(batch)
            else:
                loss, _, n_samples, loss_dict = diffusion_loss(
                    model, batch, device, loss_weights,
                    current_stage=current_stage,
                    prediction_type=prediction_type
                )
                
            total_loss += loss.item()
            total_samples += n_samples
            
            for key in loss_accumulator:
                loss_accumulator[key] += loss_dict[key]
    
    avg_loss = total_loss / len(validationData) if len(validationData) > 0 else 0
    
    # 计算各项平均损失
    avg_loss_dict = {k: v / len(validationData) for k, v in loss_accumulator.items()}
    
    return avg_loss, 0, total_samples, avg_loss_dict


def check_data_folders(folder):
    """检查数据文件夹结构"""
    assert osp.isdir(osp.join(folder, 'train')), "Cannot find training data"  # 检查train子文件夹是否存在
    assert osp.isdir(osp.join(folder, 'val')), "Cannot find validation data"  # 检查val子文件夹是否存在

def load_checkpoint(model, checkpoint_path, device):
    """加载检查点"""
    if not osp.exists(checkpoint_path):
        raise ValueError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError("Invalid checkpoint format")
    
    epoch = checkpoint.get('epoch', -1)
    print(f"Loaded checkpoint from epoch {epoch}")
    return checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', help="Batch size per GPU", required=True, type=int)
    parser.add_argument('--dataFolder', help="Directory with training and validation data", default=None)
    parser.add_argument('--fileDir', help="Directory to save training data")
    parser.add_argument('--resume', help="Path to checkpoint to resume training", default=None)
    parser.add_argument('--stage', help="Training stage to start from (1 or 2)", type=int, default=1, choices=[1, 2])
    parser.add_argument('--stage1_epochs', help="Number of epochs for stage 1", type=int, default=200)
    parser.add_argument('--stage2_epochs', help="Number of epochs for stage 2", type=int, default=300)
    # parser.add_argument('--prediction_type', help="Model prediction type", type=str, default='v', choices=['epsilon', 'x0', 'v'])
    args = parser.parse_args()

    # 检查数据文件夹
    dataFolder = args.dataFolder
    if not osp.isdir(dataFolder):
        raise ValueError("Please provide a valid data folder")
    
    check_data_folders(dataFolder)
    
    if not osp.isdir(args.fileDir):
        raise ValueError("Please provide a valid file directory to save training data")

    # 设备配置
    device = 'cpu'
    if torch.cuda.is_available():
        print("Using GPU....")
        device = torch.device('cuda')

    batch_size = args.batchSize
    if torch.cuda.device_count() > 1:
        batch_size = batch_size * torch.cuda.device_count()
    print(f"Total batch size : {batch_size}")

    torch_seed = np.random.randint(low=0, high=1000)  # 生成随机种子
    torch.manual_seed(torch_seed)  # 设置PyTorch随机种子，确保结果可复现
    
    # =================== 模型配置 ===================
    # 根据论文发现：模型容量不足时，只有x_pred能训练，epsilon/v_pred会失败
    # 因此增加模型容量以支持所有预测类型
    
    # 从命令行参数获取配置
    # prediction_type = args.prediction_type
    
    prediction_type = 'x0'  # 'epsilon', 'x0', or 'v'
    loss_type = 'v'  # 'epsilon', 'x0', or 'v'
    
    model_args = dict(
        n_layers=6,  # 6 -> 12 (增加深度)
        n_heads=8,   # 8 -> 12 (增加注意力头)
        d_k=192,      # 192 -> 256
        d_v=96,      # 96 -> 128
        d_model=512,  # 512 -> 768 (增加模型维度)
        d_inner=1024, # 2048 -> 3072 (4x d_model，标准Transformer比例)
        pad_idx=None,
        n_position=15*15,
        dropout=0.1,
        train_shape=[12, 12],
        n_path_steps=24,  # 24个中间点；pMF状态和网络输出为完整26点
        diffusion_steps=50,
        prediction_type=prediction_type,  # 'epsilon', 'x0', or 'v' - 模型输出什么
        loss_type=loss_type  # 与prediction_type保持一致
        # 
        # 【预测类型】
        #     --prediction_type=x0: 直接预测，容量要求低但精度受限
        #     --prediction_type=epsilon: DDPM标准，稳定
        #     --prediction_type=v: Stable Diffusion方案，最推荐
    )
    
    print("\n" + "="*70)
    print("模型配置")
    print("="*70)
    print(f"训练方法: Standard Diffusion (DDIM)")
    print(f"预测类型: {prediction_type}")
    print(f"模型层数: {model_args['n_layers']}")
    print(f"注意力头: {model_args['n_heads']}")
    print(f"模型维度: {model_args['d_model']}")
    print("="*70 + "\n")

    # 初始化扩散模型
    model = PathDiffusionTransformer(**model_args)

    if torch.cuda.device_count() > 1:  # 检查是否有多个GPU
        print("Using ", torch.cuda.device_count(), "GPUs")  # 打印使用的GPU数量
        model = nn.DataParallel(model)  # 使用DataParallel包装模型，实现数据并行
    model.to(device=device)  # 将模型移动到指定设备(CPU或GPU)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # =================== 数据加载 ===================
    # env_list = ["env000004", "env000005"]
    # env_list = ["env000008"]
    # env_list = ["env000008", "env000009"]
    # env_list = ["env000010", "env000011"]
    # env_list = ["env000012", "env000013"]
    env_list = ["env000012", "env000013",
                "env000012_optimized", "env000013_optimized"]
    
    # =================== EMA配置 ===================
    use_ema = False  # 是否使用EMA
    ema_decays = [0.99, 0.999]  # EMA衰减率列表，可以设置多个值如[0.999, 0.9999, 0.99999]
    ema_models = []
    
    if use_ema:
        if not isinstance(ema_decays, (list, tuple)) or len(ema_decays) == 0:
            raise ValueError("ema_decays must be a non-empty list.")
        
        # 获取要包装的模型（处理DataParallel情况）
        model_to_ema = model.module if isinstance(model, nn.DataParallel) else model
        
        for decay in ema_decays:
            ema_m = ModelEmaV2(model_to_ema, decay=float(decay))
            ema_m.to(device)
            ema_models.append(ema_m)
        
        print(f"✓ EMA启用 (decays={ema_decays})")
    
    # =================== 两阶段训练配置 ===================
    stage1_config = {
        'epochs': args.stage1_epochs,
        'lr_mul': 1e-1,  # 学习率倍增器
        'loss_weights': {
            'main': 1e-2,          # 主预测损失
            'tangent': 0e3,   # 切线约束损失（确保起点和终点方向一致）
        }
    }
    
    # 阶段2配置：Langevin 分布迁移
    # 目标分布 q_T(x|c) ∝ exp[-C(x,c) / T]。
    # main loss 的目标只来自 Langevin 迁移控制点；
    # cost 只通过 Langevin 内循环塑造目标样本，不再作为直接最小化项压到网络上。
    stage2_config = {
        'epochs': args.stage2_epochs,
        'lr_mul': 1e-3,  # 小学习率微调（不是从头训练！）
        'loss_weights': {
            'main': 1e-2,        # 保留主损失作为正则化（不能为0！）
            'tangent': 0e-2,   # 切线约束损失（确保起点和终点方向一致）
            'langevin_gen_particles': 1,    # k_small：每次 refresh 每个条件生成的粒子数
            'langevin_refresh_interval': 5, # 每隔多少个 pMF step 刷新一批 Langevin 样本
            'langevin_train_batch_size': 0, # 0 表示使用当前 dataloader batch size
            'langevin_buffer_size': 4096,   # detached CPU replay buffer 容量
            'langevin_empty_cache_after_refresh': True,
            'langevin_stats_ema_alpha': 0.05,
            'langevin_kappa': 1.0,          # T = (P90(C)-P10(C)) / kappa，先从1开始
            'langevin_steps': 10,           # M：初期建议5~10，稳定后再加到20
            'langevin_rho_grad': 0.05,      # eta_grad = rho_grad * D / G
            'langevin_rho_noise': 0.02,     # eta_noise = (rho_noise * D)^2 / (2*T*d)
            'langevin_rho_max': 0.1,        # g_max = rho_max * D / eta
            'langevin_temperature_min': 1e-6,
            'langevin_eta_min': 1e-5,
            'langevin_eta_max': 10.0,       # 0.05 会让 cost 漂移过弱；由 g_max 控制单步最大位移
            'langevin_noise_scale': 0.3,    # 诊断显示 noise=1 容易抵消很弱的梯度漂移
            'langevin_diagnostics': True,   # 额外记录纯梯度链、步长 SNR 等指标
            'stage2_val_samples': 1,        # 验证时每个条件采样几条轨迹来计算 cost
            'stage2_val_sample_steps': 3,   # 验证采样的 pMF refine 步数，越大越慢但更稳定
        }
    }
    
    # 初始化当前阶段（在数据加载前）
    current_stage = args.stage
    
    # 第二阶段 Langevin 迁移需要 cost map
    if current_stage == 1:
        loss_weights = stage1_config['loss_weights']
    else:
        loss_weights = stage2_config['loss_weights']

    if current_stage == 2 and loss_weights.get('main', 0.0) <= 0.0:
        print("⚠ Warning: Stage 2 main loss weight <= 0; pMF distillation may not learn the Langevin targets.")
    
    compute_stability = (
        current_stage == 2 and loss_weights.get('langevin_steps', 0) > 0
    )
    
    trainDataset = UnevenPathDataLoader(
        env_list=env_list,
        dataFolder=osp.join(dataFolder, 'train'),
        compute_stability_map=compute_stability
    )
    trainingData = DataLoader(
        trainDataset, 
        num_workers=15, 
        collate_fn=PaddedSequence, 
        batch_size=batch_size,
        shuffle=True
    )

    valDataset = UnevenPathDataLoader(
        env_list=env_list,
        dataFolder=osp.join(dataFolder, 'val'),
        compute_stability_map=compute_stability
    )
    validationData = DataLoader(
        valDataset, 
        num_workers=5, 
        collate_fn=PaddedSequence, 
        batch_size=batch_size
    )
    
    # =================== 继续两阶段训练配置 ===================
    trainDataFolder = args.fileDir
    
    # 总训练轮数
    n_epochs = stage1_config['epochs'] + stage2_config['epochs']
    
    # 初始化学习率（当前阶段已在数据加载前初始化）
    if current_stage == 1:
        current_lr_mul = stage1_config['lr_mul']
    else:
        current_lr_mul = stage2_config['lr_mul']
    
    print("\n=== 两阶段训练配置 ===")
    print(f"\n阶段1 ({stage1_config['epochs']} epochs, lr_mul={stage1_config['lr_mul']}):")
    for key, weight in stage1_config['loss_weights'].items():
        print(f"  {key}: {weight}")
    print(f"\n阶段2 ({stage2_config['epochs']} epochs, lr_mul={stage2_config['lr_mul']}):")
    for key, weight in stage2_config['loss_weights'].items():
        print(f"  {key}: {weight}")
    print(f"\n当前起始阶段: {current_stage}")
    print()
    
    # 保存模型配置
    config = {
        'model_args': model_args,
        'stage1_config': stage1_config,
        'stage2_config': stage2_config,
        'total_epochs': n_epochs
    }
    json.dump(
        config,
        open(osp.join(trainDataFolder, 'model_params.json'), 'w'),
        sort_keys=True,
        indent=4
    )
    
    writer = SummaryWriter(log_dir=trainDataFolder)
    
    # 根据阶段配置参数冻结
    if current_stage == 1:
        # 阶段1：基础 pMF 训练
        if isinstance(model, nn.DataParallel):
            model.module.unfreeze_for_stage1()
        else:
            model.unfreeze_for_stage1()
    else:
        # 阶段2：按照模型内部策略设置可训练参数
        if isinstance(model, nn.DataParallel):
            model.module.freeze_for_stage2()
        else:
            model.freeze_for_stage2()
    
    # 优化器（只优化requires_grad=True的参数）
    if isinstance(model, nn.DataParallel):
        trainable_params = model.module.get_trainable_parameters()
    else:
        trainable_params = model.get_trainable_parameters()
    
    optimizer = Optim.ScheduledOptim(
        optim.AdamW(
            trainable_params,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        ),
        lr_mul=current_lr_mul,
        d_model=512,
        n_warmup_steps=1000
    )
    
    print(f"✓ 阶段{current_stage}：可训练参数数量 = {sum(p.numel() for p in trainable_params)}")

    
    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    stage1_best_loss = float('inf')
    
    if args.resume:
        checkpoint = load_checkpoint(model, args.resume, device)
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        stage1_best_loss = checkpoint.get('stage1_best_loss', float('inf'))
        
        # 恢复阶段信息
        saved_stage = checkpoint.get('stage', 1)
        saved_epoch = checkpoint.get('epoch', -1)
        # 注意：当前保存的 epoch 是“阶段内索引”（stage_epoch），
        # 恢复训练循环需要“全局索引”（global epoch）。
        if saved_stage == 2 and saved_epoch >= 0:
            saved_epoch_global = stage1_config['epochs'] + saved_epoch
        else:
            saved_epoch_global = saved_epoch
        
        # 尝试加载EMA模型状态（如果启用EMA且checkpoint中包含相同数量的EMA模型）
        if use_ema and len(ema_models) > 0:
            # 方法1：尝试从同一个checkpoint加载（旧版本，所有EMA在一个文件中）
            if 'ema_state_dicts' in checkpoint and 'ema_decays' in checkpoint:
                ema_state_dicts = checkpoint['ema_state_dicts']
                checkpoint_decays = checkpoint['ema_decays']
                
                if len(ema_state_dicts) == len(ema_models):
                    for i, (ema_m, state_dict) in enumerate(zip(ema_models, ema_state_dicts)):
                        try:
                            ema_m.module.load_state_dict(state_dict)
                            print(f"✓ Loaded EMA model {i} (decay={checkpoint_decays[i]}) from checkpoint")
                        except Exception as e:
                            print(f"Warning: Failed to load EMA model {i}: {e}")
                else:
                    print(f"Warning: EMA model count mismatch (checkpoint: {len(ema_state_dicts)}, current: {len(ema_models)})")
            
            # 方法2：尝试从分离的EMA checkpoint加载（新版本）
            else:
                stage_prefix = f'stage{saved_stage}_best'
                for i, (ema_m, decay) in enumerate(zip(ema_models, ema_decays)):
                    ema_checkpoint_path = osp.join(trainDataFolder, f'{stage_prefix}_ema_{decay}.pth')
                    if osp.exists(ema_checkpoint_path):
                        try:
                            ema_checkpoint = torch.load(ema_checkpoint_path, map_location=device)
                            ema_m.module.load_state_dict(ema_checkpoint['model_state_dict'])
                            print(f"✓ Loaded EMA model (decay={decay}) from {osp.basename(ema_checkpoint_path)}")
                        except Exception as e:
                            print(f"Warning: Failed to load EMA model (decay={decay}): {e}")
                    else:
                        print(f"Note: EMA checkpoint not found: {osp.basename(ema_checkpoint_path)}")
        
        if args.stage == 1 and saved_stage == 2:
            print("Warning: Checkpoint is from stage 2, but --stage=1 specified")
            print("   Continuing from stage 2 instead")
            current_stage = 2
            loss_weights = stage2_config['loss_weights']
            current_lr_mul = stage2_config['lr_mul']
            # 继续第二阶段的训练
            start_epoch = saved_epoch_global + 1
            
        elif args.stage == 2 and saved_stage == 1:
            print("Loading stage 1 checkpoint, starting stage 2 from epoch 0")
            current_stage = 2
            loss_weights = stage2_config['loss_weights']
            current_lr_mul = stage2_config['lr_mul']
            # 从第二阶段的起始位置开始（stage1_epochs）
            start_epoch = stage1_config['epochs']
            stage1_best_loss = best_val_loss  # 保存第一阶段的最佳损失
            best_val_loss = float('inf')  # 重置第二阶段的最佳损失
            print(f"✓ Stage 1 checkpoint loaded (val_loss={stage1_best_loss:.4f})")
            print(f"✓ Starting Stage 2 from epoch 0")
            print(f"✓ Model parameters initialized from Stage 1 checkpoint")
            
        elif args.stage == 1 and saved_stage == 1:
            # 继续第一阶段的训练
            current_stage = 1
            loss_weights = stage1_config['loss_weights']
            current_lr_mul = stage1_config['lr_mul']
            start_epoch = saved_epoch_global + 1
            
        else:  # args.stage == 2 and saved_stage == 2
            # 继续第二阶段的训练
            current_stage = 2
            loss_weights = stage2_config['loss_weights']
            current_lr_mul = stage2_config['lr_mul']
            start_epoch = saved_epoch_global + 1
        
        # 根据恢复的阶段更新优化器学习率
        optimizer.lr_mul = current_lr_mul
        
        # 如果从stage 1切换到stage 2，重新初始化优化器（不加载旧的优化器状态）
        if args.stage == 2 and saved_stage == 1:
            print("Resetting optimizer for stage 2 (fresh start)")
            # 不加载旧的优化器状态，使用全新的优化器
        elif 'optimizer_state_dict' in checkpoint:
            try:
                optimizer._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'n_steps' in checkpoint:
                    optimizer.n_steps = checkpoint['n_steps']
                print(f"✓ Loaded optimizer state (stage {current_stage}, lr_mul={current_lr_mul})")
            except Exception as e:
                print(f"Warning: Failed to load optimizer state: {e}")
    
    # =================== 训练循环 ===================
    train_losses = []
    val_losses = []
    # 为每个阶段定义独立的保存路径
    stage1_best_path = osp.join(trainDataFolder, 'stage1_best_model.pth')
    stage2_best_path = osp.join(trainDataFolder, 'stage2_best_model.pth')
    
    # 为每个EMA模型跟踪最佳验证损失
    ema_best_val_losses = [float('inf')] * len(ema_models) if use_ema else []
    
    print(f"\n{'='*60}")
    print(f"Starting training from epoch {start_epoch}")
    print(f"Total epochs: {n_epochs}")
    print(f"Starting stage: {current_stage}")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, n_epochs):
        # 检查是否需要切换到阶段2
        if current_stage == 1 and epoch >= stage1_config['epochs']:
            print(f"\n{'='*60}")
            print(f"Switching to Stage 2 at epoch {epoch}")
            print(f"{'='*60}")
            current_stage = 2
            loss_weights = stage2_config['loss_weights']
            
            # 记录阶段1的最佳损失
            stage1_best_loss = best_val_loss
            print(f"✓ Stage 1 best validation loss: {stage1_best_loss:.4f}")
            
            # 加载阶段1的最优模型参数
            if osp.exists(stage1_best_path):
                print(f"✓ Loading Stage 1 best model from {osp.basename(stage1_best_path)}...")
                checkpoint = torch.load(stage1_best_path, map_location=device)
                if isinstance(model, nn.DataParallel):
                    model.module.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ Loaded Stage 1 best model successfully")
            else:
                print(f"⚠ Warning: Stage 1 best model not found, continuing with current parameters")
            
            # 加载阶段1的最优EMA模型（如果启用）
            if use_ema and len(ema_models) > 0:
                print(f"✓ Loading Stage 1 best EMA models...")
                for i, (ema_m, decay) in enumerate(zip(ema_models, ema_decays)):
                    ema_checkpoint_path = osp.join(trainDataFolder, f'stage1_best_ema_{decay}.pth')
                    if osp.exists(ema_checkpoint_path):
                        try:
                            ema_checkpoint = torch.load(ema_checkpoint_path, map_location=device)
                            ema_m.module.load_state_dict(ema_checkpoint['model_state_dict'])
                            print(f"  ✓ Loaded EMA model (decay={decay}) from {osp.basename(ema_checkpoint_path)}")
                        except Exception as e:
                            print(f"  Warning: Failed to load EMA model (decay={decay}): {e}")
                    else:
                        print(f"  Note: EMA checkpoint not found: {osp.basename(ema_checkpoint_path)}, will use current EMA state")
            
            # 阶段2：按照模型内部策略设置可训练参数
            if isinstance(model, nn.DataParallel):
                model.module.freeze_for_stage2()
                trainable_params = model.module.get_trainable_parameters()
            else:
                model.freeze_for_stage2()
                trainable_params = model.get_trainable_parameters()
            
            trainable_count = sum(p.numel() for p in trainable_params)
            print(f"✓ 阶段2训练参数量: {trainable_count:,}")
            
            # 更新优化器
            old_lr_mul = optimizer.lr_mul
            optimizer = Optim.ScheduledOptim(
                optim.AdamW(
                    trainable_params,
                    betas=(0.95, 0.999),
                    eps=1e-8,
                    weight_decay=0.01
                ),
                lr_mul=stage2_config['lr_mul'],
                d_model=512,
                n_warmup_steps=50
            )
            print(f"✓ Reset optimizer with new learning rate: {old_lr_mul} → {stage2_config['lr_mul']}")
            print(
                "✓ Enabled Langevin migration: "
                f"k_small={stage2_config['loss_weights']['langevin_gen_particles']}, "
                f"buffer={stage2_config['loss_weights']['langevin_buffer_size']}, "
                f"refresh={stage2_config['loss_weights']['langevin_refresh_interval']}, "
                f"M={stage2_config['loss_weights']['langevin_steps']}, "
                f"kappa={stage2_config['loss_weights']['langevin_kappa']}"
            )
            print()

            
            # 重置最佳验证损失，用于阶段2
            best_val_loss = float('inf')
            
            # 重置EMA模型的最佳验证损失（开始新阶段）
            if use_ema:
                ema_best_val_losses = [float('inf')] * len(ema_models)
                print(f"✓ Reset EMA best validation losses for Stage 2")
            
            # 重新创建EMA模型（基于加载的最佳模型）
            if use_ema and len(ema_models) > 0:
                print(f"✓ Re-initializing EMA models based on Stage 1 best model...")
                model_to_ema = model.module if isinstance(model, nn.DataParallel) else model
                
                # 重新创建EMA模型（会基于当前模型初始化）
                new_ema_models = []
                for decay in ema_decays:
                    ema_m = ModelEmaV2(model_to_ema, decay=float(decay))
                    ema_m.to(device)
                    new_ema_models.append(ema_m)
                
                # 尝试加载阶段1的最优EMA模型状态
                for i, (ema_m, decay) in enumerate(zip(new_ema_models, ema_decays)):
                    ema_checkpoint_path = osp.join(trainDataFolder, f'stage1_best_ema_{decay}.pth')
                    if osp.exists(ema_checkpoint_path):
                        try:
                            ema_checkpoint = torch.load(ema_checkpoint_path, map_location=device)
                            ema_m.module.load_state_dict(ema_checkpoint['model_state_dict'])
                            print(f"  ✓ Loaded Stage 1 best EMA state (decay={decay})")
                        except Exception as e:
                            print(f"  Note: Using fresh EMA initialization for decay={decay}: {e}")
                
                ema_models = new_ema_models
            
            # 重新创建数据加载器以启用stability计算
            print("正在重新加载数据集以启用stability map计算...")
            trainDataset = UnevenPathDataLoader(
                env_list=env_list,
                dataFolder=osp.join(dataFolder, 'train'),
                compute_stability_map=True
            )
            trainingData = DataLoader(
                trainDataset, 
                num_workers=15, 
                collate_fn=PaddedSequence, 
                batch_size=batch_size,
                shuffle=True
            )
            
            valDataset = UnevenPathDataLoader(
                env_list=env_list,
                dataFolder=osp.join(dataFolder, 'val'),
                compute_stability_map=True
            )
            validationData = DataLoader(
                valDataset, 
                num_workers=5, 
                collate_fn=PaddedSequence, 
                batch_size=batch_size
            )
            print("✓ 数据集重新加载完成（已启用stability map计算）\n")
        
        # 计算当前阶段内的相对epoch
        if current_stage == 1:
            stage_epoch = epoch
        else:
            stage_epoch = epoch - stage1_config['epochs']
        
        # 训练
        train_loss, _, _, train_loss_dict = train_epoch(
            model, trainingData, optimizer, device, stage_epoch, loss_weights, current_stage,
            ema_models=ema_models if use_ema else None
        )
        
        # 验证（使用主模型）
        val_loss, _, _, val_loss_dict = eval_epoch(
            model, validationData, device, loss_weights, current_stage=current_stage
        )
        
        # 验证EMA模型
        ema_val_losses = []
        ema_val_loss_dicts = []
        if use_ema and len(ema_models) > 0:
            for i, ema_m in enumerate(ema_models):
                ema_val_loss, _, _, ema_val_loss_dict = eval_epoch(
                    ema_m.module, validationData, device, loss_weights, current_stage=current_stage
                )
                ema_val_losses.append(ema_val_loss)
                ema_val_loss_dicts.append(ema_val_loss_dict)
        
        # 记录
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"\n[Stage {current_stage}] Epoch {stage_epoch}:")
        print(f"  Train Loss: {train_loss:.9f}")
        print(f"    - Main: {train_loss_dict['main']:.6f}")
        print(f"    - Tangent: {train_loss_dict['tangent']:.6f}")
        if current_stage == 2:
            print(
                "    - Langevin: "
                f"T={train_loss_dict.get('langevin_T', 0.0):.3e}, "
                f"eta={train_loss_dict.get('langevin_eta', 0.0):.3e}, "
                f"SNR={train_loss_dict.get('langevin_step_snr', 0.0):.2f}, "
                f"gmax={train_loss_dict.get('langevin_gmax', 0.0):.3e}, "
                f"cost {train_loss_dict.get('langevin_cost_before', 0.0):.3e}"
                f"→{train_loss_dict.get('langevin_cost_after', 0.0):.3e}, "
                f"gd-only={train_loss_dict.get('langevin_cost_grad_only', 0.0):.3e}, "
                f"ΔC={train_loss_dict.get('langevin_cost_delta', 0.0):.2e}, "
                f"D ratio={train_loss_dict.get('langevin_diversity_ratio', 0.0):.3f}, "
                f"buffer={train_loss_dict.get('langevin_buffer_size', 0.0):.0f}"
            )
            print(
                "    - Langevin step scale: "
                f"G={train_loss_dict.get('langevin_G', 0.0):.3e}, "
                f"grad_step={train_loss_dict.get('langevin_grad_step_norm', 0.0):.3e}, "
                f"noise_step={train_loss_dict.get('langevin_noise_step_norm', 0.0):.3e}, "
                f"eta_grad={train_loss_dict.get('langevin_eta_grad', 0.0):.3e}, "
                f"eta_noise={train_loss_dict.get('langevin_eta_noise', 0.0):.3e}, "
                f"eta_at_max={train_loss_dict.get('langevin_eta_at_max', 0.0):.0f}"
            )
        if current_stage == 2:
            print(f"  Val Cost:   {val_loss:.9f}")
            print(f"    - Sample Cost: {val_loss_dict['main']:.6f}")
        else:
            print(f"  Val Loss:   {val_loss:.9f}")
            print(f"    - Main: {val_loss_dict['main']:.6f}")
            print(f"    - Tangent: {val_loss_dict['tangent']:.6f}")
        
        # 打印EMA验证损失
        if use_ema and len(ema_val_losses) > 0:
            for i, (ema_val_loss, decay) in enumerate(zip(ema_val_losses, ema_decays)):
                ema_metric_name = "EMA Val Cost" if current_stage == 2 else "EMA Val Loss"
                print(f"  {ema_metric_name} (decay={decay}): {ema_val_loss:.6f}")
        
        # TensorBoard - 总损失（使用阶段内epoch）
        writer.add_scalar('Loss/train', train_loss, stage_epoch)
        writer.add_scalar('Loss/val', val_loss, stage_epoch)
        if current_stage == 2:
            writer.add_scalar('Cost/val_sample_cost', val_loss, stage_epoch)
        
        # TensorBoard - 各项损失
        writer.add_scalar('Loss/train_main', train_loss_dict['main'], stage_epoch)
        writer.add_scalar('Loss/train_tangent', train_loss_dict['tangent'], stage_epoch)

        for key in [
            'langevin_T',
            'langevin_delta_cost',
            'langevin_G',
            'langevin_eta_grad',
            'langevin_eta_noise',
            'langevin_eta',
            'langevin_eta_at_max',
            'langevin_gmax',
            'langevin_grad_step_norm',
            'langevin_noise_step_norm',
            'langevin_step_snr',
            'langevin_D_before',
            'langevin_D_after',
            'langevin_diversity_ratio',
            'langevin_cov_trace_before',
            'langevin_cov_trace_after',
            'langevin_cov_trace_ratio',
            'langevin_cost_before',
            'langevin_cost_after',
            'langevin_cost_grad_only',
            'langevin_cost_delta',
            'langevin_cost_delta_grad_only',
            'langevin_buffer_size',
        ]:
            writer.add_scalar(f'Langevin/train_{key}', train_loss_dict.get(key, 0.0), stage_epoch)
        
        writer.add_scalar('Loss/val_main', val_loss_dict['main'], stage_epoch)
        writer.add_scalar('Loss/val_tangent', val_loss_dict['tangent'], stage_epoch)
        
        # TensorBoard - EMA验证损失
        if use_ema and len(ema_val_losses) > 0:
            for i, (ema_val_loss, decay) in enumerate(zip(ema_val_losses, ema_decays)):
                writer.add_scalar(f'Loss/ema_val_{decay}', ema_val_loss, stage_epoch)
        
        writer.add_scalar('LR', optimizer._optimizer.param_groups[0]['lr'], stage_epoch)
        
        current_val_metric = val_loss
        if current_val_metric < best_val_loss:
            best_val_loss = current_val_metric
            
            if isinstance(model, nn.DataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            # 根据当前阶段选择保存路径
            best_model_path = stage1_best_path if current_stage == 1 else stage2_best_path
            
            # 保存标准模型
            checkpoint = {
                'epoch': stage_epoch,
                'stage': current_stage,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer._optimizer.state_dict(),
                'n_steps': optimizer.n_steps,
                'train_loss': train_loss,
                'val_loss': current_val_metric,
                'val_metric_name': 'cost' if current_stage == 2 else 'loss',
                'stage1_best_loss': stage1_best_loss,
                'torch_seed': torch_seed
            }
            
            torch.save(checkpoint, best_model_path)
            metric_name = 'val_cost' if current_stage == 2 else 'val_loss'
            print(
                f"  ✓ Saved best model for Stage {current_stage} to "
                f"{osp.basename(best_model_path)} ({metric_name}={current_val_metric:.6f})"
            )
        
        # 分别保存每个EMA模型（基于各自的训练指标在Stage 2）
        if use_ema and len(ema_models) > 0:
            for i, (ema_m, decay, ema_val_loss) in enumerate(zip(ema_models, ema_decays, ema_val_losses)):
                ema_val_metric = ema_val_loss
                if ema_val_metric < ema_best_val_losses[i]:
                    ema_best_val_losses[i] = ema_val_metric
                    
                    # 获取EMA模型的state_dict
                    ema_state_dict = ema_m.module.state_dict()
                    
                    # 构造EMA模型文件名
                    stage_prefix = f'stage{current_stage}_best'
                    ema_model_filename = f'{stage_prefix}_ema_{decay}.pth'
                    ema_model_path = osp.join(trainDataFolder, ema_model_filename)
                    
                    # 保存EMA模型
                    ema_checkpoint = {
                        'epoch': stage_epoch,
                        'stage': current_stage,
                        'model_state_dict': ema_state_dict,
                        'ema_decay': decay,
                        'train_loss': train_loss,
                        'val_loss': ema_val_metric,
                        'val_metric_name': 'cost' if current_stage == 2 else 'loss',
                        'stage1_best_loss': stage1_best_loss,
                        'torch_seed': torch_seed
                    }
                    torch.save(ema_checkpoint, ema_model_path)
                    metric_name = 'val_cost' if current_stage == 2 else 'val_loss'
                    print(
                        f"  ✓ Saved best EMA model (decay={decay}) to "
                        f"{ema_model_filename} ({metric_name}={ema_val_metric:.6f})"
                    )
        
        # 定期保存检查点（每5个epoch）- 作为备份
        if (stage_epoch + 1) % 5 == 0:
            if isinstance(model, nn.DataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            # 保存标准模型checkpoint
            checkpoint_path = osp.join(trainDataFolder, f'checkpoint_stage{current_stage}_epoch_{stage_epoch}.pth')
            torch.save({
                'epoch': stage_epoch,
                'stage': current_stage,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer._optimizer.state_dict(),
                'n_steps': optimizer.n_steps,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_metric_name': 'cost' if current_stage == 2 else 'loss',
                'stage1_best_loss': stage1_best_loss,
                'torch_seed': torch_seed
            }, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")
            
            # 分别保存每个EMA checkpoint
            if use_ema and len(ema_models) > 0:
                for i, (ema_m, decay) in enumerate(zip(ema_models, ema_decays)):
                    ema_state_dict = ema_m.module.state_dict()
                    ema_checkpoint_filename = f'checkpoint_stage{current_stage}_epoch_{stage_epoch}_ema_{decay}.pth'
                    ema_checkpoint_path = osp.join(trainDataFolder, ema_checkpoint_filename)
                    
                    torch.save({
                        'epoch': stage_epoch,
                        'stage': current_stage,
                        'model_state_dict': ema_state_dict,
                        'ema_decay': decay,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'val_metric_name': 'cost' if current_stage == 2 else 'loss',
                        'stage1_best_loss': stage1_best_loss,
                        'torch_seed': torch_seed
                    }, ema_checkpoint_path)
                    print(f"  Saved EMA checkpoint (decay={decay}) to {ema_checkpoint_filename}")
        
        # 保存训练进度
        pickle.dump({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }, open(osp.join(trainDataFolder, 'training_progress.pkl'), 'wb'))
        
        print()
    
    # =================== 训练完成 ===================
    writer.close()
    
    print(f"\n{'='*60}")
    print("Training completed!")
    if current_stage == 2:
        print(f"Stage 1 best validation loss: {stage1_best_loss:.6f}")
    best_metric_name = 'cost' if current_stage == 2 else 'loss'
    print(f"Stage {current_stage} best validation {best_metric_name}: {best_val_loss:.6f}")
    print(f"Best model saved to: {best_model_path}")
    print(f"{'='*60}\n")
    
    # 保存最终模型
    if isinstance(model, nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    
    final_model_path = osp.join(trainDataFolder, 'final_model.pth')
    final_stage = 2 if n_epochs > stage1_config['epochs'] else 1
    final_stage_epoch = (n_epochs - 1) if final_stage == 1 else (n_epochs - 1 - stage1_config['epochs'])
    final_metric_name = 'cost' if final_stage == 2 else 'loss'
    
    # 保存标准最终模型
    torch.save({
        'epoch': final_stage_epoch,
        'stage': final_stage,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer._optimizer.state_dict(),
        'n_steps': optimizer.n_steps,
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'val_metric_name': final_metric_name,
        'stage1_best_loss': stage1_best_loss,
        'torch_seed': torch_seed
    }, final_model_path)
    
    print(f"Final model saved to: {final_model_path}")
    
    # 保存每个EMA最终模型
    if use_ema and len(ema_models) > 0:
        for i, (ema_m, decay) in enumerate(zip(ema_models, ema_decays)):
            ema_state_dict = ema_m.module.state_dict()
            final_ema_filename = f'final_model_ema_{decay}.pth'
            final_ema_path = osp.join(trainDataFolder, final_ema_filename)
            
            torch.save({
                'epoch': final_stage_epoch,
                'stage': final_stage,
                'model_state_dict': ema_state_dict,
                'ema_decay': decay,
                'train_loss': train_losses[-1],
                'val_loss': val_losses[-1],
                'val_metric_name': final_metric_name,
                'stage1_best_loss': stage1_best_loss,
                'torch_seed': torch_seed
            }, final_ema_path)
            print(f"Final EMA model (decay={decay}) saved to: {final_ema_filename}")
    
    # Print summary of best models
    if stage1_best_loss < float('inf'):
        print(f"Stage 1 best validation loss: {stage1_best_loss:.4f}")
        print(f"  Saved to: {stage1_best_path}")
    if current_stage == 2:
        print(f"Stage 2 best validation loss: {best_val_loss:.4f}")
        print(f"  Saved to: {stage2_best_path}")

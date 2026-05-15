"""
train_dit.py - 训练不平坦地面路径预测模型(基于扩散模型)
"""

import numpy as np
import pickle
from contextlib import nullcontext
import os

import torch
import torch.optim as optim

import json
import argparse

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from os import path as osp

from transformer import Models, Optim
from dataLoader_dit import UnevenPathDataLoader, PaddedSequence
from dataLoader_dit import hashTable, receptive_field

from torch.utils.tensorboard import SummaryWriter
from timm.utils import ModelEmaV2

from ESDF3d_atpoint import compute_esdf_batch
# from grad_optimizer import TrajectoryOptimizerSE2
from dit.Models import PathDiffusionTransformer

# B样条工具
from bspline_utils import (
    fit_bspline_least_squares,
    reconstruct_from_control_points,
    DifferentiableBSpline
)
from torch.func import functional_call, jvp
import math

def _safe_tensor_quantile(values, q, max_elements=2_000_000):
    """大张量安全的分位数估计，用于诊断监控。"""
    flat = values.detach().float().flatten()
    if flat.numel() == 0:
        return 0.0
    if flat.numel() > max_elements:
        # 诊断指标不需要精确到每个参数；确定性等间隔采样可避免 torch.quantile 的大张量限制。
        idx = torch.linspace(
            0,
            flat.numel() - 1,
            steps=max_elements,
            device=flat.device,
            dtype=torch.long,
        )
        flat = flat[idx]
    return torch.quantile(flat, q).item()


# =================== Fisher-aware Adam 优化器 ===================
class FisherAdamW(torch.optim.Optimizer):
    """
    Fisher 感知的 AdamW 优化器
    
    在 Adam update 后，用 Fisher 对角矩阵对 update 进行逐参数缩放。
    这样可以让高 Fisher 参数的更新幅度小，低 Fisher 参数的更新幅度大。
    
    核心思想：
        u_j^{Adam} = m_j / (sqrt(v_j) + eps)
        u_j = u_j^{Adam} / (F_j + fisher_eps)^alpha
        theta_j <- theta_j - lr * u_j
    
    其中 alpha=1 对应理论上的 Fisher 逆预条件。
    """
    
    def __init__(
        self,
        named_params,
        fisher_diag,
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        fisher_eps=1e-8,
        fisher_alpha=1.0,
        diagnostics_every=10,
        diagnostic_fisher_diag=None,
    ):
        """
        Fisher-aware AdamW: 使用稳健归一化 Fisher 度量进行二阶优化
        
        更新方向满足：
            min_s (1/2) s^T M s  s.t.  g^T s = -delta
        其中 M = tilde{F} 是经过稳健归一化的 Fisher 度量
        闭式解：s* ∝ -M^{-1} g = -(tilde{F})^{-1} g
        
        Args:
            named_params: 模型的 named_parameters() 迭代器
            fisher_diag: dict，键为参数名，值为稳健归一化后的 Fisher 张量
            lr: 学习率
            betas: Adam 的 beta1, beta2
            eps: Adam 的 eps
            weight_decay: 权重衰减系数
            fisher_eps: Fisher 缩放的数值稳定化常数（阻尼项）
            fisher_alpha: Fisher 缩放的指数（固定为 1.0，对应二阶信息最小损伤原理）
            diagnostics_every: 每隔多少个 optimizer step 采样一次更新能量诊断
            diagnostic_fisher_diag: 可选，仅用于 high/low Fisher 诊断分组；
                若为空，则使用 fisher_diag。Direct Safe Adam baseline 可传入
                真实 PMF-FIM 做诊断，同时用全 1 fisher_diag 关闭缩放。
        """
        named_params = [
            (n, p) for n, p in named_params if p.requires_grad
        ]
        diagnostic_fisher_diag = diagnostic_fisher_diag or fisher_diag
        
        params = [p for _, p in named_params]
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            fisher_eps=fisher_eps,
            fisher_alpha=fisher_alpha,
        )
        
        super().__init__(params, defaults)
        
        # 保存 Fisher 对角映射（参数id -> Fisher 张量）
        self.fisher_by_id = {}
        self.diagnostic_fisher_by_id = {}
        for name, p in named_params:
            if name not in fisher_diag:
                raise ValueError(f"Missing Fisher entry for parameter: {name}")
            if name not in diagnostic_fisher_diag:
                raise ValueError(f"Missing diagnostic Fisher entry for parameter: {name}")
            self.fisher_by_id[id(p)] = fisher_diag[name].detach()
            self.diagnostic_fisher_by_id[id(p)] = diagnostic_fisher_diag[name].detach()

        with torch.no_grad():
            all_fisher = torch.cat([f.detach().float().cpu().flatten() for f in diagnostic_fisher_diag.values()])
            self.fisher_q20 = _safe_tensor_quantile(all_fisher, 0.2)
            self.fisher_q80 = _safe_tensor_quantile(all_fisher, 0.8)

        self.diagnostics_every = max(int(diagnostics_every), 0)
        self._diagnostic_step = 0
        self.reset_diagnostics()

    def reset_diagnostics(self):
        """重置 Fisher-aware 更新诊断的 epoch 累积量。"""
        self._diagnostics = {
            'sampled_steps': 0,
            'raw_high_energy': 0.0,
            'raw_low_energy': 0.0,
            'scaled_high_energy': 0.0,
            'scaled_low_energy': 0.0,
            'actual_high_energy': 0.0,
            'actual_low_energy': 0.0,
            'high_count': 0,
            'low_count': 0,
        }

    def diagnostics_summary(self):
        """返回当前 epoch 的 Fisher-aware 更新诊断。"""
        d = self._diagnostics
        high_count = max(d['high_count'], 1)
        low_count = max(d['low_count'], 1)
        raw_high_mean = d['raw_high_energy'] / high_count
        raw_low_mean = d['raw_low_energy'] / low_count
        scaled_high_mean = d['scaled_high_energy'] / high_count
        scaled_low_mean = d['scaled_low_energy'] / low_count
        actual_high_mean = d['actual_high_energy'] / high_count
        actual_low_mean = d['actual_low_energy'] / low_count

        return {
            'fisher_update_sampled_steps': float(d['sampled_steps']),
            'fisher_update_high_energy': d['actual_high_energy'],
            'fisher_update_low_energy': d['actual_low_energy'],
            'fisher_update_high_mean': actual_high_mean,
            'fisher_update_low_mean': actual_low_mean,
            'fisher_update_high_low_ratio': actual_high_mean / (actual_low_mean + 1e-30),
            'fisher_raw_update_high_low_ratio': raw_high_mean / (raw_low_mean + 1e-30),
            'fisher_scaled_update_high_low_ratio': scaled_high_mean / (scaled_low_mean + 1e-30),
            'fisher_high_attenuation': scaled_high_mean / (raw_high_mean + 1e-30),
            'fisher_low_amplification': scaled_low_mean / (raw_low_mean + 1e-30),
        }
    
    @torch.no_grad()
    def step(self, closure=None):
        """执行一步优化。"""
        loss = None
        
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._diagnostic_step += 1
        collect_diagnostics = (
            self.diagnostics_every > 0
            and self._diagnostic_step % self.diagnostics_every == 0
        )
        if collect_diagnostics:
            self._diagnostics['sampled_steps'] += 1
        
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            fisher_eps = group["fisher_eps"]
            fisher_alpha = group["fisher_alpha"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.detach()
                
                if grad.is_sparse:
                    raise RuntimeError("FisherAdamW does not support sparse gradients.")
                
                state = self.state[p]
                
                # 初始化状态
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                
                state["step"] += 1
                step = state["step"]
                
                # Adam 一阶和二阶矩更新
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                
                # 偏置修正
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                
                # Adam update
                denom = exp_avg_sq.sqrt()
                denom.div_(math.sqrt(bias_correction2))
                denom.add_(eps)
                
                adam_update = exp_avg / bias_correction1 / denom
                
                # Fisher 感知缩放
                fisher = self.fisher_by_id[id(p)].to(
                    device=p.device,
                    dtype=p.dtype
                )
                diagnostic_fisher = self.diagnostic_fisher_by_id[id(p)].to(
                    device=p.device,
                    dtype=p.dtype
                )
                
                # Fisher 缩放因子：(F_j + eps)^alpha
                fisher_scale = (fisher + fisher_eps).pow(fisher_alpha)
                
                # 应用 Fisher 缩放
                update = adam_update / fisher_scale

                self._accumulate_update_diagnostics(
                    fisher=diagnostic_fisher,
                    adam_update=adam_update,
                    scaled_update=update,
                    lr=lr,
                    collect=collect_diagnostics,
                )
                
                # 权重衰减（decoupled）
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)
                
                # 更新参数
                p.add_(update, alpha=-lr)
        
        return loss

    def _accumulate_update_diagnostics(self, fisher, adam_update, scaled_update, lr, collect):
        """采样记录高/低 Fisher 区域的 Adam 更新能量。"""
        if not collect:
            return

        with torch.no_grad():
            high_mask = fisher >= self.fisher_q80
            low_mask = fisher <= self.fisher_q20
            high_count = int(high_mask.sum().item())
            low_count = int(low_mask.sum().item())
            if high_count == 0 or low_count == 0:
                return

            actual_update = scaled_update * lr
            d = self._diagnostics
            d['high_count'] += high_count
            d['low_count'] += low_count
            d['raw_high_energy'] += adam_update[high_mask].detach().float().pow(2).sum().item()
            d['raw_low_energy'] += adam_update[low_mask].detach().float().pow(2).sum().item()
            d['scaled_high_energy'] += scaled_update[high_mask].detach().float().pow(2).sum().item()
            d['scaled_low_energy'] += scaled_update[low_mask].detach().float().pow(2).sum().item()
            d['actual_high_energy'] += actual_update[high_mask].detach().float().pow(2).sum().item()
            d['actual_low_energy'] += actual_update[low_mask].detach().float().pow(2).sum().item()


# =================== B样条控制点转换 ===================
def trajectory_to_control_points(trajectory, num_middle_points=24):
    """
    将轨迹转换为B样条控制点（仅返回中间控制点，不包含起终点）
    
    【语义说明】
    - 完整B样条需要26个控制点 = 起点(1) + 中间点(24) + 终点(1)
    - 起点和终点是固定的，由start_pose和goal_pose决定
    - 网络只需要预测中间24个自由控制点
    
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


def control_points_to_trajectory(control_points, num_output_points=100):
    """
    从控制点重建密集轨迹
    
    Args:
        control_points: (B, num_control_points, 2) 控制点张量（只包含x,y）
        num_output_points: 重建后的轨迹点数量
    
    Returns:
        trajectory: (B, num_output_points, 2) 重建的轨迹（只包含x,y）
    """
    B, num_cp, _ = control_points.shape
    device = control_points.device
    
    # 将张量转换为numpy
    control_points_np = control_points.cpu().numpy()
    
    trajectory_list = []
    for i in range(B):
        cp = control_points_np[i]  # (num_cp, 2)
        
        # 重建轨迹
        traj = reconstruct_from_control_points(
            cp,
            num_output_points=num_output_points,
            degree=3
        )
        trajectory_list.append(traj)
    
    # 转换回张量
    trajectory = torch.tensor(np.stack(trajectory_list, axis=0), dtype=torch.float32, device=device)
    
    return trajectory  # (B, num_output_points, 2)


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

def compute_smoothness_loss(trajectory, include_angle=False, threshold=None):
    """
    改进的平滑性损失 - 只惩罚超过阈值的急转弯
    Args:
        trajectory: (B, N, 3) - 轨迹序列，格式为(x, y, theta)
        include_angle: bool - 是否在平滑性计算中包含角度维度
        threshold: float - 加速度阈值，只惩罚超过此值的加速度（None则惩罚所有）
    Returns:
        smoothness_loss: 标量
    """
    # 位置部分 (x, y) 的一阶和二阶差分
    pos_first_diff = trajectory[:, 1:, :2] - trajectory[:, :-1, :2]  # (B, N-1, 2)
    pos_second_diff = pos_first_diff[:, 1:, :] - pos_first_diff[:, :-1, :]  # (B, N-2, 2)
    
    # 计算加速度的L2范数
    acceleration_norm = torch.norm(pos_second_diff, dim=2)  # (B, N-2)
    
    if threshold is not None:
        # 只惩罚超过阈值的加速度（软阈值）
        # 使用ReLU：max(0, |a| - threshold)^2
        excess_acceleration = torch.relu(acceleration_norm - threshold)
        pos_smoothness = torch.mean(excess_acceleration ** 2)
    else:
        # 原始版本：惩罚所有加速度
        pos_smoothness = torch.mean(acceleration_norm ** 2)
    
    if include_angle:
        # 角度部分：对theta做周期性感知的差分
        theta = trajectory[:, :, 2]  # (B, N)
        angle_first_diff = torch.atan2(torch.sin(theta[:, 1:] - theta[:, :-1]),
                                       torch.cos(theta[:, 1:] - theta[:, :-1]))  # (B, N-1)
        angle_second_diff = torch.atan2(torch.sin(angle_first_diff[:, 1:] - angle_first_diff[:, :-1]),
                                        torch.cos(angle_first_diff[:, 1:] - angle_first_diff[:, :-1]))  # (B, N-2)
        angle_acc_norm = torch.abs(angle_second_diff)  # (B, N-2)
        
        if threshold is not None:
            angle_threshold = threshold * 0.1
            excess_angle_acc = torch.relu(angle_acc_norm - angle_threshold)
            angle_smoothness = torch.mean(excess_angle_acc ** 2)
        else:
            angle_smoothness = torch.mean(angle_acc_norm ** 2)
        
        smoothness_loss = pos_smoothness + angle_smoothness
    else:
        smoothness_loss = pos_smoothness
    
    return smoothness_loss

def compute_angle_smoothness_loss(trajectory):
    """
    计算角度平滑性损失 - 惩罚角速度本身和角加速度
    使用theta的周期性感知差分计算角速度/角加速度
    
    **重要**：当trajectory包含固定的起点和终点时（22个点），
    排除边界段的角速度约束（索引0和20），避免强制预测点角度向固定的起终点角度靠拢。
    
    Args:
        trajectory: (B, N, 3) - 轨迹序列，格式为(x, y, theta)
    Returns:
        angle_loss: 标量
    """
    theta = trajectory[:, :, 2]  # (B, N)
    
    # 相邻点角度差（周期性感知）
    delta = torch.atan2(torch.sin(theta[:, 1:] - theta[:, :-1]),
                        torch.cos(theta[:, 1:] - theta[:, :-1]))  # (B, N-1)
    angular_velocity = torch.abs(delta)  # (B, N-1)
    
    # **关键修改**：如果轨迹是22个点（包含起终点），排除边界段的角速度约束
    N = trajectory.shape[1]
    velocity_weights = torch.ones_like(angular_velocity)
    
    if N == 22:  # 完整轨迹（起点+20预测点+终点）
        # 完全排除边界段：不约束"起点→第一个预测点"和"最后预测点→终点"的角度变化
        # 让第一个和最后一个预测点的角度可以自由适应实际运动方向
        velocity_weights[:, 0] = 0.0   # 排除起点到第一个预测点
        velocity_weights[:, -1] = 0.0  # 排除最后一个预测点到终点
    
    velocity_loss = torch.mean((angular_velocity ** 2) * velocity_weights)
    
    # 计算角速度的变化（角加速度）
    delta_diff = torch.atan2(torch.sin(delta[:, 1:] - delta[:, :-1]),
                             torch.cos(delta[:, 1:] - delta[:, :-1]))  # (B, N-2)
    angle_acc = torch.abs(delta_diff)  # (B, N-2)
    
    # **角加速度也需要排除边界相关的项**
    # 对于22个点的轨迹，angle_acc有20个值（索引0-19）
    # 索引0对应的是"起点→第一预测点→第二预测点"的角加速度
    # 索引19对应的是"倒数第三预测点→倒数第二预测点→终点"的角加速度
    if N == 22:
        acc_weights = torch.ones_like(angle_acc)
        acc_weights[:, 0] = 0.0   # 排除涉及起点的角加速度
        acc_weights[:, -1] = 0.0  # 排除涉及终点的角加速度
        acceleration_loss = torch.mean((angle_acc ** 2) * acc_weights)
    else:
        acceleration_loss = torch.mean(angle_acc ** 2)
    
    # 混合损失：角速度 + 角加速度
    # 只约束预测点内部的角度平滑性，不约束与固定起终点的衔接
    angle_loss = velocity_loss + 0.5 * acceleration_loss
    
    return angle_loss

def compute_angle_consistency_loss(trajectory):
    """
    计算角度一致性损失 - 防止倒车（使用heading向量与运动方向的点积）
    该损失确保运动方向与车辆朝向一致（防止倒车）
    
    **重要**：当trajectory包含固定的起点和终点时（22个点），
    只计算预测点内部的角度一致性（索引1到20），排除边界段（0→1和20→21）
    以避免与固定的起终点角度产生冲突。
    
    Args:
        trajectory: (B, N, 3) - 轨迹序列，格式为(x, y, theta)
    Returns:
        angle_loss: 标量
    """
    # 提取位置和朝向向量
    positions = trajectory[:, :, :2]  # (B, N, 2)
    theta = trajectory[:, :, 2]  # (B, N)
    
    # 车辆朝向向量：(cos(θ), sin(θ))
    heading_vectors = torch.stack([torch.cos(theta), torch.sin(theta)], dim=2)  # (B, N, 2)
    
    # 归一化朝向向量
    eps = 1e-6
    heading_norms = torch.norm(heading_vectors, dim=2, keepdim=True)  # (B, N, 1)
    heading_vectors_norm = heading_vectors / (heading_norms + eps)  # (B, N, 2)
    
    # 计算运动方向向量（从当前点指向下一个点）
    motion_vectors = positions[:, 1:, :] - positions[:, :-1, :]  # (B, N-1, 2)
    
    # 计算运动向量的长度，用于归一化和过滤静止点
    motion_lengths = torch.norm(motion_vectors, dim=2, keepdim=True)  # (B, N-1, 1)
    
    # 只对运动距离足够大的点计算损失
    valid_mask = (motion_lengths.squeeze(2) > eps)  # (B, N-1)
    
    # **关键修改**：如果轨迹是22个点（包含起终点），排除边界段
    # 只计算索引1到20之间的角度一致性（预测点内部）
    N = trajectory.shape[1]
    if N == 22:  # 完整轨迹（起点+20预测点+终点）
        # 排除第一段（起点→第一个预测点，索引0）和最后一段（最后一个预测点→终点，索引20）
        boundary_mask = torch.ones_like(valid_mask, dtype=torch.bool)
        boundary_mask[:, 0] = False   # 排除起点→第一个预测点
        boundary_mask[:, -1] = False  # 排除最后一个预测点→终点
        valid_mask = valid_mask & boundary_mask
    
    # 如果没有有效的运动点，返回0损失
    if not valid_mask.any():
        return torch.tensor(0.0, device=trajectory.device)
    
    # 归一化运动向量
    motion_vectors_norm = motion_vectors / (motion_lengths + eps)  # (B, N-1, 2)
    
    # 使用当前点的归一化朝向向量
    current_headings = heading_vectors_norm[:, :-1, :]  # (B, N-1, 2)
    
    # 计算朝向向量与运动向量的点积（余弦相似度）
    # dot = cos(θ)，当θ=0时（前进）dot=1，当θ=π时（倒车）dot=-1
    cos_similarity = (current_headings * motion_vectors_norm).sum(dim=2)  # (B, N-1)
    
    # 只计算有效点的损失
    # 使用 (1 - cos_similarity) 作为损失：
    # - 前进时 cos ≈ 1，损失 ≈ 0
    # - 倒车时 cos ≈ -1，损失 ≈ 2
    masked_loss = (1 - cos_similarity) * valid_mask.float()  # (B, N-1)
    angle_loss = masked_loss.sum() / (valid_mask.sum() + eps)
    
    # 最终检查NaN
    if torch.isnan(angle_loss) or torch.isinf(angle_loss):
        return torch.tensor(0.0, device=trajectory.device)
    
    return angle_loss

def compute_uniformity_loss(trajectory):
    """
    计算均匀性损失 - 相邻点之间距离的方差
    Args:
        trajectory: (B, N, 3) - 轨迹序列，格式为(x, y, theta)
    Returns:
        uniformity_loss: 标量
    """
    # 计算相邻点的欧几里得距离
    pos_diff = trajectory[:, 1:, :2] - trajectory[:, :-1, :2]  # (B, N-1, 2)
    distances = torch.norm(pos_diff, dim=2)  # (B, N-1)
    # 距离的方差（希望距离均匀）
    mean_dist = torch.mean(distances, dim=1, keepdim=True)  # (B, 1)
    uniformity_loss = torch.mean((distances - mean_dist) ** 2)
    return uniformity_loss

def compute_sincos_normalization_loss(trajectory):
    """
    兼容保留：theta范围正则（替代sin/cos归一化损失）
    
    Args:
        trajectory: (B, N, 3) - 轨迹序列，格式为(x, y, theta)
    Returns:
        norm_loss: 标量
    """
    theta = trajectory[:, :, 2]  # (B, N)
    angle_overflow = torch.abs(theta) - torch.pi
    angle_overflow = torch.clamp(angle_overflow, min=0.0)
    norm_loss = torch.mean(angle_overflow ** 2)
    return norm_loss

def compute_curvature_constraint_loss(trajectory, max_curvature=2.0):
    """
    曲率约束损失 - 只惩罚超过最大曲率的点
    使用三点法计算曲率：κ = 2*sin(θ) / d
    其中θ是转角，d是弦长
    
    Args:
        trajectory: (B, N, 3) - 轨迹序列，格式为(x, y, theta)
        max_curvature: float - 最大允许曲率（单位：1/米）
    Returns:
        curvature_loss: 标量
    """
    positions = trajectory[:, :, :2]  # (B, N, 2)
    
    # 计算三个连续点形成的向量
    vec1 = positions[:, 1:-1, :] - positions[:, :-2, :]  # (B, N-2, 2) P1->P2
    vec2 = positions[:, 2:, :] - positions[:, 1:-1, :]    # (B, N-2, 2) P2->P3
    
    # 计算转角（使用向量夹角）
    # cos(θ) = (v1·v2) / (|v1||v2|)
    dot_product = (vec1 * vec2).sum(dim=2)  # (B, N-2)
    norm1 = torch.norm(vec1, dim=2)  # (B, N-2)
    norm2 = torch.norm(vec2, dim=2)  # (B, N-2)
    
    eps = 1e-6
    cos_angle = dot_product / (norm1 * norm2 + eps)
    # 数值稳定性：不仅clamp到[-1,1]，还要避免正好在边界（acos梯度无穷大）
    cos_angle = torch.clamp(cos_angle, -0.9999, 0.9999)
    angle = torch.acos(cos_angle)  # (B, N-2) 转角 [0, π]
    
    # 计算弦长（P1到P3的距离）
    chord_length = torch.norm(positions[:, 2:, :] - positions[:, :-2, :], dim=2) + eps  # (B, N-2)
    
    # 近似曲率：κ ≈ 2*sin(θ/2) / chord_length
    # 简化：κ ≈ θ / chord_length (小角度近似)
    curvature = angle / chord_length  # (B, N-2)
    
    # 只惩罚超过最大曲率的点
    excess_curvature = curvature - max_curvature  # (B, N-2)
    # curvature_loss = torch.mean(torch.relu(excess_curvature) ** 2)
    curvature_loss = torch.mean(F.softplus(excess_curvature, beta=2.0))
    
    return curvature_loss


def _get_stage2_fisher_proxy_state():
    """获取/初始化第二阶段的 Fisher 代理状态。"""
    if not hasattr(diffusion_loss, '_stage2_fisher_proxy_state'):
        diffusion_loss._stage2_fisher_proxy_state = {
            'ref_params': None,
            'fisher_diag': None,
            'proxy_scale': 0.01,
            'fisher_eps': 1e-8,
        }
    return diffusion_loss._stage2_fisher_proxy_state


def estimate_diag_fisher_from_main_loss(
    model,
    dataloader,
    compute_loss_fn,
    device,
    max_batches=None,
    f_min=0.1,
    f_max=10.0,
    neutral_fisher=1.0,
):
    """
    从第一阶段 main_loss 估计对角 Fisher 信息矩阵。
    
    Fisher_j = E[(∂main_loss/∂θ_j)²]
    
    Args:
        model: 神经网络模型
        dataloader: 数据加载器
        compute_loss_fn: 计算 main_loss 的函数 (model, batch) -> loss
        device: 设备
        max_batches: 最大 batch 数（用于快速估计），None 表示用全部数据
        f_min, f_max: Fisher 对角裁剪范围（防止数值病态）
    
    Returns:
        fisher_dict: dict，键为参数名，值为 Fisher 对角张量
    """
    model.eval()
    
    fisher = {
        name: torch.zeros_like(p, device=device)
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    # 统计每个参数被有效梯度更新的次数（grad is not None）
    touched = {name: 0 for name in fisher.keys()}
    
    num_batches = 0
    
    with torch.enable_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            model.zero_grad(set_to_none=True)
            
            # 移动 batch 到设备
            if isinstance(batch, dict):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
            else:
                batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
            
            # 计算 main_loss
            main_loss = compute_loss_fn(model, batch)
            
            # 确保 main_loss 是标量
            if main_loss.dim() > 0:
                main_loss = main_loss.mean()
            
            # 反向传播
            main_loss.backward()
            
            # 累积梯度平方
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[name] += p.grad.detach().pow(2)
                    touched[name] += 1
            
            num_batches += 1
    
    # 平均化
    for name in fisher:
        if touched[name] > 0:
            fisher[name] /= touched[name]
        else:
            # 若该参数在 Fisher 估计阶段从未拿到梯度，则使用中性缩放（不放大不缩小）
            fisher[name].fill_(neutral_fisher)
    
    # 【诊断】统计有多少参数被有效梯度更新
    touched_count = sum(1 for cnt in touched.values() if cnt > 0)
    total_params = len(touched)
    print(f"[Fisher Diagnostic] Touched {touched_count}/{total_params} params in {num_batches} batches")
    
    # 逐参数张量稳健归一化：log-quantile 方法
    # 目标：构造稳健的 Fisher 度量 tilde{F}，满足
    #   1. 正值：tilde{F}_j > 0
    #   2. 尺度稳定：median(tilde{F}) ≈ 1
    #   3. 保留重要性顺序：高 Fisher → 高 tilde{F}，但长尾不破坏分布
    active_names = [name for name, cnt in touched.items() if cnt > 0]
    if len(active_names) > 0:
        all_raw = torch.cat([fisher[name].detach().float().flatten() for name in active_names])
        print(f"[Fisher Diagnostic] Raw fisher (before robust normalization):")
        print(f"  - min={all_raw.min().item():.3e}, max={all_raw.max().item():.3e}, "
              f"mean={all_raw.mean().item():.3e}, median={all_raw.median().item():.3e}, "
              f"positive_frac={(all_raw > 0).float().mean().item():.2%}, zero_frac={(all_raw <= 0).float().mean().item():.2%}")

        eps_log = 1e-30
        import numpy as np
        log_f_min = np.log(f_min)
        log_f_max = np.log(f_max)
        q_percentile = 0.95  # 使用 95% 分位数作为离散度尺度

        per_tensor_scales = []
        per_tensor_log_medians = []
        
        for name in active_names:
            v = fisher[name].detach().float()
            positive = v[v > 0]
            
            if positive.numel() == 0:
                # 这个张量完全没有正值，保持为 1.0
                fisher[name] = torch.ones_like(v)
                per_tensor_scales.append(torch.tensor(1.0, device=v.device))
                per_tensor_log_medians.append(torch.tensor(0.0, device=v.device))
                continue

            # 在 log 空间计算
            log_pos = torch.log(positive.clamp_min(eps_log))
            log_median = log_pos.median()
            per_tensor_log_medians.append(log_median)
            
            # 计算 log 偏差
            log_deviation = log_pos - log_median
            
            # 用分位数作为尺度（避免极值主导）
            abs_deviation = log_deviation.abs()
            scale_quantile = torch.quantile(abs_deviation, q_percentile).clamp_min(1e-6)
            per_tensor_scales.append(scale_quantile)
            
            # 对所有值（包括 0）做归一化
            log_all = torch.log(v.clamp_min(eps_log))
            log_centered = log_all - log_median
            log_normalized = log_centered / scale_quantile
            
            # 裁剪到固定范围
            log_clipped = torch.clamp(log_normalized, log_f_min, log_f_max)
            
            # 指数回来得到 tilde{F}
            fisher[name] = torch.exp(log_clipped).detach()

        per_tensor_scales = torch.stack(per_tensor_scales)
        per_tensor_log_medians = torch.stack(per_tensor_log_medians)
        
        print(f"[Fisher Diagnostic] Per-tensor log-space statistics:")
        print(f"  - log_median: min={per_tensor_log_medians.min().item():.3e}, "
              f"max={per_tensor_log_medians.max().item():.3e}, mean={per_tensor_log_medians.mean().item():.3e}")
        print(f"  - scale (q={q_percentile}): min={per_tensor_scales.min().item():.3e}, "
              f"max={per_tensor_scales.max().item():.3e}, median={per_tensor_scales.median().item():.3e}")
        
        # 【诊断】归一化后、裁剪前
        fisher_normalized_all = torch.cat([fisher[name].detach().float().flatten() for name in active_names])
        print(f"[Fisher Diagnostic] After log-quantile per-tensor normalization (before clipping):")
        print(f"  - min={fisher_normalized_all.min().item():.3e}, max={fisher_normalized_all.max().item():.3e}, "
              f"mean={fisher_normalized_all.mean().item():.3e}, median={fisher_normalized_all.median().item():.3e}")
        
        # 【诊断】裁剪后的最终 Fisher 分布
        fisher_clipped_all = torch.cat([fisher[name].detach().float().flatten() for name in active_names])
        frac_at_min = (fisher_clipped_all <= (f_min + 1e-7)).float().mean().item()
        frac_at_max = (fisher_clipped_all >= (f_max - 1e-7)).float().mean().item()
        
        # 计算分位数，处理大张量/内存问题
        try:
            q10 = torch.quantile(fisher_clipped_all.float(), 0.1).item()
            q90 = torch.quantile(fisher_clipped_all.float(), 0.9).item()
        except Exception:
            sorted_vals = torch.sort(fisher_clipped_all.float())[0]
            q10 = sorted_vals[int(len(sorted_vals) * 0.1)].item()
            q90 = sorted_vals[int(len(sorted_vals) * 0.9)].item()
        
        print(f"[Fisher Diagnostic] After clipping [f_min={f_min}, f_max={f_max}]:")
        print(f"  - min={fisher_clipped_all.min().item():.3e}, max={fisher_clipped_all.max().item():.3e}, "
              f"mean={fisher_clipped_all.mean().item():.3e}, median={fisher_clipped_all.median().item():.3e}")
        print(f"  - q10={q10:.3e}, q90={q90:.3e}")
        print(f"  - clip@min={frac_at_min:.2%}, clip@max={frac_at_max:.2%}")
        
        # 【诊断】Fisher 逆的缩放因子分布（对应 Adam 更新放大倍数）
        fisher_inv_scale = 1.0 / (fisher_clipped_all + 1e-8)
        print(f"[Fisher Diagnostic] Fisher inverse scaling factors (1/(F+eps)) with alpha=1.0:")
        print(f"  - min={fisher_inv_scale.min().item():.3e}, max={fisher_inv_scale.max().item():.3e}, "
              f"mean={fisher_inv_scale.mean().item():.3e}, median={fisher_inv_scale.median().item():.3e}")
        print(f"  - ratio max/min = {(fisher_inv_scale.max() / fisher_inv_scale.min()).item():.2f}x")
    else:
        # 极端情况：全部参数都没有有效梯度
        for name in fisher:
            fisher[name].fill_(neutral_fisher)
            fisher[name] = fisher[name].detach()
    
    return fisher


def _store_fixed_fisher_stats(proxy_state, fisher_diag, f_min, f_max):
    """保存固定 Fisher 的分布统计，供 TensorBoard 和后续诊断使用。"""
    with torch.no_grad():
        all_fisher = torch.cat([v.detach().float().flatten() for v in fisher_diag.values()])
        fisher_inv = 1.0 / (all_fisher + 1e-8)
        proxy_state['fisher_diag_mean_fixed'] = all_fisher.mean().item()
        proxy_state['fisher_diag_median_fixed'] = all_fisher.median().item()
        proxy_state['fisher_diag_q10_fixed'] = _safe_tensor_quantile(all_fisher, 0.1)
        proxy_state['fisher_diag_q20_fixed'] = _safe_tensor_quantile(all_fisher, 0.2)
        proxy_state['fisher_diag_q80_fixed'] = _safe_tensor_quantile(all_fisher, 0.8)
        proxy_state['fisher_diag_q90_fixed'] = _safe_tensor_quantile(all_fisher, 0.9)
        proxy_state['fisher_diag_min_fixed'] = all_fisher.min().item()
        proxy_state['fisher_diag_max_fixed'] = all_fisher.max().item()
        proxy_state['fisher_clip_at_min_fixed'] = (all_fisher <= (f_min + 1e-7)).float().mean().item()
        proxy_state['fisher_clip_at_max_fixed'] = (all_fisher >= (f_max - 1e-7)).float().mean().item()
        proxy_state['fisher_inv_scale_mean_fixed'] = fisher_inv.mean().item()
        proxy_state['fisher_inv_scale_median_fixed'] = fisher_inv.median().item()
        proxy_state['fisher_inv_scale_ratio_fixed'] = (fisher_inv.max() / fisher_inv.min()).item()


def _make_unit_fisher_diag_like(fisher_diag):
    """构造全 1 Fisher，用于关闭 Fisher scaling 的 Direct Safe Adam baseline。"""
    return {name: torch.ones_like(value) for name, value in fisher_diag.items()}


def _attach_stage2_fisher_list_to_proxy_state(model, proxy_state, fisher_diag):
    """按当前可训练参数顺序保存 Fisher 张量，便于高/低 Fisher 参数漂移诊断。"""
    trainable_named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    proxy_state['trainable_param_names'] = [name for name, _ in trainable_named_params]
    proxy_state['fisher_diag'] = [
        fisher_diag[name].detach().to(device=p.device, dtype=p.dtype)
        for name, p in trainable_named_params
        if name in fisher_diag
    ]


def compute_stage2_fisher_param_diagnostics(model):
    """计算 Stage 2 当前参数相对 Stage1 锚点在高/低 Fisher 区域的漂移。"""
    proxy_state = _get_stage2_fisher_proxy_state()
    ref_params = proxy_state.get('ref_params')
    fisher_diag = proxy_state.get('fisher_diag')
    q20 = proxy_state.get('fisher_diag_q20_fixed')
    q80 = proxy_state.get('fisher_diag_q80_fixed')

    if ref_params is None or fisher_diag is None or q20 is None or q80 is None:
        return {}

    base_model = model.module if hasattr(model, 'module') else model
    trainable_params = [p for p in base_model.parameters() if p.requires_grad]
    if len(trainable_params) != len(ref_params) or len(trainable_params) != len(fisher_diag):
        return {}

    high_energy = 0.0
    low_energy = 0.0
    total_energy = 0.0
    high_count = 0
    low_count = 0
    total_count = 0

    with torch.no_grad():
        for param, ref_param, fisher in zip(trainable_params, ref_params, fisher_diag):
            fisher = fisher.to(device=param.device, dtype=param.dtype)
            delta2 = (param - ref_param.to(device=param.device, dtype=param.dtype)).detach().float().pow(2)
            high_mask = fisher >= q80
            low_mask = fisher <= q20

            if high_mask.any():
                high_energy += delta2[high_mask].sum().item()
                high_count += int(high_mask.sum().item())
            if low_mask.any():
                low_energy += delta2[low_mask].sum().item()
                low_count += int(low_mask.sum().item())
            total_energy += delta2.sum().item()
            total_count += delta2.numel()

    high_mean = high_energy / max(high_count, 1)
    low_mean = low_energy / max(low_count, 1)
    total_mean = total_energy / max(total_count, 1)

    return {
        'fisher_param_drift_high_energy': high_energy,
        'fisher_param_drift_low_energy': low_energy,
        'fisher_param_drift_high_mean': high_mean,
        'fisher_param_drift_low_mean': low_mean,
        'fisher_param_drift_mean_weighted': total_mean,
        'fisher_param_drift_high_low_ratio': high_mean / (low_mean + 1e-30),
    }


def _predict_stage2_probe_x0(model, map_input, z_t, t, r, start_normalized, goal_normalized, prediction_type):
    """用当前模型输出构造轻量 x0 预测探针。"""
    model_output = model(map_input, z_t, t, r, start_normalized, goal_normalized)
    t_v = t.view(-1, 1, 1)
    if prediction_type == 'epsilon':
        return (z_t - t_v * model_output) / (1.0 - t_v + 1e-5)
    if prediction_type == 'x0':
        return model_output
    if prediction_type == 'v':
        return z_t - t_v * model_output
    return model_output


def compute_stage2_output_probe_metrics(model, dataloader, device, prediction_type='x0', max_batches=2):
    """
    轻量输出探针：
    - drift: 同一 z,t,r 下，当前模型与 Stage1 锚点模型的 x0 预测 MSE
    - diversity: 同一条件下，两组不同噪声输入得到的 x0 预测 MSE
    """
    proxy_state = _get_stage2_fisher_proxy_state()
    ref_params = proxy_state.get('ref_params')
    if ref_params is None:
        return {}

    base_model = model.module if hasattr(model, 'module') else model
    trainable_params = [p for p in base_model.parameters() if p.requires_grad]
    if len(trainable_params) != len(ref_params):
        return {}

    was_training = model.training
    model.eval()
    drift_sum = 0.0
    diversity_sum = 0.0
    batches = 0

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break

                map_input = batch['map'].float().to(device)
                trajectory = batch['trajectory'].to(device)
                start_pose = batch['start_pose'].to(device)
                goal_pose = batch['goal_pose'].to(device)
                batch_size_local = map_input.shape[0]

                middle_cp = trajectory_to_control_points(trajectory, num_middle_points=24)
                middle_cp_normalized = torch.clamp(middle_cp / 20.0, -1.0, 1.0)

                start_normalized = torch.zeros(batch_size_local, 4, device=device)
                start_normalized[:, :2] = torch.clamp(start_pose[:, :2] / 20.0, -1.0, 1.0)
                start_normalized[:, 2] = torch.cos(start_pose[:, 2])
                start_normalized[:, 3] = torch.sin(start_pose[:, 2])

                goal_normalized = torch.zeros(batch_size_local, 4, device=device)
                goal_normalized[:, :2] = torch.clamp(goal_pose[:, :2] / 20.0, -1.0, 1.0)
                goal_normalized[:, 2] = torch.cos(goal_pose[:, 2])
                goal_normalized[:, 3] = torch.sin(goal_pose[:, 2])

                t = base_model.sample_timesteps(batch_size_local, device=device)
                t = torch.clamp(t, min=1e-4, max=1.0 - 1e-4)
                r = torch.rand_like(t) * t
                noise_1 = torch.randn_like(middle_cp_normalized)
                noise_2 = torch.randn_like(middle_cp_normalized)
                t_v = t.view(-1, 1, 1)
                z_1 = (1.0 - t_v) * middle_cp_normalized + t_v * noise_1
                z_2 = (1.0 - t_v) * middle_cp_normalized + t_v * noise_2

                pred_current_1 = _predict_stage2_probe_x0(
                    model, map_input, z_1, t, r, start_normalized, goal_normalized, prediction_type
                )
                pred_current_2 = _predict_stage2_probe_x0(
                    model, map_input, z_2, t, r, start_normalized, goal_normalized, prediction_type
                )
                diversity_sum += F.mse_loss(pred_current_1, pred_current_2).item()

                current_params = [p.detach().clone() for p in trainable_params]
                try:
                    for param, ref_param in zip(trainable_params, ref_params):
                        param.copy_(ref_param.to(device=param.device, dtype=param.dtype))

                    pred_ref_1 = _predict_stage2_probe_x0(
                        model, map_input, z_1, t, r, start_normalized, goal_normalized, prediction_type
                    )
                finally:
                    for param, current_param in zip(trainable_params, current_params):
                        param.copy_(current_param)

                drift_sum += F.mse_loss(pred_current_1, pred_ref_1).item()
                batches += 1
    finally:
        model.train(was_training)

    if batches == 0:
        return {}

    return {
        'fisher_output_drift_x0_mse': drift_sum / batches,
        'fisher_output_diversity_x0_mse': diversity_sum / batches,
    }



def stage2_optimize_control_points(
    model, map_input, start_normalized, goal_normalized, start_pose, goal_pose,
    stability_cost_map, map_info, device, prediction_type='epsilon', num_iterations=10, lr=0.01
):
    """
    【第二阶段优化】对每个batch样本进行采样和优化，返回1条优化的控制点
    
    关键：保证对每个样本返回形状(B, 24, 2)的优化中间控制点
    
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
        prediction_type: 预测类型
        num_iterations: 优化迭代次数
        lr: 学习率
    
    Returns:
        optimized_middle_cp: (B, 24, 2) 优化后的中间控制点，每个样本一条
    """
    from grad_optimizer import cost_on_dense_trajectory
    
    # 从 start_pose 推导 batch size
    B = start_pose.shape[0]
    
    # B样条层
    bspline_layer = DifferentiableBSpline(
        num_control_points=26,
        num_output_points=100,
        degree=3
    ).to(device)
    
    with torch.enable_grad():
        # 【优化】整个batch采样和优化
        with torch.no_grad():
            sampled_control_points = model.sample(
                map_input,  # (B, C, H, W)
                start_normalized,  # (B, 4)
                goal_normalized,  # (B, 4)
                num_samples=1,
                num_steps=3,
                solver='pmf_refined',
                reconstruct_trajectory=False,
                num_traj_points=100
            )  # (B, 26, 2)
        
        # 提取中间控制点（去掉起终点）
        x0_sample = sampled_control_points[:, 1:-1, :].clone().detach()  # (B, 24, 2)
        x0_sample.requires_grad_(True)
        
        # 为整个batch创建优化器
        optimizer = torch.optim.AdamW([x0_sample], lr=lr)
        
        # 起点和终点
        start_cp = start_pose[:, :2].unsqueeze(1)  # (B, 1, 2)
        goal_cp = goal_pose[:, :2].unsqueeze(1)    # (B, 1, 2)
        
        # batch级优化循环
        for iter in range(num_iterations):
            optimizer.zero_grad()
            
            # 拼接完整控制点（整个batch）
            full_control_points = torch.cat([
                start_cp,       # (B, 1, 2)
                x0_sample,      # (B, 24, 2)
                goal_cp         # (B, 1, 2)
            ], dim=1)  # (B, 26, 2)
            
            # 使用B样条重建轨迹
            reconstructed_traj = bspline_layer(full_control_points)  # (B, 100, 2)
            
            # 计算cost（会对整个batch求均值）
            cost = cost_on_dense_trajectory(
                reconstructed_traj, 
                start_pose,  # (B, 3)
                goal_pose,   # (B, 3)
                stability_cost_map, 
                map_info, 
                device
            )  # 标量
            
            # 反向传播
            cost.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_([x0_sample], 1.0)
            
            # 优化步
            optimizer.step()
        
        # 返回优化后的控制点
        optimized_middle_cp = x0_sample.detach()  # (B, 24, 2)
    
    return optimized_middle_cp



def diffusion_loss(model, batch, device, loss_weights=None, epoch=0, total_epochs=100, is_training=True, current_stage=1, use_dense_trajectory=False, num_dense_points=100, prediction_type='x0'):
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
        loss_weights: dict with keys ['main', 'smoothness', 'curvature', 'angle_smoothness', 'angle_consistency', 'uniformity', 'capsize']
        epoch: 当前epoch（用于动态权重）
        current_stage: 当前训练阶段（1或2），只在阶段2执行高级损失计算
        total_epochs: 总epoch数（用于动态权重）
        is_training: bool - True时使用回归损失，False时返回实际cost值
        use_dense_trajectory: bool - True时使用密集轨迹计算主损失，False时使用原始控制点
        num_dense_points: int - 密集轨迹的点数（仅在use_dense_trajectory=True时使用）
        prediction_type: str - 预测类型 ('epsilon', 'x0', 'v')
    """
    # 默认权重
    if loss_weights is None:
        loss_weights = {
            'main': 1.0,
            'smoothness': 0.1,
            'curvature': 0.0,
            'angle_smoothness': 0.05,
            'angle_consistency': 0.05,
            'uniformity': 0.01,
            'capsize': 0.0,
            'consistency': 0.1  # 时间一致性损失权重
        }
    map_input = batch['map'].float().to(device)
    trajectory = batch['trajectory'].to(device)  # (B, 100, 3)
    start_pose = batch['start_pose'].to(device)  # (B, 3)
    goal_pose = batch['goal_pose'].to(device)  # (B, 3)
    
    B = map_input.shape[0]
    
    # =================== B样条控制点转换 ===================
    # 【语义说明】
    # - 完整B样条：26个控制点 = 起点(1) + 中间点(24) + 终点(1)
    # - 起点/终点：固定的，从start_pose/goal_pose提取
    # - 中间24个点：网络预测的自由控制点
    # 
    # 【第二阶段特殊处理】
    # - 第一阶段（current_stage==1）：仅使用模仿学习数据
    # - 第二阶段（current_stage==2）：在损失层面混合模仿数据和优化数据
    #   * 分别计算两个loss：loss_imitation 和 loss_optimized
    #   * 加权组合：total_loss = alpha * loss_imitation + (1-alpha) * loss_optimized
    #   * 保留两种数据的特性，防止模态坍塌
    
    # 始终获取模仿学习的中间控制点
    middle_cp_imitation = trajectory_to_control_points(trajectory, num_middle_points=24)  # (B, 24, 2)
    
    # 在第二阶段，也可扩展获取优化数据（当前未启用）
    middle_cp_optimized = None
    # if current_stage == 2:
    #     # 检查是否有稳定性代价地图可用
    #     if 'cost_map' in batch and loss_weights.get('capsize', 0.0) > 0:
    #         # 准备归一化的起点和终点（仅用于优化函数）
    #         start_normalized_temp = torch.zeros(B, 4, device=device)
    #         start_normalized_temp[:, :2] = start_pose[:, :2] / 20.0
    #         start_normalized_temp[:, 2] = torch.cos(start_pose[:, 2])
    #         start_normalized_temp[:, 3] = torch.sin(start_pose[:, 2])
    #         start_normalized_temp[:, :2] = torch.clamp(start_normalized_temp[:, :2], -1.0, 1.0)
            
    #         goal_normalized_temp = torch.zeros(B, 4, device=device)
    #         goal_normalized_temp[:, :2] = goal_pose[:, :2] / 20.0
    #         goal_normalized_temp[:, 2] = torch.cos(goal_pose[:, 2])
    #         goal_normalized_temp[:, 3] = torch.sin(goal_pose[:, 2])
    #         goal_normalized_temp[:, :2] = torch.clamp(goal_normalized_temp[:, :2], -1.0, 1.0)
            
    #         # 执行采样和优化，获取优化数据
    #         middle_cp_optimized = stage2_optimize_control_points(
    #             model=model,
    #             map_input=map_input,
    #             start_normalized=start_normalized_temp,
    #             goal_normalized=goal_normalized_temp,
    #             start_pose=start_pose,
    #             goal_pose=goal_pose,
    #             stability_cost_map=batch['cost_map'].to(device)[0],
    #             map_info={
    #                 'resolution': 0.4,
    #                 'origin': (-20.0, -20.0, -np.pi),
    #                 'size': (100, 100, 36)
    #             },
    #             device=device,
    #             prediction_type=prediction_type,
    #             num_iterations=10,
    #             lr=0.1
    #         )  # (B, 24, 2) 优化后的控制点
    
    # 使用模仿数据作为主要的中间控制点
    middle_cp = middle_cp_imitation
    
    # 【第二阶段特殊处理】在batch维度堆叠模仿数据和优化数据
    # 这样可以在一次前向传播中同时计算两个损失
    stage2_mix_loss = False
    # if current_stage == 2 and middle_cp_optimized is not None:
    #     # 堆叠：(B, 24, 2) + (B, 24, 2) -> (2B, 24, 2)
    #     middle_cp = torch.cat([middle_cp_imitation, middle_cp_optimized], dim=0)
        
    #     # 同时扩展其他条件
    #     map_input = torch.cat([map_input, map_input], dim=0)
    #     start_pose = torch.cat([start_pose, start_pose], dim=0)
    #     goal_pose = torch.cat([goal_pose, goal_pose], dim=0)
    #     trajectory = torch.cat([trajectory, trajectory], dim=0)
        
    #     B = B * 2  # 更新batch size
    #     stage2_mix_loss = True
    
    # 归一化控制点：坐标范围通常在 [-20, 20]，归一化到 [-1, 1]
    middle_cp_normalized = middle_cp / 20.0  # (B, 24, 2) 或 (2B, 24, 2)
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
    
    # ===== pixel Mean Flow: 连续时间采样 =====
    # Stage 1 保持原始训练分布；Stage 2 固定为 one-step endpoint，
    # 让安全微调同时把网络推向 t=1,r=0 的单步生成器。
    if current_stage == 2:
        t = torch.ones(B, device=device)
        r = torch.zeros(B, device=device)
    else:
        base_model = model.module if hasattr(model, 'module') else model
        t1 = base_model.sample_timesteps(B, device=device)
        t2 = base_model.sample_timesteps(B, device=device)

        t = torch.maximum(t1, t2)
        r = torch.minimum(t1, t2)
        t = torch.clamp(t, min=1e-4, max=1.0 - 1e-4) # 避免极端值
        r = torch.clamp(r, min=1e-4, max=1.0 - 1e-4) # 确保 r <= t
    noise_cp = torch.randn_like(middle_cp_normalized)

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
    target_v = noise_cp - middle_cp_normalized # dz/dt 的真值

    # noisy_cp 必须在 requires_grad 环境下生成，确保导数链条完整
    with torch.enable_grad():
        t.requires_grad_(True)
        t_v = t.view(-1, 1, 1)
        # 显式重算 noisy_cp 确保它是 t 的函数
        z_t = (1.0 - t_v) * middle_cp_normalized + t_v * noise_cp
        
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
        main_loss_all = F.mse_loss(V_theta, target_v, reduction='none')  # (B, 24, 2)
    elif loss_type == 'x0':
        # x0_rec = z_t - t * V_theta
        pred_x0_corrected = z_t - t.view(-1, 1, 1) * V_theta
        main_loss_all = F.mse_loss(pred_x0_corrected, middle_cp_normalized, reduction='none')  # (B, 24, 2)
    else:
        # epsilon 空间的 pMF 修正写法：pred_eps_corrected = V_theta + pred_x0_corrected
        # 但推荐统一使用 v-loss 以符合论文实现
        main_loss_all = F.mse_loss(V_theta, target_v, reduction='none') # (B, 24, 2)
    
    # 【第二阶段特殊处理】分离模仿和优化两部分的损失，然后加权
    if stage2_mix_loss:
        # 分离损失：(2B, 24, 2) -> (B, 24, 2) + (B, 24, 2)
        main_loss_imitation = main_loss_all[:len(main_loss_all)//2].mean()  # 前半部分
        main_loss_optimized = main_loss_all[len(main_loss_all)//2:].mean()   # 后半部分
        
        # # 根据训练进度动态调整混合系数
        # # 早期更多使用模仿损失，保持在原分布附近
        # # 后期逐渐增加优化损失的比例
        # progress = epoch / (total_epochs + 1e-8)
        # # alpha从0.8衰减到0.2：模仿损失比例从80%到20%
        # alpha_loss = 0.8 - 0.6 * progress  # 范围 [0.8, 0.2]
        
        alpha_loss = 0.98
        
        # 加权组合两个损失
        main_loss = alpha_loss * main_loss_imitation + (1.0 - alpha_loss) * main_loss_optimized
    else:
        # 第一阶段或第二阶段但没有优化数据：只用模仿数据
        main_loss = main_loss_all.mean()
        
    # if loss_weights['main'] <= 0.0:
    #     # main_loss = main_loss * 0.0
    #     main_loss = torch.zeros_like(main_loss)
        
    # main_loss = main_loss * 1e-1
    
    # =================== 计算辅助损失（可选） ===================
    # 注意：由于我们现在只预测控制点，辅助损失的计算需要先重建轨迹
    
    # 初始化所有损失为零tensor（保持梯度连接）
    dummy_loss = main_loss.sum() * 0.0
    smoothness_loss = dummy_loss.clone()
    curvature_loss = dummy_loss.clone()
    angle_smoothness_loss = dummy_loss.clone()
    angle_consistency_loss = dummy_loss.clone()
    uniformity_loss = dummy_loss.clone()
    capsize_loss = dummy_loss.clone()
    capsize_safe_loss = dummy_loss.clone()
    capsize_proxy_penalty = dummy_loss.clone()
    fisher_diag_mean = dummy_loss.clone()
    param_drift_mean = dummy_loss.clone()
    tangent_loss = dummy_loss.clone()
    consistency_loss = dummy_loss.clone()
    
    if loss_weights['tangent'] > 0.0:
        tangent_loss = compute_tangent_loss(middle_cp_normalized, start_normalized, goal_normalized)
    
    if loss_weights['capsize'] > 0.0:
        # =================== 第二阶段：物理约束 ===================
        # 训练阶段使用“当前前向预测的 x0”重建轨迹，避免随机采样链的高方差导致
        # 任一单项辅助损失都把分布推向单模态（与条件解绑）。
        if is_training:
            # 从当前训练图得到 x0 预测（归一化坐标），保持与当前条件和时间步一致
            # x0_middle_pred = z_t - t.view(-1, 1, 1) * V_theta  # (B, 24, 2)
            x0_middle_pred = z_t - t.view(-1, 1, 1) * u_out  # (B, 24, 2)
            x0_middle_denorm = torch.clamp(x0_middle_pred, -1.0, 1.0) * 20.0

            # 组合完整控制点并重建密集轨迹
            start_cp = start_pose[:, :2].unsqueeze(1)  # (B,1,2)
            goal_cp = goal_pose[:, :2].unsqueeze(1)    # (B,1,2)
            full_ctrl_points = torch.cat([start_cp, x0_middle_denorm, goal_cp], dim=1)  # (B,26,2)

            bspline_layer = DifferentiableBSpline(
                num_control_points=24 + 2,
                num_output_points=100,
                degree=3
            ).to(device)
            reconstructed_traj = bspline_layer(full_ctrl_points)  # (B,100,2)
            
            # reconstructed_traj = model.sample_differentiable(
            #     map_input,
            #     start_normalized,
            #     goal_normalized,
            #     num_steps=3,
            #     solver='pmf_refined',
            #     reconstruct_trajectory=True,
            #     num_traj_points=100
            # )  # (B, 100, 2) - 已经是真实坐标（非归一化）

            start_pose_expanded = start_pose
            goal_pose_expanded = goal_pose
        else:
            # 验证阶段保留采样链评估，更接近推理分布
            reconstructed_traj = model.sample(
                map_input,
                start_normalized,
                goal_normalized,
                num_samples=10,
                num_steps=3,
                # solver='pmf_refined',
                solver='pmf_onestep',
                reconstruct_trajectory=True,
                num_traj_points=100
            )
            # model.sample 在 num_samples>1 时返回 (B*num_samples, N, 2)
            start_pose_expanded = start_pose.repeat_interleave(10, dim=0)
            goal_pose_expanded = goal_pose.repeat_interleave(10, dim=0)
        
        # # reconstructed_traj = model.sample_differentiable(
        # #     map_input,
        # #     start_normalized,
        # #     goal_normalized,
        # #     num_steps=3,
        # #     solver='pmf_refined',
        # #     reconstruct_trajectory=True,
        # #     num_traj_points=100
        # # )  # (B, N, 2) - 只有(x,y)，已经是真实坐标（非归一化）
        
        # 使用整批 cost_map，避免错误地只取 batch[0]
        stability_cost_map = batch['cost_map'].to(device)
        map_size = (100, 100, 36) # W, H, D for (x, y, yaw)
        resolution = 0.4
        origin = (-20.0, -20.0, -np.pi) # x, y, yaw
        map_info = {
            'resolution': resolution,
            'origin': origin,
            'size': map_size
        }
        
        # # # =================== 新方法：多步优化 + MSE损失 ===================
        # # if is_training:
        # #     # 训练模式：使用多步优化 + MSE损失
        # #     from grad_optimizer import optimize_control_points_multistep
            
        # #     # 对网络预测的控制点进行多步优化
        # #     optimized_middle_cp, _ = optimize_control_points_multistep(
        # #         middle_control_points=x0_middle_denorm,  # (B, 24, 2)
        # #         start_pose=start_pose,
        # #         goal_pose=goal_pose,
        # #         stability_cost_map=stability_cost_map,
        # #         map_info=map_info,
        # #         iterations=10,
        # #         lr=0.1,
        # #         grad_clip_norm=1.0,
        # #         device=device,
        # #         verbose=False
        # #     )
            
        # #     # 计算MSE损失：网络预测的控制点 vs 优化后的控制点
        # #     capsize_loss = F.mse_loss(x0_middle_denorm, optimized_middle_cp.detach())
            
        # # else:
        # #     # 验证模式：使用原来的cost计算方式
        # #     from grad_optimizer import cost_on_dense_trajectory
        # #     capsize_loss = cost_on_dense_trajectory(
        # #         reconstructed_traj, start_pose, goal_pose,
        # #         stability_cost_map, map_info, device
        # #     )
        # if is_training:
        #     from grad_optimizer import cost_on_dense_trajectory_phr_alm

        #     # 持久化 PHR-ALM 对偶变量（跨 batch）
        #     if not hasattr(diffusion_loss, '_phr_alm_state'):
        #         diffusion_loss._phr_alm_state = {
        #             'lambda_ineq': None,
        #             'lambda_eq': None,
        #             'mu': 1.0,
        #         }

        #     phr_state = diffusion_loss._phr_alm_state
        #     capsize_loss, _, lambda_ineq_new, lambda_eq_new, mu_new = cost_on_dense_trajectory_phr_alm(
        #         reconstructed_traj,
        #         start_pose_expanded,
        #         goal_pose_expanded,
        #         stability_cost_map,
        #         map_info,
        #         lambda_ineq=phr_state['lambda_ineq'],
        #         lambda_eq=phr_state['lambda_eq'],
        #         mu=phr_state['mu'],
        #         update_dual=True,
        #         device=device
        #     )

        #     phr_state['lambda_ineq'] = lambda_ineq_new
        #     phr_state['lambda_eq'] = lambda_eq_new
        #     phr_state['mu'] = mu_new
            
        #     capsize_loss = capsize_loss * 1e-8
        # else:
        #     from grad_optimizer import cost_on_dense_trajectory
        #     capsize_loss = cost_on_dense_trajectory(
        #         reconstructed_traj, start_pose_expanded, goal_pose_expanded,
        #         stability_cost_map, map_info, device
        #     )
    
        from grad_optimizer import cost_on_dense_trajectory
        safe_loss = cost_on_dense_trajectory(
            reconstructed_traj, start_pose_expanded, goal_pose_expanded,
            stability_cost_map, map_info, device
        )

        # 数值稳定性保护：capsize_loss 出现 NaN/Inf 时回退为 0（跳过该分量）
        if not torch.isfinite(safe_loss):
            print("⚠ Warning: capsize_loss is NaN/Inf, fallback to 0 for this batch")
            safe_loss = dummy_loss.clone()

        # =========================================================
        # 第二阶段：Fisher 感知梯度更新
        #   使用“阶段切换时”预估的 PMF-FIM（固定不变）。
        #   这里不再在 batch 内重估 Fisher，只记录固定 Fisher 统计量。
        # =========================================================
        capsize_safe_loss = safe_loss
        capsize_loss = safe_loss
        if is_training:
            proxy_state = _get_stage2_fisher_proxy_state()
            base_model = model.module if hasattr(model, 'module') else model
            trainable_params = [p for p in base_model.parameters() if p.requires_grad]

            # 首次进入第二阶段时，初始化参考参数（用于监控参数漂移）
            if proxy_state['ref_params'] is None or len(proxy_state['ref_params']) != len(trainable_params):
                proxy_state['ref_params'] = [p.detach().clone() for p in trainable_params]
                proxy_state['fisher_diag'] = [torch.ones_like(p) for p in trainable_params]

            # Fisher 统计使用阶段切换时固定值，不在 stage2 训练中更新
            fisher_diag_mean_fixed = proxy_state.get('fisher_diag_mean_fixed', None)
            if fisher_diag_mean_fixed is not None:
                fisher_diag_mean = torch.tensor(float(fisher_diag_mean_fixed), device=device)
            else:
                fisher_diag_mean = dummy_loss.clone()

            drift_means = []
            for param, ref_param in zip(trainable_params, proxy_state['ref_params']):
                delta = param - ref_param
                drift_means.append(delta.pow(2).mean())

            if len(drift_means) > 0:
                param_drift_mean = torch.stack(drift_means).mean()
            else:
                param_drift_mean = dummy_loss.clone()

            # 【重要】不再添加 Fisher 惩罚项到 loss
            # 而是让优化器在更新步骤中应用 Fisher 缩放
            capsize_proxy_penalty = dummy_loss.clone()
            capsize_loss = safe_loss

    
    # 占位符监控指标
    angle_range_ratio = 1.0
    mean_norm_sq = 1.0
    angle_norm_error = torch.tensor(0.0, device=device)
    main_loss_pos = main_loss
    main_loss_ang = torch.tensor(0.0, device=device)
    
    # 如果需要计算辅助损失（例如平滑性），可以从控制点重建轨迹
    # 但为了训练效率，暂时跳过
    skip_angle_losses = True  # 控制点阶段跳过角度相关损失
    
    # main_loss 数值稳定性保护（尤其在 stage2 main=0 时避免无关分支污染）
    if isinstance(main_loss, torch.Tensor) and (not torch.isfinite(main_loss)):
        print("⚠ Warning: main_loss is NaN/Inf, fallback to 0 for this batch")
        main_loss = dummy_loss.clone()

    # 混合损失
    if loss_weights['main'] <= 0.0:
        total_loss = loss_weights['tangent'] * tangent_loss \
                    + loss_weights['capsize'] * capsize_loss
    else:
        total_loss = loss_weights['main'] * main_loss \
                    + loss_weights['tangent'] * tangent_loss \
                    + loss_weights['capsize'] * capsize_loss
    # 检查损失异常：不中断训练，回退为零损失并跳过本 batch 更新
    if not torch.isfinite(total_loss):
        total_loss_val = total_loss.item() if isinstance(total_loss, torch.Tensor) else float('nan')
        main_loss_val = main_loss.item() if isinstance(main_loss, torch.Tensor) else float('nan')
        print(f"⚠ Warning: total_loss is NaN/Inf ({total_loss_val}), main_loss={main_loss_val}. Skip this batch.")
        total_loss = dummy_loss.clone()
    
    # 返回各项损失用于记录
    loss_dict = {
        'main': main_loss.item(),
        'smoothness': smoothness_loss.item(),
        'curvature': curvature_loss.item(),
        'angle_smoothness': angle_smoothness_loss.item(),
        'angle_consistency': angle_consistency_loss.item(),
        'uniformity': uniformity_loss.item(),
        'tangent': tangent_loss.item(),
        'capsize': capsize_loss.item(),
        'capsize_safe': capsize_safe_loss.item(),
        'capsize_proxy_penalty': capsize_proxy_penalty.item(),
        'fisher_diag_mean': fisher_diag_mean.item(),
        'param_drift_mean': param_drift_mean.item(),
        'consistency': consistency_loss.item(),
        'angle_norm_mean': mean_norm_sq ** 0.5,
        'angle_norm_error': angle_norm_error.item() if isinstance(angle_norm_error, torch.Tensor) else 0.0,
        'main_loss_pos': main_loss_pos.item() if isinstance(main_loss_pos, torch.Tensor) else 0.0,
        'main_loss_ang': main_loss_ang.item() if isinstance(main_loss_ang, torch.Tensor) else 0.0,
    }
    
    # 调整返回的样本数：如果在第二阶段进行了混合，返回原始batch size
    n_samples = B // 2 if stage2_mix_loss else B
    
    return total_loss, 0, n_samples, loss_dict

def train_epoch(model, trainingData, optimizer, device, stage_epoch=0, loss_weights=None, current_stage=1, total_stage_epochs=50, ema_models=None, use_dense_trajectory=False, num_dense_points=100):
    """
    单轮训练函数
    
    Args:
        ema_models: EMA模型列表，用于更新EMA参数（可选）
        use_dense_trajectory: bool - 是否使用密集轨迹计算主损失
        num_dense_points: int - 密集轨迹的点数
    """
    model.train()
    total_loss = 0
    total_samples = 0
    fisher_optimizer = optimizer._optimizer if isinstance(getattr(optimizer, '_optimizer', None), FisherAdamW) else None
    if current_stage == 2 and fisher_optimizer is not None:
        fisher_optimizer.reset_diagnostics()
    
    # 累积各项损失
    loss_accumulator = {
        'main': 0, 
        'smoothness': 0, 
        'curvature': 0,
        'angle_smoothness': 0, 
        'angle_consistency': 0, 
        'uniformity': 0, 
        'tangent': 0, 
        'capsize': 0, 
        'capsize_safe': 0,
        'capsize_proxy_penalty': 0,
        'fisher_diag_mean': 0,
        'param_drift_mean': 0,
        'consistency': 0,
    }
    
    pbar = tqdm(trainingData, mininterval=2, desc=f"Training Epoch {stage_epoch} (Stage {current_stage})")
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()
        
        loss, _, n_samples, loss_dict = diffusion_loss(
            model, batch, device, loss_weights, 
            epoch=stage_epoch, total_epochs=total_stage_epochs,
            is_training=True,  # 训练模式：capsize loss使用回归损失
            current_stage=current_stage,  # 传递当前阶段
            use_dense_trajectory=use_dense_trajectory,
            num_dense_points=num_dense_points,
            prediction_type=prediction_type
        )

        # 先检查 loss 再统计，避免把 NaN 累积进 epoch 指标
        if not torch.isfinite(loss):
            print("Warning: NaN/Inf loss, skipping batch")
            continue

        total_loss += loss.item()
        total_samples += n_samples
        
        for key, value in loss_dict.items():
            loss_accumulator.setdefault(key, 0)
            loss_accumulator[key] += value
        
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

        # 注意：不在每步刷新参考参数，参考参数在切换到 Stage 2 时固定为
        # 从 Stage 1 加载的初始参数（见切换逻辑处初始化）
        
        # 更新进度条
        grad_info = f'{original_grad_norm:.3e}→{clip_value:.3e}' if was_clipped else f'{original_grad_norm:.3e}'
        
        # 根据当前阶段显示不同的信息
        if current_stage == 1:
            # 阶段1：显示主损失、平滑性
            pbar.set_postfix({
                'Total': f'{loss.item():.5f}',
                'Main': f'{loss_dict["main"]:.5f}',
                # 'Smooth': f'{loss_dict["smoothness"]:.4f}',
                'Tangent': f'{loss_dict["tangent"]:.4f}',
                'GradNorm': grad_info
            })
        else:
            # 阶段2：显示主损失与物理约束损失
            pbar.set_postfix({
                'Total': f'{loss.item():.5f}',
                'Main': f'{loss_dict["main"]:.5f}',
                'Capsize': f'{loss_dict["capsize"]:.4f}',
                'Safe': f'{loss_dict["capsize_safe"]:.4f}',
                'FisherP': f'{loss_dict["capsize_proxy_penalty"]:.3e}',
                'Fdiag': f'{loss_dict["fisher_diag_mean"]:.3e}',
                'Drift': f'{loss_dict["param_drift_mean"]:.3e}',
                'GradNorm': grad_info
            })
    
    avg_loss = total_loss / len(trainingData) if len(trainingData) > 0 else 0
    
    # 计算各项平均损失
    avg_loss_dict = {k: v / len(trainingData) for k, v in loss_accumulator.items()}
    if current_stage == 2 and fisher_optimizer is not None:
        avg_loss_dict.update(fisher_optimizer.diagnostics_summary())
    
    return avg_loss, 0, total_samples, avg_loss_dict


def eval_epoch(model, validationData, device, loss_weights=None, epoch=0, total_epochs=100, current_stage=1, use_late_timesteps=False, use_dense_trajectory=False, num_dense_points=100):
    """
    单轮评估函数
    
    Args:
        use_dense_trajectory: bool - 是否使用密集轨迹计算主损失
        num_dense_points: int - 密集轨迹的点数
    """
    model.eval()
    total_loss = 0
    total_samples = 0
    
    # 累积各项损失
    loss_accumulator = {
        'main': 0, 
        'smoothness': 0, 
        'curvature': 0,
        'angle_smoothness': 0, 
        'angle_consistency': 0, 
        'uniformity': 0, 
        'tangent': 0, 
        'capsize': 0,
        'capsize_safe': 0,
        'capsize_proxy_penalty': 0,
        'fisher_diag_mean': 0,
        'param_drift_mean': 0,
    }
    
    with torch.no_grad():
        for batch in tqdm(validationData, mininterval=2, desc="Validation"):
            
            loss, _, n_samples, loss_dict = diffusion_loss(
                model, batch, device, loss_weights,
                epoch=epoch, total_epochs=total_epochs,
                is_training=False,  # 验证模式：capsize loss返回实际cost值
                current_stage=current_stage,  # 传递当前阶段
                use_dense_trajectory=use_dense_trajectory,
                num_dense_points=num_dense_points,
                prediction_type=prediction_type
            )
                
            total_loss += loss.item()
            total_samples += n_samples
            
            for key, value in loss_dict.items():
                loss_accumulator.setdefault(key, 0)
                loss_accumulator[key] += value
    
    avg_loss = total_loss / len(validationData) if len(validationData) > 0 else 0
    
    # 计算各项平均损失
    avg_loss_dict = {k: v / len(validationData) for k, v in loss_accumulator.items()}
    
    return avg_loss, 0, total_samples, avg_loss_dict

def check_data_folders(folder):
    """检查数据文件夹结构"""
    assert osp.isdir(osp.join(folder, 'train')), "Cannot find training data"  # 检查train子文件夹是否存在
    assert osp.isdir(osp.join(folder, 'val')), "Cannot find validation data"  # 检查val子文件夹是否存在

def print_model_parameters(model, stage_info=""):
    """打印模型参数的训练状态"""
    print(f"{stage_info} - Model Parameter Status:")
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"  ✓ {name}: {param.numel()} parameters (trainable)")
        else:
            print(f"  ✗ {name}: {param.numel()} parameters (frozen)")
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    print(f"  Trainable ratio: {trainable_params/total_params:.1%}")
    print()

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
    parser.add_argument(
        '--stage2_anchor',
        help="Stage 1 anchor checkpoint for rigorous Stage 2 training/resume. "
             "When resuming from a Stage 2 checkpoint, this must point to the fixed Stage 1 reference used for ref_params and PMF-FIM.",
        default=None
    )
    parser.add_argument('--stage', help="Training stage to start from (1 or 2)", type=int, default=1, choices=[1, 2])
    parser.add_argument('--stage1_epochs', help="Number of epochs for stage 1", type=int, default=200)
    parser.add_argument('--stage2_epochs', help="Number of epochs for stage 2", type=int, default=300)
    parser.add_argument(
        '--fisher_mode',
        help="Stage 2 optimizer mode: fisher uses PMF-FIM scaling; none uses unit Fisher for Direct Safe Adam baseline while keeping PMF-FIM diagnostics.",
        type=str,
        default='fisher',
        choices=['fisher', 'none']
    )
    # parser.add_argument('--prediction_type', help="Model prediction type", type=str, default='v', choices=['epsilon', 'x0', 'v'])
    args = parser.parse_args()
    stage2_anchor_path = args.stage2_anchor
    if stage2_anchor_path is not None and not osp.exists(stage2_anchor_path):
        raise ValueError(f"Stage 2 anchor checkpoint not found: {stage2_anchor_path}")

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
        n_path_steps=24,  # 24个B样条控制点 (x,y)
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
    
    # =================== 密集轨迹配置 ===================
    use_dense_trajectory = False  # 是否使用密集轨迹计算主损失
    num_dense_points = 100  # 密集轨迹的点数
    
    print(f"✓ 密集轨迹: {'启用' if use_dense_trajectory else '禁用'}")
    if use_dense_trajectory:
        print(f"  - 密集点数: {num_dense_points}")
    
    # =================== 两阶段训练配置 ===================
    # 阶段1配置：注重基础轨迹预测
    # stage1_config = {
    #     'epochs': args.stage1_epochs,
    #     'lr_mul': 1e-1,  # 学习率倍增器
    #     'loss_weights': {
    #         'smoothness': 1e-5,   # 平滑性损失
    #         'curvature': 0e-4,    # 曲率约束（限制最大曲率）- 新增
    #         'angle_smoothness': 1e-2,  # 角度平滑性损失
    #         'angle_consistency': 0e-6,  # 角度一致性损失(防止倒车)
    #         'uniformity': 0e-4,   # 均匀性损失（点间距离方差）
    #         'sincos_norm': 1e-1,   # sin/cos归一化损失（确保sin²+cos²≈1）
    #         'capsize': 0.0        # 倾覆监督损失（第一阶段不启用）
    #     }
    # }
    stage1_config = {
        'epochs': args.stage1_epochs,
        'lr_mul': 1e-1,  # 学习率倍增器
        'loss_weights': {
            'main': 1e-2,          # 主预测损失
            'smoothness': 0e-5,   # 平滑性损失
            'curvature': 0e-4,    # 曲率约束（限制最大曲率）- 新增
            'angle_smoothness': 0e-2,  # 角度平滑性损失
            'angle_consistency': 0e-6,  # 角度一致性损失(防止倒车)
            'uniformity': 0e-4,   # 均匀性损失（点间距离方差）
            'tangent': 0e3,   # 切线约束损失（确保起点和终点方向一致）
            'capsize': 0.0        # 倾覆监督损失（第一阶段不启用）
        }
    }
    
    # 阶段2配置：通过 one-step endpoint 微调学习低cost轨迹
    # **关键**：diffusion_loss 在 Stage 2 固定 t=1,r=0，使 main loss 起到
    # one-step PMF 蒸馏/模仿正则作用，capsize loss 直接优化单步输出轨迹。
    stage2_config = {
        'epochs': args.stage2_epochs,
        'lr_mul': 1e-3,  # 小学习率微调（不是从头训练！）
        'loss_weights': {
            'main': 0e-2,        # one-step PMF 蒸馏/模仿正则，配合 Stage 2 固定 t=1,r=0
            'smoothness': 0e-5,   # 平滑性损失
            'curvature': 0e-4,    # 曲率约束（限制最大曲率）- 新增
            'angle_smoothness': 0e-2,  # 角度平滑性损失
            'angle_consistency': 0e-6,  # 角度一致性损失
            'uniformity': 0e-4,   # 均匀性损失
            'tangent': 0e-2,   # 切线约束损失（确保起点和终点方向一致）
            'capsize': 1e-2,
            'consistency': 0.0,   # 时间一致性损失（当前不启用）
        }
    }
    
    # 初始化当前阶段（在数据加载前）
    current_stage = args.stage
    
    # 根据当前阶段决定是否计算stability map（第二阶段需要倾覆损失时启用）
    if current_stage == 1:
        loss_weights = stage1_config['loss_weights']
    else:
        loss_weights = stage2_config['loss_weights']

    if current_stage == 2 and loss_weights.get('main', 0.0) <= 0.0:
        print("⚠ Warning: Stage 2 main loss weight <= 0, training is capsize-only and may collapse.")
    
    compute_stability = (
        current_stage == 2 and (
            loss_weights.get('capsize', 0.0) > 0
        )
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
    print(f"  fisher_mode: {args.fisher_mode}")
    print(f"\n当前起始阶段: {current_stage}")
    print()
    
    # 保存模型配置
    config = {
        'model_args': model_args,
        'stage1_config': stage1_config,
        'stage2_config': stage2_config,
        'fisher_mode': args.fisher_mode,
        'stage2_anchor_path': stage2_anchor_path,
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
        # 阶段1：全参数训练（包括map cross-attention）
        if isinstance(model, nn.DataParallel):
            model.module.unfreeze_for_stage1()
        else:
            model.unfreeze_for_stage1()
    else:
        # 阶段2：只fine-tune map cross-attention + 主预测头
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

    def _load_model_state_dict_from_checkpoint_path(checkpoint_path):
        """只加载模型参数，不改变训练控制流。"""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' not in checkpoint:
            raise ValueError(f"Invalid checkpoint format: {checkpoint_path}")
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint

    def _clone_model_state_dict_to_cpu():
        base_model = model.module if isinstance(model, nn.DataParallel) else model
        return {name: value.detach().cpu().clone() for name, value in base_model.state_dict().items()}

    def _restore_model_state_dict_from_cpu(state_dict):
        base_model = model.module if isinstance(model, nn.DataParallel) else model
        base_model.load_state_dict(state_dict)

    def _setup_stage2_fixed_fisher_and_optimizer(current_optimizer, anchor_checkpoint_path=None):
        """在 Stage 2 启动时一次性估计固定 PMF-FIM，并切换到 FisherAdamW。"""
        proxy_state = _get_stage2_fisher_proxy_state()

        # 已经是 FisherAdamW 且已有固定 Fisher 时，直接复用
        if isinstance(current_optimizer._optimizer, FisherAdamW) and proxy_state.get('fisher_diag_precomputed') is not None:
            print("✓ Stage 2 fixed Fisher already initialized, skip re-estimation")
            return current_optimizer

        restore_state = None
        if anchor_checkpoint_path is not None:
            print(f"✓ Loading Stage 2 anchor for fixed PMF-FIM/reference: {anchor_checkpoint_path}")
            restore_state = _clone_model_state_dict_to_cpu()
            anchor_checkpoint = _load_model_state_dict_from_checkpoint_path(anchor_checkpoint_path)
            anchor_stage = anchor_checkpoint.get('stage', None)
            if anchor_stage is not None and anchor_stage != 1:
                print(f"⚠ Warning: Stage 2 anchor checkpoint reports stage={anchor_stage}, expected stage=1")

        if isinstance(model, nn.DataParallel):
            stage2_anchor_params = model.module.get_trainable_parameters()
        else:
            stage2_anchor_params = model.get_trainable_parameters()

        proxy_state['ref_params'] = [p.detach().clone() for p in stage2_anchor_params]
        proxy_state['fisher_diag'] = [torch.ones_like(p) for p in stage2_anchor_params]

        def compute_main_loss_fn(model_for_fisher, batch_for_fisher):
            """用于 Fisher 估计的 main_loss：与第一阶段训练目标保持一致。"""
            fisher_loss_weights = {
                'main': 1.0,
                'smoothness': 0.0,
                'curvature': 0.0,
                'angle_smoothness': 0.0,
                'angle_consistency': 0.0,
                'uniformity': 0.0,
                'tangent': 0.0,
                'capsize': 0.0,
                'consistency': 0.0,
            }
            fisher_main_loss, _, _, _ = diffusion_loss(
                model_for_fisher,
                batch_for_fisher,
                device,
                loss_weights=fisher_loss_weights,
                epoch=0,
                total_epochs=1,
                is_training=True,
                current_stage=1,
                use_dense_trajectory=False,
                num_dense_points=100,
                prediction_type=prediction_type,
            )
            return fisher_main_loss

        print("📊 Stage 2 startup: estimating fixed PMF-FIM from stage1 main loss...")
        fisher_f_min = 0.3  # 温和的下界（后续可改为 0.1）
        fisher_f_max = 3.0  # 温和的上界（后续可改为 10.0）
        fisher_dataset = UnevenPathDataLoader(
            env_list=env_list,
            dataFolder=osp.join(dataFolder, 'train'),
            compute_stability_map=False
        )
        fisher_loader = DataLoader(
            fisher_dataset,
            num_workers=15,
            collate_fn=PaddedSequence,
            batch_size=batch_size,
            shuffle=True
        )

        fisher_diag = estimate_diag_fisher_from_main_loss(
            model=model,
            dataloader=fisher_loader,
            compute_loss_fn=compute_main_loss_fn,
            device=device,
            max_batches=100,
            f_min=fisher_f_min,
            f_max=fisher_f_max,
        )

        if restore_state is not None:
            _restore_model_state_dict_from_cpu(restore_state)
            print("✓ Restored resume/current model parameters after anchor Fisher estimation")

        proxy_state['fisher_diag_precomputed'] = fisher_diag
        _store_fixed_fisher_stats(proxy_state, fisher_diag, fisher_f_min, fisher_f_max)
        _attach_stage2_fisher_list_to_proxy_state(model, proxy_state, fisher_diag)
        print(
            f"✓ Fixed Fisher stats: min={proxy_state['fisher_diag_min_fixed']:.3e}, "
            f"max={proxy_state['fisher_diag_max_fixed']:.3e}, "
            f"mean={proxy_state['fisher_diag_mean_fixed']:.3e}, "
            f"median={proxy_state['fisher_diag_median_fixed']:.3e}, "
            f"clip@min={proxy_state['fisher_clip_at_min_fixed']:.2%}, "
            f"clip@max={proxy_state['fisher_clip_at_max_fixed']:.2%}"
        )
        proxy_state['fisher_mode'] = args.fisher_mode
        optimizer_fisher_diag = (
            fisher_diag if args.fisher_mode == 'fisher'
            else _make_unit_fisher_diag_like(fisher_diag)
        )

        base_optimizer = FisherAdamW(
            named_params=list(model.named_parameters()),
            fisher_diag=optimizer_fisher_diag,
            lr=1e-4,
            betas=(0.95, 0.999),
            eps=1e-8,
            weight_decay=0.01,
            fisher_eps=1e-8,
            fisher_alpha=1.0,  # 二阶优化：精确 Fisher 逆预条件
            diagnostics_every=10,
            diagnostic_fisher_diag=fisher_diag,
        )
        new_optimizer = Optim.ScheduledOptim(
            base_optimizer,
            lr_mul=stage2_config['lr_mul'],
            d_model=512,
            n_warmup_steps=50,
        )
        if args.fisher_mode == 'fisher':
            print("✓ Stage 2 optimizer switched to FisherAdamW with robust normalized PMF-FIM (alpha=1.0)")
        else:
            print("✓ Stage 2 optimizer switched to Direct Safe Adam baseline (unit Fisher scaling, PMF-FIM diagnostics kept)")
        return new_optimizer

    
    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    stage1_best_loss = float('inf')
    pending_optimizer_state = None
    pending_optimizer_n_steps = None
    
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

        checkpoint_fisher_mode = checkpoint.get('fisher_mode', None)
        if checkpoint_fisher_mode is not None and checkpoint_fisher_mode != args.fisher_mode:
            raise ValueError(
                f"Checkpoint fisher_mode={checkpoint_fisher_mode}, but current --fisher_mode={args.fisher_mode}. "
                "Use the same fisher_mode for rigorous resume."
            )
        checkpoint_anchor_path = checkpoint.get('stage2_anchor_path', None)
        if checkpoint_anchor_path is not None:
            if stage2_anchor_path is None:
                stage2_anchor_path = checkpoint_anchor_path
                print(f"✓ Reusing Stage 2 anchor from checkpoint metadata: {stage2_anchor_path}")
            elif osp.abspath(stage2_anchor_path) != osp.abspath(checkpoint_anchor_path):
                print(
                    f"⚠ Warning: --stage2_anchor differs from checkpoint metadata:\n"
                    f"  checkpoint: {checkpoint_anchor_path}\n"
                    f"  current:    {stage2_anchor_path}"
                )
        
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
            if stage2_anchor_path is None:
                stage2_anchor_path = args.resume
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
            if stage2_anchor_path is None:
                raise ValueError(
                    "Rigorous Stage 2 resume requires --stage2_anchor pointing to the fixed Stage 1 anchor checkpoint. "
                    "Example: --resume checkpoint_stage2_epoch_29.pth --stage2_anchor stage1_best_model.pth"
                )
        
        # 根据恢复的阶段更新优化器学习率
        optimizer.lr_mul = current_lr_mul
        
        # 如果从stage 1切换到stage 2，重新初始化优化器（不加载旧的优化器状态）
        if args.stage == 2 and saved_stage == 1:
            print("Resetting optimizer for stage 2 (fresh start)")
            # 不加载旧的优化器状态，使用全新的优化器
        elif 'optimizer_state_dict' in checkpoint:
            if current_stage == 2:
                pending_optimizer_state = checkpoint['optimizer_state_dict']
                pending_optimizer_n_steps = checkpoint.get('n_steps', None)
                print("✓ Deferred Stage 2 optimizer state loading until FisherAdamW is initialized")
            else:
                try:
                    optimizer._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if 'n_steps' in checkpoint:
                        optimizer.n_steps = checkpoint['n_steps']
                    print(f"✓ Loaded optimizer state (stage {current_stage}, lr_mul={current_lr_mul})")
                except Exception as e:
                    print(f"Warning: Failed to load optimizer state: {e}")

    # 关键兜底：如果训练一开始就在 Stage 2（或恢复到 Stage 2），
    # 也必须初始化固定 PMF-FIM + FisherAdamW，避免 Fdiag=0 和未启用 Fisher 更新。
    if current_stage == 2:
        optimizer = _setup_stage2_fixed_fisher_and_optimizer(optimizer, anchor_checkpoint_path=stage2_anchor_path)
        if pending_optimizer_state is not None:
            try:
                optimizer._optimizer.load_state_dict(pending_optimizer_state)
                if pending_optimizer_n_steps is not None:
                    optimizer.n_steps = pending_optimizer_n_steps
                print(f"✓ Loaded deferred Stage 2 optimizer state (n_steps={optimizer.n_steps})")
            except Exception as e:
                print(f"Warning: Failed to load deferred Stage 2 optimizer state: {e}")
    
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
                if stage2_anchor_path is None:
                    stage2_anchor_path = stage1_best_path
                    print(f"✓ Stage 2 anchor set to Stage 1 best model: {stage2_anchor_path}")
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
            
            # 【关键】阶段2：只训练map cross-attention + 主预测头
            if isinstance(model, nn.DataParallel):
                model.module.freeze_for_stage2()
                trainable_params = model.module.get_trainable_parameters()
            else:
                model.freeze_for_stage2()
                trainable_params = model.get_trainable_parameters()
            
            # 在切换到 Stage 2 时，将参考锚点设置为此处加载的 Stage1 模型参数
            # 使得后续的 Fisher 感知优化以 Stage1 参数为固定参考
            proxy_state = _get_stage2_fisher_proxy_state()
            proxy_state['ref_params'] = [p.detach().clone() for p in trainable_params]
            # 初始化 fisher_diag 为与参数同形状的全 1 张量，随后会从数据估计
            proxy_state['fisher_diag'] = [torch.ones_like(p) for p in trainable_params]

            trainable_count = sum(p.numel() for p in trainable_params)
            print(f"✓ 阶段2训练参数量: {trainable_count:,}")
            
            # 【关键】在数据加载器 reload 之前定义一个计算 main_loss 的函数
            def compute_main_loss_fn(model, batch):
                """用于 Fisher 估计的 main_loss：与第一阶段训练目标保持一致。"""
                fisher_loss_weights = {
                    'main': 1.0,
                    'smoothness': 0.0,
                    'curvature': 0.0,
                    'angle_smoothness': 0.0,
                    'angle_consistency': 0.0,
                    'uniformity': 0.0,
                    'tangent': 0.0,
                    'capsize': 0.0,
                    'consistency': 0.0,
                }
                fisher_main_loss, _, _, _ = diffusion_loss(
                    model,
                    batch,
                    device,
                    loss_weights=fisher_loss_weights,
                    epoch=0,
                    total_epochs=1,
                    is_training=True,
                    current_stage=1,
                    use_dense_trajectory=False,
                    num_dense_points=100,
                    prediction_type=prediction_type,
                )
                return fisher_main_loss
            
            # 【重要】估计 Fisher 对角
            print("📊 正在从第一阶段数据估计 Fisher 对角...")
            # 重新创建数据加载器以启用stability计算
            print("正在加载第一阶段训练数据用于 Fisher 估计...")
            temp_dataset = UnevenPathDataLoader(
                env_list=env_list,
                dataFolder=osp.join(dataFolder, 'train'),
                compute_stability_map=False  # Fisher 估计不需要 stability map
            )
            temp_dataloader = DataLoader(
                temp_dataset, 
                num_workers=15, 
                collate_fn=PaddedSequence, 
                batch_size=batch_size,
                shuffle=True
            )
            
            fisher_diag = estimate_diag_fisher_from_main_loss(
                model=model,
                dataloader=temp_dataloader,
                compute_loss_fn=compute_main_loss_fn,
                device=device,
                max_batches=100,  # 用前100个batch估计，加快估计速度
                f_min=0.3,  # 温和的下界（与 Stage2 启动一致）
                f_max=3.0   # 温和的上界（与 Stage2 启动一致）
            )
            print(f"✓ Fisher 对角估计完成")
            
            # 保存 Fisher 到 proxy_state
            proxy_state['fisher_diag_precomputed'] = fisher_diag
            _store_fixed_fisher_stats(proxy_state, fisher_diag, f_min=0.3, f_max=3.0)
            _attach_stage2_fisher_list_to_proxy_state(model, proxy_state, fisher_diag)
            print(
                f"✓ Fixed Fisher stats: min={proxy_state['fisher_diag_min_fixed']:.3e}, "
                f"max={proxy_state['fisher_diag_max_fixed']:.3e}, "
                f"mean={proxy_state['fisher_diag_mean_fixed']:.3e}, "
                f"median={proxy_state['fisher_diag_median_fixed']:.3e}, "
                f"clip@min={proxy_state['fisher_clip_at_min_fixed']:.2%}, "
                f"clip@max={proxy_state['fisher_clip_at_max_fixed']:.2%}"
            )
            proxy_state['fisher_mode'] = args.fisher_mode
            optimizer_fisher_diag = (
                fisher_diag if args.fisher_mode == 'fisher'
                else _make_unit_fisher_diag_like(fisher_diag)
            )
            
            # 更新优化器为 FisherAdamW
            old_lr_mul = optimizer.lr_mul
            
            # 创建一个简单的学习率调度器 wrapper（在 FisherAdamW 之上）
            base_optimizer = FisherAdamW(
                named_params=list(model.named_parameters()),
                fisher_diag=optimizer_fisher_diag,
                lr=1e-4,  # 初始学习率（会被调度器覆盖）
                betas=(0.95, 0.999),
                eps=1e-8,
                weight_decay=0.01,
                fisher_eps=1e-8,
                fisher_alpha=1.0,  # 二阶优化的标准参数
                diagnostics_every=10,
                diagnostic_fisher_diag=fisher_diag,
            )
            
            # 用 ScheduledOptim wrapper 包装 FisherAdamW
            optimizer = Optim.ScheduledOptim(
                base_optimizer,
                lr_mul=stage2_config['lr_mul'],
                d_model=512,
                n_warmup_steps=50
            )
            if args.fisher_mode == 'fisher':
                print(f"✓ Reset optimizer to FisherAdamW with Fisher-aware scaling (alpha=1.0)")
            else:
                print(f"✓ Reset optimizer to Direct Safe Adam baseline (unit Fisher scaling, PMF-FIM diagnostics kept)")
            print(f"✓ Learning rate: {old_lr_mul} → {stage2_config['lr_mul']}")
            print(f"✓ Enabled capsize loss: {stage2_config['loss_weights']['capsize']}")
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
        
        # 计算当前阶段的总epoch数
        total_stage_epochs = stage1_config['epochs'] if current_stage == 1 else stage2_config['epochs']
        
        # 训练
        train_loss, _, _, train_loss_dict = train_epoch(
            model, trainingData, optimizer, device, stage_epoch, loss_weights, current_stage, total_stage_epochs,
            ema_models=ema_models if use_ema else None, use_dense_trajectory=use_dense_trajectory, num_dense_points=num_dense_points
        )
        
        # 验证（使用主模型）
        val_loss, _, _, val_loss_dict = eval_epoch(
            model, validationData, device, loss_weights, stage_epoch, total_stage_epochs, 
            current_stage=current_stage, use_dense_trajectory=use_dense_trajectory, num_dense_points=num_dense_points
        )
        
        # 验证EMA模型
        ema_val_losses = []
        ema_val_loss_dicts = []
        if use_ema and len(ema_models) > 0:
            for i, ema_m in enumerate(ema_models):
                ema_val_loss, _, _, ema_val_loss_dict = eval_epoch(
                    ema_m.module, validationData, device, loss_weights, stage_epoch, total_stage_epochs,
                    current_stage=current_stage, use_dense_trajectory=use_dense_trajectory, num_dense_points=num_dense_points
                )
                ema_val_losses.append(ema_val_loss)
                ema_val_loss_dicts.append(ema_val_loss_dict)

        stage2_fisher_metrics = {}
        if current_stage == 2:
            stage2_fisher_metrics.update(compute_stage2_fisher_param_diagnostics(model))
            stage2_fisher_metrics.update(
                compute_stage2_output_probe_metrics(
                    model,
                    validationData,
                    device,
                    prediction_type=prediction_type,
                    max_batches=2,
                )
            )
        # 记录
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"\n[Stage {current_stage}] Epoch {stage_epoch}:")
        print(f"  Train Loss: {train_loss:.9f}")
        print(f"    - Main: {train_loss_dict['main']:.6f}")
        # print(f"    - Smoothness: {train_loss_dict['smoothness']:.6f}")
        # print(f"    - Curvature: {train_loss_dict['curvature']:.6f}")
        # print(f"    - Angle Smoothness: {train_loss_dict['angle_smoothness']:.6f}")
        # print(f"    - Angle Consistency: {train_loss_dict['angle_consistency']:.6f}")
        # print(f"    - Uniformity: {train_loss_dict['uniformity']:.6f}")
        print(f"    - Tangent: {train_loss_dict['tangent']:.6f}")
        print(f"    - Capsize: {train_loss_dict['capsize']:.9f}")
        print(f"    - Capsize Safe: {train_loss_dict['capsize_safe']:.9f}")
        print(f"    - Fisher Penalty: {train_loss_dict['capsize_proxy_penalty']:.3e}")
        print(f"    - Fisher Diag Mean: {train_loss_dict['fisher_diag_mean']:.3e}")
        print(f"    - Param Drift Mean: {train_loss_dict['param_drift_mean']:.3e}")
        if current_stage == 2:
            print(
                f"    - Fisher Update High/Low: "
                f"{train_loss_dict.get('fisher_update_high_low_ratio', 0.0):.3e} "
                f"(raw={train_loss_dict.get('fisher_raw_update_high_low_ratio', 0.0):.3e})"
            )
            print(
                f"    - Fisher Drift High/Low: "
                f"{stage2_fisher_metrics.get('fisher_param_drift_high_low_ratio', 0.0):.3e}, "
                f"Output Drift: {stage2_fisher_metrics.get('fisher_output_drift_x0_mse', 0.0):.3e}, "
                f"Diversity: {stage2_fisher_metrics.get('fisher_output_diversity_x0_mse', 0.0):.3e}"
            )
        print(f"  Val Loss:   {val_loss:.9f}")
        print(f"    - Main: {val_loss_dict['main']:.6f}")
        # print(f"    - Smoothness: {val_loss_dict['smoothness']:.6f}")
        # print(f"    - Curvature: {val_loss_dict['curvature']:.6f}")
        # print(f"    - Angle Smoothness: {val_loss_dict['angle_smoothness']:.6f}")
        # print(f"    - Angle Consistency: {val_loss_dict['angle_consistency']:.6f}")
        # print(f"    - Uniformity: {val_loss_dict['uniformity']:.6f}")
        print(f"    - Tangent: {val_loss_dict['tangent']:.6f}")
        print(f"    - Capsize: {val_loss_dict['capsize']:.9f}")
        print(f"    - Capsize Safe: {val_loss_dict['capsize_safe']:.9f}")
        # 打印EMA验证损失
        if use_ema and len(ema_val_losses) > 0:
            for i, (ema_val_loss, decay) in enumerate(zip(ema_val_losses, ema_decays)):
                print(f"  EMA Val Loss (decay={decay}): {ema_val_loss:.6f}")
        
        # TensorBoard - 总损失（使用阶段内epoch）
        writer.add_scalar('Loss/train', train_loss, stage_epoch)
        writer.add_scalar('Loss/val', val_loss, stage_epoch)
        
        # TensorBoard - 各项损失
        writer.add_scalar('Loss/train_main', train_loss_dict['main'], stage_epoch)
        writer.add_scalar('Loss/train_smoothness', train_loss_dict['smoothness'], stage_epoch)
        writer.add_scalar('Loss/train_curvature', train_loss_dict['curvature'], stage_epoch)
        writer.add_scalar('Loss/train_angle_smoothness', train_loss_dict['angle_smoothness'], stage_epoch)
        writer.add_scalar('Loss/train_angle_consistency', train_loss_dict['angle_consistency'], stage_epoch)
        writer.add_scalar('Loss/train_uniformity', train_loss_dict['uniformity'], stage_epoch)
        writer.add_scalar('Loss/train_tangent', train_loss_dict['tangent'], stage_epoch)
        writer.add_scalar('Loss/train_capsize', train_loss_dict['capsize'], stage_epoch)
        writer.add_scalar('Loss/train_capsize_safe', train_loss_dict['capsize_safe'], stage_epoch)
        writer.add_scalar('Loss/train_fisher_penalty', train_loss_dict['capsize_proxy_penalty'], stage_epoch)
        writer.add_scalar('Loss/train_fisher_diag_mean', train_loss_dict['fisher_diag_mean'], stage_epoch)
        writer.add_scalar('Loss/train_param_drift_mean', train_loss_dict['param_drift_mean'], stage_epoch)

        if current_stage == 2:
            proxy_state = _get_stage2_fisher_proxy_state()
            writer.add_scalar('Fisher/scaling_enabled', 1.0 if args.fisher_mode == 'fisher' else 0.0, stage_epoch)
            fixed_fisher_tags = {
                'Fisher/fixed_min': 'fisher_diag_min_fixed',
                'Fisher/fixed_max': 'fisher_diag_max_fixed',
                'Fisher/fixed_mean': 'fisher_diag_mean_fixed',
                'Fisher/fixed_median': 'fisher_diag_median_fixed',
                'Fisher/fixed_q10': 'fisher_diag_q10_fixed',
                'Fisher/fixed_q90': 'fisher_diag_q90_fixed',
                'Fisher/fixed_clip_at_min': 'fisher_clip_at_min_fixed',
                'Fisher/fixed_clip_at_max': 'fisher_clip_at_max_fixed',
                'Fisher/inv_scale_mean': 'fisher_inv_scale_mean_fixed',
                'Fisher/inv_scale_median': 'fisher_inv_scale_median_fixed',
                'Fisher/inv_scale_ratio': 'fisher_inv_scale_ratio_fixed',
            }
            for tag, key in fixed_fisher_tags.items():
                if key in proxy_state:
                    writer.add_scalar(tag, proxy_state[key], stage_epoch)

            train_fisher_tags = {
                'Fisher/update_sampled_steps': 'fisher_update_sampled_steps',
                'Fisher/update_high_energy': 'fisher_update_high_energy',
                'Fisher/update_low_energy': 'fisher_update_low_energy',
                'Fisher/update_high_mean': 'fisher_update_high_mean',
                'Fisher/update_low_mean': 'fisher_update_low_mean',
                'Fisher/update_high_low_ratio': 'fisher_update_high_low_ratio',
                'Fisher/raw_update_high_low_ratio': 'fisher_raw_update_high_low_ratio',
                'Fisher/scaled_update_high_low_ratio': 'fisher_scaled_update_high_low_ratio',
                'Fisher/high_attenuation': 'fisher_high_attenuation',
                'Fisher/low_amplification': 'fisher_low_amplification',
            }
            for tag, key in train_fisher_tags.items():
                writer.add_scalar(tag, train_loss_dict.get(key, 0.0), stage_epoch)

            stage2_metric_tags = {
                'Fisher/param_drift_high_energy': 'fisher_param_drift_high_energy',
                'Fisher/param_drift_low_energy': 'fisher_param_drift_low_energy',
                'Fisher/param_drift_high_mean': 'fisher_param_drift_high_mean',
                'Fisher/param_drift_low_mean': 'fisher_param_drift_low_mean',
                'Fisher/param_drift_mean_weighted': 'fisher_param_drift_mean_weighted',
                'Fisher/param_drift_high_low_ratio': 'fisher_param_drift_high_low_ratio',
                'Fisher/output_drift_x0_mse': 'fisher_output_drift_x0_mse',
                'Fisher/output_diversity_x0_mse': 'fisher_output_diversity_x0_mse',
            }
            for tag, key in stage2_metric_tags.items():
                writer.add_scalar(tag, stage2_fisher_metrics.get(key, 0.0), stage_epoch)
        
        # ===== R² × S¹ 监控指标 =====
        writer.add_scalar('Manifold/train_angle_norm_mean', train_loss_dict.get('angle_norm_mean', 0.0), stage_epoch)
        writer.add_scalar('Manifold/train_angle_norm_error', train_loss_dict.get('angle_norm_error', 0.0), stage_epoch)
        writer.add_scalar('Manifold/train_main_loss_pos', train_loss_dict.get('main_loss_pos', 0.0), stage_epoch)
        writer.add_scalar('Manifold/train_main_loss_ang', train_loss_dict.get('main_loss_ang', 0.0), stage_epoch)
        
        writer.add_scalar('Loss/val_main', val_loss_dict['main'], stage_epoch)
        writer.add_scalar('Loss/val_smoothness', val_loss_dict['smoothness'], stage_epoch)
        writer.add_scalar('Loss/val_curvature', val_loss_dict['curvature'], stage_epoch)
        writer.add_scalar('Loss/val_angle_smoothness', val_loss_dict['angle_smoothness'], stage_epoch)
        writer.add_scalar('Loss/val_angle_consistency', val_loss_dict['angle_consistency'], stage_epoch)
        writer.add_scalar('Loss/val_uniformity', val_loss_dict['uniformity'], stage_epoch)
        writer.add_scalar('Loss/val_tangent', val_loss_dict['tangent'], stage_epoch)
        writer.add_scalar('Loss/val_capsize', val_loss_dict['capsize'], stage_epoch)
        writer.add_scalar('Loss/val_capsize_safe', val_loss_dict['capsize_safe'], stage_epoch)
        # ===== R² × S¹ 监控指标（验证集）=====
        writer.add_scalar('Manifold/val_angle_norm_mean', val_loss_dict.get('angle_norm_mean', 0.0), stage_epoch)
        writer.add_scalar('Manifold/val_angle_norm_error', val_loss_dict.get('angle_norm_error', 0.0), stage_epoch)
        writer.add_scalar('Manifold/val_main_loss_pos', val_loss_dict.get('main_loss_pos', 0.0), stage_epoch)
        writer.add_scalar('Manifold/val_main_loss_ang', val_loss_dict.get('main_loss_ang', 0.0), stage_epoch)
        
        # TensorBoard - EMA验证损失
        if use_ema and len(ema_val_losses) > 0:
            for i, (ema_val_loss, decay) in enumerate(zip(ema_val_losses, ema_decays)):
                writer.add_scalar(f'Loss/ema_val_{decay}', ema_val_loss, stage_epoch)
        
        writer.add_scalar('LR', optimizer._optimizer.param_groups[0]['lr'], stage_epoch)
        
        # 保存最佳标准模型
        # # Stage 2：以验证集安全率为主（越高越好），main仅作为次级tie-break
        # if current_stage == 2:
        #     current_val_metric = -val_loss_dict.get('reward_rate', 0.0) + 1e-3 * val_loss_dict['main']
        # else:
        #     current_val_metric = val_loss
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
                'stage1_best_loss': stage1_best_loss,
                'torch_seed': torch_seed,
                'fisher_mode': args.fisher_mode,
                'stage2_anchor_path': stage2_anchor_path,
            }
            
            torch.save(checkpoint, best_model_path)
            if current_stage == 2:
                print(
                    f"  ✓ Saved best model for Stage {current_stage} to {osp.basename(best_model_path)} "
                    # f"(val_safe_rate={val_loss_dict.get('reward_rate', 0.0):.4f}, val_main={val_loss_dict['main']:.6f}, metric={current_val_metric:.6f})"
                )
            else:
                print(f"  ✓ Saved best model for Stage {current_stage} to {osp.basename(best_model_path)} (val_loss={current_val_metric:.6f})")
        
        # 分别保存每个EMA模型（基于各自的训练指标在Stage 2）
        if use_ema and len(ema_models) > 0:
            for i, (ema_m, decay, ema_val_loss) in enumerate(zip(ema_models, ema_decays, ema_val_losses)):
                # if current_stage == 2:
                #     # 同样使用训练集指标（EMA是基于training trajectory学出来的）
                #     # 注意：这里我们还没有EMA模型的训练时指标，只能用val指标
                #     # 但理想情况下应该在训练阶段累积EMA模型的reward_rate
                #     ema_val_metric = -ema_val_loss_dicts[i].get('reward_rate', 0.0) + 1e-3 * ema_val_loss_dicts[i]['main']
                # else:
                #     ema_val_metric = ema_val_loss
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
                        'stage1_best_loss': stage1_best_loss,
                        'torch_seed': torch_seed,
                        'fisher_mode': args.fisher_mode,
                        'stage2_anchor_path': stage2_anchor_path,
                    }
                    torch.save(ema_checkpoint, ema_model_path)
                    if current_stage == 2:
                        print(
                            f"  ✓ Saved best EMA model (decay={decay}) to {ema_model_filename} "
                            # f"(val_safe_rate={ema_val_loss_dicts[i].get('reward_rate', 0.0):.4f}, val_main={ema_val_loss_dicts[i]['main']:.6f}, metric={ema_val_metric:.6f})"
                        )
                    else:
                        print(f"  ✓ Saved best EMA model (decay={decay}) to {ema_model_filename} (val_loss={ema_val_metric:.6f})")
        
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
                'stage1_best_loss': stage1_best_loss,
                'torch_seed': torch_seed,
                'fisher_mode': args.fisher_mode,
                'stage2_anchor_path': stage2_anchor_path,
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
                        'stage1_best_loss': stage1_best_loss,
                        'torch_seed': torch_seed,
                        'fisher_mode': args.fisher_mode,
                        'stage2_anchor_path': stage2_anchor_path,
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
    print(f"Stage {current_stage} best validation loss: {best_val_loss:.6f}")
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
    
    # 保存标准最终模型
    torch.save({
        'epoch': final_stage_epoch,
        'stage': final_stage,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer._optimizer.state_dict(),
        'n_steps': optimizer.n_steps,
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'stage1_best_loss': stage1_best_loss,
        'torch_seed': torch_seed,
        'fisher_mode': args.fisher_mode,
        'stage2_anchor_path': stage2_anchor_path,
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

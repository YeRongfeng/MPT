"""
train_grpo.py - 训练不平坦地面路径预测模型(基于扩散模型)
"""

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


def _predict_x0_from_model_output(z_t, t, model_output, prediction_type='x0'):
    """将模型输出统一转换为 x0 预测。"""
    t_v = t.view(-1, 1, 1)
    if prediction_type == 'epsilon':
        return (z_t - t_v * model_output) / (1.0 - t_v + 1e-5)
    if prediction_type == 'x0':
        return model_output
    if prediction_type == 'v':
        return z_t - t_v * model_output
    raise ValueError(f"Unsupported prediction_type: {prediction_type}")


def generate_gr_awf_pseudo_targets(
    model,
    map_input,
    start_normalized,
    goal_normalized,
    start_pose,
    goal_pose,
    middle_cp_normalized,
    batch,
    device,
    prediction_type='x0',
    num_path=4,
    tau=1.0,
    t_eval_min=0.5,
    t_eval_max=1.0,
    num_dense_points=100,
    relabel_mode='topm_sample',
    top_m=None,
    weight_uniform_mix=0.2,
    quality_guard=True,
    max_cost_ratio=1.05,
    min_span=0.05,
):
    """
    GR-AWF 第二阶段伪标签生成：
    1) 组并行探索（无梯度）
    2) 组内相对优势 -> softmax 权重
    3) 用组内重采样/选择，得到替换 batch GT 的 pseudo x0（避免均值塌缩）

    Returns:
        pseudo_middle_cp_normalized: (B, N, D)
        stats: 监控信息
    """
    from grad_optimizer import cost_on_dense_trajectory

    B, N, D = middle_cp_normalized.shape
    K = max(1, int(num_path))

    # ===== Stage-1: 并行探索（no-grad） =====
    map_stacked = map_input.repeat_interleave(K, dim=0)
    start_norm_stacked = start_normalized.repeat_interleave(K, dim=0)
    goal_norm_stacked = goal_normalized.repeat_interleave(K, dim=0)
    start_pose_stacked = start_pose.repeat_interleave(K, dim=0)
    goal_pose_stacked = goal_pose.repeat_interleave(K, dim=0)
    x0_anchor = middle_cp_normalized.repeat_interleave(K, dim=0)

    prev_training = model.training
    model.eval()
    with torch.no_grad():
        t_eval = torch.rand(B * K, device=device) * (t_eval_max - t_eval_min) + t_eval_min
        t_eval = torch.clamp(t_eval, min=1e-4, max=1.0 - 1e-4)
        r_eval = torch.rand_like(t_eval) * t_eval

        eps_explore = torch.randn_like(x0_anchor)
        t_eval_v = t_eval.view(-1, 1, 1)
        z_eval = (1.0 - t_eval_v) * x0_anchor + t_eval_v * eps_explore

        model_out_eval = model(
            map_stacked,
            z_eval,
            t_eval,
            r_eval,
            start_norm_stacked,
            goal_norm_stacked,
        )
        x0_pseudo = _predict_x0_from_model_output(
            z_eval, t_eval, model_out_eval, prediction_type=prediction_type
        ).detach()
        x0_pseudo = torch.clamp(x0_pseudo, -1.0, 1.0)

    # 恢复原训练状态
    model.train(prev_training)

    # ===== Stage-2: 组内相对优势打分（黑盒 cost） =====
    x0_pseudo_denorm = x0_pseudo * 20.0
    start_cp = start_pose_stacked[:, :2].unsqueeze(1)
    goal_cp = goal_pose_stacked[:, :2].unsqueeze(1)
    full_control_points = torch.cat([start_cp, x0_pseudo_denorm, goal_cp], dim=1)  # (B*K, 26, 2)

    bspline_layer = DifferentiableBSpline(
        num_control_points=26,
        num_output_points=num_dense_points,
        degree=3
    ).to(device)
    dense_traj = bspline_layer(full_control_points)  # (B*K, num_dense_points, 2)

    stability_cost_map = batch['cost_map'].to(device)
    if stability_cost_map.shape[0] == B:
        stability_cost_map = stability_cost_map.repeat_interleave(K, dim=0)

    map_info = {
        'resolution': 0.4,
        'origin': (-20.0, -20.0, -np.pi),
        'size': (100, 100, 36)
    }

    with torch.no_grad():
        costs_flat = cost_on_dense_trajectory(
            dense_traj,
            start_pose_stacked,
            goal_pose_stacked,
            stability_cost_map,
            map_info,
            device,
            return_per_sample=True,
        )  # (B*K,)

        costs = costs_flat.view(B, K)
        cost_mean = costs.mean(dim=1, keepdim=True)
        cost_std = costs.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
        advantages = (cost_mean - costs) / cost_std
        weights = F.softmax(advantages / max(float(tau), 1e-6), dim=1)  # (B, K), sum=1

        # 与均匀分布做 convex mixing，避免过早塌缩到单一路径
        uniform_mix = float(weight_uniform_mix)
        uniform_mix = min(max(uniform_mix, 0.0), 1.0)
        if uniform_mix > 0.0:
            weights = (1.0 - uniform_mix) * weights + uniform_mix * (torch.ones_like(weights) / K)

    # ===== Stage-3: 组内重采样/选择为新的 pseudo GT（detach） =====
    x0_group = x0_pseudo.view(B, K, N, D)

    # Top-M 截断（工程折中）：去掉明显劣质轨迹，再在幸存者内重采样
    if top_m is None:
        M = max(1, K // 2)
    else:
        M = max(1, min(int(top_m), K))

    if relabel_mode == 'soft_all':
        # 全量软加权（显存充裕时最稳），但可能有“均值化”风险
        pseudo_middle_cp_normalized = (x0_group * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        selected_cost = (weights * costs).sum(dim=1)  # (B,)
    elif relabel_mode == 'sample':
        selected_idx = torch.multinomial(weights, num_samples=1).squeeze(1)  # (B,)
        batch_idx = torch.arange(B, device=device)
        pseudo_middle_cp_normalized = x0_group[batch_idx, selected_idx]
        selected_cost = costs.gather(1, selected_idx.unsqueeze(1)).squeeze(1)
    elif relabel_mode == 'top1':
        # 不推荐：容易坍塌，仅保留兼容
        selected_idx = torch.argmax(weights, dim=1)
        batch_idx = torch.arange(B, device=device)
        pseudo_middle_cp_normalized = x0_group[batch_idx, selected_idx]
        selected_cost = costs.gather(1, selected_idx.unsqueeze(1)).squeeze(1)
    else:
        # 默认 topm_sample：Top-M + 重新加权 + multinomial
        top_adv_vals, top_adv_idx = torch.topk(advantages, k=M, dim=1)  # (B, M)
        top_weights = F.softmax(top_adv_vals / max(float(tau), 1e-6), dim=1)  # (B, M)

        uniform_mix = float(weight_uniform_mix)
        uniform_mix = min(max(uniform_mix, 0.0), 1.0)
        if uniform_mix > 0.0:
            top_weights = (1.0 - uniform_mix) * top_weights + uniform_mix * (torch.ones_like(top_weights) / M)

        picked_local = torch.multinomial(top_weights, num_samples=1).squeeze(1)  # (B,)
        selected_idx = top_adv_idx.gather(1, picked_local.unsqueeze(1)).squeeze(1)  # (B,)
        batch_idx = torch.arange(B, device=device)
        pseudo_middle_cp_normalized = x0_group[batch_idx, selected_idx]
        selected_cost = costs.gather(1, selected_idx.unsqueeze(1)).squeeze(1)

    # 质量门控：防止伪标签退化为“点”或明显劣化轨迹
    keep_ratio = 1.0
    if quality_guard:
        with torch.no_grad():
            # 基线：当前 batch 原 GT 控制点的 cost
            anchor_full_cp = torch.cat([
                start_pose[:, :2].unsqueeze(1),
                middle_cp_normalized * 20.0,
                goal_pose[:, :2].unsqueeze(1)
            ], dim=1)  # (B, 26, 2)
            anchor_dense = bspline_layer(anchor_full_cp)  # (B, num_dense_points, 2)
            anchor_map = batch['cost_map'].to(device)
            anchor_cost = cost_on_dense_trajectory(
                anchor_dense,
                start_pose,
                goal_pose,
                anchor_map,
                map_info,
                device,
                return_per_sample=True,
            )  # (B,)

            # 空间展宽（归一化坐标）过小，视为坍塌到点
            span = torch.norm(
                pseudo_middle_cp_normalized.max(dim=1).values - pseudo_middle_cp_normalized.min(dim=1).values,
                dim=1
            )  # (B,)

            keep_by_cost = selected_cost <= (anchor_cost * max(float(max_cost_ratio), 1e-3))
            keep_by_span = span >= max(float(min_span), 1e-6)
            keep_mask = keep_by_cost & keep_by_span

            keep_ratio = keep_mask.float().mean().item()
            keep_mask_exp = keep_mask.view(B, 1, 1)
            pseudo_middle_cp_normalized = torch.where(
                keep_mask_exp,
                pseudo_middle_cp_normalized,
                middle_cp_normalized
            )

    pseudo_middle_cp_normalized = pseudo_middle_cp_normalized.detach()

    entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=1)  # (B,)

    stats = {
        'mean_cost': costs_flat.mean().item(),
        'adv_mean': advantages.mean().item(),
        'adv_std': advantages.std(unbiased=False).item(),
        'w_max': weights.max().item(),
        'w_min': weights.min().item(),
        'w_entropy': entropy.mean().item(),
        'top_m': float(M),
        'keep_ratio': keep_ratio,
    }
    return pseudo_middle_cp_normalized, stats



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
        is_training: bool - 训练模式下第二阶段会执行 GR-AWF 重标注；验证模式不重标注
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

    # =================== Stage 2: GR-AWF 先替换 batch GT，再走与阶段1一致的损失流程 ===================
    stage2_relabel_active = False
    stage2_relabel_stats = None
    if is_training and current_stage == 2 and ('cost_map' in batch):
        num_path = int(loss_weights.get('num_path', 4))
        tau = float(loss_weights.get('tau', 1.0))
        t_eval_min = float(loss_weights.get('t_eval_min', 0.5))
        t_eval_max = float(loss_weights.get('t_eval_max', 1.0))
        relabel_mode = str(loss_weights.get('relabel_mode', 'topm_sample'))
        top_m = int(loss_weights.get('top_m', max(1, num_path // 2)))
        weight_uniform_mix = float(loss_weights.get('weight_uniform_mix', 0.2))
        quality_guard = bool(loss_weights.get('quality_guard', True))
        max_cost_ratio = float(loss_weights.get('max_cost_ratio', 1.05))
        min_span = float(loss_weights.get('min_span', 0.05))
        pseudo_replace_ratio = float(loss_weights.get('pseudo_replace_ratio', 0.7))
        pseudo_replace_ratio = min(max(pseudo_replace_ratio, 0.0), 1.0)

        pseudo_cp_normalized, stage2_relabel_stats = generate_gr_awf_pseudo_targets(
            model=model,
            map_input=map_input,
            start_normalized=start_normalized,
            goal_normalized=goal_normalized,
            start_pose=start_pose,
            goal_pose=goal_pose,
            middle_cp_normalized=middle_cp_normalized,
            batch=batch,
            device=device,
            prediction_type=prediction_type,
            num_path=num_path,
            tau=tau,
            t_eval_min=t_eval_min,
            t_eval_max=t_eval_max,
            num_dense_points=num_dense_points,
            relabel_mode=relabel_mode,
            top_m=top_m,
            weight_uniform_mix=weight_uniform_mix,
            quality_guard=quality_guard,
            max_cost_ratio=max_cost_ratio,
            min_span=min_span,
        )

        # 软替换：保留一部分原始 GT 作为锚点，减少自举塌缩
        middle_cp_normalized = (
            (1.0 - pseudo_replace_ratio) * middle_cp_normalized
            + pseudo_replace_ratio * pseudo_cp_normalized
        )
        stage2_relabel_active = True
    
    # ===== pixel Mean Flow: 连续时间采样 =====
    # 采样 t 和 r (0 <= r <= t <= 1)
    if hasattr(model, 'module'):
        t = model.module.sample_timesteps(B, device=device)
    else:
        t = model.sample_timesteps(B, device=device)
    t = torch.clamp(t, min=1e-4, max=1.0 - 1e-4) # 避免极端值
    r = torch.rand_like(t) * t
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
    tangent_loss = dummy_loss.clone()
    consistency_loss = dummy_loss.clone()
    
    if loss_weights['tangent'] > 0.0:
        tangent_loss = compute_tangent_loss(middle_cp_normalized, start_normalized, goal_normalized)
    
    # 旧的第二阶段 capsize 反传分支已移除。
    # 现在第二阶段仅通过 GR-AWF 重标注 pseudo GT，然后复用第一阶段主损失流程。

    # Stage 2 验证：使用 pred_x0 对应轨迹的 cost 作为 val_loss（不反传）
    if (not is_training) and current_stage == 2 and ('cost_map' in batch):
        from grad_optimizer import cost_on_dense_trajectory

        x0_middle_pred = z_t - t.view(-1, 1, 1) * u_out  # (B, 24, 2), normalized
        x0_middle_denorm = torch.clamp(x0_middle_pred, -1.0, 1.0) * 20.0

        start_cp = start_pose[:, :2].unsqueeze(1)  # (B, 1, 2)
        goal_cp = goal_pose[:, :2].unsqueeze(1)    # (B, 1, 2)
        full_ctrl_points = torch.cat([start_cp, x0_middle_denorm, goal_cp], dim=1)  # (B, 26, 2)

        bspline_layer = DifferentiableBSpline(
            num_control_points=26,
            num_output_points=100,
            degree=3
        ).to(device)
        reconstructed_traj = bspline_layer(full_ctrl_points)  # (B, 100, 2)

        map_info = {
            'resolution': 0.4,
            'origin': (-20.0, -20.0, -np.pi),
            'size': (100, 100, 36)
        }
        stability_cost_map = batch['cost_map'].to(device)
        capsize_loss = cost_on_dense_trajectory(
            reconstructed_traj,
            start_pose,
            goal_pose,
            stability_cost_map,
            map_info,
            device
        )

        if not torch.isfinite(capsize_loss):
            capsize_loss = dummy_loss.clone()

        # 验证阶段以 cost 作为主指标（用于 early stopping / best model）
        total_loss = capsize_loss

        loss_dict = {
            'main': main_loss.item(),
            'smoothness': smoothness_loss.item(),
            'curvature': curvature_loss.item(),
            'angle_smoothness': angle_smoothness_loss.item(),
            'angle_consistency': angle_consistency_loss.item(),
            'uniformity': uniformity_loss.item(),
            'tangent': tangent_loss.item(),
            'capsize': capsize_loss.item(),
            'consistency': consistency_loss.item(),
            'angle_norm_mean': 1.0,
            'angle_norm_error': 0.0,
            'main_loss_pos': main_loss.item(),
            'main_loss_ang': 0.0,
        }
        return total_loss, 0, B, loss_dict

    
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

    # 混合损失（旧 stage2 capsize 反传项已移除）
    if loss_weights['main'] <= 0.0:
        total_loss = loss_weights['tangent'] * tangent_loss
    else:
        total_loss = loss_weights['main'] * main_loss \
                    + loss_weights['tangent'] * tangent_loss
    
    # 检查损失异常：不中断训练，回退为零损失并跳过本 batch 更新
    if not torch.isfinite(total_loss):
        total_loss_val = total_loss.item() if isinstance(total_loss, torch.Tensor) else float('nan')
        main_loss_val = main_loss.item() if isinstance(main_loss, torch.Tensor) else float('nan')
        print(f"⚠ Warning: total_loss is NaN/Inf ({total_loss_val}), main_loss={main_loss_val}. Skip this batch.")
        total_loss = dummy_loss.clone()
    
    # stage2 relabel 模式下，用重标注阶段的平均 cost 作为监控项（不参与反传）
    capsize_log_value = capsize_loss.item()
    if stage2_relabel_stats is not None:
        capsize_log_value = float(stage2_relabel_stats.get('mean_cost', capsize_log_value))
    w_max_log = float(stage2_relabel_stats.get('w_max', 0.0)) if stage2_relabel_stats is not None else 0.0
    w_min_log = float(stage2_relabel_stats.get('w_min', 0.0)) if stage2_relabel_stats is not None else 0.0
    w_entropy_log = float(stage2_relabel_stats.get('w_entropy', 0.0)) if stage2_relabel_stats is not None else 0.0
    keep_ratio_log = float(stage2_relabel_stats.get('keep_ratio', 0.0)) if stage2_relabel_stats is not None else 0.0

    # 返回各项损失用于记录
    loss_dict = {
        'main': main_loss.item(),
        'smoothness': smoothness_loss.item(),
        'curvature': curvature_loss.item(),
        'angle_smoothness': angle_smoothness_loss.item(),
        'angle_consistency': angle_consistency_loss.item(),
        'uniformity': uniformity_loss.item(),
        'tangent': tangent_loss.item(),
        'capsize': capsize_log_value,
        'consistency': consistency_loss.item(),
        'angle_norm_mean': mean_norm_sq ** 0.5,
        'angle_norm_error': angle_norm_error.item() if isinstance(angle_norm_error, torch.Tensor) else 0.0,
        'main_loss_pos': main_loss_pos.item() if isinstance(main_loss_pos, torch.Tensor) else 0.0,
        'main_loss_ang': main_loss_ang.item() if isinstance(main_loss_ang, torch.Tensor) else 0.0,
        'w_max': w_max_log,
        'w_min': w_min_log,
        'w_entropy': w_entropy_log,
        'keep_ratio': keep_ratio_log,
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
        'consistency': 0,
    }
    
    pbar = tqdm(trainingData, mininterval=2, desc=f"Training Epoch {stage_epoch} (Stage {current_stage})")
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()
        
        loss, _, n_samples, loss_dict = diffusion_loss(
            model, batch, device, loss_weights, 
            epoch=stage_epoch, total_epochs=total_stage_epochs,
            is_training=True,  # 训练模式：第二阶段启用 GR-AWF 重标注
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
                'Wmax': f'{loss_dict.get("w_max", 0.0):.2f}',
                'Wmin': f'{loss_dict.get("w_min", 0.0):.2f}',
                'WEnt': f'{loss_dict.get("w_entropy", 0.0):.2f}',
                'Keep': f'{loss_dict.get("keep_ratio", 0.0):.2f}',
                'GradNorm': grad_info
            })
    
    avg_loss = total_loss / len(trainingData) if len(trainingData) > 0 else 0
    
    # 计算各项平均损失
    avg_loss_dict = {k: v / len(trainingData) for k, v in loss_accumulator.items()}
    
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
    }
    
    with torch.no_grad():
        for batch in tqdm(validationData, mininterval=2, desc="Validation"):
            
            loss, _, n_samples, loss_dict = diffusion_loss(
                model, batch, device, loss_weights,
                epoch=epoch, total_epochs=total_epochs,
                is_training=False,  # 验证模式：不做 GR-AWF 重标注
                current_stage=current_stage,  # 传递当前阶段
                use_dense_trajectory=use_dense_trajectory,
                num_dense_points=num_dense_points,
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
    
    # 阶段2配置：通过梯度优化学习低cost轨迹
    # **核心改变**：使用完整DDIM采样链计算cost，而非单步预测
    # **关键**：保留main loss作为正则化，capsize loss作为优化目标
    stage2_config = {
        'epochs': args.stage2_epochs,
        'lr_mul': 1e-3,  # 小学习率微调（不是从头训练！）
        'loss_weights': {
            'main': 1.0,         # GR-AWF 主损失（加权流匹配）
            'smoothness': 0e-5,   # 平滑性损失
            'curvature': 0e-4,    # 曲率约束（限制最大曲率）- 新增
            'angle_smoothness': 0e-2,  # 角度平滑性损失
            'angle_consistency': 0e-6,  # 角度一致性损失
            'uniformity': 0e-4,   # 均匀性损失
            'tangent': 0e-2,   # 切线约束损失（确保起点和终点方向一致）
            'capsize': 0.0,    # 旧第二阶段损失已弃用，仅保留日志字段
            'consistency': 0.0,   # 时间一致性损失（当前不启用）
            # ===== GR-AWF 超参数 =====
            'num_path': 8,      # 组大小 K（Baseline 推荐）
            'tau': 1.0,         # softmax 温度（Baseline 推荐）
            't_eval_min': 0.8,  # 探索时间步（固定 0.8）
            't_eval_max': 0.8,  # 探索时间步（固定 0.8）
            'relabel_mode': 'topm_sample',  # topm_sample/sample/soft_all/top1
            'top_m': 2,                     # Top-M 截断数量（K=8 时建议先用 2）
            'weight_uniform_mix': 0.1,      # 与均匀分布混合，抗塌缩
            'pseudo_replace_ratio': 0.5,    # 伪标签替换比例（软替换，防止自举坍塌）
            'quality_guard': True,          # 启用伪标签质量门控
            'max_cost_ratio': 1.05,         # 仅接受不劣于基线太多的伪标签
            'min_span': 0.05,               # 轨迹展宽下限（归一化坐标）
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
        print("⚠ Warning: Stage 2 main loss weight <= 0.")
    
    # GR-AWF 重标注在第二阶段需要 cost_map
    compute_stability = (current_stage == 2)
    
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
            
            # 【关键】阶段2：只训练map cross-attention + 主预测头
            if isinstance(model, nn.DataParallel):
                model.module.freeze_for_stage2()
                trainable_params = model.module.get_trainable_parameters()
            else:
                model.freeze_for_stage2()
                trainable_params = model.get_trainable_parameters()
            
            trainable_count = sum(p.numel() for p in trainable_params)
            print(f"✓ 阶段2训练参数量: {trainable_count:,}")
            
            # 更新优化器（只优化cross-attention + 主预测头）
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
            print(f"✓ Enabled GR-AWF relabeling for Stage 2")
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
        print(f"  Val Loss:   {val_loss:.9f}")
        print(f"    - Main: {val_loss_dict['main']:.6f}")
        # print(f"    - Smoothness: {val_loss_dict['smoothness']:.6f}")
        # print(f"    - Curvature: {val_loss_dict['curvature']:.6f}")
        # print(f"    - Angle Smoothness: {val_loss_dict['angle_smoothness']:.6f}")
        # print(f"    - Angle Consistency: {val_loss_dict['angle_consistency']:.6f}")
        # print(f"    - Uniformity: {val_loss_dict['uniformity']:.6f}")
        print(f"    - Tangent: {val_loss_dict['tangent']:.6f}")
        print(f"    - Capsize: {val_loss_dict['capsize']:.9f}")
        
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
                'torch_seed': torch_seed
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
                        'torch_seed': torch_seed
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
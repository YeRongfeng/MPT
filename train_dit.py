"""
train_dit.py - 训练不平坦地面路径预测模型(基于扩散模型)
"""

import numpy as np
import pickle

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

from ESDF3d_atpoint import compute_esdf_batch
from grad_optimizer import TrajectoryOptimizerSE2
from dit.Models import PathDiffusionTransformer

def compute_timestep_weight(t_normalized, k=1.0):
    """
    计算时间步权重（适配新参数化）
    
    新参数化下，t_normalized从1（纯数据）到0（纯噪声）
    我们希望模型更关注干净数据区域（t_normalized接近1）
    
    Args:
        t_normalized: (B,) 归一化的时间步，范围[0, 1]
            - t=1.0: 纯数据（最重要）
            - t=0.0: 纯噪声（相对不重要）
        k: 加权指数，越大则对干净数据区域的偏好越强
    
    Returns:
        weight: (B,) 时间步权重，范围[0, 1]
    
    例子（k=1.0）：
        t=1.0 (纯数据): w = 1.0 (最高权重)
        t=0.5 (中等): w = 0.5
        t=0.0 (纯噪声): w = 0.0 (最低权重)
    """
    # w(t) = t^k，让模型更关注干净数据区域
    weight = t_normalized ** k
    return weight

def compute_smoothness_loss(trajectory, include_angle=False, threshold=None):
    """
    改进的平滑性损失 - 只惩罚超过阈值的急转弯
    Args:
        trajectory: (B, N, 4) - 轨迹序列，格式为(x, y, sin(θ), cos(θ))
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
        # 角度部分：直接对sin/cos分量计算平滑性，不恢复角度
        sin_theta = trajectory[:, :, 2]  # (B, N)
        cos_theta = trajectory[:, :, 3]  # (B, N)
        
        # 对sin和cos分别计算一阶差分
        sin_first_diff = sin_theta[:, 1:] - sin_theta[:, :-1]  # (B, N-1)
        cos_first_diff = cos_theta[:, 1:] - cos_theta[:, :-1]  # (B, N-1)
        
        # 对sin和cos分别计算二阶差分
        sin_second_diff = sin_first_diff[:, 1:] - sin_first_diff[:, :-1]  # (B, N-2)
        cos_second_diff = cos_first_diff[:, 1:] - cos_first_diff[:, :-1]  # (B, N-2)
        
        # 计算角度变化的幅度（sin和cos的二阶导数的L2范数）
        angle_acc_norm = torch.sqrt(sin_second_diff ** 2 + cos_second_diff ** 2)  # (B, N-2)
        
        if threshold is not None:
            # 角加速度阈值（基于sin/cos空间的变化率）
            angle_threshold = threshold * 0.1  # 调整比例
            excess_angle_acc = torch.relu(angle_acc_norm - angle_threshold)
            angle_smoothness = torch.mean(excess_angle_acc ** 2)
        else:
            angle_smoothness = torch.mean(angle_acc_norm ** 2)
        
        # 合并位置和角度的平滑性损失
        smoothness_loss = pos_smoothness + angle_smoothness
    else:
        smoothness_loss = pos_smoothness
    
    return smoothness_loss

def compute_angle_smoothness_loss(trajectory):
    """
    计算角度平滑性损失 - 惩罚角速度本身和角加速度
    使用单位圆上的切向速度来计算角速度，避免角度恢复
    
    **重要**：当trajectory包含固定的起点和终点时（22个点），
    排除边界段的角速度约束（索引0和20），避免强制预测点角度向固定的起终点角度靠拢，
    导致与实际运动方向冲突。
    
    Args:
        trajectory: (B, N, 4) - 轨迹序列，格式为(x, y, sin(θ), cos(θ))
    Returns:
        angle_loss: 标量
    """
    sin_theta = trajectory[:, :, 2]  # (B, N)
    cos_theta = trajectory[:, :, 3]  # (B, N)
    
    # 方法：使用相邻帧之间的点积和叉积来计算角度变化
    # cos(Δθ) = sin(t)*sin(t+1) + cos(t)*cos(t+1)
    # sin(Δθ) = sin(t+1)*cos(t) - sin(t)*cos(t+1)
    
    sin_t = sin_theta[:, :-1]  # (B, N-1)
    cos_t = cos_theta[:, :-1]
    sin_t1 = sin_theta[:, 1:]  # (B, N-1)
    cos_t1 = cos_theta[:, 1:]
    
    # 计算角度变化 Δθ 的 sin 和 cos
    # 这代表相邻点之间的角速度（假设时间步长为1）
    cos_delta = sin_t * sin_t1 + cos_t * cos_t1  # (B, N-1)
    sin_delta = sin_t1 * cos_t - sin_t * cos_t1  # (B, N-1)
    
    # 角速度的幅度 (通过 sin 和 cos 计算)
    # |Δθ| ≈ sqrt((1 - cos(Δθ))^2 + sin(Δθ)^2) 
    # 简化：使用 sin²(Δθ) + (1-cos(Δθ))² 来近似角度大小的平方
    angular_velocity = torch.sqrt(sin_delta ** 2 + (1 - cos_delta) ** 2 + 1e-8)  # (B, N-1)
    
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
    cos_delta_diff = cos_delta[:, 1:] - cos_delta[:, :-1]  # (B, N-2)
    sin_delta_diff = sin_delta[:, 1:] - sin_delta[:, :-1]  # (B, N-2)
    
    # 角加速度的L2范数
    angle_acc = torch.sqrt(cos_delta_diff ** 2 + sin_delta_diff ** 2 + 1e-8)  # (B, N-2)
    
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
    计算角度一致性损失 - 防止倒车（使用sin/cos向量点积）
    该损失确保运动方向与车辆朝向一致（防止倒车）
    使用向量点积直接在sin/cos空间计算，避免角度恢复
    
    **重要**：当trajectory包含固定的起点和终点时（22个点），
    只计算预测点内部的角度一致性（索引1到20），排除边界段（0→1和20→21）
    以避免与固定的起终点角度产生冲突。
    
    Args:
        trajectory: (B, N, 4) - 轨迹序列，格式为(x, y, sin(θ), cos(θ))
    Returns:
        angle_loss: 标量
    """
    # 提取位置和朝向向量
    positions = trajectory[:, :, :2]  # (B, N, 2)
    sin_theta = trajectory[:, :, 2]  # (B, N)
    cos_theta = trajectory[:, :, 3]  # (B, N)
    
    # 车辆朝向向量：(cos(θ), sin(θ))
    heading_vectors = torch.stack([cos_theta, sin_theta], dim=2)  # (B, N, 2)
    
    # 归一化朝向向量（防止sin²+cos²≠1时点积失真）
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
        trajectory: (B, N, 4) - 轨迹序列，格式为(x, y, sin(θ), cos(θ))
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
    计算sin/cos归一化损失 - 确保 sin²(θ) + cos²(θ) ≈ 1
    这是一个正则化项，强制模型输出符合三角恒等式
    
    Args:
        trajectory: (B, N, 4) - 轨迹序列，格式为(x, y, sin(θ), cos(θ))
    Returns:
        norm_loss: 标量
    """
    sin_theta = trajectory[:, :, 2]  # (B, N)
    cos_theta = trajectory[:, :, 3]  # (B, N)
    
    # 计算 sin²(θ) + cos²(θ)
    norm_sq = sin_theta ** 2 + cos_theta ** 2  # (B, N)
    
    # 损失为偏离1的平方：(sin² + cos² - 1)²
    norm_loss = torch.mean((norm_sq - 1.0) ** 2)
    
    return norm_loss

def compute_curvature_constraint_loss(trajectory, max_curvature=2.0):
    """
    曲率约束损失 - 只惩罚超过最大曲率的点
    使用三点法计算曲率：κ = 2*sin(θ) / d
    其中θ是转角，d是弦长
    
    Args:
        trajectory: (B, N, 4) - 轨迹序列，格式为(x, y, sin(θ), cos(θ))
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


def diffusion_loss(model, batch, device, loss_weights=None, epoch=0, total_epochs=100, is_training=True, current_stage=1):
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
    trajectory = batch['trajectory'].to(device)  # (B, 22, 3) - 完整路径（起点+20中间点+终点）
    start_pose = batch['start_pose'].to(device)  # (B, 3)
    goal_pose = batch['goal_pose'].to(device)  # (B, 3)
    
    B = map_input.shape[0]
    
    # 只取中间20个点（不包括起点和终点）
    middle_trajectory = trajectory[:, 1:-1, :]  # (B, 20, 3)
    
    # 转换为sin/cos编码并归一化
    # 输入: (B, 20, 3) -> 输出: (B, 20, 4)
    traj_normalized = torch.zeros(B, 20, 4, device=device)
    
    # 坐标范围通常在 [-20, 20]，归一化到 [-1, 1]
    traj_normalized[:, :, :2] = middle_trajectory[:, :, :2] / 20.0  # x,y: [-20,20] → [-1,1]
    
    # 角度转换为sin/cos编码（已经在[-1,1]范围内）
    traj_normalized[:, :, 2] = torch.sin(middle_trajectory[:, :, 2])  # sin(θ)
    traj_normalized[:, :, 3] = torch.cos(middle_trajectory[:, :, 2])  # cos(θ)
    
    # 裁剪位置到合理范围（sin/cos自然在[-1,1]内）
    traj_normalized[:, :, :2] = torch.clamp(traj_normalized[:, :, :2], -1.0, 1.0)
    
    # 归一化起点终点坐标（转换为4维：x, y, sin(θ), cos(θ)）
    start_normalized = torch.zeros(B, 4, device=device)
    start_normalized[:, :2] = start_pose[:, :2] / 20.0
    start_normalized[:, 2] = torch.sin(start_pose[:, 2])  # sin(θ)
    start_normalized[:, 3] = torch.cos(start_pose[:, 2])  # cos(θ)
    start_normalized[:, :2] = torch.clamp(start_normalized[:, :2], -1.0, 1.0)
    
    goal_normalized = torch.zeros(B, 4, device=device)
    goal_normalized[:, :2] = goal_pose[:, :2] / 20.0
    goal_normalized[:, 2] = torch.sin(goal_pose[:, 2])  # sin(θ)
    goal_normalized[:, 3] = torch.cos(goal_pose[:, 2])  # cos(θ)
    goal_normalized[:, :2] = torch.clamp(goal_normalized[:, :2], -1.0, 1.0)
    
    # 改进的时间步采样策略（参考文生图模型）
    # 使用sigmoid正态分布采样，避免极端时间步，提高训练稳定性
    if hasattr(model, 'module'):  # DataParallel
        t = model.module.sample_timesteps(B, device=device)
    else:
        t = model.sample_timesteps(B, device=device)
    
    # 生成噪声（不裁剪，保持标准正态分布）
    noise = torch.randn_like(traj_normalized)
    
    # 加噪（不裁剪，让扩散过程自然进行）
    noisy_traj = model.q_sample(traj_normalized, t, noise)
    
    # 获取模型预测
    model_output = model(map_input, noisy_traj, t, start_normalized, goal_normalized)
    
    # 步骤1：根据prediction_type解析模型输出
    prediction_type = getattr(model, 'prediction_type', 'epsilon')
    loss_type = getattr(model, 'loss_type', prediction_type)
    
    # ===== 新参数化训练 =====
    # 
    # 新扩散参数化（时间t线性插值）：
    #   加噪过程：z = t * x + (1 - t) * e
    #   速度定义：v = (x - z) / (1 - t)
    # 
    # 三种预测类型的关系：
    #   1. 预测噪声 epsilon: x = (z - (1-t)*e) / t
    #   2. 预测数据 x0: 直接得到 x
    #   3. 预测速度 v: x = z + (1-t)*v
    # 
    # 从任一预测可以推导其他两个：
    #   - 从 x 和 z 可得 e = (z - t*x) / (1-t)
    #   - 从 x 和 z 可得 v = (x - z) / (1-t)
    
    # 获取归一化的时间步 t (0到1)
    device = noisy_traj.device
    if hasattr(model, 'module'):  # DataParallel
        t_normalized = model.module.timesteps_normalized[t][:, None, None]
        t_eps = model.module.t_eps
    else:
        t_normalized = model.timesteps_normalized[t][:, None, None]
        t_eps = model.t_eps
    
    # 根据模型预测类型，计算出 x0, epsilon, v 三者
    if prediction_type == 'epsilon':
        # 模型预测噪声 e
        pred_epsilon = model_output
        # 从e反推x: x = (z - (1-t)*e) / t
        pred_x0 = (noisy_traj - (1 - t_normalized) * pred_epsilon) / torch.clamp(t_normalized, min=t_eps)
        # 计算v: v = (x - z) / (1-t)
        pred_v = (pred_x0 - noisy_traj) / torch.clamp(1 - t_normalized, min=t_eps)
        
    elif prediction_type == 'x0':
        # 模型直接预测 x
        pred_x0 = model_output
        # 从x反推e: e = (z - t*x) / (1-t)
        pred_epsilon = (noisy_traj - t_normalized * pred_x0) / torch.clamp(1 - t_normalized, min=t_eps)
        # 计算v: v = (x - z) / (1-t)
        pred_v = (pred_x0 - noisy_traj) / torch.clamp(1 - t_normalized, min=t_eps)
        
    elif prediction_type == 'v':
        # 模型预测 v
        pred_v = model_output
        # 从v反推x: x = z + (1-t)*v
        pred_x0 = noisy_traj + (1 - t_normalized) * pred_v
        # 从x和z反推e: e = (z - t*x) / (1-t)
        pred_epsilon = (noisy_traj - t_normalized * pred_x0) / torch.clamp(1 - t_normalized, min=t_eps)
        
    else:
        raise ValueError(f"Unknown prediction_type: {prediction_type}. Must be 'epsilon', 'x0', or 'v'")
    
    # 根据loss_type选择目标并计算损失
    if loss_type == 'epsilon':
        target = noise
        prediction = pred_epsilon
    elif loss_type == 'x0':
        target = traj_normalized
        prediction = pred_x0
    elif loss_type == 'v':
        # 真实的v目标：v = (x - z) / (1-t)
        target = (traj_normalized - noisy_traj) / torch.clamp(1 - t_normalized, min=t_eps)
        prediction = pred_v
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Must be 'epsilon', 'x0', or 'v'")
    
    # =================== 计算辅助损失 ===================
    # 使用预测的x0（去噪后的轨迹）来计算辅助损失
    
    # 初始化所有损失为与模型输出相关的零tensor（保持梯度连接）
    dummy_loss = pred_x0.sum() * 0.0
    main_loss = dummy_loss.clone()
    smoothness_loss = dummy_loss.clone()
    curvature_loss = dummy_loss.clone()
    angle_smoothness_loss = dummy_loss.clone()
    angle_consistency_loss = dummy_loss.clone()
    uniformity_loss = dummy_loss.clone()
    capsize_loss = dummy_loss.clone()
    sincos_norm_loss = dummy_loss.clone()
    
    # 检查模型输出是否已经学会生成有效的sin/cos值
    # 如果 sin²+cos² 太小，说明模型还在初始化阶段，跳过角度相关的辅助损失
    sin_cos_norm_sq = pred_x0[:, :, 2]**2 + pred_x0[:, :, 3]**2  # (B, 20)
    mean_norm_sq = sin_cos_norm_sq.mean().item()
    skip_angle_losses = (mean_norm_sq < 0.01)  # 如果平均 sin²+cos² < 0.01，跳过角度计算
    
    # 只有当模型输出有效时才计算需要角度的辅助损失
    if not skip_angle_losses:
        # 反归一化到原始坐标空间: (B, 20, 4) -> (B, 20, 4)
        pred_x0_denorm = torch.zeros_like(pred_x0)
        pred_x0_denorm[:, :, :2] = pred_x0[:, :, :2] * 20.0  # x,y: [-1,1] → [-20,20]
        
        # 保持sin/cos编码，不恢复角度
        pred_x0_denorm[:, :, 2] = pred_x0[:, :, 2]  # sin(θ)
        pred_x0_denorm[:, :, 3] = pred_x0[:, :, 3]  # cos(θ)
        
        # 构建包含起点和终点的完整轨迹 (B, 22, 4)
        # 起点和终点应该被detach，因为它们是固定的边界条件，不应参与梯度计算
        # 直接从原始的start_pose和goal_pose构建（避免归一化再反归一化的精度损失）
        start_denorm = torch.zeros(B, 1, 4, device=device)
        start_denorm[:, 0, :2] = start_pose[:, :2].detach()  # x,y (已经是原始尺度)
        start_denorm[:, 0, 2] = torch.sin(start_pose[:, 2]).detach()  # sin(θ)
        start_denorm[:, 0, 3] = torch.cos(start_pose[:, 2]).detach()  # cos(θ)
        
        goal_denorm = torch.zeros(B, 1, 4, device=device)
        goal_denorm[:, 0, :2] = goal_pose[:, :2].detach()  # x,y (已经是原始尺度)
        goal_denorm[:, 0, 2] = torch.sin(goal_pose[:, 2]).detach()  # sin(θ)
        goal_denorm[:, 0, 3] = torch.cos(goal_pose[:, 2]).detach()  # cos(θ)
        
        # 拼接：[起点, 20个预测点, 终点] -> (B, 22, 4)
        full_traj_denorm = torch.cat([start_denorm, pred_x0_denorm, goal_denorm], dim=1)
    
    if loss_weights['main'] > 0:
        # 改进的损失计算（参考文生图模型）
        # 先对每个样本的空间维度求平均，再对batch求平均
        # 避免大batch的数值问题和梯度不均匀
        loss_per_sample = F.mse_loss(prediction, target, reduction='none')  # (B, N, 4)
        loss_per_sample = loss_per_sample.mean(dim=(1, 2))  # (B,) 先对N和4维度平均
        
        # 可选：应用时间步加权（让模型更关注干净数据区域）
        # 参考文生图模型：w(t) = t^k，对t接近1（干净数据）的样本给更高权重
        # 如果不需要加权，注释掉下面这行
        # t_weights = compute_timestep_weight(t_normalized.squeeze(), k=1.0)  # (B,)
        # loss_per_sample = loss_per_sample * t_weights
        
        main_loss = loss_per_sample.mean()  # 标量，再对batch平均
        
        if loss_type == 'epsilon':
            main_loss = main_loss * 0.5  # epsilon损失缩放0.5
        elif loss_type == 'x0':
            main_loss = main_loss * 0.1  # x0损失缩放0.1
        elif loss_type == 'v':
            main_loss = main_loss * 1e-3  # v损失缩放0.001
    
    # 计算各项损失（改进版平滑性损失 + 曲率约束）
    # 只在模型输出有效时才计算这些损失
    if not skip_angle_losses:  # 只有当 sin²+cos² 足够大时才计算
        if loss_weights['smoothness'] > 0:
            # 使用改进的平滑性损失：只惩罚超过阈值的加速度
            # 使用完整轨迹（包含起终点）来确保边界平滑
            # acceleration_threshold = 0.5  # 根据你的数据调整
            # smoothness_loss = compute_smoothness_loss(full_traj_denorm, threshold=acceleration_threshold)
            smoothness_loss = compute_smoothness_loss(full_traj_denorm, threshold=None)
        
        if loss_weights.get('curvature', 0) > 0:
            # 曲率约束：限制最大曲率（例如2.0 rad/m）
            # 使用完整轨迹来约束曲率
            max_curvature = 1.4  # 根据你的场景调整
            curvature_loss = compute_curvature_constraint_loss(full_traj_denorm, max_curvature=max_curvature)
        
        if loss_weights['angle_smoothness'] > 0:
            # 角度平滑性：使用完整轨迹确保起终点衔接平滑
            angle_smoothness_loss = compute_angle_smoothness_loss(full_traj_denorm)
        
        if loss_weights['angle_consistency'] > 0:
            # 角度一致性（防止倒车）：使用完整轨迹
            angle_consistency_loss = compute_angle_consistency_loss(full_traj_denorm)
        
        if loss_weights['uniformity'] > 0:
            # 均匀性：使用完整轨迹确保整体点间距均匀
            uniformity_loss = compute_uniformity_loss(full_traj_denorm)
    
    # 计算sin/cos归一化损失（不依赖角度恢复，总是计算）
    if loss_weights.get('sincos_norm', 0) > 0:
        sincos_norm_loss = compute_sincos_normalization_loss(pred_x0)
    
    # 计算倾覆监督损失（简化版本：直接使用优化器cost）
    # 只在阶段2执行，阶段1专注于基础轨迹生成
    if not skip_angle_losses and loss_weights['capsize'] > 0 and 'cost_map' in batch and is_training and current_stage == 2:
        cost_map = batch['cost_map'].to(device)  # (B, num_layers, max_anchors)
        
        # 地图配置
        map_size = (100, 100, 36)  # W, H, D for (x, y, yaw)
        resolution = 0.4
        origin = (-20.0, -20.0, -np.pi)  # x, y, yaw
        map_info = {
            'resolution': resolution,
            'origin': origin,
            'size': map_size
        }
        
        # 地图配置
        map_size = (100, 100, 36)  # W, H, D for (x, y, yaw)
        resolution = 0.4
        origin = (-20.0, -20.0, -np.pi)  # x, y, yaw
        map_info = {
            'resolution': resolution,
            'origin': origin,
            'size': map_size
        }
        
        # 反归一化预测的轨迹
        pred_x0_denorm = torch.zeros_like(pred_x0)
        pred_x0_denorm[:, :, :2] = pred_x0[:, :, :2] * 20.0
        pred_x0_denorm[:, :, 2] = pred_x0[:, :, 2]
        pred_x0_denorm[:, :, 3] = pred_x0[:, :, 3]
        
        # 计算物理损失（基于优化器代价）
        physics_costs = []
        for i in range(B):
            start_fixed = start_pose[i].detach()
            goal_fixed = goal_pose[i].detach()
            
            # 构建完整轨迹
            predicted_angles = torch.atan2(pred_x0_denorm[i, :, 2], pred_x0_denorm[i, :, 3])
            predicted_traj_3d = torch.stack([
                pred_x0_denorm[i, :, 0],
                pred_x0_denorm[i, :, 1],
                predicted_angles
            ], dim=1)
            
            full_traj = torch.cat([
                start_fixed.unsqueeze(0),
                predicted_traj_3d,
                goal_fixed.unsqueeze(0)
            ], dim=0)
            
            stability_cost_map = cost_map[i].permute(2, 0, 1)
            try:
                optimizer = TrajectoryOptimizerSE2(
                    full_traj.detach(),
                    stability_cost_map,
                    map_info,
                    device=device
                )
                cost = optimizer.cost_on_poses(full_traj)
                
                if not (torch.isnan(cost) or torch.isinf(cost) or cost.item() > 1000):
                    physics_costs.append(cost)
            except Exception as e:
                pass
        
        if len(physics_costs) > 0:
            # 直接使用优化器返回的cost作为损失
            capsize_loss = torch.stack(physics_costs).mean()
        else:
            capsize_loss = dummy_loss.clone()
        
        # 时间一致性损失（可选，保留以增强训练稳定性）
        consistency_loss = dummy_loss.clone()
        valid_mask = t > 0
        if valid_mask.any():
            t_minus_1 = torch.where(valid_mask, t - 1, t)
            
            # 在t-1时刻加噪
            noisy_traj_tm1 = model.q_sample(traj_normalized, t_minus_1, noise)
            
            # 预测t-1时刻的x0
            model_output_tm1 = model(map_input, noisy_traj_tm1, t_minus_1, start_normalized, goal_normalized)
            
            # 获取归一化的时间步
            if hasattr(model, 'module'):  # DataParallel
                t_tm1_normalized = model.module.timesteps_normalized[t_minus_1][:, None, None]
                t_eps = model.module.t_eps
            else:
                t_tm1_normalized = model.timesteps_normalized[t_minus_1][:, None, None]
                t_eps = model.t_eps
            
            if prediction_type == 'epsilon':
                pred_epsilon_tm1 = model_output_tm1
                pred_x0_tm1 = (noisy_traj_tm1 - (1 - t_tm1_normalized) * pred_epsilon_tm1) / torch.clamp(t_tm1_normalized, min=t_eps)
            elif prediction_type == 'x0':
                pred_x0_tm1 = model_output_tm1
            elif prediction_type == 'v':
                pred_v_tm1 = model_output_tm1
                pred_x0_tm1 = noisy_traj_tm1 + (1 - t_tm1_normalized) * pred_v_tm1
            
            # 时间一致性：t和t-1时刻预测的x0应该接近
            consistency_loss = F.mse_loss(
                pred_x0[valid_mask], 
                pred_x0_tm1[valid_mask].detach()  # detach避免循环依赖
            )
            consistency_loss = dummy_loss.clone()
    elif not is_training and not skip_angle_losses and loss_weights['capsize'] > 0 and 'cost_map' in batch:
        # =====================
        # 验证模式：使用完整采样得到的轨迹计算cost
        # =====================
        cost_map = batch['cost_map'].to(device)
        map_size = (100, 100, 36)
        resolution = 0.4
        origin = (-20.0, -20.0, -np.pi)
        map_info = {
            'resolution': resolution,
            'origin': origin,
            'size': map_size
        }
        
        sample_costs = []
        for i in range(B):
            try:
                # 获取单个样本的输入
                map_single = map_input[i:i+1]  # (1, C, H, W)
                start_single = start_normalized[i:i+1]  # (1, 4)
                goal_single = goal_normalized[i:i+1]  # (1, 4)
                
                # 完整采样（使用DDIM加速，50步）
                with torch.no_grad():
                    if isinstance(model, nn.DataParallel):
                        sampled_traj = model.module.sample(
                            map_single, start_single, goal_single, 
                            num_samples=1, ddim_steps=50
                        )[0]  # (20, 4)
                    else:
                        sampled_traj = model.sample(
                            map_single, start_single, goal_single,
                            num_samples=1, ddim_steps=50
                        )[0]  # (20, 4)
                
                # 反归一化采样结果
                sampled_denorm = torch.zeros_like(sampled_traj)
                sampled_denorm[:, :2] = sampled_traj[:, :2] * 20.0  # x,y
                sampled_denorm[:, 2:] = sampled_traj[:, 2:]  # sin/cos
                
                # 构建完整轨迹并计算cost
                start_fixed = start_pose[i].detach()
                goal_fixed = goal_pose[i].detach()
                
                # 从sin/cos恢复角度
                sampled_angles = torch.atan2(sampled_denorm[:, 2], sampled_denorm[:, 3])
                sampled_traj_3d = torch.stack([
                    sampled_denorm[:, 0],  # x
                    sampled_denorm[:, 1],  # y
                    sampled_angles         # θ
                ], dim=1)  # (20, 3)
                
                full_sampled_traj = torch.cat([
                    start_fixed.unsqueeze(0),  # 起点
                    sampled_traj_3d,           # 采样的20个点
                    goal_fixed.unsqueeze(0)    # 终点
                ], dim=0)  # (22, 3)
                
                # 计算cost
                stability_cost_map = cost_map[i].permute(2, 0, 1)
                optimizer_traj = TrajectoryOptimizerSE2(
                    full_sampled_traj.detach(),
                    stability_cost_map,
                    map_info,
                    device=device
                )
                cost = optimizer_traj.cost_on_poses(full_sampled_traj)
                
                if not (torch.isnan(cost) or torch.isinf(cost) or cost.item() > 1000):
                    sample_costs.append(cost.item())
            except Exception as e:
                print(f"Warning: Failed to compute cost for sample {i}: {e}")
                continue
        
        # 返回平均cost作为capsize_loss
        if len(sample_costs) > 0:
            capsize_loss = dummy_loss + sum(sample_costs) / len(sample_costs)
        else:
            capsize_loss = dummy_loss.clone()
        
        consistency_loss = dummy_loss.clone()
    else:
        # 如果不满足条件，初始化为dummy
        if 'capsize_loss' not in locals():
            capsize_loss = dummy_loss.clone()
        if 'consistency_loss' not in locals():
            consistency_loss = dummy_loss.clone()
    
    # 混合损失（添加时间一致性）
    total_loss = (
        loss_weights['main'] * main_loss +
        loss_weights['smoothness'] * smoothness_loss +
        loss_weights.get('curvature', 0.0) * curvature_loss +
        loss_weights['angle_smoothness'] * angle_smoothness_loss +
        loss_weights['angle_consistency'] * angle_consistency_loss +
        loss_weights['uniformity'] * uniformity_loss +
        loss_weights.get('sincos_norm', 0.0) * sincos_norm_loss +
        loss_weights['capsize'] * capsize_loss +
        loss_weights.get('consistency', 0.0) * consistency_loss  # 新增：时间一致性损失
    )
    
    # # 调试信息（5%概率打印，帮助监控训练初期）
    # if torch.rand(1).item() < 0.05:
    #     print(f"\n[训练步骤调试 | sin²+cos²平均: {mean_norm_sq:.4f} | 跳过角度损失: {skip_angle_losses}]")
    #     print(f"  主损失: {main_loss.item():.6f}")
    #     if not skip_angle_losses:
    #         print(f"  平滑性损失: {smoothness_loss.item():.6f}")
    #         print(f"  曲率损失: {curvature_loss.item():.6f}")
    #         print(f"  角度平滑性损失: {angle_smoothness_loss.item():.6f}")
    #         print(f"  角度一致性损失: {angle_consistency_loss.item():.6f}")
    #         print(f"  均匀性损失: {uniformity_loss.item():.6f}")
    #     print(f"  倾覆损失: {capsize_loss.item():.6f}")
    #     print(f"  一致性损失: {consistency_loss.item():.6f}")
    #     print(f"  总损失: {total_loss.item():.6f}")
    
    # 检查损失异常
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print(f"❌ 损失异常: {total_loss.item()}")
        print(f"  sin²+cos²平均: {mean_norm_sq:.4f} (跳过角度损失: {skip_angle_losses})")
        print(f"  主损失: {main_loss.item()}, 平滑: {smoothness_loss.item()}, 曲率: {curvature_loss.item()}, "
              f"角度平滑性: {angle_smoothness_loss.item()}, 角度一致性: {angle_consistency_loss.item()}, "
              f"均匀: {uniformity_loss.item()}, 倾覆: {capsize_loss.item()}, 一致性: {consistency_loss.item()}")
        raise ValueError("Loss is NaN or Inf")
    
    # 返回各项损失用于记录
    loss_dict = {
        'main': main_loss.item(),
        'smoothness': smoothness_loss.item(),
        'curvature': curvature_loss.item(),
        'angle_smoothness': angle_smoothness_loss.item(),
        'angle_consistency': angle_consistency_loss.item(),
        'uniformity': uniformity_loss.item(),
        'sincos_norm': sincos_norm_loss.item(),
        'capsize': capsize_loss.item(),
        'consistency': consistency_loss.item()  # 新增
    }
    
    return total_loss, 0, B, loss_dict


def train_epoch(model, trainingData, optimizer, device, stage_epoch=0, loss_weights=None, current_stage=1, total_stage_epochs=50):
    """
    单轮训练函数
    """
    model.train()
    total_loss = 0
    total_samples = 0
    
    # 累积各项损失（添加consistency）
    loss_accumulator = {'main': 0, 'smoothness': 0, 'curvature': 0, 'angle_smoothness': 0, 'angle_consistency': 0, 'uniformity': 0, 'sincos_norm': 0, 'capsize': 0, 'consistency': 0}
    
    pbar = tqdm(trainingData, mininterval=2, desc=f"Training Epoch {stage_epoch} (Stage {current_stage})")
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()
        
        loss, _, n_samples, loss_dict = diffusion_loss(
            model, batch, device, loss_weights, 
            epoch=stage_epoch, total_epochs=total_stage_epochs,
            is_training=True,  # 训练模式：capsize loss使用回归损失
            current_stage=current_stage  # 传递当前阶段
        )
        
        total_loss += loss.item()
        total_samples += n_samples
        
        for key in loss_accumulator.keys():
            loss_accumulator[key] += loss_dict[key]
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss, skipping batch")
            continue
        
        loss.backward()
        
        # 梯度裁剪 - 第二阶段使用适中裁剪（采样链梯度累积大）
        if current_stage == 1:
            clip_value = 1.0  # 阶段1：正常裁剪
        else:
            clip_value = 1.0  # 阶段2：中等裁剪（采样链累积多步梯度）
        
        original_grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=clip_value, norm_type=2
        )
        was_clipped = original_grad_norm > clip_value
        
        optimizer.step_and_update_lr()
        
        # 更新进度条
        grad_info = f'{original_grad_norm:.2f}→{clip_value}' if was_clipped else f'{original_grad_norm:.2f}'
        
        # 根据当前阶段显示不同的信息
        if current_stage == 1:
            # 阶段1：显示主损失、平滑性
            pbar.set_postfix({
                'Total': f'{loss.item():.4f}',
                'Main': f'{loss_dict["main"]:.4f}',
                'Smooth': f'{loss_dict["smoothness"]:.5f}',
                'SinCos': f'{loss_dict["sincos_norm"]:.5f}',
                'GradNorm': grad_info
            })
        else:
            # 阶段2：显示倾覆损失
            pbar.set_postfix({
                'Total': f'{loss.item():.4e}', # 科学计数法
                'Capsize': f'{loss_dict["capsize"]*loss_weights["capsize"]:.6e}', # 科学计数法
                'SinCos': f'{loss_dict["sincos_norm"]:.5f}',
                'GradNorm': grad_info
            })
    
    avg_loss = total_loss / len(trainingData) if len(trainingData) > 0 else 0
    
    # 计算各项平均损失
    avg_loss_dict = {k: v / len(trainingData) for k, v in loss_accumulator.items()}
    
    return avg_loss, 0, total_samples, avg_loss_dict


def eval_epoch(model, validationData, device, loss_weights=None, epoch=0, total_epochs=100, current_stage=1, use_late_timesteps=False):
    """
    单轮评估函数
    """
    model.eval()
    total_loss = 0
    total_samples = 0
    
    # 累积各项损失
    loss_accumulator = {'main': 0, 'smoothness': 0, 'curvature': 0, 'angle_smoothness': 0, 'angle_consistency': 0, 'uniformity': 0, 'sincos_norm': 0, 'capsize': 0}
    
    with torch.no_grad():
        for batch in tqdm(validationData, mininterval=2, desc="Validation"):
            
            loss, _, n_samples, loss_dict = diffusion_loss(
                model, batch, device, loss_weights,
                epoch=epoch, total_epochs=total_epochs,
                is_training=False,  # 验证模式：capsize loss返回实际cost值
                current_stage=current_stage  # 传递当前阶段
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
    parser.add_argument('--stage1_epochs', help="Number of epochs for stage 1", type=int, default=70)
    parser.add_argument('--stage2_epochs', help="Number of epochs for stage 2", type=int, default=50)
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
        n_path_steps=20,  # 不计起点 + 20个中间点 + 不计终点 (使用4维编码: x,y,sin(θ),cos(θ))
        diffusion_steps=50,
        prediction_type=prediction_type,  # 'epsilon', 'x0', or 'v' - 模型输出什么
        loss_type=loss_type,  # 与prediction_type保持一致
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
    env_list = ["env000008", "env000009"]
    
    # =================== 两阶段训练配置 ===================
    # 阶段1配置：注重基础轨迹预测
    # stage1_config = {
    #     'epochs': args.stage1_epochs,
    #     'lr_mul': 1e-1,  # 学习率倍增器
    #     'loss_weights': {
    #         'main': 1e0,          # 主预测损失
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
            'main': 1e1,          # 主预测损失
            'smoothness': 0e-5,   # 平滑性损失
            'curvature': 0e-4,    # 曲率约束（限制最大曲率）- 新增
            'angle_smoothness': 0e-2,  # 角度平滑性损失
            'angle_consistency': 0e-6,  # 角度一致性损失(防止倒车)
            'uniformity': 0e-4,   # 均匀性损失（点间距离方差）
            'sincos_norm': 1e-1,   # sin/cos归一化损失（确保sin²+cos²≈1）
            'capsize': 0.0        # 倾覆监督损失（第一阶段不启用）
        }
    }
    
    # 阶段2配置：通过梯度优化学习低cost轨迹
    # **核心改变**：使用完整DDIM采样链计算cost，而非单步预测
    # **关键**：保留main loss作为正则化，capsize loss作为优化目标
    stage2_config = {
        'epochs': args.stage2_epochs,
        'lr_mul': 1e-4,  # 小学习率微调（不是从头训练！）
        'loss_weights': {
            'main': 0e-1,        # 保留主损失作为正则化（不能为0！）
            'smoothness': 0e-5,   # 平滑性损失
            'curvature': 0e-4,    # 曲率约束（限制最大曲率）- 新增
            'angle_smoothness': 0e-2,  # 角度平滑性损失
            'angle_consistency': 0e-6,  # 角度一致性损失
            'uniformity': 0e-4,   # 均匀性损失
            'sincos_norm': 1e-2,   # sin/cos归一化损失（确保sin²+cos²≈1）
            'capsize': 1e-4,      # 倾覆优化损失（主要优化目标）
            'consistency': 0.0    # 时间一致性损失（当前不启用）
        }
    }
    
    # 初始化当前阶段（在数据加载前）
    current_stage = args.stage
    
    # 根据当前阶段决定是否计算stability map（第二阶段需要倾覆损失时启用）
    if current_stage == 1:
        loss_weights = stage1_config['loss_weights']
    else:
        loss_weights = stage2_config['loss_weights']
    
    compute_stability = (current_stage == 2 and loss_weights.get('capsize', 0.0) > 0)
    
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
        
        if args.stage == 1 and saved_stage == 2:
            print("Warning: Checkpoint is from stage 2, but --stage=1 specified")
            print("   Continuing from stage 2 instead")
            current_stage = 2
            loss_weights = stage2_config['loss_weights']
            current_lr_mul = stage2_config['lr_mul']
            # 继续第二阶段的训练
            start_epoch = saved_epoch + 1
            
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
            start_epoch = saved_epoch + 1
            
        else:  # args.stage == 2 and saved_stage == 2
            # 继续第二阶段的训练
            current_stage = 2
            loss_weights = stage2_config['loss_weights']
            current_lr_mul = stage2_config['lr_mul']
            start_epoch = saved_epoch + 1
        
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
            print(f"✓ Enabled capsize loss: {stage2_config['loss_weights']['capsize']}")
            print()

            
            # 重置最佳验证损失，用于阶段2
            best_val_loss = float('inf')
            
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
        )
        
        # 验证
        val_loss, _, _, val_loss_dict = eval_epoch(
            model, validationData, device, loss_weights, stage_epoch, total_stage_epochs, 
            current_stage=current_stage
        )
        
        # 记录
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"\n[Stage {current_stage}] Epoch {stage_epoch}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"    - Main: {train_loss_dict['main']:.4f}")
        print(f"    - Smoothness: {train_loss_dict['smoothness']:.4f}")
        print(f"    - Curvature: {train_loss_dict['curvature']:.4f}")
        print(f"    - Angle Smoothness: {train_loss_dict['angle_smoothness']:.4f}")
        print(f"    - Angle Consistency: {train_loss_dict['angle_consistency']:.4f}")
        print(f"    - Uniformity: {train_loss_dict['uniformity']:.4f}")
        print(f"    - Sin/Cos Norm: {train_loss_dict['sincos_norm']:.4f}")
        print(f"    - Capsize: {train_loss_dict['capsize']:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"    - Main: {val_loss_dict['main']:.4f}")
        print(f"    - Smoothness: {val_loss_dict['smoothness']:.4f}")
        print(f"    - Curvature: {val_loss_dict['curvature']:.4f}")
        print(f"    - Angle Smoothness: {val_loss_dict['angle_smoothness']:.4f}")
        print(f"    - Angle Consistency: {val_loss_dict['angle_consistency']:.4f}")
        print(f"    - Uniformity: {val_loss_dict['uniformity']:.4f}")
        print(f"    - Sin/Cos Norm: {val_loss_dict['sincos_norm']:.4f}")
        print(f"    - Capsize: {val_loss_dict['capsize']:.4f}")
        
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
        writer.add_scalar('Loss/train_sincos_norm', train_loss_dict['sincos_norm'], stage_epoch)
        writer.add_scalar('Loss/train_capsize', train_loss_dict['capsize'], stage_epoch)
        
        writer.add_scalar('Loss/val_main', val_loss_dict['main'], stage_epoch)
        writer.add_scalar('Loss/val_smoothness', val_loss_dict['smoothness'], stage_epoch)
        writer.add_scalar('Loss/val_curvature', val_loss_dict['curvature'], stage_epoch)
        writer.add_scalar('Loss/val_angle_smoothness', val_loss_dict['angle_smoothness'], stage_epoch)
        writer.add_scalar('Loss/val_angle_consistency', val_loss_dict['angle_consistency'], stage_epoch)
        writer.add_scalar('Loss/val_uniformity', val_loss_dict['uniformity'], stage_epoch)
        writer.add_scalar('Loss/val_sincos_norm', val_loss_dict['sincos_norm'], stage_epoch)
        writer.add_scalar('Loss/val_capsize', val_loss_dict['capsize'], stage_epoch)
        
        writer.add_scalar('LR', optimizer._optimizer.param_groups[0]['lr'], stage_epoch)
        
        # 保存最佳模型（根据阶段选择不同的保存路径）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            if isinstance(model, nn.DataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            # 根据当前阶段选择保存路径
            best_model_path = stage1_best_path if current_stage == 1 else stage2_best_path
            
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
            }, best_model_path)
            
            print(f"  ✓ Saved best model for Stage {current_stage} to {osp.basename(best_model_path)} (val_loss={val_loss:.4f})")
        
        # 定期保存检查点（每5个epoch）- 作为备份
        if (stage_epoch + 1) % 5 == 0:
            if isinstance(model, nn.DataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
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
        print(f"Stage 1 best validation loss: {stage1_best_loss:.4f}")
    print(f"Stage {current_stage} best validation loss: {best_val_loss:.4f}")
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
    
    # Print summary of best models
    if stage1_best_loss < float('inf'):
        print(f"Stage 1 best validation loss: {stage1_best_loss:.4f}")
        print(f"  Saved to: {stage1_best_path}")
    if current_stage == 2:
        print(f"Stage 2 best validation loss: {best_val_loss:.4f}")
        print(f"  Saved to: {stage2_best_path}")
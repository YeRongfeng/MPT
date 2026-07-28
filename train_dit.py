"""
train_dit.py - 训练不平坦地面路径预测模型(基于扩散模型)
"""

import numpy as np
import pickle
import random
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
from map_config import MAP_CONFIG, MAP_HALF_EXTENT, discover_environments

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
    - 本函数只提取中间24点；训练时再拼入标签起终点，构成完整26点监督目标
    
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


def adaptive_pmf_loss(loss_all, norm_p=1.0, norm_eps=0.01):
    """
    Pixel/MeanFlow 风格的 per-sample adaptive weighting。

    Args:
        loss_all: (B, ..., D) unreduced MSE loss
    Returns:
        adp_loss: 用于反传的 adaptive loss
        raw_loss: 原始 per-sample MSE 均值，仅用于诊断
        weight_mean: stop-grad 权重均值，仅用于诊断
    """
    loss_per_sample = loss_all.flatten(start_dim=1).mean(dim=1)
    weight = (loss_per_sample + norm_eps).pow(norm_p).detach()
    adp_loss = (loss_per_sample / weight).mean()
    raw_loss = loss_per_sample.mean().detach()
    weight_mean = weight.mean().detach()
    return adp_loss, raw_loss, weight_mean


def sparse_boundary_violation_loss(
    points_normalized,
    safe_bound=0.98,
):
    """Only penalize point coordinates that violate the safe box.

    The loss is normalized by the number of active violating coordinates,
    rather than all B*N*2 coordinates, so rare violations are not diluted.
    """
    if points_normalized.ndim != 3 or points_normalized.shape[-1] != 2:
        raise ValueError(
            "Expected normalized points shaped (B,N,2), got "
            f"{tuple(points_normalized.shape)}"
        )
    if not 0.0 < safe_bound <= 1.0:
        raise ValueError(f"safe_bound must be in (0, 1], got {safe_bound}")

    violation = F.relu(points_normalized.abs() - safe_bound)
    active = violation > 0
    active_count = active.sum()
    boundary_loss = (
        violation.square().sum()
        / active_count.to(violation.dtype).clamp_min(1.0)
    )

    per_sample_max = violation.flatten(start_dim=1).amax(dim=1)
    oob_samples = per_sample_max > 0
    if bool(oob_samples.any()):
        oob_sample_loss = per_sample_max[oob_samples].square().mean()
    else:
        oob_sample_loss = violation.sum() * 0.0

    diagnostics = {
        'oob_sample_loss': oob_sample_loss,
        'oob_rate': oob_samples.float().mean(),
        'max_oob': per_sample_max.max(),
        'active_boundary_fraction': active.float().mean(),
    }
    return boundary_loss, diagnostics


def stage2_optimize_control_points(
    model, map_input, start_normalized, goal_normalized, start_pose, goal_pose,
    stability_cost_map, map_info, device, prediction_type='epsilon', num_iterations=10, lr=0.01
):
    """
    【第二阶段优化】对每个batch样本进行采样和优化，返回1条优化的控制点
    
    在缩放零和边残差空间优化，并在计算B样条代价时解码为26个控制点。
    
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
    base_model = model.module if hasattr(model, 'module') else model
    representation = base_model.trajectory_representation
    coordinate_scale = base_model.coordinate_scale
    
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
        
        sampled_control_points_normalized = sampled_control_points / coordinate_scale
        residual_sample = representation.encode(
            sampled_control_points_normalized,
            start_normalized[:, :2],
            goal_normalized[:, :2],
        ).detach()
        residual_sample.requires_grad_(True)
        
        # 为整个batch创建优化器
        optimizer = torch.optim.AdamW([residual_sample], lr=lr)
        
        # batch级优化循环
        for iter in range(num_iterations):
            optimizer.zero_grad()
            
            full_control_points = representation.decode(
                residual_sample,
                start_normalized[:, :2],
                goal_normalized[:, :2],
            ) * coordinate_scale
            
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
            torch.nn.utils.clip_grad_norm_([residual_sample], 1.0)
            
            # 优化步
            optimizer.step()
        
        # 返回优化后的控制点
        optimized_control_points = representation.decode(
            residual_sample,
            start_normalized[:, :2],
            goal_normalized[:, :2],
        ) * coordinate_scale
        optimized_middle_cp = optimized_control_points[:, 1:-1, :].detach()
    
    return optimized_middle_cp



def diffusion_loss(
    model,
    batch,
    device,
    loss_weights=None,
    epoch=0,
    total_epochs=100,
    is_training=True,
    current_stage=1,
    num_dense_points=100,
    prediction_type='x0',
    sampling_generator=None,
):
    """
    训练损失：25个物理空间25×缩放零和边残差上的pMF主损失。
    
    Args:
        loss_weights: dict with keys
            ['main', 'main_norm_p', 'boundary', 'boundary_safe_bound', 'capsize']
        epoch: 当前epoch（用于动态权重）
        current_stage: 当前训练阶段（1或2），只在阶段2执行高级损失计算
        total_epochs: 总epoch数（用于动态权重）
        is_training: bool - True时使用回归损失，False时返回实际cost值
        num_dense_points: int - 感知/安全损失使用的B样条密集采样点数
        prediction_type: str - 预测类型 ('epsilon', 'x0', 'v')
    """
    # 默认权重
    if loss_weights is None:
        loss_weights = {
            'main': 1.0,
            'main_norm_p': 1.0,
            'boundary': 0.0,
            'boundary_safe_bound': 0.98,
            'capsize': 0.0,
        }
    map_input = batch['map'].float().to(device)
    trajectory = batch['trajectory'].to(device)  # (B, 100, 3)
    start_pose = batch['start_pose'].to(device)  # (B, 3)
    goal_pose = batch['goal_pose'].to(device)  # (B, 3)
    
    B = map_input.shape[0]
    base_model = model.module if hasattr(model, 'module') else model
    loss_type = getattr(base_model, 'loss_type', prediction_type)
    representation = base_model.trajectory_representation
    coordinate_scale = base_model.coordinate_scale
    
    # =================== B样条控制点转换 ===================
    # 【语义说明】
    # - 完整B样条：26个控制点 = 起点(1) + 中间点(24) + 终点(1)
    # - 标签起终点与中间24点组成完整26点控制点
    # - pMF状态和网络输出是25条物理边的25×缩放零和残差
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
    #         start_normalized_temp[:, :2] = start_pose[:, :2] / coordinate_scale
    #         start_normalized_temp[:, 2] = torch.cos(start_pose[:, 2])
    #         start_normalized_temp[:, 3] = torch.sin(start_pose[:, 2])
    #         start_normalized_temp[:, :2] = torch.clamp(start_normalized_temp[:, :2], -1.0, 1.0)
            
    #         goal_normalized_temp = torch.zeros(B, 4, device=device)
    #         goal_normalized_temp[:, :2] = goal_pose[:, :2] / coordinate_scale
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
    #             map_info=MAP_CONFIG.cost_map_info(),
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
    
    # 居中地图坐标按统一的半边长归一化到 [-1, 1]。
    middle_cp_normalized = middle_cp / coordinate_scale
    middle_cp_normalized = torch.clamp(middle_cp_normalized, -1.0, 1.0)
    
    # 归一化起点终点坐标（转换为4维：x, y, cos(θ), sin(θ)） - 用于条件
    start_normalized = torch.zeros(B, 4, device=device)
    start_normalized[:, :2] = start_pose[:, :2] / coordinate_scale
    start_normalized[:, 2] = torch.cos(start_pose[:, 2])  # cos(θ)
    start_normalized[:, 3] = torch.sin(start_pose[:, 2])  # sin(θ)
    start_normalized[:, :2] = torch.clamp(start_normalized[:, :2], -1.0, 1.0)
    
    goal_normalized = torch.zeros(B, 4, device=device)
    goal_normalized[:, :2] = goal_pose[:, :2] / coordinate_scale
    goal_normalized[:, 2] = torch.cos(goal_pose[:, 2])  # cos(θ)
    goal_normalized[:, 3] = torch.sin(goal_pose[:, 2])  # sin(θ)
    goal_normalized[:, :2] = torch.clamp(goal_normalized[:, :2], -1.0, 1.0)

    full_control_points_normalized = torch.cat(
        [
            start_normalized[:, :2].unsqueeze(1),
            middle_cp_normalized,
            goal_normalized[:, :2].unsqueeze(1),
        ],
        dim=1,
    )  # (B, 26, 2)

    x0_residual = representation.encode(
        full_control_points_normalized,
        start_normalized[:, :2],
        goal_normalized[:, :2],
    )  # (B, 25, 2)
    
    # ===== pixel Mean Flow: 连续时间采样 =====
    # 采样 t 和 r (0 <= r <= t <= 1)
    if hasattr(model, 'module'):
        t = model.module.sample_timesteps(
            B, device=device, generator=sampling_generator
        )
    else:
        t = model.sample_timesteps(
            B, device=device, generator=sampling_generator
        )
    t = torch.clamp(t, min=1e-4, max=1.0 - 1e-4) # 避免极端值
    r = torch.rand(
        t.shape,
        device=t.device,
        dtype=t.dtype,
        generator=sampling_generator,
    ) * t
    noise_residual = torch.randn(
        x0_residual.shape,
        device=x0_residual.device,
        dtype=x0_residual.dtype,
        generator=sampling_generator,
    )
    noise_residual = representation.project_zero_sum(noise_residual)

    def assert_zero_sum(name, tensor, tolerance=1e-5):
        # Check the mean rather than the raw sum: the constraint is defined as
        # mean==0 and the raw sum magnifies harmless float32 roundoff by 25.
        max_error = tensor.mean(dim=1).abs().max().detach()
        if not torch.isfinite(max_error) or max_error.item() > tolerance:
            raise AssertionError(
                f"{name} left the zero-sum subspace: "
                f"max mean error={max_error.item():.3e}"
            )

    assert_zero_sum("x0_residual", x0_residual)
    assert_zero_sum("noise_residual", noise_residual)

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
            # JVP/forward-AD 不支持部分 fused SDPA kernel；CPU后端也可能自动
            # 选择 fused attention，因此不再只按 is_cuda 条件启用该保护。
            sdp_ctx = torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True
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

        p_x0 = representation.project_zero_sum(p_x0)
        # 返回平均速度 u = (z_t - x_0) / t
        return representation.project_zero_sum(
            (z_arg - p_x0) / (t_v + 1e-5)
        )

    # 2. 准备 JVP 的输入 (Primals) 和 变化率 (Tangents)
    target_v = representation.project_zero_sum(
        noise_residual - x0_residual
    )  # dz/dt 的真值

    # noisy_cp 必须在 requires_grad 环境下生成，确保导数链条完整
    with torch.enable_grad():
        t.requires_grad_(True)
        t_v = t.view(-1, 1, 1)
        # 显式重算 noisy_cp 确保它是 t 的函数
        z_t = (
            (1.0 - t_v) * x0_residual
            + t_v * noise_residual
        )
        z_t = representation.project_zero_sum(z_t)
        assert_zero_sum("z_t", z_t)
        
        # Primals: 当前点
        primals = (z_t, t, r)
        # Tangents: 方向。当 t 变化 1 时，z 变化 target_v，r 不变
        tangents = (target_v, torch.ones_like(t), torch.zeros_like(r))
        
        # 3. 计算 JVP (全导数)
        # u_out: 算出的 u
        # du_dt_full: 算出的全导数 du/dt
        u_out, du_dt_full = jvp(u_fn, primals, tangents)
        u_out = representation.project_zero_sum(u_out)
        assert_zero_sum("u_out", u_out)
        
        # 对修正项进行幅度裁剪，增加稳定性
        du_dt_full = torch.clamp(du_dt_full, -5.0, 5.0)

    # 4. 构建 V_theta 并计算 Loss
    # V = u + (t - r) * stopgrad(du/dt)
    
    V_theta = representation.project_zero_sum(
        u_out
        + (t.view(-1, 1, 1) - r.view(-1, 1, 1)).detach()
        * du_dt_full.detach()
    )

    if loss_type == 'v':
        main_loss_all = F.mse_loss(V_theta, target_v, reduction='none')  # (B, 25, 2)
    elif loss_type == 'x0':
        # x0_rec = z_t - t * V_theta
        pred_x0_corrected = representation.project_zero_sum(
            z_t - t.view(-1, 1, 1) * V_theta
        )
        main_loss_all = F.mse_loss(
            pred_x0_corrected, x0_residual, reduction='none'
        )
    else:
        # epsilon 空间的 pMF 修正写法：pred_eps_corrected = V_theta + pred_x0_corrected
        # 但推荐统一使用 v-loss 以符合论文实现
        main_loss_all = F.mse_loss(V_theta, target_v, reduction='none') # (B, 25, 2)

    # 第一组消融实验只改变主pMF损失的自适应指数；其余损失和网络保持不变。
    main_norm_p = float(loss_weights.get('main_norm_p', 1.0))
    main_loss_adp_all, main_raw_mse_all, main_adp_weight_mean_all = adaptive_pmf_loss(
        main_loss_all, norm_p=main_norm_p
    )
    
    # 【第二阶段特殊处理】分离模仿和优化两部分的损失，然后加权
    if stage2_mix_loss:
        # 分离损失：(2B, 24, 2) -> (B, 24, 2) + (B, 24, 2)
        split_idx = len(main_loss_all) // 2
        main_loss_imitation, main_raw_mse_imitation, main_adp_weight_imitation = adaptive_pmf_loss(
            main_loss_all[:split_idx], norm_p=main_norm_p
        )
        main_loss_optimized, main_raw_mse_optimized, main_adp_weight_optimized = adaptive_pmf_loss(
            main_loss_all[split_idx:], norm_p=main_norm_p
        )
        
        # # 根据训练进度动态调整混合系数
        # # 早期更多使用模仿损失，保持在原分布附近
        # # 后期逐渐增加优化损失的比例
        # progress = epoch / (total_epochs + 1e-8)
        # # alpha从0.8衰减到0.2：模仿损失比例从80%到20%
        # alpha_loss = 0.8 - 0.6 * progress  # 范围 [0.8, 0.2]
        
        alpha_loss = 0.98
        
        # 加权组合两个损失
        main_loss = alpha_loss * main_loss_imitation + (1.0 - alpha_loss) * main_loss_optimized
        main_raw_mse = alpha_loss * main_raw_mse_imitation + (1.0 - alpha_loss) * main_raw_mse_optimized
        main_adp_weight_mean = alpha_loss * main_adp_weight_imitation + (1.0 - alpha_loss) * main_adp_weight_optimized
    else:
        # 第一阶段或第二阶段但没有优化数据：只用模仿数据
        main_loss = main_loss_adp_all
        main_raw_mse = main_raw_mse_all
        main_adp_weight_mean = main_adp_weight_mean_all

    # 从u恢复25×缩放零和边残差，再解码到物理控制点。
    x0_residual_pred = representation.project_zero_sum(
        z_t - (t.view(-1, 1, 1) + 1e-5) * u_out
    )

    # State-scale and zero-predictor diagnostics. These are deliberately
    # unweighted raw MSE statistics, independent of the adaptive pMF loss.
    residual_rms = x0_residual.detach().square().mean().sqrt()
    unscaled_residual_rms = residual_rms / representation.residual_scale
    noise_rms = noise_residual.detach().square().mean().sqrt()
    zero_pred_mse = x0_residual.detach().square().mean()
    model_x0_mse = (
        x0_residual_pred.detach() - x0_residual.detach()
    ).square().mean()
    explained_ratio = 1.0 - model_x0_mse / zero_pred_mse.clamp_min(1e-12)
    noise_to_residual_ratio = noise_rms / residual_rms.clamp_min(1e-12)
    radial_diagnostics = representation.radial_diagnostics_from_residual(
        x0_residual_pred,
        start_normalized[:, :2],
        goal_normalized[:, :2],
    )
    gt_radial_diagnostics = representation.radial_diagnostics_from_residual(
        x0_residual.detach(),
        start_normalized[:, :2],
        goal_normalized[:, :2],
    )
    radial_x_norm = radial_diagnostics['x_norm'].detach()
    radial_rho = radial_diagnostics['rho'].detach()
    hard_boundary_margin = radial_diagnostics['hard_boundary_margin'].detach()
    gt_radial_x_norm = gt_radial_diagnostics['x_norm'].detach()

    needs_control_points = (
        loss_weights.get('boundary', 0.0) > 0.0
        or (is_training and loss_weights.get('capsize', 0.0) > 0.0)
    )
    x0_control_pred = None
    if needs_control_points:
        x0_control_pred = representation.decode(
            x0_residual_pred,
            start_normalized[:, :2],
            goal_normalized[:, :2],
        )
    dummy_loss = main_loss.sum() * 0.0
    boundary_loss = dummy_loss.clone()
    boundary_diagnostics = {
        'oob_sample_loss': dummy_loss.detach().clone(),
        'oob_rate': dummy_loss.detach().clone(),
        'max_oob': dummy_loss.detach().clone(),
        'active_boundary_fraction': dummy_loss.detach().clone(),
        'dense_oob_rate': dummy_loss.detach().clone(),
        'dense_max_oob': dummy_loss.detach().clone(),
    }
    pred_dense_normalized = None
    if loss_weights.get('boundary', 0.0) > 0.0:
        safe_bound = float(loss_weights.get('boundary_safe_bound', 0.98))
        boundary_loss, boundary_diagnostics = sparse_boundary_violation_loss(
            x0_control_pred,
            safe_bound=safe_bound,
        )
        # Control points define the conservative training constraint. The
        # dense B-spline metrics report actual curve violations separately.
        bspline_layer = DifferentiableBSpline(
            num_control_points=26,
            num_output_points=num_dense_points,
            degree=3,
        ).to(device=x0_control_pred.device, dtype=x0_control_pred.dtype)
        pred_dense_normalized = bspline_layer(x0_control_pred)
        _, dense_boundary_diagnostics = sparse_boundary_violation_loss(
            pred_dense_normalized,
            safe_bound=safe_bound,
        )
        boundary_diagnostics['dense_oob_rate'] = dense_boundary_diagnostics['oob_rate']
        boundary_diagnostics['dense_max_oob'] = dense_boundary_diagnostics['max_oob']

    capsize_loss = dummy_loss.clone()

    if loss_weights.get('capsize', 0.0) > 0.0:
        # =================== 第二阶段：物理约束 ===================
        # 训练阶段使用“当前前向预测的 x0”重建轨迹，避免随机采样链的高方差导致
        # 任一单项辅助损失都把分布推向单模态（与条件解绑）。
        if is_training:
            if pred_dense_normalized is None:
                if x0_control_pred is None:
                    raise RuntimeError("Decoded control points are required for capsize loss.")
                bspline_layer = DifferentiableBSpline(
                    num_control_points=26,
                    num_output_points=num_dense_points,
                    degree=3,
                ).to(device=x0_control_pred.device, dtype=x0_control_pred.dtype)
                pred_dense_normalized = bspline_layer(x0_control_pred)
            reconstructed_traj = pred_dense_normalized * coordinate_scale
            
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
                solver='pmf_refined',
                reconstruct_trajectory=True,
                num_traj_points=100
            )
            # model.sample 在 num_samples>1 时返回 (B*num_samples, N, 2)
            start_pose_expanded = start_pose.repeat_interleave(10, dim=0)
            goal_pose_expanded = goal_pose.repeat_interleave(10, dim=0)
        
        # 使用整批 cost_map，避免错误地只取 batch[0]
        stability_cost_map = batch['cost_map'].to(device)
        map_info = MAP_CONFIG.cost_map_info()
        
        from grad_optimizer import cost_on_dense_trajectory
        capsize_loss = cost_on_dense_trajectory(
            reconstructed_traj, start_pose_expanded, goal_pose_expanded,
            stability_cost_map, map_info, device
        )

        # 数值稳定性保护：capsize_loss 出现 NaN/Inf 时回退为 0（跳过该分量）
        if not torch.isfinite(capsize_loss):
            print("⚠ Warning: capsize_loss is NaN/Inf, fallback to 0 for this batch")
            capsize_loss = dummy_loss.clone()

    
    # main_loss 数值稳定性保护（尤其在 stage2 main=0 时避免无关分支污染）
    if isinstance(main_loss, torch.Tensor) and (not torch.isfinite(main_loss)):
        print("⚠ Warning: main_loss is NaN/Inf, fallback to 0 for this batch")
        main_loss = dummy_loss.clone()

    total_loss = (
        loss_weights['main'] * main_loss
        + loss_weights.get('boundary', 0.0) * boundary_loss
        + loss_weights.get('capsize', 0.0) * capsize_loss
    )
    
    # 检查损失异常：不中断训练，回退为零损失并跳过本 batch 更新
    if not torch.isfinite(total_loss):
        total_loss_val = total_loss.item() if isinstance(total_loss, torch.Tensor) else float('nan')
        main_loss_val = main_loss.item() if isinstance(main_loss, torch.Tensor) else float('nan')
        print(f"⚠ Warning: total_loss is NaN/Inf ({total_loss_val}), main_loss={main_loss_val}. Skip this batch.")
        total_loss = dummy_loss.clone()
    
    # 返回各项损失用于记录
    loss_dict = {
        'main': main_loss.item(),
        'main_raw_mse': main_raw_mse.item() if isinstance(main_raw_mse, torch.Tensor) else float(main_raw_mse),
        'main_adp_weight_mean': main_adp_weight_mean.item() if isinstance(main_adp_weight_mean, torch.Tensor) else float(main_adp_weight_mean),
        'residual_rms': residual_rms.item(),
        'unscaled_residual_rms': unscaled_residual_rms.item(),
        'noise_rms': noise_rms.item(),
        'noise_to_residual_ratio': noise_to_residual_ratio.item(),
        'radial_x_norm_mean': radial_x_norm.mean().item(),
        'radial_x_norm_max': radial_x_norm.max().item(),
        'radial_rho_mean': radial_rho.mean().item(),
        'radial_rho_min': radial_rho.min().item(),
        'radial_active_cp_mean': radial_diagnostics[
            'active_control_point'
        ].float().mean().item(),
        'radial_active_y_fraction': radial_diagnostics[
            'active_coordinate'
        ].float().mean().item(),
        'radial_active_upper_fraction': radial_diagnostics[
            'active_is_upper'
        ].float().mean().item(),
        'hard_boundary_margin_mean': hard_boundary_margin.mean().item(),
        'hard_boundary_margin_min': hard_boundary_margin.min().item(),
        'gt_radial_x_norm_mean': gt_radial_x_norm.mean().item(),
        'gt_radial_x_norm_median': torch.quantile(
            gt_radial_x_norm, 0.50
        ).item(),
        'gt_radial_x_norm_p95': torch.quantile(
            gt_radial_x_norm, 0.95
        ).item(),
        'gt_radial_x_norm_p99': torch.quantile(
            gt_radial_x_norm, 0.99
        ).item(),
        'gt_radial_x_norm_max': gt_radial_x_norm.max().item(),
        'gt_radial_above_090_fraction': (
            gt_radial_x_norm > 0.90
        ).float().mean().item(),
        'gt_radial_above_098_fraction': (
            gt_radial_x_norm > 0.98
        ).float().mean().item(),
        'gt_infeasible_fraction': (gt_radial_x_norm > 1.0 + 1e-5).float().mean().item(),
        'zero_pred_mse': zero_pred_mse.item(),
        'model_x0_mse': model_x0_mse.item(),
        'explained_ratio': explained_ratio.item(),
        'boundary': boundary_loss.item(),
        'oob_sample_loss': boundary_diagnostics['oob_sample_loss'].item(),
        'oob_rate': boundary_diagnostics['oob_rate'].item(),
        'max_oob': boundary_diagnostics['max_oob'].item(),
        'active_boundary_fraction': boundary_diagnostics['active_boundary_fraction'].item(),
        'dense_oob_rate': boundary_diagnostics['dense_oob_rate'].item(),
        'dense_max_oob': boundary_diagnostics['dense_max_oob'].item(),
        'capsize': capsize_loss.item(),
    }
    
    # 调整返回的样本数：如果在第二阶段进行了混合，返回原始batch size
    n_samples = B // 2 if stage2_mix_loss else B
    
    return total_loss, 0, n_samples, loss_dict

def train_epoch(
    model,
    trainingData,
    optimizer,
    device,
    stage_epoch=0,
    loss_weights=None,
    current_stage=1,
    total_stage_epochs=50,
    ema_models=None,
    num_dense_points=100,
    prediction_type='x0',
):
    """
    单轮训练函数
    
    Args:
        ema_models: EMA模型列表，用于更新EMA参数（可选）
        num_dense_points: int - 安全损失所用B样条密集采样点数
    """
    model.train()
    total_loss = 0
    total_samples = 0
    
    # 累积各项损失
    loss_accumulator = {
        'main': 0, 
        'main_raw_mse': 0,
        'main_adp_weight_mean': 0,
        'residual_rms': 0,
        'unscaled_residual_rms': 0,
        'noise_rms': 0,
        'noise_to_residual_ratio': 0,
        'radial_x_norm_mean': 0,
        'radial_x_norm_max': 0,
        'radial_rho_mean': 0,
        'radial_rho_min': 0,
        'radial_active_cp_mean': 0,
        'radial_active_y_fraction': 0,
        'radial_active_upper_fraction': 0,
        'hard_boundary_margin_mean': 0,
        'hard_boundary_margin_min': 0,
        'gt_radial_x_norm_mean': 0,
        'gt_radial_x_norm_median': 0,
        'gt_radial_x_norm_p95': 0,
        'gt_radial_x_norm_p99': 0,
        'gt_radial_x_norm_max': 0,
        'gt_radial_above_090_fraction': 0,
        'gt_radial_above_098_fraction': 0,
        'gt_infeasible_fraction': 0,
        'zero_pred_mse': 0,
        'model_x0_mse': 0,
        'explained_ratio': 0,
        'boundary': 0,
        'oob_sample_loss': 0,
        'oob_rate': 0,
        'max_oob': 0,
        'active_boundary_fraction': 0,
        'dense_oob_rate': 0,
        'dense_max_oob': 0,
        'capsize': 0, 
    }
    head_gradient_totals = {
        'raw_v_weight_grad_norm': 0.0,
        'raw_v_bias_grad_norm': 0.0,
        'raw_s_weight_grad_norm': 0.0,
        'raw_s_bias_grad_norm': 0.0,
    }
    head_gradient_batches = 0
    raw_v_norm_first_batch = float('nan')
    
    pbar = tqdm(trainingData, mininterval=2, desc=f"Training Epoch {stage_epoch} (Stage {current_stage})")
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()

        captured_raw_v_norms = []
        capture_handle = None
        if batch_idx == 0:
            base_model = model.module if hasattr(model, 'module') else model

            def capture_raw_v_norm(_module, _inputs, output):
                raw_v_norm = torch.linalg.vector_norm(
                    output.detach().flatten(start_dim=1), dim=1
                ).mean()
                captured_raw_v_norms.append(raw_v_norm.item())

            capture_handle = base_model.main_pred.register_forward_hook(
                capture_raw_v_norm
            )
        try:
            loss, _, n_samples, loss_dict = diffusion_loss(
                model, batch, device, loss_weights,
                epoch=stage_epoch, total_epochs=total_stage_epochs,
                is_training=True,  # 训练模式：capsize loss使用回归损失
                current_stage=current_stage,  # 传递当前阶段
                num_dense_points=num_dense_points,
                prediction_type=prediction_type
            )
        finally:
            if capture_handle is not None:
                capture_handle.remove()
        if captured_raw_v_norms:
            raw_v_norm_first_batch = captured_raw_v_norms[-1]

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

        base_model = model.module if hasattr(model, 'module') else model
        raw_v_layer = base_model.main_pred[-1]
        raw_s_layer = base_model.radial_slack_pred[-1]

        def parameter_grad_norm(parameter):
            if parameter is None or parameter.grad is None:
                return 0.0
            return parameter.grad.detach().norm().item()

        head_gradient_totals['raw_v_weight_grad_norm'] += parameter_grad_norm(
            raw_v_layer.weight
        )
        head_gradient_totals['raw_v_bias_grad_norm'] += parameter_grad_norm(
            raw_v_layer.bias
        )
        head_gradient_totals['raw_s_weight_grad_norm'] += parameter_grad_norm(
            raw_s_layer.weight
        )
        head_gradient_totals['raw_s_bias_grad_norm'] += parameter_grad_norm(
            raw_s_layer.bias
        )
        head_gradient_batches += 1
        
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
            pbar.set_postfix({
                'Total': f'{loss.item():.5f}',
                'Main': f'{loss_dict["main"]:.5f}',
                'Boundary': f'{loss_dict["boundary"]:.5f}',
                'OOB': f'{100.0 * loss_dict["oob_rate"]:.1f}%',
                'GradNorm': grad_info
            })
        else:
            # 阶段2：显示主损失与物理约束损失
            pbar.set_postfix({
                'Total': f'{loss.item():.5f}',
                'Main': f'{loss_dict["main"]:.5f}',
                'Boundary': f'{loss_dict["boundary"]:.5f}',
                'OOB': f'{100.0 * loss_dict["oob_rate"]:.1f}%',
                'Capsize': f'{loss_dict["capsize"]:.4f}',
                'GradNorm': grad_info
            })
    
    avg_loss = total_loss / len(trainingData) if len(trainingData) > 0 else 0
    
    # 计算各项平均损失
    avg_loss_dict = {k: v / len(trainingData) for k, v in loss_accumulator.items()}
    grad_denominator = max(head_gradient_batches, 1)
    avg_loss_dict.update({
        key: value / grad_denominator
        for key, value in head_gradient_totals.items()
    })
    avg_loss_dict['raw_v_norm_first_batch'] = raw_v_norm_first_batch
    
    return avg_loss, 0, total_samples, avg_loss_dict


def eval_epoch(
    model,
    validationData,
    device,
    loss_weights=None,
    epoch=0,
    total_epochs=100,
    current_stage=1,
    use_late_timesteps=False,
    num_dense_points=100,
    validation_seed=2026,
    prediction_type='x0',
):
    """
    单轮评估函数
    
    Args:
        num_dense_points: int - 安全损失所用B样条密集采样点数
    """
    model.eval()
    total_loss = 0
    total_samples = 0
    
    # 累积各项损失
    loss_accumulator = {
        'main': 0, 
        'main_raw_mse': 0,
        'main_adp_weight_mean': 0,
        'residual_rms': 0,
        'unscaled_residual_rms': 0,
        'noise_rms': 0,
        'noise_to_residual_ratio': 0,
        'radial_x_norm_mean': 0,
        'radial_x_norm_max': 0,
        'radial_rho_mean': 0,
        'radial_rho_min': 0,
        'radial_active_cp_mean': 0,
        'radial_active_y_fraction': 0,
        'radial_active_upper_fraction': 0,
        'hard_boundary_margin_mean': 0,
        'hard_boundary_margin_min': 0,
        'gt_radial_x_norm_mean': 0,
        'gt_radial_x_norm_median': 0,
        'gt_radial_x_norm_p95': 0,
        'gt_radial_x_norm_p99': 0,
        'gt_radial_x_norm_max': 0,
        'gt_radial_above_090_fraction': 0,
        'gt_radial_above_098_fraction': 0,
        'gt_infeasible_fraction': 0,
        'zero_pred_mse': 0,
        'model_x0_mse': 0,
        'explained_ratio': 0,
        'boundary': 0,
        'oob_sample_loss': 0,
        'oob_rate': 0,
        'max_oob': 0,
        'active_boundary_fraction': 0,
        'dense_oob_rate': 0,
        'dense_max_oob': 0,
        'capsize': 0,
    }
    
    # Reset for every validation pass so each epoch and every EMA model see
    # exactly the same (t, r, epsilon) sequence batch by batch.
    validation_generator = torch.Generator(device=device)
    validation_generator.manual_seed(int(validation_seed))

    with torch.no_grad():
        for batch in tqdm(validationData, mininterval=2, desc="Validation"):
            
            loss, _, n_samples, loss_dict = diffusion_loss(
                model, batch, device, loss_weights,
                epoch=epoch, total_epochs=total_epochs,
                is_training=False,  # 验证模式：capsize loss返回实际cost值
                current_stage=current_stage,  # 传递当前阶段
                num_dense_points=num_dense_points,
                prediction_type=prediction_type,
                sampling_generator=validation_generator,
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


def build_stage1_config(
    epochs,
    main_norm_p=1.0,
    boundary_weight=0.1,
    boundary_safe_bound=0.98,
):
    """Canonical Stage 1 configuration shared with Fisher training."""
    return {
        'epochs': int(epochs),
        'lr_mul': 1e-1,
        'loss_weights': {
            'main': 1e-2,
            'main_norm_p': float(main_norm_p),
            'boundary': float(boundary_weight),
            'boundary_safe_bound': float(boundary_safe_bound),
        },
    }


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

def load_model_state_dict_compat(module, state_dict, strict=True):
    """
    加载模型参数。

    strict=True 时保持 PyTorch 默认行为；strict=False 时只加载名称和形状都匹配的参数，
    方便从旧版 raw-map cross-attention 结构迁移到 regime-latent 结构。
    """
    if strict:
        return module.load_state_dict(state_dict)

    current_state = module.state_dict()
    compatible_state = {}
    skipped = []
    for name, value in state_dict.items():
        if name in current_state and current_state[name].shape == value.shape:
            compatible_state[name] = value
        else:
            skipped.append(name)

    missing, unexpected = module.load_state_dict(compatible_state, strict=False)
    print(
        f"Partial checkpoint load: loaded {len(compatible_state)} tensors, "
        f"skipped {len(skipped)} incompatible tensors, missing {len(missing)}, unexpected {len(unexpected)}"
    )
    if skipped:
        print("  Skipped examples:", ", ".join(skipped[:8]))
    return missing, unexpected

def load_checkpoint(model, checkpoint_path, device):
    """加载检查点"""
    if not osp.exists(checkpoint_path):
        raise ValueError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model_to_load = model.module if isinstance(model, nn.DataParallel) else model
        if isinstance(model, nn.DataParallel):
            load_model_state_dict_compat(model_to_load, checkpoint['model_state_dict'], strict=False)
        else:
            load_model_state_dict_compat(model_to_load, checkpoint['model_state_dict'], strict=False)
    else:
        raise ValueError("Invalid checkpoint format")
    
    epoch = checkpoint.get('epoch', -1)
    print(f"Loaded checkpoint from epoch {epoch}")
    return checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', help="Batch size per GPU", required=True, type=int)
    parser.add_argument(
        '--dataFolder',
        help="Directory with training and validation data",
        default=str(MAP_CONFIG.dataset_root),
    )
    parser.add_argument('--fileDir', help="Directory to save training data")
    parser.add_argument('--resume', help="Path to checkpoint to resume training", default=None)
    parser.add_argument('--stage', help="Training stage to start from (1 or 2)", type=int, default=1, choices=[1, 2])
    parser.add_argument('--stage1_epochs', help="Number of epochs for stage 1", type=int, default=200)
    parser.add_argument('--stage2_epochs', help="Number of epochs for stage 2", type=int, default=300)
    parser.add_argument(
        '--main_norm_p',
        help="Adaptive exponent for the main pMF loss (first ablation: 1.0, 0.5, 0.0)",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        '--boundary_weight',
        help="Weight of the sparse active-only control-point boundary loss",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        '--boundary_safe_bound',
        help="Safe normalized map bound; violations of |coordinate| > bound are penalized",
        type=float,
        default=0.98,
    )
    parser.add_argument('--seed', help="Fixed random seed for controlled ablations", type=int, default=2026)
    parser.add_argument(
        '--checkpoint_every',
        help="Periodic checkpoint interval; 0 disables periodic backups",
        type=int,
        default=25,
    )
    # parser.add_argument('--prediction_type', help="Model prediction type", type=str, default='v', choices=['epsilon', 'x0', 'v'])
    args = parser.parse_args()

    if not 0.0 <= args.main_norm_p <= 1.0:
        raise ValueError("--main_norm_p must be in [0, 1]")
    if args.boundary_weight < 0.0:
        raise ValueError("--boundary_weight must be >= 0")
    if not 0.0 < args.boundary_safe_bound <= 1.0:
        raise ValueError("--boundary_safe_bound must be in (0, 1]")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint_every must be >= 0")

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

    # 单变量消融必须共享模型初始化、噪声序列和DataLoader随机顺序。
    torch_seed = int(args.seed)
    random.seed(torch_seed)
    np.random.seed(torch_seed)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)
    
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
        n_path_steps=25,  # 25条物理边的25×缩放零和残差token
        diffusion_steps=50,
        coordinate_scale=MAP_HALF_EXTENT,
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
    print(f"训练方法: pMF / Rectified Flow")
    print(f"条件结构: Implicit Guidance Encoder")
    print(f"预测类型: {prediction_type}")
    print(f"模型层数: {model_args['n_layers']}")
    print(f"注意力头: {model_args['n_heads']}")
    print(f"模型维度: {model_args['d_model']}")
    print(f"Guidance tokens: {model_args['n_path_steps']} (aligned to physical edges)")
    print("轨迹表示: 25×物理零和边残差（严格端点）")
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
    # 自动发现目录，避免手写环境列表漏掉地图。dataset20 的100张地图全部训练。
    train_env_list = discover_environments(
        osp.join(dataFolder, "train"),
        expected_count=MAP_CONFIG.expected_environments,
    )
    val_env_list = discover_environments(
        osp.join(dataFolder, "val"),
        expected_count=MAP_CONFIG.expected_environments,
    )
    print(
        f"✓ 数据集环境: train={len(train_env_list)}, "
        f"val={len(val_env_list)}, root={dataFolder}"
    )
    
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
    
    num_dense_points = 100
    print(f"✓ 几何/安全损失的B样条密集采样点数: {num_dense_points}")
    
    # =================== 两阶段训练配置 ===================
    stage1_config = build_stage1_config(
        epochs=args.stage1_epochs,
        main_norm_p=args.main_norm_p,
        boundary_weight=args.boundary_weight,
        boundary_safe_bound=args.boundary_safe_bound,
    )
    
    # 阶段2配置：通过梯度优化学习低cost轨迹
    # **核心改变**：使用完整DDIM采样链计算cost，而非单步预测
    # **关键**：保留main loss作为正则化，capsize loss作为优化目标
    stage2_config = {
        'epochs': args.stage2_epochs,
        'lr_mul': 1e-3,  # 小学习率微调（不是从头训练！）
        'loss_weights': {
            'main': 1e-2,        # 保留主损失作为正则化（不能为0！）
            'main_norm_p': args.main_norm_p,
            'boundary': args.boundary_weight,
            'boundary_safe_bound': args.boundary_safe_bound,
            'capsize': 1e-2,
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
        env_list=train_env_list,
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
        env_list=val_env_list,
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
        'total_epochs': n_epochs,
        'seed': torch_seed,
        'checkpoint_every': args.checkpoint_every,
        'map_config': MAP_CONFIG.to_dict(),
        'train_environment_count': len(train_env_list),
        'val_environment_count': len(val_env_list),
    }
    json.dump(
        config,
        open(osp.join(trainDataFolder, 'model_params.json'), 'w'),
        sort_keys=True,
        indent=4
    )
    
    writer = SummaryWriter(log_dir=trainDataFolder)
    writer.add_scalar('Config/main_norm_p', args.main_norm_p, 0)
    writer.add_scalar('Config/boundary_weight', args.boundary_weight, 0)
    writer.add_scalar('Config/boundary_safe_bound', args.boundary_safe_bound, 0)
    writer.add_scalar('Config/random_seed', torch_seed, 0)
    
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
                model_to_load = model.module if isinstance(model, nn.DataParallel) else model
                if isinstance(model, nn.DataParallel):
                    load_model_state_dict_compat(model_to_load, checkpoint['model_state_dict'], strict=False)
                else:
                    load_model_state_dict_compat(model_to_load, checkpoint['model_state_dict'], strict=False)
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
                env_list=train_env_list,
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
                env_list=val_env_list,
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
            ema_models=ema_models if use_ema else None, num_dense_points=num_dense_points
        )
        
        # 验证（使用主模型）
        val_loss, _, _, val_loss_dict = eval_epoch(
            model, validationData, device, loss_weights, stage_epoch, total_stage_epochs, 
            current_stage=current_stage,
            num_dense_points=num_dense_points,
            validation_seed=torch_seed,
        )
        
        # 验证EMA模型
        ema_val_losses = []
        ema_val_loss_dicts = []
        if use_ema and len(ema_models) > 0:
            for i, ema_m in enumerate(ema_models):
                ema_val_loss, _, _, ema_val_loss_dict = eval_epoch(
                    ema_m.module, validationData, device, loss_weights, stage_epoch, total_stage_epochs,
                    current_stage=current_stage,
                    num_dense_points=num_dense_points,
                    validation_seed=torch_seed,
                )
                ema_val_losses.append(ema_val_loss)
                ema_val_loss_dicts.append(ema_val_loss_dict)
        
        # 记录
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"\n[Stage {current_stage}] Epoch {stage_epoch}:")
        print(f"  Train Loss: {train_loss:.9f}")
        print(f"    - Main: {train_loss_dict['main']:.6f}")
        print(
            "    - Scaled residual/noise RMS and ratio: "
            f"{train_loss_dict['residual_rms']:.6f} / "
            f"{train_loss_dict['noise_rms']:.6f}"
            f" / {train_loss_dict['noise_to_residual_ratio']:.3f}"
        )
        print(f"    - Unscaled residual RMS: {train_loss_dict['unscaled_residual_rms']:.6f}")
        print(
            "    - Radial x-norm mean/max, rho mean/min: "
            f"{train_loss_dict['radial_x_norm_mean']:.4f} / "
            f"{train_loss_dict['radial_x_norm_max']:.4f} / "
            f"{train_loss_dict['radial_rho_mean']:.4f} / "
            f"{train_loss_dict['radial_rho_min']:.4f}"
        )
        print(
            "    - Hard margin mean/min, GT infeasible: "
            f"{train_loss_dict['hard_boundary_margin_mean']:.6f} / "
            f"{train_loss_dict['hard_boundary_margin_min']:.6f} / "
            f"{train_loss_dict['gt_infeasible_fraction']:.3%}"
        )
        print(
            "    - GT alpha mean/median/p95/p99, >.90/>.98: "
            f"{train_loss_dict['gt_radial_x_norm_mean']:.4f} / "
            f"{train_loss_dict['gt_radial_x_norm_median']:.4f} / "
            f"{train_loss_dict['gt_radial_x_norm_p95']:.4f} / "
            f"{train_loss_dict['gt_radial_x_norm_p99']:.4f} / "
            f"{train_loss_dict['gt_radial_above_090_fraction']:.3%} / "
            f"{train_loss_dict['gt_radial_above_098_fraction']:.3%}"
        )
        print(
            "    - Active constraint CP / y / upper: "
            f"{train_loss_dict['radial_active_cp_mean']:.2f} / "
            f"{train_loss_dict['radial_active_y_fraction']:.3%} / "
            f"{train_loss_dict['radial_active_upper_fraction']:.3%}"
        )
        print(
            "    - First raw-v norm; head grad Wv/bv/Ws/bs: "
            f"{train_loss_dict['raw_v_norm_first_batch']:.6f}; "
            f"{train_loss_dict['raw_v_weight_grad_norm']:.6f} / "
            f"{train_loss_dict['raw_v_bias_grad_norm']:.3e} / "
            f"{train_loss_dict['raw_s_weight_grad_norm']:.6f} / "
            f"{train_loss_dict['raw_s_bias_grad_norm']:.6f}"
        )
        print(
            "    - Zero MSE, Model MSE, Explained: "
            f"{train_loss_dict['zero_pred_mse']:.6f} / "
            f"{train_loss_dict['model_x0_mse']:.6f} / "
            f"{train_loss_dict['explained_ratio']:.3f}"
        )
        print(
            "    - Boundary / OOB rate / max OOB: "
            f"{train_loss_dict['boundary']:.6f} / "
            f"{train_loss_dict['oob_rate']:.3%} / "
            f"{train_loss_dict['max_oob']:.6f}"
        )
        print(
            "    - Dense-curve OOB rate / max OOB: "
            f"{train_loss_dict['dense_oob_rate']:.3%} / "
            f"{train_loss_dict['dense_max_oob']:.6f}"
        )
        print(f"    - Capsize: {train_loss_dict['capsize']:.9f}")
        print(f"  Val Loss:   {val_loss:.9f}")
        print(f"    - Main: {val_loss_dict['main']:.6f}")
        print(
            "    - Scaled residual/noise RMS and ratio: "
            f"{val_loss_dict['residual_rms']:.6f} / "
            f"{val_loss_dict['noise_rms']:.6f}"
            f" / {val_loss_dict['noise_to_residual_ratio']:.3f}"
        )
        print(f"    - Unscaled residual RMS: {val_loss_dict['unscaled_residual_rms']:.6f}")
        print(
            "    - Radial x-norm mean/max, rho mean/min: "
            f"{val_loss_dict['radial_x_norm_mean']:.4f} / "
            f"{val_loss_dict['radial_x_norm_max']:.4f} / "
            f"{val_loss_dict['radial_rho_mean']:.4f} / "
            f"{val_loss_dict['radial_rho_min']:.4f}"
        )
        print(
            "    - Hard margin mean/min, GT infeasible: "
            f"{val_loss_dict['hard_boundary_margin_mean']:.6f} / "
            f"{val_loss_dict['hard_boundary_margin_min']:.6f} / "
            f"{val_loss_dict['gt_infeasible_fraction']:.3%}"
        )
        print(
            "    - GT alpha mean/median/p95/p99, >.90/>.98: "
            f"{val_loss_dict['gt_radial_x_norm_mean']:.4f} / "
            f"{val_loss_dict['gt_radial_x_norm_median']:.4f} / "
            f"{val_loss_dict['gt_radial_x_norm_p95']:.4f} / "
            f"{val_loss_dict['gt_radial_x_norm_p99']:.4f} / "
            f"{val_loss_dict['gt_radial_above_090_fraction']:.3%} / "
            f"{val_loss_dict['gt_radial_above_098_fraction']:.3%}"
        )
        print(
            "    - Active constraint CP / y / upper: "
            f"{val_loss_dict['radial_active_cp_mean']:.2f} / "
            f"{val_loss_dict['radial_active_y_fraction']:.3%} / "
            f"{val_loss_dict['radial_active_upper_fraction']:.3%}"
        )
        print(
            "    - Zero MSE, Model MSE, Explained: "
            f"{val_loss_dict['zero_pred_mse']:.6f} / "
            f"{val_loss_dict['model_x0_mse']:.6f} / "
            f"{val_loss_dict['explained_ratio']:.3f}"
        )
        print(
            "    - Boundary / OOB rate / max OOB: "
            f"{val_loss_dict['boundary']:.6f} / "
            f"{val_loss_dict['oob_rate']:.3%} / "
            f"{val_loss_dict['max_oob']:.6f}"
        )
        print(
            "    - Dense-curve OOB rate / max OOB: "
            f"{val_loss_dict['dense_oob_rate']:.3%} / "
            f"{val_loss_dict['dense_max_oob']:.6f}"
        )
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
        writer.add_scalar('Loss/train_main_raw_mse', train_loss_dict.get('main_raw_mse', 0.0), stage_epoch)
        writer.add_scalar('Loss/train_main_adp_weight_mean', train_loss_dict.get('main_adp_weight_mean', 0.0), stage_epoch)
        writer.add_scalar('Loss/train_boundary', train_loss_dict['boundary'], stage_epoch)
        writer.add_scalar('Loss/train_capsize', train_loss_dict['capsize'], stage_epoch)
        for metric_name in (
            'raw_v_norm_first_batch',
            'raw_v_weight_grad_norm',
            'raw_v_bias_grad_norm',
            'raw_s_weight_grad_norm',
            'raw_s_bias_grad_norm',
        ):
            writer.add_scalar(
                f'Diagnostics/train_{metric_name}',
                train_loss_dict[metric_name],
                stage_epoch,
            )
        
        writer.add_scalar('Loss/val_main', val_loss_dict['main'], stage_epoch)
        writer.add_scalar('Loss/val_main_raw_mse', val_loss_dict.get('main_raw_mse', 0.0), stage_epoch)
        writer.add_scalar('Loss/val_main_adp_weight_mean', val_loss_dict.get('main_adp_weight_mean', 0.0), stage_epoch)
        writer.add_scalar('Loss/val_boundary', val_loss_dict['boundary'], stage_epoch)
        writer.add_scalar('Loss/val_capsize', val_loss_dict['capsize'], stage_epoch)

        for split_name, metrics in (
            ('train', train_loss_dict),
            ('val', val_loss_dict),
        ):
            for metric_name in (
                'residual_rms',
                'unscaled_residual_rms',
                'noise_rms',
                'noise_to_residual_ratio',
                'radial_x_norm_mean',
                'radial_x_norm_max',
                'radial_rho_mean',
                'radial_rho_min',
                'radial_active_cp_mean',
                'radial_active_y_fraction',
                'radial_active_upper_fraction',
                'hard_boundary_margin_mean',
                'hard_boundary_margin_min',
                'gt_radial_x_norm_mean',
                'gt_radial_x_norm_median',
                'gt_radial_x_norm_p95',
                'gt_radial_x_norm_p99',
                'gt_radial_x_norm_max',
                'gt_radial_above_090_fraction',
                'gt_radial_above_098_fraction',
                'gt_infeasible_fraction',
                'zero_pred_mse',
                'model_x0_mse',
                'explained_ratio',
                'oob_sample_loss',
                'oob_rate',
                'max_oob',
                'active_boundary_fraction',
                'dense_oob_rate',
                'dense_max_oob',
            ):
                writer.add_scalar(
                    f'Diagnostics/{split_name}_{metric_name}',
                    metrics[metric_name],
                    stage_epoch,
                )
        
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
                'main_norm_p': args.main_norm_p,
                'boundary_weight': args.boundary_weight,
                'boundary_safe_bound': args.boundary_safe_bound,
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
                        'main_norm_p': args.main_norm_p,
                        'boundary_weight': args.boundary_weight,
                        'boundary_safe_bound': args.boundary_safe_bound,
                    }
                    torch.save(ema_checkpoint, ema_model_path)
                    if current_stage == 2:
                        print(
                            f"  ✓ Saved best EMA model (decay={decay}) to {ema_model_filename} "
                            # f"(val_safe_rate={ema_val_loss_dicts[i].get('reward_rate', 0.0):.4f}, val_main={ema_val_loss_dicts[i]['main']:.6f}, metric={ema_val_metric:.6f})"
                        )
                    else:
                        print(f"  ✓ Saved best EMA model (decay={decay}) to {ema_model_filename} (val_loss={ema_val_metric:.6f})")
        
        # 可选周期备份；消融实验可设为0，仅保留best/final以控制磁盘占用。
        if args.checkpoint_every > 0 and (stage_epoch + 1) % args.checkpoint_every == 0:
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
                'main_norm_p': args.main_norm_p,
                'boundary_weight': args.boundary_weight,
                'boundary_safe_bound': args.boundary_safe_bound,
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
                        'main_norm_p': args.main_norm_p,
                        'boundary_weight': args.boundary_weight,
                        'boundary_safe_bound': args.boundary_safe_bound,
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
        'main_norm_p': args.main_norm_p,
        'boundary_weight': args.boundary_weight,
        'boundary_safe_bound': args.boundary_safe_bound,
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
                'torch_seed': torch_seed,
                'main_norm_p': args.main_norm_p,
                'boundary_weight': args.boundary_weight,
                'boundary_safe_bound': args.boundary_safe_bound,
            }, final_ema_path)
            print(f"Final EMA model (decay={decay}) saved to: {final_ema_filename}")
    
    # Print summary of best models
    if stage1_best_loss < float('inf'):
        print(f"Stage 1 best validation loss: {stage1_best_loss:.4f}")
        print(f"  Saved to: {stage1_best_path}")
    if current_stage == 2:
        print(f"Stage 2 best validation loss: {best_val_loss:.4f}")
        print(f"  Saved to: {stage2_best_path}")

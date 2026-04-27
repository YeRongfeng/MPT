"""
train_cdit.py - 训练不平坦地面路径预测模型(基于扩散模型)
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
from dit.Models import CostConditionedPathDiffusionTransformer

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


def diffusion_loss(model, batch, device, loss_weights=None, epoch=0, total_epochs=100, is_training=True, current_stage=1, use_dense_trajectory=False, num_dense_points=100, prediction_type='x0', cfg_drop_prob=0.0):
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
        cfg_drop_prob: float - 训练时将cost条件替换为null token的概率（CFG训练）
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
    cost = batch['cost'].to(device)  # (B,)
    
    cost_normalized = cost  # (B,)
    
    # cost_mean = 1.4
    # cost_std = 0.2
    # cost_normalized = (cost - cost_mean) / cost_std  # 标准化成本，确保数值稳定性

    # =================== CFG训练：cost null token dropout ===================
    # 以一定概率将样本的cost条件替换为 learnable null token（由模型内部处理）
    # 推理时可配合 guidance 权重 w 做条件引导
    cost_drop_mask = None
    if is_training and cfg_drop_prob > 0.0:
        cfg_drop_prob = float(max(0.0, min(1.0, cfg_drop_prob)))
        cost_drop_mask = (torch.rand_like(cost) < cfg_drop_prob)
    
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
    def u_fn(z_arg, t_arg, r_arg, cost_arg):
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
            # 【关键改变】添加 cost_arg 参数支持成本条件化
            with sdp_ctx:
                model_output = functional_call(
                    model, 
                    {**params, **buffers}, 
                    (map_input, z_arg, t_arg, r_arg, start_normalized, goal_normalized, cost_arg, cost_drop_mask, False)
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
        
        # Primals: 当前点（包含成本）
        primals = (z_t, t, r, cost_normalized)
        # Tangents: 方向。当 t 变化 1 时，z 变化 target_v，r 不变，cost 不变
        tangents = (target_v, torch.ones_like(t), torch.zeros_like(r), torch.zeros_like(cost_normalized))
            # 注意：cost 在 primals 中，但 jvp 只会对 requires_grad=True 的项进行求导
            # cost 在这里不需要 requires_grad，因为我们不计算关于 cost 的导数
        
        # 3. 计算 JVP (全导数)
        # u_out: 算出的 u
        # du_dt_full: 算出的全导数 du/dt
        u_out, du_dt_full = jvp(u_fn, primals, tangents)
        
        # 对修正项进行幅度裁剪，增加稳定性
        du_dt_full = torch.clamp(du_dt_full, -5.0, 5.0)

    # 4. 构建 V_theta 并计算主损失
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
    
    # 只保留主损失
    main_loss = main_loss_all.mean()
    if isinstance(main_loss, torch.Tensor) and (not torch.isfinite(main_loss)):
        print("⚠ Warning: main_loss is NaN/Inf, fallback to 0 for this batch")
        main_loss = torch.zeros_like(main_loss)

    total_loss = loss_weights.get('main', 1.0) * main_loss if loss_weights is not None else main_loss

    loss_dict = {
        'main': main_loss.item(),
    }

    return total_loss, 0, B, loss_dict

def train_epoch(model, trainingData, optimizer, device, stage_epoch=0, loss_weights=None, current_stage=1, total_stage_epochs=50, ema_models=None, use_dense_trajectory=False, num_dense_points=100, cfg_drop_prob=0.0):
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
    
    # 只累积主损失
    loss_accumulator = {'main': 0.0}
    
    pbar = tqdm(trainingData, mininterval=2, desc=f"Training Epoch {stage_epoch}")
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()
        
        loss, _, n_samples, loss_dict = diffusion_loss(
            model, batch, device, loss_weights,
            epoch=stage_epoch, total_epochs=total_stage_epochs,
            is_training=True,
            current_stage=current_stage,
            use_dense_trajectory=use_dense_trajectory,
            num_dense_points=num_dense_points,
            prediction_type=prediction_type,
            cfg_drop_prob=cfg_drop_prob
        )

        # 先检查 loss 再统计，避免把 NaN 累积进 epoch 指标
        if not torch.isfinite(loss):
            print("Warning: NaN/Inf loss, skipping batch")
            continue

        total_loss += loss.item()
        total_samples += n_samples
        
        loss_accumulator['main'] += loss_dict['main']
        
        loss.backward()
        
        clip_value = 1.0
        
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
        
        pbar.set_postfix({
            'Total': f'{loss.item():.5f}',
            'Main': f'{loss_dict["main"]:.5f}',
            'GradNorm': grad_info
        })
    
    avg_loss = total_loss / len(trainingData) if len(trainingData) > 0 else 0
    
    # 计算各项平均损失
    avg_loss_dict = {k: v / len(trainingData) for k, v in loss_accumulator.items()}
    
    return avg_loss, 0, total_samples, avg_loss_dict


def eval_epoch(model, validationData, device, loss_weights=None, epoch=0, total_epochs=100, current_stage=1, use_late_timesteps=False, use_dense_trajectory=False, num_dense_points=100, cfg_drop_prob=0.0):
    """
    单轮评估函数
    
    Args:
        use_dense_trajectory: bool - 是否使用密集轨迹计算主损失
        num_dense_points: int - 密集轨迹的点数
    """
    model.eval()
    total_loss = 0
    total_samples = 0
    
    # 只累积主损失
    loss_accumulator = {'main': 0.0}
    
    with torch.no_grad():
        for batch in tqdm(validationData, mininterval=2, desc="Validation"):
            
            loss, _, n_samples, loss_dict = diffusion_loss(
                model, batch, device, loss_weights,
                epoch=epoch, total_epochs=total_epochs,
                is_training=False,
                current_stage=current_stage,
                use_dense_trajectory=use_dense_trajectory,
                num_dense_points=num_dense_points,
                prediction_type=prediction_type,
                cfg_drop_prob=cfg_drop_prob
            )
                
            total_loss += loss.item()
            total_samples += n_samples
            
            loss_accumulator['main'] += loss_dict['main']
    
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
    parser.add_argument('--epochs', help="Number of training epochs", type=int, default=200)
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

    # 初始化扩散模型（使用成本条件化版本）
    model = CostConditionedPathDiffusionTransformer(**model_args)

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
    
    # =================== 单阶段训练配置 ===================
    # 统一配置：注重路径预测 + 成本条件化 
    training_config = {
        'epochs': args.epochs,
        'lr_mul': 3e-2,  # 学习率倍增器
        'cfg_drop_prob': 0.15,  # CFG训练：cost条件替换为null token的概率（建议0.1~0.2）
        'loss_weights': {
            'main': 1e-2,          # 主预测损失
            'smoothness': 0e-5,   # 平滑性损失
            'curvature': 0e-4,    # 曲率约束（限制最大曲率）
            'angle_smoothness': 0e-2,  # 角度平滑性损失
            'angle_consistency': 0e-6,  # 角度一致性损失(防止倒车)
            'uniformity': 0e-4,   # 均匀性损失（点间距离方差）
            'tangent': 0e3,   # 切线约束损失
            'capsize': 0.0,        # 倾覆监督损失（可选）
            'consistency': 0.0,   # 时间一致性损失（当前不启用）
        }
    }
    
    # 为了兼容现有代码，设置stage1和stage2为相同配置（单阶段训练）
    
    # 使用统一的损失权重
    loss_weights = training_config['loss_weights']
    
    # 根据损失权重决定是否计算stability map
    compute_stability = loss_weights.get('capsize', 0.0) > 0
    
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
    
    # =================== 单阶段训练配置 ===================
    trainDataFolder = args.fileDir
    current_stage = 1
    n_epochs = args.epochs
    
    print("\n=== 单阶段训练配置 ===")
    print(f"\n训练 ({training_config['epochs']} epochs, lr_mul={training_config['lr_mul']}):")
    print(f"  cfg_drop_prob: {training_config['cfg_drop_prob']}")
    for key, weight in training_config['loss_weights'].items():
        print(f"  {key}: {weight}")
    print()
    
    # 保存模型配置
    config = {
        'model_args': model_args,
        'training_config': training_config,
        'total_epochs': args.epochs
    }
    json.dump(
        config,
        open(osp.join(trainDataFolder, 'model_params.json'), 'w'),
        sort_keys=True,
        indent=4
    )
    
    writer = SummaryWriter(log_dir=trainDataFolder)
    
    # 单阶段训练：全参数训练
    print("✓ 单阶段训练：全参数训练模式")
    
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
        lr_mul=training_config['lr_mul'],
        d_model=512,
        n_warmup_steps=1000
    )
    
    print(f"✓ 可训练参数数量 = {sum(p.numel() for p in trainable_params)}")

    
    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        checkpoint = load_checkpoint(model, args.resume, device)
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        start_epoch = checkpoint.get('epoch', -1) + 1
        
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
            
            # 从分离的EMA checkpoint加载
            else:
                for i, (ema_m, decay) in enumerate(zip(ema_models, ema_decays)):
                    ema_checkpoint_path = osp.join(trainDataFolder, f'best_ema_{decay}.pth')
                    if osp.exists(ema_checkpoint_path):
                        try:
                            ema_checkpoint = torch.load(ema_checkpoint_path, map_location=device)
                            ema_m.module.load_state_dict(ema_checkpoint['model_state_dict'])
                            print(f"✓ Loaded EMA model (decay={decay}) from {osp.basename(ema_checkpoint_path)}")
                        except Exception as e:
                            print(f"Warning: Failed to load EMA model (decay={decay}): {e}")
        
        # 加载优化器状态
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'n_steps' in checkpoint:
                    optimizer.n_steps = checkpoint['n_steps']
                print(f"✓ Loaded optimizer state")
            except Exception as e:
                print(f"Warning: Failed to load optimizer state: {e}")
    
    # =================== 训练循环 ===================
    train_losses = []
    val_losses = []
    # 单阶段保存路径
    best_model_path = osp.join(trainDataFolder, 'best_model.pth')
    
    # 为每个EMA模型跟踪最佳验证损失
    ema_best_val_losses = [float('inf')] * len(ema_models) if use_ema else []
    
    print(f"\n{'='*60}")
    print(f"Starting single-stage training from epoch {start_epoch}")
    print(f"Total epochs: {args.epochs}")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, args.epochs):
        # 单阶段训练
        stage_epoch = epoch
        
        # 训练
        train_loss, _, _, train_loss_dict = train_epoch(
            model, trainingData, optimizer, device, stage_epoch, loss_weights, current_stage, args.epochs,
            ema_models=ema_models if use_ema else None, use_dense_trajectory=use_dense_trajectory, num_dense_points=num_dense_points,
            cfg_drop_prob=training_config.get('cfg_drop_prob', 0.0)
        )
        
        # 验证（使用主模型）
        val_loss, _, _, val_loss_dict = eval_epoch(
            model, validationData, device, loss_weights, stage_epoch, args.epochs, 
            current_stage=current_stage, use_dense_trajectory=use_dense_trajectory, num_dense_points=num_dense_points,
            cfg_drop_prob=0.0
        )
        
        # 验证EMA模型
        ema_val_losses = []
        ema_val_loss_dicts = []
        if use_ema and len(ema_models) > 0:
            for i, ema_m in enumerate(ema_models):
                ema_val_loss, _, _, ema_val_loss_dict = eval_epoch(
                    ema_m.module, validationData, device, loss_weights, stage_epoch, args.epochs,
                    current_stage=current_stage, use_dense_trajectory=use_dense_trajectory, num_dense_points=num_dense_points,
                    cfg_drop_prob=0.0
                )
                ema_val_losses.append(ema_val_loss)
                ema_val_loss_dicts.append(ema_val_loss_dict)
        
        # 记录
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"\nEpoch {stage_epoch}:")
        print(f"  Train Loss: {train_loss:.9f}")
        print(f"    - Main: {train_loss_dict['main']:.6f}")
        print(f"  Val Loss:   {val_loss:.9f}")
        print(f"    - Main: {val_loss_dict['main']:.6f}")
        
        # 打印EMA验证损失
        if use_ema and len(ema_val_losses) > 0:
            for i, (ema_val_loss, decay) in enumerate(zip(ema_val_losses, ema_decays)):
                print(f"  EMA Val Loss (decay={decay}): {ema_val_loss:.6f}")
        
        # TensorBoard - 总损失（使用阶段内epoch）
        writer.add_scalar('Loss/train', train_loss, stage_epoch)
        writer.add_scalar('Loss/val', val_loss, stage_epoch)
        
        # TensorBoard - 主损失
        writer.add_scalar('Loss/train_main', train_loss_dict['main'], stage_epoch)
        writer.add_scalar('Loss/val_main', val_loss_dict['main'], stage_epoch)
        
        # TensorBoard - EMA验证损失
        if use_ema and len(ema_val_losses) > 0:
            for i, (ema_val_loss, decay) in enumerate(zip(ema_val_losses, ema_decays)):
                writer.add_scalar(f'Loss/ema_val_{decay}', ema_val_loss, stage_epoch)
        
        writer.add_scalar('LR', optimizer._optimizer.param_groups[0]['lr'], stage_epoch)
        
        # 保存最佳标准模型
        current_val_metric = val_loss
        if current_val_metric < best_val_loss:
            best_val_loss = current_val_metric
            
            if isinstance(model, nn.DataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            # 保存最佳模型
            checkpoint = {
                'epoch': stage_epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer._optimizer.state_dict(),
                'n_steps': optimizer.n_steps,
                'train_loss': train_loss,
                'val_loss': current_val_metric,
                'torch_seed': torch_seed
            }
            
            torch.save(checkpoint, best_model_path)
            print(f"  ✓ Saved best model to {osp.basename(best_model_path)} (val_loss={current_val_metric:.6f})")
        
        # 分别保存每个EMA模型
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
                        'model_state_dict': ema_state_dict,
                        'ema_decay': decay,
                        'train_loss': train_loss,
                        'val_loss': ema_val_metric,
                        'torch_seed': torch_seed
                    }
                    torch.save(ema_checkpoint, ema_model_path)
                    print(f"  ✓ Saved best EMA model (decay={decay}) to {ema_model_filename} (val_loss={ema_val_metric:.6f})")
        
        # 定期保存检查点（每5个epoch）- 作为备份
        if (stage_epoch + 1) % 5 == 0:
            if isinstance(model, nn.DataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            # 保存标准模型checkpoint
            checkpoint_path = osp.join(trainDataFolder, f'checkpoint_epoch_{stage_epoch}.pth')
            torch.save({
                'epoch': stage_epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer._optimizer.state_dict(),
                'n_steps': optimizer.n_steps,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'torch_seed': torch_seed
            }, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")
            
            # 分别保存每个EMA checkpoint
            if use_ema and len(ema_models) > 0:
                for i, (ema_m, decay) in enumerate(zip(ema_models, ema_decays)):
                    ema_state_dict = ema_m.module.state_dict()
                    ema_checkpoint_filename = f'checkpoint_epoch_{stage_epoch}_ema_{decay}.pth'
                    ema_checkpoint_path = osp.join(trainDataFolder, ema_checkpoint_filename)
                    
                    torch.save({
                        'epoch': stage_epoch,
                        'model_state_dict': ema_state_dict,
                        'ema_decay': decay,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
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
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best model saved to: {best_model_path}")
    print(f"{'='*60}\n")
    
    # 保存最终模型
    if isinstance(model, nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    
    final_model_path = osp.join(trainDataFolder, 'final_model.pth')
    final_stage_epoch = n_epochs - 1
    
    # 保存标准最终模型
    torch.save({
        'epoch': final_stage_epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer._optimizer.state_dict(),
        'n_steps': optimizer.n_steps,
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
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
                'model_state_dict': ema_state_dict,
                'ema_decay': decay,
                'train_loss': train_losses[-1],
                'val_loss': val_losses[-1],
                'torch_seed': torch_seed
            }, final_ema_path)
            print(f"Final EMA model (decay={decay}) saved to: {final_ema_filename}")
    
    # Print summary of best model
    if best_val_loss < float('inf'):
        print(f"\nBest validation loss: {best_val_loss:.4f}")
        print(f"  Saved to: {best_model_path}")
"""
train_bspline_dit.py - 基于B样条参数化的扩散模型训练

【核心改进】
1. 参数化建模：预测15个B样条控制点（角度），而非直接预测60个轨迹点
2. 运动学约束：通过B样条插值+运动学积分自动满足SE(2)约束
3. 平滑性保证：B样条天然平滑，解决了直接预测时的角度一致性问题
4. 降维学习：15维控制点 << 60维直接输出，更容易学习

【与train_dit.py的区别】
- train_dit.py: 直接预测 (x,y,θ)，在噪声状态下难以保持运动学约束
- train_bspline_dit.py: 预测控制点 → B样条 → 运动学积分 → 轨迹，自动保持约束
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
from dit.BSplineModels import BSplineDiffusionTransformer, BSplineInterpolator

def fit_bspline_control_points_from_trajectory(trajectory, n_control_points=15):
    """
    从轨迹拟合B样条控制点（简化版：均匀采样角度）
    
    Args:
        trajectory: (B, N, 3) 轨迹点 [x, y, theta]
        n_control_points: 控制点数量
        
    Returns:
        control_points: (B, n_control_points) 角度控制点
    """
    B, N, _ = trajectory.shape
    device = trajectory.device
    
    # 均匀采样轨迹上的角度作为控制点初始值
    indices = torch.linspace(0, N-1, n_control_points, device=device).long()
    
    control_angles = []
    for b in range(B):
        sampled_angles = trajectory[b, indices, 2]  # (n_control_points,)
        control_angles.append(sampled_angles)
    
    control_points = torch.stack(control_angles, dim=0)  # (B, n_control_points)
    
    return control_points


def bspline_diffusion_loss(model, batch, device, loss_weights, current_stage=1):
    """
    B样条扩散模型损失函数
    
    核心思想：
    1. 在控制点空间做扩散（降维：15维 vs 60维）
    2. 从控制点重建轨迹（B样条+运动学积分）
    3. 在轨迹空间计算损失
    
    优势：
    - 自动满足运动学约束（通过积分）
    - B样条保证平滑性
    - 控制点噪声被B样条平滑，不破坏约束
    
    Args:
        model: BSplineDiffusionTransformer
        batch: 数据batch
        device: 设备
        loss_weights: 损失权重
        current_stage: 训练阶段（1或2）
    """
    map_input = batch['map'].float().to(device)
    trajectory = batch['trajectory'].to(device)  # (B, 22, 3)
    start_pose = batch['start_pose'].to(device)  # (B, 3)
    goal_pose = batch['goal_pose'].to(device)  # (B, 3)
    
    B = map_input.shape[0]
    n_control = model.n_angle_control_points
    
    # ========== 步骤1：从GT轨迹拟合控制点 ==========
    gt_control_points = fit_bspline_control_points_from_trajectory(
        trajectory, n_control_points=n_control
    )  # (B, n_control)
    
    # 计算弧长（用于运动学积分）
    positions = trajectory[:, :, :2]
    distances = torch.norm(positions[:, 1:] - positions[:, :-1], dim=2)
    arc_lengths = distances.sum(dim=1)  # (B,)
    
    # ========== 步骤2：扩散过程 ==========
    t = torch.randint(0, model.diffusion_steps, (B,), device=device)
    noise = torch.randn_like(gt_control_points)
    noisy_control_points = model.q_sample(gt_control_points, t, noise)
    
    # ========== 步骤3：模型预测 ==========
    pred_output = model(map_input, noisy_control_points, t, start_pose, goal_pose)
    
    # 恢复x0
    if model.prediction_type == 'epsilon':
        sqrt_alpha_t = torch.sqrt(model.alphas_cumprod[t])[:, None]
        sqrt_one_minus_alpha_t = torch.sqrt(1 - model.alphas_cumprod[t])[:, None]
        pred_x0_control = (noisy_control_points - sqrt_one_minus_alpha_t * pred_output) / sqrt_alpha_t
    elif model.prediction_type == 'x0':
        pred_x0_control = pred_output
    else:
        raise NotImplementedError(f"prediction_type={model.prediction_type} not supported")
    
    # ========== 步骤4：从控制点重建轨迹 ==========
    pred_trajectories = []
    for i in range(B):
        pred_traj = BSplineInterpolator.integrate_kinematics_from_angle_control_points(
            pred_x0_control[i:i+1],
            start_pose[i:i+1],
            arc_lengths[i].item(),
            n_path_points=21  # 起点 + 20中间点
        )  # (1, 20, 3)
        pred_trajectories.append(pred_traj)
    pred_trajectory = torch.cat(pred_trajectories, dim=0)  # (B, 20, 3)
    
    # ========== 步骤5：计算损失 ==========
    gt_middle = trajectory[:, 1:-1, :]  # (B, 20, 3) GT中间点
    
    # 主损失：轨迹重建
    pos_loss = F.mse_loss(pred_trajectory[:, :, :2], gt_middle[:, :, :2])
    angle_loss = F.mse_loss(
        torch.sin(pred_trajectory[:, :, 2]), torch.sin(gt_middle[:, :, 2])
    ) + F.mse_loss(
        torch.cos(pred_trajectory[:, :, 2]), torch.cos(gt_middle[:, :, 2])
    )
    main_loss = pos_loss + 0.5 * angle_loss
    
    # 控制点平滑性
    angle_diff = pred_x0_control[:, 1:] - pred_x0_control[:, :-1]
    control_smooth_loss = torch.mean(angle_diff**2)
    
    # 起终点约束
    endpoint_loss = F.mse_loss(pred_trajectory[:, 0, :2], start_pose[:, :2]) + \
                   F.mse_loss(pred_trajectory[:, -1, :2], goal_pose[:, :2])
    
    # ========== 步骤6：Stage 2 Cost优化 ==========
    cost_loss = torch.tensor(0.0, device=device)
    if current_stage == 2 and loss_weights.get('cost', 0) > 0:
        if 'esdf' in batch:
            # 构建完整22点轨迹（起点 + 20中间点 + 终点）
            full_trajectories = []
            for i in range(B):
                full_traj = torch.cat([
                    start_pose[i:i+1, :],     # 起点
                    pred_trajectory[i],        # 20个中间点
                    goal_pose[i:i+1, :]       # 终点
                ], dim=0)  # (22, 3)
                full_trajectories.append(full_traj)
            full_trajectories = torch.stack(full_trajectories, dim=0)  # (B, 22, 3)
            
            # B样条插值到100个密集点
            dense_trajectories = []
            for i in range(B):
                # B样条插值
                interpolator = BSplineInterpolator()
                u = torch.linspace(0, 1, 100, device=device)
                
                # x 坐标 B样条
                x_control = full_trajectories[i, :, 0].unsqueeze(0)  # (1, 22)
                x_dense = interpolator.evaluate_bspline(x_control, u, degree=3)  # (1, 100)
                
                # y 坐标 B样条
                y_control = full_trajectories[i, :, 1].unsqueeze(0)
                y_dense = interpolator.evaluate_bspline(y_control, u, degree=3)
                
                # yaw 角度 B样条（周期性处理）
                yaw_control = full_trajectories[i, :, 2].unsqueeze(0)
                # 展开角度到连续域
                yaw_unwrapped = torch.zeros_like(yaw_control)
                yaw_unwrapped[0, 0] = yaw_control[0, 0]
                for j in range(1, 22):
                    diff = yaw_control[0, j] - yaw_control[0, j-1]
                    # 处理周期性跳变
                    if diff > torch.pi:
                        diff -= 2 * torch.pi
                    elif diff < -torch.pi:
                        diff += 2 * torch.pi
                    yaw_unwrapped[0, j] = yaw_unwrapped[0, j-1] + diff
                
                yaw_dense = interpolator.evaluate_bspline(yaw_unwrapped, u, degree=3)
                
                # 重新包装到 [-π, π]
                yaw_dense = torch.atan2(torch.sin(yaw_dense), torch.cos(yaw_dense))
                
                # 组合成完整轨迹
                dense_traj = torch.stack([
                    x_dense.squeeze(0),
                    y_dense.squeeze(0),
                    yaw_dense.squeeze(0)
                ], dim=1)  # (100, 3)
                
                dense_trajectories.append(dense_traj)
            
            dense_trajectories = torch.stack(dense_trajectories, dim=0)  # (B, 100, 3)
            
            # 计算密集轨迹的cost
            from evaluator import cost_on_dense_trajectory
            costs = cost_on_dense_trajectory(
                dense_trajectories, batch['map'], batch['esdf'], device
            )  # (B,)
            
            # Cost损失：鼓励cost接近0
            cost_loss = torch.mean(costs)
    
    # 总损失
    total_loss = (
        loss_weights.get('main', 1.0) * main_loss +
        loss_weights.get('smoothness', 0.1) * control_smooth_loss +
        loss_weights.get('endpoint', 1.0) * endpoint_loss +
        loss_weights.get('cost', 0.0) * cost_loss
    )
    
    loss_dict = {
        'main': main_loss.item(),
        'smoothness': control_smooth_loss.item(),
        'endpoint': endpoint_loss.item(),
        'cost': cost_loss.item() if isinstance(cost_loss, torch.Tensor) else 0.0,
        'total': total_loss.item()
    }
    
    return total_loss, loss_dict

def train_epoch(model, trainingData, optimizer, device, stage_epoch=0, loss_weights=None, current_stage=1, total_stage_epochs=50):
    """
    单轮训练函数
    """
    model.train()
    total_loss = 0
    total_samples = 0
    
    # 累积各项损失（添加consistency）
    loss_accumulator = {'main': 0, 'smoothness': 0, 'endpoint': 0, 'cost': 0, 'total': 0}
    
    pbar = tqdm(trainingData, mininterval=2, desc=f"Training Epoch {stage_epoch} (Stage {current_stage})")
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()
        
        loss, loss_dict = bspline_diffusion_loss(
            model, batch, device, loss_weights,
            current_stage=current_stage
        )
        n_samples = batch['map'].shape[0]
        
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
            # 阶段1：显示主损失、平滑性、起终点
            pbar.set_postfix({
                'Total': f'{loss.item():.4f}',
                'Main': f'{loss_dict["main"]:.4f}',
                'Smooth': f'{loss_dict["smoothness"]:.5f}',
                'Endpoint': f'{loss_dict["endpoint"]:.5f}',
                'GradNorm': grad_info
            })
        else:
            # 阶段2：显示cost损失
            pbar.set_postfix({
                'Total': f'{loss.item():.4e}',
                'Cost': f'{loss_dict["cost"]:.6e}',
                'Main': f'{loss_dict["main"]:.4f}',
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
    loss_accumulator = {'main': 0, 'smoothness': 0, 'endpoint': 0, 'cost': 0, 'total': 0}
    
    with torch.no_grad():
        for batch in tqdm(validationData, mininterval=2, desc="Validation"):
            
            loss, loss_dict = bspline_diffusion_loss(
                model, batch, device, loss_weights,
                current_stage=current_stage
            )
            n_samples = batch['map'].shape[0]
                
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
    parser.add_argument('--prediction_type', help="Model prediction type", type=str, default='v', choices=['epsilon', 'x0', 'v'])
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
    
    # B样条配置
    n_control_points = 15  # B样条控制点数量（建议12-15）
    
    prediction_type = 'x0'  # B样条模型推荐使用'x0'预测
    
    model_args = dict(
        n_layers=6,
        n_heads=8,
        d_k=128,
        d_v=128,
        d_model=512,
        d_inner=2048,
        dropout=0.1,
        n_angle_control_points=n_control_points,  # B样条控制点数量
        n_path_points=21,  # 输出路径点数（包含起点）
        diffusion_steps=50,
        prediction_type=prediction_type,
        n_position=225,
        train_shape=(15, 15)
        # 
        # 【预测类型组合】
        #   --prediction_type=x0: 直接预测，容量要求低但精度受限
        #   --prediction_type=epsilon: DDPM标准，稳定
        #   --prediction_type=v: Stable Diffusion方案，最推荐
    )
    
    print("\n" + "="*70)
    print("B样条扩散模型配置")
    print("="*70)
    print(f"控制点数量: {n_control_points} (vs 60维直接预测)")
    print(f"预测类型: {prediction_type}")
    print(f"模型层数: {model_args['n_layers']}")
    print(f"注意力头: {model_args['n_heads']}")
    print(f"模型维度: {model_args['d_model']}")
    print("="*70 + "\n")

    # 初始化B样条扩散模型
    model = BSplineDiffusionTransformer(**model_args)

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
    env_list = ["env000008"]
    # env_list = ["env000008", "env000009"]
    
    # =================== 两阶段训练配置 ===================
    # 阶段1配置：B样条控制点空间轨迹重建
    stage1_config = {
        'epochs': args.stage1_epochs,
        'lr_mul': 1e-1,  # 学习率倍增器
        'loss_weights': {
            'main': 3e-4,           # 轨迹重建损失（位置+角度）
            'smoothness': 1e-1,     # 控制点平滑性损失（二阶差分）
            'endpoint': 0e-5,       # 起终点约束损失
        }
    }
    
    # 阶段2配置：通过梯度优化学习低cost轨迹
    # 在控制点空间优化，B样条保证平滑性
    stage2_config = {
        'epochs': args.stage2_epochs,
        'lr_mul': 1e-4,  # 小学习率微调
        'loss_weights': {
            'main': 0.1,           # 保留重建损失作为正则化
            'smoothness': 0.01,    # 保留平滑性约束
            'endpoint': 0.5,       # 保留起终点约束
            'cost': 1e-4,          # TODO: 需要在bspline_diffusion_loss中添加Stage 2的cost计算
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
        print(f"    - Endpoint: {train_loss_dict['endpoint']:.4f}")
        print(f"    - Cost: {train_loss_dict['cost']:.6e}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"    - Main: {val_loss_dict['main']:.4f}")
        print(f"    - Smoothness: {val_loss_dict['smoothness']:.4f}")
        print(f"    - Endpoint: {val_loss_dict['endpoint']:.4f}")
        print(f"    - Cost: {val_loss_dict['cost']:.6e}")
        
        # TensorBoard - 总损失（使用阶段内epoch）
        writer.add_scalar('Loss/train', train_loss, stage_epoch)
        writer.add_scalar('Loss/val', val_loss, stage_epoch)
        
        # TensorBoard - 各项损失
        writer.add_scalar('Loss/train_main', train_loss_dict['main'], stage_epoch)
        writer.add_scalar('Loss/train_smoothness', train_loss_dict['smoothness'], stage_epoch)
        writer.add_scalar('Loss/train_endpoint', train_loss_dict['endpoint'], stage_epoch)
        writer.add_scalar('Loss/train_cost', train_loss_dict['cost'], stage_epoch)
        
        writer.add_scalar('Loss/val_main', val_loss_dict['main'], stage_epoch)
        writer.add_scalar('Loss/val_smoothness', val_loss_dict['smoothness'], stage_epoch)
        writer.add_scalar('Loss/val_endpoint', val_loss_dict['endpoint'], stage_epoch)
        writer.add_scalar('Loss/val_cost', val_loss_dict['cost'], stage_epoch)
        
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
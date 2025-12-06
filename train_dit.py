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
from dataLoader_uneven import UnevenPathDataLoader, PaddedSequence
from dataLoader_uneven import hashTable, receptive_field

from torch.utils.tensorboard import SummaryWriter

from ESDF3d_atpoint import compute_esdf_batch
from grad_optimizer import TrajectoryOptimizerSE2
from dit.Models import PathDiffusionTransformer

def diffusion_loss(model, batch, device):
    """
    扩散模型损失函数
    
    Args:
        model: PathDiffusionTransformer
        batch: 包含 map, trajectory 的字典
    Returns:
        loss: MSE损失
        stats: 统计信息 (用于兼容原有接口)
    """
    map_input = batch['map'].float().to(device)  # (B, 6, H, W)
    trajectory = batch['trajectory'].to(device)  # (B, num_steps, 3)
    
    B = map_input.shape[0]
    
    # 提取路径：去掉终点，只保留中间20步
    # trajectory[:, 0] = 起点, trajectory[:, 1:21] = 20个中间点
    path_gt = trajectory[:, 1:-1, :]  # (B, 20, 3) 包含起点

    # 归一化路径坐标到 [-1, 1] 范围（适合扩散模型）
    # 假设地图范围是 [-20, 20]
    path_gt_normalized = path_gt.clone()
    path_gt_normalized[:, :, :2] = path_gt[:, :, :2] / 20.0  # x, y归一化
    path_gt_normalized[:, :, 2] = path_gt[:, :, 2] / np.pi   # yaw归一化
    
    # 随机采样时间步
    t = torch.randint(0, model.diffusion_steps, (B,), device=device)
    
    # 生成噪声
    noise = torch.randn_like(path_gt_normalized)
    
    # 加噪
    noisy_path = model.q_sample(path_gt_normalized, t, noise)
    
    # 预测噪声
    pred_noise = model(map_input, noisy_path, t)
    
    # MSE损失
    loss = F.mse_loss(pred_noise, noise)
    
    # 返回兼容的统计信息（用于保持原有训练循环）
    n_correct = 0  # 扩散模型不需要分类准确率
    n_samples = B
    batch_stats = (0, 0, 0, 0)  # (total_pos, total_neg, correct_pos, correct_neg)
    
    return loss, n_correct, n_samples, batch_stats

def train_epoch(model, trainingData, optimizer, device, epoch=0):
    """
    单轮训练函数 - 扩散模型版本
    """
    model.train()
    total_loss = 0
    total_samples = 0
    
    pbar = tqdm(trainingData, mininterval=2, desc=f"Training Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()
        
        # 计算扩散损失
        loss, _, n_samples, _ = diffusion_loss(model, batch, device)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss, skipping batch")
            continue
        
        loss.backward()
        
        # 梯度裁剪
        original_grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0, norm_type=2
        )
        was_clipped = original_grad_norm > 1.0
        
        optimizer.step_and_update_lr()
        
        total_loss += loss.item()
        total_samples += n_samples
        
        # 更新进度条
        grad_info = f'{original_grad_norm:.2f}→1.0' if was_clipped else f'{original_grad_norm:.2f}'
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'GradNorm': grad_info,
            'LR': f'{optimizer._optimizer.param_groups[0]["lr"]:.2e}'
        })
    
    avg_loss = total_loss / len(trainingData) if len(trainingData) > 0 else 0
    return avg_loss, 0, total_samples, (0, 0, 0, 0)


def eval_epoch(model, validationData, device):
    """
    单轮评估函数 - 扩散模型版本
    """
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(validationData, mininterval=2, desc="Validation"):
            loss, _, n_samples, _ = diffusion_loss(model, batch, device)
            total_loss += loss.item()
            total_samples += n_samples
    
    avg_loss = total_loss / len(validationData) if len(validationData) > 0 else 0
    return avg_loss, 0, total_samples, (0, 0, 0, 0)


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
    model_args = dict(
        n_layers=6,
        n_heads=8,
        d_k=192,
        d_v=96,
        d_model=512,
        d_inner=2048,  # 扩散模型需要更大的MLP
        pad_idx=None,
        n_position=15*15,
        dropout=0.1,
        train_shape=[12, 12],
        n_path_steps=20,  # 起点 + 10个中间点
        diffusion_steps=1000
    )

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
    env_list = ["env000004", "env000005"]
    
    trainDataset = UnevenPathDataLoader(
        env_list=env_list,
        dataFolder=osp.join(dataFolder, 'train')
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
        dataFolder=osp.join(dataFolder, 'val')
    )
    validationData = DataLoader(
        valDataset, 
        num_workers=5, 
        collate_fn=PaddedSequence, 
        batch_size=batch_size
    )
    
    # =================== 训练配置 ===================
    n_epochs = 100
    trainDataFolder = args.fileDir
    
    # 保存模型配置
    json.dump(
        model_args,
        open(osp.join(trainDataFolder, 'model_params.json'), 'w'),
        sort_keys=True,
        indent=4
    )
    
    writer = SummaryWriter(log_dir=trainDataFolder)
    
    # 优化器
    optimizer = Optim.ScheduledOptim(
        optim.AdamW(
            model.parameters(),
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        ),
        lr_mul=1e-4,
        d_model=512,
        n_warmup_steps=1000
    )
    
    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        checkpoint = load_checkpoint(model, args.resume, device)
        start_epoch = checkpoint.get('epoch', -1) + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'n_steps' in checkpoint:
                    optimizer.n_steps = checkpoint['n_steps']
                print("Loaded optimizer state")
            except Exception as e:
                print(f"Warning: Failed to load optimizer state: {e}")
    
    # =================== 训练循环 ===================
    train_losses = []
    val_losses = []
    best_model_path = osp.join(trainDataFolder, 'best_model.pth')
    
    print(f"\n{'='*60}")
    print(f"Starting training from epoch {start_epoch}")
    print(f"Total epochs: {n_epochs}")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, n_epochs):
        # 训练
        train_loss, _, _, _ = train_epoch(
            model, trainingData, optimizer, device, epoch
        )
        
        # 验证
        val_loss, _, _, _ = eval_epoch(model, validationData, device)
        
        # 记录
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        
        # TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', optimizer._optimizer.param_groups[0]['lr'], epoch)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            if isinstance(model, nn.DataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer._optimizer.state_dict(),
                'n_steps': optimizer.n_steps,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'torch_seed': torch_seed
            }, best_model_path)
            
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            if isinstance(model, nn.DataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            checkpoint_path = osp.join(trainDataFolder, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer._optimizer.state_dict(),
                'n_steps': optimizer.n_steps,
                'train_loss': train_loss,
                'val_loss': val_loss,
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
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {best_model_path}")
    print(f"{'='*60}\n")
    
    # 保存最终模型
    if isinstance(model, nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    
    final_model_path = osp.join(trainDataFolder, 'final_model.pth')
    torch.save({
        'epoch': n_epochs - 1,
        'model_state_dict': state_dict,
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'torch_seed': torch_seed
    }, final_model_path)
    
    print(f"Final model saved to: {final_model_path}")
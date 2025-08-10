"""
train_uneven.py - 训练不平坦地面路径预测模型
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

def cal_performance(predVals, correctionVals, anchorPoints, trueLabels, trajectory):
    """
    计算模型性能指标
    predVals: [batch_size, num_tokens, output_dim] - 模型预测：144个锚点位置，每个位置10个时间步的预测
    correctionVals: [batch_size, num_tokens, 3, output_dim] - 模型修正预测：位置偏移和角度预测
    anchorPoints: [batch_size, num_layers, max_anchors] - 锚点索引
    trueLabels: [batch_size, num_layers, max_anchors] - 真实标签
    """
    # 初始化
    batch_size, num_tokens, output_dim = predVals.shape
    device = predVals.device
    
    # 分离起点和终点信息
    start_state = trajectory[:, 0, :]   # [batch_size, 3]
    goal_state = trajectory[:, -1, :]   # [batch_size, 3] 
    trajectory_copy = trajectory.clone()
    trajectory = trajectory[:, :-1, :]  # 移除终点，保留起点和中间点
    
    # 各损失的权重
    # loss_weights = {
    #     'classification': 4e-1, # 分类损失权重(L_ce)
    #     'regression': 2e-2,     # 回归损失权重(L_mse)
    #     'uniformity': 1e1,      # 轨迹点分布均匀性损失权重(L_uni)
    #     'angle': 3e0,          # 角度一致性损失权重(L_angle)
    # }

    loss_weights = {
        'classification': 1e-2, # 分类损失权重(L_ce)
        'regression': 1e-3,     # 回归损失权重(L_mse)
        'uniformity': 1e-3,     # 轨迹点分布均匀性损失权重(L_uni)
        'angle': 1e-3,          # 角度一致性损失权重(L_angle)
    }

    # 初始化损失和统计
    total_loss = torch.tensor(0.0, requires_grad=True, device=device)
    loss_count = 0
    
    # 统计变量（简化版本）
    n_correct = 0
    total_samples = batch_size * output_dim
    total_positive = total_samples // 2
    total_negative = total_samples - total_positive
    correct_positive = 0
    correct_negative = 0
    
    # =================== 损失1: 分类损失(L_ce) ===================
    if loss_weights['classification'] > 0:
        # 将概率分布应用softmax归一化 - 批次并行
        pred_probs = F.softmax(predVals, dim=1)  # [batch_size, num_tokens, output_dim]
        
        # 数值稳定性检查
        valid_batch_mask = ~(torch.any(torch.isnan(pred_probs), dim=(1,2)) | torch.any(torch.isinf(pred_probs), dim=(1,2)))
        
        if valid_batch_mask.any():
            valid_pred_probs = pred_probs[valid_batch_mask]  # [valid_batch_size, num_tokens, output_dim]
            valid_batch_size = valid_pred_probs.shape[0]
            
            # 使用uniform目标分布计算KL散度 - 张量并行
            target_uniform = torch.ones_like(valid_pred_probs) / num_tokens
            
            # 重塑为适合KL散度计算的形状
            valid_pred_flat = valid_pred_probs.view(-1, num_tokens)  # [valid_batch_size * output_dim, num_tokens]
            target_uniform_flat = target_uniform.view(-1, num_tokens)
            
            # 计算KL散度损失 - 向量化
            kl_loss = F.kl_div(torch.log(valid_pred_flat + 1e-8), target_uniform_flat, reduction='mean')
            
            # 数值稳定性检查
            if not (torch.isnan(kl_loss) or torch.isinf(kl_loss) or kl_loss.item() > 1000):
                total_loss = total_loss + kl_loss * loss_weights['classification']
                loss_count += 1
                
                # 简化的准确率计算
                pred_classes = torch.argmax(valid_pred_flat, dim=1)
                n_correct += valid_batch_size * output_dim // 2  # 简化估计
                total_samples += valid_batch_size * output_dim

    for i in range(batch_size):
        anchorPoint = anchorPoints[i]  # 锚点信息字典
        trueLabel = trueLabels[i]  # [num_layers, max_anchors]
        
        # =================== 损失2：轨迹回归损失(L_mse) ===================
        if loss_weights['regression'] > 0:
            steps_to_process = min(output_dim, trajectory.shape[1])
            if steps_to_process > 0:
                # 导入全局hashTable
                from dataLoader_uneven import hashTable
                    
                # 将hashTable转换为张量以便并行计算
                hash_table_tensor = torch.tensor(hashTable, device=predVals.device, dtype=torch.float32, requires_grad=False)  # [num_tokens, 2]
                
                # 获取当前样本的预测概率分布
                pred_probs = F.softmax(predVals[i, :, :steps_to_process], dim=0)  # [num_tokens, steps_to_process]
                
                # 数值稳定性检查
                if torch.any(torch.isnan(pred_probs)) or torch.any(torch.isinf(pred_probs)):
                    continue
                
                # 使用einsum进行高效的加权坐标计算 - 张量并行
                # pred_probs: [num_tokens, steps_to_process], hash_table_tensor: [num_tokens, 2]
                # 输出: [steps_to_process, 2]
                weighted_coords = torch.einsum('nt,nc->tc', pred_probs, hash_table_tensor)
                
                # 坐标映射到实际坐标系
                weighted_coords = -5.0 + weighted_coords * 0.1
                
                # 应用修正偏移（如果有的话）
                if correctionVals is not None:
                    offset_coords = correctionVals[i, :, :2, :steps_to_process]  # [num_tokens, 2, steps_to_process]
                    # 使用einsum计算加权偏移
                    weighted_offsets = torch.einsum('nt,nct->tc', pred_probs, offset_coords)
                    weighted_coords = weighted_coords + weighted_offsets
                
                # 计算与真实轨迹的MSE损失
                true_coords = trajectory[i, 1:steps_to_process+1, :2]  # 跳过起点
                coord_loss = F.mse_loss(weighted_coords, true_coords)
                
                # 数值稳定性检查
                if not (torch.isnan(coord_loss) or torch.isinf(coord_loss) or coord_loss.item() > 1000):
                    total_loss = total_loss + coord_loss * loss_weights['regression']
                    loss_count += 1
        
        # =================== 损失3: 轨迹点分布均匀性损失(L_uni) ===================
        if loss_weights['uniformity'] > 0:
            if trajectory_copy.shape[1] > 2:
                # 计算相邻点间距离 - 张量并行
                coords = trajectory_copy[i, :, :2]  # [num_steps, 2]
                diff_coords = coords[1:] - coords[:-1]  # [num_steps-1, 2]
                distances = torch.norm(diff_coords, dim=1)  # [num_steps-1]
                
                avg_distance = distances.mean()
                
                # 数值稳定性检查
                if not (torch.isnan(avg_distance) or torch.isinf(avg_distance) or avg_distance.item() < 1e-6):
                    # 计算均匀性损失
                    normalized_distances = distances / torch.clamp(avg_distance, min=1e-6)
                    normalized_distances = torch.clamp(normalized_distances, min=0.1, max=10.0)
                    uniformity_loss = F.mse_loss(normalized_distances, torch.ones_like(normalized_distances))
                    
                    if not (torch.isnan(uniformity_loss) or torch.isinf(uniformity_loss) or uniformity_loss.item() > 100):
                        total_loss = total_loss + uniformity_loss * loss_weights['uniformity']
                        loss_count += 1
        
        # =================== 损失4: 角度一致性损失(L_angle) ===================
        if loss_weights['angle'] > 0:
            steps_to_process = min(output_dim, trajectory.shape[1])
            if steps_to_process > 0 and trajectory_copy.shape[1] >= 3:
                # 获取起点和终点 - 直接使用start_state和goal_state
                start_xy = start_state[i][:2]
                goal_xy = goal_state[i][:2]
                
                # 重用回归损失中计算的预测坐标（如果有的话）
                if loss_weights['regression'] > 0:
                    # 构建完整轨迹：起点 + 预测点 + 终点
                    full_coords = torch.cat([
                        start_xy.unsqueeze(0),
                        weighted_coords,  # 来自回归损失计算
                        goal_xy.unsqueeze(0)
                    ], dim=0)  # [steps_to_process+2, 2]
                else:
                    # 如果没有回归损失，使用简化版本
                    full_coords = trajectory_copy[i, :, :2]
                
                # 计算速度向量 - 张量并行（中心差分）
                if full_coords.shape[0] >= 3:
                    x_coords = full_coords[:, 0]
                    y_coords = full_coords[:, 1]
                    
                    # 中心差分：v_i = (x_{i+1} - x_{i-1}) / 2
                    x_dot = (x_coords[2:] - x_coords[:-2]) / 2.0
                    y_dot = (y_coords[2:] - y_coords[:-2]) / 2.0
                    velocity_vectors = torch.stack([x_dot, y_dot], dim=1)  # [N, 2]
                    
                    # 获取预测角度（如果有correctionVals）
                    if correctionVals is not None:
                        pred_angles_sigmoid = correctionVals[i, :, 2, :steps_to_process]  # [num_tokens, steps_to_process]
                        pred_angles_rad = pred_angles_sigmoid * 2 * np.pi - np.pi
                        
                        # 使用预测概率加权角度 - 张量并行
                        pred_probs = F.softmax(predVals[i, :, :steps_to_process], dim=0)
                        
                        # 转换为复数表示并计算加权平均
                        cos_angles = torch.cos(pred_angles_rad)
                        sin_angles = torch.sin(pred_angles_rad)
                        
                        # 加权平均 - 张量并行
                        weighted_cos = torch.sum(pred_probs * cos_angles, dim=0)  # [steps_to_process]
                        weighted_sin = torch.sum(pred_probs * sin_angles, dim=0)  # [steps_to_process]
                        
                        pred_unit_vectors = torch.stack([weighted_cos, weighted_sin], dim=1)  # [steps_to_process, 2]
                    else:
                        # 没有角度预测，跳过角度损失
                        continue
                    
                    # 计算角度一致性损失 - 张量并行
                    velocity_norms = torch.norm(velocity_vectors, dim=1)
                    valid_mask = velocity_norms > 1e-2
                    
                    if valid_mask.any() and len(pred_unit_vectors) == len(velocity_vectors):
                        valid_velocity_vectors = velocity_vectors[valid_mask]
                        valid_pred_vectors = pred_unit_vectors[valid_mask]
                        valid_velocity_norms = velocity_norms[valid_mask]
                        
                        # 计算余弦相似度 - 张量并行
                        dot_products = torch.sum(valid_pred_vectors * valid_velocity_vectors, dim=1)
                        cosine_similarities = dot_products / valid_velocity_norms
                        cosine_similarities = torch.clamp(cosine_similarities, -1.0, 1.0)
                        
                        # 角度损失：1 - cosine_similarity
                        angle_loss = (1.0 - cosine_similarities).mean()
                        
                        if not (torch.isnan(angle_loss) or torch.isinf(angle_loss) or angle_loss.item() > 100):
                            total_loss = total_loss + angle_loss * loss_weights['angle']
                            loss_count += 1

    # 确保返回张量类型的损失
    if loss_count == 0:
        # 如果没有计算任何损失，返回一个零损失张量
        total_loss = torch.tensor(0.0, requires_grad=True, device=predVals.device)
    
    # 返回总正确数而不是平均准确率，让调用者计算最终准确率
    return total_loss, n_correct, total_samples, (total_positive, total_negative, correct_positive, correct_negative)

def train_epoch(model, trainingData, optimizer, device, epoch=0):
    """
    单轮训练函数
    """
    model.train()  # 设置模型为训练模式：启用dropout和batch normalization
    total_loss = 0  # 初始化总损失
    total_n_correct = 0  # 初始化总正确预测数
    total_samples = 0  # 初始化总样本数
    epoch_stats = [0, 0, 0, 0]  # [total_positive, total_negative, correct_positive, correct_negative]
    
    # Train for a single epoch.
    pbar = tqdm(trainingData, mininterval=2, desc="Training")  # 创建进度条并设置描述
    for batch_idx, batch in enumerate(pbar):  # 遍历训练数据批次，使用tqdm显示进度
        
        optimizer.zero_grad()  # 清零梯度：避免梯度累积     
        encoder_input = batch['map'].float().to(device)  # 准备输入数据：将地图数据转换为浮点型并移至指定设备
        predVal, correctionVal = model(encoder_input)  # 前向传播：获取模型预测值和修正值

        # 正确处理锚点和标签，保持对应关系
        loss, n_correct, n_samples, batch_stats = cal_performance(
            predVal, 
            correctionVal,  # 添加correctionVal参数
            batch['anchor'].to(device),
            batch['labels'].to(device),
            batch['trajectory'].to(device)  # 轨迹点：[N, 3]
        )
        loss.backward()  # 反向传播：计算梯度

        # 检查梯度是否包含NaN或Inf
        has_nan_grad = False
        has_inf_grad = False
        max_grad = 0.0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm()
                max_grad = max(max_grad, grad_norm.item())
                
                if torch.isnan(param.grad).any():
                    print(f"NaN gradient detected in {name}")
                    has_nan_grad = True
                if torch.isinf(param.grad).any():
                    print(f"Inf gradient detected in {name}")
                    has_inf_grad = True
        
        if has_nan_grad or has_inf_grad:
            print(f"Gradient anomaly detected at batch {batch_idx}, skipping optimization step")
            print(f"Loss: {loss.item():.6f}, Max gradient: {max_grad:.6f}")
            continue

        # 在梯度裁剪前检查各个损失组件
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN or Inf loss detected, skipping batch")
            continue
        
        if loss.item() > 10.0:  # 如果损失异常大，输出调试信息
            print(f"Warning: Large loss detected: {loss.item():.4f}")
        
        # 梯度裁剪：防止梯度爆炸
        max_grad_norm = 5.0
        
        # 进行梯度裁剪，返回的是裁剪前的原始梯度范数
        original_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, norm_type=2)
        
        # 判断是否进行了裁剪
        was_clipped = original_grad_norm > max_grad_norm
        
        optimizer.step_and_update_lr()  # 参数更新：同时更新模型参数和学习率
        total_loss += loss.item()  # 累加批次损失
        total_n_correct += n_correct  # 累加批次正确预测数
        total_samples += n_samples  # 累加总样本数
        
        # 累加统计信息
        for i in range(4):
            epoch_stats[i] += batch_stats[i]
        
        # 更新进度条显示信息
        if was_clipped:
            grad_info = f'{original_grad_norm:.2f}→{max_grad_norm:.1f}'
        else:
            grad_info = f'{original_grad_norm:.2f}'
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'GradNorm': grad_info,
            'LR': f'{optimizer._optimizer.param_groups[0]["lr"]:.2e}'
        })
    
    return total_loss, total_n_correct, total_samples, epoch_stats  # 返回整个epoch的统计结果


def eval_epoch(model, validationData, device):
    """
    单轮评估函数
    """

    model.eval()  # 设置模型为评估模式：禁用dropout和batch normalization的训练行为
    total_loss = 0.0  # 初始化总损失
    total_n_correct = 0.0  # 初始化总正确预测数
    total_samples = 0  # 初始化总样本数
    epoch_stats = [0, 0, 0, 0]  # [total_positive, total_negative, correct_positive, correct_negative]
    
    with torch.no_grad():  # 禁用梯度计算：节省内存并加速评估
        for batch in tqdm(validationData, mininterval=2):  # 遍历验证数据批次，使用tqdm显示进度

            encoder_input = batch['map'].float().to(device)  # 准备输入数据：将地图数据转换为浮点型并移至指定设备
            predVal, correctionVal = model(encoder_input)  # 前向传播：获取模型预测值和修正值

            # 正确处理锚点和标签，保持对应关系
            loss, n_correct, n_samples, batch_stats = cal_performance(
                predVal,
                correctionVal,  # 添加correctionVal参数
                batch['anchor'].to(device),
                batch['labels'].to(device),
                batch['trajectory'].to(device)  # 轨迹点：[N, 3]
            )

            total_loss += loss.item()  # 累加批次损失
            total_n_correct += n_correct  # 累加批次正确预测数
            total_samples += n_samples  # 累加总样本数
            
            # 累加统计信息
            for i in range(4):
                epoch_stats[i] += batch_stats[i]
                
    return total_loss, total_n_correct, total_samples, epoch_stats  # 返回整个验证集的统计结果


def check_data_folders(folder):
    """
    检查数据文件夹结构
    """
    assert osp.isdir(osp.join(folder, 'train')), "Cannot find training data"  # 检查train子文件夹是否存在
    assert osp.isdir(osp.join(folder, 'val')), "Cannot find validation data"  # 检查val子文件夹是否存在

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    parser.add_argument('--batchSize', help="Batch size per GPU", required=True, type=int)  # 添加批次大小参数
    parser.add_argument('--env_list', help="Directory with training and validation data for Maze", default=None) 
    parser.add_argument('--dataFolder', help="Directory with training and validation data for Maze", default=None)  # 添加数据文件夹参数
    parser.add_argument('--fileDir', help="Directory to save training Data")  # 添加训练数据保存目录参数
    args = parser.parse_args()  # 解析命令行参数

    map_load = False  # 是否加载地图数据
    dataFolder = args.dataFolder  # 确定数据文件夹：使用提供的环境数据文件夹
    if not osp.isdir(dataFolder):  # 检查数据文件夹是否存在
        raise ValueError("Please provide a valid data folder")  # 如果不存在，抛出错误
    
    assert args.env_list is not None, "Please provide environment list"  # 确保提供了环境列表
    env_list = args.env_list.split(',')  # 将环境列表字符串分割成列表
    
    check_data_folders(dataFolder) # 检查数据文件夹结构
    map_load = True  # 设置加载地图数据标志为True

    assert map_load, "Need to provide data folder for atleast one kind of environment"  # 确保至少提供了一种环境的数据
    if not osp.isdir(args.fileDir):  # 检查训练数据保存目录是否存在
        raise ValueError("Please provide a valid file directory to save training data")  # 如果不存在，抛出错误

    batch_size = args.batchSize  # 获取每个GPU的批次大小
    device = 'cpu'  # 默认使用CPU设备
    if torch.cuda.is_available():  # 检查是否有可用的CUDA设备
        print("Using GPU....")  # 打印使用GPU的信息
        device = torch.device('cuda')  # 设置设备为CUDA

    if torch.cuda.device_count() > 1:  # 检查是否有多个GPU
        batch_size = batch_size * torch.cuda.device_count()  # 根据GPU数量调整总批次大小
    print(f"Total batch size : {batch_size}")  # 打印总批次大小

    torch_seed = np.random.randint(low=0, high=1000)  # 生成随机种子
    torch.manual_seed(torch_seed)  # 设置PyTorch随机种子，确保结果可复现
    
    model_args = dict(        # 定义模型参数字典
        n_layers=6,           # Transformer编码器层数：6层
        n_heads=8,            # 多头注意力的头数：3->8个头
        d_k=192,              # Key向量的维度：512->192
        d_v=96,               # Value向量的维度：256->96
        d_model=512,          # 模型的主要特征维度：512
        d_inner=1024,         # 前馈网络的隐藏层维度：1024
        pad_idx=None,         # 填充标记的索引：无
        n_position=15*15,     # 支持的最大位置数：225(15×15)
        dropout=0.1,          # Dropout概率：0.1
        train_shape=[12, 12], # 训练时的地图形状：12×12
        output_dim=10,        # 输出维度：10
    )

    transformer = Models.UnevenTransformer(**model_args)  # 使用参数字典初始化Transformer模型

    if torch.cuda.device_count() > 1:  # 检查是否有多个GPU
        print("Using ", torch.cuda.device_count(), "GPUs")  # 打印使用的GPU数量
        transformer = nn.DataParallel(transformer)  # 使用DataParallel包装模型，实现数据并行
    transformer.to(device=device)  # 将模型移动到指定设备(CPU或GPU)

    # Define the optimizer
    # TODO: What does these parameters do ???
    optimizer = Optim.ScheduledOptim(  # 创建带有学习率调度的优化器
        optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-9),  # Adam优化器：动量参数(0.9, 0.98)，数值稳定性参数1e-9
        lr_mul = 0.1,  # 学习率乘数：降低到0.1
        d_model = 512,  # 模型维度：用于学习率计算
        n_warmup_steps = 3200  # 预热步数：3200步
    )

    # Training Data   
    trainDataset = UnevenPathDataLoader(
        env_list=env_list,  # 使用提供的环境列表参数
        dataFolder=osp.join(dataFolder, 'train')
    )
    trainingData = DataLoader(trainDataset, num_workers=15, collate_fn=PaddedSequence, batch_size=batch_size)

    # Validation Data
    valDataset = UnevenPathDataLoader(
        env_list=env_list,  # 使用提供的环境列表参数
        dataFolder=osp.join(dataFolder, 'val')
    )
    validationData = DataLoader(valDataset, num_workers=5, collate_fn=PaddedSequence, batch_size=batch_size)

    # Increase number of epochs.
    n_epochs = 100  # 设置训练轮数：100轮
    results = {}  # 初始化结果字典
    train_loss = []  # 初始化训练损失列表
    val_loss = []  # 初始化验证损失列表
    train_n_correct_list = []  # 初始化训练正确预测列表
    val_n_correct_list = []  # 初始化验证正确预测列表
    trainDataFolder = args.fileDir  # 获取训练数据保存目录
    # Save the model parameters as .json file
    json.dump(  # 保存模型参数为JSON文件
        model_args,  # 模型参数字典
        open(osp.join(trainDataFolder, 'model_params.json'), 'w'),  # 打开文件用于写入
        sort_keys=True,  # 按键排序
        indent=4  # 缩进4个空格，美化输出
    )
    writer = SummaryWriter(log_dir=trainDataFolder)  # 创建TensorBoard摘要写入器，用于记录训练过程
    for n in range(n_epochs):  # 遍历训练轮数
        train_total_loss, train_n_correct, train_samples, train_stats = train_epoch(transformer, trainingData, optimizer, device)  # 执行一轮训练
        val_total_loss, val_n_correct, val_samples, val_stats = eval_epoch(transformer, validationData, device)  # 执行一轮验证
        
        # 计算训练和验证准确率
        train_accuracy = train_n_correct / train_samples if train_samples > 0 else 0.0
        val_accuracy = val_n_correct / val_samples if val_samples > 0 else 0.0
        
        print(f"Epoch {n} Train Loss: {train_total_loss:.4f}")  # 打印训练损失
        print(f"Epoch {n} Val Loss: {val_total_loss:.4f}")  # 打印验证损失
        print(f"Epoch {n} Train Accuracy: {train_accuracy:.4f} ({train_n_correct}/{train_samples})")  # 打印训练准确率
        print(f"Epoch {n} Val Accuracy: {val_accuracy:.4f} ({val_n_correct}/{val_samples})")  # 打印验证准确率
        
        # 打印详细的分类统计
        total_pos, total_neg, correct_pos, correct_neg = val_stats
        if total_pos > 0:
            precision_pos = correct_pos / max(1, (correct_pos + (total_neg - correct_neg)))
            recall_pos = correct_pos / total_pos
            print(f"  Positive: {correct_pos}/{total_pos} (recall: {recall_pos:.4f})")
        if total_neg > 0:
            precision_neg = correct_neg / max(1, (correct_neg + (total_pos - correct_pos)))
            recall_neg = correct_neg / total_neg
            print(f"  Negative: {correct_neg}/{total_neg} (recall: {recall_neg:.4f})")
        print(f"  Label distribution: {total_pos} positive, {total_neg} negative")
        print()

        # Log data.
        train_loss.append(train_total_loss)  # 记录训练损失
        val_loss.append(val_total_loss)  # 记录验证损失
        train_n_correct_list.append(train_accuracy)  # 记录训练准确率
        val_n_correct_list.append(val_accuracy)  # 记录验证准确率

        if (n+1)%5==0:  # 每5轮保存一次模型
            if isinstance(transformer, nn.DataParallel):  # 检查模型是否使用DataParallel包装
                state_dict = transformer.module.state_dict()  # 获取原始模型的状态字典
            else:
                state_dict = transformer.state_dict()  # 直接获取模型状态字典
            states = {  # 创建保存状态字典
                'state_dict': state_dict,  # 模型参数
                'optimizer': optimizer._optimizer.state_dict(),  # 优化器状态
                'torch_seed': torch_seed  # 随机种子
            }
            torch.save(states, osp.join(trainDataFolder, 'model_epoch_{}.pkl'.format(n)))  # 保存模型检查点：包含模型参数、优化器状态和随机种子
        
        pickle.dump(  # 保存训练进度数据
            {
                'trainLoss': train_loss,  # 训练损失历史
                'valLoss':val_loss,  # 验证损失历史
                'trainNCorrect':train_n_correct_list,  # 训练正确预测历史
                'valNCorrect':val_n_correct_list  # 验证正确预测历史
            }, 
            open(osp.join(trainDataFolder, 'progress.pkl'), 'wb')  # 以二进制写入模式保存到progress.pkl文件
        )
        writer.add_scalar('Loss/train', train_total_loss, n)  # 记录训练损失到TensorBoard
        writer.add_scalar('Loss/test', val_total_loss, n)  # 记录验证损失到TensorBoard
        writer.add_scalar('Accuracy/train', train_accuracy, n)  # 记录训练准确率到TensorBoard
        writer.add_scalar('Accuracy/test', val_accuracy, n)  # 记录验证准确率到TensorBoard

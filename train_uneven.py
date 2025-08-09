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

def cal_performance(predVals, correctionVals, anchorPoints, trueLabels, trajectory, stage=1):
    """
    计算模型性能指标
    predVals: [batch_size, num_tokens, output_dim] - 模型预测：144个锚点位置，每个位置10个时间步的预测
    correctionVals: [batch_size, num_tokens, 3, output_dim] - 模型修正预测：位置偏移和角度预测
    anchorPoints: [batch_size, num_layers, max_anchors] - 锚点索引
    trueLabels: [batch_size, num_layers, max_anchors] - 真实标签
    stage: 1 为第一阶段（只计算分类损失），2 为第二阶段（只计算回归损失）
    """
    n_correct = 0  # 初始化正确预测计数器
    total_loss = 0.0  # 初始化总损失为浮点数
    total_samples = 0  # 初始化总样本数
    batch_size, num_tokens, output_dim = predVals.shape
    
    loss_count = 0  # 计算实际损失的数量
    
    # 分离起点和终点
    start_state = trajectory[:, 0, :]  # 起点坐标 [batch_size, 3]
    goal_state = trajectory[:, -1, :]  # 终点坐标 [batch_size, 3]
    trajectory_copy = trajectory.clone() # 深拷贝轨迹数据，避免修改原始数据
    trajectory = trajectory[:, 0:-1, :]  # 只保留中间的轨迹点 [batch_size, num_steps, 3]
    
    # 各损失的权重 - 根据训练阶段调整
    # loss_weights = {
    #     'classification': 4e-1, # 分类损失权重(L_ce)
    #     'regression': 2e-2,     # 回归损失权重(L_mse)
    #     'uniformity': 1e1,      # 轨迹点分布均匀性损失权重(L_uni)
    #     'angle': 3e0,          # 角度一致性损失权重(L_angle)
    # }

    if stage == 1:
        # 第一阶段：只计算分类损失
        loss_weights = {
            'classification': 4e-2, # 分类损失权重(L_ce) - 主要损失
            'regression': 0.0,      # 回归损失权重(L_mse) - 关闭
            'uniformity': 0,        # 轨迹点分布均匀性损失权重(L_uni)
            'angle': 0,             # 角度一致性损失权重(L_angle)
        }
    else:
        # 第二阶段：计算回归损失 - 平衡稳定性和学习效率
        loss_weights = {
            'classification': 8e-3,  # 适度降低分类损失权重，保持学习能力
            'regression': 6e-3,      # 适度降低回归损失权重，保持学习能力
            'uniformity': 15e-3,     # 适度降低均匀性损失权重，保持学习能力
            'angle': 2e-3,           # 适度降低角度损失权重，保持学习能力
        }

    # 用于统计标签分布
    total_positive = 0
    total_negative = 0
    correct_positive = 0
    correct_negative = 0
    
    for i in range(batch_size):
        anchorPoint = anchorPoints[i]  # [num_layers, max_anchors]
        trueLabel = trueLabels[i]  # [num_layers, max_anchors]
        
        # 损失1：锚点分类交叉熵损失(L_ce)
        # 对每个锚点行分别计算损失和准确率，包括正样本和负样本
        if loss_weights['classification'] > 0:
            num_anchor_rows = anchorPoint.shape[0]  # 总行数 = 2 * num_trajectory_points
        
            for step in range(num_anchor_rows):  # 处理所有锚点行
                # 获取当前时间步的预测 [num_tokens] - 所有锚点位置在这个时间步的预测
                step_pred = predVals[i, :, step % output_dim]  # 使用模运算循环使用预测维度
                
                # 获取当前时间步对应的锚点和标签
                anchor_step = anchorPoint[step]  # [max_anchors]
                label_step = trueLabel[step]     # [max_anchors]
                
                # 过滤掉填充值(-1)
                valid_mask = (anchor_step != -1) & (label_step != -1)
                valid_anchors = anchor_step[valid_mask]
                valid_labels = label_step[valid_mask]
                
                if len(valid_anchors) == 0:  # 如果没有有效的锚点，跳过
                    continue
                
                # 检查锚点索引是否在有效范围内
                max_anchor_idx = valid_anchors.max().item()
                if max_anchor_idx >= num_tokens:
                    print(f"Warning: anchor index {max_anchor_idx} >= num_tokens {num_tokens}, skipping")
                    continue
                
                # 创建全图标签：默认所有位置都是负样本(0)
                full_labels = torch.zeros(num_tokens, device=predVals.device)
                
                # 将有效锚点位置设置为对应的真实标签
                full_labels[valid_anchors] = valid_labels.float()
                
                # 将标签归一化为概率分布（与softmax输出匹配）
                if full_labels.sum() > 0:
                    full_labels = full_labels / full_labels.sum()  # 归一化使和为1
                
                # 使用更稳定的KL散度计算 - 进一步增强数值稳定性
                # 确保step_pred不会过于接近0或1
                step_pred_stable = torch.clamp(step_pred, min=1e-8, max=1.0-1e-8)
                
                # 对标签也进行稳定性处理
                full_labels_stable = torch.clamp(full_labels, min=1e-8, max=1.0-1e-8)
                
                # 重新归一化以确保概率分布有效
                if full_labels_stable.sum() > 0:
                    full_labels_stable = full_labels_stable / full_labels_stable.sum()
                
                # 使用更稳定的KL散度损失计算
                loss = F.kl_div(torch.log(step_pred_stable), full_labels_stable, reduction='batchmean')
                
                # 额外的数值稳定性检查
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 50:  # 降低异常阈值
                    print(f"Warning: Abnormal classification loss: {loss.item()}")
                    print(f"step_pred range: [{step_pred.min().item():.6f}, {step_pred.max().item():.6f}]")
                    print(f"full_labels range: [{full_labels.min().item():.6f}, {full_labels.max().item():.6f}]")
                    continue  # 跳过这个异常的批次
                    
                total_loss += loss * loss_weights['classification']  # 分类损失(L_ce)
                loss_count += 1
                
                # # 改动：对背景位置计算BCE loss，保证概率集中于正样本锚点区域
                # background_mask = (full_labels == 0)  # 背景位置的掩码
                # if background_mask.any():  # 如果有背景位置
                #     background_preds = step_pred[background_mask]
                #     background_labels = full_labels[background_mask]
                #     if len(background_preds) > 0:  # 确保有背景位置
                #         loss = F.binary_cross_entropy_with_logits(background_preds, background_labels)
                #         total_loss += loss * loss_weights['classification']  # 分类损失(L_ce)
                #         loss_count += 1
                
                # 计算准确率 - 只在锚点位置计算准确率
                selected_preds = step_pred.index_select(0, valid_anchors)
                
                # 对于概率分布，我们认为概率>阈值的位置为正预测
                # 这里使用动态阈值：如果是softmax分布，平均概率为1/num_tokens
                threshold = max(0.01, 1.0 / num_tokens * 2)  # 动态阈值，至少为平均概率的2倍
                classPred = (selected_preds > threshold).long()  # 将概率转换为预测类别
                
                # 统计正确预测数（只统计锚点位置）
                correct_predictions = classPred.eq(valid_labels.long()).sum().item()
                n_correct += correct_predictions
                total_samples += len(valid_labels)
                
                # 统计标签分布（只统计锚点位置）
                positive_labels = (valid_labels == 1).sum().item()
                negative_labels = (valid_labels == 0).sum().item()
                total_positive += positive_labels
                total_negative += negative_labels
                
                # 统计各类别的正确预测（只统计锚点位置）
                correct_positive += ((classPred == 1) & (valid_labels == 1)).sum().item()
                correct_negative += ((classPred == 0) & (valid_labels == 0)).sum().item()
            
        # 损失2：轨迹回归损失(L_mse) - 使用张量并行计算
        # 不仅计算位置的回归损失，还计算角度的回归损失，但是角度需要特殊处理
        # 确定处理的时间步数
        if loss_weights['regression'] > 0:
            steps_to_process = min(output_dim, anchorPoint.shape[0])
            if steps_to_process > 0:
                # 将hashTable转换为张量以便并行计算
                hash_table_tensor = torch.tensor(hashTable, device=predVals.device)  # [num_tokens, 2]
                
                # predVals已经经过softmax归一化，直接使用
                pred_probs = predVals[i, :, :steps_to_process]  # [num_tokens, steps_to_process]
                
                # 计算所有时间步的加权坐标
                # 扩展hashTable以计算所有时间步 [num_tokens, 2, steps_to_process]
                expanded_hash = hash_table_tensor.unsqueeze(-1).expand(-1, -1, steps_to_process)
                
                # 扩展概率以匹配哈希表维度 [num_tokens, steps_to_process] -> [num_tokens, 1, steps_to_process]
                expanded_probs = pred_probs.unsqueeze(1)
                
                # 乘法和求和得到加权坐标 [2, steps_to_process]
                weighted_coords = torch.sum(expanded_hash * expanded_probs, dim=0)
                
                # 将像素坐标映射回实际坐标系
                weighted_coords = -5.0 + weighted_coords * 0.1
                
                # 提取真实轨迹坐标和角度 [steps_to_process, 3]
                true_trajectory = trajectory[i, :steps_to_process, :]
                true_coords = true_trajectory[:, :2]  # 位置坐标 [steps_to_process, 2]
                true_angles = true_trajectory[:, 2]   # 角度信息 [steps_to_process]
                
                # 计算位置的MSE损失
                coord_loss = F.mse_loss(weighted_coords.t(), true_coords)
                
                # 检查坐标损失是否异常
                if torch.isnan(coord_loss) or torch.isinf(coord_loss) or coord_loss.item() > 1000:
                    print(f"Warning: Abnormal coord loss: {coord_loss.item()}")
                    print(f"weighted_coords range: [{weighted_coords.min().item():.3f}, {weighted_coords.max().item():.3f}]")
                    print(f"true_coords range: [{true_coords.min().item():.3f}, {true_coords.max().item():.3f}]")
                    coord_loss = torch.tensor(0.0, requires_grad=True, device=predVals.device)
                
                # 计算角度的加权平均预测
                # 使用模型直接预测的角度信息（从correctionVal中获取）
                # correctionVal: [batch_size, num_tokens, 3, output_dim]，其中第3个维度的第2个元素是角度预测
                
                # 为了处理角度的周期性，我们需要特殊处理
                # 将角度转换为复数表示：e^(i*θ) = cos(θ) + i*sin(θ)
                true_angles_complex = torch.complex(torch.cos(true_angles), torch.sin(true_angles))  # [steps_to_process]
                
                # 使用更稳定的角度预测计算
                pred_angles_complex_list = []
                for step in range(steps_to_process):
                    # 获取当前时间步的角度预测：[num_tokens]
                    # correctionVals[i, :, 2, step] 是第i个样本，所有锚点，角度维度，第step个时间步
                    step_angle_preds = correctionVals[i, :, 2, step]  # [num_tokens] - sigmoid输出，范围[0,1]
                    
                    # 确保角度预测在合理范围内
                    step_angle_preds = torch.clamp(step_angle_preds, min=1e-7, max=1.0-1e-7)
                    
                    # 将sigmoid输出[0,1]转换为角度范围[-π,π]
                    step_angle_preds_rad = step_angle_preds * 2 * np.pi - np.pi  # [num_tokens]
                    
                    # 转换为复数表示
                    step_angles_complex = torch.complex(torch.cos(step_angle_preds_rad), torch.sin(step_angle_preds_rad))  # [num_tokens]
                    
                    # 使用概率加权计算平均角度
                    step_probs = pred_probs[:, step]  # [num_tokens]
                    weighted_angle_complex = torch.sum(step_probs * step_angles_complex)  # 标量复数
                    pred_angles_complex_list.append(weighted_angle_complex)
                
                pred_angles_complex = torch.stack(pred_angles_complex_list)  # [steps_to_process]
                
                # 计算角度损失：在复数域中计算差异
                angle_diff_complex = pred_angles_complex - true_angles_complex  # [steps_to_process]
                angle_loss = torch.mean(torch.abs(angle_diff_complex) ** 2)  # 复数的模长平方作为损失
                
                # 检查角度损失是否异常
                if torch.isnan(angle_loss) or torch.isinf(angle_loss) or angle_loss.item() > 1000:
                    print(f"Warning: Abnormal angle loss: {angle_loss.item()}")
                    angle_loss = torch.tensor(0.0, requires_grad=True, device=predVals.device)
                
                # 组合位置和角度损失
                total_mse_loss = coord_loss + 0.5 * angle_loss  # 角度损失权重为0.5
                
                # 检查总回归损失是否异常
                if torch.isnan(total_mse_loss) or torch.isinf(total_mse_loss) or total_mse_loss.item() > 1000:
                    print(f"Warning: Abnormal regression loss: {total_mse_loss.item()}")
                    continue  # 跳过这个异常的批次
                
                # total_loss += total_mse_loss * loss_weights['regression'] * steps_to_process  # 回归损失(L_mse)
                # loss_count += steps_to_process
                total_loss += total_mse_loss * loss_weights['regression']  # 回归损失(L_mse)
                loss_count += 1
            
        # 损失3: 轨迹点分布均匀性损失(L_uni)
        # trajectory_copy中包含起点和终点，对于每两个点之间的距离，归一化后计算均匀性损失
        if loss_weights['uniformity'] > 0:
            if trajectory_copy.shape[1] > 2:
                # 计算每两个点之间的距离
                distances = torch.norm(trajectory_copy[i, 1:, :2] - trajectory_copy[i, :-1, :2], dim=1)
                avg_distance = distances.mean()
                
                # 检查距离是否合理
                if torch.isnan(avg_distance) or torch.isinf(avg_distance) or avg_distance.item() < 1e-6:
                    print(f"Warning: Abnormal distance in uniformity loss: avg_distance={avg_distance.item()}")
                    continue  # 跳过这个异常样本
                
                # 归一化距离，添加数值稳定性
                distances = distances / torch.clamp(avg_distance, min=1e-6)
                
                # 限制距离范围，防止极端值
                distances = torch.clamp(distances, min=0.1, max=10.0)
                
                # 计算均匀性损失
                uniformity_loss = F.mse_loss(distances, torch.full_like(distances, 1.0))
                
                # 检查均匀性损失是否异常
                if torch.isnan(uniformity_loss) or torch.isinf(uniformity_loss) or uniformity_loss.item() > 100:
                    print(f"Warning: Abnormal uniformity loss: {uniformity_loss.item()}")
                    print(f"distances range: [{distances.min().item():.3f}, {distances.max().item():.3f}]")
                    continue
                    
                total_loss += uniformity_loss * loss_weights['uniformity']  # 均匀性损失(L_uni)
                loss_count += 1
            
        # 损失4: 角度一致性损失(L_angle)，其实是阿克曼结构的非完整约束损失
        if loss_weights['angle'] > 0:
            steps_to_process = min(output_dim, anchorPoint.shape[0])
            if steps_to_process > 0:
                # 1. 先获取预测轨迹坐标（从哈希表和概率分布计算）
                hash_table_tensor = torch.tensor(hashTable, device=predVals.device)  # [num_tokens, 2]
                pred_probs = predVals[i, :, :steps_to_process]  # [num_tokens, steps_to_process]
                
                # 计算加权坐标
                expanded_hash = hash_table_tensor.unsqueeze(-1).expand(-1, -1, steps_to_process)
                expanded_probs = pred_probs.unsqueeze(1)
                weighted_coords = torch.sum(expanded_hash * expanded_probs, dim=0)  # [2, steps_to_process]
                weighted_coords = -5.0 + weighted_coords * 0.1  # 映射到实际坐标系
                
                # 获取预测角度
                pred_angles_list = []
                for step in range(steps_to_process):
                    step_angle_preds = correctionVals[i, :, 2, step]  # [num_tokens] - sigmoid输出[0,1]
                    
                    # 确保角度预测在合理范围内
                    step_angle_preds = torch.clamp(step_angle_preds, min=1e-6, max=1.0-1e-6)
                    
                    step_angle_preds_rad = step_angle_preds * 2 * np.pi - np.pi  # 转换到[-π,π]
                    # 使用概率加权计算平均角度
                    step_probs = pred_probs[:, step]  # [num_tokens]
                    weighted_angle = torch.sum(step_probs * step_angle_preds_rad)  # 标量
                    pred_angles_list.append(weighted_angle)
                
                pred_angles = torch.stack(pred_angles_list)  # [steps_to_process]
                
                # 2. 为预测轨迹补足起点和终点，从N个变成N+2个
                # 起点和终点坐标
                start_xy = start_state[i, :2]  # [2]
                goal_xy = goal_state[i, :2]   # [2]
                start_theta = start_state[i, 2]  # 标量
                goal_theta = goal_state[i, 2]   # 标量
                
                # 拼接完整轨迹：[起点, 中间点, 终点] -> [N+2, 2]
                full_coords = torch.cat([
                    start_xy.unsqueeze(0),        # [1, 2]
                    weighted_coords.t(),          # [N, 2]
                    goal_xy.unsqueeze(0)          # [1, 2]
                ], dim=0)  # [N+2, 2]

                # 3. 隔点计算速度向量：x_dot_i = (x_i+1 - x_i-1) / 2dt, y_dot_i = (y_i+1 - y_i-1) / 2dt, dt = 1
                # 对于中间的N个点，计算每个点的速度向量
                if full_coords.shape[0] >= 3:  # 至少需要3个点才能计算中间点的速度
                    x_coords = full_coords[:, 0]  # [N+2]
                    y_coords = full_coords[:, 1]  # [N+2]
                    
                    # 计算中间N个点的速度向量
                    x_dot = (x_coords[2:] - x_coords[:-2]) / 2.0  # [N]
                    y_dot = (y_coords[2:] - y_coords[:-2]) / 2.0  # [N]
                    
                    # 4. 归一化速度向量
                    velocity_norms = torch.sqrt(x_dot**2 + y_dot**2)  # [N]
                    
                    # 检查速度向量是否合理
                    if torch.any(torch.isnan(velocity_norms)) or torch.any(torch.isinf(velocity_norms)):
                        print(f"Warning: NaN or Inf in velocity norms")
                        continue
                    
                    # 防止除零，并限制极小值
                    velocity_norms = torch.clamp(velocity_norms, min=1e-6)
                    x_dot_normalized = x_dot / velocity_norms  # [N]
                    y_dot_normalized = y_dot / velocity_norms  # [N]
                    
                    # 检查归一化后的速度向量
                    if torch.any(torch.isnan(x_dot_normalized)) or torch.any(torch.isnan(y_dot_normalized)):
                        print(f"Warning: NaN in normalized velocity vectors")
                        continue
                    
                    # 5. 计算角度对应的单位向量
                    cos_theta = torch.cos(pred_angles)  # [N]
                    sin_theta = torch.sin(pred_angles)  # [N]
                    
                    # 6. 计算损失：L_angle = mse((cos(theta), sin(theta)), (x_dot_normalized, y_dot_normalized))
                    angle_vector = torch.stack([cos_theta, sin_theta], dim=1)  # [N, 2]
                    velocity_vector = torch.stack([x_dot_normalized, y_dot_normalized], dim=1)  # [N, 2]
                    
                    angle_loss = F.mse_loss(angle_vector, velocity_vector)
                    
                    # 检查角度一致性损失是否异常
                    if torch.isnan(angle_loss) or torch.isinf(angle_loss) or angle_loss.item() > 100:
                        print(f"Warning: Abnormal angle consistency loss: {angle_loss.item()}")
                        print(f"velocity_norms range: [{velocity_norms.min().item():.6f}, {velocity_norms.max().item():.6f}]")
                        print(f"pred_angles range: [{pred_angles.min().item():.3f}, {pred_angles.max().item():.3f}]")
                        continue
                    
                    total_loss += angle_loss * loss_weights['angle']  # 角度一致性损失(L_angle)
                    loss_count += 1

    # 确保返回张量类型的损失
    if loss_count == 0:
        # 如果没有计算任何损失，返回一个零损失张量
        total_loss = torch.tensor(0.0, requires_grad=True, device=predVals.device)
    
    # 返回总正确数而不是平均准确率，让调用者计算最终准确率
    return total_loss, n_correct, total_samples, (total_positive, total_negative, correct_positive, correct_negative)

def train_epoch(model, trainingData, optimizer, device, epoch=0, stage=1):
    """
    单轮训练函数
    stage: 1 为第一阶段（训练除correctionPred外的所有参数），2 为第二阶段（只训练correctionPred）
    """
    if stage == 1:
        # 第一阶段：训练除correctionPred外的所有参数
        model.train()  # 设置模型为训练模式：启用dropout和batch normalization
        # model.eval() 
        # model.correctionPred.train()
    else:
        # 第二阶段：只训练correctionPred
        # model.train()  # 设置模型为训练模式：启用dropout和batch normalization
        model.eval() 
        model.correctionPred.train()
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
            batch['trajectory'].to(device),  # 轨迹点：[N, 3]
            stage=stage  # 传递阶段信息
        )
        
        # 在反向传播前检查损失值的合理性
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN or Inf loss detected, skipping batch")
            continue
        
        # 对于过大的损失值，进行缩放以保持梯度稳定性
        if loss.item() > 0.8:  # 适当提高阈值，避免过度抑制学习
            print(f"Warning: Large loss detected ({loss.item():.4f}), applying scaling")
            loss = loss * 0.5  # 温和缩放，保持学习能力
        
        loss.backward()  # 反向传播：计算梯度
        
        # 在梯度裁剪前检查各个损失组件
        if loss.item() > 10.0:  # 如果损失异常大，输出调试信息
            print(f"Warning: Large loss detected: {loss.item():.4f}")
        
        # 梯度裁剪：防止梯度爆炸 - 平衡稳定性和学习效率
        max_grad_norm = 2.0  # 适度控制，平衡稳定性和学习速度
        
        # 进行梯度裁剪，返回的是裁剪前的原始梯度范数
        original_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, norm_type=2)
        
        # 对patch embedding层进行更精细的梯度检查和处理
        patch_embedding_grad_norms = []
        sensitive_layers = {4, 8, 12, 15}  # 已知的敏感层索引
        
        for name, param in model.named_parameters():
            if 'to_patch_embedding' in name and param.grad is not None:
                param_grad_norm = param.grad.data.norm(2).item()
                patch_embedding_grad_norms.append((name, param_grad_norm))
                
                # 从参数名中提取层索引（如果包含数字）
                layer_idx = None
                if '.' in name:
                    parts = name.split('.')
                    for part in parts:
                        if part.isdigit():
                            layer_idx = int(part)
                            break
                
                # 对敏感层使用更强的梯度控制
                if layer_idx in sensitive_layers:
                    if param_grad_norm > 0.8:  # 敏感层更严格的阈值
                        param.grad.data *= 0.3  # 更强的缩放
                elif param_grad_norm > 1.0:  # 其他层的标准处理
                    param.grad.data *= 0.5
        
        # 判断是否进行了裁剪
        was_clipped = original_grad_norm > max_grad_norm
        
        # 如果梯度范数异常大，输出详细调试信息
        if original_grad_norm > 1.8:  # 进一步降低阈值，更早发现问题
            print(f"Warning: Large gradient norm detected: {original_grad_norm:.2f}")
            print(f"  Batch loss: {loss.item():.6f}")
            print(f"  Stage: {stage}")
            
            # 检查各个参数的梯度
            max_grad = 0.0
            max_param_name = ""
            gradient_info = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_grad_norm = param.grad.data.norm(2).item()
                    if param_grad_norm > 0.5:  # 收集所有较大的梯度
                        gradient_info.append((name, param_grad_norm))
                    if param_grad_norm > max_grad:
                        max_grad = param_grad_norm
                        max_param_name = name
            
            print(f"  Largest gradient: {max_grad:.4f} in {max_param_name}")
            # 显示前5个最大的梯度
            gradient_info.sort(key=lambda x: x[1], reverse=True)
            print(f"  Top gradients:")
            for name, grad_norm in gradient_info[:5]:
                print(f"    {grad_norm:.4f} - {name}")
            print()

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
        
        # 计算patch embedding层的平均梯度
        avg_patch_grad = 0.0
        if patch_embedding_grad_norms:
            avg_patch_grad = sum([grad for _, grad in patch_embedding_grad_norms]) / len(patch_embedding_grad_norms)
        
        stage_info = "S1" if stage == 1 else "S2"  # 显示训练阶段
        pbar.set_postfix({
            'Stage': stage_info,
            'Loss': f'{loss.item():.4f}',
            'GradNorm': grad_info,
            'PatchGrad': f'{avg_patch_grad:.2f}',  # 显示patch embedding层平均梯度
            'LR': f'{optimizer._optimizer.param_groups[0]["lr"]:.2e}'
        })
    
    return total_loss, total_n_correct, total_samples, epoch_stats  # 返回整个epoch的统计结果


def eval_epoch(model, validationData, device, stage=1):
    """
    单轮评估函数
    stage: 1 为第一阶段（只计算分类损失），2 为第二阶段（只计算回归损失）
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
                batch['trajectory'].to(device),  # 轨迹点：[N, 3]
                stage=stage  # 传递阶段信息
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

def print_model_parameters(model, stage_info=""):
    """
    打印模型参数的训练状态
    """
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

def load_stage1_checkpoint(model, checkpoint_path, device):
    """
    加载第一阶段的模型检查点
    """
    if not osp.exists(checkpoint_path):
        raise ValueError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading stage 1 checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 检查检查点格式
    if 'state_dict' not in checkpoint:
        raise ValueError("Invalid checkpoint format: 'state_dict' not found")
    
    # 加载模型状态
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    
    # 返回检查点信息
    stage = checkpoint.get('stage', 1)
    epoch = checkpoint.get('epoch', -1)
    print(f"Loaded checkpoint: Stage {stage}, Epoch {epoch}")
    print("Stage 1 parameters loaded successfully, ready for stage 2 training")
    
    return checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    parser.add_argument('--batchSize', help="Batch size per GPU", required=True, type=int)  # 添加批次大小参数
    parser.add_argument('--env_list', help="Directory with training and validation data for Maze", default=None) 
    parser.add_argument('--dataFolder', help="Directory with training and validation data for Maze", default=None)  # 添加数据文件夹参数
    parser.add_argument('--fileDir', help="Directory to save training Data")  # 添加训练数据保存目录参数
    parser.add_argument('--load_stage1_model', help="Path to stage1 model checkpoint to load and start stage2 training", default=None)  # 加载第一阶段的模型参数，直接开始第二阶段训练
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
    
    # 双阶段训练配置
    stage1_epochs = 70  # 第一阶段训练轮数：训练除correctionPred外的所有参数
    stage2_epochs = 80  # 第二阶段训练轮数：只训练correctionPred参数
    total_epochs = stage1_epochs + stage2_epochs
    
    # 初始化数据记录变量
    # Define the optimizer
    # TODO: What does these parameters do ???
    # optimizer = Optim.ScheduledOptim(  # 创建带有学习率调度的优化器
    #     # optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-9),  # Adam优化器：动量参数(0.9, 0.98)，数值稳定性参数1e-9
    #     optim.Adam(filter(lambda p: p.requires_grad, transformer.parameters()), # 只优化需要梯度的参数
    #                betas=(0.9, 0.98), eps=1e-9),  # Adam优化器：动量参数(0.9, 0.98)，数值稳定性参数1e-9
    #     lr_mul = 0.1,  # 学习率乘数：降低到0.1
    #     d_model = 512,  # 模型维度：用于学习率计算
    #     n_warmup_steps = 3200  # 预热步数：3200步
    # )

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
    # n_epochs = 100  # 设置训练轮数：100轮
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
    
    # 检查是否需要加载第一阶段的模型并直接开始第二阶段训练
    start_stage = 1
    checkpoint = None  # 初始化 checkpoint 变量
    
    if args.load_stage1_model:
        # 加载第一阶段的检查点并直接开始第二阶段训练
        print(f"Loading stage 1 checkpoint: {args.load_stage1_model}")
        checkpoint = load_stage1_checkpoint(transformer, args.load_stage1_model, device)
        start_stage = 2  # 直接开始第二阶段训练
        print("Will skip stage 1 and directly start stage 2 training")
    
    # 第一阶段：冻结correctionPred，训练其他参数
    if start_stage == 1:
        print("=== Stage 1: Training all parameters except correctionPred ===")
        for param in transformer.parameters():
            param.requires_grad = True
        for param in transformer.correctionPred.parameters():
            param.requires_grad = False
        
        # 打印参数状态
        print_model_parameters(transformer, "Stage 1")
        
        # 第一阶段优化器
        stage1_optimizer = Optim.ScheduledOptim(
            optim.Adam(filter(lambda p: p.requires_grad, transformer.parameters()),
                       betas=(0.9, 0.98), eps=1e-9),
            lr_mul = 0.1,
            d_model = 512,
            n_warmup_steps = 3200
        )
        
        # 第一阶段训练
        for n in range(stage1_epochs):
            train_total_loss, train_n_correct, train_samples, train_stats = train_epoch(
                transformer, trainingData, stage1_optimizer, device, epoch=n, stage=1
            )
            val_total_loss, val_n_correct, val_samples, val_stats = eval_epoch(transformer, validationData, device, stage=1)
            
            # 计算训练和验证准确率
            train_accuracy = train_n_correct / train_samples if train_samples > 0 else 0.0
            val_accuracy = val_n_correct / val_samples if val_samples > 0 else 0.0
            
            print(f"Stage 1 - Epoch {n} Train Loss: {train_total_loss:.4f}")
            print(f"Stage 1 - Epoch {n} Val Loss: {val_total_loss:.4f}")
            print(f"Stage 1 - Epoch {n} Train Accuracy: {train_accuracy:.4f} ({train_n_correct}/{train_samples})")
            print(f"Stage 1 - Epoch {n} Val Accuracy: {val_accuracy:.4f} ({val_n_correct}/{val_samples})")
            
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
            train_loss.append(train_total_loss)
            val_loss.append(val_total_loss)
            train_n_correct_list.append(train_accuracy)
            val_n_correct_list.append(val_accuracy)

            if (n+1)%5==0:
                if isinstance(transformer, nn.DataParallel):
                    state_dict = transformer.module.state_dict()
                else:
                    state_dict = transformer.state_dict()
                states = {
                    'state_dict': state_dict,
                    'optimizer': stage1_optimizer._optimizer.state_dict(),
                    'torch_seed': torch_seed,
                    'stage': 1,
                    'epoch': n
                }
                torch.save(states, osp.join(trainDataFolder, f'stage1_model_epoch_{n}.pkl'))
            
            pickle.dump(
                {
                    'trainLoss': train_loss,
                    'valLoss':val_loss,
                    'trainNCorrect':train_n_correct_list,
                    'valNCorrect':val_n_correct_list
                }, 
                open(osp.join(trainDataFolder, 'stage1_progress.pkl'), 'wb')
            )
            writer.add_scalar('Stage1/Loss/train', train_total_loss, n)
            writer.add_scalar('Stage1/Loss/test', val_total_loss, n)
            writer.add_scalar('Stage1/Accuracy/train', train_accuracy, n)
            writer.add_scalar('Stage1/Accuracy/test', val_accuracy, n)
    
    # 第二阶段：冻结其他参数，只训练correctionPred
    print("=== Stage 2: Training only correctionPred parameters ===")
    # for param in transformer.parameters():
    #     param.requires_grad = False
    # for param in transformer.correctionPred.parameters():
    #     param.requires_grad = True
    transformer.train()  # 设置模型为训练模式
    
    # 打印参数状态
    print_model_parameters(transformer, "Stage 2")
    
    # 第二阶段优化器
    stage2_optimizer = Optim.ScheduledOptim(
        optim.Adam(filter(lambda p: p.requires_grad, transformer.parameters()),
                   betas=(0.9, 0.98), eps=1e-9),
        lr_mul = 0.1,
        d_model = 512,
        n_warmup_steps = 3200
    )
    
    # 第二阶段训练
    for n in range(stage2_epochs):
        train_total_loss, train_n_correct, train_samples, train_stats = train_epoch(
            transformer, trainingData, stage2_optimizer, device, epoch=n, stage=2
        )
        val_total_loss, val_n_correct, val_samples, val_stats = eval_epoch(transformer, validationData, device, stage=2)
        
        # 计算训练和验证准确率
        train_accuracy = train_n_correct / train_samples if train_samples > 0 else 0.0
        val_accuracy = val_n_correct / val_samples if val_samples > 0 else 0.0
        
        print(f"Stage 2 - Epoch {n} Train Loss: {train_total_loss:.4f}")
        print(f"Stage 2 - Epoch {n} Val Loss: {val_total_loss:.4f}")
        print(f"Stage 2 - Epoch {n} Train Accuracy: {train_accuracy:.4f} ({train_n_correct}/{train_samples})")
        print(f"Stage 2 - Epoch {n} Val Accuracy: {val_accuracy:.4f} ({val_n_correct}/{val_samples})")
        
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

        # Log data (继续累加到之前的数据中)
        train_loss.append(train_total_loss)
        val_loss.append(val_total_loss)
        train_n_correct_list.append(train_accuracy)
        val_n_correct_list.append(val_accuracy)

        if (n+1)%5==0:
            if isinstance(transformer, nn.DataParallel):
                state_dict = transformer.module.state_dict()
            else:
                state_dict = transformer.state_dict()
            states = {
                'state_dict': state_dict,
                'optimizer': stage2_optimizer._optimizer.state_dict(),
                'torch_seed': torch_seed,
                'stage': 2,
                'epoch': n
            }
            torch.save(states, osp.join(trainDataFolder, f'stage2_model_epoch_{n}.pkl'))
        
        pickle.dump(
            {
                'trainLoss': train_loss,
                'valLoss':val_loss,
                'trainNCorrect':train_n_correct_list,
                'valNCorrect':val_n_correct_list
            }, 
            open(osp.join(trainDataFolder, 'stage2_progress.pkl'), 'wb')
        )
        writer.add_scalar('Stage2/Loss/train', train_total_loss, stage1_epochs + n)
        writer.add_scalar('Stage2/Loss/test', val_total_loss, stage1_epochs + n)
        writer.add_scalar('Stage2/Accuracy/train', train_accuracy, stage1_epochs + n)
        writer.add_scalar('Stage2/Accuracy/test', val_accuracy, stage1_epochs + n)
    
    # 保存最终的完整模型
    if isinstance(transformer, nn.DataParallel):
        state_dict = transformer.module.state_dict()
    else:
        state_dict = transformer.state_dict()
    final_states = {
        'state_dict': state_dict,
        'torch_seed': torch_seed,
        'total_epochs': total_epochs,
        'stage1_epochs': stage1_epochs,
        'stage2_epochs': stage2_epochs
    }
    
    # 根据训练阶段保存优化器状态
    if start_stage == 1:
        # 如果执行了两个阶段的训练，保存两个阶段的优化器状态
        final_states['stage1_optimizer'] = stage1_optimizer._optimizer.state_dict()
        final_states['stage2_optimizer'] = stage2_optimizer._optimizer.state_dict()
    else:
        # 如果从第二阶段开始训练，从加载的检查点中读取第一阶段的优化器状态
        if 'optimizer' in checkpoint:
            final_states['stage1_optimizer'] = checkpoint['optimizer']
        final_states['stage2_optimizer'] = stage2_optimizer._optimizer.state_dict()
    
    torch.save(final_states, osp.join(trainDataFolder, 'final_model.pkl'))
    
    # 保存最终的训练进度
    pickle.dump(
        {
            'trainLoss': train_loss,
            'valLoss':val_loss,
            'trainNCorrect':train_n_correct_list,
            'valNCorrect':val_n_correct_list,
            'stage1_epochs': stage1_epochs,
            'stage2_epochs': stage2_epochs
        }, 
        open(osp.join(trainDataFolder, 'final_progress.pkl'), 'wb')
    )

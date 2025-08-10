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
    
    # 各损失的权重
    # loss_weights = {
    #     'classification': 4e-1, # 分类损失权重(L_ce)
    #     'regression': 2e-2,     # 回归损失权重(L_mse)
    #     'uniformity': 1e1,      # 轨迹点分布均匀性损失权重(L_uni)
    #     'angle': 3e0,          # 角度一致性损失权重(L_angle)
    # }

    loss_weights = {
        'classification': 1e-2, # 分类损失权重(L_ce) - 提高权重避免梯度消失
        'regression': 1e-3,     # 回归损失权重(L_mse) - 提高权重
        'uniformity': 1e-3,     # 轨迹点分布均匀性损失权重(L_uni) - 提高权重
        'angle': 1e-6,          # 角度一致性损失权重(L_angle) - 用极小权重调试
    }

    # 用于统计标签分布
    total_positive = 0
    total_negative = 0
    correct_positive = 0
    correct_negative = 0
    
    for i in range(batch_size):
        anchorPoint = anchorPoints[i]  # [num_layers, max_anchors]
        trueLabel = trueLabels[i]  # [num_layers, max_anchors]
        
        # 损失1：锚点分类交叉熵损失(L_ce) - 使用张量并行计算
        # 由于DataLoader已经保证了固定尺寸，可以直接进行批量处理
        if loss_weights['classification'] > 0:
            num_anchor_rows = anchorPoint.shape[0]  # 总行数 = 2 * num_trajectory_points
            
            # 检查锚点索引是否在有效范围内
            valid_anchor_mask = (anchorPoint >= 0) & (anchorPoint < num_tokens) & (trueLabel != -1)
            if not valid_anchor_mask.any():
                # 如果没有有效的锚点，跳过这个样本
                continue
            
            # 批量处理所有时间步的损失计算
            step_indices = torch.arange(num_anchor_rows, device=predVals.device) % output_dim
            
            # 获取所有时间步的预测 [num_anchor_rows, num_tokens]
            all_step_preds = predVals[i, :, step_indices].t()  # [num_anchor_rows, num_tokens]
            
            # 为每个时间步创建全图标签张量 [num_anchor_rows, num_tokens] - 完全并行化
            full_labels_batch = torch.zeros(num_anchor_rows, num_tokens, device=predVals.device)
            
            # 创建有效掩码 [num_anchor_rows, MAX_POSITIVE_ANCHORS]
            valid_mask = (anchorPoint != -1) & (trueLabel != -1) & (anchorPoint < num_tokens)
            
            # 使用高级索引批量设置所有标签
            if valid_mask.any():
                # 获取所有有效位置的索引
                step_indices, anchor_indices = torch.where(valid_mask)  # 有效位置的(step, anchor)索引对
                
                if len(step_indices) > 0:
                    # 批量获取有效的锚点索引和标签值
                    valid_anchor_positions = anchorPoint[step_indices, anchor_indices]  # 锚点在tokens中的位置
                    valid_label_values = trueLabel[step_indices, anchor_indices].float()  # 对应的标签值
                    
                    # 批量设置标签值
                    full_labels_batch[step_indices, valid_anchor_positions] = valid_label_values
                    
                    # 按行归一化为概率分布
                    row_sums = full_labels_batch.sum(dim=1, keepdim=True)  # [num_anchor_rows, 1]
                    nonzero_mask = (row_sums > 0).squeeze(1)  # [num_anchor_rows]
                    if nonzero_mask.any():
                        full_labels_batch[nonzero_mask] = full_labels_batch[nonzero_mask] / row_sums[nonzero_mask]
            
            # 数值稳定性处理
            all_step_preds_stable = torch.clamp(all_step_preds, min=1e-8, max=1.0-1e-8)
            full_labels_stable = torch.clamp(full_labels_batch, min=1e-8, max=1.0-1e-8)
            
            # 重新归一化以确保概率分布有效
            label_sums = full_labels_stable.sum(dim=1, keepdim=True)
            label_sums = torch.clamp(label_sums, min=1e-8)
            full_labels_stable = full_labels_stable / label_sums
            
            # 批量计算KL散度损失 [num_anchor_rows]
            kl_losses = F.kl_div(
                torch.log(all_step_preds_stable), 
                full_labels_stable, 
                reduction='none'
            ).sum(dim=1)  # 对每个时间步求和
            
            # 过滤掉异常值并求平均
            valid_loss_mask = ~(torch.isnan(kl_losses) | torch.isinf(kl_losses) | (kl_losses > 50))
            if valid_loss_mask.any():
                classification_loss = kl_losses[valid_loss_mask].mean()
                total_loss += classification_loss * loss_weights['classification']
                loss_count += 1
            else:
                print(f"Warning: All classification losses are abnormal, skipping")
            
            # 批量计算准确率 - 完全并行化处理
            threshold = max(0.01, 1.0 / num_tokens * 2)
            
            # 创建有效锚点和标签的掩码 [num_anchor_rows, MAX_POSITIVE_ANCHORS]
            valid_anchor_mask = (anchorPoint != -1) & (trueLabel != -1) & (anchorPoint < num_tokens)
            
            if valid_anchor_mask.any():
                # 批量收集所有有效的预测值和标签
                all_valid_preds = []
                all_valid_labels = []
                all_valid_positive_labels = []
                all_valid_negative_labels = []
                
                # 使用高级索引批量提取有效的预测和标签
                for step in range(num_anchor_rows):
                    step_mask = valid_anchor_mask[step]  # [MAX_POSITIVE_ANCHORS]
                    if step_mask.any():
                        valid_anchors = anchorPoint[step][step_mask]  # 有效锚点索引
                        valid_labels = trueLabel[step][step_mask]     # 有效标签
                        step_pred = all_step_preds[step]             # [num_tokens]
                        
                        # 提取对应锚点的预测值
                        selected_preds = step_pred[valid_anchors]    # 有效锚点的预测值
                        
                        all_valid_preds.append(selected_preds)
                        all_valid_labels.append(valid_labels)
                        all_valid_positive_labels.append(valid_labels == 1)
                        all_valid_negative_labels.append(valid_labels == 0)
                
                if all_valid_preds:
                    # 将所有有效数据拼接成单个张量进行批量处理
                    batch_valid_preds = torch.cat(all_valid_preds)      # [total_valid_samples]
                    batch_valid_labels = torch.cat(all_valid_labels)    # [total_valid_samples]
                    batch_positive_mask = torch.cat(all_valid_positive_labels)  # [total_valid_samples]
                    batch_negative_mask = torch.cat(all_valid_negative_labels)  # [total_valid_samples]
                    
                    # 批量计算预测类别
                    batch_class_preds = (batch_valid_preds > threshold).long()  # [total_valid_samples]
                    
                    # 批量统计正确预测数
                    batch_correct = batch_class_preds.eq(batch_valid_labels.long())  # [total_valid_samples]
                    n_correct += batch_correct.sum().item()
                    total_samples += len(batch_valid_labels)
                    
                    # 批量统计标签分布
                    total_positive += batch_positive_mask.sum().item()
                    total_negative += batch_negative_mask.sum().item()
                    
                    # 批量统计各类别的正确预测
                    correct_positive_mask = (batch_class_preds == 1) & batch_positive_mask
                    correct_negative_mask = (batch_class_preds == 0) & batch_negative_mask
                    correct_positive += correct_positive_mask.sum().item()
                    correct_negative += correct_negative_mask.sum().item()
            
        # 损失2：轨迹回归损失(L_mse) - 使用张量并行计算
        # 不仅计算位置的回归损失，还计算角度的回归损失，但是角度需要特殊处理
        # 确定处理的时间步数
        if loss_weights['regression'] > 0:
            steps_to_process = min(output_dim, anchorPoint.shape[0])
            if steps_to_process > 0:
                # 将hashTable转换为张量以便并行计算
                hash_table_tensor = torch.tensor(hashTable, device=predVals.device, dtype=torch.float32, requires_grad=False)  # [num_tokens, 2]
                
                # predVals已经经过softmax归一化，检查是否包含异常值
                pred_probs = predVals[i, :, :steps_to_process]  # [num_tokens, steps_to_process]
                
                # 数值稳定性检查
                if torch.any(torch.isnan(pred_probs)) or torch.any(torch.isinf(pred_probs)):
                    print(f"Warning: NaN or Inf in pred_probs, skipping regression loss")
                    continue
                
                # 确保概率值在合理范围内，但不重新归一化（保持模型的输出分布）
                pred_probs = torch.clamp(pred_probs, min=1e-8, max=1.0-1e-8)
                
                # 对每个时间步分别计算加权坐标，只对概率大于阈值的锚点进行加权计算
                weighted_coords_list = []
                num_tokens = pred_probs.shape[0]
                
                for step in range(steps_to_process):
                    step_prob = pred_probs[:, step]  # [num_tokens]
                    
                    # 检查当前时间步的概率是否有异常值
                    if torch.any(torch.isnan(step_prob)) or torch.any(torch.isinf(step_prob)):
                        print(f"Warning: NaN or Inf in step_prob at step {step}, using uniform distribution")
                        step_prob = torch.ones_like(step_prob) / num_tokens
                    
                    # 动态阈值，至少为平均概率的2倍
                    threshold = max(0.01, 1.0 / num_tokens * 2)
                    
                    # 只对概率大于阈值的锚点进行加权计算
                    valid_mask = step_prob > threshold
                    valid_indices = torch.where(valid_mask)[0]
                    
                    if len(valid_indices) > 0:
                        # 计算有效锚点的概率和坐标加权
                        valid_probs = step_prob[valid_indices]
                        total_prob = valid_probs.sum()
                        
                        # 归一化概率并计算加权坐标
                        if total_prob > 1e-8:
                            normalized_probs = valid_probs / total_prob
                            weighted_x = torch.sum(hash_table_tensor[valid_indices, 0] * normalized_probs)
                            weighted_y = torch.sum(hash_table_tensor[valid_indices, 1] * normalized_probs)
                        else:
                            # 如果总概率太小，使用所有锚点的均匀权重
                            weighted_x = hash_table_tensor[:, 0].mean()
                            weighted_y = hash_table_tensor[:, 1].mean()
                    else:
                        # 如果没有锚点超过阈值，使用所有锚点的加权平均
                        weighted_x = torch.sum(hash_table_tensor[:, 0] * step_prob)
                        weighted_y = torch.sum(hash_table_tensor[:, 1] * step_prob)
                    
                    # 检查加权坐标是否异常
                    if torch.isnan(weighted_x) or torch.isnan(weighted_y) or torch.isinf(weighted_x) or torch.isinf(weighted_y):
                        print(f"Warning: NaN or Inf in weighted coords at step {step}, using center coordinates")
                        weighted_x = hash_table_tensor[:, 0].mean()
                        weighted_y = hash_table_tensor[:, 1].mean()
                    
                    weighted_coords_list.append(torch.stack([weighted_x, weighted_y]))
                
                # 将列表转换为张量 [2, steps_to_process]
                weighted_coords = torch.stack(weighted_coords_list, dim=1)

                # 依据概率进行修正量的计算 - 使用并行张量计算
                # 提取修正值 [num_tokens, 2, steps_to_process]
                offset_coords = correctionVals[i, :, :2, :steps_to_process]  # x, y 偏移量
                
                # 检查修正值是否有异常
                if torch.any(torch.isnan(offset_coords)) or torch.any(torch.isinf(offset_coords)):
                    print(f"Warning: NaN or Inf in offset_coords, setting to zero")
                    offset_coords = torch.zeros_like(offset_coords)
                
                # 扩展概率分布以匹配偏移量维度 [num_tokens, steps_to_process] -> [num_tokens, 1, steps_to_process]
                expanded_probs_correction = pred_probs.unsqueeze(1)  # [num_tokens, 1, steps_to_process]
                
                # 并行计算所有时间步的加权偏移量 [2, steps_to_process]
                weighted_offsets = torch.sum(expanded_probs_correction * offset_coords, dim=0)
                
                # 检查加权偏移量是否异常
                if torch.any(torch.isnan(weighted_offsets)) or torch.any(torch.isinf(weighted_offsets)):
                    print(f"Warning: NaN or Inf in weighted_offsets, setting to zero")
                    weighted_offsets = torch.zeros_like(weighted_offsets)
                
                # 计算修正后的坐标：基础坐标 + 加权偏移量 [2, steps_to_process]
                corrected_coords_raw = weighted_coords + weighted_offsets
                
                # 检查修正后的坐标是否异常
                if torch.any(torch.isnan(corrected_coords_raw)) or torch.any(torch.isinf(corrected_coords_raw)):
                    print(f"Warning: NaN or Inf in corrected_coords_raw, using only weighted_coords")
                    corrected_coords_raw = weighted_coords
                
                # 转置以匹配预期格式 [steps_to_process, 2]
                corrected_coords = corrected_coords_raw.t()
                
                # 将像素坐标映射回实际坐标系
                corrected_coords = -5.0 + corrected_coords * 0.1
                
                # 提取真实轨迹坐标和角度 [steps_to_process, 3]
                true_trajectory = trajectory[i, 1:steps_to_process+1, :]
                true_coords = true_trajectory[:, :2]  # 位置坐标 [steps_to_process, 2]
                true_angles = true_trajectory[:, 2]   # 角度信息 [steps_to_process]
                
                # 计算修正后位置的MSE损失
                coord_loss = F.mse_loss(corrected_coords, true_coords)
                
                # 检查坐标损失是否异常
                if torch.isnan(coord_loss) or torch.isinf(coord_loss) or coord_loss.item() > 1000:
                    print(f"Warning: Abnormal coord loss: {coord_loss.item()}")
                    print(f"corrected_coords range: [{corrected_coords.min().item():.3f}, {corrected_coords.max().item():.3f}]")
                    print(f"true_coords range: [{true_coords.min().item():.3f}, {true_coords.max().item():.3f}]")
                    coord_loss = torch.tensor(0.0, requires_grad=True, device=predVals.device)
                
                # 计算角度的加权平均预测
                # 使用模型直接预测的角度信息（从correctionVal中获取）
                # correctionVal: [batch_size, num_tokens, 3, output_dim]，其中第3个维度的第2个元素是角度预测
                
                # 为了处理角度的周期性，我们需要特殊处理
                # 将角度转换为复数表示：e^(i*θ) = cos(θ) + i*sin(θ)
                true_angles_complex = torch.complex(torch.cos(true_angles), torch.sin(true_angles))  # [steps_to_process]
                
                # 提取所有时间步的角度预测 [num_tokens, steps_to_process]
                step_angle_preds = correctionVals[i, :, 2, :steps_to_process]  # sigmoid输出，范围[0,1]
                
                # 确保角度预测在合理范围内
                step_angle_preds = torch.clamp(step_angle_preds, min=1e-7, max=1.0-1e-7)
                
                # 将sigmoid输出[0,1]转换为角度范围[-π,π] [num_tokens, steps_to_process]
                step_angle_preds_rad = step_angle_preds * 2 * np.pi - np.pi
                
                # 转换为复数表示 [num_tokens, steps_to_process]
                step_angles_complex = torch.complex(torch.cos(step_angle_preds_rad), torch.sin(step_angle_preds_rad))
                
                # 对每个时间步分别计算加权角度，只对概率大于阈值的锚点进行加权
                pred_angles_complex_list = []
                num_tokens = pred_probs.shape[0]
                
                for step in range(steps_to_process):
                    step_prob = pred_probs[:, step]  # [num_tokens]
                    step_complex = step_angles_complex[:, step]  # [num_tokens]
                    
                    # 动态阈值，至少为平均概率的2倍
                    threshold = max(0.01, 1.0 / num_tokens * 2)
                    
                    # 只对概率大于阈值的锚点进行加权计算
                    valid_mask = step_prob > threshold
                    valid_indices = torch.where(valid_mask)[0]
                    
                    if len(valid_indices) > 0:
                        # 计算有效锚点的概率和角度加权
                        valid_probs = step_prob[valid_indices]
                        total_prob = valid_probs.sum()
                        
                        # 归一化概率并计算加权角度
                        if total_prob > 0:
                            normalized_probs = valid_probs / total_prob
                            pred_angle_complex = torch.sum(step_complex[valid_indices] * normalized_probs)
                        else:
                            # 如果总概率为0，使用所有锚点的加权平均
                            pred_angle_complex = torch.sum(step_complex * step_prob)
                    else:
                        # 如果没有锚点超过阈值，使用所有锚点的加权平均
                        pred_angle_complex = torch.sum(step_complex * step_prob)
                    
                    pred_angles_complex_list.append(pred_angle_complex)
                
                # 将列表转换为张量 [steps_to_process]
                pred_angles_complex = torch.stack(pred_angles_complex_list)
                
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
                print(f"Debug: Starting angle loss computation for sample {i}, steps_to_process={steps_to_process}")
                
                # 1. 先获取预测轨迹坐标（从哈希表和概率分布计算）
                hash_table_tensor = torch.tensor(hashTable, device=predVals.device, dtype=torch.float32, requires_grad=False)  # [num_tokens, 2]
                pred_probs = predVals[i, :, :steps_to_process]  # [num_tokens, steps_to_process]
                
                print(f"Debug: pred_probs shape: {pred_probs.shape}, range: [{pred_probs.min().item():.6f}, {pred_probs.max().item():.6f}]")
                
                # 数值稳定性检查
                if torch.any(torch.isnan(pred_probs)) or torch.any(torch.isinf(pred_probs)):
                    print(f"Warning: NaN or Inf in pred_probs for angle loss, skipping")
                    continue
                
                # 确保概率值在合理范围内，但不重新归一化（保持模型的输出分布）
                pred_probs = torch.clamp(pred_probs, min=1e-8, max=1.0-1e-8)
                
                # 对每个时间步分别计算加权坐标，只对概率大于阈值的锚点进行加权计算
                weighted_coords_list = []
                num_tokens = pred_probs.shape[0]
                
                for step in range(steps_to_process):
                    step_prob = pred_probs[:, step]  # [num_tokens]
                    
                    # 检查当前时间步的概率是否有异常值
                    if torch.any(torch.isnan(step_prob)) or torch.any(torch.isinf(step_prob)):
                        print(f"Warning: NaN or Inf in step_prob for angle loss at step {step}")
                        step_prob = torch.ones_like(step_prob) / num_tokens
                    
                    # 动态阈值，至少为平均概率的2倍
                    threshold = max(0.01, 1.0 / num_tokens * 2)
                    
                    # 只对概率大于阈值的锚点进行加权计算
                    valid_mask = step_prob > threshold
                    valid_indices = torch.where(valid_mask)[0]
                    
                    if len(valid_indices) > 0:
                        # 计算有效锚点的概率和坐标加权
                        valid_probs = step_prob[valid_indices]
                        total_prob = valid_probs.sum()
                        
                        # 归一化概率并计算加权坐标
                        if total_prob > 1e-8:
                            normalized_probs = valid_probs / total_prob
                            weighted_x = torch.sum(hash_table_tensor[valid_indices, 0] * normalized_probs)
                            weighted_y = torch.sum(hash_table_tensor[valid_indices, 1] * normalized_probs)
                        else:
                            # 如果总概率太小，使用所有锚点的均匀权重
                            weighted_x = hash_table_tensor[:, 0].mean()
                            weighted_y = hash_table_tensor[:, 1].mean()
                    else:
                        # 如果没有锚点超过阈值，使用所有锚点的加权平均
                        weighted_x = torch.sum(hash_table_tensor[:, 0] * step_prob)
                        weighted_y = torch.sum(hash_table_tensor[:, 1] * step_prob)
                    
                    # 检查加权坐标是否异常
                    if torch.isnan(weighted_x) or torch.isnan(weighted_y) or torch.isinf(weighted_x) or torch.isinf(weighted_y):
                        print(f"Warning: NaN or Inf in weighted coords for angle loss at step {step}")
                        weighted_x = hash_table_tensor[:, 0].mean()
                        weighted_y = hash_table_tensor[:, 1].mean()
                    
                    weighted_coords_list.append(torch.stack([weighted_x, weighted_y]))
                
                # 将列表转换为张量 [2, steps_to_process]
                weighted_coords = torch.stack(weighted_coords_list, dim=1)
                weighted_coords = -5.0 + weighted_coords * 0.1  # 映射到实际坐标系
                
                # 获取预测角度 - 对每个时间步分别计算，只对概率大于阈值的锚点进行加权
                # 提取所有时间步的角度预测 [num_tokens, steps_to_process]
                step_angle_preds_all = correctionVals[i, :, 2, :steps_to_process]  # sigmoid输出[0,1]
                
                # 检查角度预测是否有异常
                if torch.any(torch.isnan(step_angle_preds_all)) or torch.any(torch.isinf(step_angle_preds_all)):
                    print(f"Warning: NaN or Inf in angle predictions, setting to default")
                    step_angle_preds_all = torch.full_like(step_angle_preds_all, 0.5)  # 默认角度0
                
                # 确保角度预测在合理范围内
                step_angle_preds_all = torch.clamp(step_angle_preds_all, min=1e-6, max=1.0-1e-6)
                
                # 转换到[-π,π] [num_tokens, steps_to_process]
                step_angle_preds_rad_all = step_angle_preds_all * 2 * np.pi - np.pi
                
                # 对每个时间步分别计算加权角度，只对概率大于阈值的锚点进行加权
                pred_angles_list = []
                for step in range(steps_to_process):
                    step_prob = pred_probs[:, step]  # [num_tokens]
                    step_angle_preds_rad = step_angle_preds_rad_all[:, step]  # [num_tokens]
                    
                    # 动态阈值，至少为平均概率的2倍
                    threshold = max(0.01, 1.0 / num_tokens * 2)
                    
                    # 只对概率大于阈值的锚点进行加权计算
                    valid_mask = step_prob > threshold
                    valid_indices = torch.where(valid_mask)[0]
                    
                    if len(valid_indices) > 0:
                        # 计算有效锚点的概率和角度加权
                        valid_probs = step_prob[valid_indices]
                        total_prob = valid_probs.sum()
                        
                        # 归一化概率并计算加权角度
                        if total_prob > 0:
                            normalized_probs = valid_probs / total_prob
                            pred_angle = torch.sum(step_angle_preds_rad[valid_indices] * normalized_probs)
                        else:
                            # 如果总概率为0，使用所有锚点的加权平均
                            pred_angle = torch.sum(step_angle_preds_rad * step_prob)
                    else:
                        # 如果没有锚点超过阈值，使用所有锚点的加权平均
                        pred_angle = torch.sum(step_angle_preds_rad * step_prob)
                    
                    pred_angles_list.append(pred_angle)
                
                # 将列表转换为张量 [steps_to_process]
                pred_angles = torch.stack(pred_angles_list)
                
                # 2. 为预测轨迹补足起点和终点，从N个变成N+2个
                # 起点和终点坐标
                start_xy = start_state[i, :2]  # [2]
                goal_xy = goal_state[i, :2]   # [2]
                start_theta = start_state[i, 2]  # 标量
                goal_theta = goal_state[i, 2]   # 标量
                
                print(f"Debug: start_xy: {start_xy}, goal_xy: {goal_xy}")
                print(f"Debug: weighted_coords shape: {weighted_coords.shape}, range: [{weighted_coords.min().item():.6f}, {weighted_coords.max().item():.6f}]")
                
                # 拼接完整轨迹：[起点, 中间点, 终点] -> [N+2, 2]
                full_coords = torch.cat([
                    start_xy.unsqueeze(0),        # [1, 2]
                    weighted_coords.t(),          # [N, 2]
                    goal_xy.unsqueeze(0)          # [1, 2]
                ], dim=0)  # [N+2, 2]
                
                print(f"Debug: full_coords shape: {full_coords.shape}, range: [{full_coords.min().item():.6f}, {full_coords.max().item():.6f}]")

                # 3. 隔点计算速度向量：x_dot_i = (x_i+1 - x_i-1) / 2dt, y_dot_i = (y_i+1 - y_i-1) / 2dt, dt = 1
                # 对于中间的N个点，计算每个点的速度向量
                if full_coords.shape[0] >= 3:  # 至少需要3个点才能计算中间点的速度
                    x_coords = full_coords[:, 0]  # [N+2]
                    y_coords = full_coords[:, 1]  # [N+2]
                    
                    # 计算中间N个点的速度向量
                    x_dot = (x_coords[2:] - x_coords[:-2]) / 2.0  # [N]
                    y_dot = (y_coords[2:] - y_coords[:-2]) / 2.0  # [N]
                    
                    print(f"Debug: x_dot shape: {x_dot.shape}, range: [{x_dot.min().item():.6f}, {x_dot.max().item():.6f}]")
                    print(f"Debug: y_dot shape: {y_dot.shape}, range: [{y_dot.min().item():.6f}, {y_dot.max().item():.6f}]")
                    
                    # 4. 改进的角度一致性约束：避免归一化极小除数的问题
                    # 不直接归一化速度向量，而是使用向量内积的性质来计算角度一致性
                    
                    # 计算预测角度对应的单位向量
                    cos_theta = torch.cos(pred_angles)  # [N]
                    sin_theta = torch.sin(pred_angles)  # [N]
                    pred_unit_vectors = torch.stack([cos_theta, sin_theta], dim=1)  # [N, 2]
                    
                    print(f"Debug: pred_angles shape: {pred_angles.shape}, range: [{pred_angles.min().item():.6f}, {pred_angles.max().item():.6f}]")
                    print(f"Debug: pred_unit_vectors shape: {pred_unit_vectors.shape}, range: [{pred_unit_vectors.min().item():.6f}, {pred_unit_vectors.max().item():.6f}]")
                    
                    # 计算速度向量 [N, 2]
                    velocity_vectors = torch.stack([x_dot, y_dot], dim=1)  # [N, 2]
                    
                    print(f"Debug: velocity_vectors shape: {velocity_vectors.shape}, range: [{velocity_vectors.min().item():.6f}, {velocity_vectors.max().item():.6f}]")
                    
                    # 方法1：使用余弦相似度损失，避免直接归一化
                    # cos(angle_between_vectors) = (a · b) / (|a| * |b|)
                    # 我们希望 cos(angle) 接近 1，即向量方向一致
                    
                    # 计算向量内积 [N]
                    dot_products = torch.sum(pred_unit_vectors * velocity_vectors, dim=1)  # [N]
                    
                    print(f"Debug: dot_products shape: {dot_products.shape}, range: [{dot_products.min().item():.6f}, {dot_products.max().item():.6f}]")
                    
                    # 计算速度向量的模长 [N]
                    velocity_norms = torch.sqrt(x_dot**2 + y_dot**2)  # [N]
                    
                    print(f"Debug: velocity_norms shape: {velocity_norms.shape}, range: [{velocity_norms.min().item():.6f}, {velocity_norms.max().item():.6f}]")
                    
                    # 检查速度向量模长是否合理
                    if torch.any(torch.isnan(velocity_norms)) or torch.any(torch.isinf(velocity_norms)):
                        print(f"Warning: NaN or Inf in velocity norms")
                        continue
                    
                    # 使用数值稳定的方式处理小模长：
                    # 不是直接跳过，而是使用加权损失，对小模长的贡献降权
                    min_norm_threshold = 1e-3  # 最小模长阈值
                    
                    # 计算权重：模长越大，权重越大，但不为零
                    # weight = min(1.0, norm / min_threshold)^2
                    normalized_norms = velocity_norms / min_norm_threshold
                    weights = torch.clamp(normalized_norms, min=0.01, max=1.0) ** 2  # [N]
                    
                    print(f"Debug: weights shape: {weights.shape}, range: [{weights.min().item():.6f}, {weights.max().item():.6f}]")
                    
                    # 计算余弦相似度 = dot_product / (|pred| * |velocity|)
                    # 由于pred_unit_vectors已经是单位向量，|pred| = 1
                    cosine_similarities = dot_products / torch.clamp(velocity_norms, min=1e-8)  # [N]
                    
                    print(f"Debug: cosine_similarities shape: {cosine_similarities.shape}, range: [{cosine_similarities.min().item():.6f}, {cosine_similarities.max().item():.6f}]")
                    
                    # 检查余弦相似度是否合理
                    if torch.any(torch.isnan(cosine_similarities)) or torch.any(torch.isinf(cosine_similarities)):
                        print(f"Warning: NaN or Inf in cosine similarities")
                        continue
                    
                    # 角度一致性损失：希望余弦相似度接近1（角度差接近0）
                    # 使用 1 - cosine_similarity 作为损失
                    cosine_losses = 1.0 - cosine_similarities  # [N]
                    
                    print(f"Debug: cosine_losses shape: {cosine_losses.shape}, range: [{cosine_losses.min().item():.6f}, {cosine_losses.max().item():.6f}]")
                    
                    # 应用权重：小模长的损失贡献较小
                    weighted_cosine_losses = cosine_losses * weights  # [N]
                    
                    print(f"Debug: weighted_cosine_losses shape: {weighted_cosine_losses.shape}, range: [{weighted_cosine_losses.min().item():.6f}, {weighted_cosine_losses.max().item():.6f}]")
                    
                    # 计算最终的角度损失
                    angle_loss = weighted_cosine_losses.mean()
                    
                    print(f"Debug: angle_loss: {angle_loss.item():.6f}")
                    
                    # 检查角度一致性损失是否异常
                    if torch.isnan(angle_loss) or torch.isinf(angle_loss) or angle_loss.item() > 100:
                        print(f"Warning: Abnormal angle consistency loss: {angle_loss.item()}")
                        print(f"velocity_norms range: [{velocity_norms.min().item():.6f}, {velocity_norms.max().item():.6f}]")
                        print(f"cosine_similarities range: [{cosine_similarities.min().item():.6f}, {cosine_similarities.max().item():.6f}]")
                        continue
                    
                    # 检查角度损失的梯度是否会导致问题
                    angle_loss_contribution = angle_loss * loss_weights['angle']
                    print(f"Debug: angle_loss_contribution: {angle_loss_contribution.item():.6f}")
                    
                    total_loss += angle_loss_contribution  # 角度一致性损失(L_angle)
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
        
        for name, param in model.parameters():
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

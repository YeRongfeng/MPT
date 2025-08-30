"""
train_mamba.py - 训练不平坦地面路径预测模型
"""

import numpy as np
import pickle

import torch
import torch.optim as optim

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
import argparse

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from os import path as osp

from vision_mamba import Models, Optim
from dataLoader_uneven import UnevenPathDataLoader, PaddedSequence
from dataLoader_uneven import hashTable, receptive_field

from torch.utils.tensorboard import SummaryWriter

from ESDF3d_atpoint import compute_esdf_batch
from grad_optimizer import TrajectoryOptimizerSE2

def cal_performance(predVals, correctionVals, normals, yaw_stabilities, cost_map, anchorPoints, trueLabels, trajectory, stage=1):
    """
    计算loss损失
    
    predVals: [batch_size, num_tokens, output_dim] - 模型预测：144个锚点位置，每个位置10个时间步的预测
    correctionVals: [batch_size, num_tokens, 3, output_dim] - 模型修正预测：位置偏移和角度预测
    anchorPoints: 锚点信息字典，包含hashTable等
    trueLabels: [batch_size, num_layers, max_anchors] - 真实标签
    trajectory: [batch_size, num_steps, 3] - 轨迹数据
    stage: 1 为第一阶段, 2 为第二阶段
    """
    # 初始化
    batch_size, num_tokens, output_dim = predVals.shape
    device = predVals.device
    
    # 分离起点和终点信息
    start_state = trajectory[:, 0, :]   # [batch_size, 3]
    goal_state = trajectory[:, -1, :]   # [batch_size, 3] 
    trajectory_copy = trajectory.clone()
    trajectory = trajectory[:, :-1, :]  # 移除终点，保留起点和中间点
    
    # 根据训练阶段设置损失权重
    if stage == 1:
        loss_weights = {
            'classification': 1e-2,  # 第一阶段专注轨迹回归
            'regression': 1e-3, #1e-3
            'uniformity': 1e-4, #1e-4
            'angle': 3e-4,      #3e-4
            'smoothness': 1e-4, #1e-4
            'capsize': 0e-2,
            'curvature': 1e-5,  #1e-5
            'stability': 0e-3,  # 轨迹点稳定性结果预测
        }
        # loss_weights = {
        #     'classification': 1e-2,  # 第一阶段专注轨迹回归
        #     'regression': 3e-5,
        #     'uniformity': 7e-5,
        #     'angle': 8e-5,
        #     'smoothness': 1e-5,
        #     'capsize': 0e-2,
        #     'curvature': 1e-5,
        #     'stability': 0e-3,  # 轨迹点稳定性结果预测
        # }
    else:
        # loss_weights = {
        #     'classification': 1e-3,  # 第二阶段专注安全性优化
        #     'regression': 0e-4,
        #     'uniformity': 3e-4,
        #     'angle': 3e-4,
        #     'smoothness': 1e-4,
        #     'capsize': 1e-3,
        # }
        loss_weights = {
            'classification': 0e-1,  # 第二阶段专注安全性优化
            'regression': 0e-4,
            'uniformity': 0e-4,
            'angle': 0e-4,
            'smoothness': 0e-4,
            'capsize': 3e-7,
            'curvature': 0e-3,
            'stability': 0e-3,  # 轨迹点稳定性结果预测
        }
    
    # 初始化损失和统计
    total_loss = torch.tensor(0.0, requires_grad=True, device=device)
    loss_count = 0
    
    # 统计变量（根据实际标签计算）
    n_correct = 0
    total_samples = 0  # 根据实际有效标签数量计算
    total_positive = 0
    total_negative = 0
    correct_positive = 0
    correct_negative = 0

    for i in range(batch_size):
        anchorPoint = anchorPoints[i]  # [num_layers, max_anchors]
        trueLabel = trueLabels[i]  # [num_layers, max_anchors]
        
        if loss_weights['regression'] > 0 or loss_weights['angle'] > 0 or \
            loss_weights['smoothness'] > 0 or loss_weights['capsize'] > 0 or \
            loss_weights['curvature'] > 0 or loss_weights['stability'] > 0:
            # 如果需要坐标，那么需要在这里提前计算好坐标
            available_trajectory_steps = trajectory.shape[1] - 1  # 减去起点，得到可预测的步数
            steps_to_process = min(output_dim, available_trajectory_steps)
            
            if steps_to_process > 0:
                # 导入全局hashTable
                from dataLoader_uneven import hashTable
                    
                # 将hashTable转换为张量以便并行计算
                hash_table_tensor = torch.tensor(hashTable, device=predVals.device, dtype=torch.float32, requires_grad=False)  # [num_tokens, 2]
                
                # 获取当前样本的预测概率分布
                # pred_probs = F.softmax(predVals[i, :, :steps_to_process], dim=0)  # [num_tokens, steps_to_process]
                pred_probs = predVals[i, :, :steps_to_process]
                
                # 数值稳定性检查
                if torch.any(torch.isnan(pred_probs)) or torch.any(torch.isinf(pred_probs)):
                    continue
                
                # 使用einsum进行高效的加权坐标计算
                # pred_probs: [num_tokens, steps_to_process], hash_table_tensor: [num_tokens, 2]
                # 输出: [steps_to_process, 2]
                weighted_coords = torch.einsum('nt,nc->tc', pred_probs, hash_table_tensor)
                
                # 坐标映射到实际坐标系
                weighted_coords = -5.0 + weighted_coords * 0.1
                
                # 应用修正偏移（基于概率计算）
                if correctionVals is not None:
                    # 使用概率计算加权偏移
                    offset_coords = correctionVals[i, :, :2, :steps_to_process]  # [num_tokens, 2, steps_to_process]
                    weighted_offsets = torch.einsum('nt,nct->tc', pred_probs, offset_coords)
                    weighted_coords += weighted_offsets
                    
                    # 获取预测角度
                    pred_angles_sigmoid = correctionVals[i, :, 2, :steps_to_process]  # [num_tokens, steps_to_process]
                    # 使用概率分布加权角度预测 
                    weighted_angles_sigmoid = torch.sum(pred_probs * pred_angles_sigmoid, dim=0)  # [steps_to_process]
                    # 将sigmoid输出[0,1]转换为角度范围[-π, π]
                    weighted_pred_angles = weighted_angles_sigmoid * 2 * np.pi - np.pi  # [steps_to_process]
        
        # =================== 损失1：锚点分类交叉熵损失(L_ce) ===================
        # 修改版本：让真实标签对应点的概率总和接近1，而不要求均匀分布
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
            
            # 创建有效掩码 [num_anchor_rows, MAX_POSITIVE_ANCHORS]
            valid_mask = (anchorPoint != -1) & (trueLabel != -1) & (anchorPoint < num_tokens)
            
            # 使用高级索引批量提取真实标签位置的预测概率 - 完全张量并行
            classification_loss = torch.tensor(0.0, device=predVals.device)
            valid_steps_count = 0
            
            if valid_mask.any():
                # 获取所有有效位置的索引
                step_indices_idx, anchor_indices = torch.where(valid_mask)  # 有效位置的(step, anchor)索引对
                
                if len(step_indices_idx) > 0:
                    # 批量获取有效的锚点索引和标签值
                    valid_anchor_positions = anchorPoint[step_indices_idx, anchor_indices]  # 锚点在tokens中的位置
                    valid_label_values = trueLabel[step_indices_idx, anchor_indices].float()  # 对应的标签值
                    
                    # 批量获取对应的预测概率
                    pred_probs_at_anchors = all_step_preds[step_indices_idx, valid_anchor_positions]  # [total_valid]
                    
                    # 只保留正样本（标签值为1的锚点）
                    positive_mask = (valid_label_values == 1)
                    if positive_mask.any():
                        positive_step_indices = step_indices_idx[positive_mask]  # [num_positive]
                        positive_probs = pred_probs_at_anchors[positive_mask]  # [num_positive]
                        
                        # 使用scatter_add按时间步分组求和 - 完全张量并行
                        step_prob_sums = torch.zeros(num_anchor_rows, device=predVals.device)
                        step_prob_sums.scatter_add_(0, positive_step_indices, positive_probs)  # [num_anchor_rows]
                        
                        # 计算每个时间步是否有正样本
                        step_has_positive = torch.zeros(num_anchor_rows, device=predVals.device, dtype=torch.bool)
                        step_has_positive.scatter_(0, positive_step_indices, True)
                        
                        # 只对有正样本的时间步计算损失
                        valid_step_sums = step_prob_sums[step_has_positive]  # [valid_steps]
                        
                        if len(valid_step_sums) > 0:
                            # 计算MSE损失：让每个时间步的正样本概率总和接近1
                            target_ones = torch.ones_like(valid_step_sums)
                            step_losses = F.mse_loss(valid_step_sums, target_ones, reduction='none')  # [valid_steps]
                            classification_loss = step_losses.mean()
                            valid_steps_count = len(valid_step_sums)
            
            # 如果有有效的步骤损失，添加到总损失中
            if valid_steps_count > 0:
                # 数值稳定性检查
                if not (torch.isnan(classification_loss) or torch.isinf(classification_loss) or classification_loss.item() > 50):
                    total_loss = total_loss + classification_loss * loss_weights['classification']
                    loss_count += 1
                else:
                    print(f"Warning: Classification loss is abnormal: {classification_loss.item()}, skipping")
            else:
                print(f"Warning: No valid positive labels found for classification loss")
            
            # 批量计算准确率
            # threshold = max(0.01, 1.0 / num_tokens * 2)
            threshold = 0.5 / num_tokens
            
            # 创建有效锚点和标签的掩码 [num_anchor_rows, MAX_POSITIVE_ANCHORS]
            valid_anchor_mask_acc = (anchorPoint != -1) & (trueLabel != -1) & (anchorPoint < num_tokens)
            
            if valid_anchor_mask_acc.any():
                # 使用高级索引一次性提取所有有效的预测和标签
                step_indices_acc, anchor_indices_acc = torch.where(valid_anchor_mask_acc)
                
                if len(step_indices_acc) > 0:
                    # 批量获取有效锚点位置和标签值
                    valid_anchor_positions = anchorPoint[step_indices_acc, anchor_indices_acc]  # [total_valid]
                    valid_label_values = trueLabel[step_indices_acc, anchor_indices_acc]        # [total_valid]
                    
                    # 批量获取对应的预测值
                    valid_step_preds = all_step_preds[step_indices_acc]                         # [total_valid, num_tokens]
                    selected_preds = valid_step_preds[torch.arange(len(step_indices_acc), device=predVals.device), valid_anchor_positions]  # [total_valid]
                    
                    # 批量计算预测类别和统计
                    batch_class_preds = (selected_preds > threshold).long()                     # [total_valid]
                    batch_correct = batch_class_preds.eq(valid_label_values.long())             # [total_valid]
                    
                    # 批量统计各种指标
                    n_correct += batch_correct.sum().item()
                    total_samples += len(valid_label_values)
                    
                    # 批量计算正负样本统计
                    positive_mask = (valid_label_values == 1)
                    negative_mask = (valid_label_values == 0)
                    
                    total_positive += positive_mask.sum().item()
                    total_negative += negative_mask.sum().item()
                    
                    # 批量统计各类别的正确预测
                    correct_positive_mask = (batch_class_preds == 1) & positive_mask
                    correct_negative_mask = (batch_class_preds == 0) & negative_mask
                    correct_positive += correct_positive_mask.sum().item()
                    correct_negative += correct_negative_mask.sum().item()
        
        # =================== 损失2：轨迹回归损失(L_mse) ===================
        if loss_weights['regression'] > 0:
            # 注意：trajectory已经移除了终点，只包含起点和中间点
            available_trajectory_steps = trajectory.shape[1] - 1  # 减去起点，得到可预测的步数
            steps_to_process = min(output_dim, available_trajectory_steps)

            if steps_to_process > 0 and weighted_coords is not None:
                # 计算与真实轨迹的坐标MSE损失
                # trajectory[i, 0] 是起点，trajectory[i, 1:steps_to_process+1] 是要预测的中间点
                true_coords = trajectory[i, 1:1+steps_to_process, :2]  # 从起点后第1个点开始，取steps_to_process个点
                coord_loss = F.mse_loss(weighted_coords, true_coords)
                
                # 添加角度回归监督
                angle_loss = torch.tensor(0.0, device=predVals.device)
                if weighted_pred_angles is not None and trajectory_copy.shape[2] >= 3:
                    # 获取真实轨迹角度（跳过起点，对应预测的中间点）
                    true_angles = trajectory_copy[i, 1:1+steps_to_process, 2]  # [steps_to_process]
                    
                    # 计算角度差异，处理角度的周期性
                    angle_diff = weighted_pred_angles - true_angles
                    # 将角度差异规范化到[-π, π]范围内
                    angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
                    # 计算角度回归损失（使用平方损失）
                    angle_loss = torch.mean(angle_diff ** 2)
                
                # 合并坐标损失和角度损失
                total_regression_loss = coord_loss + 0.1 * angle_loss  # 角度损失权重可调整
                
                # 数值稳定性检查
                if not (torch.isnan(total_regression_loss) or torch.isinf(total_regression_loss) or total_regression_loss.item() > 1000):
                    total_loss = total_loss + total_regression_loss * loss_weights['regression']
                    loss_count += 1
        
        # =================== 损失3: 轨迹点分布均匀性损失(L_uni) ===================
        if loss_weights['uniformity'] > 0:
            if weighted_coords is not None and weighted_coords.shape[1] > 0:
                # 计算相邻点间距离
                coords = weighted_coords[:, :2]  # [num_steps, 2]
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
                
                # 使用预测坐标构建完整轨迹（如果有回归损失计算）
                if weighted_coords is not None:
                    # 构建完整轨迹：起点 + 预测点 + 终点
                    full_coords = torch.cat([
                        start_xy.unsqueeze(0),
                        weighted_coords,  # 来自回归损失计算的预测坐标
                        goal_xy.unsqueeze(0)
                    ], dim=0)  # [steps_to_process+2, 2]
                else:
                    # 如果没有回归损失计算，使用真实轨迹坐标
                    full_coords = trajectory_copy[i, :, :2]
                
                # 计算速度向量（中心差分）
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
                        
                        # 使用预测概率加权角度
                        pred_probs = predVals[i, :, :steps_to_process]
                        
                        # 转换为复数表示并计算加权平均
                        cos_angles = torch.cos(pred_angles_rad)
                        sin_angles = torch.sin(pred_angles_rad)
                        
                        # 加权平均
                        weighted_cos = torch.sum(pred_probs * cos_angles, dim=0)  # [steps_to_process]
                        weighted_sin = torch.sum(pred_probs * sin_angles, dim=0)  # [steps_to_process]
                        
                        pred_unit_vectors = torch.stack([weighted_cos, weighted_sin], dim=1)  # [steps_to_process, 2]
                    else:
                        # 没有角度预测，跳过角度损失
                        continue
                    
                    # 计算角度一致性损失
                    velocity_norms = torch.norm(velocity_vectors, dim=1)
                    valid_mask = velocity_norms > 1e-2
                    
                    if valid_mask.any() and len(pred_unit_vectors) == len(velocity_vectors):
                        valid_velocity_vectors = velocity_vectors[valid_mask]
                        valid_pred_vectors = pred_unit_vectors[valid_mask]
                        valid_velocity_norms = velocity_norms[valid_mask]
                        
                        # 计算余弦相似度
                        dot_products = torch.sum(valid_pred_vectors * valid_velocity_vectors, dim=1)
                        cosine_similarities = dot_products / valid_velocity_norms
                        cosine_similarities = torch.clamp(cosine_similarities, -1.0, 1.0)
                        
                        # 角度损失：1 - cosine_similarity
                        angle_loss = (1.0 - cosine_similarities).mean()
                        
                        if not (torch.isnan(angle_loss) or torch.isinf(angle_loss) or angle_loss.item() > 100):
                            total_loss = total_loss + angle_loss * loss_weights['angle']
                            loss_count += 1
        
        # =================== 损失5：平滑度损失(L_smo) ===================
        if loss_weights['smoothness'] > 0:
            steps_to_process = min(output_dim, trajectory.shape[1])
            if steps_to_process > 0 and trajectory_copy.shape[1] >= 3:
                # 获取起点和终点 - 直接使用start_state和goal_state
                start_xy = start_state[i][:2]
                goal_xy = goal_state[i][:2]
                
                # 使用预测坐标构建完整轨迹（如果有回归损失计算）
                if weighted_coords is not None:
                    # 构建完整轨迹：起点 + 预测点 + 终点
                    full_coords = torch.cat([
                        start_xy.unsqueeze(0),
                        weighted_coords,  # 来自回归损失计算的预测坐标
                        goal_xy.unsqueeze(0)
                    ], dim=0)  # [steps_to_process+2, 2]
                else:
                    # 如果没有回归损失计算，使用真实轨迹坐标
                    full_coords = trajectory_copy[i, :, :2]
                    
                # 计算二阶差分平滑度损失
                if full_coords.shape[0] >= 3:
                    x_coords = full_coords[:, 0]
                    y_coords = full_coords[:, 1]
                    
                    # 二阶差分：s_i = (x_{i+1} - 2*x_i + x_{i-1})^2 + (y_{i+1} - 2*y_i + y_{i-1})^2
                    x_smoothness = (x_coords[2:] - 2 * x_coords[1:-1] + x_coords[:-2]) ** 2
                    y_smoothness = (y_coords[2:] - 2 * y_coords[1:-1] + y_coords[:-2]) ** 2
                    
                    smoothness_loss = torch.mean(x_smoothness + y_smoothness)
                    
                    if not (torch.isnan(smoothness_loss) or torch.isinf(smoothness_loss) or smoothness_loss.item() > 100):
                        total_loss = total_loss + smoothness_loss * loss_weights['smoothness']
                        loss_count += 1
        
        # =================== 损失6：倾覆监督损失(L_badbin) ===================
        if loss_weights['capsize'] > 0:
            nx, ny, nz = normals[i, 0, :, :], normals[i, 1, :, :], normals[i, 2, :, :]  # 获取法线信息
            stability_cost_map = cost_map[i, :, :, :]  # [num_layers, max_anchors]

            full_coords = torch.cat([
                start_state[i][:2].unsqueeze(0),  # 起点
                weighted_coords,  # 来自回归损失计算的预测坐标
                goal_state[i][:2].unsqueeze(0)  # 终点
            ], dim=0)  # [steps_to_process+2, 2]
            full_yaws = torch.cat([
                start_state[i][2].unsqueeze(0),  # 起点角度
                weighted_pred_angles,  # 来自回归损失计算的预测角度
                goal_state[i][2].unsqueeze(0)  # 终点角度
            ], dim=0)  # [steps_to_process+2, 1]
            
            # 将坐标和角度按列堆叠为 [steps_to_process+2, 3]
            full_traj = torch.stack([
                full_coords[:, 0],  # x坐标
                full_coords[:, 1],  # y坐标
                full_yaws  # 预测角度
            ], dim=1).to(dtype=torch.float32, device=predVals.device)
            
            map_size = (100, 100, 36) # W, H, D for (x, y, yaw)
            resolution = 0.1
            origin = (-5.0, -5.0, -np.pi) # x, y, yaw
            map_info = {
                'resolution': resolution,
                'origin': origin,
                'size': map_size
            }
            optimizer = TrajectoryOptimizerSE2(full_traj, stability_cost_map, map_info, device=device)
            # 获得优化轨迹的损失
            capsize_loss = torch.tensor(0.0, device=predVals.device, requires_grad=True)
            # 直接基于模型输出的 full_traj 计算代价并保留计算图，以便梯度能回传到模型
            capsize_loss = optimizer.cost_on_poses(full_traj)
            # 数值稳定性检查
            if not (torch.isnan(capsize_loss) or torch.isinf(capsize_loss) or capsize_loss.item() > 1000):
                total_loss = total_loss + capsize_loss * loss_weights['capsize']
                loss_count += 1

            # # 计算各预测点的倾覆esdf值
            # nx, ny, nz = normals[i, 0, :, :], normals[i, 1, :, :], normals[i, 2, :, :]  # 获取法线信息
            # # 将坐标和角度按列堆叠为 [steps_to_process, 3]
            # queries = torch.stack([
            #     weighted_coords[:, 0],  # x坐标
            #     weighted_coords[:, 1],  # y坐标
            #     weighted_pred_angles    # 预测角度
            # ], dim=1).to(dtype=torch.float32, device=predVals.device)  # [steps_to_process, 3]
            # # 将queries重排为 [steps_to_process, 3]
            # queries = queries.t()  # [steps_to_process, 3]
            # esdf_results = compute_esdf_batch(nx, ny, nz, queries,
            #                                   resolution=0.1, origin=(-5.0, -5.0),
            #                                   yaw_weight=1.4, search_radius=5.0, chunk_cells=3000, device=predVals.device)
            # # 提取每个返回项的第一个元素
            # if isinstance(esdf_results, (list, tuple)):
            #     first_elems = []
            #     for item in esdf_results:
            #         if isinstance(item, (list, tuple)):
            #             first_val = item[0]
            #         else:
            #             first_val = item
            #         # 将非 tensor 转为 tensor，并放到正确设备、类型
            #         if not isinstance(first_val, torch.Tensor):
            #             first_val = torch.tensor(first_val, device=predVals.device, dtype=torch.float32)
            #         else:
            #             first_val = first_val.to(device=predVals.device, dtype=torch.float32)
            #         first_elems.append(first_val)
            #     if len(first_elems) == 0:
            #         # 兜底：避免空列表导致后续错误
            #         capsize_esdf = torch.tensor([], device=predVals.device, dtype=torch.float32)
            #     else:
            #         capsize_esdf = torch.stack(first_elems)  # [steps, ...]
            # else:
            #     # 如果直接返回单个值或单个 tensor，直接使用
            #     capsize_esdf = esdf_results if isinstance(esdf_results, torch.Tensor) else torch.tensor(esdf_results, device=predVals.device, dtype=torch.float32)
            
            # d_safe = 0.  # 安全距离（米）
            # kalpa = 0.6  # 安全损失衰减速率
            
            # # 如果 capsize_esdf 为空则跳过；否则按元素计算并取平均作为损失（可按需改为逐点加权）
            # if capsize_esdf.numel() > 0:
            #     # 将倾覆损失的梯度设置为可训练
            #     capsize_loss = torch.tensor(0.0, device=predVals.device, requires_grad=True)
                
            #     capsize_loss_per_point = torch.exp(-(capsize_esdf - d_safe) / kalpa)
            #     capsize_loss = capsize_loss_per_point.mean()
            #     # 数值稳定性检查
            #     if not (torch.isnan(capsize_loss) or torch.isinf(capsize_loss) or capsize_loss.item() > 1e6):
            #         total_loss = total_loss + capsize_loss * loss_weights['capsize']
            #         loss_count += 1
            # else:
            #     # 没有有效的 esdf 返回，跳过该损失
            #     pass
            
        # =================== 损失7：曲率损失(L_curvature) ===================
        if loss_weights['curvature'] > 0:
            steps_to_process = min(output_dim, trajectory.shape[1])
            if steps_to_process > 0 and trajectory_copy.shape[1] >= 3:
                # 获取起点和终点 - 直接使用start_state和goal_state
                start_xy = start_state[i][:2]
                goal_xy = goal_state[i][:2]
                
                # 使用预测坐标构建完整轨迹（如果有回归损失计算）
                if weighted_coords is not None:
                    # 构建完整轨迹：起点 + 预测点 + 终点
                    full_coords = torch.cat([
                        start_xy.unsqueeze(0),
                        weighted_coords,  # 来自回归损失计算的预测坐标
                        goal_xy.unsqueeze(0)
                    ], dim=0)  # [steps_to_process+2, 2]
                else:
                    # 如果没有回归损失计算，使用真实轨迹坐标
                    full_coords = trajectory_copy[i, :, :2]
                    
                # 计算曲率损失（使用差分近似速度和加速度）
                if full_coords.shape[0] >= 3:
                    x_coords = full_coords[:, 0].to(device=device, dtype=torch.float32)
                    y_coords = full_coords[:, 1].to(device=device, dtype=torch.float32)
                    
                    # 速度：前向差分 v_k = x_{k+1} - x_k  -> 长度 N-1
                    vx = x_coords[1:] - x_coords[:-1]   # [N-1]
                    vy = y_coords[1:] - y_coords[:-1]   # [N-1]
                    
                    # 加速度：差分速度 a_k = v_{k+1} - v_k -> 长度 N-2
                    ax = vx[1:] - vx[:-1]   # [N-2]
                    ay = vy[1:] - vy[:-1]   # [N-2]
                    
                    # 对齐以计算曲率：使用 xdot = vx[:-1], xdd = ax
                    xdot = vx[:-1]  # [N-2]
                    ydot = vy[:-1]  # [N-2]
                    xdd = ax
                    ydd = ay
                    
                    # 计算曲率：|x' y'' - y' x''| / (x'^2 + y'^2)^{3/2}
                    num = torch.abs(xdot * ydd - ydot * xdd)
                    denom = (xdot ** 2 + ydot ** 2) ** 1.5 + 1e-9
                    # 更稳健的分母下界，避免速度接近0时曲率爆炸
                    denom = torch.clamp(denom, min=1e-3)
                    curvature = num / denom
                    
                    # 曲率惩罚：只惩罚超过阈值的部分
                    curvature_threshold = 2.1
                    curvature_cost = torch.mean(F.relu(curvature - curvature_threshold))
                    
                    # 数值稳定性检查并加入损失
                    if not (torch.isnan(curvature_cost) or torch.isinf(curvature_cost) or curvature_cost.item() > 100):
                        total_loss = total_loss + curvature_cost * loss_weights['curvature']
                        loss_count += 1
                        
        # =================== 损失8：轨迹点稳定性损失(L_stability) ===================
        if loss_weights['stability'] > 0:
            # 这个损失是模型最终会输出一个稳定性预测值，用于间接监督模型对问题的建模理解
            if correctionVals is not None:
                stability_preds_raw = correctionVals[i, :, 3, :steps_to_process]  # [num_tokens, steps]
                pred_probs = predVals[i, :, :steps_to_process]  # [num_tokens, steps]
                # 对 token 维度加权求和，得到每个 time step 的稳定性预测（[steps]）
                stability_preds = torch.sum(stability_preds_raw * pred_probs, dim=0)
                stability_preds = stability_preds.to(dtype=torch.float32, device=predVals.device)
                
                # 将坐标和角度按列堆叠为 [steps_to_process+2, 3]
                weighted_traj = torch.stack([
                    weighted_coords[:, 0],  # x坐标
                    weighted_coords[:, 1],  # y坐标
                    weighted_pred_angles  # 预测角度
                ], dim=1).to(dtype=torch.float32, device=predVals.device)
                
                # 将预测的轨迹点位姿去cost_map中查找稳定性值
                # 先将轨迹点转换为cost_map索引
                map_size = (100, 100, 36)  # 假设cost_map是100x100x36的
                resolution = 0.1  # 假设分辨率是0.1米
                origin = (-5.0, -5.0, -np.pi)  # 假设原点是(-5.0, -5.0, -π)
                id_x = ((weighted_traj[:, 0] - origin[0]) / resolution).long()  # x坐标索引
                id_y = ((weighted_traj[:, 1] - origin[1]) / resolution).long()  # y坐标索引
                id_yaw = ((weighted_traj[:, 2] - origin[2]) / (2 * np.pi / map_size[2])).long()  # yaw索引
                
                # 确保索引在有效范围内
                id_x = torch.clamp(id_x, 0, map_size[0] - 1)
                id_y = torch.clamp(id_y, 0, map_size[1] - 1)
                id_yaw = torch.clamp(id_yaw, 0, map_size[2] - 1)
                
                # 使用索引从cost_map中获取稳定性值
                yaw_stability = yaw_stabilities[i, :, :, :]  # [num_layers, max_anchors]
                stability_values = yaw_stability[id_x, id_y, id_yaw]  # [steps_to_process]
                
                # 计算稳定性损失：让预测的稳定性值接近真实值
                stability_loss = F.mse_loss(stability_preds, stability_values)
                # 数值稳定性检查
                if not (torch.isnan(stability_loss) or torch.isinf(stability_loss) or stability_loss.item() > 100):
                    total_loss = total_loss + stability_loss * loss_weights['stability']
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
        # encoder_input = batch['map'].float().to(device)  # 准备输入数据：将地图数据转换为浮点型并移至指定设备
        # predVal, correctionVal = model(encoder_input)  # 前向传播：获取模型预测值和修正值
        map_input = batch['map'].float().to(device)
        predVal, correctionVal = model(map_input)  # 前向传播：获取模型预测值和修正值
        # predVal = model(encoder_input)  # 前向传播：获取模型预测值和修正值
        normals = map_input[:, :3, :, :]  # 获取法线信息 [N, 3, H, W]

        # 正确处理锚点和标签，保持对应关系
        loss, n_correct, n_samples, batch_stats = cal_performance(
            predVal, 
            correctionVal,  # 添加correctionVal参数
            normals,  # 添加法线信息
            # batch['yaw_stability'].to(device),  # 添加yaw_stability参数
            None,
            batch['cost_map'].to(device),  # 添加cost_map参数
            batch['anchor'].to(device),
            batch['labels'].to(device),
            batch['trajectory'].to(device),  # 轨迹点：[N, 3]
            stage=stage  # 传递阶段信息
        )
        
        # 在反向传播前检查损失值的合理性
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN or Inf loss detected, skipping batch")
            continue
        
        loss.backward()  # 反向传播：计算梯度
        
        # 梯度裁剪：防止梯度爆炸 - 平衡稳定性和学习效率
        max_grad_norm = 5.0  # 适度控制，平衡稳定性和学习速度
        if stage == 2:
            max_grad_norm = 0.5  # 第二阶段更严格的梯度裁剪
        
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
        
        stage_info = "S1" if stage == 1 else "S2"  # 显示训练阶段
        pbar.set_postfix({
            'Stage': stage_info,
            'Loss': f'{loss.item():.4f}',
            'GradNorm': grad_info,
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

            # encoder_input = batch['map'].float().to(device)  # 准备输入数据：将地图数据转换为浮点型并移至指定设备
            # predVal, correctionVal = model(encoder_input)  # 前向传播：获取模型预测值和修正值
            map_input = batch['map'].float().to(device)
            predVal, correctionVal = model(map_input)  # 前向传播：获取模型预测值和修正值
            # predVal = model(encoder_input)  # 前向传播：获取模型预测值和修正值
            normals = map_input[:, :3, :, :]  # 获取法线信息 [N, 3, H, W]

            # 正确处理锚点和标签，保持对应关系
            loss, n_correct, n_samples, batch_stats = cal_performance(
                predVal, 
                correctionVal,  # 添加correctionVal参数
                normals,  # 添加法线信息
                # batch['yaw_stability'].to(device),  # 添加yaw_stability参数
                None,
                batch['cost_map'].to(device),  # 添加cost_map参数
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
    # parser.add_argument('--env_list', help="Directory with training and validation data for Maze", default=None) 
    parser.add_argument('--dataFolder', help="Directory with training and validation data for Maze", default=None)  # 添加数据文件夹参数
    parser.add_argument('--fileDir', help="Directory to save training Data")  # 添加训练数据保存目录参数
    parser.add_argument('--load_stage1_model', help="Path to stage1 model checkpoint to load and start stage2 training", default=None)  # 加载第一阶段的模型参数，直接开始第二阶段训练
    parser.add_argument('--resume_stage1_model', help="Path to resume stage1 training (continue from checkpoint)", default=None)  # 恢复阶段1训练
    args = parser.parse_args()  # 解析命令行参数

    map_load = False  # 是否加载地图数据
    dataFolder = args.dataFolder  # 确定数据文件夹：使用提供的环境数据文件夹
    if not osp.isdir(dataFolder):  # 检查数据文件夹是否存在
        raise ValueError("Please provide a valid data folder")  # 如果不存在，抛出错误
    
    # assert args.env_list is not None, "Please provide environment list"  # 确保提供了环境列表
    # env_list = args.env_list.split(',')  # 将环境列表字符串分割成列表

    env_num = 40
    env_list = [f"env{i:06d}" for i in range(env_num)]  # 生成环境列表，格式为 env000000, env000001, ..., env000099
    # print(f"Training on {len(env_list)} environments: {env_list}")  # 打印环境列表长度和内容

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
    
    # model_args = dict(        # 定义模型参数字典
    #     n_layers=6,           # Transformer编码器层数：6层
    #     n_heads=6,            # 多头注意力的头数：8->6个头
    #     d_k=128,              # Key向量的维度：192->128
    #     d_v=64,               # Value向量的维度：96->64
    #     d_model=512,          # 模型的主要特征维度：512
    #     d_inner=768,         # 前馈网络的隐藏层维度：1024->768
    #     pad_idx=None,         # 填充标记的索引：无
    #     n_position=15*15,     # 支持的最大位置数：225(15×15)
    #     dropout=0.1,          # Dropout概率：0.1
    #     train_shape=[12, 12], # 训练时的地图形状：12×12
    #     output_dim=10,        # 输出维度：10
    # )
    
    model_args = dict(        # 定义模型参数字典
        n_layers=6,          # Mamba编码器层数：12层
        d_state=16,           # Mamba状态维度：16
        dt_rank=32,           # 动态张量分解秩：32
        d_model=512,          # 模型的主要特征维度：512
        pad_idx=None,         # 填充标记的索引：无
        n_position=15*15,     # 支持的最大位置数：225(15×15)
        dropout=0.1,          # Dropout概率：0.1
        drop_path=0.25,        # DropPath概率：0.3
        train_shape=[12, 12], # 训练时的地图形状：12×12
        output_dim=10,        # 输出维度：10
    )

    transformer = Models.UnevenTransformer(**model_args)  # 使用参数字典初始化Transformer模型

    if torch.cuda.device_count() > 1:  # 检查是否有多个GPU
        print("Using ", torch.cuda.device_count(), "GPUs")  # 打印使用的GPU数量
        transformer = nn.DataParallel(transformer)  # 使用DataParallel包装模型，实现数据并行
    transformer.to(device=device)  # 将模型移动到指定设备(CPU或GPU)
    
    # 双阶段训练配置
    stage1_epochs = 100  # 第一阶段训练轮数：训练除correctionPred外的所有参数
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
    
    # 记录各阶段最优验证损失，用于保存best model
    best_val_loss_stage1 = float('inf')
    best_val_loss_stage2 = float('inf')
    best_stage1_path = osp.join(trainDataFolder, 'best_stage1_model.pkl')
    best_stage2_path = osp.join(trainDataFolder, 'best_stage2_model.pkl')
    
    # 检查是否需要加载第一阶段的模型并直接开始第二阶段训练
    start_stage = 1
    checkpoint = None  # 初始化 checkpoint 变量
    resume_stage1 = False
    resume_start_epoch = 0
    
    # 优先处理直接跳过到stage2的加载选项；否则检查是否需要从stage1检查点恢复训练
    if args.load_stage1_model:
        # 加载第一阶段的检查点并直接开始第二阶段训练
        print(f"Loading stage 1 checkpoint: {args.load_stage1_model}")
        checkpoint = load_stage1_checkpoint(transformer, args.load_stage1_model, device)
        start_stage = 2  # 直接开始第二阶段训练
        print("Will skip stage 1 and directly start stage 2 training")
    elif args.resume_stage1_model:
        # 从stage1检查点恢复训练（继续stage1）
        print(f"Resuming stage 1 training from checkpoint: {args.resume_stage1_model}")
        checkpoint = load_stage1_checkpoint(transformer, args.resume_stage1_model, device)
        resume_stage1 = True
        resume_start_epoch = checkpoint.get('epoch', -1) + 1
        # 从checkpoint中恢复已保存的最佳验证loss（如果有）
        try:
            best_val_loss_stage1 = float(checkpoint.get('val_loss', best_val_loss_stage1))
        except:
            pass
        print(f"Will resume Stage 1 from epoch {resume_start_epoch}")
    
    # 第一阶段：冻结correctionPred，训练其他参数
    if start_stage == 1:
        print("=== Stage 1: Training all parameters except correctionPred ===")
        for param in transformer.parameters():
            param.requires_grad = True
        # for param in transformer.correctionPred.parameters():
        #     param.requires_grad = False
        
        # 打印参数状态
        print_model_parameters(transformer, "Stage 1")
        
        # 第一阶段优化器
        stage1_optimizer = Optim.ScheduledOptim(
            optim.Adam(filter(lambda p: p.requires_grad, transformer.parameters()),
                       betas=(0.9, 0.98), eps=1e-9),
            # lr_mul = 0.1,
            lr_mul = 3e-2,
            d_model = 512,
            n_warmup_steps = 800
            # n_warmup_steps = 3200
        )
        # 如果是从checkpoint恢复stage1训练，并且checkpoint中有optimizer状态，则恢复优化器
        if resume_stage1 and checkpoint is not None and 'optimizer' in checkpoint:
            try:
                stage1_optimizer._optimizer.load_state_dict(checkpoint['optimizer'])
                print("Loaded stage1 optimizer state from checkpoint")
            except Exception as e:
                print(f"Warning: Failed to load stage1 optimizer state: {e}")
        
        # 第一阶段训练
        start_epoch = resume_start_epoch if resume_stage1 else 0
        for n in range(start_epoch, stage1_epochs):
            train_total_loss, train_n_correct, train_samples, train_stats = train_epoch(
                transformer, trainingData, stage1_optimizer, device, epoch=n, stage=1
            )
            val_total_loss, val_n_correct, val_samples, val_stats = eval_epoch(transformer, validationData, device, stage=1)
            
            # 保存 stage1 的 best model（按验证损失）
            try:
                if val_total_loss < best_val_loss_stage1:
                    best_val_loss_stage1 = val_total_loss
                    if isinstance(transformer, nn.DataParallel):
                        state_dict = transformer.module.state_dict()
                    else:
                        state_dict = transformer.state_dict()
                    torch.save({
                        'state_dict': state_dict,
                        'torch_seed': torch_seed,
                        'stage': 1,
                        'epoch': n,
                        'val_loss': val_total_loss
                    }, best_stage1_path)
                    print(f"Saved new best Stage1 model to {best_stage1_path} (val_loss={val_total_loss:.4f})")
            except Exception as e:
                print(f"Warning: Failed to save best stage1 model: {e}")
            
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
    # for param in transformer.classPred.parameters():
    #     param.requires_grad = True
    transformer.train()  # 设置模型为训练模式
    # for param in transformer.encoder.map_fe.parameters():
    #     param.requires_grad = False
    # for param in transformer.encoder.pose_injector.parameters():
    #     param.requires_grad = False
    
    # 打印参数状态
    print_model_parameters(transformer, "Stage 2")
    
    # 第二阶段优化器
    stage2_optimizer = Optim.ScheduledOptim(
        optim.Adam(filter(lambda p: p.requires_grad, transformer.parameters()),
                   betas=(0.9, 0.98), eps=1e-9),
        # lr_mul = 1.0,
        lr_mul = 1e-3,
        d_model = 512,
        n_warmup_steps = 800
    )
    
    # 第二阶段训练
    for n in range(stage2_epochs):
        train_total_loss, train_n_correct, train_samples, train_stats = train_epoch(
            transformer, trainingData, stage2_optimizer, device, epoch=n, stage=2
        )
        val_total_loss, val_n_correct, val_samples, val_stats = eval_epoch(transformer, validationData, device, stage=2)
        
        # 保存 stage2 的 best model（按验证损失）
        try:
            if val_total_loss < best_val_loss_stage2:
                best_val_loss_stage2 = val_total_loss
                if isinstance(transformer, nn.DataParallel):
                    state_dict = transformer.module.state_dict()
                else:
                    state_dict = transformer.state_dict()
                torch.save({
                    'state_dict': state_dict,
                    'torch_seed': torch_seed,
                    'stage': 2,
                    'epoch': n,
                    'val_loss': val_total_loss
                }, best_stage2_path)
                print(f"Saved new best Stage2 model to {best_stage2_path} (val_loss={val_total_loss:.4f})")
        except Exception as e:
            print(f"Warning: Failed to save best stage2 model: {e}")
        
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

        if (n+1)%1==0:
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

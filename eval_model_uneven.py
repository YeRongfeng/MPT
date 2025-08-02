'''A script for generating patches
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage.io
import numpy as np
import pickle

from os import path as osp
import argparse
import json
import time

from transformer import Models
from dataLoader_uneven import get_encoder_input, receptive_field

device='cuda' if torch.cuda.is_available() else 'cpu'

def getHashTable(mapSize):
    H, W = mapSize
    anchor_spacing = 8
    boundary_offset = 6
    
    H_1 = np.floor((H-2)/2)-2
    W_1 = np.floor((W-2)/2)-2
    H_2 = np.floor((H_1-2)/2)+2
    W_2 = np.floor((W_1-2)/2)+2
    output_grid_h = int(np.floor((H_2-1)/2)+1)
    output_grid_w = int(np.floor((W_2-1)/2)+1)
    
    # 生成锚点哈希表：像素坐标 (row, col)
    return [(anchor_spacing*r+boundary_offset, anchor_spacing*c+boundary_offset)
            for c in range(output_grid_w) for r in range(output_grid_h)]

def get_patch(model, start_pos, goal_pos, normal_x, normal_y, normal_z):
    # Identitfy Anchor points
    encoder_input = get_encoder_input(normal_z, goal_pos, start_pos, normal_x, normal_y)
    hashTable = getHashTable(normal_z.shape)
    
    print(f"Input map shape: {normal_z.shape}")
    print(f"Encoder input shape: {encoder_input.shape}")
    
    # 将numpy数组转换为PyTorch张量并确保正确的数据类型和维度顺序
    if isinstance(encoder_input, np.ndarray):
        # 从 [H, W, C] 转换为 [C, H, W]
        encoder_input = torch.from_numpy(encoder_input).permute(2, 0, 1).float()
    elif isinstance(encoder_input, torch.Tensor):
        # 确保维度顺序正确
        if encoder_input.dim() == 3 and encoder_input.shape[-1] == 4:
            encoder_input = encoder_input.permute(2, 0, 1).float()
        else:
            encoder_input = encoder_input.float()
    else:
        encoder_input = torch.tensor(encoder_input, dtype=torch.float32)
        if encoder_input.dim() == 3 and encoder_input.shape[-1] == 4:
            encoder_input = encoder_input.permute(2, 0, 1)
    
    print(f"Encoder input tensor shape: {encoder_input.shape}")
    
    try:
        predVal = model(encoder_input[None,:].cuda())  # Shape: (batch_size, channels, height, width) -> (batch_size, num_tokens, output_dim)
    except Exception as e:
        print(f"Model forward error: {e}")
        print(f"Model expected input size might be different from {normal_z.shape}")
        raise e
    
    # 从模型输出获取维度信息
    batch_size, num_tokens, output_dim = predVal.shape
    
    # 对每个output_dim维度分别进行全局softmax归一化
    predProb_list = []
    patch_maps = []
    map_size = normal_z.shape
    
    for dim in range(output_dim):
        # 对第dim个输出维度进行softmax归一化
        predProb = F.softmax(predVal[0, :, dim], dim=0)  # Shape: (num_tokens,)
        predProb_list.append(predProb)
        
        # 对第dim个输出维度获取预测锚点 (取概率最大的作为预测类别)
        predClass = (predProb > 0.1).long()  # 可以调整阈值或使用其他策略
        possAnchor = [hashTable[i] for i, label in enumerate(predClass) if label==1]
        
        # 为第dim个输出维度生成patch_map
        patch_map = np.zeros_like(normal_z, dtype=np.float32)
        for pos in possAnchor:
            goal_start_x = max(0, pos[0]- receptive_field//2)
            goal_start_y = max(0, pos[1]- receptive_field//2)
            goal_end_x = min(map_size[1], pos[0]+ receptive_field//2)
            goal_end_y = min(map_size[0], pos[1]+ receptive_field//2)
            patch_map[goal_start_y:goal_end_y, goal_start_x:goal_end_x] = 1.0
        
        patch_maps.append(patch_map)
    
    # 将列表转换为numpy数组：(output_dim, height, width)
    patch_maps = np.stack(patch_maps, axis=0)
    predProb_list = torch.stack(predProb_list, dim=0)  # Shape: (output_dim, num_tokens)
    
    # 将n张概率图（每个坐标对应一个锚点），独立地进行对锚点坐标概率加权运算，回归到n个坐标上
    predTraj = []
    for dim in range(output_dim):
        predProb = predProb_list[dim]


        # 对每个全部锚点位置进行加权
        weighted_x = sum(pos[0] * predProb[idx].item() for idx, pos in enumerate(hashTable))
        weighted_y = sum(pos[1] * predProb[idx].item() for idx, pos in enumerate(hashTable))

        # 映射回到实际坐标系
        weighted_x = -5 + weighted_x * 0.1  # 列对应x坐标
        weighted_y = -5 + weighted_y * 0.1  # 行对应y坐标
        
        # 将加权结果添加到预测轨迹中
        predTraj.append((weighted_x, weighted_y))

    return patch_maps, predProb_list, predTraj
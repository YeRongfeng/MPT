import matplotlib.pyplot as plt
import os
from os import path as osp
import numpy as np
import pickle

from skimage import io

import sys
sys.modules['numpy._core'] = np
sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath
sys.modules['numpy._core.multiarray'] = np.core.multiarray

import torch
import torch.nn.functional as F
import json

from transformer import Models
from dataLoader_uneven import get_encoder_input, receptive_field
from eval_model_uneven import getHashTable, get_patch

# Define the network
device='cuda' if torch.cuda.is_available() else 'cpu'



def plot_single_trajectory(ax, elevation_masked, trajectory, predTraj=None, output_dim=None, is_pred=False):
    """绘制单个轨迹子图的辅助函数"""
    # 显示地形图
    ax.imshow(elevation_masked, extent=[-5, 5, -5, 5],
              origin='lower', cmap='terrain', aspect='equal')
    ax.grid(True, alpha=0.3)
    
    start_pos = trajectory[0, :]
    goal_pos = trajectory[-1, :]
    
    # 如果是预测图，则绘制预测轨迹
    if is_pred and predTraj is not None:
        # 先为预测轨迹补足起点和终点，只使用x,y坐标
        predTraj = np.vstack((start_pos[:2], predTraj, goal_pos[:2]))
        # print(f"Predicted trajectory shape: {predTraj.shape}")
        output_dim = predTraj.shape[0]  # 已经包括起点和终点
        
        for i in range(output_dim-1):
            color = plt.cm.rainbow(i / (output_dim - 2))  # 使用rainbow颜色映射
            ax.plot([predTraj[i][0], predTraj[i+1][0]], 
                    [predTraj[i][1], predTraj[i+1][1]], 
                    color=color, zorder=3, linewidth=2, marker='o', markersize=3)
    # 否则绘制真实轨迹
    else:
        for i in range(trajectory.shape[0] - 1):
            color = plt.cm.rainbow(i / (trajectory.shape[0] - 2))  # 使用rainbow颜色映射
            ax.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], color=color, zorder=3, linewidth=2, marker='o', markersize=3)
    
    # 绘制起点和终点
    ax.scatter(start_pos[0], start_pos[1], color='r', zorder=3, s=50)
    ax.scatter(goal_pos[0], goal_pos[1], color='purple', zorder=3, s=50)
    
    # 绘制朝向
    arrow_scale = 0.3
    ax.arrow(start_pos[0], start_pos[1],
             np.cos(start_pos[2]) * arrow_scale,
             np.sin(start_pos[2]) * arrow_scale,
             head_width=0.1, head_length=0.2, fc='r', ec='r', zorder=3)
    ax.arrow(goal_pos[0], goal_pos[1],
             np.cos(goal_pos[2]) * arrow_scale,
             np.sin(goal_pos[2]) * arrow_scale,
             head_width=0.1, head_length=0.2, fc='purple', ec='purple', zorder=3)
    
    # 设置标题并关闭坐标轴
    title = 'Predicted Trajectory' if is_pred else 'Ground Truth Trajectory'
    ax.set_title(title, fontsize=12, pad=8)
    ax.axis('off')

def plot_elevation_map(pathNums, envType, save_path='predictions'):
    """绘制多组轨迹对比图"""
    if not isinstance(pathNums, list):
        pathNums = [pathNums]  # 确保pathNums是列表
    
    # 限制最多显示6组对比
    pathNums = pathNums[:6]
    num_pairs = len(pathNums)
    
    # 创建大图和网格，不使用constrained_layout以便手动控制布局
    fig = plt.figure(figsize=(18, 13))  # 略微增加高度，为标题和颜色条预留空间
    gs = plt.GridSpec(3, 4, figure=fig, left=0.05, right=0.95, 
                     bottom=0.08, top=0.9, wspace=0.15, hspace=0.3)  # 手动设置边距和间距
    
    # 加载环境数据，所有子图共用同一个环境
    envFolder = osp.join('data/test_training/val', envType)
    env_path = osp.join(envFolder, f'map.p')
    with open(env_path, 'rb') as f:
        env = pickle.load(f)
        tensor = env['tensor']
        elevation = tensor[:, :, 0]
        normal_x = tensor[:, :, 1]
        normal_y = tensor[:, :, 2]
        normal_z = tensor[:, :, 3]
        elevation_masked = np.ma.masked_invalid(elevation)
    
    # 为每对轨迹创建子图
    for idx, pathNum in enumerate(pathNums):
        # 计算当前对应在网格中的位置
        row = idx // 2
        col = (idx % 2) * 2
        
        # 加载真实轨迹数据
        path_file = osp.join(envFolder, f'path_{pathNum}.p')
        with open(path_file, 'rb') as f:
            path_data = pickle.load(f)
            trajectory = path_data['path']  # [N, 3]
        
        start_pos = trajectory[0, :]
        goal_pos = trajectory[-1, :]
        
        # 获取预测轨迹
        # patch_map, predProb, predTraj = get_patch(transformer, start_pos[:2], goal_pos[:2], normal_x, normal_y, normal_z)
        patch_map, predProb, predTraj = get_patch(transformer, start_pos, goal_pos, normal_x, normal_y, normal_z)
        output_dim = patch_map.shape[0]
        
        # 创建左侧子图 - 预测轨迹
        ax_pred = fig.add_subplot(gs[row, col])
        # 为预测轨迹补足起点和终点
        plot_single_trajectory(ax_pred, elevation_masked, trajectory, predTraj, output_dim, is_pred=True)
        
        # 创建右侧子图 - 真实轨迹
        ax_true = fig.add_subplot(gs[row, col+1])
        plot_single_trajectory(ax_true, elevation_masked, trajectory, is_pred=False)
        
        # 添加路径编号 - 调整位置和样式使其更加醒目
        ax_pred.text(0.02, 0.98, f"Path #{pathNum}", transform=ax_pred.transAxes, 
                   fontsize=11, weight='bold', verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.3, edgecolor='gray'))
    
    # 为轨迹的颜色添加全局颜色条 - 调整位置到图形底部，更合理的位置
    norm = plt.Normalize(0, output_dim - 1)
    cmap = plt.cm.rainbow
    # 调整颜色条位置，放置在整个图形的底部
    cbar_ax = fig.add_axes([0.3, 0.02, 0.4, 0.015])  # [left, bottom, width, height]
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
                       cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Trajectory Step', fontsize=12)
    
    # 添加超级标题，调整位置确保不被裁剪
    plt.subplots_adjust(top=0.92)  # 为标题腾出更多空间
    fig.suptitle(f'Trajectory Predictions for {envType.capitalize()} Environment', 
                fontsize=18, y=0.98, fontweight='bold')
    
    # 保存图像
    save_path = osp.join(save_path, f'{envType}')
    os.makedirs(save_path, exist_ok=True)
    
    # 生成路径编号列表的简短表示，例如 1_2_3
    path_ids_str = "_".join(str(p) for p in pathNums)
    plt.savefig(osp.join(save_path, f'multi_trajectories_{path_ids_str}.png'), dpi=300)

if __name__ == "__main__":
    epoch = 19
    envType_list = ['desert']
    # envType_list = ['hill']
    save_path = 'predictions'

    modelFolder = 'data/uneven'
    # modelFolder = 'data/uneven_old'
    modelFile = osp.join(modelFolder, f'model_params.json')
    model_param = json.load(open(modelFile))

    transformer = Models.UnevenTransformer(**model_param)
    _ = transformer.to(device)

    checkpoint = torch.load(osp.join(modelFolder, f'model_epoch_{epoch}.pkl'))
    transformer.load_state_dict(checkpoint['state_dict'])

    _ = transformer.eval()

    envType_random = np.random.choice(envType_list, size=1)[0]
    # 随机选择6个不同的路径
    path_indexes = np.random.choice(range(500), size=6, replace=False)
    # path_indexes = np.array([7, 33, 86, 150, 196, 197])  # 示例路径索引
    # path_indexes = np.array([21, 42, 63, 84, 105, 126])  # 示例路径索引
    print(f"Evaluating environment: {envType_random}")
    print(f"Evaluating path indexes: {path_indexes}")

    # 绘制6组对比图
    plot_elevation_map(path_indexes.tolist(), envType_random, save_path)
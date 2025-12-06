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

dataset_path = 'data/sim_dataset/train'
# dataset_path = 'data/sim_dataset/val'

def plot_all_trajectories(envType, save_path='predictions'):
    """
    绘制指定环境中的所有轨迹在同一张图上
    Args:
        envType: str 环境类型/名称
        save_path: str 保存路径
    """
    # 加载环境数据
    envFolder = osp.join(dataset_path, envType)
    env_path = osp.join(envFolder, f'map.p')
    
    if not os.path.exists(env_path):
        print(f"Error: {env_path} does not exist!")
        return
    
    with open(env_path, 'rb') as f:
        env = pickle.load(f)
    
    # 获取所有轨迹文件 - 使用与predict.py相同的命名规则
    all_files = os.listdir(envFolder)
    traj_files = sorted([f for f in all_files if f.startswith('path_') and f.endswith('.p')])
    
    if len(traj_files) == 0:
        print(f"Error: No trajectory files found in {envFolder}")
        print(f"Looking for files matching 'path_*.p'")
        print(f"Available files: {all_files[:20]}")
        return
    
    print(f"Found {len(traj_files)} trajectories in {envType}")
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'All Trajectories in {envType} ({len(traj_files)} paths)', fontsize=14, pad=10)
    
    # 遍历所有轨迹文件
    successful_plots = 0
    for traj_file in traj_files:
        try:
            traj_path = osp.join(envFolder, traj_file)
            with open(traj_path, 'rb') as f:
                traj_data = pickle.load(f)
                trajectory = traj_data['path']  # [N, 3]
            
            start_pos = trajectory[0, :]
            goal_pos = trajectory[-1, :]
            
            # 绘制轨迹线 - 使用蓝色，带透明度
            ax.plot(trajectory[:, 0], trajectory[:, 1], 
                    color='blue', alpha=0.3, linewidth=1.5, zorder=2)
            
            # 绘制起点 - 绿色
            ax.scatter(start_pos[0], start_pos[1], 
                      color='green', alpha=0.5, s=30, zorder=3, edgecolors='none')
            
            # 绘制终点 - 红色
            ax.scatter(goal_pos[0], goal_pos[1], 
                      color='red', alpha=0.5, s=30, zorder=3, edgecolors='none')
            
            successful_plots += 1
            
        except Exception as e:
            print(f"Error loading {traj_file}: {e}")
    
    print(f"Successfully plotted {successful_plots} trajectories")
    
    if successful_plots == 0:
        print("No trajectories were successfully plotted!")
        return
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', alpha=0.3, linewidth=2, label='Trajectories'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
               markersize=8, alpha=0.5, label='Start Points'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
               markersize=8, alpha=0.5, label='Goal Points')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # 保存图像
    save_dir = osp.join(save_path, 'dataset_overview')
    os.makedirs(save_dir, exist_ok=True)
    save_file = osp.join(save_dir, f'{envType}_all_trajectories.png')
    
    plt.tight_layout()
    plt.savefig(save_file, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_file}")

def plot_trajectory_heatmap(envType, save_path='predictions', grid_size=100):
    """
    绘制轨迹热力图，显示路径密度分布
    Args:
        envType: str 环境类型/名称
        save_path: str 保存路径
        grid_size: int 网格大小
    """
    # 加载环境数据
    envFolder = osp.join(dataset_path, envType)
    
    if not os.path.exists(envFolder):
        print(f"Error: Directory {envFolder} does not exist!")
        return
    
    # 获取所有轨迹文件
    all_files = os.listdir(envFolder)
    traj_files = sorted([f for f in all_files if f.startswith('path_') and f.endswith('.p')])
    
    if len(traj_files) == 0:
        print(f"Error: No trajectory files found in {envFolder}")
        return
    
    print(f"Found {len(traj_files)} trajectories for heatmap")
    
    # 创建热力图网格
    heatmap = np.zeros((grid_size, grid_size))
    
    # 遍历所有轨迹，统计每个网格的访问次数
    for traj_file in traj_files:
        try:
            traj_path = osp.join(envFolder, traj_file)
            with open(traj_path, 'rb') as f:
                traj_data = pickle.load(f)
                trajectory = traj_data['path']  # [N, 3]
            
            # 将坐标转换为网格索引
            x_indices = ((trajectory[:, 0] + 20) / 40 * grid_size).astype(int)
            y_indices = ((trajectory[:, 1] + 20) / 40 * grid_size).astype(int)
            
            # 确保索引在有效范围内
            x_indices = np.clip(x_indices, 0, grid_size - 1)
            y_indices = np.clip(y_indices, 0, grid_size - 1)
            
            # 增加热力图计数
            for x_idx, y_idx in zip(x_indices, y_indices):
                heatmap[y_idx, x_idx] += 1
                
        except Exception as e:
            continue
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # 绘制热力图
    im = ax.imshow(heatmap, extent=[-20, 20, -20, 20], origin='lower', 
                   cmap='hot', alpha=0.7, interpolation='gaussian')
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'Trajectory Density Heatmap - {envType}', fontsize=14, pad=10)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Visit Count', fontsize=12)
    
    # 保存图像
    save_dir = osp.join(save_path, 'dataset_overview')
    os.makedirs(save_dir, exist_ok=True)
    save_file = osp.join(save_dir, f'{envType}_heatmap.png')
    
    plt.tight_layout()
    plt.savefig(save_file, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_file}")

if __name__ == "__main__":
    envType = 'env000005'  # 指定环境
    save_path = 'predictions'
    
    print(f"Visualizing dataset for environment: {envType}")
    
    # 绘制所有轨迹在同一张图上
    plot_all_trajectories(envType, save_path)
    
    # 绘制轨迹密度热力图
    plot_trajectory_heatmap(envType, save_path)
    
    print("Visualization complete!")
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

dataset_path = 'data/dataset20/train'
# dataset_path = 'data/sim_dataset/train'
# dataset_path = 'data/sim_dataset/val'

is_dense = True  # 数据集是否为密集采样轨迹

def spline_interpolate(control_points, num_samples=100):
    """
    使用三次自然样条对控制点进行插值，生成平滑轨迹
    
    Args:
        control_points: 样条控制点，形状为(num_control_points, 3)，包含[x, y, yaw]
        num_samples: 插值生成的轨迹点数量，默认100
    
    Returns:
        插值后的轨迹点，形状为(num_samples, 3)，保证经过所有控制点
    """
    if control_points is None or len(control_points) < 2:
        print("Insufficient control points for spline interpolation")
        return None
    
    num_control = len(control_points)
    
    # 生成控制点参数（均匀参数化）
    t_control = np.linspace(0, 1, num_control)
    # 生成采样点参数
    t_samples = np.linspace(0, 1, num_samples)
    
    # 分开处理xy坐标和yaw角
    x_control = control_points[:, 0]
    y_control = control_points[:, 1]
    yaw_control = control_points[:, 2]
    
    # 对x, y使用三次样条插值
    x_interpolated = _evaluate_scalar_spline(x_control, t_samples, t_control)
    y_interpolated = _evaluate_scalar_spline(y_control, t_samples, t_control)
    
    # 对yaw进行周期性感知的样条插值
    yaw_interpolated = _evaluate_yaw_spline(yaw_control, t_samples, t_control)
    
    # 组合结果
    trajectory = np.column_stack([x_interpolated, y_interpolated, yaw_interpolated])
    
    return trajectory

def _solve_natural_cubic_M(y_values, t_control):
    """
    求解自然三次样条的二阶导数 M
    使用三对角矩阵算法（Thomas algorithm）
    
    Args:
        y_values: 控制点的y值，形状为(N,)
        t_control: 控制点的参数，形状为(N,)
    
    Returns:
        二阶导数M，形状为(N,)
    """
    N = len(y_values)
    if N < 2:
        return np.zeros(N)
    
    # 构建三对角系统 A*M = b
    # 自然边界条件：M[0] = M[-1] = 0
    h = np.diff(t_control)  # 根据实际参数计算间隔
    h = np.clip(h, 1e-6, None)  # 避免除零
    
    # 构建对角线
    diag = 2 * (h[:-1] + h[1:])
    diag = np.concatenate([[1], diag, [1]])  # 边界条件
    
    # 构建上下对角线
    upper = np.concatenate([[0], h[1:], [0]])
    lower = np.concatenate([[0], h[:-1], [0]])
    
    # 构建右侧向量
    b = np.zeros(N)
    for i in range(1, N - 1):
        b[i] = 6 * ((y_values[i + 1] - y_values[i]) / h[i] - 
                    (y_values[i] - y_values[i - 1]) / h[i - 1])
    
    # 边界条件（自然样条）
    b[0] = 0
    b[-1] = 0
    
    # 使用 Thomas 算法求解三对角系统
    M = _solve_tridiagonal(lower, diag, upper, b)
    
    return M

def _solve_tridiagonal(lower, diag, upper, b):
    """
    使用 Thomas 算法求解三对角线性系统
    
    Args:
        lower: 下对角线
        diag: 主对角线
        upper: 上对角线
        b: 右侧向量
    
    Returns:
        解向量 x
    """
    N = len(b)
    c_prime = np.zeros(N - 1)
    d_prime = np.zeros(N)
    x = np.zeros(N)
    
    # 前向消元
    c_prime[0] = upper[0] / diag[0]
    d_prime[0] = b[0] / diag[0]
    
    for i in range(1, N - 1):
        denom = diag[i] - lower[i] * c_prime[i - 1]
        c_prime[i] = upper[i] / denom
        d_prime[i] = (b[i] - lower[i] * d_prime[i - 1]) / denom
    
    d_prime[-1] = (b[-1] - lower[-1] * d_prime[-2]) / (diag[-1] - lower[-1] * c_prime[-2])
    
    # 回代
    x[-1] = d_prime[-1]
    for i in range(N - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    
    return x

def _evaluate_scalar_spline(y_control, t_eval, t_control):
    """
    对一维标量序列进行三次样条插值
    
    Args:
        y_control: 控制点的y值，形状为(N,)
        t_eval: 评估点的参数，形状为(M,)
        t_control: 控制点的参数，形状为(N,)
    
    Returns:
        插值后的y值，形状为(M,)
    """
    N = len(y_control)
    M_values = _solve_natural_cubic_M(y_control, t_control)  # 传入 t_control
    
    h = np.diff(t_control)
    h = np.clip(h, 1e-6, None)  # 避免除零
    
    # 找到每个评估点所在的区间
    idx = np.searchsorted(t_control[1:], t_eval, side='left')
    idx = np.clip(idx, 0, N - 2)
    
    # 获取区间端点
    t_k = t_control[idx]
    t_k1 = t_control[idx + 1]
    h_k = h[idx]
    dt = t_eval - t_k
    
    y_k = y_control[idx]
    y_k1 = y_control[idx + 1]
    M_k = M_values[idx]
    M_k1 = M_values[idx + 1]
    
    # 三次样条插值公式
    term1 = M_k * (t_k1 - t_eval)**3 / (6 * h_k)
    term2 = M_k1 * dt**3 / (6 * h_k)
    term3 = (y_k - M_k * h_k**2 / 6) * (t_k1 - t_eval) / h_k
    term4 = (y_k1 - M_k1 * h_k**2 / 6) * dt / h_k
    
    S = term1 + term2 + term3 + term4
    
    return S

def _evaluate_yaw_spline(yaw_control, t_eval, t_control):
    """
    对yaw角进行周期性感知的三次样条插值
    
    Args:
        yaw_control: 控制点的yaw角，形状为(N,)
        t_eval: 评估点的参数，形状为(M,)
        t_control: 控制点的参数，形状为(N,)
    
    Returns:
        插值后的yaw角，形状为(M,)，规范化到[-pi, pi]
    """
    # 展开角度序列，消除周期性跳跃
    yaw_unwrapped = _unwrap_angles(yaw_control)
    
    # 对展开后的角度进行标量样条插值
    yaw_interpolated_unwrapped = _evaluate_scalar_spline(yaw_unwrapped, t_eval, t_control)
    
    # 将插值结果重新规范化到 [-π, π]
    yaw_interpolated = np.arctan2(np.sin(yaw_interpolated_unwrapped), 
                                   np.cos(yaw_interpolated_unwrapped))
    
    return yaw_interpolated

def _unwrap_angles(angles):
    """
    展开角度序列，消除周期性跳跃
    
    Args:
        angles: 角度序列，形状为(N,)
    
    Returns:
        展开后的角度序列，形状为(N,)
    """
    if len(angles) <= 1:
        return angles.copy()
    
    unwrapped = np.zeros_like(angles)
    unwrapped[0] = angles[0]
    
    for i in range(1, len(angles)):
        diff = angles[i] - angles[i-1]
        # 规范化角度差到 [-π, π]
        diff = np.arctan2(np.sin(diff), np.cos(diff))
        unwrapped[i] = unwrapped[i-1] + diff
    
    return unwrapped

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
    cnt = 0
    for traj_file in traj_files:
        # cnt += 1
        # if cnt % 50 == 0:
        #     break  # 仅绘制前50条轨迹以防图像过于拥挤
        try:
            traj_path = osp.join(envFolder, traj_file)
            with open(traj_path, 'rb') as f:
                traj_data = pickle.load(f)
                trajectory = traj_data['path']  # [N, 3] - 这些是控制点
            
            # 如果轨迹不是密集采样，使用三次样条插值生成平滑轨迹（100个点）
            if is_dense is not True:
                trajectory_smooth = spline_interpolate(trajectory, num_samples=100)
            else:
                trajectory_smooth = trajectory
            
            start_pos = trajectory_smooth[0, :]
            goal_pos = trajectory_smooth[-1, :]
            
            # 绘制轨迹线 - 使用蓝色，带透明度
            ax.plot(trajectory_smooth[:, 0], trajectory_smooth[:, 1], 
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
                trajectory = traj_data['path']  # [N, 3] - 控制点
            
            # 使用三次样条插值生成平滑轨迹（100个点）
            trajectory_smooth = spline_interpolate(trajectory, num_samples=100)
            
            if trajectory_smooth is None:
                trajectory_smooth = trajectory
            
            # 将坐标转换为网格索引
            x_indices = ((trajectory_smooth[:, 0] + 20) / 40 * grid_size).astype(int)
            y_indices = ((trajectory_smooth[:, 1] + 20) / 40 * grid_size).astype(int)
            
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
    envType = 'env000000'  # 指定环境
    save_path = 'predictions'
    
    print(f"Visualizing dataset for environment: {envType}")
    
    # 绘制所有轨迹在同一张图上
    plot_all_trajectories(envType, save_path)
    
    # 绘制轨迹密度热力图
    plot_trajectory_heatmap(envType, save_path)
    
    print("Visualization complete!")
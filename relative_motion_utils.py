"""
相对位移转换工具
"""
import numpy as np
import torch

def trajectory_to_relative_motion(trajectory):
    """
    将绝对坐标轨迹转换为相对位移
    
    Args:
        trajectory: (N, 3) numpy array or torch tensor, [x, y, θ]
        
    Returns:
        relative_motion: (N-1, 3) 相对位移 [Δx, Δy, Δθ]
        start_pose: (3,) 起点位置
    """
    is_torch = isinstance(trajectory, torch.Tensor)
    
    if is_torch:
        # PyTorch版本
        delta = trajectory[1:] - trajectory[:-1]
        # 归一化角度到 [-π, π]
        delta[:, 2] = torch.atan2(torch.sin(delta[:, 2]), torch.cos(delta[:, 2]))
        return delta, trajectory[0]
    else:
        # NumPy版本
        delta = trajectory[1:] - trajectory[:-1]
        delta[:, 2] = np.arctan2(np.sin(delta[:, 2]), np.cos(delta[:, 2]))
        return delta, trajectory[0]


def relative_motion_to_trajectory(relative_motion, start_pose):
    """
    将相对位移转换回绝对坐标轨迹
    
    Args:
        relative_motion: (N-1, 3) [Δx, Δy, Δθ]
        start_pose: (3,) [x_0, y_0, θ_0]
        
    Returns:
        trajectory: (N, 3) [x, y, θ]
    """
    is_torch = isinstance(relative_motion, torch.Tensor)
    
    if is_torch:
        N = relative_motion.shape[0] + 1
        trajectory = torch.zeros(N, 3, dtype=relative_motion.dtype, device=relative_motion.device)
        trajectory[0] = start_pose
        
        for i in range(N-1):
            trajectory[i+1] = trajectory[i] + relative_motion[i]
            # 归一化角度
            trajectory[i+1, 2] = torch.atan2(
                torch.sin(trajectory[i+1, 2]),
                torch.cos(trajectory[i+1, 2])
            )
    else:
        N = relative_motion.shape[0] + 1
        trajectory = np.zeros((N, 3))
        trajectory[0] = start_pose
        
        for i in range(N-1):
            trajectory[i+1] = trajectory[i] + relative_motion[i]
            trajectory[i+1, 2] = np.arctan2(
                np.sin(trajectory[i+1, 2]),
                np.cos(trajectory[i+1, 2])
            )
    
    return trajectory


def compute_relative_motion_statistics(dataset_path):
    """
    计算数据集中相对位移的统计信息（用于归一化）
    """
    import pickle
    from os import path as osp
    import os
    
    all_deltas = []
    
    # 遍历所有环境
    for env_name in os.listdir(dataset_path):
        env_folder = osp.join(dataset_path, env_name)
        if not osp.isdir(env_folder):
            continue
        
        # 遍历所有路径
        for path_file in os.listdir(env_folder):
            if not path_file.startswith('path_'):
                continue
            
            with open(osp.join(env_folder, path_file), 'rb') as f:
                data = pickle.load(f)
                trajectory = data['path']
                
                delta, _ = trajectory_to_relative_motion(trajectory)
                all_deltas.append(delta)
    
    all_deltas = np.concatenate(all_deltas, axis=0)
    
    stats = {
        'delta_xy_mean': all_deltas[:, :2].mean(axis=0),
        'delta_xy_std': all_deltas[:, :2].std(axis=0),
        'delta_xy_max': np.abs(all_deltas[:, :2]).max(),
        'delta_theta_mean': all_deltas[:, 2].mean(),
        'delta_theta_std': all_deltas[:, 2].std(),
        'delta_theta_max': np.abs(all_deltas[:, 2]).max()
    }
    
    print("相对位移统计:")
    print(f"  Δx,Δy 均值: {stats['delta_xy_mean']}")
    print(f"  Δx,Δy 标准差: {stats['delta_xy_std']}")
    print(f"  Δx,Δy 最大值: {stats['delta_xy_max']:.2f}")
    print(f"  Δθ 均值: {stats['delta_theta_mean']:.4f}")
    print(f"  Δθ 标准差: {stats['delta_theta_std']:.4f}")
    print(f"  Δθ 最大值: {stats['delta_theta_max']:.4f}")
    
    return stats
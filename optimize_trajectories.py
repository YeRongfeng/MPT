"""
optimize_trajectories.py - 优化轨迹数据集并保存到新环境

【功能说明】
1. 从指定环境导入轨迹数据
2. 使用梯度优化器对轨迹进行优化
3. 将优化后的轨迹保存到新环境
4. 保持原有的轨迹文件结构和元数据（除path和cost外）
5. 复制原地图到新环境

【优化方法】
- 使用grad_optimizer中的optimize_control_points函数
- 基于稳定性代价地图进行优化
- 保留起点和终点位置
"""

import os
import numpy as np
import torch
import pickle
import shutil
from tqdm import tqdm

from grad_optimizer import (
    optimize_control_points_multistep,
    cost_on_dense_trajectory
)
from bspline_utils import (
    fit_bspline_least_squares,
    reconstruct_from_control_points,
    DifferentiableBSpline
)

# =================== 配置参数 ===================
# 支持处理多个数据文件夹（同时处理train和val）
source_data_folders = [
    '/home/yrf/MPT/data/sim_dataset/train',
    '/home/yrf/MPT/data/sim_dataset/val',
]

# 源环境名称列表（None表示处理所有环境）
source_env_names = ['env000012', 'env000013']  # 指定特定环境列表
# source_env_names = None  # 处理所有环境

# 新环境后缀（源名称_后缀）
dest_env_suffix = '_optimized'

# 目标基础文件夹（与源文件夹结构相同）
dest_base_folders = [
    '/home/yrf/MPT/data/sim_dataset/train',
    '/home/yrf/MPT/data/sim_dataset/val',
]

# 优化参数
optimization_config = {
    'iterations': 20,
    'lr': 0.01,
    'grad_clip_norm': 1.0,
    'verbose': False
}

# Map配置
map_info = {
    'origin': (-20.0, -20.0, -np.pi),
    'resolution': 0.4,
    'size': (100, 100, 36)  # (W, H, D)
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# =================== 主程序 ===================

def trajectory_to_control_points(trajectory, num_middle_points=24):
    """
    将轨迹转换为B样条控制点（仅返回中间控制点，不包含起终点）
    
    Args:
        trajectory: (N, 3) 轨迹张量 [x, y, theta]
        num_middle_points: 中间控制点数量（默认24，不包含起终点）
    
    Returns:
        middle_control_points: (num_middle_points, 2) 中间控制点张量（只包含x,y，不包含起终点）
    """
    # 转换为numpy
    if torch.is_tensor(trajectory):
        trajectory_np = trajectory.cpu().numpy()
    else:
        trajectory_np = trajectory
    
    # 拟合完整的26个控制点（包含起终点）
    full_cp, _, _ = fit_bspline_least_squares(
        trajectory_np[:, :2],  # 只使用x,y
        num_control_points=num_middle_points + 2,  # 26个控制点
        degree=3
    )
    
    # 只保留中间24个控制点，排除起终点
    middle_cp = full_cp[1:-1]  # 去掉第一个和最后一个
    
    return middle_cp  # (num_middle_points, 2)


def control_points_to_trajectory(control_points, start_xy, goal_xy, num_output_points=100):
    """
    从控制点重建密集轨迹
    
    Args:
        control_points: (num_control_points, 2) 控制点
        start_xy: (2,) 起点xy坐标
        goal_xy: (2,) 终点xy坐标
        num_output_points: 重建后的轨迹点数量
    
    Returns:
        trajectory: (num_output_points, 2) 重建的轨迹（xy坐标）
    """
    if torch.is_tensor(control_points):
        control_points = control_points.cpu().numpy()
    
    # 拼接完整的控制点（包括起终点）
    full_cp = np.vstack([
        start_xy.reshape(1, -1),
        control_points,
        goal_xy.reshape(1, -1)
    ])
    
    # 重建轨迹
    traj = reconstruct_from_control_points(
        full_cp,
        num_output_points=num_output_points,
        degree=3
    )
    
    return traj  # (num_output_points, 2)


def reconstruct_trajectory_with_yaw(trajectory_xy, start_yaw, goal_yaw):
    """
    从xy坐标重建完整轨迹（包括yaw角）
    
    Args:
        trajectory_xy: (N, 2) xy坐标
        start_yaw: float 起点yaw
        goal_yaw: float 终点yaw
    
    Returns:
        trajectory: (N, 3) 完整轨迹 [x, y, yaw]
    """
    N = trajectory_xy.shape[0]
    trajectory = np.zeros((N, 3))
    trajectory[:, :2] = trajectory_xy
    
    # 从xy差分计算yaw
    dxy = np.diff(trajectory_xy, axis=0, prepend=trajectory_xy[0:1])  # (N, 2)
    trajectory[1:, 2] = np.arctan2(dxy[1:, 1], dxy[1:, 0])
    trajectory[0, 2] = start_yaw
    
    return trajectory


def optimize_environment_trajectories(source_env_path, dest_env_path, map_info, optimization_config, device):
    """
    优化单个环境中的所有轨迹
    
    Args:
        source_env_path: str 源环境文件夹路径
        dest_env_path: str 目标环境文件夹路径
        map_info: dict 地图信息
        optimization_config: dict 优化配置
        device: 计算设备
    
    Returns:
        stats: dict 统计信息
    """
    print(f'\n处理环境: {source_env_path}')
    
    # 创建目标环境文件夹
    os.makedirs(dest_env_path, exist_ok=True)
    
    # 复制地图文件和其他非.p文件
    print(f'  复制环境文件...')
    for filename in os.listdir(source_env_path):
        source_file_path = os.path.join(source_env_path, filename)
        dest_file_path = os.path.join(dest_env_path, filename)
        
        # 跳过目录和以.p后缀的轨迹文件
        if os.path.isdir(source_file_path) or filename.endswith('.p'):
            continue
        
        # 复制其他文件（如map.p和配置文件等）
        if os.path.isfile(source_file_path):
            shutil.copy(source_file_path, dest_file_path)
    
    # 加载地图数据获取cost map
    source_map_path = os.path.join(source_env_path, 'map.p')
    if not os.path.exists(source_map_path):
        print(f'  警告: 未找到地图文件')
        return None
    
    with open(source_map_path, 'rb') as f:
        map_data = pickle.load(f)
    
    # 读取stability_map.npz文件作为cost地图
    stability_map_path = os.path.join(source_env_path, 'stability_map.npz')
    if not os.path.exists(stability_map_path):
        print(f'  警告: 未找到稳定性地图文件: {stability_map_path}')
        return None
    
    # 加载stability_map
    stability_map_data = np.load(stability_map_path)
    cost_map = stability_map_data['stability_map']  # 直接使用作为cost_map
    
    # 反转代价地图：稳定性越高，代价越低
    cost_map = 1.0 - cost_map
    cost_map_tensor = torch.from_numpy(cost_map).float().to(device)
    
    # 获取轨迹文件列表
    path_files = sorted([f for f in os.listdir(source_env_path) if f.startswith('path_') and f.endswith('.p')])
    print(f'  找到 {len(path_files)} 条轨迹')
    
    if not path_files:
        print(f'  此环境中没有轨迹文件，跳过')
        return None
    
    stats = {
        'total': len(path_files),
        'optimized': 0,
        'errors': 0,
        'original_costs': [],
        'optimized_costs': [],
        'cost_improvements': []
    }
    
    # 遍历每个轨迹文件
    for path_file in tqdm(path_files, desc=f'优化 {os.path.basename(source_env_path)} 中的轨迹'):
        try:
            source_path = os.path.join(source_env_path, path_file)
            dest_path = os.path.join(dest_env_path, path_file)
            
            # 读取轨迹数据
            with open(source_path, 'rb') as f:
                path_data = pickle.load(f)
            
            trajectory_orig = path_data['path'].astype(np.float32)  # (N, 3) [x, y, yaw]
            
            # 【数据验证】
            if trajectory_orig.shape[1] < 3:
                print(f'    警告: {path_file} 轨迹维度不足，跳过')
                stats['errors'] += 1
                continue
            
            # 提取起终点信息
            start_pose = trajectory_orig[0, :3]  # (3,) [x, y, yaw]
            goal_pose = trajectory_orig[-1, :3]  # (3,) [x, y, yaw]
            
            # 从数据集中读取原始轨迹的cost（已保存）
            original_cost = path_data.get('cost', 0.0)
            
            # 创建tensor用于优化
            start_pose_tensor = torch.from_numpy(start_pose).unsqueeze(0).float().to(device)
            goal_pose_tensor = torch.from_numpy(goal_pose).unsqueeze(0).float().to(device)
            
            # 转换为控制点
            middle_cp = trajectory_to_control_points(trajectory_orig, num_middle_points=24)  # (24, 2)
            middle_cp_tensor = torch.from_numpy(middle_cp).float().to(device).unsqueeze(0)  # (1, 24, 2)
            
            # 优化控制点
            optimized_middle_cp, cost_history = optimize_control_points_multistep(
                middle_control_points=middle_cp_tensor,
                start_pose=start_pose_tensor,
                goal_pose=goal_pose_tensor,
                stability_cost_map=cost_map_tensor,
                map_info=map_info,
                iterations=optimization_config['iterations'],
                lr=optimization_config['lr'],
                grad_clip_norm=optimization_config['grad_clip_norm'],
                verbose=optimization_config['verbose'],
                device=device
            )
            
            # 重建优化后的轨迹（xy坐标）
            trajectory_opt_xy = control_points_to_trajectory(
                optimized_middle_cp.squeeze(0),  # 去掉batch维度：(1, 24, 2) -> (24, 2)
                start_pose[:2],
                goal_pose[:2],
                num_output_points=trajectory_orig.shape[0]
            )
            
            # 重建完整轨迹（包括yaw）
            trajectory_opt = reconstruct_trajectory_with_yaw(
                trajectory_opt_xy,
                start_pose[2],
                goal_pose[2]
            )
            
            # 计算优化后的cost
            trajectory_opt_tensor = torch.from_numpy(trajectory_opt).unsqueeze(0).float().to(device)
            with torch.no_grad():
                optimized_cost = cost_on_dense_trajectory(
                    trajectory_opt_tensor,
                    start_pose_tensor,
                    goal_pose_tensor,
                    cost_map_tensor,
                    map_info,
                    device=device
                ).item() * 1e2  # 放大cost数值
            
            # 更新轨迹数据：保留原有结构，仅更新path和cost
            path_data_new = path_data.copy()
            path_data_new['path'] = trajectory_opt.astype(np.float32)
            path_data_new['cost'] = float(optimized_cost)
            
            # 保存优化后的文件
            with open(dest_path, 'wb') as f:
                pickle.dump(path_data_new, f)
            
            stats['optimized'] += 1
            stats['original_costs'].append(original_cost)
            stats['optimized_costs'].append(optimized_cost)
            stats['cost_improvements'].append((original_cost - optimized_cost) / (original_cost + 1e-8))
        
        except Exception as e:
            print(f'    错误处理 {path_file}: {str(e)}')
            stats['errors'] += 1
    
    return stats


def main():
    """主程序"""
    print('=' * 80)
    print('轨迹优化器')
    print('=' * 80)
    print(f'源环境: {source_env_names if source_env_names else "所有环境"}')
    print(f'源文件夹: {source_data_folders}')
    print(f'目标环境后缀: {dest_env_suffix}')
    print(f'目标文件夹: {dest_base_folders}')
    print(f'计算设备: {device}')
    print('=' * 80)
    
    # 统计信息汇总
    global_stats = {
        'total_folders': len(source_data_folders),
        'processed_folders': 0,
        'total_envs': 0,
        'processed_envs': 0,
        'total_trajectories': 0,
        'optimized_trajectories': 0,
        'failed_trajectories': 0,
        'all_improvements': [],
        'all_original_costs': [],
        'all_optimized_costs': []
    }
    
    # 处理每个数据文件夹对
    for src_folder, dest_folder in zip(source_data_folders, dest_base_folders):
        print(f'\n处理数据文件夹: {src_folder} -> {dest_folder}')
        
        if not os.path.exists(src_folder):
            print(f'警告: 源文件夹不存在 {src_folder}')
            continue
        
        # 获取环境列表
        if source_env_names is None:
            # 处理所有环境
            all_envs = [d for d in os.listdir(src_folder) if os.path.isdir(os.path.join(src_folder, d)) and d.startswith('env')]
            envs_to_process = sorted(all_envs)
        else:
            envs_to_process = source_env_names
        
        print(f'  要处理的环境数: {len(envs_to_process)}')
        
        # 处理每个环境
        for source_env_name in tqdm(envs_to_process, desc=f'优化环境 ({os.path.basename(src_folder)})'):
            source_env_path = os.path.join(src_folder, source_env_name)
            dest_env_name = source_env_name + dest_env_suffix
            dest_env_path = os.path.join(dest_folder, dest_env_name)
            
            if not os.path.exists(source_env_path):
                print(f'  警告: 源环境不存在 {source_env_path}')
                continue
            
            # 处理环境
            stats = optimize_environment_trajectories(
                source_env_path,
                dest_env_path,
                map_info,
                optimization_config,
                device
            )
            
            if stats is None:
                print(f'  警告: 处理失败 {source_env_path}')
                continue
            
            # 更新全局统计
            global_stats['total_envs'] += 1
            global_stats['processed_envs'] += 1 if stats['optimized'] > 0 else 0
            global_stats['total_trajectories'] += stats['total']
            global_stats['optimized_trajectories'] += stats['optimized']
            global_stats['failed_trajectories'] += stats['errors']
            global_stats['all_improvements'].extend(stats['cost_improvements'])
            global_stats['all_original_costs'].extend(stats['original_costs'])
            global_stats['all_optimized_costs'].extend(stats['optimized_costs'])
        
        global_stats['processed_folders'] += 1
    
    # 显示全局统计
    print('\n' + '=' * 80)
    print('优化总体统计')
    print('=' * 80)
    print(f'处理的文件夹数: {global_stats["processed_folders"]}/{global_stats["total_folders"]}')
    print(f'处理的环境数: {global_stats["processed_envs"]}/{global_stats["total_envs"]}')
    print(f'总轨迹数: {global_stats["total_trajectories"]}')
    print(f'优化成功: {global_stats["optimized_trajectories"]}')
    print(f'处理失败: {global_stats["failed_trajectories"]}')
    
    # 原始cost统计
    if global_stats['all_original_costs']:
        original_costs = np.array(global_stats['all_original_costs'])
        print(f'\n原始轨迹成本统计:')
        print(f'  - 最小: {original_costs.min():.6f}')
        print(f'  - 最大: {original_costs.max():.6f}')
        print(f'  - 平均: {original_costs.mean():.6f}')
        print(f'  - 中位数: {np.median(original_costs):.6f}')
        print(f'  - 标准差: {original_costs.std():.6f}')
    
    # 优化后cost统计
    if global_stats['all_optimized_costs']:
        optimized_costs = np.array(global_stats['all_optimized_costs'])
        print(f'\n优化后轨迹成本统计:')
        print(f'  - 最小: {optimized_costs.min():.6f}')
        print(f'  - 最大: {optimized_costs.max():.6f}')
        print(f'  - 平均: {optimized_costs.mean():.6f}')
        print(f'  - 中位数: {np.median(optimized_costs):.6f}')
        print(f'  - 标准差: {optimized_costs.std():.6f}')
    
    # 成本改进统计
    if global_stats['all_improvements']:
        improvements = np.array(global_stats['all_improvements'])
        original_costs = np.array(global_stats['all_original_costs'])
        optimized_costs = np.array(global_stats['all_optimized_costs'])
        
        print(f'\n成本改进统计:')
        print(f'  - 平均改进率: {improvements.mean() * 100:.2f}%')
        print(f'  - 最大改进率: {improvements.max() * 100:.2f}%')
        print(f'  - 最小改进率: {improvements.min() * 100:.2f}%')
        print(f'  - 标准差: {improvements.std() * 100:.2f}%')
        print(f'  - 平均cost降低值: {(original_costs.mean() - optimized_costs.mean()):.6f}')
        print(f'  - 总cost降低值: {(original_costs.sum() - optimized_costs.sum()):.6f}')
    
    print(f'\n✓ 优化完成! 所有结果已保存')


if __name__ == '__main__':
    main()

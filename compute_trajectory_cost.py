"""
compute_trajectory_cost.py - 计算数据集中所有轨迹的cost，并更新到数据文件中

【功能说明】
1. 遍历数据集中的所有轨迹文件（path_*.p）
2. 对每条轨迹计算稳定性代价
3. 将cost保存到轨迹文件的字段中
4. 支持批处理和显示进度

【成本计算原理】
- 使用梯度优化器中的 cost_on_dense_trajectory 函数
- 基于地面稳定性地图（yaw stability map）计算
- cost越低表示轨迹越稳定
"""

import os
import numpy as np
import torch
import pickle
from tqdm import tqdm

from grad_optimizer import cost_on_dense_trajectory

# =================== 配置参数 ===================
# 支持处理多个数据文件夹（同时处理train和val）
data_folders = [
    '/home/yrf/MPT/data/sim_dataset/train',
    '/home/yrf/MPT/data/sim_dataset/val',
]

# 指定要处理的环境（None表示处理所有）
# env_list = None  # 处理所有环境
env_list = ['env000012', 'env000013']  # 或指定特定环境

# Map配置
map_info = {
    'origin': (-20.0, -20.0, -np.pi),
    'resolution': 0.4,
    'size': (100, 100, 36)  # (W, H, D)
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# =================== 主程序 ===================

def compute_trajectory_cost(trajectory, cost_map, map_info, device='cpu'):
    """
    计算单条轨迹的cost
    
    Args:
        trajectory: (N, 3) 轨迹数组 [x, y, yaw] (支持numpy数组或Tensor)
        cost_map: (D, H, W) 稳定性代价地图
        map_info: dict 地图信息
        device: 计算设备
    
    Returns:
        cost: float 轨迹的总成本
    """
    # 转换为tensor（如果不是的话）
    if not torch.is_tensor(trajectory):
        trajectory = torch.from_numpy(trajectory).float()
    else:
        trajectory = trajectory.float()
    
    if not torch.is_tensor(cost_map):
        cost_map = torch.from_numpy(cost_map).float()
    else:
        cost_map = cost_map.float()
    
    # 转移到指定设备
    trajectory = trajectory.to(device)
    cost_map = cost_map.to(device)
    
    # 添加batch维度
    trajectory_batch = trajectory.unsqueeze(0)  # (1, N, 3)
    
    # 虚拟start_pose和goal_pose（仅用于函数调用）
    start_pose = trajectory[0:1, :3]  # (1, 3)
    goal_pose = trajectory[-1:, :3]   # (1, 3)
    
    # 计算cost
    with torch.no_grad():
        cost = cost_on_dense_trajectory(
            trajectory_batch,
            start_pose,
            goal_pose,
            cost_map,
            map_info,
            device=device
        )
    
    # 放大cost数值（scale=1e2）
    return float(cost.cpu().item()) * 1e2


def process_environment(env_path, map_info, device):
    """
    处理单个环境中的所有轨迹
    
    Args:
        env_path: str 环境文件夹路径
        map_info: dict 地图信息
        device: 计算设备
    
    Returns:
        stats: dict 统计信息
    """
    print(f'\n处理环境: {env_path}')
    
    # 读取地图数据
    map_path = os.path.join(env_path, 'map.p')
    if not os.path.exists(map_path):
        print(f'  未找到地图文件: {map_path}')
        return None
    
    with open(map_path, 'rb') as f:
        map_data = pickle.load(f)
    
    # 读取stability_map.npz文件作为cost地图
    stability_map_path = os.path.join(env_path, 'stability_map.npz')
    if not os.path.exists(stability_map_path):
        print(f'  未找到稳定性地图文件: {stability_map_path}')
        return None
    
    # 加载stability_map
    stability_map_data = np.load(stability_map_path)
    cost_map = stability_map_data['stability_map']  # 直接使用作为cost_map
    
    # 反转代价地图：稳定性越高，代价越低
    cost_map = 1.0 - cost_map
    
    # 获取轨迹文件列表
    path_files = sorted([f for f in os.listdir(env_path) if f.startswith('path_') and f.endswith('.p')])
    print(f'  找到 {len(path_files)} 条轨迹')
    
    if not path_files:
        print(f'  此环境中没有轨迹文件，跳过')
        return None
    
    stats = {
        'total': len(path_files),
        'processed': 0,
        'errors': 0,
        'min_cost': float('inf'),
        'max_cost': float('-inf'),
        'mean_cost': 0.0,
        'costs': []
    }
    
    # 遍历每个轨迹文件
    for path_file in tqdm(path_files, desc=f'处理 {os.path.basename(env_path)} 中的轨迹'):
        try:
            path_path = os.path.join(env_path, path_file)
            
            # 读取轨迹数据
            with open(path_path, 'rb') as f:
                path_data = pickle.load(f)
            
            trajectory = path_data['path']  # (N, 3) [x, y, yaw]
            
            # 【数据验证】
            if trajectory.shape[1] < 3:
                print(f'    警告: {path_file} 轨迹维度不足，跳过')
                stats['errors'] += 1
                continue
            
            # 计算cost
            cost = compute_trajectory_cost(trajectory, cost_map, map_info, device)
            
            # 【异常值处理】
            if np.isnan(cost) or np.isinf(cost):
                print(f'    警告: {path_file} 计算得到异常cost: {cost}，设置为1.0')
                cost = 1.0
                stats['errors'] += 1
            
            # 更新轨迹文件：添加cost字段
            path_data['cost'] = float(cost)
            
            # 保存更新后的文件
            with open(path_path, 'wb') as f:
                pickle.dump(path_data, f)
            
            stats['processed'] += 1
            stats['costs'].append(cost)
            stats['min_cost'] = min(stats['min_cost'], cost)
            stats['max_cost'] = max(stats['max_cost'], cost)
        
        except Exception as e:
            print(f'    错误处理 {path_file}: {str(e)}')
            stats['errors'] += 1
    
    # 计算统计信息
    if stats['costs']:
        stats['mean_cost'] = float(np.mean(stats['costs']))
    
    return stats


def main():
    """主程序"""
    print('=' * 80)
    print('轨迹成本计算器')
    print('=' * 80)
    print(f'数据文件夹: {data_folders}')
    print(f'地图信息: origin={map_info["origin"]}, resolution={map_info["resolution"]}')
    print(f'计算设备: {device}')
    print('=' * 80)
    
    # 统计信息汇总
    total_stats = {
        'processed_envs': 0,
        'total_trajectories': 0,
        'processed_trajectories': 0,
        'total_errors': 0,
        'all_costs': []
    }
    
    # 处理每个数据文件夹
    for data_folder in data_folders:
        print(f'\n处理数据文件夹: {data_folder}')
        
        if not os.path.exists(data_folder):
            print(f'警告: 数据文件夹不存在 {data_folder}')
            continue
        
        # 获取环境列表
        if env_list is None:
            # 处理所有环境
            all_envs = [d for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d)) and d.startswith('env')]
            env_list_to_process = sorted(all_envs)
        else:
            env_list_to_process = env_list
        
        print(f'要处理的环境数: {len(env_list_to_process)}')
        
        # 处理每个环境
        for env_name in tqdm(env_list_to_process, desc=f'处理环境 ({os.path.basename(data_folder)})'):
            env_path = os.path.join(data_folder, env_name)
            
            if not os.path.exists(env_path):
                print(f'警告: 环境路径不存在 {env_path}')
                continue
            
            stats = process_environment(env_path, map_info, device)
            
            if stats is None:
                continue
            
            # 更新总统计
            total_stats['processed_envs'] += 1
            total_stats['total_trajectories'] += stats['total']
            total_stats['processed_trajectories'] += stats['processed']
            total_stats['total_errors'] += stats['errors']
            total_stats['all_costs'].extend(stats['costs'])
            
            # 显示此环境的统计
            print(f'\n  ✓ 环境 {env_name} 统计:')
            print(f'    - 总轨迹数: {stats["total"]}')
            print(f'    - 处理成功: {stats["processed"]}')
            print(f'    - 处理失败: {stats["errors"]}')
            print(f'    - 最小cost: {stats["min_cost"]:.6f}')
            print(f'    - 最大cost: {stats["max_cost"]:.6f}')
            print(f'    - 平均cost: {stats["mean_cost"]:.6f}')
    
    # 显示总体统计
    print('\n' + '=' * 80)
    print('总体统计')
    print('=' * 80)
    print(f'处理的环境数: {total_stats["processed_envs"]}')
    print(f'总轨迹数: {total_stats["total_trajectories"]}')
    print(f'成功处理: {total_stats["processed_trajectories"]}')
    print(f'处理失败: {total_stats["total_errors"]}')
    
    if total_stats['all_costs']:
        print(f'\n成本统计:')
        print(f'  - 最小cost: {min(total_stats["all_costs"]):.6f}')
        print(f'  - 最大cost: {max(total_stats["all_costs"]):.6f}')
        print(f'  - 平均cost: {np.mean(total_stats["all_costs"]):.6f}')
        print(f'  - 中位数cost: {np.median(total_stats["all_costs"]):.6f}')
        print(f'  - 标准差: {np.std(total_stats["all_costs"]):.6f}')
    
    print('\n✓ 数据集成本计算完成!')


if __name__ == '__main__':
    main()

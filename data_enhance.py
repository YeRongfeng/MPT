'''
data_enhance.py - 数据增强脚本（通过地图变换生成更多训练数据）
'''

import os
import numpy as np
import pickle
import shutil
from tqdm import tqdm

def apply_map_transform(map_tensor, transform_type):
    """
    对地图应用指定的变换
    
    Args:
        map_tensor: [H, W, C] 地图张量
        transform_type: 变换类型 (0-7)
            0: 原始 (无变换)
            1: x轴翻转
            2: y轴翻转
            3: xy互换
            4: x轴翻转 + y轴翻转
            5: x轴翻转 + xy互换
            6: y轴翻转 + xy互换
            7: x轴翻转 + y轴翻转 + xy互换
    
    Returns:
        transformed_map: 变换后的地图张量
    """
    map_copy = map_tensor.copy()
    
    # 解析变换类型
    flip_x = transform_type & 1  # 最低位
    flip_y = (transform_type >> 1) & 1  # 第二位
    swap_xy = (transform_type >> 2) & 1  # 第三位
    
    # 应用变换
    if swap_xy:
        map_copy = np.transpose(map_copy, (1, 0, 2))  # 交换x和y维度
    
    if flip_x:
        map_copy = np.flip(map_copy, axis=0)  # 沿x轴翻转
    
    if flip_y:
        map_copy = np.flip(map_copy, axis=1)  # 沿y轴翻转
    
    return map_copy

def apply_trajectory_transform(trajectory, transform_type, map_size=100):
    """
    对轨迹应用对应的变换
    
    Args:
        trajectory: [N, 3] 轨迹数据 [x, y, yaw]
        transform_type: 变换类型 (0-7)
        map_size: 地图尺寸（假设为正方形）
    
    Returns:
        transformed_trajectory: 变换后的轨迹
    """
    traj_copy = trajectory.copy()
    x, y, yaw = traj_copy[:, 0], traj_copy[:, 1], traj_copy[:, 2]
    
    # 解析变换类型
    flip_x = transform_type & 1
    flip_y = (transform_type >> 1) & 1
    swap_xy = (transform_type >> 2) & 1
    
    # 应用变换（注意：地图坐标范围是[-20, 20]）
    map_range = 40  # 地图范围：-20到20
    
    if swap_xy:
        x, y = y, x  # 交换x和y坐标
        yaw = yaw + np.pi/2  # 旋转90度
    
    if flip_x:
        x = -x  # x轴翻转
        yaw = np.pi - yaw  # yaw角度相应调整
    
    if flip_y:
        y = -y  # y轴翻转
        yaw = -yaw  # yaw角度相应调整
    
    # 将yaw角度标准化到[-π, π]范围
    yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
    
    traj_copy[:, 0] = x
    traj_copy[:, 1] = y
    traj_copy[:, 2] = yaw
    
    return traj_copy

def enhance_data(data_folder):
    """
    对指定文件夹中的数据进行增强
    
    Args:
        data_folder: 数据文件夹路径
    """
    # 获取现有环境列表
    existing_envs = [d for d in os.listdir(data_folder) 
                    if d.startswith('env') and os.path.isdir(os.path.join(data_folder, d))]
    existing_envs.sort()
    
    print(f"Found {len(existing_envs)} existing environments: {existing_envs}")
    
    # 确定下一个可用的环境编号
    if existing_envs:
        last_env_num = int(existing_envs[-1][3:])  # 提取数字部分
        next_env_num = last_env_num + 1
    else:
        next_env_num = 0
    
    # 对每个现有环境进行数据增强
    for env_name in existing_envs:
        env_path = os.path.join(data_folder, env_name)
        print(f"\nProcessing environment: {env_name}")
        
        # 读取原始地图数据
        map_path = os.path.join(env_path, 'map.p')
        if not os.path.exists(map_path):
            print(f"Map file not found in {env_name}, skipping...")
            continue
            
        with open(map_path, 'rb') as f:
            map_data = pickle.load(f)
        
        # 读取所有轨迹文件
        path_files = [f for f in os.listdir(env_path) 
                     if f.startswith('path_') and f.endswith('.p')]
        
        if not path_files:
            print(f"No trajectory files found in {env_name}, skipping...")
            continue
        
        print(f"Found {len(path_files)} trajectory files")
        
        # 对每种变换（除了原始状态0）创建新环境
        for transform_type in range(1, 8):
            new_env_name = f'env{next_env_num:06d}'
            new_env_path = os.path.join(data_folder, new_env_name)
            
            print(f"Creating augmented environment: {new_env_name} (transform type {transform_type})")
            
            # 创建新环境目录
            os.makedirs(new_env_path, exist_ok=True)
            
            # 变换并保存地图
            transformed_map = apply_map_transform(map_data['tensor'], transform_type)
            new_map_data = {
                'tensor': transformed_map,
                'original_env': env_name,
                'transform_type': transform_type
            }
            
            new_map_path = os.path.join(new_env_path, 'map.p')
            with open(new_map_path, 'wb') as f:
                pickle.dump(new_map_data, f)
            
            # 变换并保存所有轨迹文件
            for path_file in tqdm(path_files, desc=f"Processing trajectories for {new_env_name}"):
                path_path = os.path.join(env_path, path_file)
                
                with open(path_path, 'rb') as f:
                    path_data = pickle.load(f)
                
                # 变换轨迹
                if 'path' in path_data:
                    transformed_trajectory = apply_trajectory_transform(
                        path_data['path'], transform_type
                    )
                    
                    # 创建新的轨迹数据
                    new_path_data = {
                        'path': transformed_trajectory,
                        'map_name': new_env_name,
                        'original_env': path_data.get('map_name', env_name),
                        'transform_type': transform_type
                    }
                    
                    # 如果原数据有valid标签，保留它
                    if 'valid' in path_data:
                        new_path_data['valid'] = path_data['valid']
                    
                    # 保存变换后的轨迹
                    new_path_path = os.path.join(new_env_path, path_file)
                    with open(new_path_path, 'wb') as f:
                        pickle.dump(new_path_data, f)
            
            next_env_num += 1
    
    print(f"\nData enhancement completed. Total environments: {next_env_num}")

if __name__ == "__main__":
    # 设置数据文件夹路径
    data_folders = [
        '/home/yrf/MPT/data/sim_dataset/train',
        '/home/yrf/MPT/data/sim_dataset/val'
    ]
    
    for data_folder in data_folders:
        if os.path.exists(data_folder):
            print(f"Enhancing data in: {data_folder}")
            enhance_data(data_folder)
        else:
            print(f"Data folder not found: {data_folder}")
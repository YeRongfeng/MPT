'''
data_clean.py - 数据清洗脚本（用于剔除有问题的轨迹，并给轨迹打标签）
'''

import os
import numpy as np
import torch
import pickle

from tqdm import tqdm
from dataLoader_uneven import compute_map_yaw_bins

# data_Folder = '/home/yrf/MPT/data/terrain/train'
# data_Folder = '/home/yrf/MPT/data/sim_dataset/val'
data_Folder = '/home/sdu/MPT/data/sim_dataset/train'
# data_Folder = '/home/yrf/MPT/data/terrain_dataset/val'
env_list = ['env{:06d}'.format(i) for i in range(0, 2)]
cnt = 0

for env in env_list:
    env_path = os.path.join(data_Folder, env)
    print(f'Processing environment: {env_path}')
    
    # 读取地图数据
    map_path = os.path.join(env_path, 'map.p')
    with open(map_path, 'rb') as f:
        map_data = pickle.load(f)
    map_tensor = map_data['tensor']
    elevation = map_tensor[:, :, 0]
    normal_x = map_tensor[:, :, 1]
    normal_y = map_tensor[:, :, 2]
    normal_z = map_tensor[:, :, 3]
    
    yaw_stability = compute_map_yaw_bins(normal_x, normal_y, normal_z, yaw_bins=36)  # [H, W, 36]

    path_files = [f for f in os.listdir(env_path) if f.startswith('path_') and f.endswith('.p')]
    # 使用 tqdm 进度条显示处理进度
    print(f'Found {len(path_files)} path files in {env_path}')
    if not path_files:
        print(f'No path files found in {env_path}. Skipping...')
        continue
    # 遍历每个路径文件
    for path_file in tqdm(path_files, desc=f'Processing paths in {env}'):
        ''' 读取轨迹文件：`path_{id}.p`
        {
            'path': np.array,            # [N, 3] 轨迹点 [x, y, yaw]
            'map_name': 'env000000'         # 关联的地图名称
        }
        '''
        path_path = os.path.join(env_path, path_file)
        with open(path_path, 'rb') as f:
            path_data = pickle.load(f)
        trajectory = path_data['path']
        
        path = trajectory[:, :3]  # 只取前3列（x, y, yaw）
        valid = True
        for i in range(path.shape[0]):
            x, y, yaw = path[i]
            x_idx = int((x + 20) / 0.4)
            y_idx = int((y + 20) / 0.4)
            yaw_idx = int((yaw + np.pi) / (2 * np.pi / 36)) % 36

            # print(f'x: {x}, y: {y}, yaw: {yaw}')
            # print(f'x_idx: {x_idx}, y_idx: {y_idx}, yaw_idx: {yaw_idx}')
            
            if 0 <= x_idx < yaw_stability.shape[0] and 0 <= y_idx < yaw_stability.shape[1]:
                yaw_stability_value = yaw_stability[x_idx, y_idx, yaw_idx]
                if yaw_stability_value == 0:
                    # print(f'Invalid trajectory at index {i} in {path_file}: yaw stability is zero.')
                    valid = False
                    cnt += 1
                    break
        
        # 覆写原文件，在文件内部添加标签"valid"
        with open(path_path, 'wb') as f:
            pickle.dump({'valid': valid, 'path': path, 'map_name': path_data['map_name']}, f)
        # print(f'Saved cleaned trajectory to {path_file}')
print(f'Total invalid trajectories found: {cnt}')

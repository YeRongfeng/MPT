"""调试脚本：检查数据集中轨迹的实际弧长分布"""
import numpy as np
import pickle
from os import path as osp

dataset_path = 'data/sim_dataset/val'
envType = 'env000008'
envFolder = osp.join(dataset_path, envType)

arc_lengths = []

# 检查前50条轨迹
for pathNum in range(50):
    path_file = osp.join(envFolder, f'path_{pathNum}.p')
    try:
        with open(path_file, 'rb') as f:
            path_data = pickle.load(f)
            trajectory = path_data['path']  # [N, 3]
        
        # 计算弧长
        positions = trajectory[:, :2]  # (N, 2)
        distances = np.linalg.norm(positions[1:] - positions[:-1], axis=1)  # (N-1,)
        total_length = distances.sum()
        
        arc_lengths.append(total_length)
        
        if pathNum < 5:
            print(f"Path {pathNum}: length={total_length:.2f}, shape={trajectory.shape}")
    except:
        break

arc_lengths = np.array(arc_lengths)
print(f"\n=== Arc Length Statistics ===")
print(f"Mean: {arc_lengths.mean():.2f}")
print(f"Std: {arc_lengths.std():.2f}")
print(f"Min: {arc_lengths.min():.2f}")
print(f"Max: {arc_lengths.max():.2f}")
print(f"Median: {np.median(arc_lengths):.2f}")

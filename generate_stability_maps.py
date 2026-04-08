#!/usr/bin/env python3
"""
离线批量生成每个环境的 yaw_stability 和 cost_map，并保存到环境目录下。
默认输出文件名：stability_map.npz
"""

import os
import pickle
from os import path as osp

import numpy as np
from tqdm import tqdm

from dataLoader_dit import compute_map_yaw_bins, generate_sdf_from_yaw_stability


# =====================
# 可直接在脚本内修改的配置
# =====================
# dataset_root = '/home/yrf/MPT/data/terrain'
dataset_root = '/home/yrf/MPT/data/sim_dataset'
# dataset_root = '/home/yrf/MPT/data/terrain_dataset'

# 自动处理的子目录（通常为 train + val）
splits = ['train', 'val']

# 指定环境名列表；设为 None 表示每个 split 下自动扫描所有包含 map.p 的环境
# env_list = None
env_list = ["env000012", "env000013"]

output_name = 'stability_map.npz'
yaw_bins = 36
voxel_size_xy = 0.1
yaw_weight = 1.4
overwrite = False


def discover_envs(data_folder):
    if not osp.isdir(data_folder):
        return []

    envs = []
    for name in sorted(os.listdir(data_folder)):
        env_path = osp.join(data_folder, name)
        if not osp.isdir(env_path):
            continue
        if osp.exists(osp.join(env_path, 'map.p')):
            envs.append(name)
    return envs


def process_one_env(env_path, output_name, yaw_bins, voxel_size_xy, yaw_weight, overwrite=False):
    output_file = osp.join(env_path, output_name)
    if osp.exists(output_file) and not overwrite:
        return 'skip', output_file

    map_file = osp.join(env_path, 'map.p')
    if not osp.exists(map_file):
        return 'missing_map', map_file

    with open(map_file, 'rb') as f:
        map_data = pickle.load(f)

    tensor = map_data['tensor']
    normal_x = tensor[:, :, 1]
    normal_y = tensor[:, :, 2]
    normal_z = tensor[:, :, 3]

    yaw_stability = compute_map_yaw_bins(normal_x, normal_y, normal_z, yaw_bins=yaw_bins)
    cost_map = generate_sdf_from_yaw_stability(
        yaw_stability,
        voxel_size_xy=voxel_size_xy,
        yaw_weight=yaw_weight,
    )

    if hasattr(yaw_stability, 'detach'):
        yaw_stability = yaw_stability.detach().cpu().numpy()
    if hasattr(cost_map, 'detach'):
        cost_map = cost_map.detach().cpu().numpy()

    np.savez_compressed(
        output_file,
        yaw_stability=yaw_stability.astype(np.float32),
        cost_map=cost_map.astype(np.float32),
        yaw_bins=np.int32(yaw_bins),
        voxel_size_xy=np.float32(voxel_size_xy),
        yaw_weight=np.float32(yaw_weight),
    )

    return 'ok', output_file


def main():
    if not osp.isdir(dataset_root):
        raise NotADirectoryError(f'数据根目录不存在: {dataset_root}')

    split_folders = []
    for split in splits:
        split_folder = osp.join(dataset_root, split)
        if osp.isdir(split_folder):
            split_folders.append((split, split_folder))
        else:
            print(f'WARNING: split目录不存在，跳过: {split_folder}')

    if not split_folders:
        print('未发现可处理的 split 目录。')
        return

    print(f'数据根目录: {dataset_root}')
    print(f'待处理 splits: {[name for name, _ in split_folders]}')

    ok_count = 0
    skip_count = 0
    missing_count = 0
    total_tasks = 0

    # 先统计总任务数用于进度条
    split_to_envs = {}
    for split_name, split_folder in split_folders:
        current_env_list = env_list if env_list is not None else discover_envs(split_folder)
        split_to_envs[split_name] = current_env_list
        total_tasks += len(current_env_list)

    if total_tasks == 0:
        print('未发现可处理环境。')
        return

    task_idx = 0
    with tqdm(total=total_tasks, desc='Generating stability maps') as pbar:
        for split_name, split_folder in split_folders:
            current_env_list = split_to_envs[split_name]
            if not current_env_list:
                print(f'[{split_name}] 没有可处理环境，跳过。')
                continue

            print(f'[{split_name}] 待处理环境数量: {len(current_env_list)}')

            for env_name in current_env_list:
                task_idx += 1
                env_path = osp.join(split_folder, env_name)
                status, info = process_one_env(
                    env_path=env_path,
                    output_name=output_name,
                    yaw_bins=yaw_bins,
                    voxel_size_xy=voxel_size_xy,
                    yaw_weight=yaw_weight,
                    overwrite=overwrite,
                )

                if status == 'ok':
                    ok_count += 1
                    print(f'[{task_idx}/{total_tasks}] OK: {split_name}/{env_name} -> {info}')
                elif status == 'skip':
                    skip_count += 1
                    print(f'[{task_idx}/{total_tasks}] SKIP: {split_name}/{env_name} (已存在)')
                else:
                    missing_count += 1
                    print(f'[{task_idx}/{total_tasks}] MISSING MAP: {split_name}/{env_name} -> {info}')

                pbar.update(1)

    print('\n生成完成:')
    print(f'  成功: {ok_count}')
    print(f'  跳过: {skip_count}')
    print(f'  缺失map: {missing_count}')


if __name__ == '__main__':
    main()

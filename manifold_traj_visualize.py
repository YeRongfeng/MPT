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
from dataLoader_uneven import compute_map_yaw_bins
from grad_optimizer import TrajectoryOptimizerSE2
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_yaw_stability_edge(yaw_stability, fraction_of_local_max=0.5):
    """
    生成真正的 3D 边界（在 H x W x B 的体素空间上）：
    思路：
    - 把每个像素 (x,y) 在 yaw 维上，选择大于 fraction_of_local_max * local_max 的 bins 作为“占据”体素，
      这样保留同一 (x,y) 下的显著模态，得到二值体素体积 occupied(H,W,B)。
    - 在该 3D 二值体积上做连通域标记（3D），再对每个连通域提取表面（boundary voxels）：
      表面定义为该连通域内某体素存在任一 6-邻居（上下左右、yaw 前后）不属于同一连通域。
    - yaw 维度是回环的（circular），在 yaw 方向上邻居使用 roll（wrap）。
    返回值为 (H, W, B) 的二值数组（numpy.float32），表示 3D 边界体素。
    参数 fraction_of_local_max: 控制每个 (x,y) 保留那些相对于该点最大值的显著 bins（0..1）
    """
    # 转为 numpy
    if torch.is_tensor(yaw_stability):
        ys_np = yaw_stability.cpu().numpy()
    else:
        ys_np = yaw_stability

    H, W, B = ys_np.shape

    # 依据局部最大值阈值化：保留相对显著的 bins（避免把所有微弱噪声也视为体素）
    local_max = ys_np.max(axis=2, keepdims=True)  # shape (H, W, 1)
    # 防止除以0：若 local_max == 0 则该 (x,y) 全部为 0 -> 不保留任何 bin
    thr = local_max * float(fraction_of_local_max)
    occupied = (ys_np >= thr) & (local_max > 0)   # bool (H, W, B)

    # 若没有显著体素，直接返回空边界
    if not occupied.any():
        return np.zeros((H, W, B), dtype=np.float32)

    # 连通域标记（使用 skimage.measure.label 支持 3D 连通性）
    from skimage.measure import label
    # connectivity=1 -> 6-邻居 (face connectivity) 对应 3D 六联通
    labels = label(occupied, connectivity=1)

    # 为了检测边界，在 x/y 方向使用 pad：外部用 -1 标记（表示图像外部），在 yaw 方向不填充（周期）
    p = np.pad(labels, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=-1)
    center = p[1:-1, 1:-1, :]  # (H, W, B)

    # 6 个邻居
    nb_up    = p[:-2,   1:-1, :]  # i-1, j
    nb_down  = p[2:,    1:-1, :]  # i+1, j
    nb_left  = p[1:-1, :-2,   :]  # i, j-1
    nb_right = p[1:-1, 2:,    :]  # i, j+1
    # yaw 前后用 roll（回环）
    nb_yaw_prev = np.roll(center, 1, axis=2)
    nb_yaw_next = np.roll(center, -1, axis=2)

    # 如果某 voxel 属于连通域（label>0），且存在任一邻居 label != center -> 为表面 voxel
    neighbor_stack = np.stack([nb_up, nb_down, nb_left, nb_right, nb_yaw_prev, nb_yaw_next], axis=0)
    # 排除 pad 出来的图像外部（-1），但把 map 内部的空白（0）视为外部空间 -> 属于表面
    neighbor_diff = ((neighbor_stack != center) & (neighbor_stack != -1))  # shape (6, H, W, B)
    boundary_mask = (center > 0) & neighbor_diff.any(axis=0)

    yaw_stability_edge = boundary_mask.astype(np.float32)  # (H, W, B)

    return yaw_stability_edge

if __name__ == "__main__":
    # stage = 1
    # epoch = 4
    stage = 2
    epoch = 39
    envNum = np.random.randint(0, 99)  # 随机选择环境id
    envList = ['env000022']  # 生成环境列表，格式为 env000000, env000001, ..., env000009
    dataset_path = 'data/terrain/train'
    save_path = 'predictions'
    path_id = 0

    modelFolder = 'data/uneven'
    # modelFolder = 'data/uneven_old'
    modelFile = osp.join(modelFolder, f'model_params.json')
    model_param = json.load(open(modelFile))

    transformer = Models.UnevenTransformer(**model_param)
    _ = transformer.to(device)

    # checkpoint = torch.load(osp.join(modelFolder, f'model_epoch_{epoch}.pkl'))
    if stage == 1:
        checkpoint = torch.load(osp.join(modelFolder, f'stage1_model_epoch_{epoch}.pkl'))
    else:
        checkpoint = torch.load(osp.join(modelFolder, f'stage2_model_epoch_{epoch}.pkl'))
    transformer.load_state_dict(checkpoint['state_dict'])

    _ = transformer.eval()
    
    env_path = osp.join(dataset_path, envList[0])
    
    # 获取地图信息
    map_file = osp.join(env_path, 'map.p')
    with open(map_file, 'rb') as f:
        env_data = pickle.load(f)
    map_tensor = env_data['tensor']  # shape (H, W, B)
    elevation = map_tensor[:, :, 0]  # 高度图 (H, W)
    normal_x = map_tensor[:, :, 1]  # 法向量 x 分量 (H, W)
    normal_y = map_tensor[:, :, 2]  # 法向量 y 分量 (H, W)
    normal_z = map_tensor[:, :, 3]  # 法向量 z 分量 (H, W)
    
    yaw_stability = compute_map_yaw_bins(normal_x, normal_y, normal_z, yaw_bins=36)  # shape (H, W, B)
    yaw_stability_edge = get_yaw_stability_edge(yaw_stability, fraction_of_local_max=0.5)
    
    # 读取轨迹
    path_file = f'path_{path_id}.p'
    path_path = osp.join(env_path, path_file)
    with open(path_path, 'rb') as f:
        path_data = pickle.load(f)
    trajectory = path_data['path']

    map_size = (100, 100, 36) # W, H, D for (x, y, yaw)
    resolution = 0.1
    origin = (-5.0, -5.0, -np.pi) # x, y, yaw
    map_info = {
        'resolution': resolution,
        'origin': origin,
        'size': map_size
    }
    zeros_map = torch.zeros(tuple(map_size), dtype=torch.float32, device=device)
    optimizer = TrajectoryOptimizerSE2(trajectory, zeros_map, map_info, device=device)
    # 获取优化前的轨迹
    with torch.no_grad():
        initial_poses_torch = torch.tensor(trajectory, device=device, dtype=torch.float32)
        Sx_i, Sy_i, Syaw_i, *_ = optimizer._evaluate_spline_se2(initial_poses_torch, optimizer.t_dense)
        initial_trajectory = torch.stack([Sx_i, Sy_i, Syaw_i], dim=1).cpu().numpy()
        
    # 生成模型预测轨迹
    goal_pos = trajectory[-1, :]  # 终点位置
    start_pos = trajectory[0, :]  # 起点位置
    encoder_input = get_encoder_input(normal_z, goal_pos, start_pos, normal_x, normal_y)
    patch_maps, predProb_list, predTraj = get_patch(transformer, start_pos, goal_pos, normal_x, normal_y, normal_z)
    # 确保 predTraj 是一个连续的 numpy/torch Tensor，然后再拼接起止点
    if isinstance(predTraj, list):
        pred_arr = np.asarray(predTraj, dtype=np.float32) if len(predTraj) > 0 else np.zeros((0, 3), dtype=np.float32)
        pred_traj_t = torch.from_numpy(pred_arr).to(device)
    elif torch.is_tensor(predTraj):
        pred_traj_t = predTraj.to(device).float()
    else:
        pred_traj_t = torch.from_numpy(np.asarray(predTraj, dtype=np.float32)).to(device)

    start_t = torch.tensor(start_pos, dtype=torch.float32, device=device).unsqueeze(0)
    goal_t  = torch.tensor(goal_pos,  dtype=torch.float32, device=device).unsqueeze(0)
    predTraj = torch.cat((start_t, pred_traj_t, goal_t), dim=0)
    # 获取完整的预测轨迹
    with torch.no_grad():
        Sx_p, Sy_p, Syaw_p, *_ = optimizer._evaluate_spline_se2(predTraj, optimizer.t_dense)
        pred_trajectory = torch.stack([Sx_p, Sy_p, Syaw_p], dim=1).cpu().numpy()
    # 将原始的预测轨迹转换为 numpy 数组
    predTraj = predTraj.cpu().numpy()

    # --- 图1：在yaw_stability图中绘制轨迹 ---
    H, W, B = yaw_stability_edge.shape

    # 找到所有边界点的索引 (row, col, bin)
    rows, cols, bins = np.where(yaw_stability_edge > 0)
    # rows = rows / float(H) * 10.0 - 5.0  # 将行索引映射到实际坐标
    # cols = cols / float(W) * 10.0 - 5.0  # 将列索引映射到实际坐标
    # bins = bins / float(B) * 2 * np.pi - np.pi  # 将 bin 映射为 [-pi, pi]
    # 使用 map_info 的 origin/resolution 做精确映射到世界坐标（避免坐标轴反向/交换问题）
    res = map_info['resolution']
    origin_x, origin_y, origin_yaw = map_info['origin']
    # cols -> x_world，rows -> y_world
    # 注意：图像行索引通常从上到下（row=0 在上），而地图 origin 通常是左下角，
    # 所以 world_y = origin_y + (H - 1 - row) * res
    x_world = origin_x + cols * res
    y_world = origin_y + (H - 1 - rows) * res
    # yaw: bins 映射到角度/弧度；origin_yaw 通常为 -pi
    yaw_world = origin_yaw + bins * (2.0 * np.pi / float(B))
    # 覆盖变量以便后续绘图直接使用 cols(rows)=x(y)
    cols = x_world
    rows = y_world
    bins = yaw_world
    # # 可选：输出少量样本用于检查（运行时可注释掉）
    # print("Voxel->world sample (x,y,yaw):", cols[:5], rows[:5], bins[:5])

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig1 = plt.figure(figsize=(12, 9))
    ax1 = fig1.add_subplot(111, projection='3d')
    # 若点太多则下采样，避免绘图过慢或内存占用过大
    max_points = 200000
    n_pts = rows.shape[0]
    if n_pts > max_points:
        idx = np.random.choice(n_pts, max_points, replace=False)
        rows = rows[idx]; cols = cols[idx]; bins = bins[idx]
    yaw_rad = bins
    # 绘制 3D 散点图
    scatter = ax1.scatter(cols, rows, yaw_rad, c='red', s=8, marker='o',
                            edgecolors='none', linewidths=0, alpha=0.4, rasterized=True)
    
    # 绘制初始轨迹
    ax1.plot(initial_trajectory[:, 0], initial_trajectory[:, 1], initial_trajectory[:, 2],
             color='blue', linewidth=2, label='Initial Trajectory')
    
    # 绘制预测轨迹
    ax1.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2],
             color='green', linewidth=2, label='Predicted Trajectory')
    
    ax1.set_xlabel('X (cols)')
    ax1.set_ylabel('Y (rows)')
    ax1.set_zlabel('Yaw (rad)')
    ax1.set_title('Yaw Stability Edge (3D: x, y, yaw)')
    ax1.view_init(elev=30, azim=-60)
    plt.tight_layout()
    plt.show()

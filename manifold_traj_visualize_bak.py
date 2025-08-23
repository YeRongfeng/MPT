import matplotlib.pyplot as plt
import os
from os import path as osp
import plotly.graph_objects as go
import numpy as np
import pandas as pd
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

def get_yaw_stability_edge(yaw_stability, fraction_of_local_max=0.5, valid_mask=None):
    """
    生成 3D 边界体素 (H, W, B)，改进了与地图边界 (map edge) 交接处的处理：
    - valid_mask: 可选 (H,W) bool 数组，True 表示该 (x,y) 在地图内部（有效）。
      如果提供则能正确区分“地图外部”与“yaw_stability==0 的内部区域”；
      如果不提供，函数会尝试使用 local_max>0 做近似（可能不完美，会打印提示）。

    返回 float32 数组 shape (H, W, B)，1 表示该 voxel 位于边界。
    """
    if torch.is_tensor(yaw_stability):
        ys_np = yaw_stability.cpu().numpy()
    else:
        ys_np = np.asarray(yaw_stability)

    H, W, B = ys_np.shape
    
    ys_np = 1-ys_np

    # 局部最大值（按 yaw bins）
    local_max = ys_np.max(axis=2, keepdims=True)  # (H, W, 1)
    thr = local_max * float(fraction_of_local_max)
    occupied = (ys_np >= thr) & (local_max > 0)   # bool (H, W, B)

    # 推断或校验 valid_mask（地图有效区域）
    if valid_mask is None:
        # 警告：没有提供地图有效区域掩码时，我们不能区分“地图外部”与“内部的全零”。
        # 为了避免把地图外框整圈错误识别成边界，这里采用较保守的策略：
        # treat all XY as valid (避免大面积误判)。建议最好传入真实的 valid_mask。
        valid_mask = np.ones((H, W), dtype=bool)
        # 若你希望用 local_max>0 来作为 valid_mask，请显式传入 valid_mask=local_max[...,0]>0
        # print("Warning: valid_mask not provided — using all-True fallback. For correct behavior provide valid_mask (H,W).")
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        assert valid_mask.shape == (H, W), "valid_mask must have shape (H, W)"

    # 标识地图内部但 yaw_stability 全零的 XY 单元
    zero_inside_xy = (~(local_max.squeeze(axis=2) > 0)) & valid_mask  # (H, W)

    # 连通域标记（3D，6-邻居）
    from skimage.measure import label
    labels = label(occupied, connectivity=1)  # 0 表示背景（未占据）

    # 在三维上 pad：pad 的值设为 -1 表示“外部（超出索引）”
    p = np.pad(labels, ((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=-1)
    center = p[1:-1, 1:-1, :]  # (H, W, B)

    # 6 个邻居（xy 四向直接从 pad 中取；yaw 用 roll 做循环邻居）
    nb_up    = p[:-2,   1:-1, :]  # i-1, j
    nb_down  = p[2:,    1:-1, :]  # i+1, j
    nb_left  = p[1:-1, :-2,   :]  # i, j-1
    nb_right = p[1:-1, 2:,    :]  # i, j+1
    # yaw 前后用 roll（周期）
    nb_yaw_prev = np.roll(center, 1, axis=2)
    nb_yaw_next = np.roll(center, -1, axis=2)

    neighbor_stack = np.stack([nb_up, nb_down, nb_left, nb_right, nb_yaw_prev, nb_yaw_next], axis=0)  # (6,H,W,B)

    # 基本差异：邻居标签 != 中心标签（包括 pad(-1) 与任何正标签比较）
    neighbor_diff = (neighbor_stack != center)  # (6,H,W,B)

    # 现在处理那些来源于 pad(-1) 的邻居（即超出边界的位置）
    # 我们只在“该超出边界处对应的地图内侧正好是 yaw_stability==0（zero_inside）”时
    # 把该 pad 视作有效的“零区邻居”，从而标记该接缝为边界；否则忽略 pad 差异以避免整圈误判。
    # 为此，构造一个 (H,W,B) 的 zero_inside 布尔广播（在 yaw 维上复制）
    zero_inside_3d = np.repeat(zero_inside_xy[:, :, None], B, axis=2)  # (H, W, B)
    # 将 zero_inside 做 pad（在 xy 方向 pad False）
    p_zero = np.pad(zero_inside_3d, ((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=False)
    # 使用显式索引以保证在小尺寸时也返回正确且一致的形状 (H, W, B)
    nbz_up   = p_zero[0:H,     1:W+1, :]
    nbz_down = p_zero[2:2+H,   1:W+1, :]
    nbz_left = p_zero[1:1+H,   0:W,   :]
    nbz_right= p_zero[1:1+H,   2:2+W, :]
    # yaw 前后用 roll（周期）
    nb_yaw_prev = np.roll(center, 1, axis=2)
    nb_yaw_next = np.roll(center, -1, axis=2)

    # yaw 的邻居对应的 zero_inside 实际上就是 center 对应的 zero_inside（因为 zero_inside 没有 yaw 变化）
    # 对于 out-of-bounds 的判断，我们只在四方向上需要检查 pad(-1) 的情况；yaw 的 pad 不产生 -1（因为我们 rolled center）
    # 在堆叠前检查形状一致性，便于定位潜在问题
    target_shape = center.shape
    for arr, name in [(nbz_up, 'nbz_up'), (nbz_down, 'nbz_down'), (nbz_left, 'nbz_left'), (nbz_right, 'nbz_right')]:
        if arr.shape != target_shape:
            raise ValueError(f"{name} shape {arr.shape} does not match center shape {target_shape}")

    neighbor_zero_stack = np.stack([nb_up, nb_down, nb_left, nb_right, nb_yaw_prev, nb_yaw_next], axis=0)  # (6,H,W,B)

    # 现在对于那些 neighbor == -1 的位置，如果 neighbor_zero_stack 为 True -> 保留 neighbor_diff 为 True
    # 否则把 neighbor_diff 对应位置设为 False（即把 pad 差异忽略）
    pad_mask = (neighbor_stack == -1)  # (6,H,W,B)
    # 如果 pad 且 neighbor_zero_stack == False -> 取消差异
    neighbor_diff = np.where(pad_mask & (~neighbor_zero_stack), False, neighbor_diff)

    # 最终：任一邻居差异则为表面 voxel（并且 center>0）
    boundary_mask = (center > 0) & neighbor_diff.any(axis=0)

    yaw_stability_edge = boundary_mask.astype(np.float32)
    return yaw_stability_edge


if __name__ == "__main__":
    # stage = 1
    # epoch = 4
    stage = 2
    epoch = 79
    envNum = np.random.randint(0, 99)  # 随机选择环境id
    envList = ['env000009']  # 生成环境列表，格式为 env000000, env000001, ..., env000009
    dataset_path = 'data/terrain/train'
    save_path = 'predictions'
    path_id = 18

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
    y_world = origin_y + rows * res
    # yaw: bins 映射到角度/弧度；origin_yaw 通常为 -pi
    yaw_world = origin_yaw + bins * (2.0 * np.pi / float(B))
    # 覆盖变量以便后续绘图直接使用 cols(rows)=x(y)
    cols = x_world
    rows = y_world
    bins = yaw_world
    # # 可选：输出少量样本用于检查（运行时可注释掉）
    # print("Voxel->world sample (x,y,yaw):", cols[:5], rows[:5], bins[:5])
    
    # 覆盖变量以便后续绘图直接使用 cols(rows)=x(y)
    cols = x_world
    rows = y_world
    bins = yaw_world

    # 为后续绘图统一命名并检查形状（修复 x_pts/y_pts/yaw_rad 未定义问题）
    x_pts = np.asarray(cols)
    y_pts = np.asarray(rows)
    yaw_rad = np.asarray(bins)
    if not (x_pts.shape == y_pts.shape == yaw_rad.shape):
        raise ValueError(f"Point arrays shape mismatch: {x_pts.shape}, {y_pts.shape}, {yaw_rad.shape}")

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import matplotlib as mpl

    # --------- 可调参数（按需修改） ----------
    figsize = (8.0, 6.0)
    dpi = 200
    font_family = "serif"
    base_fontsize = 10
    max_points = 200000
    point_size = 20           # <-- 放大点：你要的点大小（原来 6）
    side_axis_frac = 0.05     # 伪轴距离数据边界的相对偏移（0.05 = 5%）
    # ----------------------------------------

    mpl.rcParams.update({
        "font.family": font_family,
        "font.size": base_fontsize,
        "axes.linewidth": 0.8,
        "axes.labelsize": base_fontsize + 1,
        "legend.fontsize": base_fontsize,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,
    })

    # 下采样（避免渲染过慢）
    n_pts = x_pts.shape[0]
    if n_pts > max_points:
        idx = np.random.choice(n_pts, max_points, replace=False)
        x_pts = x_pts[idx]; y_pts = y_pts[idx]; yaw_rad = yaw_rad[idx]

    # 颜色映射
    norm = mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi)
    cmap = mpl.cm.viridis
    colors = cmap(norm(yaw_rad))

    # 帮助函数：在 3D 中实现等比例刻度
    def set_axes_equal(ax: Axes3D):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        max_range = max([x_range, y_range, z_range])
        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)
        ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
        ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
        ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

    # 创建图
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # 显式设置 z 轴范围以保证在任何情况下都可见（yaw 范围 -pi..pi）
    ax.set_zlim(-np.pi, np.pi)

    # 绘制边界体素（放大点）
    sc = ax.scatter(x_pts, y_pts, yaw_rad,
                    s=point_size,                # 放大点
                    c=colors,
                    marker='o',
                    linewidths=0,
                    alpha=0.8,
                    # alpha=1.0,
                    edgecolors='none',
                    rasterized=True,
                    # depthshade=True,
                    zorder=1)

    # 绘制轨迹与控制点（保持层次）
    ax.plot(initial_trajectory[:, 0], initial_trajectory[:, 1], initial_trajectory[:, 2],
            linewidth=2.2, label='Initial trajectory', zorder=3)
    ax.scatter(trajectory[1:-1, 0], trajectory[1:-1, 1], trajectory[1:-1, 2],
            s=36, marker='o', edgecolor='white', linewidth=0.6, zorder=4)

    ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2],
            linewidth=2.6, label='Predicted trajectory', zorder=5)
    ax.scatter(predTraj[1:-1, 0], predTraj[1:-1, 1], predTraj[1:-1, 2],
            s=44, marker='D', edgecolor='white', linewidth=0.6, zorder=6)

    # 起点/终点
    ax.scatter(initial_trajectory[0, 0], initial_trajectory[0, 1], initial_trajectory[0, 2],
            s=110, marker='^', label='Start', zorder=8)
    ax.scatter(initial_trajectory[-1, 0], initial_trajectory[-1, 1], initial_trajectory[-1, 2],
            s=120, marker='*', label='Goal', zorder=8)

    ax.text(initial_trajectory[0, 0], initial_trajectory[0, 1], initial_trajectory[0, 2] + 0.05,
            "Start", fontsize=base_fontsize, va='bottom', ha='center', zorder=8)
    ax.text(initial_trajectory[-1, 0], initial_trajectory[-1, 1], initial_trajectory[-1, 2] + 0.05,
            "Goal", fontsize=base_fontsize, va='bottom', ha='center', zorder=8)

    # colorbar
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = plt.colorbar(mappable, ax=ax, pad=0.12, shrink=0.6)
    cbar.set_label('Yaw (rad)', fontsize=base_fontsize)
    cbar.ax.tick_params(labelsize=base_fontsize - 1)

    # 轴标签 / 标题
    ax.set_xlabel('X (m)', labelpad=6)
    ax.set_ylabel('Y (m)', labelpad=6)
    ax.set_zlabel('Yaw (rad)', labelpad=8)
    ax.set_title('Yaw Stability Edge (x, y, yaw)', pad=10)

    # 视角
    ax.view_init(elev=30, azim=-60)
    set_axes_equal(ax)

    # 关闭网格，优化面板边框显示
    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('k'); ax.yaxis.pane.set_edgecolor('k'); ax.zaxis.pane.set_edgecolor('k')

    # --------- 绘制“左侧伪 z 轴”（保证在任何视角都可见） ----------
    # 选择伪轴位置：放在 x 范围最左、y 范围最下，并略向外偏移一点
    x_min, x_max = np.min(x_pts), np.max(x_pts)
    y_min, y_max = np.min(y_pts), np.max(y_pts)
    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0

    x_pos = x_min - side_axis_frac * x_range
    y_pos = y_min - side_axis_frac * y_range
    z_min, z_max = -np.pi, np.pi

    # 主轴线
    ax.plot([x_pos, x_pos], [y_pos, y_pos], [z_min, z_max], color='k', linewidth=1.0, zorder=20)

    # 刻度与刻度标签（你可以调整 n_ticks）
    n_ticks = 5
    ticks = np.linspace(z_min, z_max, n_ticks)
    tick_len = 0.015 * (x_range if x_range>0 else 1.0)  # 刻度横向长度（相对于 x 范围）
    for t in ticks:
        # 横向小刻度线
        ax.plot([x_pos, x_pos + tick_len], [y_pos, y_pos], [t, t], color='k', linewidth=1.0, zorder=20)
        # 文本标签（放在刻度线右侧）
        ax.text(x_pos + 1.6*tick_len, y_pos, t, f"{t:.2f}", fontsize=base_fontsize-1, va='center', ha='left', zorder=21)

    # 给伪轴加上轴名
    ax.text(x_pos, y_pos, z_max + 0.08*(z_max - z_min), 'Yaw (rad)', fontsize=base_fontsize, va='bottom', ha='center', zorder=21)

    # 图例（放在图外）
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))

    plt.tight_layout()
    outname_pdf = 'yaw_stability_edge_3d.pdf'
    outname_svg = 'yaw_stability_edge_3d.svg'
    plt.savefig(outname_pdf, dpi=600, bbox_inches='tight')
    plt.savefig(outname_svg, dpi=600, bbox_inches='tight')
    plt.show()

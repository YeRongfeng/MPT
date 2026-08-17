if __package__:
    from . import _repo_path
else:
    import _repo_path

import matplotlib.pyplot as plt
import os
from os import path as osp
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import pickle

from skimage import io, measure

import sys
sys.modules['numpy._core'] = np
sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath
sys.modules['numpy._core.multiarray'] = np.core.multiarray

import torch
import torch.nn.functional as F
import json
from tools._paths import PREDICTIONS_ROOT, predictions_dir

from transformer import Models
from dataLoader_uneven import get_encoder_input, receptive_field
from eval_model_uneven import getHashTable, get_patch
from dataLoader_uneven import compute_map_yaw_bins
from grad_optimizer import TrajectoryOptimizerSE2
import torch

from scipy.ndimage import binary_fill_holes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_unstable_pose(Sx, Sy, Syaw, yaw_stability):
    """
    使用 torch 实现，支持 Sx/Sy/Syaw 为 torch.Tensor（可在 GPU 上）或 numpy array。
    返回：num_capsize (int), capsize_points (list of (x,y,yaw) floats)
    """
    # 将输入统一为 torch.Tensor（放在 Sx 的 device 上或 cpu）
    if torch.is_tensor(Sx):
        device = Sx.device
    else:
        device = torch.device("cpu")
    Sx_t = torch.as_tensor(Sx, device=device, dtype=torch.float32)
    Sy_t = torch.as_tensor(Sy, device=device, dtype=torch.float32)
    Syaw_t = torch.as_tensor(Syaw, device=device, dtype=torch.float32)

    ys_t = torch.as_tensor(yaw_stability, device=device, dtype=torch.float32)

    # 检查形状为 (H, W, B)
    if ys_t.ndim != 3:
        raise ValueError("yaw_stability must have shape (H, W, B)")

    H, W, B = ys_t.shape
    res = 0.1
    origin_x, origin_y, origin_yaw = -5.0, -5.0, -np.pi

    # 计算连续索引
    col_f = (Sx_t - origin_x) / res
    row_f = (Sy_t - origin_y) / res
    bin_f = (Syaw_t - origin_yaw) / (2.0 * np.pi / B)

    col = torch.floor(col_f).long()
    row = torch.floor(row_f).long()
    # yaw 周期化
    bin_idx = (torch.floor(bin_f).long() % B)

    # 有效范围掩码（只考虑 XY 在地图内的点）
    valid_mask = (row >= 0) & (row < H) & (col >= 0) & (col < W)

    if valid_mask.any().item() == 0:
        return 0, []

    # 使用 view(-1) 保证一维索引（比 squeeze 更稳健）
    valid_idx = torch.nonzero(valid_mask, as_tuple=False).view(-1)
    r_sel = row[valid_idx]
    c_sel = col[valid_idx]
    b_sel = bin_idx[valid_idx]

    # 从 yaw_stability 中取值，使用阈值判断不稳定（与脚本其它位置保持一致）
    vals = ys_t[r_sel, c_sel, b_sel]
    interp_thresh = 0.5
    unstable_mask = (vals < interp_thresh)

    if unstable_mask.any().item() == 0:
        return 0, []

    unstable_idx = valid_idx[unstable_mask]
    xs = Sx_t[unstable_idx].detach().cpu().numpy().tolist()
    ys = Sy_t[unstable_idx].detach().cpu().numpy().tolist()
    ysaws = Syaw_t[unstable_idx].detach().cpu().numpy().tolist()

    capsize_points = list(zip(xs, ys, ysaws))
    num_capsize = len(capsize_points)

    return int(num_capsize), capsize_points

if __name__ == "__main__":
    # stage = 1
    # epoch = 4
    stage = 2
    epoch = 79
    envNum = np.random.randint(0, 99)  # 随机选择环境id
    envList = ['env000009']  # 生成环境列表，格式为 env000000, env000001, ..., env000009
    dataset_path = 'data/terrain/train'
    save_path = str(PREDICTIONS_ROOT)
    path_id = 30

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
    
    # 传入轨迹 (Sx, Sy, Syaw) 以及 yaw_stability 地图
    num_capsize, capsize_points = compute_unstable_pose(Sx_i, Sy_i, Syaw_i, yaw_stability)
    print(f"Number of unstable poses in initial trajectory: {num_capsize}")

    num_capsize_pred, capsize_points_pred = compute_unstable_pose(Sx_p, Sy_p, Syaw_p, yaw_stability)
    print(f"Number of unstable poses in predicted trajectory: {num_capsize_pred}")

    # --- 替换原有散点图代码为等值面可视化 ---
    interp_thresh_local = 0.5  # 局部阈值（与后面 interp_thresh 保持一致）

    # 直接把不可达体素作为实心体（不填洞）
    # 不可达定义：yaw_stability < interp_thresh_local
    unreachable_mask = (yaw_stability < interp_thresh_local)

    # unreachable_mask 可能为 torch.Tensor 或 numpy.ndarray，统一转为 numpy.float32
    if isinstance(unreachable_mask, torch.Tensor):
        unreachable_volume = unreachable_mask.cpu().numpy().astype(np.float32)
    else:
        unreachable_volume = np.asarray(unreachable_mask, dtype=np.float32)

    H, W, B = unreachable_volume.shape
    res = map_info['resolution']
    origin_x, origin_y, origin_yaw = map_info['origin']

    # （可选）创建坐标网格（目前未直接使用）
    x_coords = origin_x + np.arange(W) * res
    y_coords = origin_y + np.arange(H) * res
    yaw_coords = origin_yaw + np.arange(B) * (2.0 * np.pi / B)
    X_grid, Y_grid, Z_grid = np.meshgrid(x_coords, y_coords, yaw_coords, indexing='ij')

    # 使用 marching_cubes 提取等值面（level=0.5），显式传入 float 数据
    try:
        verts, faces, normals, values = measure.marching_cubes(unreachable_volume.astype(np.float32), level=0.5)
        # marching_cubes 返回的 verts 顺序为 (row, col, slice) 对应 (r, c, b)
        v_row = verts[:, 0].copy()
        v_col = verts[:, 1].copy()
        v_bin = verts[:, 2].copy()

        # 映射回世界坐标以供绘图：x <- col, y <- row, yaw <- slice
        verts[:, 0] = origin_x + v_col * res
        verts[:, 1] = origin_y + v_row * res
        verts[:, 2] = origin_yaw + v_bin * (2.0 * np.pi / B)
    except ValueError:
        print("Warning: No isosurface found. Skipping surface plot.")
        verts, faces = None, None
    except Exception as e:
        print(f"Error running marching_cubes or mapping verts: {e}")
        verts, faces = None, None

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib as mpl

    # --------- 可调参数（按需修改） ----------
    figsize = (10, 8)
    dpi = 200
    font_family = "serif"
    base_fontsize = 10
    voxel_center = False  # 体素中心对齐
    flip_y = False    # y 轴是否翻转（与体素索引一致）
    interp_thresh = 0.5  # 全局阈值（与前面 interp_thresh_local 保持一致）
    # 交点绘制参数（防止未定义错误）
    intersection_marker_size = 40
    intersection_marker_style = 'x'
    initial_intersection_color = 'lime'
    predicted_intersection_color = 'cyan'
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

    # 创建图
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # 绘制等值面
    if verts is not None:
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                        triangles=faces,
                        color='red',
                        alpha=0.3,
                        label='Unreachable Region',
                        edgecolor='none',
                        antialiased=True,
                        zorder=1)

    # 绘制轨迹与控制点（保持层次）
    ax.plot(initial_trajectory[:, 0], initial_trajectory[:, 1], initial_trajectory[:, 2],
            linewidth=2.2, label='Initial trajectory', zorder=3)
    ax.scatter(trajectory[1:-1, 0], trajectory[1:-1, 1], trajectory[1:-1, 2],
            s=36, marker='o', edgecolor='white', linewidth=0.6, zorder=4)

    # ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2],
    #         linewidth=2.6, label='Predicted trajectory', zorder=5)
    # ax.scatter(predTraj[1:-1, 0], predTraj[1:-1, 1], predTraj[1:-1, 2],
    #         s=44, marker='D', edgecolor='white', linewidth=0.6, zorder=6)

    # 起点/终点（去掉 label，只保留 marker）
    ax.scatter(initial_trajectory[0, 0], initial_trajectory[0, 1], initial_trajectory[0, 2],
            s=110, marker='^', zorder=8)
    ax.scatter(initial_trajectory[-1, 0], initial_trajectory[-1, 1], initial_trajectory[-1, 2],
            s=120, marker='*', zorder=8)

    # 去掉起点/终点的文字注释（不显示 "Start"/"Goal" 文本）
    #ax.text(initial_trajectory[0, 0], initial_trajectory[0, 1], initial_trajectory[0, 2] + 0.05,
    #        "Start", fontsize=base_fontsize, va='bottom', ha='center', zorder=8)
    #ax.text(initial_trajectory[-1, 0], initial_trajectory[-1, 1], initial_trajectory[-1, 2] + 0.05,
    #        "Goal", fontsize=base_fontsize, va='bottom', ha='center', zorder=8)

    # 轴标签 / 标题
    # （已移除坐标轴标签和标题，仅保留可视化）

    # NOTE：视角
    ax.view_init(elev=20, azim=40)
    # ax.view_init(elev=26, azim=-7)
    
    # 手动设置坐标范围以确保等比例
    max_range = np.array([initial_trajectory[:, 0].max()-initial_trajectory[:, 0].min(),
                          initial_trajectory[:, 1].max()-initial_trajectory[:, 1].min(),
                          initial_trajectory[:, 2].max()-initial_trajectory[:, 2].min()]).max() / 2.0
    mid_x = (initial_trajectory[:, 0].max()+initial_trajectory[:, 0].min()) * 0.5
    mid_y = (initial_trajectory[:, 1].max()+initial_trajectory[:, 1].min()) * 0.5
    mid_z = (initial_trajectory[:, 2].max()+initial_trajectory[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # 强制坐标范围为 X: [-5,5], Y: [-5,5], Z(yaw): [-pi, pi]
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)
    ax.set_zlim(-np.pi, np.pi)

    # 关闭网格，优化面板边框显示（保留坐标轴，但隐藏刻度文字）
    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('k'); ax.yaxis.pane.set_edgecolor('k'); ax.zaxis.pane.set_edgecolor('k')

    # 隐藏刻度文字但保留坐标轴和刻度线
    try:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    except Exception:
        # 备用方法：对每个 axis 的 tick label 设置不可见
        for t in ax.xaxis.get_major_ticks():
            t.label.set_visible(False)
        for t in ax.yaxis.get_major_ticks():
            t.label.set_visible(False)
        for t in ax.zaxis.get_major_ticks():
            t.label.set_visible(False)

    # 不再调用 ax.set_axis_off()

    # plt.tight_layout()
    # outname_pdf = 'yaw_stability_isosurface_3d.pdf'
    # outname_svg = 'yaw_stability_isosurface_3d.svg'
    # plt.savefig(outname_pdf, dpi=600, bbox_inches='tight')
    # plt.savefig(outname_svg, dpi=600, bbox_inches='tight')
    # plt.show()

    def world_to_voxel_index(x, y, yaw):
        """把世界坐标映射到体素整数索引 (r, c, b)。越界返回 None。"""
        if voxel_center:
            col_f = (x - origin_x) / res - 0.5
            row_f = (y - origin_y) / res - 0.5
        else:
            col_f = (x - origin_x) / res
            row_f = (y - origin_y) / res

        if flip_y:
            row_f = (H - 1) - row_f

        bin_f = (yaw - origin_yaw) / (2.0 * np.pi / B)
        r = int(np.floor(row_f))
        c = int(np.floor(col_f))
        b = int(np.floor(bin_f)) % B

        if r < 0 or r >= H or c < 0 or c >= W:
            return None
        return r, c, b

    def trilinear_sample(vol, row_f, col_f, bin_f):
        """占位：不再使用。保留以防其它代码调用，但直接使用最近体素值（最近邻）。"""
        r = int(np.floor(row_f)); c = int(np.floor(col_f)); b = int(np.floor(bin_f)) % B
        if r < 0 or c < 0 or r >= H or c >= W:
            return 0.0
        return float(vol[r, c, b])

    def is_inside(point):
        """基于矩形体素判断点是否落在不可达区域（返回 bool），不做插值。"""
        idx = world_to_voxel_index(point[0], point[1], point[2])
        if idx is None:
            return False
        r, c, b = idx
        return float(unreachable_volume[r, c, b]) > interp_thresh

    def locate_crossing_continuous(p1, p2, max_iters=30, tol=1e-4):
        """在连续线段上用二分定位阈值交点，使用整数体素判断。"""
        a = np.array(p1, dtype=np.float64); b = np.array(p2, dtype=np.float64)
        ia = world_to_voxel_index(a[0], a[1], a[2])
        ib = world_to_voxel_index(b[0], b[1], b[2])
        va = float(unreachable_volume[ia]) if ia is not None else 0.0
        vb = float(unreachable_volume[ib]) if ib is not None else 0.0
        fa = (va > interp_thresh); fb = (vb > interp_thresh)
        if fa == fb:
            return None
        for _ in range(max_iters):
            m = 0.5 * (a + b)
            im = world_to_voxel_index(m[0], m[1], m[2])
            vm = float(unreachable_volume[im]) if im is not None else 0.0
            fm = (vm > interp_thresh)
            if fm == fa:
                a, ia, va, fa = m, im, vm, fm
            else:
                b, ib, vb, fb = m, im, vm, fm
            if np.linalg.norm(b - a) < tol:
                break
        mid = 0.5 * (a + b)
        return (float(mid[0]), float(mid[1]), float(mid[2]))

    def find_intersections(trajectory, samples_per_segment=24):
        """用离散体素采样判断段上是否穿越不可达区域并定位交点（不插值）。"""
        intersections = []
        traj = np.asarray(trajectory, dtype=np.float64)
        for i in range(len(traj) - 1):
            p1 = traj[i]; p2 = traj[i + 1]
            ts = np.linspace(0.0, 1.0, samples_per_segment + 1)
            samples = p1[None, :] * (1 - ts[:, None]) + p2[None, :] * (ts[:, None])
            flags = []
            for s in samples:
                idx = world_to_voxel_index(s[0], s[1], s[2])
                if idx is None:
                    flags.append(False)
                else:
                    r, c, b = idx
                    flags.append(bool(unreachable_volume[r, c, b] > interp_thresh))
            for j in range(len(flags) - 1):
                if flags[j] != flags[j + 1]:
                    cross = locate_crossing_continuous(samples[j], samples[j + 1])
                    if cross is not None:
                        intersections.append(cross)
        return intersections

    # # 寻找交点（单次计算、单次输出）
    # initial_intersections = find_intersections(initial_trajectory)
    # predicted_intersections = find_intersections(pred_trajectory)

    # print(f"Initial Intersections: {len(initial_intersections)}")
    # print(f"Predicted Intersections: {len(predicted_intersections)}")

    # # 可选：绘制交点（如果需要显示）
    # if len(initial_intersections) > 0:
    #     ax.scatter(*zip(*initial_intersections),
    #                s=intersection_marker_size,
    #                marker=intersection_marker_style,
    #                color=initial_intersection_color,
    #                zorder=7)
    # if len(predicted_intersections) > 0:
    #     ax.scatter(*zip(*predicted_intersections),
    #                s=intersection_marker_size,
    #                marker=intersection_marker_style,
    #                color=predicted_intersection_color,
    #                zorder=7)

    # 清理并只生成一次图例安全句柄（若需要图例，可取消下面注释）
    import matplotlib.patches as mpatches
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    handles, labels = ax.get_legend_handles_labels()
    safe_pairs = [(h, l) for h, l in zip(handles, labels) if not isinstance(h, Poly3DCollection)]
    safe_handles = [h for h, _ in safe_pairs]
    safe_labels = [l for _, l in safe_pairs]
    if verts is not None:
        unreach_patch = mpatches.Patch(color='red', alpha=0.3, label='Unreachable Region')
        safe_handles.insert(0, unreach_patch)
        safe_labels.insert(0, unreach_patch.get_label())
    # 如果要显示图例，取消下一行注释
    # ax.legend(safe_handles, safe_labels, loc='upper left', bbox_to_anchor=(0.02, 0.98))

    # # 绘制 unstable pose（仅一次）
    # if num_capsize > 0:
    #     xs, ys, ysaws = zip(*capsize_points)
    #     ax.scatter(xs, ys, ysaws,
    #                s=intersection_marker_size,
    #                marker='o',
    #                color='lime',
    #                linewidth=0.5,
    #                alpha=1.0,
    #                zorder=9)
    # if num_capsize_pred > 0:
    #     xs, ys, ysaws = zip(*capsize_points_pred)
    #     ax.scatter(xs, ys, ysaws,
    #                s=intersection_marker_size,
    #                marker='D',
    #                color='cyan',
    #                linewidth=0.5,
    #                alpha=1.0,
    #                zorder=9)

    # # 保留坐标系背景面（pane），隐藏刻度文字但保留刻度线与面（仅设置一次）
    # try:
    #     ax.xaxis.pane.set_edgecolor('k')
    #     ax.yaxis.pane.set_edgecolor('k')
    #     ax.zaxis.pane.set_edgecolor('k')
    #     ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    #     ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    #     ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    #     ax.xaxis.pane.fill = True
    #     ax.yaxis.pane.fill = True
    #     ax.zaxis.pane.fill = True
    # except Exception:
    #     pass

    # try:
    #     ax.set_xticklabels([])
    #     ax.set_yticklabels([])
    #     ax.set_zticklabels([])
    # except Exception:
    #     for t in ax.xaxis.get_major_ticks():
    #         t.label.set_visible(False)
    #     for t in ax.yaxis.get_major_ticks():
    #         t.label.set_visible(False)
    #     for t in ax.zaxis.get_major_ticks():
    #         t.label.set_visible(False)

    # 关闭坐标轴
    ax.axis('off')

    # 最终显示
    plt.tight_layout()
    output_dir = predictions_dir("visualization", "yaw_stability")
    outname_pdf_intersections = output_dir / 'yaw_stability_intersections_3d.pdf'
    outname_svg_intersections = output_dir / 'yaw_stability_intersections_3d.svg'
    outname_png_intersections = output_dir / 'yaw_stability_intersections_3d.png'
    plt.savefig(outname_pdf_intersections, dpi=300, bbox_inches='tight')
    plt.savefig(outname_svg_intersections, dpi=300, bbox_inches='tight')
    plt.savefig(outname_png_intersections, dpi=300, bbox_inches='tight')
    plt.show()

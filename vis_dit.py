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

from dit.Models import PathDiffusionTransformer
from dataLoader_uneven import get_encoder_input, receptive_field
from eval_model_uneven import getHashTable, get_patch
import torch

from relative_motion_utils import relative_motion_to_trajectory

dataset_path = 'data/sim_dataset/val'
# dataset_path = 'data/sim_dataset/train'

diffusion_step = 50

def spline_interpolate(control_points, num_samples=100):
    """
    使用三次自然样条对控制点进行插值，生成平滑轨迹
    
    Args:
        control_points: 样条控制点，形状为(num_control_points, 3)，包含[x, y, yaw]
        num_samples: 插值生成的轨迹点数量，默认100
    
    Returns:
        插值后的轨迹点，形状为(num_samples, 3)，保证经过所有控制点
    """
    if control_points is None or len(control_points) < 2:
        print("Insufficient control points for spline interpolation")
        return None
    
    num_control = len(control_points)
    
    # 生成控制点参数（均匀参数化）
    t_control = np.linspace(0, 1, num_control)
    # 生成采样点参数
    t_samples = np.linspace(0, 1, num_samples)
    
    # 分开处理xy坐标和yaw角
    x_control = control_points[:, 0]
    y_control = control_points[:, 1]
    yaw_control = control_points[:, 2]
    
    # 对x, y使用三次样条插值
    x_interpolated = _evaluate_scalar_spline(x_control, t_samples, t_control)
    y_interpolated = _evaluate_scalar_spline(y_control, t_samples, t_control)
    
    # 对yaw进行周期性感知的样条插值
    yaw_interpolated = _evaluate_yaw_spline(yaw_control, t_samples, t_control)
    
    # 组合结果
    trajectory = np.column_stack([x_interpolated, y_interpolated, yaw_interpolated])
    
    return trajectory

def _solve_natural_cubic_M(y_values, t_control):
    """
    求解自然三次样条的二阶导数 M
    使用三对角矩阵算法（Thomas algorithm）
    
    Args:
        y_values: 控制点的y值，形状为(N,)
        t_control: 控制点的参数，形状为(N,)
    
    Returns:
        二阶导数M，形状为(N,)
    """
    N = len(y_values)
    if N < 2:
        return np.zeros(N)
    
    # 构建三对角系统 A*M = b
    # 自然边界条件：M[0] = M[-1] = 0
    h = np.diff(t_control)  # 根据实际参数计算间隔
    h = np.clip(h, 1e-6, None)  # 避免除零
    
    # 构建对角线
    diag = 2 * (h[:-1] + h[1:])
    diag = np.concatenate([[1], diag, [1]])  # 边界条件
    
    # 构建上下对角线
    upper = np.concatenate([[0], h[1:], [0]])
    lower = np.concatenate([[0], h[:-1], [0]])
    
    # 构建右侧向量
    b = np.zeros(N)
    for i in range(1, N - 1):
        b[i] = 6 * ((y_values[i + 1] - y_values[i]) / h[i] - 
                    (y_values[i] - y_values[i - 1]) / h[i - 1])
    
    # 边界条件（自然样条）
    b[0] = 0
    b[-1] = 0
    
    # 使用 Thomas 算法求解三对角系统
    M = _solve_tridiagonal(lower, diag, upper, b)
    
    return M

def _solve_tridiagonal(lower, diag, upper, b):
    """
    使用 Thomas 算法求解三对角线性系统
    
    Args:
        lower: 下对角线
        diag: 主对角线
        upper: 上对角线
        b: 右侧向量
    
    Returns:
        解向量 x
    """
    N = len(b)
    c_prime = np.zeros(N - 1)
    d_prime = np.zeros(N)
    x = np.zeros(N)
    
    # 前向消元
    c_prime[0] = upper[0] / diag[0]
    d_prime[0] = b[0] / diag[0]
    
    for i in range(1, N - 1):
        denom = diag[i] - lower[i] * c_prime[i - 1]
        c_prime[i] = upper[i] / denom
        d_prime[i] = (b[i] - lower[i] * d_prime[i - 1]) / denom
    
    d_prime[-1] = (b[-1] - lower[-1] * d_prime[-2]) / (diag[-1] - lower[-1] * c_prime[-2])
    
    # 回代
    x[-1] = d_prime[-1]
    for i in range(N - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    
    return x

def _evaluate_scalar_spline(y_control, t_eval, t_control):
    """
    对一维标量序列进行三次样条插值
    
    Args:
        y_control: 控制点的y值，形状为(N,)
        t_eval: 评估点的参数，形状为(M,)
        t_control: 控制点的参数，形状为(N,)
    
    Returns:
        插值后的y值，形状为(M,)
    """
    N = len(y_control)
    M_values = _solve_natural_cubic_M(y_control, t_control)  # 传入 t_control
    
    h = np.diff(t_control)
    h = np.clip(h, 1e-6, None)  # 避免除零
    
    # 找到每个评估点所在的区间
    idx = np.searchsorted(t_control[1:], t_eval, side='left')
    idx = np.clip(idx, 0, N - 2)
    
    # 获取区间端点
    t_k = t_control[idx]
    t_k1 = t_control[idx + 1]
    h_k = h[idx]
    dt = t_eval - t_k
    
    y_k = y_control[idx]
    y_k1 = y_control[idx + 1]
    M_k = M_values[idx]
    M_k1 = M_values[idx + 1]
    
    # 三次样条插值公式
    term1 = M_k * (t_k1 - t_eval)**3 / (6 * h_k)
    term2 = M_k1 * dt**3 / (6 * h_k)
    term3 = (y_k - M_k * h_k**2 / 6) * (t_k1 - t_eval) / h_k
    term4 = (y_k1 - M_k1 * h_k**2 / 6) * dt / h_k
    
    S = term1 + term2 + term3 + term4
    
    return S

def _evaluate_yaw_spline(yaw_control, t_eval, t_control):
    """
    对yaw角进行周期性感知的三次样条插值
    
    Args:
        yaw_control: 控制点的yaw角，形状为(N,)
        t_eval: 评估点的参数，形状为(M,)
        t_control: 控制点的参数，形状为(N,)
    
    Returns:
        插值后的yaw角，形状为(M,)，规范化到[-pi, pi]
    """
    # 展开角度序列，消除周期性跳跃
    yaw_unwrapped = _unwrap_angles(yaw_control)
    
    # 对展开后的角度进行标量样条插值
    yaw_interpolated_unwrapped = _evaluate_scalar_spline(yaw_unwrapped, t_eval, t_control)
    
    # 将插值结果重新规范化到 [-π, π]
    yaw_interpolated = np.arctan2(np.sin(yaw_interpolated_unwrapped), 
                                   np.cos(yaw_interpolated_unwrapped))
    
    return yaw_interpolated

def _unwrap_angles(angles):
    """
    展开角度序列，消除周期性跳跃
    
    Args:
        angles: 角度序列，形状为(N,)
    
    Returns:
        展开后的角度序列，形状为(N,)
    """
    if len(angles) <= 1:
        return angles.copy()
    
    unwrapped = np.zeros_like(angles)
    unwrapped[0] = angles[0]
    
    for i in range(1, len(angles)):
        diff = angles[i] - angles[i-1]
        # 规范化角度差到 [-π, π]
        diff = np.arctan2(np.sin(diff), np.cos(diff))
        unwrapped[i] = unwrapped[i-1] + diff
    
    return unwrapped

def generate_paths(model, map_input, start_point, goal_point, num_paths=5):
    """
    生成完整路径（绝对坐标版本，使用sin/cos编码）
    
    Args:
        model: 训练好的PathDiffusionTransformer
        map_input: (1, 6, H, W) 已包含起点/终点嵌入的地图
        start_point: (3,) [x, y, yaw] 起点坐标（真实坐标）
        goal_point: (3,) [x, y, yaw] 终点坐标（真实坐标）
        num_paths: 生成路径数量
    Returns:
        middle_paths: (num_paths, 20, 3) 中间20个点的绝对坐标 [x, y, theta]
    """
    model.eval()
    
    # 归一化起点终点并转换为4维 (x, y, sin(θ), cos(θ))
    start_normalized = torch.zeros(4, device=start_point.device)
    start_normalized[:2] = start_point[:2] / 20.0  # x,y归一化
    start_normalized[2] = torch.sin(start_point[2])  # sin(θ)
    start_normalized[3] = torch.cos(start_point[2])  # cos(θ)
    start_normalized[:2] = torch.clamp(start_normalized[:2], -1.0, 1.0)
    start_normalized = start_normalized.unsqueeze(0)  # (1, 4)
    
    goal_normalized = torch.zeros(4, device=goal_point.device)
    goal_normalized[:2] = goal_point[:2] / 20.0
    goal_normalized[2] = torch.sin(goal_point[2])  # sin(θ)
    goal_normalized[3] = torch.cos(goal_point[2])  # cos(θ)
    goal_normalized[:2] = torch.clamp(goal_normalized[:2], -1.0, 1.0)
    goal_normalized = goal_normalized.unsqueeze(0)  # (1, 4)
    
    # 生成中间20个点（4维编码）
    with torch.no_grad():
        normalized_traj = model.sample(
            map_input,
            start_normalized,
            goal_normalized,
            num_samples=num_paths,
            ddim_steps=diffusion_step
        )  # (num_paths, 20, 4) - 归一化的4维编码
    
    # 反归一化到真实坐标 (x, y, theta)
    traj_denorm = torch.zeros(num_paths, 20, 3, device=normalized_traj.device)
    traj_denorm[:, :, :2] = normalized_traj[:, :, :2] * 20.0  # x,y: [-1,1] → [-20,20]
    # 从sin/cos恢复角度
    traj_denorm[:, :, 2] = torch.atan2(normalized_traj[:, :, 2], normalized_traj[:, :, 3])  # θ ∈ [-π, π]
    
    return traj_denorm.cpu().numpy()  # (num_paths, 20, 3)

# Define the network
device='cuda' if torch.cuda.is_available() else 'cpu'

def plot_single_trajectory(ax, elevation_masked, trajectory, predTrajs=None, output_dim=None, is_pred=False, use_bezier_interpolate=False):
    """绘制单个轨迹子图的辅助函数
    
    Args:
        ax: matplotlib轴对象
        elevation_masked: 地形高程图
        trajectory: 真实轨迹 (N, 3)
        predTrajs: 预测轨迹列表，可以是：
                   - (20, 3) 单条轨迹
                   - (num_paths, 20, 3) 多条轨迹
        output_dim: 输出维度
        is_pred: 是否为预测模式
        use_bezier_interpolate: 是否使用样条插值平滑轨迹（变量名保持兼容）
    """
    # 显示地形图
    # ax.imshow(elevation_masked, extent=[-5, 5, -5, 5],
    ax.imshow(elevation_masked, extent=[-20, 20, -20, 20],
              origin='lower', cmap='terrain', aspect='equal')
    ax.grid(True, alpha=0.3)
    
    start_pos = trajectory[0, :]
    goal_pos = trajectory[-1, :]
    
    # 绘制轨迹线段
    if is_pred and predTrajs is not None:
        # 检查是单条轨迹还是多条轨迹
        if predTrajs.ndim == 2:  # 单条轨迹 (20, 3)
            predTrajs = predTrajs[np.newaxis, ...]  # 转换为 (1, 20, 3)
        
        num_paths = predTrajs.shape[0]
        
        # 为多条轨迹设置不同的颜色和透明度
        if num_paths == 1:
            alphas = [0.8]
            line_styles = ['-']
            linewidths = [2]
        else:
            # 多条轨迹使用不同透明度，主轨迹更明显
            alphas = [0.9] + [0.4] * (num_paths - 1)
            line_styles = ['-'] * num_paths
            linewidths = [2.5] + [1.5] * (num_paths - 1)
        
        # 绘制每条预测轨迹
        for traj_idx, predTraj in enumerate(predTrajs):
            # 如果启用样条插值，先对轨迹进行插值处理
            if use_bezier_interpolate:
                # 补足起点和终点形成完整控制点
                full_traj = np.vstack((start_pos, predTraj, goal_pos))  # (N+2, 3)
                # 进行样条插值，生成100个点的平滑轨迹
                predTraj_smooth = spline_interpolate(full_traj, num_samples=100)
                if predTraj_smooth is not None:
                    predTraj_path = predTraj_smooth[:, :2]  # 只取xy坐标
                    output_dim = predTraj_path.shape[0]
                else:
                    # 插值失败，使用原始方法
                    predTraj_xy = np.array([[point[0], point[1]] for point in predTraj])
                    predTraj_path = np.vstack((start_pos[:2], predTraj_xy, goal_pos[:2]))
                    output_dim = predTraj_path.shape[0]
            else:
                # 将predTraj从(x,y,theta)格式转换为只包含(x,y)的格式用于绘制路径
                predTraj_xy = np.array([[point[0], point[1]] for point in predTraj])
                # 先为预测轨迹补足起点和终点，只使用x,y坐标
                predTraj_path = np.vstack((start_pos[:2], predTraj_xy, goal_pos[:2]))
                output_dim = predTraj_path.shape[0]  # 已经包括起点和终点
            
            # 绘制轨迹线段
            for i in range(output_dim-1):
                # 主轨迹（第一条）使用单色，其他轨迹使用彩虹色
                if traj_idx == 0 and num_paths > 1:
                    color = 'orange'  # 主轨迹使用单一颜色橙色
                else:
                    color = plt.cm.rainbow(i / (output_dim - 2))  # 其他轨迹使用rainbow颜色映射
                ax.plot([predTraj_path[i][0], predTraj_path[i+1][0]], 
                        [predTraj_path[i][1], predTraj_path[i+1][1]], 
                        color=color, zorder=3 if traj_idx == 0 and num_paths > 1 else 2, linewidth=linewidths[traj_idx], 
                        marker='o', markersize=1.5, alpha=alphas[traj_idx],
                        linestyle=line_styles[traj_idx])
            
            # 只为第一条（主要）轨迹绘制角度箭头，避免过于杂乱
            if traj_idx == 0 or num_paths == 1:
                arrow_scale = 0.2
                for i, (x, y, theta) in enumerate(predTraj):
                    # color = plt.cm.rainbow(i / (len(predTraj) - 1)) if len(predTraj) > 1 else 'blue'
                    color = 'orange'
                    ax.arrow(x, y,
                             np.cos(theta) * arrow_scale,
                             np.sin(theta) * arrow_scale,
                             head_width=0.08, head_length=0.12, fc=color, ec=color, 
                             zorder=4, alpha=alphas[traj_idx])
    else:
        # 如果启用样条插值，先对真实轨迹进行插值处理
        if use_bezier_interpolate:
            trajectory_smooth = spline_interpolate(trajectory, num_samples=100)
            if trajectory_smooth is not None:
                trajectory_to_plot = trajectory_smooth
            else:
                trajectory_to_plot = trajectory
        else:
            trajectory_to_plot = trajectory
        
        # 绘制真实轨迹线段
        for i in range(trajectory_to_plot.shape[0] - 1):
            color = plt.cm.rainbow(i / (trajectory_to_plot.shape[0] - 2))  # 使用rainbow颜色映射
            ax.plot(trajectory_to_plot[i:i+2, 0], trajectory_to_plot[i:i+2, 1], color=color, zorder=3, linewidth=2, marker='o', markersize=1.5)
        
        # 绘制真实轨迹的角度箭头（从轨迹数据第三列读取）
        arrow_scale = 0.2
        for i in range(1, trajectory_to_plot.shape[0] - 1):  # 跳过起点和终点，它们单独处理
            color = plt.cm.rainbow((i-1) / (trajectory_to_plot.shape[0] - 3)) if trajectory_to_plot.shape[0] > 3 else 'green'
            # color = 'orange'
            x, y, theta = trajectory_to_plot[i, :]
            ax.arrow(x, y,
                     np.cos(theta) * arrow_scale,
                     np.sin(theta) * arrow_scale,
                     head_width=0.08, head_length=0.12, fc=color, ec=color, zorder=4, alpha=0.8)
    
    # 绘制起点和终点
    ax.scatter(start_pos[0], start_pos[1], color='purple', zorder=5, s=100, edgecolors='black', linewidth=1)
    ax.scatter(goal_pos[0], goal_pos[1], color='r', zorder=5, s=100, edgecolors='black', linewidth=1)
    
    # 绘制起点和终点的朝向箭头（使用更大的箭头表示起点终点）
    arrow_scale_large = 0.3
    ax.arrow(start_pos[0], start_pos[1],
             np.cos(start_pos[2]) * arrow_scale_large,
             np.sin(start_pos[2]) * arrow_scale_large,
             head_width=0.12, head_length=0.18, fc='purple', ec='black', zorder=6, linewidth=1)
    ax.arrow(goal_pos[0], goal_pos[1],
             np.cos(goal_pos[2]) * arrow_scale_large,
             np.sin(goal_pos[2]) * arrow_scale_large,
             head_width=0.12, head_length=0.18, fc='r', ec='black', zorder=6, linewidth=1)
    
    # 设置标题并关闭坐标轴
    title = 'Predicted Trajectory' if is_pred else 'Ground Truth Trajectory'
    ax.set_title(title, fontsize=12, pad=8)
    ax.axis('off')

def plot_elevation_map(pathNums, envType, save_path='predictions', num_pred_paths=1, use_bezier_interpolate=False):
    """绘制多组轨迹对比图
    
    Args:
        pathNums: 路径编号列表
        envType: 环境类型
        save_path: 保存路径
        num_pred_paths: 每个场景生成的预测轨迹数量（1=单条，>1=多条重叠显示）
        use_bezier_interpolate: 是否使用样条插值平滑轨迹（默认False，变量名保持兼容）
    """
    if not isinstance(pathNums, list):
        pathNums = [pathNums]  # 确保pathNums是列表
    
    # 限制最多显示6组对比
    pathNums = pathNums[:6]
    num_pairs = len(pathNums)
    
    # 创建大图和网格，不使用constrained_layout以便手动控制布局
    fig = plt.figure(figsize=(18, 13))  # 略微增加高度，为标题和颜色条预留空间
    gs = plt.GridSpec(3, 4, figure=fig, left=0.05, right=0.95, 
                     bottom=0.08, top=0.9, wspace=0.15, hspace=0.3)  # 手动设置边距和间距
    
    # 加载环境数据，所有子图共用同一个环境
    envFolder = osp.join(dataset_path, envType)
    # envFolder = osp.join('data/test_training/val', envType)
    env_path = osp.join(envFolder, f'map.p')
    with open(env_path, 'rb') as f:
        env = pickle.load(f)
        tensor = env['tensor']
        elevation = tensor[:, :, 0]
        normal_x = tensor[:, :, 1]
        normal_y = tensor[:, :, 2]
        normal_z = tensor[:, :, 3]
        elevation_masked = np.ma.masked_invalid(elevation)
    
    # 为每对轨迹创建子图
    for idx, pathNum in enumerate(pathNums):
        # 计算当前对应在网格中的位置
        row = idx // 2
        col = (idx % 2) * 2
        
        # 加载真实轨迹数据
        path_file = osp.join(envFolder, f'path_{pathNum}.p')
        with open(path_file, 'rb') as f:
            path_data = pickle.load(f)
            trajectory = path_data['path']  # [N, 3]
        
        start_pos = trajectory[0, :]
        goal_pos = trajectory[-1, :]
        
        # print(f"True_x range: {min(trajectory[:, 0])} to {max(trajectory[:, 0])}")
        # print(f"True_y range: {min(trajectory[:, 1])} to {max(trajectory[:, 1])}")
        
        # print(f"True Traj: {trajectory}")
        
        # 获取预测轨迹
        # patch_map, predProb, predTraj = get_patch(transformer, start_pos[:2], goal_pos[:2], normal_x, normal_y, normal_z)
        # patch_map, predProb, predTraj = get_patch(transformer, start_pos, goal_pos, normal_x, normal_y, normal_z)
        # output_dim = patch_map.shape[0]
        # print(f"normal_z shape: {normal_z.shape}")
        
        # encoder_input = get_encoder_input(normal_z, goal_pos, start_pos, normal_x, normal_y)
        # # 将numpy数组转换为PyTorch张量并确保正确的数据类型和维度顺序
        # if isinstance(encoder_input, np.ndarray):
        #     # 从 [H, W, C] 转换为 [C, H, W]
        #     encoder_input = torch.from_numpy(encoder_input).permute(2, 0, 1).float()
        # elif isinstance(encoder_input, torch.Tensor):
        #     # 确保维度顺序正确
        #     if encoder_input.dim() == 3 and encoder_input.shape[-1] == 4:
        #         encoder_input = encoder_input.permute(2, 0, 1).float()
        #     else:
        #         encoder_input = encoder_input.float()
        # else:
        #     encoder_input = torch.tensor(encoder_input, dtype=torch.float32)
        #     if encoder_input.dim() == 3 and encoder_input.shape[-1] == 4:
        #         encoder_input = encoder_input.permute(2, 0, 1)
        
        encoder_input = torch.tensor(np.concatenate((
            normal_x[:, :, None],  # [H, W, 1]
            normal_y[:, :, None],  # [H, W, 1]
            normal_z[:, :, None]   # [H, W, 1]
        ), axis=2), dtype=torch.float32)  # [H, W, 3]

        predTrajs = generate_paths(model, 
                                  map_input=encoder_input.permute(2, 0, 1)[None, :].cuda(),
                                  start_point=torch.tensor(start_pos).float().to(device),
                                  goal_point=torch.tensor(goal_pos).float().to(device),
                                  num_paths=num_pred_paths  # 生成多条轨迹
                                 )  # (num_pred_paths, 20, 3)
        
        # 如果只有一条轨迹，squeeze掉第一维
        if num_pred_paths == 1:
            predTrajs = predTrajs.squeeze(0)  # (20, 3)
            output_dim = predTrajs.shape[0]
        else:
            output_dim = predTrajs.shape[1]  # 20
        
        if num_pred_paths == 1:
            print(f"Predicted Traj: {predTrajs}")
            print(f"Start Pos: {start_pos}, Goal Pos: {goal_pos}")
        else:
            print(f"Predicted Traj[0]: {predTrajs[0]}")
            print(f"Start Pos: {start_pos}, \nGoal Pos: {goal_pos}")
            # print(f"Generated {num_pred_paths} trajectory samples")
        
        # 创建左侧子图 - 预测轨迹
        ax_pred = fig.add_subplot(gs[row, col])
        # 为预测轨迹补足起点和终点
        plot_single_trajectory(ax_pred, elevation_masked, trajectory, predTrajs, output_dim, is_pred=True, use_bezier_interpolate=use_bezier_interpolate)
        
        # 创建右侧子图 - 真实轨迹
        ax_true = fig.add_subplot(gs[row, col+1])
        plot_single_trajectory(ax_true, elevation_masked, trajectory, is_pred=False, use_bezier_interpolate=use_bezier_interpolate)
        
        # 添加路径编号 - 调整位置和样式使其更加醒目
        ax_pred.text(0.02, 0.98, f"Path #{pathNum}", transform=ax_pred.transAxes, 
                   fontsize=11, weight='bold', verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.3, edgecolor='gray'))
    
    # 为轨迹的颜色添加全局颜色条 - 调整位置到图形底部，更合理的位置
    norm = plt.Normalize(0, output_dim - 1)
    cmap = plt.cm.rainbow
    # 调整颜色条位置，放置在整个图形的底部
    cbar_ax = fig.add_axes([0.3, 0.02, 0.4, 0.015])  # [left, bottom, width, height]
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
                       cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Trajectory Step', fontsize=12)
    
    # 添加超级标题，调整位置确保不被裁剪
    plt.subplots_adjust(top=0.92)  # 为标题腾出更多空间
    title_suffix = f' ({num_pred_paths} samples per scene)' if num_pred_paths > 1 else ''
    fig.suptitle(f'Trajectory Predictions for {envType.capitalize()} Environment{title_suffix}', 
                fontsize=18, y=0.98, fontweight='bold')
    
    # 保存图像
    save_path = osp.join(save_path, f'{envType}')
    os.makedirs(save_path, exist_ok=True)
    
    # 生成路径编号列表的简短表示，例如 1_2_3
    path_ids_str = "_".join(str(p) for p in pathNums)
    plt.savefig(osp.join(save_path, f'multi_trajectories_{path_ids_str}.png'), dpi=300)
    print(f"Saved multi-trajectory comparison figure for paths {path_ids_str} in {envType} environment.")
    print("fig saved to", osp.join(save_path, f'multi_trajectories_{path_ids_str}.png'))
    

if __name__ == "__main__":
    best = True
    # best = False
    epoch = 4
    stage = 2
    
    # =================== 多轨迹配置 ===================
    # 设置为1: 只生成单条轨迹（确定性预测）
    # 设置为>1: 生成多条轨迹（展示扩散模型的多峰性）
    # num_pred_paths = 100  # 每个场景生成100条不同的轨迹
    # num_pred_paths = 20  # 每个场景生成20条不同的轨迹
    num_pred_paths = 10  # 每个场景生成5条不同的轨迹
    # num_pred_paths = 5  # 每个场景生成5条不同的轨迹
    # num_pred_paths = 1  # 单条轨迹模式
    # ================================================
    
    # =================== 样条插值配置 ===================
    # True: 使用三次样条插值生成平滑轨迹（100个点，保证经过所有控制点）
    # False: 直接使用控制点连接（默认，更快）
    use_bezier_interpolate = True  # 启用样条插值
    # use_bezier_interpolate = False  # 禁用样条插值
    # ==================================================
    # 注意：变量名保持为 use_bezier_interpolate 以兼容现有代码，实际使用样条插值

    envNum = np.random.randint(0, 99)  # 随机选择环境id
    # envType_list = [f'env{envNum:06d}']  # 生成环境列表，格式为 env000000, env000001, ..., env000009
    envType_list = ['env000008']  # 生成环境列表，格式为 env000000, env000001, ..., env000009
    save_path = 'predictions'

    modelFolder = 'data/sim'
    modelFile = osp.join(modelFolder, f'model_params.json')
    model_param = json.load(open(modelFile))

    model = PathDiffusionTransformer(**model_param['model_args'])
    _ = model.to(device)

    # checkpoint = torch.load(osp.join(modelFolder, f'model_epoch_{epoch}.pkl'))
    
    if best:
        checkpoint = torch.load(osp.join(modelFolder, f'stage{stage}_best_model.pth'))
        print(f"Loaded best stage {stage} model.")
    else:
        checkpoint = torch.load(osp.join(modelFolder, f'checkpoint_stage{stage}_epoch_{epoch}.pth'))
        print(f"Loaded stage {stage} model from epoch {epoch}.")
    
    model.load_state_dict(checkpoint['model_state_dict'])

    _ = model.eval()

    # envType_random = np.random.choice(envType_list, size=1)[0]
    # 随机选择一个路径用于概率图对比
    # path_index = np.random.choice(range(500), size=1)[0]
    # path_index_list = list(np.random.choice(range(200), size=6, replace=False))
    # path_index_list = list([163, 119, 340, 416, 148, 260])
    # path_index_list = list([0, 1, 2, 3, 4, 5])
    # path_index_list = list([2, 3, 7, 17, 23, 25])
    # path_index_list = list([0, 1, 2, 3, 4, 4])  # 测试前5条路径
    # path_index_list = list([5, 6, 7, 8, 9, 10])  # 测试前5条路径
    # path_index_list = list([10, 11, 12, 13, 14, 15])  # 测试前5条路径
    # path_index_list = list([16, 17, 18, 19, 20, 21])  # 测试前5条路径
    # path_index_list = list([22, 23, 24, 25, 26, 27])  # 测试前5条路径
    # path_index_list = list([28, 29, 30, 31, 32, 33])  # 测试前5条路径
    # path_index_list = list([34, 35, 36, 37, 38, 39])  # 测试前5条路径
    path_index_list = list([40, 41, 42, 43, 44, 45])  # 测试前5条路径
    # path_index_list = list([46, 47, 48, 49, 44, 45])  # 测试前5条路径
    # print(f"Evaluating environment: {envType_random}")
    print(f"Evaluating path index: {path_index_list}")

    # # 绘制多条轨迹的预测概率图和GT标签图对比
    # for path_index in path_index_list:
    #     plot_predProb_map(path_index, envType_random, save_path)
        
    # # 绘制多组轨迹对比图
    # plot_elevation_map(path_index_list, envType_random, save_path)
    
    # plot_predProb_map(2, envType_list[0], save_path)
    
    for env in envType_list:
        print(f"Evaluating environment: {env}")
        print(f"Generating {num_pred_paths} trajectory sample(s) per scene")
        print(f"Spline interpolation: {'Enabled' if use_bezier_interpolate else 'Disabled'}")

        # 绘制多组轨迹对比图
        plot_elevation_map(path_index_list, env, save_path, num_pred_paths=num_pred_paths, use_bezier_interpolate=use_bezier_interpolate)
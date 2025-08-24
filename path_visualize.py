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
import torch

# dataset_path = 'data/terrain_dataset/val'
dataset_path = 'data/terrain/train'

def generate_ground_truth_labels(trajectory, hashTable, output_dim):
    """
    生成ground truth标签图，与dataLoader_uneven中的逻辑相同
    使用geom2pixMatpos函数来确定正标签位置
    
    Args:
        trajectory: [N, 3] 轨迹点，包含(x, y, theta)
        hashTable: 锚点哈希表
        output_dim: 输出维度（时间步数）
    
    Returns:
        gt_labels: [output_dim, num_tokens] ground truth标签分布
    """
    from dataLoader_uneven import receptive_field, geom2pixMatpos
    
    # 直接使用dataLoader_uneven中已修复的geom2pixMatpos函数
    # 不再在这里重新实现，确保与训练时使用的完全一致
    
    num_tokens = len(hashTable)
    gt_labels = []
    
    # 只使用中间的轨迹点（去掉起点和终点）
    middle_trajectory = trajectory[1:-1, :]  # [num_steps, 3]
    
    # 确保轨迹点数量与output_dim匹配
    num_steps = min(len(middle_trajectory), output_dim)
    
    for step in range(output_dim):
        # 创建当前时间步的标签图
        step_labels = torch.zeros(num_tokens)
        
        if step < num_steps:
            # 获取当前时间步的真实轨迹点
            true_point = middle_trajectory[step]  # [3] (x, y, theta)
            pos = true_point[:2]  # 只取(x, y)坐标
            
            # 使用geom2pixMatpos函数找到正样本锚点
            positive_indices = geom2pixMatpos(pos)
            
            # 标记正样本位置
            if len(positive_indices[0]) > 0:  # 检查是否有正样本
                step_labels[positive_indices[0]] = 1.0
        
        gt_labels.append(step_labels)
    
    return torch.stack(gt_labels, dim=0)  # [output_dim, num_tokens]

# Define the network
device='cuda' if torch.cuda.is_available() else 'cpu'

def plot_single_trajectory(ax, elevation_masked, trajectory, predTraj=None, output_dim=None, is_pred=False):
    """绘制单个轨迹子图的辅助函数（使用平滑插值曲线代替折线）"""
    import math
    # 尝试使用 scipy 的样条插值；不可用时退回到线性插值并做简单平滑
    def smooth_trajectory(traj, num=200):
        """
        traj: (N,2) or (N,3) array (x,y[,theta])
        返回 (num,3) 的平滑轨迹（x,y,theta）
        """
        traj = np.asarray(traj, dtype=np.float64)
        if traj.shape[0] < 2:
            # 不足两个点，返回原样并补theta=0
            x = np.repeat(traj[:,0], num) if traj.shape[0] > 0 else np.zeros(num)
            y = np.repeat(traj[:,1], num) if traj.shape[0] > 0 else np.zeros(num)
            theta = np.zeros_like(x)
            return np.stack([x, y, theta], axis=1)

        x = traj[:,0]
        y = traj[:,1]

        try:
            from scipy import interpolate as si
            k = min(3, len(x)-1)
            # 如果只有两个点，splprep 要求 k <= 1，所以使用 k = 1
            tck, u = si.splprep([x, y], s=0.0, k=k)
            u_fine = np.linspace(0, 1, num)
            x_fine, y_fine = si.splev(u_fine, tck)
            dx, dy = si.splev(u_fine, tck, der=1)
            theta_fine = np.arctan2(dy, dx)
            return np.stack([x_fine, y_fine, theta_fine], axis=1)
        except Exception:
            # fallback: 基于累积弧长的线性插值再平滑
            dist = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
            u = np.zeros(len(x))
            u[1:] = np.cumsum(dist)
            if u[-1] == 0:
                u = np.linspace(0, 1, len(x))
            else:
                u = u / u[-1]
            u_fine = np.linspace(0, 1, num)
            x_fine = np.interp(u_fine, u, x)
            y_fine = np.interp(u_fine, u, y)
            # 简单平滑：三点滑动平均（边界重复）
            def smooth_arr(a, win=5):
                if len(a) < win:
                    return a
                pad = win//2
                a_pad = np.pad(a, (pad,pad), mode='edge')
                kernel = np.ones(win)/win
                return np.convolve(a_pad, kernel, mode='valid')
            x_s = smooth_arr(x_fine, win=7)
            y_s = smooth_arr(y_fine, win=7)
            dx = np.gradient(x_s)
            dy = np.gradient(y_s)
            theta_fine = np.arctan2(dy, dx)
            return np.stack([x_s, y_s, theta_fine], axis=1)

    # 显示地形图（保留）
    ax.imshow(elevation_masked, extent=[-5, 5, -5, 5],
              origin='lower', cmap='terrain', aspect='equal')
    ax.grid(True, alpha=0.3)

    start_pos = trajectory[0, :]
    goal_pos = trajectory[-1, :]

    # 绘制平滑轨迹
    if is_pred and predTraj is not None:
        # 合并起点、预测中间点、终点然后平滑
        pred_arr = np.asarray(predTraj, dtype=np.float64)
        if pred_arr.ndim == 1:
            pred_arr = pred_arr.reshape(-1, 3)
        pred_path = np.vstack((start_pos[:2], pred_arr[:, :2], goal_pos[:2]))
        smooth = smooth_trajectory(pred_path, num=200)
        # smooth = smooth_trajectory(trajectory, num=200)
        # 绘制连续平滑曲线，并用颜色渐变标识时间步
        N = smooth.shape[0]
        for i in range(N - 1):
        # for i in range(0, N - 1, N//11):
            color = plt.cm.rainbow(i / max(1, N - 2))
            ax.plot(smooth[i:i+2,0], smooth[i:i+2,1], color=color, zorder=3, linewidth=15)
            # ax.scatter(smooth[i, 0], smooth[i, 1], color=color, zorder=3, s=200)
        # # 在平滑曲线上绘制方向箭头，使用插值得到的theta
        # arrow_scale = 0.2
        # step = max(1, N // 12)
        # for i in range(0, N, step):
        #     x, y, theta = smooth[i]
        #     color = plt.cm.rainbow(i / max(1, N - 1))
        #     ax.arrow(x, y,
        #              math.cos(theta) * arrow_scale,
        #              math.sin(theta) * arrow_scale,
        #              head_width=0.2, head_length=0.3, fc=color, ec=color, zorder=4, alpha=0.9)
    else:
        # 真实轨迹平滑绘制
        real = np.asarray(trajectory, dtype=np.float64)
        real_path = real[:, :2]
        smooth = smooth_trajectory(real_path, num=200)
        N = smooth.shape[0]
        for i in range(N - 1):
            color = plt.cm.rainbow(i / max(1, N - 2))
            ax.plot(smooth[i:i+2,0], smooth[i:i+2,1], color=color, zorder=3, linewidth=2)
        # 绘制方向箭头（跳步绘制以减少杂乱）
        arrow_scale = 0.2
        step = max(1, N // 12)
        for i in range(0, N, step):
            x, y, theta = smooth[i]
            color = plt.cm.rainbow(i / max(1, N - 1))
            ax.arrow(x, y,
                     math.cos(theta) * arrow_scale,
                     math.sin(theta) * arrow_scale,
                     head_width=0.2, head_length=0.3, fc=color, ec=color, zorder=4, alpha=0.9)

    # 起点/终点保持明显标记
    ax.scatter(start_pos[0], start_pos[1], color='purple', zorder=6, s=900, edgecolors='black', linewidth=1)
    ax.scatter(goal_pos[0], goal_pos[1], color='r', zorder=6, s=900, edgecolors='black', linewidth=1)

    # 起点终点朝向箭头
    arrow_scale_large = 0.5
    ax.arrow(start_pos[0], start_pos[1],
             np.cos(start_pos[2]) * arrow_scale_large,
             np.sin(start_pos[2]) * arrow_scale_large,
             head_width=0.3, head_length=0.45, fc='purple', ec='black', zorder=7, linewidth=5)
    ax.arrow(goal_pos[0], goal_pos[1],
             np.cos(goal_pos[2]) * arrow_scale_large,
             np.sin(goal_pos[2]) * arrow_scale_large,
             head_width=0.3, head_length=0.45, fc='r', ec='black', zorder=7, linewidth=5)

    # 不绘制标题，保留关闭坐标轴
    # ax.set_title(title, fontsize=12, pad=8)
    ax.axis('off')

def plot_elevation_map(pathNums, envType, save_path='predictions'):
    """绘制多组轨迹对比图"""
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
        patch_map, predProb, predTraj = get_patch(transformer, start_pos, goal_pos, normal_x, normal_y, normal_z)
        output_dim = patch_map.shape[0]
        # print(f"normal_z shape: {normal_z.shape}")
        
        # print(f"Predicted Traj: {predTraj}")
        
        # 创建左侧子图 - 预测轨迹
        ax_pred = fig.add_subplot(gs[row, col])
        # 为预测轨迹补足起点和终点
        plot_single_trajectory(ax_pred, elevation_masked, trajectory, predTraj, output_dim, is_pred=True)
        
        # 创建右侧子图 - 真实轨迹
        ax_true = fig.add_subplot(gs[row, col+1])
        plot_single_trajectory(ax_true, elevation_masked, trajectory, is_pred=False)
        
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
    fig.suptitle(f'Trajectory Predictions for {envType.capitalize()} Environment', 
                fontsize=18, y=0.98, fontweight='bold')
    
    # 保存图像
    save_path = osp.join(save_path, f'{envType}')
    os.makedirs(save_path, exist_ok=True)
    
    # 生成路径编号列表的简短表示，例如 1_2_3
    path_ids_str = "_".join(str(p) for p in pathNums)
    plt.savefig(osp.join(save_path, f'multi_trajectories_{path_ids_str}.png'), dpi=300)
    
def plot_predProb_map(pathNum, envType, save_path='predictions'):
    """
    绘制单条轨迹的预测概率分布图和对应的ground truth标签图
    显示output_dim个时间步，每个时间步包含预测概率图和GT标签图
    
    Args:
        pathNum: 单个路径编号
        envType: 环境类型
        save_path: 保存路径
    """
    # 加载环境数据
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
    
    # 加载真实轨迹数据
    path_file = osp.join(envFolder, f'path_{pathNum}.p')
    with open(path_file, 'rb') as f:
        path_data = pickle.load(f)
        trajectory = path_data['path']  # [N, 3]
    
    start_pos = trajectory[0, :]
    goal_pos = trajectory[-1, :]
    
    # 获取预测轨迹和概率分布
    patch_map, predProb_list, predTraj = get_patch(transformer, start_pos, goal_pos, normal_x, normal_y, normal_z)
    output_dim = predProb_list.shape[0]  # [output_dim, num_tokens]
    
    # 获取hashTable用于生成ground truth标签
    hashTable = getHashTable(normal_z.shape)
    
    # 生成ground truth标签
    gt_labels = generate_ground_truth_labels(trajectory, hashTable, output_dim)  # [output_dim, num_tokens]
    
    # 创建子图网格：每行显示一个时间步，左侧是预测概率图，右侧是GT标签图
    fig = plt.figure(figsize=(10, 4 * output_dim))
    gs = plt.GridSpec(output_dim, 2, figure=fig, left=0.1, right=0.9, 
                     bottom=0.05, top=0.95, wspace=0.2, hspace=0.3)
    
    # 为每个时间步创建子图
    for step in range(output_dim):
        # 获取当前时间步的预测概率和GT标签
        pred_prob_step = predProb_list[step].detach().cpu().numpy()  # [num_tokens]
        gt_labels_step = gt_labels[step].numpy()  # [num_tokens]
        
        # 将1D概率分布重塑为2D图像以便可视化
        # 这里需要根据hashTable的空间分布来重塑
        map_height, map_width = normal_z.shape
        
        # 创建空的概率图和标签图
        pred_prob_map = np.zeros((map_height, map_width))
        gt_labels_map = np.zeros((map_height, map_width))
        
        # 将锚点的概率值和标签值映射到对应的像素位置
        for token_idx, (anchor_row, anchor_col) in enumerate(hashTable):
            if 0 <= anchor_row < map_height and 0 <= anchor_col < map_width:
                pred_prob_map[anchor_col, anchor_row] = pred_prob_step[token_idx]
                gt_labels_map[anchor_col, anchor_row] = gt_labels_step[token_idx]
        
        # 左侧子图 - 预测概率分布
        ax_pred = fig.add_subplot(gs[step, 0])
        im_pred = ax_pred.imshow(pred_prob_map, extent=[-5, 5, -5, 5],
                                origin='lower', cmap='viridis', aspect='equal')
        ax_pred.set_title(f'Predicted Probability - Step {step}', fontsize=12, pad=8)
        ax_pred.axis('off')
        
        # 在预测图上叠加轨迹点
        if step < len(trajectory):  # 确保索引有效
            traj_point = trajectory[step+1]  # +1因为去掉了起点
            ax_pred.scatter(traj_point[0], traj_point[1], color='red', s=50, marker='x', linewidth=3)
        
        # 右侧子图 - Ground Truth标签分布
        ax_gt = fig.add_subplot(gs[step, 1])
        im_gt = ax_gt.imshow(gt_labels_map, extent=[-5, 5, -5, 5],
                            origin='lower', cmap='Reds', aspect='equal', vmin=0, vmax=1)
        ax_gt.set_title(f'Ground Truth Labels - Step {step}', fontsize=12, pad=8)
        ax_gt.axis('off')
        
        # 在GT图上叠加轨迹点
        if step < len(trajectory):  # 确保索引有效
            traj_point = trajectory[step+1]  # +1因为去掉了起点
            ax_gt.scatter(traj_point[0], traj_point[1], color='blue', s=50, marker='x', linewidth=3)
        
        # 添加颜色条
        # if step == 0:  # 只在第一行添加颜色条说明
        #     plt.colorbar(im_pred, ax=ax_pred, shrink=0.8, label='Probability')
        #     plt.colorbar(im_gt, ax=ax_gt, shrink=0.8, label='Label')
            
        plt.colorbar(im_pred, ax=ax_pred, shrink=0.8, label='Probability')
        plt.colorbar(im_gt, ax=ax_gt, shrink=0.8, label='Label')
    
    # 添加超级标题
    fig.suptitle(f'Prediction vs Ground Truth - Path #{pathNum} in {envType.capitalize()} Environment', 
                fontsize=16, y=0.98, fontweight='bold')
    
    # 保存图像
    save_dir = osp.join(save_path, f'{envType}')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(osp.join(save_dir, f'prob_comparison_path_{pathNum}.png'), dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图像以释放内存

# ===== 修改：只绘制一条预测轨迹，在高程图上叠加 =====
def plot_single_prediction_on_elevation(pathNum, envType, save_path='predictions'):
    """
    仅在高程图上绘制一条预测轨迹（包含起点和终点），使用平滑曲线，去掉标题和边距留白。
    """
    # 加载环境数据
    envFolder = osp.join(dataset_path, envType)
    env_path = osp.join(envFolder, f'map.p')
    with open(env_path, 'rb') as f:
        env = pickle.load(f)
        tensor = env['tensor']
        elevation = tensor[:, :, 0]
        normal_x = tensor[:, :, 1]
        normal_y = tensor[:, :, 2]
        normal_z = tensor[:, :, 3]
        elevation_masked = np.ma.masked_invalid(elevation)

    # 加载真实轨迹数据
    path_file = osp.join(envFolder, f'path_{pathNum}.p')
    with open(path_file, 'rb') as f:
        path_data = pickle.load(f)
        trajectory = path_data['path']  # [N, 3]

    start_pos = trajectory[0, :]
    goal_pos = trajectory[-1, :]

    # 获取预测轨迹（使用已有 get_patch 接口）
    patch_map, predProb, predTraj = get_patch(transformer, start_pos, goal_pos, normal_x, normal_y, normal_z)

    # 绘图：单张高程图并叠加平滑的预测轨迹
    fig, ax = plt.subplots(figsize=(8, 8))
    # 使用 plot_single_trajectory 来绘制（内部已做平滑）
    plot_single_trajectory(ax, elevation_masked, trajectory, predTraj=predTraj, is_pred=True)

    # 去掉标题并压缩边距、去空白
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    save_dir = osp.join(save_path, envType)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(osp.join(save_dir, f'pred_traj_path_{pathNum}.png'), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    # stage = 1
    # epoch = 99
    stage = 2
    epoch = 79
    # envType_list = ['desert']
    envNum = np.random.randint(0, 99)  # 随机选择环境id
    envType_list = ['env000009']
    save_path = 'predictions'

    modelFolder = 'data/uneven'
    modelFile = osp.join(modelFolder, f'model_params.json')
    model_param = json.load(open(modelFile))

    transformer = Models.UnevenTransformer(**model_param)
    _ = transformer.to(device)

    if stage == 1:
        checkpoint = torch.load(osp.join(modelFolder, f'stage1_model_epoch_{epoch}.pkl'))
    else:
        checkpoint = torch.load(osp.join(modelFolder, f'stage2_model_epoch_{epoch}.pkl'))
    transformer.load_state_dict(checkpoint['state_dict'])
    _ = transformer.eval()

    # 只绘制一条预测轨迹：从列表中选择第一个路径编号
    path_index_list = list([30, 47, 48, 49, 44, 45])
    chosen_path = path_index_list[0]
    chosen_env = envType_list[0]

    print(f"绘制环境 {chosen_env} 的路径 #{chosen_path} 的预测轨迹")
    plot_single_prediction_on_elevation(chosen_path, chosen_env, save_path)
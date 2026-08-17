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

from dit.Models import CostConditionedPathDiffusionTransformer
from dataLoader_uneven import get_encoder_input, receptive_field
from eval_model_uneven import getHashTable, get_patch
import torch

from relative_motion_utils import relative_motion_to_trajectory
from tools._paths import PREDICTIONS_ROOT


DEFAULT_PREDICTIONS_DIR = str(PREDICTIONS_ROOT)

dataset_path = 'data/sim_dataset/val'
# dataset_path = 'data/sim_dataset/train'

diffusion_step = None

def generate_paths(model, map_input, start_point, goal_point, cost_scalar=0.0, num_paths=5, 
                   reconstruct_trajectory=True, num_traj_points=100, solver='heun', w=1.0):
    """
    生成完整路径（使用B样条控制点重建）- Rectified Flow版本
    
    Args:
        model: 训练好的CostConditionedPathDiffusionTransformer
        map_input: (1, 3, H, W) 地图输入
        start_point: (3,) [x, y, yaw] 起点坐标（真实坐标）
        goal_point: (3,) [x, y, yaw] 终点坐标（真实坐标）
        num_paths: 生成路径数量
        reconstruct_trajectory: 是否从控制点重建轨迹（默认True）
        num_traj_points: 重建后的轨迹点数（默认100）
        solver: ODE求解器类型 ('euler' 或 'heun')
    Returns:
        trajectories: (num_paths, N, 3) 轨迹 [x, y, theta]
            - 如果reconstruct_trajectory=True: N=num_traj_points
            - 如果reconstruct_trajectory=False: N=n_path_steps（控制点数）
    """
    model.eval()
    
    # 归一化起点终点并转换为4维 (x, y, cos(θ), sin(θ))
    start_normalized = torch.zeros(4, device=start_point.device)
    start_normalized[:2] = start_point[:2] / 20.0  # x,y归一化
    start_normalized[2] = torch.cos(start_point[2])  # cos(θ)
    start_normalized[3] = torch.sin(start_point[2])  # sin(θ)
    start_normalized[:2] = torch.clamp(start_normalized[:2], -1.0, 1.0)
    start_normalized = start_normalized.unsqueeze(0)  # (1, 4)
    
    goal_normalized = torch.zeros(4, device=goal_point.device)
    goal_normalized[:2] = goal_point[:2] / 20.0
    goal_normalized[2] = torch.cos(goal_point[2])  # cos(θ)
    goal_normalized[3] = torch.sin(goal_point[2])  # sin(θ)
    goal_normalized[:2] = torch.clamp(goal_normalized[:2], -1.0, 1.0)
    goal_normalized = goal_normalized.unsqueeze(0)  # (1, 4)
    
    # 从Rectified Flow采样轨迹（只有x,y）
    with torch.no_grad():
        sampled_traj_xy = model.sample(
            map_input,
            start_normalized,
            goal_normalized,
            cost_scalar=cost_scalar,
            num_samples=num_paths,
            num_steps=diffusion_step,  # 使用新参数名
            solver=solver,  # 选择ODE求解器
            reconstruct_trajectory=reconstruct_trajectory,
            num_traj_points=num_traj_points,
            w=w
        )  # (num_paths, N, 2) - 只有(x,y)，已经是真实坐标（非归一化）
    
    # 从xy坐标计算theta（通过差分计算切向量）
    def compute_theta_from_xy(traj_xy):
        """
        从轨迹的xy坐标计算theta角度
        Args:
            traj_xy: (N, 2) 轨迹的xy坐标
        Returns:
            theta: (N,) 每个点的切向角度
        """
        N = traj_xy.shape[0]
        theta = np.zeros(N)
        
        # 中心差分
        for i in range(1, N-1):
            dx = traj_xy[i+1, 0] - traj_xy[i-1, 0]
            dy = traj_xy[i+1, 1] - traj_xy[i-1, 1]
            theta[i] = np.arctan2(dy, dx)
        
        # 边界点用前向/后向差分
        dx = traj_xy[1, 0] - traj_xy[0, 0]
        dy = traj_xy[1, 1] - traj_xy[0, 1]
        theta[0] = np.arctan2(dy, dx)
        
        dx = traj_xy[-1, 0] - traj_xy[-2, 0]
        dy = traj_xy[-1, 1] - traj_xy[-2, 1]
        theta[-1] = np.arctan2(dy, dx)
        
        return theta
    
    # 转换为numpy并添加theta维度
    sampled_traj_xy_np = sampled_traj_xy.cpu().numpy()  # (num_paths, N, 2)
    
    trajectories = []
    for i in range(num_paths):
        traj_xy = sampled_traj_xy_np[i]  # (N, 2)
        theta = compute_theta_from_xy(traj_xy)  # (N,)
        traj_full = np.column_stack([traj_xy, theta])  # (N, 3)
        trajectories.append(traj_full)
    
    trajectories = np.stack(trajectories, axis=0)  # (num_paths, N, 3)
    
    return trajectories

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
            # 预测轨迹已经通过sample方法中的B样条重建完成，直接使用
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
        # Ground Truth直接绘制，不进行样条插值
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

def plot_elevation_map(pathNums, envType, save_path=DEFAULT_PREDICTIONS_DIR, num_pred_paths=1, use_bezier_interpolate=False, w=1.0):
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
                                  cost_scalar=cost_scalar,
                                  num_paths=num_pred_paths,  # 生成多条轨迹
                                  reconstruct_trajectory=True,  # 从控制点重建轨迹
                                  num_traj_points=100,  # 重建为100个点
                                  solver=solver,  # 传递求解器类型
                                  w=w
                                 )  # (num_pred_paths, 100, 3) - 100个重建的轨迹点
        
        # 如果只有一条轨迹，squeeze掉第一维
        if num_pred_paths == 1:
            predTrajs = predTrajs.squeeze(0)  # (100, 3)
            output_dim = predTrajs.shape[0]  # 100
        else:
            output_dim = predTrajs.shape[1]  # 100
        
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
    
    # ema = True
    ema = False
    # ema_decay = 0.99
    ema_decay = 0.999
    
    # =================== 多轨迹配置 ===================
    # 设置为1: 只生成单条轨迹（确定性预测）
    # 设置为>1: 生成多条轨迹（展示扩散模型的多峰性）
    # num_pred_paths = 100  # 每个场景生成100条不同的轨迹
    num_pred_paths = 20  # 每个场景生成20条不同的轨迹
    # num_pred_paths = 10  # 每个场景生成5条不同的轨迹
    # num_pred_paths = 5  # 每个场景生成5条不同的轨迹
    # num_pred_paths = 1  # 单条轨迹模式
    # ================================================
    
    # =================== ODE求解器配置 ===================
    # 'pmf_onestep': 一步预测方法（快速）
    # 'euler': 一阶Euler方法（快速，但精度较低）
    # 'heun': 二阶Heun方法（较慢，但精度更高）
    # solver = 'pmf_onestep'  # 快速的一步预测方法
    solver = 'pmf_refined'
    # solver = 'euler'  # 可选：速度更快
    diffusion_step = 3
    # solver = 'heun'  # 推荐：精度更高
    # diffusion_step = 50
    # ==================================================
    
    # cost_scalar=0.65  # 成本权重，路径生成时对成本期望值
    # # cfg_w = 3.5  # CFG guidance权重：1.0=不引导，>1更强条件引导
    # cfg_w = 1.5  # CFG guidance权重：1.0=不引导，>1更强条件引导
    
    # cost_mean = 1.4
    # cost_std = 0.2
    # cost_scalar = (cost_scalar - cost_mean) / cost_std  # 标准化成本权重
    
    cost_scalar = 0.65
    cfg_w = 1.5
    
    # =================== 样条插值配置 ===================
    # True: 使用三次样条插值生成平滑轨迹（100个点，保证经过所有控制点）
    # False: 直接使用控制点连接（默认，更快）
    use_bezier_interpolate = True  # 启用样条插值
    # use_bezier_interpolate = False  # 禁用样条插值
    # ==================================================
    # 注意：变量名保持为 use_bezier_interpolate 以兼容现有代码，实际使用样条插值

    envNum = np.random.randint(0, 99)  # 随机选择环境id
    # envType_list = [f'env{envNum:06d}']  # 生成环境列表，格式为 env000000, env000001, ..., env000009
    envType_list = ['env000010']  # 生成环境列表，格式为 env000000, env000001, ..., env000009
    save_path = DEFAULT_PREDICTIONS_DIR

    modelFolder = 'data/sim'
    modelFile = osp.join(modelFolder, f'model_params.json')
    model_param = json.load(open(modelFile))

    model = CostConditionedPathDiffusionTransformer(**model_param['model_args'])
    _ = model.to(device)

    # checkpoint = torch.load(osp.join(modelFolder, f'model_epoch_{epoch}.pkl'))
    
    if best:
        if ema:
            checkpoint = torch.load(osp.join(modelFolder, f'best_ema_{ema_decay}.pth'))
            print(f"Loaded best EMA model.")
        else:
            checkpoint = torch.load(osp.join(modelFolder, f'best_model.pth'))
            print(f"Loaded best model.")
    else:
        checkpoint = torch.load(osp.join(modelFolder, f'checkpoint_epoch_{epoch}.pth'))
        print(f"Loaded model from epoch {epoch}.")
    
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
        print(f"ODE Solver: {solver}")
        print(f"CFG w: {cfg_w}")
        print(f"Spline interpolation: {'Enabled' if use_bezier_interpolate else 'Disabled'}")

        # 绘制多组轨迹对比图
        plot_elevation_map(path_index_list, env, save_path, num_pred_paths=num_pred_paths, use_bezier_interpolate=use_bezier_interpolate, w=cfg_w)

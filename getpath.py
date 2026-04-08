import matplotlib.pyplot as plt
import os
from os import path as osp
import numpy as np
import pickle

import sys
sys.modules['numpy._core'] = np
sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath
sys.modules['numpy._core.multiarray'] = np.core.multiarray

import torch
import json

from dit.Models import PathDiffusionTransformer

dataset_path = 'data/sim_dataset/val'
diffusion_step = None

# Define the network
device='cuda' if torch.cuda.is_available() else 'cpu'


def generate_paths(model, map_input, start_point, goal_point, num_paths=1,
                   reconstruct_trajectory=True, num_traj_points=100, solver='heun'):
    """
    使用 PathDiffusionTransformer 生成轨迹
    Returns:
        trajectories: (num_paths, N, 3)，每个点为 [x, y, theta]
    """
    model.eval()

    # 归一化起终点为 4 维：(x, y, cos(theta), sin(theta))
    start_normalized = torch.zeros(4, device=start_point.device)
    start_normalized[:2] = start_point[:2] / 20.0
    start_normalized[2] = torch.cos(start_point[2])
    start_normalized[3] = torch.sin(start_point[2])
    start_normalized[:2] = torch.clamp(start_normalized[:2], -1.0, 1.0)
    start_normalized = start_normalized.unsqueeze(0)

    goal_normalized = torch.zeros(4, device=goal_point.device)
    goal_normalized[:2] = goal_point[:2] / 20.0
    goal_normalized[2] = torch.cos(goal_point[2])
    goal_normalized[3] = torch.sin(goal_point[2])
    goal_normalized[:2] = torch.clamp(goal_normalized[:2], -1.0, 1.0)
    goal_normalized = goal_normalized.unsqueeze(0)

    with torch.no_grad():
        sampled_traj_xy = model.sample(
            map_input,
            start_normalized,
            goal_normalized,
            num_samples=num_paths,
            num_steps=diffusion_step,
            solver=solver,
            reconstruct_trajectory=reconstruct_trajectory,
            num_traj_points=num_traj_points,
        )  # (num_paths, N, 2)

    def compute_theta_from_xy(traj_xy):
        n = traj_xy.shape[0]
        theta = np.zeros(n)
        for i in range(1, n - 1):
            dx = traj_xy[i + 1, 0] - traj_xy[i - 1, 0]
            dy = traj_xy[i + 1, 1] - traj_xy[i - 1, 1]
            theta[i] = np.arctan2(dy, dx)

        dx = traj_xy[1, 0] - traj_xy[0, 0]
        dy = traj_xy[1, 1] - traj_xy[0, 1]
        theta[0] = np.arctan2(dy, dx)

        dx = traj_xy[-1, 0] - traj_xy[-2, 0]
        dy = traj_xy[-1, 1] - traj_xy[-2, 1]
        theta[-1] = np.arctan2(dy, dx)
        return theta

    sampled_traj_xy_np = sampled_traj_xy.cpu().numpy()
    trajectories = []
    for i in range(num_paths):
        traj_xy = sampled_traj_xy_np[i]
        theta = compute_theta_from_xy(traj_xy)
        traj_full = np.column_stack([traj_xy, theta])
        trajectories.append(traj_full)

    return np.stack(trajectories, axis=0)

def plot_single_trajectory(ax, elevation_masked, start_pos, goal_pos, predTraj):
    """绘制单个轨迹子图的辅助函数"""
    # 显示地形图
    ax.imshow(elevation_masked, extent=[-20, 20, -20, 20],
              origin='lower', cmap='terrain', aspect='equal')
    ax.grid(True, alpha=0.3)
    
    # 绘制轨迹线段
    # 将predTraj从(x,y,theta)格式转换为只包含(x,y)的格式用于绘制路径
    predTraj_xy = np.array([[point[0], point[1]] for point in predTraj])
    # 先为预测轨迹补足起点和终点，只使用x,y坐标
    predTraj_path = np.vstack((start_pos[:2], predTraj_xy, goal_pos[:2]))
    output_dim = predTraj_path.shape[0]  # 已经包括起点和终点
    
    # 绘制轨迹线段
    for i in range(output_dim-1):
        color = plt.cm.rainbow(i / (output_dim - 2)) if output_dim > 2 else 'blue'
        ax.plot([predTraj_path[i][0], predTraj_path[i+1][0]], 
                [predTraj_path[i][1], predTraj_path[i+1][1]], 
                color=color, zorder=3, linewidth=2, marker='o', markersize=3)
    
    # 绘制预测轨迹的角度箭头（使用predTraj中的原始角度信息）
    arrow_scale = 0.2
    for i, (x, y, theta) in enumerate(predTraj):
        color = plt.cm.rainbow(i / (len(predTraj) - 1)) if len(predTraj) > 1 else 'blue'
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
    title = 'Predicted Trajectory'
    ax.set_title(title, fontsize=12, pad=8)
    ax.axis('off')

def plot_elevation_map(start_pos, goal_pos, envType, save_path='predictions'):
    """
    绘制一张预测轨迹图
    Args:
        start_pos: numpy array [x, y, theta] 起点位置和朝向
        goal_pos: numpy array [x, y, theta] 终点位置和朝向
        envType: str 环境类型/名称
        save_path: str 保存路径
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

    encoder_input = torch.tensor(np.concatenate((
        normal_x[:, :, None],
        normal_y[:, :, None],
        normal_z[:, :, None]
    ), axis=2), dtype=torch.float32)

    predTrajs = generate_paths(
        model,
        map_input=encoder_input.permute(2, 0, 1)[None, :].to(device),
        start_point=torch.tensor(start_pos).float().to(device),
        goal_point=torch.tensor(goal_pos).float().to(device),
        num_paths=1,
        reconstruct_trajectory=True,
        num_traj_points=100,
        solver=solver,
    )
    predTraj = predTrajs.squeeze(0)
    
    print(f"Start: ({start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f})")
    print(f"Goal: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f}, {goal_pos[2]:.2f})")
    print(f"Predicted trajectory has {len(predTraj)} waypoints")
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # 绘制轨迹
    plot_single_trajectory(ax, elevation_masked, start_pos, goal_pos, predTraj)
    
    # 保存图像
    save_dir = osp.join(save_path, f'{envType}')
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成文件名
    filename = f'custom_start_{start_pos[0]:.1f}_{start_pos[1]:.1f}_goal_{goal_pos[0]:.1f}_{goal_pos[1]:.1f}.png'
    save_file = osp.join(save_dir, filename)
    
    plt.tight_layout()
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_file}")

if __name__ == "__main__":
    best = True
    stage = 2
    epoch = 4

    ema = False
    ema_decay = 0.999

    solver = 'pmf_onestep'
    diffusion_step = 3
    
    envType = 'env000010'  # 指定环境
    save_path = 'predictions'

    modelFolder = 'data/sim'
    modelFile = osp.join(modelFolder, f'model_params.json')
    model_param = json.load(open(modelFile))

    model = PathDiffusionTransformer(**model_param['model_args'])
    _ = model.to(device)

    if best:
        if ema:
            checkpoint = torch.load(osp.join(modelFolder, f'stage{stage}_best_ema_{ema_decay}.pth'))
            print(f"Loaded best EMA stage {stage} model.")
        else:
            checkpoint = torch.load(osp.join(modelFolder, f'stage{stage}_best_model.pth'))
            print(f"Loaded best stage {stage} model.")
    else:
        checkpoint = torch.load(osp.join(modelFolder, f'checkpoint_stage{stage}_epoch_{epoch}.pth'))
        print(f"Loaded stage {stage} model from epoch {epoch}.")
    model.load_state_dict(checkpoint['model_state_dict'])

    _ = model.eval()

    # 自定义起点和终点 (x, y, theta)
    # theta 是朝向角度，单位是弧度
    # 位姿数据： start_map=(-0.034, 0.081, 173.520°), goal_map=(0.045, 10.083, 89.578°)

    start_pos = np.array([-0.034, 0.081, 173.520/180*np.pi])  # 起点在左下角，朝向右
    goal_pos = np.array([0.045, 10.083, 89.578/180*np.pi])  # 终点在右上角，朝向东北

    # # 或者定义多组起点终点进行测试
    # test_cases = [
    #     (np.array([-15.0, -15.0, 0.0]), np.array([15.0, 15.0, np.pi/2])),
    #     (np.array([-10.0, 10.0, -np.pi/4]), np.array([10.0, -10.0, np.pi])),
    #     (np.array([0.0, -15.0, np.pi/2]), np.array([0.0, 15.0, np.pi/2])),
    # ]
    
    print(f"Evaluating environment: {envType}")
    
    # 绘制单个轨迹
    plot_elevation_map(start_pos, goal_pos, envType, save_path)
    
    # # 或者绘制多组测试用例
    # for i, (start, goal) in enumerate(test_cases):
    #     print(f"\n--- Test case {i+1} ---")
    #     plot_elevation_map(start, goal, envType, save_path)
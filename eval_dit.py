import os
from os import path as osp
import numpy as np
import pickle
import json
import sys
sys.modules['numpy._core'] = np
sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath
sys.modules['numpy._core.multiarray'] = np.core.multiarray

import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Tuple
import time

from dit.Models import PathDiffusionTransformer
from evaluator import TrajectoryEvaluator
from dataLoader_dit import compute_map_yaw_bins, generate_sdf_from_yaw_stability


def generate_paths(model, map_input, start_point, goal_point, num_paths=1, device='cuda',
                   solver='pmf_refined', num_steps=3,
                   reconstruct_trajectory=True, num_traj_points=100,
                   use_guided_sampling=False, cost_map=None, map_info=None,
                   guidance_scale=0.15, guidance_start_step=0.7):
    """
    生成完整路径（绝对坐标版本，使用新 Rectified Flow 采样接口）
    
    Args:
        model: 训练好的PathDiffusionTransformer
        map_input: (1, 3, H, W) 已包含地形信息的地图
        start_point: (3,) [x, y, yaw] 起点坐标（真实坐标）
        goal_point: (3,) [x, y, yaw] 终点坐标（真实坐标）
        num_paths: 生成路径数量
        device: 计算设备
        solver: ODE 求解器类型 ('pmf_onestep' | 'pmf_refined' | 'euler' | 'heun')
        num_steps: ODE 求解步数
        reconstruct_trajectory: 是否从控制点重建轨迹
        num_traj_points: 重建后的轨迹点数
        use_guided_sampling: 是否使用guided sampling（若模型不支持将直接报错）
        cost_map: 稳定性代价地图 (用于guided sampling)
        map_info: 地图配置信息 (用于guided sampling)
        guidance_scale: 引导强度
        guidance_start_step: 开始引导的时间点比例
    Returns:
        trajectories: (num_paths, N, 3) 绝对坐标 [x, y, theta]
        inference_time: 单次推理耗时（秒）
    """
    model.eval()
    
    # 归一化起点终点并转换为4维 (x, y, cos(θ), sin(θ))
    start_normalized = torch.zeros(4, device=device)
    start_normalized[:2] = start_point[:2] / 20.0  # x,y归一化
    start_normalized[2] = torch.cos(start_point[2])  # cos(θ)
    start_normalized[3] = torch.sin(start_point[2])  # sin(θ)
    start_normalized[:2] = torch.clamp(start_normalized[:2], -1.0, 1.0)
    start_normalized = start_normalized.unsqueeze(0)  # (1, 4)
    
    goal_normalized = torch.zeros(4, device=device)
    goal_normalized[:2] = goal_point[:2] / 20.0
    goal_normalized[2] = torch.cos(goal_point[2])  # cos(θ)
    goal_normalized[3] = torch.sin(goal_point[2])  # sin(θ)
    goal_normalized[:2] = torch.clamp(goal_normalized[:2], -1.0, 1.0)
    goal_normalized = goal_normalized.unsqueeze(0)  # (1, 4)
    
    # 开始计时
    start_time = time.time()
    
    if use_guided_sampling:
        if not hasattr(model, 'guided_sample'):
            raise NotImplementedError("当前模型未实现 guided_sample，无法启用guided sampling")
        if cost_map is None or map_info is None:
            raise ValueError("cost_map and map_info are required for guided sampling")
        
        traj_xy = model.guided_sample(
            map_input,
            start_normalized,
            goal_normalized,
            cost_map,
            map_info,
            num_samples=num_paths,
            num_steps=num_steps,
            solver=solver,
            reconstruct_trajectory=reconstruct_trajectory,
            num_traj_points=num_traj_points,
            guidance_scale=guidance_scale,
            guidance_start_step=guidance_start_step
        )
    else:
        with torch.no_grad():
            traj_xy = model.sample(
                map_input,
                start_normalized,
                goal_normalized,
                num_samples=num_paths,
                num_steps=num_steps,
                solver=solver,
                reconstruct_trajectory=reconstruct_trajectory,
                num_traj_points=num_traj_points
            )  # (num_paths, N, 2)
    
    # 结束计时
    inference_time = time.time() - start_time

    # 计算 yaw（从轨迹切向量估计）
    if not isinstance(traj_xy, torch.Tensor):
        traj_xy = torch.tensor(traj_xy, dtype=torch.float32, device=device)
    else:
        traj_xy = traj_xy.to(device)

    B, N, _ = traj_xy.shape
    if N < 2:
        yaw = torch.zeros(B, N, device=traj_xy.device)
    else:
        dx = torch.zeros(B, N, device=traj_xy.device)
        dy = torch.zeros(B, N, device=traj_xy.device)
        dx[:, 1:-1] = traj_xy[:, 2:, 0] - traj_xy[:, :-2, 0]
        dy[:, 1:-1] = traj_xy[:, 2:, 1] - traj_xy[:, :-2, 1]
        dx[:, 0] = traj_xy[:, 1, 0] - traj_xy[:, 0, 0]
        dy[:, 0] = traj_xy[:, 1, 1] - traj_xy[:, 0, 1]
        dx[:, -1] = traj_xy[:, -1, 0] - traj_xy[:, -2, 0]
        dy[:, -1] = traj_xy[:, -1, 1] - traj_xy[:, -2, 1]
        yaw = torch.atan2(dy, dx)

    traj_full = torch.cat([traj_xy, yaw.unsqueeze(-1)], dim=-1)  # (B, N, 3)
    
    return traj_full.detach().cpu().numpy(), inference_time


def load_environment_data(env_folder: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载环境数据"""
    env_path = osp.join(env_folder, 'map.p')
    with open(env_path, 'rb') as f:
        env = pickle.load(f)
        tensor = env['tensor']
        elevation = tensor[:, :, 0]
        normal_x = tensor[:, :, 1]
        normal_y = tensor[:, :, 2]
        normal_z = tensor[:, :, 3]
    print(f"Loaded environment data from {env_folder}: elevation shape {elevation.shape}, normal shape {normal_x.shape}")
    return elevation, normal_x, normal_y, normal_z


def load_trajectory_data(env_folder: str, path_num: int) -> np.ndarray:
    """加载真实轨迹数据"""
    path_file = osp.join(env_folder, f'path_{path_num}.p')
    with open(path_file, 'rb') as f:
        path_data = pickle.load(f)
        trajectory = path_data['path']  # [N, 3]
    return trajectory


def prepare_map_input(normal_x: np.ndarray, normal_y: np.ndarray, normal_z: np.ndarray, device='cuda') -> torch.Tensor:
    """准备地图输入张量"""
    encoder_input = torch.tensor(np.concatenate((
        normal_x[:, :, None],  # [H, W, 1]
        normal_y[:, :, None],  # [H, W, 1]
        normal_z[:, :, None]   # [H, W, 1]
    ), axis=2), dtype=torch.float32)  # [H, W, 3]
    
    # 转换为 [1, 3, H, W] 格式
    map_input = encoder_input.permute(2, 0, 1).unsqueeze(0).to(device)
    return map_input


def evaluate_single_path(model_stage1: PathDiffusionTransformer,
                        model_stage2: PathDiffusionTransformer,
                        evaluator: TrajectoryEvaluator,
                        env_folder: str,
                        path_num: int,
                        normal_x: np.ndarray,
                        normal_y: np.ndarray,
                        normal_z: np.ndarray,
                        num_pred_paths: int = 1,
                        device: str = 'cuda',
                        use_map_evaluation: bool = True,
                        use_guided_sampling: bool = False,
                        cost_map: torch.Tensor = None,
                        map_info: dict = None,
                        guidance_scale: float = 0.15,
                        guidance_start_step: float = 0.7,
                        solver: str = 'pmf_refined',
                        num_steps: int = 3,
                        reconstruct_trajectory: bool = True,
                        num_traj_points: int = 100) -> Tuple[Dict[str, float], List[Dict[str, float]], List[Dict[str, float]], np.ndarray, np.ndarray, float, float]:
    """
    评估单条路径（同时评估两个阶段的模型）
    
    Args:
        model_stage1: 阶段一的DIT模型
        model_stage2: 阶段二的DIT模型
        evaluator: 轨迹评估器
        env_folder: 环境文件夹路径
        path_num: 路径编号
        normal_x, normal_y, normal_z: 法向量数据
        num_pred_paths: 生成的预测轨迹数量
        device: 计算设备
        use_map_evaluation: 是否使用地图进行碰撞检测评估
        
    Returns:
        gt_metrics: 真实轨迹的评估指标
        stage1_metrics_list: 阶段一预测轨迹的评估指标列表
        stage2_metrics_list: 阶段二预测轨迹的评估指标列表
        stage1_trajectories: 阶段一预测的轨迹数组
        stage2_trajectories: 阶段二预测的轨迹数组
    """
    # 加载真实轨迹
    trajectory = load_trajectory_data(env_folder, path_num)
    
    # 提取起点和终点
    start_pos = trajectory[0, :]
    goal_pos = trajectory[-1, :]
    
    # 准备地图输入
    map_input = prepare_map_input(normal_x, normal_y, normal_z, device)
    
    if use_guided_sampling:
        # 启用guided sampling时：对比stage1标准采样 vs stage1 guided采样
        # stage1: 标准采样
        stage1_trajs, stage1_time = generate_paths(
            model_stage1,
            map_input=map_input,
            start_point=torch.tensor(start_pos).float().to(device),
            goal_point=torch.tensor(goal_pos).float().to(device),
            num_paths=num_pred_paths,
            device=device,
            use_guided_sampling=False,  # 标准采样
            cost_map=None,
            map_info=None,
            guidance_scale=guidance_scale,
            guidance_start_step=guidance_start_step,
            solver=solver,
            num_steps=num_steps,
            reconstruct_trajectory=reconstruct_trajectory,
            num_traj_points=num_traj_points
        )  # (num_pred_paths, N, 3), float
        
        # stage2: stage1模型 + guided采样
        stage2_trajs, stage2_time = generate_paths(
            model_stage1,  # 注意：使用stage1模型
            map_input=map_input,
            start_point=torch.tensor(start_pos).float().to(device),
            goal_point=torch.tensor(goal_pos).float().to(device),
            num_paths=num_pred_paths,
            device=device,
            use_guided_sampling=True,  # guided采样
            cost_map=cost_map,
            map_info=map_info,
            guidance_scale=guidance_scale,
            guidance_start_step=guidance_start_step,
            solver=solver,
            num_steps=num_steps,
            reconstruct_trajectory=reconstruct_trajectory,
            num_traj_points=num_traj_points
        )  # (num_pred_paths, N, 3), float
    else:
        # 标准模式：对比stage1 vs stage2
        # stage1: 标准采样
        stage1_trajs, stage1_time = generate_paths(
            model_stage1,
            map_input=map_input,
            start_point=torch.tensor(start_pos).float().to(device),
            goal_point=torch.tensor(goal_pos).float().to(device),
            num_paths=num_pred_paths,
            device=device,
            use_guided_sampling=False,
            cost_map=None,
            map_info=None,
            guidance_scale=guidance_scale,
            guidance_start_step=guidance_start_step,
            solver=solver,
            num_steps=num_steps,
            reconstruct_trajectory=reconstruct_trajectory,
            num_traj_points=num_traj_points
        )  # (num_pred_paths, N, 3), float
        
        # stage2: stage2模型 + 标准采样
        stage2_trajs, stage2_time = generate_paths(
            model_stage2,  # 使用stage2模型
            map_input=map_input,
            start_point=torch.tensor(start_pos).float().to(device),
            goal_point=torch.tensor(goal_pos).float().to(device),
            num_paths=num_pred_paths,
            device=device,
            use_guided_sampling=False,
            cost_map=None,
            map_info=None,
            guidance_scale=guidance_scale,
            guidance_start_step=guidance_start_step,
            solver=solver,
            num_steps=num_steps,
            reconstruct_trajectory=reconstruct_trajectory,
            num_traj_points=num_traj_points
        )  # (num_pred_paths, N, 3), float
    
    # 评估真实轨迹
    gt_trajectory_tensor = torch.tensor(trajectory, dtype=torch.float32, device=device)
    gt_metrics = evaluator.evaluate_trajectory(gt_trajectory_tensor)
    
    # 评估阶段一预测轨迹
    stage1_metrics_list = []
    for i in range(num_pred_paths):
        pred_traj_single = stage1_trajs[i]  # (N, 3)
        pred_trajectory_tensor = torch.tensor(pred_traj_single, dtype=torch.float32, device=device)
        pred_metrics = evaluator.evaluate_trajectory(pred_trajectory_tensor)
        stage1_metrics_list.append(pred_metrics)
    
    # 评估阶段二预测轨迹
    stage2_metrics_list = []
    for i in range(num_pred_paths):
        pred_traj_single = stage2_trajs[i]  # (N, 3)
        pred_trajectory_tensor = torch.tensor(pred_traj_single, dtype=torch.float32, device=device)
        pred_metrics = evaluator.evaluate_trajectory(pred_trajectory_tensor)
        stage2_metrics_list.append(pred_metrics)
    
    return gt_metrics, stage1_metrics_list, stage2_metrics_list, stage1_trajs, stage2_trajs, stage1_time, stage2_time


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    聚合多条轨迹的评估指标
    
    Args:
        metrics_list: 评估指标列表
        
    Returns:
        聚合后的统计指标（均值、标准差、最小值、最大值）
    """
    if not metrics_list:
        return {}
    
    # 获取所有指标名称
    metric_keys = metrics_list[0].keys()
    
    aggregated = {}
    for key in metric_keys:
        values = [m[key] for m in metrics_list if key in m]
        if values:
            aggregated[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
    
    return aggregated


def save_results_to_csv(results: List[Dict], output_path: str):
    """保存结果到CSV文件"""
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def print_comparison_summary(gt_metrics_agg: Dict, stage1_metrics_agg: Dict, stage2_metrics_agg: Dict, use_guided_sampling: bool = False):
    """打印真实轨迹、阶段一和阶段二的三方对比摘要"""
    print("\n" + "="*100)
    if use_guided_sampling:
        print("Ground Truth vs Stage1(Standard) vs Stage1(Guided) Comparison".center(100))
    else:
        print("Ground Truth vs Stage1 vs Stage2 Trajectories Comparison".center(100))
    print("="*100 + "\n")
    
    # 选择关键指标进行对比
    key_metrics = [
        'collision_risk_mean',
        'unstable_point_ratio',
        'path_length',
        'estimated_time',
        'smoothness_total',
        'jerk_x',
        'jerk_y',
        'jerk_yaw',
        'curvature_mean',
        'speed_mean',
        'out_of_bounds_ratio',
        'heading_error_mean'
    ]
    
    for metric in key_metrics:
        if metric in gt_metrics_agg:
            print(f"\n{metric}:")
            print(f"  GT:      {gt_metrics_agg[metric]['mean']:12.6f} ± {gt_metrics_agg[metric]['std']:10.6f}")
            
            if metric in stage1_metrics_agg:
                label = "  S1(Std):" if use_guided_sampling else "  Stage1: "
                print(f"{label} {stage1_metrics_agg[metric]['mean']:12.6f} ± {stage1_metrics_agg[metric]['std']:10.6f}", end="")
                gt_val = gt_metrics_agg[metric]['mean']
                s1_val = stage1_metrics_agg[metric]['mean']
                if abs(gt_val) > 1e-9:
                    diff_pct = ((s1_val - gt_val) / abs(gt_val)) * 100
                    status = "↓" if diff_pct < 0 else "↑"
                    print(f"  [{status} {abs(diff_pct):6.2f}% vs GT]")
                else:
                    print()
            
            if metric in stage2_metrics_agg:
                label = "  S1(Gui):" if use_guided_sampling else "  Stage2: "
                print(f"{label} {stage2_metrics_agg[metric]['mean']:12.6f} ± {stage2_metrics_agg[metric]['std']:10.6f}", end="")
                gt_val = gt_metrics_agg[metric]['mean']
                s2_val = stage2_metrics_agg[metric]['mean']
                if abs(gt_val) > 1e-9:
                    diff_pct = ((s2_val - gt_val) / abs(gt_val)) * 100
                    status = "↓" if diff_pct < 0 else "↑"
                    print(f"  [{status} {abs(diff_pct):6.2f}% vs GT]")
                else:
                    print()
                
                # Stage2 vs Stage1 对比
                if metric in stage1_metrics_agg:
                    s1_val = stage1_metrics_agg[metric]['mean']
                    if abs(s1_val) > 1e-9:
                        diff_pct_s2_s1 = ((s2_val - s1_val) / abs(s1_val)) * 100
                        status_s2_s1 = "↓" if diff_pct_s2_s1 < 0 else "↑"
                        vs_label = "vs S1(Std)" if use_guided_sampling else "vs Stage1"
                        print(f"           {'':12}   {'':10}  [{status_s2_s1} {abs(diff_pct_s2_s1):6.2f}% {vs_label}]")
    
    print("\n" + "="*100 + "\n")


def main():
    # =================== 配置参数 ===================
    dataset_path = 'data/sim_dataset/val'
    # dataset_path = 'data/sim_dataset/train'
    save_path = 'evaluation_results'
    os.makedirs(save_path, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 模型配置
    best = True
    # best = False
    epoch = 4
    stage = 2
    
    # ema = True
    ema = False
    # ema_decay = 0.99
    ema_decay = 0.999
    
    # 评估配置
    num_pred_paths = 10  # 每个场景生成多少条预测轨迹
    use_map_evaluation = True  # 是否使用地图进行碰撞检测评估
    # use_map_evaluation = False  # 设置为 False 只评估几何特性
    
    # Guided Sampling 配置
    use_guided_sampling = False  # 是否使用guided sampling（推理时优化）
    # use_guided_sampling = True  # 启用guided sampling
    guidance_scale = 0.15  # 引导强度 (推荐: 0.08-0.2)
    guidance_start_step = 0.7  # 开始引导的时间点 (推荐: 0.5-0.7)

    # ODE 求解器配置（与新模型接口对齐）
    # solver = 'pmf_refined'  # 'pmf_onestep' | 'pmf_refined' | 'euler' | 'heun'
    solver = 'pmf_onestep'  # 'pmf_onestep' | 'pmf_refined' | 'euler' | 'heun'
    diffusion_step = 3
    reconstruct_trajectory = True
    num_traj_points = 100
    
    # 选择要评估的环境和路径
    # envNum = np.random.randint(0, 99)
    # env_list = [f'env{envNum:06d}']
    env_list = ['env000010']  # 可以添加多个环境
    
    # 选择要评估的路径编号
    # path_nums = list(range(50))  # 评估前50条路径
    # path_nums = list(range(40, 50))  # 评估指定范围的路径
    path_nums = list(range(0, 100))  # 评估指定范围的路径
    # path_nums = [40, 41, 42, 43, 44, 45]  # 或指定特定路径
    
    print(f"Device: {device}")
    print(f"Evaluating {len(env_list)} environments")
    print(f"Evaluating {len(path_nums)} paths per environment")
    print(f"Generating {num_pred_paths} predictions per path")
    print(f"Map-based collision detection: {'Enabled' if use_map_evaluation else 'Disabled'}")
    print(f"Guided sampling: {'Enabled' if use_guided_sampling else 'Disabled'}")
    print(f"Solver: {solver}, steps: {diffusion_step}, reconstruct: {reconstruct_trajectory}, points: {num_traj_points}")
    if use_guided_sampling:
        print(f"  → Guidance scale: {guidance_scale}")
        print(f"  → Guidance start step: {guidance_start_step}")
    # ================================================
    
    # 加载模型
    modelFolder = 'data/sim'
    # modelFolder = 'data'
    modelFile = osp.join(modelFolder, 'model_params.json')
    model_param = json.load(open(modelFile))
    
    # 加载阶段一模型
    model_stage1 = PathDiffusionTransformer(**model_param['model_args'])
    model_stage1 = model_stage1.to(device)
    
    # if best:
    #     checkpoint_stage1 = torch.load(osp.join(modelFolder, 'stage1_best_model.pth'))
    #     print(f"Loaded best stage 1 model.")
    # else:
    #     checkpoint_stage1 = torch.load(osp.join(modelFolder, f'checkpoint_stage1_epoch_{epoch}.pth'))
    #     print(f"Loaded stage 1 model from epoch {epoch}.")
    
    checkpoint_stage1 = torch.load(osp.join(modelFolder, 'stage1_best_model.pth'))
    print(f"Loaded best stage 1 model.")
    
    model_stage1.load_state_dict(checkpoint_stage1['model_state_dict'])
    model_stage1.eval()
    
    # 加载阶段二模型
    model_stage2 = PathDiffusionTransformer(**model_param['model_args'])
    model_stage2 = model_stage2.to(device)
    
    if best:
        if ema:
            checkpoint_stage2 = torch.load(osp.join(modelFolder, f'stage2_best_ema_{ema_decay}.pth'))
            print(f"Loaded best EMA stage {stage} model.")
        else:
            checkpoint_stage2 = torch.load(osp.join(modelFolder, 'stage2_best_model.pth'))
            print(f"Loaded best stage 2 model.")
    else:
        checkpoint_stage2 = torch.load(osp.join(modelFolder, f'checkpoint_stage2_epoch_{epoch}.pth'))
        print(f"Loaded stage 2 model from epoch {epoch}.")
    
    model_stage2.load_state_dict(checkpoint_stage2['model_state_dict'])
    model_stage2.eval()
    
    # 创建评估器（稍后会为每个环境创建带地图的评估器）
    # 这里先创建一个占位符，实际评估时会为每个环境单独创建
    evaluator = None
    
    # 存储所有结果
    all_gt_metrics = []
    all_stage1_metrics = []
    all_stage2_metrics = []
    detailed_results = []
    
    # 存储推理时间
    stage1_inference_times = []
    stage2_inference_times = []
    
    # 批量评估
    total_paths = len(env_list) * len(path_nums)
    pbar = tqdm(total=total_paths, desc="Evaluating trajectories")
    
    for env_name in env_list:
        env_folder = osp.join(dataset_path, env_name)
        
        # 加载环境数据（对同一环境只加载一次）
        elevation, normal_x, normal_y, normal_z = load_environment_data(env_folder)
        
        # 为当前环境创建评估器
        if use_map_evaluation:
            print(f"\nComputing stability map and ESDF for {env_name}...")
            
            # 转换为 torch 张量
            normal_x_torch = torch.tensor(normal_x, dtype=torch.float32, device=device)
            normal_y_torch = torch.tensor(normal_y, dtype=torch.float32, device=device)
            normal_z_torch = torch.tensor(normal_z, dtype=torch.float32, device=device)
            
            # 计算 yaw stability map
            yaw_stability = compute_map_yaw_bins(
                normal_x_torch, 
                normal_y_torch, 
                normal_z_torch, 
                yaw_bins=36
            )  # [H, W, 36]
            
            # 生成 ESDF cost map
            cost_map = generate_sdf_from_yaw_stability(
                yaw_stability, 
                voxel_size_xy=0.4,  # 匹配数据集的分辨率
                yaw_weight=1.4
            )  # [H, W, 36] - signed distance field
            
            # 创建地图信息
            H, W, D = cost_map.shape
            map_info = {
                'resolution': 0.4,  # 0.4米/像素
                'origin': (-20.0, -20.0, -np.pi),  # 地图原点 (x, y, yaw)
                'size': (W, H, D)  # (width, height, yaw_bins)
            }
            
            # 转换地图格式为 (D, H, W)
            cost_map_transposed = cost_map.permute(2, 0, 1)  # [36, H, W]
            yaw_stability_transposed = yaw_stability.permute(2, 0, 1)  # [36, H, W]
            
            # 创建带地图的评估器
            # occupancy_map 使用 ESDF；yaw_stability_map 使用二值稳定性图
            evaluator = TrajectoryEvaluator(
                occupancy_map=cost_map_transposed,
                # yaw_stability_map=yaw_stability_transposed,
                yaw_stability_map=cost_map_transposed,
                map_info=map_info,
                device=device
            )
            print(f"Created evaluator with ESDF occupancy map and binary yaw stability map for {env_name}")
        else:
            # 创建不带地图的评估器（只评估几何特性）
            evaluator = TrajectoryEvaluator(device=device)
            print(f"Created evaluator without map for {env_name}")
        
        for path_num in path_nums:
            try:
                # 准备guided sampling所需的参数
                if use_guided_sampling:
                    cost_map_for_guided = cost_map_transposed.unsqueeze(0) if use_map_evaluation else None
                    map_info_for_guided = map_info if use_map_evaluation else None
                else:
                    cost_map_for_guided = None
                    map_info_for_guided = None
                
                # 评估单条路径（同时评估两个阶段）
                gt_metrics, stage1_metrics_list, stage2_metrics_list, stage1_trajs, stage2_trajs, stage1_time, stage2_time = evaluate_single_path(
                    model_stage1=model_stage1,
                    model_stage2=model_stage2,
                    evaluator=evaluator,
                    env_folder=env_folder,
                    path_num=path_num,
                    normal_x=normal_x,
                    normal_y=normal_y,
                    normal_z=normal_z,
                    num_pred_paths=num_pred_paths,
                    device=device,
                    use_map_evaluation=use_map_evaluation,
                    use_guided_sampling=use_guided_sampling,
                    cost_map=cost_map_for_guided,
                    map_info=map_info_for_guided,
                    guidance_scale=guidance_scale,
                    guidance_start_step=guidance_start_step,
                    solver=solver,
                    num_steps=diffusion_step,
                    reconstruct_trajectory=reconstruct_trajectory,
                    num_traj_points=num_traj_points
                )
                
                # 收集结果
                all_gt_metrics.append(gt_metrics)
                all_stage1_metrics.extend(stage1_metrics_list)
                all_stage2_metrics.extend(stage2_metrics_list)
                
                # 收集推理时间
                stage1_inference_times.append(stage1_time)
                stage2_inference_times.append(stage2_time)
                
                # 保存阶段一详细结果
                for i, pred_metrics in enumerate(stage1_metrics_list):
                    result_dict = {
                        'environment': env_name,
                        'path_num': path_num,
                        'pred_sample': i,
                        'type': 'stage1_standard' if use_guided_sampling else 'stage1_predicted'
                    }
                    result_dict.update(pred_metrics)
                    detailed_results.append(result_dict)
                
                # 保存阶段二详细结果
                for i, pred_metrics in enumerate(stage2_metrics_list):
                    result_dict = {
                        'environment': env_name,
                        'path_num': path_num,
                        'pred_sample': i,
                        'type': 'stage1_guided' if use_guided_sampling else 'stage2_predicted'
                    }
                    result_dict.update(pred_metrics)
                    detailed_results.append(result_dict)
                
                # 添加真实轨迹结果
                gt_result_dict = {
                    'environment': env_name,
                    'path_num': path_num,
                    'pred_sample': -1,
                    'type': 'ground_truth'
                }
                gt_result_dict.update(gt_metrics)
                detailed_results.append(gt_result_dict)
                
                pbar.update(1)
                
            except Exception as e:
                print(f"\nError evaluating {env_name}/path_{path_num}: {e}")
                pbar.update(1)
                continue
    
    pbar.close()
    
    # 聚合统计结果
    print("\n" + "="*100)
    print("Computing aggregate statistics...")
    
    gt_metrics_agg = aggregate_metrics(all_gt_metrics)
    stage1_metrics_agg = aggregate_metrics(all_stage1_metrics)
    stage2_metrics_agg = aggregate_metrics(all_stage2_metrics)
    
    # 打印三方对比摘要
    print_comparison_summary(gt_metrics_agg, stage1_metrics_agg, stage2_metrics_agg, use_guided_sampling)
    
    # 打印推理时间统计
    print("\n" + "="*100)
    print("Inference Time Statistics".center(100))
    print("="*100 + "\n")
    
    if stage1_inference_times:
        stage1_avg_time = np.mean(stage1_inference_times)
        stage1_std_time = np.std(stage1_inference_times)
        model_label = "Stage1 (Standard Sampling):" if use_guided_sampling else "Stage1 Model:"
        print(f"{model_label}")
        print(f"  Average inference time: {stage1_avg_time:.4f}s ± {stage1_std_time:.4f}s")
        print(f"  Min: {np.min(stage1_inference_times):.4f}s, Max: {np.max(stage1_inference_times):.4f}s")
        print(f"  Total samples: {len(stage1_inference_times)}")
    
    print()
    
    if stage2_inference_times:
        stage2_avg_time = np.mean(stage2_inference_times)
        stage2_std_time = np.std(stage2_inference_times)
        if use_guided_sampling:
            model_label = f"Stage1 (Guided Sampling):"
            print(f"{model_label}")
            print(f"  Average inference time: {stage2_avg_time:.4f}s ± {stage2_std_time:.4f}s")
            print(f"  Min: {np.min(stage2_inference_times):.4f}s, Max: {np.max(stage2_inference_times):.4f}s")
            print(f"  Total samples: {len(stage2_inference_times)}")
            print(f"  Guidance scale: {guidance_scale}, start step: {guidance_start_step}")
        else:
            model_label = "Stage2 Model:"
            print(f"{model_label}")
            print(f"  Average inference time: {stage2_avg_time:.4f}s ± {stage2_std_time:.4f}s")
            print(f"  Min: {np.min(stage2_inference_times):.4f}s, Max: {np.max(stage2_inference_times):.4f}s")
            print(f"  Total samples: {len(stage2_inference_times)}")
    
    print("\n" + "="*100 + "\n")
    
    # 保存详细结果到CSV
    csv_path = osp.join(save_path, 'detailed_evaluation_results.csv')
    save_results_to_csv(detailed_results, csv_path)
    
    # 保存聚合统计到JSON
    summary_results = {
        'ground_truth': gt_metrics_agg,
        'stage1_predicted': stage1_metrics_agg,
        'stage2_predicted': stage2_metrics_agg,
        'inference_time': {
            'stage1': {
                'mean': float(np.mean(stage1_inference_times)) if stage1_inference_times else 0.0,
                'std': float(np.std(stage1_inference_times)) if stage1_inference_times else 0.0,
                'min': float(np.min(stage1_inference_times)) if stage1_inference_times else 0.0,
                'max': float(np.max(stage1_inference_times)) if stage1_inference_times else 0.0
            },
            'stage2': {
                'mean': float(np.mean(stage2_inference_times)) if stage2_inference_times else 0.0,
                'std': float(np.std(stage2_inference_times)) if stage2_inference_times else 0.0,
                'min': float(np.min(stage2_inference_times)) if stage2_inference_times else 0.0,
                'max': float(np.max(stage2_inference_times)) if stage2_inference_times else 0.0
            }
        },
        'evaluation_config': {
            'dataset_path': dataset_path,
            'environments': env_list,
            'num_paths': len(path_nums),
            'num_pred_samples': num_pred_paths,
            'total_evaluated': len(all_gt_metrics),
            'use_map_evaluation': use_map_evaluation,
            'use_guided_sampling': use_guided_sampling,
            'guidance_scale': guidance_scale if use_guided_sampling else None,
            'guidance_start_step': guidance_start_step if use_guided_sampling else None
        }
    }
    
    json_path = osp.join(save_path, 'evaluation_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary_results, f, indent=2)
    print(f"Summary saved to {json_path}")
    
    # 生成简化的对比表格
    comparison_data = []
    key_metrics = [
        'collision_risk_mean', 'unstable_point_ratio', 'path_length', 'estimated_time',
        'smoothness_total', 'jerk_x', 'jerk_y', 'jerk_yaw', 'curvature_mean', 'speed_mean',
        'out_of_bounds_ratio', 'heading_error_mean'
    ]
    
    for metric in key_metrics:
        if metric in gt_metrics_agg:
            row = {'metric': metric}
            
            # GT 数据
            row['gt_mean'] = gt_metrics_agg[metric]['mean']
            row['gt_std'] = gt_metrics_agg[metric]['std']
            
            # Stage1 数据
            if metric in stage1_metrics_agg:
                row['stage1_mean'] = stage1_metrics_agg[metric]['mean']
                row['stage1_std'] = stage1_metrics_agg[metric]['std']
                row['stage1_vs_gt_pct'] = ((stage1_metrics_agg[metric]['mean'] - gt_metrics_agg[metric]['mean']) / 
                                          (abs(gt_metrics_agg[metric]['mean']) + 1e-9)) * 100
            
            # Stage2 数据
            if metric in stage2_metrics_agg:
                row['stage2_mean'] = stage2_metrics_agg[metric]['mean']
                row['stage2_std'] = stage2_metrics_agg[metric]['std']
                row['stage2_vs_gt_pct'] = ((stage2_metrics_agg[metric]['mean'] - gt_metrics_agg[metric]['mean']) / 
                                          (abs(gt_metrics_agg[metric]['mean']) + 1e-9)) * 100
                
                # Stage2 vs Stage1
                if metric in stage1_metrics_agg:
                    row['stage2_vs_stage1_pct'] = ((stage2_metrics_agg[metric]['mean'] - stage1_metrics_agg[metric]['mean']) / 
                                                   (abs(stage1_metrics_agg[metric]['mean']) + 1e-9)) * 100
            
            comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_csv = osp.join(save_path, 'metrics_comparison.csv')
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"Comparison table saved to {comparison_csv}")
    
    print("\n" + "="*100)
    print(f"Evaluation complete! Results saved to {save_path}/")
    print("="*100 + "\n")
    
    # 打印一些有趣的统计
    print("\nKey Findings:")
    print(f"  • Evaluated {len(all_gt_metrics)} ground truth trajectories")
    print(f"  • Evaluated {len(all_stage1_metrics)} stage1 predicted trajectories")
    print(f"  • Evaluated {len(all_stage2_metrics)} stage2 predicted trajectories")
    
    # 检查哪些指标改善了，哪些恶化了
    print("\n" + "-"*100)
    if use_guided_sampling:
        print("Stage1 (Standard) vs GT Performance:")
    else:
        print("Stage1 vs GT Performance:")
    print("-"*100)
    stage1_improved = []
    stage1_worsened = []
    for metric in key_metrics:
        if metric in gt_metrics_agg and metric in stage1_metrics_agg:
            gt_val = gt_metrics_agg[metric]['mean']
            s1_val = stage1_metrics_agg[metric]['mean']
            diff_pct = ((s1_val - gt_val) / (abs(gt_val) + 1e-9)) * 100
            
            # 对于某些指标，低更好
            if metric in ['collision_risk_mean', 'unstable_point_ratio', 'out_of_bounds_ratio', 
                         'heading_error_mean', 'curvature_violation_ratio', 'dangerous_point_ratio',
                         'jerk_x', 'jerk_y', 'jerk_yaw', 'smoothness_total']:
                if diff_pct < 0:
                    stage1_improved.append((metric, abs(diff_pct)))
                elif diff_pct > 0:
                    stage1_worsened.append((metric, abs(diff_pct)))
    
    if stage1_improved:
        print("\nImproved metrics (lower is better):")
        for metric, pct in stage1_improved[:5]:
            print(f"  ✓ {metric}: {pct:.2f}% better than GT")
    
    if stage1_worsened:
        print("\nWorsened metrics:")
        for metric, pct in stage1_worsened[:5]:
            print(f"  ✗ {metric}: {pct:.2f}% worse than GT")
    
    print("\n" + "-"*100)
    if use_guided_sampling:
        print("Stage1 (Guided) vs GT Performance:")
    else:
        print("Stage2 vs GT Performance:")
    print("-"*100)
    stage2_improved = []
    stage2_worsened = []
    for metric in key_metrics:
        if metric in gt_metrics_agg and metric in stage2_metrics_agg:
            gt_val = gt_metrics_agg[metric]['mean']
            s2_val = stage2_metrics_agg[metric]['mean']
            diff_pct = ((s2_val - gt_val) / (abs(gt_val) + 1e-9)) * 100
            
            if metric in ['collision_risk_mean', 'unstable_point_ratio', 'out_of_bounds_ratio', 
                         'heading_error_mean', 'curvature_violation_ratio', 'dangerous_point_ratio',
                         'jerk_x', 'jerk_y', 'jerk_yaw', 'smoothness_total']:
                if diff_pct < 0:
                    stage2_improved.append((metric, abs(diff_pct)))
                elif diff_pct > 0:
                    stage2_worsened.append((metric, abs(diff_pct)))
    
    if stage2_improved:
        print("\nImproved metrics (lower is better):")
        for metric, pct in stage2_improved[:5]:
            print(f"  ✓ {metric}: {pct:.2f}% better than GT")
    
    if stage2_worsened:
        print("\nWorsened metrics:")
        for metric, pct in stage2_worsened[:5]:
            print(f"  ✗ {metric}: {pct:.2f}% worse than GT")
    
    print("\n" + "-"*100)
    if use_guided_sampling:
        print("Stage1 (Guided) vs Stage1 (Standard) Performance:")
    else:
        print("Stage2 vs Stage1 Performance:")
    print("-"*100)
    stage2_vs_stage1_improved = []
    stage2_vs_stage1_worsened = []
    for metric in key_metrics:
        if metric in stage1_metrics_agg and metric in stage2_metrics_agg:
            s1_val = stage1_metrics_agg[metric]['mean']
            s2_val = stage2_metrics_agg[metric]['mean']
            diff_pct = ((s2_val - s1_val) / (abs(s1_val) + 1e-9)) * 100
            
            if metric in ['collision_risk_mean', 'unstable_point_ratio', 'out_of_bounds_ratio', 
                         'heading_error_mean', 'curvature_violation_ratio', 'dangerous_point_ratio',
                         'jerk_x', 'jerk_y', 'jerk_yaw', 'smoothness_total']:
                if diff_pct < 0:
                    stage2_vs_stage1_improved.append((metric, abs(diff_pct)))
                elif diff_pct > 0:
                    stage2_vs_stage1_worsened.append((metric, abs(diff_pct)))
    
    if stage2_vs_stage1_improved:
        print("\nImproved metrics (lower is better):")
        compare_label = "Stage1 (Standard)" if use_guided_sampling else "Stage1"
        for metric, pct in stage2_vs_stage1_improved[:5]:
            print(f"  ✓ {metric}: {pct:.2f}% better than {compare_label}")
    
    if stage2_vs_stage1_worsened:
        print("\nWorsened metrics:")
        compare_label = "Stage1 (Standard)" if use_guided_sampling else "Stage1"
        for metric, pct in stage2_vs_stage1_worsened[:5]:
            print(f"  ✗ {metric}: {pct:.2f}% worse than {compare_label}")


if __name__ == '__main__':
    main()

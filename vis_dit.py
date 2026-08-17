import matplotlib.pyplot as plt
import os
from os import path as osp
import json
import math
import numpy as np
import pickle

import sys
sys.modules['numpy._core'] = np
sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath
sys.modules['numpy._core.multiarray'] = np.core.multiarray

import torch
import argparse

from dit.Models import PathMeanFlowTransformer
from dataLoader_dit import (
    MASK_INPUT_SEMANTICS,
    MASK_NOISE_SEED_OFFSET,
    build_masked_normal_input,
    erode_mask_for_vehicle,
    generate_random_mask,
    normalize_mask,
)
from map_config import (
    DENSE_TRAJECTORY_POINTS,
    MAP_BOUNDS,
    MAP_CONFIG,
    SAFETY_COST_CONFIG,
)
from grad_optimizer import trajectory_validity_metrics
from boundary_constrained_path import (
    PATH_REPRESENTATION_SEMANTIC_VERSION,
)
from tools._paths import PREDICTIONS_ROOT
from stage2_critic import PathAlignedCandidateCritic


DEFAULT_STAGE1_CHECKPOINT = "data/path_meanflow/stage1_best.pth"


def _first_existing_path(*candidates):
    return next((value for value in candidates if osp.isfile(value)), candidates[0])


DEFAULT_STAGE2_CHECKPOINT = _first_existing_path(
    "data/path_meanflow_stage2/stage2_best.pth",
    "tests/audits/privileged_source_transport_t2_20260804/"
    "stage2_best_calibration.pth",
)
DEFAULT_STAGE2_CRITIC_CHECKPOINT = _first_existing_path(
    "data/path_meanflow_stage2/critic/critic_dual_head_6400.pth",
    "tests/audits/stage2_stability_critic_dual_head_v1_20260804/"
    "critic_dual_head_6400.pth",
)
DEFAULT_STAGE1_VISUALIZATION_DIR = str(PREDICTIONS_ROOT / "path_meanflow_stage1")
DEFAULT_STAGE2_VISUALIZATION_DIR = str(PREDICTIONS_ROOT / "path_meanflow_stage2")
DEVELOPMENT_DIRECT_COST_STAGE = "development_direct_cost_sgd_pair"


def generate_paths(model, map_input, start_point, goal_point, num_paths=5, 
                   reconstruct_trajectory=True,
                   num_traj_points=DENSE_TRAJECTORY_POINTS,
                   solver='pmf_onestep', num_steps=1, source_noise=None,
                   return_details=False):
    """
    生成完整路径（使用B样条控制点重建）- Rectified Flow版本
    
    Args:
        model: 训练好的 Path MeanFlow 生成器
        map_input: (1, 3, H, W) 地图输入
        start_point: (3,) [x, y, yaw] 起点坐标（真实坐标）
        goal_point: (3,) [x, y, yaw] 终点坐标（真实坐标）
        num_paths: 生成路径数量
        reconstruct_trajectory: 是否从控制点重建轨迹（默认True）
        num_traj_points: 重建后的轨迹点数（默认使用统一稠密采样配置）
        solver: 采样器类型
        num_steps: refined/euler/heun 的积分步数；pmf_onestep 时忽略
    Returns:
        trajectories: (num_paths, N, 3) 轨迹 [x, y, theta]
            - 如果reconstruct_trajectory=True: N=num_traj_points
            - 如果reconstruct_trajectory=False: N=26（完整控制点数，首尾已替换）
    """
    model.eval()
    base_model = model.module if hasattr(model, "module") else model
    coordinate_scale = base_model.coordinate_scale
    
    # 归一化起点终点并转换为4维 (x, y, cos(θ), sin(θ))
    start_normalized = torch.zeros(4, device=start_point.device)
    start_normalized[:2] = start_point[:2] / coordinate_scale
    start_normalized[2] = torch.cos(start_point[2])  # cos(θ)
    start_normalized[3] = torch.sin(start_point[2])  # sin(θ)
    start_normalized[:2] = torch.clamp(start_normalized[:2], -1.0, 1.0)
    start_normalized = start_normalized.unsqueeze(0)  # (1, 4)
    
    goal_normalized = torch.zeros(4, device=goal_point.device)
    goal_normalized[:2] = goal_point[:2] / coordinate_scale
    goal_normalized[2] = torch.cos(goal_point[2])  # cos(θ)
    goal_normalized[3] = torch.sin(goal_point[2])  # sin(θ)
    goal_normalized[:2] = torch.clamp(goal_normalized[:2], -1.0, 1.0)
    goal_normalized = goal_normalized.unsqueeze(0)  # (1, 4)
    
    # 从Rectified Flow采样轨迹（只有x,y）
    with torch.no_grad():
        state = base_model.sample(
            map_input,
            start_normalized,
            goal_normalized,
            num_samples=num_paths,
            num_steps=num_steps,
            solver=solver,
            reconstruct_trajectory=False,
            num_traj_points=num_traj_points,
            source_noise=source_noise,
            return_residual=True,
        )
        start_batch = start_normalized.repeat_interleave(num_paths, dim=0)
        goal_batch = goal_normalized.repeat_interleave(num_paths, dim=0)
        geometry = base_model.evaluate_trajectory_state(
            state, start_batch, goal_batch
        )
    packed = torch.cat(
        [
            geometry["position"],
            geometry["yaw"].unsqueeze(-1),
            geometry["first_derivative"],
            geometry["second_derivative"],
            geometry["curvature"].unsqueeze(-1),
        ],
        dim=-1,
    )
    trajectories = packed.detach().cpu().numpy()
    if not return_details:
        return trajectories
    with torch.no_grad():
        curvature_audit = base_model.audit_trajectory_state_curvature(
            state,
            start_batch,
            goal_batch,
        )
    return {
        "trajectories": trajectories,
        "state": state.detach(),
        "geometry": {key: value.detach() for key, value in geometry.items()},
        "curvature_audit": curvature_audit.detach(),
        "start_condition": start_normalized.detach(),
        "goal_condition": goal_normalized.detach(),
    }


@torch.no_grad()
def select_stage2_candidate(
    critic,
    observed_map,
    generated,
    validity,
):
    """Apply the frozen deployable Stage-2 selector to one K-candidate pool."""
    geometry = generated["geometry"]
    predicted_risk, safe_logit = critic(
        observed_map,
        generated["start_condition"],
        generated["goal_condition"],
        geometry["position"].unsqueeze(0),
        geometry["curvature"].unsqueeze(0),
    )
    predicted_risk = predicted_risk[0]
    safe_probability = torch.sigmoid(safe_logit[0])
    maximum_curvature = validity["max_curvature"].to(
        device=safe_probability.device,
        dtype=safe_probability.dtype,
    )
    curvature_preference = torch.minimum(
        torch.ones_like(maximum_curvature),
        float(SAFETY_COST_CONFIG.curvature_limit)
        / maximum_curvature.clamp_min(1e-12),
    )
    forbidden_ok = validity["forbidden_region_ok"].to(
        device=safe_probability.device
    ).bool()
    score = safe_probability * curvature_preference
    filtered_score = torch.where(
        forbidden_ok,
        score,
        torch.full_like(score, -math.inf),
    )
    selected_index = int(filtered_score.argmax()) if bool(forbidden_ok.any()) else None

    stability_ok = validity["stability_ok"].bool()
    curvature_ok = validity["curvature_ok"].bool()
    candidates = []
    for index in range(len(score)):
        candidates.append(
            {
                "index": int(index),
                "forbidden_ok": bool(forbidden_ok[index]),
                "stability_ok_privileged_diagnostic": bool(stability_ok[index]),
                "capsizing_safe_privileged_diagnostic": bool(
                    forbidden_ok[index] and stability_ok[index]
                ),
                "curvature_ok_diagnostic": bool(curvature_ok[index]),
                "critic_safe_probability": float(safe_probability[index]),
                "critic_predicted_risk": float(predicted_risk[index]),
                "max_curvature_inv_m": float(maximum_curvature[index]),
                "curvature_preference": float(curvature_preference[index]),
                "selection_score": float(score[index]),
            }
        )
    ranking = sorted(
        (entry for entry in candidates if entry["forbidden_ok"]),
        key=lambda entry: entry["selection_score"],
        reverse=True,
    )
    return {
        "rule": (
            "forbidden hard filter; argmax(sigmoid(safe_logit) * "
            "min(1, curvature_limit/max_curvature))"
        ),
        "selected_index": selected_index,
        "rejected_all_forbidden": selected_index is None,
        "candidates": candidates,
        "ranking": [entry["index"] for entry in ranking],
    }


def build_partial_map_input(
    normal_x,
    normal_y,
    normal_z,
    mask,
    *,
    noise_seed,
):
    """与训练数据完全一致地构造“高斯噪声法向量 + 可通行 mask”输入。"""
    normals = np.stack([normal_x, normal_y, normal_z], axis=-1).astype(np.float32)
    return torch.from_numpy(
        build_masked_normal_input(normals, mask, noise_seed)
    ).permute(2, 0, 1).float()


def raw_box_metrics(trajectories):
    """Report unmodified box violations, as required for the no-radial model."""
    xy = np.asarray(trajectories)[..., :2]
    low = np.array([MAP_BOUNDS[0], MAP_BOUNDS[2]])
    high = np.array([MAP_BOUNDS[1], MAP_BOUNDS[3]])
    violation = np.maximum(low - xy, 0.0) + np.maximum(xy - high, 0.0)
    point_oob = np.any(violation > 0.0, axis=-1)
    return {
        "trajectory_oob_rate": float(np.mean(np.any(point_oob, axis=-1))),
        "point_oob_rate": float(np.mean(point_oob)),
        "max_oob_m": float(np.max(violation)),
    }

def plot_single_trajectory(
    ax,
    elevation_masked,
    trajectory,
    predTrajs=None,
    output_dim=None,
    is_pred=False,
    use_bezier_interpolate=False,
    selected_index=None,
    forbidden_ok=None,
):
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
    ax.imshow(
        elevation_masked,
        extent=MAP_BOUNDS,
        origin='lower',
        cmap='terrain',
        aspect='equal',
        # The masked cells are the hard visual contract for forbidden_ok.
        interpolation='nearest',
    )
    ax.grid(True, alpha=0.3)
    
    start_pos = trajectory[0, :]
    goal_pos = trajectory[-1, :]
    
    # 绘制轨迹线段
    if is_pred and predTrajs is not None:
        # 检查是单条轨迹还是多条轨迹
        if predTrajs.ndim == 2:  # 单条轨迹 (20, 3)
            predTrajs = predTrajs[np.newaxis, ...]  # 转换为 (1, 20, 3)
        
        num_paths = predTrajs.shape[0]
        if forbidden_ok is None:
            # 没有 evaluation cost map 时不能把“未审计”冒充 forbidden-pass。
            forbidden_status = [None] * num_paths
        else:
            forbidden_ok = np.asarray(forbidden_ok, dtype=bool)
            if forbidden_ok.shape != (num_paths,):
                raise ValueError(
                    "forbidden_ok must have one entry per predicted trajectory"
                )
            forbidden_status = [bool(value) for value in forbidden_ok]
        draw_order = [index for index in range(num_paths) if index != selected_index]
        if selected_index is not None:
            draw_order.append(int(selected_index))

        for traj_idx in draw_order:
            predTraj = predTrajs[traj_idx]
            predTraj_path = np.asarray(predTraj)[..., :2]
            is_selected = selected_index is not None and traj_idx == selected_index
            is_forbidden_ok = forbidden_status[traj_idx]
            if is_forbidden_ok is True:
                color = "#159895"
                linewidth = 3.2 if is_selected else 1.25
                alpha = 1.0 if is_selected else 0.55
                zorder = 5 if is_selected else 3
                label = (
                    f"selected #{traj_idx} (forbidden-pass)"
                    if is_selected
                    else "forbidden-pass candidates"
                )
            elif is_forbidden_ok is False:
                color = "#8b8f94"
                linewidth = 3.2 if is_selected else 1.0
                alpha = 1.0 if is_selected else 0.45
                zorder = 5 if is_selected else 2
                label = (
                    f"selected #{traj_idx} (forbidden-fail)"
                    if is_selected
                    else "forbidden-fail candidates"
                )
            else:
                color = "#8b8f94"
                linewidth = 1.0
                alpha = 0.45
                zorder = 2
                label = "forbidden-unverified candidates"
            existing_labels = set(ax.get_legend_handles_labels()[1])
            ax.plot(
                predTraj_path[:, 0],
                predTraj_path[:, 1],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder,
                label=None if label in existing_labels else label,
            )

            if is_selected or (num_paths == 1 and selected_index is None):
                arrow_scale = 0.2
                arrow_stride = max(1, len(predTraj) // 12)
                for point in predTraj[::arrow_stride]:
                    x, y, theta = point[:3]
                    ax.arrow(
                        x,
                        y,
                        np.cos(theta) * arrow_scale,
                        np.sin(theta) * arrow_scale,
                        head_width=0.08,
                        head_length=0.12,
                        fc=color,
                        ec=color,
                        zorder=zorder + 1,
                        alpha=alpha,
                    )
        if num_paths > 1:
            ax.legend(loc="lower left", fontsize=7, framealpha=0.85)
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

def plot_elevation_map(
    model,
    pathNums,
    envType,
    save_path=DEFAULT_STAGE1_VISUALIZATION_DIR,
    num_pred_paths=1,
    use_bezier_interpolate=False,
    mask_seed=2026,
    source_seed=2027,
    p_mask=0.5,
    mask_mode="stage1",
    solver="pmf_onestep",
    num_steps=1,
    device="cpu",
    checkpoint_metadata=None,
    vehicle_radius_meters=SAFETY_COST_CONFIG.vehicle_radius_meters,
    dataset_root=MAP_CONFIG.dataset_root,
    split="val",
    critic=None,
    critic_metadata=None,
):
    """绘制多组轨迹对比图
    
    Args:
        pathNums: 路径编号列表
        envType: 环境类型
        save_path: 保存路径
        num_pred_paths: 每个场景生成的预测轨迹数量（1=单条，>1=多条重叠显示）
        use_bezier_interpolate: 是否使用样条插值平滑轨迹（默认False，变量名保持兼容）
        mask_mode: full 使用完整地图；stage1 要求示范轨迹不被遮挡；
            stage2 只保护起终点，允许遮挡旧示范路线。
    """
    if mask_mode not in ("full", "stage1", "stage2"):
        raise ValueError(
            "mask_mode 必须为 full、stage1 或 stage2，"
            f"实际为 {mask_mode!r}"
        )
    require_trajectory_clear = mask_mode != "stage2"

    if not isinstance(pathNums, list):
        pathNums = [pathNums]  # 确保pathNums是列表
    
    # 限制最多显示6组对比
    pathNums = pathNums[:6]
    num_pairs = len(pathNums)
    if num_pairs == 0:
        raise ValueError("至少需要提供一个 path 编号")
    if int(num_pred_paths) < 1:
        raise ValueError("num_pred_paths 必须大于 0")

    # 每个 condition 占“预测/真值”两列；条件不足 2 个时不保留空列。
    pairs_per_row = min(2, num_pairs)
    num_rows = int(np.ceil(num_pairs / pairs_per_row))
    num_columns = pairs_per_row * 2
    fig = plt.figure(
        figsize=(9 * pairs_per_row, 4.8 * num_rows + 1.2)
    )
    gs = plt.GridSpec(
        num_rows,
        num_columns,
        figure=fig,
        left=0.05,
        right=0.95,
        bottom=0.10,
        top=0.88,
        wspace=0.15,
        hspace=0.3,
    )
    
    # 加载环境数据，所有子图共用同一个环境
    envFolder = osp.join(str(dataset_root), str(split), envType)
    # envFolder = osp.join('data/test_training/val', envType)
    env_path = osp.join(envFolder, f'map.p')
    with open(env_path, 'rb') as f:
        env = pickle.load(f)
        tensor = env['tensor']
        elevation = tensor[:, :, 0]
        normal_x = tensor[:, :, 1]
        normal_y = tensor[:, :, 2]
        normal_z = tensor[:, :, 3]
        map_mask = (
            normalize_mask(
                env["mask"], tensor.shape[:2], source=f"{env_path}['mask']"
            )
            if "mask" in env
            else None
        )
        elevation_masked = np.ma.masked_invalid(elevation)
    stability_file = osp.join(envFolder, "stability_map.npz")
    if osp.exists(stability_file):
        with np.load(stability_file) as stability_data:
            evaluation_cost_map = stability_data["cost_map"].astype(np.float32)
    else:
        evaluation_cost_map = None
    if critic is not None and evaluation_cost_map is None:
        raise FileNotFoundError(
            "Stage-2 selection visualization requires the precomputed "
            f"stability map: {stability_file}"
        )
    condition_records = []
    
    # 为每对轨迹创建子图
    for idx, pathNum in enumerate(pathNums):
        # 计算当前对应在网格中的位置
        row = idx // pairs_per_row
        col = (idx % pairs_per_row) * 2
        
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
        
        base_model = model.module if hasattr(model, "module") else model
        if getattr(base_model, "map_channels", 3) == 4:
            if mask_mode == "full":
                mask = np.ones_like(elevation, dtype=np.float32)
                mask_metadata = {
                    "accepted_type": "full_map",
                    "semantic_mode": "full",
                    "masked_fraction": 0.0,
                    "demo_blocked_fraction": 0.0,
                    "demo_max_contiguous_blocked_fraction": 0.0,
                    "sampling_attempts": 0,
                }
            elif "mask" in path_data:
                mask = normalize_mask(
                    path_data["mask"],
                    elevation.shape,
                    source=f"{path_file}['mask']",
                )
                mask = erode_mask_for_vehicle(
                    mask, vehicle_radius_meters=vehicle_radius_meters
                )
                mask_metadata = {"accepted_type": "path_data"}
            elif map_mask is not None:
                mask = erode_mask_for_vehicle(
                    map_mask, vehicle_radius_meters=vehicle_radius_meters
                )
                mask_metadata = {"accepted_type": "map_data"}
            else:
                mask, mask_metadata = generate_random_mask(
                    elevation.shape,
                    mask_seed + int(pathNum) * 1_000_003,
                    trajectory[:, :2],
                    p_mask=p_mask,
                    require_trajectory_clear=require_trajectory_clear,
                    vehicle_radius_meters=vehicle_radius_meters,
                    return_metadata=True,
                )
            print(
                f"Mask path={pathNum} mode={mask_mode} "
                f"type={mask_metadata.get('accepted_type', 'dataset')} "
                f"masked={float(mask_metadata.get('masked_fraction', 1.0 - mask.mean())):.1%} "
                "demo_blocked="
                f"{float(mask_metadata.get('demo_blocked_fraction', 0.0)):.1%}"
            )
            encoder_input = build_partial_map_input(
                normal_x,
                normal_y,
                normal_z,
                mask,
                noise_seed=(
                    mask_seed
                    + int(pathNum) * 1_000_003
                    + MASK_NOISE_SEED_OFFSET
                ),
            )
            prediction_elevation = np.ma.masked_where(
                mask < 0.5, elevation
            )
        else:
            if mask_mode != "full":
                raise ValueError(
                    "旧 3 通道模型不支持显式 mask 可视化；请使用 full 模式。"
                )
            mask = np.ones_like(elevation, dtype=np.float32)
            encoder_input = torch.from_numpy(
                np.stack([normal_x, normal_y, normal_z], axis=0)
            ).float()
            prediction_elevation = elevation_masked

        source_generator = torch.Generator(device=torch.device(device))
        source_generator.manual_seed(
            int(source_seed) + int(pathNum) * 1_000_003
        )
        source_noise = torch.randn(
            int(num_pred_paths),
            base_model.num_edges,
            2,
            generator=source_generator,
            device=device,
        )
        generated = generate_paths(
            model,
            map_input=encoder_input[None, :].to(device),
            start_point=torch.tensor(start_pos).float().to(device),
            goal_point=torch.tensor(goal_pos).float().to(device),
            num_paths=num_pred_paths,
            reconstruct_trajectory=True,
            num_traj_points=DENSE_TRAJECTORY_POINTS,
            solver=solver,
            num_steps=num_steps,
            source_noise=source_noise,
            return_details=True,
        )
        predTrajs = generated["trajectories"]
        box_metrics = raw_box_metrics(predTrajs)
        print("Raw no-repair box metrics:", box_metrics)
        record = {
            "path": int(pathNum),
            "mask_mode": mask_mode,
            "num_samples": int(num_pred_paths),
            "masked_fraction": float(1.0 - mask.mean()),
            **box_metrics,
        }
        panel_metrics = None
        if evaluation_cost_map is not None:
            # generate_paths 为 Matplotlib 返回 numpy；hard validity 使用 torch。
            pred_tensor = torch.as_tensor(
                predTrajs,
                dtype=torch.float32,
                device=device,
            )
            validity = trajectory_validity_metrics(
                pred_tensor[..., :2],
                evaluation_cost_map,
                MAP_CONFIG.cost_map_info(),
                analytic_yaw=pred_tensor[..., 2],
                analytic_curvature=pred_tensor[..., 7],
                analytic_curvature_audit=generated["curvature_audit"],
                mask=mask,
                start_pose=torch.tensor(
                    start_pos,
                    dtype=pred_tensor.dtype,
                    device=pred_tensor.device,
                ),
                goal_pose=torch.tensor(
                    goal_pos,
                    dtype=pred_tensor.dtype,
                    device=pred_tensor.device,
                ),
                condition_ids=torch.zeros(
                    pred_tensor.shape[0],
                    dtype=torch.long,
                    device=pred_tensor.device,
                ),
            )
            forbidden_ok = validity["forbidden_region_ok"].bool()
            stability_ok = validity["stability_ok"].bool()
            capsizing_safe = forbidden_ok & stability_ok
            panel_metrics = {
                "proposal_strict_valid_rate": float(validity["strict_valid_rate"]),
                "strict_safe_at_k": float(validity["safe_at_k"]),
                "proposal_capsizing_safe_rate": float(capsizing_safe.float().mean()),
                "capsizing_safe_at_k": bool(capsizing_safe.any()),
                "forbidden_pass_rate": float(
                    forbidden_ok.float().mean()
                ),
                "stability_pass_rate": float(
                    stability_ok.float().mean()
                ),
                "curvature_pass_rate": float(
                    validity["curvature_ok"].float().mean()
                ),
                "yaw_pass_rate": float(
                    validity["endpoint_yaw_ok"].float().mean()
                ),
                "max_curvature_median": float(
                    validity["max_curvature"].median()
                ),
                "max_endpoint_yaw_error_rad": float(
                    torch.maximum(
                        validity["start_yaw_error"],
                        validity["goal_yaw_error"],
                    ).max()
                ),
            }
            record.update(panel_metrics)
            forbidden_ok_for_plot = (
                forbidden_ok.detach().cpu().numpy()
            )
            record.update(
                {
                    "forbidden_ok_indices": np.flatnonzero(
                        forbidden_ok_for_plot
                    ).astype(int).tolist(),
                    "forbidden_fail_indices": np.flatnonzero(
                        ~forbidden_ok_for_plot
                    ).astype(int).tolist(),
                    "mask_signed_distance_min": [
                        float(value)
                        for value in validity[
                            "mask_signed_distance_min"
                        ].detach().cpu()
                    ],
                    "forbidden_violation_max": [
                        float(value)
                        for value in validity[
                            "forbidden_violation_max"
                        ].detach().cpu()
                    ],
                    "mask_segment_sample_count": [
                        int(value)
                        for value in validity[
                            "mask_segment_sample_count"
                        ].detach().cpu()
                    ],
                }
            )
            print("Analytic validity:", panel_metrics)
        else:
            record["validity_metrics"] = "stability_map.npz unavailable"
            print("Analytic validity: skipped (stability_map.npz unavailable)")
        selection = None
        selected_index = None
        forbidden_ok_for_plot = None
        if evaluation_cost_map is not None:
            forbidden_ok_for_plot = (
                validity["forbidden_region_ok"]
                .bool()
                .detach()
                .cpu()
                .numpy()
            )
        if critic is not None:
            selection = select_stage2_candidate(
                critic,
                encoder_input[None, :].to(device),
                generated,
                validity,
            )
            selected_index = selection["selected_index"]
            forbidden_for_plot = [
                candidate["forbidden_ok"]
                for candidate in selection["candidates"]
            ]
            record["stage2_selection"] = selection
            if selected_index is None:
                print("Stage-2 selector: REJECT (all candidates failed forbidden)")
            else:
                selected = selection["candidates"][selected_index]
                ranked = selection["ranking"][:5]
                print(
                    "Stage-2 selector:",
                    {
                        "selected": selected_index,
                        "p_safe": selected["critic_safe_probability"],
                        "predicted_risk": selected["critic_predicted_risk"],
                        "max_curvature": selected["max_curvature_inv_m"],
                        "curvature_preference": selected["curvature_preference"],
                        "score": selected["selection_score"],
                        "privileged_capsizing_safe_diagnostic": selected[
                            "capsizing_safe_privileged_diagnostic"
                        ],
                        "top5": ranked,
                    },
                )
        condition_records.append(record)
        
        # 如果只有一条轨迹，squeeze掉第一维
        if num_pred_paths == 1:
            predTrajs = predTrajs.squeeze(0)
            output_dim = predTrajs.shape[0]
        else:
            output_dim = predTrajs.shape[1]
        
        if num_pred_paths == 1:
            print(
                f"Generated one raw trajectory with {predTrajs.shape[0]} points; "
                f"start={start_pos.tolist()}, goal={goal_pos.tolist()}"
            )
        else:
            print(
                f"Generated {num_pred_paths} raw trajectories; "
                f"start={start_pos.tolist()}, goal={goal_pos.tolist()}"
            )
        
        # 创建左侧子图 - 预测轨迹
        ax_pred = fig.add_subplot(gs[row, col])
        # 预测轨迹已在Models.py::sample()中覆盖首尾。
        plot_single_trajectory(
            ax_pred,
            prediction_elevation,
            trajectory,
            predTrajs,
            output_dim,
            is_pred=True,
            use_bezier_interpolate=use_bezier_interpolate,
            selected_index=selected_index,
            forbidden_ok=forbidden_ok_for_plot,
        )
        if panel_metrics is None:
            ax_pred.set_title(f"{mask_mode}: trajectory samples")
        elif selection is not None and selected_index is None:
            ax_pred.set_title(
                f"Stage 2: REJECT (all K={num_pred_paths} fail forbidden)\n"
                f"Capsizing Safe@K={panel_metrics['capsizing_safe_at_k']}",
                fontsize=9,
            )
        elif selection is not None:
            selected = selection["candidates"][selected_index]
            truth = (
                "safe"
                if selected["capsizing_safe_privileged_diagnostic"]
                else "unsafe"
            )
            ax_pred.set_title(
                f"Selected #{selected_index} | "
                f"p_safe {selected['critic_safe_probability']:.3f} | "
                f"kappa {selected['max_curvature_inv_m']:.2f} | "
                f"score {selected['selection_score']:.3f}\n"
                f"Privileged truth: {truth} | "
                f"Capsizing Safe@{num_pred_paths}: "
                f"{panel_metrics['capsizing_safe_at_k']}",
                fontsize=9,
            )
        else:
            ax_pred.set_title(
                f"{mask_mode}: forbidden-pass="
                f"{panel_metrics['forbidden_pass_rate']:.1%}, "
                f"strict-valid="
                f"{panel_metrics['proposal_strict_valid_rate']:.1%}, "
                f"strict Safe@K={panel_metrics['strict_safe_at_k']:.0%}",
                fontsize=9,
            )
        
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
    workflow_label = "Stage 2" if critic is not None else mask_mode.capitalize()
    selector_label = " | forbidden + critic x curvature" if critic is not None else ""
    fig.suptitle(
        f"Path MeanFlow {workflow_label} | {envType} | K={num_pred_paths}"
        f"{selector_label}",
        fontsize=13,
        y=0.98,
        fontweight="bold",
    )
    
    # 保存图像
    save_path = osp.join(save_path, f'{envType}')
    os.makedirs(save_path, exist_ok=True)
    
    # 生成路径编号列表的简短表示，例如 1_2_3
    path_ids_str = "_".join(str(p) for p in pathNums)
    figure_path = osp.join(
        save_path,
        f"boundary_constrained_{mask_mode}_trajectories_{path_ids_str}.png",
    )
    summary_path = osp.join(
        save_path,
        f"boundary_constrained_{mask_mode}_trajectories_{path_ids_str}.json",
    )
    plt.savefig(figure_path, dpi=300)
    plt.close(fig)
    summary = {
        "checkpoint": checkpoint_metadata or {},
        "critic": critic_metadata or {},
        "dataset_root": osp.abspath(str(dataset_root)),
        "split": str(split),
        "environment": envType,
        "mask_mode": mask_mode,
        "solver": solver,
        "num_steps": int(num_steps),
        "mask_seed": int(mask_seed),
        "source_seed": int(source_seed),
        "conditions": condition_records,
    }
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False)
    print(f"Saved multi-trajectory comparison figure for paths {path_ids_str} in {envType} environment.")
    print("Figure:", figure_path)
    print("Metrics:", summary_path)
    return figure_path, summary_path


def load_generator_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    protocol = checkpoint.get("protocol") or {}
    raw_model_args = checkpoint.get("model_args") or protocol.get("model_args")
    if raw_model_args is None:
        raise ValueError(
            "Checkpoint lacks model_args both at the top level and in protocol."
        )
    checkpoint_stage = str(checkpoint.get("stage", "stage1")).lower()
    is_stage2 = (
        "stage2" in checkpoint_stage
        or checkpoint_stage == DEVELOPMENT_DIRECT_COST_STAGE
    )
    representation_semantics = checkpoint.get("representation_semantic_version")
    input_semantics = checkpoint.get("input_mask_semantics")
    if is_stage2:
        # Compact development checkpoints may omit the frozen source protocol.
        representation_semantics = (
            representation_semantics or PATH_REPRESENTATION_SEMANTIC_VERSION
        )
        input_semantics = input_semantics or MASK_INPUT_SEMANTICS
    if representation_semantics != PATH_REPRESENTATION_SEMANTIC_VERSION:
        raise ValueError(
            "Checkpoint is incompatible with the first-order "
            "boundary-constrained B-spline path representation."
        )
    if input_semantics != MASK_INPUT_SEMANTICS:
        raise ValueError(
            "Checkpoint 使用旧 mask/法向量填充语义，不能用于当前可视化输入。"
        )
    model_args = {
        key: value
        for key, value in raw_model_args.items()
        if key in PathMeanFlowTransformer.CONFIG_KEYS
    }
    model = PathMeanFlowTransformer(**model_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    model.use_gradient_checkpoint = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if getattr(model, "use_radial_output", True):
        raise ValueError(
            "Boundary-constrained path visualization rejects legacy radial "
            "checkpoints."
        )
    metadata = {
        "path": osp.abspath(checkpoint_path),
        "stage": checkpoint.get("stage"),
        "arm": checkpoint.get("arm"),
        "epoch": checkpoint.get("epoch"),
        "global_update": checkpoint.get("global_update"),
        "protocol_sha256": checkpoint.get("protocol_sha256"),
        "checkpoint_semantics": checkpoint.get("checkpoint_semantics"),
        "optimizer_update": checkpoint.get("optimizer_update"),
        "train_loss": checkpoint.get("train_loss"),
        "val_loss": checkpoint.get("val_loss"),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "representation_semantic_version": representation_semantics,
        "input_mask_semantics": input_semantics,
        "source_checkpoint": protocol.get("checkpoint"),
        "source_checkpoint_sha256": protocol.get("checkpoint_sha256"),
        "protocol_version": protocol.get("version"),
    }
    return model, checkpoint, metadata, is_stage2


def load_stage2_critic(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    config = checkpoint.get("critic_config") or {}
    critic = PathAlignedCandidateCritic(
        map_channels=int(config.get("map_channels", 4)),
        hidden_dim=int(config.get("hidden_dim", 128)),
    ).to(device)
    critic.load_state_dict(checkpoint["critic_state_dict"], strict=True)
    critic.eval()
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    safe_semantics = checkpoint.get("safe_head_semantics")
    if safe_semantics != "P(stability_ok | forbidden_ok, deployment_inputs)":
        raise ValueError(
            "The selected Stage-2 rule requires the forbidden-conditioned "
            "stability safe head from the dual-head critic."
        )
    metadata = {
        "path": osp.abspath(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
        "best_validation_loss": checkpoint.get("best_validation_loss"),
        "stability_critic_semantics": checkpoint.get(
            "stability_critic_semantics"
        ),
        "safe_head_semantics": safe_semantics,
        "risk_head_semantics": checkpoint.get("risk_head_semantics"),
        "source_checkpoint": checkpoint.get("source_checkpoint"),
        "source_checkpoint_sha256": checkpoint.get("source_checkpoint_sha256"),
    }
    return critic, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Visualize Boundary-Constrained Path MeanFlow checkpoints with "
            "analytic path geometry and hard-validity diagnostics."
        )
    )
    parser.add_argument(
        "--workflow",
        choices=("stage1", "stage2"),
        default=None,
        help="默认从 checkpoint 推断；未指定 checkpoint 时默认 stage2。",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
    )
    parser.add_argument(
        "--critic-checkpoint",
        "--critic_checkpoint",
        dest="critic_checkpoint",
        default=DEFAULT_STAGE2_CRITIC_CHECKPOINT,
    )
    parser.add_argument(
        "--without-selector",
        action="store_true",
        help="只画生成器候选池，不运行 forbidden/critic/curvature 选择器。",
    )
    parser.add_argument("--environment", required=True)
    parser.add_argument("--paths", type=int, nargs="+", default=[0, 3, 6, 9, 12, 15])
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument(
        "--output",
        default=None,
    )
    parser.add_argument(
        "--dataset-root",
        "--dataset_root",
        dest="dataset_root",
        default=None,
        help="数据集根目录；Stage 2 默认 data/dataset1_val。",
    )
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--mask_seed", type=int, default=2026)
    parser.add_argument("--source_seed", type=int, default=2027)
    parser.add_argument(
        "--p_mask",
        type=float,
        default=None,
        help="默认 Stage 1 为 0.5，Stage 2 为 1.0。",
    )
    parser.add_argument(
        "--mask_mode",
        choices=("full", "stage1", "stage2"),
        default=None,
        help=(
            "full=完整地图；stage1=示范轨迹不被遮挡；"
            "stage2=有界阻断 mask。默认根据 checkpoint stage 推断。"
        ),
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=(1, 2),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--solver",
        choices=("pmf_onestep", "pmf_refined", "euler", "heun"),
        default="pmf_onestep",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=3,
        help="pmf_refined/euler/heun 的积分步数；pmf_onestep 时忽略。",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto、cpu、cuda 或 cuda:N。",
    )
    parser.add_argument(
        "--vehicle_radius_meters",
        type=float,
        default=SAFETY_COST_CONFIG.vehicle_radius_meters,
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前 PyTorch 无法使用 CUDA。")

    workflow_hint = args.workflow
    if args.stage is not None:
        legacy_workflow = f"stage{args.stage}"
        if workflow_hint is not None and workflow_hint != legacy_workflow:
            raise ValueError("--stage 与 --workflow 冲突。")
        workflow_hint = legacy_workflow
    checkpoint_path = args.checkpoint or (
        DEFAULT_STAGE1_CHECKPOINT
        if workflow_hint == "stage1"
        else DEFAULT_STAGE2_CHECKPOINT
    )
    model, checkpoint, checkpoint_metadata, checkpoint_is_stage2 = (
        load_generator_checkpoint(checkpoint_path, device)
    )
    inferred_workflow = "stage2" if checkpoint_is_stage2 else "stage1"
    workflow = workflow_hint or inferred_workflow
    if workflow != inferred_workflow:
        raise ValueError(
            f"--workflow={workflow} 与 checkpoint stage={checkpoint.get('stage')} 冲突。"
        )
    mask_mode = args.mask_mode or workflow
    if args.stage is not None and mask_mode != workflow:
        raise ValueError("--stage 与 --mask_mode 冲突。")
    num_samples = args.num_samples or (16 if workflow == "stage2" else 32)
    if num_samples <= 0:
        raise ValueError("--num_samples 必须大于 0。")
    output = args.output or (
        DEFAULT_STAGE2_VISUALIZATION_DIR
        if workflow == "stage2"
        else DEFAULT_STAGE1_VISUALIZATION_DIR
    )
    dataset_root = args.dataset_root or (
        "data/dataset1_val" if workflow == "stage2" else str(MAP_CONFIG.dataset_root)
    )
    p_mask = args.p_mask if args.p_mask is not None else (
        1.0 if workflow == "stage2" else 0.5
    )
    if not 0.0 <= p_mask <= 1.0:
        raise ValueError("--p_mask 必须位于 [0, 1]。")
    critic = None
    critic_metadata = None
    if workflow == "stage2" and not args.without_selector:
        critic, critic_metadata = load_stage2_critic(
            args.critic_checkpoint,
            device,
        )
    print(
        "Loaded Boundary-Constrained Path MeanFlow checkpoint:",
        {
            "stage": checkpoint_metadata["stage"],
            "epoch": checkpoint_metadata["epoch"],
            "update": checkpoint_metadata["global_update"],
            "val_loss": checkpoint_metadata["val_loss"],
            "mask_mode": mask_mode,
            "workflow": workflow,
            "num_samples": num_samples,
            "dataset": osp.abspath(dataset_root),
            "p_mask": p_mask,
            "selector": critic is not None,
            "device": device,
        },
    )
    plot_elevation_map(
        model,
        args.paths,
        args.environment,
        output,
        num_pred_paths=num_samples,
        use_bezier_interpolate=True,
        mask_seed=args.mask_seed,
        source_seed=args.source_seed,
        p_mask=p_mask,
        mask_mode=mask_mode,
        solver=args.solver,
        num_steps=args.num_steps,
        device=device,
        checkpoint_metadata=checkpoint_metadata,
        vehicle_radius_meters=args.vehicle_radius_meters,
        dataset_root=dataset_root,
        split=args.split,
        critic=critic,
        critic_metadata=critic_metadata,
    )

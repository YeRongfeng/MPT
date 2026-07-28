import math

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

import tqdm

# 导入评估器
from evaluator import TrajectoryEvaluator
# B样条工具
from bspline_utils import (
    fit_bspline_least_squares,
    reconstruct_from_control_points,
    DifferentiableBSpline
)
from map_config import (
    MAP_BOUNDS,
    MAP_CONFIG,
    MAP_HALF_EXTENT,
    MAP_RESOLUTION,
    MAP_YAW_BINS,
    SAFETY_COST_CONFIG,
)

# 从xy坐标计算theta（通过差分计算切向量）
def compute_theta_from_xy(traj_xy):
    """
    从轨迹的xy坐标计算theta角度
    Args:
        traj_xy: (N, 2) 或 (B, N, 2) 轨迹的xy坐标
    Returns:
        theta: (N,) 或 (B, N) 每个点的切向角度
    """
    # 【关键修复】保持输入张量的梯度链
    # 只在必要时转换为张量，避免断开 requires_grad 链
    if not torch.is_tensor(traj_xy):
        traj_xy = torch.tensor(traj_xy, dtype=torch.float32)
    else:
        # 如果已是张量，确保数据类型正确（保持梯度）
        traj_xy = traj_xy.float()

    if traj_xy.ndim == 2:
        traj_xy = traj_xy.unsqueeze(0)  # (1, N, 2)
        squeeze_back = True
    elif traj_xy.ndim == 3:
        squeeze_back = False
    else:
        raise ValueError(f"traj_xy must be 2D or 3D, got shape={traj_xy.shape}")

    B, N, _ = traj_xy.shape
    if N < 2:
        theta = torch.zeros((B, N), dtype=traj_xy.dtype, device=traj_xy.device)
        return theta.squeeze(0) if squeeze_back else theta

    # 向量化差分（可微分，避免in-place写入导致梯度断开）
    # 中间点：中心差分；两端点：前向/后向差分
    dxy_mid = traj_xy[:, 2:, :] - traj_xy[:, :-2, :]      # (B, N-2, 2)
    dxy_start = traj_xy[:, 1:2, :] - traj_xy[:, 0:1, :]   # (B, 1, 2)
    dxy_end = traj_xy[:, -1:, :] - traj_xy[:, -2:-1, :]   # (B, 1, 2)
    dxy = torch.cat([dxy_start, dxy_mid, dxy_end], dim=1) # (B, N, 2)

    theta = torch.atan2(dxy[:, :, 1], dxy[:, :, 0])       # (B, N)
    return theta.squeeze(0) if squeeze_back else theta

def cost_on_dense_trajectory(trajectory, start_pose, goal_pose, occupancy_map, map_info, device='cpu', return_per_sample=False):
        """
        对已生成的密集轨迹直接计算 Cost（支持批量处理）。
        trajectory: (B, N, 2) 或 (B, N, 3)
        occupancy_map: torch tensor (D, H, W) cost map
        map_info: dict with 'origin', 'resolution', and 'size'
        """
        # normalize inputs to tensor on device
        if not torch.is_tensor(trajectory):
            trajectory = torch.tensor(trajectory, dtype=torch.float32, device=device)
        else:
            trajectory = trajectory.to(device)

        # 数值稳定性：清理上游偶发 NaN/Inf，避免传播导致整批损失失效
        trajectory = torch.nan_to_num(
            trajectory,
            nan=0.0,
            posinf=MAP_HALF_EXTENT,
            neginf=-MAP_HALF_EXTENT,
        )

        if trajectory.ndim == 2:  # 单条轨迹情况
            trajectory = trajectory.unsqueeze(0)  # 添加 batch 维度

        B, N, _ = trajectory.shape  # Batch size, number of points

        # ensure we have x, y
        traj_xy = trajectory[:, :, :2]  # (B, N, 2)
        theta = compute_theta_from_xy(traj_xy).to(device)  # (B, N)
        trajectory = torch.cat([traj_xy, theta.unsqueeze(-1)], dim=-1)  # (B, N, 3)

        # 1. 提取状态
        x = trajectory[:, :, 0]  # (B, N)
        y = trajectory[:, :, 1]  # (B, N)
        yaw = trajectory[:, :, 2]  # (B, N)

        # 2. 计算网格索引
        origin = map_info['origin']
        resolution = map_info['resolution']
        W, H, D = map_info['size']

        # 使用连续坐标进行可微分采样，避免 long()/离散索引导致梯度为0
        x_idx_f = (x - origin[0]) / resolution
        y_idx_f = (y - origin[1]) / resolution
        yaw_rel = torch.remainder(yaw - origin[2], 2 * np.pi)
        yaw_idx_f = yaw_rel / (2 * np.pi / D)

        # clamp 到有效范围（保持 float，可导）
        x_idx_f = torch.clamp(x_idx_f, 0.0, float(W - 1))
        y_idx_f = torch.clamp(y_idx_f, 0.0, float(H - 1))
        yaw_idx_f = torch.clamp(yaw_idx_f, 0.0, float(D - 1))

        # 3. 构造 grid 并采样
        # PyTorch expects grid last-dim as (x, y, z) for 5D grid_sample with input (N, C, D, H, W)
        # 所以按 (x_norm, y_norm, z_norm) 的顺序构造
        x_norm = (x_idx_f / (W - 1)) * 2.0 - 1.0
        y_norm = (y_idx_f / (H - 1)) * 2.0 - 1.0
        z_norm = (yaw_idx_f / (D - 1)) * 2.0 - 1.0

        # stack 为 (B, N, 3) in order (x, y, z)
        grid = torch.stack([x_norm, y_norm, z_norm], dim=-1)  # (B, N, 3)

        # grid_sample 对于 5D 输入 (N, C, D, H, W) 期望 grid 为 (N, D_out, H_out, W_out, 3)
        # 我们要对每个轨迹点采样，因此将 grid reshape 为 (B, N, 1, 1, 3)
        grid = grid.view(B, N, 1, 1, 3)

        # occupancy_map 统一为 (B, C=1, D, H, W)
        # 关键：优先使用 expand / 索引映射，避免 repeat 造成巨量显存占用
        occ = occupancy_map.to(device)

        # 维度语义对齐：兼容 (H,W,D)/(B,H,W,D) 输入，统一转为 (D,H,W)/(B,D,H,W)
        if occ.ndim == 3:
            if occ.shape[-1] == D and occ.shape[0] == H and occ.shape[1] == W:
                occ = occ.permute(2, 0, 1).contiguous()  # (H,W,D) -> (D,H,W)
        elif occ.ndim == 4:
            if occ.shape[-1] == D and occ.shape[1] == H and occ.shape[2] == W:
                occ = occ.permute(0, 3, 1, 2).contiguous()  # (B,H,W,D) -> (B,D,H,W)

        if occ.ndim == 3:
            # (D,H,W) -> (1,1,D,H,W) -> (B,1,D,H,W)
            occ = occ.unsqueeze(0).unsqueeze(0)
            if B > 1:
                occ = occ.expand(B, -1, -1, -1, -1)
        elif occ.ndim == 4:
            # (B_map,D,H,W) -> (B,1,D,H,W)
            b_map = occ.shape[0]
            if b_map == B:
                occ = occ.unsqueeze(1)
            elif b_map == 1:
                occ = occ.unsqueeze(1).expand(B, -1, -1, -1, -1)
            elif B % b_map == 0:
                # 常见场景：轨迹做了 num_samples 展开（B = b_map * k）
                k = B // b_map
                map_idx = torch.arange(B, device=device) // k  # [0,0,...,1,1,...]
                occ = occ[map_idx].unsqueeze(1)
            else:
                # 退化回退：按比例映射到最近的 map batch
                map_idx = torch.linspace(0, b_map - 1, steps=B, device=device).long()
                occ = occ[map_idx].unsqueeze(1)
        elif occ.ndim == 5:
            # 已是 (B_map,C,D,H,W)
            b_map = occ.shape[0]
            if b_map == B:
                pass
            elif b_map == 1:
                occ = occ.expand(B, -1, -1, -1, -1)
            elif B % b_map == 0:
                k = B // b_map
                map_idx = torch.arange(B, device=device) // k
                occ = occ[map_idx]
            else:
                map_idx = torch.linspace(0, b_map - 1, steps=B, device=device).long()
                occ = occ[map_idx]
        else:
            raise ValueError(f"Unsupported occupancy_map shape: {tuple(occ.shape)}")

        # 使用 5D grid_sample 进行三维采样，输出形状 (B, C, N, 1, 1)
        occ_vals = F.grid_sample(occ, grid, mode='bilinear', padding_mode='border', align_corners=True)
        # 将 shape 调整为 (B, N)
        occ_vals = occ_vals.squeeze(-1).squeeze(-1)  # (B, C, N)
        occ_vals = occ_vals.squeeze(1)  # (B, N)

        # 4. 计算障碍物代价
        d_safe = 0.15
        # kalpa = 0.7
        # z = (-(occ_vals - d_safe) / (kalpa + 1e-12))
        # occupancy_values = torch.sigmoid(torch.clamp(z, min=-50.0, max=50.0))
        
        alpha = 10.0 
        occupancy_values = torch.nn.functional.softplus(-alpha * (occ_vals - d_safe)) / alpha
        
        # occupancy_values shape: (B, N) -> 对每条轨迹在点维度上求均值，得到 (B,)
        obstacle_cost = torch.mean(occupancy_values, dim=1) * 1e0  # Batch mean over points

        # 5. 其他代价计算（平滑性、曲率等）
        dx = x[:, 1:] - x[:, :-1]
        dy = y[:, 1:] - y[:, :-1]
        dyaw = torch.atan2(torch.sin(yaw[:, 1:] - yaw[:, :-1]), torch.cos(yaw[:, 1:] - yaw[:, :-1]))

        ddx = dx[:, 1:] - dx[:, :-1]
        ddy = dy[:, 1:] - dy[:, :-1]
        ddyaw = torch.atan2(torch.sin(dyaw[:, 1:] - dyaw[:, :-1]), torch.cos(dyaw[:, 1:] - dyaw[:, :-1]))

        smoothness_cost = torch.mean(ddx**2 + ddy**2 + ddyaw**2, dim=1)

        # 仅使用几何曲率项（按你的要求保持控制变量）
        # 使用三点外接圆离散曲率：κ = 4A/(abc) = 2|v1×v2|/(|v1||v2||p3-p1|)
        # 该形式对参数化更稳健，不依赖 ds_floor，低速/短步长时不会被“放松约束”。
        p1 = trajectory[:, :-2, :2]
        p2 = trajectory[:, 1:-1, :2]
        p3 = trajectory[:, 2:, :2]

        v1 = p2 - p1
        v2 = p3 - p2
        chord = p3 - p1

        a = torch.sqrt(torch.sum(v1 * v1, dim=-1) + 1e-12)
        b = torch.sqrt(torch.sum(v2 * v2, dim=-1) + 1e-12)
        c = torch.sqrt(torch.sum(chord * chord, dim=-1) + 1e-12)
        cross = torch.abs(v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0])

        geom_curvature = 2.0 * cross / (a * b * c + 1e-12)

        curvature_limit = 1.4 # 越小曲率半径越大，转弯越平缓；越大允许更急的转弯
        curvature_violation = F.relu(geom_curvature - curvature_limit)

        # # Huber 罚比纯平方更稳（超大超限时线性增长），同时保留低超限区间的二次收敛特性
        # huber_delta = 0.3 # 曲率超限多少才进入线性增长区间，单位同 curvature_violation（即曲率值）
        # curvature_cost = torch.mean(
        #     torch.where(
        #         curvature_violation < huber_delta,
        #         0.5 * (curvature_violation ** 2),
        #         huber_delta * (curvature_violation - 0.5 * huber_delta)
        #     ),
        #     dim=1
        # )
        curvature_cost = torch.mean(curvature_violation**2, dim=1)

        # 支持 start_pose/goal_pose 为标量 (3,) 或 batch (B,3)
        if torch.is_tensor(start_pose):
            if start_pose.ndim == 1:
                start_yaw = start_pose[2].to(device)
            else:
                start_yaw = start_pose[:, 2].to(device)
        else:
            start_yaw = torch.tensor(start_pose, device=device, dtype=torch.float32)
            if start_yaw.ndim == 1:
                start_yaw = start_yaw[2]

        if torch.is_tensor(goal_pose):
            if goal_pose.ndim == 1:
                goal_yaw = goal_pose[2].to(device)
            else:
                goal_yaw = goal_pose[:, 2].to(device)
        else:
            goal_yaw = torch.tensor(goal_pose, device=device, dtype=torch.float32)
            if goal_yaw.ndim == 1:
                goal_yaw = goal_yaw[2]

        # 计算起点yaw偏差和终点yaw偏差（支持 batch 或 broadcast）
        yaw_diff_start = torch.atan2(torch.sin(yaw[:, 0] - start_yaw), torch.cos(yaw[:, 0] - start_yaw))
        yaw_diff_end = torch.atan2(torch.sin(yaw[:, -1] - goal_yaw), torch.cos(yaw[:, -1] - goal_yaw))
        yaw_endpoint_cost = yaw_diff_start**2 + yaw_diff_end**2

        ddx_full = torch.zeros_like(x)
        ddy_full = torch.zeros_like(y)
        ddyaw_full = torch.zeros_like(yaw)
        ddx_full[:, 2:] = ddx
        ddy_full[:, 2:] = ddy
        ddyaw_full[:, 2:] = ddyaw
        jerk_x = ddx_full[:, 2:] - ddx_full[:, 1:-1]
        jerk_y = ddy_full[:, 2:] - ddy_full[:, 1:-1]
        jerk_yaw = ddyaw_full[:, 2:] - ddyaw_full[:, 1:-1]
        jerk_cost = torch.mean(jerk_x**2 + jerk_y**2 + 1e0 * jerk_yaw**2, dim=1)

        # out of bound penalty: 轨迹点必须在地图边界内至少 g_safe 个像素的范围内
        g_safe = 1  # 安全边界，单位像素
        x_min = float(origin[0]) + g_safe * resolution
        x_max = float(origin[0] + (W - 1) * resolution) - g_safe * resolution
        y_min = float(origin[1]) + g_safe * resolution
        y_max = float(origin[1] + (H - 1) * resolution) - g_safe * resolution
        g_x_low = F.relu(x_min - x)
        g_x_high = F.relu(x - x_max)
        g_y_low = F.relu(y_min - y)
        g_y_high = F.relu(y - y_max)
        out_of_bound_cost = torch.mean(g_x_low + g_x_high + g_y_low + g_y_high, dim=1) * 1e3

        # 6. 权重组合
        weights = {
            'obstacle': 1e-1,
            'smoothness': 0e-7,
            'curvature': 1e-3,
            'endpoints': 1e-3,
            'jerk': 1e-4,
            'out_of_bound': 1e-3,
        }

        total_cost = (
            weights['obstacle'] * obstacle_cost +
            weights['smoothness'] * smoothness_cost +
            weights['curvature'] * curvature_cost +
            weights['endpoints'] * yaw_endpoint_cost +
            weights['jerk'] * jerk_cost +
            weights['out_of_bound'] * out_of_bound_cost
        )

        # 支持返回逐样本代价（GR-AWF 组内相对优势）或 batch 均值（兼容原逻辑）
        if return_per_sample:
            return total_cost

        # 返回标量：对 batch 上的 total_cost 求均值
        # 注意：外部训练还会乘以 loss_weights['capsize']，这里不再额外缩小
        return torch.mean(total_cost)


def top_tail_mean(values, ratio=0.05):
    """返回每个样本最坏 ``ratio`` 比例点的均值。

    Args:
        values: 形状为 (B, N) 的逐点代价/违规量。
        ratio: 轨迹尾部比例，范围为 (0, 1]。
    """
    if values.ndim != 2:
        raise ValueError(f"values must have shape (B, N), got {tuple(values.shape)}")
    if values.shape[1] == 0:
        raise ValueError("values must contain at least one trajectory point")
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")

    k = max(1, math.ceil(values.shape[1] * ratio))
    return torch.topk(values, k=k, dim=1, largest=True, sorted=False).values.mean(dim=1)


def discrete_turning_curvature(traj_xy, eps=1e-12):
    """用“相邻线段转角 / 局部弧长”计算离散曲率。

    相比三点外接圆公式，该定义在三点共线但运动方向相反时不会
    退化为零：正向直线的转角为 0，180 度掉头的转角为 pi。

    Args:
        traj_xy: (B, N, 2) 轨迹。

    Returns:
        (B, N-2) 离散曲率。
    """
    if traj_xy.ndim != 3 or traj_xy.shape[-1] != 2:
        raise ValueError(f"traj_xy must have shape (B,N,2), got {tuple(traj_xy.shape)}")
    if traj_xy.shape[1] < 3:
        raise ValueError("traj_xy must contain at least three points")

    incoming = traj_xy[:, 1:-1, :] - traj_xy[:, :-2, :]
    outgoing = traj_xy[:, 2:, :] - traj_xy[:, 1:-1, :]
    incoming_length = torch.linalg.vector_norm(incoming, dim=-1)
    outgoing_length = torch.linalg.vector_norm(outgoing, dim=-1)

    cross = incoming[..., 0] * outgoing[..., 1] - incoming[..., 1] * outgoing[..., 0]
    dot = torch.sum(incoming * outgoing, dim=-1)
    turning_angle = torch.atan2(torch.abs(cross), dot)  # [0, pi]
    local_arc_length = 0.5 * (incoming_length + outgoing_length)
    return turning_angle / local_arc_length.clamp_min(eps)


def _match_cost_map_batch(occ, batch_size, device):
    """将 (B_map, C, D, H, W) 代价图对齐到轨迹 batch。"""
    b_map = occ.shape[0]
    if b_map == batch_size:
        return occ
    if b_map == 1:
        return occ.expand(batch_size, -1, -1, -1, -1)
    if batch_size % b_map == 0:
        repeats_per_map = batch_size // b_map
        map_idx = torch.arange(batch_size, device=device) // repeats_per_map
        return occ[map_idx]

    raise ValueError(
        f"occupancy-map batch ({b_map}) cannot be aligned with trajectory batch "
        f"({batch_size})"
    )


def _sample_cost_map_on_dense_trajectory(traj_xy, occupancy_map, map_info, device):
    """
    根据 xy 切向计算 yaw，并在 (x, y, yaw) 代价图上可微采样。

    Returns:
        x, y, yaw, occ_vals，形状均为 (B, N)。
    """
    batch_size, num_points, _ = traj_xy.shape
    origin = map_info['origin']
    resolution = float(map_info['resolution'])
    W, H, D = map_info['size']

    if resolution <= 0.0:
        raise ValueError(f"map resolution must be positive, got {resolution}")
    if min(W, H, D) < 2:
        raise ValueError(f"map dimensions must all be at least 2, got {(W, H, D)}")

    x = traj_xy[:, :, 0]
    y = traj_xy[:, :, 1]
    yaw = compute_theta_from_xy(traj_xy).to(device)

    x_idx = torch.clamp((x - origin[0]) / resolution, 0.0, float(W - 1))
    y_idx = torch.clamp((y - origin[1]) / resolution, 0.0, float(H - 1))
    yaw_rel = torch.remainder(yaw - origin[2], 2.0 * np.pi)
    yaw_idx = torch.clamp(yaw_rel / (2.0 * np.pi / D), 0.0, float(D - 1))

    grid = torch.stack([
        (x_idx / (W - 1)) * 2.0 - 1.0,
        (y_idx / (H - 1)) * 2.0 - 1.0,
        (yaw_idx / (D - 1)) * 2.0 - 1.0,
    ], dim=-1).view(batch_size, num_points, 1, 1, 3)

    occ = torch.as_tensor(occupancy_map, dtype=traj_xy.dtype, device=device)

    # 兼容 (D,H,W)/(H,W,D) 以及它们的 batch 形式。
    if occ.ndim == 3:
        if occ.shape[-1] == D and occ.shape[0] == H and occ.shape[1] == W:
            occ = occ.permute(2, 0, 1).contiguous()
        if tuple(occ.shape) != (D, H, W):
            raise ValueError(
                f"occupancy_map spatial shape must be (D,H,W) or (H,W,D), got "
                f"{tuple(occ.shape)}"
            )
        occ = occ.unsqueeze(0).unsqueeze(0)
    elif occ.ndim == 4:
        if occ.shape[-1] == D and occ.shape[1] == H and occ.shape[2] == W:
            occ = occ.permute(0, 3, 1, 2).contiguous()
        if tuple(occ.shape[1:]) != (D, H, W):
            raise ValueError(
                f"batched occupancy_map spatial shape must be (D,H,W) or (H,W,D), "
                f"got {tuple(occ.shape[1:])}"
            )
        occ = occ.unsqueeze(1)
    elif occ.ndim == 5:
        if occ.shape[1] != 1 or tuple(occ.shape[2:]) != (D, H, W):
            raise ValueError(
                "5D occupancy_map must have shape (B_map,1,D,H,W), got "
                f"{tuple(occ.shape)}"
            )
    else:
        raise ValueError(f"Unsupported occupancy_map shape: {tuple(occ.shape)}")

    occ = _match_cost_map_batch(occ, batch_size, device)
    occ_vals = F.grid_sample(
        occ,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )
    occ_vals = occ_vals[:, 0, :, 0, 0]
    return x, y, yaw, occ_vals


def _pose_yaw_for_batch(pose, batch_size, dtype, device, name):
    """提取 pose yaw，并兼容单值、batch 以及每场景多轨迹的 batch。"""
    pose = torch.as_tensor(pose, dtype=dtype, device=device)
    if pose.ndim == 1:
        if pose.numel() < 3:
            raise ValueError(f"{name} must contain at least (x, y, yaw)")
        return pose[2]
    if pose.ndim != 2 or pose.shape[1] < 3:
        raise ValueError(f"{name} must have shape (3,) or (B,3), got {tuple(pose.shape)}")

    yaw = pose[:, 2]
    if yaw.shape[0] == batch_size:
        return yaw
    if batch_size % yaw.shape[0] == 0:
        return yaw.repeat_interleave(batch_size // yaw.shape[0])
    raise ValueError(
        f"{name} batch ({yaw.shape[0]}) cannot be aligned with trajectory batch "
        f"({batch_size})"
    )


def cost_on_dense_trajectory_tail_risk(
    trajectory,
    start_pose,
    goal_pose,
    occupancy_map,
    map_info,
    device='cpu',
    return_per_sample=False,
    *,
    tail_ratio=SAFETY_COST_CONFIG.tail_ratio,
    curvature_tail_ratio=SAFETY_COST_CONFIG.curvature_tail_ratio,
    out_of_bound_tail_ratio=SAFETY_COST_CONFIG.out_of_bound_tail_ratio,
    d_safe=SAFETY_COST_CONFIG.d_safe_meters,
    alpha=SAFETY_COST_CONFIG.softplus_alpha,
    curvature_limit=SAFETY_COST_CONFIG.curvature_limit,
    boundary_safe_pixels=SAFETY_COST_CONFIG.boundary_safe_pixels,
    obstacle_weight=SAFETY_COST_CONFIG.obstacle_weight,
    curvature_weight=SAFETY_COST_CONFIG.curvature_weight,
    out_of_bound_weight=SAFETY_COST_CONFIG.out_of_bound_weight,
    endpoint_weight=SAFETY_COST_CONFIG.endpoint_weight,
    quality_weight=SAFETY_COST_CONFIG.quality_weight,
    quality_term=SAFETY_COST_CONFIG.quality_term,
    return_components=False,
):
    """
    面向“整条轨迹必须可行”的稠密轨迹 tail-risk cost。

    与 :func:`cost_on_dense_trajectory` 的主要区别：

    1. 障碍、曲率和越界量均按物理阈值无量纲化；
    2. 安全、曲率和越界分别使用自己的 tail 比例，避免为覆盖不稳定点
       而增大的安全 tail 同时把曲率过度平滑；
    3. 曲率使用转角/局部弧长及线性相对超限，能识别共线掉头；
    4. 可行性项使用显式权重组合，轨迹质量只作为小权重正则项。

    ``occupancy_map`` 延续原函数的语义：值越大表示安全余度越大，
    低于 ``d_safe`` 视为风险。

    Args:
        trajectory: (N,2/3) 或 (B,N,2/3) 稠密轨迹。传入的 yaw 不使用，
            yaw 由 xy 切向可微计算。
        tail_ratio: 安全余度使用的最坏点比例，由共享安全配置给出。
        curvature_tail_ratio: 曲率使用的最坏点比例。
        out_of_bound_tail_ratio: 越界距离使用的最坏点比例。
        quality_term: ``'jerk'``、``'smoothness'`` 或 ``'none'``。
        return_components: 为 True 时返回 ``(cost, components)`` 便于诊断。

    Returns:
        默认返回 batch 均值标量；``return_per_sample=True`` 时返回 (B,)。
    """
    if d_safe <= 0.0:
        raise ValueError(f"d_safe must be positive, got {d_safe}")
    tail_ratios = {
        'tail_ratio': tail_ratio,
        'curvature_tail_ratio': curvature_tail_ratio,
        'out_of_bound_tail_ratio': out_of_bound_tail_ratio,
    }
    for name, ratio in tail_ratios.items():
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"{name} must be in (0, 1], got {ratio}")
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if curvature_limit <= 0.0:
        raise ValueError(f"curvature_limit must be positive, got {curvature_limit}")
    if boundary_safe_pixels < 0:
        raise ValueError(
            f"boundary_safe_pixels must be non-negative, got {boundary_safe_pixels}"
        )
    feasibility_weights = {
        'obstacle_weight': obstacle_weight,
        'curvature_weight': curvature_weight,
        'out_of_bound_weight': out_of_bound_weight,
        'endpoint_weight': endpoint_weight,
    }
    for name, weight in feasibility_weights.items():
        if weight < 0.0:
            raise ValueError(f"{name} must be non-negative, got {weight}")
    if quality_weight < 0.0:
        raise ValueError(f"quality_weight must be non-negative, got {quality_weight}")
    if quality_term not in {'jerk', 'smoothness', 'none'}:
        raise ValueError(
            "quality_term must be one of {'jerk', 'smoothness', 'none'}, "
            f"got {quality_term!r}"
        )

    if not torch.is_tensor(trajectory):
        trajectory = torch.tensor(trajectory, dtype=torch.float32, device=device)
    else:
        trajectory = trajectory.to(device=device, dtype=torch.float32)
    trajectory = torch.nan_to_num(
        trajectory,
        nan=0.0,
        posinf=MAP_HALF_EXTENT,
        neginf=-MAP_HALF_EXTENT,
    )

    if trajectory.ndim == 2:
        trajectory = trajectory.unsqueeze(0)
    if trajectory.ndim != 3 or trajectory.shape[-1] < 2:
        raise ValueError(
            f"trajectory must have shape (N,2/3) or (B,N,2/3), got "
            f"{tuple(trajectory.shape)}"
        )
    if trajectory.shape[1] < 3:
        raise ValueError("trajectory must contain at least three points")

    traj_xy = trajectory[:, :, :2]
    batch_size = traj_xy.shape[0]
    x, y, yaw, occ_vals = _sample_cost_map_on_dense_trajectory(
        traj_xy, occupancy_map, map_info, device
    )

    # 1) 无量纲障碍/安全余度违规。
    obs_violation = F.softplus(alpha * (d_safe - occ_vals)) / (alpha * d_safe)
    obstacle_cost = top_tail_mean(obs_violation, tail_ratio)

    # 2) 无量纲线性曲率超限。转角定义避免外接圆公式将共线掉头误判为零曲率。
    geom_curvature = discrete_turning_curvature(traj_xy)
    curvature_violation = F.relu(geom_curvature / curvature_limit - 1.0)
    curvature_cost = top_tail_mean(curvature_violation, curvature_tail_ratio)

    # 3) 越界距离按一个地图像素的物理长度归一化。
    origin = map_info['origin']
    resolution = float(map_info['resolution'])
    W, H, _ = map_info['size']
    x_min = float(origin[0]) + boundary_safe_pixels * resolution
    x_max = float(origin[0] + (W - 1) * resolution) - boundary_safe_pixels * resolution
    y_min = float(origin[1]) + boundary_safe_pixels * resolution
    y_max = float(origin[1] + (H - 1) * resolution) - boundary_safe_pixels * resolution
    out_violation = (
        F.relu(x_min - x)
        + F.relu(x - x_max)
        + F.relu(y_min - y)
        + F.relu(y - y_max)
    ) / resolution
    out_of_bound_cost = top_tail_mean(out_violation, out_of_bound_tail_ratio)

    # 4) endpoint yaw 按最大角距 pi 归一化，并使用线性误差。
    start_yaw = _pose_yaw_for_batch(
        start_pose, batch_size, trajectory.dtype, device, 'start_pose'
    )
    goal_yaw = _pose_yaw_for_batch(
        goal_pose, batch_size, trajectory.dtype, device, 'goal_pose'
    )
    yaw_diff_start = torch.atan2(
        torch.sin(yaw[:, 0] - start_yaw), torch.cos(yaw[:, 0] - start_yaw)
    )
    yaw_diff_end = torch.atan2(
        torch.sin(yaw[:, -1] - goal_yaw), torch.cos(yaw[:, -1] - goal_yaw)
    )
    endpoint_cost = (torch.abs(yaw_diff_start) + torch.abs(yaw_diff_end)) / np.pi

    weighted_obstacle_cost = obstacle_weight * obstacle_cost
    weighted_curvature_cost = curvature_weight * curvature_cost
    weighted_out_of_bound_cost = out_of_bound_weight * out_of_bound_cost
    weighted_endpoint_cost = endpoint_weight * endpoint_cost
    feasible_cost = (
        weighted_obstacle_cost
        + weighted_curvature_cost
        + weighted_out_of_bound_cost
        + weighted_endpoint_cost
    )

    # 质量项不当作硬约束；它仅用小权重打破可行轨迹之间的平局。
    dx = x[:, 1:] - x[:, :-1]
    dy = y[:, 1:] - y[:, :-1]
    dyaw = torch.atan2(
        torch.sin(yaw[:, 1:] - yaw[:, :-1]),
        torch.cos(yaw[:, 1:] - yaw[:, :-1]),
    )
    ddx = dx[:, 1:] - dx[:, :-1]
    ddy = dy[:, 1:] - dy[:, :-1]
    ddyaw = torch.atan2(
        torch.sin(dyaw[:, 1:] - dyaw[:, :-1]),
        torch.cos(dyaw[:, 1:] - dyaw[:, :-1]),
    )
    smoothness_cost = torch.mean(ddx ** 2 + ddy ** 2 + ddyaw ** 2, dim=1)

    if trajectory.shape[1] >= 4:
        jerk_x = ddx[:, 1:] - ddx[:, :-1]
        jerk_y = ddy[:, 1:] - ddy[:, :-1]
        jerk_yaw = torch.atan2(
            torch.sin(ddyaw[:, 1:] - ddyaw[:, :-1]),
            torch.cos(ddyaw[:, 1:] - ddyaw[:, :-1]),
        )
        jerk_cost = torch.mean(jerk_x ** 2 + jerk_y ** 2 + jerk_yaw ** 2, dim=1)
    else:
        jerk_cost = torch.zeros_like(feasible_cost)

    if quality_term == 'jerk':
        quality_cost = jerk_cost
    elif quality_term == 'smoothness':
        quality_cost = smoothness_cost
    else:
        quality_cost = torch.zeros_like(feasible_cost)

    total_cost = feasible_cost + quality_weight * quality_cost
    components = {
        'obstacle': obstacle_cost,
        'curvature': curvature_cost,
        'curvature_max': torch.max(geom_curvature, dim=1).values,
        'out_of_bound': out_of_bound_cost,
        'endpoint': endpoint_cost,
        'weighted_obstacle': weighted_obstacle_cost,
        'weighted_curvature': weighted_curvature_cost,
        'weighted_out_of_bound': weighted_out_of_bound_cost,
        'weighted_endpoint': weighted_endpoint_cost,
        'feasible': feasible_cost,
        'smoothness': smoothness_cost,
        'jerk': jerk_cost,
        'quality': quality_cost,
        'total': total_cost,
    }

    if return_per_sample:
        result = total_cost
        result_components = components
    else:
        result = total_cost.mean()
        result_components = {name: value.mean() for name, value in components.items()}

    if return_components:
        return result, result_components
    return result

def cost_on_dense_trajectory_phr_alm(
    trajectory,
    start_pose,
    goal_pose,
    occupancy_map,
    map_info,
    control_points=None,
    lambda_ineq=None,
    lambda_eq=None,
    mu=1.0,
    update_dual=False,
    mu_growth=1.5,
    mu_max=100.0,
    curvature_limit=1.4,
    enable_curvature_constraint=False,
    enable_nonholonomic_constraint=True,
    enable_control_point_bounds_constraint=False,
    path_length_weight=0e-1,
    dual_clip=1e6,
    device='cpu'
):
        """
        基于 `cost_on_dense_trajectory()` 的 PHR-ALM 版本。

        说明：
        - 目标函数包含（smoothness + jerk + trajectory length）。
        - 将障碍约束、端点航向约束（以及可选曲率/控制点边界/非完整约束）转为 ALM 约束项：
            g_obs <= 0, h_yaw = 0, [可选] g_curv <= 0, [可选] g_cp_box <= 0, [可选] h_nonholo = 0
        - 支持在函数内部更新 `lambda` 和 `mu`（`update_dual=True`）。

        Returns:
            total_loss,
            terms(dict),
            lambda_ineq,
            lambda_eq,
            mu
        """
        if lambda_ineq is None:
            lambda_ineq = {}
        if lambda_eq is None:
            lambda_eq = {}

        if not torch.is_tensor(trajectory):
            trajectory = torch.tensor(trajectory, dtype=torch.float32, device=device)
        else:
            trajectory = trajectory.to(device)
        trajectory = torch.nan_to_num(
            trajectory,
            nan=0.0,
            posinf=MAP_HALF_EXTENT,
            neginf=-MAP_HALF_EXTENT,
        )
        if trajectory.ndim == 2:
            trajectory = trajectory.unsqueeze(0)

        B, N, _ = trajectory.shape
        traj_xy = trajectory[:, :, :2]
        yaw = compute_theta_from_xy(traj_xy).to(device)
        trajectory = torch.cat([traj_xy, yaw.unsqueeze(-1)], dim=-1)

        x = trajectory[:, :, 0]
        y = trajectory[:, :, 1]

        origin = map_info['origin']
        resolution = map_info['resolution']
        W, H, D = map_info['size']

        x_idx_f = torch.clamp((x - origin[0]) / resolution, 0.0, float(W - 1))
        y_idx_f = torch.clamp((y - origin[1]) / resolution, 0.0, float(H - 1))
        yaw_rel = torch.remainder(yaw - origin[2], 2 * np.pi)
        yaw_idx_f = torch.clamp(yaw_rel / (2 * np.pi / D), 0.0, float(D - 1))

        x_norm = (x_idx_f / (W - 1)) * 2.0 - 1.0
        y_norm = (y_idx_f / (H - 1)) * 2.0 - 1.0
        z_norm = (yaw_idx_f / (D - 1)) * 2.0 - 1.0
        grid = torch.stack([x_norm, y_norm, z_norm], dim=-1).view(B, N, 1, 1, 3)

        occ = occupancy_map.to(device)
        if occ.ndim == 3:
            if occ.shape[-1] == D and occ.shape[0] == H and occ.shape[1] == W:
                occ = occ.permute(2, 0, 1).contiguous()
        elif occ.ndim == 4:
            if occ.shape[-1] == D and occ.shape[1] == H and occ.shape[2] == W:
                occ = occ.permute(0, 3, 1, 2).contiguous()

        if occ.ndim == 3:
            occ = occ.unsqueeze(0).unsqueeze(0)
            if B > 1:
                occ = occ.expand(B, -1, -1, -1, -1)
        elif occ.ndim == 4:
            b_map = occ.shape[0]
            if b_map == B:
                occ = occ.unsqueeze(1)
            elif b_map == 1:
                occ = occ.unsqueeze(1).expand(B, -1, -1, -1, -1)
            elif B % b_map == 0:
                k = B // b_map
                map_idx = torch.arange(B, device=device) // k
                occ = occ[map_idx].unsqueeze(1)
            else:
                map_idx = torch.linspace(0, b_map - 1, steps=B, device=device).long()
                occ = occ[map_idx].unsqueeze(1)
        elif occ.ndim == 5:
            b_map = occ.shape[0]
            if b_map == B:
                pass
            elif b_map == 1:
                occ = occ.expand(B, -1, -1, -1, -1)
            elif B % b_map == 0:
                k = B // b_map
                map_idx = torch.arange(B, device=device) // k
                occ = occ[map_idx]
            else:
                map_idx = torch.linspace(0, b_map - 1, steps=B, device=device).long()
                occ = occ[map_idx]
        else:
            raise ValueError(f"Unsupported occupancy_map shape: {tuple(occ.shape)}")

        occ_vals = F.grid_sample(occ, grid, mode='bilinear', padding_mode='border', align_corners=True)
        occ_vals = occ_vals.squeeze(-1).squeeze(-1).squeeze(1)

        # 采样到的占据值
        d_safe = 0.15
        g_obs = d_safe - occ_vals  # 约束 occ_vals >= d_safe  等价于 g_obs <= 0

        dx = x[:, 1:] - x[:, :-1]
        dy = y[:, 1:] - y[:, :-1]
        dyaw = torch.atan2(torch.sin(yaw[:, 1:] - yaw[:, :-1]), torch.cos(yaw[:, 1:] - yaw[:, :-1]))
        ddx = dx[:, 1:] - dx[:, :-1]
        ddy = dy[:, 1:] - dy[:, :-1]
        ddyaw = torch.atan2(torch.sin(dyaw[:, 1:] - dyaw[:, :-1]), torch.cos(dyaw[:, 1:] - dyaw[:, :-1]))
        smoothness_cost = torch.mean(ddx**2 + ddy**2 + ddyaw**2, dim=1)

        ddx_full = torch.zeros_like(x)
        ddy_full = torch.zeros_like(y)
        ddyaw_full = torch.zeros_like(yaw)
        ddx_full[:, 2:] = ddx
        ddy_full[:, 2:] = ddy
        ddyaw_full[:, 2:] = ddyaw
        jerk_x = ddx_full[:, 2:] - ddx_full[:, 1:-1]
        jerk_y = ddy_full[:, 2:] - ddy_full[:, 1:-1]
        jerk_yaw = ddyaw_full[:, 2:] - ddyaw_full[:, 1:-1]
        jerk_cost = torch.mean(jerk_x**2 + jerk_y**2 + 1e3 * jerk_yaw**2, dim=1)

        # 轨迹总长度目标：sum ||p_{i+1} - p_i||
        seg_len = torch.sqrt(dx**2 + dy**2 + 1e-12)  # (B, N-1)
        path_length_cost = torch.sum(seg_len, dim=1)  # (B,)

        weights = {
            'smoothness': 0e2,
            'jerk': 1e2,
            'path_length': path_length_weight,
        }
        objective_cost = torch.mean(
            weights['smoothness'] * smoothness_cost +
            weights['jerk'] * jerk_cost +
            weights['path_length'] * path_length_cost
        )

        # constraints for PHR-ALM
        p1 = traj_xy[:, :-2, :]
        p2 = traj_xy[:, 1:-1, :]
        p3 = traj_xy[:, 2:, :]
        v1 = p2 - p1
        v2 = p3 - p2
        chord = p3 - p1
        a = torch.sqrt(torch.sum(v1 * v1, dim=-1) + 1e-12)
        b = torch.sqrt(torch.sum(v2 * v2, dim=-1) + 1e-12)
        c = torch.sqrt(torch.sum(chord * chord, dim=-1) + 1e-12)
        cross = torch.abs(v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0])
        geom_curvature = 2.0 * cross / (a * b * c + 1e-12)
        g_curv = geom_curvature - curvature_limit
        if not enable_curvature_constraint:
            g_curv = torch.zeros_like(g_curv)

        # 控制点边界硬约束（可选）：xmin<=x<=xmax, ymin<=y<=ymax
        g_cp_box = None
        g_safe = W // 20  # 安全边界，控制点必须在地图内至少 g_safe 个像素的范围内
        if enable_control_point_bounds_constraint and control_points is not None:
            if not torch.is_tensor(control_points):
                control_points = torch.tensor(control_points, dtype=torch.float32, device=device)
            else:
                control_points = control_points.to(device)
            if control_points.ndim == 2:
                control_points = control_points.unsqueeze(0)

            cp_x = control_points[:, :, 0]
            cp_y = control_points[:, :, 1]

            x_min = float(origin[0]) + g_safe * resolution
            x_max = float(origin[0] + (W - 1) * resolution) - g_safe * resolution
            y_min = float(origin[1]) + g_safe * resolution
            y_max = float(origin[1] + (H - 1) * resolution) - g_safe * resolution

            g_cp_x_low = x_min - cp_x
            g_cp_x_high = cp_x - x_max
            g_cp_y_low = y_min - cp_y
            g_cp_y_high = cp_y - y_max
            g_cp_box = torch.cat([g_cp_x_low, g_cp_x_high, g_cp_y_low, g_cp_y_high], dim=1)

        if torch.is_tensor(start_pose):
            start_pose = start_pose.to(device)
        else:
            start_pose = torch.tensor(start_pose, dtype=torch.float32, device=device)
        if torch.is_tensor(goal_pose):
            goal_pose = goal_pose.to(device)
        else:
            goal_pose = torch.tensor(goal_pose, dtype=torch.float32, device=device)

        start_yaw = start_pose[2] if start_pose.ndim == 1 else start_pose[:, 2]
        goal_yaw = goal_pose[2] if goal_pose.ndim == 1 else goal_pose[:, 2]
        h_yaw_start = torch.atan2(torch.sin(yaw[:, 0] - start_yaw), torch.cos(yaw[:, 0] - start_yaw))
        h_yaw_end = torch.atan2(torch.sin(yaw[:, -1] - goal_yaw), torch.cos(yaw[:, -1] - goal_yaw))
        h_yaw = torch.stack([h_yaw_start, h_yaw_end], dim=-1)

        # 差速车非完整约束（无需时间参数化）：dx*sin(theta)-dy*cos(theta)=0
        if enable_nonholonomic_constraint:
            yaw_seg = yaw[:, :-1]  # 与 dx,dy 对齐
            h_nonholo = dx * torch.sin(yaw_seg) - dy * torch.cos(yaw_seg)  # (B, N-1)
        else:
            h_nonholo = torch.zeros_like(dx)

        if 'obstacle_clearance' not in lambda_ineq or lambda_ineq['obstacle_clearance'].shape != g_obs.shape:
            lambda_ineq['obstacle_clearance'] = torch.zeros_like(g_obs, device=device)
        if 'curvature' not in lambda_ineq or lambda_ineq['curvature'].shape != g_curv.shape:
            lambda_ineq['curvature'] = torch.zeros_like(g_curv, device=device)
        if g_cp_box is not None:
            if 'control_point_bounds' not in lambda_ineq or lambda_ineq['control_point_bounds'].shape != g_cp_box.shape:
                lambda_ineq['control_point_bounds'] = torch.zeros_like(g_cp_box, device=device)
        if 'yaw_endpoint' not in lambda_eq or lambda_eq['yaw_endpoint'].shape != h_yaw.shape:
            lambda_eq['yaw_endpoint'] = torch.zeros_like(h_yaw, device=device)
        if 'non_holo' not in lambda_eq or lambda_eq['non_holo'].shape != h_nonholo.shape:
            lambda_eq['non_holo'] = torch.zeros_like(h_nonholo, device=device)

        lam_obs = lambda_ineq['obstacle_clearance'].detach()
        lam_ineq = lambda_ineq['curvature'].detach()
        lam_cp = lambda_ineq['control_point_bounds'].detach() if g_cp_box is not None else None
        lam_eq = lambda_eq['yaw_endpoint'].detach()
        lam_eq_nonholo = lambda_eq['non_holo'].detach()
        eps = 1e-12

        active_obs = torch.relu(lam_obs + mu * g_obs)
        active = torch.relu(lam_ineq + mu * g_curv)
        active_cp = torch.relu(lam_cp + mu * g_cp_box) if g_cp_box is not None else None
        phr_penalty = (
            torch.sum((active_obs ** 2) / (2.0 * mu + eps)) +
            torch.sum((active ** 2) / (2.0 * mu + eps))
        )
        if active_cp is not None:
            phr_penalty = phr_penalty + torch.sum((active_cp ** 2) / (2.0 * mu + eps))
        eq_penalty = (
            torch.sum(lam_eq * h_yaw + 0.5 * mu * h_yaw ** 2) +
            torch.sum(lam_eq_nonholo * h_nonholo + 0.5 * mu * h_nonholo ** 2)
        )

        total_loss = objective_cost + phr_penalty + eq_penalty

        if update_dual:
            with torch.no_grad():
                g_curv_safe = torch.nan_to_num(g_curv.detach(), nan=0.0, posinf=1e3, neginf=-1e3)
                g_obs_safe = torch.nan_to_num(g_obs.detach(), nan=0.0, posinf=1e3, neginf=-1e3)
                if g_cp_box is not None:
                    g_cp_safe = torch.nan_to_num(g_cp_box.detach(), nan=0.0, posinf=1e3, neginf=-1e3)
                h_yaw_safe = torch.nan_to_num(h_yaw.detach(), nan=0.0, posinf=1e3, neginf=-1e3)
                h_nonholo_safe = torch.nan_to_num(h_nonholo.detach(), nan=0.0, posinf=1e3, neginf=-1e3)
                lambda_ineq['obstacle_clearance'] = torch.relu(lambda_ineq['obstacle_clearance'] + mu * g_obs_safe)
                lambda_ineq['obstacle_clearance'] = torch.clamp(lambda_ineq['obstacle_clearance'], 0.0, dual_clip)
                lambda_ineq['curvature'] = torch.relu(lambda_ineq['curvature'] + mu * g_curv_safe)
                lambda_ineq['curvature'] = torch.clamp(lambda_ineq['curvature'], 0.0, dual_clip)
                if g_cp_box is not None:
                    lambda_ineq['control_point_bounds'] = torch.relu(lambda_ineq['control_point_bounds'] + mu * g_cp_safe)
                    lambda_ineq['control_point_bounds'] = torch.clamp(lambda_ineq['control_point_bounds'], 0.0, dual_clip)
                lambda_eq['yaw_endpoint'] = lambda_eq['yaw_endpoint'] + mu * h_yaw_safe
                lambda_eq['yaw_endpoint'] = torch.clamp(lambda_eq['yaw_endpoint'], -dual_clip, dual_clip)
                lambda_eq['non_holo'] = lambda_eq['non_holo'] + mu * h_nonholo_safe
                lambda_eq['non_holo'] = torch.clamp(lambda_eq['non_holo'], -dual_clip, dual_clip)
                mu = min(float(mu) * float(mu_growth), float(mu_max))

        terms = {
            'objective': objective_cost.detach(),
            'path_length': torch.mean(path_length_cost).detach(),
            'phr_penalty': phr_penalty.detach(),
            'eq_penalty': eq_penalty.detach(),
            'total_loss': total_loss.detach(),
            'mu': float(mu),
            'obstacle_violation_max': torch.max(torch.relu(g_obs.detach())).item(),
            'curvature_violation_max': torch.max(torch.relu(g_curv.detach())).item(),
            'control_point_bounds_violation_max': (
                torch.max(torch.relu(g_cp_box.detach())).item() if g_cp_box is not None else 0.0
            ),
            'yaw_eq_violation_max': torch.max(torch.abs(h_yaw.detach())).item(),
            'non_holo_eq_violation_max': torch.max(torch.abs(h_nonholo.detach())).item(),
        }
        return total_loss, terms, lambda_ineq, lambda_eq, mu

def visualize_terrain_trajectory(ax, trajectory, elev, nx, ny, nz, positions=None, yaws=None, traj_label='Trajectory', traj_color='y', traj_linestyle='-'):
    """可视化带有地形信息和轨迹的绘图"""
    # 确保张量转为numpy数组
    if isinstance(nx, torch.Tensor):
        elev = elev.cpu().numpy()
        nx = nx.cpu().numpy()
        ny = ny.cpu().numpy()
        nz = nz.cpu().numpy()
    
    # 创建地形高度图（使用法向量的Z分量）
    height_map = elev
    
    # 显示地形
    im = ax.imshow(height_map, cmap='terrain', extent=MAP_BOUNDS, origin='lower')
    plt.colorbar(im, ax=ax, label='Terrain Height')
    
    # 绘制密集轨迹（最醒目，线宽加粗，zorder最高，颜色可选深蓝/黑/自定义）
    ax.plot(trajectory[:, 0], trajectory[:, 1], color=(traj_color if traj_color != 'y' else '#0044cc'), linestyle=traj_linestyle, linewidth=4, label=traj_label, zorder=20)
    
    if positions is not None:
        # 只绘制控制点散点和虚线，不画yaw箭头
        ax.scatter(positions[:, 0], positions[:, 1], c='red', s=40, alpha=0.35, label='Control Points', edgecolors='none', zorder=5)
        ax.plot(positions[:, 0], positions[:, 1], 'r--', alpha=0.35, linewidth=1, zorder=4)
    # 控制点不再绘制yaw箭头
    
    # 在密集轨迹上绘制yaw箭头（更大更明显，间隔更小）
    if yaws is not None and trajectory is not None and len(yaws) == len(trajectory):
        arrow_gap = max(1, len(trajectory)//12)  # 箭头更密集
        for i in range(0, len(trajectory), arrow_gap):
            arrow_length = 0.0225 * MAP_CONFIG.size_meters
            dx = arrow_length * np.cos(yaws[i])
            dy = arrow_length * np.sin(yaws[i])
            ax.arrow(trajectory[i, 0], trajectory[i, 1], dx, dy,
                     head_width=0.13, head_length=0.18, fc='#00e0e0', ec='#00e0e0', alpha=0.95, linewidth=2, zorder=30)

    ax.set_title(f'{traj_label} on Terrain')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.axis('equal')
    ax.grid(True)
    ax.legend()

from ESDF3d_atpoint import compute_esdf_batch, query_is_unreachable_by_match_batch

_bspline_layer_cache = {}  # 全局缓存B样条层

def _default_phr_alm_constraints(traj_xy, start_pose, goal_pose, curvature_limit=1.4, device='cuda'):
    """
    默认约束构造（用于 PHR-ALM）：
      - 不等式约束: 曲率约束 g_curv <= 0
      - 等式约束: 起终点航向一致性 h_yaw = 0
    """
    yaw = compute_theta_from_xy(traj_xy).to(device)  # (B, N)

    p1 = traj_xy[:, :-2, :]
    p2 = traj_xy[:, 1:-1, :]
    p3 = traj_xy[:, 2:, :]
    v1 = p2 - p1
    v2 = p3 - p2
    chord = p3 - p1

    a = torch.sqrt(torch.sum(v1 * v1, dim=-1) + 1e-12)
    b = torch.sqrt(torch.sum(v2 * v2, dim=-1) + 1e-12)
    c = torch.sqrt(torch.sum(chord * chord, dim=-1) + 1e-12)
    cross = torch.abs(v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0])
    geom_curvature = 2.0 * cross / (a * b * c + 1e-12)

    g_curv = geom_curvature - curvature_limit  # <= 0

    if torch.is_tensor(start_pose):
        start_pose = start_pose.to(device)
    else:
        start_pose = torch.tensor(start_pose, dtype=torch.float32, device=device)
    if torch.is_tensor(goal_pose):
        goal_pose = goal_pose.to(device)
    else:
        goal_pose = torch.tensor(goal_pose, dtype=torch.float32, device=device)

    start_yaw = start_pose[2] if start_pose.ndim == 1 else start_pose[:, 2]
    goal_yaw = goal_pose[2] if goal_pose.ndim == 1 else goal_pose[:, 2]
    h_yaw_start = torch.atan2(torch.sin(yaw[:, 0] - start_yaw), torch.cos(yaw[:, 0] - start_yaw))
    h_yaw_end = torch.atan2(torch.sin(yaw[:, -1] - goal_yaw), torch.cos(yaw[:, -1] - goal_yaw))
    h_yaw = torch.stack([h_yaw_start, h_yaw_end], dim=-1)  # (B, 2)

    return {
        'ineq': {'curvature': g_curv},
        'eq': {'yaw_endpoint': h_yaw}
    }


def optimize_control_points_multistep(
    middle_control_points, 
    start_pose, 
    goal_pose, 
    stability_cost_map, 
    map_info,
    iterations=50,
    lr=0.1,
    grad_clip_norm=1.0,
    device='cuda',
    verbose=False
):
    """
    对控制点进行多步梯度优化（GPU优化版本）
    
    Args:
        middle_control_points: (B, num_middle_points, 2) 中间控制点（不包含起终点）
        start_pose: (B, 3) 或 (3,) 起点 [x, y, yaw]
        goal_pose: (B, 3) 或 (3,) 终点 [x, y, yaw]
        stability_cost_map: (D, H, W) 稳定性代价地图
        map_info: dict 包含地图信息
        iterations: 优化迭代次数
        lr: 学习率
        grad_clip_norm: 梯度裁剪范数
        device: 设备
        verbose: 是否打印优化过程
    
    Returns:
        optimized_middle_control_points: (B, num_middle_points, 2) 优化后的中间控制点
        cost_history: list 优化过程中的cost历史
    """
    B = middle_control_points.shape[0]
    num_middle_points = middle_control_points.shape[1]
    
    # 确保输入在正确的设备上
    middle_control_points = middle_control_points.to(device)
    if torch.is_tensor(start_pose):
        start_pose = start_pose.to(device)
    else:
        start_pose = torch.tensor(start_pose, dtype=torch.float32, device=device)
    if torch.is_tensor(goal_pose):
        goal_pose = goal_pose.to(device)
    else:
        goal_pose = torch.tensor(goal_pose, dtype=torch.float32, device=device)
    
    # 克隆控制点并设置为可训练
    middle_cp_opt = middle_control_points.clone().detach()
    middle_cp_opt.requires_grad_(True)
    
    # 提取起点和终点的x,y坐标
    if start_pose.ndim == 1:
        start_cp = start_pose[:2].unsqueeze(0).unsqueeze(0)  # (1, 1, 2)
        goal_cp = goal_pose[:2].unsqueeze(0).unsqueeze(0)    # (1, 1, 2)
    else:
        start_cp = start_pose[:, :2].unsqueeze(1)  # (B, 1, 2)
        goal_cp = goal_pose[:, :2].unsqueeze(1)    # (B, 1, 2)
    
    # 使用缓存的B样条层，避免重复创建
    cache_key = (num_middle_points + 2, 100, 3, device)
    if cache_key not in _bspline_layer_cache:
        _bspline_layer_cache[cache_key] = DifferentiableBSpline(
            num_control_points=num_middle_points + 2,
            num_output_points=100,
            degree=3
        ).to(device)
    bspline_layer = _bspline_layer_cache[cache_key]
    
    # 创建优化器
    # optimizer = torch.optim.SGD([middle_cp_opt], lr=lr, momentum=0.9)  # 用SGD替代Adam更快
    optimizer = torch.optim.AdamW([middle_cp_opt], lr=lr)
    
    # 记录优化历史
    cost_history = []
    
    # 提前拼接起终点，避免重复操作
    start_cp_expanded = start_cp.expand(B, -1, -1)
    goal_cp_expanded = goal_cp.expand(B, -1, -1)
    
    # 优化循环 - 使用torch.no_grad()包装不需要梯度的部分
    with torch.set_grad_enabled(True):
        for iter in range(iterations):
            optimizer.zero_grad()
            
            # 拼接完整控制点 (B, num_control_points, 2)
            control_points = torch.cat([
                start_cp_expanded,
                middle_cp_opt,
                goal_cp_expanded
            ], dim=1)
            
            # 使用B样条重建轨迹 (B, 100, 2)
            traj_xy = bspline_layer(control_points)
            
            # 计算cost
            # cost = cost_on_dense_trajectory(
            cost = cost_on_dense_trajectory_tail_risk(
                traj_xy, start_pose, goal_pose,
                stability_cost_map, map_info, device
            )
            
            # 记录cost (detach避免保存计算图)
            cost_history.append(cost.detach().item())
            
            # 反向传播
            cost.backward()
            
            # 梯度裁剪
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_([middle_cp_opt], grad_clip_norm)
            
            # 更新参数
            optimizer.step()
            
            if verbose and (iter % 10 == 0 or iter == iterations - 1):
                print(f"  Iter {iter}/{iterations}, Cost: {cost.item():.6f}")
    
    return middle_cp_opt.detach(), cost_history

def optimize_control_points_phr_alm(
    middle_control_points,
    start_pose,
    goal_pose,
    stability_cost_map,
    map_info,
    lambda_ineq=None,
    lambda_eq=None,
    mu=1.0,
    outer_iterations=8,
    inner_iterations=40,
    lr=0.01,
    mu_growth=1.5,
    mu_max=100.0,
    enable_curvature_constraint=False,
    enable_nonholonomic_constraint=True,
    enable_control_point_bounds_constraint=True,
    grad_clip_norm=1.0,
    device='cuda',
    verbose=False,
):
    """
    使用 PHR-ALM 的控制点优化接口（lambda/mu 作为输入输出）。

    Args:
        middle_control_points: (B, M, 2) 中间控制点
        start_pose: (B,3) 或 (3,)
        goal_pose: (B,3) 或 (3,)
        stability_cost_map: 稳定性代价地图
        map_info: 地图信息
        lambda_ineq: dict[str, Tensor]，不等式约束乘子（可为 None，自动初始化）
        lambda_eq: dict[str, Tensor]，等式约束乘子（可为 None，自动初始化）
        mu: ALM 罚参数（输入并在函数内更新）
        outer_iterations: ALM 外层轮数
        inner_iterations: 每轮 ALM 的内层梯度步数
        lr: 内层学习率
        mu_growth: 每轮外层的 mu 放大倍率
        mu_max: mu 上限
        enable_curvature_constraint: 是否启用曲率约束（默认 False，禁用）
        enable_nonholonomic_constraint: 是否启用差速车非完整约束（默认 True）
        enable_control_point_bounds_constraint: 是否启用控制点边界硬约束（默认 True）
        grad_clip_norm: 梯度裁剪
        device: 设备
        verbose: 打印日志

    Returns:
        optimized_middle_control_points,
        history,
        lambda_ineq,
        lambda_eq,
        mu
    """
    B = middle_control_points.shape[0]
    num_middle_points = middle_control_points.shape[1]

    middle_cp_opt = middle_control_points.to(device).clone().detach()
    middle_cp_opt.requires_grad_(True)

    if torch.is_tensor(start_pose):
        start_pose = start_pose.to(device)
    else:
        start_pose = torch.tensor(start_pose, dtype=torch.float32, device=device)
    if torch.is_tensor(goal_pose):
        goal_pose = goal_pose.to(device)
    else:
        goal_pose = torch.tensor(goal_pose, dtype=torch.float32, device=device)

    if start_pose.ndim == 1:
        start_cp = start_pose[:2].unsqueeze(0).unsqueeze(0)
        goal_cp = goal_pose[:2].unsqueeze(0).unsqueeze(0)
    else:
        start_cp = start_pose[:, :2].unsqueeze(1)
        goal_cp = goal_pose[:, :2].unsqueeze(1)

    start_cp_expanded = start_cp.expand(B, -1, -1)
    goal_cp_expanded = goal_cp.expand(B, -1, -1)

    cache_key = (num_middle_points + 2, 100, 3, device)
    if cache_key not in _bspline_layer_cache:
        _bspline_layer_cache[cache_key] = DifferentiableBSpline(
            num_control_points=num_middle_points + 2,
            num_output_points=100,
            degree=3
        ).to(device)
    bspline_layer = _bspline_layer_cache[cache_key]

    optimizer = torch.optim.AdamW([middle_cp_opt], lr=lr)

    if lambda_ineq is None:
        lambda_ineq = {}
    if lambda_eq is None:
        lambda_eq = {}

    history = []
    mu = float(mu)

    for outer_iter in range(outer_iterations):
        for inner_iter in range(inner_iterations):
            optimizer.zero_grad()

            control_points = torch.cat([
                start_cp_expanded,
                middle_cp_opt,
                goal_cp_expanded
            ], dim=1)
            traj_xy = bspline_layer(control_points)

            total_loss, terms, lambda_ineq, lambda_eq, _ = cost_on_dense_trajectory_phr_alm(
                traj_xy,
                start_pose,
                goal_pose,
                stability_cost_map,
                map_info,
                control_points=control_points,
                lambda_ineq=lambda_ineq,
                lambda_eq=lambda_eq,
                mu=mu,
                update_dual=False,
                mu_growth=mu_growth,
                mu_max=mu_max,
                enable_curvature_constraint=enable_curvature_constraint,
                enable_nonholonomic_constraint=enable_nonholonomic_constraint,
                enable_control_point_bounds_constraint=enable_control_point_bounds_constraint,
                device=device
            )
            total_loss.backward()

            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_([middle_cp_opt], grad_clip_norm)
            optimizer.step()

            history.append({
                'outer_iter': outer_iter,
                'inner_iter': inner_iter,
                'objective': float(terms['objective'].item()),
                'phr_penalty': float(terms['phr_penalty'].item()),
                'eq_penalty': float(terms['eq_penalty'].item()),
                'total_loss': float(terms['total_loss'].item()),
                'mu': float(mu),
                'obstacle_violation_max': float(terms['obstacle_violation_max']),
                'curvature_violation_max': float(terms['curvature_violation_max']),
                'control_point_bounds_violation_max': float(terms['control_point_bounds_violation_max']),
                'yaw_eq_violation_max': float(terms['yaw_eq_violation_max']),
                'non_holo_eq_violation_max': float(terms['non_holo_eq_violation_max']),
            })

        # ---- ALM 外层更新：在函数内部更新 lambda / mu ----
        with torch.no_grad():
            control_points = torch.cat([
                start_cp_expanded,
                middle_cp_opt,
                goal_cp_expanded
            ], dim=1)
            traj_xy = bspline_layer(control_points)
            _, _, lambda_ineq, lambda_eq, mu = cost_on_dense_trajectory_phr_alm(
                traj_xy,
                start_pose,
                goal_pose,
                stability_cost_map,
                map_info,
                control_points=control_points,
                lambda_ineq=lambda_ineq,
                lambda_eq=lambda_eq,
                mu=mu,
                update_dual=True,
                mu_growth=mu_growth,
                mu_max=mu_max,
                enable_curvature_constraint=enable_curvature_constraint,
                enable_nonholonomic_constraint=enable_nonholonomic_constraint,
                enable_control_point_bounds_constraint=enable_control_point_bounds_constraint,
                device=device
            )

        if verbose:
            last = history[-1]
            print(
                f"[PHR-ALM] outer={outer_iter+1}/{outer_iterations}, "
                f"total={last['total_loss']:.6f}, obj={last['objective']:.6f}, mu={mu:.4f}"
            )

    return middle_cp_opt.detach(), history, lambda_ineq, lambda_eq, mu

def generate_stability_cost_map(nx, ny, nz, map_info, device='cuda'):
    """
    阶段 A: 生成 (x, y, yaw) 稳定性代价地图.
    """
    map_size = map_info['size'] # (W, H, D)
    origin = map_info['origin'] # (x, y, yaw)
    resolution = map_info['resolution']
    
    W, H, D = map_size
    
    print(f"Generating stability cost map of size {W}x{H}x{D}...")
    
    # 1. 创建所有查询点的网格
    # a. 创建每个维度的坐标
    x_coords = torch.linspace(origin[0], origin[0] + (W-1)*resolution, W, device=device)
    y_coords = torch.linspace(origin[1], origin[1] + (H-1)*resolution, H, device=device)
    yaw_range = 2 * np.pi
    yaw_coords = torch.linspace(origin[2], origin[2] + yaw_range * (D-1)/D, D, device=device)
    
    # b. 使用 meshgrid 创建三维坐标网格
    grid_y, grid_x, grid_yaw = torch.meshgrid(y_coords, x_coords, yaw_coords, indexing='ij')
    
    # c. 将网格展平为查询点列表 (num_points, 3)
    queries = torch.stack([
        grid_x.flatten(),
        grid_y.flatten(),
        grid_yaw.flatten()
    ], dim=1)
    
    # 2. 批量计算稳定性 (这里可以利用你原来的批量处理函数)
    print(f"Evaluating stability for {queries.shape[0]} points...")
    # 注意：如果点太多导致显存爆炸，需要分块(chunk)处理
    # 这里采用外部分块调用 compute_esdf_batch，并兼容不同返回类型
    
    num_points = queries.shape[0]
    chunk_q = 12000  # 每次处理的查询点数，必要时减小（例如 2000 或 1000）
    capsize_parts = []
    # 保证 nx, ny, nz 在调用端设备上（compute_esdf_batch 内部可能也会移动）
    nx_dev = nx.to(device)
    ny_dev = ny.to(device)
    nz_dev = nz.to(device)
    
    # 使用 tqdm 显示分块进度
    total_chunks = (num_points + chunk_q - 1) // chunk_q
    for i in tqdm.tqdm(range(0, num_points, chunk_q), desc="compute_esdf_batch chunks", total=total_chunks):
        q_chunk = queries[i:i+chunk_q].to(device)
        try:
            res_chunk = compute_esdf_batch(nx_dev, ny_dev, nz_dev, q_chunk,
                                           resolution=resolution,
                                           origin=(origin[0], origin[1]),
                                           yaw_weight=SAFETY_COST_CONFIG.yaw_esdf_weight,
                                           search_radius=5.0,
                                           chunk_cells=1000,
                                           device=device)
        except RuntimeError as e:
            # GPU OOM 或其它错误，尝试清理并在 CPU 上重试（更慢但稳妥）
            print(f"Warning: compute_esdf_batch failed on GPU for chunk {i}-{i+chunk_q}, retrying on CPU: {e}")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            nx_cpu = nx_dev.cpu()
            ny_cpu = ny_dev.cpu()
            nz_cpu = nz_dev.cpu()
            q_chunk_cpu = q_chunk.cpu()
            res_chunk = compute_esdf_batch(nx_cpu, ny_cpu, nz_cpu, q_chunk_cpu,
                                           resolution=resolution,
                                           origin=(origin[0], origin[1]),
                                           yaw_weight=SAFETY_COST_CONFIG.yaw_esdf_weight,
                                           search_radius=5.0,
                                           chunk_cells=500,
                                           device='cpu')

        # 调用 compute_esdf_batch 后，兼容各种返回类型并优先提取每个条目的第一个值 (esdf_val)
        tensor_res = None
        if torch.is_tensor(res_chunk):
            tensor_res = res_chunk
        elif isinstance(res_chunk, np.ndarray):
            tensor_res = torch.from_numpy(res_chunk)
        elif isinstance(res_chunk, (list, tuple)):
            # 处理 compute_esdf_batch 返回的 list/tuple，每项通常为 (esdf_val, (i,j) or None, yaw_dist or None)
            if len(res_chunk) == 0:
                raise RuntimeError("compute_esdf_batch returned empty list/tuple")
            esdf_list = []
            for item in res_chunk:
                # 若 item 本身是 (esdf_val, ij, yaw_dist)
                if isinstance(item, (list, tuple)) and len(item) > 0:
                    v = item[0]
                else:
                    v = item
                if torch.is_tensor(v):
                    esdf_list.append(v.detach().cpu().float())
                elif isinstance(v, np.ndarray):
                    esdf_list.append(torch.from_numpy(v).float())
                else:
                    # 标量或其他可转为 float 的类型
                    try:
                        esdf_list.append(torch.tensor(float(v), dtype=torch.float32))
                    except Exception:
                        raise RuntimeError("Cannot convert element of compute_esdf_batch result to tensor: " + str(type(v)))
            # 将每个标量/标量tensor 堆叠为 1D tensor，长度应等于 q_chunk.shape[0]
            tensor_res = torch.stack(esdf_list).reshape(-1)
        elif isinstance(res_chunk, dict):
            # 先尝试常见字段名
            found = False
            for key in ('capsize', 'cap', 'esdf', 'distance', 'dist', 'value', 'values'):
                if key in res_chunk:
                    v = res_chunk[key]
                    if torch.is_tensor(v):
                        tensor_res = v
                    elif isinstance(v, np.ndarray):
                        tensor_res = torch.from_numpy(v)
                    found = True
                    break
            if not found:
                # 递归查找 dict 内的第一个 tensor/ndarray
                for v in res_chunk.values():
                    if torch.is_tensor(v):
                        tensor_res = v; break
                    if isinstance(v, np.ndarray):
                        tensor_res = torch.from_numpy(v); break
            if tensor_res is None:
                raise RuntimeError("compute_esdf_batch returned dict but contained no recognizable tensor/ndarray")
        else:
            raise RuntimeError("compute_esdf_batch returned unsupported type: " + str(type(res_chunk)))

        # 统一为 CPU 上的 float32 tensor，减少显存占用
        tensor_res = tensor_res.detach().cpu().float()

        # 尝试根据常见布局提取 capsize 值（第0通道或直接 1D 列表）
        expected_len = q_chunk.shape[0]
        if tensor_res.ndim == 1:
            cap_chunk = tensor_res.reshape(-1)
        elif tensor_res.ndim == 0:
            cap_chunk = tensor_res.reshape(-1).repeat(expected_len)[:expected_len]
        else:
            # 优先取最后一维的第0通道（例如 [..., C] -> [...,0]），否则展平
            try:
                if tensor_res.shape[-1] >= 1:
                    cap_chunk = tensor_res[..., 0].reshape(-1)
                else:
                    cap_chunk = tensor_res.reshape(-1)
            except Exception:
                cap_chunk = tensor_res.reshape(-1)

        # 保证长度与 q_chunk 匹配，否则截取或填充 Inf（后续会处理）
        if cap_chunk.numel() < expected_len:
            pad = torch.full((expected_len - cap_chunk.numel(),), float('inf'), dtype=cap_chunk.dtype)
            cap_chunk = torch.cat([cap_chunk, pad], dim=0)
        elif cap_chunk.numel() > expected_len:
            cap_chunk = cap_chunk[:expected_len]

        capsize_parts.append(cap_chunk)

    capsize_esdf_flat = torch.cat(capsize_parts, dim=0)  # (num_points,)

    # 如果有 inf（因填充），将其替换为一个很大的距离值
    if torch.isfinite(capsize_esdf_flat).all() is False:
        capsize_esdf_flat = torch.where(torch.isfinite(capsize_esdf_flat),
                                        capsize_esdf_flat,
                                        torch.full_like(capsize_esdf_flat, 1e3))

    # 恢复到原有成本计算流程
    d_safe = 0.
    kalpa = 0.6
    z = (-(capsize_esdf_flat - d_safe) / (kalpa + 1e-12))
    costs = torch.sigmoid(torch.clamp(z, min=-50.0, max=50.0))

    # 3. 将一维的成本列表重塑为三维地图 (D, H, W)
    cost_map = costs.reshape(H, W, D).permute(2, 0, 1) # reshape 成 (H, W, D)，然后 permute 成 (D, H, W)
    
    print("Stability cost map generated.")
    return cost_map

def _stability_grid_indices(x, y, yaw, yaw_bins=MAP_YAW_BINS):
    """Convert a world SE(2) pose to configured stability-map indices."""
    origin_x, origin_y, origin_yaw = MAP_CONFIG.cost_map_origin
    x_idx = int(np.floor((float(x) - origin_x) / MAP_RESOLUTION))
    y_idx = int(np.floor((float(y) - origin_y) / MAP_RESOLUTION))
    yaw_step = 2.0 * np.pi / int(yaw_bins)
    yaw_idx = int(np.floor((float(yaw) - origin_yaw) / yaw_step)) % int(yaw_bins)
    return x_idx, y_idx, yaw_idx


def check_trajectory_reachability(trajectory_points, yaw_values, yaw_stability):
    """
    使用与 data_clean.py 一致的方法检查轨迹点的可达性
    
    Args:
        trajectory_points: numpy array of shape (N, 2) - x, y coordinates
        yaw_values: numpy array of shape (N,) - yaw angles
        yaw_stability: torch tensor of shape (H, W, 36) - stability map
    
    Returns:
        capsize_mask: numpy array of shape (N,) - True for unreachable points
    """
    capsize_mask = []
    
    for i in range(len(trajectory_points)):
        x, y = trajectory_points[i]
        yaw = yaw_values[i]
        
        x_idx, y_idx, yaw_idx = _stability_grid_indices(
            x, y, yaw, yaw_stability.shape[2]
        )
        
        # 边界检查和稳定性判断
        if 0 <= y_idx < yaw_stability.shape[0] and 0 <= x_idx < yaw_stability.shape[1]:
            yaw_stability_value = yaw_stability[y_idx, x_idx, yaw_idx]
            is_unreachable = (yaw_stability_value == 0)
        else:
            is_unreachable = True  # 超出地图边界
            
        capsize_mask.append(is_unreachable)
    
    return np.array(capsize_mask, dtype=bool)


def generate_paths(model, map_input, start_point, goal_point, num_paths=5, diffusion_step=50,
                   reconstruct_trajectory=True, num_traj_points=100, solver='heun'):
    """
    生成完整路径（使用B样条控制点重建）- Rectified Flow版本
    
    Args:
        model: 训练好的PathDiffusionTransformer
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
            - 如果reconstruct_trajectory=False: N=26（含起终点的完整控制点）
    """
    model.eval()
    base_model = model.module if hasattr(model, 'module') else model
    coordinate_scale = float(base_model.coordinate_scale)
    if not np.isclose(coordinate_scale, MAP_HALF_EXTENT, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"模型 coordinate_scale={coordinate_scale}，"
            f"当前地图配置要求 {MAP_HALF_EXTENT}"
        )

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
        sampled_traj_xy = model.sample(
            map_input,
            start_normalized,
            goal_normalized,
            num_samples=num_paths,
            num_steps=diffusion_step,  # 使用新参数名
            solver=solver,  # 选择ODE求解器
            reconstruct_trajectory=reconstruct_trajectory,
            num_traj_points=num_traj_points
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


def extract_middle_control_points(full_control_points, expected_middle_points=24):
    """从含起终点的完整 B 样条控制点中提取网络输出的中间点。

    ``model.sample(..., reconstruct_trajectory=False)`` 返回的是
    ``[start, middle_1, ..., middle_24, goal]``，而不是单独的 24 个中间点。
    """
    if not torch.is_tensor(full_control_points):
        full_control_points = torch.as_tensor(full_control_points, dtype=torch.float32)
    if full_control_points.ndim != 2 or full_control_points.shape[1] < 2:
        raise ValueError(
            "full_control_points must have shape (num_control_points,2/3), got "
            f"{tuple(full_control_points.shape)}"
        )

    expected_full_points = expected_middle_points + 2
    if full_control_points.shape[0] != expected_full_points:
        raise ValueError(
            f"expected {expected_full_points} complete control points "
            f"(1+{expected_middle_points}+1), got {full_control_points.shape[0]}"
        )
    return full_control_points[1:-1, :2]

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- 0. 加载地形数据 ---
    # from dataLoader_uneven import UnevenPathDataLoader
    from dataLoader_dit import UnevenPathDataLoader
    # env_list = ['env000010']
    envNum = np.random.randint(0, 99)
    env_list = [f'env{envNum:06d}']
    dataFolder = str(MAP_CONFIG.dataset_root / 'val')
    # dataset = UnevenPathDataLoader(env_list, dataFolder)
    dataset = UnevenPathDataLoader(env_list, dataFolder, True)
    requested_path_index = 42
    path_index = min(requested_path_index, len(dataset) - 1)
    if path_index != requested_path_index:
        print(
            f"Requested path index {requested_path_index} exceeds dataset size "
            f"{len(dataset)}; using {path_index} instead."
        )
    sample = dataset[path_index]
    if sample is None:
        raise ValueError(f"Sample at index {path_index} is invalid.")
    print("Loaded path index:", path_index)
    
    nx = sample['map'][0, :, :].to(device)
    ny = sample['map'][1, :, :].to(device)
    nz = sample['map'][2, :, :].to(device)
    
    nz = torch.abs(nz)  # 使用法向量的绝对值

    # --- 1. 定义代价地图参数并生成地图 ---
    map_info = MAP_CONFIG.cost_map_info()
    map_size = map_info['size']
    resolution = map_info['resolution']
    origin = map_info['origin']

    # !! 核心步骤：生成稳定性代价地图 !!
    # stability_cost_map = generate_stability_cost_map(nx, ny, nz, map_info, device)
    stability_cost_map = sample['cost_map'].to(device)  # 使用预先计算的稳定性代价地图

    # --- 2. 定义初始轨迹控制点 ---
    # 原始样本中的控制点/轨迹命名为 initial_control_points_raw（更直观）
    initial_control_points_raw = sample['trajectory'].cpu().numpy()  # (x, y, yaw)

    # # --- 使用网络推理结果作为初始点 ---
    # 生成模型预测轨迹
    from dataLoader_uneven import get_encoder_input
    from eval_model_uneven import get_patch
    from transformer import Models
    # from vision_mamba import Models
    import os.path as osp
    import json
    
    best = True
    # best = False
    # stage = 1
    # epoch = 39
    stage = 1
    epoch = 24
    
    # =================== ODE求解器配置 ===================
    # 'pmf_onestep': 一步预测方法（快速）
    # 'euler': 一阶Euler方法（快速，但精度较低）
    # 'heun': 二阶Heun方法（较慢，但精度更高）
    solver = 'pmf_onestep'  # 快速的一步预测方法
    # solver = 'pmf_refined'
    # solver = 'euler'  # 可选：速度更快
    diffusion_step = 3
    # solver = 'heun'  # 推荐：精度更高
    # diffusion_step = 50
    # ==================================================

    modelFolder = 'data/sim'
    # modelFolder = 'data'
    # modelFolder = 'data/uneven_old'
    modelFile = osp.join(modelFolder, f'model_params.json')
    model_param = json.load(open(modelFile))
    
    from dit.Models import PathDiffusionTransformer
    
    model = PathDiffusionTransformer(**model_param['model_args'])
    _ = model.to(device)
    
    if best:
        checkpoint = torch.load(
            osp.join(modelFolder, f'stage{stage}_best_model.pth'),
            map_location=device,
        )
        print(f"Loaded best stage {stage} model.")
    else:
        checkpoint = torch.load(
            osp.join(modelFolder, f'checkpoint_stage{stage}_epoch_{epoch}.pth'),
            map_location=device,
        )
        print(f"Loaded stage {stage} model from epoch {epoch}.")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    trajectory = sample['trajectory'].cpu().numpy()  # (N, 3)
    goal_pos = trajectory[-1, :]  # 终点位置
    start_pos = trajectory[0, :]  # 起点位置
    
    normal_x = nx.cpu().numpy()
    normal_y = ny.cpu().numpy()
    normal_z = nz.cpu().numpy()
    
    encoder_input = torch.tensor(np.concatenate((
        normal_x[:, :, None],  # [H, W, 1]
        normal_y[:, :, None],  # [H, W, 1]
        normal_z[:, :, None]   # [H, W, 1]
    ), axis=2), dtype=torch.float32)  # [H, W, 3]

    predTrajs = generate_paths(model, 
                                  map_input=encoder_input.permute(2, 0, 1)[None, :].to(device),
                                  start_point=torch.tensor(start_pos).float().to(device),
                                  goal_point=torch.tensor(goal_pos).float().to(device),
                                  num_paths=1,  # 生成多条轨迹
                                  diffusion_step=diffusion_step,
                                  reconstruct_trajectory=False,  # 直接返回含起终点的完整控制点
                                  num_traj_points=100,  # 此模式下不使用
                                  solver=solver  # 传递求解器类型
                                 )  # (num_pred_paths, 26, 3) - 完整控制点
        
    
    predTraj = predTrajs[0]
    
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

    # --- 3. 初始化并运行优化器 ---
    B = 1  # batch size
    base_model = model.module if hasattr(model, 'module') else model
    # n_path_steps is the number of physical edges (25), so decoding produces
    # 26 complete control points and therefore 24 optimizable middle points.
    expected_middle_points = int(base_model.n_path_steps) - 1

    # sample(reconstruct_trajectory=False) 已经返回 [start, 24 middle, goal]。
    # 优化器的输入只能是中间24点，后续再拼接固定起终点。
    middle_control_points = extract_middle_control_points(
        pred_traj_t,
        expected_middle_points=expected_middle_points,
    ).unsqueeze(0).to(device).clone()  # (B,24,2)

    # Extract start/goal x,y correctly (start_t/goal_t have shape (1,3)) -> (1,2)
    start_cp = start_t[:, :2].detach() # 起点，取出坐标(x,y)
    goal_cp = goal_t[:, :2].detach()   # 终点，取出坐标(x,y)

    # Make middle control points trainable
    middle_control_points = middle_control_points.clone().detach()
    middle_control_points.requires_grad_(True)
        
    # -------------------------
    #      创建一次性 B 样条层
    # -------------------------
    bspline_layer = DifferentiableBSpline(
        num_control_points=middle_control_points.shape[1] + 2,  # 24 + 2 = 26
        num_output_points=100,
        degree=3
    ).to(device)
    
    # start_cp and goal_cp already have batch dim (1,2), so unsqueeze at dim=1 to get (1,1,2)
    control_points = torch.cat([
        start_cp.unsqueeze(1),  # (B,1,2)
        middle_control_points,   # (B,24,2)
        goal_cp.unsqueeze(1)     # (B,1,2)
    ], dim=1)  # (B,26,2)
    if control_points.shape[1] != expected_middle_points + 2:
        raise RuntimeError(
            f"invalid full control-point count: expected {expected_middle_points + 2}, "
            f"got {control_points.shape[1]}"
        )
    
    initial_traj = bspline_layer(control_points)  # (B,100,2)
    # compute yaw (theta) from xy and append as third dimension -> (B,100,3)
    init_xy = initial_traj
    init_yaw_vec = compute_theta_from_xy(init_xy[0]).to(init_xy.device)
    init_yaw = init_yaw_vec.unsqueeze(0).unsqueeze(2)  # (1, N, 1)
    initial_traj = torch.cat([init_xy, init_yaw], dim=2)
    # Save a numpy snapshot of the initial trajectory BEFORE optimization (remove batch dim)
    initial_traj_snapshot_np = initial_traj[0].detach().cpu().numpy()
    # Save a numpy snapshot of the initial control points BEFORE optimization
    initial_control_points_np = control_points[0].detach().cpu().numpy()  # (num_control_points, 2)
    # compute approximate yaw for control points for plotting arrows
    initial_control_yaws = compute_theta_from_xy(torch.tensor(initial_control_points_np)).detach().cpu().numpy()
    # Helper: ensure trajectories are (N,3) numpy arrays (x,y,yaw)
    def _ensure_traj_np(tr):
        arr = np.asarray(tr)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim == 2 and arr.shape[1] == 2:
            theta = compute_theta_from_xy(torch.tensor(arr)).detach().cpu().numpy()
            arr = np.column_stack([arr, theta])
        return arr
    
    # =================== 使用集成优化方法 ===================
    print("\n" + "="*100)
    print("Starting Multi-Step Optimization".center(100))
    print("="*100 + "\n")
    
    # 设置优化参数
    iterations = 300
    lr = 0.01
    grad_clip_norm = 1.0
    
    # 调用集成的多步优化方法
    optimized_middle_cp, cost_history = optimize_control_points_multistep(
        middle_control_points=middle_control_points,  # (B, num_middle_points, 2)
        start_pose=start_t[0],  # (3,) [x, y, yaw]
        goal_pose=goal_t[0],    # (3,) [x, y, yaw]
        stability_cost_map=stability_cost_map,
        map_info=map_info,
        iterations=iterations,
        lr=lr,
        grad_clip_norm=grad_clip_norm,
        device=device,
        verbose=True  # 打印优化过程
    )
    
    # optimized_middle_cp, history, lambda_ineq, lambda_eq, final_mu = optimize_control_points_phr_alm(
    #     middle_control_points=middle_control_points,  # (B, num_middle_points, 2)
    #     start_pose=start_t[0],  # (3,) [x, y, yaw]
    #     goal_pose=goal_t[0],    # (3,) [x, y, yaw]
    #     stability_cost_map=stability_cost_map,
    #     map_info=map_info,
    #     outer_iterations=100,
    #     inner_iterations=10,
    #     lr=lr,
    #     grad_clip_norm=grad_clip_norm,
    #     device=device,
    #     verbose=True  # 打印优化过程
    # )
    
    # 重建完整控制点（包含起终点）
    control_points = torch.cat([
        start_cp.unsqueeze(1),     # (1,1,2)
        optimized_middle_cp,       # (1, num_middle_points, 2) - 已经包含batch维度
        goal_cp.unsqueeze(1)       # (1,1,2)
    ], dim=1)  # (1, num_control_points, 2)
    
    # 重建优化后的轨迹
    optimized_traj = bspline_layer(control_points)  # (B,100,2)
    # append yaw to optimized trajectory
    opt_xy = optimized_traj
    opt_yaw_vec = compute_theta_from_xy(opt_xy[0]).to(opt_xy.device)
    opt_yaw = opt_yaw_vec.unsqueeze(0).unsqueeze(2)
    optimized_traj = torch.cat([opt_xy, opt_yaw], dim=2)  # (B,100,3)

    optimized_trajectory = optimized_traj[0, :, :2].detach().cpu().numpy()  # (100, 2)
    optimized_yaw_dense = compute_theta_from_xy(torch.tensor(optimized_trajectory)).detach().cpu().numpy()  # (100,)
    # Save a numpy snapshot of the optimized control points (26,2) and their yaws for plotting
    optimized_control_points_np = control_points[0].detach().cpu().numpy()
    optimized_control_yaws = compute_theta_from_xy(torch.tensor(optimized_control_points_np)).detach().cpu().numpy()

    # =================== 评估优化前后的轨迹质量 ===================
    print("\n" + "="*100)
    print("Trajectory Quality Evaluation: Before vs After Optimization".center(100))
    print("="*100 + "\n")
    
    # 计算 yaw_stability 地图（用于评估）
    from dataLoader_dit import compute_map_yaw_bins
    yaw_stability = compute_map_yaw_bins(
        nx, ny, nz, yaw_bins=MAP_YAW_BINS
    )
    
    # 将 yaw_stability 从 (H, W, D) 转置为 (D, H, W) 格式供评估器使用
    # compute_map_yaw_bins 返回 (H, W, D)，评估器期望 (D, H, W)
    yaw_stability_transposed = yaw_stability.permute(2, 0, 1)
    
    # 创建评估器
    evaluator = TrajectoryEvaluator(
        occupancy_map=stability_cost_map,
        # yaw_stability_map=yaw_stability_transposed,
        yaw_stability_map=stability_cost_map,
        map_info=map_info,
        device=device
    )
    
    # 评估优化前的轨迹（DIT预测的轨迹）
    metrics_before = evaluator.evaluate_trajectory(initial_traj[0, :, :].to(device))
    
    # 评估优化后的轨迹（optimized_traj 已包含 yaw -> (B,N,3)）
    metrics_after = evaluator.evaluate_trajectory(optimized_traj[0].to(device))
    
    # =================== 手动验证：对比两种计算方法 ===================
    print("\n" + "="*100)
    print("Manual Verification: Evaluator vs Direct Calculation".center(100))
    print("="*100 + "\n")
    
    # 使用 check_trajectory_reachability_consistent 手动计算初始轨迹的不可达点
    # Use the snapshot captured before optimization to avoid any accidental overwrites
    initial_traj_np = initial_traj_snapshot_np
    initial_mid_points = initial_traj_np[1:-1, :2]  # 去掉首尾
    initial_mid_yaws = initial_traj_np[1:-1, 2]
    
    def check_trajectory_reachability_consistent_inline(trajectory_points, yaw_values, yaw_stability):
        """内联版本的检查函数，用于调试"""
        capsize_mask = []
        for i in range(len(trajectory_points)):
            x, y = trajectory_points[i]
            yaw = yaw_values[i]
            
            x_idx, y_idx, yaw_idx = _stability_grid_indices(
                x, y, yaw, yaw_stability.shape[2]
            )
            
            # 边界检查和稳定性判断
            # yaw_stability 的形状是 (H, W, D)，其中 H=y方向, W=x方向, D=yaw方向
            if 0 <= y_idx < yaw_stability.shape[0] and 0 <= x_idx < yaw_stability.shape[1]:
                yaw_stability_value = yaw_stability[y_idx, x_idx, yaw_idx]  # 修正：[y_idx, x_idx, yaw_idx]
                if torch.is_tensor(yaw_stability_value):
                    is_unreachable = bool((yaw_stability_value == 0).cpu().numpy())
                else:
                    is_unreachable = bool(yaw_stability_value == 0)
            else:
                is_unreachable = True
            capsize_mask.append(is_unreachable)
        
        return np.array(capsize_mask, dtype=bool)
    
    manual_mask_initial = check_trajectory_reachability_consistent_inline(
        initial_mid_points, initial_mid_yaws, yaw_stability
    )
    manual_ratio_initial = (np.sum(manual_mask_initial) / len(manual_mask_initial)) if len(manual_mask_initial) > 0 else 0.0
    
    # 对比优化后的轨迹
    optimized_traj_np = optimized_traj[0].detach().cpu().numpy()
    optimized_traj_np = _ensure_traj_np(optimized_traj_np)
    optimized_mid_points = optimized_traj_np[1:-1, :2]
    optimized_mid_yaws = optimized_traj_np[1:-1, 2]
    
    manual_mask_optimized = check_trajectory_reachability_consistent_inline(
        optimized_mid_points, optimized_mid_yaws, yaw_stability
    )
    manual_ratio_optimized = (np.sum(manual_mask_optimized) / len(manual_mask_optimized)) if len(manual_mask_optimized) > 0 else 0.0
    
    print(f"Initial Trajectory (without endpoints):")
    print(f"  Evaluator unstable_point_ratio: {metrics_before['unstable_point_ratio']:.6f} ({metrics_before['unstable_point_ratio']*100:.2f}%)")
    print(f"  Manual calculation (direct):     {manual_ratio_initial:.6f} ({manual_ratio_initial*100:.2f}%)")
    print(f"  Difference: {abs(metrics_before['unstable_point_ratio'] - manual_ratio_initial):.6f}")
    print()
    print(f"Optimized Trajectory (without endpoints):")
    print(f"  Evaluator unstable_point_ratio: {metrics_after['unstable_point_ratio']:.6f} ({metrics_after['unstable_point_ratio']*100:.2f}%)")
    print(f"  Manual calculation (direct):     {manual_ratio_optimized:.6f} ({manual_ratio_optimized*100:.2f}%)")
    print(f"  Difference: {abs(metrics_after['unstable_point_ratio'] - manual_ratio_optimized):.6f}")
    print()
    
    # 调试：打印一些样本点的对比
    print("Sample point comparison (initial trajectory, first 5 mid-points):")
    for i in range(min(5, len(initial_mid_points))):
        x, y = initial_mid_points[i]
        yaw = initial_mid_yaws[i]
        x_idx, y_idx, yaw_idx = _stability_grid_indices(
            x, y, yaw, yaw_stability.shape[2]
        )
        
        if 0 <= y_idx < yaw_stability.shape[0] and 0 <= x_idx < yaw_stability.shape[1]:
            stability_val = yaw_stability[y_idx, x_idx, yaw_idx].item() if torch.is_tensor(yaw_stability[y_idx, x_idx, yaw_idx]) else yaw_stability[y_idx, x_idx, yaw_idx]
            print(f"  Point {i}: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f} -> idx=({y_idx}, {x_idx}, {yaw_idx}) -> stability={stability_val}")
    
    print("\n" + "="*100 + "\n")
    
    # 评估真实轨迹（Ground Truth）
    gt_trajectory_tensor = torch.tensor(trajectory, dtype=torch.float32, device=device)
    metrics_gt = evaluator.evaluate_trajectory(gt_trajectory_tensor)
    
    # 定义关键指标
    key_metrics = [
        'collision_risk_mean',
        'unstable_point_ratio',
        # 'path_length',
        # 'estimated_time',
        # 'smoothness_total',
        'jerk_x',
        'jerk_y',
        'jerk_yaw',
        'curvature_mean',
        'speed_mean',
        # 'out_of_bounds_ratio',
        # 'heading_error_mean'
    ]
    
    # 打印对比结果
    print("\nDetailed Metrics Comparison:")
    print("-" * 100)
    print(f"{'Metric':<30} {'GT':<15} {'Before Opt':<15} {'After Opt':<15} {'Improvement':<20}")
    print("-" * 100)
    
    improvements = []
    
    for metric in key_metrics:
        if metric in metrics_gt and metric in metrics_before and metric in metrics_after:
            gt_val = metrics_gt[metric]
            before_val = metrics_before[metric]
            after_val = metrics_after[metric]
            
            # 计算改进百分比
            if abs(before_val) > 1e-9:
                improvement_pct = ((before_val - after_val) / abs(before_val)) * 100
            else:
                improvement_pct = 0.0
            
            # 确定改进方向
            # 对于这些指标，值越小越好
            better_lower = metric in [
                'collision_risk_mean', 'unstable_point_ratio', 'out_of_bounds_ratio',
                'heading_error_mean', 'jerk_x', 'jerk_y', 'jerk_yaw', 'smoothness_total'
            ]
            
            if better_lower:
                is_improvement = after_val < before_val
                status = "↓" if is_improvement else "↑"
            else:
                is_improvement = after_val > before_val
                status = "↑" if is_improvement else "↓"
            
            improvement_str = f"{status} {abs(improvement_pct):.2f}%"
            if is_improvement:
                improvement_str = f"✓ {improvement_str}"
                improvements.append((metric, abs(improvement_pct)))
            else:
                improvement_str = f"✗ {improvement_str}"
            
            print(f"{metric:<30} {gt_val:>14.6f} {before_val:>14.6f} {after_val:>14.6f} {improvement_str:<20}")
    
    print("-" * 100)
    
    # 打印总结
    print("\n" + "="*100)
    print("Optimization Summary".center(100))
    print("="*100 + "\n")
    
    # 统计改进的指标数量
    num_improved = sum(1 for m in key_metrics 
                      if m in metrics_before and m in metrics_after 
                      and ((m in ['collision_risk_mean', 'unstable_point_ratio', 'out_of_bounds_ratio',
                                  'heading_error_mean', 'jerk_x', 'jerk_y', 'jerk_yaw', 'smoothness_total']
                           and metrics_after[m] < metrics_before[m])
                          or (m not in ['collision_risk_mean', 'unstable_point_ratio', 'out_of_bounds_ratio',
                                       'heading_error_mean', 'jerk_x', 'jerk_y', 'jerk_yaw', 'smoothness_total']
                              and metrics_after[m] > metrics_before[m])))
    
    total_metrics = len([m for m in key_metrics if m in metrics_before and m in metrics_after])
    
    print(f"Total metrics evaluated: {total_metrics}")
    print(f"Metrics improved: {num_improved} ({num_improved/total_metrics*100:.1f}%)")
    print(f"Metrics worsened: {total_metrics - num_improved} ({(total_metrics-num_improved)/total_metrics*100:.1f}%)")
    
    # 列出改进最大的指标
    if improvements:
        print("\nTop improvements:")
        improvements.sort(key=lambda x: x[1], reverse=True)
        for i, (metric, pct) in enumerate(improvements[:5], 1):
            print(f"  {i}. {metric}: {pct:.2f}%")
    
    # 关键安全指标对比
    print("\n" + "-"*100)
    print("Key Safety Metrics:".center(100))
    print("-"*100)
    
    safety_metrics = ['collision_risk_mean', 'unstable_point_ratio']
    for metric in safety_metrics:
        if metric in metrics_before and metric in metrics_after:
            before_val = metrics_before[metric]
            after_val = metrics_after[metric]
            change = ((after_val - before_val) / (abs(before_val) + 1e-9)) * 100
            status = "IMPROVED" if after_val < before_val else "WORSENED"
            print(f"  {metric}: {before_val:.6f} → {after_val:.6f} ({change:+.2f}%) [{status}]")
    
    print("\n" + "="*100 + "\n")

    # --- 4. 可视化结果 ---
    import mpl_toolkits.mplot3d  # 确保 3D 支持

    # 准备基础数据
    elev_map_np = sample['elevation'].cpu().numpy()
    # 计算要展示的 yaw 切片索引（优先根据初始控制点的第一个 yaw，回退到中间切片）
    D = map_size[2]
    try:
        init_yaw = float(initial_control_points_raw[0, 2])
        rel = (init_yaw - origin[2]) % (2 * np.pi)
        yaw_idx = int(round(rel / (2 * np.pi) * D)) % D
    except Exception:
        yaw_idx = D // 2
    cost_slice_to_show = stability_cost_map[yaw_idx].cpu().numpy()  # (H, W)
    map_extent = list(MAP_BOUNDS)

    # --- 先单独检测一下原轨迹的控制点，是否存在会倾覆的点 ---
    initial_yaw_dense = initial_traj[0, :, 2]  # 初始轨迹的 yaw_dense
    initial_control_points_tensor = torch.tensor(initial_control_points_raw, device=device, dtype=torch.float32)

    # 计算初始轨迹的中间点（去除首尾）- 使用控制点
    initial_control_mid_points = initial_control_points_raw[1:-1, :2]  # (N-2, 2)
    initial_control_mid_yaws = initial_control_points_raw[1:-1, 2]     # (N-2,)

    # yaw_stability 已在评估部分计算，此处无需重复
    # from dataLoader_uneven import compute_map_yaw_bins
    # yaw_stability = compute_map_yaw_bins(nx, ny, nz, yaw_bins=MAP_YAW_BINS)

    def check_trajectory_reachability_consistent(trajectory_points, yaw_values, yaw_stability):
        """使用与 data_clean.py 完全一致的方法检查轨迹点的可达性"""
        capsize_mask = []
        
        for i in range(len(trajectory_points)):
            x, y = trajectory_points[i]
            yaw = yaw_values[i]
            
            x_idx, y_idx, yaw_idx = _stability_grid_indices(
                x, y, yaw, yaw_stability.shape[2]
            )
            
            # 边界检查和稳定性判断
            # yaw_stability 的形状是 (H, W, D)，其中 H=y方向, W=x方向, D=yaw方向
            if 0 <= y_idx < yaw_stability.shape[0] and 0 <= x_idx < yaw_stability.shape[1]:
                yaw_stability_value = yaw_stability[y_idx, x_idx, yaw_idx]  # 修正：[y_idx, x_idx, yaw_idx]
                if torch.is_tensor(yaw_stability_value):
                    is_unreachable = bool((yaw_stability_value == 0).cpu().numpy())
                else:
                    is_unreachable = bool(yaw_stability_value == 0)
            else:
                is_unreachable = True  # 超出地图边界
                
            capsize_mask.append(is_unreachable)
        
        capsize_mask = np.array(capsize_mask, dtype=bool)
        capsize_points = trajectory_points[capsize_mask]
        
        return capsize_mask, capsize_points

    # 检查初始控制点（中间点）的可达性
    capsize_mask, capsize_points = check_trajectory_reachability_consistent(
        initial_control_mid_points, initial_control_mid_yaws, yaw_stability
    )
    num_capsize = np.sum(capsize_mask)
    print(f"Number of points which is unreachable in <initial_control_poses>: {num_capsize} out of {len(capsize_mask)}, percentage: {num_capsize / len(capsize_mask) * 100:.2f}%")
    # 为绘图创建独立的 numpy 副本，避免后续变量覆盖或形状问题
    initial_plot_np = initial_traj_snapshot_np.copy()
    initial_plot_np = _ensure_traj_np(initial_plot_np)

    optimized_plot_np = optimized_traj[0].detach().cpu().numpy().copy()
    optimized_plot_np = _ensure_traj_np(optimized_plot_np)

    # (debug prints removed)

    # --- Figure 1: 两个子图（分别显示初始轨迹 / 优化后轨迹，各自叠在彩色高程上） ---
    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 8))

    # 优化前轨迹 (使用 snapshot 的独立副本，明确标注)
    visualize_terrain_trajectory(
        axes1[0], 
        initial_plot_np, 
        nz,
        nx, 
        ny, 
        nz,
        positions=initial_control_points_np,  # 使用 control points (num_control_points=26)
        yaws=initial_control_yaws,  # 使用 control points 的 yaw
        traj_label='Initial Trajectory',
        traj_color='c',
        traj_linestyle='--'
    )
    axes1[0].set_title('Initial Trajectory')

    # 检查初始密集轨迹的可达性（去除端点）
    initial_traj_mid_points = initial_traj_np[1:-1, :2]     # (K-2, 2)
    initial_traj_mid_yaws = initial_traj_np[1:-1, 2]       # (K-2,)

    capsize_mask, capsize_points = check_trajectory_reachability_consistent(
        initial_traj_mid_points, initial_traj_mid_yaws, yaw_stability
    )
    # 不可达点高亮，zorder高于轨迹线，点更大
    if capsize_points.shape[0] > 0:
        axes1[0].scatter(capsize_points[:, 0], capsize_points[:, 1], c='#00ff00', s=60, label='unreachable points', zorder=40, edgecolors='black', linewidths=0.7)
    axes1[0].legend() # 添加图例

    # 添加对于可达性的统计信息
    num_capsize = np.sum(capsize_mask)
    print(f"Number of points which is unreachable in  <initial_trajectory> : {num_capsize} out of {len(capsize_mask)}, percentage: {num_capsize / len(capsize_mask) * 100:.2f}%")
    # 可视化优化后轨迹 (只用优化后的控制点)
    visualize_terrain_trajectory(
        axes1[1],
        optimized_plot_np,
        nz,
        nx,
        ny,
        nz,
        positions=optimized_control_points_np,  # 只用26个控制点
        yaws=optimized_control_yaws,           # 只用26个yaw
        traj_label='Optimized Trajectory',
        traj_color='m',
        traj_linestyle='-'
    )
    axes1[1].set_title('Optimized Trajectory')

    # 检查优化后密集轨迹的可达性（去除端点）
    optimized_traj_mid_points = optimized_trajectory[1:-1]     # (K-2, 2)
    optimized_traj_mid_yaws = optimized_yaw_dense[1:-1]       # (K-2,)

    capsize_mask, capsize_points = check_trajectory_reachability_consistent(
        optimized_traj_mid_points, optimized_traj_mid_yaws, yaw_stability
    )
    if capsize_points.shape[0] > 0:
        axes1[1].scatter(capsize_points[:, 0], capsize_points[:, 1], c='#00ff00', s=60, label='unreachable points', zorder=40, edgecolors='black', linewidths=0.7)
    axes1[1].legend()
    
    # 添加对于可达性的统计信息
    num_capsize = np.sum(capsize_mask)
    print(f"Number of points which is unreachable in <optimized_trajectory>: {num_capsize} out of {len(capsize_mask)}, percentage: {num_capsize / len(capsize_mask) * 100:.2f}%")

    # --- Figure 2: 两个 3D 子图（流形空间：x, y, yaw）---
    # 构建占据体小量采样点以供 3D 可视化（下采样以减少点数）
    D, H, W = stability_cost_map.shape
    st_map_np = stability_cost_map.cpu().numpy()  # (D, H, W)
    xs = origin[0] + (np.arange(W) + 0.5) * resolution
    ys = origin[1] + (np.arange(H) + 0.5) * resolution
    yaws = origin[2] + (np.arange(D) * (2*np.pi / D))
    # 下采样网格
    sx = np.linspace(0, W-1, min(40, W)).astype(int)
    sy = np.linspace(0, H-1, min(40, H)).astype(int)
    sd = np.linspace(0, D-1, min(40, D)).astype(int)
    Xg, Yg, Zg = np.meshgrid(xs[sx], ys[sy], yaws[sd], indexing='xy')
    vals = st_map_np[np.ix_(sd, sy, sx)]  # (Sd, Sy, Sx)
    vals_flat = vals.reshape(-1)
    coords_flat = np.stack([Xg.reshape(-1), Yg.reshape(-1), Zg.reshape(-1)], axis=1)
    # 只绘制高代价点，用颜色映射显示 cost
    mask_vis = vals_flat > np.percentile(vals_flat, 60)
    vis_coords = coords_flat[mask_vis]
    vis_vals = vals_flat[mask_vis]

    fig2 = plt.figure(figsize=(16, 8))
    ax3d_1 = fig2.add_subplot(1, 2, 1, projection='3d')
    ax3d_2 = fig2.add_subplot(1, 2, 2, projection='3d')

    ax3d_1.plot(initial_traj_np[:,0], initial_traj_np[:,1], initial_traj_np[:,2], color='cyan', linestyle='--', linewidth=2, label='Initial Traj (SE2)')
    # 只绘制26个初始控制点
    ax3d_1.scatter(initial_control_points_np[:,0], initial_control_points_np[:,1], initial_control_yaws, c='blue', s=40, marker='o', label='Initial Controls (26)', zorder=6)
    ax3d_1.set_title("Initial Trajectory in SE(2) Manifold")
    ax3d_1.set_xlabel("X (m)"); ax3d_1.set_ylabel("Y (m)"); ax3d_1.set_zlabel("Yaw (rad)")
    ax3d_1.legend(); ax3d_1.grid(True, linestyle='--', alpha=0.3)

    ax3d_2.plot(optimized_traj_np[:,0], optimized_traj_np[:,1], optimized_traj_np[:,2], color='magenta', linestyle='-', linewidth=2, label='Optimized Traj (SE2)')
    # 只绘制26个优化后控制点
    ax3d_2.scatter(optimized_control_points_np[:,0], optimized_control_points_np[:,1], optimized_control_yaws, c='red', s=40, marker='x', label='Optimized Controls (26)', zorder=6)
    ax3d_2.set_title("Optimized Trajectory in SE(2) Manifold")
    ax3d_2.set_xlabel("X (m)"); ax3d_2.set_ylabel("Y (m)"); ax3d_2.set_zlabel("Yaw (rad)")
    ax3d_2.legend(); ax3d_2.grid(True, linestyle='--', alpha=0.3)

    # # --- Figure 3: 损失曲线 ---
    # fig3, ax3 = plt.subplots(1, 1, figsize=(8, 4))
    # ax3.plot(cost_history, marker='o', linewidth=1)
    # ax3.set_title("Cost Function Convergence")
    # ax3.set_xlabel("Iteration"); ax3.set_ylabel("Cost")
    # ax3.grid(True); ax3.set_yscale('log')
    # plt.tight_layout()
    
    # savefig
    fig1.savefig(f'figure_1_trajectory_comparison_{path_index}.png', dpi=300)
    fig2.savefig(f'figure_2_3d_manifold_{path_index}.png', dpi=300)
    # fig3.savefig(f'figure_3_cost_convergence_{path_index}.png', dpi=300)
    
    plt.show()

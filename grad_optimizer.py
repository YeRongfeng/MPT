import math

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import tqdm

# Stage 2 checkpoint 必须绑定具体路径修正目标/硬判据语义。
PRIVILEGED_PATH_CORRECTION_SEMANTICS = (
    "gauge44_stable_yaw_log1p_curvature_dense200_physical_bounds_segment_mask_v13"
)
PRIVILEGED_COST_SEMANTICS = PRIVILEGED_PATH_CORRECTION_SEMANTICS

# 导入评估器
from evaluator import TrajectoryEvaluator
# B样条工具
from bspline_utils import (
    fit_bspline_least_squares,
    reconstruct_from_control_points,
    DifferentiableBSpline
)
from boundary_constrained_path import (
    BoundaryConstrainedPathRepresentation,
)
from map_config import (
    DENSE_TRAJECTORY_POINTS,
    MAP_BOUNDS,
    MAP_CONFIG,
    MAP_HALF_EXTENT,
    MAP_RESOLUTION,
    MAP_YAW_BINS,
    SAFETY_COST_CONFIG,
)
from tools._paths import test_dir

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

        curvature_limit = SAFETY_COST_CONFIG.curvature_limit
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


def stable_analytic_curvature_violation(
    first_derivative,
    second_derivative,
    *,
    curvature_limit=SAFETY_COST_CONFIG.curvature_limit,
    minimum_segment_meters=SAFETY_COST_CONFIG.curvature_min_segment_meters,
    num_trajectory_points=DENSE_TRAJECTORY_POINTS,
):
    """Return a well-conditioned surrogate for ``curvature > limit``.

    The exact inequality is

    ``|p' x p''| - kappa_max * ||p'||^3 <= 0``.

    Optimizing the usual quotient directly is ill-conditioned near a local
    stationary point.  We therefore optimize its algebraic positive margin
    and normalize by a denominator with a physical speed floor.  The zero set
    and violation sign are unchanged; exact hard acceptance still uses the
    dense analytic curvature audit.
    """
    first = torch.as_tensor(first_derivative)
    second = torch.as_tensor(
        second_derivative,
        dtype=first.dtype,
        device=first.device,
    )
    if first.shape != second.shape or first.ndim != 3 or first.shape[-1] != 2:
        raise ValueError(
            "analytic derivatives must have matching shape (B,N,2), got "
            f"{tuple(first.shape)} and {tuple(second.shape)}"
        )
    if first.shape[1] != int(num_trajectory_points):
        raise ValueError(
            "num_trajectory_points does not match analytic derivatives: "
            f"{num_trajectory_points} versus {first.shape[1]}"
        )
    if float(curvature_limit) <= 0.0:
        raise ValueError("curvature_limit must be positive")
    if float(minimum_segment_meters) <= 0.0:
        raise ValueError("minimum_segment_meters must be positive")

    speed = torch.linalg.vector_norm(first, dim=-1)
    cross = torch.abs(
        first[..., 0] * second[..., 1]
        - first[..., 1] * second[..., 0]
    )
    speed_cubed = speed.pow(3)
    algebraic_margin = cross - float(curvature_limit) * speed_cubed
    # For a uniform t grid, ||p'|| * dt approximates segment length.
    # This ties the conditioning floor to the existing physical degeneracy
    # threshold instead of introducing an unrelated tunable constant.
    speed_floor = float(minimum_segment_meters) * max(
        int(num_trajectory_points) - 1,
        1,
    )
    normalization = float(curvature_limit) * (
        speed_cubed + speed_floor**3
    )
    violation = F.relu(algebraic_margin) / normalization
    return violation, speed


class _StableAtan2(torch.autograd.Function):
    """Exact atan2 forward with a bounded low-speed backward denominator."""

    @staticmethod
    def forward(ctx, y, x, denominator_floor):
        ctx.save_for_backward(y, x)
        ctx.denominator_floor = float(denominator_floor)
        return torch.atan2(y, x)

    @staticmethod
    def backward(ctx, grad_output):
        y, x = ctx.saved_tensors
        denominator = (x.square() + y.square()).clamp_min(
            ctx.denominator_floor**2
        )
        grad_y = grad_output * x / denominator
        grad_x = -grad_output * y / denominator
        return grad_y, grad_x, None


def stable_analytic_yaw(
    first_derivative,
    *,
    minimum_segment_meters=SAFETY_COST_CONFIG.curvature_min_segment_meters,
    num_trajectory_points=DENSE_TRAJECTORY_POINTS,
):
    """Return exact tangent yaw with bounded gradients near zero speed."""
    first = torch.as_tensor(first_derivative)
    if first.ndim != 3 or first.shape[-1] != 2:
        raise ValueError(
            "analytic first derivative must have shape (B,N,2), got "
            f"{tuple(first.shape)}"
        )
    if first.shape[1] != int(num_trajectory_points):
        raise ValueError(
            "num_trajectory_points does not match analytic derivative: "
            f"{num_trajectory_points} versus {first.shape[1]}"
        )
    speed_floor = float(minimum_segment_meters) * max(
        int(num_trajectory_points) - 1,
        1,
    )
    yaw = _StableAtan2.apply(
        first[..., 1],
        first[..., 0],
        speed_floor,
    )
    speed = torch.linalg.vector_norm(first, dim=-1)
    backward_scale = torch.clamp(
        speed.square() / max(speed_floor**2, 1e-12),
        max=1.0,
    )
    return yaw, backward_scale


def discrete_turning_curvature(
    traj_xy,
    min_segment_length=SAFETY_COST_CONFIG.curvature_min_segment_meters,
):
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
    degenerate = (
        (incoming_length < float(min_segment_length))
        | (outgoing_length < float(min_segment_length))
    )
    # 重复/极短点不能被误判为零曲率；给出有限但显著的违规值。
    turning_angle = torch.where(
        degenerate,
        torch.full_like(turning_angle, np.pi),
        turning_angle,
    )
    local_arc_length = 0.5 * (incoming_length + outgoing_length)
    return turning_angle / local_arc_length.clamp_min(
        float(min_segment_length)
    )


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


def _sample_cost_map_at_xy_yaw(xy, yaw, occupancy_map, map_info, device):
    """按显式 ``(x, y, yaw)`` 在 yaw-aware 代价图上可微采样。

    轨迹 cost 和端点任务可行性预检查共用本函数，避免两处的坐标轴、
    周期 yaw 插值或边界处理悄悄不一致。
    """
    xy = torch.as_tensor(xy, dtype=torch.float32, device=device)
    yaw = torch.as_tensor(yaw, dtype=xy.dtype, device=device)
    if xy.ndim != 3 or xy.shape[-1] != 2:
        raise ValueError(f"xy 必须为 (B,N,2)，实际为 {tuple(xy.shape)}")
    if yaw.shape != xy.shape[:2]:
        raise ValueError(
            f"yaw 必须为 (B,N)，实际为 {tuple(yaw.shape)}，"
            f"xy 为 {tuple(xy.shape)}"
        )
    batch_size, num_points, _ = xy.shape
    origin = map_info['origin']
    resolution = float(map_info['resolution'])
    W, H, D = map_info['size']

    if resolution <= 0.0:
        raise ValueError(f"map resolution must be positive, got {resolution}")
    if min(W, H, D) < 2:
        raise ValueError(f"map dimensions must all be at least 2, got {(W, H, D)}")

    x = xy[:, :, 0]
    y = xy[:, :, 1]

    # ``origin`` is the lower map boundary, while each H/W element describes
    # a grid cell selected by floor((coord-origin)/resolution).  ESDF samples
    # therefore live at cell centers, not at the lower cell edges.
    x_idx = torch.clamp(
        (x - origin[0]) / resolution - 0.5,
        0.0,
        float(W - 1),
    )
    y_idx = torch.clamp(
        (y - origin[1]) / resolution - 0.5,
        0.0,
        float(H - 1),
    )
    yaw_rel = torch.remainder(yaw - origin[2], 2.0 * np.pi)
    yaw_idx = yaw_rel / (2.0 * np.pi / D)

    grid = torch.stack([
        (x_idx / (W - 1)) * 2.0 - 1.0,
        (y_idx / (H - 1)) * 2.0 - 1.0,
        # 后面会追加第0个 yaw bin，深度变为 D+1；align_corners=True
        # 时用 D 作分母，确保最后一格可周期插值回第0格。
        (yaw_idx / D) * 2.0 - 1.0,
    ], dim=-1).view(batch_size, num_points, 1, 1, 3)

    occ = torch.as_tensor(occupancy_map, dtype=xy.dtype, device=device)

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
    occ = torch.cat([occ, occ[:, :, :1]], dim=2)
    occ_vals = F.grid_sample(
        occ,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )
    occ_vals = occ_vals[:, 0, :, 0, 0]
    return occ_vals


def _sample_cost_map_on_dense_trajectory(
    traj_xy,
    analytic_yaw,
    occupancy_map,
    map_info,
    device,
):
    """
    使用 B-spline 解析切向 yaw，并在 (x, y, yaw) 代价图上可微采样。

    Returns:
        x, y, yaw, occ_vals，形状均为 (B, N)。
    """
    traj_xy = torch.as_tensor(traj_xy, dtype=torch.float32, device=device)
    if analytic_yaw is None:
        raise ValueError(
            "Production geometry requires analytic_yaw from B-spline p'(t); "
            "finite-difference yaw is not accepted."
        )
    yaw = torch.as_tensor(
        analytic_yaw, dtype=traj_xy.dtype, device=device
    )
    if yaw.shape != traj_xy.shape[:2]:
        raise ValueError(
            f"analytic_yaw must be {tuple(traj_xy.shape[:2])}, got "
            f"{tuple(yaw.shape)}"
        )
    occ_vals = _sample_cost_map_at_xy_yaw(
        traj_xy,
        yaw,
        occupancy_map,
        map_info,
        device,
    )
    x = traj_xy[:, :, 0]
    y = traj_xy[:, :, 1]
    return x, y, yaw, occ_vals


def _finite_difference_yaw(traj_xy):
    """Differentiable tangent yaw for legacy dense-trajectory soft costs.

    Production hard-validity paths pass analytic B-spline yaw explicitly.
    The older generic dense-path optimizer has only sampled xy positions, so
    its soft objective uses centered differences internally.
    """

    traj_xy = torch.as_tensor(traj_xy, dtype=torch.float32)
    if traj_xy.ndim != 3 or traj_xy.shape[-1] != 2:
        raise ValueError(
            f"traj_xy must have shape (B,N,2), got {tuple(traj_xy.shape)}"
        )
    if traj_xy.shape[1] < 2:
        raise ValueError("traj_xy must contain at least two points")
    tangent = torch.empty_like(traj_xy)
    tangent[:, 0] = traj_xy[:, 1] - traj_xy[:, 0]
    tangent[:, -1] = traj_xy[:, -1] - traj_xy[:, -2]
    if traj_xy.shape[1] > 2:
        tangent[:, 1:-1] = traj_xy[:, 2:] - traj_xy[:, :-2]
    return torch.atan2(tangent[..., 1], tangent[..., 0])


def _yaw_candidates_for_interval(center, tolerance, origin, bins):
    """返回周期 yaw 区间端点及其中所有插值网格节点。

    yaw 维采用分段线性插值，所以闭区间内的最大 stability 必然出现在
    区间端点或 yaw 网格节点；不需要依赖任意的密集采样数量。
    """
    center = float(center)
    tolerance = float(tolerance)
    if tolerance < 0.0:
        raise ValueError("yaw tolerance 不能为负数")
    step = 2.0 * math.pi / int(bins)
    values = [center - tolerance, center, center + tolerance]
    lower = center - tolerance
    upper = center + tolerance
    first = math.ceil((lower - float(origin)) / step)
    last = math.floor((upper - float(origin)) / step)
    values.extend(float(origin) + index * step for index in range(first, last + 1))
    return sorted(set(values))


def endpoint_stability_feasibility(
    start_pose,
    goal_pose,
    stability_cost_map,
    map_info,
    *,
    d_safe=SAFETY_COST_CONFIG.d_safe_meters,
    start_yaw_tolerance_rad=0.0,
    goal_yaw_tolerance_rad=0.0,
    constraint_epsilon=SAFETY_COST_CONFIG.hard_constraint_epsilon,
):
    """检查固定端点姿态是否满足 stability hard threshold。

    当前 44D 表示固定起终点位置和 yaw，因此默认只评估 pose 中的 exact
    yaw（tolerance=0）。只有调用者所审计的表示确实允许端点 yaw 变化时，
    才应显式传入非零 tolerance；此时返回区间内的最佳 stability margin。
    """
    start_pose = torch.as_tensor(start_pose, dtype=torch.float32).flatten()
    goal_pose = torch.as_tensor(goal_pose, dtype=torch.float32).flatten()
    if start_pose.numel() < 3 or goal_pose.numel() < 3:
        raise ValueError("start_pose 和 goal_pose 必须至少包含 (x,y,yaw)")

    _, _, yaw_bins = map_info["size"]
    yaw_origin = float(map_info["origin"][2])
    device = torch.as_tensor(stability_cost_map).device

    def best_margin(pose, tolerance):
        candidates = _yaw_candidates_for_interval(
            float(pose[2]),
            tolerance,
            yaw_origin,
            yaw_bins,
        )
        yaws = torch.tensor(candidates, dtype=torch.float32, device=device)
        xy = pose[:2].to(device).view(1, 1, 2).repeat(1, len(candidates), 1)
        sampled = _sample_cost_map_at_xy_yaw(
            xy,
            yaws.view(1, -1),
            stability_cost_map,
            map_info,
            device,
        )[0]
        return float(sampled.max().detach().cpu())

    start_best = best_margin(start_pose, start_yaw_tolerance_rad)
    goal_best = best_margin(goal_pose, goal_yaw_tolerance_rad)
    threshold = float(d_safe) - float(constraint_epsilon)
    start_feasible = bool(np.isfinite(start_best) and start_best >= threshold)
    goal_feasible = bool(np.isfinite(goal_best) and goal_best >= threshold)
    return {
        "feasible": start_feasible and goal_feasible,
        "start_feasible": start_feasible,
        "goal_feasible": goal_feasible,
        "start_best_stability_margin": start_best,
        "goal_best_stability_margin": goal_best,
        "required_stability_margin": float(d_safe),
        "start_yaw_tolerance_rad": float(start_yaw_tolerance_rad),
        "goal_yaw_tolerance_rad": float(goal_yaw_tolerance_rad),
    }


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
    if quality_term not in {'jerk', 'third_difference', 'smoothness', 'none'}:
        raise ValueError(
            "quality_term must be jerk/third_difference/smoothness/none, "
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
    dense_yaw = _finite_difference_yaw(traj_xy)
    x, y, yaw, occ_vals = _sample_cost_map_on_dense_trajectory(
        traj_xy,
        dense_yaw,
        occupancy_map,
        map_info,
        device,
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

    if quality_term in {'jerk', 'third_difference'}:
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


def _sample_mask(traj_xy, mask, map_info, *, mode="bilinear"):
    """沿轨迹可微采样二维可通行 mask（1=允许，0=禁止）。"""
    if mode not in {"bilinear", "nearest"}:
        raise ValueError(f"unsupported mask sampling mode: {mode!r}")
    mask = torch.as_tensor(
        mask, dtype=traj_xy.dtype, device=traj_xy.device
    )
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(
            f"mask must have shape (H,W) or (B,H,W), got {tuple(mask.shape)}"
        )
    batch_size = traj_xy.shape[0]
    if mask.shape[0] == 1:
        mask = mask.expand(batch_size, -1, -1)
    elif mask.shape[0] != batch_size:
        if batch_size % mask.shape[0] != 0:
            raise ValueError("mask batch cannot be aligned to trajectories")
        mask = mask.repeat_interleave(batch_size // mask.shape[0], dim=0)

    origin = map_info['origin']
    resolution = float(map_info['resolution'])
    width, height, _ = map_info['size']
    expected_shape = (int(height), int(width))
    if tuple(mask.shape[-2:]) != expected_shape:
        raise ValueError(
            "mask spatial shape must match map_info size (height,width): "
            f"expected {expected_shape}, got {tuple(mask.shape[-2:])}"
        )
    # Keep this distance field on the same spatial contract as the stability
    # ESDF: origin is the lower map boundary and array entries are cell centers.
    x_idx = (traj_xy[..., 0] - float(origin[0])) / resolution - 0.5
    y_idx = (traj_xy[..., 1] - float(origin[1])) / resolution - 0.5
    grid = torch.stack(
        [
            (x_idx / max(width - 1, 1)) * 2.0 - 1.0,
            (y_idx / max(height - 1, 1)) * 2.0 - 1.0,
        ],
        dim=-1,
    ).unsqueeze(2)
    values = F.grid_sample(
        mask.unsqueeze(1),
        grid,
        mode=mode,
        # 地图外边界由 box 项唯一处理；mask 场只描述地图内部禁区。
        padding_mode='border',
        align_corners=True,
    )
    return values[:, 0, :, 0]


def _interpolate_segment_points(traj_xy, *, spacing_m):
    """Return differentiable interior samples for each trajectory segment."""
    if traj_xy.ndim != 3 or traj_xy.shape[-1] != 2:
        raise ValueError(
            f"traj_xy must have shape (B,N,2), got {tuple(traj_xy.shape)}"
        )
    if float(spacing_m) <= 0.0:
        raise ValueError("spacing_m must be positive")
    batch_size, num_points, _ = traj_xy.shape
    if num_points < 2:
        return traj_xy.new_empty((batch_size, 0, 2))

    segment_length = torch.linalg.vector_norm(
        traj_xy[:, 1:] - traj_xy[:, :-1],
        dim=-1,
    )
    max_intervals = max(
        1,
        int(
            torch.ceil(
                segment_length.detach().amax() / float(spacing_m)
            ).item()
        ),
    )
    if max_intervals <= 1:
        return traj_xy.new_empty((batch_size, 0, 2))

    fractions = torch.arange(
        1,
        max_intervals,
        dtype=traj_xy.dtype,
        device=traj_xy.device,
    ) / float(max_intervals)
    points = (
        traj_xy[:, :-1, None, :]
        + fractions[None, None, :, None]
        * (traj_xy[:, 1:, None, :] - traj_xy[:, :-1, None, :])
    )
    return points.reshape(batch_size, -1, 2)


def _sample_mask_on_segments(
    traj_xy,
    mask,
    map_info,
    *,
    spacing_m,
    mode="bilinear",
):
    """Sample a mask field between consecutive trajectory points."""
    segment_points = _interpolate_segment_points(
        traj_xy,
        spacing_m=spacing_m,
    )
    if segment_points.shape[1] == 0:
        return segment_points.new_empty(segment_points.shape[:2])
    return _sample_mask(
        segment_points,
        mask,
        map_info,
        mode=mode,
    )


def _safe_map_bounds(map_info, boundary_safe_pixels):
    """Return symmetric physical map bounds after a pixel-sized inset."""
    resolution = float(map_info["resolution"])
    width, height, _ = map_info["size"]
    if resolution <= 0.0:
        raise ValueError("map resolution must be positive")
    if float(boundary_safe_pixels) < 0.0:
        raise ValueError("boundary_safe_pixels cannot be negative")

    x_lower = float(map_info["origin"][0])
    y_lower = float(map_info["origin"][1])
    x_upper = x_lower + width * resolution
    y_upper = y_lower + height * resolution
    if "bounds" in map_info:
        bounds = tuple(map(float, map_info["bounds"]))
        expected = (x_lower, x_upper, y_lower, y_upper)
        if len(bounds) != 4 or any(
            not math.isclose(actual, wanted, abs_tol=1e-6)
            for actual, wanted in zip(bounds, expected)
        ):
            raise ValueError(
                "map bounds are inconsistent with origin, size, and resolution"
            )

    inset = float(boundary_safe_pixels) * resolution
    x_min, x_max = x_lower + inset, x_upper - inset
    y_min, y_max = y_lower + inset, y_upper - inset
    if x_min > x_max or y_min > y_max:
        raise ValueError("boundary inset leaves no valid map interior")
    return x_min, x_max, y_min, y_max


def _box_signed_distance_at_xy(traj_xy, map_info, boundary_safe_pixels):
    """Return physical signed distance to the inset rectangular map box."""
    x_min, x_max, y_min, y_max = _safe_map_bounds(
        map_info,
        boundary_safe_pixels,
    )
    x = traj_xy[..., 0]
    y = traj_xy[..., 1]
    return torch.stack(
        [
            x - x_min,
            x_max - x,
            y - y_min,
            y_max - y,
        ],
        dim=-1,
    ).amin(dim=-1)


def build_forbidden_distance_map(mask, map_info, device=None):
    """计算禁区内部的无符号深度，仅保留给旧实验兼容使用。

    二值 mask 在禁区内部梯度为零，单独采样它无法把深处轨迹推出障碍物。
    当前 Stage 2 主线改用 :func:`build_signed_mask_distance_map`，不会调用本
    函数。
    """
    from scipy.ndimage import distance_transform_edt

    mask = torch.as_tensor(mask, dtype=torch.float32)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(
            "mask 必须为 (H,W) 或 (B,H,W)，"
            f"实际为 {tuple(mask.shape)}"
        )
    distances = [
        distance_transform_edt(sample.detach().cpu().numpy() <= 0.5)
        * float(map_info["resolution"])
        for sample in mask
    ]
    return torch.as_tensor(
        np.stack(distances),
        dtype=torch.float32,
        device=device or mask.device,
    )


def build_signed_mask_distance_map(mask, map_info, device=None):
    """构造带符号的配置空间 mask 距离场。

    符号约定固定为：

    - ``d_mask > 0``：可通行区域；
    - ``d_mask < 0``：mask 禁区；
    - 最小化基于 ``-d_mask`` 的惩罚会推动轨迹令 ``d_mask`` 增大。

    输入必须已经是车辆 footprint 腐蚀后的唯一配置空间 mask；本函数不再
    做二次腐蚀。
    """
    from scipy.ndimage import distance_transform_edt

    mask = torch.as_tensor(mask, dtype=torch.float32)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(
            "mask 必须为 (H,W) 或 (B,H,W)，"
            f"实际为 {tuple(mask.shape)}"
        )
    resolution = float(map_info["resolution"])
    width, height, _ = map_info["size"]
    expected_shape = (int(height), int(width))
    if tuple(mask.shape[-2:]) != expected_shape:
        raise ValueError(
            "mask spatial shape must match map_info size (height,width): "
            f"expected {expected_shape}, got {tuple(mask.shape[-2:])}"
        )
    large_distance = resolution * math.hypot(width, height)
    signed_distances = []
    for sample in mask:
        allowed = sample.detach().cpu().numpy() > 0.5
        if allowed.all():
            # 完整地图没有内部 mask 障碍。使用常数场，确保 mask cost 及其
            # 空间梯度严格为零；矩形外边界只由 box 项处理。
            signed_distances.append(
                np.full(allowed.shape, large_distance, dtype=np.float32)
            )
            continue
        if not allowed.any():
            signed_distances.append(
                np.full(allowed.shape, -large_distance, dtype=np.float32)
            )
            continue
        distance_inside_allowed = distance_transform_edt(allowed)
        distance_inside_forbidden = distance_transform_edt(~allowed)
        signed = (
            distance_inside_allowed - distance_inside_forbidden
        ) * resolution
        # 完全对称障碍的 medial axis 上，纯 EDT 可能形成离散平台而给出零
        # 梯度。仅在禁区内部加入远小于一个像素距离的确定性斜率来打破并列；
        # 它不改变正负类别，也不参与完整地图。
        row_index, col_index = np.indices(allowed.shape)
        centered_ramp = (
            col_index
            + 0.61803398875 * row_index
            - 0.5 * (allowed.shape[1] + 0.61803398875 * allowed.shape[0])
        )
        signed = np.where(
            allowed,
            signed,
            signed
            + SAFETY_COST_CONFIG.mask_medial_axis_tiebreak_per_pixel
            * resolution
            * centered_ramp,
        )
        signed_distances.append(signed)
    return torch.as_tensor(
        np.stack(signed_distances),
        dtype=torch.float32,
        device=device or mask.device,
    )


def privileged_planning_cost(
    trajectory,
    start_pose,
    goal_pose,
    stability_cost_map,
    map_info,
    *,
    analytic_yaw=None,
    analytic_curvature=None,
    analytic_first_derivative=None,
    analytic_second_derivative=None,
    mask=None,
    signed_mask_distance_map=None,
    return_per_sample=False,
    return_components=False,
    stability_tail_ratio=SAFETY_COST_CONFIG.tail_ratio,
    curvature_tail_ratio=SAFETY_COST_CONFIG.curvature_tail_ratio,
    forbidden_tail_ratio=SAFETY_COST_CONFIG.forbidden_tail_ratio,
    d_safe=SAFETY_COST_CONFIG.d_safe_meters,
    alpha=SAFETY_COST_CONFIG.softplus_alpha,
    curvature_limit=SAFETY_COST_CONFIG.curvature_limit,
    boundary_safe_pixels=SAFETY_COST_CONFIG.boundary_safe_pixels,
    forbidden_weight=SAFETY_COST_CONFIG.forbidden_weight,
    stability_weight=SAFETY_COST_CONFIG.obstacle_weight,
    curvature_weight=SAFETY_COST_CONFIG.curvature_weight,
    endpoint_yaw_weight=0.0,
    regularization_weight=SAFETY_COST_CONFIG.quality_weight,
    short_segment_weight=SAFETY_COST_CONFIG.short_segment_weight,
):
    """Stage 2 独立特权 cost，不在内部调用旧的总 cost 函数。

    主目标只有统一禁区、稳定性和解析曲率；端点 yaw 是表示不变量，
    只保留 assertion/诊断，不再进入 Stage 2 cost。proposal tether 在外层
    优化器中加入。
    """
    trajectory = torch.as_tensor(trajectory, dtype=torch.float32)
    if trajectory.ndim == 2:
        trajectory = trajectory.unsqueeze(0)
    if trajectory.ndim != 3 or trajectory.shape[-1] < 2:
        raise ValueError(
            "trajectory 必须为 (N,2/3) 或 (B,N,2/3)，"
            f"实际为 {tuple(trajectory.shape)}"
        )
    if trajectory.shape[1] < 4:
        raise ValueError("特权 cost 至少需要 4 个稠密轨迹点")
    traj_xy = trajectory[..., :2]
    device = traj_xy.device
    batch_size, num_points, _ = traj_xy.shape

    ratios = {
        "stability_tail_ratio": stability_tail_ratio,
        "curvature_tail_ratio": curvature_tail_ratio,
        "forbidden_tail_ratio": forbidden_tail_ratio,
    }
    for name, ratio in ratios.items():
        if not 0.0 < float(ratio) <= 1.0:
            raise ValueError(f"{name} 必须位于 (0,1]")
    if alpha <= 0.0 or d_safe <= 0.0 or curvature_limit <= 0.0:
        raise ValueError("alpha、d_safe 和 curvature_limit 必须为正数")
    weights = {
        "forbidden_weight": forbidden_weight,
        "stability_weight": stability_weight,
        "curvature_weight": curvature_weight,
        "endpoint_yaw_weight": endpoint_yaw_weight,
        "regularization_weight": regularization_weight,
        "short_segment_weight": short_segment_weight,
    }
    for name, weight in weights.items():
        if float(weight) < 0.0:
            raise ValueError(f"{name} 不能为负数")

    derivatives_provided = (
        analytic_first_derivative is not None,
        analytic_second_derivative is not None,
    )
    if derivatives_provided[0] != derivatives_provided[1]:
        raise ValueError(
            "analytic_first_derivative and analytic_second_derivative "
            "must be provided together"
        )
    if derivatives_provided[0]:
        yaw_for_cost, yaw_backward_scale = stable_analytic_yaw(
            analytic_first_derivative,
            minimum_segment_meters=(
                SAFETY_COST_CONFIG.curvature_min_segment_meters
            ),
            num_trajectory_points=num_points,
        )
    else:
        yaw_for_cost = analytic_yaw
        yaw_backward_scale = torch.full(
            traj_xy.shape[:2],
            float("nan"),
            dtype=traj_xy.dtype,
            device=device,
        )

    x, y, yaw, stability = _sample_cost_map_on_dense_trajectory(
        traj_xy,
        yaw_for_cost,
        stability_cost_map,
        map_info,
        device,
    )
    resolution = float(map_info["resolution"])

    # stability map 的值越大越安全。软 cost 与 hard threshold 共用 d_safe。
    stability_point_violation = (
        F.softplus(alpha * (d_safe - stability)) / (alpha * d_safe)
    )
    stability_cost = top_tail_mean(
        stability_point_violation,
        stability_tail_ratio,
    )

    if analytic_curvature is None:
        raise ValueError(
            "Production geometry requires analytic_curvature from B-spline "
            "p'(t), p''(t); finite-difference curvature is not accepted."
        )
    curvature = torch.as_tensor(
        analytic_curvature, dtype=traj_xy.dtype, device=device
    )
    if curvature.shape != traj_xy.shape[:2]:
        raise ValueError(
            f"analytic_curvature must be {tuple(traj_xy.shape[:2])}, got "
            f"{tuple(curvature.shape)}"
        )
    if derivatives_provided[0]:
        raw_turning_violation, analytic_speed = (
            stable_analytic_curvature_violation(
                analytic_first_derivative,
                analytic_second_derivative,
                curvature_limit=curvature_limit,
                minimum_segment_meters=(
                    SAFETY_COST_CONFIG.curvature_min_segment_meters
                ),
                num_trajectory_points=num_points,
            )
        )
    else:
        # Compatibility path for diagnostics that only carry precomputed
        # curvature. Production Stage 2 forwards always provide derivatives.
        raw_turning_violation = F.relu(
            curvature / float(curvature_limit) - 1.0
        )
        analytic_speed = torch.full_like(curvature, float("nan"))
    # Near the hard boundary log1p(v) ~= v, while severe cusp-like outliers
    # receive a robust, unbounded penalty whose derivative cannot dominate an
    # entire model update solely because of their distance from the boundary.
    turning_violation = torch.log1p(raw_turning_violation)
    segment_length = torch.linalg.vector_norm(
        traj_xy[:, 1:] - traj_xy[:, :-1],
        dim=-1,
    )
    minimum_segment = float(
        SAFETY_COST_CONFIG.curvature_min_segment_meters
    )
    short_segment_violation = F.relu(
        minimum_segment - segment_length
    ) / max(resolution, 1e-9)
    short_segment_point_violation = torch.empty_like(curvature)
    short_segment_point_violation[:, 0] = short_segment_violation[:, 0]
    short_segment_point_violation[:, -1] = short_segment_violation[:, -1]
    short_segment_point_violation[:, 1:-1] = torch.maximum(
        short_segment_violation[:, :-1],
        short_segment_violation[:, 1:],
    )
    # 极短线段与转角都属于同一个局部几何可执行性项。
    curvature_point_violation = (
        turning_violation
        + float(short_segment_weight) * short_segment_point_violation
    )
    curvature_cost = top_tail_mean(
        curvature_point_violation,
        curvature_tail_ratio,
    )

    width, height, _ = map_info["size"]
    x_min, x_max, y_min, y_max = _safe_map_bounds(
        map_info,
        boundary_safe_pixels,
    )
    box_signed_distance = torch.stack(
        [
            x - x_min,
            x_max - x,
            y - y_min,
            y_max - y,
        ],
        dim=-1,
    ).amin(dim=-1)
    # Evaluate every original point and every interior segment sample.  The
    # latter is required for both the external mask and the physical box:
    # checking only the 200 stored points can miss a forbidden crossing.
    segment_points = _interpolate_segment_points(
        traj_xy,
        spacing_m=0.5 * resolution,
    )
    segment_box_signed_distance = _box_signed_distance_at_xy(
        segment_points,
        map_info,
        boundary_safe_pixels,
    )
    box_constraint_signed_distance = torch.cat(
        [box_signed_distance, segment_box_signed_distance],
        dim=1,
    )
    box_violation_meters = F.relu(-box_constraint_signed_distance)
    # 完整地图时 mask 距离为足够大的正常数，因此统一距离自然退化为 box。
    mask_signed_distance = torch.full_like(
        x,
        resolution * math.hypot(width, height),
    )
    mask_segment_signed_distance = torch.empty(
        (batch_size, 0),
        dtype=traj_xy.dtype,
        device=device,
    )
    if mask is not None:
        if signed_mask_distance_map is None:
            signed_mask_distance_map = build_signed_mask_distance_map(
                mask,
                map_info,
                device=device,
            )
        mask_signed_distance = _sample_mask(
            traj_xy,
            signed_mask_distance_map,
            map_info,
        )
        if segment_points.shape[1] > 0:
            mask_segment_signed_distance = _sample_mask(
                segment_points,
                signed_mask_distance_map,
                map_info,
                mode="bilinear",
            )
    segment_composite_clearance = (
        segment_box_signed_distance
        if mask is None
        else torch.minimum(
            mask_segment_signed_distance,
            segment_box_signed_distance,
        )
    )
    # 这是 mask clearance 与 box clearance 的复合安全裕度；其符号严格正确，
    # 但在两类边界交汇处不声称是交集边界的精确欧氏 signed distance。
    composite_clearance = torch.minimum(
        mask_signed_distance,
        box_signed_distance,
    )
    composite_clearance_forbidden = torch.cat(
        [composite_clearance, segment_composite_clearance],
        dim=1,
    )
    mask_signed_distance_for_metrics = torch.cat(
        [mask_signed_distance, mask_segment_signed_distance],
        dim=1,
    )
    forbidden_point_violation = F.softplus(
        -alpha * composite_clearance_forbidden / resolution
    ) / alpha
    forbidden_cost = top_tail_mean(
        forbidden_point_violation,
        forbidden_tail_ratio,
    )

    start_yaw = _pose_yaw_for_batch(
        start_pose,
        batch_size,
        trajectory.dtype,
        device,
        "start_pose",
    )
    goal_yaw = _pose_yaw_for_batch(
        goal_pose,
        batch_size,
        trajectory.dtype,
        device,
        "goal_pose",
    )
    yaw_diff_start = torch.atan2(
        torch.sin(yaw[:, 0] - start_yaw),
        torch.cos(yaw[:, 0] - start_yaw),
    )
    yaw_diff_end = torch.atan2(
        torch.sin(yaw[:, -1] - goal_yaw),
        torch.cos(yaw[:, -1] - goal_yaw),
    )
    endpoint_cost = (
        torch.abs(yaw_diff_start) + torch.abs(yaw_diff_end)
    ) / np.pi

    path_length = segment_length.sum(dim=1)
    map_diagonal = math.hypot(
        width * resolution,
        height * resolution,
    )
    normalized_length = path_length / max(map_diagonal, 1e-6)

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
    third_x = ddx[:, 1:] - ddx[:, :-1]
    third_y = ddy[:, 1:] - ddy[:, :-1]
    third_yaw = torch.atan2(
        torch.sin(ddyaw[:, 1:] - ddyaw[:, :-1]),
        torch.cos(ddyaw[:, 1:] - ddyaw[:, :-1]),
    )
    third_difference_cost = torch.mean(
        third_x.square() + third_y.square() + third_yaw.square(),
        dim=1,
    )

    weighted_forbidden = float(forbidden_weight) * forbidden_cost
    weighted_stability = float(stability_weight) * stability_cost
    weighted_curvature = float(curvature_weight) * curvature_cost
    if float(endpoint_yaw_weight) != 0.0:
        raise ValueError(
            "endpoint_yaw_weight must be zero: endpoint yaw is a 44D "
            "representation invariant and is no longer an optimization term."
        )
    weighted_endpoint_yaw = torch.zeros_like(endpoint_cost)
    weighted_regularization = (
        float(regularization_weight) * third_difference_cost
    )
    task_cost = (
        weighted_forbidden
        + weighted_stability
        + weighted_curvature
    )
    total = task_cost + weighted_regularization

    def point_summary(prefix, values):
        return {
            f"{prefix}_mean": values.mean(dim=1),
            f"{prefix}_max": values.amax(dim=1),
        }

    components = {
        "forbidden_region": forbidden_cost,
        "stability": stability_cost,
        "curvature": curvature_cost,
        "endpoint_yaw": endpoint_cost,
        "regularization": third_difference_cost,
        "path_length": path_length,
        "normalized_length": normalized_length,
        "third_difference_regularization": third_difference_cost,
        "weighted_forbidden_region": weighted_forbidden,
        "weighted_stability": weighted_stability,
        "weighted_curvature": weighted_curvature,
        "weighted_endpoint_yaw": weighted_endpoint_yaw,
        "weighted_regularization": weighted_regularization,
        "task_cost": task_cost,
        "regularized_task_cost": total,
        "total": total,
        "min_stability_margin": stability.amin(dim=1),
        "curvature_max": curvature.amax(dim=1),
        "analytic_speed_min": analytic_speed.amin(dim=1),
        "analytic_yaw_backward_scale_min": yaw_backward_scale.amin(dim=1),
        "composite_clearance_min": composite_clearance_forbidden.amin(dim=1),
        "box_signed_distance_min": box_constraint_signed_distance.amin(dim=1),
        "mask_signed_distance_min": mask_signed_distance_for_metrics.amin(dim=1),
        **point_summary("stability_violation", stability_point_violation),
        **point_summary("curvature_violation", curvature_point_violation),
        **point_summary("turning_violation", turning_violation),
        **point_summary(
            "raw_turning_violation",
            raw_turning_violation,
        ),
        **point_summary(
            "short_segment_violation",
            short_segment_point_violation,
        ),
        **point_summary(
            "forbidden_violation",
            forbidden_point_violation,
        ),
        "stability_violation_ratio": (
            stability < float(d_safe)
        ).float().mean(dim=1),
        "curvature_violation_ratio": (
            curvature > float(curvature_limit)
        ).float().mean(dim=1),
        "forbidden_violation_ratio": (
            composite_clearance_forbidden < 0.0
        ).float().mean(dim=1),
        # 以下两个仅用于失败原因细分，不再是独立优化项。
        "box_component_violation_max": box_violation_meters.amax(dim=1),
        "box_component_violation_ratio": (
            box_constraint_signed_distance < 0.0
        ).float().mean(dim=1),
        "mask_component_violation_max": F.relu(
            -mask_signed_distance_for_metrics
        ).amax(dim=1),
        "mask_component_violation_ratio": (
            mask_signed_distance_for_metrics < 0.0
        ).float().mean(dim=1),
    }
    if not return_per_sample:
        total = total.mean()
        components = {
            key: value.mean() for key, value in components.items()
        }
    if return_components:
        return total, components
    return total


_AUDIT_COMPONENT_PAIRS = {
    "forbidden_region": (
        "forbidden_region",
        "weighted_forbidden_region",
    ),
    "stability": ("stability", "weighted_stability"),
    "curvature": ("curvature", "weighted_curvature"),
    "endpoint_yaw": ("endpoint_yaw", "weighted_endpoint_yaw"),
    "optional_regularization": (
        "regularization",
        "weighted_regularization",
    ),
}


def _summarize_cost_component_gradients(components, variable):
    """汇总指定优化空间中各 cost 分项的值和梯度范数。"""
    batch_size = variable.shape[0]

    def quantile(values, q):
        values = values.detach().float().flatten()
        return float(torch.quantile(values, q))

    def gradient_norm(values):
        if not values.requires_grad:
            return torch.zeros(
                batch_size,
                dtype=variable.dtype,
                device=variable.device,
            )
        gradient = torch.autograd.grad(
            values.sum(),
            variable,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if gradient is None:
            return torch.zeros(
                batch_size,
                dtype=variable.dtype,
                device=variable.device,
            )
        return torch.linalg.vector_norm(gradient.flatten(1), dim=1)

    result = {}
    for name, (raw_key, weighted_key) in _AUDIT_COMPONENT_PAIRS.items():
        raw = components[raw_key]
        weighted = components[weighted_key]
        raw_gradient = gradient_norm(raw)
        weighted_gradient = gradient_norm(weighted)
        result[name] = {
            "value_mean": float(raw.detach().mean()),
            "value_median": quantile(raw, 0.5),
            "value_p95": quantile(raw, 0.95),
            "gradient_norm_mean": float(raw_gradient.mean()),
            "gradient_norm_median": quantile(raw_gradient, 0.5),
            "gradient_norm_p95": quantile(raw_gradient, 0.95),
            "weighted_contribution_mean": float(weighted.detach().mean()),
            "weighted_gradient_norm_mean": float(weighted_gradient.mean()),
            "weighted_gradient_norm_p95": quantile(
                weighted_gradient,
                0.95,
            ),
        }
    return result


def privileged_cost_scale_diagnostics(
    trajectory,
    start_pose,
    goal_pose,
    stability_cost_map,
    map_info,
    *,
    analytic_yaw,
    analytic_curvature,
    analytic_first_derivative=None,
    analytic_second_derivative=None,
    mask,
    signed_mask_distance_map=None,
):
    """统计各分项对稠密轨迹空间的梯度范数。"""
    trajectory = torch.as_tensor(
        trajectory,
        dtype=torch.float32,
    ).detach().clone().requires_grad_(True)
    _, components = privileged_planning_cost(
        trajectory,
        start_pose,
        goal_pose,
        stability_cost_map,
        map_info,
        analytic_yaw=analytic_yaw,
        analytic_curvature=analytic_curvature,
        analytic_first_derivative=analytic_first_derivative,
        analytic_second_derivative=analytic_second_derivative,
        mask=mask,
        signed_mask_distance_map=signed_mask_distance_map,
        return_per_sample=True,
        return_components=True,
    )
    result = _summarize_cost_component_gradients(components, trajectory)
    result["third_difference_is_hard_constraint"] = (
        SAFETY_COST_CONFIG.third_difference_is_hard_constraint
    )
    return result


def privileged_optimizer_scale_diagnostics(
    proposal_residual,
    start_pose,
    goal_pose,
    stability_cost_map,
    map_info,
    *,
    coordinate_scale=MAP_HALF_EXTENT,
    mask,
    signed_mask_distance_map=None,
    num_dense_points=DENSE_TRAJECTORY_POINTS,
):
    """同时报告 dense-space 与优化器真实 44D state-space 梯度。"""
    residual = torch.as_tensor(
        proposal_residual,
        dtype=torch.float32,
    ).detach().clone()
    residual = residual.requires_grad_(True)
    count = residual.shape[0]
    start_pose = torch.as_tensor(
        start_pose,
        dtype=residual.dtype,
        device=residual.device,
    )
    goal_pose = torch.as_tensor(
        goal_pose,
        dtype=residual.dtype,
        device=residual.device,
    )
    if start_pose.ndim == 1:
        start_pose = start_pose.unsqueeze(0)
    if goal_pose.ndim == 1:
        goal_pose = goal_pose.unsqueeze(0)
    if start_pose.shape[0] == 1:
        start_pose = start_pose.expand(count, -1)
    if goal_pose.shape[0] == 1:
        goal_pose = goal_pose.expand(count, -1)

    bridge = BoundaryConstrainedPathRepresentation(
        dense_points=int(num_dense_points)
    ).to(residual.device)
    geometry = bridge.evaluate(residual, start_pose, goal_pose)
    dense = geometry["position"]
    _, components = privileged_planning_cost(
        dense,
        start_pose,
        goal_pose,
        stability_cost_map,
        map_info,
        analytic_yaw=geometry["yaw"],
        analytic_curvature=geometry["curvature"],
        analytic_first_derivative=geometry["first_derivative"],
        analytic_second_derivative=geometry["second_derivative"],
        mask=mask,
        signed_mask_distance_map=signed_mask_distance_map,
        return_per_sample=True,
        return_components=True,
    )
    return {
        "dense_space": _summarize_cost_component_gradients(
            components,
            dense,
        ),
        "state_space": _summarize_cost_component_gradients(
            components,
            residual,
        ),
        "primary_balance_space": "state_space",
        "third_difference_is_hard_constraint": (
            SAFETY_COST_CONFIG.third_difference_is_hard_constraint
        ),
    }


def privileged_cost_contract():
    """返回 Stage 2 软优化项与最终 hard acceptance 的明确对应关系。"""
    return {
        "semantic_version": PRIVILEGED_COST_SEMANTICS,
        "dense_trajectory_points": DENSE_TRAJECTORY_POINTS,
        "spatial_sampling": (
            "origin_is_lower_boundary; H/W arrays live at cell centers"
        ),
        "selection_priority": [
            "accepted(strict_valid_and_quality_ok)",
            "strict_valid",
            "minimum_normalized_violation",
        ],
        "replay_acceptance": "strict_valid and length_quality_gate",
        "safe_at_k_grouping": "condition_id",
        "objective_decomposition": {
            "task_cost": (
                "forbidden_region + stability + analytic_curvature"
            ),
            "expert_objective": "task_cost + proposal_tether",
            "strict_validity_includes_proposal_tether": False,
        },
        "proposal_tether": {
            "optimization": "proposal_weight * mean((R-R_proposal)^2)",
            "hard": False,
            "location": "optimize_privileged_trajectories outer objective",
        },
        "forbidden_region": {
            "margin": "min(mask_signed_clearance, box_signed_clearance)",
            "external_mask_contract": (
                "caller supplies the already-eroded configuration-space mask; "
                "the cost does not generate or replace it"
            ),
            "trajectory_sampling": (
                "original trajectory points plus differentiable interior samples "
                "on every segment at spacing <= 0.5 * map resolution"
            ),
            "soft_sampling": "bilinear signed-distance sampling",
            "hard_sampling": "nearest signed-distance sampling",
            "terminology": (
                "unified free-space margin / composite signed clearance; "
                "not claimed to be the exact Euclidean signed distance "
                "of the intersection"
            ),
            "optimization": (
                "tail_mean(softplus(-composite_clearance/resolution))"
            ),
            "hard": (
                "min_composite_clearance >= "
                f"{-SAFETY_COST_CONFIG.hard_constraint_epsilon}"
            ),
            "subcomponents_for_failure_diagnosis": ["mask", "box"],
        },
        "stability": {
            "optimization": "tail_mean(softplus(d_safe-stability_margin))",
            "yaw_forward": "exact analytic atan2(p_prime_y,p_prime_x)",
            "yaw_backward": (
                "atan2 denominator floored by physical minimum-segment speed"
            ),
            "hard": (
                "min_stability_margin >= "
                f"{SAFETY_COST_CONFIG.d_safe_meters}"
            ),
        },
        "curvature": {
            "optimization": (
                "tail_mean(log1p(stable_algebraic_turning_violation) + "
                "short_segment_violation)"
            ),
            "optimization_zero_set": (
                "abs(cross(p_prime,p_double_prime)) <= "
                "curvature_limit * norm(p_prime)^3"
            ),
            "conditioning_speed_floor": (
                SAFETY_COST_CONFIG.curvature_min_segment_meters
                * (DENSE_TRAJECTORY_POINTS - 1)
            ),
            "hard": (
                "analytic_audit_max_curvature <= "
                f"{SAFETY_COST_CONFIG.curvature_limit}"
            ),
            "hard_audit_points": "representation.audit_curvature default grid",
        },
        "length": {
            "optimization": False,
            "diagnostic": "path_length/map_diagonal",
            "quality_gate": (
                "path_length/map_diagonal <= "
                f"{SAFETY_COST_CONFIG.max_path_length_ratio}"
            ),
            "part_of_strict_physical_validity": False,
        },
        "endpoint_yaw": {
            "optimization": "none; representation invariant",
            "hard": True,
            "start_tolerance_rad": SAFETY_COST_CONFIG.start_yaw_tolerance_rad,
            "goal_tolerance_rad": SAFETY_COST_CONFIG.goal_yaw_tolerance_rad,
        },
        "third_difference_regularization": {
            "optimization": (
                "optional sample-count-dependent regularizer; default weight=0"
            ),
            "weight": SAFETY_COST_CONFIG.quality_weight,
            "hard": SAFETY_COST_CONFIG.third_difference_is_hard_constraint,
            "physical_jerk": False,
        },
    }


@torch.no_grad()
def trajectory_validity_metrics(
    trajectory,
    stability_cost_map,
    map_info,
    *,
    analytic_yaw=None,
    analytic_curvature=None,
    analytic_curvature_audit=None,
    mask=None,
    signed_mask_distance_map=None,
    d_safe=SAFETY_COST_CONFIG.hard_stability_margin_meters,
    curvature_limit=SAFETY_COST_CONFIG.curvature_limit,
    boundary_safe_pixels=SAFETY_COST_CONFIG.boundary_safe_pixels,
    constraint_epsilon=SAFETY_COST_CONFIG.hard_constraint_epsilon,
    max_path_length_ratio=SAFETY_COST_CONFIG.max_path_length_ratio,
    start_pose=None,
    goal_pose=None,
    endpoint_yaw_is_hard_constraint=(
        SAFETY_COST_CONFIG.endpoint_yaw_is_hard_constraint
    ),
    start_yaw_tolerance_rad=SAFETY_COST_CONFIG.start_yaw_tolerance_rad,
    goal_yaw_tolerance_rad=SAFETY_COST_CONFIG.goal_yaw_tolerance_rad,
    condition_ids=None,
):
    """按与特权 cost 相同的采样与坐标执行 hard validity。

    倾覆硬条件是稳定性余量严格大于 ``d_safe``。论文评价默认
    ``hard_stability_margin_meters=0``：余量 > 0 即未倾覆。Stage 2 软代价
    仍用 ``d_safe_meters``，搜索器也可另设规划余量。

    ``analytic_curvature`` 必须与稠密轨迹逐点对齐，用于积分统计。可选的
    ``analytic_curvature_audit`` 可以使用更密的解析采样；提供时，它决定
    hard curvature 最大值、``curvature_ok`` 和 strict validity，避免稀疏
    积分网格漏掉局部峰值。三阶差分只报告、不作为硬约束。长度是独立
    质量门槛，不混入 ``strict_valid``。safe@K 只有在显式提供
    condition_ids（或 batch=1）时才计算，避免把多个 condition 混成一组。
    """
    trajectory = torch.as_tensor(trajectory, dtype=torch.float32)
    if trajectory.ndim == 2:
        trajectory = trajectory.unsqueeze(0)
    traj_xy = trajectory[..., :2]
    x, y, yaw, stability = _sample_cost_map_on_dense_trajectory(
        traj_xy,
        analytic_yaw,
        stability_cost_map,
        map_info,
        traj_xy.device,
    )
    if analytic_curvature is None:
        raise ValueError(
            "Production hard-validity requires analytic_curvature from "
            "B-spline p'(t), p''(t)."
        )
    curvature = torch.as_tensor(
        analytic_curvature,
        dtype=traj_xy.dtype,
        device=traj_xy.device,
    )
    if curvature.shape != traj_xy.shape[:2]:
        raise ValueError(
            f"analytic_curvature must be {tuple(traj_xy.shape[:2])}, got "
            f"{tuple(curvature.shape)}"
        )
    batch_size = traj_xy.shape[0]
    if analytic_curvature_audit is None:
        curvature_audit = curvature
    else:
        curvature_audit = torch.as_tensor(
            analytic_curvature_audit,
            dtype=traj_xy.dtype,
            device=traj_xy.device,
        )
        if (
            curvature_audit.ndim != 2
            or curvature_audit.shape[0] != batch_size
            or curvature_audit.shape[1] == 0
        ):
            raise ValueError(
                "analytic_curvature_audit must have shape (B,M), M>0; "
                f"expected B={batch_size}, got {tuple(curvature_audit.shape)}"
            )
    hard_max_curvature = curvature_audit.amax(dim=1)
    resolution = float(map_info['resolution'])
    width, height, _ = map_info['size']
    stability_violation = F.relu(float(d_safe) - stability)
    curvature_violation = F.relu(curvature - float(curvature_limit))
    hard_curvature_violation = F.relu(
        hard_max_curvature - float(curvature_limit)
    )
    box_signed_distance = _box_signed_distance_at_xy(
        traj_xy,
        map_info,
        boundary_safe_pixels,
    )
    segment_points = _interpolate_segment_points(
        traj_xy,
        spacing_m=0.5 * resolution,
    )
    segment_box_signed_distance = _box_signed_distance_at_xy(
        segment_points,
        map_info,
        boundary_safe_pixels,
    )
    box_constraint_signed_distance = torch.cat(
        [box_signed_distance, segment_box_signed_distance],
        dim=1,
    )
    box_violation = F.relu(-box_constraint_signed_distance)
    stable = stability.amin(dim=1) > float(d_safe)
    curvature_ok = hard_curvature_violation <= float(constraint_epsilon)
    box_component_ok = (
        box_constraint_signed_distance.amin(dim=1)
        >= -float(constraint_epsilon)
    )
    mask_signed_distance = torch.full_like(
        x,
        float(map_info["resolution"]) * max(width, height),
    )
    mask_hard_signed_distance = mask_signed_distance
    mask_segment_signed_distance = torch.empty(
        (batch_size, 0),
        dtype=traj_xy.dtype,
        device=traj_xy.device,
    )
    mask_segment_soft_signed_distance = torch.empty(
        (batch_size, 0),
        dtype=traj_xy.dtype,
        device=traj_xy.device,
    )
    if mask is None:
        mask_component_ok = torch.ones_like(box_component_ok)
    else:
        if signed_mask_distance_map is None:
            signed_mask_distance_map = build_signed_mask_distance_map(
                mask,
                map_info,
                device=traj_xy.device,
            )
        mask_signed_distance = _sample_mask(
            traj_xy,
            signed_mask_distance_map,
            map_info,
        )
        mask_hard_signed_distance = _sample_mask(
            traj_xy,
            signed_mask_distance_map,
            map_info,
            mode="nearest",
        )
        if segment_points.shape[1] > 0:
            mask_segment_signed_distance = _sample_mask(
                segment_points,
                signed_mask_distance_map,
                map_info,
                mode="nearest",
            )
            mask_segment_soft_signed_distance = _sample_mask(
                segment_points,
                signed_mask_distance_map,
                map_info,
                mode="bilinear",
            )
        mask_constraint_signed_distance = torch.cat(
            [mask_hard_signed_distance, mask_segment_signed_distance],
            dim=1,
        )
        mask_component_ok = (
            mask_constraint_signed_distance.amin(dim=1)
            >= -float(constraint_epsilon)
        )
    if mask is None:
        mask_constraint_signed_distance = mask_hard_signed_distance
    segment_composite_clearance = (
        segment_box_signed_distance
        if mask is None
        else torch.minimum(
            mask_segment_signed_distance,
            segment_box_signed_distance,
        )
    )
    segment_composite_soft_clearance = (
        segment_box_signed_distance
        if mask is None
        else torch.minimum(
            mask_segment_soft_signed_distance,
            segment_box_signed_distance,
        )
    )
    composite_clearance = torch.minimum(
        mask_signed_distance,
        box_signed_distance,
    )
    hard_composite_clearance = torch.cat(
        [
            torch.minimum(mask_hard_signed_distance, box_signed_distance),
            segment_composite_clearance,
        ],
        dim=1,
    )
    forbidden_region_ok = (
        hard_composite_clearance.amin(dim=1)
        >= -float(constraint_epsilon)
    )

    segment_length = torch.linalg.vector_norm(
        traj_xy[:, 1:] - traj_xy[:, :-1],
        dim=-1,
    )
    normalized_forbidden_violation = F.relu(
        -composite_clearance
    ) / max(resolution, 1e-6)
    hard_normalized_forbidden_violation = F.relu(
        -hard_composite_clearance
    ) / max(resolution, 1e-6)
    normalized_stability_violation = stability_violation / max(
        float(d_safe), 1e-6
    )
    normalized_curvature_violation = curvature_violation / max(
        float(curvature_limit), 1e-6
    )
    normalized_hard_curvature_violation = (
        hard_curvature_violation / max(float(curvature_limit), 1e-6)
    )
    violation_density = torch.sqrt(
        normalized_forbidden_violation.square()
        + normalized_stability_violation.square()
        + normalized_curvature_violation.square()
    )
    violation_max_normalized = torch.sqrt(
        hard_normalized_forbidden_violation.amax(dim=1).square()
        + normalized_stability_violation.amax(dim=1).square()
        + normalized_hard_curvature_violation.square()
    )
    interior_count = (
        segment_points.shape[1] // max(traj_xy.shape[1] - 1, 1)
    )
    if interior_count > 0:
        intervals = interior_count + 1
        interior_forbidden = F.relu(
            -segment_composite_soft_clearance
        ) / max(resolution, 1e-6)
        non_forbidden_density = torch.sqrt(
            normalized_stability_violation.square()
            + normalized_curvature_violation.square()
        )
        fractions = torch.arange(
            1,
            intervals,
            dtype=traj_xy.dtype,
            device=traj_xy.device,
        ) / float(intervals)
        interior_non_forbidden = (
            non_forbidden_density[:, :-1, None]
            + fractions[None, None, :]
            * (
                non_forbidden_density[:, 1:, None]
                - non_forbidden_density[:, :-1, None]
            )
        )
        density_grid = torch.cat(
            [
                violation_density[:, :-1, None],
                torch.sqrt(
                    interior_forbidden.reshape(
                        batch_size,
                        traj_xy.shape[1] - 1,
                        interior_count,
                    ).square()
                    + interior_non_forbidden.square()
                ),
                violation_density[:, 1:, None],
            ],
            dim=-1,
        )
        subsegment_length = segment_length / float(intervals)
        violation_integral_normalized_m = (
            0.5
            * (density_grid[..., :-1] + density_grid[..., 1:])
            * subsegment_length[..., None]
        ).sum(dim=(-1, -2))
    else:
        violation_integral_normalized_m = (
            0.5
            * (violation_density[:, :-1] + violation_density[:, 1:])
            * segment_length
        ).sum(dim=1)
    robust_margin_normalized = torch.stack(
        [
            hard_composite_clearance.amin(dim=1) / max(resolution, 1e-6),
            (stability.amin(dim=1) - float(d_safe))
            / max(float(d_safe), 1e-6),
            (float(curvature_limit) - hard_max_curvature)
            / max(float(curvature_limit), 1e-6),
        ],
        dim=1,
    ).amin(dim=1)
    path_length = segment_length.sum(dim=1)
    map_diagonal = math.hypot(
        width * resolution,
        height * resolution,
    )
    path_length_ratio = path_length / max(map_diagonal, 1e-6)
    length_ok = path_length_ratio <= float(max_path_length_ratio)

    dx = x[:, 1:] - x[:, :-1]
    dy = y[:, 1:] - y[:, :-1]
    ddx = dx[:, 1:] - dx[:, :-1]
    ddy = dy[:, 1:] - dy[:, :-1]
    third_difference_regularization = (
        (ddx[:, 1:] - ddx[:, :-1]).square()
        + (ddy[:, 1:] - ddy[:, :-1]).square()
    ).mean(dim=1)
    if (
        endpoint_yaw_is_hard_constraint
        and (start_pose is None or goal_pose is None)
    ):
        raise ValueError(
            "启用 endpoint yaw hard constraint 时必须提供 start_pose/goal_pose"
        )
    # 关闭 yaw 硬约束时允许省略姿态；此时用轨迹自身端点切向作为占位，
    # 只为保持诊断字段数值良好，不会参与 strict validity。
    if start_pose is None:
        start_pose = torch.cat([traj_xy[:, 0], yaw[:, 0:1]], dim=1)
    if goal_pose is None:
        goal_pose = torch.cat([traj_xy[:, -1], yaw[:, -1:]], dim=1)
    start_yaw = _pose_yaw_for_batch(
        start_pose,
        batch_size,
        trajectory.dtype,
        traj_xy.device,
        "start_pose",
    )
    goal_yaw = _pose_yaw_for_batch(
        goal_pose,
        batch_size,
        trajectory.dtype,
        traj_xy.device,
        "goal_pose",
    )
    start_yaw_error = torch.abs(
        torch.atan2(
            torch.sin(yaw[:, 0] - start_yaw),
            torch.cos(yaw[:, 0] - start_yaw),
        )
    )
    goal_yaw_error = torch.abs(
        torch.atan2(
            torch.sin(yaw[:, -1] - goal_yaw),
            torch.cos(yaw[:, -1] - goal_yaw),
        )
    )
    if endpoint_yaw_is_hard_constraint:
        endpoint_yaw_ok = (
            (start_yaw_error <= float(start_yaw_tolerance_rad))
            & (goal_yaw_error <= float(goal_yaw_tolerance_rad))
        )
    else:
        endpoint_yaw_ok = torch.ones_like(box_component_ok)
    finite_ok = (
        torch.isfinite(traj_xy).flatten(1).all(dim=1)
        & torch.isfinite(stability).all(dim=1)
        & torch.isfinite(curvature).all(dim=1)
        & torch.isfinite(curvature_audit).all(dim=1)
        & torch.isfinite(mask_constraint_signed_distance).all(dim=1)
    )
    strict_valid = (
        finite_ok
        & stable
        & curvature_ok
        & forbidden_region_ok
        & endpoint_yaw_ok
    )
    quality_ok = length_ok
    accepted = strict_valid & quality_ok

    safe_at_k_by_condition = None
    accepted_at_k_by_condition = None
    safe_at_k_rate = None
    accepted_at_k_rate = None
    if condition_ids is not None:
        condition_ids = torch.as_tensor(
            condition_ids,
            device=traj_xy.device,
        ).flatten()
        if condition_ids.numel() != batch_size:
            raise ValueError(
                "condition_ids 数量必须等于轨迹 batch，"
                f"实际为 {condition_ids.numel()} 和 {batch_size}"
            )
        unique_ids = torch.unique(condition_ids, sorted=True)
        safe_at_k_by_condition = torch.stack(
            [strict_valid[condition_ids == key].any() for key in unique_ids]
        )
        accepted_at_k_by_condition = torch.stack(
            [accepted[condition_ids == key].any() for key in unique_ids]
        )
        safe_at_k_rate = safe_at_k_by_condition.float().mean()
        accepted_at_k_rate = accepted_at_k_by_condition.float().mean()
    elif batch_size == 1:
        safe_at_k_by_condition = strict_valid.clone()
        accepted_at_k_by_condition = accepted.clone()
        safe_at_k_rate = strict_valid.float().mean()
        accepted_at_k_rate = accepted.float().mean()

    return {
        # ``valid`` 保留为专家接受别名；物理可行性请显式读取 strict_valid。
        'valid': accepted,
        'accepted': accepted,
        'strict_valid': strict_valid,
        'quality_ok': quality_ok,
        'valid_rate': accepted.float().mean(),
        'strict_valid_rate': strict_valid.float().mean(),
        'safe_at_k': safe_at_k_rate,
        'safe_at_k_by_condition': safe_at_k_by_condition,
        'accepted_at_k': accepted_at_k_rate,
        'accepted_at_k_by_condition': accepted_at_k_by_condition,
        'min_stability_margin': stability.amin(dim=1),
        # Hard maximum may come from a denser analytic audit grid.
        'max_curvature': hard_max_curvature,
        'dense_max_curvature': curvature.amax(dim=1),
        'curvature_audit_points': torch.full(
            (batch_size,),
            int(curvature_audit.shape[1]),
            dtype=torch.long,
            device=traj_xy.device,
        ),
        'stability_violation_max': stability_violation.amax(dim=1),
        'stability_violation_mean': stability_violation.mean(dim=1),
        'stability_violation_ratio': (
            stability_violation > float(constraint_epsilon)
        ).float().mean(dim=1),
        'curvature_violation_max': hard_curvature_violation,
        'curvature_violation_mean': curvature_violation.mean(dim=1),
        'curvature_violation_ratio': (
            curvature_violation > float(constraint_epsilon)
        ).float().mean(dim=1),
        # Unified Stage-2 operator-audit quantities.  The integral has physical
        # arc-length units (metres); all violation components are normalized by
        # their corresponding hard-constraint scale.
        'violation_max_normalized': violation_max_normalized,
        'violation_integral_normalized_m': (
            violation_integral_normalized_m
        ),
        'robust_margin_normalized': robust_margin_normalized,
        'box_violation_max': box_violation.amax(dim=1),
        'box_violation_mean': box_violation.mean(dim=1),
        'box_violation_ratio': (
            box_violation > float(constraint_epsilon)
        ).float().mean(dim=1),
        'mask_signed_distance_min': mask_constraint_signed_distance.amin(dim=1),
        'mask_segment_sample_count': torch.full(
            (batch_size,),
            int(mask_segment_signed_distance.shape[1]),
            dtype=torch.long,
            device=traj_xy.device,
        ),
        'box_signed_distance_min': box_signed_distance.amin(dim=1),
        'composite_clearance_min': hard_composite_clearance.amin(dim=1),
        'forbidden_violation_max': F.relu(
            -hard_composite_clearance
        ).amax(dim=1),
        'forbidden_violation_mean': F.relu(
            -hard_composite_clearance
        ).mean(dim=1),
        'forbidden_violation_ratio': (
            hard_composite_clearance < -float(constraint_epsilon)
        ).float().mean(dim=1),
        'mask_violation_max': F.relu(
            -mask_constraint_signed_distance
        ).amax(dim=1),
        'mask_violation_mean': F.relu(
            -mask_constraint_signed_distance
        ).mean(dim=1),
        'mask_violation_ratio': (
            mask_constraint_signed_distance < -float(constraint_epsilon)
        ).float().mean(dim=1),
        'path_length': path_length,
        'path_length_ratio': path_length_ratio,
        'third_difference_regularization': third_difference_regularization,
        'start_yaw_error': start_yaw_error,
        'goal_yaw_error': goal_yaw_error,
        'endpoint_yaw_ok': endpoint_yaw_ok,
        'third_difference_is_hard_constraint': torch.zeros_like(accepted),
        'stability_ok': stable,
        'curvature_ok': curvature_ok,
        'forbidden_region_ok': forbidden_region_ok,
        # 仅用于区分失败来自内部 mask 还是矩形边界，不再独立进入 strict。
        'box_component_ok': box_component_ok,
        'mask_component_ok': mask_component_ok,
        # 兼容现有日志读取；语义明确为子诊断，不是两个独立 hard 项。
        'box_ok': box_component_ok,
        'mask_ok': mask_component_ok,
        'length_ok': length_ok,
        'finite_ok': finite_ok,
    }


def apply_privileged_path_correction(
    proposal_residual,
    start_pose,
    goal_pose,
    stability_cost_map,
    map_info,
    *,
    coordinate_scale=MAP_HALF_EXTENT,
    mask=None,
    signed_mask_distance_map=None,
    iterations=80,
    lr=5e-2,
    grad_clip_norm=5.0,
    proposal_weight=2e-3,
    proximity_metric="state_mse",
    robust_identity_margin=0.05,
    num_dense_points=DENSE_TRAJECTORY_POINTS,
    cost_kwargs=None,
):
    """用完整地图在无约束 44D state 中逐条纠正当前策略 proposal。

    已严格可行且最弱归一化约束裕度不小于
    ``robust_identity_margin`` 的 proposal 是修正器的精确固定点。刚越过
    hard threshold 的边界可行解仍允许通过近端优化增加安全裕度。
    """
    if proximity_metric not in {"state_mse", "physical_path_mse"}:
        raise ValueError(
            "proximity_metric must be 'state_mse' or "
            f"'physical_path_mse', got {proximity_metric!r}"
        )
    proposal = torch.as_tensor(
        proposal_residual, dtype=torch.float32,
        device=torch.as_tensor(proposal_residual).device,
    )
    if proposal.ndim != 3 or proposal.shape[1:] != (22, 2):
        raise ValueError(
            f"proposal state must have shape (K,22,2), got {tuple(proposal.shape)}"
        )
    device = proposal.device
    optimized = proposal.clone().detach().requires_grad_(True)
    count = proposal.shape[0]

    start_pose = torch.as_tensor(start_pose, dtype=torch.float32, device=device)
    goal_pose = torch.as_tensor(goal_pose, dtype=torch.float32, device=device)
    if start_pose.ndim == 1:
        start_pose = start_pose.unsqueeze(0)
    if goal_pose.ndim == 1:
        goal_pose = goal_pose.unsqueeze(0)
    if start_pose.shape[0] == 1:
        start_pose = start_pose.expand(count, -1)
    if goal_pose.shape[0] == 1:
        goal_pose = goal_pose.expand(count, -1)

    bridge = BoundaryConstrainedPathRepresentation(
        dense_points=int(num_dense_points)
    ).to(device)
    # residual 已有显式 proposal tether；这里禁止 AdamW 的隐式 weight decay。
    optimizer = torch.optim.Adam([optimized], lr=float(lr))
    best_accepted = proposal.clone()
    best_accepted_cost = torch.full((count,), float('inf'), device=device)
    best_accepted_found = torch.zeros(count, dtype=torch.bool, device=device)
    best_valid = proposal.clone()
    best_valid_cost = torch.full((count,), float('inf'), device=device)
    best_valid_found = torch.zeros(count, dtype=torch.bool, device=device)
    best_invalid = proposal.clone()
    best_invalid_score = torch.full((count,), float('inf'), device=device)
    best_invalid_cost = torch.full((count,), float('inf'), device=device)
    history = []
    cost_kwargs = dict(cost_kwargs or {})
    if mask is not None and signed_mask_distance_map is None:
        signed_mask_distance_map = build_signed_mask_distance_map(
            mask, map_info, device=device
        )
    with torch.no_grad():
        initial_geometry = bridge.evaluate(
            proposal, start_pose, goal_pose
        )
        initial_validity = trajectory_validity_metrics(
            initial_geometry["position"],
            stability_cost_map,
            map_info,
            analytic_yaw=initial_geometry["yaw"],
            analytic_curvature=initial_geometry["curvature"],
            analytic_curvature_audit=bridge.audit_curvature(
                proposal, start_pose, goal_pose
            ),
            mask=mask,
            signed_mask_distance_map=signed_mask_distance_map,
            start_pose=start_pose,
            goal_pose=goal_pose,
            condition_ids=torch.zeros(
                count, dtype=torch.long, device=device
            ),
        )
        initial_robust_margin = initial_validity.get(
            "robust_margin_normalized",
            torch.full(
                (count,), float("-inf"), dtype=proposal.dtype, device=device
            ),
        )
        robust_identity_mask = (
            initial_validity["strict_valid"]
            & (
                initial_robust_margin
                >= float(robust_identity_margin)
            )
        )
        # The unmodified proposal is iteration zero and must participate in
        # best-candidate retention even before the first optimizer step.
        initial_strict = initial_validity["strict_valid"]
        initial_accepted = initial_validity.get(
            "accepted", initial_strict
        )
        best_valid_found |= initial_strict
        best_accepted_found |= initial_accepted
        best_valid[initial_strict] = proposal[initial_strict]
        best_accepted[initial_accepted] = proposal[initial_accepted]

    def update_best(candidate_residual, candidate_geometry, candidate_total):
        """按 accepted、strict-valid、invalid 三层优先级保存候选。"""
        nonlocal best_accepted, best_accepted_cost, best_accepted_found
        nonlocal best_valid, best_valid_cost, best_valid_found
        nonlocal best_invalid, best_invalid_score, best_invalid_cost
        validity = trajectory_validity_metrics(
            candidate_geometry["position"],
            stability_cost_map,
            map_info,
            analytic_yaw=candidate_geometry["yaw"],
            analytic_curvature=candidate_geometry["curvature"],
            analytic_curvature_audit=bridge.audit_curvature(
                candidate_residual, start_pose, goal_pose
            ),
            mask=mask,
            signed_mask_distance_map=signed_mask_distance_map,
            start_pose=start_pose,
            goal_pose=goal_pose,
            condition_ids=torch.zeros(
                count,
                dtype=torch.long,
                device=device,
            ),
        )
        hard_valid = validity["strict_valid"]
        accepted = validity.get("accepted", hard_valid)
        accepted_improved = accepted & (
            candidate_total < best_accepted_cost
        )
        best_accepted_cost = torch.where(
            accepted_improved,
            candidate_total,
            best_accepted_cost,
        )
        best_accepted[accepted_improved] = candidate_residual[
            accepted_improved
        ]
        best_accepted_found |= accepted

        valid_improved = hard_valid & (candidate_total < best_valid_cost)
        best_valid_cost = torch.where(
            valid_improved,
            candidate_total,
            best_valid_cost,
        )
        best_valid[valid_improved] = candidate_residual[valid_improved]
        best_valid_found |= hard_valid

        resolution = float(map_info["resolution"])
        violation_score = (
            validity["forbidden_violation_max"] / max(resolution, 1e-6)
            + validity["stability_violation_max"]
            / max(SAFETY_COST_CONFIG.d_safe_meters, 1e-6)
            + validity["curvature_violation_max"]
            / max(SAFETY_COST_CONFIG.curvature_limit, 1e-6)
            + F.relu(
                validity["path_length_ratio"]
                / max(SAFETY_COST_CONFIG.max_path_length_ratio, 1e-6)
                - 1.0
            )
        )
        violation_score = torch.where(
            validity["finite_ok"],
            violation_score,
            torch.full_like(violation_score, float("inf")),
        )
        score_improved = violation_score < best_invalid_score
        score_tied = torch.isclose(
            violation_score,
            best_invalid_score,
            rtol=0.0,
            atol=1e-9,
        )
        invalid_improved = (~hard_valid) & (
            score_improved
            | (score_tied & (candidate_total < best_invalid_cost))
        )
        best_invalid_score = torch.where(
            invalid_improved,
            violation_score,
            best_invalid_score,
        )
        best_invalid_cost = torch.where(
            invalid_improved,
            candidate_total,
            best_invalid_cost,
        )
        best_invalid[invalid_improved] = candidate_residual[invalid_improved]
        return validity

    for _ in range(int(iterations)):
        optimizer.zero_grad(set_to_none=True)
        residual = torch.where(
            robust_identity_mask[:, None, None],
            proposal,
            optimized,
        )
        geometry = bridge.evaluate(residual, start_pose, goal_pose)
        regularized_task_cost = privileged_planning_cost(
            geometry["position"],
            start_pose,
            goal_pose,
            stability_cost_map,
            map_info,
            analytic_yaw=geometry["yaw"],
            analytic_curvature=geometry["curvature"],
            analytic_first_derivative=geometry["first_derivative"],
            analytic_second_derivative=geometry["second_derivative"],
            mask=mask,
            signed_mask_distance_map=signed_mask_distance_map,
            return_per_sample=True,
            **cost_kwargs,
        )
        if proximity_metric == "physical_path_mse":
            tether = (
                geometry["position"] - initial_geometry["position"]
            ).square().mean(dim=(1, 2))
        else:
            tether = (residual - proposal).square().mean(dim=(1, 2))
        expert_objective = (
            regularized_task_cost + float(proposal_weight) * tether
        )
        with torch.no_grad():
            update_best(residual, geometry, expert_objective)
        expert_objective.mean().backward()
        if grad_clip_norm and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_([optimized], float(grad_clip_norm))
        optimizer.step()
        history.append(float(expert_objective.detach().mean().item()))

    with torch.no_grad():
        # Include the iterate produced by the final optimizer step.
        candidate = optimized
        candidate_geometry = bridge.evaluate(
            candidate, start_pose, goal_pose
        )
        candidate_regularized_task_cost = privileged_planning_cost(
            candidate_geometry["position"],
            start_pose,
            goal_pose,
            stability_cost_map,
            map_info,
            analytic_yaw=candidate_geometry["yaw"],
            analytic_curvature=candidate_geometry["curvature"],
            analytic_first_derivative=(
                candidate_geometry["first_derivative"]
            ),
            analytic_second_derivative=(
                candidate_geometry["second_derivative"]
            ),
            mask=mask,
            signed_mask_distance_map=signed_mask_distance_map,
            return_per_sample=True,
            **cost_kwargs,
        )
        if proximity_metric == "physical_path_mse":
            candidate_tether = (
                candidate_geometry["position"]
                - initial_geometry["position"]
            ).square().mean(dim=(1, 2))
        else:
            candidate_tether = (
                candidate - proposal
            ).square().mean(dim=(1, 2))
        candidate_expert_objective = (
            candidate_regularized_task_cost
            + float(proposal_weight)
            * candidate_tether
        )
        update_best(
            candidate,
            candidate_geometry,
            candidate_expert_objective,
        )

        strict_or_invalid = torch.where(
            best_valid_found[:, None, None],
            best_valid,
            best_invalid,
        )
        # 满足 replay 质量门槛的轨迹优先于仅满足物理 hard constraints 的轨迹；
        # 后者仍优先于所有 invalid 轨迹并保留用于失败诊断。
        selected = torch.where(
            best_accepted_found[:, None, None],
            best_accepted,
            strict_or_invalid,
        )
        selected = torch.where(
            robust_identity_mask[:, None, None],
            proposal,
            selected,
        )
        final_geometry = bridge.evaluate(selected, start_pose, goal_pose)
        final_curvature_audit = bridge.audit_curvature(
            selected, start_pose, goal_pose
        )
        bridge.assert_endpoint_yaw(selected, start_pose, goal_pose)
        dense = final_geometry["position"]
        final_regularized_task_cost, final_components = privileged_planning_cost(
            dense,
            start_pose,
            goal_pose,
            stability_cost_map,
            map_info,
            analytic_yaw=final_geometry["yaw"],
            analytic_curvature=final_geometry["curvature"],
            analytic_first_derivative=final_geometry["first_derivative"],
            analytic_second_derivative=final_geometry["second_derivative"],
            mask=mask,
            signed_mask_distance_map=signed_mask_distance_map,
            return_per_sample=True,
            return_components=True,
            **cost_kwargs,
        )
        if proximity_metric == "physical_path_mse":
            final_proposal_tether = (
                final_geometry["position"]
                - initial_geometry["position"]
            ).square().mean(dim=(1, 2))
        else:
            final_proposal_tether = (
                selected - proposal
            ).square().mean(dim=(1, 2))
        final_expert_objective = (
            final_regularized_task_cost
            + float(proposal_weight) * final_proposal_tether
        )
    return {
        'corrected_path_coordinates': selected.detach(),
        'corrected_residual': selected.detach(),
        'trajectory': dense.detach(),
        'yaw': final_geometry["yaw"].detach(),
        'curvature': final_geometry["curvature"].detach(),
        'curvature_audit': final_curvature_audit.detach(),
        'first_derivative': final_geometry["first_derivative"].detach(),
        'second_derivative': final_geometry["second_derivative"].detach(),
        # cost 保留为兼容别名，含义是 task/optional-reg，不含 proposal tether。
        'cost': final_regularized_task_cost.detach(),
        'task_cost': final_components["task_cost"].detach(),
        'regularized_task_cost': final_regularized_task_cost.detach(),
        'proposal_tether': final_proposal_tether.detach(),
        'proximity_metric': proximity_metric,
        'path_correction_objective': final_expert_objective.detach(),
        'expert_objective': final_expert_objective.detach(),
        'components': {k: v.detach() for k, v in final_components.items()},
        'history': history,
        'best_strict_valid_found': best_valid_found.detach(),
        'best_accepted_found': best_accepted_found.detach(),
        'selected_accepted_candidate': best_accepted_found.detach(),
        'selected_strict_valid_candidate': (
            best_accepted_found | best_valid_found
        ).detach(),
        'best_invalid_violation_score': best_invalid_score.detach(),
        'robust_identity_mask': robust_identity_mask.detach(),
        'initial_robust_margin_normalized': (
            initial_robust_margin.detach()
        ),
    }


# Compatibility alias for earlier training and analysis scripts.
optimize_privileged_trajectories = apply_privileged_path_correction


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
    curvature_limit=SAFETY_COST_CONFIG.curvature_limit,
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

def overlay_path_mask(ax, mask, extent=MAP_BOUNDS):
    """在地图上用半透明红色标出 mask=0（禁止或未观测）区域。"""
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape={mask.shape}")
    blocked = np.ma.masked_where(mask > 0.5, np.ones_like(mask))
    ax.imshow(
        blocked,
        cmap=ListedColormap(["#ff3b30"]),
        alpha=0.32,
        extent=extent,
        origin="lower",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
        zorder=2,
    )
    # Proxy artist makes the overlay semantics explicit in the legend.
    ax.scatter(
        [],
        [],
        marker="s",
        s=70,
        facecolor="#ff3b30",
        edgecolor="#a40000",
        alpha=0.45,
        label="Mask = 0 (blocked / unobserved)",
    )


def visualize_input_mask(
    ax,
    mask,
    initial_trajectory=None,
    optimized_trajectory=None,
    extent=MAP_BOUNDS,
):
    """单独绘制二值输入 mask，并可选叠加优化前后的轨迹。"""
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape={mask.shape}")
    binary_mask = (mask > 0.5).astype(np.float32)
    image = ax.imshow(
        binary_mask,
        cmap=ListedColormap(["#d73027", "#1a9850"]),
        extent=extent,
        origin="lower",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    colorbar = plt.colorbar(image, ax=ax, ticks=[0.0, 1.0])
    colorbar.ax.set_yticklabels(
        ["0: blocked / unobserved", "1: allowed / observed"]
    )
    if initial_trajectory is not None:
        ax.plot(
            initial_trajectory[:, 0],
            initial_trajectory[:, 1],
            color="cyan",
            linestyle="--",
            linewidth=3,
            label="Initial Trajectory",
            zorder=10,
        )
    if optimized_trajectory is not None:
        ax.plot(
            optimized_trajectory[:, 0],
            optimized_trajectory[:, 1],
            color="magenta",
            linestyle="-",
            linewidth=3,
            label="Optimized Trajectory",
            zorder=11,
        )
    ax.set_title("Input Mask")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()


def visualize_terrain_trajectory(
    ax,
    trajectory,
    elev,
    nx,
    ny,
    nz,
    positions=None,
    yaws=None,
    mask=None,
    traj_label='Trajectory',
    traj_color='y',
    traj_linestyle='-',
):
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

    if mask is not None:
        overlay_path_mask(ax, mask)
    
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

def _default_phr_alm_constraints(
    traj_xy,
    start_pose,
    goal_pose,
    curvature_limit=SAFETY_COST_CONFIG.curvature_limit,
    device='cuda',
):
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
    optimizer = torch.optim.Adam([middle_cp_opt], lr=lr)
    
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

    optimizer = torch.optim.Adam([middle_cp_opt], lr=lr)

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


def run_formal_expert_standalone():
    """Run the exact Stage-2 expert collection/correction path for one context.

    The proposal sampling, endpoint-feasible context resampling, eight-particle
    source batch, privileged correction, hard acceptance, and source-target
    lineage all come from ``collect_privileged_distillation_round``.  This
    avoids the legacy 24-control-point optimizer that used to live in the
    standalone demo.
    """
    from pathlib import Path

    from dataLoader_dit import UnevenPathDataLoader
    from posterior_pipeline import (
        PrivilegedConstraintDistillationConfig,
        _require_current_mask_semantics,
        _require_demo_target_semantics,
        _require_main_method_model,
        collect_privileged_distillation_round,
        load_model,
        normalize_poses,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(
        "data/two_stage_meanflow_gauge44_v1/stage1_best.pth"
    )
    model, _, checkpoint = load_model(checkpoint_path, device)
    _require_main_method_model(model)
    _require_current_mask_semantics(checkpoint)
    _require_demo_target_semantics(checkpoint)
    model.use_gradient_checkpoint = False
    model.eval()
    print(f"Loaded current Stage 1 model: {checkpoint_path}")

    env_number = int(np.random.randint(0, 99))
    env_list = [f"env{env_number:06d}"]
    print(f"Standalone environment: {env_list[0]}")
    data_folder = str(MAP_CONFIG.dataset_root / "val")
    dataset = UnevenPathDataLoader(
        env_list,
        data_folder,
        compute_stability_map=True,
        use_precomputed_stability=True,
        compute_stability_if_missing=True,
        partial_observation=True,
        include_mask=True,
        # Visualization explicitly requests an active mask.  The expert
        # optimizer and its frozen hyperparameters are otherwise unchanged.
        p_mask=1.0,
        mask_mode="stage2_independent",
        encode_path_coordinates=True,
    )
    requested_index = 10
    requested_index = min(requested_index, len(dataset) - 1)

    expert_config = PrivilegedConstraintDistillationConfig()
    print(
        "Formal Stage 2 expert configuration: "
        f"particles={expert_config.particles_per_context}, "
        f"steps={expert_config.privileged_steps}, "
        f"lr={expert_config.privileged_lr}, "
        f"proposal_weight={expert_config.proposal_weight}"
    )
    # A formal collection round spans many contexts and records 0/8 contexts
    # as failures instead of terminating the round.  The standalone viewer
    # follows the same policy adaptively: audit each failed context, then move
    # to another context until an accepted pair is available for plotting.
    max_context_attempts = len(dataset)
    records = []
    correction_failures = []
    collect_metrics = None
    attempted_pairs = 0
    accepted_pairs = 0
    failure_category_counts = {}
    contexts_evaluated = 0
    for attempt in range(max_context_attempts):
        contexts_evaluated = attempt + 1
        candidate_index = (requested_index + attempt) % len(dataset)
        (
            attempt_records,
            attempt_failures,
            attempt_metrics,
        ) = collect_privileged_distillation_round(
            model,
            dataset,
            [candidate_index],
            distillation_round=0,
            config=expert_config,
            device=device,
            seed=10_000 + attempt,
            available_contexts=len(dataset),
        )
        attempted_pairs += int(attempt_metrics["pairs_attempted"])
        accepted_pairs += len(attempt_records)
        correction_failures.extend(attempt_failures)
        for failure in attempt_failures:
            for category in failure.get("failure_categories", []):
                failure_category_counts[category] = (
                    failure_category_counts.get(category, 0) + 1
                )
        if attempt_records:
            records = attempt_records
            collect_metrics = attempt_metrics
            break
        failed_dataset_indices = sorted(
            {
                int(failure["dataset_index"])
                for failure in attempt_failures
                if failure.get("dataset_index") is not None
            }
        )
        print(
            "Formal expert context rejected: "
            f"attempt={attempt + 1}/{max_context_attempts}, "
            f"dataset_indices={failed_dataset_indices}, "
            f"accepted=0/{expert_config.particles_per_context}; "
            "continuing exactly as multi-context Stage 2 collection does."
        )
    if not records:
        raise RuntimeError(
            "The formal Stage 2 expert produced no accepted source-target "
            f"pair after {max_context_attempts} contexts "
            f"({attempted_pairs} particles). "
            f"failure_category_counts={failure_category_counts}"
        )
    print(
        "Cumulative standalone collection audit: "
        f"contexts={contexts_evaluated}, "
        f"accepted={accepted_pairs}/{attempted_pairs}, "
        f"failed={attempted_pairs - accepted_pairs}, "
        f"failure_category_counts={failure_category_counts}"
    )

    # The training pipeline retains all accepted pairs.  A single figure needs
    # one representative pair, so select the accepted target with minimum
    # formal task cost; never display a failed correction as an expert.
    selected = min(records, key=lambda item: float(item["final_cost"]))
    dataset_index = int(selected["dataset_index"])
    context = dataset.get_item(
        dataset_index,
        mask_variant=int(selected["mask_variant"]),
        noise_seed=int(selected["mask_noise_seed"]),
        return_mask_metadata=True,
    )
    metadata = context["mask_metadata"]
    print(
        "Selected accepted expert pair: "
        f"dataset_index={dataset_index}, "
        f"particle={selected['particle_index']}, "
        f"cost={float(selected['initial_cost']):.6f}"
        f"->{float(selected['final_cost']):.6f}, "
        f"accepted_pairs={len(records)}/"
        f"{expert_config.particles_per_context}"
    )
    print(
        "Mask sampling: "
        f"source={metadata['source']}, "
        f"active={metadata['mask_active']}, "
        f"type={metadata['accepted_type']}, "
        f"masked_fraction={metadata['masked_fraction']:.4f}"
    )

    physical_start = context["start_pose"].float().to(device).unsqueeze(0)
    physical_goal = context["goal_pose"].float().to(device).unsqueeze(0)
    start_model, goal_model = normalize_poses(
        physical_start,
        physical_goal,
        model.coordinate_scale,
        device,
    )
    proposal_state = selected["proposal_residual"].to(device).unsqueeze(0)
    expert_state = selected["target_residual"].to(device).unsqueeze(0)
    with torch.no_grad():
        proposal_geometry = model.evaluate_trajectory_state(
            proposal_state,
            start_model,
            goal_model,
        )
        expert_geometry = model.evaluate_trajectory_state(
            expert_state,
            start_model,
            goal_model,
        )

    map_info = MAP_CONFIG.cost_map_info()
    stability_cost_map = context["cost_map"].to(device)
    input_mask = context["mask"].to(device)
    signed_mask_distance = build_signed_mask_distance_map(
        input_mask,
        map_info,
        device=device,
    )
    condition_ids = torch.zeros(1, dtype=torch.long, device=device)
    proposal_validity = trajectory_validity_metrics(
        proposal_geometry["position"],
        stability_cost_map,
        map_info,
        analytic_yaw=proposal_geometry["yaw"],
        analytic_curvature=proposal_geometry["curvature"],
        analytic_curvature_audit=model.audit_trajectory_state_curvature(
            proposal_state,
            start_model,
            goal_model,
        ),
        mask=input_mask,
        signed_mask_distance_map=signed_mask_distance,
        start_pose=physical_start,
        goal_pose=physical_goal,
        condition_ids=condition_ids,
    )
    expert_validity = trajectory_validity_metrics(
        expert_geometry["position"],
        stability_cost_map,
        map_info,
        analytic_yaw=expert_geometry["yaw"],
        analytic_curvature=expert_geometry["curvature"],
        analytic_curvature_audit=model.audit_trajectory_state_curvature(
            expert_state,
            start_model,
            goal_model,
        ),
        mask=input_mask,
        signed_mask_distance_map=signed_mask_distance,
        start_pose=physical_start,
        goal_pose=physical_goal,
        condition_ids=condition_ids,
    )

    def scalar(metrics, key):
        return float(metrics[key][0].detach().cpu())

    print("\nFormal hard-contract diagnostics:")
    print(
        "  strict_valid: "
        f"{bool(proposal_validity['strict_valid'][0])} -> "
        f"{bool(expert_validity['strict_valid'][0])}"
    )
    print(
        "  accepted: "
        f"{bool(proposal_validity['accepted'][0])} -> "
        f"{bool(expert_validity['accepted'][0])}"
    )
    for key in (
        "violation_max_normalized",
        "violation_integral_normalized_m",
        "min_stability_margin",
        "max_curvature",
        "composite_clearance_min",
        "path_length_ratio",
    ):
        print(
            f"  {key}: {scalar(proposal_validity, key):.6f}"
            f" -> {scalar(expert_validity, key):.6f}"
        )
    print(
        "  collection success rate: "
        f"{collect_metrics['path_correction_success_rate']:.2%}"
    )

    proposal_traj = torch.cat(
        [
            proposal_geometry["position"],
            proposal_geometry["yaw"].unsqueeze(-1),
        ],
        dim=-1,
    )[0].detach().cpu().numpy()
    expert_traj = torch.cat(
        [
            expert_geometry["position"],
            expert_geometry["yaw"].unsqueeze(-1),
        ],
        dim=-1,
    )[0].detach().cpu().numpy()
    proposal_controls = (
        proposal_geometry["control_points"][0].detach().cpu().numpy()
    )
    expert_controls = (
        expert_geometry["control_points"][0].detach().cpu().numpy()
    )
    input_mask_np = input_mask.detach().cpu().numpy()
    elevation = context["elevation"]
    normals = context["normals"]
    nx = normals[0]
    ny = normals[1]
    nz = torch.abs(normals[2])

    fig1, axes = plt.subplots(1, 3, figsize=(22, 7))
    visualize_terrain_trajectory(
        axes[0],
        proposal_traj,
        elevation,
        nx,
        ny,
        nz,
        positions=proposal_controls,
        yaws=proposal_traj[:, 2],
        mask=input_mask_np,
        traj_label="Stage 1 Proposal",
        traj_color="c",
        traj_linestyle="--",
    )
    axes[0].set_title("Stage 1 Proposal")
    visualize_terrain_trajectory(
        axes[1],
        expert_traj,
        elevation,
        nx,
        ny,
        nz,
        positions=expert_controls,
        yaws=expert_traj[:, 2],
        mask=input_mask_np,
        traj_label="Formal Stage 2 Expert",
        traj_color="m",
        traj_linestyle="-",
    )
    axes[1].set_title("Formal Stage 2 Expert")
    visualize_input_mask(
        axes[2],
        input_mask_np,
        initial_trajectory=proposal_traj,
        optimized_trajectory=expert_traj,
    )
    axes[2].set_title("Stage 2 Input Mask")
    fig1.tight_layout()

    fig2 = plt.figure(figsize=(16, 8))
    ax_before = fig2.add_subplot(1, 2, 1, projection="3d")
    ax_after = fig2.add_subplot(1, 2, 2, projection="3d")
    ax_before.plot(
        proposal_traj[:, 0],
        proposal_traj[:, 1],
        proposal_traj[:, 2],
        color="cyan",
        linestyle="--",
        linewidth=2,
        label="Stage 1 Proposal",
    )
    ax_after.plot(
        expert_traj[:, 0],
        expert_traj[:, 1],
        expert_traj[:, 2],
        color="magenta",
        linewidth=2,
        label="Formal Stage 2 Expert",
    )
    for axis, title in (
        (ax_before, "Stage 1 Proposal in SE(2)"),
        (ax_after, "Formal Stage 2 Expert in SE(2)"),
    ):
        axis.set_title(title)
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
        axis.set_zlabel("Yaw (rad)")
        axis.legend()
        axis.grid(True, linestyle="--", alpha=0.3)
    fig2.tight_layout()

    figure_dir = test_dir(
        "grad_optimizer", f"dataset_{dataset_index:06d}"
    )
    figure1_path = figure_dir / "trajectory_comparison.png"
    figure2_path = figure_dir / "se2_manifold.png"
    fig1.savefig(figure1_path, dpi=300)
    fig2.savefig(figure2_path, dpi=300)
    print(f"Saved: {figure1_path}")
    print(f"Saved: {figure2_path}")
    plt.show()


def grad_optimizer_direct_cost_forward(
    model,
    batch,
    device,
    *,
    source,
):
    """Independent grad_optimizer deployment-point cost implementation.

    This deliberately does not call ``posterior_pipeline._direct_cost_forward``.
    ``audit_direct_cost_equivalence.py`` keeps the two entrypoints locked at
    cost, gradient, optimizer-step, and multi-step output level.
    """
    map_input = batch["map"].float().to(device)
    physical_start = batch["start_pose"].float().to(device)
    physical_goal = batch["goal_pose"].float().to(device)
    cost_map = batch["cost_map"].float().to(device)
    mask = batch["mask"].float().to(device)
    count = map_input.shape[0]

    source = torch.as_tensor(
        source,
        dtype=map_input.dtype,
        device=device,
    )
    expected_shape = (count, model.num_edges, 2)
    if tuple(source.shape) != expected_shape:
        raise ValueError(
            f"Explicit source must have shape {expected_shape}, "
            f"got {tuple(source.shape)}"
        )
    source = model.project_zero_sum(source)

    def normalize_pose(pose):
        normalized = torch.zeros(
            pose.shape[0],
            4,
            dtype=pose.dtype,
            device=device,
        )
        normalized[:, :2] = pose[:, :2] / float(model.coordinate_scale)
        normalized[:, 2:] = torch.stack(
            [torch.cos(pose[:, 2]), torch.sin(pose[:, 2])],
            dim=-1,
        )
        return normalized

    start_model = normalize_pose(physical_start)
    goal_model = normalize_pose(physical_goal)
    signed_mask = build_signed_mask_distance_map(
        mask,
        MAP_CONFIG.cost_map_info(),
        device=device,
    )
    one = torch.ones(count, dtype=map_input.dtype, device=device)
    zero = torch.zeros(count, dtype=map_input.dtype, device=device)
    state = model.project_zero_sum(
        model(
            map_input,
            source,
            one,
            zero,
            start_model,
            goal_model,
        )
    )
    geometry = model.evaluate_trajectory_state(
        state,
        start_model,
        goal_model,
    )
    curvature_audit = model.audit_trajectory_state_curvature(
        state,
        start_model,
        goal_model,
    )
    _, components = privileged_planning_cost(
        geometry["position"],
        physical_start,
        physical_goal,
        cost_map,
        MAP_CONFIG.cost_map_info(),
        analytic_yaw=geometry["yaw"],
        analytic_curvature=geometry["curvature"],
        analytic_first_derivative=geometry["first_derivative"],
        analytic_second_derivative=geometry["second_derivative"],
        mask=mask,
        signed_mask_distance_map=signed_mask,
        return_per_sample=True,
        return_components=True,
    )
    return {
        "state": state,
        "source": source,
        "geometry": geometry,
        "curvature_audit": curvature_audit,
        "components": components,
        "task_cost": components["task_cost"],
        "start_pose": physical_start,
        "goal_pose": physical_goal,
        "cost_map": cost_map,
        "mask": mask,
        "signed_mask_distance": signed_mask,
        "condition_ids": torch.arange(
            count,
            dtype=torch.long,
            device=device,
        ),
        "contexts": count,
        "sources_per_context": 1,
    }


def build_direct_cost_standalone_parser():
    """Build the lightweight direct-cost diagnostic CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Directly fine-tune the deployed Path MeanFlow one-step output "
            "with differentiable privileged path cost."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default="data/two_stage_meanflow_gauge44_v1/stage1_best.pth",
    )
    parser.add_argument(
        "--output-dir",
        default="tests/audits/grad_optimizer_direct_cost_preview",
    )
    parser.add_argument("--updates", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--early-stop-cost",
        type=float,
        default=1e-4,
        help="Stop after the fixed trained paths are strict and below this cost.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=2,
        help="Required consecutive admitted iterates; set 0 to disable.",
    )
    parser.add_argument("--train-contexts", type=int, default=1)
    parser.add_argument("--validation-contexts", type=int, default=4)
    parser.add_argument("--sources-per-context", type=int, default=1)
    parser.add_argument(
        "--pair-attempts",
        type=int,
        default=8,
        help=(
            "Maximum endpoint-feasible condition/source pairs audited to "
            "find one that the formal expert can make strict-valid."
        ),
    )
    parser.add_argument("--expert-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--mask-seed", type=int, default=2026)
    parser.add_argument(
        "--min-masked-fraction",
        type=float,
        default=0.02,
        help="Only use visibly active masks with at least this blocked fraction.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cpu', or a CUDA device such as 'cuda:0'.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving it.",
    )
    return parser


def run_direct_cost_standalone(argv=None):
    """Run a controlled, small direct-cost model fine-tuning diagnostic.

    The expert is used only to admit and visualize one demonstrably correctable
    condition/source pair.  Network training never consumes its target.  It
    updates Path MeanFlow only from privileged differentiable task cost at the
    deployed one-step endpoint ``t=1, r=0``.  Disjoint validation contexts
    expose immediate regression or a preliminary held-out gain.
    Unlike distribution-level Stage-2 training, this admission deliberately
    reuses exactly the same source on every update so its primary question
    matches path correction: can the network move one fixed proposal downhill?
    """
    import json
    from pathlib import Path

    from torch.utils.data import DataLoader
    from dataLoader_dit import mask_start_goal_connected

    from posterior_pipeline import (
        _require_current_mask_semantics,
        _require_demo_target_semantics,
        _require_main_method_model,
        evaluate_direct_cost_stage2,
        load_model,
        make_partial_dataset,
        normalize_poses,
    )

    args = build_direct_cost_standalone_parser().parse_args(argv)
    positive = {
        "updates": args.updates,
        "lr": args.lr,
        "train-contexts": args.train_contexts,
        "validation-contexts": args.validation_contexts,
        "sources-per-context": args.sources_per_context,
        "pair-attempts": args.pair_attempts,
        "expert-steps": args.expert_steps,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.grad_clip < 0.0:
        raise ValueError("--grad-clip cannot be negative")
    if args.early_stop_cost < 0.0:
        raise ValueError("--early-stop-cost cannot be negative")
    if args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience cannot be negative")
    if args.train_contexts != 1 or args.sources_per_context != 1:
        raise ValueError(
            "This paired path-level admission requires exactly "
            "--train-contexts 1 and --sources-per-context 1"
        )
    if not 0.0 <= args.min_masked_fraction < 1.0:
        raise ValueError("--min-masked-fraction must be in [0,1)")

    if args.device == "auto":
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, _, checkpoint = load_model(checkpoint_path, device)
    _require_main_method_model(model)
    _require_current_mask_semantics(checkpoint)
    _require_demo_target_semantics(checkpoint)
    model.use_gradient_checkpoint = False
    # Match deployed one-step inference: dropout is disabled, gradients are not.
    model.eval()
    print(f"Loaded Stage 1 model: {checkpoint_path}")
    print(
        "Direct-cost preview: "
        f"updates={args.updates}, lr={args.lr:g}, "
        f"grad_clip={args.grad_clip:g}, "
        f"train_contexts={args.train_contexts}, "
        f"validation_contexts={args.validation_contexts}, "
        f"sources/context={args.sources_per_context}"
    )

    train_dataset, train_envs = make_partial_dataset(
        str(MAP_CONFIG.dataset_root),
        "train",
        compute_stability_map=True,
        mask_seed=args.mask_seed,
        p_mask=1.0,
        mask_mode="stage2_independent",
        vehicle_radius_meters=SAFETY_COST_CONFIG.vehicle_radius_meters,
    )
    validation_dataset, validation_envs = make_partial_dataset(
        str(MAP_CONFIG.dataset_root),
        "val",
        compute_stability_map=True,
        mask_seed=args.mask_seed,
        p_mask=1.0,
        mask_mode="stage2_independent",
        vehicle_radius_meters=SAFETY_COST_CONFIG.vehicle_radius_meters,
    )

    model_keys = ("map", "start_pose", "goal_pose", "cost_map", "mask")

    def collect_visible_contexts(
        dataset,
        count,
        selection_seed,
        *,
        require_endpoint_feasible,
    ):
        order = torch.randperm(
            len(dataset),
            generator=torch.Generator().manual_seed(selection_seed),
        ).tolist()
        selected = []
        selected_indices = []
        raw_items = []
        for dataset_index in order:
            item = dataset.get_item(
                dataset_index,
                mask_variant=1,
                noise_seed=selection_seed + 100_000 + dataset_index,
                return_mask_metadata=True,
            )
            metadata = item["mask_metadata"]
            masked_fraction = float(metadata["masked_fraction"])
            if (
                not bool(metadata["mask_active"])
                or masked_fraction < args.min_masked_fraction
            ):
                continue
            if require_endpoint_feasible:
                feasibility = endpoint_stability_feasibility(
                    item["start_pose"],
                    item["goal_pose"],
                    item["cost_map"],
                    MAP_CONFIG.cost_map_info(),
                )
                if not feasibility["feasible"]:
                    continue
                if not mask_start_goal_connected(
                    item["mask"],
                    item["start_pose"],
                    item["goal_pose"],
                ):
                    continue
            selected.append(
                {key: item[key].detach().cpu() for key in model_keys}
            )
            raw_items.append(item)
            selected_indices.append(int(dataset_index))
            if len(selected) == count:
                break
        if len(selected) != count:
            raise RuntimeError(
                f"Only found {len(selected)}/{count} contexts with an active "
                f"mask fraction >= {args.min_masked_fraction:.3f}"
            )
        return selected, raw_items, selected_indices

    train_candidates, train_candidate_raw, train_candidate_indices = (
        collect_visible_contexts(
            train_dataset,
            args.pair_attempts,
            args.seed + 1_000,
            require_endpoint_feasible=True,
        )
    )
    validation_items, validation_raw, validation_indices = (
        collect_visible_contexts(
            validation_dataset,
            args.validation_contexts,
            args.seed + 2_000,
            require_endpoint_feasible=False,
        )
    )
    print(
        "Selected validation masks: "
        + ", ".join(
            f"{index}:{float(item['mask_metadata']['masked_fraction']):.1%}"
            for index, item in zip(validation_indices, validation_raw)
        )
    )

    validation_loader = DataLoader(
        validation_items,
        batch_size=len(validation_items),
        shuffle=False,
        num_workers=0,
    )
    fixed_train_source_seed = args.seed + 20_000
    validation_seed = args.seed + 30_000

    def evaluate_split(loader, source_seed):
        return evaluate_direct_cost_stage2(
            model,
            loader,
            device,
            sources_per_context=args.sources_per_context,
            seed=source_seed,
        )

    def batch_from_item(item):
        return next(
            iter(
                DataLoader(
                    [item],
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                )
            )
        )

    @torch.no_grad()
    def snapshot_from_output(output):
        validity = trajectory_validity_metrics(
            output["geometry"]["position"],
            output["cost_map"],
            MAP_CONFIG.cost_map_info(),
            analytic_yaw=output["geometry"]["yaw"],
            analytic_curvature=output["geometry"]["curvature"],
            analytic_curvature_audit=output["curvature_audit"],
            mask=output["mask"],
            signed_mask_distance_map=output["signed_mask_distance"],
            start_pose=output["start_pose"],
            goal_pose=output["goal_pose"],
            condition_ids=output["condition_ids"],
        )
        return {
            "trajectory": torch.cat(
                [
                    output["geometry"]["position"],
                    output["geometry"]["yaw"].unsqueeze(-1),
                ],
                dim=-1,
            )[0].detach().cpu().numpy(),
            "control_points": output["geometry"]["control_points"][0]
            .detach()
            .cpu()
            .numpy(),
            "state": output["state"][0].detach().cpu().numpy(),
            "task_cost": float(output["task_cost"][0]),
            "strict_valid": bool(validity["strict_valid"][0]),
            "forbidden_ok": bool(validity["forbidden_region_ok"][0]),
            "stability_ok": bool(validity["stability_ok"][0]),
            "curvature_ok": bool(validity["curvature_ok"][0]),
            "violation_max_normalized": float(
                validity["violation_max_normalized"][0]
            ),
            "violation_integral_normalized_m": float(
                validity["violation_integral_normalized_m"][0]
            ),
        }

    @torch.no_grad()
    def fixed_snapshot(item, source):
        output = grad_optimizer_direct_cost_forward(
            model,
            batch_from_item(item),
            device,
            source=source,
        )
        return snapshot_from_output(output)

    def metrics_from_snapshot(snapshot):
        strict = float(snapshot["strict_valid"])
        return {
            "task_cost": float(snapshot["task_cost"]),
            "safe_at_1": strict,
            "safe_at_k": strict,
            "strict_valid_rate": strict,
            "violation_max_normalized": float(
                snapshot["violation_max_normalized"]
            ),
            "violation_integral_normalized_m": float(
                snapshot["violation_integral_normalized_m"]
            ),
            "forbidden_ok_rate": float(snapshot["forbidden_ok"]),
            "stability_ok_rate": float(snapshot["stability_ok"]),
            "curvature_ok_rate": float(snapshot["curvature_ok"]),
        }

    # Admit an exact pair only when the formal path-space expert can turn that
    # same Stage-1 proposal into a strict-valid path.  The target is retained
    # solely for the audit/plot below and is never used by the network loss.
    pair_audits = []
    train_items = None
    train_raw = None
    train_indices = None
    fixed_train_source = None
    baseline_train_snapshot = None
    expert_snapshot = None
    train_batch = None
    for attempt, (candidate, raw_item, dataset_index) in enumerate(
        zip(
            train_candidates,
            train_candidate_raw,
            train_candidate_indices,
        )
    ):
        pair_seed = fixed_train_source_seed + attempt
        source_cpu = torch.randn(
            1,
            model.num_edges,
            2,
            generator=torch.Generator().manual_seed(pair_seed),
        )
        candidate_source = model.project_zero_sum(source_cpu.to(device))
        candidate_batch = batch_from_item(candidate)
        with torch.no_grad():
            proposal_output = grad_optimizer_direct_cost_forward(
                model,
                candidate_batch,
                device,
                source=candidate_source,
            )
            proposal_snapshot = snapshot_from_output(proposal_output)
        corrected = apply_privileged_path_correction(
            proposal_output["state"].detach(),
            proposal_output["start_pose"],
            proposal_output["goal_pose"],
            proposal_output["cost_map"],
            MAP_CONFIG.cost_map_info(),
            mask=proposal_output["mask"],
            signed_mask_distance_map=(
                proposal_output["signed_mask_distance"]
            ),
            iterations=args.expert_steps,
            lr=5e-2,
            proposal_weight=2e-3,
        )
        corrected_validity = trajectory_validity_metrics(
            corrected["trajectory"],
            proposal_output["cost_map"],
            MAP_CONFIG.cost_map_info(),
            analytic_yaw=corrected["yaw"],
            analytic_curvature=corrected["curvature"],
            analytic_curvature_audit=corrected["curvature_audit"],
            mask=proposal_output["mask"],
            signed_mask_distance_map=(
                proposal_output["signed_mask_distance"]
            ),
            start_pose=proposal_output["start_pose"],
            goal_pose=proposal_output["goal_pose"],
            condition_ids=torch.zeros(
                1,
                dtype=torch.long,
                device=device,
            ),
        )
        expert_strict = bool(corrected_validity["strict_valid"][0])
        expert_accepted = bool(corrected_validity["accepted"][0])
        audit = {
            "attempt": attempt + 1,
            "dataset_index": int(dataset_index),
            "source_seed": int(pair_seed),
            "proposal_cost": float(proposal_output["task_cost"][0]),
            "proposal_strict": bool(proposal_snapshot["strict_valid"]),
            "expert_cost": float(corrected["task_cost"][0]),
            "expert_strict": expert_strict,
            "expert_accepted": expert_accepted,
        }
        pair_audits.append(audit)
        print(
            "Paired expert audit "
            f"{attempt + 1}/{args.pair_attempts}: "
            f"index={dataset_index}, "
            f"cost={audit['proposal_cost']:.6f}"
            f"->{audit['expert_cost']:.6f}, "
            f"strict={audit['proposal_strict']}->{expert_strict}, "
            f"accepted={expert_accepted}"
        )
        if (
            proposal_snapshot["strict_valid"]
            or not expert_strict
            or not expert_accepted
        ):
            continue

        physical_start = proposal_output["start_pose"]
        physical_goal = proposal_output["goal_pose"]
        start_model, goal_model = normalize_poses(
            physical_start,
            physical_goal,
            model.coordinate_scale,
            device,
        )
        with torch.no_grad():
            corrected_geometry = model.evaluate_trajectory_state(
                corrected["corrected_residual"],
                start_model,
                goal_model,
            )
        expert_snapshot = {
            "trajectory": torch.cat(
                [
                    corrected_geometry["position"],
                    corrected_geometry["yaw"].unsqueeze(-1),
                ],
                dim=-1,
            )[0].detach().cpu().numpy(),
            "control_points": corrected_geometry["control_points"][0]
            .detach()
            .cpu()
            .numpy(),
            "state": corrected["corrected_residual"][0]
            .detach()
            .cpu()
            .numpy(),
            "task_cost": float(corrected["task_cost"][0]),
            "strict_valid": expert_strict,
            "forbidden_ok": bool(
                corrected_validity["forbidden_region_ok"][0]
            ),
            "stability_ok": bool(corrected_validity["stability_ok"][0]),
            "curvature_ok": bool(corrected_validity["curvature_ok"][0]),
            "violation_max_normalized": float(
                corrected_validity["violation_max_normalized"][0]
            ),
            "violation_integral_normalized_m": float(
                corrected_validity[
                    "violation_integral_normalized_m"
                ][0]
            ),
        }
        train_items = [candidate]
        train_raw = [raw_item]
        train_indices = [int(dataset_index)]
        fixed_train_source = candidate_source.detach()
        baseline_train_snapshot = proposal_snapshot
        train_batch = candidate_batch
        fixed_train_source_seed = pair_seed
        break

    if train_items is None:
        raise RuntimeError(
            "Could not find a Stage-1-invalid pair that the formal expert "
            f"makes strict-valid in {args.pair_attempts} endpoint-feasible "
            f"attempts. audits={pair_audits}"
        )
    selected_endpoint_feasibility = endpoint_stability_feasibility(
        train_raw[0]["start_pose"],
        train_raw[0]["goal_pose"],
        train_raw[0]["cost_map"],
        MAP_CONFIG.cost_map_info(),
    )
    print(
        "Selected paired admission: "
        f"dataset_index={train_indices[0]}, "
        f"source_seed={fixed_train_source_seed}, "
        f"mask_blocked="
        f"{float(train_raw[0]['mask_metadata']['masked_fraction']):.1%}, "
        f"endpoint_margins=("
        f"{selected_endpoint_feasibility['start_best_stability_margin']:.4f}, "
        f"{selected_endpoint_feasibility['goal_best_stability_margin']:.4f})"
    )

    baseline_train = metrics_from_snapshot(baseline_train_snapshot)
    baseline_validation = evaluate_split(
        validation_loader,
        validation_seed,
    )
    validation_source_cpu = torch.randn(
        1,
        model.num_edges,
        2,
        generator=torch.Generator().manual_seed(args.seed + 40_000),
    )
    fixed_validation_source = model.project_zero_sum(
        validation_source_cpu.to(device)
    )
    baseline_validation_snapshot = fixed_snapshot(
        validation_items[0],
        fixed_validation_source,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    cost_history = []
    gradient_history = []
    admitted_streak = 0
    optimizer_steps = 0
    effective_admission_cost = max(
        float(args.early_stop_cost),
        1.05 * float(expert_snapshot["task_cost"]),
    )
    print(
        "Direct admission threshold: strict-valid and "
        f"task_cost <= {effective_admission_cost:.6f} "
        "(no worse than 1.05x the paired expert, with configured floor)."
    )
    for update in range(1, args.updates + 1):
        optimizer.zero_grad(set_to_none=True)
        output = grad_optimizer_direct_cost_forward(
            model,
            train_batch,
            device,
            # The exact tensor admitted by the paired expert audit is reused;
            # no device-specific RNG reconstruction is involved.
            source=fixed_train_source,
        )
        loss = output["task_cost"].mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"Non-finite task cost at update {update}"
            )
        loss.backward()
        if args.grad_clip > 0.0:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )
        else:
            squared = [
                parameter.grad.detach().square().sum()
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            gradient_norm = torch.sqrt(torch.stack(squared).sum())
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError(
                f"Non-finite parameter gradient at update {update}"
            )
        cost_history.append(float(loss.detach()))
        gradient_history.append(float(gradient_norm.detach()))
        current_validity = trajectory_validity_metrics(
            output["geometry"]["position"].detach(),
            output["cost_map"],
            MAP_CONFIG.cost_map_info(),
            analytic_yaw=output["geometry"]["yaw"].detach(),
            analytic_curvature=output["geometry"]["curvature"].detach(),
            analytic_curvature_audit=output["curvature_audit"].detach(),
            mask=output["mask"],
            signed_mask_distance_map=output["signed_mask_distance"],
            start_pose=output["start_pose"],
            goal_pose=output["goal_pose"],
            condition_ids=output["condition_ids"],
        )
        admitted = (
            bool(current_validity["strict_valid"].all())
            and cost_history[-1] <= effective_admission_cost
        )
        admitted_streak = admitted_streak + 1 if admitted else 0
        print(
            f"update={update:03d}/{args.updates} "
            f"train_cost={cost_history[-1]:.6f} "
            f"grad_preclip={gradient_history[-1]:.3f} "
            f"strict={bool(current_validity['strict_valid'].all())}"
        )
        if (
            args.early_stop_patience > 0
            and admitted_streak >= args.early_stop_patience
        ):
            print(
                "Fixed-source admission reached for "
                f"{admitted_streak} consecutive updates; stopping early."
            )
            break
        optimizer.step()
        optimizer_steps += 1

    final_train_snapshot = fixed_snapshot(
        train_items[0],
        fixed_train_source,
    )
    final_train = metrics_from_snapshot(final_train_snapshot)
    final_validation = evaluate_split(
        validation_loader,
        validation_seed,
    )
    final_validation_snapshot = fixed_snapshot(
        validation_items[0],
        fixed_validation_source,
    )

    def print_comparison(name, before, after):
        print(f"\n{name} fixed evaluation:")
        for key, format_spec in (
            ("task_cost", ".6f"),
            ("safe_at_1", ".2%"),
            ("safe_at_k", ".2%"),
            ("strict_valid_rate", ".2%"),
            ("violation_max_normalized", ".6f"),
            ("violation_integral_normalized_m", ".6f"),
            ("forbidden_ok_rate", ".2%"),
            ("stability_ok_rate", ".2%"),
            ("curvature_ok_rate", ".2%"),
        ):
            print(
                f"  {key}: "
                f"{format(before[key], format_spec)} -> "
                f"{format(after[key], format_spec)}"
            )

    print_comparison("TRAIN", baseline_train, final_train)
    print_comparison(
        "INDEPENDENT VALIDATION",
        baseline_validation,
        final_validation,
    )
    print("\nFixed trained trajectory (same condition and source):")
    for key in (
        "task_cost",
        "strict_valid",
        "violation_max_normalized",
        "violation_integral_normalized_m",
    ):
        print(
            f"  {key}: {baseline_train_snapshot[key]} -> "
            f"{final_train_snapshot[key]}"
        )
    state_drift_rms = float(
        np.sqrt(
            np.mean(
                (
                    final_train_snapshot["state"]
                    - baseline_train_snapshot["state"]
                )
                ** 2
            )
        )
    )
    path_drift_rms_m = float(
        np.sqrt(
            np.mean(
                (
                    final_train_snapshot["trajectory"][:, :2]
                    - baseline_train_snapshot["trajectory"][:, :2]
                )
                ** 2
            )
        )
    )
    print(f"  state_drift_rms: {state_drift_rms:.6f}")
    print(f"  path_drift_rms_m: {path_drift_rms_m:.6f}")

    selected_raw = train_raw[0]
    mask_np = selected_raw["mask"].detach().cpu().numpy()
    normals = selected_raw["normals"]
    elevation = selected_raw["elevation"]
    nx, ny = normals[0], normals[1]
    nz = torch.abs(normals[2])
    fig, axes = plt.subplots(2, 2, figsize=(18, 15))
    visualize_terrain_trajectory(
        axes[0, 0],
        baseline_train_snapshot["trajectory"],
        elevation,
        nx,
        ny,
        nz,
        positions=baseline_train_snapshot["control_points"],
        yaws=baseline_train_snapshot["trajectory"][:, 2],
        mask=mask_np,
        traj_label="Stage 1",
        traj_color="c",
        traj_linestyle="--",
    )
    axes[0, 0].set_title(
        "Stage 1 | "
        f"cost={baseline_train_snapshot['task_cost']:.3f}, "
        f"strict={baseline_train_snapshot['strict_valid']}"
    )
    visualize_terrain_trajectory(
        axes[0, 1],
        final_train_snapshot["trajectory"],
        elevation,
        nx,
        ny,
        nz,
        positions=final_train_snapshot["control_points"],
        yaws=final_train_snapshot["trajectory"][:, 2],
        mask=mask_np,
        traj_label="Direct Cost Fine-Tuned",
        traj_color="m",
        traj_linestyle="-",
    )
    axes[0, 1].set_title(
        "Direct Cost | "
        f"cost={final_train_snapshot['task_cost']:.3f}, "
        f"strict={final_train_snapshot['strict_valid']}"
    )
    axes[0, 1].plot(
        expert_snapshot["trajectory"][:, 0],
        expert_snapshot["trajectory"][:, 1],
        color="#7CFC00",
        linestyle=":",
        linewidth=3,
        label=(
            "Formal Expert Reference "
            f"(cost={expert_snapshot['task_cost']:.3f})"
        ),
        zorder=25,
    )
    axes[0, 1].legend()
    visualize_input_mask(
        axes[1, 0],
        mask_np,
        initial_trajectory=baseline_train_snapshot["trajectory"],
        optimized_trajectory=final_train_snapshot["trajectory"],
    )
    axes[1, 0].plot(
        expert_snapshot["trajectory"][:, 0],
        expert_snapshot["trajectory"][:, 1],
        color="#7CFC00",
        linestyle=":",
        linewidth=3,
        label="Formal Expert Reference",
        zorder=12,
    )
    axes[1, 0].legend()
    axes[1, 0].set_title(
        "Fixed Trained Input Mask | "
        f"blocked={float(selected_raw['mask_metadata']['masked_fraction']):.1%}"
    )
    update_axis = np.arange(1, len(cost_history) + 1)
    axes[1, 1].plot(
        update_axis,
        cost_history,
        color="#1f77b4",
        marker="o",
        label="Train task cost",
    )
    axes[1, 1].set_xlabel("Network update")
    axes[1, 1].set_ylabel("Privileged task cost")
    axes[1, 1].axhline(
        expert_snapshot["task_cost"],
        color="#2ca02c",
        linestyle=":",
        linewidth=2,
        label="Formal expert task cost",
    )
    axes[1, 1].grid(True, linestyle="--", alpha=0.3)
    gradient_axis = axes[1, 1].twinx()
    gradient_axis.plot(
        update_axis,
        gradient_history,
        color="#d62728",
        marker="x",
        alpha=0.75,
        label="Gradient norm (pre-clip)",
    )
    gradient_axis.set_ylabel("Gradient norm (pre-clip)")
    axes[1, 1].set_title("Direct Cost Optimization")
    handles_a, labels_a = axes[1, 1].get_legend_handles_labels()
    handles_b, labels_b = gradient_axis.get_legend_handles_labels()
    axes[1, 1].legend(handles_a + handles_b, labels_a + labels_b)
    fig.tight_layout()

    figure_path = output_dir / "direct_cost_preview.png"
    metrics_path = output_dir / "direct_cost_preview_metrics.json"
    fig.savefig(figure_path, dpi=220)
    report = {
        "method": "direct_onestep_privileged_task_cost_no_anchor_no_replay",
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "updates": args.updates,
        "updates_completed": len(cost_history),
        "optimizer_steps_completed": optimizer_steps,
        "learning_rate": args.lr,
        "grad_clip": args.grad_clip,
        "early_stop_cost": args.early_stop_cost,
        "effective_admission_cost": effective_admission_cost,
        "early_stop_patience": args.early_stop_patience,
        "sources_per_context": args.sources_per_context,
        "train_environments": train_envs,
        "validation_environments": validation_envs,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "train_masked_fractions": [
            float(item["mask_metadata"]["masked_fraction"])
            for item in train_raw
        ],
        "validation_masked_fractions": [
            float(item["mask_metadata"]["masked_fraction"])
            for item in validation_raw
        ],
        "baseline_train": baseline_train,
        "final_train": final_train,
        "baseline_validation": baseline_validation,
        "final_validation": final_validation,
        "baseline_selected_train": {
            key: value
            for key, value in baseline_train_snapshot.items()
            if key not in {"trajectory", "control_points", "state"}
        },
        "final_selected_train": {
            key: value
            for key, value in final_train_snapshot.items()
            if key not in {"trajectory", "control_points", "state"}
        },
        "baseline_selected_validation": {
            key: value
            for key, value in baseline_validation_snapshot.items()
            if key not in {"trajectory", "control_points", "state"}
        },
        "final_selected_validation": {
            key: value
            for key, value in final_validation_snapshot.items()
            if key not in {"trajectory", "control_points", "state"}
        },
        "fixed_train_source_seed": fixed_train_source_seed,
        "fixed_train_source": (
            fixed_train_source.detach().cpu().tolist()
        ),
        "paired_expert_audits": pair_audits,
        "selected_endpoint_feasibility": (
            selected_endpoint_feasibility
        ),
        "expert_reference": {
            key: value
            for key, value in expert_snapshot.items()
            if key not in {"trajectory", "control_points", "state"}
        },
        "state_drift_rms": state_drift_rms,
        "path_drift_rms_m": path_drift_rms_m,
        "cost_history": cost_history,
        "gradient_norm_pre_clip_history": gradient_history,
        "training_uses_expert_target": False,
        "diagnostic_uses_expert_reference": True,
        "uses_expert": False,
        "uses_replay": False,
        "uses_stage1_anchor": False,
        "inference_uses_privileged_map": False,
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"\nSaved figure: {figure_path}")
    print(f"Saved metrics: {metrics_path}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)
    return report


if __name__ == "__main__":
    run_direct_cost_standalone()


# Retained only as a historical diagnostic reference.  Direct execution uses
# ``run_direct_cost_standalone`` above, never this legacy control-point path.
if False and __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- 0. 加载地形数据 ---
    # from dataLoader_uneven import UnevenPathDataLoader
    from dataLoader_dit import UnevenPathDataLoader
    # env_list = ['env000010']
    envNum = np.random.randint(0, 99)
    env_list = [f'env{envNum:06d}']
    dataFolder = str(MAP_CONFIG.dataset_root / 'val')
    # dataset = UnevenPathDataLoader(env_list, dataFolder)
    # Standalone demo follows the current four-channel Stage-1 input contract:
    # masked normal map plus an explicit mask channel.  Force mask activation
    # here so repeated visualization runs cannot select the Bernoulli
    # "complete observation" branch and produce an all-green mask.
    dataset = UnevenPathDataLoader(
        env_list,
        dataFolder,
        compute_stability_map=True,
        use_precomputed_stability=True,
        compute_stability_if_missing=True,
        partial_observation=True,
        include_mask=True,
        p_mask=1.0,
    )
    requested_path_index = 10
    path_index = min(requested_path_index, len(dataset) - 1)
    if path_index != requested_path_index:
        print(
            f"Requested path index {requested_path_index} exceeds dataset size "
            f"{len(dataset)}; using {path_index} instead."
        )
    sample = dataset.get_item(
        path_index,
        return_mask_metadata=True,
    )
    if sample is None:
        raise ValueError(f"Sample at index {path_index} is invalid.")
    print("Loaded path index:", path_index)
    mask_metadata = sample["mask_metadata"]
    print(
        "Mask sampling: "
        f"source={mask_metadata['source']}, "
        f"active={mask_metadata['mask_active']}, "
        f"type={mask_metadata['accepted_type']}, "
        f"masked_fraction={mask_metadata['masked_fraction']:.4f}"
    )
    
    nx = sample['map'][0, :, :].to(device)
    ny = sample['map'][1, :, :].to(device)
    nz = sample['map'][2, :, :].to(device)
    input_mask = sample['map'][3, :, :].to(device)
    
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
    # 生成当前 44D Path MeanFlow 模型的预测轨迹。旧 data/sim 配置包含
    # d_k/d_v/n_path_steps 等 diffusion-era 参数，并对应 25×2、三通道、
    # radial checkpoint；它不能迁移到当前 22×2 一阶边界表示。
    from pathlib import Path
    from posterior_pipeline import (
        _require_current_mask_semantics,
        _require_demo_target_semantics,
        _require_main_method_model,
        load_model,
    )
    
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

    checkpoint_path = Path(
        "data/two_stage_meanflow_gauge44_v1/stage1_best.pth"
    )
    model, _, checkpoint = load_model(checkpoint_path, device)
    _require_main_method_model(model)
    _require_current_mask_semantics(checkpoint)
    _require_demo_target_semantics(checkpoint)
    model.use_gradient_checkpoint = False
    model.eval()
    print(f"Loaded current Stage 1 model: {checkpoint_path}")
    
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
                                  map_input=sample["map"].unsqueeze(0).to(device),
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
    # The current representation still decodes 26 complete B-spline controls,
    # while its learned free state is 22×2.  This standalone optimizer operates
    # on the 24 controls between the fixed endpoint positions.
    expected_middle_points = int(base_model.num_control_points) - 2

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
    
    initial_geometry = bspline_layer.evaluate_geometry(control_points)
    init_xy = initial_geometry["position"]
    initial_traj = torch.cat(
        [init_xy, initial_geometry["yaw"].unsqueeze(2)],
        dim=2,
    )
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
    optimized_geometry = bspline_layer.evaluate_geometry(control_points)
    opt_xy = optimized_geometry["position"]
    optimized_traj = torch.cat(
        [opt_xy, optimized_geometry["yaw"].unsqueeze(2)],
        dim=2,
    )

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
    metrics_before = evaluator.evaluate_trajectory(
        initial_traj[0].to(device),
        control_points=torch.cat(
            [
                control_points.new_tensor(initial_control_points_np),
                torch.zeros_like(control_points[0, :, :1]),
            ],
            dim=1,
        ),
        analytic_first_derivative=initial_geometry["first_derivative"][0],
        analytic_second_derivative=initial_geometry["second_derivative"][0],
        analytic_curvature=initial_geometry["curvature"][0],
    )
    
    # 评估优化后的轨迹（optimized_traj 已包含 yaw -> (B,N,3)）
    metrics_after = evaluator.evaluate_trajectory(
        optimized_traj[0].to(device),
        control_points=torch.cat(
            [control_points[0], torch.zeros_like(control_points[0, :, :1])],
            dim=1,
        ),
        analytic_first_derivative=optimized_geometry["first_derivative"][0],
        analytic_second_derivative=optimized_geometry["second_derivative"][0],
        analytic_curvature=optimized_geometry["curvature"][0],
    )
    
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
    gt_control_points_np, _, _ = fit_bspline_least_squares(
        np.asarray(trajectory)[:, :2],
        num_control_points=control_points.shape[1],
        degree=bspline_layer.degree,
    )
    gt_control_points = torch.as_tensor(
        gt_control_points_np,
        dtype=control_points.dtype,
        device=device,
    ).unsqueeze(0)
    gt_geometry = bspline_layer.evaluate_geometry(gt_control_points)
    gt_trajectory_tensor = torch.cat(
        [
            gt_geometry["position"],
            gt_geometry["yaw"].unsqueeze(2),
        ],
        dim=2,
    )[0]
    metrics_gt = evaluator.evaluate_trajectory(
        gt_trajectory_tensor,
        control_points=torch.cat(
            [
                gt_control_points[0],
                torch.zeros_like(gt_control_points[0, :, :1]),
            ],
            dim=1,
        ),
        analytic_first_derivative=gt_geometry["first_derivative"][0],
        analytic_second_derivative=gt_geometry["second_derivative"][0],
        analytic_curvature=gt_geometry["curvature"][0],
    )
    
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

    input_mask_np = input_mask.detach().cpu().numpy()
    masked_fraction = float(np.mean(input_mask_np <= 0.5))
    print(
        f"Input mask: {masked_fraction * 100:.2f}% blocked/unobserved, "
        f"{(1.0 - masked_fraction) * 100:.2f}% allowed/observed"
    )

    # --- Figure 1: 优化前、优化后以及模型实际接收的二值 mask ---
    fig1, axes1 = plt.subplots(1, 3, figsize=(22, 7))

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
        mask=input_mask_np,
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
        mask=input_mask_np,
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

    visualize_input_mask(
        axes1[2],
        input_mask_np,
        initial_trajectory=initial_plot_np,
        optimized_trajectory=optimized_plot_np,
    )
    fig1.tight_layout()

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
    figure_dir = test_dir("grad_optimizer", f"path_{path_index:06d}")
    fig1.savefig(figure_dir / "trajectory_comparison.png", dpi=300)
    fig2.savefig(figure_dir / "se2_manifold.png", dpi=300)
    # fig3.savefig(f'figure_3_cost_convergence_{path_index}.png', dpi=300)
    
    plt.show()

"""
dataLoader_uneven.py - MPT路径规划数据加载器模块
"""

# 【核心依赖库导入】
import torch  # PyTorch深度学习框架：张量计算和自动微分
from torch.utils.data import Dataset  # 数据集基类：提供数据加载的标准接口

import skimage.io  # 图像IO操作：读取地图图像文件
import pickle  # 序列化库：加载路径数据文件
import numpy as np  # 数值计算库：数组操作和数学计算
from functools import lru_cache
import hashlib

import os  # 操作系统接口：文件和目录操作
from os import path as osp  # 路径操作：文件路径处理
from einops import rearrange  # 张量重排：高效的维度操作

from torch.nn.utils.rnn import pad_sequence  # 序列填充：处理变长序列的批处理

from utils import geom2pix  # 坐标转换工具：几何坐标到像素坐标的转换
from relative_motion_utils import trajectory_to_relative_motion  # 相对运动工具：轨迹转换为相对运动表示
from bspline_utils import (
    create_bspline_basis_matrix,
    reconstruct_from_control_points,
)
from boundary_constrained_path import (
    BoundaryConstrainedPathRepresentation,
)
from map_config import (
    DENSE_TRAJECTORY_POINTS,
    MAP_BOUNDS,
    MAP_CONFIG,
    MAP_GRID_SIZE,
    MAP_HALF_EXTENT,
    MAP_RESOLUTION,
    MAP_YAW_BINS,
    SAFETY_COST_CONFIG,
)
from uav_mask import generate_uav_observation_mask

# 添加兼容性处理
import sys
sys.modules['numpy._core'] = np
sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath
sys.modules['numpy._core.multiarray'] = np.core.multiarray

# 【全局参数配置】地图物理尺度统一来自 map_config.py
input_size = MAP_GRID_SIZE
output_grid = 12
anchor_spacing = 8
boundary_offset = 6

map_size = MAP_CONFIG.map_shape
receptive_field = 38   # 感受野大小：每个锚点影响的像素范围 TODO
res = MAP_RESOLUTION

# 【理论最大正样本数自动计算】
# 感受野和锚点间距都由像素数乘 MAP_RESOLUTION 得到。
# 当前 dataset20 中分别为 38*0.2=7.6 米、8*0.2=1.6 米。
# 每个轴向最大锚点数：ceil(7.6 / 1.6) = 5个
# 理论最大正样本数：5 × 5 = 25个
import math
receptive_field_size = receptive_field * res  # 感受野的实际大小（米）
anchor_spacing_size = anchor_spacing * res    # 锚点间距的实际大小（米）
max_anchors_per_axis = math.ceil(receptive_field_size / anchor_spacing_size)  # 每个轴向最大锚点数
MAX_POSITIVE_ANCHORS = max_anchors_per_axis * max_anchors_per_axis  # 理论最大正样本数

# Stability caches are part of the physical-label contract, not merely a
# performance optimization.  Bump this token whenever normal orientation,
# array axes, ESDF construction, or continuous sampling semantics change.
STABILITY_MAP_FORMAT_VERSION = 3
STABILITY_MAP_SEMANTIC_VERSION = (
    "yaw_esdf_hwy_upward_nz_no_transpose_cell_center_xy_uniform_v3"
)


def stability_source_map_sha256(map_file, chunk_bytes=1024 * 1024):
    """Return the content hash binding a stability cache to ``map.p``."""
    digest = hashlib.sha256()
    with open(map_file, "rb") as handle:
        while True:
            chunk = handle.read(int(chunk_bytes))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stability_cache_is_compatible(
    cached,
    *,
    map_shape,
    resolution,
    yaw_bins,
    yaw_weight,
    source_map_sha256,
):
    """Validate both geometry and semantic provenance of a loaded NPZ."""
    try:
        return (
            cached["yaw_stability"].shape == (*tuple(map_shape), int(yaw_bins))
            and cached["cost_map"].shape == (*tuple(map_shape), int(yaw_bins))
            and int(cached["format_version"].item())
            == STABILITY_MAP_FORMAT_VERSION
            and str(cached["semantic_version"].item())
            == STABILITY_MAP_SEMANTIC_VERSION
            and str(cached["source_map_sha256"].item())
            == str(source_map_sha256)
            and int(cached["yaw_bins"].item()) == int(yaw_bins)
            and np.isclose(
                float(cached["voxel_size_xy"].item()),
                float(resolution),
                atol=1e-6,
                rtol=0.0,
            )
            and np.isclose(
                float(cached["yaw_weight"].item()),
                float(yaw_weight),
                atol=1e-6,
                rtol=0.0,
            )
        )
    except (KeyError, TypeError, ValueError):
        return False

# 被 mask 挡住的法向量使用全局小方差高斯噪声；训练动态采样，验证固定。
# 固定偏移用于让 mask 形状与验证噪声彼此独立。
MASK_NOISE_SEED_OFFSET = 7_919
MASK_INPUT_SEMANTICS = "stage1_uav_observation_mask_gaussian_v1"
MASK_GENERATION_SEMANTICS = "uav_footprint_observation_circle_obstacles_v1"
STAGE2_MASK_GENERATION_SEMANTICS = (
    "informed_ellipse_two_stage_obstacle_curriculum_segment_aware_v5"
)
MASK_NOISE_STD = 0.25

# 不完整地图允许大面积缺失，但必须保留类似 Informed RRT 的示范椭圆；
# 主动障碍随后只在轨迹中段局部挖去。
INFORMED_ELLIPSE_CLEARANCE_METERS = 0.5
STAGE2_MAX_MASKED_FRACTION = 0.60
STAGE2_MAX_DEMO_BLOCKED_FRACTION = 0.15
STAGE2_MAX_CONTIGUOUS_BLOCKED_FRACTION = 0.10


def normalize_mask(mask, shape, source="mask"):
    """校验唯一的输入 mask，并统一为 1=允许、0=禁止的 float32 数组。

    mask=0 可以源于地图未观测，也可以源于锥桶等实体障碍；模型不区分原因。
    """
    value = np.squeeze(np.asarray(mask))
    if value.shape != tuple(shape):
        raise ValueError(
            f"{source} 应为 {tuple(shape)}，实际为 {value.shape}"
        )
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{source} 含 NaN/Inf")
    return (value > 0.5).astype(np.float32)


def build_masked_normal_input(normals, mask, seed=None, noise_std=MASK_NOISE_STD):
    """构造四通道输入；mask=0 处法向量替换为 N(0,noise_std²)。

    ``seed=None`` 用于训练，每次访问动态重采样；传入固定 seed 用于验证、
    测试和可视化，保证相同样本可复现。
    """
    normals = np.asarray(normals, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    if normals.ndim != 3 or normals.shape[-1] != 3:
        raise ValueError(f"normals 应为 (H,W,3)，实际为 {normals.shape}")
    if mask.shape != normals.shape[:2]:
        raise ValueError(
            f"mask 应为 {normals.shape[:2]}，实际为 {mask.shape}"
        )
    mask = (mask > 0.5).astype(np.float32)
    if noise_std <= 0.0:
        raise ValueError("noise_std 必须为正数")
    # seed=None 时每次访问动态重采样；显式 seed 用于验证和测试复现。
    rng = np.random.default_rng(None if seed is None else np.uint64(seed))
    gaussian = rng.normal(
        0.0, float(noise_std), size=normals.shape
    ).astype(np.float32)
    masked_normals = np.where(mask[..., None] > 0.5, normals, gaussian)
    return np.concatenate([masked_normals, mask[..., None]], axis=-1)


def fit_demo_bspline_control_points(
    trajectory,
    num_control_points=26,
    degree=3,
):
    """拟合与 Stage 2 hard contract 一致的平滑示范控制点。

    旧实现先做欠定最小二乘，再单独旋转首尾控制柄；这会让第二个控制点与
    后续控制点形成尖角，在端点制造并不存在于原路径中的曲率峰值。现在把
    端点位置、exact endpoint yaw 和内部控制点放进同一个正则化最小二乘问题。
    """
    trajectory = np.asarray(trajectory, dtype=np.float32)
    if trajectory.ndim != 2 or trajectory.shape[0] < 2:
        raise ValueError(
            f"示范轨迹必须至少为 (2,2)，实际为 {trajectory.shape}"
        )
    if trajectory.shape[0] == 2:
        # mask 单元测试和旧接口可能只提供起终点；线性补一个中点即可。
        trajectory = np.stack(
            [trajectory[0], 0.5 * (trajectory[0] + trajectory[1]), trajectory[1]],
            axis=0,
        )
    if trajectory.shape[1] < 3:
        # 旧二维测试数据没有任务 yaw，使用首尾线段方向作为端点切向。
        start_vector = trajectory[1, :2] - trajectory[0, :2]
        goal_vector = trajectory[-1, :2] - trajectory[-2, :2]
        start_yaw = float(np.arctan2(start_vector[1], start_vector[0]))
        goal_yaw = float(np.arctan2(goal_vector[1], goal_vector[0]))
        trajectory = np.concatenate(
            [
                trajectory[:, :2],
                np.linspace(
                    start_yaw,
                    goal_yaw,
                    trajectory.shape[0],
                    dtype=np.float32,
                )[:, None],
            ],
            axis=1,
        )
    return _fit_demo_bspline_control_points_cached(
        trajectory.astype(np.float32, copy=False).tobytes(),
        tuple(trajectory.shape),
        int(num_control_points),
        int(degree),
        float(SAFETY_COST_CONFIG.start_yaw_tolerance_rad),
        float(SAFETY_COST_CONFIG.goal_yaw_tolerance_rad),
        float(SAFETY_COST_CONFIG.curvature_limit),
    ).copy()


def _arc_length_resample_numpy(points, count):
    """按弧长重采样折线，避免原始点密度改变最小二乘权重。"""
    points = np.asarray(points, dtype=np.float64)
    segment_length = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_length)])
    if cumulative[-1] <= 1e-8:
        raise ValueError("示范轨迹总长度接近零，无法拟合 B 样条")
    target = np.linspace(0.0, cumulative[-1], int(count))
    return np.stack(
        [
            np.interp(target, cumulative, points[:, axis])
            for axis in range(2)
        ],
        axis=1,
    )


def _closest_allowed_tangent(task_yaw, raw_vector, tolerance):
    """在 hard yaw 容差内选取最贴近原示范切向的方向。"""
    raw_yaw = float(np.arctan2(raw_vector[1], raw_vector[0]))
    difference = float(
        np.arctan2(
            np.sin(raw_yaw - float(task_yaw)),
            np.cos(raw_yaw - float(task_yaw)),
        )
    )
    selected_yaw = float(task_yaw) + float(
        np.clip(difference, -float(tolerance), float(tolerance))
    )
    return np.asarray(
        [np.cos(selected_yaw), np.sin(selected_yaw)],
        dtype=np.float64,
    )


def _max_discrete_curvature_numpy(points):
    """与 Stage 2 ``discrete_turning_curvature`` 相同的 numpy 诊断。"""
    points = np.asarray(points, dtype=np.float64)
    incoming = points[1:-1] - points[:-2]
    outgoing = points[2:] - points[1:-1]
    incoming_length = np.linalg.norm(incoming, axis=1)
    outgoing_length = np.linalg.norm(outgoing, axis=1)
    cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
    dot = np.sum(incoming * outgoing, axis=1)
    angle = np.arctan2(np.abs(cross), dot)
    minimum = float(SAFETY_COST_CONFIG.curvature_min_segment_meters)
    degenerate = (incoming_length < minimum) | (outgoing_length < minimum)
    angle[degenerate] = np.pi
    local_length = 0.5 * (incoming_length + outgoing_length)
    return float(np.max(angle / np.maximum(local_length, minimum)))


@lru_cache(maxsize=32_768)
def _fit_demo_bspline_control_points_cached(
    trajectory_bytes,
    trajectory_shape,
    num_control_points,
    degree,
    start_yaw_tolerance,
    goal_yaw_tolerance,
    curvature_limit,
):
    """缓存确定性拟合；默认单进程 DataLoader 下每条路径只求解一次。"""
    from scipy.optimize import lsq_linear

    trajectory = np.frombuffer(
        trajectory_bytes,
        dtype=np.float32,
    ).reshape(trajectory_shape).astype(np.float64)
    raw_points = trajectory[:, :2]
    # 50 个弧长点足以稳定约束 26 个控制点，同时避免每个 epoch 重复大矩阵拟合。
    fit_points = _arc_length_resample_numpy(raw_points, 50)
    evaluation_points = _arc_length_resample_numpy(
        raw_points,
        DENSE_TRAJECTORY_POINTS,
    )
    count = int(num_control_points)
    if count < 5:
        raise ValueError("带端点 yaw 约束的 B 样条至少需要 5 个控制点")
    basis, _ = create_bspline_basis_matrix(
        len(fit_points),
        count,
        int(degree),
    )
    start = fit_points[0]
    goal = fit_points[-1]
    start_direction = _closest_allowed_tangent(
        trajectory[0, 2],
        raw_points[1] - raw_points[0],
        start_yaw_tolerance,
    )
    goal_direction = _closest_allowed_tangent(
        trajectory[-1, 2],
        raw_points[-1] - raw_points[-2],
        goal_yaw_tolerance,
    )

    # 未知量为首尾控制柄长度，以及 C2...C[-3] 的 xy。
    unknowns = 2 + 2 * (count - 4)
    design = np.zeros((2 * len(fit_points), unknowns), dtype=np.float64)
    target = np.zeros(2 * len(fit_points), dtype=np.float64)
    for sample_index in range(len(fit_points)):
        constant = (
            (basis[sample_index, 0] + basis[sample_index, 1]) * start
            + (basis[sample_index, -2] + basis[sample_index, -1]) * goal
        )
        for axis in range(2):
            row = 2 * sample_index + axis
            target[row] = fit_points[sample_index, axis] - constant[axis]
            design[row, 0] = basis[sample_index, 1] * start_direction[axis]
            design[row, 1] = -basis[sample_index, -2] * goal_direction[axis]
            for control_index in range(2, count - 2):
                column = 2 + 2 * (control_index - 2) + axis
                design[row, column] = basis[sample_index, control_index]

    control_constant = np.zeros((count, 2), dtype=np.float64)
    control_constant[:2] = start
    control_constant[-2:] = goal
    transform = np.zeros((2 * count, unknowns), dtype=np.float64)
    transform[2:4, 0] = start_direction
    transform[-4:-2, 1] = -goal_direction
    for control_index in range(2, count - 2):
        for axis in range(2):
            transform[
                2 * control_index + axis,
                2 + 2 * (control_index - 2) + axis,
            ] = 1.0

    second_difference = np.zeros(
        (2 * (count - 2), 2 * count),
        dtype=np.float64,
    )
    for control_index in range(count - 2):
        for axis in range(2):
            row = 2 * control_index + axis
            second_difference[row, 2 * control_index + axis] = 1.0
            second_difference[row, 2 * (control_index + 1) + axis] = -2.0
            second_difference[row, 2 * (control_index + 2) + axis] = 1.0

    path_length = float(
        np.linalg.norm(np.diff(fit_points, axis=0), axis=1).sum()
    )
    lower = np.full(unknowns, -np.inf)
    upper = np.full(unknowns, np.inf)
    lower[:2] = max(path_length / max(count - 1, 1) * 0.25, 1e-4)
    upper[:2] = max(path_length * 0.25, lower[0] * 2.0)

    def solve(smooth_weight):
        scale = np.sqrt(float(smooth_weight))
        augmented_design = np.concatenate(
            [design, scale * second_difference @ transform],
            axis=0,
        )
        augmented_target = np.concatenate(
            [
                target,
                -scale
                * second_difference
                @ control_constant.reshape(-1),
            ],
            axis=0,
        )
        solution = lsq_linear(
            augmented_design,
            augmented_target,
            bounds=(lower, upper),
        ).x
        control = (
            control_constant.reshape(-1) + transform @ solution
        ).reshape(count, 2)
        dense = reconstruct_from_control_points(
            control,
            DENSE_TRAJECTORY_POINTS,
            degree=int(degree),
        )
        curvature = _max_discrete_curvature_numpy(dense)
        rmse = float(np.sqrt(np.mean(np.square(dense - evaluation_points))))
        return control, curvature, rmse

    candidates = [solve(0.005)]
    if candidates[0][1] > float(curvature_limit):
        # 仅对约 1% 的困难示范尝试更强平滑，主路径仍只需求解一次。
        candidates.extend(
            solve(weight)
            for weight in (0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1, 0.2)
        )
    feasible = [
        item
        for item in candidates
        if item[1]
        <= float(curvature_limit)
        + float(SAFETY_COST_CONFIG.hard_constraint_epsilon)
    ]
    selected = (
        min(feasible, key=lambda item: item[2])
        if feasible
        else min(candidates, key=lambda item: item[1])
    )
    return selected[0].astype(np.float32)


def dense_demo_bspline(
    trajectory,
    num_points=DENSE_TRAJECTORY_POINTS,
):
    """按 mask 构造专用的容差-yaw规则重建稠密示范 B 样条。

    Stage 1 的真实 44D target 由 ``BoundaryConstrainedPathRepresentation``
    构造，并保持 exact endpoint yaw；两者不可再混称为同一个拟合器。
    """
    control_points = fit_demo_bspline_control_points(trajectory)
    return reconstruct_from_control_points(
        control_points, num_output_points=int(num_points), degree=3
    ).astype(np.float32)


def erode_mask_for_vehicle(
    mask,
    *,
    vehicle_radius_meters=SAFETY_COST_CONFIG.vehicle_radius_meters,
    resolution=MAP_RESOLUTION,
):
    """把几何 mask 转成车辆中心可行的配置空间 mask。"""
    from scipy.ndimage import binary_erosion

    mask = np.asarray(mask, dtype=np.float32) > 0.5
    radius_pixels = int(np.ceil(float(vehicle_radius_meters) / float(resolution)))
    if radius_pixels <= 0:
        return mask.astype(np.float32)
    yy, xx = np.mgrid[
        -radius_pixels : radius_pixels + 1,
        -radius_pixels : radius_pixels + 1,
    ]
    footprint = np.square(xx) + np.square(yy) <= radius_pixels**2
    # 地图外边界由独立 box cost 处理；这里只膨胀 mask 内部遮挡。
    eroded = binary_erosion(mask, structure=footprint, border_value=1)
    return eroded.astype(np.float32)


def _sample_trajectory_for_mask(trajectory_xy, *, spacing_m=None):
    """按固定物理间隔生成有序折线样本，端点和每个折点均保留。"""
    points = np.asarray(trajectory_xy, dtype=np.float32)[..., :2]
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError(f"trajectory_xy 应为 (N,2)，实际为 {points.shape}")
    if spacing_m is None:
        spacing_m = 0.5 * MAP_RESOLUTION
    if float(spacing_m) <= 0.0:
        raise ValueError("spacing_m 必须为正数")

    dense = [points[:1]]
    for start, end in zip(points[:-1], points[1:]):
        distance = float(np.linalg.norm(end - start))
        steps = max(1, int(np.ceil(distance / float(spacing_m))))
        # 上一段的终点就是本段的起点；只追加本段的非重复采样点。
        dense.append(
            np.linspace(
                start,
                end,
                steps + 1,
                endpoint=True,
                dtype=np.float32,
            )[1:]
        )
    return np.concatenate(dense, axis=0)


def _trajectory_mask_pixel_sequence(trajectory_xy, shape):
    """返回与连续折线顺序一致的 (row, col) mask 像素序列。"""
    shape = tuple(int(value) for value in shape)
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError(f"mask shape 必须是正二维尺寸，实际为 {shape}")
    dense = _sample_trajectory_for_mask(trajectory_xy)
    rows_cols = [geom2pix(point, size=shape) for point in dense]
    return np.asarray(rows_cols, dtype=np.int64)


def rasterize_trajectory(trajectory_xy, shape):
    """把连续折线按半像素加密后栅格化。"""
    pixels = _trajectory_mask_pixel_sequence(trajectory_xy, shape)
    raster = np.zeros(tuple(shape), dtype=bool)
    raster[pixels[:, 0], pixels[:, 1]] = True
    return raster


def trajectory_is_allowed(trajectory_xy, mask):
    """检查连续轨迹是否始终位于 mask=1。"""
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError(f"mask 应为二维数组，实际为 {mask.shape}")
    pixels = _trajectory_mask_pixel_sequence(trajectory_xy, mask.shape)
    return bool(np.all(mask[pixels[:, 0], pixels[:, 1]] > 0.5))


def trajectory_mask_blockage_metrics(trajectory_xy, mask):
    """统计 mask 沿稠密轨迹造成的总阻断量和最长连续阻断段。

    这不是有效性判据。Stage 2 用它限制随机干预的任务变化幅度：允许局部
    障碍挡住旧路线，但不把局部纠正问题变成完全不同的全局重规划问题。
    """
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError(f"mask 应为二维数组，实际为 {mask.shape}")
    pixels = _trajectory_mask_pixel_sequence(trajectory_xy, mask.shape)
    blocked = mask[pixels[:, 0], pixels[:, 1]] <= 0.5
    if blocked.size == 0:
        return {
            "blocked_fraction": 0.0,
            "max_contiguous_blocked_fraction": 0.0,
        }
    padded = np.pad(blocked.astype(np.int8), (1, 1))
    changes = np.flatnonzero(np.diff(padded))
    longest = (
        int(np.max(changes[1::2] - changes[::2]))
        if changes.size
        else 0
    )
    return {
        "blocked_fraction": float(blocked.mean()),
        "max_contiguous_blocked_fraction": float(longest / blocked.size),
    }


def informed_demo_ellipse(
    shape,
    trajectory_xy,
    *,
    clearance_meters=INFORMED_ELLIPSE_CLEARANCE_METERS,
):
    """返回包含示范路径的 Informed-RRT 风格椭圆区域。

    起终点是两个焦点，长轴长度取稠密示范路径长度并增加固定余量。按照
    三角不等式，示范路径位于该椭圆内；不完整地图只能遮挡椭圆外部。
    """
    trajectory = np.asarray(trajectory_xy, dtype=np.float32)[..., :2]
    if trajectory.ndim != 2 or len(trajectory) < 2:
        raise ValueError(f"trajectory_xy 至少应为 (2,2)，实际为 {trajectory.shape}")
    height, width = tuple(shape)
    rows, cols = np.mgrid[0:height, 0:width]
    grid_x = (
        MAP_CONFIG.origin_xy[0] + (cols.astype(np.float32) + 0.5) * MAP_RESOLUTION
    )
    grid_y = (
        MAP_CONFIG.origin_xy[1] + (rows.astype(np.float32) + 0.5) * MAP_RESOLUTION
    )
    start = trajectory[0]
    goal = trajectory[-1]
    path_length = float(
        np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum()
    )
    focal_distance = float(np.linalg.norm(goal - start))
    major_axis = max(path_length, focal_distance) + 2.0 * float(
        clearance_meters
    )
    distance_sum = np.hypot(grid_x - start[0], grid_y - start[1]) + np.hypot(
        grid_x - goal[0], grid_y - goal[1]
    )
    return distance_sum <= major_axis


def endpoint_yaw_corridors(
    trajectory,
    *,
    corridor_meters=SAFETY_COST_CONFIG.endpoint_mask_corridor_meters,
):
    """构造起点前向和终点反向的局部中心线通道。

    输入含 yaw 时严格使用任务端点 yaw；旧的二维测试轨迹则用首尾线段方向
    回退。mask 已经按车辆半径转为配置空间，因此这里只检查车辆中心线。
    """
    trajectory = np.asarray(trajectory, dtype=np.float32)
    if trajectory.ndim != 2 or trajectory.shape[0] < 2 or trajectory.shape[1] < 2:
        raise ValueError(f"trajectory 至少应为 (2,2)，实际为 {trajectory.shape}")
    length = float(corridor_meters)
    if length < 0.0:
        raise ValueError("corridor_meters 不能为负数")

    if trajectory.shape[1] >= 3:
        start_direction = np.array(
            [np.cos(trajectory[0, 2]), np.sin(trajectory[0, 2])],
            dtype=np.float32,
        )
        goal_direction = np.array(
            [np.cos(trajectory[-1, 2]), np.sin(trajectory[-1, 2])],
            dtype=np.float32,
        )
    else:
        start_direction = trajectory[1, :2] - trajectory[0, :2]
        goal_direction = trajectory[-1, :2] - trajectory[-2, :2]
        start_direction /= max(float(np.linalg.norm(start_direction)), 1e-8)
        goal_direction /= max(float(np.linalg.norm(goal_direction)), 1e-8)

    # 半像素采样，和连续轨迹 mask 检查保持相同空间精度。
    steps = max(1, int(np.ceil(length / (0.5 * MAP_RESOLUTION))))
    distance = np.linspace(0.0, length, steps + 1, dtype=np.float32)
    start_corridor = (
        trajectory[0, :2][None] + distance[:, None] * start_direction[None]
    )
    goal_corridor = (
        trajectory[-1, :2][None] - distance[:, None] * goal_direction[None]
    )
    return start_corridor.astype(np.float32), goal_corridor.astype(np.float32)


def mask_start_goal_connected(mask, start_xy, goal_xy):
    """检查腐蚀后的配置空间 mask 中，起终点是否属于同一可通行分量。

    这里使用八邻域，只做随机 mask 是否显然不可解的必要条件检查。通过该
    检查不代表一定存在满足地形、曲率和航向约束的轨迹。
    """
    from scipy.ndimage import label

    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError(f"mask 应为二维数组，实际为 {mask.shape}")
    start_row, start_col = geom2pix(
        np.asarray(start_xy, dtype=np.float32)[:2], size=mask.shape
    )
    goal_row, goal_col = geom2pix(
        np.asarray(goal_xy, dtype=np.float32)[:2], size=mask.shape
    )
    allowed = mask > 0.5
    if not allowed[start_row, start_col] or not allowed[goal_row, goal_col]:
        return False
    components, _ = label(
        allowed,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    component = int(components[start_row, start_col])
    return component != 0 and component == int(components[goal_row, goal_col])


def generate_random_mask(
    shape,
    seed,
    trajectory_xy,
    *,
    p_mask=0.5,
    require_trajectory_clear=True,
    vehicle_radius_meters=SAFETY_COST_CONFIG.vehicle_radius_meters,
    endpoint_corridor_meters=SAFETY_COST_CONFIG.endpoint_mask_corridor_meters,
    max_attempts=128,
    return_metadata=False,
):
    """按训练阶段生成唯一 mask。

    不完整地图与主动障碍最终合并为同一张 mask：前者只能遮挡示范的
    Informed-RRT 风格椭圆之外，后者只在轨迹中段放置小型局部障碍。
    Stage 1 只生成不完整地图；Stage 2 可随机组合两种语义。
    """
    if not 0.0 <= p_mask <= 1.0:
        raise ValueError("p_mask 必须在 [0,1]")
    height, width = tuple(shape)
    yy, xx = np.mgrid[0:height, 0:width]
    dense_trajectory = dense_demo_bspline(trajectory_xy)
    informed_region = informed_demo_ellipse(
        (height, width),
        dense_trajectory,
    )
    # 随机缺失先额外保护一个车辆半径，随后配置空间腐蚀也不会侵入目标椭圆。
    incomplete_carve_protection = informed_demo_ellipse(
        (height, width),
        dense_trajectory,
        clearance_meters=(
            INFORMED_ELLIPSE_CLEARANCE_METERS
            + float(vehicle_radius_meters)
        ),
    )
    task_endpoints = np.asarray(trajectory_xy, dtype=np.float32)[[0, -1], :2]
    start_corridor, goal_corridor = endpoint_yaw_corridors(
        trajectory_xy,
        corridor_meters=endpoint_corridor_meters,
    )

    def task_rejection_reason(mask):
        # 起终点始终必须可用；Stage 2 只取消对旧示范中间段的条件化。
        endpoints_clear = all(
            trajectory_is_allowed(point[None], mask)
            for point in task_endpoints
        )
        if not endpoints_clear:
            return "endpoint_blocked", {
                "blocked_fraction": 0.0,
                "max_contiguous_blocked_fraction": 0.0,
            }
        if not trajectory_is_allowed(start_corridor, mask):
            return "start_corridor_blocked", {
                "blocked_fraction": 0.0,
                "max_contiguous_blocked_fraction": 0.0,
            }
        if not trajectory_is_allowed(goal_corridor, mask):
            return "goal_corridor_blocked", {
                "blocked_fraction": 0.0,
                "max_contiguous_blocked_fraction": 0.0,
            }
        blockage = trajectory_mask_blockage_metrics(dense_trajectory, mask)
        if require_trajectory_clear:
            # Stage 1 必须与 get_item 的最终校验使用完全相同的连续折线
            # 栅格化；仅查询 200 个样条点会漏掉相邻点之间的薄障碍。
            if not trajectory_is_allowed(dense_trajectory, mask):
                return "demo_blocked", blockage
            return None, blockage
        if not mask_start_goal_connected(
            mask,
            task_endpoints[0],
            task_endpoints[1],
        ):
            return "start_goal_disconnected", blockage
        if (
            blockage["blocked_fraction"]
            > STAGE2_MAX_DEMO_BLOCKED_FRACTION
        ):
            return "demo_blockage_too_large", blockage
        if (
            blockage["max_contiguous_blocked_fraction"]
            > STAGE2_MAX_CONTIGUOUS_BLOCKED_FRACTION
        ):
            return "demo_blockage_too_long", blockage
        return None, blockage

    def mask_type(has_large, has_small):
        if has_large and has_small:
            return "mixed"
        if has_large:
            return "large_only"
        if has_small:
            return "small_only"
        return "complete"

    proposed_type_counts = {}
    rejected_type_counts = {}
    rejection_reason_counts = {}
    raw_connectivity_counts = {"connected": 0, "disconnected": 0}

    def increment(counter, key):
        counter[key] = counter.get(key, 0) + 1

    base_seed = int(seed)

    # 顶层 Bernoulli：x=0 表示本样本完全不加遮挡，返回全 1 mask。
    activation_rng = np.random.default_rng(base_seed)
    mask_active = bool(activation_rng.binomial(1, float(p_mask)))
    if not mask_active:
        full = np.ones((height, width), dtype=np.float32)
        metadata = {
            "mask_active": False,
            "has_large_cutout": False,
            "has_small_obstacles": False,
            "has_route_local_obstacle": False,
            "has_global_entity_obstacles": False,
            "trajectory_obstacle_mode": "none",
            "semantic_mode": "complete",
            "attempts": 1,
            "sampling_attempts": 1,
            "resamples": 0,
            "masked_fraction": 0.0,
            "endpoint_corridor_meters": float(endpoint_corridor_meters),
            "accepted_type": "complete",
            "mask_profile": (
                "stage1_demo_valid"
                if require_trajectory_clear
                else "stage2_local_intervention"
            ),
            "demo_blocked_fraction": 0.0,
            "demo_max_contiguous_blocked_fraction": 0.0,
            "informed_ellipse_fraction": float(informed_region.mean()),
            "proposed_type_counts": {"complete": 1},
            "rejected_type_counts": {},
            "rejection_reason_counts": {},
            "raw_connectivity_counts": {"connected": 1, "disconnected": 0},
            "used_fallback": False,
        }
        return (full, metadata) if return_metadata else full

    def carve_ellipse(
        mask,
        rng,
        radius_range,
        center=None,
        protected_region=None,
    ):
        if center is None:
            cx = rng.uniform(0, width - 1)
            cy = rng.uniform(0, height - 1)
        else:
            cx, cy = center
        rx = rng.uniform(*radius_range) * width
        ry = rng.uniform(*radius_range) * height
        angle = rng.uniform(0.0, 2.0 * np.pi)
        dx, dy = xx - cx, yy - cy
        u = np.cos(angle) * dx + np.sin(angle) * dy
        v = -np.sin(angle) * dx + np.cos(angle) * dy
        carved = (u / rx) ** 2 + (v / ry) ** 2 <= 1.0
        if protected_region is not None:
            carved &= ~protected_region
        mask[carved] = False

    for attempt in range(int(max_attempts)):
        rng = np.random.default_rng(base_seed + 17 + attempt * 104_729)
        mask = np.ones((height, width), dtype=bool)
        # 两阶段都训练不完整地图和实体障碍语义。Stage 1 的轨迹附近障碍
        # 保持安全间距；Stage 2 才把障碍直接放到中段轨迹上制造绕行目标。
        has_incomplete_observation = (
            True if require_trajectory_clear else bool(rng.random() < 0.70)
        )
        has_route_local_obstacle = (
            bool(rng.random() < 0.65)
            if require_trajectory_clear
            else bool(rng.random() < 0.50)
        )
        has_global_entity_obstacles = bool(
            require_trajectory_clear and rng.random() < 0.25
        )
        if not (has_incomplete_observation or has_route_local_obstacle):
            has_incomplete_observation = True

        has_large = (
            has_incomplete_observation and bool(rng.random() < 0.65)
        )
        has_small = (
            has_incomplete_observation and bool(rng.random() < 0.65)
        )
        if has_incomplete_observation and not (has_large or has_small):
            has_large = bool(rng.integers(0, 2))
            has_small = not has_large
        # 主动障碍也属于最终 mask 中的小孔洞。
        has_small = (
            has_small
            or has_route_local_obstacle
            or has_global_entity_obstacles
        )
        large_count = (1, 4)
        large_radius = (0.14, 0.34)
        small_count = (3, 11)
        small_radius = (0.015, 0.055)
        semantic_mode = (
            "mixed"
            if has_incomplete_observation
            and (has_route_local_obstacle or has_global_entity_obstacles)
            else (
                "incomplete_observation"
                if has_incomplete_observation
                else "route_obstacle"
            )
        )
        trajectory_obstacle_mode = (
            "near_route_nonblocking"
            if require_trajectory_clear and has_route_local_obstacle
            else (
                "on_route_blocking"
                if has_route_local_obstacle
                else "none"
            )
        )
        candidate_type = mask_type(has_large, has_small)
        increment(proposed_type_counts, candidate_type)

        if has_large:
            for _ in range(int(rng.integers(*large_count))):
                carve_ellipse(
                    mask,
                    rng,
                    large_radius,
                    protected_region=incomplete_carve_protection,
                )
        if has_route_local_obstacle:
            # 只在旧路线中间 60% 放一个小障碍，制造局部绕行监督；后面的
            # 阻断比例门槛仍会拒绝尺寸偶然过大的候选。
            lower = max(1, int(round(0.20 * len(dense_trajectory))))
            upper = min(
                len(dense_trajectory) - 1,
                int(round(0.80 * len(dense_trajectory))),
            )
            route_index = int(rng.integers(lower, max(lower + 1, upper)))
            center_xy = dense_trajectory[route_index].copy()
            if require_trajectory_clear:
                tangent = (
                    dense_trajectory[min(route_index + 2, len(dense_trajectory) - 1)]
                    - dense_trajectory[max(route_index - 2, 0)]
                )
                tangent /= max(float(np.linalg.norm(tangent)), 1e-6)
                normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
                side = -1.0 if rng.random() < 0.5 else 1.0
                center_xy += (
                    side * rng.uniform(0.9, 1.4) * normal
                ).astype(np.float32)
            route_row, route_col = geom2pix(
                center_xy,
                size=(height, width),
            )
            carve_ellipse(
                mask,
                rng,
                (0.015, 0.030),
                center=(route_col, route_row),
            )
        if has_global_entity_obstacles:
            # 少量全局小障碍提高位置泛化；若碰到示范 footprint，Stage 1
            # 的既有 hard rejection 会重新采样，不产生矛盾监督。
            for _ in range(int(rng.integers(1, 4))):
                carve_ellipse(mask, rng, (0.010, 0.025))
        if has_small:
            if has_incomplete_observation:
                for _ in range(int(rng.integers(*small_count))):
                    carve_ellipse(
                        mask,
                        rng,
                        small_radius,
                        protected_region=incomplete_carve_protection,
                    )

        # Stage 1 只排除几乎删掉全图的候选；Stage 2 在车辆腐蚀后使用更
        # 明确的上限，避免隐藏一个极窄但像素上仍连通的通道。
        stage1_too_sparse = bool(
            require_trajectory_clear and mask.mean() < 0.20
        )
        mask_float = erode_mask_for_vehicle(
            mask.astype(np.float32),
            vehicle_radius_meters=vehicle_radius_meters,
        )
        connected = mask_start_goal_connected(
            mask_float,
            task_endpoints[0],
            task_endpoints[1],
        )
        increment(
            raw_connectivity_counts,
            "connected" if connected else "disconnected",
        )
        if stage1_too_sparse:
            rejection_reason = "too_sparse"
            blockage = trajectory_mask_blockage_metrics(
                dense_trajectory, mask_float
            )
        elif (
            not require_trajectory_clear
            and 1.0 - float(mask_float.mean())
            > STAGE2_MAX_MASKED_FRACTION
        ):
            rejection_reason = "masked_fraction_too_large"
            blockage = trajectory_mask_blockage_metrics(
                dense_trajectory, mask_float
            )
        else:
            rejection_reason, blockage = task_rejection_reason(mask_float)
        if rejection_reason is None:
            metadata = {
                "mask_active": True,
                "has_large_cutout": has_large,
                "has_small_obstacles": has_small,
                "has_route_local_obstacle": has_route_local_obstacle,
                "has_global_entity_obstacles": has_global_entity_obstacles,
                "trajectory_obstacle_mode": trajectory_obstacle_mode,
                "semantic_mode": semantic_mode,
                "attempts": attempt + 1,
                "sampling_attempts": attempt + 1,
                "resamples": attempt,
                "masked_fraction": float(1.0 - mask_float.mean()),
                "endpoint_corridor_meters": float(endpoint_corridor_meters),
                "accepted_type": candidate_type,
                "mask_profile": (
                    "stage1_demo_valid"
                    if require_trajectory_clear
                    else "stage2_local_intervention"
                ),
                "demo_blocked_fraction": blockage["blocked_fraction"],
                "demo_max_contiguous_blocked_fraction": blockage[
                    "max_contiguous_blocked_fraction"
                ],
                "informed_ellipse_fraction": float(informed_region.mean()),
                "proposed_type_counts": proposed_type_counts,
                "rejected_type_counts": rejected_type_counts,
                "rejection_reason_counts": rejection_reason_counts,
                "raw_connectivity_counts": raw_connectivity_counts,
                "used_fallback": False,
            }
            return (mask_float, metadata) if return_metadata else mask_float
        increment(rejected_type_counts, candidate_type)
        increment(rejection_reason_counts, rejection_reason)

    # 极少数候选持续过度遮挡时，回退为逐个尝试小孔洞；接受条件仍由
    # require_trajectory_clear 决定，不会沿轨迹人工刻出走廊。
    rng = np.random.default_rng(base_seed + int(max_attempts) * 104_729)
    mask = np.ones((height, width), dtype=bool)
    accepted = 0
    fallback_attempts = 0
    for _ in range(64):
        fallback_attempts += 1
        increment(proposed_type_counts, "small_only")
        candidate = mask.copy()
        carve_ellipse(
            candidate,
            rng,
            (0.015, 0.055),
            protected_region=incomplete_carve_protection,
        )
        candidate_float = candidate.astype(np.float32)
        candidate_float = erode_mask_for_vehicle(
            candidate_float,
            vehicle_radius_meters=vehicle_radius_meters,
        )
        connected = mask_start_goal_connected(
            candidate_float,
            task_endpoints[0],
            task_endpoints[1],
        )
        increment(
            raw_connectivity_counts,
            "connected" if connected else "disconnected",
        )
        if (
            not require_trajectory_clear
            and 1.0 - float(candidate_float.mean())
            > STAGE2_MAX_MASKED_FRACTION
        ):
            rejection_reason = "masked_fraction_too_large"
        else:
            rejection_reason, _ = task_rejection_reason(candidate_float)
        if rejection_reason is None:
            mask = candidate
            accepted += 1
        else:
            increment(rejected_type_counts, "small_only")
            increment(rejection_reason_counts, rejection_reason)
        if accepted >= 6:
            break
    mask_float = erode_mask_for_vehicle(
        mask.astype(np.float32),
        vehicle_radius_meters=vehicle_radius_meters,
    )
    blockage = trajectory_mask_blockage_metrics(
        dense_trajectory, mask_float
    )
    metadata = {
        "mask_active": True,
        "has_large_cutout": False,
        "has_small_obstacles": accepted > 0,
        "has_route_local_obstacle": False,
        "has_global_entity_obstacles": False,
        "trajectory_obstacle_mode": "none",
        "semantic_mode": "incomplete_observation",
        "attempts": int(max_attempts),
        "sampling_attempts": int(max_attempts) + fallback_attempts,
        "resamples": int(max_attempts) + fallback_attempts - 1,
        "masked_fraction": float(1.0 - mask_float.mean()),
        "endpoint_corridor_meters": float(endpoint_corridor_meters),
        "accepted_type": "small_only" if accepted > 0 else "complete",
        "mask_profile": (
            "stage1_demo_valid"
            if require_trajectory_clear
            else "stage2_local_intervention"
        ),
        "demo_blocked_fraction": blockage["blocked_fraction"],
        "demo_max_contiguous_blocked_fraction": blockage[
            "max_contiguous_blocked_fraction"
        ],
        "informed_ellipse_fraction": float(informed_region.mean()),
        "proposed_type_counts": proposed_type_counts,
        "rejected_type_counts": rejected_type_counts,
        "rejection_reason_counts": rejection_reason_counts,
        "raw_connectivity_counts": raw_connectivity_counts,
        "used_fallback": True,
    }
    return (mask_float, metadata) if return_metadata else mask_float

# 【锚点网格系统构建】
# 将连续的地图空间离散化为12x12的锚点网格，用于Transformer的token化处理

# X轴锚点坐标：从6像素开始，每8像素一个锚点，再转换为地图几何坐标。
X = (
    np.arange(
        boundary_offset,
        output_grid * anchor_spacing + boundary_offset,
        anchor_spacing,
    )
    * res
    - MAP_HALF_EXTENT
)

# Y轴使用相同的居中坐标约定。
Y = (
    np.arange(
        boundary_offset,
        output_grid * anchor_spacing + boundary_offset,
        anchor_spacing,
    )
    * res
    - MAP_HALF_EXTENT
)

# 创建2D网格：生成所有锚点的几何坐标
grid_2d = np.meshgrid(X, Y)  # 创建X-Y坐标网格

# 网格点重排：将2D网格转换为(N, 2)的点列表，N=144个锚点
# 注意：必须与hashTable的生成顺序保持一致！
# hashTable按行优先顺序：for r in range(output_grid) for c in range(output_grid)
# 因此grid_points也必须按相同顺序重排
XX, YY = grid_2d[0], grid_2d[1]  # XX是x坐标矩阵，YY是y坐标矩阵
grid_points = np.array([[XX[r, c], YY[r, c]] 
                       for r in range(output_grid) for c in range(output_grid)])  # 形状：(144, 2)

# print(grid_points)

# 哈希表：锚点索引到像素坐标的映射表
# 修正：与grid_points保持一致的[x, y]顺序（而不是[r, c]=[y, x]顺序）
hashTable = [(anchor_spacing*c+boundary_offset, anchor_spacing*r+boundary_offset)
             for r in range(output_grid) for c in range(output_grid)]

# print(hashTable[:5])  # 打印前5个锚点坐标

# 【网格系统说明】
# 1. 锚点分布：12x12=144个锚点均匀分布在地图上
# 2. 像素间距：每个锚点间隔8像素，物理间距由 MAP_RESOLUTION 派生
# 3. 边界偏移：起始偏移6像素，确保锚点不在地图边缘
# 4. 坐标对应：每个锚点代表一个8x8像素的区域
# 5. 索引映射：通过hashTable实现1D索引到2D像素坐标的转换

def geom2pixMatpos(pos, res=MAP_RESOLUTION, size=MAP_CONFIG.map_shape):
    # 计算输入位置到所有锚点的距离
    distances = np.linalg.norm(grid_points - pos, axis=1)  # 形状：(100,)
    
    # 筛选距离阈值内的锚点索引
    # indices = np.where(distances <= receptive_field * res * 0.5)  # 阈值：18 * 0.1 * 0.5 = 0.9米
    # indices = np.where(distances <= receptive_field * res * 0.4)  # 阈值：18 * 0.1 * 0.4 = 0.72米

    # 筛选感受野区域内包含输入位置的锚点索引（感受野区域是矩形）
    indices = np.where((grid_points[:, 0] >= pos[0] - receptive_field * res / 2) &
                       (grid_points[:, 0] <= pos[0] + receptive_field * res / 2) &
                       (grid_points[:, 1] >= pos[1] - receptive_field * res / 2) &
                       (grid_points[:, 1] <= pos[1] + receptive_field * res / 2))

    return indices  # 返回正样本锚点索引元组

def geom2pix(pos, res=MAP_RESOLUTION, size=MAP_CONFIG.map_shape):
    """
    几何坐标到像素坐标的转换函数
    
    Args:
        pos: 几何坐标 (x, y)，单位：米
        res: 地图分辨率，米/像素
        size: 地图尺寸 (height, width)
    
    Returns:
        tuple: 像素坐标 (row, col)
    """
    x, y = pos

    # 居中地图坐标 [-half_extent, half_extent] -> 像素坐标。
    col = int((x - MAP_CONFIG.origin_xy[0]) / res)
    row = int((y - MAP_CONFIG.origin_xy[1]) / res)
    
    # 边界检查
    row = max(0, min(size[0] - 1, row))
    col = max(0, min(size[1] - 1, col))
    
    return (row, col)


def PaddedSequence(batch):
    """
    固定尺寸批处理整理函数（用于Transformer训练）
    
    【核心功能】
    由于DataLoader已经确保所有样本的anchor和labels具有固定尺寸(MAX_POSITIVE_ANCHORS)，
    此函数只需简单地将批处理数据堆叠即可，无需复杂的填充处理。
    
    【处理流程】
    1. 过滤无效样本：移除None值的样本
    2. 数据堆叠：直接堆叠所有数据到批处理维度
    
    Args:
        batch (list): 批处理样本列表
            每个元素包含：
            - 'map': 地图张量，形状为(1, C, H, W)
            - 'anchor': 锚点序列，形状为(2*N, MAX_POSITIVE_ANCHORS) - 已固定尺寸
            - 'labels': 标签序列，形状为(2*N, MAX_POSITIVE_ANCHORS) - 已固定尺寸
            - 'trajectory': 轨迹序列，形状为(N+2, 3)
    
    Returns:
        dict: 整理后的批处理数据
            - 'map': 批处理地图，形状为(B, C, H, W)
            - 'anchor': 批处理锚点序列，形状为(B, 2*N, MAX_POSITIVE_ANCHORS)
            - 'labels': 批处理标签序列，形状为(B, 2*N, MAX_POSITIVE_ANCHORS)
            - 'length': 每个样本的序列长度，形状为(B,) - 用于兼容性
            - 'trajectory': 批处理轨迹序列，形状为(B, N+2, 3)
    """
    # 过滤有效样本：移除None值，确保数据完整性
    valid_batch = [batch_i for batch_i in batch if batch_i is not None]
    relative_motions = torch.stack([item['relative_motion'] for item in batch])
    start_poses = torch.stack([item['start_pose'] for item in batch])
    goal_poses = torch.stack([item['goal_pose'] for item in batch])
    
    # 由于所有样本已经具有固定尺寸，直接堆叠即可
    data = {
        'map': torch.stack([batch_i['map'] for batch_i in valid_batch]),  # [B, C, H, W]
        'relative_motion': torch.stack([batch_i['relative_motion'] for batch_i in valid_batch]),
        'start_pose': torch.stack([batch_i['start_pose'] for batch_i in valid_batch]),
        'goal_pose': torch.stack([batch_i['goal_pose'] for batch_i in valid_batch]),
        'anchor': torch.stack([batch_i['anchor'] for batch_i in valid_batch]),  # [B, 2*N, MAX_POSITIVE_ANCHORS]
        'labels': torch.stack([batch_i['labels'] for batch_i in valid_batch]),  # [B, 2*N, MAX_POSITIVE_ANCHORS]
        'length': torch.tensor([batch_i['anchor'].shape[0] for batch_i in valid_batch]),  # [B,] - 序列长度
        'trajectory': torch.stack([batch_i['trajectory'] for batch_i in valid_batch]),  # [B, N+2, 3]
        'cost': torch.stack([batch_i['cost'] for batch_i in valid_batch])  # 路径成本标量
    }
    
    # 如果有stability相关数据（Stage 2训练需要），则添加到返回字典
    if valid_batch and 'normals' in valid_batch[0]:
        data['normals'] = torch.stack([batch_i['normals'] for batch_i in valid_batch])  # [B, 3, H, W]
    if valid_batch and 'cost_map' in valid_batch[0]:
        data['cost_map'] = torch.stack([batch_i['cost_map'] for batch_i in valid_batch])  # [B, num_layers, max_anchors]
    
    return data

def get_encoder_input(normal_z, goal_state, start_state, normal_x, normal_y):
    """
    构造编码输入，包含数据验证和修复机制
    """
    
    # # 确保 nz 全为非负数
    # normal_z = torch.abs(normal_z)  # 确保法向量Z分量非负
    
    # # 输入数据验证和修复
    # def validate_and_fix_normal_component(component, component_name, default_value=0.0):
    #     """验证和修复法向量分量"""
    #     if not np.all(np.isfinite(component)):
    #         invalid_count = np.sum(~np.isfinite(component))
    #         print(f"Warning: {component_name} contains {invalid_count} invalid values, applying fixes")
    #         component = np.nan_to_num(component, nan=default_value, posinf=1.0, neginf=-1.0)
        
    #     # 约束到合理范围
    #     component = np.clip(component, -1.0, 1.0)
    #     return component
    
    # # 修复各个法向量分量
    # normal_x = validate_and_fix_normal_component(normal_x, "normal_x", 0.0)
    # normal_y = validate_and_fix_normal_component(normal_y, "normal_y", 0.0) 
    # normal_z = validate_and_fix_normal_component(normal_z, "normal_z", 1.0)  # Z默认向上
    
    # 逐点归一化法向量，确保单位长度
    for i in range(normal_z.shape[0]):
        for j in range(normal_z.shape[1]):
            norm_vec = np.array([normal_x[i, j], normal_y[i, j], normal_z[i, j]])
            norm_length = np.linalg.norm(norm_vec)
            
            if norm_length > 1e-8:
                norm_vec = norm_vec / norm_length
            else:
                norm_vec = np.array([0.0, 0.0, 1.0])  # 默认向上
            
            normal_x[i, j] = norm_vec[0]
            normal_y[i, j] = norm_vec[1]
            normal_z[i, j] = norm_vec[2]
    
    # 构造编码输入[H, W, 6]
    goal_pos = goal_state[:2]  # 终点位置 (x, y)
    start_pos = start_state[:2]  # 起点位置 (x, y)
    goal_angle = goal_state[2]  # 终点朝向
    start_angle = start_state[2]  # 起点朝向
    
    goal_index = geom2pix(goal_pos, res=res, size=normal_z.shape[:2])
    start_index = geom2pix(start_pos, res=res, size=normal_z.shape[:2])

    # 起点区域（使用start_index）
    start_start_y = max(0, start_index[0] - receptive_field//2)
    start_start_x = max(0, start_index[1] - receptive_field//2)
    start_end_y = min(normal_z.shape[0], start_index[0] + receptive_field//2)
    start_end_x = min(normal_z.shape[1], start_index[1] + receptive_field//2)

    # 终点区域（使用goal_index）
    goal_start_y = max(0, goal_index[0] - receptive_field//2)
    goal_start_x = max(0, goal_index[1] - receptive_field//2)
    goal_end_y = min(normal_z.shape[0], goal_index[0] + receptive_field//2)
    goal_end_x = min(normal_z.shape[1], goal_index[1] + receptive_field//2)
    
    # 上下文地图： 起点终点的 位置标记 + 朝向的余弦值 + 朝向的正弦值，组成3通道输入
    context_map = np.zeros((*normal_z.shape[:2], 3))  # [H, W, 3]
    context_map[goal_start_y:goal_end_y, goal_start_x:goal_end_x, 0] = 1.0  # 终点标记为1
    context_map[start_start_y:start_end_y, start_start_x:start_end_x, 0] = -1.0  # 起点标记为-1
    
    # 检查角度值是否有效
    if not (np.isfinite(goal_angle) and np.isfinite(start_angle)):
        print("Warning: Invalid angles detected, using default values")
        goal_angle = 0.0
        start_angle = 0.0
    
    context_map[goal_start_y:goal_end_y, goal_start_x:goal_end_x, 1] = np.cos(goal_angle)  # 终点朝向
    context_map[goal_start_y:goal_end_y, goal_start_x:goal_end_x, 2] = np.sin(goal_angle)  # 终点朝向
    context_map[start_start_y:start_end_y, start_start_x:start_end_x, 1] = np.cos(start_angle)  # 起点朝向
    context_map[start_start_y:start_end_y, start_start_x:start_end_x, 2] = np.sin(start_angle)  # 起点朝向

    # 拼接为6通道
    encoded_input = np.concatenate((normal_x[:, :, None], normal_y[:, :, None], normal_z[:, :, None], context_map[:, :, :3]), axis=2)
    
    # 最终检查编码输入的有效性
    if not np.all(np.isfinite(encoded_input)):
        print("Warning: Final encoded input contains invalid values, applying final cleanup")
        encoded_input = np.nan_to_num(encoded_input, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return encoded_input

# def get_encoder_input(normal_z, goal_pos, start_pos, normal_x, normal_y):
#     goal_index = geom2pix(goal_pos, res=res, size=normal_z.shape[:2])
#     start_index = geom2pix(start_pos, res=res, size=normal_z.shape[:2])

#     # 起点区域（使用start_index）
#     start_start_y = max(0, start_index[0] - receptive_field//2)
#     start_start_x = max(0, start_index[1] - receptive_field//2)
#     start_end_y = min(normal_z.shape[0], start_index[0] + receptive_field//2)
#     start_end_x = min(normal_z.shape[1], start_index[1] + receptive_field//2)

#     # 终点区域（使用goal_index）
#     goal_start_y = max(0, goal_index[0] - receptive_field//2)
#     goal_start_x = max(0, goal_index[1] - receptive_field//2)
#     goal_end_y = min(normal_z.shape[0], goal_index[0] + receptive_field//2)
#     goal_end_x = min(normal_z.shape[1], goal_index[1] + receptive_field//2)
    
#     # 上下文地图： 起点终点的位置标记
#     context_map = np.zeros(normal_z.shape[:2])  # [H, W]
#     context_map[goal_start_y:goal_end_y, goal_start_x:goal_end_x] = 1.0  # 终点标记为1
#     context_map[start_start_y:start_end_y, start_start_x:start_end_x] = -1.0  # 起点标记为-1

#     # 构造θ=<nx,ny>->(cosθ, sinθ)的映射
#     angle_map = np.zeros((normal_z.shape[0], normal_z.shape[1], 2))  # [H, W, 2]
#     for i in range(normal_z.shape[0]):
#         for j in range(normal_z.shape[1]):
#             n_xy_norm = np.linalg.norm([normal_x[i, j], normal_y[i, j]])
#             if n_xy_norm == 0:
#                 # 如果法向量为零，避免除以零
#                 angle_map[i, j, 0] = 0.0
#                 angle_map[i, j, 1] = 0.0
#             else:
#                 # 归一化法向量
#                 angle_map[i, j, 0] = normal_x[i, j] / n_xy_norm
#                 angle_map[i, j, 1] = normal_y[i, j] / n_xy_norm

#     # 拼接为4通道
#     encoded_input = np.concatenate((normal_z[:, :, None], context_map[:, :, None], angle_map[:, :, :2]), axis=2)  # [H, W, 4]
#     return encoded_input

def compute_terrain_direction(nx, ny, nz, cos_theta, sin_theta):
    """
    根据地形法向量调整运动方向，确保方向在可达范围内
    
    Args:
        nx: 法向量x分量 (标量或数组)
        ny: 法向量y分量 (标量或数组)  
        nz: 法向量z分量 (标量或数组)
        cos_theta: 目标方向角度的余弦值 (标量或数组)
        sin_theta: 目标方向角度的正弦值 (标量或数组)
    
    Returns:
        np.ndarray: 调整后的[cos_theta, sin_theta]
    """
    # 转换为numpy数组以支持向量化操作
    nx = np.asarray(nx)
    ny = np.asarray(ny)
    nz = np.asarray(nz)
    cos_theta = np.asarray(cos_theta)
    sin_theta = np.asarray(sin_theta)
    
    # 如果输入为空，直接返回
    if nx.size == 0 or ny.size == 0 or nz.size == 0:
        return np.array([cos_theta, sin_theta])
    
    # 从cos和sin计算当前角度
    current_theta = np.arctan2(sin_theta, cos_theta)
    
    # 地形约束参数
    h = 8           # 机器人高度
    min_edge = 10   # 最小边长约束
    max_edge = 20   # 最大边长约束
    
    # 计算地形倾斜角度对应的tan值
    # 这里假设地形倾斜角度与法向量相关
    terrain_slope = np.arctan2(np.sqrt(nx**2 + ny**2), np.abs(nz))
    
    # 计算约束参数b
    b_vals = h * np.tan(terrain_slope)
    
    # 根据b值分类地形约束类型
    mask_reachable = b_vals < min_edge  # 完全可达
    mask_partial = (b_vals >= min_edge) & (b_vals < max_edge)  # 部分可达
    mask_complex = (b_vals >= max_edge) & (b_vals < np.sqrt(max_edge**2 + min_edge**2))  # 复杂约束
    mask_unreachable = b_vals >= np.sqrt(max_edge**2 + min_edge**2)  # 完全不可达
    
    # 初始化调整后的角度
    adjusted_theta = current_theta.copy()
    
    # 处理完全不可达区域：设置为零方向
    if np.any(mask_unreachable):
        adjusted_theta[mask_unreachable] = 0.0
    
    # # 处理完全可达区域：应用地形约束但不限制方向
    # if np.any(mask_reachable):
    #     reachable_indices = np.where(mask_reachable)[0]
    #     reachable_theta = current_theta[mask_reachable]
    #     reachable_nx = nx[mask_reachable]
    #     reachable_ny = ny[mask_reachable]
    #     reachable_nz = nz[mask_reachable]
        
    #     # 计算地形法向量的影响：从局部坐标系转换到全局坐标系
    #     normal_proj_angles = np.arctan2(reachable_ny, reachable_nx)
        
    #     # 对于完全可达区域，我们仍然要考虑地形倾斜的影响
    #     # 使用简化的地形校正：theta_global = theta_local + terrain_correction
    #     terrain_correction = normal_proj_angles * 0.3  # 地形影响权重可调
        
    #     # 应用地形校正
    #     corrected_theta = reachable_theta + terrain_correction
        
    #     # 标准化角度到[-π, π]
    #     corrected_theta = np.where(corrected_theta > np.pi, corrected_theta - 2*np.pi, corrected_theta)
    #     corrected_theta = np.where(corrected_theta < -np.pi, corrected_theta + 2*np.pi, corrected_theta)
        
    #     # 更新调整后的角度
    #     if adjusted_theta.ndim == 0:  # 标量情况
    #         adjusted_theta = corrected_theta[0] if len(corrected_theta) > 0 else adjusted_theta
    #     else:  # 数组情况
    #         adjusted_theta[mask_reachable] = corrected_theta
    
    # 处理部分可达区域
    if np.any(mask_partial):
        # print(f"进入部分可达区域处理，共{np.sum(mask_partial)}个点")
        partial_b = b_vals[mask_partial]
        partial_theta = current_theta[mask_partial]
        partial_nx = nx[mask_partial]
        partial_ny = ny[mask_partial]
        partial_nz = nz[mask_partial]
        
        # print(f"部分可达区域 b值范围: {partial_b.min():.2f} - {partial_b.max():.2f}")
        # print(f"部分可达区域 角度范围: {partial_theta.min():.3f} - {partial_theta.max():.3f} rad")
        
        # 计算约束边界角度
        s1_vals = np.arcsin(min_edge / partial_b)
        e1_vals = np.pi - s1_vals
        s2_vals = -s1_vals
        e2_vals = -np.pi + s1_vals
        
        # print(f"局部约束边界 s1: {s1_vals.min():.3f}-{s1_vals.max():.3f}, e1: {e1_vals.min():.3f}-{e1_vals.max():.3f}")
        # print(f"局部约束边界 s2: {s2_vals.min():.3f}-{s2_vals.max():.3f}, e2: {e2_vals.min():.3f}-{e2_vals.max():.3f}")
        
        # 考虑地形法向量的影响：从局部坐标系转换到全局坐标系
        # theta_global = arctan2(ny,nx) + arctan(nz*tan(theta_local))
        normal_proj_angles = np.arctan2(partial_ny, partial_nx)
        # print(f"法向量投影角度范围: {normal_proj_angles.min():.3f} - {normal_proj_angles.max():.3f} rad")
        
        # 计算全局坐标系下的边界参数，注意处理arctan的角度范围
        # 对于 arctan(nz*tan(theta_local))，需要根据原始角度的象限来调整结果
        def safe_arctan_transform(nz_vals, theta_local_vals):
            """安全的arctan变换，考虑角度的正确象限"""
            tan_vals = np.tan(theta_local_vals)
            arctan_result = np.arctan(nz_vals * tan_vals)
            
            # 如果原始角度在第二或第三象限（cos < 0），需要调整arctan结果
            # 第二象限：θ ∈ (π/2, π) => cos < 0, sin > 0
            # 第三象限：θ ∈ (-π, -π/2) => cos < 0, sin < 0
            cos_local = np.cos(theta_local_vals)
            sin_local = np.sin(theta_local_vals)
            
            # 调整第二象限的角度：arctan结果需要加π
            second_quadrant = (cos_local < 0) & (sin_local > 0)
            arctan_result = np.where(second_quadrant, arctan_result + np.pi, arctan_result)
            
            # 调整第三象限的角度：arctan结果需要减π
            third_quadrant = (cos_local < 0) & (sin_local < 0)
            arctan_result = np.where(third_quadrant, arctan_result - np.pi, arctan_result)
            
            return arctan_result
        
        # 计算全局坐标系下的边界参数
        s1_transform = safe_arctan_transform(partial_nz, s1_vals)
        e1_transform = safe_arctan_transform(partial_nz, e1_vals)
        s2_transform = safe_arctan_transform(partial_nz, s2_vals)
        e2_transform = safe_arctan_transform(partial_nz, e2_vals)
        
        s1_global = normal_proj_angles + s1_transform
        e1_global = normal_proj_angles + e1_transform
        s2_global = normal_proj_angles + s2_transform
        e2_global = normal_proj_angles + e2_transform
        
        # 标准化角度到[-π, π]
        s1_global = np.where(s1_global > np.pi, s1_global - 2*np.pi, s1_global)
        s1_global = np.where(s1_global < -np.pi, s1_global + 2*np.pi, s1_global)
        e1_global = np.where(e1_global > np.pi, e1_global - 2*np.pi, e1_global)
        e1_global = np.where(e1_global < -np.pi, e1_global + 2*np.pi, e1_global)
        s2_global = np.where(s2_global > np.pi, s2_global - 2*np.pi, s2_global)
        s2_global = np.where(s2_global < -np.pi, s2_global + 2*np.pi, s2_global)
        e2_global = np.where(e2_global > np.pi, e2_global - 2*np.pi, e2_global)
        e2_global = np.where(e2_global < -np.pi, e2_global + 2*np.pi, e2_global)
        
        # print(f"全局约束边界 s1: {s1_global.min():.3f}-{s1_global.max():.3f}, e1: {e1_global.min():.3f}-{e1_global.max():.3f}")
        # print(f"全局约束边界 s2: {s2_global.min():.3f}-{s2_global.max():.3f}, e2: {e2_global.min():.3f}-{e2_global.max():.3f}")
        
        # 检查是否在不可达区域（现在使用全局坐标系下的角度）
        unreachable_mask = ((partial_theta > s1_global) & (partial_theta < e1_global)) | \
                          ((partial_theta > e2_global) & (partial_theta < s2_global))
        
        # print(f"不可达角度检查: {np.sum(unreachable_mask)} 个角度被判定为不可达")
        
        # 对不可达角度进行调整：找到最近的可达边界
        if np.any(unreachable_mask):
            # print(f"开始修正不可达角度...")
            unreachable_indices = np.where(unreachable_mask)[0]
            partial_indices = np.where(mask_partial)[0]  # 获取部分可达区域在原数组中的索引
            
            for idx in unreachable_indices:
                current_angle = partial_theta[idx]
                # print(f"  修正角度 {idx}: 当前角度={current_angle:.3f} rad")
                # 找到最近的边界（全局坐标系下）
                boundaries = [s1_global[idx], e1_global[idx], s2_global[idx], e2_global[idx]]
                # print(f"    边界值: s1={s1_global[idx]:.3f}, e1={e1_global[idx]:.3f}, s2={s2_global[idx]:.3f}, e2={e2_global[idx]:.3f}")
                closest_boundary = min(boundaries, key=lambda x: abs(current_angle - x))
                # print(f"    最近边界: {closest_boundary:.3f} rad")
                # 调整角度 - 使用原数组中的正确索引
                original_idx = partial_indices[idx]
                if adjusted_theta.ndim == 0:  # 标量情况
                    adjusted_theta = closest_boundary
                else:  # 数组情况
                    adjusted_theta[original_idx] = closest_boundary
                # print(f"    角度已修正为: {closest_boundary:.3f} rad，原索引={original_idx}")
        # else:
            # print("没有不可达角度需要修正")
    
    # 处理复杂约束区域
    if np.any(mask_complex):
        complex_b = b_vals[mask_complex]
        complex_theta = current_theta[mask_complex]
        complex_nx = nx[mask_complex]
        complex_ny = ny[mask_complex]
        complex_nz = nz[mask_complex]
        
        # 计算复杂约束的边界参数
        r1_vals = np.arcsin(min_edge / complex_b)
        r2_vals = np.arccos(max_edge / complex_b)
        
        # 计算所有边界角度
        s1_vals = -r2_vals
        e1_vals = r2_vals
        s2_vals = r1_vals
        e2_vals = np.pi - r1_vals
        p1_vals = np.pi - r2_vals
        p2_vals = -np.pi + r2_vals
        s3_vals = -np.pi + r1_vals
        e3_vals = -r1_vals
        
        # 考虑地形法向量的影响：从局部坐标系转换到全局坐标系
        # theta_global = arctan2(ny,nx) + arctan(nz*tan(theta_local))
        normal_proj_angles = np.arctan2(complex_ny, complex_nx)
        
        # 计算全局坐标系下的边界参数，注意处理arctan的角度范围
        def safe_arctan_transform(nz_vals, theta_local_vals):
            """安全的arctan变换，考虑角度的正确象限"""
            tan_vals = np.tan(theta_local_vals)
            arctan_result = np.arctan(nz_vals * tan_vals)
            
            # 如果原始角度在第二或第三象限（cos < 0），需要调整arctan结果
            cos_local = np.cos(theta_local_vals)
            sin_local = np.sin(theta_local_vals)
            
            # 调整第二象限的角度：arctan结果需要加π
            second_quadrant = (cos_local < 0) & (sin_local > 0)
            arctan_result = np.where(second_quadrant, arctan_result + np.pi, arctan_result)
            
            # 调整第三象限的角度：arctan结果需要减π
            third_quadrant = (cos_local < 0) & (sin_local < 0)
            arctan_result = np.where(third_quadrant, arctan_result - np.pi, arctan_result)
            
            return arctan_result
        
        # 计算全局坐标系下的边界参数
        s1_transform = safe_arctan_transform(complex_nz, s1_vals)
        e1_transform = safe_arctan_transform(complex_nz, e1_vals)
        s2_transform = safe_arctan_transform(complex_nz, s2_vals)
        e2_transform = safe_arctan_transform(complex_nz, e2_vals)
        p1_transform = safe_arctan_transform(complex_nz, p1_vals)
        p2_transform = safe_arctan_transform(complex_nz, p2_vals)
        s3_transform = safe_arctan_transform(complex_nz, s3_vals)
        e3_transform = safe_arctan_transform(complex_nz, e3_vals)
        
        s1_global = normal_proj_angles + s1_transform
        e1_global = normal_proj_angles + e1_transform
        s2_global = normal_proj_angles + s2_transform
        e2_global = normal_proj_angles + e2_transform
        p1_global = normal_proj_angles + p1_transform
        p2_global = normal_proj_angles + p2_transform
        s3_global = normal_proj_angles + s3_transform
        e3_global = normal_proj_angles + e3_transform
        
        # 标准化所有角度到[-π, π]
        def normalize_angle(angle):
            angle = np.where(angle > np.pi, angle - 2*np.pi, angle)
            angle = np.where(angle < -np.pi, angle + 2*np.pi, angle)
            return angle
        
        s1_global = normalize_angle(s1_global)
        e1_global = normalize_angle(e1_global)
        s2_global = normalize_angle(s2_global)
        e2_global = normalize_angle(e2_global)
        p1_global = normalize_angle(p1_global)
        p2_global = normalize_angle(p2_global)
        s3_global = normalize_angle(s3_global)
        e3_global = normalize_angle(e3_global)
        
        # 检查是否在不可达区域（复杂约束的多个区间，使用全局坐标系）
        unreachable_mask = (
            ((complex_theta > s1_global) & (complex_theta < e1_global)) |
            ((complex_theta > s2_global) & (complex_theta < e2_global)) |
            ((complex_theta > s3_global) & (complex_theta < e3_global)) |
            (complex_theta < p2_global) |
            (complex_theta > p1_global)
        )
        
        # 对不可达角度进行调整：找到最近的可达边界
        if np.any(unreachable_mask):
            unreachable_indices = np.where(unreachable_mask)[0]
            complex_indices = np.where(mask_complex)[0]  # 获取复杂约束区域在原数组中的索引
            
            for idx in unreachable_indices:
                current_angle = complex_theta[idx]
                # 找到最近的边界（包括所有复杂约束的边界，全局坐标系下）
                boundaries = [
                    s1_global[idx], e1_global[idx], s2_global[idx], e2_global[idx],
                    s3_global[idx], e3_global[idx], p1_global[idx], p2_global[idx]
                ]
                closest_boundary = min(boundaries, key=lambda x: abs(current_angle - x))
                # 调整角度 - 使用原数组中的正确索引
                original_idx = complex_indices[idx]
                if adjusted_theta.ndim == 0:  # 标量情况
                    adjusted_theta = closest_boundary
                else:  # 数组情况
                    adjusted_theta[original_idx] = closest_boundary
    
    # 将调整后的角度转换回cos和sin
    adjusted_cos = np.cos(adjusted_theta)
    adjusted_sin = np.sin(adjusted_theta)
    
    return np.array([adjusted_cos, adjusted_sin])

def compute_map_yaw_bins(normal_x, normal_y, normal_z, yaw_bins=18):
    """
    计算地图上每个点的分箱角度是否会倾覆（PyTorch版本，高效批量处理）
    
    基于相同的物理约束来判断每个yaw_bin角度是否会导致倾覆，
    不会倾覆的角度分箱标为1，会倾覆的标为0。
    
    Args:
        normal_x: 地形法向量x分量 (H, W) 
        normal_y: 地形法向量y分量 (H, W)
        normal_z: 地形法向量z分量 (H, W)
        yaw_bins: 朝向角度的分箱数量，默认为18

    Returns:
        torch.Tensor: 每个点每个角度分箱的倾覆状态，形状为(H, W, yaw_bins)，1表示不会倾覆，0表示会倾覆
    """
    # 确保输入是torch张量
    if not isinstance(normal_x, torch.Tensor):
        normal_x = torch.tensor(normal_x, dtype=torch.float32)
        normal_y = torch.tensor(normal_y, dtype=torch.float32)
        normal_z = torch.tensor(normal_z, dtype=torch.float32)

    # PCL surface normals have an arbitrary sign.  The saved dataset already
    # uses the intended horizontal convention, while nz is predominantly
    # negative.  Orient the normal upward before *all* angular calculations;
    # using abs(nz) only in the slope magnitude but signed nz below rotates the
    # yaw feasibility intervals even though the underlying plane is unchanged.
    normal_z = torch.abs(normal_z)
    
    device = normal_x.device
    H, W = normal_x.shape
    
    # 地形约束参数
    # h = 8.0         # 机器人高度
    # min_edge = 10.0 # 最小边长约束
    # max_edge = 20.0 # 最大边长约束
    # h = 35.0         # 机器人高度
    # min_edge = 8.0 # 最小边长约束
    # max_edge = 15.0 # 最大边长约束
    h = 15.0         # 机器人高度
    min_edge = 8.0 # 最小边长约束
    max_edge = 15.0 # 最大边长约束
    
    # 计算地形倾斜角度对应的tan值
    terrain_slope = torch.arctan2(torch.sqrt(normal_x**2 + normal_y**2), torch.abs(normal_z))
    
    # 计算约束参数b
    b_vals = h * torch.tan(terrain_slope)
    
    # 根据b值分类地形约束类型
    mask_reachable = b_vals < min_edge  # 完全可达
    mask_partial = (b_vals >= min_edge) & (b_vals < max_edge)  # 部分可达
    mask_complex = (b_vals >= max_edge) & (b_vals < torch.sqrt(torch.tensor(max_edge**2 + min_edge**2, device=device)))  # 复杂约束
    mask_unreachable = b_vals >= torch.sqrt(torch.tensor(max_edge**2 + min_edge**2, device=device))  # 完全不可达
    
    # 初始化结果数组：所有角度分箱都标记为不会倾覆(1)
    yaw_stability = torch.ones((H, W, yaw_bins), dtype=torch.float32, device=device)
    
    # 定义角度分箱的中心角度：从-π到π均匀分布
    bin_angles = torch.linspace(-torch.pi, torch.pi, yaw_bins + 1, device=device)[:-1]  # 去掉最后一个
    
    def safe_arctan_transform(nz_vals, theta_local_vals):
        """安全的arctan变换，考虑角度的正确象限"""
        tan_vals = torch.tan(theta_local_vals)
        arctan_result = torch.arctan(nz_vals * tan_vals)
        
        # 如果原始角度在第二或第三象限（cos < 0），需要调整arctan结果
        cos_local = torch.cos(theta_local_vals)
        sin_local = torch.sin(theta_local_vals)
        
        # 调整第二象限的角度：arctan结果需要加π
        second_quadrant = (cos_local < 0) & (sin_local > 0)
        arctan_result = torch.where(second_quadrant, arctan_result + torch.pi, arctan_result)
        
        # 调整第三象限的角度：arctan结果需要减π
        third_quadrant = (cos_local < 0) & (sin_local < 0)
        arctan_result = torch.where(third_quadrant, arctan_result - torch.pi, arctan_result)
        
        return arctan_result

    def normalize_angle(angle):
        """标准化角度到[-π, π]"""
        angle = torch.where(angle > torch.pi, angle - 2*torch.pi, angle)
        angle = torch.where(angle < -torch.pi, angle + 2*torch.pi, angle)
        return angle
    
    def check_angle_in_range_vectorized(angles, starts, ends):
        """向量化检查角度是否在范围内"""
        # angles: (yaw_bins,), starts: (N,), ends: (N,)
        # 返回: (N, yaw_bins) 布尔张量
        angles = angles.unsqueeze(0)  # (1, yaw_bins)
        starts = starts.unsqueeze(1)  # (N, 1)
        ends = ends.unsqueeze(1)      # (N, 1)
        
        angles = normalize_angle(angles)
        starts = normalize_angle(starts)
        ends = normalize_angle(ends)
        
        # 正常情况：start <= end
        normal_case = starts <= ends
        in_range_normal = (angles >= starts) & (angles <= ends) & normal_case
        
        # 跨越边界情况：start > end
        cross_boundary = starts > ends
        in_range_cross = ((angles >= starts) | (angles <= ends)) & cross_boundary
        
        return in_range_normal | in_range_cross
    
    # 处理完全不可达区域：所有角度都标记为会倾覆(0)
    yaw_stability[mask_unreachable] = 0.0
    
    # 处理部分可达区域 - 向量化处理
    if torch.any(mask_partial):
        # 获取部分可达区域的坐标和值
        partial_indices = torch.where(mask_partial)
        partial_b = b_vals[mask_partial]
        partial_nx = normal_x[mask_partial]
        partial_ny = normal_y[mask_partial]
        partial_nz = normal_z[mask_partial]
        
        # 批量计算约束边界角度（局部坐标系）
        s1_vals = torch.arcsin(min_edge / partial_b)
        e1_vals = torch.pi - s1_vals
        s2_vals = -s1_vals
        e2_vals = -torch.pi + s1_vals
        
        # 考虑地形法向量的影响：从局部坐标系转换到全局坐标系
        normal_proj_angles = torch.arctan2(partial_ny, partial_nx)
        
        # 计算全局坐标系下的边界参数
        s1_transforms = safe_arctan_transform(partial_nz, s1_vals)
        e1_transforms = safe_arctan_transform(partial_nz, e1_vals)
        s2_transforms = safe_arctan_transform(partial_nz, s2_vals)
        e2_transforms = safe_arctan_transform(partial_nz, e2_vals)
        
        s1_globals = normalize_angle(normal_proj_angles + s1_transforms)
        e1_globals = normalize_angle(normal_proj_angles + e1_transforms)
        s2_globals = normalize_angle(normal_proj_angles + s2_transforms)
        e2_globals = normalize_angle(normal_proj_angles + e2_transforms)
        
        # 向量化检查每个角度分箱是否在不可达区域
        in_unreachable_region1 = check_angle_in_range_vectorized(bin_angles, s1_globals, e1_globals)  # (N, yaw_bins)
        in_unreachable_region2 = check_angle_in_range_vectorized(bin_angles, e2_globals, s2_globals)  # (N, yaw_bins)
        
        unreachable_mask = in_unreachable_region1 | in_unreachable_region2  # (N, yaw_bins)
        
        # 更新yaw_stability
        for idx, (i, j) in enumerate(zip(partial_indices[0], partial_indices[1])):
            yaw_stability[i, j, unreachable_mask[idx]] = 0.0
    
    # 处理复杂约束区域 - 向量化处理
    if torch.any(mask_complex):
        # 获取复杂约束区域的坐标和值
        complex_indices = torch.where(mask_complex)
        complex_b = b_vals[mask_complex]
        complex_nx = normal_x[mask_complex]
        complex_ny = normal_y[mask_complex]
        complex_nz = normal_z[mask_complex]
        
        # 批量计算复杂约束的边界参数
        r1_vals = torch.arcsin(min_edge / complex_b)
        r2_vals = torch.arccos(max_edge / complex_b)
        
        # 计算所有边界角度（局部坐标系）
        s1_vals = -r2_vals
        e1_vals = r2_vals
        s2_vals = r1_vals
        e2_vals = torch.pi - r1_vals
        p1_vals = torch.pi - r2_vals
        p2_vals = -torch.pi + r2_vals
        s3_vals = -torch.pi + r1_vals
        e3_vals = -r1_vals
        
        # 考虑地形法向量的影响：从局部坐标系转换到全局坐标系
        normal_proj_angles = torch.arctan2(complex_ny, complex_nx)
        
        # 计算全局坐标系下的边界参数
        s1_transforms = safe_arctan_transform(complex_nz, s1_vals)
        e1_transforms = safe_arctan_transform(complex_nz, e1_vals)
        s2_transforms = safe_arctan_transform(complex_nz, s2_vals)
        e2_transforms = safe_arctan_transform(complex_nz, e2_vals)
        p1_transforms = safe_arctan_transform(complex_nz, p1_vals)
        p2_transforms = safe_arctan_transform(complex_nz, p2_vals)
        s3_transforms = safe_arctan_transform(complex_nz, s3_vals)
        e3_transforms = safe_arctan_transform(complex_nz, e3_vals)
        
        s1_globals = normalize_angle(normal_proj_angles + s1_transforms)
        e1_globals = normalize_angle(normal_proj_angles + e1_transforms)
        s2_globals = normalize_angle(normal_proj_angles + s2_transforms)
        e2_globals = normalize_angle(normal_proj_angles + e2_transforms)
        p1_globals = normalize_angle(normal_proj_angles + p1_transforms)
        p2_globals = normalize_angle(normal_proj_angles + p2_transforms)
        s3_globals = normalize_angle(normal_proj_angles + s3_transforms)
        e3_globals = normalize_angle(normal_proj_angles + e3_transforms)
        
        # 向量化检查每个角度分箱是否在不可达区域
        in_unreachable1 = check_angle_in_range_vectorized(bin_angles, s1_globals, e1_globals)
        in_unreachable2 = check_angle_in_range_vectorized(bin_angles, s2_globals, e2_globals)
        in_unreachable3 = check_angle_in_range_vectorized(bin_angles, s3_globals, e3_globals)
        
        # 处理p1和p2边界（单侧边界）
        bin_angles_expanded = bin_angles.unsqueeze(0)  # (1, yaw_bins)
        p1_expanded = p1_globals.unsqueeze(1)  # (N, 1)
        p2_expanded = p2_globals.unsqueeze(1)  # (N, 1)
        
        in_unreachable_p1 = bin_angles_expanded > p1_expanded
        in_unreachable_p2 = bin_angles_expanded < p2_expanded
        
        unreachable_mask = (in_unreachable1 | in_unreachable2 | in_unreachable3 | 
                           in_unreachable_p1 | in_unreachable_p2)  # (N, yaw_bins)
        
        # 更新yaw_stability
        for idx, (i, j) in enumerate(zip(complex_indices[0], complex_indices[1])):
            yaw_stability[i, j, unreachable_mask[idx]] = 0.0

    # Contract: axis 0 remains map row/y and axis 1 remains map column/x.
    # Do not transpose here; downstream converts HWD -> DHW explicitly.
    return yaw_stability

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def generate_sdf_from_yaw_stability(
    yaw_stability,            # torch.Tensor or np.ndarray, shape (H,W,Y), 1 safe / 0 occupied
    voxel_size_xy=0.1,
    yaw_weight=0.2,
    use_scipy=True,
):
    """
    返回带符号的 ESDF（signed distance）：内部为负，边界为 0，外部为正。
    与 generate_cost_map_from_yaw_stability 类似，但只返回 esdf（不做 sigmoid / cost 映射）。
    支持 scipy 快速路径或 PyTorch 退回实现（分块最近邻）。
    返回类型与输入一致（输入为 torch 则返回 torch.Tensor，否则返回 numpy.ndarray）。
    """
    input_is_torch = isinstance(yaw_stability, torch.Tensor)
    if input_is_torch:
        device = yaw_stability.device
        ys = yaw_stability.detach().to('cpu').numpy()
    else:
        ys = np.asarray(yaw_stability)

    H, W, Y = ys.shape
    occupied = (ys <= 0.5)

    # scipy's EDT assumes an implicit background when no target voxel exists.
    # Handle uniform maps explicitly so they do not acquire a position-dependent
    # pseudo-distance unrelated to the finite map.
    max_xy = math.hypot(H * voxel_size_xy, W * voxel_size_xy)
    max_yaw = math.pi * yaw_weight
    max_dist = math.sqrt(max_xy**2 + max_yaw**2) + 1.0
    if not occupied.any():
        esdf = np.full((H, W, Y), max_dist, dtype=np.float32)
        if input_is_torch:
            return torch.from_numpy(esdf).to(device=device, dtype=torch.float32)
        return esdf
    if occupied.all():
        esdf = np.full((H, W, Y), -max_dist, dtype=np.float32)
        if input_is_torch:
            return torch.from_numpy(esdf).to(device=device, dtype=torch.float32)
        return esdf

    if use_scipy:
        try:
            from scipy.ndimage import distance_transform_edt
            delta_theta = 2.0 * math.pi / float(Y)
            sampling = (voxel_size_xy, voxel_size_xy, yaw_weight * delta_theta)

            # 三倍平铺处理周期性
            occupied_tiled = np.concatenate([occupied, occupied, occupied], axis=2)  # (H,W,3Y)

            inv = ~occupied_tiled
            dist_to_occupied = distance_transform_edt(inv, sampling=sampling)
            dist_to_free = distance_transform_edt(occupied_tiled, sampling=sampling)

            signed_tiled = dist_to_occupied - dist_to_free  # 内部为负，外部为正
            esdf = signed_tiled[:, :, Y:2*Y].astype(np.float32)  # 取中段

            if input_is_torch:
                return torch.from_numpy(esdf).to(device=device, dtype=torch.float32)
            return esdf

        except Exception as e:
            # 退回到 PyTorch 实现
            print("WARNING: scipy EDT failed or unavailable, falling back to PyTorch SDF. Error:", e)
            use_scipy = False

    # PyTorch 退回实现（分块最近邻）
    if input_is_torch:
        ys_t = yaw_stability.to(dtype=torch.float32)
        device = ys_t.device
    else:
        ys_t = torch.from_numpy(ys.astype(np.float32))

    occupied_mask = (ys_t <= 0.5)
    occ_idx = torch.nonzero(occupied_mask, as_tuple=False)  # (M,3)
    free_idx = torch.nonzero(~occupied_mask, as_tuple=False)  # (F,3)
    M = occ_idx.shape[0]
    F = free_idx.shape[0]
    N = H * W * Y

    # 特殊情况
    if M == 0 and F == 0:
        esdf_map = torch.zeros((H, W, Y), dtype=torch.float32)
        if input_is_torch:
            return esdf_map.to(device=device)
        return esdf_map.cpu().numpy()

    if M == 0:
        esdf_map = torch.ones((H, W, Y), dtype=torch.float32) * float(max_dist)
        if input_is_torch:
            return esdf_map.to(device=device)
        return esdf_map.cpu().numpy()

    if F == 0:
        esdf_map = torch.ones((H, W, Y), dtype=torch.float32) * float(-max_dist)
        if input_is_torch:
            return esdf_map.to(device=device)
        return esdf_map.cpu().numpy()

    # 预计算 yaw 投影（与 cost 函数一致）
    ks = torch.arange(Y, dtype=torch.float32)
    theta_k = (ks + 0.5) * (2.0 * math.pi / float(Y))
    yaw_cos = torch.cos(theta_k) * yaw_weight
    yaw_sin = torch.sin(theta_k) * yaw_weight

    # 构建占据点与空闲点坐标 (x, y, yaw_x, yaw_y)
    occ_i = occ_idx[:,0].to(dtype=torch.float32)
    occ_j = occ_idx[:,1].to(dtype=torch.float32)
    occ_k = occ_idx[:,2].to(dtype=torch.long)
    occ_x = occ_i * voxel_size_xy
    occ_y = occ_j * voxel_size_xy
    occ_yaw_x = yaw_cos[occ_k]
    occ_yaw_y = yaw_sin[occ_k]
    occ_coords = torch.stack([occ_x, occ_y, occ_yaw_x, occ_yaw_y], dim=1).to(dtype=torch.float32)

    free_i = free_idx[:,0].to(dtype=torch.float32)
    free_j = free_idx[:,1].to(dtype=torch.float32)
    free_k = free_idx[:,2].to(dtype=torch.long)
    free_x = free_i * voxel_size_xy
    free_y = free_j * voxel_size_xy
    free_yaw_x = yaw_cos[free_k]
    free_yaw_y = yaw_sin[free_k]
    free_coords = torch.stack([free_x, free_y, free_yaw_x, free_yaw_y], dim=1).to(dtype=torch.float32)

    # 分块计算最小距离
    chunk_size = 200_000
    min_dists_occ = torch.empty(N, dtype=torch.float32)
    min_dists_free = torch.empty(N, dtype=torch.float32)
    device_coords = occ_coords.device

    if torch.cuda.is_available():
        occ_coords = occ_coords.cuda()
        free_coords = free_coords.cuda()
        device_coords = occ_coords.device

    def idx_to_coords_block(start, end):
        idxs = torch.arange(start, end, device=device_coords, dtype=torch.long)
        i = (idxs // (W * Y)).to(torch.float32)
        rem = idxs % (W * Y)
        j = (rem // Y).to(torch.float32)
        k = (rem % Y).to(torch.long)
        x = i * voxel_size_xy
        y = j * voxel_size_xy
        yaw_x = yaw_cos[k].to(device=device_coords)
        yaw_y = yaw_sin[k].to(device=device_coords)
        coords_block = torch.stack([x, y, yaw_x, yaw_y], dim=1)
        return coords_block

    start = 0
    while start < N:
        end = min(N, start + chunk_size)
        coords_block = idx_to_coords_block(start, end)

        if coords_block.device != occ_coords.device:
            coords_block = coords_block.to(device=occ_coords.device)
        dists_occ = torch.cdist(coords_block, occ_coords)  # (B, M)
        min_block_occ, _ = torch.min(dists_occ, dim=1)
        min_dists_occ[start:end] = min_block_occ.cpu()

        if coords_block.device != free_coords.device:
            coords_block = coords_block.to(device=free_coords.device)
        dists_free = torch.cdist(coords_block, free_coords)  # (B, F)
        min_block_free, _ = torch.min(dists_free, dim=1)
        min_dists_free[start:end] = min_block_free.cpu()

        start = end

    dist_occ_map = min_dists_occ.view(H, W, Y).to(dtype=torch.float32)
    dist_free_map = min_dists_free.view(H, W, Y).to(dtype=torch.float32)

    esdf_map = (dist_occ_map - dist_free_map).to(dtype=torch.float32)  # signed ESDF

    if input_is_torch:
        return esdf_map.to(device=device)
    else:
        return esdf_map.cpu().numpy()

def generate_cost_map_from_yaw_stability(
    yaw_stability,            # torch.Tensor or np.ndarray, shape (H,W,Y), 1 safe / 0 occupied
    voxel_size_xy=0.1,
    yaw_weight=0.2,
    d_safe=0.0,
    kalpa=0.6,
    use_scipy=True,
    return_esdf=False,
):
    """
    快速版本：优先使用 scipy.ndimage.distance_transform_edt（C 实现）。
    结果为带符号距离：内部为负，边界为0，外部为正。
    对 yaw 周期性做三倍平铺处理以正确计算环绕距离。
    """
    # --- 标准化输入 ---
    input_is_torch = isinstance(yaw_stability, torch.Tensor)
    if input_is_torch:
        device = yaw_stability.device
        ys = yaw_stability.detach().to('cpu').numpy()
    else:
        ys = np.asarray(yaw_stability)

    H, W, Y = ys.shape
    occupied = (ys <= 0.5)   # True 表示占据（不安全 / capsized）

    # 快速路径：scipy 的 EDT（推荐）
    if use_scipy:
        try:
            from scipy.ndimage import distance_transform_edt
            # yaw 每格对应的弧度
            delta_theta = 2.0 * math.pi / float(Y)
            sampling = (voxel_size_xy, voxel_size_xy, yaw_weight * delta_theta)

            # 三倍平铺以考虑周期性
            occupied_tiled = np.concatenate([occupied, occupied, occupied], axis=2)  # (H,W,3Y)

            # 计算到占据点的距离（外部距离）和到空闲点的距离（内部距离）
            # 注意：distance_transform_edt 的输入布尔数组的语义：
            # distance_transform_edt(input) 计算到 False(0) 的距离。因此：
            # - inv = ~occupied_tiled -> False 在占据点，distance_transform_edt(inv) 得到到占据点的距离
            # - distance_transform_edt(occupied_tiled) 得到到空闲点的距离
            inv = ~occupied_tiled
            dist_to_occupied = distance_transform_edt(inv, sampling=sampling)   # 到占据点（外部为正）
            dist_to_free = distance_transform_edt(occupied_tiled, sampling=sampling)  # 到空闲（内部为正）

            signed_tiled = dist_to_occupied - dist_to_free  # 内部为负，外部为正
            esdf = signed_tiled[:, :, Y:2*Y].astype(np.float32)  # (H,W,Y)

            # 计算 cost（保留原有变换）
            z = (-(esdf - d_safe) / (kalpa + 1e-12))
            z = np.clip(z, -50.0, 50.0)
            costs = 1.0 / (1.0 + np.exp(-z))  # sigmoid

            if input_is_torch:
                costs_t = torch.from_numpy(costs).to(device=device, dtype=torch.float32)
                if return_esdf:
                    esdf_t = torch.from_numpy(esdf).to(device=device, dtype=torch.float32)
                    return costs_t, esdf_t
                return costs_t
            else:
                if return_esdf:
                    return costs, esdf
                return costs

        except Exception as e:
            # 如果 scipy 不可用或调用失败，退回下面的 PyTorch 近似实现（告警）
            print("WARNING: scipy EDT 快速路径失败或不可用，退回 PyTorch 实现。错误：", e)
            use_scipy = False

    # --- 退回：改良的 PyTorch 分块最近邻（慢但健壮） ---
    if input_is_torch:
        ys_t = yaw_stability.to(dtype=torch.float32)
        device = ys_t.device
    else:
        ys_t = torch.from_numpy(ys.astype(np.float32))

    occupied_mask = (ys_t <= 0.5)
    occ_idx = torch.nonzero(occupied_mask, as_tuple=False)  # (M,3)
    free_idx = torch.nonzero(~occupied_mask, as_tuple=False)  # (F,3)
    M = occ_idx.shape[0]
    F = free_idx.shape[0]
    N = H * W * Y

    # 特殊情况处理：全无占据或全被占据
    max_xy = math.hypot(H * voxel_size_xy, W * voxel_size_xy)
    max_yaw = math.pi * yaw_weight
    max_dist = math.sqrt(max_xy**2 + max_yaw**2) + 1.0

    if M == 0 and F == 0:
        # 极端不可用情况，返回零地图
        esdf_map = torch.zeros((H, W, Y), dtype=torch.float32)
        z = (-(esdf_map - d_safe) / (kalpa + 1e-12))
        costs = torch.sigmoid(torch.clamp(z, min=-50.0, max=50.0))
        if input_is_torch:
            if return_esdf:
                return costs, esdf_map
            return costs
        else:
            c_np = costs.cpu().numpy()
            if return_esdf:
                return c_np, esdf_map.cpu().numpy()
            return c_np

    if M == 0:
        # 没有占据点 -> 所有点到占据的距离为大值，到空闲距离为0 => 带符号为 +max_dist
        esdf_map = torch.ones((H, W, Y), dtype=torch.float32) * float(max_dist)
        z = (-(esdf_map - d_safe) / (kalpa + 1e-12))
        costs = torch.sigmoid(torch.clamp(z, min=-50.0, max=50.0))
        if input_is_torch:
            if return_esdf:
                return costs, esdf_map
            return costs
        else:
            c_np = costs.cpu().numpy()
            if return_esdf:
                return c_np, esdf_map.cpu().numpy()
            return c_np

    if F == 0:
        # 全部被占据 -> 到空闲距离为大值，到占据距离为0 => 带符号为 -max_dist
        esdf_map = torch.ones((H, W, Y), dtype=torch.float32) * float(-max_dist)
        z = (-(esdf_map - d_safe) / (kalpa + 1e-12))
        costs = torch.sigmoid(torch.clamp(z, min=-50.0, max=50.0))
        if input_is_torch:
            if return_esdf:
                return costs, esdf_map
            return costs
        else:
            c_np = costs.cpu().numpy()
            if return_esdf:
                return c_np, esdf_map.cpu().numpy()
            return c_np

    # 预计算 yaw 坐标
    ks = torch.arange(Y, dtype=torch.float32)
    theta_k = (ks + 0.5) * (2.0 * math.pi / float(Y))
    yaw_cos = torch.cos(theta_k) * yaw_weight
    yaw_sin = torch.sin(theta_k) * yaw_weight

    # 构建占据点和空闲点的坐标（x, y, yaw_x, yaw_y）
    occ_i = occ_idx[:,0].to(dtype=torch.float32)
    occ_j = occ_idx[:,1].to(dtype=torch.float32)
    occ_k = occ_idx[:,2].to(dtype=torch.long)
    occ_x = occ_i * voxel_size_xy
    occ_y = occ_j * voxel_size_xy
    occ_yaw_x = yaw_cos[occ_k]
    occ_yaw_y = yaw_sin[occ_k]
    occ_coords = torch.stack([occ_x, occ_y, occ_yaw_x, occ_yaw_y], dim=1).to(dtype=torch.float32)

    free_i = free_idx[:,0].to(dtype=torch.float32)
    free_j = free_idx[:,1].to(dtype=torch.float32)
    free_k = free_idx[:,2].to(dtype=torch.long)
    free_x = free_i * voxel_size_xy
    free_y = free_j * voxel_size_xy
    free_yaw_x = yaw_cos[free_k]
    free_yaw_y = yaw_sin[free_k]
    free_coords = torch.stack([free_x, free_y, free_yaw_x, free_yaw_y], dim=1).to(dtype=torch.float32)

    # 分块遍历所有 N 点，按 chunk_size 控制显存，分别计算到占据/到空闲的最小距离
    chunk_size = 200_000  # 可调整
    min_dists_occ = torch.empty(N, dtype=torch.float32)
    min_dists_free = torch.empty(N, dtype=torch.float32)
    device = occ_coords.device
    # 如果在GPU上，把 occ_coords/free_coords 放GPU加速
    if torch.cuda.is_available():
        occ_coords = occ_coords.cuda()
        free_coords = free_coords.cuda()
        device = occ_coords.device

    def idx_to_coords_block(start, end):
        idxs = torch.arange(start, end, device=device, dtype=torch.long)
        i = (idxs // (W * Y)).to(torch.float32)
        rem = idxs % (W * Y)
        j = (rem // Y).to(torch.float32)
        k = (rem % Y).to(torch.long)
        x = i * voxel_size_xy
        y = j * voxel_size_xy
        yaw_x = yaw_cos[k].to(device=device)
        yaw_y = yaw_sin[k].to(device=device)
        coords_block = torch.stack([x, y, yaw_x, yaw_y], dim=1)
        return coords_block

    start = 0
    while start < N:
        end = min(N, start + chunk_size)
        coords_block = idx_to_coords_block(start, end)
        # 把 coords_block 与 occ_coords/free_coords 放在同设备
        if coords_block.device != occ_coords.device:
            coords_block = coords_block.to(device=occ_coords.device)
        # 计算到占据点距离并取最小
        dists_occ = torch.cdist(coords_block, occ_coords)  # (B, M)
        min_block_occ, _ = torch.min(dists_occ, dim=1)
        min_dists_occ[start:end] = min_block_occ.cpu()

        # 计算到空闲点距离并取最小
        if coords_block.device != free_coords.device:
            coords_block = coords_block.to(device=free_coords.device)
        dists_free = torch.cdist(coords_block, free_coords)  # (B, F)
        min_block_free, _ = torch.min(dists_free, dim=1)
        min_dists_free[start:end] = min_block_free.cpu()

        start = end

    dist_occ_map = min_dists_occ.view(H, W, Y).to(dtype=torch.float32)
    dist_free_map = min_dists_free.view(H, W, Y).to(dtype=torch.float32)

    esdf_map = (dist_occ_map - dist_free_map).to(dtype=torch.float32)  # 带符号 ESDF：内部为负，外部为正
    z = (-(esdf_map - d_safe) / (kalpa + 1e-12))
    costs_t = torch.sigmoid(torch.clamp(z, min=-50.0, max=50.0))

    if input_is_torch:
        if return_esdf:
            return costs_t.to(device=device), esdf_map.to(device=device)
        return costs_t.to(device=device)
    else:
        c_np = costs_t.cpu().numpy()
        if return_esdf:
            return c_np, esdf_map.cpu().numpy()
        return c_np


class UnevenPathDataLoader(Dataset):
    """
    UnevenPathDataLoader: 不平坦地面的路径数据加载器
    
    【核心功能】
    从不平坦地面的路径规划数据集中加载训练样本，用于训练模型在复杂地形下的路径规划能力。
    
    【数据集特点】
    1. 地图类型：包含不平坦地面的复杂环境
    2. 地图尺寸：10m×10m，分辨率0.01m
    3. 路径类型：包含成功和失败的路径规划任务
    4. 数据格式：Pickle文件，包含地图和路径信息:
    (1)地图文件：`map.p`
    ```python
    {
        'tensor': np.array,          # [H, W, 4] 四通道张量
        'bounds': (min_x, max_x, min_y, max_y),
        'resolution': 0.2,           # 栅格分辨率
        'map_name': 'desert',        # 地图名称
        'channels': ['elevation', 'normal_x', 'normal_y', 'normal_z'],
        'shape': (height, width, 4)
    }
    ```
    例如：
    ```python
    {'tensor': array([
        [[ 1.91332285e+00,  1.66046888e-01, -2.47567687e-02, 9.85807061e-01],
         [ 1.89383935e+00,  1.66046888e-01, -2.47567687e-02, 9.85807061e-01],
         [ 1.87698874e+00,  1.80939287e-01, -3.28837261e-02, 9.82944369e-01],
         ...,
         [ 7.89104671e-01,  7.29798600e-02, -3.43800820e-02, 9.96740639e-01],
         [ 7.80242312e-01,  9.87786874e-02, -4.03737016e-02, 9.94290054e-01],
         [ 7.70666865e-01,  9.87786874e-02, -4.03737016e-02, 9.94290054e-01]],

        ...,

        [[ 1.12184868e+00, -3.71789820e-02,  4.94028963e-02, 9.98086691e-01],
         [ 1.12459070e+00, -3.71789820e-02,  4.94028963e-02, 9.98086691e-01],
         [ 1.12810575e+00, -5.22360252e-03,  3.20198573e-02, 9.99473572e-01],
         ...,  
         [ 8.72300527e-01,  9.26663429e-02, -1.00500155e-02, 9.95646477e-01],
         [ 8.60928348e-01,  1.48325890e-01,  3.07161896e-03, 9.88933742e-01],
         [ 8.48599134e-01,  1.48325890e-01,  3.07161896e-03, 9.88933742e-01]]], shape=(100, 100, 4)), 
      'bounds': (-5.0, 5.0, -5.0, 5.0), 
      'resolution': 0.1, 
      'map_name': 'desert', 
      'channels': ['elevation', 'normal_x', 'normal_y', 'normal_z'], 
      'shape': (100, 100, 4)}
    ```
    (2)轨迹文件：`path_{id}.p`
    ```python
    {
        'valid': True,               # 是否有效路径
        'path': np.array,            # [N, 3] 轨迹点 [x, y, yaw]
        'map_name': 'desert'         # 关联的地图名称
    }
    ```
    (3)目录格式：
    ```
    ├── desert/
        |── map.p
        ├── path_0.p
        ├── path_1.p
        └── ...
    ├── forest/
        ├── map.p
        ├── path_0.p
        ├── path_1.p
        └── ...
    └── ...
    ```
    
    【数据加载示例】
    ```python
    import pickle
    import numpy as np

    # 加载地图数据
    with open('map.p', 'rb') as f:
        map_data = pickle.load(f)

    tensor = map_data['tensor']  # [H, W, 4]
    elevation = tensor[:, :, 0]
    normal_x = tensor[:, :, 1]
    normal_y = tensor[:, :, 2]
    normal_z = tensor[:, :, 3]

    # 加载轨迹数据
    with open('path_0.p', 'rb') as f:
        path_data = pickle.load(f)

    trajectory = path_data['path']  # [N, 3]
    map_name = path_data['map_name']  # 'desert'
    ```

    """

    def __init__(
        self,
        env_list,
        dataFolder,
        compute_stability_map=False,
        use_precomputed_stability=False,
        stability_map_filename='stability_map.npz',
        compute_stability_if_missing=False,
        partial_observation=False,
        include_mask=False,
        mask_seed=2026,
        p_mask=0.5,
        dynamic_mask_noise=False,
        mask_mode="stage1_demo_valid",
        mask_source="legacy",
        vehicle_radius_meters=SAFETY_COST_CONFIG.vehicle_radius_meters,
        encode_path_coordinates=False,
        encode_gauge_state=None,
    ):
        self.num_env = len(env_list)
        self.env_list = env_list
        self.dataFolder = dataFolder
        self.compute_stability_map = compute_stability_map
        self.use_precomputed_stability = use_precomputed_stability
        self.stability_map_filename = stability_map_filename
        self.compute_stability_if_missing = compute_stability_if_missing
        self.partial_observation = bool(partial_observation)
        self.include_mask = bool(include_mask)
        self.mask_seed = int(mask_seed)
        self.p_mask = float(p_mask)
        self.dynamic_mask_noise = bool(dynamic_mask_noise)
        if mask_mode not in {"stage1_demo_valid", "stage2_independent"}:
            raise ValueError(
                "mask_mode 必须为 stage1_demo_valid 或 stage2_independent"
            )
        self.mask_mode = mask_mode
        if mask_source not in {"uav", "legacy"}:
            raise ValueError("mask_source 必须为 uav 或 legacy")
        self.mask_source = mask_source
        self.vehicle_radius_meters = float(vehicle_radius_meters)
        if encode_gauge_state is not None:
            if encode_path_coordinates:
                raise ValueError(
                    "同时设置了 encode_path_coordinates 和旧参数 "
                    "encode_gauge_state。"
                )
            encode_path_coordinates = bool(encode_gauge_state)
        self.encode_path_coordinates = bool(encode_path_coordinates)
        self._path_representation = (
            BoundaryConstrainedPathRepresentation()
            if self.encode_path_coordinates
            else None
        )
        self._path_coordinate_cache = {}
        if self.vehicle_radius_meters < 0.0:
            raise ValueError("vehicle_radius_meters 不能为负数")
        if not 0.0 <= self.p_mask <= 1.0:
            raise ValueError("p_mask 必须在 [0,1]")
        self.env_index = {env_name: i for i, env_name in enumerate(env_list)}
        self.indexDict = []
        self.env_static_cache = {}  # 每个环境的静态缓存：地图通道/编码输入/stability等

        for env_name in env_list:
            env_path = osp.join(dataFolder, env_name)
            if not osp.isdir(env_path):
                raise FileNotFoundError(f"环境目录不存在: {env_path}")
            if not osp.isfile(osp.join(env_path, "map.p")):
                raise FileNotFoundError(f"环境缺少 map.p: {env_path}")

            # 保存真实路径编号，不再假定 path 文件必须从0开始连续编号。
            path_indices = []
            for filename in os.listdir(env_path):
                if not (filename.startswith("path_") and filename.endswith(".p")):
                    continue
                index_text = filename[len("path_"):-len(".p")]
                if index_text.isdigit():
                    path_indices.append(int(index_text))
            if not path_indices:
                raise FileNotFoundError(f"环境中没有 path_*.p: {env_path}")
            for path_index in sorted(path_indices):
                self.indexDict.append((self.env_index[env_name], path_index))
        
        if self.compute_stability_map:
            if self.use_precomputed_stability:
                stability_info = f"(启用stability map，优先加载预计算文件: {self.stability_map_filename})"
            else:
                stability_info = "(启用stability map在线计算)"
        else:
            stability_info = "(禁用stability map)"

        print(f"不平坦地面数据加载器初始化完成：{self.num_env}个环境，{len(self.indexDict)}个路径样本 {stability_info}")
    
    def __len__(self):
        return len(self.indexDict)

    def _make_partial_mask(
        self,
        idx,
        shape,
        trajectory_xy,
        mask_variant=0,
        bounds=MAP_BOUNDS,
        resolution=MAP_RESOLUTION,
        return_metadata=False,
    ):
        """生成可按轮次变化、同时可由索引和 variant 精确重建的 mask。"""
        variant = int(mask_variant)
        seed = self.mask_seed + int(idx) * 1_000_003 + variant * 15_485_863
        if self.mask_source == "uav":
            return generate_uav_observation_mask(
                shape,
                trajectory_xy,
                seed,
                bounds=bounds,
                resolution=resolution,
                p_mask=self.p_mask,
                vehicle_radius_m=self.vehicle_radius_meters,
                require_trajectory_clear=(self.mask_mode == "stage1_demo_valid"),
                return_metadata=return_metadata,
            )
        return generate_random_mask(
            shape,
            seed,
            trajectory_xy,
            p_mask=self.p_mask,
            require_trajectory_clear=(self.mask_mode == "stage1_demo_valid"),
            vehicle_radius_meters=self.vehicle_radius_meters,
            return_metadata=return_metadata,
        )

    def _get_env_static_data(self, env_name):
        """
        获取环境静态数据（带缓存），避免重复加载同一环境地图。
        """
        if env_name in self.env_static_cache:
            return self.env_static_cache[env_name]

        env_path = osp.join(self.dataFolder, env_name)
        map_file = osp.join(env_path, 'map.p')

        with open(map_file, 'rb') as f:
            map_data = pickle.load(f)

        map_tensor = map_data['tensor']
        if tuple(map_tensor.shape[:2]) != MAP_CONFIG.map_shape:
            raise ValueError(
                f"{map_file} 的地图尺寸为 {tuple(map_tensor.shape[:2])}，"
                f"配置要求 {MAP_CONFIG.map_shape}"
            )
        map_resolution = float(map_data.get("resolution", MAP_RESOLUTION))
        if not np.isclose(map_resolution, MAP_RESOLUTION, rtol=0.0, atol=1e-6):
            raise ValueError(
                f"{map_file} 的分辨率为 {map_resolution}，"
                f"配置要求 {MAP_RESOLUTION}"
            )
        map_bounds = tuple(map_data.get("bounds", MAP_BOUNDS))
        if len(map_bounds) != 4 or not np.allclose(
            map_bounds, MAP_BOUNDS, rtol=0.0, atol=1e-6
        ):
            raise ValueError(
                f"{map_file} 的边界为 {map_bounds}，配置要求 {MAP_BOUNDS}"
            )
        elevation = map_tensor[:, :, 0].astype(np.float32)
        normal_x = map_tensor[:, :, 1].astype(np.float32)
        normal_y = map_tensor[:, :, 2].astype(np.float32)
        normal_z = map_tensor[:, :, 3].astype(np.float32)

        encoded_input = np.concatenate((
            normal_x[:, :, None],
            normal_y[:, :, None],
            normal_z[:, :, None]
        ), axis=2).astype(np.float32)
        if not np.all(np.isfinite(encoded_input)):
            raise ValueError(f"{map_file} 的法向量含 NaN/Inf")
        map_mask = (
            normalize_mask(
                map_data["mask"],
                map_tensor.shape[:2],
                source=f"{map_file}['mask']",
            )
            if "mask" in map_data
            else None
        )

        env_static = {
            'elevation': elevation,
            'normal_x': normal_x,
            'normal_y': normal_y,
            'normal_z': normal_z,
            'encoded_input': encoded_input,
            'map_shape': map_tensor.shape[:2],
            'bounds': map_bounds,
            'resolution': map_resolution,
            'mask': map_mask,
        }

        if self.compute_stability_map and self.use_precomputed_stability:
            stability_file = osp.join(env_path, self.stability_map_filename)
            if osp.exists(stability_file):
                with np.load(stability_file) as stability_data:
                    compatible = stability_cache_is_compatible(
                        stability_data,
                        map_shape=map_tensor.shape[:2],
                        resolution=map_resolution,
                        yaw_bins=MAP_YAW_BINS,
                        yaw_weight=SAFETY_COST_CONFIG.yaw_esdf_weight,
                        source_map_sha256=stability_source_map_sha256(map_file),
                    )
                    if compatible:
                        env_static['yaw_stability'] = stability_data[
                            'yaw_stability'
                        ].astype(np.float32)
                        env_static['cost_map'] = stability_data[
                            'cost_map'
                        ].astype(np.float32)
                    elif not self.compute_stability_if_missing:
                        raise ValueError(
                            f"stability cache 语义或源地图不匹配: "
                            f"{stability_file}. 请用当前代码和 --overwrite "
                            "重建，或显式允许在线回退计算。"
                        )
            elif not self.compute_stability_if_missing:
                raise FileNotFoundError(
                    f"预计算stability文件不存在: {stability_file}. "
                    f"请先离线生成，或设置 compute_stability_if_missing=True 允许回退在线计算。"
                )

        self.env_static_cache[env_name] = env_static
        return env_static
    
    def get_item(
        self,
        idx,
        *,
        mask_variant=0,
        noise_seed=None,
        return_mask_metadata=False,
    ):
        """读取一个样本，并允许 Stage 2 显式指定本轮 mask/noise 版本。

        普通 DataLoader 仍通过 ``__getitem__`` 使用 variant=0。Stage 2 收集
        时传入轮次 variant 和固定 noise seed，replay 随后用这两个值重建
        完全相同的四通道条件，而不是读取当前轮的新 mask。
        """
        env_index, path_index = self.indexDict[idx]
        env_name = self.env_list[env_index]
        env_path = osp.join(self.dataFolder, env_name)
        path_file = osp.join(env_path, f'path_{path_index}.p')

        env_static = self._get_env_static_data(env_name)
        
        # 1. 加载地图数据
        elevation = env_static['elevation']
        normal_x = env_static['normal_x']
        normal_y = env_static['normal_y']
        normal_z = env_static['normal_z']
        
        # 2. 加载路径数据
        with open(path_file, 'rb') as f:
            path_data = pickle.load(f)
        # valid = path_data['valid']  # 是否有效路径
        # if not valid:
        #     return None  # 如果路径无效，返回None    
        cost = path_data.get('cost', 0.0)  # 路径成本（如果有）
        
        trajectory = path_data['path']  # [N+2, 3]
        path_coordinates = None
        path_fit_rmse = None
        path_fit_max_curvature = None
        path_fit_curvature_feasible = None
        path_fit_smoothness_weight = None
        path_fit_minimum_candidate_max_curvature = None
        path_fit_minimum_curvature_smoothness_weight = None
        if self._path_representation is not None:
            cached = self._path_coordinate_cache.get(int(idx))
            if cached is None:
                state, diagnostics = self._path_representation.fit_demonstration(
                    trajectory
                )
                cached = (
                    state.to(dtype=torch.float32),
                    float(diagnostics.control_reconstruction_rmse),
                    float(diagnostics.max_curvature),
                    bool(diagnostics.curvature_feasible),
                    float(diagnostics.smoothness_weight),
                    float(diagnostics.minimum_candidate_max_curvature),
                    float(
                        diagnostics.minimum_curvature_smoothness_weight
                    ),
                )
                self._path_coordinate_cache[int(idx)] = cached
            (
                path_coordinates,
                path_fit_rmse,
                path_fit_max_curvature,
                path_fit_curvature_feasible,
                path_fit_smoothness_weight,
                path_fit_minimum_candidate_max_curvature,
                path_fit_minimum_curvature_smoothness_weight,
            ) = cached
        
        # 3. 生成编码输入
        path = trajectory[:, :3]  # [N+2, 3]
        
        # 转换为相对位移
        relative_motion, start_pose = trajectory_to_relative_motion(trajectory)
        # relative_motion: (21, 3) [Δx, Δy, Δθ]
        # start_pose: (3,) [x_0, y_0, θ_0]
        
        # 提取中间步骤（去掉第一步和最后一步）
        middle_motion = relative_motion[1:-1]  # (19, 3)
        goal_pose = trajectory[-1]  # (3,)
        
        # # 对xy进行对换
        # path[:, [0, 1]] = path[:, [1, 0]]
        
        # 对角度进行标准化，确保在[-pi, pi]范围内
        path[:, 2] = (path[:, 2] + np.pi) % (2 * np.pi) - np.pi
        
        # start_pose = torch.tensor([path[0, 0], path[0, 1], np.cos(path[0, 2]), np.sin(path[0, 2])], dtype=torch.float32)  # [4] 起点位姿 [x, y, cos(yaw), sin(yaw)]
        # goal_pose = torch.tensor([path[-1, 0], path[-1, 1], np.cos(path[-1, 2]), np.sin(path[-1, 2])], dtype=torch.float32)  # [4] 终点位姿 [x, y, cos(yaw), sin(yaw)]
        # pose_input = torch.stack((start_pose, goal_pose), dim=0)  # [2, 4] 起点和终点位姿
        
        full_encoded_input = env_static['encoded_input']  # [H, W, 3]
        # 优先使用样本或地图直接提供的唯一 mask。它已经同时表达地图缺失和
        # 实体障碍；只有旧数据完全没有 mask 时，才生成不完整观测 mask。
        if "mask" in path_data:
            mask = normalize_mask(
                path_data["mask"],
                env_static["map_shape"],
                source=f"{path_file}['mask']",
            )
            mask_source = "path"
        elif env_static["mask"] is not None:
            mask = env_static["mask"].copy()
            mask_source = "map"
        elif self.partial_observation:
            mask, mask_metadata = self._make_partial_mask(
                idx,
                env_static['map_shape'],
                trajectory,
                mask_variant=mask_variant,
                bounds=env_static['bounds'],
                resolution=env_static['resolution'],
                return_metadata=True,
            )
            mask_source = "generated"
        else:
            mask = np.ones(
                env_static['map_shape'], dtype=np.float32
            )
            mask_source = "complete"
        dense_trajectory = dense_demo_bspline(trajectory)
        # 外部 mask 也统一转换为考虑车辆半径的配置空间；随机 mask 已经处理，
        # 重复腐蚀会过度扩大障碍，因此只处理直接提供的 mask。
        if "mask" in path_data or env_static["mask"] is not None:
            mask = erode_mask_for_vehicle(
                mask,
                vehicle_radius_meters=self.vehicle_radius_meters,
            )
        if mask_source != "generated":
            is_complete = bool(np.all(mask > 0.5))
            accepted_type = "complete" if is_complete else "external"
            blockage = trajectory_mask_blockage_metrics(
                dense_trajectory, mask
            )
            connected = mask_start_goal_connected(
                mask,
                trajectory[0, :2],
                trajectory[-1, :2],
            )
            mask_metadata = {
                "mask_active": not is_complete,
                "has_large_cutout": False,
                "has_small_obstacles": False,
                "has_route_local_obstacle": False,
                "has_global_entity_obstacles": False,
                "trajectory_obstacle_mode": "none",
                "semantic_mode": "external" if not is_complete else "complete",
                "attempts": 1,
                "sampling_attempts": 1,
                "resamples": 0,
                "masked_fraction": float(1.0 - mask.mean()),
                "endpoint_corridor_meters": float(
                    SAFETY_COST_CONFIG.endpoint_mask_corridor_meters
                ),
                "accepted_type": accepted_type,
                "mask_profile": self.mask_mode,
                "demo_blocked_fraction": blockage["blocked_fraction"],
                "demo_max_contiguous_blocked_fraction": blockage[
                    "max_contiguous_blocked_fraction"
                ],
                "informed_ellipse_fraction": float(
                    informed_demo_ellipse(
                        env_static["map_shape"],
                        dense_trajectory,
                    ).mean()
                ),
                "proposed_type_counts": {accepted_type: 1},
                "rejected_type_counts": {},
                "rejection_reason_counts": {},
                "raw_connectivity_counts": {
                    "connected": int(connected),
                    "disconnected": int(not connected),
                },
                "used_fallback": False,
            }
        mask_metadata["source"] = mask_source
        endpoints_valid = all(
            trajectory_is_allowed(point[None], mask)
            for point in dense_trajectory[[0, -1]]
        )
        if not endpoints_valid:
            raise ValueError(f"{path_file} 的起点或终点位于 mask=0")
        start_corridor, goal_corridor = endpoint_yaw_corridors(trajectory)
        if not trajectory_is_allowed(start_corridor, mask):
            raise ValueError(f"{path_file} 的起步 yaw 通道位于 mask=0")
        if not trajectory_is_allowed(goal_corridor, mask):
            raise ValueError(f"{path_file} 的到达 yaw 通道位于 mask=0")
        demo_mask_valid = trajectory_is_allowed(dense_trajectory, mask)
        if self.mask_mode == "stage1_demo_valid" and not demo_mask_valid:
            raise ValueError(
                f"{path_file} 的稠密 B 样条或车辆 footprint 经过 mask=0；"
                "训练数据不满足约束"
            )

        if noise_seed is not None:
            effective_noise_seed = int(noise_seed)
        elif self.dynamic_mask_noise:
            effective_noise_seed = None
        else:
            effective_noise_seed = (
                self.mask_seed
                + int(idx) * 1_000_003
                + int(mask_variant) * 15_485_863
                + MASK_NOISE_SEED_OFFSET
            )
        encoded_input = build_masked_normal_input(
            full_encoded_input,
            mask,
            effective_noise_seed,
        )

        if self.include_mask:
            # build_masked_normal_input 已经附加统一 mask 通道。
            pass
        else:
            encoded_input = encoded_input[:, :, :3]
        
        # encoded_input = get_encoder_input(
        #     np.abs(normal_z),        # 确保使用的法向量z分量为正值
        #     goal_state=path[-1, :],  # 终点位姿
        #     start_state=path[0, :],  # 起点位姿
        #     normal_x=normal_x, 
        #     normal_y=normal_y
        # )
        
        # encoded_input = get_encoder_input(
        #     normal_z, 
        #     goal_pos=path[-1, :2],  # 终点位置
        #     start_pos=path[0, :2],  # 起点位置
        #     normal_x=normal_x, 
        #     normal_y=normal_y
        # )
        
        # goal_index = geom2pix(path[-1, :2], res=map_data['resolution'], size=elevation.shape[:2])
        # start_index = geom2pix(path[0, :2], res=map_data['resolution'], size=elevation.shape[:2])
        
        # # 起点区域（使用start_index）
        # start_start_y = max(0, start_index[0] - receptive_field//2)
        # start_start_x = max(0, start_index[1] - receptive_field//2)
        # start_end_y = min(elevation.shape[0], start_index[0] + receptive_field//2)
        # start_end_x = min(elevation.shape[1], start_index[1] + receptive_field//2)
        
        # # 终点区域（使用goal_index）
        # goal_start_y = max(0, goal_index[0] - receptive_field//2)
        # goal_start_x = max(0, goal_index[1] - receptive_field//2)
        # goal_end_y = min(elevation.shape[0], goal_index[0] + receptive_field//2)
        # goal_end_x = min(elevation.shape[1], goal_index[1] + receptive_field//2)
        
        # # 构造编码输入
        # context_map = np.zeros(elevation.shape[:2])  # [H, W]
        # context_map[goal_start_y:goal_end_y, goal_start_x:goal_end_x] = 1.0  # 终点标记为1
        # context_map[start_start_y:start_end_y, start_start_x:start_end_x] = -1.0  # 起点标记为-1
        
        # # 构造θ=<nx,ny>->(cosθ, sinθ)的映射
        # angle_map = np.zeros((elevation.shape[0], elevation.shape[1], 2))  # [H, W, 2]
        # for i in range(elevation.shape[0]):
        #     for j in range(elevation.shape[1]):
        #         n_xy_norm = np.linalg.norm([normal_x[i, j], normal_y[i, j]])
                
        #         angle_map[i, j, 0] = normal_x[i, j] / n_xy_norm
        #         angle_map[i, j, 1] = normal_y[i, j] / n_xy_norm
        
        # # 拼接为4通道
        # encoded_input = np.concatenate((normal_z[:, :, None], context_map[:, :, None], angle_map[:, :, :2]), axis=2)  # [H, W, 4]
        
        # 4. 提取正/负样本锚点
        # 为每个轨迹点创建独立的正样本图层
        # trajectory = trajectory[1:-1, :]  # [N, 3]
        path_xy = trajectory[1:-1, :2]  # [N, 2] 去掉起点和终点
        num_trajectory_points = len(path_xy)
        
        # print(f"轨迹点数量: {num_trajectory_points}")
        # print(f"轨迹形状: {trajectory.shape}")
        
        # 为每个轨迹点找到对应的锚点（固定尺寸处理）
        positive_anchors_per_point = []  # 每个轨迹点对应的锚点列表
        for pos in path_xy:
            indices, = geom2pixMatpos(pos, res=res, size=env_static['map_shape'])
            current_anchors = list(set(indices))
            
            # 固定尺寸处理：确保每个轨迹点的正样本数量都是MAX_POSITIVE_ANCHORS
            if len(current_anchors) > MAX_POSITIVE_ANCHORS:
                # 如果超过最大值，随机采样到固定数量
                current_anchors = np.random.choice(current_anchors, size=MAX_POSITIVE_ANCHORS, replace=False).tolist()
            else:
                # 如果不足最大值，用-1填充到固定数量
                current_anchors.extend([-1] * (MAX_POSITIVE_ANCHORS - len(current_anchors)))
            
            positive_anchors_per_point.append(current_anchors)
        
        # 现在所有轨迹点的正样本数量都是MAX_POSITIVE_ANCHORS
        max_anchors = MAX_POSITIVE_ANCHORS
                       
        # 生成负样本：为每个轨迹点生成对应的负样本
        # 每个轨迹点的负样本应该是该点正样本的补集，而不是全体正样本的补集
        all_anchor_indices = set(range(len(hashTable)))
        negative_anchors_per_point = []
        # max_anchors = max(0, len(all_anchor_indices) - max_anchors)  # 负样本锚点数量
        
        for i in range(num_trajectory_points):
            # 获取当前轨迹点的正样本锚点
            current_positive_anchors = set([a for a in positive_anchors_per_point[i] if a != -1])
            
            # 当前轨迹点的负样本候选：全体锚点减去当前点的正样本
            available_negative_anchors = list(all_anchor_indices - current_positive_anchors)
            
            # 为当前轨迹点生成负样本
            if len(available_negative_anchors) >= max_anchors:
                neg_anchors = np.random.choice(available_negative_anchors, size=max_anchors, replace=False).tolist()
            else:
                neg_anchors = available_negative_anchors + [-1] * (max_anchors - len(available_negative_anchors))
            negative_anchors_per_point.append(neg_anchors)
        
        # 构建最终的锚点和标签张量
        all_positive = torch.tensor(positive_anchors_per_point)  # [num_trajectory_points, max_anchors]
        all_negative = torch.tensor(negative_anchors_per_point)  # [num_trajectory_points, max_anchors]
        
        anchor = torch.cat([all_positive, all_negative], dim=0)  # [2*num_trajectory_points, max_anchors]
        
        # 创建标签：前半部分为正样本(1)，后半部分为负样本(0)
        positive_labels = torch.ones_like(all_positive)
        negative_labels = torch.zeros_like(all_negative)
        labels = torch.cat([positive_labels, negative_labels], dim=0)  # [2*num_trajectory_points, max_anchors]
        
        # 将填充位置(-1)的标签设为-1，训练时忽略
        labels[anchor == -1] = -1
        
        # 5. 计算或加载 yaw_bins 倾覆状态（根据参数决定）
        yaw_stability = None
        cost_map = None
        if self.compute_stability_map:
            # 先尝试从缓存读取（包括预计算文件或历史在线计算结果）
            yaw_stability = env_static.get('yaw_stability', None)
            cost_map = env_static.get('cost_map', None)

            # 如果缓存没有，则在线计算一次并写回缓存
            if yaw_stability is None or cost_map is None:
                yaw_stability = compute_map_yaw_bins(
                    normal_x, normal_y, normal_z, yaw_bins=MAP_YAW_BINS
                )
                cost_map = generate_sdf_from_yaw_stability(
                    yaw_stability,
                    voxel_size_xy=MAP_RESOLUTION,
                    yaw_weight=SAFETY_COST_CONFIG.yaw_esdf_weight,
                )
                if isinstance(yaw_stability, torch.Tensor):
                    yaw_stability = yaw_stability.detach().cpu().numpy().astype(np.float32)
                if isinstance(cost_map, torch.Tensor):
                    cost_map = cost_map.detach().cpu().numpy().astype(np.float32)
                env_static['yaw_stability'] = yaw_stability
                env_static['cost_map'] = cost_map

        # 转换为PyTorch张量
        result = {
            'map': torch.as_tensor(encoded_input, dtype=torch.float).permute(2, 0, 1),  # 地图：(C, H, W) - 转换为channels-first格式
            'anchor': anchor,  # 锚点索引：(N, M)
            'labels': labels,  # 锚点标签：(N, M)
            'relative_motion': torch.from_numpy(middle_motion).float(),  # ✅ 改为相对位移
            'start_pose': torch.from_numpy(start_pose).float(),  # ✅ 添加起点
            'goal_pose': torch.from_numpy(goal_pose).float(),    # ✅ 添加终点
            'trajectory': torch.as_tensor(trajectory, dtype=torch.float),  # 轨迹点：[N, 3]
            'elevation': torch.as_tensor(elevation, dtype=torch.float),  # 高程图：[H, W]
            'cost': torch.tensor(cost, dtype=torch.float)  # 路径成本
        }
        if path_coordinates is not None:
            result["path_coordinates"] = path_coordinates.clone()
            # Compatibility key for checkpoints and analysis scripts created
            # before terminology standardization.
            result["trajectory_state_44d"] = path_coordinates.clone()
            result["trajectory_state_fit_rmse_m"] = torch.tensor(
                path_fit_rmse, dtype=torch.float32
            )
            result["trajectory_state_fit_max_curvature"] = torch.tensor(
                path_fit_max_curvature, dtype=torch.float32
            )
            result["trajectory_state_fit_curvature_feasible"] = torch.tensor(
                path_fit_curvature_feasible, dtype=torch.bool
            )
            result["trajectory_state_fit_smoothness_weight"] = torch.tensor(
                path_fit_smoothness_weight, dtype=torch.float32
            )
            result[
                "trajectory_state_fit_minimum_candidate_max_curvature"
            ] = torch.tensor(
                path_fit_minimum_candidate_max_curvature,
                dtype=torch.float32,
            )
            result[
                "trajectory_state_fit_minimum_curvature_smoothness_weight"
            ] = torch.tensor(
                path_fit_minimum_curvature_smoothness_weight,
                dtype=torch.float32,
            )
        result['mask'] = torch.as_tensor(
            mask, dtype=torch.float
        )
        # 仅作数据聚合诊断，不会作为模型输入。
        result['demo_mask_valid'] = torch.tensor(
            demo_mask_valid, dtype=torch.bool
        )
        result['dataset_index'] = torch.tensor(idx, dtype=torch.long)
        result['env_index'] = torch.tensor(env_index, dtype=torch.long)
        result['path_index'] = torch.tensor(path_index, dtype=torch.long)
        result['mask_variant'] = torch.tensor(
            int(mask_variant), dtype=torch.long
        )
        result['mask_noise_seed'] = torch.tensor(
            -1 if effective_noise_seed is None else effective_noise_seed,
            dtype=torch.long,
        )
        if return_mask_metadata:
            # 仅供逐轮特权约束蒸馏统计；不放入普通 DataLoader，避免嵌套字典参与
            # batch collate。
            result['mask_metadata'] = mask_metadata
        
        # 条件性地添加stability相关数据
        if self.compute_stability_map:
            result['normals'] = torch.stack([
                torch.as_tensor(normal_x, dtype=torch.float),
                torch.as_tensor(normal_y, dtype=torch.float),
                torch.as_tensor(normal_z, dtype=torch.float)
            ], dim=0)  # 法线：(3, H, W)
            result['yaw_stability'] = torch.as_tensor(yaw_stability, dtype=torch.float)  # yaw分箱倾覆状态：[H, W, 36]
            result['cost_map'] = torch.as_tensor(cost_map, dtype=torch.float)  # 成本图：[H, W, yaw_bins]

        return result

    def __getitem__(self, idx):
        """标准数据集接口；Stage 1 和普通评估固定使用默认 mask 版本。"""
        return self.get_item(idx)

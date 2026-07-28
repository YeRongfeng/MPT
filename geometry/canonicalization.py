"""Task-centric planar canonicalization.

Continuous points use

    p_c = R(-theta) (p - start_xy),
    theta = atan2(goal_y - start_y, goal_x - start_x).

Raster maps require resampling, so their equivariance is approximate.  The
returned validity mask makes samples outside the transformed source raster
explicit instead of silently treating padding as terrain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


Bounds = Tuple[float, float, float, float]


@dataclass(frozen=True)
class TaskFrame:
    """Planar task frame, with tensors broadcastable over leading dimensions."""

    origin: torch.Tensor
    theta: torch.Tensor


@dataclass(frozen=True)
class CanonicalMap:
    """Resampled map and the fractionally sampled source-domain validity mask."""

    values: torch.Tensor
    valid_mask: torch.Tensor


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap radians to [-pi, pi), preserving dtype and device."""

    return torch.remainder(angle + torch.pi, 2.0 * torch.pi) - torch.pi


def _require_last_dim(value: torch.Tensor, size: int, name: str) -> None:
    if value.ndim < 1 or value.shape[-1] != size:
        raise ValueError(f"{name} must end in {size}, got {tuple(value.shape)}")


def task_frame(start_pose: torch.Tensor, goal_pose: torch.Tensor) -> TaskFrame:
    """Build the deterministic frame from (...,3) poses."""

    _require_last_dim(start_pose, 3, "start_pose")
    _require_last_dim(goal_pose, 3, "goal_pose")
    if start_pose.shape != goal_pose.shape:
        raise ValueError(
            f"start_pose and goal_pose must match, got "
            f"{tuple(start_pose.shape)} and {tuple(goal_pose.shape)}"
        )
    delta = goal_pose[..., :2] - start_pose[..., :2]
    distance = torch.linalg.vector_norm(delta, dim=-1)
    if torch.any(distance <= torch.finfo(delta.dtype).eps):
        raise ValueError("Canonicalization is undefined for coincident endpoints")
    return TaskFrame(
        origin=start_pose[..., :2],
        theta=torch.atan2(delta[..., 1], delta[..., 0]),
    )


def _rotate_xy(xy: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    _require_last_dim(xy, 2, "xy")
    while angle.ndim < xy.ndim - 1:
        angle = angle.unsqueeze(-1)
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    x, y = xy.unbind(dim=-1)
    return torch.stack((cos * x - sin * y, sin * x + cos * y), dim=-1)


def canonicalize_pose(
    pose: torch.Tensor,
    start_pose: torch.Tensor,
    goal_pose: torch.Tensor,
) -> torch.Tensor:
    """Transform (...,3) poses into the task frame."""

    _require_last_dim(pose, 3, "pose")
    frame = task_frame(start_pose, goal_pose)
    xy = _rotate_xy(pose[..., :2] - frame.origin, -frame.theta)
    yaw = wrap_angle(pose[..., 2] - frame.theta)
    return torch.cat((xy, yaw.unsqueeze(-1)), dim=-1)


def decanonicalize_pose(
    pose: torch.Tensor,
    start_pose: torch.Tensor,
    goal_pose: torch.Tensor,
) -> torch.Tensor:
    """Transform (...,3) task-frame poses back to the global frame."""

    _require_last_dim(pose, 3, "pose")
    frame = task_frame(start_pose, goal_pose)
    xy = _rotate_xy(pose[..., :2], frame.theta) + frame.origin
    yaw = wrap_angle(pose[..., 2] + frame.theta)
    return torch.cat((xy, yaw.unsqueeze(-1)), dim=-1)


def canonicalize_trajectory(
    trajectory: torch.Tensor,
    start_pose: torch.Tensor,
    goal_pose: torch.Tensor,
) -> torch.Tensor:
    """Canonicalize (...,N,2|3) trajectory points."""

    if trajectory.shape[-1] not in (2, 3):
        raise ValueError(
            f"trajectory must end in 2 or 3, got {tuple(trajectory.shape)}"
        )
    frame = task_frame(start_pose, goal_pose)
    origin = frame.origin.unsqueeze(-2)
    theta = frame.theta.unsqueeze(-1)
    xy = _rotate_xy(trajectory[..., :2] - origin, -theta)
    if trajectory.shape[-1] == 2:
        return xy
    yaw = wrap_angle(trajectory[..., 2] - theta)
    return torch.cat((xy, yaw.unsqueeze(-1)), dim=-1)


def decanonicalize_trajectory(
    trajectory: torch.Tensor,
    start_pose: torch.Tensor,
    goal_pose: torch.Tensor,
) -> torch.Tensor:
    """Transform (...,N,2|3) trajectory points back to global coordinates."""

    if trajectory.shape[-1] not in (2, 3):
        raise ValueError(
            f"trajectory must end in 2 or 3, got {tuple(trajectory.shape)}"
        )
    frame = task_frame(start_pose, goal_pose)
    theta = frame.theta.unsqueeze(-1)
    xy = _rotate_xy(trajectory[..., :2], theta) + frame.origin.unsqueeze(-2)
    if trajectory.shape[-1] == 2:
        return xy
    yaw = wrap_angle(trajectory[..., 2] + theta)
    return torch.cat((xy, yaw.unsqueeze(-1)), dim=-1)


def canonicalize_residual(
    residual: torch.Tensor,
    start_pose: torch.Tensor,
    goal_pose: torch.Tensor,
) -> torch.Tensor:
    """Rotate (...,E,2) edge residual vectors; translation has no effect."""

    _require_last_dim(residual, 2, "residual")
    frame = task_frame(start_pose, goal_pose)
    return _rotate_xy(residual, -frame.theta.unsqueeze(-1))


def decanonicalize_residual(
    residual: torch.Tensor,
    start_pose: torch.Tensor,
    goal_pose: torch.Tensor,
) -> torch.Tensor:
    """Rotate (...,E,2) edge residual vectors back to the global frame."""

    _require_last_dim(residual, 2, "residual")
    frame = task_frame(start_pose, goal_pose)
    return _rotate_xy(residual, frame.theta.unsqueeze(-1))


def transform_pose_se2(
    pose: torch.Tensor,
    angle: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    """Apply ``p' = R(angle)p + translation`` and ``yaw'=yaw+angle``."""

    _require_last_dim(pose, 3, "pose")
    _require_last_dim(translation, 2, "translation")
    xy = _rotate_xy(pose[..., :2], angle) + translation
    return torch.cat(
        (xy, wrap_angle(pose[..., 2] + angle).unsqueeze(-1)), dim=-1
    )


def transform_trajectory_se2(
    trajectory: torch.Tensor,
    angle: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    """Apply an SE(2) transform to ``(...,N,2|3)`` trajectories."""

    if trajectory.shape[-1] not in (2, 3):
        raise ValueError(
            f"trajectory must end in 2 or 3, got {tuple(trajectory.shape)}"
        )
    while translation.ndim < trajectory.ndim:
        translation = translation.unsqueeze(-2)
    xy = _rotate_xy(trajectory[..., :2], angle) + translation
    if trajectory.shape[-1] == 2:
        return xy
    while angle.ndim < trajectory.ndim - 1:
        angle = angle.unsqueeze(-1)
    yaw = wrap_angle(trajectory[..., 2] + angle)
    return torch.cat((xy, yaw.unsqueeze(-1)), dim=-1)


def inverse_transform_trajectory_se2(
    trajectory: torch.Tensor,
    angle: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    """Apply the inverse of ``p' = R(angle)p + translation``."""

    if trajectory.shape[-1] not in (2, 3):
        raise ValueError(
            f"trajectory must end in 2 or 3, got {tuple(trajectory.shape)}"
        )
    while translation.ndim < trajectory.ndim:
        translation = translation.unsqueeze(-2)
    xy = _rotate_xy(trajectory[..., :2] - translation, -angle)
    if trajectory.shape[-1] == 2:
        return xy
    while angle.ndim < trajectory.ndim - 1:
        angle = angle.unsqueeze(-1)
    yaw = wrap_angle(trajectory[..., 2] - angle)
    return torch.cat((xy, yaw.unsqueeze(-1)), dim=-1)


def transform_residual_se2(
    residual: torch.Tensor, angle: torch.Tensor
) -> torch.Tensor:
    """Rotate residual vectors under an active SE(2) transform."""

    _require_last_dim(residual, 2, "residual")
    return _rotate_xy(residual, angle)


def _pixel_centers(
    bounds: Bounds,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xmin, xmax, ymin, ymax = bounds
    if not xmin < xmax or not ymin < ymax:
        raise ValueError(f"Invalid bounds: {bounds}")
    xs = torch.linspace(xmin, xmax, width, device=device, dtype=dtype)
    ys = torch.linspace(ymin, ymax, height, device=device, dtype=dtype)
    return xs, ys


def canonicalize_map(
    map_tensor: torch.Tensor,
    start_pose: torch.Tensor,
    goal_pose: torch.Tensor,
    *,
    source_bounds: Bounds,
    output_bounds: Optional[Bounds] = None,
    output_size: Optional[Sequence[int]] = None,
    normal_channels: Optional[Tuple[int, int, int]] = (0, 1, 2),
    mode: str = "bilinear",
    enabled: bool = True,
) -> CanonicalMap:
    """Resample a map into the task frame and rotate its horizontal normals.

    Args:
        map_tensor: ``(C,H,W)`` or ``(B,C,H,W)``.
        start_pose, goal_pose: ``(3,)`` or ``(B,3)`` global poses.
        source_bounds: physical coordinates of first/last pixel centers.
        output_bounds: canonical coordinates of first/last output centers.
        output_size: output ``(H,W)``; defaults to input size.
        normal_channels: indices ``(nx,ny,nz)``.  Set to ``None`` for scalar
            fields such as elevation or a validity grid.  ``nz`` is sampled
            but not rotated.
        enabled: when false, return the exact input tensor without resampling.

    Elevation is a scalar under planar SE(2): it must be spatially resampled
    but its value is neither rotated nor offset.  Put it in a non-normal
    channel, or call this function with ``normal_channels=None``.
    """

    if map_tensor.ndim not in (3, 4):
        raise ValueError(
            f"map_tensor must be (C,H,W) or (B,C,H,W), got {tuple(map_tensor.shape)}"
        )
    unbatched = map_tensor.ndim == 3
    values = map_tensor.unsqueeze(0) if unbatched else map_tensor
    batch, channels, in_h, in_w = values.shape
    if not enabled:
        mask_shape = (1, in_h, in_w) if unbatched else (batch, 1, in_h, in_w)
        return CanonicalMap(
            values=map_tensor,
            valid_mask=torch.ones(
                mask_shape, device=map_tensor.device, dtype=map_tensor.dtype
            ),
        )

    if not values.is_floating_point():
        raise TypeError("map_tensor must have a floating dtype")
    if output_bounds is None:
        output_bounds = source_bounds
    out_h, out_w = (
        (in_h, in_w)
        if output_size is None
        else (int(output_size[0]), int(output_size[1]))
    )
    if out_h <= 1 or out_w <= 1:
        raise ValueError("output_size dimensions must be greater than one")

    start_b = start_pose.unsqueeze(0) if start_pose.ndim == 1 else start_pose
    goal_b = goal_pose.unsqueeze(0) if goal_pose.ndim == 1 else goal_pose
    if start_b.shape != (batch, 3) or goal_b.shape != (batch, 3):
        raise ValueError(
            f"Expected batched poses {(batch, 3)}, got "
            f"{tuple(start_b.shape)} and {tuple(goal_b.shape)}"
        )
    frame = task_frame(start_b, goal_b)
    xs, ys = _pixel_centers(
        output_bounds,
        out_h,
        out_w,
        device=values.device,
        dtype=values.dtype,
    )
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    canonical_xy = torch.stack((xx, yy), dim=-1).expand(batch, -1, -1, -1)
    global_xy = _rotate_xy(
        canonical_xy, frame.theta[:, None, None]
    ) + frame.origin[:, None, None, :]

    xmin, xmax, ymin, ymax = source_bounds
    grid_x = 2.0 * (global_xy[..., 0] - xmin) / (xmax - xmin) - 1.0
    grid_y = 2.0 * (global_xy[..., 1] - ymin) / (ymax - ymin) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)
    sampled = F.grid_sample(
        values,
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=True,
    )
    validity_source = torch.ones(
        (batch, 1, in_h, in_w), device=values.device, dtype=values.dtype
    )
    valid = F.grid_sample(
        validity_source,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).clamp_(0.0, 1.0)

    if normal_channels is not None:
        nx_index, ny_index, nz_index = normal_channels
        if len({nx_index, ny_index, nz_index}) != 3:
            raise ValueError("normal_channels must contain three unique indices")
        if min(normal_channels) < 0 or max(normal_channels) >= channels:
            raise ValueError(
                f"normal_channels {normal_channels} invalid for {channels} channels"
            )
        nx = sampled[:, nx_index].clone()
        ny = sampled[:, ny_index].clone()
        cos = torch.cos(frame.theta)[:, None, None]
        sin = torch.sin(frame.theta)[:, None, None]
        sampled[:, nx_index] = cos * nx + sin * ny
        sampled[:, ny_index] = -sin * nx + cos * ny
        # sampled[:, nz_index] intentionally remains unchanged.

    if unbatched:
        return CanonicalMap(values=sampled[0], valid_mask=valid[0])
    return CanonicalMap(values=sampled, valid_mask=valid)


def transform_map_se2(
    map_tensor: torch.Tensor,
    angle: torch.Tensor,
    translation: torch.Tensor,
    *,
    source_bounds: Bounds,
    output_bounds: Optional[Bounds] = None,
    output_size: Optional[Sequence[int]] = None,
    normal_channels: Optional[Tuple[int, int, int]] = (0, 1, 2),
    mode: str = "bilinear",
) -> CanonicalMap:
    """Actively transform a raster map by SE(2).

    The output field is ``M'(p') = M(R(-angle)(p'-translation))`` and sampled
    horizontal normal vectors are rotated by ``R(angle)``.
    """

    unbatched = map_tensor.ndim == 3
    values = map_tensor.unsqueeze(0) if unbatched else map_tensor
    batch = values.shape[0]
    angle_b = angle.reshape(-1)
    if angle_b.numel() == 1:
        angle_b = angle_b.expand(batch)
    translation_b = (
        translation.unsqueeze(0) if translation.ndim == 1 else translation
    )
    if translation_b.shape[0] == 1 and batch > 1:
        translation_b = translation_b.expand(batch, -1)
    if angle_b.shape != (batch,) or translation_b.shape != (batch, 2):
        raise ValueError(
            f"Expected angle {(batch,)} and translation {(batch,2)}, got "
            f"{tuple(angle_b.shape)} and {tuple(translation_b.shape)}"
        )
    inverse_angle = -angle_b
    inverse_origin = -_rotate_xy(translation_b, inverse_angle)
    synthetic_start = torch.cat(
        (
            inverse_origin,
            torch.zeros(batch, 1, device=values.device, dtype=values.dtype),
        ),
        dim=1,
    )
    direction = torch.stack(
        (torch.cos(inverse_angle), torch.sin(inverse_angle)), dim=1
    )
    synthetic_goal = torch.cat(
        (
            inverse_origin + direction,
            torch.zeros(batch, 1, device=values.device, dtype=values.dtype),
        ),
        dim=1,
    )
    result = canonicalize_map(
        values,
        synthetic_start,
        synthetic_goal,
        source_bounds=source_bounds,
        output_bounds=output_bounds,
        output_size=output_size,
        normal_channels=normal_channels,
        mode=mode,
    )
    if unbatched:
        return CanonicalMap(
            values=result.values[0], valid_mask=result.valid_mask[0]
        )
    return result

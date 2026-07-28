"""Shared analytic output wrapper for all A/B/C/D variants."""

from __future__ import annotations

import torch

from dit.Models import PhysicalScaledEdgeResidualRepresentation
from geometry.canonicalization import decanonicalize_residual
from map_config import MAP_HALF_EXTENT


def decode_global_feasible(
    residual_model_frame: torch.Tensor,
    global_start_pose: torch.Tensor,
    global_goal_pose: torch.Tensor,
    *,
    model_coordinate_scale: float,
    canonical: bool,
    representation: PhysicalScaledEdgeResidualRepresentation | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return global residual/control points inside the original map square.

    The model residual is dimensionless at ``model_coordinate_scale``.  It is
    rotated to global axes when needed, converted to the legacy 10 m
    normalization, and passed through the unchanged radial projector.
    """

    if global_start_pose.shape != global_goal_pose.shape:
        raise ValueError("Global start/goal pose shapes must match")
    if global_start_pose.shape != (len(residual_model_frame), 3):
        raise ValueError(
            f"Expected global poses {(len(residual_model_frame), 3)}, got "
            f"{tuple(global_start_pose.shape)}"
        )
    if model_coordinate_scale <= 0.0:
        raise ValueError("model_coordinate_scale must be positive")
    representation = representation or (
        PhysicalScaledEdgeResidualRepresentation().to(
            device=residual_model_frame.device,
            dtype=residual_model_frame.dtype,
        )
    )
    residual_global = (
        decanonicalize_residual(
            residual_model_frame, global_start_pose, global_goal_pose
        )
        if canonical
        else residual_model_frame
    )
    residual_global = residual_global * (
        float(model_coordinate_scale) / MAP_HALF_EXTENT
    )
    start_n = global_start_pose[:, :2] / MAP_HALF_EXTENT
    goal_n = global_goal_pose[:, :2] / MAP_HALF_EXTENT
    residual_global = representation.radial_project_residual(
        residual_global, start_n, goal_n
    )
    control_global = (
        representation.decode(residual_global, start_n, goal_n)
        * MAP_HALF_EXTENT
    )
    return residual_global, control_global

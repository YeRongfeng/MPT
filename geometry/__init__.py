"""Geometry utilities for task-conditioned trajectory generation."""

from .canonicalization import (
    CanonicalMap,
    TaskFrame,
    canonicalize_map,
    canonicalize_pose,
    canonicalize_residual,
    canonicalize_trajectory,
    decanonicalize_pose,
    decanonicalize_residual,
    decanonicalize_trajectory,
    inverse_transform_trajectory_se2,
    task_frame,
    transform_map_se2,
    transform_pose_se2,
    transform_residual_se2,
    transform_trajectory_se2,
    wrap_angle,
)

__all__ = [
    "CanonicalMap",
    "TaskFrame",
    "canonicalize_map",
    "canonicalize_pose",
    "canonicalize_residual",
    "canonicalize_trajectory",
    "decanonicalize_pose",
    "decanonicalize_residual",
    "decanonicalize_trajectory",
    "inverse_transform_trajectory_se2",
    "task_frame",
    "transform_map_se2",
    "transform_pose_se2",
    "transform_residual_se2",
    "transform_trajectory_se2",
    "wrap_angle",
]

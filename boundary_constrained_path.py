"""First-order boundary-constrained B-spline path representation.

It provides:

* standard-normal 22x2 (44-D) free path coordinates;
* affine decoding to 26 clamped cubic B-spline control points;
* deterministic canonical geometric path parameterization;
* analytic position, first derivative, second derivative, yaw, and curvature.

The source distribution is a First-Order Boundary-Projected Gaussian Path
Prior.  It is not described as a Brownian bridge or a conditional Gaussian
bridge in the production method.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.interpolate import BSpline
from torch import nn

from bspline_utils import create_bspline_basis_matrix
from map_config import SAFETY_COST_CONFIG


# Keep the value stable: this is an on-disk compatibility token, not the
# publication-facing method name.  Renaming must not invalidate existing
# checkpoints that use the same mathematical representation.
PATH_REPRESENTATION_SEMANTIC_VERSION = (
    "gauge_fixed_first_order_projected_bridge_44d_v1"
)
DEMONSTRATION_FIT_SEMANTIC_VERSION = (
    "gauge_fixed_quintic_demo_fit_exact_yaw_smooth_candidates_"
    "kappa2p1_audit1001_float32_v4"
)


@dataclass(frozen=True)
class PathCoordinateFitDiagnostics:
    chord_length: float
    path_length: float
    endpoint_slope: float
    control_reconstruction_rmse: float
    max_curvature: float
    curvature_feasible: bool
    smoothness_weight: float
    minimum_candidate_max_curvature: float
    minimum_curvature_smoothness_weight: float


def _polyline_sample(
    points: np.ndarray,
    parameters: np.ndarray,
) -> tuple[np.ndarray, float]:
    points = np.asarray(points, dtype=np.float64)
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment)])
    length = float(cumulative[-1])
    if length <= 1e-10:
        raise ValueError("Demonstration path length is degenerate")
    normalized = cumulative / length
    parameters = np.asarray(parameters, dtype=np.float64)
    sampled = np.stack(
        [
            np.interp(parameters, normalized, points[:, axis])
            for axis in range(2)
        ],
        axis=1,
    )
    return sampled, length


def canonical_parameter_map(
    t: np.ndarray,
    endpoint_slope: float,
) -> np.ndarray:
    """Fixed monotone quintic map for canonical path parameterization."""
    t = np.asarray(t, dtype=np.float64)
    slope = float(endpoint_slope)
    if not 0.0 < slope <= 1.0 + 1e-9:
        raise ValueError(f"endpoint_slope must lie in (0,1], got {slope}")
    slope = min(slope, 1.0)
    smoothstep = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
    result = slope * t + (1.0 - slope) * smoothstep
    if np.any(np.diff(result) <= 0.0):
        raise AssertionError(
            "Canonical quintic parameter map must be strictly monotone"
        )
    return result


def _basis_derivatives(
    samples: int,
    controls: int = 26,
    degree: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    basis, knots = create_bspline_basis_matrix(samples, controls, degree)
    parameter = np.linspace(0.0, 1.0, samples)
    first = np.empty_like(basis)
    second = np.empty_like(basis)
    for index in range(controls):
        coefficient = np.zeros(controls, dtype=np.float64)
        coefficient[index] = 1.0
        spline = BSpline(knots, coefficient, degree)
        first[:, index] = spline.derivative(1)(parameter)
        second[:, index] = spline.derivative(2)(parameter)
    return basis, first, second, knots


class FirstOrderBoundaryConstrainedBSpline(nn.Module):
    """Affine 44-D free-coordinate representation of boundary-feasible paths."""

    num_control_points = 26
    degree = 3
    state_points = 22
    state_dimension = 44
    semantic_version = PATH_REPRESENTATION_SEMANTIC_VERSION
    demonstration_fit_curvature_audit_points = 1001
    # These candidates are part of the demonstration-target construction, not
    # trainable hyperparameters.  The unregularized fit is retained whenever it
    # is already curvature-feasible.  Only a failing fit enters this frozen
    # coarse-to-fine regularization ladder.
    demonstration_fit_smoothness_weights = (
        0.0,
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
        3e-2,
        1e-1,
        3e-1,
        1.0,
        3.0,
        10.0,
        30.0,
        100.0,
    )

    def __init__(
        self,
        *,
        dense_points: int = 200,
        fit_points: int = 50,
    ):
        super().__init__()
        if dense_points < 4:
            raise ValueError("dense_points must be at least 4")
        if fit_points < self.num_control_points:
            raise ValueError("fit_points must be at least 26")
        basis, first, second, knots = _basis_derivatives(
            dense_points,
            self.num_control_points,
            self.degree,
        )
        (
            _,
            fit_audit_first,
            fit_audit_second,
            _,
        ) = _basis_derivatives(
            self.demonstration_fit_curvature_audit_points,
            self.num_control_points,
            self.degree,
        )
        fit_basis, fit_knots = create_bspline_basis_matrix(
            fit_points,
            self.num_control_points,
            self.degree,
        )
        if not np.allclose(knots, fit_knots):
            raise AssertionError("Inconsistent knot vectors")
        beta = float(
            self.degree / (knots[self.degree + 1] - knots[self.degree])
        )
        if not math.isclose(beta, 69.0):
            raise AssertionError(f"Unexpected endpoint derivative beta={beta}")

        weights = np.ones(dense_points, dtype=np.float64)
        weights[[0, -1]] = 0.5
        weights /= dense_points - 1
        gram = basis.T @ (weights[:, None] * basis)
        fixed_indices = np.array([0, 1, 24, 25], dtype=np.int64)
        free_indices = np.arange(2, 24, dtype=np.int64)
        free_from_fixed = np.linalg.solve(
            gram[np.ix_(free_indices, free_indices)],
            gram[np.ix_(free_indices, fixed_indices)],
        )

        edge_count = self.num_control_points - 1
        indices = np.arange(self.num_control_points, dtype=np.float64)
        position_prior_covariance = (
            np.minimum(indices[:, None], indices[None, :])
            - indices[:, None] * indices[None, :] / edge_count
        ) / edge_count**2
        projection = np.zeros(
            (self.num_control_points, self.num_control_points),
            dtype=np.float64,
        )
        projection[np.ix_(free_indices, free_indices)] = np.eye(
            len(free_indices)
        )
        projection[np.ix_(free_indices, fixed_indices)] = free_from_fixed
        projected_covariance = (
            projection @ position_prior_covariance @ projection.T
        )
        free_covariance = projected_covariance[
            np.ix_(free_indices, free_indices)
        ]
        free_covariance = 0.5 * (
            free_covariance + free_covariance.T
        )
        source_factor = np.linalg.cholesky(free_covariance)

        self.dense_points = int(dense_points)
        self.fit_points = int(fit_points)
        self.beta = beta
        self.register_buffer(
            "basis", torch.as_tensor(basis, dtype=torch.float64)
        )
        self.register_buffer(
            "first_basis", torch.as_tensor(first, dtype=torch.float64)
        )
        self.register_buffer(
            "second_basis", torch.as_tensor(second, dtype=torch.float64)
        )
        # Target fitting uses a denser, fixed grid than deployed trajectory
        # sampling so narrow curvature peaks cannot slip through the fit
        # admission check.  These deterministic matrices are not checkpoint
        # state; keeping them non-persistent preserves decoder compatibility.
        self.register_buffer(
            "fit_audit_first_basis",
            torch.as_tensor(fit_audit_first, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "fit_audit_second_basis",
            torch.as_tensor(fit_audit_second, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "fit_basis", torch.as_tensor(fit_basis, dtype=torch.float64)
        )
        self.register_buffer(
            "free_from_fixed",
            torch.as_tensor(free_from_fixed, dtype=torch.float64),
        )
        self.register_buffer(
            "source_factor",
            torch.as_tensor(source_factor, dtype=torch.float64),
        )
        self.register_buffer(
            "source_factor_inverse",
            torch.as_tensor(
                np.linalg.inv(source_factor), dtype=torch.float64
            ),
        )
        self.register_buffer(
            "canonical_chord",
            torch.stack(
                [
                    torch.linspace(0.0, 1.0, self.num_control_points),
                    torch.zeros(self.num_control_points),
                ],
                dim=1,
            ).to(torch.float64),
        )

    @staticmethod
    def _validate_pose(pose: torch.Tensor, name: str) -> torch.Tensor:
        pose = torch.as_tensor(pose)
        if pose.ndim == 1:
            pose = pose.unsqueeze(0)
        if pose.ndim != 2 or pose.shape[1] < 3:
            raise ValueError(f"{name} must have shape (B,>=3)")
        return pose

    def _condition(
        self,
        start_pose: torch.Tensor,
        goal_pose: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        start_pose = self._validate_pose(start_pose, "start_pose").to(
            device=device, dtype=dtype
        )
        goal_pose = self._validate_pose(goal_pose, "goal_pose").to(
            device=device, dtype=dtype
        )
        if start_pose.shape[0] != goal_pose.shape[0]:
            raise ValueError("start_pose and goal_pose batch mismatch")
        chord = goal_pose[:, :2] - start_pose[:, :2]
        distance = torch.linalg.vector_norm(chord, dim=1)
        if torch.any(distance <= 1e-8):
            raise ValueError("Start and goal positions must differ")
        theta = torch.atan2(chord[:, 1], chord[:, 0])
        start_relative = start_pose[:, 2] - theta
        goal_relative = goal_pose[:, 2] - theta
        start_direction = torch.stack(
            [torch.cos(start_relative), torch.sin(start_relative)], dim=1
        )
        goal_direction = torch.stack(
            [torch.cos(goal_relative), torch.sin(goal_relative)], dim=1
        )
        return (
            start_pose,
            goal_pose,
            distance,
            theta,
            start_direction,
            goal_direction,
        )

    def fixed_canonical_controls(
        self,
        start_direction: torch.Tensor,
        goal_direction: torch.Tensor,
    ) -> torch.Tensor:
        batch = start_direction.shape[0]
        fixed = torch.empty(
            batch,
            4,
            2,
            dtype=start_direction.dtype,
            device=start_direction.device,
        )
        fixed[:, 0] = 0.0
        fixed[:, 1] = start_direction / self.beta
        fixed[:, 2] = (
            torch.tensor(
                [1.0, 0.0],
                dtype=start_direction.dtype,
                device=start_direction.device,
            )
            - goal_direction / self.beta
        )
        fixed[:, 3] = torch.tensor(
            [1.0, 0.0],
            dtype=start_direction.dtype,
            device=start_direction.device,
        )
        return fixed

    def canonical_source_mean(
        self,
        start_direction: torch.Tensor,
        goal_direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fixed = self.fixed_canonical_controls(
            start_direction, goal_direction
        )
        chord = self.canonical_chord.to(
            device=fixed.device, dtype=fixed.dtype
        )
        chord_fixed = chord[[0, 1, 24, 25]]
        chord_free = chord[2:24]
        fixed_delta = fixed - chord_fixed.unsqueeze(0)
        free_mean = chord_free.unsqueeze(0) - torch.einsum(
            "ij,bjk->bik",
            self.free_from_fixed.to(
                device=fixed.device, dtype=fixed.dtype
            ),
            fixed_delta,
        )
        return free_mean, fixed

    def canonical_control_points(
        self,
        state: torch.Tensor,
        start_direction: torch.Tensor,
        goal_direction: torch.Tensor,
    ) -> torch.Tensor:
        state = torch.as_tensor(state)
        if state.ndim == 2:
            state = state.unsqueeze(0)
        if state.ndim != 3 or state.shape[1:] != (22, 2):
            raise ValueError("state must have shape (B,22,2)")
        if start_direction.shape != (state.shape[0], 2):
            raise ValueError("start_direction must have shape (B,2)")
        if goal_direction.shape != (state.shape[0], 2):
            raise ValueError("goal_direction must have shape (B,2)")
        mean, fixed = self.canonical_source_mean(
            start_direction, goal_direction
        )
        factor = self.source_factor.to(
            device=state.device, dtype=state.dtype
        )
        free = mean + torch.einsum("ij,bjk->bik", factor, state)
        controls = torch.empty(
            state.shape[0],
            self.num_control_points,
            2,
            device=state.device,
            dtype=state.dtype,
        )
        controls[:, 0] = fixed[:, 0]
        controls[:, 1] = fixed[:, 1]
        controls[:, 2:24] = free
        controls[:, 24] = fixed[:, 2]
        controls[:, 25] = fixed[:, 3]
        return controls

    @staticmethod
    def _to_global(
        canonical: torch.Tensor,
        start_xy: torch.Tensor,
        distance: torch.Tensor,
        theta: torch.Tensor,
        *,
        translate: bool,
    ) -> torch.Tensor:
        cosine = torch.cos(theta)
        sine = torch.sin(theta)
        x = canonical[..., 0]
        y = canonical[..., 1]
        global_value = torch.stack(
            [
                distance[:, None] * (cosine[:, None] * x - sine[:, None] * y),
                distance[:, None] * (sine[:, None] * x + cosine[:, None] * y),
            ],
            dim=-1,
        )
        if translate:
            global_value = global_value + start_xy[:, None]
        return global_value

    def decode_control_points(
        self,
        state: torch.Tensor,
        start_pose: torch.Tensor,
        goal_pose: torch.Tensor,
    ) -> torch.Tensor:
        state = torch.as_tensor(state)
        if state.ndim == 2:
            state = state.unsqueeze(0)
        condition = self._condition(
            start_pose,
            goal_pose,
            dtype=state.dtype,
            device=state.device,
        )
        start, _, distance, theta, start_direction, goal_direction = condition
        canonical = self.canonical_control_points(
            state, start_direction, goal_direction
        )
        return self._to_global(
            canonical,
            start[:, :2],
            distance,
            theta,
            translate=True,
        )

    def evaluate(
        self,
        state: torch.Tensor,
        start_pose: torch.Tensor,
        goal_pose: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        control = self.decode_control_points(state, start_pose, goal_pose)
        basis = self.basis.to(device=control.device, dtype=control.dtype)
        first_basis = self.first_basis.to(
            device=control.device, dtype=control.dtype
        )
        second_basis = self.second_basis.to(
            device=control.device, dtype=control.dtype
        )
        position = torch.einsum("nc,bcd->bnd", basis, control)
        first = torch.einsum("nc,bcd->bnd", first_basis, control)
        second = torch.einsum("nc,bcd->bnd", second_basis, control)
        speed = torch.linalg.vector_norm(first, dim=-1)
        yaw = torch.atan2(first[..., 1], first[..., 0])
        numerator = torch.abs(
            first[..., 0] * second[..., 1]
            - first[..., 1] * second[..., 0]
        )
        curvature = numerator / speed.clamp_min(1e-10).pow(3)
        return {
            "control_points": control,
            "position": position,
            "first_derivative": first,
            "second_derivative": second,
            "speed": speed,
            "yaw": yaw,
            "curvature": curvature,
        }

    def endpoint_yaw_errors(
        self,
        state: torch.Tensor,
        start_pose: torch.Tensor,
        goal_pose: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return wrapped analytic endpoint-yaw errors for test/assertions."""
        state = torch.as_tensor(state)
        start_pose = self._validate_pose(start_pose, "start_pose").to(
            device=state.device, dtype=state.dtype
        )
        goal_pose = self._validate_pose(goal_pose, "goal_pose").to(
            device=state.device, dtype=state.dtype
        )
        geometry = self.evaluate(state, start_pose, goal_pose)
        start_error = torch.atan2(
            torch.sin(geometry["yaw"][:, 0] - start_pose[:, 2]),
            torch.cos(geometry["yaw"][:, 0] - start_pose[:, 2]),
        ).abs()
        goal_error = torch.atan2(
            torch.sin(geometry["yaw"][:, -1] - goal_pose[:, 2]),
            torch.cos(geometry["yaw"][:, -1] - goal_pose[:, 2]),
        ).abs()
        return start_error, goal_error

    def audit_curvature(
        self,
        state: torch.Tensor,
        start_pose: torch.Tensor,
        goal_pose: torch.Tensor,
    ) -> torch.Tensor:
        """Return analytic curvature on the fixed 1001-point fit audit grid."""
        control = self.decode_control_points(state, start_pose, goal_pose)
        first = torch.einsum(
            "nc,bcd->bnd",
            self.fit_audit_first_basis.to(
                device=control.device, dtype=control.dtype
            ),
            control,
        )
        second = torch.einsum(
            "nc,bcd->bnd",
            self.fit_audit_second_basis.to(
                device=control.device, dtype=control.dtype
            ),
            control,
        )
        speed = torch.linalg.vector_norm(first, dim=-1)
        cross = torch.abs(
            first[..., 0] * second[..., 1]
            - first[..., 1] * second[..., 0]
        )
        return cross / speed.clamp_min(1e-10).pow(3)

    def assert_endpoint_yaw(
        self,
        state: torch.Tensor,
        start_pose: torch.Tensor,
        goal_pose: torch.Tensor,
        *,
        tolerance: float = 5e-4,
    ) -> None:
        """Assert the representation invariant using analytic derivatives."""
        start_error, goal_error = self.endpoint_yaw_errors(
            state, start_pose, goal_pose
        )
        maximum = torch.maximum(start_error, goal_error).amax()
        if not bool(torch.isfinite(maximum)) or float(maximum) > tolerance:
            raise AssertionError(
                "First-order boundary path invariant failed: "
                f"max error={float(maximum):.6g} rad, tolerance={tolerance}"
            )

    def encode_canonical_controls(
        self,
        canonical_controls: torch.Tensor,
        start_direction: torch.Tensor,
        goal_direction: torch.Tensor,
    ) -> torch.Tensor:
        canonical_controls = torch.as_tensor(canonical_controls)
        if canonical_controls.ndim == 2:
            canonical_controls = canonical_controls.unsqueeze(0)
        mean, fixed = self.canonical_source_mean(
            start_direction, goal_direction
        )
        boundary = canonical_controls[:, [0, 1, 24, 25]]
        tolerance = (
            2e-5 if canonical_controls.dtype == torch.float32 else 1e-10
        )
        if not torch.allclose(boundary, fixed, rtol=0.0, atol=tolerance):
            error = float((boundary - fixed).abs().max())
            raise ValueError(
                f"Controls violate first-order boundary, max error={error}"
            )
        inverse = self.source_factor_inverse.to(
            device=canonical_controls.device,
            dtype=canonical_controls.dtype,
        )
        return torch.einsum(
            "ij,bjk->bik",
            inverse,
            canonical_controls[:, 2:24] - mean,
        )

    def fit_demonstration(
        self,
        trajectory: np.ndarray,
    ) -> tuple[torch.Tensor, PathCoordinateFitDiagnostics]:
        """Encode one demonstration into curvature-audited 44-D coordinates.

        The four boundary controls remain fixed, so every candidate preserves
        the exact endpoint positions and yaws of this representation.  A plain
        position-only least-squares fit can nevertheless create large second
        derivatives while retaining millimetre-scale position RMSE.  We keep
        that fit when it is feasible; otherwise we deterministically try a
        frozen ladder of second-difference regularizers and choose the
        feasible candidate with the smallest position RMSE.  If the ladder
        finds no feasible candidate, the original position-only fit is
        returned and explicitly marked infeasible in the diagnostics.  The
        caller can therefore report the unresolved sample without silently
        deleting it or inventing a projected target.
        """
        trajectory = np.asarray(trajectory, dtype=np.float64)
        if trajectory.ndim != 2 or trajectory.shape[0] < 3:
            raise ValueError("trajectory must have shape (N,>=2), N>=3")
        points = trajectory[:, :2]
        fit_t = np.linspace(0.0, 1.0, self.fit_points)
        _, path_length = _polyline_sample(points, fit_t)
        start = points[0]
        goal = points[-1]
        chord = goal - start
        chord_length = float(np.linalg.norm(chord))
        if chord_length <= 1e-8:
            raise ValueError("Start and goal positions must differ")
        theta = math.atan2(chord[1], chord[0])
        if trajectory.shape[1] >= 3:
            start_yaw = float(trajectory[0, 2])
            goal_yaw = float(trajectory[-1, 2])
        else:
            first = points[1] - points[0]
            last = points[-1] - points[-2]
            start_yaw = math.atan2(first[1], first[0])
            goal_yaw = math.atan2(last[1], last[0])
        endpoint_slope = chord_length / path_length
        warped = canonical_parameter_map(fit_t, endpoint_slope)
        physical_target, _ = _polyline_sample(points, warped)
        cosine = math.cos(theta)
        sine = math.sin(theta)
        relative = physical_target - start
        canonical_target = np.stack(
            [
                (cosine * relative[:, 0] + sine * relative[:, 1])
                / chord_length,
                (-sine * relative[:, 0] + cosine * relative[:, 1])
                / chord_length,
            ],
            axis=1,
        )
        start_direction = np.array(
            [math.cos(start_yaw - theta), math.sin(start_yaw - theta)]
        )
        goal_direction = np.array(
            [math.cos(goal_yaw - theta), math.sin(goal_yaw - theta)]
        )
        fixed = np.stack(
            [
                np.zeros(2),
                start_direction / self.beta,
                np.array([1.0, 0.0]) - goal_direction / self.beta,
                np.array([1.0, 0.0]),
            ]
        )
        fixed_indices = np.asarray([0, 1, 24, 25], dtype=np.int64)
        free_indices = np.arange(2, 24, dtype=np.int64)
        fit_basis = self.fit_basis.detach().cpu().numpy()
        fixed_basis = fit_basis[:, fixed_indices]
        interior_basis = fit_basis[:, free_indices]
        target_without_boundary = canonical_target - fixed_basis @ fixed

        second_difference = np.zeros((24, 26), dtype=np.float64)
        for index in range(24):
            second_difference[index, index : index + 3] = (1.0, -2.0, 1.0)
        smooth_free = second_difference[:, free_indices]
        smooth_fixed = second_difference[:, fixed_indices]
        fixed_smoothness = smooth_fixed @ fixed

        curvature_limit = float(SAFETY_COST_CONFIG.curvature_limit)
        curvature_epsilon = float(
            SAFETY_COST_CONFIG.hard_constraint_epsilon
        )
        training_start_pose = torch.tensor(
            [[start[0], start[1], start_yaw]], dtype=torch.float32
        )
        training_goal_pose = torch.tensor(
            [[goal[0], goal[1], goal_yaw]], dtype=torch.float32
        )
        start_direction_tensor = torch.as_tensor(
            start_direction, dtype=self.source_factor.dtype
        ).unsqueeze(0)
        goal_direction_tensor = torch.as_tensor(
            goal_direction, dtype=self.source_factor.dtype
        ).unsqueeze(0)

        def solve_candidate(smoothness_weight: float):
            weight = float(smoothness_weight)
            if weight == 0.0:
                free = np.linalg.lstsq(
                    interior_basis,
                    target_without_boundary,
                    rcond=None,
                )[0]
            else:
                scale = math.sqrt(weight)
                augmented_basis = np.concatenate(
                    [interior_basis, scale * smooth_free], axis=0
                )
                augmented_target = np.concatenate(
                    [
                        target_without_boundary,
                        -scale * fixed_smoothness,
                    ],
                    axis=0,
                )
                free = np.linalg.lstsq(
                    augmented_basis,
                    augmented_target,
                    rcond=None,
                )[0]
            candidate_controls = np.zeros((26, 2), dtype=np.float64)
            candidate_controls[fixed_indices] = fixed
            candidate_controls[free_indices] = free
            reconstructed = fit_basis @ candidate_controls
            reconstruction_rmse = float(
                np.sqrt(
                    np.mean(np.square(reconstructed - canonical_target))
                )
                * chord_length
            )
            control_tensor = torch.as_tensor(
                candidate_controls, dtype=self.source_factor.dtype
            ).unsqueeze(0)
            # The actual Stage-1 target is float32.  Candidate admission must
            # audit that encoded-and-quantized state after the float32 decoder
            # round trip, rather than the pre-encoding float64 controls.
            training_state = self.encode_canonical_controls(
                control_tensor,
                start_direction_tensor,
                goal_direction_tensor,
            ).to(torch.float32)
            with torch.no_grad():
                max_curvature = float(
                    self.audit_curvature(
                        training_state,
                        training_start_pose,
                        training_goal_pose,
                    ).amax()
                )
            return {
                "controls": candidate_controls,
                "training_state": training_state[0].clone(),
                "reconstruction_rmse": reconstruction_rmse,
                "max_curvature": max_curvature,
                "curvature_feasible": bool(
                    np.isfinite(max_curvature)
                    and max_curvature <= curvature_limit + curvature_epsilon
                ),
                "smoothness_weight": weight,
            }

        candidates = [solve_candidate(0.0)]
        if not candidates[0]["curvature_feasible"]:
            candidates.extend(
                solve_candidate(weight)
                for weight in self.demonstration_fit_smoothness_weights[1:]
            )
        feasible_candidates = [
            candidate
            for candidate in candidates
            if candidate["curvature_feasible"]
        ]
        if feasible_candidates:
            selected = min(
                feasible_candidates,
                key=lambda candidate: (
                    candidate["reconstruction_rmse"],
                    candidate["smoothness_weight"],
                ),
            )
        else:
            selected = candidates[0]
        finite_candidates = [
            candidate
            for candidate in candidates
            if np.isfinite(candidate["max_curvature"])
        ]
        minimum_curvature_candidate = (
            min(
                finite_candidates,
                key=lambda candidate: (
                    candidate["max_curvature"],
                    candidate["reconstruction_rmse"],
                    candidate["smoothness_weight"],
                ),
            )
            if finite_candidates
            else candidates[0]
        )
        state = selected["training_state"]
        return state, PathCoordinateFitDiagnostics(
            chord_length=chord_length,
            path_length=path_length,
            endpoint_slope=endpoint_slope,
            control_reconstruction_rmse=selected[
                "reconstruction_rmse"
            ],
            max_curvature=selected["max_curvature"],
            curvature_feasible=selected["curvature_feasible"],
            smoothness_weight=selected["smoothness_weight"],
            minimum_candidate_max_curvature=minimum_curvature_candidate[
                "max_curvature"
            ],
            minimum_curvature_smoothness_weight=minimum_curvature_candidate[
                "smoothness_weight"
            ],
        )


# Canonical production-facing short name.
BoundaryConstrainedPathRepresentation = (
    FirstOrderBoundaryConstrainedBSpline
)

# Compatibility aliases for experiments and checkpoints created before the
# terminology was standardized.  New production code must use the names above.
REPRESENTATION_SEMANTIC_VERSION = PATH_REPRESENTATION_SEMANTIC_VERSION
GaugeFitDiagnostics = PathCoordinateFitDiagnostics
quintic_gauge = canonical_parameter_map
GaugeFixedFirstOrderBridge = FirstOrderBoundaryConstrainedBSpline
GaugeFixedBridge = FirstOrderBoundaryConstrainedBSpline

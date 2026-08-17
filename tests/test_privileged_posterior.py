import unittest
import inspect
from unittest.mock import Mock, call, patch

import numpy as np
import torch
from torch import nn

from boundary_constrained_path import BoundaryConstrainedPathRepresentation
from dataLoader_dit import (
    STAGE2_MAX_CONTIGUOUS_BLOCKED_FRACTION,
    STAGE2_MAX_DEMO_BLOCKED_FRACTION,
    STAGE2_MAX_MASKED_FRACTION,
    UnevenPathDataLoader,
    build_masked_normal_input,
    dense_demo_bspline,
    endpoint_yaw_corridors,
    erode_mask_for_vehicle,
    fit_demo_bspline_control_points,
    generate_random_mask,
    informed_demo_ellipse,
    mask_start_goal_connected,
    normalize_mask,
    trajectory_mask_blockage_metrics,
    trajectory_is_allowed,
)
from dit.Models import PathDiffusionTransformer
from grad_optimizer import (
    _sample_cost_map_on_dense_trajectory,
    _sample_mask,
    build_forbidden_distance_map,
    build_signed_mask_distance_map,
    discrete_turning_curvature,
    endpoint_stability_feasibility,
    optimize_privileged_trajectories,
    privileged_planning_cost,
    top_tail_mean,
    trajectory_validity_metrics,
)
from map_config import MAP_CONFIG, SAFETY_COST_CONFIG
from posterior_pipeline import (
    DaggerReplayBuffer,
    ReplayPairDataset,
    _log_numeric_tree,
    _log_stage1_update_metrics,
    _select_endpoint_feasible_contexts,
    noise_invariance_diagnostics,
    prior_transport_loss,
)


def _small_model(use_radial_output=False):
    model = PathDiffusionTransformer(
        n_layers=1,
        n_heads=2,
        d_model=32,
        d_inner=64,
        dropout=0.0,
        map_channels=4,
        use_radial_output=use_radial_output,
    )
    model.use_gradient_checkpoint = False
    return model.eval()


def _analytic_geometry(trajectory, yaw=0.0, curvature=0.0):
    """为低层 cost/validity 单测显式提供解析几何契约。"""
    trajectory = torch.as_tensor(trajectory)
    shape = trajectory.shape[:-1]
    if not torch.is_tensor(yaw):
        yaw = torch.full(shape, float(yaw), dtype=trajectory.dtype)
    else:
        yaw = yaw.to(dtype=trajectory.dtype, device=trajectory.device)
    if not torch.is_tensor(curvature):
        curvature = torch.full(
            shape, float(curvature), dtype=trajectory.dtype
        )
    else:
        curvature = curvature.to(
            dtype=trajectory.dtype, device=trajectory.device
        )
    return {
        "analytic_yaw": yaw,
        "analytic_curvature": curvature,
    }


class PrivilegedPosteriorTests(unittest.TestCase):
    def test_stage1_reports_but_does_not_drop_curvature_invalid_fit(self):
        class ToyMeanFlow(nn.Module):
            num_edges = 22
            coordinate_scale = 10.0

            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(0.5))
                self.trajectory_representation = (
                    BoundaryConstrainedPathRepresentation(dense_points=200)
                )

            @staticmethod
            def project_zero_sum(value):
                return value

            @staticmethod
            def sample_timesteps(batch_size, device, generator=None):
                del generator
                return torch.full((batch_size,), 0.75, device=device)

            def forward(self, map_input, state, t, r, start, goal):
                del map_input, t, r, start, goal
                return self.scale * state

        model = ToyMeanFlow()
        target = torch.zeros(2, 22, 2)
        target[1, :, 1] = torch.where(
            torch.arange(22) % 2 == 0,
            torch.tensor(4.0),
            torch.tensor(-4.0),
        )
        start = torch.tensor([[-2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
        goal = torch.tensor([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        batch = {
            "path_coordinates": target,
            "map": torch.zeros(2, 4, 4, 4),
            "start_pose": start,
            "goal_pose": goal,
        }
        curvature = model.trajectory_representation.audit_curvature(
            target, start, goal
        )
        expected_invalid = int(
            (
                curvature.amax(dim=1)
                > SAFETY_COST_CONFIG.curvature_limit
                + SAFETY_COST_CONFIG.hard_constraint_epsilon
            ).sum()
        )
        self.assertGreater(expected_invalid, 0)

        loss, terms = prior_transport_loss(
            model,
            batch,
            torch.device("cpu"),
            generator=torch.Generator().manual_seed(9),
        )
        self.assertIsNotNone(loss)
        self.assertEqual(terms["invalid_demo_curvature"], expected_invalid)
        self.assertEqual(terms["skipped_demo_curvature"], 0)
        self.assertEqual(terms["used_examples"], 2)
        loss.backward()
        self.assertIsNotNone(model.scale.grad)

    def test_tensorboard_numeric_tree_logs_only_finite_scalars(self):
        writer = Mock()
        _log_numeric_tree(
            writer,
            "stage1",
            {
                "loss": {"train": 1.25, "validation": torch.tensor(2.5)},
                "nan": float("nan"),
                "text": "skip",
                "list": [1, 2],
            },
            3,
        )
        self.assertEqual(
            writer.add_scalar.call_args_list,
            [
                call("stage1/loss/train", 1.25, 3),
                call("stage1/loss/validation", 2.5, 3),
            ],
        )

    def test_stage1_tensorboard_logs_loss_on_each_optimizer_update(self):
        writer = Mock()
        _log_stage1_update_metrics(
            writer,
            loss=torch.tensor(1.5),
            terms={
                "flow": torch.tensor(1.25),
                "endpoint": torch.tensor(1.0),
            },
            gradient_norm=torch.tensor(0.75),
            learning_rate=1e-4,
            global_update=17,
        )
        self.assertEqual(
            writer.add_scalar.call_args_list,
            [
                call("stage1/loss/train_step", 1.5, 17),
                call("stage1/loss/flow_step", 1.25, 17),
                call("stage1/loss/endpoint_step", 1.0, 17),
                call("stage1/optimization/gradient_norm_step", 0.75, 17),
                call("stage1/optimization/learning_rate_step", 1e-4, 17),
            ],
        )

    def test_demo_bspline_respects_endpoint_yaw_hard_tolerance(self):
        x = np.linspace(-1.0, 1.0, 40, dtype=np.float32)
        trajectory = np.stack(
            [
                x,
                np.zeros_like(x),
                np.linspace(np.pi / 2, -np.pi / 2, x.size),
            ],
            axis=-1,
        )
        control_points = fit_demo_bspline_control_points(trajectory)
        start_yaw = np.arctan2(
            *(control_points[1] - control_points[0])[::-1]
        )
        goal_yaw = np.arctan2(
            *(control_points[-1] - control_points[-2])[::-1]
        )
        angle_error = lambda actual, target: abs(
            np.arctan2(
                np.sin(actual - target),
                np.cos(actual - target),
            )
        )
        self.assertLessEqual(
            angle_error(start_yaw, np.pi / 2),
            0.35 + 1e-5,
        )
        self.assertLessEqual(
            angle_error(goal_yaw, -np.pi / 2),
            0.35 + 1e-5,
        )

    def test_constrained_demo_fit_does_not_create_endpoint_curvature_spike(self):
        angle = np.linspace(0.0, np.pi / 2, 30, dtype=np.float32)
        radius = 2.0
        trajectory = np.stack(
            [
                radius * np.sin(angle),
                radius * (1.0 - np.cos(angle)),
                angle,
            ],
            axis=-1,
        )
        dense = torch.as_tensor(
            dense_demo_bspline(trajectory),
            dtype=torch.float32,
        ).unsqueeze(0)
        max_curvature = float(discrete_turning_curvature(dense).amax())
        self.assertLessEqual(max_curvature, 2.1 + 1e-5)

    def test_single_mask_is_normalized_without_extra_obstacle_channel(self):
        source = np.ones((3, 4), dtype=np.float32)
        source[0, 0] = 0.0
        source[1, 2] = -1.0
        result = normalize_mask(source, (3, 4))
        self.assertEqual(result[0, 0], 0.0)
        self.assertEqual(result[1, 2], 0.0)
        self.assertEqual(result[2, 3], 1.0)

    def test_masked_normals_are_replaced_by_deterministic_gaussian_noise(self):
        normals = np.full((3, 4, 3), 0.25, dtype=np.float32)
        mask = np.ones((3, 4), dtype=np.float32)
        mask[1:, 2:] = 0.0
        first = build_masked_normal_input(normals, mask, seed=17)
        second = build_masked_normal_input(normals, mask, seed=17)

        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.array_equal(first[..., 3], mask))
        self.assertTrue(np.all(first[mask > 0.5, :3] == 0.25))
        self.assertFalse(np.all(first[mask < 0.5, :3] == 0.25))
        self.assertLess(float(first[mask < 0.5, :3].std()), 0.4)

    def test_training_noise_can_be_resampled_dynamically(self):
        normals = np.zeros((4, 4, 3), dtype=np.float32)
        mask = np.zeros((4, 4), dtype=np.float32)
        first = build_masked_normal_input(normals, mask, seed=None)
        second = build_masked_normal_input(normals, mask, seed=None)
        self.assertFalse(np.array_equal(first[..., :3], second[..., :3]))

    def test_random_masks_cover_large_small_and_mixed_patterns(self):
        trajectory = np.stack(
            [np.linspace(-8.0, 8.0, 25), np.zeros(25)], axis=-1
        ).astype(np.float32)
        patterns = set()
        inactive_count = 0
        for seed in range(80):
            mask, metadata = generate_random_mask(
                (100, 100),
                seed,
                trajectory,
                return_metadata=True,
            )
            self.assertTrue(
                trajectory_is_allowed(dense_demo_bspline(trajectory), mask)
            )
            if not metadata["mask_active"]:
                inactive_count += 1
                self.assertEqual(metadata["masked_fraction"], 0.0)
                continue
            self.assertGreater(metadata["masked_fraction"], 0.0)
            self.assertEqual(
                metadata["sampling_attempts"],
                sum(metadata["proposed_type_counts"].values()),
            )
            self.assertEqual(
                metadata["resamples"],
                metadata["sampling_attempts"] - 1,
            )
            patterns.add(
                (
                    metadata["has_large_cutout"],
                    metadata["has_small_obstacles"],
                )
            )
        self.assertIn((True, False), patterns)
        self.assertIn((False, True), patterns)
        self.assertIn((True, True), patterns)
        self.assertGreater(inactive_count, 0)

    def test_bernoulli_zero_returns_complete_map(self):
        trajectory = np.array([[-5.0, 0.0], [5.0, 0.0]], dtype=np.float32)
        mask, metadata = generate_random_mask(
            (100, 100),
            3,
            trajectory,
            p_mask=0.0,
            return_metadata=True,
        )
        self.assertTrue(np.all(mask == 1.0))
        self.assertFalse(metadata["mask_active"])

    def test_vehicle_radius_erodes_mask_around_obstacle(self):
        mask = np.ones((100, 100), dtype=np.float32)
        mask[50, 50] = 0.0
        eroded = erode_mask_for_vehicle(mask)
        self.assertEqual(eroded[50, 50], 0.0)
        self.assertEqual(eroded[50, 49], 0.0)
        self.assertEqual(eroded[0, 0], 1.0)

    def test_random_mask_is_deterministic_for_same_sample(self):
        trajectory = np.array([[-5.0, -5.0], [5.0, 5.0]], dtype=np.float32)
        first = generate_random_mask((100, 100), 1234, trajectory)
        second = generate_random_mask((100, 100), 1234, trajectory)
        self.assertTrue(np.array_equal(first, second))

    def test_stage2_independent_masks_can_block_old_demo(self):
        trajectory = np.stack(
            [np.linspace(-8.0, 8.0, 25), np.zeros(25)], axis=-1
        ).astype(np.float32)
        dense = dense_demo_bspline(trajectory)
        blocked = 0
        for seed in range(80):
            mask = generate_random_mask(
                (100, 100),
                seed,
                trajectory,
                p_mask=1.0,
                require_trajectory_clear=False,
            )
            start_corridor, goal_corridor = endpoint_yaw_corridors(trajectory)
            self.assertTrue(trajectory_is_allowed(start_corridor, mask))
            self.assertTrue(trajectory_is_allowed(goal_corridor, mask))
            blocked += int(not trajectory_is_allowed(dense, mask))
        self.assertGreater(blocked, 0)

    def test_stage2_masks_are_bounded_local_interventions(self):
        trajectory = np.stack(
            [np.linspace(-8.0, 8.0, 25), np.zeros(25)], axis=-1
        ).astype(np.float32)
        dense = dense_demo_bspline(trajectory)
        locally_blocked = 0
        for seed in range(120):
            mask, metadata = generate_random_mask(
                (100, 100),
                seed,
                trajectory,
                p_mask=1.0,
                require_trajectory_clear=False,
                return_metadata=True,
            )
            blockage = trajectory_mask_blockage_metrics(dense, mask)
            self.assertTrue(
                mask_start_goal_connected(
                    mask, trajectory[0], trajectory[-1]
                )
            )
            self.assertLessEqual(
                metadata["masked_fraction"],
                STAGE2_MAX_MASKED_FRACTION + 1e-7,
            )
            self.assertLessEqual(
                blockage["blocked_fraction"],
                STAGE2_MAX_DEMO_BLOCKED_FRACTION + 1e-7,
            )
            self.assertLessEqual(
                blockage["max_contiguous_blocked_fraction"],
                STAGE2_MAX_CONTIGUOUS_BLOCKED_FRACTION + 1e-7,
            )
            locally_blocked += int(blockage["blocked_fraction"] > 0.0)
        self.assertGreater(locally_blocked, 0)

    def test_mask_blockage_metrics_checks_segment_interiors(self):
        mask = np.ones((100, 100), dtype=np.float32)
        # The two stored endpoints are allowed, but the physical segment crosses
        # a forbidden vertical strip between them.
        mask[:, 49:51] = 0.0
        trajectory = np.asarray(
            [[-4.0, 0.0], [4.0, 0.0]],
            dtype=np.float32,
        )

        blockage = trajectory_mask_blockage_metrics(trajectory, mask)

        self.assertFalse(trajectory_is_allowed(trajectory, mask))
        self.assertGreater(blockage["blocked_fraction"], 0.0)
        self.assertGreater(
            blockage["max_contiguous_blocked_fraction"],
            0.0,
        )

    def test_stage1_obstacles_are_near_route_but_do_not_block_demo(self):
        x = np.linspace(-7.0, 7.0, 25, dtype=np.float32)
        trajectory = np.stack(
            [x, 0.8 * np.sin(x / 2.0)], axis=-1
        )
        dense = dense_demo_bspline(trajectory)
        informed = informed_demo_ellipse((100, 100), dense)
        near_route_obstacles = 0
        for seed in range(20):
            mask, metadata = generate_random_mask(
                (100, 100),
                seed,
                trajectory,
                p_mask=1.0,
                require_trajectory_clear=True,
                return_metadata=True,
            )
            self.assertTrue(trajectory_is_allowed(dense, mask))
            self.assertEqual(metadata["demo_blocked_fraction"], 0.0)
            if metadata["trajectory_obstacle_mode"] == "near_route_nonblocking":
                near_route_obstacles += 1
                self.assertTrue(np.any(mask[informed] <= 0.5))
        self.assertGreater(near_route_obstacles, 0)

    def test_endpoint_corridors_follow_given_start_and_goal_yaw(self):
        x = np.linspace(-6.0, 6.0, 25, dtype=np.float32)
        trajectory = np.stack(
            [x, np.zeros_like(x), np.zeros_like(x)],
            axis=-1,
        )
        trajectory[0, 2] = np.pi / 2
        trajectory[-1, 2] = -np.pi / 2

        start_corridor, goal_corridor = endpoint_yaw_corridors(
            trajectory,
            corridor_meters=0.5,
        )

        self.assertGreater(start_corridor[-1, 1], start_corridor[0, 1])
        self.assertGreater(goal_corridor[-1, 1], goal_corridor[0, 1])
        for seed in range(20):
            mask = generate_random_mask(
                (100, 100),
                seed,
                trajectory,
                p_mask=1.0,
                require_trajectory_clear=False,
            )
            self.assertTrue(trajectory_is_allowed(start_corridor, mask))
            self.assertTrue(trajectory_is_allowed(goal_corridor, mask))

    def test_stage2_mask_variant_changes_each_round_and_is_reproducible(self):
        loader = UnevenPathDataLoader.__new__(UnevenPathDataLoader)
        loader.mask_seed = 2026
        loader.p_mask = 1.0
        loader.mask_mode = "stage2_independent"
        loader.vehicle_radius_meters = 0.2
        trajectory = np.stack(
            [np.linspace(-8.0, 8.0, 25), np.zeros(25)], axis=-1
        ).astype(np.float32)
        round_one = loader._make_partial_mask(
            7, (100, 100), trajectory, mask_variant=1
        )
        round_one_again = loader._make_partial_mask(
            7, (100, 100), trajectory, mask_variant=1
        )
        round_two = loader._make_partial_mask(
            7, (100, 100), trajectory, mask_variant=2
        )
        self.assertTrue(np.array_equal(round_one, round_one_again))
        self.assertFalse(np.array_equal(round_one, round_two))

    def test_configuration_mask_connectivity_uses_eroded_mask(self):
        mask = np.ones((100, 100), dtype=np.float32)
        mask[:, 49:51] = 0.0
        self.assertFalse(
            mask_start_goal_connected(mask, (-4.0, 0.0), (4.0, 0.0))
        )
        mask[50, 49:51] = 1.0
        self.assertTrue(
            mask_start_goal_connected(mask, (-4.0, 0.0), (4.0, 0.0))
        )

    def test_forbidden_distance_has_gradient_signal_inside_mask(self):
        mask = torch.ones(1, 100, 100)
        mask[:, 40:60, 40:60] = 0.0
        distance = build_forbidden_distance_map(
            mask, MAP_CONFIG.cost_map_info()
        )
        self.assertGreater(float(distance[0, 50, 50]), 0.0)
        self.assertEqual(float(distance[0, 0, 0]), 0.0)

    def test_signed_mask_distance_and_cost_gradient_have_correct_sign(self):
        map_info = {
            "origin": (-1.0, -1.0, -np.pi),
            "resolution": 0.1,
            "size": (20, 20, 4),
        }
        mask = torch.ones(20, 20)
        mask[:, 10:] = 0.0
        signed = build_signed_mask_distance_map(mask, map_info)
        self.assertGreater(float(signed[0, 10, 2]), 0.0)
        self.assertLess(float(signed[0, 10, 17]), 0.0)

        trajectory = torch.stack(
            [
                torch.linspace(0.15, 0.45, 20),
                torch.linspace(-0.6, 0.6, 20),
            ],
            dim=-1,
        ).unsqueeze(0).requires_grad_(True)
        direction = torch.zeros_like(trajectory)
        direction[..., 0] = -1.0  # 朝左移动，即朝可通行区移动。
        stability = torch.ones(20, 20, 4)
        start = torch.tensor([0.15, -0.6, np.pi / 2])
        goal = torch.tensor([0.45, 0.6, np.pi / 2])

        def mask_only_cost(points):
            return privileged_planning_cost(
                points,
                start,
                goal,
                stability,
                map_info,
                mask=mask,
                signed_mask_distance_map=signed,
                forbidden_weight=1.0,
                stability_weight=0.0,
                curvature_weight=0.0,
                endpoint_yaw_weight=0.0,
                regularization_weight=0.0,
                **_analytic_geometry(points),
            )

        cost = mask_only_cost(trajectory)
        gradient = torch.autograd.grad(cost, trajectory)[0]
        analytical = float((gradient * direction).sum())
        epsilon = 1e-3
        finite_difference = float(
            (
                mask_only_cost(trajectory.detach() + epsilon * direction)
                - mask_only_cost(trajectory.detach() - epsilon * direction)
            )
            / (2.0 * epsilon)
        )
        self.assertLess(analytical, 0.0)
        self.assertLess(finite_difference, 0.0)
        self.assertAlmostEqual(
            analytical,
            finite_difference,
            delta=max(1e-3, abs(finite_difference) * 0.03),
        )

    def test_signed_mask_sampling_uses_physical_cell_centers(self):
        map_info = {
            "origin": (-1.0, -1.0, -np.pi),
            "resolution": 0.1,
            "size": (20, 20, 4),
        }
        column_values = torch.arange(20, dtype=torch.float32).repeat(20, 1)
        cell_centers = torch.tensor(
            [[[-0.95, -0.95], [-0.85, -0.95], [0.95, -0.95]]]
        )
        sampled = _sample_mask(cell_centers, column_values, map_info)
        self.assertTrue(
            torch.allclose(sampled, torch.tensor([[0.0, 1.0, 19.0]]))
        )

        mask = torch.ones(20, 20)
        mask[:, 10:] = 0.0
        signed = build_signed_mask_distance_map(mask, map_info)
        interface = _sample_mask(
            torch.tensor([[[0.0, 0.0]]]), signed, map_info
        )
        self.assertLess(abs(float(interface)), 1e-4)

    def test_box_boundary_inset_is_symmetric_in_physical_bounds(self):
        map_info = {
            "origin": (-1.0, -1.0, -np.pi),
            "resolution": 0.1,
            "size": (20, 20, 4),
            "bounds": (-1.0, 1.0, -1.0, 1.0),
        }
        y = torch.linspace(-0.5, 0.5, 20)
        trajectories = torch.stack(
            [
                torch.stack([torch.full_like(y, -0.85), y], dim=-1),
                torch.stack([torch.full_like(y, 0.85), y], dim=-1),
            ]
        )
        yaw = torch.full((2, 20), np.pi / 2)
        curvature = torch.zeros(2, 20)
        validity = trajectory_validity_metrics(
            trajectories,
            torch.ones(20, 20, 4),
            map_info,
            analytic_yaw=yaw,
            analytic_curvature=curvature,
            start_pose=torch.tensor(
                [[-0.85, -0.5, np.pi / 2], [0.85, -0.5, np.pi / 2]]
            ),
            goal_pose=torch.tensor(
                [[-0.85, 0.5, np.pi / 2], [0.85, 0.5, np.pi / 2]]
            ),
        )
        self.assertTrue(torch.equal(
            validity["box_component_ok"], torch.tensor([True, True])
        ))

    def test_signed_mask_gradient_points_outward_on_both_axes_and_medial_axis(self):
        map_info = {
            "origin": (-1.0, -1.0, -np.pi),
            "resolution": 0.1,
            "size": (20, 20, 4),
        }
        stability = torch.ones(20, 20, 4)

        def mask_gradient(mask, trajectory):
            signed = build_signed_mask_distance_map(mask, map_info)
            trajectory = trajectory.unsqueeze(0).requires_grad_(True)
            cost = privileged_planning_cost(
                trajectory,
                trajectory.detach()[0, 0].new_tensor(
                    [trajectory[0, 0, 0], trajectory[0, 0, 1], 0.0]
                ),
                trajectory.detach()[0, -1].new_tensor(
                    [trajectory[0, -1, 0], trajectory[0, -1, 1], 0.0]
                ),
                stability,
                map_info,
                mask=mask,
                signed_mask_distance_map=signed,
                forbidden_weight=1.0,
                stability_weight=0.0,
                curvature_weight=0.0,
                endpoint_yaw_weight=0.0,
                regularization_weight=0.0,
                **_analytic_geometry(trajectory),
            )
            return torch.autograd.grad(cost, trajectory)[0].mean(dim=(0, 1))

        vertical = torch.ones(20, 20)
        vertical[:, 8:12] = 0.0
        y = torch.linspace(-0.6, 0.6, 20)
        left_gradient = mask_gradient(
            vertical,
            torch.stack([torch.full_like(y, -0.15), y], dim=-1),
        )
        right_gradient = mask_gradient(
            vertical,
            torch.stack([torch.full_like(y, 0.05), y], dim=-1),
        )
        center_gradient = mask_gradient(
            vertical,
            torch.stack([torch.full_like(y, -0.05), y], dim=-1),
        )
        # 梯度下降方向分别向左、向右；medial axis 也不再严格为零。
        self.assertGreater(float(left_gradient[0]), 0.0)
        self.assertLess(float(right_gradient[0]), 0.0)
        self.assertGreater(abs(float(center_gradient[0])), 1e-6)

        horizontal = torch.ones(20, 20)
        horizontal[8:12, :] = 0.0
        x = torch.linspace(-0.6, 0.6, 20)
        lower_gradient = mask_gradient(
            horizontal,
            torch.stack([x, torch.full_like(x, -0.15)], dim=-1),
        )
        self.assertGreater(float(lower_gradient[1]), 0.0)
        self.assertGreater(
            abs(float(lower_gradient[1])),
            10.0 * abs(float(lower_gradient[0])),
        )

    def test_complete_mask_has_constant_distance_and_zero_mask_gradient(self):
        map_info = {
            "origin": (-1.0, -1.0, -np.pi),
            "resolution": 0.1,
            "size": (20, 20, 4),
        }
        mask = torch.ones(20, 20)
        signed = build_signed_mask_distance_map(mask, map_info)
        self.assertTrue(torch.equal(signed, torch.full_like(signed, signed[0, 0, 0])))
        self.assertTrue(
            np.all(
                erode_mask_for_vehicle(
                    mask.numpy(),
                    vehicle_radius_meters=0.2,
                    resolution=0.1,
                )
                == 1.0
            )
        )
        trajectory = torch.stack(
            [torch.linspace(-0.8, 0.8, 20), torch.zeros(20)],
            dim=-1,
        ).unsqueeze(0).requires_grad_(True)
        stability = torch.ones(20, 20, 4)
        cost_with_mask = privileged_planning_cost(
            trajectory,
            torch.tensor([-0.8, 0.0, 0.0]),
            torch.tensor([0.8, 0.0, 0.0]),
            stability,
            map_info,
            mask=mask,
            signed_mask_distance_map=signed,
            forbidden_weight=1.0,
            stability_weight=0.0,
            curvature_weight=0.0,
            endpoint_yaw_weight=0.0,
            regularization_weight=0.0,
            **_analytic_geometry(trajectory),
        )
        gradient_with_mask = torch.autograd.grad(
            cost_with_mask,
            trajectory,
            retain_graph=True,
        )[0]
        cost_without_mask = privileged_planning_cost(
            trajectory,
            torch.tensor([-0.8, 0.0, 0.0]),
            torch.tensor([0.8, 0.0, 0.0]),
            stability,
            map_info,
            mask=None,
            forbidden_weight=1.0,
            stability_weight=0.0,
            curvature_weight=0.0,
            endpoint_yaw_weight=0.0,
            regularization_weight=0.0,
            **_analytic_geometry(trajectory),
        )
        gradient_without_mask = torch.autograd.grad(
            cost_without_mask,
            trajectory,
        )[0]
        # 全 1 mask 对统一禁区项没有额外影响，剩余梯度只来自 box 距离。
        self.assertAlmostEqual(
            float(cost_with_mask),
            float(cost_without_mask),
            places=7,
        )
        self.assertTrue(
            torch.allclose(
                gradient_with_mask,
                gradient_without_mask,
                atol=1e-8,
                rtol=0.0,
            )
        )

    def test_unified_forbidden_region_covers_box_and_mask(self):
        map_info = {
            "origin": (-1.0, -1.0, -np.pi),
            "resolution": 0.1,
            "size": (20, 20, 4),
        }
        stability = torch.ones(20, 20, 4)
        outside = torch.stack(
            [torch.full((20,), 1.0), torch.linspace(-0.5, 0.5, 20)],
            dim=-1,
        ).unsqueeze(0)
        poses = (
            torch.tensor([1.0, -0.5, np.pi / 2]),
            torch.tensor([1.0, 0.5, np.pi / 2]),
        )
        _, components = privileged_planning_cost(
            outside,
            poses[0],
            poses[1],
            stability,
            map_info,
            mask=torch.ones(20, 20),
            return_components=True,
            **_analytic_geometry(outside, yaw=np.pi / 2),
        )
        validity = trajectory_validity_metrics(
            outside,
            stability,
            map_info,
            mask=torch.ones(20, 20),
            start_pose=poses[0],
            goal_pose=poses[1],
            **_analytic_geometry(outside, yaw=np.pi / 2),
        )
        self.assertGreater(float(components["forbidden_region"]), 0.0)
        self.assertFalse(bool(validity["forbidden_region_ok"][0]))
        self.assertTrue(bool(validity["mask_component_ok"][0]))
        self.assertFalse(bool(validity["box_component_ok"][0]))

    def test_forbidden_region_checks_interiors_of_segments(self):
        map_info = {
            "origin": (-1.0, -1.0, -np.pi),
            "resolution": 0.1,
            "size": (20, 20, 4),
        }
        mask = torch.ones(20, 20)
        # The stored trajectory points remain on either side of this strip;
        # only the straight segment between them enters mask=0.
        mask[:, 9:11] = 0.0
        trajectory = torch.tensor(
            [
                [-0.8, 0.0],
                [-0.3, 0.0],
                [0.3, 0.0],
                [0.8, 0.0],
            ],
            dtype=torch.float32,
        ).unsqueeze(0)
        stability = torch.ones(20, 20, 4)
        start = torch.tensor([-0.8, 0.0, 0.0])
        goal = torch.tensor([0.8, 0.0, 0.0])
        geometry = _analytic_geometry(trajectory)

        _, components = privileged_planning_cost(
            trajectory,
            start,
            goal,
            stability,
            map_info,
            mask=mask,
            return_components=True,
            **geometry,
        )
        validity = trajectory_validity_metrics(
            trajectory,
            stability,
            map_info,
            mask=mask,
            start_pose=start,
            goal_pose=goal,
            **geometry,
        )

        self.assertGreater(
            int(validity["mask_segment_sample_count"][0]),
            0,
        )
        self.assertGreater(float(components["mask_component_violation_ratio"]), 0.0)
        self.assertGreater(float(components["forbidden_violation_max"]), 0.0)
        self.assertFalse(bool(validity["mask_component_ok"][0]))
        self.assertFalse(bool(validity["forbidden_region_ok"][0]))

    def test_forbidden_region_reports_physical_box_failure_separately(self):
        map_info = {
            "origin": (-1.0, -1.0, -np.pi),
            "resolution": 0.1,
            "size": (20, 20, 4),
        }
        trajectory = torch.tensor(
            [
                [-0.8, -0.8],
                [-0.1, -1.2],
                [0.2, -1.2],
                [0.8, 0.8],
            ],
            dtype=torch.float32,
        ).unsqueeze(0)
        stability = torch.ones(20, 20, 4)
        start = torch.tensor([-0.8, -0.8, 0.0])
        goal = torch.tensor([0.8, 0.8, 0.0])

        validity = trajectory_validity_metrics(
            trajectory,
            stability,
            map_info,
            mask=torch.ones(20, 20),
            start_pose=start,
            goal_pose=goal,
            **_analytic_geometry(trajectory),
        )

        self.assertGreater(int(validity["mask_segment_sample_count"][0]), 0)
        self.assertGreater(float(validity["box_violation_max"][0]), 0.0)
        self.assertTrue(bool(validity["mask_component_ok"][0]))
        self.assertFalse(bool(validity["box_component_ok"][0]))
        self.assertFalse(bool(validity["forbidden_region_ok"][0]))

    def test_length_is_diagnostic_only_and_regularization_defaults_to_zero(self):
        map_info = {
            "origin": (-1.0, -1.0, -np.pi),
            "resolution": 0.1,
            "size": (20, 20, 4),
        }
        x = torch.linspace(-0.5, 0.5, 20)
        direct = torch.stack([x, torch.zeros_like(x)], dim=-1)
        detour = torch.stack(
            [x, 0.4 * torch.sin(torch.linspace(0.0, np.pi, 20))],
            dim=-1,
        )
        trajectories = torch.stack([direct, detour])
        total, components = privileged_planning_cost(
            trajectories,
            torch.tensor([[-0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]]),
            torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
            torch.ones(20, 20, 4),
            map_info,
            mask=torch.ones(20, 20),
            forbidden_weight=0.0,
            stability_weight=0.0,
            curvature_weight=0.0,
            endpoint_yaw_weight=0.0,
            return_per_sample=True,
            return_components=True,
            **_analytic_geometry(trajectories),
        )
        self.assertTrue(torch.equal(total, torch.zeros_like(total)))
        self.assertGreater(
            float(components["normalized_length"][1]),
            float(components["normalized_length"][0]),
        )
        self.assertTrue(
            torch.equal(
                components["weighted_regularization"],
                torch.zeros_like(total),
            )
        )

    def test_stability_sampling_axes_and_periodic_yaw(self):
        map_info = {
            "origin": (-1.0, -1.0, -np.pi),
            "resolution": 0.1,
            "size": (6, 5, 4),
        }
        cost_map = torch.empty(5, 6, 4)
        for row in range(5):
            for col in range(6):
                for yaw_bin in range(4):
                    cost_map[row, col, yaw_bin] = (
                        100 * row + 10 * col + yaw_bin
                    )
        trajectory = torch.tensor(
            [[[-0.85, -0.75], [-0.75, -0.75], [-0.65, -0.75]]]
        )
        _, _, _, sampled = _sample_cost_map_on_dense_trajectory(
            trajectory,
            torch.zeros(trajectory.shape[:2]),
            cost_map,
            map_info,
            trajectory.device,
        )
        self.assertAlmostEqual(float(sampled[0, 0]), 212.0, places=4)
        self.assertAlmostEqual(float(sampled[0, 2]), 232.0, places=4)

        periodic_map = torch.zeros(5, 6, 4)
        periodic_map[..., 3] = 1.0
        epsilon = 1e-3
        near_positive_pi = torch.tensor(
            [[[0.2, 0.0], [0.1, epsilon], [0.0, 2 * epsilon]]]
        )
        near_negative_pi = torch.tensor(
            [[[0.2, 0.0], [0.1, -epsilon], [0.0, -2 * epsilon]]]
        )
        positive = _sample_cost_map_on_dense_trajectory(
            near_positive_pi,
            torch.full(
                near_positive_pi.shape[:2], np.pi - epsilon
            ),
            periodic_map,
            map_info,
            near_positive_pi.device,
        )[3]
        negative = _sample_cost_map_on_dense_trajectory(
            near_negative_pi,
            torch.full(
                near_negative_pi.shape[:2], -np.pi + epsilon
            ),
            periodic_map,
            map_info,
            near_negative_pi.device,
        )[3]
        self.assertLess(float(positive.mean()), 0.05)
        self.assertLess(float(negative.mean()), 0.05)
        self.assertLess(float((positive - negative).abs().max()), 0.05)

    def test_endpoint_stability_feasibility_uses_hard_yaw_interval(self):
        map_info = {
            "origin": (-1.0, -1.0, -np.pi),
            "resolution": 0.1,
            "size": (20, 20, 8),
        }
        cost_map = torch.full((20, 20, 8), 0.2)
        # 起点在所有允许 yaw 下都低于 d_safe，固定端点使任务无解。
        cost_map[10, 5, :] = 0.1
        result = endpoint_stability_feasibility(
            # row=10,col=5 and row=10,col=15 cell centers.
            torch.tensor([-0.45, 0.05, 0.0]),
            torch.tensor([0.55, 0.05, 0.0]),
            cost_map,
            map_info,
            d_safe=0.15,
            start_yaw_tolerance_rad=0.35,
            goal_yaw_tolerance_rad=0.35,
        )
        self.assertFalse(result["feasible"])
        self.assertFalse(result["start_feasible"])
        self.assertTrue(result["goal_feasible"])
        self.assertAlmostEqual(
            result["start_best_stability_margin"],
            0.1,
            places=5,
        )

    def test_stage2_resamples_endpoint_stability_infeasible_contexts(self):
        class TinyDataset:
            def __len__(self):
                return 4

            def get_item(
                self,
                index,
                *,
                mask_variant,
                noise_seed,
                return_mask_metadata,
            ):
                return {
                    "start_pose": torch.tensor([float(index), 0.0, 0.0]),
                    "goal_pose": torch.tensor([1.0, 0.0, 0.0]),
                    "cost_map": torch.ones(2, 2, 2),
                    "mask_metadata": {},
                }

        infeasible = {
            "feasible": False,
            "start_feasible": False,
            "goal_feasible": True,
            "start_best_stability_margin": 0.0,
            "goal_best_stability_margin": 1.0,
        }
        feasible = {
            "feasible": True,
            "start_feasible": True,
            "goal_feasible": True,
            "start_best_stability_margin": 1.0,
            "goal_best_stability_margin": 1.0,
        }
        with patch(
            "posterior_pipeline.endpoint_stability_feasibility",
            side_effect=[infeasible, feasible, feasible],
        ):
            selected, metrics = _select_endpoint_feasible_contexts(
                TinyDataset(),
                [0, 1],
                dagger_round=0,
                seed=11,
            )
        self.assertEqual(len(selected), 2)
        self.assertEqual(metrics["context_candidates_drawn"], 3)
        self.assertEqual(metrics["endpoint_stability_rejections"], 1)
        self.assertAlmostEqual(
            metrics["raw_endpoint_stability_feasibility_rate"],
            2 / 3,
        )
        self.assertAlmostEqual(
            metrics["average_context_sampling_attempts"],
            1.5,
        )

    def test_curvature_degenerate_segments_are_finite_and_invalid(self):
        trajectory = torch.zeros(1, 5, 2)
        curvature = discrete_turning_curvature(trajectory)
        self.assertTrue(bool(torch.isfinite(curvature).all()))
        self.assertGreater(float(curvature.min()), 100.0)

    def test_short_segment_curvature_branch_has_nonzero_gradient(self):
        trajectory = torch.tensor(
            [[
                [-0.5, 0.0],
                [-0.4995, 0.0],
                [-0.2, 0.0],
                [0.2, 0.0],
                [0.5, 0.0],
            ]],
            requires_grad=True,
        )
        cost, components = privileged_planning_cost(
            trajectory,
            torch.tensor([-0.5, 0.0, 0.0]),
            torch.tensor([0.5, 0.0, 0.0]),
            torch.ones(20, 20, 4),
            {
                "origin": (-1.0, -1.0, -np.pi),
                "resolution": 0.1,
                "size": (20, 20, 4),
            },
            mask=torch.ones(20, 20),
            forbidden_weight=0.0,
            stability_weight=0.0,
            curvature_weight=1.0,
            endpoint_yaw_weight=0.0,
            regularization_weight=0.0,
            return_components=True,
            **_analytic_geometry(trajectory),
        )
        gradient = torch.autograd.grad(cost, trajectory)[0]
        self.assertGreater(
            float(components["short_segment_violation_max"]),
            0.0,
        )
        self.assertGreater(float(torch.linalg.vector_norm(gradient)), 0.0)

    def test_top_tail_mean_uses_ceil_and_keeps_worst_points(self):
        values = torch.arange(20, dtype=torch.float32).unsqueeze(0)
        result = top_tail_mean(values, ratio=0.11)
        self.assertEqual(float(result), 18.0)

    def test_single_mask_violation_is_not_hidden_by_mean(self):
        map_info = {
            "origin": (-1.0, -1.0, -np.pi),
            "resolution": 0.1,
            "size": (20, 20, 4),
        }
        mask = torch.ones(20, 20)
        mask[:, 10:] = 0.0
        signed = build_signed_mask_distance_map(mask, map_info)
        trajectory = torch.stack(
            [
                torch.full((200,), -0.5),
                torch.linspace(-0.8, 0.8, 200),
            ],
            dim=-1,
        ).unsqueeze(0)
        trajectory[:, 100, 0] = 0.5
        stability = torch.ones(20, 20, 4)
        start = torch.tensor([-0.5, -0.8, np.pi / 2])
        goal = torch.tensor([-0.5, 0.8, np.pi / 2])
        _, components = privileged_planning_cost(
            trajectory,
            start,
            goal,
            stability,
            map_info,
            mask=mask,
            signed_mask_distance_map=signed,
            return_components=True,
            **_analytic_geometry(trajectory, yaw=np.pi / 2),
        )
        validity = trajectory_validity_metrics(
            trajectory,
            stability,
            map_info,
            mask=mask,
            signed_mask_distance_map=signed,
            start_pose=start,
            goal_pose=goal,
            **_analytic_geometry(trajectory, yaw=np.pi / 2),
        )
        self.assertGreater(
            float(components["forbidden_violation_max"]),
            float(components["forbidden_violation_mean"]),
        )
        self.assertGreater(
            float(components["forbidden_violation_ratio"]),
            0.0,
        )
        self.assertFalse(bool(validity["forbidden_region_ok"][0]))

    def test_safe_at_k_is_grouped_by_condition(self):
        map_info = {
            "origin": (-1.0, -1.0, -np.pi),
            "resolution": 0.1,
            "size": (20, 20, 4),
        }
        stability = torch.ones(20, 20, 4)
        stability[:, 10:, :] = -1.0
        y = torch.linspace(-0.6, 0.6, 20)
        left = torch.stack([torch.full_like(y, -0.5), y], dim=-1)
        right = torch.stack([torch.full_like(y, 0.5), y], dim=-1)
        trajectories = torch.stack([left, right, right, right])
        starts = torch.tensor(
            [
                [-0.5, -0.6, np.pi / 2],
                [0.5, -0.6, np.pi / 2],
                [0.5, -0.6, np.pi / 2],
                [0.5, -0.6, np.pi / 2],
            ]
        )
        goals = starts.clone()
        goals[:, 1] = 0.6
        grouped = trajectory_validity_metrics(
            trajectories,
            stability,
            map_info,
            mask=torch.ones(20, 20),
            start_pose=starts,
            goal_pose=goals,
            condition_ids=torch.tensor([0, 0, 1, 1]),
            **_analytic_geometry(trajectories, yaw=np.pi / 2),
        )
        self.assertTrue(
            torch.equal(
                grouped["safe_at_k_by_condition"],
                torch.tensor([True, False]),
            )
        )
        self.assertEqual(float(grouped["safe_at_k"]), 0.5)
        ungrouped = trajectory_validity_metrics(
            trajectories,
            stability,
            map_info,
            mask=torch.ones(20, 20),
            start_pose=starts,
            goal_pose=goals,
            **_analytic_geometry(trajectories, yaw=np.pi / 2),
        )
        self.assertIsNone(ungrouped["safe_at_k"])

    def test_endpoint_yaw_is_hard_while_length_is_separate_quality_gate(self):
        map_info = {
            "origin": (-1.0, -1.0, -np.pi),
            "resolution": 0.1,
            "size": (20, 20, 4),
        }
        trajectory = torch.stack(
            [torch.linspace(-0.5, 0.5, 20), torch.zeros(20)],
            dim=-1,
        ).unsqueeze(0)
        stability = torch.ones(20, 20, 4)
        wrong_yaw = trajectory_validity_metrics(
            trajectory,
            stability,
            map_info,
            mask=torch.ones(20, 20),
            start_pose=torch.tensor([-0.5, 0.0, np.pi / 2]),
            goal_pose=torch.tensor([0.5, 0.0, np.pi / 2]),
            **_analytic_geometry(trajectory),
        )
        self.assertFalse(bool(wrong_yaw["endpoint_yaw_ok"][0]))
        self.assertFalse(bool(wrong_yaw["strict_valid"][0]))

        excessive_length = trajectory_validity_metrics(
            trajectory,
            stability,
            map_info,
            mask=torch.ones(20, 20),
            start_pose=torch.tensor([-0.5, 0.0, 0.0]),
            goal_pose=torch.tensor([0.5, 0.0, 0.0]),
            max_path_length_ratio=0.01,
            **_analytic_geometry(trajectory),
        )
        self.assertTrue(bool(excessive_length["strict_valid"][0]))
        self.assertFalse(bool(excessive_length["quality_ok"][0]))
        self.assertFalse(bool(excessive_length["accepted"][0]))

    def test_optimizer_preserves_earlier_hard_valid_candidate(self):
        proposal = torch.randn(1, 22, 2) * 0.1
        call_count = 0

        def fake_cost(
            trajectory,
            *_args,
            return_per_sample=False,
            return_components=False,
            **_kwargs,
        ):
            per_sample = trajectory.square().mean(dim=(1, 2))
            result = per_sample if return_per_sample else per_sample.mean()
            if return_components:
                return result, {
                    "task_cost": per_sample,
                    "total": per_sample,
                }
            return result

        def fake_validity(trajectory, *_args, **_kwargs):
            nonlocal call_count
            batch = trajectory.shape[0]
            first_candidate_only = call_count == 0
            call_count += 1
            strict = torch.full(
                (batch,),
                first_candidate_only,
                dtype=torch.bool,
                device=trajectory.device,
            )
            zeros = torch.zeros(batch, device=trajectory.device)
            ones = torch.ones(batch, dtype=torch.bool, device=trajectory.device)
            return {
                "strict_valid": strict,
                "accepted": strict,
                "forbidden_violation_max": zeros,
                "stability_violation_max": zeros,
                "curvature_violation_max": zeros,
                "start_yaw_error": zeros,
                "goal_yaw_error": zeros,
                "path_length_ratio": zeros,
                "finite_ok": ones,
            }

        with patch("grad_optimizer.privileged_planning_cost", side_effect=fake_cost):
            with patch(
                "grad_optimizer.trajectory_validity_metrics",
                side_effect=fake_validity,
            ):
                result = optimize_privileged_trajectories(
                    proposal,
                    torch.tensor([[-0.5, 0.0, 0.0]]),
                    torch.tensor([[0.5, 0.0, 0.0]]),
                    torch.ones(20, 20, 4),
                    {
                        "origin": (-1.0, -1.0, -np.pi),
                        "resolution": 0.1,
                        "size": (20, 20, 4),
                    },
                    iterations=2,
                    lr=0.1,
                    mask=None,
                )
        self.assertTrue(bool(result["best_strict_valid_found"][0]))
        self.assertTrue(
            torch.allclose(
                result["corrected_residual"],
                proposal,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                result["expert_objective"],
                result["regularized_task_cost"]
                + 2e-3 * result["proposal_tether"],
            )
        )

    def test_model_constructor_contains_only_active_configuration(self):
        parameters = set(
            inspect.signature(PathDiffusionTransformer.__init__).parameters
        ) - {"self"}
        self.assertEqual(parameters, set(PathDiffusionTransformer.CONFIG_KEYS))
        self.assertFalse(hasattr(_small_model(), "map_proj_25"))

    def test_legacy_radial_output_is_hard_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not support"):
            _small_model(use_radial_output=True)

    def test_fixed_source_noise_reproduces_same_proposal(self):
        model = _small_model()
        map_input = torch.zeros(1, 4, 32, 32)
        map_input[:, 3] = 1.0
        start = torch.tensor([[-0.8, -0.8, 1.0, 0.0]])
        goal = torch.tensor([[0.8, 0.8, 1.0, 0.0]])
        source = model.project_zero_sum(torch.randn(1, 22, 2))
        first = model.sample_pmf_onestep(
            map_input, start, goal, source_noise=source, return_residual=True
        )
        second = model.sample_pmf_onestep(
            map_input, start, goal, source_noise=source, return_residual=True
        )
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(tuple(first.shape), (1, 22, 2))

    def test_model_level_mask_noise_invariance_metrics_are_reported(self):
        mask = torch.ones(100, 100)
        mask[:30, :30] = 0.0
        map_input = torch.zeros(4, 100, 100)
        map_input[3] = mask
        dataset = [
            {
                "map": map_input,
                "mask": mask,
                "start_pose": torch.tensor([-5.0, -5.0, 0.0]),
                "goal_pose": torch.tensor([5.0, 5.0, 0.0]),
                "cost_map": torch.ones(100, 100, 36),
            }
        ]
        metrics = noise_invariance_diagnostics(
            _small_model(),
            dataset,
            torch.device("cpu"),
            max_contexts=1,
            noise_draws=3,
            seed=9,
        )
        self.assertEqual(metrics["contexts"], 1)
        self.assertIn("control_point_mean_deviation_m", metrics)
        self.assertIn("strict_valid_change_rate", metrics)
        self.assertEqual(metrics["contexts_with_stability_map"], 1)

    def test_noise_invariance_does_not_require_privileged_stability_map(self):
        mask = torch.ones(100, 100)
        mask[20:40, 20:40] = 0.0
        map_input = torch.zeros(4, 100, 100)
        map_input[3] = mask
        dataset = [
            {
                "map": map_input,
                "mask": mask,
                "start_pose": torch.tensor([-5.0, -5.0, 0.0]),
                "goal_pose": torch.tensor([5.0, 5.0, 0.0]),
            }
        ]

        metrics = noise_invariance_diagnostics(
            _small_model(),
            dataset,
            torch.device("cpu"),
            max_contexts=1,
            noise_draws=2,
            seed=11,
        )

        self.assertEqual(metrics["contexts"], 1)
        self.assertEqual(metrics["contexts_with_stability_map"], 0)
        self.assertIn("strict_valid_change_rate", metrics)

    @staticmethod
    def _buffer_entry(round_index, particle_index, priority):
        source = torch.full((25, 2), float(particle_index))
        proposal = source + 10.0
        target = source + 20.0
        return {
            "dataset_index": particle_index,
            "particle_index": particle_index,
            "round": round_index,
            "mask_variant": round_index + 1,
            "mask_noise_seed": 10_000 + particle_index,
            "source_noise": source,
            "proposal_residual": proposal,
            "target_residual": target,
            "initial_cost": priority,
            "final_cost": priority / 2.0,
            "initial_valid": False,
            "final_valid": True,
            "priority": priority,
        }

    def test_replay_is_bounded_and_keeps_recent_and_severe_history(self):
        buffer = DaggerReplayBuffer(
            5,
            recent_rounds=1,
            recent_fraction=0.6,
            priority_fraction=0.2,
            seed=7,
        )
        old = [self._buffer_entry(0, i, 100.0 if i == 0 else 1.0) for i in range(5)]
        current = [self._buffer_entry(1, 10 + i, 2.0) for i in range(5)]
        buffer.add(old, current_round=0)
        buffer.add(current, current_round=1)

        self.assertEqual(len(buffer), 5)
        self.assertTrue(any(entry["round"] == 1 for entry in buffer.entries))
        self.assertTrue(
            any(
                entry["round"] == 0 and entry["particle_index"] == 0
                for entry in buffer.entries
            )
        )

    def test_replay_preserves_source_proposal_target_lineage(self):
        entry = self._buffer_entry(2, 3, 4.0)
        buffer = DaggerReplayBuffer(2)
        buffer.add([entry], current_round=2)
        stored = buffer.entries[0]

        self.assertTrue(torch.equal(stored["source_noise"], entry["source_noise"]))
        self.assertTrue(
            torch.equal(stored["proposal_residual"], entry["proposal_residual"])
        )
        self.assertTrue(
            torch.equal(stored["target_residual"], entry["target_residual"])
        )

    def test_replay_reconstructs_saved_mask_and_noise_variant(self):
        class FakeDataset:
            def get_item(self, index, *, mask_variant, noise_seed):
                return {
                    "map": torch.tensor(
                        [float(index), float(mask_variant), float(noise_seed)]
                    ),
                    "start_pose": torch.zeros(3),
                    "goal_pose": torch.ones(3),
                }

        entry = self._buffer_entry(2, 5, 1.0)
        sample = ReplayPairDataset([entry], FakeDataset())[0]
        self.assertTrue(
            torch.equal(
                sample["map"],
                torch.tensor([5.0, 3.0, 10005.0]),
            )
        )


if __name__ == "__main__":
    unittest.main()

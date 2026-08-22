import numpy as np

from visualize_uav_observation import (
    SensorConfig,
    build_uav_survey_path,
    simulate_downward_lidar_scan,
    simulate_uav_observation,
)


def test_uav_observation_returns_are_subset_of_visible_and_support():
    elevation = np.zeros((40, 40), dtype=np.float32)
    result = simulate_uav_observation(
        elevation,
        (-4.0, 4.0, -4.0, 4.0),
        0.2,
        (0.0, 0.0),
        0.0,
        SensorConfig(range_m=4.0, tilt_deg=0.0, vertical_fov_deg=80.0, point_density=1.0),
        np.random.default_rng(7),
    )
    assert result["returns"].any()
    assert np.all(~result["returns"] | result["visible"])
    assert np.all(~result["support"] | result["candidate"])


def test_uav_observation_can_capture_terrain_occlusion():
    elevation = np.zeros((40, 40), dtype=np.float32)
    elevation[:, 20] = 3.0
    result = simulate_uav_observation(
        elevation,
        (-4.0, 4.0, -4.0, 4.0),
        0.2,
        (-2.0, 0.0),
        0.0,
        SensorConfig(range_m=8.0, tilt_deg=25.0, vertical_fov_deg=50.0, altitude_m=4.0, point_density=1.0),
        np.random.default_rng(8),
    )
    assert np.any(result["candidate"] & ~result["visible"])


def test_downward_scan_is_a_circular_footprint_not_a_forward_fan():
    elevation = np.zeros((80, 80), dtype=np.float32)
    config = SensorConfig(scan_radius_m=2.0, point_density=1.0, support_radius_cells=0)
    result = simulate_downward_lidar_scan(
        elevation,
        (-8.0, 8.0, -8.0, 8.0),
        0.2,
        (0.0, 0.0),
        config,
        np.random.default_rng(9),
    )
    candidate = result["candidate"]
    ys, xs = np.nonzero(candidate)
    assert candidate.any()
    assert np.allclose(xs.mean(), (candidate.shape[1] - 1) / 2.0, atol=1.0)
    assert np.allclose(ys.mean(), (candidate.shape[0] - 1) / 2.0, atol=1.0)
    assert np.array_equal(result["returns"], candidate)


def test_downward_scan_support_can_be_accumulated():
    elevation = np.zeros((80, 80), dtype=np.float32)
    config = SensorConfig(scan_radius_m=1.2, point_density=1.0, support_radius_cells=0)
    first = simulate_downward_lidar_scan(
        elevation, (-8.0, 8.0, -8.0, 8.0), 0.2, (-2.0, 0.0), config, np.random.default_rng(10)
    )
    second = simulate_downward_lidar_scan(
        elevation, (-8.0, 8.0, -8.0, 8.0), 0.2, (2.0, 0.0), config, np.random.default_rng(11)
    )
    accumulated = first["support"].copy()
    before = int(accumulated.sum())
    accumulated |= second["support"]
    assert int(accumulated.sum()) > before
    assert np.all(accumulated[first["support"]])


def test_uav_survey_route_is_map_scale_and_independent_of_vehicle_path():
    vehicle_path = np.asarray([[-8.0, 5.0], [-4.0, 5.5], [0.0, 3.0], [5.0, 2.0]], dtype=np.float32)
    route = build_uav_survey_path(
        (-10.0, 10.0, -10.0, 10.0),
        vehicle_path,
        np.random.default_rng(12),
        spacing_m=1.5,
    )
    assert route.shape[0] >= 20
    assert np.ptp(route[:, 0]) > 5.0 or np.ptp(route[:, 1]) > 5.0
    assert np.all(route[:, 0] >= -10.0) and np.all(route[:, 0] <= 10.0)
    assert np.all(route[:, 1] >= -10.0) and np.all(route[:, 1] <= 10.0)
    for point in vehicle_path:
        assert np.min(np.linalg.norm(route - point[None, :], axis=1)) <= 3.5


def test_uav_survey_route_is_densified_without_rendering_gaps():
    vehicle_path = np.asarray([[-8.0, 5.0], [0.0, 3.0], [7.0, 2.0]], dtype=np.float32)
    route = build_uav_survey_path(
        (-10.0, 10.0, -10.0, 10.0),
        vehicle_path,
        np.random.default_rng(13),
        spacing_m=1.5,
    )
    assert np.max(np.linalg.norm(np.diff(route, axis=0), axis=1)) <= 0.75 * 1.5 + 1e-6

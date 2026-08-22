import numpy as np

from dataLoader_dit import dense_demo_bspline, erode_mask_for_vehicle, trajectory_is_allowed
from uav_mask import generate_uav_observation_mask


def _route():
    x = np.linspace(-8.0, 7.0, 25, dtype=np.float32)
    return np.stack([x, 1.5 * np.sin(x / 2.5)], axis=1)


def test_uav_mask_is_deterministic_and_not_an_ellipse_cutout():
    route = _route()
    first, metadata = generate_uav_observation_mask(
        (100, 100), route, 17, return_metadata=True
    )
    second = generate_uav_observation_mask((100, 100), route, 17)
    assert np.array_equal(first, second)
    assert metadata["semantic_mode"] == "uav_observation_with_local_obstacles"
    assert 0.0 < metadata["observed_fraction"] < 1.0
    assert metadata["scan_center_count"] > metadata["corridor_anchor_count"]


def test_uav_mask_keeps_stage1_reference_route_in_vehicle_configuration_space():
    route = _route()
    mask = generate_uav_observation_mask(
        (100, 100), route, 29, require_trajectory_clear=True
    )
    vehicle_mask = erode_mask_for_vehicle(mask, vehicle_radius_meters=0.35)
    assert trajectory_is_allowed(dense_demo_bspline(route), vehicle_mask)


def test_uav_mask_zero_probability_is_complete():
    mask, metadata = generate_uav_observation_mask(
        (100, 100), _route(), 31, p_mask=0.0, return_metadata=True
    )
    assert np.all(mask == 1.0)
    assert metadata["mask_active"] is False


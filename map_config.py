"""Shared dataset and map geometry configuration.

Change the values in ``MAP_CONFIG`` when switching map scales.  Training,
loading, normalization, sampling, and physical-cost code import the derived
values from this module so they cannot silently drift apart.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class MapConfig:
    dataset_root: Path
    expected_environments: int
    size_meters: float
    grid_size: int
    yaw_bins: int = 36

    def __post_init__(self) -> None:
        if self.expected_environments <= 0:
            raise ValueError("expected_environments must be positive")
        if self.size_meters <= 0.0:
            raise ValueError("size_meters must be positive")
        if self.grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if self.yaw_bins <= 1:
            raise ValueError("yaw_bins must be greater than one")

    @property
    def half_extent(self) -> float:
        """Coordinate normalization scale for centered square maps."""
        return self.size_meters / 2.0

    @property
    def resolution(self) -> float:
        """Map resolution in meters per pixel."""
        return self.size_meters / self.grid_size

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        half = self.half_extent
        return (-half, half, -half, half)

    @property
    def origin_xy(self) -> Tuple[float, float]:
        half = self.half_extent
        return (-half, -half)

    @property
    def map_shape(self) -> Tuple[int, int]:
        return (self.grid_size, self.grid_size)

    @property
    def cost_map_size(self) -> Tuple[int, int, int]:
        return (self.grid_size, self.grid_size, self.yaw_bins)

    @property
    def cost_map_origin(self) -> Tuple[float, float, float]:
        return (*self.origin_xy, -3.141592653589793)

    def cost_map_info(self) -> Dict[str, Any]:
        return {
            "resolution": self.resolution,
            "origin": self.cost_map_origin,
            "size": self.cost_map_size,
            "bounds": self.bounds,
        }

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["dataset_root"] = str(self.dataset_root)
        result.update(
            {
                "half_extent": self.half_extent,
                "resolution": self.resolution,
                "bounds": self.bounds,
                "cost_map_size": self.cost_map_size,
                "cost_map_origin": self.cost_map_origin,
            }
        )
        return result


@dataclass(frozen=True)
class SafetyCostConfig:
    """Shared physical thresholds and Stage-2 tail-risk tradeoff.

    Physical thresholds stay fixed when the map extent changes because the
    vehicle itself is not being rescaled.  The curvature weight compensates
    for the change in gradient balance caused by coordinate normalization:
    relative curvature/ESDF gradients scale approximately with 1 / size^2.
    """

    map_size_meters: float
    reference_map_size_meters: float = 40.0
    # Safety should cover the observed unstable population (about 12%).
    tail_ratio: float = 0.15
    # Curvature only needs to suppress local spikes; applying the 15% safety
    # tail here made Stage 2 round off too much of the trajectory.
    curvature_tail_ratio: float = 0.05
    out_of_bound_tail_ratio: float = 0.05
    d_safe_meters: float = 0.15
    softplus_alpha: float = 10.0
    curvature_limit: float = 1.4
    boundary_safe_pixels: int = 1
    obstacle_weight: float = 1.0
    reference_curvature_weight: float = 1.0
    out_of_bound_weight: float = 1.0
    endpoint_weight: float = 1.0
    # Low-priority tie-breaker: once safety and curvature are acceptable,
    # prefer trajectories with smoother vehicle-control changes.
    quality_weight: float = 1e-4
    quality_term: str = "jerk"
    yaw_esdf_weight: float = 1.4

    @property
    def curvature_weight(self) -> float:
        scale_ratio = self.map_size_meters / self.reference_map_size_meters
        return self.reference_curvature_weight * scale_ratio**2

    def tail_risk_kwargs(self) -> Dict[str, Any]:
        return {
            "tail_ratio": self.tail_ratio,
            "curvature_tail_ratio": self.curvature_tail_ratio,
            "out_of_bound_tail_ratio": self.out_of_bound_tail_ratio,
            "d_safe": self.d_safe_meters,
            "alpha": self.softplus_alpha,
            "curvature_limit": self.curvature_limit,
            "boundary_safe_pixels": self.boundary_safe_pixels,
            "obstacle_weight": self.obstacle_weight,
            "curvature_weight": self.curvature_weight,
            "out_of_bound_weight": self.out_of_bound_weight,
            "endpoint_weight": self.endpoint_weight,
            "quality_weight": self.quality_weight,
            "quality_term": self.quality_term,
        }

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["curvature_weight"] = self.curvature_weight
        return result


def discover_environments(split_folder, expected_count=None):
    """Return all valid environment names in a train/validation split."""
    split_path = Path(split_folder)
    if not split_path.is_dir():
        raise FileNotFoundError(f"数据划分目录不存在: {split_path}")

    environments = []
    for env_path in sorted(path for path in split_path.iterdir() if path.is_dir()):
        if not (env_path / "map.p").is_file():
            raise FileNotFoundError(f"环境目录缺少 map.p: {env_path}")
        if not any(env_path.glob("path_*.p")):
            raise FileNotFoundError(f"环境目录缺少 path_*.p: {env_path}")
        environments.append(env_path.name)

    if not environments:
        raise ValueError(f"数据划分中没有发现环境: {split_path}")
    if expected_count is not None and len(environments) != expected_count:
        raise ValueError(
            f"{split_path} 发现 {len(environments)} 个环境，"
            f"配置要求 {expected_count} 个"
        )
    return environments


# Dataset20: 100 centered 20m x 20m maps sampled on a 100 x 100 grid.
# This is the single edit point for future map/dataset scale changes.
MAP_CONFIG = MapConfig(
    dataset_root=Path("/home/yrf/MPT/data/dataset0"),
    expected_environments=100,
    size_meters=20.0,
    grid_size=100,
)

# Stage-2 safety-cost tuning is centralized here.  For the current 20 m map,
# the scale compensation gives curvature_weight=(20/40)^2=0.25.
SAFETY_COST_CONFIG = SafetyCostConfig(map_size_meters=MAP_CONFIG.size_meters)

# Uppercase aliases are convenient for code that prefers macro-like constants.
MAP_SIZE_METERS = MAP_CONFIG.size_meters
MAP_GRID_SIZE = MAP_CONFIG.grid_size
MAP_HALF_EXTENT = MAP_CONFIG.half_extent
MAP_RESOLUTION = MAP_CONFIG.resolution
MAP_BOUNDS = MAP_CONFIG.bounds
MAP_ORIGIN_XY = MAP_CONFIG.origin_xy
MAP_YAW_BINS = MAP_CONFIG.yaw_bins

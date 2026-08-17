"""Archived Stage-2 path-aligned candidate reweighting.

Stage 1 remains frozen and generates a finite candidate set.  The critic uses
only deployable observations and candidate geometry.  It converts predicted
risk into a probability distribution with an empirical KL constraint, so
Stage 2 changes candidate mass without changing the Stage-1 generator.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from map_config import MAP_CONFIG, MAP_HALF_EXTENT, SAFETY_COST_CONFIG


STAGE2_REWEIGHTING_SEMANTICS = (
    "frozen_stage1_path_aligned_privileged_critic_kl_reweighting_v1"
)
STAGE2_CRITIC_CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class Stage2ReweightingConfig:
    candidates: int = 16
    kl_budget: float = 0.20
    critic_hidden_dim: int = 128
    map_channels: int = 4
    classification_score_weight: float = 1.0
    selection_mode: str = "sample"
    kl_solver_iterations: int = 80
    kl_solver_tolerance: float = 1e-8

    def __post_init__(self):
        if self.candidates <= 1:
            raise ValueError("Stage 2 requires at least two candidates")
        if self.kl_budget < 0.0:
            raise ValueError("KL budget must be non-negative")
        if self.critic_hidden_dim <= 0:
            raise ValueError("critic_hidden_dim must be positive")
        if self.critic_hidden_dim % 8 != 0:
            raise ValueError("critic_hidden_dim must be divisible by 8")
        if self.map_channels != 4:
            raise ValueError("The current critic requires the four-channel map contract")
        if self.selection_mode not in {"sample", "argmin"}:
            raise ValueError("selection_mode must be 'sample' or 'argmin'")
        if self.kl_solver_iterations <= 0:
            raise ValueError("kl_solver_iterations must be positive")
        if self.kl_solver_tolerance <= 0.0:
            raise ValueError("kl_solver_tolerance must be positive")

    @classmethod
    def from_dict(cls, values):
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in dict(values).items() if key in fields})

    def to_dict(self):
        return asdict(self)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def physical_path_grid(path: torch.Tensor) -> torch.Tensor:
    """Convert physical (x,y) cell centers to grid_sample coordinates.

    The last grid dimension is (x, y), matching width then height.  This is the
    same spatial contract used by the production privileged cost sampler.
    """
    path = torch.as_tensor(path)
    if path.shape[-1] != 2:
        raise ValueError(f"path must end in (x,y), got {tuple(path.shape)}")
    origin_x, origin_y = MAP_CONFIG.origin_xy
    resolution = float(MAP_CONFIG.resolution)
    width, height, _ = MAP_CONFIG.cost_map_size
    x_index = (path[..., 0] - float(origin_x)) / resolution - 0.5
    y_index = (path[..., 1] - float(origin_y)) / resolution - 0.5
    return torch.stack(
        [
            x_index / max(width - 1, 1) * 2.0 - 1.0,
            y_index / max(height - 1, 1) * 2.0 - 1.0,
        ],
        dim=-1,
    )


class PathAlignedCandidateCritic(nn.Module):
    """Predict candidate risk after aligning observed map features to paths."""

    def __init__(self, map_channels: int = 4, hidden_dim: int = 128):
        super().__init__()
        if map_channels != 4:
            raise ValueError("PathAlignedCandidateCritic requires four map channels")
        self.map_channels = int(map_channels)
        self.hidden_dim = int(hidden_dim)
        self.map_encoder = nn.Sequential(
            nn.Conv2d(map_channels, 32, 5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        # 64 encoded map + 4 raw map + 8 path geometry channels.
        self.path_encoder = nn.Sequential(
            nn.Conv1d(76, 96, 5, padding=2),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.Conv1d(96, hidden_dim, 5, padding=2),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
        )
        self.attention_pool = nn.Conv1d(hidden_dim, 1, 1)
        self.pose_encoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 64 * 2 + 32, 256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 96),
            nn.SiLU(),
        )
        self.risk_head = nn.Linear(96, 1)
        self.safe_head = nn.Linear(96, 1)

    @staticmethod
    def path_features(path: torch.Tensor, curvature: torch.Tensor) -> torch.Tensor:
        delta = torch.diff(path, dim=-2, prepend=path[..., :1, :])
        segment_length = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        tangent = delta / segment_length.clamp_min(1e-6)
        return torch.cat(
            [
                path / float(MAP_HALF_EXTENT),
                delta / float(MAP_CONFIG.resolution),
                tangent,
                segment_length / float(MAP_CONFIG.resolution),
                curvature.unsqueeze(-1)
                / float(SAFETY_COST_CONFIG.curvature_limit),
            ],
            dim=-1,
        )

    @staticmethod
    def sample_along(feature_map: torch.Tensor, path: torch.Tensor) -> torch.Tensor:
        if feature_map.ndim != 4:
            raise ValueError("feature_map must have shape (B,C,H,W)")
        if path.ndim != 4 or path.shape[-1] != 2:
            raise ValueError("path must have shape (B,K,N,2)")
        batch, candidates, points = path.shape[:3]
        if feature_map.shape[0] != batch:
            raise ValueError("map and path batch dimensions differ")
        grid = physical_path_grid(path).reshape(batch * candidates, points, 1, 2)
        expanded = feature_map.repeat_interleave(candidates, dim=0)
        sampled = F.grid_sample(
            expanded,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled[..., 0].reshape(
            batch, candidates, feature_map.shape[1], points
        )

    def forward(
        self,
        observed_map: torch.Tensor,
        start_condition: torch.Tensor,
        goal_condition: torch.Tensor,
        candidate_path: torch.Tensor,
        candidate_curvature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if candidate_path.ndim != 4 or candidate_path.shape[-1] != 2:
            raise ValueError("candidate_path must have shape (B,K,N,2)")
        if candidate_curvature.shape != candidate_path.shape[:-1]:
            raise ValueError("candidate_curvature must have shape (B,K,N)")
        batch, candidates, points = candidate_path.shape[:3]
        encoded = self.map_encoder(observed_map)
        aligned = self.sample_along(encoded, candidate_path)
        raw = self.sample_along(observed_map, candidate_path)
        geometry = self.path_features(
            candidate_path, candidate_curvature
        ).permute(0, 1, 3, 2)
        sequence = torch.cat([aligned, raw, geometry], dim=2).reshape(
            batch * candidates, 76, points
        )
        sequence = self.path_encoder(sequence)
        mean_pool = sequence.mean(dim=-1)
        max_pool = sequence.amax(dim=-1)
        attention = torch.softmax(self.attention_pool(sequence), dim=-1)
        attention_pool = (sequence * attention).sum(dim=-1)
        path_embedding = torch.cat(
            [mean_pool, max_pool, attention_pool], dim=-1
        ).reshape(batch, candidates, -1)
        global_map = torch.cat(
            [encoded.mean(dim=(-1, -2)), encoded.amax(dim=(-1, -2))], dim=-1
        )
        pose = self.pose_encoder(
            torch.cat([start_condition, goal_condition], dim=-1)
        )
        context = torch.cat([global_map, pose], dim=-1)[:, None].expand(
            -1, candidates, -1
        )
        hidden = self.fusion(torch.cat([path_embedding, context], dim=-1))
        predicted_risk = F.softplus(self.risk_head(hidden).squeeze(-1))
        safe_logit = self.safe_head(hidden).squeeze(-1)
        return predicted_risk, safe_logit


def critic_energy(
    predicted_risk: torch.Tensor,
    safe_logit: torch.Tensor,
    classification_score_weight: float = 1.0,
) -> torch.Tensor:
    if predicted_risk.shape != safe_logit.shape:
        raise ValueError("risk and safe logits must have identical shapes")
    return predicted_risk + float(classification_score_weight) * F.softplus(
        -safe_logit
    )


def empirical_kl_from_uniform(weights: torch.Tensor) -> torch.Tensor:
    if weights.ndim != 2:
        raise ValueError("weights must have shape (B,K)")
    candidates = weights.shape[1]
    return torch.sum(
        torch.where(
            weights > 0,
            weights * torch.log(weights.clamp_min(1e-30) * candidates),
            torch.zeros_like(weights),
        ),
        dim=1,
    )


def bounded_kl_weights(
    energy: torch.Tensor,
    budget: float = 0.20,
    *,
    iterations: int = 80,
    tolerance: float = 1e-8,
) -> torch.Tensor:
    """Return the most concentrated Boltzmann weights within a KL budget."""
    energy = torch.as_tensor(energy)
    squeeze = energy.ndim == 1
    if squeeze:
        energy = energy.unsqueeze(0)
    if energy.ndim != 2 or energy.shape[1] == 0:
        raise ValueError("energy must have shape (B,K) with K > 0")
    if not bool(torch.isfinite(energy).all()):
        raise ValueError("energy contains non-finite values")
    if budget < 0.0:
        raise ValueError("KL budget must be non-negative")
    if iterations <= 0 or tolerance <= 0.0:
        raise ValueError("invalid KL solver controls")

    batch, candidates = energy.shape
    uniform = torch.full_like(energy, 1.0 / candidates)
    if budget <= tolerance or candidates == 1:
        return uniform[0] if squeeze else uniform
    centered = energy - energy.amin(dim=1, keepdim=True)
    variable = centered.amax(dim=1) > tolerance
    if not bool(variable.any()):
        return uniform[0] if squeeze else uniform

    low = torch.zeros(batch, device=energy.device, dtype=energy.dtype)
    high = torch.ones_like(low)
    # Find a per-row upper inverse temperature that reaches the budget, unless
    # ties make the maximum attainable KL smaller than the requested budget.
    for _ in range(60):
        weights = torch.softmax(-high[:, None] * centered, dim=1)
        kl = empirical_kl_from_uniform(weights)
        grow = variable & (kl < float(budget) - tolerance) & (high < 1e12)
        if not bool(grow.any()):
            break
        high = torch.where(grow, high * 2.0, high)
    for _ in range(int(iterations)):
        middle = (low + high) * 0.5
        weights = torch.softmax(-middle[:, None] * centered, dim=1)
        kl = empirical_kl_from_uniform(weights)
        too_concentrated = kl > float(budget)
        high = torch.where(too_concentrated, middle, high)
        low = torch.where(too_concentrated, low, middle)
    weights = torch.softmax(-low[:, None] * centered, dim=1)
    weights = torch.where(variable[:, None], weights, uniform)
    return weights[0] if squeeze else weights


class ReweightedPathPlanner(nn.Module):
    """Compose a frozen Path MeanFlow generator with the Stage-2 critic."""

    def __init__(
        self,
        stage1: nn.Module,
        critic: PathAlignedCandidateCritic,
        config: Stage2ReweightingConfig | None = None,
    ):
        super().__init__()
        self.stage1 = stage1
        self.critic = critic
        self.config = config or Stage2ReweightingConfig()
        if int(getattr(stage1, "map_channels", -1)) != self.config.map_channels:
            raise ValueError("Stage 1 and critic map-channel contracts differ")
        for parameter in self.stage1.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def plan(
        self,
        observed_map: torch.Tensor,
        start_condition: torch.Tensor,
        goal_condition: torch.Tensor,
        *,
        source_noise: torch.Tensor | None = None,
        selection_mode: Literal["sample", "argmin"] | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Generate K candidates and return one deployable selected path."""
        self.stage1.eval()
        self.critic.eval()
        batch = observed_map.shape[0]
        candidates = int(self.config.candidates)
        expected_source_shape = (
            batch * candidates,
            int(self.stage1.num_edges),
            2,
        )
        if source_noise is None:
            source_noise = torch.randn(
                expected_source_shape,
                device=observed_map.device,
                dtype=observed_map.dtype,
                generator=generator,
            )
        else:
            source_noise = torch.as_tensor(
                source_noise,
                device=observed_map.device,
                dtype=observed_map.dtype,
            )
            if tuple(source_noise.shape) != expected_source_shape:
                raise ValueError(
                    f"source_noise must be {expected_source_shape}, got "
                    f"{tuple(source_noise.shape)}"
                )
        state = self.stage1.sample(
            observed_map,
            start_condition,
            goal_condition,
            num_samples=candidates,
            solver="pmf_onestep",
            source_noise=source_noise,
            return_residual=True,
        )
        start_k = start_condition.repeat_interleave(candidates, dim=0)
        goal_k = goal_condition.repeat_interleave(candidates, dim=0)
        geometry = self.stage1.evaluate_trajectory_state(state, start_k, goal_k)
        paths = geometry["position"].reshape(batch, candidates, -1, 2)
        curvature = geometry["curvature"].reshape(batch, candidates, -1)
        predicted_risk, safe_logit = self.critic(
            observed_map,
            start_condition,
            goal_condition,
            paths,
            curvature,
        )
        energy = critic_energy(
            predicted_risk,
            safe_logit,
            self.config.classification_score_weight,
        )
        weights = bounded_kl_weights(
            energy,
            self.config.kl_budget,
            iterations=self.config.kl_solver_iterations,
            tolerance=self.config.kl_solver_tolerance,
        )
        mode = selection_mode or self.config.selection_mode
        if mode == "sample":
            selected_index = torch.multinomial(
                weights, num_samples=1, replacement=True, generator=generator
            ).squeeze(1)
        elif mode == "argmin":
            selected_index = energy.argmin(dim=1)
        else:
            raise ValueError(f"Unknown selection mode: {mode}")
        flat_index = (
            torch.arange(batch, device=observed_map.device) * candidates
            + selected_index
        )
        return {
            "selected_index": selected_index,
            "selected_state": state[flat_index],
            "selected_path": paths[
                torch.arange(batch, device=observed_map.device), selected_index
            ],
            "selected_curvature": curvature[
                torch.arange(batch, device=observed_map.device), selected_index
            ],
            "candidate_state": state.reshape(batch, candidates, *state.shape[1:]),
            "candidate_path": paths,
            "candidate_curvature": curvature,
            "predicted_risk": predicted_risk,
            "safe_logit": safe_logit,
            "energy": energy,
            "weights": weights,
        }


def save_critic_checkpoint(
    path: str | Path,
    critic: PathAlignedCandidateCritic,
    config: Stage2ReweightingConfig,
    *,
    source_checkpoint: str | Path,
    optimizer_state_dict=None,
    **metadata,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_version": STAGE2_CRITIC_CHECKPOINT_VERSION,
            "stage": "stage2_path_aligned_reweighting",
            "stage2_reweighting_semantics": STAGE2_REWEIGHTING_SEMANTICS,
            "critic_state_dict": critic.state_dict(),
            # Compatibility with the admitted research checkpoint.
            "model_state_dict": critic.state_dict(),
            "critic_class": "PathAlignedCandidateCritic",
            "critic_config": config.to_dict(),
            "source_checkpoint": str(Path(source_checkpoint).resolve()),
            "source_checkpoint_sha256": sha256_file(source_checkpoint),
            "optimizer_state_dict": optimizer_state_dict,
            **metadata,
        },
        path,
    )


def load_critic_checkpoint(
    path: str | Path,
    device: torch.device | str,
    *,
    expected_source_checkpoint: str | Path | None = None,
) -> tuple[PathAlignedCandidateCritic, Stage2ReweightingConfig, dict]:
    checkpoint = torch.load(path, map_location=device)
    semantics = checkpoint.get("stage2_reweighting_semantics")
    if semantics not in {None, STAGE2_REWEIGHTING_SEMANTICS}:
        raise ValueError("Stage-2 critic semantic version mismatch")
    config = Stage2ReweightingConfig.from_dict(
        checkpoint.get("critic_config", {})
    )
    critic = PathAlignedCandidateCritic(
        map_channels=config.map_channels,
        hidden_dim=config.critic_hidden_dim,
    ).to(device)
    state_dict = checkpoint.get("critic_state_dict") or checkpoint.get(
        "model_state_dict"
    )
    if state_dict is None:
        raise ValueError("Stage-2 critic checkpoint lacks model weights")
    critic.load_state_dict(state_dict, strict=True)
    if expected_source_checkpoint is not None:
        expected_hash = sha256_file(expected_source_checkpoint)
        actual_hash = checkpoint.get("source_checkpoint_sha256")
        if actual_hash != expected_hash:
            raise ValueError("Stage-2 critic and Stage-1 checkpoint differ")
    return critic, config, checkpoint


def load_reweighted_planner(
    stage1_checkpoint: str | Path,
    critic_checkpoint: str | Path,
    device: torch.device | str,
) -> ReweightedPathPlanner:
    # Runtime import avoids coupling the pure critic module to training code.
    from posterior_pipeline import (
        _require_current_mask_semantics,
        _require_demo_target_semantics,
        _require_main_method_model,
        load_model,
    )

    stage1, _, checkpoint = load_model(Path(stage1_checkpoint), torch.device(device))
    _require_main_method_model(stage1)
    _require_current_mask_semantics(checkpoint)
    _require_demo_target_semantics(checkpoint)
    critic, config, _ = load_critic_checkpoint(
        critic_checkpoint,
        device,
        expected_source_checkpoint=stage1_checkpoint,
    )
    planner = ReweightedPathPlanner(stage1, critic, config).to(device)
    planner.eval()
    return planner

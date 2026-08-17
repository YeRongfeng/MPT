"""Deployable path-aligned stability critic used by Stage 2.

The critic receives only the observed terrain, endpoint conditions and a
candidate path. Privileged full-terrain stability is a training label and is
never part of the forward input.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from map_config import MAP_CONFIG, MAP_HALF_EXTENT, SAFETY_COST_CONFIG


STAGE2_CRITIC_CHECKPOINT_VERSION = 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def physical_path_grid(path: torch.Tensor) -> torch.Tensor:
    """Convert physical (x, y) cell centers to grid_sample coordinates."""
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
    """Predict stability probability and conditional severity per candidate."""

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


def load_critic_checkpoint(path: str | Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("critic_config", {})
    critic = PathAlignedCandidateCritic(
        map_channels=int(config.get("map_channels", 4)),
        hidden_dim=int(config.get("hidden_dim", 128)),
    ).to(device)
    state = checkpoint.get("critic_state_dict", checkpoint.get("model_state_dict"))
    if state is None:
        raise KeyError(f"Critic checkpoint has no state dict: {path}")
    critic.load_state_dict(state, strict=True)
    critic.eval()
    return critic, checkpoint

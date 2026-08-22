"""Run a saved Path MeanFlow checkpoint on the shared baseline tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from dataLoader_dit import MASK_NOISE_SEED_OFFSET, build_masked_normal_input
from map_config import DENSE_TRAJECTORY_POINTS
from posterior_pipeline import load_model


_MODEL_CACHE: Dict[str, Any] = {}


def load_path_meanflow(checkpoint: str, device: Optional[str] = None):
    checkpoint = str(Path(checkpoint))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    key = f"{checkpoint}::{device}"
    if key not in _MODEL_CACHE:
        model, _, ckpt = load_model(checkpoint, torch.device(device))
        model.eval()
        _MODEL_CACHE[key] = (model, ckpt, torch.device(device))
    return _MODEL_CACHE[key]


def plan_path_meanflow(
    task,
    *,
    checkpoint: str,
    source_seed: int,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    model, ckpt, torch_device = load_path_meanflow(checkpoint, device)
    normals = np.asarray(task.normals, dtype=np.float32)
    map_input = torch.from_numpy(
        build_masked_normal_input(
            normals,
            task.mask,
            int(source_seed) + int(task.path_id) * 1_000_003 + MASK_NOISE_SEED_OFFSET,
        )
    ).permute(2, 0, 1).float().unsqueeze(0).to(torch_device)

    start = torch.as_tensor(task.start, dtype=torch.float32, device=torch_device)
    goal = torch.as_tensor(task.goal, dtype=torch.float32, device=torch_device)
    scale = float(model.coordinate_scale)

    def _pose4(pose):
        out = torch.zeros(1, 4, device=torch_device)
        out[0, :2] = torch.clamp(pose[:2] / scale, -1.0, 1.0)
        out[0, 2] = torch.cos(pose[2])
        out[0, 3] = torch.sin(pose[2])
        return out

    generator = torch.Generator(device=torch_device)
    generator.manual_seed(int(source_seed) + int(task.path_id) * 1_000_003)
    noise = torch.randn(
        1,
        model.num_edges,
        2,
        generator=generator,
        device=torch_device,
    )
    with torch.no_grad():
        state = model.sample(
            map_input,
            _pose4(start),
            _pose4(goal),
            num_samples=1,
            num_steps=1,
            solver="pmf_onestep",
            reconstruct_trajectory=False,
            num_traj_points=DENSE_TRAJECTORY_POINTS,
            source_noise=noise,
            return_residual=True,
        )
        geometry = model.evaluate_trajectory_state(
            state, _pose4(start), _pose4(goal)
        )
    xy = geometry["position"][0].detach().cpu().numpy()
    yaw = geometry["yaw"][0].detach().cpu().numpy()
    path = np.concatenate([xy, yaw[:, None]], axis=1).astype(np.float32)
    return {
        "found": True,
        "path": path,
        "curvature": geometry["curvature"][0].detach().cpu().numpy(),
        "expansions": 1,
        "failure_reason": None,
        "input": "partial_masked_normals",
        "checkpoint": checkpoint,
        "checkpoint_stage": ckpt.get("stage"),
    }

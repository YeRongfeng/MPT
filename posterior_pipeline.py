"""Boundary-Constrained Path MeanFlow 两阶段训练流水线。

Stage 1 执行 Conditional Path MeanFlow Pretraining。
Stage 2 从 Stage 1 checkpoint 继续训练，并在部署点直接反向传播
privileged task cost；source-transport/critic 路径仅作为历史实验代码保留。

完整地图、朝向相关稳定性场和路径修正器只在 Stage 2 数据构造时出现；
推理时模型仍只接收部分观测地图、显式 mask、起终点和源噪声。
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call, jvp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
    WeightedRandomSampler,
)
from tqdm import tqdm

from dataLoader_dit import (
    MASK_GENERATION_SEMANTICS,
    MASK_INPUT_SEMANTICS,
    MASK_NOISE_STD,
    STAGE2_MAX_CONTIGUOUS_BLOCKED_FRACTION,
    STAGE2_MAX_DEMO_BLOCKED_FRACTION,
    STAGE2_MAX_MASKED_FRACTION,
    UnevenPathDataLoader,
    dense_demo_bspline,
    mask_start_goal_connected,
    rasterize_trajectory,
)
from dit.Models import PathMeanFlowTransformer
from boundary_constrained_path import (
    DEMONSTRATION_FIT_SEMANTIC_VERSION,
    PATH_REPRESENTATION_SEMANTIC_VERSION,
)
from grad_optimizer import (
    PRIVILEGED_COST_SEMANTICS,
    apply_privileged_path_correction,
    build_signed_mask_distance_map,
    endpoint_stability_feasibility,
    privileged_cost_contract,
    privileged_optimizer_scale_diagnostics,
    privileged_planning_cost,
    trajectory_validity_metrics,
)
from map_config import (
    DENSE_TRAJECTORY_POINTS,
    MAP_CONFIG,
    MAP_HALF_EXTENT,
    SAFETY_COST_CONFIG,
    discover_environments,
)
from mgda_utils import solve_mgda_active_set

# Stage 1 示范路径采用规范参数化、一阶边界约束和确定性平滑候选拟合。
# token 与 target 构造绑定；拟合规则改变时必须阻止旧 checkpoint 续训。
DEMONSTRATION_PATH_COORDINATE_SEMANTICS = (
    DEMONSTRATION_FIT_SEMANTIC_VERSION
)
DEMO_TARGET_SEMANTICS = DEMONSTRATION_PATH_COORDINATE_SEMANTICS
STAGE1_ENDPOINT_CURVATURE_OBJECTIVE = (
    "deployment_endpoint_audit1001_saturating_top_tail_relative_excess_v2"
)
# Stage 2 的 44D 表示固定起终点位置和 yaw；因此上下文筛选只按
# exact endpoint yaw 检查 stability hard threshold。“允许 yaw 区间”只能作为
# 显式的任务可行性审计，不能与当前表示可行性混为一谈。
STAGE2_CONTEXT_SEMANTICS = (
    "resample_fixed_analytic_exact_endpoint_yaw_stability_v3"
)
DIRECT_COST_STAGE2_SEMANTICS = (
    "direct_onestep_stable_yaw_log1p_curvature_no_anchor_no_replay_v4"
)
DIRECT_COST_CONTEXT_SEMANTICS = (
    "independent_stage2_train_root_validation_root_mask_no_expert_filtering_v1"
)
DIRECT_COST_TRAINING_PROTOCOL_SEMANTICS = (
    "d0_task_cost_first_independent_validation_root_stable_yaw_curvature_v1"
)


def _make_tensorboard_writer(output, stage, *, purge_step=None):
    """为两个训练阶段创建相互独立、支持续训的 TensorBoard 日志目录。"""
    log_dir = Path(output) / "tensorboard" / stage
    log_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {"log_dir": str(log_dir), "flush_secs": 30}
    if purge_step is not None:
        kwargs["purge_step"] = int(purge_step)
    return SummaryWriter(**kwargs)


def _log_numeric_tree(writer, prefix, values, step):
    """递归记录字典中的有限标量，忽略列表、文本和 NaN。"""
    if values is None:
        return
    if isinstance(values, dict):
        for key, value in values.items():
            _log_numeric_tree(
                writer,
                f"{prefix}/{key}" if prefix else str(key),
                value,
                step,
            )
        return
    if isinstance(values, torch.Tensor) and values.numel() == 1:
        values = values.detach().cpu().item()
    if isinstance(values, (bool, int, float, np.number)):
        value = float(values)
        if np.isfinite(value):
            writer.add_scalar(prefix, value, int(step))


def _direct_cost_distribution_summary(values):
    """Summarize a detached cost distribution for epoch-level telemetry."""
    values = torch.as_tensor(values, dtype=torch.float32).flatten().cpu()
    if values.numel() == 0:
        return {}
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "median": float(torch.quantile(values, 0.50)),
        "p10": float(torch.quantile(values, 0.10)),
        "p90": float(torch.quantile(values, 0.90)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _direct_cost_epoch_statistics(
    source_samples,
    condition_samples,
    environment_condition_samples,
    environment_names,
):
    """Build source-, condition-, and environment-level epoch summaries."""
    statistics = {}
    for metric_name in source_samples:
        source_values = torch.cat(source_samples[metric_name])
        condition_values = torch.cat(condition_samples[metric_name])
        environment_means = []
        environment_detail = {}
        for environment_index, values in environment_condition_samples[
            metric_name
        ].items():
            if not values:
                continue
            environment_values = torch.as_tensor(
                values, dtype=torch.float32
            )
            environment_means.append(environment_values.mean())
            if metric_name == "task_cost":
                environment_name = str(environment_names[environment_index])
                environment_detail[environment_name] = {
                    "task_cost": _direct_cost_distribution_summary(
                        environment_values
                    )
                }
        statistics[metric_name] = {
            "source": _direct_cost_distribution_summary(source_values),
            "condition": _direct_cost_distribution_summary(condition_values),
            "environment": _direct_cost_distribution_summary(
                torch.stack(environment_means)
                if environment_means
                else torch.empty(0)
            ),
        }
        if metric_name == "task_cost":
            statistics["environment_detail"] = environment_detail
    return statistics


def _log_stage1_update_metrics(
    writer,
    *,
    loss,
    terms,
    gradient_norm,
    learning_rate,
    global_update,
):
    """Expose essential Stage 1 telemetry before the first epoch finishes."""
    _log_numeric_tree(
        writer,
        "stage1",
        {
            "loss": {
                "train_step": loss,
                "flow_step": terms.get("flow"),
                "endpoint_step": terms.get("endpoint"),
                "curvature_aux_step": terms.get("curvature_aux"),
                "curvature_total_step": terms.get("curvature_weighted"),
            },
            "optimization": {
                "gradient_norm_step": gradient_norm,
                "learning_rate_step": learning_rate,
            },
        },
        global_update,
    )


@dataclass
class PrivilegedConstraintDistillationConfig:
    """Stage 2 迭代特权约束蒸馏与修正数据聚合配置。"""

    rounds: int = 20
    contexts_per_round: int = 64
    particles_per_context: int = 8
    privileged_steps: int = 80
    privileged_lr: float = 5e-2
    proposal_weight: float = 2e-3
    buffer_size: int = 8192
    buffer_recent_rounds: int = 3
    buffer_recent_fraction: float = 0.6
    buffer_priority_fraction: float = 0.2
    updates_per_round: int = 100
    current_round_fraction: float = 0.5


def default_model_args():
    """返回真正改变模型结构或输入语义的最小参数集合。"""
    return {
        "n_layers": 6,
        "n_heads": 8,
        "d_model": 512,
        "d_inner": 1024,
        "dropout": 0.1,
        "coordinate_scale": MAP_HALF_EXTENT,
        "map_channels": 4,
        "use_radial_output": False,
    }


MODEL_ARG_KEYS = PathMeanFlowTransformer.CONFIG_KEYS

# On-disk compatibility name used by existing checkpoints.
REPRESENTATION_SEMANTIC_VERSION = PATH_REPRESENTATION_SEMANTIC_VERSION

# Compatibility alias for older experiment imports.
DaggerConfig = PrivilegedConstraintDistillationConfig


def normalize_poses(start_pose, goal_pose, coordinate_scale, device):
    """将物理位姿 ``(x,y,yaw)`` 转成模型条件 ``(x,y,cos,sin)``。"""
    start_pose = torch.as_tensor(start_pose, dtype=torch.float32, device=device)
    goal_pose = torch.as_tensor(goal_pose, dtype=torch.float32, device=device)
    if start_pose.ndim == 1:
        start_pose = start_pose.unsqueeze(0)
    if goal_pose.ndim == 1:
        goal_pose = goal_pose.unsqueeze(0)

    start = torch.zeros(start_pose.shape[0], 4, device=device)
    goal = torch.zeros(goal_pose.shape[0], 4, device=device)
    start[:, :2] = start_pose[:, :2] / float(coordinate_scale)
    goal[:, :2] = goal_pose[:, :2] / float(coordinate_scale)
    start[:, 2:] = torch.stack(
        [torch.cos(start_pose[:, 2]), torch.sin(start_pose[:, 2])], dim=-1
    )
    goal[:, 2:] = torch.stack(
        [torch.cos(goal_pose[:, 2]), torch.sin(goal_pose[:, 2])], dim=-1
    )
    return start, goal


def path_coordinates_from_batch(batch, device=None):
    """Read free path coordinates with legacy dataset-key compatibility."""
    key = (
        "path_coordinates"
        if "path_coordinates" in batch
        else "trajectory_state_44d"
    )
    if key not in batch:
        raise KeyError(
            "Dataset did not provide path_coordinates; enable canonical "
            "path-coordinate encoding."
        )
    coordinates = batch[key].float()
    return coordinates if device is None else coordinates.to(device)


def make_partial_dataset(
    data_folder,
    split,
    *,
    compute_stability_map=False,
    mask_seed=2026,
    p_mask=0.5,
    mask_mode="stage1_demo_valid",
    vehicle_radius_meters=SAFETY_COST_CONFIG.vehicle_radius_meters,
    environment_names=None,
    dynamic_mask_noise=None,
    expected_environment_count=MAP_CONFIG.expected_environments,
    compute_stability_if_missing=False,
):
    """创建部分观测数据集。

    网络输入固定为 ``[normal_x, normal_y, normal_z, mask]``。
    第四通道 ``mask`` 的统一语义是 1=允许进入、0=禁止进入；它同时合并
    不完整观测和实体障碍。``cost_map`` 仅在 Stage 2 特权纠正时加载。
    """
    split_folder = os.path.join(data_folder, split)
    available_envs = discover_environments(
        split_folder,
        expected_count=expected_environment_count,
    )
    if environment_names is None:
        envs = available_envs
    else:
        envs = [str(name) for name in environment_names]
        if len(envs) != len(set(envs)):
            raise ValueError("environment_names contains duplicates")
        missing = sorted(set(envs) - set(available_envs))
        if missing:
            raise ValueError(
                f"Requested environments are absent from {split_folder}: {missing}"
            )
        if not envs:
            raise ValueError("environment_names must not be empty")
    dataset = UnevenPathDataLoader(
        envs,
        split_folder,
        compute_stability_map=compute_stability_map,
        use_precomputed_stability=compute_stability_map,
        compute_stability_if_missing=compute_stability_if_missing,
        partial_observation=True,
        include_mask=True,
        mask_seed=mask_seed,
        p_mask=p_mask,
        # 训练时每次访问动态重采样遮挡区噪声；验证时固定。
        dynamic_mask_noise=(
            split == "train"
            if dynamic_mask_noise is None
            else bool(dynamic_mask_noise)
        ),
        mask_mode=mask_mode,
        vehicle_radius_meters=vehicle_radius_meters,
        encode_path_coordinates=True,
    )
    return dataset, envs


def frozen_stage1_environment_split(
    environments,
    *,
    seed,
    train_count,
    validation_count,
):
    """Create a deterministic, disjoint Stage-1 terrain split."""
    names = sorted(str(name) for name in environments)
    if len(names) != len(set(names)):
        raise ValueError("environments contains duplicates")
    train_count = int(train_count)
    validation_count = int(validation_count)
    if train_count <= 0 or validation_count <= 0:
        raise ValueError("Stage-1 train/validation environment counts must be positive")
    if train_count + validation_count > len(names):
        raise ValueError(
            "Stage-1 environment split requests "
            f"{train_count + validation_count}/{len(names)} environments"
        )
    shuffled = names.copy()
    random.Random(int(seed)).shuffle(shuffled)
    train = sorted(shuffled[:train_count])
    validation = sorted(
        shuffled[train_count : train_count + validation_count]
    )
    if set(train) & set(validation):
        raise AssertionError("Stage-1 environment split is not disjoint")
    return {
        "semantic_version": "stage1_physical_environment_disjoint_v1",
        "seed": int(seed),
        "train": train,
        "validation": validation,
        "unused": sorted(set(names) - set(train) - set(validation)),
    }


def stage1_environment_selection(args):
    """Resolve explicit environment-disjoint Stage-1 datasets from CLI args."""
    train_available = discover_environments(
        Path(args.dataFolder) / "train",
        expected_count=MAP_CONFIG.expected_environments,
    )
    val_available = discover_environments(
        Path(args.dataFolder) / "val",
        expected_count=MAP_CONFIG.expected_environments,
    )
    if train_available != val_available:
        raise ValueError("Stage-1 train/val environment names differ")
    split_seed = getattr(args, "stage1_split_seed", None)
    if split_seed is None:
        return {
            "semantic_version": "legacy_same_environment_train_val_v1",
            "seed": None,
            "train": train_available,
            "validation": val_available,
            "unused": [],
        }
    return frozen_stage1_environment_split(
        train_available,
        seed=split_seed,
        train_count=args.stage1_train_environments,
        validation_count=args.stage1_val_environments,
    )


def stage2_environment_selection(args):
    """Freeze full train-root and independent validation-root environments."""
    train_available = discover_environments(
        Path(args.dataFolder) / "train",
        expected_count=MAP_CONFIG.expected_environments,
    )
    validation_data = Path(
        getattr(args, "stage2_validation_data", "data/dataset1_val")
    )
    validation_split = str(
        getattr(args, "stage2_validation_split", "train")
    )
    validation_available = discover_environments(
        validation_data / validation_split,
        expected_count=None,
    )

    def order_key(stream, environment):
        payload = (
            f"{int(args.stage2_split_seed)}|{stream}|{environment}"
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"), environment

    ordered_train = sorted(
        train_available,
        key=lambda environment: order_key("train", environment),
    )
    ordered_validation = sorted(
        validation_available,
        key=lambda environment: order_key("validation", environment),
    )
    train_end = int(args.stage2_train_environments)
    validation_count = int(args.stage2_validation_environments)
    if train_end > len(ordered_train):
        raise ValueError(
            "Stage-2 train requests "
            f"{train_end}/{len(ordered_train)} environments"
        )
    if validation_count > len(ordered_validation):
        raise ValueError(
            "Stage-2 validation requests "
            f"{validation_count}/{len(ordered_validation)} environments "
            f"from {validation_data / validation_split}"
        )
    return {
        "semantic_version": "stage2_independent_train_validation_roots_v1",
        "seed": int(args.stage2_split_seed),
        "train_data_root": str(Path(args.dataFolder).resolve()),
        "train_split": "train",
        "validation_data_root": str(validation_data.resolve()),
        "validation_split": validation_split,
        "train": sorted(ordered_train[:train_end]),
        "validation": sorted(ordered_validation[:validation_count]),
        "unused": sorted(ordered_train[train_end:]),
        "validation_unused": sorted(ordered_validation[validation_count:]),
    }


def fixed_validation_indices_per_environment(
    dataset,
    environments,
    *,
    count,
    seed,
):
    """Select one deterministic validation path from each physical terrain."""
    by_environment = {str(environment): [] for environment in environments}
    for dataset_index in range(len(dataset)):
        environment_index, _ = dataset.indexDict[int(dataset_index)]
        environment = dataset.env_list[environment_index]
        if environment in by_environment:
            by_environment[environment].append(int(dataset_index))
    selected_environments = sorted(environments)[: int(count)]
    result = []
    for environment in selected_environments:
        candidates = by_environment[environment]
        if not candidates:
            raise ValueError(f"Stage-2 validation terrain has no paths: {environment}")
        generator = random.Random(f"{int(seed)}:{environment}:stage2-validation")
        result.append(candidates[generator.randrange(len(candidates))])
    if len(result) != int(count):
        raise ValueError(
            f"Requested {count} validation terrains, selected {len(result)}"
        )
    return result


def save_checkpoint(path, model, model_args, **metadata):
    """保存模型、精简后的模型参数和训练状态。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_args": model_args,
            **metadata,
        },
        path,
    )


def load_model(checkpoint_path, device):
    """只加载与当前 44D 表示语义完全一致的 checkpoint。"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    actual = checkpoint.get("representation_semantic_version")
    if actual != REPRESENTATION_SEMANTIC_VERSION:
        raise ValueError(
            "Checkpoint 轨迹表示与当前 44D 主线不兼容："
            f"期望 {REPRESENTATION_SEMANTIC_VERSION!r}，实际 {actual!r}。"
            "旧 Stage 1/Stage 2 checkpoint 已作废，请从头训练 Stage 1。"
        )
    stored_args = checkpoint.get("model_args", {})
    model_args = {
        **default_model_args(),
        **{key: value for key, value in stored_args.items() if key in MODEL_ARG_KEYS},
    }
    model = PathMeanFlowTransformer(**model_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, model_args, checkpoint


def _require_main_method_model(model):
    """阻止三通道或 radial 基线被误用于两阶段主方法。"""
    if model.map_channels != 4 or model.use_radial_output:
        raise ValueError(
            "两阶段主方法要求 map_channels=4 且 use_radial_output=False；"
            "旧三通道/radial checkpoint 只能作为基线。"
        )


def _require_current_mask_semantics(checkpoint):
    """旧 mask/固定平地填充 checkpoint 不能静默用于当前输入分布。"""
    actual = checkpoint.get("input_mask_semantics")
    if actual != MASK_INPUT_SEMANTICS:
        raise ValueError(
            "Checkpoint 的 mask 输入语义与当前实现不兼容："
            f"期望 {MASK_INPUT_SEMANTICS!r}，实际 {actual!r}。"
            "请从头重新训练 Stage 1。"
        )


def _require_demo_target_semantics(checkpoint):
    """禁止把旧的无正则示范拟合目标混入当前训练主线。"""
    actual = checkpoint.get("demo_target_semantics")
    if actual != DEMO_TARGET_SEMANTICS:
        raise ValueError(
            "Checkpoint 的示范 B 样条目标语义与当前实现不兼容："
            f"期望 {DEMO_TARGET_SEMANTICS!r}，实际 {actual!r}。"
            "示范拟合规则已改变，请从头重新训练 Stage 1。"
        )


def _require_mask_configuration(checkpoint, args, *, p_mask=None):
    """恢复训练时保证 mask 形状分布与 checkpoint 完全一致。"""
    stored_seed = checkpoint.get("mask_seed")
    stored_probability = checkpoint.get("p_mask")
    stored_radius = checkpoint.get("vehicle_radius_meters")
    expected_probability = args.p_mask if p_mask is None else p_mask
    if (
        stored_seed != args.mask_seed
        or stored_probability is None
        or not np.isclose(float(stored_probability), expected_probability)
        or stored_radius is None
        or not np.isclose(float(stored_radius), args.vehicle_radius_meters)
    ):
        raise ValueError(
            "Checkpoint 的 mask 配置与当前命令不一致："
            f"checkpoint(seed={stored_seed}, p={stored_probability}, "
            f"radius={stored_radius})，current(seed={args.mask_seed}, "
            f"p={expected_probability}, radius={args.vehicle_radius_meters})。"
        )


def _require_privileged_cost_semantics(checkpoint):
    """Stage 2 resume 不允许混用旧距离场、稠密点数或接受标准。"""
    actual = checkpoint.get("privileged_cost_semantics")
    if actual != PRIVILEGED_COST_SEMANTICS:
        raise ValueError(
            "Stage 2 checkpoint 的特权 cost 语义已过期："
            f"期望 {PRIVILEGED_COST_SEMANTICS!r}，实际 {actual!r}。"
            "请从新的 Stage 1 checkpoint 重新启动 Stage 2。"
        )


def _require_stage2_mask_generation_semantics(checkpoint):
    """避免旧 replay 用新算法重建出不同的 Stage 2 mask。"""
    actual = checkpoint.get("mask_generation_semantics")
    if actual != MASK_GENERATION_SEMANTICS:
        raise ValueError(
            "Stage 2 checkpoint 的 mask 生成语义已过期："
            f"期望 {MASK_GENERATION_SEMANTICS!r}，实际 {actual!r}。"
            "局部干预 mask 规则已改变，请从 Stage 1 重新启动 Stage 2。"
        )


def _require_stage2_context_semantics(checkpoint):
    """禁止恢复仍把端点稳定性不可行条件交给专家的旧 Stage 2。"""
    actual = checkpoint.get("stage2_context_semantics")
    if actual != STAGE2_CONTEXT_SEMANTICS:
        raise ValueError(
            "Stage 2 checkpoint 的任务条件筛选语义已过期："
            f"期望 {STAGE2_CONTEXT_SEMANTICS!r}，实际 {actual!r}。"
            "请从 Stage 1 checkpoint 重新启动 Stage 2。"
        )


def demonstration_target_path_coordinates(
    model,
    batch,
    device,
    *,
    return_curvature_validity=False,
):
    """读取 44D 示范路径坐标，并用解析曲率执行数据准入。"""
    target = path_coordinates_from_batch(batch, device)
    if target.ndim != 3 or target.shape[1:] != (model.num_edges, 2):
        raise ValueError(
            f"Expected demonstration state (B,{model.num_edges},2), got "
            f"{tuple(target.shape)}"
        )
    if not return_curvature_validity:
        return target

    start_pose = batch["start_pose"].float().to(device)
    goal_pose = batch["goal_pose"].float().to(device)
    representation = model.trajectory_representation
    geometry = representation.evaluate(
        target, start_pose, goal_pose
    )
    representation.assert_endpoint_yaw(
        target, start_pose, goal_pose
    )
    max_curvature = representation.audit_curvature(
        target, start_pose, goal_pose
    ).amax(dim=1)
    curvature_valid = (
        max_curvature
        <= (
            SAFETY_COST_CONFIG.curvature_limit
            + SAFETY_COST_CONFIG.hard_constraint_epsilon
        )
    )
    return target, curvature_valid, max_curvature


# Compatibility alias for earlier experiment imports.
demonstration_target_residual = demonstration_target_path_coordinates


def stage1_endpoint_curvature_auxiliary(
    model,
    endpoint_state,
    start_condition,
    goal_condition,
    *,
    tail_ratio=0.01,
):
    """Robust hard-curvature surrogate on the deployed endpoint output."""
    tail_ratio = float(tail_ratio)
    if not 0.0 < tail_ratio <= 1.0:
        raise ValueError("tail_ratio must be in (0,1]")
    audit_curvature = model.audit_trajectory_state_curvature(
        endpoint_state,
        start_condition,
        goal_condition,
    )
    relative_excess = torch.relu(
        audit_curvature / float(SAFETY_COST_CONFIG.curvature_limit) - 1.0
    )
    # Bounded barrier: preserve an O(1) slope near the hard boundary while
    # preventing highly curved random initial outputs from dominating Stage 1.
    robust_excess = relative_excess / (1.0 + relative_excess)
    tail_count = max(
        1,
        int(math.ceil(robust_excess.shape[1] * tail_ratio)),
    )
    penalty = torch.topk(
        robust_excess,
        k=tail_count,
        dim=1,
        largest=True,
        sorted=False,
    ).values.mean()
    pass_rate = (
        audit_curvature.amax(dim=1)
        <= float(SAFETY_COST_CONFIG.curvature_limit)
        + float(SAFETY_COST_CONFIG.hard_constraint_epsilon)
    ).float().mean()
    return penalty, pass_rate, audit_curvature


def meanflow_transport_loss(
    model,
    map_input,
    start_pose,
    goal_pose,
    source,
    target,
    device,
    *,
    endpoint_probability=0.25,
    generator=None,
    endpoint_curvature_weight=0.0,
    endpoint_curvature_tail_ratio=0.01,
):
    """计算显式 source-target 配对的 Path MeanFlow 目标。

    ``target`` 对应 t=0，``source`` 对应 t=1。Stage 2 只调用这个监督目标，
    不混入 privileged cost loss。
    """
    map_input = map_input.float().to(device)
    source = model.project_zero_sum(source.float().to(device))
    target = model.project_zero_sum(target.float().to(device))
    start, goal = normalize_poses(
        start_pose, goal_pose, model.coordinate_scale, device
    )
    batch_size = source.shape[0]
    t1 = model.sample_timesteps(batch_size, device, generator=generator)
    t2 = model.sample_timesteps(batch_size, device, generator=generator)
    t = torch.maximum(t1, t2)
    r = torch.minimum(t1, t2)

    # 一部分样本直接覆盖部署点 t=1,r=0，降低单步推理误差。
    endpoint = (
        torch.rand(batch_size, device=device, generator=generator)
        < endpoint_probability
    )
    t = torch.where(endpoint, torch.ones_like(t), t).requires_grad_(True)
    r = torch.where(endpoint, torch.zeros_like(r), r)
    target_v = model.project_zero_sum(source - target)
    z_t = model.project_zero_sum(
        (1.0 - t[:, None, None]) * target + t[:, None, None] * source
    )

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    def average_velocity(z_arg, t_arg, r_arg):
        output = functional_call(
            model,
            {**params, **buffers},
            (map_input, z_arg, t_arg, r_arg, start, goal),
        )
        output = model.project_zero_sum(output)
        return model.project_zero_sum(
            (z_arg - output) / (t_arg[:, None, None] + 1e-5)
        )

    was_training = model.training
    # JVP 内禁用 dropout/BN 状态更新，但保留参数梯度。
    model.eval()
    try:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_mem_efficient=False,
            enable_math=True,
        ):
            velocity, derivative = jvp(
                average_velocity,
                (z_t, t, r),
                (target_v, torch.ones_like(t), torch.zeros_like(r)),
            )
    finally:
        model.train(was_training)

    mean_velocity = model.project_zero_sum(
        velocity
        + (t - r)[:, None, None] * derivative.detach().clamp(-5.0, 5.0)
    )
    flow_loss = (mean_velocity - target_v).square().mean()

    one = torch.ones(batch_size, device=device)
    zero = torch.zeros(batch_size, device=device)
    endpoint_pred = model(map_input, source, one, zero, start, goal)
    endpoint_loss = (
        model.project_zero_sum(endpoint_pred) - target
    ).square().mean()
    curvature_weight = float(endpoint_curvature_weight)
    if curvature_weight < 0.0:
        raise ValueError("endpoint_curvature_weight must be non-negative")
    tail_ratio = float(endpoint_curvature_tail_ratio)
    if not 0.0 < tail_ratio <= 1.0:
        raise ValueError("endpoint_curvature_tail_ratio must be in (0,1]")
    if curvature_weight > 0.0:
        (
            curvature_aux,
            curvature_pass_rate,
            _,
        ) = stage1_endpoint_curvature_auxiliary(
            model,
            endpoint_pred,
            start,
            goal,
            tail_ratio=tail_ratio,
        )
    else:
        curvature_aux = endpoint_loss.new_zeros(())
        curvature_pass_rate = endpoint_loss.new_full((), float("nan"))
    curvature_weighted = curvature_aux * curvature_weight
    return flow_loss + 0.25 * endpoint_loss + curvature_weighted, {
        "flow": flow_loss.detach(),
        "endpoint": endpoint_loss.detach(),
        "curvature_aux": curvature_aux.detach(),
        "curvature_weighted": curvature_weighted.detach(),
        "endpoint_curvature_pass_rate": curvature_pass_rate.detach(),
    }


def prior_transport_loss(
    model,
    batch,
    device,
    generator=None,
    *,
    endpoint_curvature_weight=0.0,
    endpoint_curvature_tail_ratio=0.01,
):
    """Stage 1：用全部示范监督 MeanFlow，并显式报告拟合违规。

    Curvature is a property of the fitted target, not a learnability label.
    An unresolved fit remains in the supervised data and is counted here;
    silently deleting it would turn a representation diagnostic into a
    target-dependent training-distribution filter.
    """
    target, curvature_valid, max_curvature = demonstration_target_residual(
        model,
        batch,
        device,
        return_curvature_validity=True,
    )
    invalid = int((~curvature_valid).sum().detach().cpu())
    source = model.project_zero_sum(
        torch.randn(
            target.shape,
            dtype=target.dtype,
            device=device,
            generator=generator,
        )
    )
    loss, terms = meanflow_transport_loss(
        model,
        batch["map"],
        batch["start_pose"],
        batch["goal_pose"],
        source,
        target,
        device,
        endpoint_probability=0.25,
        generator=generator,
        endpoint_curvature_weight=endpoint_curvature_weight,
        endpoint_curvature_tail_ratio=endpoint_curvature_tail_ratio,
    )
    terms["invalid_demo_curvature"] = invalid
    terms["skipped_demo_curvature"] = 0
    terms["used_examples"] = int(target.shape[0])
    terms["demo_max_curvature"] = float(max_curvature.max().detach().cpu())
    return loss, terms


@torch.no_grad()
def stage1_representation_diagnostics(model, batch, device, *, seed):
    """Smoke telemetry for free path coordinates and analytic invariants."""
    target = path_coordinates_from_batch(batch, device)
    map_input = batch["map"].float().to(device)
    start, goal = normalize_poses(
        batch["start_pose"],
        batch["goal_pose"],
        model.coordinate_scale,
        device,
    )
    generator = torch.Generator(device=device).manual_seed(int(seed))
    source = torch.randn(
        target.shape,
        dtype=target.dtype,
        device=device,
        generator=generator,
    )
    interpolation = 0.5 * target + 0.5 * source
    one = torch.ones(target.shape[0], device=device)
    zero = torch.zeros_like(one)
    was_training = model.training
    model.eval()
    output = model(map_input, source, one, zero, start, goal)
    model.train(was_training)

    physical_start = batch["start_pose"].float().to(device)
    physical_goal = batch["goal_pose"].float().to(device)
    states = {
        "target": target,
        "source": source,
        "interpolation": interpolation,
        "network_output": output,
    }
    result = {}
    for name, state in states.items():
        geometry = model.evaluate_trajectory_state(
            state, start, goal
        )
        start_error = torch.atan2(
            torch.sin(geometry["yaw"][:, 0] - physical_start[:, 2]),
            torch.cos(geometry["yaw"][:, 0] - physical_start[:, 2]),
        ).abs()
        goal_error = torch.atan2(
            torch.sin(geometry["yaw"][:, -1] - physical_goal[:, 2]),
            torch.cos(geometry["yaw"][:, -1] - physical_goal[:, 2]),
        ).abs()
        yaw_error = torch.maximum(start_error, goal_error)
        if float(yaw_error.max()) > 5e-4:
            raise AssertionError(
                f"{name} endpoint yaw invariant failed: "
                f"{float(yaw_error.max()):.6g} rad"
            )
        segment = geometry["position"][:, 1:] - geometry["position"][:, :-1]
        path_length = torch.linalg.vector_norm(segment, dim=-1).sum(dim=1)
        chord = torch.linalg.vector_norm(
            physical_goal[:, :2] - physical_start[:, :2], dim=1
        )
        max_curvature = geometry["curvature"].amax(dim=1)
        result[name] = {
            "state_mean": float(state.mean()),
            "state_std": float(state.std()),
            "state_abs_max": float(state.abs().max()),
            "path_to_chord_mean": float(
                (path_length / chord.clamp_min(1e-6)).mean()
            ),
            "max_curvature_median": float(max_curvature.median()),
            "max_curvature_q95": float(
                torch.quantile(max_curvature, 0.95)
            ),
            "min_speed": float(geometry["speed"].amin()),
            "endpoint_yaw_error_max_rad": float(yaw_error.max()),
        }
    return result


def mask_leakage_diagnostics(dataset, max_samples=128):
    """量化拒绝采样使轨迹邻域更易处于 mask=1 的程度。

    这是泄漏代理指标而非严格互信息估计：比较轨迹附近（排除轨迹本身）和
    远离轨迹区域的可通行率。数值明显为正时应优先重新规划 mask-conditioned
    demonstrations，而不是宣称拒绝采样完全无泄漏。
    """
    from scipy.ndimage import binary_dilation

    sample_count = min(len(dataset), int(max_samples))
    indices = np.linspace(0, len(dataset) - 1, sample_count, dtype=int)
    near_rates = []
    far_rates = []
    complete = 0
    radius_pixels = max(1, int(np.ceil(0.6 / MAP_CONFIG.resolution)))
    for index in indices:
        item = dataset[int(index)]
        mask = item["mask"].numpy() > 0.5
        dense = dense_demo_bspline(item["trajectory"].numpy())
        route = rasterize_trajectory(dense, mask.shape)
        near = binary_dilation(route, iterations=radius_pixels) & ~route
        far = ~binary_dilation(route, iterations=2 * radius_pixels)
        if near.any():
            near_rates.append(float(mask[near].mean()))
        if far.any():
            far_rates.append(float(mask[far].mean()))
        complete += int(mask.all())
    near_mean = float(np.mean(near_rates)) if near_rates else float("nan")
    far_mean = float(np.mean(far_rates)) if far_rates else float("nan")
    return {
        "samples": sample_count,
        "complete_map_rate": complete / max(sample_count, 1),
        "near_route_allowed_rate": near_mean,
        "far_region_allowed_rate": far_mean,
        "near_far_gap": near_mean - far_mean,
    }


def _dataset_item_with_mask_metadata(
    dataset,
    index,
    *,
    mask_variant=0,
    noise_seed=2026,
):
    """兼容 Subset，读取带 mask 生成元数据的底层样本。"""
    if isinstance(dataset, Subset):
        return _dataset_item_with_mask_metadata(
            dataset.dataset,
            int(dataset.indices[index]),
            mask_variant=mask_variant,
            noise_seed=noise_seed,
        )
    return dataset.get_item(
        int(index),
        mask_variant=mask_variant,
        noise_seed=noise_seed,
        return_mask_metadata=True,
    )


def mask_generation_diagnostics(
    dataset,
    *,
    max_samples=256,
    mask_variant=0,
    seed=2026,
):
    """统计 mask 拒绝采样前后的类型分布和原始配置空间连通率。"""
    sample_count = min(len(dataset), int(max_samples))
    indices = np.linspace(0, len(dataset) - 1, sample_count, dtype=int)
    attempts = []
    connected = 0
    proposed = Counter()
    accepted = Counter()
    accepted_semantics = Counter()
    accepted_trajectory_obstacle_modes = Counter()
    rejected = Counter()
    rejection_reasons = Counter()
    raw_connectivity = Counter()
    demo_blocked_fractions = []
    demo_max_blocked_fractions = []
    for offset, index in enumerate(indices):
        item = _dataset_item_with_mask_metadata(
            dataset,
            int(index),
            mask_variant=mask_variant,
            noise_seed=int(seed) + offset,
        )
        metadata = item["mask_metadata"]
        attempts.append(int(metadata["sampling_attempts"]))
        proposed.update(metadata["proposed_type_counts"])
        accepted.update([metadata["accepted_type"]])
        accepted_semantics.update([metadata.get("semantic_mode", "unknown")])
        accepted_trajectory_obstacle_modes.update(
            [metadata.get("trajectory_obstacle_mode", "unknown")]
        )
        rejected.update(metadata["rejected_type_counts"])
        rejection_reasons.update(metadata["rejection_reason_counts"])
        raw_connectivity.update(metadata.get("raw_connectivity_counts", {}))
        demo_blocked_fractions.append(
            float(metadata.get("demo_blocked_fraction", 0.0))
        )
        demo_max_blocked_fractions.append(
            float(
                metadata.get(
                    "demo_max_contiguous_blocked_fraction",
                    0.0,
                )
            )
        )
        connected += int(
            mask_start_goal_connected(
                item["mask"],
                item["start_pose"][:2],
                item["goal_pose"][:2],
            )
        )

    def proportions(counter):
        total = sum(counter.values())
        return {
            key: value / max(total, 1)
            for key, value in sorted(counter.items())
        }

    return {
        "samples": sample_count,
        # raw 指拒绝采样前的候选；accepted 单独报告最终样本连通率。
        "raw_connectivity_rate": raw_connectivity.get("connected", 0)
        / max(sum(raw_connectivity.values()), 1),
        "accepted_connectivity_rate": connected / max(sample_count, 1),
        "average_sampling_attempts": float(np.mean(attempts)) if attempts else 0.0,
        "average_resamples": (
            float(np.mean(np.asarray(attempts) - 1)) if attempts else 0.0
        ),
        "raw_type_counts": dict(proposed),
        "raw_type_proportions": proportions(proposed),
        "accepted_type_counts": dict(accepted),
        "accepted_type_proportions": proportions(accepted),
        "accepted_semantic_counts": dict(accepted_semantics),
        "accepted_semantic_proportions": proportions(accepted_semantics),
        "accepted_trajectory_obstacle_mode_counts": dict(
            accepted_trajectory_obstacle_modes
        ),
        "accepted_trajectory_obstacle_mode_proportions": proportions(
            accepted_trajectory_obstacle_modes
        ),
        "rejected_type_counts": dict(rejected),
        "rejected_type_proportions": proportions(rejected),
        "rejection_reason_counts": dict(rejection_reasons),
        "mean_demo_blocked_fraction": float(
            np.mean(demo_blocked_fractions)
        ) if demo_blocked_fractions else 0.0,
        "demo_blocked_context_rate": float(
            np.mean(np.asarray(demo_blocked_fractions) > 0.0)
        ) if demo_blocked_fractions else 0.0,
        "max_demo_blocked_fraction": float(
            np.max(demo_blocked_fractions)
        ) if demo_blocked_fractions else 0.0,
        "mean_demo_max_contiguous_blocked_fraction": float(
            np.mean(demo_max_blocked_fractions)
        ) if demo_max_blocked_fractions else 0.0,
        "max_demo_max_contiguous_blocked_fraction": float(
            np.max(demo_max_blocked_fractions)
        ) if demo_max_blocked_fractions else 0.0,
    }


@torch.no_grad()
def noise_invariance_diagnostics(
    model,
    dataset,
    device,
    *,
    max_contexts=8,
    noise_draws=4,
    seed=2026,
):
    """固定 mask/任务/source，仅改变 mask=0 噪声并比较模型输出。

    返回控制点平均偏差、稠密轨迹最大偏差、几何模式变化率和 strict-valid
    状态变化率。Stage 1 不依赖特权 stability map；数据未提供 cost map 时，
    使用处处安全的中性场，仅检查 mask、边界、曲率和端点 yaw 的状态变化。
    """
    if max_contexts <= 0:
        return None
    if noise_draws < 2:
        raise ValueError("noise_draws 至少为 2")
    context_count = min(len(dataset), int(max_contexts))
    indices = np.linspace(0, len(dataset) - 1, context_count, dtype=int)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    control_deviations = []
    dense_max_deviations = []
    mode_changes = []
    validity_changes = []
    contexts_with_stability_map = 0
    was_training = model.training
    model.eval()

    for index in indices:
        item = dataset[int(index)]
        base_map = item["map"].float().to(device)
        mask = item["mask"].float().to(device)
        cost_map = item.get("cost_map")
        if cost_map is None:
            # Stage 1 只训练部分观测轨迹生成器，不应为了一个可选诊断强制
            # 读取特权 stability 文件。中性场让 stability hard check 恒通过。
            cost_map = torch.full(
                MAP_CONFIG.cost_map_size,
                float(SAFETY_COST_CONFIG.d_safe_meters) + 1.0,
                dtype=torch.float32,
                device=device,
            )
        else:
            contexts_with_stability_map += 1
            cost_map = cost_map.to(device)
        maps = base_map.unsqueeze(0).repeat(noise_draws, 1, 1, 1)
        noise = torch.randn(
            noise_draws,
            3,
            *mask.shape,
            device=device,
            generator=generator,
        ) * MASK_NOISE_STD
        maps[:, :3] = torch.where(
            mask[None, None] > 0.5,
            maps[:, :3],
            noise,
        )

        start, goal = normalize_poses(
            item["start_pose"],
            item["goal_pose"],
            model.coordinate_scale,
            device,
        )
        start = start.repeat(noise_draws, 1)
        goal = goal.repeat(noise_draws, 1)
        source = model.project_zero_sum(
            torch.randn(
                1,
                model.num_edges,
                2,
                device=device,
                generator=generator,
            )
        ).repeat(noise_draws, 1, 1)
        residual = model.sample_pmf_onestep(
            maps,
            start,
            goal,
            source_noise=source,
            return_residual=True,
        )
        control = (
            model.decode_residual_edges(residual, start, goal)
            * model.coordinate_scale
        )
        geometry = _decode_geometry(model, residual, start, goal)
        dense = geometry["position"]

        control_center = control.mean(dim=0, keepdim=True)
        dense_center = dense.mean(dim=0, keepdim=True)
        control_deviations.append(
            float(torch.linalg.vector_norm(control - control_center, dim=-1).mean())
        )
        dense_max_deviations.append(
            float(torch.linalg.vector_norm(dense - dense_center, dim=-1).max())
        )

        chord = (
            item["goal_pose"][:2] - item["start_pose"][:2]
        ).float().to(device)
        relative = dense - item["start_pose"][:2].float().to(device)[None, None]
        signed = (
            chord[0] * relative[..., 1] - chord[1] * relative[..., 0]
        ) / chord.square().sum().clamp_min(1e-6)
        score = signed.mean(dim=1)
        mode = torch.where(
            score > 0.05,
            torch.ones_like(score, dtype=torch.long),
            torch.where(
                score < -0.05,
                -torch.ones_like(score, dtype=torch.long),
                torch.zeros_like(score, dtype=torch.long),
            ),
        )
        mode_changes.append(float((mode != mode[0]).float().mean()))

        validity = trajectory_validity_metrics(
            dense,
            cost_map,
            MAP_CONFIG.cost_map_info(),
            analytic_yaw=geometry["yaw"],
            analytic_curvature=geometry["curvature"],
            analytic_curvature_audit=(
                model.audit_trajectory_state_curvature(
                    residual, start, goal
                )
            ),
            mask=mask,
            start_pose=item["start_pose"].to(device),
            goal_pose=item["goal_pose"].to(device),
            condition_ids=torch.zeros(
                noise_draws,
                dtype=torch.long,
                device=device,
            ),
        )["strict_valid"]
        validity_changes.append(
            float((validity != validity[0]).float().mean())
        )

    model.train(was_training)
    return {
        "contexts": context_count,
        "contexts_with_stability_map": contexts_with_stability_map,
        "noise_draws": int(noise_draws),
        "control_point_mean_deviation_m": float(np.mean(control_deviations)),
        "dense_trajectory_max_deviation_m": float(
            np.mean(dense_max_deviations)
        ),
        "mode_change_rate": float(np.mean(mode_changes)),
        "strict_valid_change_rate": float(np.mean(validity_changes)),
    }


def train_stage1(args, device):
    """Stage 1：仅用部分观测条件和示范轨迹训练基础生成器。"""
    output = Path(args.fileDir)
    output.mkdir(parents=True, exist_ok=True)
    environment_split = stage1_environment_selection(args)
    objective_config = {
        "semantic_version": STAGE1_ENDPOINT_CURVATURE_OBJECTIVE,
        "endpoint_curvature_weight": float(
            args.stage1_endpoint_curvature_weight
        ),
        "endpoint_curvature_tail_ratio": float(
            args.stage1_endpoint_curvature_tail_ratio
        ),
        "mask_noise_mode": args.stage1_mask_noise_mode,
    }
    dynamic_train_noise = args.stage1_mask_noise_mode == "legacy_random"
    train_set, train_envs = make_partial_dataset(
        args.dataFolder,
        "train",
        mask_seed=args.mask_seed,
        p_mask=args.p_mask,
        mask_mode="stage1_demo_valid",
        vehicle_radius_meters=args.vehicle_radius_meters,
        environment_names=environment_split["train"],
        dynamic_mask_noise=dynamic_train_noise,
    )
    val_set, val_envs = make_partial_dataset(
        args.dataFolder,
        "val",
        mask_seed=args.mask_seed,
        p_mask=args.p_mask,
        mask_mode="stage1_demo_valid",
        vehicle_radius_meters=args.vehicle_radius_meters,
        environment_names=environment_split["validation"],
        dynamic_mask_noise=False,
    )
    noise_eval_set, _ = make_partial_dataset(
        args.dataFolder,
        "val",
        # 噪声不变性是 Stage 1 的基础监控，不读取训练期特权信息。
        compute_stability_map=False,
        mask_seed=args.mask_seed,
        p_mask=args.p_mask,
        mask_mode="stage1_demo_valid",
        vehicle_radius_meters=args.vehicle_radius_meters,
        environment_names=environment_split["validation"],
        dynamic_mask_noise=False,
    )
    if args.max_contexts is not None:
        train_set = Subset(train_set, range(min(len(train_set), args.max_contexts)))
        val_set = Subset(val_set, range(min(len(val_set), args.max_contexts)))
        noise_eval_set = Subset(
            noise_eval_set,
            range(min(len(noise_eval_set), args.max_contexts)),
        )
    leakage_metrics = mask_leakage_diagnostics(train_set)
    mask_generation_metrics = mask_generation_diagnostics(
        train_set,
        max_samples=256,
        mask_variant=0,
        seed=args.seed + 60_000,
    )
    print("mask leakage diagnostics:", leakage_metrics)
    print("mask generation diagnostics:", mask_generation_metrics)
    if leakage_metrics["near_far_gap"] > 0.10:
        print(
            "Info: informed ellipse 会有意提高示范邻域可见率；"
            "near_far_gap 作为该训练先验的强度记录，不解释为独立 mask。"
        )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batchSize,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batchSize,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    start_epoch = 0
    global_update = 0
    best = float("inf")
    if args.resume:
        model, model_args, checkpoint = load_model(args.resume, device)
        _require_main_method_model(model)
        _require_current_mask_semantics(checkpoint)
        _require_demo_target_semantics(checkpoint)
        _require_mask_configuration(checkpoint, args)
        stored_environment_split = checkpoint.get(
            "stage1_environment_split"
        )
        legacy_compatible = (
            stored_environment_split is None
            and environment_split["semantic_version"]
            == "legacy_same_environment_train_val_v1"
        )
        if (
            not legacy_compatible
            and stored_environment_split != environment_split
        ):
            raise ValueError(
                "Checkpoint Stage-1 environment split does not match the "
                "current frozen split"
            )
        stored_objective = checkpoint.get("stage1_objective_config")
        legacy_objective_compatible = (
            stored_objective is None
            and objective_config["endpoint_curvature_weight"] == 0.0
            and objective_config["mask_noise_mode"] == "legacy_random"
        )
        if (
            not legacy_objective_compatible
            and stored_objective != objective_config
        ):
            raise ValueError(
                "Checkpoint Stage-1 objective/noise protocol does not match "
                "the current configuration"
            )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_update = int(checkpoint.get("global_update", 0))
        best = float(checkpoint.get("best_val_loss", checkpoint.get("val_loss", best)))
    else:
        model_args = default_model_args()
        model = PathMeanFlowTransformer(**model_args).to(device)

    writer = _make_tensorboard_writer(
        output,
        "stage1",
        # 从头重跑同一目录时隐藏此前未完成 run 留下的同 step 事件。
        purge_step=start_epoch if args.resume else 0,
    )
    _log_numeric_tree(
        writer,
        "stage1/data/mask_leakage",
        leakage_metrics,
        start_epoch,
    )
    _log_numeric_tree(
        writer,
        "stage1/data/mask_generation",
        mask_generation_metrics,
        start_epoch,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.stage1_lr, weight_decay=1e-4
    )
    if args.resume and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    noise_metrics = (
        checkpoint.get("mask_noise_invariance") if args.resume else None
    )
    representation_metrics = (
        checkpoint.get("representation_diagnostics") if args.resume else None
    )

    updates_this_run = 0
    component_keys = (
        "flow",
        "endpoint",
        "curvature_aux",
        "curvature_weighted",
        "endpoint_curvature_pass_rate",
    )
    for epoch in range(start_epoch, args.stage1_epochs):
        model.train()
        train_total = 0.0
        train_batches = 0
        train_component_totals = {key: 0.0 for key in component_keys}
        train_component_counts = {key: 0 for key in component_keys}
        train_curvature_invalid = 0
        train_curvature_skipped = 0
        train_gradient_norm = 0.0
        for batch in tqdm(train_loader, desc=f"Stage 1 train {epoch}"):
            if (
                args.stage1_max_updates is not None
                and updates_this_run >= args.stage1_max_updates
            ):
                break
            optimizer.zero_grad(set_to_none=True)
            loss, terms = prior_transport_loss(
                model,
                batch,
                device,
                endpoint_curvature_weight=(
                    args.stage1_endpoint_curvature_weight
                ),
                endpoint_curvature_tail_ratio=(
                    args.stage1_endpoint_curvature_tail_ratio
                ),
            )
            train_curvature_invalid += int(
                terms["invalid_demo_curvature"]
            )
            train_curvature_skipped += int(
                terms["skipped_demo_curvature"]
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Non-finite Stage 1 loss at update {global_update}"
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            optimizer.step()
            global_update += 1
            updates_this_run += 1
            train_total += float(loss.detach())
            for key in component_keys:
                value = float(terms[key].detach())
                if math.isfinite(value):
                    train_component_totals[key] += value
                    train_component_counts[key] += 1
            train_gradient_norm += float(gradient_norm.detach())
            train_batches += 1
            _log_stage1_update_metrics(
                writer,
                loss=loss,
                terms=terms,
                gradient_norm=gradient_norm,
                learning_rate=optimizer.param_groups[0]["lr"],
                global_update=global_update,
            )
            if updates_this_run == 1 or updates_this_run % 100 == 0:
                writer.flush()
            if updates_this_run % 100 == 0:
                target_state = path_coordinates_from_batch(batch)
                print(
                    f"stage1 update={global_update} "
                    f"loss={float(loss.detach()):.7f} "
                    f"grad={float(gradient_norm.detach()):.5f} "
                    f"target_std={float(target_state.std()):.4f} "
                    f"used={terms['used_examples']}"
                )

        model.eval()
        val_total = 0.0
        val_batches = 0
        val_component_totals = {key: 0.0 for key in component_keys}
        val_component_counts = {key: 0 for key in component_keys}
        val_curvature_invalid = 0
        val_curvature_skipped = 0
        representation_batch = None
        val_generator = torch.Generator(device=device).manual_seed(
            args.seed + epoch
        )
        for batch in tqdm(val_loader, desc=f"Stage 1 val {epoch}"):
            if representation_batch is None:
                representation_batch = batch
            loss, terms = prior_transport_loss(
                model,
                batch,
                device,
                generator=val_generator,
                endpoint_curvature_weight=(
                    args.stage1_endpoint_curvature_weight
                ),
                endpoint_curvature_tail_ratio=(
                    args.stage1_endpoint_curvature_tail_ratio
                ),
            )
            val_curvature_invalid += int(
                terms["invalid_demo_curvature"]
            )
            val_curvature_skipped += int(
                terms["skipped_demo_curvature"]
            )
            val_total += float(loss.detach())
            for key in component_keys:
                value = float(terms[key].detach())
                if math.isfinite(value):
                    val_component_totals[key] += value
                    val_component_counts[key] += 1
            val_batches += 1

        train_mean = train_total / max(train_batches, 1)
        val_mean = val_total / max(val_batches, 1)
        train_components = {
            key: (
                train_component_totals[key] / train_component_counts[key]
                if train_component_counts[key]
                else float("nan")
            )
            for key in component_keys
        }
        val_components = {
            key: (
                val_component_totals[key] / val_component_counts[key]
                if val_component_counts[key]
                else float("nan")
            )
            for key in component_keys
        }
        train_components["base_meanflow"] = (
            train_components["flow"] + 0.25 * train_components["endpoint"]
        )
        val_components["base_meanflow"] = (
            val_components["flow"] + 0.25 * val_components["endpoint"]
        )
        gradient_mean = train_gradient_norm / max(train_batches, 1)
        representation_metrics = stage1_representation_diagnostics(
            model,
            representation_batch,
            device,
            seed=args.seed + 70_000 + epoch,
        )
        try:
            noise_metrics = noise_invariance_diagnostics(
                model,
                noise_eval_set,
                device,
                max_contexts=args.noise_invariance_contexts,
                noise_draws=args.noise_invariance_draws,
                seed=args.seed + 40_000 + epoch,
            )
        except Exception as exc:
            # 诊断不是训练目标；监控失败不能让已经完成的 epoch 丢失。
            print(
                "Warning: mask 噪声不变性诊断失败，本轮 checkpoint 仍会保存："
                f" {type(exc).__name__}: {exc}"
            )
            noise_metrics = {
                "diagnostic_failed": True,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        best = min(best, val_mean)
        print(
            f"stage1 epoch={epoch} train={train_mean:.7f} val={val_mean:.7f} "
            f"base_val={val_components['base_meanflow']:.7f} "
            f"grad={gradient_mean:.5f} "
            f"curvature_invalid={train_curvature_invalid}/"
            f"{val_curvature_invalid} "
            f"curvature_skipped={train_curvature_skipped}/"
            f"{val_curvature_skipped}"
        )
        print("44D representation diagnostics:", representation_metrics)
        if noise_metrics is not None:
            print("mask noise invariance:", noise_metrics)
        _log_numeric_tree(
            writer,
            "stage1",
            {
                "loss": {
                    "train": train_mean,
                    "validation": val_mean,
                    "best_validation": best,
                    "train_components": train_components,
                    "validation_components": val_components,
                },
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm": gradient_mean,
                "demo_curvature_invalid": {
                    "train": train_curvature_invalid,
                    "validation": val_curvature_invalid,
                },
                "demo_curvature_skipped": {
                    "train": train_curvature_skipped,
                    "validation": val_curvature_skipped,
                },
                "mask_noise_invariance": noise_metrics,
                "representation": representation_metrics,
            },
            epoch,
        )
        writer.flush()
        checkpoint_metadata = {
            "representation_semantic_version": (
                REPRESENTATION_SEMANTIC_VERSION
            ),
            "stage": "stage1",
            "epoch": epoch,
            "global_update": global_update,
            "train_loss": train_mean,
            "val_loss": val_mean,
            "train_loss_components": train_components,
            "val_loss_components": val_components,
            "best_val_loss": best,
            "optimizer_state_dict": optimizer.state_dict(),
            "mask_seed": args.mask_seed,
            "p_mask": args.p_mask,
            "mask_noise_std": MASK_NOISE_STD,
            "vehicle_radius_meters": args.vehicle_radius_meters,
            "input_mask_semantics": MASK_INPUT_SEMANTICS,
            "mask_generation_semantics": MASK_GENERATION_SEMANTICS,
            "demo_target_semantics": DEMO_TARGET_SEMANTICS,
            "mask_leakage_metrics": leakage_metrics,
            "mask_generation_metrics": mask_generation_metrics,
            "mask_noise_invariance": noise_metrics,
            "representation_diagnostics": representation_metrics,
            "demo_curvature_invalid": {
                "train": train_curvature_invalid,
                "validation": val_curvature_invalid,
            },
            "demo_curvature_skipped": {
                "train": train_curvature_skipped,
                "validation": val_curvature_skipped,
            },
            "mask_distribution": (
                "incomplete_observation_outside_informed_demo_ellipse"
            ),
            "configuration_mask": "single_vehicle_eroded_mask",
            "stage1_environment_split": environment_split,
            "stage1_objective_config": objective_config,
        }
        save_checkpoint(
            output / "stage1_last.pth",
            model,
            model_args,
            **checkpoint_metadata,
        )
        if val_mean <= best:
            save_checkpoint(
                output / "stage1_best.pth",
                model,
                model_args,
                **checkpoint_metadata,
            )
        if (
            args.stage1_max_updates is not None
            and updates_this_run >= args.stage1_max_updates
        ):
            print(
                "Stage 1 stopped at requested smoke limit: "
                f"{updates_this_run} updates this run."
            )
            break

    with open(output / "stage1_method.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": (
                    "boundary_constrained_path_meanflow_with_"
                    "privileged_constraint_distillation"
                ),
                "stage1": "conditional_path_meanflow_pretraining",
                "mask_distribution": "conditioned_on_dense_demo_validity",
                "configuration_mask": "single_vehicle_eroded_mask",
                "model_args": model_args,
                "representation_semantic_version": (
                    REPRESENTATION_SEMANTIC_VERSION
                ),
                "train_environments": train_envs,
                "val_environments": val_envs,
                "environment_split": environment_split,
                "objective_config": objective_config,
                "mask_seed": args.mask_seed,
                "p_mask": args.p_mask,
                "mask_noise_std": MASK_NOISE_STD,
                "vehicle_radius_meters": args.vehicle_radius_meters,
                "mask_noise_invariance": noise_metrics,
                "representation_diagnostics": representation_metrics,
                "input_mask_semantics": MASK_INPUT_SEMANTICS,
                "demo_target_semantics": DEMO_TARGET_SEMANTICS,
                "mask_leakage_metrics": leakage_metrics,
                "mask_generation_metrics": mask_generation_metrics,
                "inference_uses_privileged_map": False,
                "inference_uses_optimizer": False,
                "radial_output": False,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    writer.close()
    return output / "stage1_best.pth"


class PathCorrectionReplayBuffer:
    """有限大小的聚合路径修正数据集。

    buffer 优先保留最近若干轮，同时从较老数据中保留高违规样本和随机历史
    样本。每条记录始终是一条独立血缘：
    ``source sample -> candidate path coordinates -> corrected path coordinates``。

    旧 tensor key 仅作为 v6 文件格式兼容字段保留，不作为方法术语。
    """

    # v6 stores every tensor lineage in 44-D free path coordinates.
    VERSION = 6
    TENSOR_KEYS = ("source_noise", "proposal_residual", "target_residual")

    def __init__(
        self,
        max_size,
        *,
        recent_rounds=3,
        recent_fraction=0.6,
        priority_fraction=0.2,
        seed=2026,
    ):
        if max_size <= 0:
            raise ValueError("replay buffer max_size 必须为正数")
        if recent_rounds <= 0:
            raise ValueError("recent_rounds 必须为正数")
        if not 0.0 <= recent_fraction <= 1.0:
            raise ValueError("recent_fraction 必须在 [0,1]")
        if not 0.0 <= priority_fraction <= 1.0:
            raise ValueError("priority_fraction 必须在 [0,1]")
        self.max_size = int(max_size)
        self.recent_rounds = int(recent_rounds)
        self.recent_fraction = float(recent_fraction)
        self.priority_fraction = float(priority_fraction)
        self.seed = int(seed)
        self.entries = []

    def __len__(self):
        return len(self.entries)

    @staticmethod
    def _cpu_entry(entry):
        """复制并规范化记录，避免 buffer 意外持有 GPU 计算图。"""
        normalized = dict(entry)
        for key in PathCorrectionReplayBuffer.TENSOR_KEYS:
            value = torch.as_tensor(entry[key], dtype=torch.float32)
            normalized[key] = value.detach().cpu().clone()
        for key in (
            "dataset_index",
            "particle_index",
            "round",
            "mask_variant",
            "mask_noise_seed",
        ):
            normalized[key] = int(entry[key])
        for key in ("initial_cost", "final_cost", "priority"):
            normalized[key] = float(entry[key])
        for key in ("initial_valid", "final_valid"):
            normalized[key] = bool(entry[key])
        return normalized

    def add(self, new_entries, current_round):
        """加入当前轮样本，并按最近/高风险/历史三类压缩。"""
        self.entries.extend(self._cpu_entry(entry) for entry in new_entries)
        self._compact(int(current_round))

    def _compact(self, current_round):
        if len(self.entries) <= self.max_size:
            return

        indexed = list(enumerate(self.entries))
        cutoff = current_round - self.recent_rounds + 1
        recent = [(i, e) for i, e in indexed if e["round"] >= cutoff]
        recent_budget = min(
            self.max_size,
            max(1 if recent else 0, round(self.max_size * self.recent_fraction)),
        )
        recent.sort(
            key=lambda item: (item[1]["round"], item[1]["priority"]),
            reverse=True,
        )
        selected = {index for index, _ in recent[:recent_budget]}

        remaining = [(i, e) for i, e in indexed if i not in selected]
        free_slots = self.max_size - len(selected)
        priority_budget = min(
            free_slots,
            round(self.max_size * self.priority_fraction),
            len(remaining),
        )
        remaining.sort(key=lambda item: item[1]["priority"], reverse=True)
        selected.update(index for index, _ in remaining[:priority_budget])

        # 其余容量随机保存历史，防止 buffer 完全退化成最近或极端违规样本。
        candidates = [i for i, _ in indexed if i not in selected]
        random.Random(self.seed + current_round).shuffle(candidates)
        selected.update(candidates[: self.max_size - len(selected)])
        self.entries = [
            entry for index, entry in indexed if index in selected
        ]

    def state_dict(self):
        return {
            "version": self.VERSION,
            "representation_semantic_version": (
                REPRESENTATION_SEMANTIC_VERSION
            ),
            "max_size": self.max_size,
            "recent_rounds": self.recent_rounds,
            "recent_fraction": self.recent_fraction,
            "priority_fraction": self.priority_fraction,
            "seed": self.seed,
            "entries": self.entries,
        }

    @classmethod
    def from_state_dict(cls, state):
        if int(state.get("version", -1)) != cls.VERSION:
            raise ValueError(
                "Replay buffer 版本不兼容；旧 25×2 replay 已作废，"
                "必须使用新 44D 表示重新采集。"
            )
        actual = state.get("representation_semantic_version")
        if actual != REPRESENTATION_SEMANTIC_VERSION:
            raise ValueError(
                "Replay buffer 轨迹表示不兼容："
                f"期望 {REPRESENTATION_SEMANTIC_VERSION!r}，"
                f"实际 {actual!r}。"
            )
        buffer = cls(
            state["max_size"],
            recent_rounds=state["recent_rounds"],
            recent_fraction=state["recent_fraction"],
            priority_fraction=state["priority_fraction"],
            seed=state["seed"],
        )
        buffer.entries = [
            buffer._cpu_entry(entry) for entry in state["entries"]
        ]
        if len(buffer.entries) > buffer.max_size:
            raise ValueError("replay 文件中的样本数超过声明容量")
        return buffer

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path):
        return cls.from_state_dict(torch.load(path, map_location="cpu"))


# Compatibility alias for earlier analysis scripts.
DaggerReplayBuffer = PathCorrectionReplayBuffer


class ReplayPairDataset(Dataset):
    """用保存的 mask/noise 版本精确重建 replay 收集时的四通道条件。"""

    def __init__(self, entries, base_dataset):
        self.entries = entries
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        context = self.base_dataset.get_item(
            entry["dataset_index"],
            mask_variant=entry["mask_variant"],
            noise_seed=entry["mask_noise_seed"],
        )
        return {
            "map": context["map"],
            "start_pose": context["start_pose"],
            "goal_pose": context["goal_pose"],
            "source_noise": entry["source_noise"],
            "target_residual": entry["target_residual"],
            "round": entry["round"],
            "priority": entry["priority"],
        }


def _decode_dense(
    model,
    residual,
    start,
    goal,
    num_points=DENSE_TRAJECTORY_POINTS,
):
    """Decode free path coordinates through the analytic path representation."""
    if int(num_points) != model.trajectory_representation.dense_points:
        raise ValueError(
            "Production analytic geometry uses a single dense sampling "
            f"contract ({model.trajectory_representation.dense_points}), got "
            f"{num_points}."
        )
    return model.evaluate_trajectory_state(
        residual, start, goal
    )["position"]


def _decode_geometry(model, state, start, goal):
    """Return the unified analytic physical geometry dictionary."""
    return model.evaluate_trajectory_state(state, start, goal)


def _sample_context_indices(total, count, seed):
    """每轮确定性采样 context；数量不足时允许有放回采样。"""
    generator = torch.Generator().manual_seed(int(seed))
    if count <= total:
        return torch.randperm(total, generator=generator)[:count].tolist()
    return torch.randint(total, (count,), generator=generator).tolist()


def _select_endpoint_feasible_contexts(
    dataset,
    context_indices,
    distillation_round=None,
    *,
    seed,
    available_contexts=None,
    dagger_round=None,
):
    """重采样固定端点处不可能满足 stability hard threshold 的任务。

    这里仅检查不可由特权路径修正器改变的起终点位置和解析航向；
    不再在 yaw 候选中排序，也不会根据中段或优化难度筛选数据。
    """
    if dagger_round is not None:
        if distillation_round is not None:
            raise ValueError(
                "同时提供了 distillation_round 和旧参数 dagger_round"
            )
        distillation_round = int(dagger_round)
    if distillation_round is None:
        raise ValueError("必须提供 distillation_round")
    requested = len(context_indices)
    if requested == 0:
        return [], {
            "context_candidates_drawn": 0,
            "endpoint_stability_rejections": 0,
            "raw_endpoint_stability_feasibility_rate": 0.0,
            "raw_start_endpoint_stability_feasibility_rate": 0.0,
            "raw_goal_endpoint_stability_feasibility_rate": 0.0,
            "average_context_sampling_attempts": 0.0,
        }
    total = len(dataset) if available_contexts is None else int(available_contexts)
    if total <= 0:
        raise ValueError("Stage 2 没有可采样的训练 context")

    initial = [int(index) for index in context_indices]
    replacement_generator = torch.Generator().manual_seed(int(seed) + 7_919)
    max_draws = max(requested * 20, requested + 64)
    selected = []
    drawn = 0
    start_rejections = 0
    goal_rejections = 0
    best_start_margins = []
    best_goal_margins = []

    while len(selected) < requested and drawn < max_draws:
        if drawn < len(initial):
            dataset_index = initial[drawn]
        else:
            dataset_index = int(
                torch.randint(
                    total,
                    (1,),
                    generator=replacement_generator,
                ).item()
            )
        mask_variant = int(distillation_round) + 1
        context_noise_seed = (
            int(seed) + int(dataset_index) * 32_452_843 + 97
        )
        context = dataset.get_item(
            dataset_index,
            mask_variant=mask_variant,
            noise_seed=context_noise_seed,
            return_mask_metadata=True,
        )
        feasibility = endpoint_stability_feasibility(
            context["start_pose"],
            context["goal_pose"],
            context["cost_map"],
            MAP_CONFIG.cost_map_info(),
            start_yaw_tolerance_rad=0.0,
            goal_yaw_tolerance_rad=0.0,
        )
        drawn += 1
        best_start_margins.append(
            feasibility["start_best_stability_margin"]
        )
        best_goal_margins.append(
            feasibility["goal_best_stability_margin"]
        )
        start_rejections += int(not feasibility["start_feasible"])
        goal_rejections += int(not feasibility["goal_feasible"])
        if not feasibility["feasible"]:
            continue
        selected.append(
            {
                "dataset_index": dataset_index,
                "mask_variant": mask_variant,
                "mask_noise_seed": context_noise_seed,
                "context": context,
            }
        )

    if len(selected) != requested:
        raise RuntimeError(
            "Stage 2 无法采到足够的端点稳定性可行条件："
            f"请求 {requested}，在 {drawn} 次候选中仅得到 {len(selected)}。"
            "请检查 stability map、d_safe 和 endpoint yaw 容差。"
        )

    endpoint_rejections = drawn - len(selected)
    metrics = {
        "context_candidates_drawn": drawn,
        "endpoint_stability_rejections": endpoint_rejections,
        "start_endpoint_stability_rejections": start_rejections,
        "goal_endpoint_stability_rejections": goal_rejections,
        "raw_endpoint_stability_feasibility_rate": (
            len(selected) / max(drawn, 1)
        ),
        "raw_start_endpoint_stability_feasibility_rate": (
            1.0 - start_rejections / max(drawn, 1)
        ),
        "raw_goal_endpoint_stability_feasibility_rate": (
            1.0 - goal_rejections / max(drawn, 1)
        ),
        "average_context_sampling_attempts": drawn / max(requested, 1),
        "candidate_start_best_stability_margin_mean": float(
            np.mean(best_start_margins)
        ),
        "candidate_goal_best_stability_margin_mean": float(
            np.mean(best_goal_margins)
        ),
    }
    return selected, metrics


def collect_privileged_distillation_round(
    model,
    dataset,
    context_indices,
    distillation_round,
    config,
    device,
    *,
    seed,
    available_contexts=None,
):
    """用当前模型采样，并让特权优化器按原下标逐条纠正。

    这里不进行集合匹配或重排。优化器输出的第 i 条轨迹就是第 i 个
    ``source_noise`` 的监督目标。
    """
    model.eval()
    model.use_gradient_checkpoint = False
    source_generator = torch.Generator(device=device).manual_seed(int(seed))
    records = []
    correction_failures = []
    initial_costs = []
    final_costs = []
    initial_valids = []
    final_valids = []
    initial_strict_valids = []
    final_strict_valids = []
    initial_mask_valids = []
    final_mask_valids = []
    initial_forbidden_valids = []
    final_forbidden_valids = []
    demo_mask_valids = []
    demo_max_curvatures = []
    connected_contexts = 0
    disconnected_pairs = 0
    mask_invalid_proposals_kept = 0
    mask_sampling_attempts = []
    raw_mask_type_counts = Counter()
    accepted_mask_type_counts = Counter()
    accepted_mask_semantic_counts = Counter()
    accepted_trajectory_obstacle_mode_counts = Counter()
    rejected_mask_type_counts = Counter()
    mask_rejection_reason_counts = Counter()
    raw_mask_connectivity_counts = Counter()
    mask_demo_blocked_fractions = []
    mask_demo_max_blocked_fractions = []
    correction_failure_reason_counts = Counter()
    cost_scale_audit = None

    selected_contexts, endpoint_context_metrics = (
        _select_endpoint_feasible_contexts(
            dataset,
            context_indices,
            distillation_round,
            seed=seed,
            available_contexts=available_contexts,
        )
    )

    for selected in tqdm(
        selected_contexts,
        desc=f"Privileged distillation collect {distillation_round}",
    ):
        # 同一路径在不同蒸馏轮使用不同 mask；variant 与噪声 seed 会随
        # 成功监督一起写入 replay，从而精确重建本轮 c_obs。
        dataset_index = selected["dataset_index"]
        mask_variant = selected["mask_variant"]
        context_noise_seed = selected["mask_noise_seed"]
        context = selected["context"]
        mask_metadata = context["mask_metadata"]
        mask_sampling_attempts.append(int(mask_metadata["sampling_attempts"]))
        raw_mask_type_counts.update(mask_metadata["proposed_type_counts"])
        accepted_mask_type_counts.update([mask_metadata["accepted_type"]])
        accepted_mask_semantic_counts.update(
            [mask_metadata.get("semantic_mode", "unknown")]
        )
        accepted_trajectory_obstacle_mode_counts.update(
            [mask_metadata.get("trajectory_obstacle_mode", "unknown")]
        )
        rejected_mask_type_counts.update(mask_metadata["rejected_type_counts"])
        mask_rejection_reason_counts.update(
            mask_metadata["rejection_reason_counts"]
        )
        raw_mask_connectivity_counts.update(
            mask_metadata.get("raw_connectivity_counts", {})
        )
        mask_demo_blocked_fractions.append(
            float(mask_metadata.get("demo_blocked_fraction", 0.0))
        )
        mask_demo_max_blocked_fractions.append(
            float(
                mask_metadata.get(
                    "demo_max_contiguous_blocked_fraction",
                    0.0,
                )
            )
        )
        demo_mask_valids.append(bool(context["demo_mask_valid"]))
        demo_state = path_coordinates_from_batch(
            context, device
        ).unsqueeze(0)
        demo_geometry = model.trajectory_representation.evaluate(
            demo_state,
            context["start_pose"].to(device).unsqueeze(0),
            context["goal_pose"].to(device).unsqueeze(0),
        )
        demo_max_curvatures.append(
            float(demo_geometry["curvature"].amax())
        )
        count = config.particles_per_context
        if not mask_start_goal_connected(
            context["mask"],
            context["start_pose"][:2],
            context["goal_pose"][:2],
        ):
            disconnected_pairs += count
            correction_failure_reason_counts["mask_disconnected"] += count
            correction_failures.append(
                {
                    "dataset_index": int(dataset_index),
                    "particle_index": None,
                    "round": int(distillation_round),
                    "mask_variant": mask_variant,
                    "mask_noise_seed": context_noise_seed,
                    "reason": "mask_disconnected",
                    "failure_categories": ["mask_disconnected"],
                    "affected_particles": count,
                }
            )
            continue
        connected_contexts += 1
        map_input = (
            context["map"].float().to(device).unsqueeze(0).repeat(count, 1, 1, 1)
        )
        start_pose = context["start_pose"].float().to(device).unsqueeze(0)
        goal_pose = context["goal_pose"].float().to(device).unsqueeze(0)
        start, goal = normalize_poses(
            start_pose, goal_pose, model.coordinate_scale, device
        )
        start_k = start.repeat(count, 1)
        goal_k = goal.repeat(count, 1)
        source = model.project_zero_sum(
            torch.randn(
                count,
                model.num_edges,
                2,
                device=device,
                generator=source_generator,
            )
        )

        signed_mask_distance = build_signed_mask_distance_map(
            context["mask"], MAP_CONFIG.cost_map_info(), device=device
        )
        with torch.no_grad():
            # 关键：proposal 来自本轮开始时的当前模型，而不是冻结 Stage 1。
            proposal = model.sample_pmf_onestep(
                map_input,
                start_k,
                goal_k,
                source_noise=source,
                return_residual=True,
            )
            proposal_geometry = _decode_geometry(
                model, proposal, start_k, goal_k
            )
            proposal_dense = proposal_geometry["position"]
            initial_cost = privileged_planning_cost(
                proposal_dense,
                start_pose,
                goal_pose,
                context["cost_map"].to(device),
                MAP_CONFIG.cost_map_info(),
                analytic_yaw=proposal_geometry["yaw"],
                analytic_curvature=proposal_geometry["curvature"],
                analytic_first_derivative=(
                    proposal_geometry["first_derivative"]
                ),
                analytic_second_derivative=(
                    proposal_geometry["second_derivative"]
                ),
                mask=context["mask"].to(device),
                signed_mask_distance_map=signed_mask_distance,
                return_per_sample=True,
            )
            initial_validity = trajectory_validity_metrics(
                proposal_dense,
                context["cost_map"].to(device),
                MAP_CONFIG.cost_map_info(),
                analytic_yaw=proposal_geometry["yaw"],
                analytic_curvature=proposal_geometry["curvature"],
                analytic_curvature_audit=(
                    model.audit_trajectory_state_curvature(
                        proposal, start_k, goal_k
                    )
                ),
                mask=context["mask"].to(device),
                signed_mask_distance_map=signed_mask_distance,
                start_pose=start_pose,
                goal_pose=goal_pose,
                condition_ids=torch.zeros(
                    count,
                    dtype=torch.long,
                    device=device,
                ),
            )

        if cost_scale_audit is None:
            cost_scale_audit = privileged_optimizer_scale_diagnostics(
                proposal,
                start_pose,
                goal_pose,
                context["cost_map"].to(device),
                MAP_CONFIG.cost_map_info(),
                coordinate_scale=model.coordinate_scale,
            mask=context["mask"].to(device),
            signed_mask_distance_map=signed_mask_distance,
        )

        # 完整地图只进入优化器，绝不传入模型。
        optimized = apply_privileged_path_correction(
            proposal,
            start_pose,
            goal_pose,
            context["cost_map"].to(device),
            MAP_CONFIG.cost_map_info(),
            coordinate_scale=model.coordinate_scale,
            mask=context["mask"].to(device),
            signed_mask_distance_map=signed_mask_distance,
            iterations=config.privileged_steps,
            lr=config.privileged_lr,
            proposal_weight=config.proposal_weight,
        )
        final_validity = trajectory_validity_metrics(
            optimized["trajectory"],
            context["cost_map"].to(device),
            MAP_CONFIG.cost_map_info(),
            analytic_yaw=optimized["yaw"],
            analytic_curvature=optimized["curvature"],
            analytic_curvature_audit=optimized["curvature_audit"],
            mask=context["mask"].to(device),
            signed_mask_distance_map=signed_mask_distance,
            start_pose=start_pose,
            goal_pose=goal_pose,
            condition_ids=torch.zeros(
                count,
                dtype=torch.long,
                device=device,
            ),
        )

        initial_cost_cpu = initial_cost.detach().cpu()
        final_cost_cpu = optimized["task_cost"].detach().cpu()
        initial_valid_cpu = initial_validity["valid"].detach().cpu()
        final_valid_cpu = final_validity["valid"].detach().cpu()
        initial_strict_valid_cpu = (
            initial_validity["strict_valid"].detach().cpu()
        )
        final_strict_valid_cpu = (
            final_validity["strict_valid"].detach().cpu()
        )
        initial_mask_valid_cpu = initial_validity["mask_ok"].detach().cpu()
        final_mask_valid_cpu = final_validity["mask_ok"].detach().cpu()
        initial_forbidden_valid_cpu = (
            initial_validity["forbidden_region_ok"].detach().cpu()
        )
        final_forbidden_valid_cpu = (
            final_validity["forbidden_region_ok"].detach().cpu()
        )
        for particle_index in range(count):
            before = float(initial_cost_cpu[particle_index])
            after = float(final_cost_cpu[particle_index])
            invalid_bonus = 2.0 if not bool(initial_valid_cpu[particle_index]) else 0.0
            improvement = max(before - after, 0.0)
            priority = before + invalid_bonus + improvement
            initial_valid = bool(initial_valid_cpu[particle_index])
            final_valid = bool(final_valid_cpu[particle_index])
            initial_mask_valid = bool(
                initial_mask_valid_cpu[particle_index]
            )
            final_mask_valid = bool(final_mask_valid_cpu[particle_index])
            if not final_valid:
                failure_reasons = []
                if not final_mask_valid:
                    failure_reasons.append("mask_unresolved")
                if not bool(
                    final_validity["box_component_ok"][particle_index]
                ):
                    failure_reasons.append("boundary_violation")
                if not bool(
                    final_validity["stability_ok"][particle_index]
                ):
                    failure_reasons.append("stability_unresolved")
                if not bool(
                    final_validity["curvature_ok"][particle_index]
                ):
                    failure_reasons.append("curvature_unresolved")
                if not bool(
                    final_validity["endpoint_yaw_ok"][particle_index]
                ):
                    failure_reasons.append("endpoint_yaw_unresolved")
                if not bool(final_validity["length_ok"][particle_index]):
                    failure_reasons.append("length_excessive")
                if not bool(final_validity["finite_ok"][particle_index]):
                    failure_reasons.append("numerical_anomaly")
                corrected_path_coordinates = optimized[
                    "corrected_path_coordinates"
                ][
                    particle_index
                ]
                final_margin = final_validity["min_stability_margin"][
                    particle_index
                ]
                final_curvature = final_validity["max_curvature"][
                    particle_index
                ]
                if (
                    not np.isfinite(before)
                    or not np.isfinite(after)
                    or not bool(
                        torch.isfinite(corrected_path_coordinates).all()
                    )
                    or not bool(torch.isfinite(final_margin))
                    or not bool(torch.isfinite(final_curvature))
                ):
                    if "numerical_anomaly" not in failure_reasons:
                        failure_reasons.append("numerical_anomaly")
                elif before - after <= max(1e-6, abs(before) * 1e-4):
                    failure_reasons.append("optimization_stalled")
                correction_failure_reason_counts.update(failure_reasons)
                correction_failures.append(
                    {
                        "dataset_index": int(dataset_index),
                        "particle_index": particle_index,
                        "round": int(distillation_round),
                        "mask_variant": mask_variant,
                        "mask_noise_seed": context_noise_seed,
                        "reason": "path_correction_invalid",
                        "failure_categories": failure_reasons,
                        "initial_cost": before,
                        "final_cost": after,
                    }
                )
                continue

            records.append(
                {
                    "dataset_index": int(dataset_index),
                    "particle_index": particle_index,
                    "round": int(distillation_round),
                    "mask_variant": mask_variant,
                    "mask_noise_seed": context_noise_seed,
                    "source_noise": source[particle_index],
                    "proposal_residual": proposal[particle_index],
                    # 同一个 particle_index 直接取优化结果，保持严格血缘。
                    "target_residual": optimized[
                        "corrected_path_coordinates"
                    ][
                        particle_index
                    ],
                    "initial_cost": before,
                    "final_cost": after,
                    "initial_valid": initial_valid,
                    "final_valid": final_valid,
                    # 当前策略即使违反 mask，只要专家成功修复，仍然进入 buffer。
                    "initial_mask_valid": initial_mask_valid,
                    "final_mask_valid": final_mask_valid,
                    "priority": priority,
                }
            )
            mask_invalid_proposals_kept += int(not initial_mask_valid)

        initial_costs.append(initial_cost_cpu)
        final_costs.append(final_cost_cpu)
        initial_valids.append(initial_valid_cpu)
        final_valids.append(final_valid_cpu)
        initial_strict_valids.append(initial_strict_valid_cpu)
        final_strict_valids.append(final_strict_valid_cpu)
        initial_mask_valids.append(initial_mask_valid_cpu)
        final_mask_valids.append(final_mask_valid_cpu)
        initial_forbidden_valids.append(initial_forbidden_valid_cpu)
        final_forbidden_valids.append(final_forbidden_valid_cpu)

    attempted_pairs = len(selected_contexts) * config.particles_per_context
    optimized_pairs = attempted_pairs - disconnected_pairs
    if initial_costs:
        initial_costs = torch.cat(initial_costs)
        final_costs = torch.cat(final_costs)
        initial_valids = torch.cat(initial_valids)
        final_valids = torch.cat(final_valids)
        initial_strict_valids = torch.cat(initial_strict_valids)
        final_strict_valids = torch.cat(final_strict_valids)
        initial_mask_valids = torch.cat(initial_mask_valids)
        final_mask_valids = torch.cat(final_mask_valids)
        initial_forbidden_valids = torch.cat(initial_forbidden_valids)
        final_forbidden_valids = torch.cat(final_forbidden_valids)
        proposal_cost = float(initial_costs.mean())
        corrected_cost = float(final_costs.mean())
        proposal_valid_rate = float(initial_valids.float().mean())
        corrected_valid_rate = float(final_valids.float().mean())
        proposal_strict_valid_rate = float(
            initial_strict_valids.float().mean()
        )
        corrected_strict_valid_rate = float(
            final_strict_valids.float().mean()
        )
        proposal_mask_violation_rate = float(
            (~initial_mask_valids).float().mean()
        )
        corrected_mask_violation_rate = float(
            (~final_mask_valids).float().mean()
        )
        proposal_forbidden_violation_rate = float(
            (~initial_forbidden_valids).float().mean()
        )
        corrected_forbidden_violation_rate = float(
            (~final_forbidden_valids).float().mean()
        )
        mask_invalid_attempted = int((~initial_mask_valids).sum())
    else:
        proposal_cost = float("nan")
        corrected_cost = float("nan")
        proposal_valid_rate = 0.0
        corrected_valid_rate = 0.0
        proposal_strict_valid_rate = 0.0
        corrected_strict_valid_rate = 0.0
        proposal_mask_violation_rate = 0.0
        corrected_mask_violation_rate = 0.0
        proposal_forbidden_violation_rate = 0.0
        corrected_forbidden_violation_rate = 0.0
        mask_invalid_attempted = 0
    def proportions(counter):
        total = sum(counter.values())
        return {
            key: value / max(total, 1)
            for key, value in sorted(counter.items())
        }

    metrics = {
        "proposal_cost": proposal_cost,
        "corrected_cost": corrected_cost,
        "proposal_task_cost": proposal_cost,
        "corrected_task_cost": corrected_cost,
        "proposal_valid_rate": proposal_valid_rate,
        "corrected_valid_rate": corrected_valid_rate,
        "proposal_strict_valid_rate": proposal_strict_valid_rate,
        "corrected_strict_valid_rate": corrected_strict_valid_rate,
        "proposal_mask_violation_rate": proposal_mask_violation_rate,
        "corrected_mask_violation_rate": corrected_mask_violation_rate,
        "proposal_forbidden_violation_rate": (
            proposal_forbidden_violation_rate
        ),
        "corrected_forbidden_violation_rate": (
            corrected_forbidden_violation_rate
        ),
        "contexts_requested": len(context_indices),
        "contexts_attempted": len(selected_contexts),
        **endpoint_context_metrics,
        "mask_connected_contexts": connected_contexts,
        "raw_mask_connectivity_rate": raw_mask_connectivity_counts.get(
            "connected", 0
        ) / max(sum(raw_mask_connectivity_counts.values()), 1),
        "accepted_mask_connectivity_rate": connected_contexts / max(
            len(selected_contexts), 1
        ),
        "average_mask_sampling_attempts": float(
            np.mean(mask_sampling_attempts)
        ) if mask_sampling_attempts else 0.0,
        "average_mask_resamples": float(
            np.mean(np.asarray(mask_sampling_attempts) - 1)
        ) if mask_sampling_attempts else 0.0,
        "raw_mask_type_counts": dict(raw_mask_type_counts),
        "raw_mask_type_proportions": proportions(raw_mask_type_counts),
        "accepted_mask_type_counts": dict(accepted_mask_type_counts),
        "accepted_mask_type_proportions": proportions(
            accepted_mask_type_counts
        ),
        "accepted_mask_semantic_counts": dict(
            accepted_mask_semantic_counts
        ),
        "accepted_mask_semantic_proportions": proportions(
            accepted_mask_semantic_counts
        ),
        "accepted_trajectory_obstacle_mode_counts": dict(
            accepted_trajectory_obstacle_mode_counts
        ),
        "accepted_trajectory_obstacle_mode_proportions": proportions(
            accepted_trajectory_obstacle_mode_counts
        ),
        "rejected_mask_type_counts": dict(rejected_mask_type_counts),
        "rejected_mask_type_proportions": proportions(
            rejected_mask_type_counts
        ),
        "mask_rejection_reason_counts": dict(mask_rejection_reason_counts),
        "mean_mask_demo_blocked_fraction": float(
            np.mean(mask_demo_blocked_fractions)
        ) if mask_demo_blocked_fractions else 0.0,
        "max_mask_demo_blocked_fraction": float(
            np.max(mask_demo_blocked_fractions)
        ) if mask_demo_blocked_fractions else 0.0,
        "mean_mask_demo_max_contiguous_blocked_fraction": float(
            np.mean(mask_demo_max_blocked_fractions)
        ) if mask_demo_max_blocked_fractions else 0.0,
        "max_mask_demo_max_contiguous_blocked_fraction": float(
            np.max(mask_demo_max_blocked_fractions)
        ) if mask_demo_max_blocked_fractions else 0.0,
        "pairs_attempted": attempted_pairs,
        "path_correction_attempts": optimized_pairs,
        "path_correction_failure_count": attempted_pairs - len(records),
        "path_correction_success_rate": len(records) / max(attempted_pairs, 1),
        "path_correction_failure_reason_counts": dict(
            correction_failure_reason_counts
        ),
        "path_correction_failure_reason_rates": {
            reason: count / max(attempted_pairs, 1)
            for reason, count in correction_failure_reason_counts.items()
        },
        "initial_cost_scale_audit": cost_scale_audit,
        "mask_invalid_proposals_attempted": mask_invalid_attempted,
        # 只统计“原 proposal 违规、且专家最终成功”的监督样本。
        "mask_invalid_proposals_kept": mask_invalid_proposals_kept,
        "pairs_collected": len(records),
        "demo_blocked_context_rate": 1.0 - (
            sum(demo_mask_valids) / max(len(demo_mask_valids), 1)
        ),
        # 只作训练分布与 hard contract 的冲突监控，不据此筛选 Stage 2。
        "demo_curvature_hard_valid_rate": float(
            np.mean(
                np.asarray(demo_max_curvatures)
                <= (
                    SAFETY_COST_CONFIG.curvature_limit
                    + SAFETY_COST_CONFIG.hard_constraint_epsilon
                )
            )
        ) if demo_max_curvatures else 0.0,
        "demo_max_curvature_median": float(
            np.median(demo_max_curvatures)
        ) if demo_max_curvatures else 0.0,
        "demo_max_curvature_q95": float(
            np.quantile(demo_max_curvatures, 0.95)
        ) if demo_max_curvatures else 0.0,
    }
    return records, correction_failures, metrics


# Compatibility alias for earlier experiment imports.
collect_dagger_round = collect_privileged_distillation_round


def _make_replay_loader(buffer, base_dataset, args, distillation_round):
    """按当前轮/历史比例从 replay 中有放回采样固定数量的更新数据。"""
    dataset = ReplayPairDataset(buffer.entries, base_dataset)
    is_current = torch.tensor(
        [entry["round"] == distillation_round for entry in buffer.entries],
        dtype=torch.bool,
    )
    current_count = int(is_current.sum())
    history_count = len(buffer) - current_count
    weights = torch.ones(len(buffer), dtype=torch.double)
    if current_count and history_count:
        weights[is_current] = args.current_round_fraction / current_count
        weights[~is_current] = (
            1.0 - args.current_round_fraction
        ) / history_count

    sampler = WeightedRandomSampler(
        weights,
        num_samples=args.updates_per_round * args.batchSize,
        replacement=True,
        generator=torch.Generator().manual_seed(
            args.seed + 20_000 + distillation_round
        ),
    )
    return DataLoader(
        dataset,
        batch_size=args.batchSize,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_on_replay(
    model,
    optimizer,
    buffer,
    base_dataset,
    args,
    distillation_round,
    device,
):
    """只用配对 MeanFlow loss 更新当前生成器。"""
    if len(buffer) == 0:
        # 本轮所有 mask 均不可连通或专家全部失败时，不构造伪监督。
        return {"loss": 0.0, "flow": 0.0, "endpoint": 0.0, "updates": 0}
    loader = _make_replay_loader(
        buffer, base_dataset, args, distillation_round
    )
    loss_generator = torch.Generator(device=device).manual_seed(
        args.seed + 30_000 + distillation_round
    )
    model.train()
    totals = {"loss": 0.0, "flow": 0.0, "endpoint": 0.0}
    for batch in tqdm(
        loader,
        desc=f"Privileged distillation update {distillation_round}",
    ):
        optimizer.zero_grad(set_to_none=True)
        loss, terms = meanflow_transport_loss(
            model,
            batch["map"],
            batch["start_pose"],
            batch["goal_pose"],
            batch["source_noise"],
            batch["target_residual"],
            device,
            endpoint_probability=args.endpoint_probability,
            generator=loss_generator,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        totals["loss"] += float(loss.detach())
        totals["flow"] += float(terms["flow"])
        totals["endpoint"] += float(terms["endpoint"])

    denominator = max(len(loader), 1)
    return {
        **{key: value / denominator for key, value in totals.items()},
        "updates": len(loader),
    }


def train_stage2_privileged_distillation(args, device):
    """Stage 2：迭代特权约束蒸馏。"""
    output = Path(args.fileDir)
    output.mkdir(parents=True, exist_ok=True)
    replay_path = Path(args.replay_buffer or output / "stage2_replay.pt")
    config = PrivilegedConstraintDistillationConfig(
        rounds=args.stage2_rounds,
        contexts_per_round=args.contexts_per_round,
        particles_per_context=args.particles_per_context,
        privileged_steps=args.privileged_steps,
        privileged_lr=args.privileged_lr,
        proposal_weight=args.proposal_weight,
        buffer_size=args.buffer_size,
        buffer_recent_rounds=args.buffer_recent_rounds,
        buffer_recent_fraction=args.buffer_recent_fraction,
        buffer_priority_fraction=args.buffer_priority_fraction,
        updates_per_round=args.updates_per_round,
        current_round_fraction=args.current_round_fraction,
    )

    start_round = 0
    if args.resume:
        model, model_args, checkpoint = load_model(args.resume, device)
        if checkpoint.get("resume_capable") is not True:
            raise ValueError(
                "--resume 必须指向含 optimizer state 的 stage2_last.pth；"
                "stage2_best.pth 仅用于评估和部署"
            )
        _require_current_mask_semantics(checkpoint)
        _require_demo_target_semantics(checkpoint)
        _require_privileged_cost_semantics(checkpoint)
        _require_stage2_prior_physical_config(checkpoint, args)
        _require_stage2_mask_generation_semantics(checkpoint)
        _require_stage2_context_semantics(checkpoint)
        if checkpoint.get("stage") not in {
            "stage2_privileged_constraint_distillation",
            "stage2_dagger",
        }:
            raise ValueError(
                "--resume 必须指向 Stage 2 特权约束蒸馏 checkpoint"
            )
        start_round = int(
            checkpoint.get(
                "distillation_round",
                checkpoint.get("dagger_round", -1),
            )
        ) + 1
        saved_replay = Path(checkpoint.get("replay_buffer_path", replay_path))
        if not replay_path.exists() and saved_replay.exists():
            replay_path = saved_replay
        if not replay_path.exists():
            raise FileNotFoundError(f"Stage 2 resume 缺少 replay: {replay_path}")
        buffer = PathCorrectionReplayBuffer.load(replay_path)
    else:
        if not args.prior_checkpoint:
            raise ValueError("Stage 2 必须通过 --prior_checkpoint 指定 Stage 1 模型")
        model, model_args, checkpoint = load_model(args.prior_checkpoint, device)
        _require_current_mask_semantics(checkpoint)
        _require_demo_target_semantics(checkpoint)
        _require_mask_configuration(checkpoint, args)
        buffer = PathCorrectionReplayBuffer(
            config.buffer_size,
            recent_rounds=config.buffer_recent_rounds,
            recent_fraction=config.buffer_recent_fraction,
            priority_fraction=config.buffer_priority_fraction,
            seed=args.seed,
        )
    _require_main_method_model(model)
    model.use_gradient_checkpoint = False

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.stage2_lr, weight_decay=1e-4
    )
    if args.resume and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    writer = _make_tensorboard_writer(
        output,
        "stage2",
        purge_step=start_round if args.resume else 0,
    )

    train_set, train_envs = make_partial_dataset(
        args.dataFolder,
        "train",
        compute_stability_map=True,
        mask_seed=args.mask_seed,
        p_mask=args.p_mask,
        mask_mode="stage2_independent",
        vehicle_radius_meters=args.vehicle_radius_meters,
    )
    available_contexts = len(train_set)
    if args.max_contexts is not None:
        available_contexts = min(available_contexts, args.max_contexts)
    noise_eval_set = Subset(train_set, range(available_contexts))
    noise_metrics = checkpoint.get("mask_noise_invariance")
    collect_metrics = checkpoint.get("collect_metrics")
    failure_path = None

    for distillation_round in range(start_round, config.rounds):
        context_indices = _sample_context_indices(
            available_contexts,
            config.contexts_per_round,
            args.seed + distillation_round,
        )
        records, correction_failures, collect_metrics = (
            collect_privileged_distillation_round(
                model,
                train_set,
                context_indices,
                distillation_round,
                config,
                device,
                seed=args.seed + 10_000 + distillation_round,
                available_contexts=available_contexts,
            )
        )
        failure_path = (
            output
            / (
                "stage2_path_correction_failures_round_"
                f"{distillation_round:03d}.json"
            )
        )
        with open(failure_path, "w", encoding="utf-8") as handle:
            json.dump(
                correction_failures,
                handle,
                indent=2,
                ensure_ascii=False,
            )
        buffer.add(records, distillation_round)
        train_metrics = train_on_replay(
            model,
            optimizer,
            buffer,
            train_set,
            args,
            distillation_round,
            device,
        )
        noise_metrics = noise_invariance_diagnostics(
            model,
            noise_eval_set,
            device,
            max_contexts=args.noise_invariance_contexts,
            noise_draws=args.noise_invariance_draws,
            seed=args.seed + 50_000 + distillation_round,
        )
        buffer.save(replay_path)

        print(
            f"stage2 distillation_round={distillation_round} "
            f"buffer={len(buffer)} "
            f"cost={collect_metrics['proposal_cost']:.5f}"
            f"->{collect_metrics['corrected_cost']:.5f} "
            f"valid={collect_metrics['proposal_valid_rate']:.1%}"
            f"->{collect_metrics['corrected_valid_rate']:.1%} "
            f"mask_violation={collect_metrics['proposal_mask_violation_rate']:.1%}"
            f"->{collect_metrics['corrected_mask_violation_rate']:.1%} "
            f"forbidden_violation="
            f"{collect_metrics['proposal_forbidden_violation_rate']:.1%}"
            f"->{collect_metrics['corrected_forbidden_violation_rate']:.1%} "
            f"demo_blocked={collect_metrics['demo_blocked_context_rate']:.1%} "
            "demo_block_fraction="
            f"{collect_metrics['mean_mask_demo_blocked_fraction']:.1%}"
            f"/{collect_metrics['max_mask_demo_blocked_fraction']:.1%} "
            "demo_curvature_ok="
            f"{collect_metrics['demo_curvature_hard_valid_rate']:.1%} "
            f"raw_mask_connected="
            f"{collect_metrics['raw_mask_connectivity_rate']:.1%} "
            f"mask_resamples={collect_metrics['average_mask_resamples']:.2f} "
            "endpoint_stability_feasible="
            f"{collect_metrics['raw_endpoint_stability_feasibility_rate']:.1%} "
            "context_resamples="
            f"{collect_metrics['average_context_sampling_attempts'] - 1.0:.2f} "
            "endpoint_corridor_rejects="
            f"{sum(collect_metrics['mask_rejection_reason_counts'].get(key, 0) for key in ('start_corridor_blocked', 'goal_corridor_blocked'))} "
            "path_correction_success="
            f"{collect_metrics['path_correction_success_rate']:.1%} "
            f"meanflow={train_metrics['loss']:.7f}"
        )
        if noise_metrics is not None:
            print("mask noise invariance:", noise_metrics)
        _log_numeric_tree(
            writer,
            "stage2",
            {
                "collection": collect_metrics,
                "meanflow": train_metrics,
                "mask_noise_invariance": noise_metrics,
                "buffer_size": len(buffer),
                "learning_rate": optimizer.param_groups[0]["lr"],
            },
            distillation_round,
        )
        writer.flush()
        save_checkpoint(
            output / "stage2_last.pth",
            model,
            model_args,
            stage="stage2_privileged_constraint_distillation",
            distillation_round=distillation_round,
            optimizer_state_dict=optimizer.state_dict(),
            replay_buffer_path=str(replay_path.resolve()),
            privileged_constraint_distillation_config=asdict(config),
            collect_metrics=collect_metrics,
            path_correction_failures_path=str(failure_path.resolve()),
            train_metrics=train_metrics,
            mask_noise_invariance=noise_metrics,
            mask_distribution=(
                "resampled_each_round_local_intervention_"
                "with_bounded_demo_blockage_and_endpoint_yaw_corridors"
            ),
            configuration_mask=(
                "single_vehicle_eroded_mask_with_endpoint_yaw_corridors"
            ),
            mask_generation_semantics=MASK_GENERATION_SEMANTICS,
            stage2_context_semantics=STAGE2_CONTEXT_SEMANTICS,
            endpoint_mask_corridor_meters=(
                SAFETY_COST_CONFIG.endpoint_mask_corridor_meters
            ),
            stage2_mask_locality_limits={
                "max_masked_fraction": STAGE2_MAX_MASKED_FRACTION,
                "max_demo_blocked_fraction": (
                    STAGE2_MAX_DEMO_BLOCKED_FRACTION
                ),
                "max_contiguous_demo_blocked_fraction": (
                    STAGE2_MAX_CONTIGUOUS_BLOCKED_FRACTION
                ),
            },
            path_correction_failure_policy=(
                "record_and_exclude_from_aggregated_dataset"
            ),
            privileged_cost_contract=privileged_cost_contract(),
            privileged_cost_semantics=PRIVILEGED_COST_SEMANTICS,
            mask_seed=args.mask_seed,
            p_mask=args.p_mask,
            mask_noise_std=MASK_NOISE_STD,
            vehicle_radius_meters=args.vehicle_radius_meters,
            input_mask_semantics=MASK_INPUT_SEMANTICS,
            demo_target_semantics=DEMO_TARGET_SEMANTICS,
            representation_semantic_version=(
                REPRESENTATION_SEMANTIC_VERSION
            ),
        )

    with open(output / "stage2_method.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": (
                    "boundary_constrained_path_meanflow_with_"
                    "privileged_constraint_distillation"
                ),
                "stage1": "conditional_path_meanflow_pretraining",
                "stage2": "iterative_privileged_constraint_distillation",
                "mask_distribution": (
                    "resampled_each_round_local_intervention_"
                    "with_bounded_demo_blockage_and_endpoint_yaw_corridors"
                ),
                "configuration_mask": (
                    "single_vehicle_eroded_mask_with_endpoint_yaw_corridors"
                ),
                "mask_generation_semantics": MASK_GENERATION_SEMANTICS,
                "stage2_context_semantics": STAGE2_CONTEXT_SEMANTICS,
                "endpoint_mask_corridor_meters": (
                    SAFETY_COST_CONFIG.endpoint_mask_corridor_meters
                ),
                "stage2_mask_locality_limits": {
                    "max_masked_fraction": STAGE2_MAX_MASKED_FRACTION,
                    "max_demo_blocked_fraction": (
                        STAGE2_MAX_DEMO_BLOCKED_FRACTION
                    ),
                    "max_contiguous_demo_blocked_fraction": (
                        STAGE2_MAX_CONTIGUOUS_BLOCKED_FRACTION
                    ),
                },
                "path_correction_failure_policy": (
                    "record_and_exclude_from_aggregated_dataset"
                ),
                "privileged_cost_contract": privileged_cost_contract(),
                "privileged_cost_semantics": PRIVILEGED_COST_SEMANTICS,
                "last_path_correction_failures": (
                    None if failure_path is None else str(failure_path.resolve())
                ),
                "last_collect_metrics": collect_metrics,
                "privileged_constraint_distillation_config": asdict(config),
                "train_environments": train_envs,
                "replay_buffer": str(replay_path.resolve()),
                "source_target_pairing": (
                    "strict_source_candidate_corrected_lineage"
                ),
                "input_mask_semantics": MASK_INPUT_SEMANTICS,
                "demo_target_semantics": DEMO_TARGET_SEMANTICS,
                "representation_semantic_version": (
                    REPRESENTATION_SEMANTIC_VERSION
                ),
                "p_mask": args.p_mask,
                "mask_noise_std": MASK_NOISE_STD,
                "vehicle_radius_meters": args.vehicle_radius_meters,
                "mask_noise_invariance": noise_metrics,
                "uses_ot": False,
                "uses_cost_only_loss": False,
                "inference_uses_privileged_map": False,
                "inference_uses_optimizer": False,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    writer.close()
    return output / "stage2_last.pth"


def _direct_cost_forward(
    model,
    batch,
    device,
    *,
    sources_per_context,
    source_generator=None,
    source=None,
):
    """Differentiable deployment-point forward pass and privileged task cost."""
    map_input = batch["map"].float().to(device)
    physical_start = batch["start_pose"].float().to(device)
    physical_goal = batch["goal_pose"].float().to(device)
    cost_map = batch["cost_map"].float().to(device)
    mask = batch["mask"].float().to(device)
    contexts = map_input.shape[0]
    repeats = int(sources_per_context)

    map_k = map_input.repeat_interleave(repeats, dim=0)
    start_k = physical_start.repeat_interleave(repeats, dim=0)
    goal_k = physical_goal.repeat_interleave(repeats, dim=0)
    cost_map_k = cost_map.repeat_interleave(repeats, dim=0)
    mask_k = mask.repeat_interleave(repeats, dim=0)
    signed_mask = build_signed_mask_distance_map(
        mask,
        MAP_CONFIG.cost_map_info(),
        device=device,
    ).repeat_interleave(repeats, dim=0)
    start_model, goal_model = normalize_poses(
        start_k,
        goal_k,
        model.coordinate_scale,
        device,
    )
    count = contexts * repeats
    if source is None:
        if source_generator is None:
            raise ValueError(
                "Either source or source_generator must be provided"
            )
        source = torch.randn(
            count,
            model.num_edges,
            2,
            dtype=map_k.dtype,
            device=device,
            generator=source_generator,
        )
    else:
        source = torch.as_tensor(
            source,
            dtype=map_k.dtype,
            device=device,
        )
        expected_shape = (count, model.num_edges, 2)
        if tuple(source.shape) != expected_shape:
            raise ValueError(
                f"Explicit source must have shape {expected_shape}, "
                f"got {tuple(source.shape)}"
            )
    source = model.project_zero_sum(source)

    # This is exactly the deployed PMF one-step endpoint (t=1,r=0), without
    # calling the @torch.no_grad sampling wrapper.
    one = torch.ones(count, dtype=map_k.dtype, device=device)
    zero = torch.zeros(count, dtype=map_k.dtype, device=device)
    state = model.project_zero_sum(
        model(map_k, source, one, zero, start_model, goal_model)
    )
    geometry = model.evaluate_trajectory_state(
        state,
        start_model,
        goal_model,
    )
    curvature_audit = model.audit_trajectory_state_curvature(
        state,
        start_model,
        goal_model,
    )
    _, components = privileged_planning_cost(
        geometry["position"],
        start_k,
        goal_k,
        cost_map_k,
        MAP_CONFIG.cost_map_info(),
        analytic_yaw=geometry["yaw"],
        analytic_curvature=geometry["curvature"],
        analytic_first_derivative=geometry["first_derivative"],
        analytic_second_derivative=geometry["second_derivative"],
        mask=mask_k,
        signed_mask_distance_map=signed_mask,
        return_per_sample=True,
        return_components=True,
    )
    return {
        "state": state,
        "source": source,
        "geometry": geometry,
        "curvature_audit": curvature_audit,
        "components": components,
        "task_cost": components["task_cost"],
        "start_pose": start_k,
        "goal_pose": goal_k,
        "cost_map": cost_map_k,
        "mask": mask_k,
        "signed_mask_distance": signed_mask,
        "condition_ids": torch.arange(
            contexts, dtype=torch.long, device=device
        ).repeat_interleave(repeats),
        "contexts": contexts,
        "sources_per_context": repeats,
    }


DIRECT_COST_MGDA_COMPONENTS = (
    ("F", "forbidden_region"),
    ("S", "stability"),
    ("K", "curvature"),
)


def _direct_cost_mgda_update(
    model,
    output_batch,
    device,
    *,
    return_geometry=False,
):
    """Build the exact raw F/S/K MGDA gradient for one training batch."""
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("Stage 2 MGDA requires trainable model parameters")

    losses = [
        output_batch["components"][component].mean()
        for _, component in DIRECT_COST_MGDA_COMPONENTS
    ]
    if not all(bool(torch.isfinite(loss)) for loss in losses):
        raise FloatingPointError("Non-finite raw Stage 2 MGDA objective")

    gradient_tuples = []
    for index, loss in enumerate(losses):
        gradient_tuples.append(
            torch.autograd.grad(
                loss,
                parameters,
                retain_graph=index < len(losses) - 1,
                allow_unused=True,
            )
        )

    gram = torch.zeros((3, 3), dtype=torch.float64, device="cpu")
    for first_index in range(3):
        for second_index in range(first_index, 3):
            value = torch.zeros((), dtype=torch.float64, device=device)
            for first, second in zip(
                gradient_tuples[first_index],
                gradient_tuples[second_index],
            ):
                if first is not None and second is not None:
                    value = value + (
                        first.detach().float() * second.detach().float()
                    ).sum().double()
            scalar = float(value.detach().cpu())
            gram[first_index, second_index] = scalar
            gram[second_index, first_index] = scalar

    solution = solve_mgda_active_set(gram)
    alpha = solution["alpha"]
    alpha_values = [float(value) for value in alpha.tolist()]
    with torch.no_grad():
        for parameter, component_gradients in zip(
            parameters,
            zip(*gradient_tuples),
        ):
            combined = None
            for coefficient, gradient in zip(
                alpha_values,
                component_gradients,
            ):
                if gradient is None or coefficient == 0.0:
                    continue
                contribution = gradient.detach() * coefficient
                combined = (
                    contribution
                    if combined is None
                    else combined + contribution
                )
            parameter.grad = None if combined is None else combined

    dots = gram @ alpha
    mgda_norm = math.sqrt(max(float(solution["objective"]), 0.0))
    component_norms = [
        math.sqrt(max(float(gram[index, index]), 0.0))
        for index in range(3)
    ]
    denominator = float(np.mean(component_norms))
    details = {
        "alpha_F": alpha_values[0],
        "alpha_S": alpha_values[1],
        "alpha_K": alpha_values[2],
        "mgda_norm": mgda_norm,
        "mgda_ratio": (
            mgda_norm / denominator if denominator > 0.0 else float("nan")
        ),
        "mgda_active_set": solution["label"],
        "mgda_common_descent": all(float(value) > 0.0 for value in dots),
        "mgda_dot_F": float(dots[0]),
        "mgda_dot_S": float(dots[1]),
        "mgda_dot_K": float(dots[2]),
    }
    if return_geometry:
        # Development-only observability.  Keep the already computed raw
        # gradients alive so an audit can compare the real Adam parameter
        # displacement with the exact MGDA direction without recomputing it.
        details.update(
            {
                "raw_objectives": tuple(
                    float(loss.detach().cpu()) for loss in losses
                ),
                "component_norms": tuple(component_norms),
                "gram": gram.clone(),
                "gradient_tuples": gradient_tuples,
            }
        )
    return details


class DirectCostEpochDataset(Dataset):
    """Expose a fresh deterministic Stage-2 mask variant each training epoch."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.mask_variant = 1

    def set_epoch(self, epoch):
        self.mask_variant = int(epoch) + 1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.get_item(
            int(index),
            mask_variant=self.mask_variant,
        )


DIRECT_COST_PAIRED_VALIDITY_KEYS = (
    "strict_valid",
    "forbidden_region_ok",
    "stability_ok",
    "curvature_ok",
)


def _direct_cost_paired_transition_metrics(current, baseline):
    """Compare exactly paired validation proposals against frozen Stage 1."""
    metrics = {}
    for key in DIRECT_COST_PAIRED_VALIDITY_KEYS:
        current_value = (
            torch.as_tensor(current[key], dtype=torch.bool).detach().cpu().reshape(-1)
        )
        baseline_value = (
            torch.as_tensor(baseline[key], dtype=torch.bool).detach().cpu().reshape(-1)
        )
        if current_value.shape != baseline_value.shape:
            raise ValueError(
                f"Paired validity shape changed for {key}: "
                f"{tuple(current_value.shape)} != {tuple(baseline_value.shape)}"
            )
        if key == "strict_valid":
            improvements = (~baseline_value) & current_value
            regressions = baseline_value & (~current_value)
            invalid_count = int((~baseline_value).sum())
            strict_count = int(baseline_value.sum())
            metrics.update(
                {
                    "invalid_to_strict_count": int(improvements.sum()),
                    "invalid_to_strict_denominator": invalid_count,
                    "invalid_to_strict_rate": float(
                        improvements.float().sum() / max(invalid_count, 1)
                    ),
                    "strict_to_invalid_count": int(regressions.sum()),
                    "strict_to_invalid_denominator": strict_count,
                    "strict_to_invalid_rate": float(
                        regressions.float().sum() / max(strict_count, 1)
                    ),
                }
            )
            continue
        prefix = key.removesuffix("_ok").removesuffix("_region")
        regressions = baseline_value & (~current_value)
        improvements = (~baseline_value) & current_value
        passed_count = int(baseline_value.sum())
        failed_count = int((~baseline_value).sum())
        metrics.update(
            {
                f"{prefix}_regression_count": int(regressions.sum()),
                f"{prefix}_regression_denominator": passed_count,
                f"{prefix}_regression_rate": float(
                    regressions.float().sum() / max(passed_count, 1)
                ),
                f"{prefix}_improvement_count": int(improvements.sum()),
                f"{prefix}_improvement_denominator": failed_count,
                f"{prefix}_improvement_rate": float(
                    improvements.float().sum() / max(failed_count, 1)
                ),
            }
        )
    return metrics


def _direct_cost_source_distribution_metrics(positions):
    """Measure source-conditioned path spread without changing validation."""
    contexts, sources, points, _ = positions.shape
    if sources < 2:
        return {
            "pairwise_path_distance_m": 0.0,
            "covariance_effective_rank": 0.0,
        }
    difference = positions[:, :, None] - positions[:, None, :]
    distance = torch.linalg.vector_norm(difference, dim=-1).mean(dim=-1)
    upper = torch.triu_indices(sources, sources, offset=1, device=positions.device)
    pairwise = distance[:, upper[0], upper[1]].mean(dim=1)
    effective_ranks = []
    for condition in range(contexts):
        samples = positions[condition].reshape(sources, points * 2)
        centered = samples - samples.mean(dim=0, keepdim=True)
        singular = torch.linalg.svdvals(centered)
        eigenvalues = singular.square()
        total = eigenvalues.sum()
        if float(total) <= 1e-20:
            effective_ranks.append(torch.zeros((), device=positions.device))
            continue
        probability = eigenvalues / total
        probability = probability[probability > 0]
        effective_ranks.append(torch.exp(-(probability * probability.log()).sum()))
    return {
        "pairwise_path_distance_m": float(pairwise.mean()),
        "covariance_effective_rank": float(torch.stack(effective_ranks).mean()),
    }


@torch.no_grad()
def evaluate_direct_cost_stage2(
    model,
    loader,
    device,
    *,
    sources_per_context,
    seed,
    baseline_snapshot=None,
    return_snapshot=False,
):
    """Evaluate fixed validation contexts/sources and paired regressions."""
    was_training = model.training
    model.eval()
    source_generator = torch.Generator(device=device).manual_seed(int(seed))
    sums = {
        "task_cost": 0.0,
        "weighted_forbidden_region": 0.0,
        "weighted_stability": 0.0,
        "weighted_curvature": 0.0,
        "violation_max_normalized": 0.0,
        "violation_integral_normalized_m": 0.0,
        "strict_valid": 0.0,
        "forbidden_region_ok": 0.0,
        "stability_ok": 0.0,
        "curvature_ok": 0.0,
        "endpoint_yaw_ok": 0.0,
    }
    trajectory_count = 0
    condition_count = 0
    safe_at_1_count = 0.0
    safe_at_k_count = 0.0
    pairwise_distance_sum = 0.0
    effective_rank_sum = 0.0
    snapshot_chunks = {key: [] for key in DIRECT_COST_PAIRED_VALIDITY_KEYS}
    try:
        for batch in loader:
            output = _direct_cost_forward(
                model,
                batch,
                device,
                sources_per_context=sources_per_context,
                source_generator=source_generator,
            )
            validity = trajectory_validity_metrics(
                output["geometry"]["position"],
                output["cost_map"],
                MAP_CONFIG.cost_map_info(),
                analytic_yaw=output["geometry"]["yaw"],
                analytic_curvature=output["geometry"]["curvature"],
                analytic_curvature_audit=output["curvature_audit"],
                mask=output["mask"],
                signed_mask_distance_map=output["signed_mask_distance"],
                start_pose=output["start_pose"],
                goal_pose=output["goal_pose"],
                condition_ids=output["condition_ids"],
            )
            components = output["components"]
            count = output["task_cost"].numel()
            contexts = output["contexts"]
            repeats = output["sources_per_context"]
            strict = validity["strict_valid"].reshape(contexts, repeats)
            safe_at_1_count += float(strict[:, 0].float().sum())
            safe_at_k_count += float(strict.any(dim=1).float().sum())
            condition_count += contexts
            trajectory_count += count
            positions = output["geometry"]["position"].reshape(
                contexts,
                repeats,
                output["geometry"]["position"].shape[-2],
                2,
            )
            source_metrics = _direct_cost_source_distribution_metrics(positions)
            pairwise_distance_sum += (
                source_metrics["pairwise_path_distance_m"] * contexts
            )
            effective_rank_sum += (
                source_metrics["covariance_effective_rank"] * contexts
            )
            for key in (
                "task_cost",
                "weighted_forbidden_region",
                "weighted_stability",
                "weighted_curvature",
            ):
                sums[key] += float(components[key].sum())
            for key in (
                "violation_max_normalized",
                "violation_integral_normalized_m",
                "strict_valid",
                "forbidden_region_ok",
                "stability_ok",
                "curvature_ok",
                "endpoint_yaw_ok",
            ):
                sums[key] += float(validity[key].float().sum())
            for key in DIRECT_COST_PAIRED_VALIDITY_KEYS:
                snapshot_chunks[key].append(validity[key].detach().cpu().bool())
    finally:
        model.train(was_training)

    if trajectory_count == 0 or condition_count == 0:
        raise RuntimeError("Direct-cost validation loader produced no samples")
    snapshot = {
        key: torch.cat(chunks, dim=0)
        for key, chunks in snapshot_chunks.items()
    }
    comparison = snapshot if baseline_snapshot is None else baseline_snapshot
    denominator = float(trajectory_count)
    metrics = {
        "conditions": int(condition_count),
        "sources_per_context": int(sources_per_context),
        "task_cost": sums["task_cost"] / denominator,
        "forbidden_cost": (
            sums["weighted_forbidden_region"] / denominator
        ),
        "stability_cost": sums["weighted_stability"] / denominator,
        "curvature_cost": sums["weighted_curvature"] / denominator,
        "violation_max_normalized": (
            sums["violation_max_normalized"] / denominator
        ),
        "violation_integral_normalized_m": (
            sums["violation_integral_normalized_m"] / denominator
        ),
        "strict_valid_rate": sums["strict_valid"] / denominator,
        "safe_at_1": safe_at_1_count / float(condition_count),
        "safe_at_k": safe_at_k_count / float(condition_count),
        "forbidden_ok_rate": (
            sums["forbidden_region_ok"] / denominator
        ),
        "stability_ok_rate": sums["stability_ok"] / denominator,
        "curvature_ok_rate": sums["curvature_ok"] / denominator,
        "yaw_ok_rate": sums["endpoint_yaw_ok"] / denominator,
        "pairwise_path_distance_m": (
            pairwise_distance_sum / float(condition_count)
        ),
        "covariance_effective_rank": (
            effective_rank_sum / float(condition_count)
        ),
    }
    metrics.update(_direct_cost_paired_transition_metrics(snapshot, comparison))
    if return_snapshot:
        return metrics, snapshot
    return metrics


def _direct_cost_validation_is_admissible(metrics, max_regression_rate=0.05):
    """Report paired Stage-1 validity preservation as a diagnostic gate."""
    regression_keys = (
        "strict_to_invalid_rate",
        "forbidden_regression_rate",
        "stability_regression_rate",
        "curvature_regression_rate",
    )
    return all(
        float(metrics.get(key, 0.0)) <= float(max_regression_rate)
        for key in regression_keys
    )


def _direct_cost_validation_key(metrics, max_regression_rate=0.05):
    """Rank checkpoints by lower validation task cost, then safety diagnostics."""
    return (
        -float(metrics["task_cost"]),
        float(metrics["safe_at_1"]),
        float(metrics["strict_valid_rate"]),
    )


def _direct_cost_config_from_args(args):
    return {
        "epochs": int(args.stage2_epochs),
        "learning_rate": float(args.stage2_lr),
        "batch_size": int(args.stage2_batch_size),
        "sources_per_context": int(args.stage2_sources_per_context),
        "grad_clip_norm": float(args.stage2_grad_clip_norm),
        "max_updates": args.stage2_max_updates,
        "eval_every_updates": int(args.stage2_eval_every_updates),
        "validation_contexts": int(args.stage2_validation_contexts),
        "validation_sources": int(args.stage2_validation_sources),
        "p_mask": float(args.stage2_p_mask),
        "split_seed": int(args.stage2_split_seed),
        "train_environments": int(args.stage2_train_environments),
        "validation_environments": int(args.stage2_validation_environments),
        "validation_data": str(
            Path(
                getattr(args, "stage2_validation_data", "data/dataset1_val")
            ).resolve()
        ),
        "validation_split": str(
            getattr(args, "stage2_validation_split", "train")
        ),
        "max_regression_rate": float(args.stage2_max_regression_rate),
        "use_mgda": bool(getattr(args, "stage2_use_mgda", True)),
    }


def _direct_cost_optimizer_name(args):
    return (
        "SGD"
        if bool(getattr(args, "stage2_use_mgda", True))
        else "Adam"
    )


def _make_direct_cost_optimizer(model, args):
    """Use plain SGD for MGDA and retain Adam for the fixed scalarization."""
    if bool(getattr(args, "stage2_use_mgda", True)):
        return torch.optim.SGD(
            model.parameters(),
            lr=float(args.stage2_lr),
            momentum=0.0,
            weight_decay=0.0,
        )
    return torch.optim.Adam(
        model.parameters(),
        lr=float(args.stage2_lr),
    )


def _require_direct_cost_resume_protocol(checkpoint, args):
    actual = checkpoint.get("stage2_training_protocol_semantics")
    if actual != DIRECT_COST_TRAINING_PROTOCOL_SEMANTICS:
        raise ValueError(
            "Stage 2 D0 training protocol semantics mismatch; restart from Stage 1"
        )
    expected_optimizer = _direct_cost_optimizer_name(args)
    stored_optimizer = checkpoint.get("stage2_optimizer")
    if stored_optimizer != expected_optimizer:
        raise ValueError(
            "Stage 2 optimizer mismatch; restart from Stage 1: "
            f"checkpoint={stored_optimizer!r}, current={expected_optimizer!r}"
        )
    stored = checkpoint.get("direct_cost_config") or {}
    current = _direct_cost_config_from_args(args)
    mutable_budget_keys = {"epochs", "max_updates"}
    for key, value in current.items():
        if key in mutable_budget_keys:
            continue
        stored_value = stored.get(key)
        if isinstance(value, float):
            equal = stored_value is not None and np.isclose(
                float(stored_value), value
            )
        else:
            equal = stored_value == value
        if not equal:
            raise ValueError(
                f"Stage 2 resume config mismatch for {key}: "
                f"checkpoint={stored_value!r}, current={value!r}"
            )


def _require_stage2_prior_physical_config(checkpoint, args):
    stored_radius = checkpoint.get("vehicle_radius_meters")
    if stored_radius is None or not np.isclose(
        float(stored_radius), float(args.vehicle_radius_meters)
    ):
        raise ValueError(
            "Stage 1 checkpoint vehicle radius differs from Stage 2 physical config"
        )


def _save_direct_cost_checkpoint(
    path,
    model,
    model_args,
    optimizer,
    args,
    *,
    epoch,
    global_update,
    validation_metrics,
    best_validation_key,
    train_metrics,
    source_generator,
    data_generator,
    source_checkpoint,
    environment_split,
    validation_indices,
    baseline_validation_metrics,
    baseline_snapshot,
    include_optimizer_state,
):
    config = _direct_cost_config_from_args(args)
    save_checkpoint(
        path,
        model,
        model_args,
        stage="stage2_direct_privileged_cost",
        stage2_epoch=int(epoch),
        stage2_global_update=int(global_update),
        optimizer_state_dict=(
            optimizer.state_dict() if include_optimizer_state else None
        ),
        resume_capable=bool(include_optimizer_state),
        direct_cost_config=config,
        direct_cost_train_metrics=train_metrics,
        direct_cost_validation_metrics=validation_metrics,
        best_validation_key=best_validation_key,
        source_generator_state=source_generator.get_state(),
        data_generator_state=data_generator.get_state(),
        source_checkpoint=str(source_checkpoint),
        stage2_objective_semantics=DIRECT_COST_STAGE2_SEMANTICS,
        stage2_context_semantics=DIRECT_COST_CONTEXT_SEMANTICS,
        stage2_training_protocol_semantics=(
            DIRECT_COST_TRAINING_PROTOCOL_SEMANTICS
        ),
        stage2_environment_split=environment_split,
        stage2_validation_data=str(
            Path(
                getattr(args, "stage2_validation_data", "data/dataset1_val")
            ).resolve()
        ),
        stage2_validation_split=str(
            getattr(args, "stage2_validation_split", "train")
        ),
        fixed_validation_indices=list(validation_indices),
        direct_cost_baseline_validation_metrics=baseline_validation_metrics,
        direct_cost_baseline_snapshot=baseline_snapshot,
        privileged_cost_contract=privileged_cost_contract(),
        privileged_cost_semantics=PRIVILEGED_COST_SEMANTICS,
        stage2_use_mgda=bool(getattr(args, "stage2_use_mgda", True)),
        stage2_optimizer=_direct_cost_optimizer_name(args),
        mgda_gradient_semantics=(
            "raw_FSK_full_parameter_exact_active_set_v1"
            if bool(getattr(args, "stage2_use_mgda", True))
            else "not_used_fixed_weight_task_cost"
        ),
        mask_generation_semantics=MASK_GENERATION_SEMANTICS,
        mask_seed=args.mask_seed,
        p_mask=args.stage2_p_mask,
        stage1_p_mask=getattr(args, "p_mask", None),
        mask_noise_std=MASK_NOISE_STD,
        vehicle_radius_meters=args.vehicle_radius_meters,
        input_mask_semantics=MASK_INPUT_SEMANTICS,
        demo_target_semantics=DEMO_TARGET_SEMANTICS,
        representation_semantic_version=REPRESENTATION_SEMANTIC_VERSION,
        uses_stage1_anchor=False,
        uses_replay=False,
        uses_expert=False,
        uses_cost_only_loss=True,
        inference_uses_privileged_map=False,
        inference_uses_optimizer=False,
    )


def train_stage2(args, device):
    """Stage 2: direct privileged task-cost backprop at deployed one-step output."""
    output = Path(args.fileDir)
    output.mkdir(parents=True, exist_ok=True)
    use_mgda = bool(getattr(args, "stage2_use_mgda", True))

    if args.resume:
        model, model_args, checkpoint = load_model(args.resume, device)
        _require_current_mask_semantics(checkpoint)
        _require_demo_target_semantics(checkpoint)
        _require_mask_configuration(
            checkpoint,
            args,
            p_mask=args.stage2_p_mask,
        )
        _require_privileged_cost_semantics(checkpoint)
        if checkpoint.get("stage") != "stage2_direct_privileged_cost":
            raise ValueError(
                "--resume 必须指向 stage2_direct_privileged_cost checkpoint"
            )
        if (
            checkpoint.get("stage2_objective_semantics")
            != DIRECT_COST_STAGE2_SEMANTICS
        ):
            raise ValueError("Stage 2 direct-cost objective semantics mismatch")
        if (
            checkpoint.get("stage2_context_semantics")
            != DIRECT_COST_CONTEXT_SEMANTICS
        ):
            raise ValueError("Stage 2 direct-cost context semantics mismatch")
        _require_direct_cost_resume_protocol(checkpoint, args)
        start_epoch = int(checkpoint.get("stage2_epoch", -1)) + 1
        global_update = int(checkpoint.get("stage2_global_update", 0))
        source_checkpoint = checkpoint.get(
            "source_checkpoint", args.resume
        )
    else:
        if not args.prior_checkpoint:
            raise ValueError(
                "Stage 2 必须通过 --prior_checkpoint 指定 Stage 1 模型"
            )
        model, model_args, checkpoint = load_model(
            args.prior_checkpoint, device
        )
        _require_current_mask_semantics(checkpoint)
        _require_demo_target_semantics(checkpoint)
        _require_stage2_prior_physical_config(checkpoint, args)
        start_epoch = 0
        global_update = 0
        source_checkpoint = args.prior_checkpoint
    _require_main_method_model(model)
    model.use_gradient_checkpoint = False
    # Disable dropout while preserving parameter gradients so the optimized
    # forward pass exactly matches deployed one-step inference.
    model.eval()

    optimizer = _make_direct_cost_optimizer(model, args)
    if args.resume and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    environment_split = stage2_environment_selection(args)
    if args.resume:
        stored_split = checkpoint.get("stage2_environment_split")
        if stored_split != environment_split:
            raise ValueError(
                "Stage 2 environment split differs from the resumed checkpoint"
            )
    train_base_set, train_envs = make_partial_dataset(
        args.dataFolder,
        "train",
        compute_stability_map=True,
        mask_seed=args.mask_seed,
        p_mask=args.stage2_p_mask,
        mask_mode="stage2_independent",
        vehicle_radius_meters=args.vehicle_radius_meters,
        environment_names=environment_split["train"],
        dynamic_mask_noise=False,
    )
    train_epoch_set = DirectCostEpochDataset(train_base_set)
    train_set = train_epoch_set
    val_set, val_envs = make_partial_dataset(
        args.stage2_validation_data,
        args.stage2_validation_split,
        compute_stability_map=True,
        mask_seed=args.mask_seed,
        p_mask=args.stage2_p_mask,
        mask_mode="stage2_independent",
        vehicle_radius_meters=args.vehicle_radius_meters,
        environment_names=environment_split["validation"],
        dynamic_mask_noise=False,
        expected_environment_count=None,
        compute_stability_if_missing=True,
    )
    if args.max_contexts is not None:
        train_set = Subset(
            train_set, range(min(len(train_set), args.max_contexts))
        )
    validation_count = int(args.stage2_validation_contexts)
    validation_indices = fixed_validation_indices_per_environment(
        val_set,
        val_envs,
        count=validation_count,
        seed=args.seed + 70_000,
    )
    if args.resume:
        stored_indices = checkpoint.get("fixed_validation_indices")
        if list(stored_indices or []) != list(validation_indices):
            raise ValueError(
                "Fixed Stage 2 validation conditions differ from checkpoint"
            )
    val_set = Subset(val_set, validation_indices)

    data_generator = torch.Generator().manual_seed(args.seed + 60_000)
    source_generator = torch.Generator(device=device).manual_seed(
        args.seed + 80_000
    )
    if args.resume:
        if checkpoint.get("data_generator_state") is not None:
            data_generator.set_state(
                checkpoint["data_generator_state"].cpu()
            )
        if checkpoint.get("source_generator_state") is not None:
            source_generator.set_state(
                checkpoint["source_generator_state"].cpu()
            )
    train_loader = DataLoader(
        train_set,
        batch_size=args.stage2_batch_size,
        shuffle=True,
        generator=data_generator,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.stage2_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    writer = _make_tensorboard_writer(
        output,
        "stage2_direct_cost",
        purge_step=global_update if args.resume else 0,
    )

    history = []
    train_epoch_history = []
    if args.resume:
        baseline_snapshot = checkpoint.get("direct_cost_baseline_snapshot")
        baseline_validation_metrics = checkpoint.get(
            "direct_cost_baseline_validation_metrics"
        )
        if baseline_snapshot is None or baseline_validation_metrics is None:
            raise ValueError(
                "Stage 2 resume checkpoint lacks the frozen paired baseline"
            )
        validation_metrics = evaluate_direct_cost_stage2(
            model,
            val_loader,
            device,
            sources_per_context=args.stage2_validation_sources,
            seed=args.seed + 90_000,
            baseline_snapshot=baseline_snapshot,
        )
    else:
        validation_metrics, baseline_snapshot = evaluate_direct_cost_stage2(
            model,
            val_loader,
            device,
            sources_per_context=args.stage2_validation_sources,
            seed=args.seed + 90_000,
            return_snapshot=True,
        )
        baseline_validation_metrics = dict(validation_metrics)
    history.append(
        {
            "epoch": start_epoch - 1,
            "update": global_update,
            "validation": validation_metrics,
        }
    )
    print(
        "direct-cost baseline "
        f"update={global_update} "
        f"Safe@1={validation_metrics['safe_at_1']:.1%} "
        f"strict={validation_metrics['strict_valid_rate']:.1%} "
        f"cost={validation_metrics['task_cost']:.5f}"
    )
    _log_numeric_tree(
        writer,
        "stage2_direct_cost/validation",
        validation_metrics,
        global_update,
    )
    if args.resume:
        best_key = tuple(
            checkpoint.get(
                "best_validation_key",
                _direct_cost_validation_key(
                    validation_metrics,
                    args.stage2_max_regression_rate,
                ),
            )
        )
    else:
        # The untouched Stage-1 checkpoint is a valid Stage-2 candidate.
        # Including update 0 prevents a regressed fine-tune from being called
        # "best" merely because it was the first updated checkpoint.
        best_key = _direct_cost_validation_key(
            validation_metrics,
            args.stage2_max_regression_rate,
        )
        _save_direct_cost_checkpoint(
            output / "stage2_best.pth",
            model,
            model_args,
            optimizer,
            args,
            epoch=-1,
            global_update=global_update,
            validation_metrics=validation_metrics,
            best_validation_key=best_key,
            train_metrics=None,
            source_generator=source_generator,
            data_generator=data_generator,
            source_checkpoint=source_checkpoint,
            environment_split=environment_split,
            validation_indices=validation_indices,
            baseline_validation_metrics=baseline_validation_metrics,
            baseline_snapshot=baseline_snapshot,
            include_optimizer_state=False,
        )
    last_eval_update = global_update
    stop_training = False
    last_train_metrics = None

    for epoch in range(start_epoch, int(args.stage2_epochs)):
        train_epoch_set.set_epoch(epoch)
        epoch_cost_metric_names = (
            "task_cost",
            "forbidden_cost",
            "stability_cost",
            "curvature_cost",
        )
        epoch_source_samples = {
            metric_name: [] for metric_name in epoch_cost_metric_names
        }
        epoch_condition_samples = {
            metric_name: [] for metric_name in epoch_cost_metric_names
        }
        epoch_environment_condition_samples = {
            metric_name: {
                environment_index: []
                for environment_index in range(len(train_envs))
            }
            for metric_name in epoch_cost_metric_names
        }
        epoch_sums = {
            "task_cost": 0.0,
            "forbidden_cost": 0.0,
            "stability_cost": 0.0,
            "curvature_cost": 0.0,
            "gradient_norm_pre_clip": 0.0,
            "gradient_norm_post_clip": 0.0,
            "gradient_clip_scale": 0.0,
            "minimum_analytic_speed": 0.0,
            "maximum_analytic_curvature": 0.0,
            "maximum_raw_turning_violation": 0.0,
            "minimum_yaw_backward_scale": 0.0,
        }
        epoch_updates = 0
        log_every_updates = int(args.stage2_log_every_updates)
        progress = tqdm(
            train_loader,
            desc=f"Stage 2 direct cost epoch {epoch + 1}",
            mininterval=10.0,
            miniters=log_every_updates,
        )
        for batch in progress:
            if (
                args.stage2_max_updates is not None
                and global_update >= int(args.stage2_max_updates)
            ):
                stop_training = True
                break
            optimizer.zero_grad(set_to_none=True)
            output_batch = _direct_cost_forward(
                model,
                batch,
                device,
                sources_per_context=args.stage2_sources_per_context,
                source_generator=source_generator,
            )
            loss = output_batch["task_cost"].mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Non-finite direct cost at update {global_update + 1}"
                )
            if use_mgda:
                mgda_details = _direct_cost_mgda_update(
                    model,
                    output_batch,
                    device,
                )
            else:
                mgda_details = None
                loss.backward()
            if float(args.stage2_grad_clip_norm) > 0.0:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(args.stage2_grad_clip_norm),
                )
            else:
                squared = [
                    parameter.grad.detach().square().sum()
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ]
                gradient_norm = torch.sqrt(torch.stack(squared).sum())
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError(
                    f"Non-finite gradient at update {global_update + 1}"
                )
            optimizer.step()
            global_update += 1
            epoch_updates += 1
            components = output_batch["components"]
            context_count = int(output_batch["contexts"])
            source_count = int(output_batch["sources_per_context"])
            batch_environment_indices = (
                batch["env_index"].detach().cpu().reshape(-1).tolist()
            )
            metric_values = {
                "task_cost": output_batch["task_cost"],
                "forbidden_cost": components["weighted_forbidden_region"],
                "stability_cost": components["weighted_stability"],
                "curvature_cost": components["weighted_curvature"],
            }
            if len(batch_environment_indices) != context_count:
                raise RuntimeError(
                    "Stage 2 batch environment metadata does not match "
                    "the number of direct-cost contexts"
                )
            for metric_name, values in metric_values.items():
                values = values.detach().float().reshape(
                    context_count, source_count
                )
                source_values = values.reshape(-1).cpu()
                condition_values = values.mean(dim=1).cpu()
                epoch_source_samples[metric_name].append(source_values)
                epoch_condition_samples[metric_name].append(condition_values)
                for environment_index, condition_value in zip(
                    batch_environment_indices,
                    condition_values.tolist(),
                ):
                    epoch_environment_condition_samples[metric_name][
                        int(environment_index)
                    ].append(float(condition_value))
            gradient_norm_value = float(gradient_norm.detach())
            clip_limit = float(args.stage2_grad_clip_norm)
            if clip_limit > 0.0:
                gradient_norm_post_clip = min(
                    gradient_norm_value,
                    clip_limit,
                )
                gradient_clip_scale = min(
                    1.0,
                    clip_limit / max(gradient_norm_value, 1e-12),
                )
            else:
                gradient_norm_post_clip = gradient_norm_value
                gradient_clip_scale = 1.0
            row = {
                "task_cost": float(loss.detach()),
                "forbidden_cost": float(
                    components["weighted_forbidden_region"].mean().detach()
                ),
                "stability_cost": float(
                    components["weighted_stability"].mean().detach()
                ),
                "curvature_cost": float(
                    components["weighted_curvature"].mean().detach()
                ),
                "gradient_norm_pre_clip": gradient_norm_value,
                "gradient_norm_post_clip": gradient_norm_post_clip,
                "gradient_clip_scale": gradient_clip_scale,
                "minimum_analytic_speed": float(
                    components["analytic_speed_min"].amin().detach()
                ),
                "maximum_analytic_curvature": float(
                    components["curvature_max"].amax().detach()
                ),
                "maximum_raw_turning_violation": float(
                    components["raw_turning_violation_max"].amax().detach()
                ),
                "minimum_yaw_backward_scale": float(
                    components["analytic_yaw_backward_scale_min"]
                    .amin()
                    .detach()
                ),
            }
            for key, value in row.items():
                epoch_sums[key] += value
                writer.add_scalar(f"stage2_direct_cost/train/{key}", value, global_update)
            if mgda_details is not None:
                for key, value in mgda_details.items():
                    if key == "mgda_active_set":
                        continue
                    writer.add_scalar(
                        f"stage2_direct_cost/train/{key}",
                        float(value),
                        global_update,
                    )
            writer.add_scalar(
                "stage2_direct_cost/train/learning_rate",
                optimizer.param_groups[0]["lr"],
                global_update,
            )
            if (
                global_update == 1
                or global_update % log_every_updates == 0
            ):
                progress.set_postfix(
                    cost=f"{row['task_cost']:.4f}",
                    grad=f"{row['gradient_norm_pre_clip']:.2f}",
                )

            if (
                global_update % int(args.stage2_eval_every_updates) == 0
            ):
                validation_metrics = evaluate_direct_cost_stage2(
                    model,
                    val_loader,
                    device,
                    sources_per_context=args.stage2_validation_sources,
                    seed=args.seed + 90_000,
                    baseline_snapshot=baseline_snapshot,
                )
                last_eval_update = global_update
                history.append(
                    {
                        "epoch": epoch,
                        "update": global_update,
                        "validation": validation_metrics,
                    }
                )
                _log_numeric_tree(
                    writer,
                    "stage2_direct_cost/validation",
                    validation_metrics,
                    global_update,
                )
                current_key = _direct_cost_validation_key(
                    validation_metrics,
                    args.stage2_max_regression_rate,
                )
                regression_admissible = _direct_cost_validation_is_admissible(
                    validation_metrics,
                    args.stage2_max_regression_rate,
                )
                if global_update % log_every_updates == 0:
                    print(
                        f"\ndirect-cost update={global_update} "
                        f"Safe@1={validation_metrics['safe_at_1']:.1%} "
                        f"Safe@K={validation_metrics['safe_at_k']:.1%} "
                        f"strict={validation_metrics['strict_valid_rate']:.1%} "
                        f"I->S={validation_metrics['invalid_to_strict_rate']:.1%} "
                        f"S->I={validation_metrics['strict_to_invalid_rate']:.1%} "
                        f"curv_reg={validation_metrics['curvature_regression_rate']:.1%} "
                        f"regression_admissible={regression_admissible} "
                        f"Vmax={validation_metrics['violation_max_normalized']:.4f} "
                        f"cost={validation_metrics['task_cost']:.5f}"
                    )
                if current_key > best_key:
                    best_key = current_key
                    _save_direct_cost_checkpoint(
                        output / "stage2_best.pth",
                        model,
                        model_args,
                        optimizer,
                        args,
                        epoch=epoch,
                        global_update=global_update,
                        validation_metrics=validation_metrics,
                        best_validation_key=best_key,
                        train_metrics=row,
                        source_generator=source_generator,
                        data_generator=data_generator,
                        source_checkpoint=source_checkpoint,
                        environment_split=environment_split,
                        validation_indices=validation_indices,
                        baseline_validation_metrics=baseline_validation_metrics,
                        baseline_snapshot=baseline_snapshot,
                        include_optimizer_state=False,
                    )
                writer.flush()

        if epoch_updates:
            epoch_statistics = _direct_cost_epoch_statistics(
                epoch_source_samples,
                epoch_condition_samples,
                epoch_environment_condition_samples,
                train_envs,
            )
            train_epoch_history.append(
                {
                    "epoch": int(epoch + 1),
                    "update": int(global_update),
                    "statistics": epoch_statistics,
                }
            )
            _log_numeric_tree(
                writer,
                "stage2_direct_cost/train_epoch",
                epoch_statistics,
                epoch + 1,
            )
            last_train_metrics = {
                key: value / epoch_updates
                for key, value in epoch_sums.items()
            }
            last_train_metrics["epoch_statistics"] = epoch_statistics
        if last_eval_update != global_update:
            validation_metrics = evaluate_direct_cost_stage2(
                model,
                val_loader,
                device,
                sources_per_context=args.stage2_validation_sources,
                seed=args.seed + 90_000,
                baseline_snapshot=baseline_snapshot,
            )
            last_eval_update = global_update
            history.append(
                {
                    "epoch": epoch,
                    "update": global_update,
                    "validation": validation_metrics,
                }
            )
            current_key = _direct_cost_validation_key(
                validation_metrics,
                args.stage2_max_regression_rate,
            )
            if current_key > best_key:
                best_key = current_key
                _save_direct_cost_checkpoint(
                    output / "stage2_best.pth",
                    model,
                    model_args,
                    optimizer,
                    args,
                    epoch=epoch,
                    global_update=global_update,
                    validation_metrics=validation_metrics,
                    best_validation_key=best_key,
                    train_metrics=last_train_metrics,
                    source_generator=source_generator,
                    data_generator=data_generator,
                    source_checkpoint=source_checkpoint,
                    environment_split=environment_split,
                    validation_indices=validation_indices,
                    baseline_validation_metrics=baseline_validation_metrics,
                    baseline_snapshot=baseline_snapshot,
                    include_optimizer_state=False,
                )
        if epoch_updates:
            _log_numeric_tree(
                writer,
                "stage2_direct_cost/validation_epoch",
                validation_metrics,
                epoch + 1,
            )
        _save_direct_cost_checkpoint(
            output / "stage2_last.pth",
            model,
            model_args,
            optimizer,
            args,
            epoch=epoch,
            global_update=global_update,
            validation_metrics=validation_metrics,
            best_validation_key=best_key,
            train_metrics=last_train_metrics,
            source_generator=source_generator,
            data_generator=data_generator,
            source_checkpoint=source_checkpoint,
            environment_split=environment_split,
            validation_indices=validation_indices,
            baseline_validation_metrics=baseline_validation_metrics,
            baseline_snapshot=baseline_snapshot,
            include_optimizer_state=True,
        )
        writer.flush()
        if stop_training:
            break

    method = {
        "method": (
            "boundary_constrained_path_meanflow_"
            "direct_differentiable_privileged_cost"
        ),
        "stage1": "conditional_path_meanflow_pretraining",
        "stage2": (
            "direct_deployment_point_raw_FSK_mgda_backpropagation"
            if use_mgda
            else "direct_deployment_point_task_cost_backpropagation"
        ),
        "stage2_objective_semantics": DIRECT_COST_STAGE2_SEMANTICS,
        "stage2_context_semantics": DIRECT_COST_CONTEXT_SEMANTICS,
        "stage2_training_protocol_semantics": (
            DIRECT_COST_TRAINING_PROTOCOL_SEMANTICS
        ),
        "stage2_use_mgda": use_mgda,
        "stage2_optimizer": _direct_cost_optimizer_name(args),
        "mgda_gradient_semantics": (
            {
                "enabled": True,
                "objectives": [
                    "forbidden_region",
                    "stability",
                    "analytic_curvature",
                ],
                "gradient_space": "shared_generator_full_parameters",
                "solver": "exact_three_objective_active_set",
                "update": "MGDA_combined_gradient_then_plain_SGD_step",
            }
            if use_mgda
            else {
                "enabled": False,
                "update": "fixed_weight_task_cost_then_Adam_step",
            }
        ),
        "optimized_sampling_point": {"t": 1.0, "r": 0.0},
        "privileged_cost_contract": privileged_cost_contract(),
        "privileged_cost_semantics": PRIVILEGED_COST_SEMANTICS,
        "train_environments": train_envs,
        "validation_environments": val_envs,
        "sealed_environments": environment_split["unused"],
        "validation_unused_environments": environment_split[
            "validation_unused"
        ],
        "train_data": str(Path(args.dataFolder).resolve()),
        "validation_data": str(Path(args.stage2_validation_data).resolve()),
        "validation_split": str(args.stage2_validation_split),
        "environment_split": environment_split,
        "fixed_validation_indices": validation_indices,
        "checkpoint_selection": (
            "minimum_validation_task_cost_then_lexicographic_max("
            "Safe@1, strict_valid_rate)"
        ),
        "regression_gate_for_checkpoint_selection": False,
        "regression_metrics_are_diagnostic": True,
        "max_regression_rate": float(args.stage2_max_regression_rate),
        "baseline_validation_metrics": baseline_validation_metrics,
        "uses_stage1_anchor": False,
        "uses_replay": False,
        "uses_expert": False,
        "uses_cost_only_loss": True,
        "uses_ot": False,
        "inference_uses_privileged_map": False,
        "inference_uses_optimizer": False,
        "history": history,
        "train_epoch_history": train_epoch_history,
        "best_validation_key": best_key,
        "global_update": global_update,
    }
    with open(
        output / "stage2_method.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(method, handle, indent=2, ensure_ascii=False)
    writer.close()
    return output / "stage2_last.pth"


def run_workflow(args):
    """运行 Stage 1、Stage 2，或连续运行完整两阶段训练。"""
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_selected_stage2():
        return train_stage2(args, device)

    if args.workflow == "stage1":
        train_stage1(args, device)
    elif args.workflow == "stage2":
        if args.stage2_resume:
            if args.resume:
                raise ValueError(
                    "Stage 2 请只使用 --stage2_resume 或 --resume 其中一个 resume checkpoint"
                )
            args.resume = args.stage2_resume
        run_selected_stage2()
    elif args.workflow == "full":
        stage1_checkpoint = train_stage1(args, device)
        args.prior_checkpoint = str(stage1_checkpoint)
        args.resume = args.stage2_resume
        run_selected_stage2()
    else:
        raise ValueError(f"不支持的 workflow: {args.workflow}")

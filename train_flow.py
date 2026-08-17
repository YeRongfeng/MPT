"""Boundary-Constrained Path MeanFlow 两阶段训练入口。

Stage 1 训练条件 Path MeanFlow。Stage 2 从 Stage 1 checkpoint 继续训练，
在部署点直接反向传播 privileged task cost。
"""

import argparse

from map_config import MAP_CONFIG, SAFETY_COST_CONFIG
from posterior_pipeline import run_workflow
from boundary_constrained_path import (
    PATH_REPRESENTATION_SEMANTIC_VERSION,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Boundary-Constrained Path MeanFlow with direct privileged cost"
        )
    )
    parser.add_argument(
        "--workflow",
        choices=["stage1", "stage2", "full"],
        default="stage1",
        help="训练 Stage 1、继续训练 Stage-2 生成器，或连续执行两个阶段。",
    )
    parser.add_argument(
        "--vehicle_radius_meters",
        type=float,
        default=SAFETY_COST_CONFIG.vehicle_radius_meters,
        help="mask 配置空间腐蚀使用的车辆保守圆半径，必须按真实几何设置。",
    )
    parser.add_argument("--dataFolder", default=str(MAP_CONFIG.dataset_root))
    parser.add_argument(
        "--fileDir",
        default="data/boundary_constrained_path_meanflow_v1",
        help=(
            "输出目录；当前表示版本为 "
            f"{PATH_REPRESENTATION_SEMANTIC_VERSION}。"
        ),
    )
    parser.add_argument("--batchSize", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mask_seed", type=int, default=2026)
    parser.add_argument(
        "--p_mask",
        type=float,
        default=0.5,
        help=(
            "样本级 Bernoulli 加 mask 的概率（不是逐像素概率）；"
            "采样为 0 时使用全 1 mask。"
        ),
    )
    parser.add_argument(
        "--max_contexts",
        type=int,
        default=None,
        help="仅用于调试：限制可访问的训练/验证 context 数。",
    )

    # Stage 1：示范轨迹 MeanFlow。
    parser.add_argument("--resume", default=None)
    parser.add_argument("--stage1_epochs", type=int, default=20)
    parser.add_argument("--stage1_lr", type=float, default=1e-4)
    parser.add_argument(
        "--stage1_max_updates",
        type=int,
        default=None,
        help="诊断用：本次 Stage 1 最多执行的 optimizer updates。",
    )
    parser.add_argument(
        "--stage1_split_seed",
        type=int,
        default=None,
        help=(
            "启用按物理地形互斥的 Stage-1 train/validation 划分；"
            "不提供时保留旧的同地形路径划分，仅用于兼容旧 checkpoint。"
        ),
    )
    parser.add_argument(
        "--stage1_train_environments",
        type=int,
        default=80,
        help="启用 --stage1_split_seed 时用于训练的地形数量。",
    )
    parser.add_argument(
        "--stage1_val_environments",
        type=int,
        default=20,
        help="启用 --stage1_split_seed 时用于验证的未见地形数量。",
    )
    parser.add_argument(
        "--stage1_mask_noise_mode",
        choices=["fixed", "legacy_random"],
        default="legacy_random",
        help=(
            "fixed 为每个样本固定遮挡区法向噪声，适合严格配对训练比较；"
            "legacy_random 保留旧训练的逐次非确定噪声。"
        ),
    )
    parser.add_argument(
        "--stage1_endpoint_curvature_weight",
        type=float,
        default=0.0,
        help=(
            "部署点 t=1,r=0 输出的 1001 点解析 curvature 辅助项权重；"
            "0 表示原始 MeanFlow 对照。"
        ),
    )
    parser.add_argument(
        "--stage1_endpoint_curvature_tail_ratio",
        type=float,
        default=0.01,
        help="curvature 相对超限有界 penalty 的最坏点比例。",
    )
    parser.add_argument(
        "--noise_invariance_contexts",
        type=int,
        default=8,
        help="Stage 1 每轮用于遮挡噪声不变性诊断的 context 数；0 表示关闭。",
    )
    parser.add_argument("--noise_invariance_draws", type=int, default=4)

    # Stage 2：在部署点直接反传 privileged task cost。
    parser.add_argument("--prior_checkpoint", default=None)
    parser.add_argument(
        "--stage2_resume",
        default=None,
        help="direct-cost Stage 2 的 resume-capable stage2_last.pth。",
    )
    parser.add_argument("--stage2_epochs", type=int, default=3)
    parser.add_argument(
        "--stage2_use_mgda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Stage 2 默认用精确三目标 MGDA 组合 F/S/K 原始梯度并通过"
            " plain SGD 更新；使用 --no-stage2_use_mgda 回到原 fixed"
            " direct-cost scalarization 和 Adam 更新。"
        ),
    )
    parser.add_argument(
        "--stage2_split_seed",
        type=int,
        default=20260802,
        help="Stage 2 train/独立 validation 环境顺序的冻结种子。",
    )
    parser.add_argument(
        "--stage2_train_environments",
        type=int,
        default=100,
        help=(
            "Stage 2 训练使用的 dataFolder/train 环境数量；"
            "当前完整 development train 为 100。"
        ),
    )
    parser.add_argument(
        "--stage2_validation_data",
        default="data/dataset1_val",
        help=(
            "独立 Stage 2 validation 数据根目录；默认读取"
            " data/dataset1_val/train。"
        ),
    )
    parser.add_argument(
        "--stage2_validation_split",
        choices=["train", "val"],
        default="train",
        help="Stage 2 validation 根目录下使用的 split，默认是 train。",
    )
    parser.add_argument(
        "--stage2_p_mask",
        type=float,
        default=1.0,
        help="Stage 2 独立 mask 的样本级采样概率。",
    )
    parser.add_argument("--stage2_lr", type=float, default=1e-5)
    parser.add_argument(
        "--stage2_batch_size",
        type=int,
        default=4,
        help="每次 direct-cost 更新使用的 context 数。",
    )
    parser.add_argument(
        "--stage2_sources_per_context",
        type=int,
        default=4,
        help="每个 context 用于 direct-cost 反传的 source 数。",
    )
    parser.add_argument(
        "--stage2_grad_clip_norm",
        type=float,
        default=1.0,
        help="参数梯度裁剪范数。",
    )
    parser.add_argument(
        "--stage2_max_updates",
        type=int,
        default=None,
        help=(
            "可选的 Stage 2 全局 optimizer update 上限；省略时由 "
            "--stage2_epochs 控制完整训练。"
        ),
    )
    parser.add_argument(
        "--stage2_eval_every_updates",
        type=int,
        default=100,
        help="每多少次更新在固定 validation contexts 上评估。",
    )
    parser.add_argument(
        "--stage2_log_every_updates",
        type=int,
        default=100,
        help=(
            "每多少次更新刷新一次 Stage 2 终端进度和摘要；"
            "TensorBoard 仍记录每个 update。"
        ),
    )
    parser.add_argument(
        "--stage2_validation_contexts",
        type=int,
        default=50,
        help=(
            "固定 validation context 数；当前每个 validation environment"
            " 选一个，dataset1_val/train 默认使用 50 个。"
        ),
    )
    parser.add_argument(
        "--stage2_validation_environments",
        type=int,
        default=50,
        help=(
            "用于训练期 checkpoint 选择的 validation 环境数量；"
            "dataset1_val/train 当前有 50 个。"
        ),
    )
    parser.add_argument(
        "--stage2_validation_sources",
        type=int,
        default=16,
        help="每个 validation context 的固定候选数。",
    )
    parser.add_argument(
        "--stage2_max_regression_rate",
        type=float,
        default=0.05,
        help="paired validation 中每个 hard validity regression 的最大比例。",
    )

    return parser


def validate_args(args):
    if args.batchSize <= 0:
        raise ValueError("--batchSize 必须为正数")
    if args.max_contexts is not None and args.max_contexts <= 0:
        raise ValueError("--max_contexts 必须为正数")
    if args.workflow in {"stage1", "full"} and args.stage1_epochs <= 0:
        raise ValueError("Stage 1 至少需要一个 epoch")
    if args.stage1_max_updates is not None and args.stage1_max_updates <= 0:
        raise ValueError("--stage1_max_updates 必须为正数")
    if args.stage1_split_seed is not None:
        if args.stage1_train_environments <= 0:
            raise ValueError("--stage1_train_environments 必须为正数")
        if args.stage1_val_environments <= 0:
            raise ValueError("--stage1_val_environments 必须为正数")
        if (
            args.stage1_train_environments + args.stage1_val_environments
            > MAP_CONFIG.expected_environments
        ):
            raise ValueError("Stage 1 训练/验证地形总数超过数据集地形数量")
    if args.stage1_endpoint_curvature_weight < 0.0:
        raise ValueError("--stage1_endpoint_curvature_weight 不能为负数")
    if not 0.0 < args.stage1_endpoint_curvature_tail_ratio <= 1.0:
        raise ValueError(
            "--stage1_endpoint_curvature_tail_ratio 必须在 (0,1]"
        )
    if args.workflow in {"stage2", "full"} and args.stage2_epochs <= 0:
        raise ValueError("Stage 2 至少需要一个 epoch")
    positive_values = {"stage1_lr": args.stage1_lr}
    if args.workflow in {"stage2", "full"}:
        positive_values.update({
            "stage2_lr": args.stage2_lr,
            "stage2_batch_size": args.stage2_batch_size,
            "stage2_sources_per_context": args.stage2_sources_per_context,
            "stage2_grad_clip_norm": args.stage2_grad_clip_norm,
            "stage2_eval_every_updates": args.stage2_eval_every_updates,
            "stage2_log_every_updates": args.stage2_log_every_updates,
            "stage2_validation_contexts": args.stage2_validation_contexts,
            "stage2_validation_environments": args.stage2_validation_environments,
            "stage2_validation_sources": args.stage2_validation_sources,
        })
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"--{name} 必须为正数")
    if args.stage2_max_updates is not None and args.stage2_max_updates <= 0:
        raise ValueError("--stage2_max_updates 必须为正数")
    if args.workflow in {"stage2", "full"}:
        if not 0.0 <= args.stage2_p_mask <= 1.0:
            raise ValueError("--stage2_p_mask 必须在 [0,1]")
        if args.stage2_train_environments <= 0:
            raise ValueError("--stage2_train_environments 必须为正数")
        if args.stage2_validation_environments <= 0:
            raise ValueError("--stage2_validation_environments 必须为正数")
        if args.stage2_train_environments > MAP_CONFIG.expected_environments:
            raise ValueError(
                "Stage 2 train 环境数量超过 dataFolder/train 的配置数量"
            )
        if args.stage2_validation_contexts > args.stage2_validation_environments:
            raise ValueError(
                "当前 Stage 2 每个 validation environment 只选一个 context，"
                "因此 --stage2_validation_contexts 不能超过"
                " --stage2_validation_environments"
            )
        if not 0.0 <= args.stage2_max_regression_rate <= 1.0:
            raise ValueError("--stage2_max_regression_rate 必须在 [0,1]")
    if args.vehicle_radius_meters < 0.0:
        raise ValueError("--vehicle_radius_meters 不能为负数")
    if args.noise_invariance_contexts < 0:
        raise ValueError("--noise_invariance_contexts 不能为负数")
    if args.noise_invariance_contexts > 0 and args.noise_invariance_draws < 2:
        raise ValueError("启用噪声诊断时 --noise_invariance_draws 至少为 2")
    for name in ("p_mask",):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name} 必须在 [0,1]")


def main():
    args = build_parser().parse_args()
    validate_args(args)
    run_workflow(args)


if __name__ == "__main__":
    main()

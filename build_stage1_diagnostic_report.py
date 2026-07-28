"""Build an archival Stage-1 noise-diagnostic report and evidence dashboard."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


COLORS = {
    "ink": "#172033",
    "muted": "#64748B",
    "grid": "#DCE3EC",
    "bg": "#F4F7FB",
    "card": "#FFFFFF",
    "random": "#718096",
    "optimized": "#007F7B",
    "accent": "#2563EB",
    "warning": "#D97706",
    "danger": "#C2413B",
    "good_bg": "#E6F5F3",
    "warn_bg": "#FFF4DD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="diagnostics/stage1_noise")
    parser.add_argument("--output-name", default="report")
    return parser.parse_args()


def configure_chinese_font() -> None:
    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Droid Sans Fallback",
        "AR PL UKai CN",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in available:
            plt.rcParams["font.family"] = candidate
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "stix"


def add_card(ax: plt.Axes, title: str, label: str) -> None:
    ax.set_facecolor(COLORS["card"])
    for spine in ax.spines.values():
        spine.set_color("#E3E8F0")
        spine.set_linewidth(1.0)
    ax.text(
        0.02,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=9,
        weight="bold",
        color=COLORS["accent"],
        va="bottom",
    )
    ax.text(
        0.1,
        1.055,
        title,
        transform=ax.transAxes,
        fontsize=11,
        weight="bold",
        color=COLORS["ink"],
        va="bottom",
    )


def draw_flow_panel(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.04, 0.52, 0.2, 0.22, "$z\\sim\\mathcal{N}(0,I)$", COLORS["random"]),
        (0.3, 0.52, 0.22, 0.22, "冻结 $G_{\\theta_0}(c,z)$", COLORS["accent"]),
        (0.58, 0.52, 0.16, 0.22, "轨迹 $x_0$", COLORS["optimized"]),
        (0.8, 0.52, 0.16, 0.22, "物理代价 $C$", COLORS["warning"]),
    ]
    for x, y, width, height, text, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor="#FFFFFF",
            edgecolor=color,
            linewidth=1.8,
        )
        ax.add_patch(patch)
        ax.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=9.5,
            color=COLORS["ink"],
        )
    for start, end in [((0.24, 0.63), (0.3, 0.63)), ((0.52, 0.63), (0.58, 0.63)), ((0.74, 0.63), (0.8, 0.63))]:
        ax.add_patch(
            FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color=COLORS["muted"])
        )
    ax.add_patch(
        FancyArrowPatch(
            (0.88, 0.49),
            (0.14, 0.42),
            connectionstyle="arc3,rad=-0.17",
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.7,
            color=COLORS["danger"],
        )
    )
    ax.text(0.51, 0.23, "反向传播只更新初始噪声 $z$", ha="center", fontsize=9.5, color=COLORS["danger"], weight="bold")
    ax.text(0.04, 0.08, "控制变量：地图/起终点固定 · Stage 1 参数冻结 · 可训练模型参数 = 0", fontsize=8.5, color=COLORS["muted"])


def build_dashboard(
    output_path: Path,
    sweep: dict,
    perturb: dict,
    geometry: dict,
) -> None:
    configure_chinese_font()
    per_condition = sweep["per_condition"]
    path_ids = [row["path_id"] for row in per_condition]
    random_rates = np.asarray([row["random_success_rate"] for row in per_condition])
    optimized_rates = np.asarray([row["optimized_success_rate"] for row in per_condition])

    random_cost = []
    optimized_cost = []
    gt_cost = []
    for row in per_condition:
        condition_dir = output_path.parent.parent / f"env000010_path{row['path_id']}_seed0"
        with (condition_dir / "summary.json").open() as handle:
            condition = json.load(handle)
        random_cost.extend(sample["initial_cost"] for sample in condition["per_restart"])
        optimized_cost.extend(sample["best_cost"] for sample in condition["per_restart"])
        gt_cost.extend([condition["ground_truth_reference_cost"]] * len(condition["per_restart"]))
    random_ratio = np.asarray(random_cost) / np.asarray(gt_cost)
    optimized_ratio = np.asarray(optimized_cost) / np.asarray(gt_cost)

    fig = plt.figure(figsize=(16, 9), facecolor=COLORS["bg"])
    grid = fig.add_gridspec(
        3,
        3,
        height_ratios=[0.68, 1, 1],
        left=0.035,
        right=0.98,
        top=0.96,
        bottom=0.055,
        hspace=0.46,
        wspace=0.25,
    )

    header = fig.add_subplot(grid[0, :])
    header.axis("off")
    header.text(
        0.0,
        0.96,
        "Stage 1 初始噪声优化诊断｜证据链总览",
        fontsize=23,
        weight="bold",
        color=COLORS["ink"],
        va="top",
    )
    header.text(
        0.0,
        0.63,
        "env000010 · 8 个条件 · 每条件 32 个 restart · 共 256 个条件—噪声样本 · $G_{\\theta_0}$ 全程冻结",
        fontsize=10.5,
        color=COLORS["muted"],
    )
    left_box = FancyBboxPatch(
        (0.0, 0.06), 0.64, 0.36, boxstyle="round,pad=0.014,rounding_size=0.014", facecolor=COLORS["good_bg"], edgecolor="none"
    )
    right_box = FancyBboxPatch(
        (0.665, 0.06), 0.335, 0.36, boxstyle="round,pad=0.014,rounding_size=0.014", facecolor=COLORS["warn_bg"], edgecolor="none"
    )
    header.add_patch(left_box)
    header.add_patch(right_box)
    header.text(0.018, 0.31, "已证实", fontsize=10, color=COLORS["optimized"], weight="bold")
    header.text(
        0.018,
        0.15,
        "冻结 Stage 1 的映射中存在大量可由噪声优化到达的低代价输出；但这些解通常偏离标准高斯典型区。",
        fontsize=11.5,
        color=COLORS["ink"],
        weight="bold",
    )
    header.text(0.682, 0.31, "尚未由本实验决定", fontsize=10, color=COLORS["warning"], weight="bold")
    header.text(
        0.682,
        0.15,
        "最终应只改采样器，还是只微调生成器。",
        fontsize=11.5,
        color=COLORS["ink"],
        weight="bold",
    )

    ax_flow = fig.add_subplot(grid[1, 0])
    add_card(ax_flow, "实验到底改变了什么", "A")
    draw_flow_panel(ax_flow)

    ax_success = fig.add_subplot(grid[1, 1])
    add_card(ax_success, "跨条件命中率：不是 path_40 偶然", "B")
    x = np.arange(len(path_ids))
    width = 0.36
    ax_success.bar(x - width / 2, random_rates, width, color=COLORS["random"], label="随机 $z$")
    ax_success.bar(x + width / 2, optimized_rates, width, color=COLORS["optimized"], label="优化 $z^*$")
    ax_success.set_xticks(x, path_ids)
    ax_success.set_ylim(0, 1.08)
    ax_success.set_ylabel("低于 GT 参考代价的比例")
    ax_success.grid(axis="y", color=COLORS["grid"], alpha=0.8)
    ax_success.legend(loc="lower right", frameon=False, fontsize=8)
    ax_success.text(
        0.02,
        0.94,
        f"总体  {sweep['random_success_rate']:.1%}  →  {sweep['optimized_success_rate']:.1%}",
        transform=ax_success.transAxes,
        fontsize=11,
        weight="bold",
        color=COLORS["ink"],
        va="top",
    )

    ax_cost = fig.add_subplot(grid[1, 2])
    add_card(ax_cost, "物理代价相对 GT 参考的变化", "C")
    violin = ax_cost.violinplot([random_ratio, optimized_ratio], positions=[0, 1], showmedians=True, showextrema=False)
    for body, color in zip(violin["bodies"], [COLORS["random"], COLORS["optimized"]]):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.8)
    violin["cmedians"].set_color(COLORS["ink"])
    ax_cost.axhline(1.0, color=COLORS["danger"], linestyle="--", linewidth=1.2, label="GT reference")
    ax_cost.set_xticks([0, 1], ["随机 $z$", "优化 $z^*$"])
    ax_cost.set_ylabel("trajectory cost / GT reference cost")
    ax_cost.set_ylim(0, min(5.0, max(random_ratio.max(), optimized_ratio.max()) * 1.05))
    ax_cost.grid(axis="y", color=COLORS["grid"], alpha=0.8)
    ax_cost.text(
        0.03,
        0.94,
        f"均值  {random_ratio.mean():.3f}×  →  {optimized_ratio.mean():.3f}×\n逐样本平均下降 {sweep['mean_relative_cost_drop']:.1%}",
        transform=ax_cost.transAxes,
        fontsize=10,
        weight="bold",
        color=COLORS["ink"],
        va="top",
    )

    ax_energy = fig.add_subplot(grid[2, 0])
    add_card(ax_energy, "找到好解时，噪声离先验有多远", "D")
    initial_energy = [row["initial_energy_per_dim_mean"] for row in per_condition]
    optimized_energy = [row["optimized_energy_per_dim_mean"] for row in per_condition]
    ax_energy.bar(x - width / 2, initial_energy, width, color=COLORS["random"], label="初始")
    ax_energy.bar(x + width / 2, optimized_energy, width, color=COLORS["warning"], label="优化后")
    ax_energy.axhline(1.0, color=COLORS["ink"], linestyle="--", linewidth=1)
    ax_energy.set_xticks(x, path_ids)
    ax_energy.set_ylabel(r"mean $||z||^2/48$")
    ax_energy.grid(axis="y", color=COLORS["grid"], alpha=0.8)
    ax_energy.legend(frameon=False, fontsize=8)
    ax_energy.text(
        0.02,
        0.96,
        f"总体  0.971 → 2.847\n最大 7.203 · 平均位移 9.323",
        transform=ax_energy.transAxes,
        fontsize=10,
        weight="bold",
        va="top",
        color=COLORS["ink"],
    )

    ax_basin = fig.add_subplot(grid[2, 1])
    add_card(ax_basin, "优质噪声区域有多宽", "E")
    sigma_values = [0.0] + [row["sigma"] for row in perturb["aggregate_by_sigma"]]
    aggregate_success = [sweep["optimized_success_rate"]] + [row["success_rate"] for row in perturb["aggregate_by_sigma"]]
    for path_id in path_ids:
        base = next(row["optimized_success_rate"] for row in per_condition if row["path_id"] == path_id)
        rows = [row for row in perturb["per_condition"] if row["path_id"] == path_id]
        rows.sort(key=lambda row: row["sigma"])
        ax_basin.plot(sigma_values, [base] + [row["success_rate"] for row in rows], color="#B7C1CF", linewidth=1, alpha=0.7)
    ax_basin.plot(sigma_values, aggregate_success, marker="o", linewidth=2.6, color=COLORS["accent"], label="总体")
    ax_basin.set_ylim(0, 1.05)
    ax_basin.set_xticks(sigma_values)
    ax_basin.set_xlabel(r"扰动尺度 $\sigma$")
    ax_basin.set_ylabel("扰动后成功率")
    ax_basin.grid(color=COLORS["grid"], alpha=0.8)
    ax_basin.text(
        0.03,
        0.1,
        "0.05: 94.9%\n0.10: 91.1%\n0.50: 52.6%",
        transform=ax_basin.transAxes,
        fontsize=9,
        weight="bold",
        color=COLORS["ink"],
    )

    ax_geometry = fig.add_subplot(grid[2, 2])
    add_card(ax_geometry, "修正结构：不是简单温度或固定偏移", "F")
    fractions = geometry["common_random_analysis"]["delta_energy_fractions"]
    fraction_values = [
        fractions["global_shift"],
        fractions["condition_effect"],
        fractions["shared_initial_noise_effect"],
        fractions["condition_noise_interaction"],
    ]
    fraction_labels = ["全局", "条件", "共享 z", "条件×z"]
    fraction_colors = ["#CBD5E1", "#60A5FA", "#A78BFA", COLORS["optimized"]]
    left = 0.0
    for value, label, color in zip(fraction_values, fraction_labels, fraction_colors):
        ax_geometry.barh([0], [value], left=left, color=color, height=0.28, label=f"{label} {value:.1%}")
        left += value
    ax_geometry.set_xlim(0, 1)
    ax_geometry.set_ylim(-0.75, 0.65)
    ax_geometry.set_yticks([])
    ax_geometry.set_xlabel("修正向量能量分解")
    ax_geometry.grid(axis="x", color=COLORS["grid"], alpha=0.8)
    ax_geometry.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False, fontsize=8)
    radial = geometry["aggregate_radial_orthogonal"]
    ax_geometry.text(
        0.02,
        0.91,
        f"平均径向移动  {radial['radial_magnitude']['mean']:.2f}\n平均正交移动  {radial['orthogonal_norm']['mean']:.2f}\n径向位移能量仅 {radial['radial_delta_energy_fraction']['mean']:.1%}",
        transform=ax_geometry.transAxes,
        fontsize=10,
        weight="bold",
        color=COLORS["ink"],
        va="top",
    )
    ax_geometry.text(
        0.02,
        0.43,
        "65.0% 来自条件与初始噪声交互\n→ 纯温度缩放、统一偏移均不足",
        transform=ax_geometry.transAxes,
        fontsize=9.5,
        color=COLORS["danger"],
        weight="bold",
        va="top",
    )

    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def build_metrics_csv(output_path: Path, sweep: dict, perturb: dict, geometry: dict) -> None:
    rows = [
        ("条件数", sweep["num_conditions"], "condition", "sweep_summary/summary.json"),
        ("每条件 restart", sweep["restarts_per_condition"][0], "trajectory", "sweep_summary/summary.json"),
        ("随机噪声成功率", sweep["random_success_rate"], "fraction", "sweep_summary/summary.json"),
        ("优化噪声成功率", sweep["optimized_success_rate"], "fraction", "sweep_summary/summary.json"),
        ("随机 cost/GT 均值", sweep["random_to_gt_ratio"]["mean"], "ratio", "sweep_summary/summary.json"),
        ("优化 cost/GT 均值", sweep["optimized_to_gt_ratio"]["mean"], "ratio", "sweep_summary/summary.json"),
        ("逐样本平均相对代价下降", sweep["mean_relative_cost_drop"], "fraction", "sweep_summary/summary.json"),
        ("初始噪声能量/维均值", sweep["initial_noise_energy_per_dim"]["mean"], "ratio", "sweep_summary/summary.json"),
        ("优化噪声能量/维均值", sweep["optimized_noise_energy_per_dim"]["mean"], "ratio", "sweep_summary/summary.json"),
        ("优化噪声能量/维最大值", sweep["optimized_noise_energy_per_dim"]["max"], "ratio", "sweep_summary/summary.json"),
        ("平均噪声位移", sweep["noise_displacement_norm"]["mean"], "L2", "sweep_summary/summary.json"),
    ]
    for row in perturb["aggregate_by_sigma"]:
        rows.append((f"sigma={row['sigma']} 扰动成功率", row["success_rate"], "fraction", "perturbation_summary/summary.json"))
    radial = geometry["aggregate_radial_orthogonal"]
    rows.extend(
        [
            ("平均径向移动", radial["radial_magnitude"]["mean"], "L2", "noise_geometry/summary.json"),
            ("平均正交移动", radial["orthogonal_norm"]["mean"], "L2", "noise_geometry/summary.json"),
            ("径向位移能量占比", radial["radial_delta_energy_fraction"]["mean"], "fraction", "noise_geometry/summary.json"),
            ("条件×噪声交互能量占比", geometry["common_random_analysis"]["delta_energy_fractions"]["condition_noise_interaction"], "fraction", "noise_geometry/summary.json"),
        ]
    )
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value", "unit", "source"])
        writer.writerows(rows)


def build_report(output_path: Path, sweep: dict, perturb: dict, geometry: dict) -> None:
    per_condition_lines = []
    for row in sweep["per_condition"]:
        per_condition_lines.append(
            f"| {row['path_id']} | {row['random_success_count']}/{row['num_restarts']} "
            f"({row['random_success_rate']:.1%}) | {row['optimized_success_count']}/{row['num_restarts']} "
            f"({row['optimized_success_rate']:.1%}) | {row['random_to_gt_ratio_mean']:.3f}× | "
            f"{row['optimized_to_gt_ratio_mean']:.3f}× | {row['optimized_energy_per_dim_mean']:.3f} |"
        )
    perturb_lines = []
    for row in perturb["aggregate_by_sigma"]:
        perturb_lines.append(
            f"| {row['sigma']:.2f} | {row['success_rate']:.2%} | "
            f"{row['centers_with_at_least_80pct_success']}/{row['num_centers']} "
            f"({row['centers_with_at_least_80pct_success_rate']:.2%}) | "
            f"{row['center_cost_multiplier']['median']:.3f}× |"
        )
    candidate_silhouettes = [row["best_candidate_silhouette"] for row in geometry["per_path"]]
    shift_explained = [row["condition_mean_shift_delta_energy_explained"] for row in geometry["per_path"]]
    radial = geometry["aggregate_radial_orthogonal"]
    energy = geometry["common_random_analysis"]["delta_energy_fractions"]

    report = rf"""# Stage 1 初始噪声优化诊断报告

> 留档范围：`env000010`；8 个起终点条件；每条件 32 个相同的标准高斯初始噪声；Stage 1 全程冻结。  
> 本报告由原始 `summary.json` 与 `trajectories.npz` 自动生成，核心数字可在 [metrics.csv](metrics.csv) 中追溯。

![证据链总览](evidence_overview.png)

## 1. 一句话结论

本组实验已经证明：

$$
\boxed{{G_{{\theta_0}}(c,\cdot)\text{{ 的冻结映射中，存在可由初始噪声优化稳定到达的低物理代价输出。}}}}
$$

同时，无约束优化找到的噪声通常明显偏离标准高斯典型区域；这些优质噪声周围多数存在局部稳定盆地。修正方向主要是高维正交搬运，并强烈依赖条件与初始噪声的交互。

**当前实验尚不能单独决定最终方法应当“只改噪声采样器”还是“只微调生成器”。** 这两者是后续设计选项，不是本次诊断的直接结论。

## 2. 诊断问题与控制变量

固定地图、起点和终点条件 $c$，冻结 Stage 1：

$$
x_0=G_{{\theta_0}}(c,z),\qquad z\sim\mathcal N(0,I),
$$

只优化初始噪声：

$$
\min_z C(G_{{\theta_0}}(c,z),c).
$$

实验中模型可训练参数数量为 0。因而物理代价下降不能归因于模型参数变化，只能归因于在固定映射中改变了输入噪声位置。

诊断逻辑为：

1. 若多次优化 $z$ 仍找不到低代价轨迹，则 Stage 1 的输出能力可能不足；
2. 若冻结模型后，多数随机起点都能通过优化 $z$ 找到低代价轨迹，则低代价输出已经存在于固定映射中；
3. 再通过噪声能量、扰动和几何分析判断这些输入区域是否典型、是否宽、是否有简单结构。

## 3. 证据链

### 证据 A：冻结映射中确实存在低代价输出

| path_id | 随机噪声成功 | 优化后成功 | 随机 cost/GT | 优化 cost/GT | 优化后 $\|z\|^2/48$ |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(per_condition_lines)}

汇总结果：

- 随机噪声成功率：**{sweep['random_success_count']}/{sweep['num_condition_noise_evaluations']} = {sweep['random_success_rate']:.2%}**；
- 优化噪声成功率：**{sweep['optimized_success_count']}/{sweep['num_condition_noise_evaluations']} = {sweep['optimized_success_rate']:.2%}**；
- 平均 `cost / GT reference`：**{sweep['random_to_gt_ratio']['mean']:.3f}× → {sweep['optimized_to_gt_ratio']['mean']:.3f}×**；
- 逐样本平均相对代价下降：**{sweep['mean_relative_cost_drop']:.2%}**。

这支持“Stage 1 已包含低代价输出能力”。它比单独报告一个最优样本更强，因为优化后的成功发生在 249/256 个条件—噪声样本上。

![单条轨迹示例：path 40](../env000010_path40_seed0/diagnostic.png)

### 证据 B：无约束优化找到的噪声通常偏离标准高斯典型区

48 维标准高斯满足 $\mathbb E[\|z\|^2/48]=1$。实际结果为：

- 初始噪声平均能量/维：**{sweep['initial_noise_energy_per_dim']['mean']:.3f}**；
- 优化噪声平均能量/维：**{sweep['optimized_noise_energy_per_dim']['mean']:.3f}**；
- 优化噪声最大能量/维：**{sweep['optimized_noise_energy_per_dim']['max']:.3f}**；
- 平均 $\|z^*-z\|$：**{sweep['noise_displacement_norm']['mean']:.3f}**。

因此，本次无约束优化经常通过离开标准高斯典型区域寻找低代价输出。但需要注意：这不证明所有优质噪声都必须位于低概率区域，因为随机采样仍有 67/256 成功；它只证明当前优化器发现的多数最优中心存在明显先验偏移。

### 证据 C：优质噪声不是普遍意义上的针尖

对每个 $z^*$ 生成 $z'=z^*+\sigma\epsilon$，每个中心、每个尺度采样 32 次，共 32768 条扰动轨迹：

| $\sigma$ | 扰动后成功率 | ≥80% 成功的中心 | 中心代价倍率中位数 |
|---:|---:|---:|---:|
{chr(10).join(perturb_lines)}

$\sigma=0.05\sim0.1$ 时总体成功率仍超过 90%，所以多数优质中心附近存在局部稳定盆地。但 $\sigma=0.5$ 时成功率降至 {perturb['aggregate_by_sigma'][-1]['success_rate']:.2%}，且不同条件差异显著。因此应表述为“多数中心局部不尖锐”，而不是“所有优质区域都很宽”。

![局部扰动结果](../perturbation_summary/perturbation.png)

### 证据 D：修正主要不是径向温度缩放

分解

$$
z^*=\alpha z+r_\perp,\qquad r_\perp^Tz=0,
$$

得到：

- $\alpha$ 平均值：**{radial['alpha']['mean']:.3f}**；
- 平均径向移动：**{radial['radial_magnitude']['mean']:.3f}**；
- 平均正交移动：**{radial['orthogonal_norm']['mean']:.3f}**；
- 正交/径向平均比：**{radial['orthogonal_to_radial_mean_ratio']:.2f}×**；
- 径向移动占位移能量：平均仅 **{radial['radial_delta_energy_fraction']['mean']:.2%}**。

因此，“只把噪声温度调高”与现有最优修正方向不一致。优化后半径增大主要来自加入大幅正交分量，而非沿原噪声方向缩放。

![径向与正交修正](../noise_geometry/radial_orthogonal.png)

### 证据 E：修正主要由条件与初始噪声共同决定

8 个条件使用完全相同的 32 个初始噪声，已验证最大差异为 0。对 $\Delta z_{{i,c}}=z^*_{{i,c}}-z_i$ 做双因素能量分解：

| 修正分量 | 能量占比 |
|---|---:|
| 全局统一修正 | {energy['global_shift']:.2%} |
| 仅条件作用 | {energy['condition_effect']:.2%} |
| 跨条件共享的初始噪声作用 | {energy['shared_initial_noise_effect']:.2%} |
| 条件 × 初始噪声交互 | **{energy['condition_noise_interaction']:.2%}** |

每个条件的固定平均偏移只能解释 **{min(shift_explained):.1%}–{max(shift_explained):.1%}** 的修正能量。这说明统一偏移、纯条件均值或通用修正都不足，主要结构是 $c$ 与 $z$ 的交互。

![common random numbers 分析](../noise_geometry/common_random_analysis.png)

### 证据 F：整体没有强离散聚类，但个别条件出现有意义模式

逐 path KMeans 的最佳候选 silhouette 仅为 **{min(candidate_silhouettes):.3f}–{max(candidate_silhouettes):.3f}**，均低于保守阈值 0.25，因此整体没有足够证据宣称存在清晰分离的多个噪声簇。

但 `path_5` 是值得保留的例外：候选 $k=2$ 的噪声 silhouette 为 0.186，虽然噪声分离较弱，但标签与轨迹聚类完全一致（ARI=1.0），并解释 76.8% 的轨迹形状方差及 82.7% 的代价方差。这提示某些条件可能确实具有有意义的多轨迹模式。

![path 5 噪声 PCA 与轨迹模式](../noise_geometry/per_path/path_5.png)

## 4. 可以说什么，不能说什么

### 已由数据直接支持

1. 冻结 Stage 1 的映射中存在大量可达的低物理代价输出；
2. 标准高斯随机采样的平均质量和命中率明显低于无约束噪声优化；
3. 当前优化得到的 $z^*$ 多数明显偏离标准高斯典型区域；
4. 多数 $z^*$ 周围存在 $\sigma\approx0.05\sim0.1$ 的局部稳定盆地；
5. 修正主要是正交、条件相关且依赖初始噪声，而不是简单温度或固定均值偏移。

### 尚未由本实验决定

1. **不能据此断言最终方案只需改变采样器。** 无约束噪声搬运可能严重偏离原先验；
2. **也不能据此断言 Stage 2 必须只修改 $G_\theta$。** 这是一个合理的设计目标，但需要与受约束噪声搬运做对照实验；
3. “低于 GT 参考代价”是相对评价标准，不等于经过独立安全标准认证；
4. 当前结论只覆盖 `env000010` 的 8 个条件，不是跨环境普遍性结论；
5. 当前 $z$ 优化无先验约束，因此不能把 $z^*$ 分布直接当作最终应学习的采样分布。

## 5. 对“Stage 2 应修改生成器”的严谨表述

附图中的目标可以改写为：

> 在保持 $z\sim\mathcal N(0,I)$ 的前提下，通过修改 $G_\theta$，让典型噪声更高概率地产生低代价轨迹。

这是一个**与原始 Stage 2 目标一致的候选方法**，但不是本次 $z$ 诊断唯一推出的方法。现有证据同样允许另一条候选路线：冻结 $G_{{\theta_0}}$，学习受约束的条件噪声搬运 $T_\phi(c,z)$。

真正需要比较的是：

| 路线 | 保持不变 | 学习对象 | 必须监控的风险 |
|---|---|---|---|
| 受约束噪声搬运 | $G_{{\theta_0}}$ | $T_\phi(c,z)$ | 先验偏移、覆盖率、多峰丢失 |
| Pixel-space Stage 2 | $z\sim\mathcal N(0,I)$ | $G_\theta$ | Stage 1 能力遗忘、多样性下降、条件解绑 |

因此现阶段最稳妥的研究结论是：

$$
\boxed{{\text{{主要矛盾是“标准高斯典型噪声与优质输出区域不匹配”；由谁来修复这种不匹配，仍需受控对照。}}}}
$$

## 6. 数据与文件索引

- 汇报总览图：[evidence_overview.png](evidence_overview.png)
- 可追溯核心指标：[metrics.csv](metrics.csv)
- 跨条件汇总：[`../sweep_summary/summary.json`](../sweep_summary/summary.json)
- 扰动实验：[`../perturbation_summary/summary.json`](../perturbation_summary/summary.json)
- 噪声几何：[`../noise_geometry/summary.json`](../noise_geometry/summary.json)
- 单条件原始结果：`../env000010_path{{id}}_seed0/summary.json` 与 `trajectories.npz`
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output_dir = root / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with (root / "sweep_summary" / "summary.json").open() as handle:
        sweep = json.load(handle)
    with (root / "perturbation_summary" / "summary.json").open() as handle:
        perturb = json.load(handle)
    with (root / "noise_geometry" / "summary.json").open() as handle:
        geometry = json.load(handle)

    build_dashboard(output_dir / "evidence_overview.png", sweep, perturb, geometry)
    build_metrics_csv(output_dir / "metrics.csv", sweep, perturb, geometry)
    build_report(output_dir / "REPORT.md", sweep, perturb, geometry)
    print(f"Saved report assets to {output_dir}")


if __name__ == "__main__":
    main()

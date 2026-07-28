"""Analyze the geometry of saved Stage-1 optimized initial noises.

Inputs are the ``trajectories.npz`` and ``summary.json`` files produced by
``diagnose_stage1_latent.py``.  No model is loaded and no new trajectory is
generated.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline geometric analysis of optimized Stage-1 noises."
    )
    parser.add_argument("--root", default="diagnostics/stage1_noise")
    parser.add_argument("--dataset", default="data/sim_dataset/val")
    parser.add_argument("--env", default="env000010")
    parser.add_argument("--path-ids", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-clusters", type=int, default=4)
    parser.add_argument(
        "--silhouette-threshold",
        type=float,
        default=0.25,
        help="Below this score, report no clear multi-cluster structure (k=1).",
    )
    parser.add_argument("--output-name", default="noise_geometry")
    return parser.parse_args()


def vector_cosine(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.sum(x * y, axis=-1) / (
        np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1) + eps
    )


def distribution_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(np.max(values)),
    }


def choose_clusters(
    values: np.ndarray, max_clusters: int, threshold: float, seed: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    n_samples = values.shape[0]
    candidate_scores: dict[int, float] = {}
    candidate_labels: dict[int, np.ndarray] = {}
    max_k = min(max_clusters, n_samples - 1)
    for k in range(2, max_k + 1):
        labels = KMeans(n_clusters=k, n_init=50, random_state=seed).fit_predict(values)
        if len(np.unique(labels)) < 2:
            continue
        candidate_labels[k] = labels
        candidate_scores[k] = float(silhouette_score(values, labels))

    if not candidate_scores:
        labels = np.zeros(n_samples, dtype=np.int64)
        return labels, labels.copy(), {
            "selected_k": 1,
            "selected_silhouette": None,
            "candidate_silhouettes": {},
            "structure": "no_valid_multicluster_solution",
        }

    best_k = max(candidate_scores, key=candidate_scores.get)
    best_score = candidate_scores[best_k]
    if best_score < threshold:
        labels = np.zeros(n_samples, dtype=np.int64)
        selected_k = 1
        structure = "weak_or_continuous"
    else:
        labels = candidate_labels[best_k]
        selected_k = best_k
        structure = "detectable_multicluster"
    return labels, candidate_labels[best_k], {
        "selected_k": selected_k,
        "selected_silhouette": best_score if selected_k > 1 else None,
        "best_candidate_k": best_k,
        "best_candidate_silhouette": best_score,
        "candidate_silhouettes": {str(k): v for k, v in candidate_scores.items()},
        "silhouette_threshold": threshold,
        "structure": structure,
    }


def cluster_associations(
    labels: np.ndarray, trajectories: np.ndarray, costs: np.ndarray, seed: int
) -> dict:
    unique = np.unique(labels)
    if len(unique) == 1:
        return {
            "cluster_sizes": [int(len(labels))],
            "trajectory_silhouette_using_noise_labels": None,
            "trajectory_variance_explained_by_noise_clusters": 0.0,
            "cost_variance_explained_by_noise_clusters": 0.0,
            "noise_trajectory_adjusted_rand": None,
        }

    trajectory_vectors = trajectories.reshape(trajectories.shape[0], -1)
    trajectory_silhouette = float(silhouette_score(trajectory_vectors, labels))

    trajectory_center = trajectory_vectors.mean(axis=0)
    total_traj_ss = np.sum((trajectory_vectors - trajectory_center) ** 2)
    between_traj_ss = 0.0
    total_cost_ss = np.sum((costs - costs.mean()) ** 2)
    between_cost_ss = 0.0
    for cluster in unique:
        mask = labels == cluster
        between_traj_ss += mask.sum() * np.sum(
            (trajectory_vectors[mask].mean(axis=0) - trajectory_center) ** 2
        )
        between_cost_ss += mask.sum() * (
            float(costs[mask].mean()) - float(costs.mean())
        ) ** 2

    trajectory_labels = KMeans(
        n_clusters=len(unique), n_init=50, random_state=seed
    ).fit_predict(trajectory_vectors)
    return {
        "cluster_sizes": [int(np.sum(labels == cluster)) for cluster in unique],
        "trajectory_silhouette_using_noise_labels": trajectory_silhouette,
        "trajectory_variance_explained_by_noise_clusters": float(
            between_traj_ss / max(total_traj_ss, 1e-12)
        ),
        "cost_variance_explained_by_noise_clusters": float(
            between_cost_ss / max(total_cost_ss, 1e-12)
        ),
        "noise_trajectory_adjusted_rand": float(
            adjusted_rand_score(labels, trajectory_labels)
        ),
    }


def plot_path_analysis(
    output_path: Path,
    path_id: int,
    optimized_pca: np.ndarray,
    pca_variance: np.ndarray,
    labels: np.ndarray,
    costs: np.ndarray,
    trajectories: np.ndarray,
    ground_truth: np.ndarray,
    cluster_note: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].scatter(
        optimized_pca[:, 0], optimized_pca[:, 1], c=labels, cmap="tab10", s=50
    )
    for index, point in enumerate(optimized_pca):
        axes[0].annotate(str(index), point, fontsize=6, alpha=0.75)
    axes[0].set_title(
        f"optimized z PCA / exploratory clusters\n{cluster_note}; "
        f"PCA explained={pca_variance.sum():.1%}"
    )
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].grid(alpha=0.2)

    cost_plot = axes[1].scatter(
        optimized_pca[:, 0], optimized_pca[:, 1], c=costs, cmap="viridis", s=55
    )
    axes[1].set_title("optimized z PCA / physical cost")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].grid(alpha=0.2)
    fig.colorbar(cost_plot, ax=axes[1], label="best cost")

    colors = plt.get_cmap("tab10")
    for cluster in np.unique(labels):
        mask = labels == cluster
        for trajectory in trajectories[mask]:
            axes[2].plot(
                trajectory[:, 0], trajectory[:, 1], color=colors(cluster), alpha=0.12
            )
        mean_trajectory = trajectories[mask].mean(axis=0)
        axes[2].plot(
            mean_trajectory[:, 0],
            mean_trajectory[:, 1],
            color=colors(cluster),
            linewidth=2.5,
            label=f"cluster {cluster} (n={mask.sum()})",
        )
    axes[2].plot(
        ground_truth[:, 0], ground_truth[:, 1], "k--", linewidth=2, label="GT"
    )
    axes[2].set_title("trajectory shapes by noise cluster")
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].grid(alpha=0.2)
    axes[2].legend(fontsize=8)
    fig.suptitle(f"{path_id=}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def common_random_analysis(deltas: np.ndarray) -> dict:
    num_conditions, num_noises, _ = deltas.shape
    matched = np.eye(num_conditions, dtype=np.float64)
    unmatched = np.eye(num_conditions, dtype=np.float64)
    for left in range(num_conditions):
        for right in range(left + 1, num_conditions):
            matched_value = float(
                vector_cosine(deltas[left], deltas[right]).mean()
            )
            left_norm = deltas[left] / (
                np.linalg.norm(deltas[left], axis=1, keepdims=True) + 1e-12
            )
            right_norm = deltas[right] / (
                np.linalg.norm(deltas[right], axis=1, keepdims=True) + 1e-12
            )
            all_cosines = left_norm @ right_norm.T
            mask = ~np.eye(num_noises, dtype=bool)
            unmatched_value = float(all_cosines[mask].mean())
            matched[left, right] = matched[right, left] = matched_value
            unmatched[left, right] = unmatched[right, left] = unmatched_value

    global_mean = deltas.mean(axis=(0, 1), keepdims=True)
    condition_effect = deltas.mean(axis=1, keepdims=True) - global_mean
    noise_effect = deltas.mean(axis=0, keepdims=True) - global_mean
    residual = deltas - global_mean - condition_effect - noise_effect
    raw_ss = float(np.sum(deltas**2))
    components = {
        "global_shift": float(num_conditions * num_noises * np.sum(global_mean**2)),
        "condition_effect": float(num_noises * np.sum(condition_effect**2)),
        "shared_initial_noise_effect": float(
            num_conditions * np.sum(noise_effect**2)
        ),
        "condition_noise_interaction": float(np.sum(residual**2)),
    }
    fractions = {key: value / max(raw_ss, 1e-12) for key, value in components.items()}
    off_diagonal = ~np.eye(num_conditions, dtype=bool)
    return {
        "matched_cosine_matrix": matched,
        "unmatched_cosine_matrix": unmatched,
        "matched_cosine_mean_off_diagonal": float(matched[off_diagonal].mean()),
        "unmatched_cosine_mean_off_diagonal": float(unmatched[off_diagonal].mean()),
        "matched_minus_unmatched_cosine": float(
            (matched - unmatched)[off_diagonal].mean()
        ),
        "delta_energy_components": components,
        "delta_energy_fractions": fractions,
    }


def plot_common_analysis(
    output_path: Path,
    path_ids: list[int],
    common: dict,
    all_delta_pca: np.ndarray,
    condition_index: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
    image = axes[0].imshow(
        common["matched_cosine_matrix"], vmin=-1, vmax=1, cmap="coolwarm"
    )
    axes[0].set_xticks(range(len(path_ids)), path_ids, rotation=45)
    axes[0].set_yticks(range(len(path_ids)), path_ids)
    axes[0].set_title(r"mean cosine of matched $\Delta z_{i,c}$")
    fig.colorbar(image, ax=axes[0])

    labels = list(common["delta_energy_fractions"].keys())
    values = list(common["delta_energy_fractions"].values())
    axes[1].bar(range(len(labels)), values)
    axes[1].set_xticks(range(len(labels)), [
        "global", "condition", "shared z", "interaction"
    ], rotation=25, ha="right")
    axes[1].set_ylabel("fraction of delta energy")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("two-way correction decomposition")
    axes[1].grid(axis="y", alpha=0.2)

    for condition, path_id in enumerate(path_ids):
        mask = condition_index == condition
        axes[2].scatter(
            all_delta_pca[mask, 0],
            all_delta_pca[mask, 1],
            s=28,
            alpha=0.7,
            label=str(path_id),
        )
    axes[2].set_title("all correction vectors PCA")
    axes[2].set_xlabel("PC1")
    axes[2].set_ylabel("PC2")
    axes[2].grid(alpha=0.2)
    axes[2].legend(title="path_id", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output_dir = root / args.output_name
    per_path_dir = output_dir / "per_path"
    per_path_dir.mkdir(parents=True, exist_ok=True)

    center_rows: list[dict] = []
    path_rows: list[dict] = []
    all_initial = []
    all_deltas = []

    for path_id in args.path_ids:
        result_dir = root / f"{args.env}_path{path_id}_seed{args.seed}"
        saved = np.load(result_dir / "trajectories.npz")
        with (result_dir / "summary.json").open() as handle:
            saved_summary = json.load(handle)

        initial = saved["initial_z"].reshape(saved["initial_z"].shape[0], -1).astype(np.float64)
        optimized = saved["optimized_z"].reshape(saved["optimized_z"].shape[0], -1).astype(np.float64)
        trajectories = saved["optimized"].astype(np.float64)
        ground_truth = saved["ground_truth"].astype(np.float64)
        costs = np.asarray(
            [row["best_cost"] for row in saved_summary["per_restart"]],
            dtype=np.float64,
        )
        delta = optimized - initial
        initial_norm_sq = np.sum(initial**2, axis=1)
        initial_norm = np.sqrt(initial_norm_sq)
        alpha = np.sum(optimized * initial, axis=1) / np.maximum(initial_norm_sq, 1e-12)
        radial_signed = (alpha - 1.0) * initial_norm
        radial_magnitude = np.abs(radial_signed)
        orthogonal = optimized - alpha[:, None] * initial
        orthogonal_norm = np.linalg.norm(orthogonal, axis=1)
        delta_norm = np.linalg.norm(delta, axis=1)
        radial_energy_fraction = radial_magnitude**2 / np.maximum(delta_norm**2, 1e-12)

        pca = PCA(n_components=2).fit(optimized)
        optimized_pca = pca.transform(optimized)
        labels, exploratory_labels, cluster_selection = choose_clusters(
            optimized, args.max_clusters, args.silhouette_threshold, args.seed
        )
        exploratory_associations = cluster_associations(
            exploratory_labels, trajectories, costs, args.seed
        )
        distances = pdist(optimized, metric="euclidean")

        mean_delta = delta.mean(axis=0)
        shift_energy_explained = float(
            np.sum(mean_delta**2) / max(np.mean(np.sum(delta**2, axis=1)), 1e-12)
        )
        path_row = {
            "path_id": path_id,
            "num_centers": len(initial),
            "alpha_mean": float(alpha.mean()),
            "alpha_median": float(np.median(alpha)),
            "radial_magnitude_mean": float(radial_magnitude.mean()),
            "orthogonal_norm_mean": float(orthogonal_norm.mean()),
            "orthogonal_to_radial_mean_ratio": float(
                orthogonal_norm.mean() / max(radial_magnitude.mean(), 1e-12)
            ),
            "radial_delta_energy_fraction_mean": float(radial_energy_fraction.mean()),
            "condition_mean_shift_norm": float(np.linalg.norm(mean_delta)),
            "condition_mean_shift_delta_energy_explained": shift_energy_explained,
            "pairwise_distance_mean": float(distances.mean()),
            "pairwise_distance_median": float(np.median(distances)),
            "pairwise_distance_min": float(distances.min()),
            "pairwise_distance_max": float(distances.max()),
            "optimized_pca_2d_explained_variance": float(
                pca.explained_variance_ratio_.sum()
            ),
            **cluster_selection,
            **{
                f"exploratory_{key}": value
                for key, value in exploratory_associations.items()
            },
        }
        path_rows.append(path_row)

        for index in range(len(initial)):
            center_rows.append(
                {
                    "path_id": path_id,
                    "restart": index,
                    "selected_cluster": int(labels[index]),
                    "exploratory_cluster": int(exploratory_labels[index]),
                    "best_cost": float(costs[index]),
                    "alpha": float(alpha[index]),
                    "initial_norm": float(initial_norm[index]),
                    "optimized_norm": float(np.linalg.norm(optimized[index])),
                    "delta_norm": float(delta_norm[index]),
                    "radial_signed": float(radial_signed[index]),
                    "radial_magnitude": float(radial_magnitude[index]),
                    "orthogonal_norm": float(orthogonal_norm[index]),
                    "radial_delta_energy_fraction": float(radial_energy_fraction[index]),
                    "optimized_pc1": float(optimized_pca[index, 0]),
                    "optimized_pc2": float(optimized_pca[index, 1]),
                }
            )

        plot_path_analysis(
            per_path_dir / f"path_{path_id}.png",
            path_id,
            optimized_pca,
            pca.explained_variance_ratio_,
            exploratory_labels,
            costs,
            trajectories,
            ground_truth,
            (
                f"candidate k={cluster_selection['best_candidate_k']}, "
                f"silhouette={cluster_selection['best_candidate_silhouette']:.3f}; "
                f"conservative k={cluster_selection['selected_k']}"
            ),
        )
        all_initial.append(initial)
        all_deltas.append(delta)

    reference_initial = all_initial[0]
    max_common_noise_error = max(
        float(np.max(np.abs(initial - reference_initial))) for initial in all_initial
    )
    common_noise_verified = max_common_noise_error < 1e-6
    if not common_noise_verified:
        raise ValueError(
            "Initial noises are not shared across conditions; common-random analysis is invalid"
        )

    deltas = np.stack(all_deltas, axis=0)
    common = common_random_analysis(deltas)
    flat_deltas = deltas.reshape(-1, deltas.shape[-1])
    delta_pca_model = PCA(n_components=2).fit(flat_deltas)
    delta_pca = delta_pca_model.transform(flat_deltas)
    condition_index = np.repeat(np.arange(len(args.path_ids)), deltas.shape[1])
    plot_common_analysis(
        output_dir / "common_random_analysis.png",
        args.path_ids,
        common,
        delta_pca,
        condition_index,
    )

    radial = np.asarray([row["radial_magnitude"] for row in center_rows])
    orthogonal = np.asarray([row["orthogonal_norm"] for row in center_rows])
    alpha_all = np.asarray([row["alpha"] for row in center_rows])
    radial_fraction = np.asarray(
        [row["radial_delta_energy_fraction"] for row in center_rows]
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for path_id in args.path_ids:
        mask = np.asarray([row["path_id"] == path_id for row in center_rows])
        axes[0].scatter(radial[mask], orthogonal[mask], s=25, alpha=0.7, label=str(path_id))
    limit = max(float(radial.max()), float(orthogonal.max()))
    axes[0].plot([0, limit], [0, limit], "k--", linewidth=1)
    axes[0].set_xlabel(r"radial movement $|\alpha-1|\,||z||$")
    axes[0].set_ylabel(r"orthogonal movement $||r_\perp||$")
    axes[0].set_title("radial vs orthogonal correction")
    axes[0].legend(title="path_id", fontsize=7, ncol=2)
    axes[0].grid(alpha=0.2)
    axes[1].hist(alpha_all, bins=25)
    axes[1].axvline(1.0, color="black", linestyle="--")
    axes[1].set_xlabel(r"$\alpha$")
    axes[1].set_title("radial scale coefficient")
    axes[1].grid(alpha=0.2)
    axes[2].hist(radial_fraction, bins=25)
    axes[2].set_xlabel("radial fraction of delta energy")
    axes[2].set_xlim(0, 1)
    axes[2].set_title("how much movement is radial?")
    axes[2].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "radial_orthogonal.png", dpi=180)
    plt.close(fig)

    summary = {
        "environment": args.env,
        "path_ids": args.path_ids,
        "num_conditions": len(args.path_ids),
        "centers_per_condition": deltas.shape[1],
        "noise_dimension": deltas.shape[2],
        "common_random_noise_verified": common_noise_verified,
        "max_initial_noise_difference_across_conditions": max_common_noise_error,
        "aggregate_radial_orthogonal": {
            "alpha": distribution_stats(alpha_all),
            "radial_magnitude": distribution_stats(radial),
            "orthogonal_norm": distribution_stats(orthogonal),
            "orthogonal_to_radial_mean_ratio": float(
                orthogonal.mean() / max(radial.mean(), 1e-12)
            ),
            "radial_delta_energy_fraction": distribution_stats(radial_fraction),
        },
        "common_random_analysis": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in common.items()
        },
        "all_delta_pca_2d_explained_variance": float(
            delta_pca_model.explained_variance_ratio_.sum()
        ),
        "per_path": path_rows,
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    for filename, rows in (("per_path.csv", path_rows), ("per_center.csv", center_rows)):
        with (output_dir / filename).open("w", newline="") as handle:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(json.dumps({key: value for key, value in summary.items() if key != "per_path"}, indent=2))
    print(f"Saved geometry analysis to {output_dir}")


if __name__ == "__main__":
    main()

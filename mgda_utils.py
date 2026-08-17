"""Small exact MGDA utilities shared by development and training code."""

from __future__ import annotations

from typing import Any

import torch


OBJECTIVE_NAMES = ("F", "S", "K")


def _as_square_symmetric_gram(gram: torch.Tensor) -> torch.Tensor:
    gram = torch.as_tensor(gram, dtype=torch.float64, device="cpu")
    if gram.shape != (3, 3):
        raise ValueError(
            f"MGDA Gram matrix must have shape (3,3), got {gram.shape}"
        )
    if not bool(torch.isfinite(gram).all()):
        raise FloatingPointError("MGDA Gram matrix contains NaN or Inf")
    return (gram + gram.T) * 0.5


def _candidate_from_alpha(
    gram: torch.Tensor,
    alpha: torch.Tensor,
    *,
    label: str,
    support: tuple[int, ...],
    feasibility_tol: float,
):
    alpha = torch.as_tensor(alpha, dtype=torch.float64, device="cpu").reshape(3)
    if not bool(torch.isfinite(alpha).all()):
        return None
    if abs(float(alpha.sum()) - 1.0) > feasibility_tol:
        return None
    if float(alpha.min()) < -feasibility_tol:
        return None
    alpha = alpha.clamp_min(0.0)
    total = float(alpha.sum())
    if total <= 0.0:
        return None
    alpha = alpha / total
    objective = torch.dot(alpha, gram @ alpha)
    return {
        "alpha": alpha,
        "objective": float(objective),
        "label": label,
        "support": support,
    }


def solve_mgda_active_set(
    gram: torch.Tensor,
    *,
    feasibility_tol: float = 1e-8,
    degeneracy_tol: float = 1e-14,
) -> dict[str, Any]:
    """Solve the three-objective MGDA simplex problem by active sets."""
    gram = _as_square_symmetric_gram(gram)
    if feasibility_tol <= 0.0 or degeneracy_tol <= 0.0:
        raise ValueError("MGDA tolerances must be positive")

    candidates = []
    ones = torch.ones(3, dtype=torch.float64)
    kkt = torch.zeros((4, 4), dtype=torch.float64)
    kkt[:3, :3] = gram
    kkt[:3, 3] = ones
    kkt[3, :3] = ones
    rhs = torch.zeros(4, dtype=torch.float64)
    rhs[3] = 1.0
    try:
        kkt_solution = torch.linalg.solve(kkt, rhs)
    except RuntimeError:
        kkt_solution = torch.linalg.lstsq(kkt, rhs).solution
    residual = torch.linalg.vector_norm(kkt @ kkt_solution - rhs)
    if float(residual) <= feasibility_tol * max(1.0, float(rhs.norm())):
        candidate = _candidate_from_alpha(
            gram,
            kkt_solution[:3],
            label="interior",
            support=(0, 1, 2),
            feasibility_tol=feasibility_tol,
        )
        if candidate is not None:
            candidates.append(candidate)

    # On an edge with alpha_i=t and alpha_j=1-t:
    # t=(H_jj-H_ij)/(H_ii-2H_ij+H_jj).
    for i, j in ((0, 1), (0, 2), (1, 2)):
        denominator = float(gram[i, i] - 2.0 * gram[i, j] + gram[j, j])
        scale = max(
            1.0,
            abs(float(gram[i, i])),
            abs(float(gram[i, j])),
            abs(float(gram[j, j])),
        )
        if denominator <= degeneracy_tol * scale:
            coefficient_i = 0.5
        else:
            coefficient_i = float(gram[j, j] - gram[i, j]) / denominator
        coefficient_j = 1.0 - coefficient_i
        alpha = torch.zeros(3, dtype=torch.float64)
        alpha[i] = coefficient_i
        alpha[j] = coefficient_j
        candidate = _candidate_from_alpha(
            gram,
            alpha,
            label=f"{OBJECTIVE_NAMES[i]}-{OBJECTIVE_NAMES[j]}",
            support=(i, j),
            feasibility_tol=feasibility_tol,
        )
        if candidate is not None:
            candidates.append(candidate)

    # Vertices cover edge minimizers outside [0, 1].
    for index, name in enumerate(OBJECTIVE_NAMES):
        alpha = torch.zeros(3, dtype=torch.float64)
        alpha[index] = 1.0
        candidate = _candidate_from_alpha(
            gram,
            alpha,
            label=name,
            support=(index,),
            feasibility_tol=feasibility_tol,
        )
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        raise FloatingPointError(
            "MGDA active-set enumeration produced no feasible candidate"
        )
    selected = min(candidates, key=lambda item: item["objective"])
    alpha = selected["alpha"]
    if selected["label"] == "interior" and float(alpha.min()) <= feasibility_tol:
        active = [
            OBJECTIVE_NAMES[index]
            for index, coefficient in enumerate(alpha.tolist())
            if coefficient > feasibility_tol
        ]
        selected_label = "-".join(active) if active else selected["label"]
    else:
        selected_label = selected["label"]
    return {
        "alpha": alpha,
        "objective": float(selected["objective"]),
        "label": selected_label,
        "support": selected["support"],
        "candidate_count": len(candidates),
        "candidates": [
            {
                "label": item["label"],
                "objective": item["objective"],
                "alpha": item["alpha"],
            }
            for item in candidates
        ],
    }

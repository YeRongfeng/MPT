# Condition decoupling: Phase 0 audits and Phase 1 token_aligned candidate

> Status: implemented and unit-tested; **no training started**.
> Date: 2026-08-15.
> This file is the handoff contract for the condition-injection experiment. It
> does not change `METHOD_NOMENCLATURE.md` or the Stage-2 protocol.

## 1. Design contract

Four responsibilities are separated in the new candidate:

| Responsibility | Implementation | Learnable? |
|---|---|---|
| Task frame (S, θ, d, u_s, u_g) | analytic preprocessing and B-spline decoder | no |
| Map condition | spatial map tokens, per-layer path-to-map CA | yes |
| MeanFlow dynamics (t, t−r) | dedicated `time_condition_mlp` → time AdaLN branch | yes |
| Residual task invariants (u_s,u_g,log d) | dedicated `task_condition_mlp` → task AdaLN branch | yes (ablatable) |
| Raw S/G tensors | **never enter a learnable module** | — |

Default fusion is `mod = mod_time(c_time) + mod_task(c_task)`. It is a
preferred clean implementation, not a theoretical requirement.
`condition_fusion ∈ {additive, joint_mlp, additive_with_interaction}` is
recorded in checkpoints; additive losing to joint conditioning must not be
read as “decoupling is wrong”.

## 2. Tier A definition boundary

The new class is called `token_aligned`, not “canonicalized”:

- it rotates horizontal normal channels into the canonical basis;
- it gives map tokens a continuous coordinate PE at their
  chord-normalized task-frame centers `p_c = d^{-1} R(-θ)(p − S)`;
- it keeps every path token's learned index embedding and **adds**
  `E_coord(C_free(z_t; u_s, u_g))`, where `C_free` is the analytic canonical
  free control point for the current state.

It does **not** resample the map raster, so the CNN still sees the original
grid orientation and metric receptive fields. Tier A answers only:

> does CA-layer map/path coordinate alignment help?

Tier A failure does not imply that full task-frame canonicalization fails.

## 3. Implemented code

- `dit/Models.py`
  - `CoordinateSincosEncoding`
  - `DualConditionSpatialTrajBlock`
  - `TokenAlignedSpatialPathMeanFlowTransformer`
    (`ARCHITECTURE_NAME="token_aligned"`, `condition_semantics` token
    `token_aligned_coordinate_pe_v1`)
  - Existing `PathDiffusionTransformer`, `SpatialMapPathMeanFlowTransformer`
    and `PathMeanFlowTransformer` are unchanged.
- `train_compact_stage1.py`
  - new architecture entry `token_aligned` (production width 6×512);
  - CLI `--task-cond-mode`, `--condition-fusion`, `--coord-pe-max-freq`;
  - checkpoint metadata records the three condition fields and refuses
    mismatched resume.
- `tests/test_token_aligned_phase1.py`
  - forward/backward smoke and module-removal checks;
  - endpoint yaw invariant;
  - **full `meanflow_transport_loss` JVP finite**;
  - **raw-S/G wiring closure test** (all derived intermediates frozen);
  - task-condition and fusion smoke matrix.
- Phase 0 scripts
  - `tests/audits/research_scripts/audit_token_aligned_phase0_cfree.py`
  - `tests/audits/research_scripts/audit_token_aligned_phase0_support.py`

## 4. Phase 0 artifacts

Output directory: `tests/audits/token_aligned_phase0/`.

### C_free(z_t) distribution (1000 frozen-train contexts × 3 sources)

Current training time distribution (logit-normal μ=−0.8, σ=0.8; 25% endpoint
atom at t=1):

| t bucket | x range | y range | samples |
|---|---|---|---|
| (0.1,0.4] | [−0.058, 1.063] | [−0.237, 0.194] | 1031 |
| (0.4,0.7] | [−0.084, 1.076] | [−0.233, 0.231] | 1102 |
| (0.7,0.9] | [−0.053, 1.042] | [−0.244, 0.185] | 97 |
| t=1 source | [−0.129, 1.118] | [−0.360, 0.345] | 698 |

The coordinate PE candidate domain `x∈[−1.5,2.5], y∈[−1.5,1.5]` had zero
out-of-domain samples. This conclusion is bound to the current projected
Gaussian `source_factor`; any source-prior change must rerun the audit.

### Planning-support canonical envelope

Canvas frozen from **frozen training split only** (split_seed 20260814,
mask_seed 20260813, p_mask=0.5, 16000 contexts). q0.995 per-side bbox:

`x_min=-1.799, x_max=2.820, y_min=-2.429, y_max=2.415`.

Frozen square canvas (q0.995 + 5% margin):

`x∈[-2.032, 3.053], y∈[-2.550, 2.536]`.

Train support-mass coverage: mean 0.99994, p10 1.000, min 0.941.
Validation support-mass coverage is **reported only**: mean 0.99999, min
0.9956. The canvas must not be revised after seeing validation.

## 5. Phase 2 attribution arms (implemented, not trained)

Four independent architectures are now available in
`train_compact_stage1.py`, each with its own `ARCHITECTURE_NAME` and
checkpoint metadata:

| architecture | condition path | geometric address | production params |
|---|---:|---|---:|
| `spatial_map` (baseline) | raw S/G + time in one `cond_mlp`, one AdaLN head | grid-index map PE + learned path index PE | 44,498,498 |
| `alignment_only` | **same as baseline** | + rotated normals, canonical map/path coordinate PE | 44,498,498 (+0.0%) |
| `alignment_only_v2` | **same as alignment_only** | + corrected analytic 12x12 feature centers (`8j+3.5`) | 44,498,498 (+0.0%) |
| `token_aligned_capacity_matched` (main Q2) | `c_time,c_task ∈ R^{d/2}`；both additively modulate SA+CA+FFN | full token-aligned geometry | 44,024,898 (−1.1%) |
| `token_aligned` | full-width `c_time`/`c_task`, additive all-sublayer heads | full token-aligned geometry | 52,913,218 (+18.9%) |
| `token_aligned_split_routing` (secondary diagnostic) | time→SA+FFN, task→CA hard routing | full token-aligned geometry | 41,886,274 (−5.9%) |

The main Q2 capacity-matched arm is only 1.1% below baseline, so any win
cannot be attributed to extra capacity.  `split_routing` is retained only to
test the stronger hard-routing prior later; it is not part of Q1/Q2.

Phase 2 comparison order (frozen):

1. **Q1, alignment value**: `spatial_map/single_12` vs `alignment_only`.
   Same condition path, same parameter count; only canonical coordinate PE
   and the required normal-channel rotation differ.
2. **Q2, decoupling value**: `alignment_only` vs
   `token_aligned_capacity_matched` (half-width additive, both conditions
   may modulate SA/CA/FFN).
3. **Q3 (only after Q2)**: specialized routing `token_aligned_split_routing`
   and full-width additive `token_aligned` as secondary diagnostics for
   routing prior and task×time interaction.

Interpretation rules (frozen):

- alignment clearly better → keep the new structure;
- alignment roughly equal → still prefer alignment, because it fixes the
  explicit CA geometric-addressing gap;
- alignment clearly and stably worse → do not immediately revert; first
  audit coordinate PE, the canonical transform, and MeanFlow source-side
  geometry for a wrong inductive bias;
- decoupled conditioning worse → does not falsify alignment; task/time
  fusion is diagnosed separately.

## 5b. Q1 result (completed, not a positive signal)

Evaluation artifact: `diagnostics/q1_evaluation.json`.

| metric | baseline best (13250) | baseline final (13750) | alignment_only best (13000) | alignment_only final (13750) |
|---|---:|---:|---:|---:|
| fixed-source val loss | 0.106475 | 0.131772 | 0.115819 | 0.131117 |
| Safe@1 | 0.255 | 0.2425 | 0.180 | 0.160 |
| Safe@8 | 0.5525 | 0.5475 | 0.350 | 0.4125 |
| Forbidden | 0.915 | 0.9075 | 0.9275 | 0.9175 |
| Stability | 0.800 | 0.7675 | 0.820 | 0.810 |
| Curvature | 0.340 | 0.340 | 0.2375 | 0.2125 |

Interpretation: `alignment_only` is stably worse on Safe@1/Safe@8 and
curvature, slightly better on forbidden/stability.  Per the frozen rule this
is the "audit, do not revert" branch.

### alignment_only_v2 result (single-variable map-address fix)

Artifacts: `diagnostics/q1_v2_evaluation.json`.

| metric | baseline best (13250) | v1 best (13000) | v2 best (10500) |
|---|---:|---:|---:|
| fixed-source val loss | 0.106475 | 0.115819 | 0.126311 |
| Safe@1 | 0.255 | 0.180 | 0.135 |
| Safe@8 | 0.5525 | 0.350 | 0.410 |
| Forbidden | 0.915 | 0.9275 | 0.9275 |
| Stability | 0.800 | 0.820 | 0.8375 |
| Curvature | 0.340 | 0.2375 | 0.195 |

Conclusion: correcting the feature-center bug alone did **not** recover the
regression; v2 is worse on Safe@1/curvature than v1 and only partially
better on Safe@8.  The wrong map address was therefore not the dominant
cause.  The next step is the architecture fix: keep path hidden state free
of `E_coord(C_free)` and inject geometry only into Map-CA via a relative
bias `b(p_map_j - C_path_i)`; do not sweep `max_freq`.

### Q1 diagnosis (inference-only)

`diagnostics/q1_geometry_diagnosis.json` found:

1. **Map CA dependency clearly increased**: map-swap trajectory RMS 0.0375→0.0785 m
   (2.1x); CA-off trajectory RMS 0.0370→0.3116 m (8.4x).
2. **Coordinate PE dominates path hidden state**: E_coord norm ≈16 vs
   PathMLP ≈9.2 and E_index ≈1.3; total path-token norm 4.0→17.1.
3. **Map feature-center implementation bug**: assumed
   `linspace(-1.0, 0.98, 12)`, but the four conv(3,p1)+maxpool(2) stages map
   feature index `j` to input pixel `8j+3.5`; normalized centers should be
   `-1 + 0.02*(8j+3.5)`.  The assumed centers are systematically wrong by up
   to ~0.15 canonical units (gradient probe mean error ≈0.205).

### alignment_only_v2 (single-variable bug fix, implemented not trained)

`AlignmentOnlyV2SpatialPathMeanFlowTransformer` changes **only** the
12x12 feature-center mapping to the analytic `8j+3.5` formula.  Everything
else is inherited from `alignment_only`: path coordinate PE, max_freq=16,
map grid PE + canonical coord PE, normal rotation, joint raw S/G+time
conditioning, and parameter count (44,498,498).

Training order after v2 is frozen:

- v2 improves substantially → the Q1 regression was largely the wrong map
  address;
- v2 still regresses → test CA-only relative geometric bias
  (`A_ij = q_i^T k_j / sqrt(d) + b(p_map_j - C_path_i)`), not simple
  max_freq sweeping.

## 6. Phase gates

- Phase 1 exit: `tests/test_token_aligned_phase1.py` passes; no training.
- Phase 2 wiring exit: `tests/test_phase2_attribution_arms.py` passes
  (exact parameter equality for `alignment_only`, ≤8% delta for the
  capacity-matched arm, forward/backward and full MeanFlow JVP for both).
- Phase 2 entry requires explicit approval of experiment scope and compute.
- Phase 3 (Tier B fixed-canvas raster canonicalization) uses the frozen
  canvas above and adds the smooth-field SE(2) raster invariance test.

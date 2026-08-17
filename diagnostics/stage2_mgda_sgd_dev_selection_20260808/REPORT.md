# Stage-2 MGDA+SGD Development Selection Audit

This audit reads only the frozen `dataset1_val/train` manifest embedded in the Stage-2 checkpoints. External-30 and final-test data were not opened.

## Selection Integrity

- Intended 10-epoch endpoint supplied to this audit: `50000` updates.
- Stored training configuration: `20` epochs, corresponding to `100000` updates.
- Actual last training step: `54059`.
- Pre-endpoint best: step `49600` with task cost `2.417366028`.
- Current `stage2_best.pth`: step `53400`; it is post-endpoint and ineligible.
- The eligible pre-endpoint best checkpoint file was overwritten and no historical copy was found.
- Selection is therefore **BLOCKED**. If 10 epochs was the independently predefined endpoint, its best checkpoint is missing. If the stored 20-epoch configuration governs, the run is incomplete.
- `stage2_last.pth` remains the reproducible 10-epoch endpoint, but choosing it now would replace the predefined best rule with a last-epoch rule.

## Frozen Validation Comparison

| model | within 10 epochs | J_F | J_S | J_K | task cost | strict | Safe@1 | Safe@K | pairwise m | eff. rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stage1 | True | 0.360617 | 0.643377 | 0.059148 | 2.476035 | 15.625% | 18.000% | 44.000% | 0.335705 | 2.230681 |
| stage2_last_10epoch | True | 0.353373 | 0.634903 | 0.031805 | 2.417669 | 28.750% | 28.000% | 74.000% | 0.353135 | 2.197829 |
| stage2_best_post_endpoint | False | 0.353106 | 0.634930 | 0.031265 | 2.416091 | 28.875% | 28.000% | 74.000% | 0.353425 | 2.195759 |

Detailed path-length, regularization, curvature, margin, paired transition, checkpoint-hash, and protocol results are in `summary.json` and `metrics.csv`.

MGDA geometry acceptance and external evaluation were not run because there is no protocol-eligible selected checkpoint yet.

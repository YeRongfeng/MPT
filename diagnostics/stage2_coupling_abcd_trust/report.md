# Stage-2 coupling A/B/C/D

- Train maps: env000015, env000040, env000064
- Held-out maps: env000068, env000085
- K=4; train/held-out samples: 36/24
- Teacher trust-region acceptance: 100.0%

| Variant | Split | Recovery (safe, median) | Mode keep | safe@K | Abnormal | Diversity RMS (m) |
|---|---|---:|---:|---:|---:|---:|
| A_stage1 | train | 0.000 | 100.0% | 0.0% | 2.8% | 0.375 |
| A_stage1 | heldout | 0.000 | 100.0% | 0.0% | 0.0% | 0.426 |
| teacher | train | 1.000 | 94.4% | 0.0% | 11.1% | 0.503 |
| teacher | heldout | 1.000 | 91.7% | 0.0% | 0.0% | 0.701 |
| B_cost_only | train | 1.077 | 77.8% | 0.0% | 30.6% | 0.206 |
| B_cost_only | heldout | 0.204 | 75.0% | 0.0% | 0.0% | 0.187 |
| C_random_coupling | train | 0.860 | 77.8% | 0.0% | 22.2% | 0.590 |
| C_random_coupling | heldout | 0.102 | 83.3% | 16.7% | 8.3% | 0.410 |
| D_induced_coupling | train | 0.782 | 86.1% | 0.0% | 19.4% | 0.493 |
| D_induced_coupling | heldout | 0.030 | 83.3% | 16.7% | 12.5% | 0.383 |

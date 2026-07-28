# Stage-2 coupling A/B/C/D

- Train maps: env000015, env000040, env000064
- Held-out maps: env000068, env000085
- K=4; train/held-out samples: 36/24
- Teacher trust-region acceptance: 26.7%

| Variant | Split | Recovery (safe, median) | Mode keep | safe@K | Abnormal | Diversity RMS (m) |
|---|---|---:|---:|---:|---:|---:|
| A_stage1 | train | 0.000 | 100.0% | 0.0% | 2.8% | 0.375 |
| A_stage1 | heldout | 0.000 | 100.0% | 0.0% | 0.0% | 0.426 |
| teacher | train | 1.000 | 100.0% | 0.0% | 2.8% | 0.453 |
| teacher | heldout | 1.000 | 100.0% | 33.3% | 0.0% | 0.528 |
| B_cost_only | train | 0.966 | 77.8% | 0.0% | 19.4% | 0.203 |
| B_cost_only | heldout | 0.325 | 83.3% | 0.0% | 0.0% | 0.209 |
| C_random_coupling | train | 0.869 | 69.4% | 0.0% | 0.0% | 0.406 |
| C_random_coupling | heldout | -0.092 | 87.5% | 0.0% | 0.0% | 0.298 |
| D_induced_coupling | train | 0.891 | 88.9% | 0.0% | 0.0% | 0.423 |
| D_induced_coupling | heldout | 0.028 | 100.0% | 0.0% | 0.0% | 0.347 |

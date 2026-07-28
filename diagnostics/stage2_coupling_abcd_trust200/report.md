# Stage-2 coupling A/B/C/D

- Train maps: env000015, env000040, env000064
- Held-out maps: env000068, env000085
- K=4; train/held-out samples: 36/24
- Teacher trust-region acceptance: 100.0%

| Variant | Split | Recovery (safe, median) | Mode keep | safe@K | Abnormal | Diversity RMS (m) |
|---|---|---:|---:|---:|---:|---:|
| A_stage1 | train | 0.000 | 100.0% | 0.0% | 2.8% | 0.375 |
| A_stage1 | heldout | 0.000 | 100.0% | 0.0% | 0.0% | 0.426 |
| teacher | train | 1.000 | 91.7% | 0.0% | 8.3% | 0.483 |
| teacher | heldout | 1.000 | 91.7% | 0.0% | 0.0% | 0.678 |
| B_cost_only | train | 1.035 | 77.8% | 0.0% | 38.9% | 0.266 |
| B_cost_only | heldout | 0.207 | 79.2% | 0.0% | 0.0% | 0.365 |
| C_random_coupling | train | 0.534 | 77.8% | 0.0% | 0.0% | 0.340 |
| C_random_coupling | heldout | 0.109 | 83.3% | 0.0% | 4.2% | 0.300 |
| D_induced_coupling | train | 0.548 | 83.3% | 0.0% | 8.3% | 0.566 |
| D_induced_coupling | heldout | 0.039 | 87.5% | 0.0% | 4.2% | 0.358 |

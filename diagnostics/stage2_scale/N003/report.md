# Stage-2 coupling A/B/C/D

- Train maps: env000015, env000040, env000064
- Held-out maps: env000017, env000077, env000090, env000088, env000069, env000091, env000046, env000072, env000087, env000092
- K=4; train/held-out samples: 12/40
- Teacher trust-region acceptance: 100.0%

| Variant | Split | Recovery (safe, median) | Mode keep | safe@K | Abnormal | Diversity RMS (m) |
|---|---|---:|---:|---:|---:|---:|
| A_stage1 | train | 0.000 | 100.0% | 0.0% | 0.0% | 0.242 |
| A_stage1 | heldout | 0.000 | 100.0% | 0.0% | 2.5% | 0.437 |
| teacher | train | 1.000 | 100.0% | 0.0% | 16.7% | 0.578 |
| teacher | heldout | 1.000 | 90.0% | 20.0% | 0.0% | 0.503 |
| B_cost_only | train | 1.058 | 75.0% | 0.0% | 25.0% | 0.409 |
| B_cost_only | heldout | 0.018 | 92.5% | 0.0% | 0.0% | 0.500 |
| D_induced_coupling | train | 0.375 | 83.3% | 0.0% | 0.0% | 0.371 |
| D_induced_coupling | heldout | 0.010 | 80.0% | 0.0% | 0.0% | 0.410 |

# Stage-2 coupling A/B/C/D

- Train maps: env000015, env000040, env000064
- Held-out maps: env000017, env000077, env000090, env000088, env000069, env000091, env000046, env000072, env000087, env000092
- K=4; train/held-out samples: 12/40
- Teacher trust-region acceptance: 100.0%

| Variant | Split | Recovery (safe, median) | Mode keep | safe@K | Abnormal | Diversity RMS (m) |
|---|---|---:|---:|---:|---:|---:|
| A_stage1 | train | 0.000 | 100.0% | 0.0% | 0.0% | 0.415 |
| A_stage1 | heldout | 0.000 | 100.0% | 0.0% | 5.0% | 0.393 |
| teacher | train | 1.000 | 100.0% | 0.0% | 0.0% | 0.510 |
| teacher | heldout | 1.000 | 87.5% | 30.0% | 2.5% | 0.448 |
| B_cost_only | train | 0.707 | 91.7% | 0.0% | 16.7% | 0.327 |
| B_cost_only | heldout | 0.023 | 90.0% | 0.0% | 7.5% | 0.426 |
| D_induced_coupling | train | 0.314 | 100.0% | 0.0% | 0.0% | 0.322 |
| D_induced_coupling | heldout | 0.033 | 95.0% | 0.0% | 2.5% | 0.393 |

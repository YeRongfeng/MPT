# Stage-2 coupling A/B/C/D

- Train maps: env000015, env000040, env000064, env000065, env000082, env000013, env000028, env000076, env000079, env000071
- Held-out maps: env000017, env000077, env000090, env000088, env000069, env000091, env000046, env000072, env000087, env000092
- K=4; train/held-out samples: 40/40
- Teacher trust-region acceptance: 100.0%

| Variant | Split | Recovery (safe, median) | Mode keep | safe@K | Abnormal | Diversity RMS (m) |
|---|---|---:|---:|---:|---:|---:|
| A_stage1 | train | 0.000 | 100.0% | 0.0% | 0.0% | 0.426 |
| A_stage1 | heldout | 0.000 | 100.0% | 0.0% | 5.0% | 0.393 |
| teacher | train | 1.000 | 90.0% | 0.0% | 2.5% | 0.471 |
| teacher | heldout | 1.000 | 87.5% | 20.0% | 2.5% | 0.447 |
| B_cost_only | train | 0.955 | 65.0% | 20.0% | 2.5% | 0.250 |
| B_cost_only | heldout | -0.043 | 82.5% | 0.0% | 5.0% | 0.447 |
| D_induced_coupling | train | 0.539 | 90.0% | 0.0% | 0.0% | 0.497 |
| D_induced_coupling | heldout | 0.020 | 92.5% | 0.0% | 0.0% | 0.496 |

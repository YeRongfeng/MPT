# Stage-2 coupling A/B/C/D

- Train maps: env000015, env000040, env000064, env000065, env000082, env000013, env000028, env000076, env000079, env000071, env000053, env000073, env000070, env000062, env000075, env000056, env000030, env000000, env000078, env000010, env000014, env000036, env000012, env000057, env000001, env000086, env000098, env000026, env000050, env000032
- Held-out maps: env000017, env000077, env000090, env000088, env000069, env000091, env000046, env000072, env000087, env000092
- K=4; train/held-out samples: 120/40
- Teacher trust-region acceptance: 100.0%

| Variant | Split | Recovery (safe, median) | Mode keep | safe@K | Abnormal | Diversity RMS (m) |
|---|---|---:|---:|---:|---:|---:|
| A_stage1 | train | 0.000 | 100.0% | 3.3% | 0.0% | 0.353 |
| A_stage1 | heldout | 0.000 | 100.0% | 0.0% | 5.0% | 0.393 |
| teacher | train | 1.000 | 87.5% | 16.7% | 4.2% | 0.430 |
| teacher | heldout | 1.000 | 87.5% | 20.0% | 2.5% | 0.447 |
| B_cost_only | train | 0.909 | 63.3% | 16.7% | 10.8% | 0.232 |
| B_cost_only | heldout | 0.039 | 87.5% | 0.0% | 25.0% | 0.632 |
| D_induced_coupling | train | 0.628 | 85.8% | 13.3% | 0.8% | 0.379 |
| D_induced_coupling | heldout | 0.093 | 87.5% | 0.0% | 15.0% | 0.556 |

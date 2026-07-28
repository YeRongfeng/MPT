# Stage-2 coupling A/B/C/D

- Train maps: env000015, env000040, env000064, env000065, env000082, env000013, env000028, env000076, env000079, env000071, env000053, env000073, env000070, env000062, env000075, env000056, env000030, env000000, env000078, env000010, env000014, env000036, env000012, env000057, env000001, env000086, env000098, env000026, env000050, env000032, env000044, env000045, env000048, env000096, env000009, env000043, env000011, env000035, env000034, env000018, env000060, env000029, env000066, env000084, env000041, env000068, env000054, env000081, env000019, env000085, env000089, env000023, env000067, env000058, env000027, env000005, env000025, env000037, env000074, env000031
- Held-out maps: env000017, env000077, env000090, env000088, env000069, env000091, env000046, env000072, env000087, env000092
- K=4; train/held-out samples: 240/40
- Teacher trust-region acceptance: 100.0%

| Variant | Split | Recovery (safe, median) | Mode keep | safe@K | Abnormal | Diversity RMS (m) |
|---|---|---:|---:|---:|---:|---:|
| A_stage1 | train | 0.000 | 100.0% | 6.7% | 0.0% | 0.347 |
| A_stage1 | heldout | 0.000 | 100.0% | 0.0% | 5.0% | 0.393 |
| teacher | train | 1.000 | 85.0% | 20.0% | 3.3% | 0.419 |
| teacher | heldout | 1.000 | 87.5% | 20.0% | 2.5% | 0.447 |
| B_cost_only | train | 0.949 | 66.7% | 28.3% | 5.0% | 0.181 |
| B_cost_only | heldout | 0.163 | 72.5% | 0.0% | 35.0% | 0.419 |
| D_induced_coupling | train | 0.610 | 81.2% | 16.7% | 0.8% | 0.361 |
| D_induced_coupling | heldout | 0.214 | 82.5% | 0.0% | 10.0% | 0.494 |

# Stage-2 coupling A/B/C/D

- Train maps: env000015, env000040, env000064
- Held-out maps: env000068, env000085
- K=4; train/held-out samples: 36/24
- Teacher trust-region acceptance: 100.0%

| Variant | Split | Recovery (safe, median) | Mode keep | safe@K | Abnormal | Diversity RMS (m) |
|---|---|---:|---:|---:|---:|---:|
| A_stage1 | train | 0.000 | 100.0% | 0.0% | 2.8% | 0.375 |
| A_stage1 | heldout | 0.000 | 100.0% | 0.0% | 0.0% | 0.426 |
| teacher | train | 1.000 | 91.7% | 0.0% | 5.6% | 0.429 |
| teacher | heldout | 1.000 | 91.7% | 0.0% | 4.2% | 0.637 |
| C_random_coupling | train | 0.795 | 75.0% | 0.0% | 0.0% | 0.416 |
| C_random_coupling | heldout | 0.093 | 70.8% | 0.0% | 4.2% | 0.395 |

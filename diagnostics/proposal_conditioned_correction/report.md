# Proposal-conditioned correction diagnostic

- Train/held-out samples: 240/40
- Epochs/batch: 20/16

| Variant | Split | Recovery median | Teacher error median | Mode keep | safe@4 | Teacher-safe subset safe@4 | Abnormal |
|---|---|---:|---:|---:|---:|---:|---:|
| D_z_to_Rstar | train | 0.680 | 0.805 | 73.3% | 18.3% | 41.7% | 0.0% |
| D_z_to_Rstar | heldout | 0.001 | 1.022 | 87.5% | 0.0% | 0.0% | 7.5% |
| P_R0_to_delta | train | 0.673 | 0.848 | 70.8% | 18.3% | 41.7% | 0.0% |
| P_R0_to_delta | heldout | 0.049 | 1.014 | 62.5% | 10.0% | 16.7% | 7.5% |

# Privileged proposal-local information upper bound

- Canonical pairs: `diagnostics/stage2_scale/N060_v6/optimizer_pairs.npz`
- Train/held-out samples: 240/40
- Local samples: 100 points along fixed Stage-1 R0
- Oracle features are used in both training and held-out inference.

| Variant | Split | Recovery median | Teacher error median | Mode keep | safe@K | Abnormal |
|---|---|---:|---:|---:|---:|---:|
| D_baseline | train | 0.610 | 0.806 | 81.2% | 16.7% | 0.8% |
| D_baseline | heldout | 0.214 | 1.005 | 82.5% | 0.0% | 10.0% |
| E_value | train | 0.665 | 0.764 | 81.7% | 16.7% | 0.0% |
| E_value | heldout | 0.251 | 0.988 | 77.5% | 0.0% | 5.0% |
| E_value_gradient | train | 0.612 | 0.765 | 82.1% | 20.0% | 0.0% |
| E_value_gradient | heldout | 0.204 | 0.997 | 82.5% | 0.0% | 10.0% |

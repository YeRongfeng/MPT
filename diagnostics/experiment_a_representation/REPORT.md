# Experiment A: trajectory representation

- Candidates: 200
- Selected high-risk samples: 50
- Iterations: 300
- LR grid: [0.003, 0.01, 0.03]
- Best global LR: {'P': 0.03, 'R': 0.03, 'raw': 0.03}
- Elapsed: 100.6 s

## Recovery relative to physical-control-point optimization

- R: median=0.987, mean=1.005, fraction >= 0.8: 100.0%, fraction >= 0.5: 100.0%
- raw: median=1.114, mean=1.120, fraction >= 0.8: 100.0%, fraction >= 0.5: 100.0%

## Best-LR diagnostic medians

| method | total cost | dangerous ratio | path length (m) | smoothness | jerk |
|---|---:|---:|---:|---:|---:|
| P | 6.3427 -> 0.9322 | 0.245 -> 0.020 | 16.89 -> 22.19 | 0.00123 -> 0.01287 | 0.00079 -> 0.00626 |
| R | 6.3427 -> 0.8192 | 0.245 -> 0.020 | 16.89 -> 20.20 | 0.00123 -> 0.00334 | 0.00079 -> 0.00116 |
| raw | 6.3427 -> 0.1314 | 0.245 -> 0.000 | 16.89 -> 35.85 | 0.00123 -> 0.02433 | 0.00079 -> 0.00768 |

## Interpretation

- R and raw both recover at least 80% of P's cost reduction on all selected samples, so the residual/radial representation is not the main Stage-2 bottleneck.
- Raw obtains its unusually low feasibility cost partly by taking large detours. Treat its result as evidence of representational capacity, not as evidence of better trajectory quality.
- The large path-length increase shows that the current objective does not sufficiently penalize detours; quality has only a small tie-breaking weight and path length is absent.

## Decision rule

- R and raw recovery >= 0.8: representation is not the main bottleneck.
- R strong but raw weak: radial feasibility map is a bottleneck.
- R also weak: edge-residual representation is a bottleneck.

See `summary.json`, `per_sample_results.csv`, `optimization_trace.csv`, and the generated plots for details.

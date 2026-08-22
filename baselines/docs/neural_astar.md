# Neural A*

## Source status

The official ICML-2021 implementation is vendored at
`baselines/third_party/neural-astar` at commit
`ff7ef1684facb451662e942aaa4978432a1cbead` (branch `icml2021`). The source
implements a trainable encoder followed by the differentiable A* search module;
it is not a one-forward path regressor.

## MPT boundary

The official data interface is a fixed-grid binary maze plus a one-hot goal map,
with a learned model checkpoint loaded by `Runner.pretrained_path`. The official
snapshot contains no checkpoint for the MPT 100x100 terrain tasks. Its declared
dependencies also target the historical PyTorch 1.5-era environment.

Therefore no random/untrained Neural A* output is treated as a baseline. The
MPT conversion is now frozen in `baselines/neural_astar_data.py`: mask to binary
traversability, start/goal maps, official Moore labels, and path-map to world
coordinates. The remaining work is to obtain an approved training/checkpoint
scope. Training has not been launched.

## Prepared conversion

`baselines/neural_astar_data.py` provides the NumPy-only conversion and can
write the official four-array train/valid/test `.npz` layout. A two-sample
`val/env000000` export was smoke-tested in `/tmp`; it is not a trained model
or an evaluation result.

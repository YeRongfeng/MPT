# Kicki neural B-spline

The official `bspline` branch is vendored at
`baselines/third_party/kicki-neural-path-planning` (commit `fc61a01`). Its
native model consumes a 256x128 two-channel free/obstacle image plus start and
goal vehicle states in a local car-like coordinate system, and emits control
points for the analytic B-spline construction.

The repository documents a separate pretrained-model bundle; no bundle is
present in this checkout. The MPT task is a 100x100, 20 m terrain grid with
SE(2) endpoints, so resizing the map and changing the vehicle-frame scaling
would be a declared adapter, not an unmodified reproduction. No Kicki result
is included until that adapter and checkpoint are fixed.

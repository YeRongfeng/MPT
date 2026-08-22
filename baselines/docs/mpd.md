# MPD-Splines

The official public source is vendored at
`baselines/third_party/mpd-splines-public` (commit `3676cbf`). The repository
implements diffusion over parametric trajectories and deployment-time cost
guidance, but its native scripts instantiate IsaacGym robot environments and
load externally generated model directories.

No model directory or IsaacGym environment is copied into MPT, and no terrain
adapter is inferred from the robot-specific configs. MPD therefore remains
source-recovered but not evaluable in the shared MPT checker.

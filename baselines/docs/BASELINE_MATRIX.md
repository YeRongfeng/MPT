# Baseline comparison matrix

This is the working inventory before reproducing the next baselines. All
implementations and artifacts belong under `baselines/`; external workspaces
are read-only references.

## Core comparison set

These are the methods currently intended for the main comparison, subject to
successful source recovery and a fair adapter.

| Method | Family | Deployment information | Online computation | Output contract | Current status | Planned order |
|---|---|---|---|---|---|---:|
| Original Uneven Planner | Search + trajectory optimization | Full terrain, privileged | Kino A* initialization plus ALM trajectory optimization | SE(2) trajectory, converted to XY for the shared checker | Original source is vendored under `baselines/third_party/uneven_planner_original`; one-shot ROS adapter and smoke exist | 1 |
| Neural A* | Learned search/planning | Official ICML-2021 implementation, Moore mechanism | Test-time learned search, not a single plain forward pass | Path output converted to the shared XY/SE(2) contract | Source and NumPy MPT conversion are vendored; MPT-compatible checkpoint is pending | 2 |
| Kicki neural B-spline | Neural vehicle-path generation | Official `bspline` branch, commit `fc61a01` | Neural inference with analytic endpoint/B-spline structure | B-spline/SE(2) path converted to the shared checker | Source is vendored; its 256x128 map input, vehicle-frame scaling, and pretrained model bundle still need an MPT adapter | 3 |
| MPD | Multi-step generative planning | Official `mpd-splines-public`, commit `3676cbf` | Multi-step generation plus deployment-time guidance | B-spline path output converted to the shared XY contract | Source is vendored; official interface is IsaacGym/robot-specific and model directories are external, so the MPT adapter is still pending | 4 |

The four methods above must be compared using their native computation pattern.
The shared evaluator standardizes validity metrics, but it must not turn a
search or guided multi-step method into an artificial one-forward baseline.

## Path MeanFlow rows

| Method | Role | Deployment information | Status | Table treatment |
|---|---|---|---|---|
| Path MeanFlow Stage 1 | Established one-call generator | Partial observation, mask, start/goal, source | Checkpoint and evaluator adapter exist | Main method ablation/reference row |
| Path MeanFlow Stage 2 | Validation-gated terrain adaptation candidate | Same deployment inputs as Stage 1; complete terrain is training/validation only | Existing checkpoint is diagnostic until the Stage 2 admission protocol passes | Report as experimental Stage 2, not as an established method property |

## Existing optional or diagnostic rows

| Method | Why it is separate | Current status | Treatment |
|---|---|---|---|
| T-Hybrid A* | Terrain-aware search baseline with an official ROS node; already has a smoke adapter | ROS smoke exists, full evaluation not done | Supplementary or an additional search row after the protocol is frozen |

## Shared evaluation boundary

All executable rows should eventually produce a path plus method-native timing
and iteration/NFE metadata. The shared hard checker reports planning support,
stability, curvature, endpoint yaw, finite values, and path length. Timeout,
no-path, numerical failure, and malformed output count as infeasible.

Full terrain is allowed for the privileged Original Uneven row and for offline
hard checking. Partial-information rows must receive the frozen observation and
mask protocol; privileged terrain must not be passed into their deployment
adapter.

The implementation order is Original Uneven (completed smoke), Neural A*,
Kicki, and finally MPD. No full benchmark run is implied by this inventory;
each method still needs a smoke contract and an approved evaluation budget
before a long run.

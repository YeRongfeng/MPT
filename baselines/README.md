# Baseline reproduction

人能读的进度和每一步说明在 [`docs/PROGRESS.md`](docs/PROGRESS.md)。
当前 baseline 的完整分层见 [`docs/BASELINE_MATRIX.md`](docs/BASELINE_MATRIX.md)。

## Repository boundary

All baseline adapters, vendored source snapshots, ROS launch/configuration
files, build directories, logs, and evaluation outputs belong under this
`baselines/` directory. External workspaces such as
`/home/sdu/uneven_planner` and
`/home/sdu/reference_project/gaofei_uneven_planner_ws/uneven_planner` are
read-only reference sources; baseline work must not edit them in place.

External planners are evaluated with the same hard checker as Path MeanFlow
(`trajectory_validity_metrics`): planning-support, stability, curvature, and
endpoint yaw. Timeout and no-path count as infeasible.

Checkpoints for Path MeanFlow itself: `data/path_meanflow/stage1_best.pth` and
`data/path_meanflow/stage2_best.pth`.

## Status

| Method | Role | Status |
|---|---|---|
| T-Hybrid A* | Search baseline（主表用这个） | 官方 ROS 节点，入口 `t_hybrid` |
| Original Uneven Planner | Privileged full-information optimizer | 原版 ROS 核心、一次性 smoke 和说明页已接入；当前仅完成单任务诊断，不能用当前生成器路径代替 |
| Path MeanFlow Stage 1/2 | 本文方法 | 已接到同一评价器 |
| Neural A* | Test-time learned search baseline | 官方 ICML-2021 源码和 NumPy MPT 转换层已固定；MPT checkpoint 待定，见 [`docs/neural_astar.md`](docs/neural_astar.md) |
| Kicki neural B-spline | Neural B-spline vehicle-path baseline | 官方 `bspline` 源码已固定；MPT 输入适配与 checkpoint 待定，见 [`docs/kicki.md`](docs/kicki.md) |
| MPD | Multi-step generative baseline | 官方 `mpd-splines-public` 源码已固定；IsaacGym/模型与 MPT 适配待定，见 [`docs/mpd.md`](docs/mpd.md) |

The Original Uneven baseline must re-run the upstream ROS planner on the
matched map and start/goal task; it must not read stored expert paths as its
answer.

## Smoke

```bash
source /opt/ros/noetic/setup.bash
# first time: catkin_make in baselines/t_hybrid_ws (see docs/hybrid_astar.md)
.venv/bin/python -m unittest tests.test_t_hybrid
.venv/bin/python -m baselines.evaluate --method t_hybrid --environments 1 --paths-per-env 1
```

Elevation grids can be exported with:

```bash
python3 -m baselines.export_elevation_pcd \
  --env-dir data/dataset1/val/env000000 \
  --output baselines/evaluation_results/env000000.pcd
```

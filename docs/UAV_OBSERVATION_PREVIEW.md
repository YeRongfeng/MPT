# Final UAV observation model

Run the interactive preview from the repository root:

```bash
MPLBACKEND=TkAgg .venv/bin/python visualize_uav_observation.py \
  --dataset-root data/dataset1 \
  --split val \
  --environment env000000
```

The three panels show the terrain with accumulated support, the current
downward scan, and the accumulated support mask. The model loads
`path_0.p` as a geometric route guide for the vehicle. The UAV uses a
randomized survey route with both map-wide exploration points and mandatory
perturbed anchors around the reference path;
it is not sampled from the vehicle route. The red marker is the vehicle and the
black triangle is the UAV. The LiDAR is mounted under the UAV, so one scan is
a ground disk (`scan-radius-m`), not a forward fan. `Next scan` moves only the
UAV and unions its new returns/support into the history; `Final area` completes
the remaining UAV survey in one click and shows the final accumulated region.
The vehicle does not track the UAV. `Advance car` moves the vehicle one step
only when the candidate position is already inside the accumulated UAV map.
`Reset` clears the history and restores a small pre-scanned patch around the
vehicle, then returns both
actors to their initial positions. `Radius +/-`, `Density +/-`, `Next env`,
and `Save` change the model or write a snapshot. `Save` writes a PNG and NPZ
under `/tmp/uav_observation_preview` by default. The survey spacing can be set
with `--survey-spacing-m`.

For Path MeanFlow mask visualization, use the physical observation source:

```bash
MPLCONFIGDIR=/tmp/mpt-mplconfig \
/home/sdu/miniconda3/envs/sem-map/bin/python vis_dit.py \
  --workflow stage1 --mask_source uav ...
```

`uav` builds the geometric mask from accumulated nadir-LiDAR footprints and
separate local circular obstacle footprints. It does not use the old random
ellipse generator. `legacy` remains available only for reproducing figures or
checkpoints trained with the previous ellipse semantics; it should not be
treated as the physical observation model. The retrained Stage 1 probe is
stored at `data/stage1_uav_mask_retrained_probe/stage1_best.pth` and records
`mask_source=uav` together with
`uav_footprint_observation_circle_obstacles_v1` in its metadata. It is a
2000-update/one-epoch CPU validation run, so it verifies the new training path
and checkpoint contract but is not a converged final model.

The Stage 1 training entry point now defaults to the same source and records it
in the checkpoint. It can also be made explicit:

```bash
.../python scripts/run_stage1_with_local_runtime.py \
  --workflow stage1 --mask_source uav ...
```

This is the observation model and its interactive interface. The route is
used as the UAV/vehicle geometry for the observation contract, while the red
vehicle reference trajectory remains separate from the UAV survey route. The
model separates map-building progress from vehicle progress so vehicle motion
is constrained by known support. Point returns and support remain separate;
altitude, footprint radius, density, and survey spacing are explicit final
parameters rather than a second temporary mask rule.

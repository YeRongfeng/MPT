# Combined Stage 1 dataset

The requested Stage 1 run uses a symlinked view at
`data/stage1_combined`. Source environments are not copied or modified:

- `data/dataset1`
- `/home/sdu/uneven_planner/dataset/public_terrain_20m/desert`
- `/home/sdu/uneven_planner/dataset/public_terrain_20m/forest`

Environment names are prefixed (`dataset1_env*`, `desert_env*`, and
`forest_env*`) so identical numeric names cannot collide. The generated
`combined_manifest.json` records the source roots and contains 300 environments
in each split. The frozen Stage 1 request is:

- split seed: `20260821`
- training environments: `240`
- validation environments: `60`
- source distribution: dataset1 `80/20`, desert `78/22`, forest `82/18`
- loader contexts: `32,000` train paths and `800` validation paths

The intended launch command is:

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
MPLCONFIGDIR=/tmp/mpt-mplconfig \
/home/sdu/miniconda3/envs/sem-map/bin/python \
  scripts/run_stage1_with_local_runtime.py \
  --workflow stage1 \
  --dataFolder data/stage1_combined \
  --fileDir data/stage1_uav_mask_retrained \
  --batchSize 16 \
  --num_workers 0 \
  --seed 20260821 \
  --mask_seed 20260821 \
  --stage1_split_seed 20260821 \
  --stage1_train_environments 240 \
  --stage1_val_environments 60 \
  --stage1_epochs 20 \
  --mask_source uav
```

The Codex sandbox may not expose the host CUDA device even when the machine has
a GPU. The full command therefore remains the reproducible host-side launch;
the current partial run is in `data/stage1_uav_mask_retrained`, while the
one-epoch audit checkpoint is in `data/stage1_uav_mask_retrained_probe`. Both
record `--mask_source uav`; neither changes the source terrain directories.

## Stability-map overlay

The source desert and forest environments are symlinked from
`/home/sdu/uneven_planner`, so their source directories are read-only inputs for
this repository. The existing precomputation tool
`tools/data/generate_stability_maps.py` was run against the local overlay
`data/stage1_combined_with_stability`; it reused the 200 dataset1 caches and
generated 400 desert/forest train/val caches locally. No files were written to
the external uneven-planner dataset.

For visualization with the same validity diagnostics, use the overlay as the
dataset root and request multiple samples per path:

```bash
.../python vis_dit.py \
  --workflow stage1 \
  --checkpoint data/stage1_uav_mask_retrained/stage1_best.pth \
  --dataset_root data/stage1_combined_with_stability \
  --mask_source uav \
  --num_samples 8
```

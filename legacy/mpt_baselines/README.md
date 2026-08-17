# Original MPT baselines

These modules implement superseded OMPL, RRT*, Dubins-car, MPNet, and UNet
workflows. They are retained for historical comparison and are not part of the
current terrain-aware Stage 1 pipeline.

The bundled OMPL extension was built for the original Python 3.8 environment
and is not compatible with the current Python 3.10 `vim` environment. OMPL
entry points require the archived Docker/runtime described in the legacy
documentation.

Run a legacy entry point from the repository root with module syntax, for
example:

```bash
python -m legacy.mpt_baselines.rrt_star_map --help
```

Historical usage notes are archived in
`docs/legacy/ORIGINAL_MPT_README.md`.

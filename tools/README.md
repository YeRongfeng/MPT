# Standalone tools

- `data/`: maintained dataset commands, currently stability-map generation.
- `_paths.py`: shared `predictions/` and `tests/audits/` output locations.

Hard-coded, one-off utilities live in `tests/audits/legacy_scripts/`. Original
MPT baseline generators and notebooks live in `legacy/mpt_baselines/`.

Run Python tools from the repository root with `python -m`, for example:

```bash
python -m tools.data.generate_stability_maps --help
```

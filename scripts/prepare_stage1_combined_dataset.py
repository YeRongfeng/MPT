#!/usr/bin/env python3
"""Create a symlinked, uniquely named Stage-1 dataset view.

The source datasets remain untouched.  The combined view lets the existing
loader consume dataset1, public desert, and public forest through one root
while preserving each physical environment as a distinct name.
"""

import argparse
import json
import os
from pathlib import Path


SOURCES = {
    "dataset1": Path("data/dataset1"),
    "desert": Path("/home/sdu/uneven_planner/dataset/public_terrain_20m/desert"),
    "forest": Path("/home/sdu/uneven_planner/dataset/public_terrain_20m/forest"),
}


def _environment_dirs(split_root):
    environments = sorted(path for path in split_root.iterdir() if path.is_dir())
    if not environments:
        raise ValueError(f"no environments found under {split_root}")
    for environment in environments:
        if not (environment / "map.p").is_file():
            raise ValueError(f"missing map.p: {environment}")
        if not any(environment.glob("path_*.p")):
            raise ValueError(f"missing path_*.p: {environment}")
    return environments


def build_view(output_root):
    output_root = Path(output_root)
    manifest = {
        "format": "stage1_combined_symlink_view_v1",
        "output_root": str(output_root.resolve()),
        "sources": {},
        "splits": {},
    }
    for source_name, source_root in SOURCES.items():
        source_root = source_root.resolve()
        if not source_root.is_dir():
            raise FileNotFoundError(source_root)
        manifest["sources"][source_name] = str(source_root)

    for split in ("train", "val"):
        split_output = output_root / split
        split_output.mkdir(parents=True, exist_ok=True)
        split_manifest = {}
        for source_name, source_root in SOURCES.items():
            source_environments = _environment_dirs(source_root / split)
            names = []
            for environment in source_environments:
                combined_name = f"{source_name}_{environment.name}"
                target = split_output / combined_name
                source = environment.resolve()
                if target.exists() or target.is_symlink():
                    if not target.is_symlink() or target.resolve() != source:
                        raise FileExistsError(
                            f"refusing to overwrite existing path: {target}"
                        )
                else:
                    target.symlink_to(source, target_is_directory=True)
                names.append(combined_name)
            split_manifest[source_name] = names
        manifest["splits"][split] = split_manifest

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "combined_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="data/stage1_combined")
    args = parser.parse_args()
    manifest = build_view(args.output_root)
    counts = {
        split: sum(len(names) for names in sources.values())
        for split, sources in manifest["splits"].items()
    }
    print(json.dumps({"output_root": manifest["output_root"], "counts": counts}, indent=2))


if __name__ == "__main__":
    main()

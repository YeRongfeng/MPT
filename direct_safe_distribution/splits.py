"""Deterministic, auditable map-level train/validation/test splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


SPLIT_NAMES = ("train", "validation", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def discover_map_ids(dataset_folder: Path) -> List[str]:
    map_ids = [
        path.name
        for path in sorted(dataset_folder.iterdir())
        if path.is_dir()
        and (path / "map.p").is_file()
        and not path.name.endswith("_optimized")
    ]
    if not map_ids:
        raise ValueError(f"No map environments found in {dataset_folder}")
    return map_ids


def deterministic_map_split(
    map_ids: Sequence[str],
    *,
    seed: int,
    train_count: int,
    validation_count: int,
    test_count: int,
) -> Dict[str, List[str]]:
    """Assign maps by a stable SHA-256 ordering, independent of Python RNG."""

    unique = sorted(set(map_ids))
    if len(unique) != len(map_ids):
        raise ValueError("map_ids contains duplicates")
    requested = train_count + validation_count + test_count
    if requested != len(unique):
        raise ValueError(
            f"Split counts sum to {requested}, but {len(unique)} maps were found"
        )
    ordered = sorted(
        unique,
        key=lambda map_id: (
            hashlib.sha256(f"{seed}|{map_id}".encode("utf-8")).digest(),
            map_id,
        ),
    )
    train_end = train_count
    val_end = train_end + validation_count
    result = {
        "train": sorted(ordered[:train_end]),
        "validation": sorted(ordered[train_end:val_end]),
        "test": sorted(ordered[val_end:]),
    }
    assert_disjoint_splits(result)
    return result


def assert_disjoint_splits(splits: Mapping[str, Iterable[str]]) -> None:
    missing = set(SPLIT_NAMES) - set(splits)
    if missing:
        raise ValueError(f"Missing split names: {sorted(missing)}")
    materialized = {name: list(splits[name]) for name in SPLIT_NAMES}
    for name, values in materialized.items():
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate map IDs within {name}")
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            overlap = sorted(set(materialized[left]) & set(materialized[right]))
            if overlap:
                raise ValueError(f"{left}/{right} overlap: {overlap}")


def build_manifest(
    dataset_folder: Path,
    *,
    seed: int = 20260728,
    train_count: int = 70,
    validation_count: int = 15,
    test_count: int = 15,
) -> Dict[str, object]:
    map_ids = discover_map_ids(dataset_folder)
    splits = deterministic_map_split(
        map_ids,
        seed=seed,
        train_count=train_count,
        validation_count=validation_count,
        test_count=test_count,
    )
    map_hashes = {
        map_id: sha256_file(dataset_folder / map_id / "map.p")
        for map_id in sorted(map_ids)
    }
    path_counts = {
        map_id: len(list((dataset_folder / map_id).glob("path_*.p")))
        for map_id in sorted(map_ids)
    }
    hash_payload = {
        "schema_version": 1,
        "dataset_folder": str(dataset_folder.resolve()),
        "seed": seed,
        "splits": splits,
        "map_sha256": map_hashes,
        "path_counts": path_counts,
    }
    manifest = {
        **hash_payload,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": canonical_json_hash(hash_payload),
        "checks": {
            "pairwise_disjoint": True,
            "all_maps_assigned_once": (
                sum(len(splits[name]) for name in SPLIT_NAMES) == len(map_ids)
            ),
            "counts": {name: len(splits[name]) for name in SPLIT_NAMES},
        },
        "protocol": {
            "stage1_allowed_splits": ["train"],
            "teacher_allowed_splits": ["train"],
            "early_stopping_allowed_splits": ["validation"],
            "final_evaluation_only_splits": ["test"],
        },
    }
    return manifest


def verify_selected_split_files(
    manifest: Mapping[str, object],
    split_names: Sequence[str],
) -> None:
    """Hash only explicitly authorized splits.

    Training must not even open strict-test map files merely to verify them.
    Full-dataset assignment checks belong to manifest creation or the final
    explicitly authorized test evaluation.
    """

    unknown = set(split_names) - set(SPLIT_NAMES)
    if unknown:
        raise ValueError(f"Unknown split names: {sorted(unknown)}")
    splits = manifest["splits"]
    assert_disjoint_splits(splits)
    dataset_folder = Path(manifest["dataset_folder"])
    selected_ids = [
        map_id for split in split_names for map_id in splits[split]
    ]
    for map_id in selected_ids:
        map_path = dataset_folder / map_id / "map.p"
        if not map_path.is_file():
            raise FileNotFoundError(f"Missing authorized map file: {map_path}")
        observed = sha256_file(dataset_folder / map_id / "map.p")
        expected = manifest["map_sha256"][map_id]
        if observed != expected:
            raise ValueError(
                f"{map_id} map hash mismatch: expected {expected}, "
                f"observed {observed}"
            )
    # Different IDs with byte-identical maps across the authorized splits are
    # also leakage.  Test hashes are not dereferenced here.
    owner_by_hash: Dict[str, str] = {}
    for split in split_names:
        for map_id in splits[split]:
            digest = manifest["map_sha256"][map_id]
            previous = owner_by_hash.get(digest)
            if previous is not None and previous != split:
                raise ValueError(
                    f"Byte-identical maps cross splits: hash={digest}, "
                    f"splits={previous}/{split}"
                )
            owner_by_hash[digest] = split


def verify_manifest_dataset(manifest: Mapping[str, object]) -> None:
    """Full verification for manifest creation/audit or authorized final test."""

    dataset_folder = Path(manifest["dataset_folder"])
    splits = manifest["splits"]
    all_ids = [
        map_id for split in SPLIT_NAMES for map_id in splits[split]
    ]
    observed_ids = discover_map_ids(dataset_folder)
    if set(all_ids) != set(observed_ids):
        raise ValueError(
            "Manifest/dataset map IDs differ: "
            f"missing={sorted(set(observed_ids) - set(all_ids))}, "
            f"extra={sorted(set(all_ids) - set(observed_ids))}"
        )
    verify_selected_split_files(manifest, SPLIT_NAMES)


def load_manifest(path: Path, *, verify_files: bool = True) -> Dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert_disjoint_splits(manifest["splits"])
    hash_payload = {
        key: manifest[key]
        for key in (
            "schema_version",
            "dataset_folder",
            "seed",
            "splits",
            "map_sha256",
            "path_counts",
        )
    }
    observed = canonical_json_hash(hash_payload)
    if observed != manifest["manifest_sha256"]:
        raise ValueError(
            f"Manifest hash mismatch: stored {manifest['manifest_sha256']}, "
            f"observed {observed}"
        )
    if verify_files:
        verify_manifest_dataset(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-folder", type=Path, default=Path("data/dataset0/train")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "diagnostics/direct_safe_distribution/split_manifest.json"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--train-count", type=int, default=70)
    parser.add_argument("--validation-count", type=int, default=15)
    parser.add_argument("--test-count", type=int, default=15)
    args = parser.parse_args()
    manifest = build_manifest(
        args.dataset_folder,
        seed=args.seed,
        train_count=args.train_count,
        validation_count=args.validation_count,
        test_count=args.test_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {args.output} ({manifest['manifest_sha256']}); "
        f"counts={manifest['checks']['counts']}"
    )


if __name__ == "__main__":
    main()

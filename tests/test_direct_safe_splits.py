import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from direct_safe_distribution.splits import (
    assert_disjoint_splits,
    build_manifest,
    deterministic_map_split,
    load_manifest,
    verify_manifest_dataset,
    verify_selected_split_files,
)


class SplitTests(unittest.TestCase):
    def test_deterministic_split_is_disjoint_and_stable(self):
        map_ids = [f"env{i:06d}" for i in range(20)]
        first = deterministic_map_split(
            map_ids, seed=7, train_count=12, validation_count=4, test_count=4
        )
        second = deterministic_map_split(
            list(reversed(map_ids)),
            seed=7,
            train_count=12,
            validation_count=4,
            test_count=4,
        )
        self.assertEqual(first, second)
        assert_disjoint_splits(first)
        self.assertEqual(set().union(*map(set, first.values())), set(map_ids))

    def test_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            assert_disjoint_splits(
                {
                    "train": ["env0"],
                    "validation": ["env1"],
                    "test": ["env0"],
                }
            )

    def test_manifest_hash_detects_tampering(self):
        with TemporaryDirectory() as temp:
            tmp_path = Path(temp)
            dataset = tmp_path / "dataset"
            for index in range(4):
                env = dataset / f"env{index:06d}"
                env.mkdir(parents=True)
                (env / "map.p").write_bytes(f"map-{index}".encode())
                (env / "path_0.p").write_bytes(b"path")
            manifest = build_manifest(
                dataset,
                seed=3,
                train_count=2,
                validation_count=1,
                test_count=1,
            )
            path = tmp_path / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(
                load_manifest(path, verify_files=True)["manifest_sha256"],
                manifest["manifest_sha256"],
            )

            changed = json.loads(path.read_text(encoding="utf-8"))
            changed["splits"]["train"][0] = "tampered"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_manifest(path, verify_files=False)

    def test_split_aware_verification_does_not_open_test_files(self):
        with TemporaryDirectory() as temp:
            dataset = Path(temp) / "dataset"
            for index in range(3):
                env = dataset / f"env{index:06d}"
                env.mkdir(parents=True)
                (env / "map.p").write_bytes(f"map-{index}".encode())
                (env / "path_0.p").write_bytes(b"path")
            manifest = build_manifest(
                dataset,
                seed=4,
                train_count=1,
                validation_count=1,
                test_count=1,
            )
            test_id = manifest["splits"]["test"][0]
            (dataset / test_id / "map.p").write_bytes(b"changed-test")
            # Train/validation verification must not dereference the test map.
            verify_selected_split_files(
                manifest, ("train", "validation")
            )
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_manifest_dataset(manifest)


if __name__ == "__main__":
    unittest.main()

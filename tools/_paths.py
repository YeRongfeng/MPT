"""Shared locations for generated project artifacts."""

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_ROOT = Path(
    os.environ.get("MPT_PREDICTIONS_DIR", REPO_ROOT / "predictions")
)
EVALUATION_RESULTS_ROOT = Path(
    os.environ.get(
        "MPT_EVALUATION_RESULTS_DIR", REPO_ROOT / "evaluation_results"
    )
)
TEST_ROOT = Path(
    os.environ.get("MPT_TEST_DIR", REPO_ROOT / "tests" / "audits")
)


def predictions_dir(*parts: str) -> Path:
    path = PREDICTIONS_ROOT.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def evaluation_results_dir(*parts: str) -> Path:
    path = EVALUATION_RESULTS_ROOT.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_dir(*parts: str) -> Path:
    path = TEST_ROOT.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path

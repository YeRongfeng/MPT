"""Make repository modules importable when a visualization is run by path."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

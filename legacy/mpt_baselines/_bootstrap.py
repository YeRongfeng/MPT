"""Compatibility helpers for bundled legacy dependencies."""

import sys
from pathlib import Path


def enable_local_packages() -> None:
    legacy_root = str(Path(__file__).resolve().parent)
    if legacy_root not in sys.path:
        sys.path.insert(0, legacy_root)

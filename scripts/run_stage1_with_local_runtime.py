#!/usr/bin/env python3
"""Run train_flow with the locally available Python 3.8 TensorBoard package.

The sem-map environment supplies Python 3.10, torch, and timm, while this
repository's virtualenv supplies TensorBoard.  Keep the compatibility shim in
one place instead of requiring an interactive PYTHONPATH recipe.
"""

import runpy
import sys
from pathlib import Path

import numpy as np


if "object" not in np.__dict__:
    np.object = object
if "bool" not in np.__dict__:
    np.bool = bool

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.append("/home/sdu/MPT/.venv/lib/python3.8/site-packages")
runpy.run_path(str(repo_root / "train_flow.py"), run_name="__main__")

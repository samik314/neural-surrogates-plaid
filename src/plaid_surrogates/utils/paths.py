"""Project paths: env vars when set, otherwise defaults inside the repo root.

PLAID_DATA_DIR -> <repo>/data, PLAID_RUNS_DIR -> <repo>/runs, HF_HOME -> <repo>/data/hf_cache.
Both data/ and runs/ are gitignored.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def repo_root() -> Path:
    """Repo root: the directory holding pyproject.toml, three levels above this file."""
    root = Path(__file__).resolve().parents[3]
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError(
            f"No repo root above {__file__}; set PLAID_DATA_DIR, PLAID_RUNS_DIR, HF_HOME."
        )
    return root


def _from_env(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else default


def data_dir() -> Path:
    """Root for converted caches (data/cache/<dataset>) and split files (data/splits)."""
    return _from_env("PLAID_DATA_DIR", repo_root() / "data")


def runs_dir() -> Path:
    """Root for run folders."""
    return _from_env("PLAID_RUNS_DIR", repo_root() / "runs")


def hf_home() -> Path:
    """Hugging Face cache location (dataset downloads)."""
    return _from_env("HF_HOME", repo_root() / "data" / "hf_cache")


def set_hf_home_env() -> Path:
    """Export HF_HOME (default if unset) for this process. Must run before importing HF libraries.

    `datasets`/`huggingface_hub` read HF_HOME once at import time, so setting it afterwards would be
    silently ignored and downloads would land in ~/.cache/huggingface. We refuse instead.
    """
    already = [m for m in ("datasets", "huggingface_hub") if m in sys.modules]
    path = hf_home()
    if already and os.environ.get("HF_HOME") != str(path):
        raise RuntimeError(
            f"{already} already imported; call set_hf_home_env() before importing them"
        )
    os.environ["HF_HOME"] = str(path)
    return path

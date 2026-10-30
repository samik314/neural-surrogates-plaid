"""Run provenance: git state, environment snapshot, pip freeze vs lock file (docs/methods.md §5)."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = (
    "torch",
    "torchvision",
    "nvidia-physicsnemo",
    "pyplaid",
    "numpy",
    "datasets",
    "transformer-engine",
)


def _git(repo_dir: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), *args], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout


def git_info(repo_dir: Path) -> dict[str, Any]:
    """Commit hash, dirty flag, diff vs HEAD, and untracked files; all None outside a git repo."""
    commit = _git(repo_dir, "rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None, "diff": None, "untracked": None}
    status = _git(repo_dir, "status", "--porcelain") or ""
    untracked = [line[3:] for line in status.splitlines() if line.startswith("??")]
    return {
        "commit": commit.strip(),
        "dirty": bool(status.strip()),
        "diff": _git(repo_dir, "diff", "HEAD", "--binary") or "",
        "untracked": untracked,
    }


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def pip_freeze() -> list[str]:
    """`pip freeze` of this interpreter; `name @ file://...` lines are pinned to versions."""
    out = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    ).stdout
    return [_normalize_freeze_line(line) for line in out.splitlines() if line.strip()]


def _normalize_freeze_line(line: str) -> str:
    # conda-installed packages show up as "name @ file:///..."; the lock stores "name==version".
    if " @ file://" in line:
        name = line.split(" @ ", 1)[0].strip()
        v = _pkg_version(name)
        return f"{name}=={v}" if v else line
    return line.strip()


def _requirement_lines(lines: list[str]) -> set[str]:
    """Package pins only: drop comments, blank lines, and pip options like --extra-index-url."""
    return {ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith(("#", "-"))}


def compare_to_lock(freeze: list[str], lock_lines: list[str]) -> dict[str, Any]:
    """Diff installed packages against the lock file (case-insensitive on names)."""
    have = {ln.lower() for ln in _requirement_lines(freeze)}
    want = {ln.lower() for ln in _requirement_lines(lock_lines)}
    return {
        "matches": have == want,
        "not_in_lock": sorted(have - want),
        "missing_from_env": sorted(want - have),
    }


def env_snapshot(lock_path: Path | None = None) -> dict[str, Any]:
    """Python/platform/package versions, CUDA and GPU info, machine, Slurm job, lock comparison."""
    import torch  # local import: git/lock helpers stay usable without loading torch

    freeze = pip_freeze()
    snap: dict[str, Any] = {
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "packages": {p: _pkg_version(p) for p in TRACKED_PACKAGES},
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "pip_freeze_sha256": hashlib.sha256("\n".join(sorted(freeze)).encode()).hexdigest(),
        "lock": None,
    }
    if lock_path is not None and lock_path.is_file():
        snap["lock"] = {
            "path": str(lock_path),
            **compare_to_lock(freeze, lock_path.read_text().splitlines()),
        }
    return snap


def write_provenance(
    run_dir: Path, repo_dir: Path, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Write provenance.json, pip_freeze.txt, and git_diff.patch (if dirty) into `run_dir`.

    `extra` holds run-specific fields (seed, precision, device, config path, ...).
    Returns the provenance dict. A dirty tree without a patch can't be promoted
    (docs/methods.md §5),
    so the patch is written whenever the tree is dirty.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    git = git_info(repo_dir)
    snap = env_snapshot(repo_dir / "requirements.lock")
    prov = {
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "git_untracked": git["untracked"],
        "environment": snap,
        **(extra or {}),
    }
    if git["dirty"]:
        (run_dir / "git_diff.patch").write_text(git["diff"])
        prov["git_diff_file"] = "git_diff.patch"
    (run_dir / "pip_freeze.txt").write_text("\n".join(pip_freeze()) + "\n")
    (run_dir / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    return prov

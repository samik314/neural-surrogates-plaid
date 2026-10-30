"""Run folders and the markdown run logs (docs/methods.md §5)."""

from __future__ import annotations

import datetime as dt
import fcntl
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from plaid_surrogates.utils.paths import repo_root, runs_dir

EXPERIMENT_COLUMNS = (
    "date",
    "run id",
    "status",
    "dataset",
    "model",
    "parent run",
    "key change",
    "seed/fold",
    "params",
    "wall time",
    "machine / GPU",
    "dev total_error",
    "worst output (RRMSE)",
    "notes",
)

SUBMISSION_COLUMNS = (
    "date",
    "run id",
    "dataset",
    "dev total_error (mean ± std if seeds/CV)",
    "leaderboard score",
    "local→leaderboard gap",
    "notes",
)


def log_row_text(row: Mapping[str, Any], dataset: str, leaderboard: str) -> str:
    """LOG_ROW.md for a submission folder: which board the file goes to, then the row to paste.

    Every 2D_MultiScHypEl "failure" of 2026-10-07..14 was a file uploaded to the Tensile2d board,
    so the target board is named first, where it can't be missed.
    """
    return (
        f"UPLOAD TO: {leaderboard or 'UNKNOWN'}\n"
        f"(the {dataset} board; check the page title before uploading)\n\n"
        "Paste into submissions/LOG.md after uploading:\n\n| "
        + " | ".join(str(row.get(c, "")) for c in SUBMISSION_COLUMNS)
        + " |\n"
    )


def experiments_log_path() -> Path:
    return repo_root() / "experiments" / "LOG.md"


def run_name(
    dataset: str,
    model: str,
    tag: str,
    seed: int,
    fold: int | None = None,
    now: dt.datetime | None = None,
    full: bool = False,
) -> str:
    """`<YYYYMMDD-HHMM>_<dataset>_<model>_<tag>_s<seed>[_f<fold> | _full]`."""
    if full and fold is not None:
        raise ValueError("a run is either a CV fold or a full retrain, not both")
    stamp = (now or dt.datetime.now()).strftime("%Y%m%d-%H%M")
    name = f"{stamp}_{dataset}_{model}_{tag}_s{seed}"
    if full:
        return f"{name}_full"
    return name if fold is None else f"{name}_f{fold}"


def make_run_dir(
    dataset: str,
    model: str,
    tag: str,
    seed: int,
    fold: int | None = None,
    root: Path | None = None,
    now: dt.datetime | None = None,
    full: bool = False,
) -> Path:
    """Create a fresh run folder; never reuses an existing one (a clash gets a -1, -2... suffix)."""
    root = root or runs_dir()
    base = run_name(dataset, model, tag, seed, fold, now, full)
    for k in range(100):
        path = root / (base if k == 0 else f"{base}-{k}")
        try:
            path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return path
    raise RuntimeError(f"could not create a unique run folder for {base}")


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def append_table_row(path: Path, columns: Sequence[str], values: Mapping[str, Any]) -> None:
    """Append one row to the markdown table whose header is exactly `columns`.

    The row goes right after the table's last row, so other tables in the file (e.g. the
    storage table below the run table in experiments/LOG.md) are left alone.
    """
    unknown = set(values) - set(columns)
    if unknown:
        raise ValueError(f"unknown log columns {sorted(unknown)}")
    header = "| " + " | ".join(columns) + " |"
    row = "| " + " | ".join(_cell(values.get(c, "")) for c in columns) + " |"
    # Read-modify-write under an exclusive POSIX lock, so parallel jobs (e.g. a seed array)
    # finishing at the same moment can't drop each other's rows.
    with open(path, "r+") as fh:
        fcntl.lockf(fh, fcntl.LOCK_EX)
        try:
            lines = fh.read().splitlines()
            try:
                h = next(k for k, line in enumerate(lines) if line.strip() == header)
            except StopIteration:
                raise ValueError(f"table header not found in {path}") from None
            end = h + 1
            while end < len(lines) and lines[end].lstrip().startswith("|"):
                end += 1
            lines.insert(end, row)
            fh.seek(0)
            fh.write("\n".join(lines) + "\n")
            fh.truncate()
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.lockf(fh, fcntl.LOCK_UN)

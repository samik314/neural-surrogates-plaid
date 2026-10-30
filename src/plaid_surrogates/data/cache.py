"""Converted-cache format: one .npz per sample + index.json. No PLAID import (docs/methods.md §6).

Layout of <cache_dir>:
    index.json               provenance, problem names, splits (benchmark order), per-sample info
    samples/<id:06d>.npz     flat keys: points, cells, sample_id, in_scalar/<n>, out_field/<n>,
                             out_scalar/<n>, tag/<n>

Invariants: split ID lists are stored exactly as given (never sorted); node order is untouched;
samples without outputs (test/OOD) carry no out_* keys at all.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

INDEX_FILE = "index.json"
SAMPLES_DIR = "samples"
FORMAT_VERSION = 1


@dataclass
class CachedSample:
    """One mesh sample in plain numpy form."""

    sample_id: int
    points: np.ndarray  # (N, 2) float64
    cells: np.ndarray  # (M, 3) int64, 0-based triangle connectivity
    input_scalars: dict[str, float]
    output_fields: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (N,) float64
    output_scalars: dict[str, float] = field(default_factory=dict)
    nodal_tags: dict[str, np.ndarray] = field(default_factory=dict)  # name -> 0-based node indices

    @property
    def n_nodes(self) -> int:
        return int(self.points.shape[0])

    @property
    def has_outputs(self) -> bool:
        return bool(self.output_fields or self.output_scalars)


def validate_sample(s: CachedSample) -> None:
    """Raise ValueError on any shape, dtype, index-range, or finiteness problem."""
    sid = f"sample {s.sample_id}"
    if s.points.ndim != 2 or s.points.shape[1] != 2 or s.points.dtype != np.float64:
        raise ValueError(
            f"{sid}: points must be (N, 2) float64, got {s.points.shape} {s.points.dtype}"
        )
    if s.cells.ndim != 2 or s.cells.shape[1] != 3 or s.cells.dtype != np.int64:
        raise ValueError(f"{sid}: cells must be (M, 3) int64, got {s.cells.shape} {s.cells.dtype}")
    n = s.n_nodes
    if not np.isfinite(s.points).all():
        raise ValueError(f"{sid}: non-finite coordinates")
    if s.cells.size and (s.cells.min() < 0 or s.cells.max() >= n):
        raise ValueError(f"{sid}: cell indices outside [0, {n})")
    for name, f in s.output_fields.items():
        if f.shape != (n,) or f.dtype != np.float64:
            raise ValueError(f"{sid}: field {name} must be ({n},) float64, got {f.shape} {f.dtype}")
        if not np.isfinite(f).all():
            raise ValueError(f"{sid}: field {name} has non-finite values")
    for kind, scalars in (("input", s.input_scalars), ("output", s.output_scalars)):
        for name, v in scalars.items():
            if not np.isfinite(v):
                raise ValueError(f"{sid}: {kind} scalar {name} is not finite ({v})")
    for name, idx in s.nodal_tags.items():
        if idx.ndim != 1 or idx.dtype != np.int64:
            raise ValueError(f"{sid}: tag {name} must be 1-D int64")
        if idx.size and (idx.min() < 0 or idx.max() >= n):
            raise ValueError(f"{sid}: tag {name} indices outside [0, {n})")


def sample_filename(sample_id: int) -> str:
    return f"{SAMPLES_DIR}/{sample_id:06d}.npz"


def _atomic_write_bytes(path: Path, write) -> None:
    # Write to a temp name then rename, so a killed job never leaves a half-written file.
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        write(fh)
    os.replace(tmp, path)


def save_sample(cache_dir: Path, s: CachedSample) -> Path:
    """Validate and write one sample; returns its path."""
    validate_sample(s)
    arrays: dict[str, np.ndarray] = {
        "sample_id": np.asarray(s.sample_id, dtype=np.int64),
        "points": s.points,
        "cells": s.cells,
    }
    for prefix, d in (("in_scalar", s.input_scalars), ("out_scalar", s.output_scalars)):
        arrays.update({f"{prefix}/{k}": np.asarray(v, dtype=np.float64) for k, v in d.items()})
    arrays.update({f"out_field/{k}": v for k, v in s.output_fields.items()})
    arrays.update({f"tag/{k}": v for k, v in s.nodal_tags.items()})
    path = cache_dir / sample_filename(s.sample_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(path, lambda fh: np.savez(fh, **arrays))
    return path


def load_sample(path: Path) -> CachedSample:
    """Read one sample written by `save_sample` (no pickle)."""
    with np.load(path, allow_pickle=False) as z:
        groups: dict[str, dict[str, Any]] = {
            "in_scalar": {},
            "out_scalar": {},
            "out_field": {},
            "tag": {},
        }
        for key in z.files:
            if "/" in key:
                prefix, name = key.split("/", 1)
                groups[prefix][name] = z[key]
        return CachedSample(
            sample_id=int(z["sample_id"]),
            points=z["points"],
            cells=z["cells"],
            input_scalars={k: float(v) for k, v in groups["in_scalar"].items()},
            output_fields=groups["out_field"],
            output_scalars={k: float(v) for k, v in groups["out_scalar"].items()},
            nodal_tags=groups["tag"],
        )


def write_index(
    cache_dir: Path,
    meta: dict[str, Any],
    splits: dict[str, list[int]],
    samples: dict[int, dict[str, Any]],
) -> Path:
    """Write index.json last, so a cache without it is recognisably incomplete."""
    for name, ids in splits.items():
        if len(set(ids)) != len(ids):
            raise ValueError(f"split {name} has duplicate IDs")
        missing = [i for i in ids if i not in samples]
        if missing:
            raise ValueError(f"split {name} references uncached samples, e.g. {missing[:5]}")
    index = {
        "format_version": FORMAT_VERSION,
        **meta,
        "splits": {name: [int(i) for i in ids] for name, ids in splits.items()},
        "samples": {str(i): info for i, info in samples.items()},
    }
    path = cache_dir / INDEX_FILE
    text = json.dumps(index, indent=1) + "\n"
    _atomic_write_bytes(path, lambda fh: fh.write(text.encode()))
    return path


def load_index(cache_dir: Path) -> dict[str, Any]:
    """Read index.json; raises FileNotFoundError if the cache is missing or incomplete."""
    return json.loads((cache_dir / INDEX_FILE).read_text())


def split_ids(index: dict[str, Any], split: str) -> list[int]:
    """Sample IDs of `split` in stored (= benchmark) order. Never sort this list."""
    try:
        return list(index["splits"][split])
    except KeyError:
        raise KeyError(f"No split {split!r}; available: {list(index['splits'])}") from None


def load_split(cache_dir: Path, split: str) -> list[CachedSample]:
    """All samples of `split`, in stored order."""
    index = load_index(cache_dir)
    return [
        load_sample(cache_dir / index["samples"][str(i)]["file"]) for i in split_ids(index, split)
    ]

"""Leaderboard submission file (docs/methods.md §3).

Format (from the official plaid-benchmarks writers): a pickle of a list with one dict per sample,
in the split's stored order (= problem.get_split order, never sorted). Each dict maps every scored
field to a 1-D float64 array on that sample's own nodes (original order) and every scored scalar
to a float.

Callers pass predictions keyed by sample ID; the writer alone decides the order, from index.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from plaid_surrogates.data.cache import load_index, split_ids

# Protocol 4 loads on any Python >= 3.4. (Arrays pickled by numpy 2.x need numpy >= 1.26 to load.)
PICKLE_PROTOCOL = 4


def build_records(
    preds_by_id: Mapping[int, Mapping[str, Any]],
    ids: Sequence[int],
    n_nodes: Mapping[int, int],
    fields: Sequence[str],
    scalars: Sequence[str],
) -> list[dict[str, Any]]:
    """Validate predictions and arrange them in `ids` order. Raises ValueError on any problem."""
    ids = [int(i) for i in ids]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate IDs in the split")
    got = {int(i) for i in preds_by_id}
    missing = [i for i in ids if i not in got]
    extra = sorted(got - set(ids))
    if missing or extra:
        raise ValueError(
            f"prediction IDs don't match the split: missing {missing[:5]}, extra {extra[:5]}"
        )
    expected = set(fields) | set(scalars)
    records: list[dict[str, Any]] = []
    for i in ids:
        rec = preds_by_id[i]
        if set(rec) != expected:
            raise ValueError(f"sample {i}: keys {sorted(rec)} != scored outputs {sorted(expected)}")
        out: dict[str, Any] = {}
        for f in fields:
            a = np.array(rec[f], dtype=np.float64)  # copy: the file never aliases model memory
            if a.ndim != 1 or a.shape[0] != n_nodes[i]:
                raise ValueError(
                    f"sample {i}: field {f} has shape {a.shape}, mesh has {n_nodes[i]} nodes"
                )
            if not np.isfinite(a).all():
                raise ValueError(f"sample {i}: field {f} has non-finite values")
            out[f] = a
        for s in scalars:
            v = np.asarray(rec[s], dtype=np.float64)
            if v.ndim != 0 or not np.isfinite(v):
                raise ValueError(f"sample {i}: scalar {s} must be one finite number, got {v!r}")
            out[s] = float(v)
        records.append(out)
    return records


def write_submission(
    records: list[dict[str, Any]],
    out_path: Path,
    meta: Mapping[str, Any],
    meta_path: Path | None = None,
) -> dict[str, Any]:
    """Pickle `records` atomically and write the metadata sidecar; returns the full metadata."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = pickle.dumps(records, protocol=PICKLE_PROTOCOL)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, out_path)
    full = {
        **meta,
        "file": out_path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "n_samples": len(records),
        "pickle_protocol": PICKLE_PROTOCOL,
        "numpy_version": np.__version__,
    }
    meta_path = meta_path or out_path.parent / "submission_meta.json"
    meta_path.write_text(json.dumps(full, indent=2, default=str) + "\n")
    return full


def read_submission(path: Path) -> list[dict[str, Any]]:
    """Load a submission file we wrote ourselves (pickle: never use on untrusted files)."""
    with open(path, "rb") as fh:
        records = pickle.load(fh)
    if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
        raise ValueError(f"{path} is not a list of dicts")
    return records


def write_split_submission(
    preds_by_id: Mapping[int, Mapping[str, Any]],
    cache_dir: Path,
    split: str | Sequence[int],
    fields: Sequence[str],
    scalars: Sequence[str],
    out_path: Path,
    meta: Mapping[str, Any],
    meta_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate, order, and write predictions for a cached split (a name, or an explicit ID list
    such as the dev split). Node counts come from index.json. Returns (records, metadata)."""
    index = load_index(cache_dir)
    ids = split_ids(index, split) if isinstance(split, str) else [int(i) for i in split]
    n_nodes = {i: int(index["samples"][str(i)]["n_nodes"]) for i in ids}
    records = build_records(preds_by_id, ids, n_nodes, fields, scalars)
    full = write_submission(
        records,
        out_path,
        {
            **meta,
            "dataset": index.get("dataset"),
            "split": split if isinstance(split, str) else "explicit id list",
            "sample_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "scored_fields": list(fields),
            "scored_scalars": list(scalars),
        },
        meta_path,
    )
    return records, full

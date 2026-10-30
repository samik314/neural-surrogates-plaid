"""Benchmark metric, exactly as the leaderboard defines it (docs/methods.md §2; PLAID paper §5).

    RRMSE_f = sqrt( mean_i [ mean_nodes (f_ref,i - f_pred,i)^2 / ||f_ref,i||_inf^2 ] )
    RRMSE_s = sqrt( mean_i [ (s_ref,i - s_pred,i)^2 / s_ref,i^2 ] )
    total_error = mean over all scored field and scalar RRMSEs

Inputs use the submission-file structure: one dict per sample, in split order, mapping each field
name to a 1-D array on that sample's nodes and each scalar name to a float. Everything is float64
numpy in physical units. No epsilons: a zero normalizer is an error, not something to smooth over.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

Record = Mapping[str, Any]


def _as_field(x: Any, name: str, i: int, n: int | None = None) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"sample {i}: field {name} must be 1-D, got shape {a.shape}")
    if n is not None and a.shape[0] != n:
        raise ValueError(f"sample {i}: field {name} has {a.shape[0]} values, mesh has {n} nodes")
    if not np.isfinite(a).all():
        raise ValueError(f"sample {i}: field {name} has non-finite values")
    return a


def _as_scalar(x: Any, name: str, i: int) -> float:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 0:
        raise ValueError(f"sample {i}: scalar {name} must be a single number, got shape {a.shape}")
    v = float(a)
    if not np.isfinite(v):
        raise ValueError(f"sample {i}: scalar {name} is not finite ({v})")
    return v


def _check_pairs(refs: Sequence[Any], preds: Sequence[Any]) -> None:
    if len(refs) != len(preds):
        raise ValueError(f"{len(refs)} reference samples but {len(preds)} predictions")
    if len(refs) == 0:
        raise ValueError("no samples to evaluate")


def _max_abs(ref: np.ndarray) -> float:
    return float(np.max(np.abs(ref)))


def _field_rrmse(
    refs: Sequence[Any], preds: Sequence[Any], normalizer: Callable[[np.ndarray], float], name: str
) -> float:
    _check_pairs(refs, preds)
    per_sample = np.empty(len(refs))
    for i, (r, p) in enumerate(zip(refs, preds, strict=True)):
        ref = _as_field(r, name, i)
        pred = _as_field(p, name, i, n=ref.shape[0])
        scale = normalizer(ref)
        if not scale > 0.0:
            raise ValueError(f"sample {i}: field {name} normalizer is {scale}; metric undefined")
        per_sample[i] = np.mean((ref - pred) ** 2) / scale**2
    return float(np.sqrt(np.mean(per_sample)))


def field_rrmse(refs: Sequence[Any], preds: Sequence[Any], name: str = "field") -> float:
    """RRMSE of one field over samples; each sample normalized by its own max |f_ref|."""
    return _field_rrmse(refs, preds, _max_abs, name)


def scalar_rrmse(refs: Sequence[Any], preds: Sequence[Any], name: str = "scalar") -> float:
    """RRMSE of one scalar over samples, relative to each sample's reference value."""
    _check_pairs(refs, preds)
    ref = np.array([_as_scalar(r, name, i) for i, r in enumerate(refs)])
    pred = np.array([_as_scalar(p, name, i) for i, p in enumerate(preds)])
    zero = np.flatnonzero(ref == 0.0)
    if zero.size:
        raise ValueError(f"scalar {name}: reference is 0 for samples {zero[:5].tolist()}")
    return float(np.sqrt(np.mean(((ref - pred) / ref) ** 2)))


def _column(records: Sequence[Record], key: str, which: str) -> list[Any]:
    missing = [i for i, r in enumerate(records) if key not in r]
    if missing:
        raise KeyError(f"{which}: {key!r} missing for samples {missing[:5]}")
    return [r[key] for r in records]


def evaluate(
    refs: Sequence[Record],
    preds: Sequence[Record],
    fields: Sequence[str],
    scalars: Sequence[str],
) -> dict[str, Any]:
    """Score predictions against references, paired by position (split order).

    Returns {"total_error": float, "rrmse": {output: float}, "n_samples": int}.
    Extra keys in the records are ignored here; the submission writer is where they're rejected.
    """
    _check_pairs(refs, preds)
    if not fields and not scalars:
        raise ValueError("nothing to score")
    rrmse: dict[str, float] = {}
    for f in fields:
        rrmse[f] = field_rrmse(_column(refs, f, "reference"), _column(preds, f, "prediction"), f)
    for s in scalars:
        rrmse[s] = scalar_rrmse(_column(refs, s, "reference"), _column(preds, s, "prediction"), s)
    return {
        "total_error": float(np.mean(list(rrmse.values()))),
        "rrmse": rrmse,
        "n_samples": len(refs),
    }


def reference_records(
    samples: Sequence[Any], fields: Sequence[str], scalars: Sequence[str]
) -> list[dict[str, Any]]:
    """Reference records (same structure as a submission) from cached samples, in given order."""
    no_out = [s.sample_id for s in samples if not s.has_outputs]
    if no_out:
        raise ValueError(f"samples without outputs can't be references: {no_out[:5]}")
    return [
        {**{f: s.output_fields[f] for f in fields}, **{k: s.output_scalars[k] for k in scalars}}
        for s in samples
    ]


# --- Diagnostic only. The first leaderboard calibration settled the definition: max|f|.
# The PLAID paper describes ||f||_inf as "the maximum component in the vector", which could be read
# as max(f) instead of max|f|. For Tensile2d U1 these differ ~3x. Never use this for selection.


def field_rrmse_maxcomponent_diagnostic(
    refs: Sequence[Any], preds: Sequence[Any], name: str = "field"
) -> float:
    """DIAGNOSTIC: like `field_rrmse` but normalized by max(f_ref) (the literal paper wording)."""
    return _field_rrmse(refs, preds, lambda ref: float(np.max(ref)), name)


def evaluate_maxcomponent_diagnostic(
    refs: Sequence[Record], preds: Sequence[Record], fields: Sequence[str], scalars: Sequence[str]
) -> dict[str, Any]:
    """DIAGNOSTIC: `evaluate` with the max(f_ref) field normalizer. Calibration only."""
    out = evaluate(refs, preds, fields=(), scalars=scalars) if scalars else {"rrmse": {}}
    rrmse = {
        f: field_rrmse_maxcomponent_diagnostic(
            _column(refs, f, "reference"), _column(preds, f, "prediction"), f
        )
        for f in fields
    }
    rrmse.update(out["rrmse"])
    return {
        "total_error": float(np.mean(list(rrmse.values()))),
        "rrmse": rrmse,
        "n_samples": len(refs),
    }

"""Standardization statistics, fit on the current training portion only (docs/methods.md §4).

- input scalars: mean/std over training *samples* (one value per sample);
- coordinates:   per-axis mean/std pooled over all training *nodes*;
- output fields: per-field mean/std pooled over all training *nodes*.

Stats are float64 and saved with every run (normalization.json). The network sees standardized
values; predictions are decoded back to physical units before any metric. A channel with zero
spread gets std 1 (it is then only centred), recorded in `constant_channels`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from plaid_surrogates.data.cache import CachedSample

FORMAT_VERSION = 1


def _safe_std(std: np.ndarray) -> np.ndarray:
    return np.where(std > 0.0, std, 1.0)


@dataclass
class Normalizer:
    """Per-channel affine standardization for inputs, coordinates, and output fields."""

    input_names: tuple[str, ...]
    field_names: tuple[str, ...]
    in_mean: np.ndarray  # (C_in,)
    in_std: np.ndarray
    coord_mean: np.ndarray  # (2,)
    coord_std: np.ndarray
    field_mean: np.ndarray  # (C_out,)
    field_std: np.ndarray
    n_samples: int = 0
    n_nodes: int = 0

    @classmethod
    def fit(
        cls,
        samples: Sequence[CachedSample],
        input_names: Sequence[str],
        field_names: Sequence[str],
    ) -> Normalizer:
        """Fit on training samples (float64, two-pass for the variances)."""
        if not samples:
            raise ValueError("no training samples")
        missing = [s.sample_id for s in samples if not s.has_outputs]
        if missing:
            raise ValueError(f"training samples without outputs: {missing[:5]}")
        scal = np.array([[s.input_scalars[n] for n in input_names] for s in samples], np.float64)
        n_nodes = sum(s.n_nodes for s in samples)

        coord_mean = sum(s.points.sum(axis=0) for s in samples) / n_nodes
        coord_var = sum(((s.points - coord_mean) ** 2).sum(axis=0) for s in samples) / n_nodes
        field_sum = np.zeros(len(field_names))
        for s in samples:
            field_sum += [s.output_fields[f].sum() for f in field_names]
        field_mean = field_sum / n_nodes
        field_var = np.zeros(len(field_names))
        for s in samples:
            field_var += [
                ((s.output_fields[f] - m) ** 2).sum()
                for f, m in zip(field_names, field_mean, strict=True)
            ]
        field_var /= n_nodes
        return cls(
            input_names=tuple(input_names),
            field_names=tuple(field_names),
            in_mean=scal.mean(axis=0),
            in_std=scal.std(axis=0),
            coord_mean=np.asarray(coord_mean, np.float64),
            coord_std=np.sqrt(np.asarray(coord_var, np.float64)),
            field_mean=field_mean,
            field_std=np.sqrt(field_var),
            n_samples=len(samples),
            n_nodes=int(n_nodes),
        )

    def constant_channels(self) -> list[str]:
        """Names of channels whose fitted std is 0 (encoded with std 1)."""
        pairs = (
            [(f"input:{n}", s) for n, s in zip(self.input_names, self.in_std, strict=True)]
            + [(f"coord:{a}", s) for a, s in zip("xy", self.coord_std, strict=True)]
            + [(f"field:{n}", s) for n, s in zip(self.field_names, self.field_std, strict=True)]
        )
        return [name for name, s in pairs if s <= 0.0]

    def encode_inputs(self, s: CachedSample) -> np.ndarray:
        """(C_in,) standardized input scalars of one sample."""
        x = np.array([s.input_scalars[n] for n in self.input_names], np.float64)
        return (x - self.in_mean) / _safe_std(self.in_std)

    def encode_coords(self, points: np.ndarray) -> np.ndarray:
        """(N, 2) standardized coordinates, node order unchanged."""
        return (points - self.coord_mean) / _safe_std(self.coord_std)

    def encode_fields(self, s: CachedSample) -> np.ndarray:
        """(N, C_out) standardized target fields, columns in `field_names` order."""
        y = np.stack([s.output_fields[f] for f in self.field_names], axis=1)
        return (y - self.field_mean) / _safe_std(self.field_std)

    def field_scale(self) -> np.ndarray:
        """Per-field std actually used by encode/decode (1 for constant channels)."""
        return _safe_std(self.field_std)

    def decode_fields(self, y: np.ndarray) -> dict[str, np.ndarray]:
        """Network output (N, C_out) -> {field: (N,) float64 in physical units}."""
        y = np.asarray(y, np.float64)
        if y.ndim != 2 or y.shape[1] != len(self.field_names):
            raise ValueError(f"expected (N, {len(self.field_names)}), got {y.shape}")
        phys = y * _safe_std(self.field_std) + self.field_mean
        return {f: np.ascontiguousarray(phys[:, k]) for k, f in enumerate(self.field_names)}

    def to_dict(self) -> dict[str, Any]:
        def per_name(names: Sequence[str], mean: np.ndarray, std: np.ndarray) -> dict[str, Any]:
            return {
                n: {"mean": float(m), "std": float(s)}
                for n, m, s in zip(names, mean, std, strict=True)
            }

        return {
            "format_version": FORMAT_VERSION,
            "fit_on": {"n_samples": self.n_samples, "n_nodes": self.n_nodes},
            "input_scalars": per_name(self.input_names, self.in_mean, self.in_std),
            "coords": per_name(("x", "y"), self.coord_mean, self.coord_std),
            "fields": per_name(self.field_names, self.field_mean, self.field_std),
            "constant_channels": self.constant_channels(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Normalizer:
        def unpack(block: dict[str, Any]) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
            names = tuple(block)
            return (
                names,
                np.array([block[n]["mean"] for n in names], np.float64),
                np.array([block[n]["std"] for n in names], np.float64),
            )

        if d.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported normalization format {d.get('format_version')}")
        in_names, in_mean, in_std = unpack(d["input_scalars"])
        _, coord_mean, coord_std = unpack(d["coords"])
        f_names, f_mean, f_std = unpack(d["fields"])
        return cls(
            input_names=in_names,
            field_names=f_names,
            in_mean=in_mean,
            in_std=in_std,
            coord_mean=coord_mean,
            coord_std=coord_std,
            field_mean=f_mean,
            field_std=f_std,
            n_samples=int(d["fit_on"]["n_samples"]),
            n_nodes=int(d["fit_on"]["n_nodes"]),
        )

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> Normalizer:
        return cls.from_dict(json.loads(Path(path).read_text()))

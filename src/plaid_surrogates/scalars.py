"""Scalar predictors, chosen per scored scalar in the config (docs/methods.md §4).

Options for each scalar:
- "training mean": the mean over the run's training portion (no model);
- any non-fitted candidate in `identities.CANDIDATES[dataset][scalar]`, e.g. "max(U2) over Top"
  (computed from the *predicted* fields plus the sample's tags) or "-P" (inputs only).

Every option is also scored on dev at each evaluation (diagnostic), so the choice is made on dev
(Rule 4) and can change as the fields improve. Saved per run as scalar_predictor.json.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from plaid_surrogates.data.cache import CachedSample
from plaid_surrogates.identities import CANDIDATES, FITTED, Candidate
from plaid_surrogates.metrics import scalar_rrmse

TRAINING_MEAN = "training mean"
FORMAT_VERSION = 2


def options(dataset: str, scalar: str) -> list[str]:
    """Allowed predictor names for one scalar, training mean first."""
    cands = CANDIDATES.get(dataset, {}).get(scalar, {})
    return [TRAINING_MEAN, *(n for n in cands if n not in FITTED)]


def _candidate(dataset: str, scalar: str, name: str) -> Candidate:
    return CANDIDATES[dataset][scalar][name]


class ScalarPredictor:
    """Per-scalar choice of predictor, plus the fitted training means."""

    def __init__(
        self, dataset: str, methods: Mapping[str, str], training_means: Mapping[str, float]
    ) -> None:
        for scalar, method in methods.items():
            if method not in options(dataset, scalar):
                raise ValueError(
                    f"{scalar}: unknown predictor {method!r}; options {options(dataset, scalar)}"
                )
        missing = set(training_means) - set(methods)
        if missing:
            raise ValueError(f"no predictor chosen for {sorted(missing)}")
        self.dataset = dataset
        self.methods = dict(methods)
        self.training_means = {k: float(v) for k, v in training_means.items()}

    @classmethod
    def fit(
        cls,
        dataset: str,
        methods: Mapping[str, str],
        train_samples: Sequence[CachedSample],
        scalars: Sequence[str],
    ) -> ScalarPredictor:
        """Check that every scored scalar has a predictor and fit the training means."""
        missing = [s for s in scalars if s not in methods]
        extra = sorted(set(methods) - set(scalars))
        if missing or extra:
            raise ValueError(f"config scalars: missing {missing}, not scored {extra}")
        means = {k: float(np.mean([s.output_scalars[k] for s in train_samples])) for k in scalars}
        return cls(dataset, {k: methods[k] for k in scalars}, means)

    def _value(self, scalar: str, method: str, fields: Mapping[str, np.ndarray], s: CachedSample):
        if method == TRAINING_MEAN:
            return self.training_means[scalar]
        cand = _candidate(self.dataset, scalar, method)
        return cand(fields, s, 0.0)

    def predict(self, fields: Mapping[str, np.ndarray], s: CachedSample) -> dict[str, float]:
        """Chosen predictor for every scalar, from predicted fields and the sample's tags/inputs."""
        return {k: float(self._value(k, m, fields, s)) for k, m in self.methods.items()}

    def option_rrmse(
        self,
        fields_by_id: Mapping[int, Mapping[str, np.ndarray]],
        samples: Sequence[CachedSample],
    ) -> dict[str, dict[str, float | None]]:
        """Dev RRMSE of every option of every scalar (None if an option can't be evaluated)."""
        out: dict[str, dict[str, float | None]] = {}
        for scalar in self.methods:
            refs = [s.output_scalars[scalar] for s in samples]
            out[scalar] = {}
            for method in options(self.dataset, scalar):
                try:
                    preds = [
                        self._value(scalar, method, fields_by_id[s.sample_id], s) for s in samples
                    ]
                except KeyError:  # e.g. a tag the dataset doesn't have
                    out[scalar][method] = None
                    continue
                out[scalar][method] = scalar_rrmse(refs, preds, scalar)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "dataset": self.dataset,
            "methods": self.methods,
            "training_means": self.training_means,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], dataset: str) -> ScalarPredictor:
        """Read scalar_predictor.json; format 1 (early runs) was training means only."""
        if d.get("format_version") == FORMAT_VERSION:
            return cls(d["dataset"], d["methods"], d["training_means"])
        if d.get("kind") == TRAINING_MEAN:
            means = d["values"]
            return cls(dataset, {k: TRAINING_MEAN for k in means}, means)
        raise ValueError(f"unknown scalar_predictor format: {sorted(d)}")

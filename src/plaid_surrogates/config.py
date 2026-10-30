"""Experiment configs: one YAML per experiment family, loaded into frozen dataclasses.

Loading is strict: an unknown or missing key is an error, so a typo in a YAML file can never
silently fall back to a value hidden in code (docs/methods.md §6: no magic numbers in code).
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from plaid_surrogates.data.features import FEATURE_SETS
from plaid_surrogates.losses import LOSSES

PRECISIONS = ("fp32", "bf16")
SCHEDULES = ("cosine", "constant")


@dataclass(frozen=True)
class ModelConfig:
    """Transolver hyperparameters. Every constructor argument is set explicitly from here."""

    n_layers: int
    n_hidden: int
    n_head: int
    slice_num: int
    mlp_ratio: int
    dropout: float
    act: str
    plus: bool
    use_te: bool

    def __post_init__(self) -> None:
        if self.n_hidden % self.n_head:
            raise ValueError(f"n_hidden {self.n_hidden} must be divisible by n_head {self.n_head}")
        if min(self.n_layers, self.n_hidden, self.n_head, self.slice_num, self.mlp_ratio) < 1:
            raise ValueError("model sizes must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


@dataclass(frozen=True)
class OptimConfig:
    """AdamW + LR schedule. One update every `grad_accum` meshes (batch size is always 1)."""

    lr: float
    weight_decay: float
    epochs: int
    grad_accum: int
    schedule: str
    grad_clip: float  # max global grad norm; 0 disables clipping

    def __post_init__(self) -> None:
        if self.schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}, got {self.schedule!r}")
        if self.lr <= 0 or self.weight_decay < 0 or self.grad_clip < 0:
            raise ValueError("need lr > 0, weight_decay >= 0, grad_clip >= 0")
        if self.epochs < 1 or self.grad_accum < 1:
            raise ValueError("epochs and grad_accum must be >= 1")


@dataclass(frozen=True)
class TrainConfig:
    """One training run, minus the seed (given on the command line)."""

    dataset: str
    tag: str
    precision: str
    device: str | None  # None -> PLAID_DEVICE, else cuda -> mps -> cpu
    eval_every: int  # epochs between exact dev evaluations (the last epoch is always evaluated)
    features: str  # feature set (data/features.py)
    scalars: dict[str, str]  # scored scalar -> predictor name (scalars.py; checked by train.py)
    loss: str  # training loss (losses.py); model selection is always on exact dev total_error
    model: ModelConfig
    optim: OptimConfig

    def __post_init__(self) -> None:
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, got {self.precision!r}")
        if self.features not in FEATURE_SETS:
            raise ValueError(f"features must be one of {FEATURE_SETS}, got {self.features!r}")
        if self.loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}, got {self.loss!r}")
        if self.eval_every < 1:
            raise ValueError("eval_every must be >= 1")
        if not self.tag or "_" in self.tag or "/" in self.tag:
            raise ValueError(f"tag must be non-empty, without '_' or '/', got {self.tag!r}")


def _build(cls: type, data: Any, where: str) -> Any:
    """Construct dataclass `cls` from a dict, recursing into nested dataclasses; strict keys."""
    if not isinstance(data, dict):
        raise TypeError(f"{where}: expected a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]
    unknown = sorted(set(data) - set(names))
    missing = [n for n in names if n not in data]
    if unknown or missing:
        raise KeyError(f"{where}: unknown keys {unknown}, missing keys {missing}")
    kwargs: dict[str, Any] = {}
    for name in names:
        hint, value = hints[name], data[name]
        if dataclasses.is_dataclass(hint):
            kwargs[name] = _build(hint, value, f"{where}.{name}")
        elif typing.get_origin(hint) is dict:
            ok = isinstance(value, dict) and all(
                isinstance(k, str) and isinstance(v, str) for k, v in value.items()
            )
            if not ok:
                raise TypeError(f"{where}.{name}: expected a mapping of str to str, got {value!r}")
            kwargs[name] = dict(value)
        elif hint is float and isinstance(value, int) and not isinstance(value, bool):
            kwargs[name] = float(value)  # YAML writes 1 for 1.0
        else:
            expected = typing.get_args(hint) or (hint,)
            if not isinstance(value, expected) or (hint is int and isinstance(value, bool)):
                raise TypeError(f"{where}.{name}: expected {hint}, got {value!r}")
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: Path) -> TrainConfig:
    """Read and validate a training config YAML."""
    return _build(TrainConfig, yaml.safe_load(Path(path).read_text()), str(path))


def config_to_dict(cfg: TrainConfig) -> dict[str, Any]:
    """Plain-dict form for config.json and checkpoints; `_build` accepts it back."""
    return dataclasses.asdict(cfg)


def config_from_dict(data: dict[str, Any]) -> TrainConfig:
    """Inverse of `config_to_dict` (used when reloading a run).

    Runs saved before these keys existed fill them with what they used: no `scalars` (the
    training mean for every scalar, which an empty mapping means to `ScalarPredictor.from_dict`),
    no `features` (input scalars only), no `loss` (standardized MSE).
    """
    legacy = {"scalars": {}, "features": "input_scalars", "loss": "standardized_mse"}
    return _build(TrainConfig, {**legacy, **data}, "config")

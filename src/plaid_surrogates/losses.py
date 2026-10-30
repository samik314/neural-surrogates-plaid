"""Training losses (docs/methods.md §4). Metrics live in metrics.py and never use these.

1. `standardized_mse` (baseline): mean squared error over nodes and channels, on fields
   standardized with training-portion statistics.
2. `MetricLoss` (metric-consistent): per sample i and field f,
       e_if = mean_nodes((pred - ref)^2) / (max|ref_if|^2 + delta)      (physical units)
   and L = mean_f w_f e_if with w_f = 1 / (sqrt(EMA_f) + eps), detached. The metric is
   mean_f sqrt(mean_i e_if), whose gradient weights field f by 1 / RRMSE_f; the EMA of e_if over
   recent training samples estimates RRMSE_f^2.
   The network works in standardized units, so e_if is computed exactly by rescaling: with
   x = (y - mean_f) / std_f, (pred - ref)^2 in physical units = std_f^2 (x_pred - x_ref)^2, hence
   e_if = mean_nodes((x_pred - x_ref)^2) * c_if with c_if = std_f^2 / (max|ref_if|^2 + delta),
   fixed per training sample (`metric_scale`).
"""

from __future__ import annotations

import numpy as np
import torch

LOSSES = ("standardized_mse", "metric")

# Stabilizers only (metrics.py never uses them). delta = 0: the data audit found no near-zero
# per-sample maxima (min / median >= 0.57 for every 2D_MultiScHypEl field; Tensile2d likewise).
METRIC_DELTA = 0.0
METRIC_EPS = 1e-8
METRIC_EMA_DECAY = 0.99  # ~100 training samples


def standardized_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean of (pred - target)^2 over every node and channel; shapes must match exactly."""
    if pred.shape != target.shape:
        raise ValueError(
            f"shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}"
        )
    return torch.mean((pred - target) ** 2)


def metric_scale(
    field_std: np.ndarray, ref_phys: np.ndarray, delta: float = METRIC_DELTA
) -> np.ndarray:
    """c_f = std_f^2 / (max|ref_f|^2 + delta) for one sample; ref_phys is (N, C) physical."""
    maxabs = np.max(np.abs(np.asarray(ref_phys, np.float64)), axis=0)
    denom = maxabs**2 + delta
    if not np.all(denom > 0.0):
        bad = np.flatnonzero(~(denom > 0.0)).tolist()
        raise ValueError(f"metric loss undefined: zero max|ref| in channels {bad}")
    return np.asarray(field_std, np.float64) ** 2 / denom


class MetricLoss:
    """Metric-consistent loss with detached per-field weights from an EMA of e_if."""

    def __init__(
        self, n_fields: int, decay: float = METRIC_EMA_DECAY, eps: float = METRIC_EPS
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self.decay, self.eps = decay, eps
        self.ema = torch.full((n_fields,), float("nan"), dtype=torch.float64)

    def per_field(
        self, pred: torch.Tensor, target: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """e_f for one sample: pred, target (1, N, C) standardized; scale (C,) from metric_scale."""
        if pred.shape != target.shape:
            raise ValueError(
                f"shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}"
            )
        if scale.shape != (pred.shape[-1],):
            raise ValueError(f"scale shape {tuple(scale.shape)} vs {pred.shape[-1]} channels")
        return ((pred - target) ** 2).mean(dim=tuple(range(pred.dim() - 1))) * scale

    def __call__(
        self, pred: torch.Tensor, target: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """L = mean_f w_f e_f; updates the EMA with this sample's e_f, then weights with it."""
        e = self.per_field(pred, target, scale)
        e_det = e.detach().double().cpu()
        if torch.isnan(self.ema).any():
            self.ema = e_det.clone()
        else:
            self.ema = self.decay * self.ema + (1.0 - self.decay) * e_det
        w = (1.0 / (torch.sqrt(self.ema) + self.eps)).to(e.device, e.dtype)
        return torch.mean(w * e)

    def ema_rrmse(self) -> np.ndarray:
        """sqrt(EMA_f): running estimate of each field's training RRMSE."""
        return torch.sqrt(self.ema).numpy()

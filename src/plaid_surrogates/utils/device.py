"""Device selection: cuda -> mps -> cpu, overridable by argument or PLAID_DEVICE."""

from __future__ import annotations

import os

import torch

_VALID = ("cuda", "mps", "cpu")


def _available(name: str) -> bool:
    if name == "cuda":
        return torch.cuda.is_available()
    if name == "mps":
        return torch.backends.mps.is_available()
    return True


def get_device(prefer: str | None = None) -> torch.device:
    """Return `prefer` (or $PLAID_DEVICE) if given, else the first available of cuda, mps, cpu.

    An explicitly requested device that is unavailable raises instead of silently falling back,
    so a job that asked for a GPU never quietly trains on CPU.
    """
    prefer = prefer or os.environ.get("PLAID_DEVICE") or None
    if prefer is not None:
        name = prefer.split(":")[0]
        if name not in _VALID:
            raise ValueError(f"Unknown device {prefer!r}; expected one of {_VALID}")
        if not _available(name):
            raise RuntimeError(f"Requested device {prefer!r} is not available on this machine")
        return torch.device(prefer)
    for name in _VALID:
        if _available(name):
            return torch.device(name)
    raise AssertionError("unreachable: cpu is always available")


def supports_float64(device: torch.device) -> bool:
    """MPS has no float64; everything else does. Metrics are always float64 numpy regardless."""
    return device.type != "mps"


def device_description(device: torch.device) -> str:
    """Human-readable device name for logs (GPU model on CUDA)."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return device.type

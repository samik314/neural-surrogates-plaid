import pytest
import torch

from plaid_surrogates.utils.device import get_device, supports_float64


def test_default_is_a_known_device(monkeypatch):
    monkeypatch.delenv("PLAID_DEVICE", raising=False)
    assert get_device().type in {"cuda", "mps", "cpu"}


def test_default_prefers_cuda_when_available(monkeypatch):
    monkeypatch.delenv("PLAID_DEVICE", raising=False)
    if torch.cuda.is_available():
        assert get_device().type == "cuda"


def test_explicit_cpu():
    assert get_device("cpu") == torch.device("cpu")


def test_env_var_override(monkeypatch):
    monkeypatch.setenv("PLAID_DEVICE", "cpu")
    assert get_device().type == "cpu"


def test_unknown_device_raises():
    with pytest.raises(ValueError):
        get_device("tpu")


def test_unavailable_device_raises_instead_of_falling_back():
    missing = next((d for d in ("cuda", "mps") if not get_device_available(d)), None)
    if missing is None:
        pytest.skip("both cuda and mps available")
    with pytest.raises(RuntimeError):
        get_device(missing)


def get_device_available(name: str) -> bool:
    return torch.cuda.is_available() if name == "cuda" else torch.backends.mps.is_available()


def test_float64_support():
    assert supports_float64(torch.device("cpu"))
    assert not supports_float64(torch.device("mps"))

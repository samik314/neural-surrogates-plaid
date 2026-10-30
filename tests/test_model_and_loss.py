import numpy as np
import pytest
import torch

from plaid_surrogates.config import ModelConfig
from plaid_surrogates.losses import MetricLoss, metric_scale, standardized_mse
from plaid_surrogates.metrics import field_rrmse
from plaid_surrogates.models.build import build_transolver, count_params

SMALL = ModelConfig(
    n_layers=1,
    n_hidden=16,
    n_head=2,
    slice_num=4,
    mlp_ratio=2,
    dropout=0.0,
    act="gelu",
    plus=False,
    use_te=False,
)


def test_transolver_forward_shapes_and_params():
    torch.manual_seed(0)
    model = build_transolver(SMALL, functional_dim=6, out_dim=5)
    assert model.use_te is False and count_params(model) > 0
    y = model(torch.randn(1, 37, 6), embedding=torch.randn(1, 37, 2))
    assert y.shape == (1, 37, 5) and torch.isfinite(y).all()


def test_transolver_handles_varying_node_counts():
    model = build_transolver(SMALL, functional_dim=3, out_dim=2).eval()
    for n in (5, 120):
        assert model(torch.randn(1, n, 3), embedding=torch.randn(1, n, 2)).shape == (1, n, 2)


def test_standardized_mse_value_and_shape_check():
    pred = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    target = torch.zeros(1, 2, 2)
    assert standardized_mse(pred, target).item() == pytest.approx((1 + 4 + 9 + 16) / 4)
    assert standardized_mse(target, target).item() == 0.0
    with pytest.raises(ValueError, match="shape mismatch"):
        standardized_mse(pred, torch.zeros(1, 2, 3))


def _phys_case(rng, n=50, c=3):
    """Physical ref/pred plus the standardized versions, with field-scale-like magnitudes."""
    mean, std = np.array([0.5, -2e6, 1e7]), np.array([0.3, 1e6, 4e6])
    ref = mean + std * rng.standard_normal((n, c))
    pred = ref + 0.05 * std * rng.standard_normal((n, c))
    xs = [torch.from_numpy((a - mean) / std).unsqueeze(0) for a in (pred, ref)]
    return mean, std, ref, pred, xs


def test_metric_scale_equals_physical_units_and_metric():
    rng = np.random.default_rng(0)
    loss = MetricLoss(3)
    e_all, refs, preds = [], [], []
    for _ in range(4):
        _, std, ref, pred, (xp, xr) = _phys_case(rng)
        e = loss.per_field(xp, xr, torch.from_numpy(metric_scale(std, ref))).numpy()
        direct = np.mean((pred - ref) ** 2, axis=0) / np.max(np.abs(ref), axis=0) ** 2
        np.testing.assert_allclose(e, direct, rtol=1e-12)
        e_all.append(e), refs.append(ref), preds.append(pred)
    # sqrt(mean_i e_if) is exactly the benchmark RRMSE of field f
    for f in range(3):
        rr = field_rrmse([r[:, f] for r in refs], [p[:, f] for p in preds])
        assert np.sqrt(np.mean([e[f] for e in e_all])) == pytest.approx(rr, rel=1e-12)


def test_metric_scale_rejects_zero_max():
    with pytest.raises(ValueError, match="channels \\[1\\]"):
        metric_scale(np.ones(2), np.array([[1.0, 0.0], [2.0, 0.0]]))


def test_metric_loss_gradient_matches_metric_and_weights_are_detached():
    rng = np.random.default_rng(1)
    _, std, ref, _, (xp, xr) = _phys_case(rng)
    scale = torch.from_numpy(metric_scale(std, ref))
    a = xp.clone().requires_grad_(True)
    MetricLoss(3, eps=0.0)(a, xr, scale).backward()
    # first sample: EMA = e, w = 1/sqrt(e): dL = mean_f de_f / sqrt(e_f) = 2 d(mean_f sqrt e_f)
    b = xp.clone().requires_grad_(True)
    torch.mean(torch.sqrt(MetricLoss(3).per_field(b, xr, scale))).backward()
    torch.testing.assert_close(a.grad, 2.0 * b.grad, rtol=1e-10, atol=0.0)


def test_metric_loss_ema_update():
    rng = np.random.default_rng(2)
    loss = MetricLoss(3, decay=0.9)
    es = []
    for _ in range(2):
        _, std, ref, _, (xp, xr) = _phys_case(rng)
        scale = torch.from_numpy(metric_scale(std, ref))
        es.append(loss.per_field(xp, xr, scale))
        loss(xp, xr, scale)
    np.testing.assert_allclose(loss.ema.numpy(), (0.9 * es[0] + 0.1 * es[1]).numpy(), rtol=1e-12)
    np.testing.assert_allclose(loss.ema_rrmse(), np.sqrt(loss.ema.numpy()))
    assert not loss.ema.requires_grad
    with pytest.raises(ValueError, match="decay"):
        MetricLoss(3, decay=1.0)
    with pytest.raises(ValueError, match="scale shape"):
        loss.per_field(xp, xr, scale[:2])

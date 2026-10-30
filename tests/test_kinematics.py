import numpy as np
import pytest

from plaid_surrogates.data import kinematics as kin
from plaid_surrogates.data.cache import CachedSample


def test_fit_affine_recovers_gradient_offset_and_fluctuation():
    rng = np.random.default_rng(0)
    x = rng.random((200, 2))
    a = np.array([[0.1, 0.05], [-0.02, -0.08]])
    c = np.array([0.7, -0.3])
    fluct = 1e-3 * np.column_stack([np.sin(2 * np.pi * x[:, 0]), np.cos(2 * np.pi * x[:, 1])])
    u = x @ a.T + c
    a_fit, c_fit, res = kin.fit_affine(x, u)
    np.testing.assert_allclose(a_fit, a, atol=1e-12)
    np.testing.assert_allclose(c_fit, c, atol=1e-12)
    assert np.abs(res).max() < 1e-12
    _, _, res2 = kin.fit_affine(x, u + fluct)
    assert 1e-4 < np.abs(res2).max() < 3e-3


def test_cauchy_green_of_pure_stretch():
    a = np.diag([0.1, -0.1])  # F = diag(1.1, 0.9)
    assert kin.cauchy_green(a) == pytest.approx([1.21, 0.0, 0.81])


def test_sample_kinematics_reports_offset_and_c_consistency():
    xs = np.linspace(0, 1, 5)
    pts = np.array([[x, y] for y in xs for x in xs])
    a = np.diag([0.1, 0.0])
    u = pts @ a.T  # pinned at the origin
    s = CachedSample(
        0,
        pts,
        np.zeros((0, 3), np.int64),
        {"C11": 1.21, "C12": 0.0, "C22": 1.0},
        output_fields={"u1": u[:, 0], "u2": u[:, 1]},
    )
    r = kin.sample_kinematics(s)
    assert r["offset_norm"] == pytest.approx(0.0, abs=1e-12)
    assert r["C_fit_err"] == pytest.approx(0.0, abs=1e-12)
    assert r["fluct_max"] == pytest.approx(0.0, abs=1e-12)
    assert r["n_nodes_u_zero"] == 5  # the whole x = 0 edge has u = 0
    assert r["min_norm_u_at"][0] == 0.0


def test_boundary_fit_and_unit_scale_detection():
    xs = np.linspace(0, 1, 6)
    pts = np.array([[x, y] for y in xs for x in xs])
    on_edge = np.flatnonzero((pts == 0).any(axis=1) | (pts == 1).any(axis=1)).astype(np.int64)
    f = np.array([[1.1, 0.05], [0.0, 0.95]])  # true F
    c = f.T @ f
    u_true = pts @ (f - np.eye(2)).T
    interior = ~np.isin(np.arange(len(pts)), on_edge)
    u_true[interior] += 0.01  # a fluctuation inside only (KUBC: none on the boundary)
    u = 10.0 * u_true  # stored in 10x units
    s = CachedSample(
        0,
        pts,
        np.zeros((0, 3), np.int64),
        {"C11": c[0, 0], "C12": c[0, 1], "C22": c[1, 1]},
        output_fields={"u1": u[:, 0], "u2": u[:, 1]},
        nodal_tags={"Ext_bound": on_edge},
    )
    r = kin.sample_kinematics(s)
    assert r["bnd_fluct_max"] == pytest.approx(0.0, abs=1e-12)
    assert r["bnd_C_fit_err"] > 1.0  # unit mismatch shows up
    assert r["bnd_FtF_scale"] == pytest.approx(10.0, rel=0.03)
    assert r["bnd_FtF_err"] < 1e-2

import numpy as np
import pytest

from plaid_surrogates.data.cache import CachedSample
from plaid_surrogates.data.features import (
    boundary_distances,
    build_inputs,
    feature_names,
    point_segment_distance,
    second_hole_distance,
)
from plaid_surrogates.data.normalize import Normalizer
from tests.test_inspect import grid


def _rve(removed=frozenset({(1, 1)}), n=3):
    """Unit-square RVE with one square hole; tags like 2D_MultiScHypEl."""
    pts, cells = grid(n, removed)
    outer = np.flatnonzero((pts == 0).any(axis=1) | (pts == 1).any(axis=1)).astype(np.int64)
    hole = np.flatnonzero(
        ~np.isin(np.arange(len(pts)), outer)
    ).astype(np.int64)  # in a 3x3 grid every non-outer node is on the hole
    s = CachedSample(
        0,
        pts,
        cells,
        {"C11": 1.1, "C12": 0.0, "C22": 0.9},
        output_fields={f: np.ones(len(pts)) for f in ("u1", "u2")},
        nodal_tags={"Ext_bound": outer, "Holes": hole},
    )
    return s, outer, hole


def test_point_segment_distance():
    a, b = np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]])
    p = np.array([[0.5, 2.0], [-3.0, 4.0], [2.0, 0.0], [0.3, 0.0]])
    np.testing.assert_allclose(point_segment_distance(p, a, b), [2.0, 5.0, 1.0, 0.0])


def test_distances_to_outer_and_hole():
    s, outer, hole = _rve()
    d = boundary_distances(s, "2D_MultiScHypEl")
    assert d.shape == (s.n_nodes, 2)
    np.testing.assert_allclose(d[outer, 0], 0.0, atol=1e-15)
    np.testing.assert_allclose(d[hole, 1], 0.0, atol=1e-15)
    # hole nodes sit 1/3 from the outer edge; outer corners sit sqrt(2)/3 from the hole corner
    np.testing.assert_allclose(d[hole, 0], 1 / 3)
    corner = int(np.flatnonzero((s.points == [0.0, 0.0]).all(axis=1))[0])
    assert d[corner, 1] == pytest.approx(np.sqrt(2) / 3)


@pytest.mark.parametrize(
    ("feature_set", "n_cols"),
    [("input_scalars", 3), ("input_scalars+tags", 6), ("input_scalars+tags+dist", 8)],
)
def test_feature_sets_shape_and_names(feature_set, n_cols):
    s, outer, hole = _rve()
    norm = Normalizer.fit([s, s], ["C11", "C12", "C22"], ["u1", "u2"])
    fx, coords = build_inputs(s, norm, feature_set, "2D_MultiScHypEl")
    names = feature_names(norm, feature_set, "2D_MultiScHypEl")
    assert fx.shape == (s.n_nodes, n_cols) == (s.n_nodes, len(names)) and coords.shape[1] == 2
    if "tags" in feature_set:
        k = names.index
        assert np.all(fx[outer, k("tag:outer")] == 1) and np.all(fx[hole, k("tag:hole")] == 1)
        assert np.all(fx[:, [k("tag:outer"), k("tag:hole"), k("tag:interior")]].sum(1) == 1)


def test_unknown_feature_set_or_dataset():
    s, *_ = _rve()
    norm = Normalizer.fit([s], ["C11", "C12", "C22"], ["u1", "u2"])
    with pytest.raises(ValueError, match="unknown feature set"):
        build_inputs(s, norm, "everything", "2D_MultiScHypEl")
    with pytest.raises(ValueError, match="no tag categories"):
        build_inputs(s, norm, "input_scalars+tags", "SomethingElse")


def test_distances_are_cached_per_sample():
    s, *_ = _rve()
    a = boundary_distances(s, "2D_MultiScHypEl")
    assert boundary_distances(s, "2D_MultiScHypEl") is a  # second call reuses the array


def _two_holes():
    """Unit square, 5 x 5 cells, holes at [0.2, 0.4]^2 and [0.6, 0.8]^2."""
    removed = {(1, 1), (3, 3)}
    pts, cells = grid(5, removed)
    outer = np.flatnonzero((pts == 0).any(axis=1) | (pts == 1).any(axis=1)).astype(np.int64)
    corners = {(i + di, j + dj) for i, j in removed for di in (0, 1) for dj in (0, 1)}
    hole = np.array(
        [k for k, (x, y) in enumerate(pts) if (round(x * 5), round(y * 5)) in corners],
        dtype=np.int64,
    )
    s = CachedSample(
        1,
        pts,
        cells,
        {"C11": 1.1, "C12": 0.0, "C22": 0.9},
        output_fields={f: np.ones(len(pts)) for f in ("u1", "u2")},
        nodal_tags={"Ext_bound": outer, "Holes": hole},
    )
    return s


def test_second_nearest_hole_distance():
    s = _two_holes()
    d2 = second_hole_distance(s, "2D_MultiScHypEl")
    origin = int(np.flatnonzero((s.points == [0.0, 0.0]).all(axis=1))[0])
    # nearest hole corner (0.2, 0.2); the SECOND hole's nearest point is its corner (0.6, 0.6)
    assert d2[origin] == pytest.approx(np.hypot(0.6, 0.6))
    # a node on hole 1 is 0 from hole 1, and its second-nearest is hole 2
    on_h1 = int(np.flatnonzero((s.points == [0.4, 0.4]).all(axis=1))[0])
    assert d2[on_h1] == pytest.approx(np.hypot(0.2, 0.2))
    norm = Normalizer.fit([s], ["C11", "C12", "C22"], ["u1", "u2"])
    fs = "input_scalars+tags+dist+lig"
    fx, _ = build_inputs(s, norm, fs, "2D_MultiScHypEl")
    names = feature_names(norm, fs, "2D_MultiScHypEl")
    assert names[-1] == "dist:hole2" and fx.shape == (s.n_nodes, 9)
    np.testing.assert_allclose(fx[:, -1], d2, rtol=1e-6)


def test_ligament_feature_needs_holes():
    s, *_ = _rve()  # one hole only
    with pytest.raises(ValueError, match="need at least 2"):
        second_hole_distance(s, "2D_MultiScHypEl")
    norm = Normalizer.fit([s], ["C11", "C12", "C22"], ["u1", "u2"])
    with pytest.raises(ValueError, match="needs a 'hole' category"):
        build_inputs(s, norm, "input_scalars+tags+dist+lig", "Tensile2d")

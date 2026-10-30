import numpy as np
import pytest

from plaid_surrogates.data.cache import CachedSample
from plaid_surrogates.data.features import build_inputs, feature_names
from plaid_surrogates.data.normalize import Normalizer


def _s(i, points, u, a):
    n = len(points)
    return CachedSample(
        sample_id=i,
        points=np.asarray(points, np.float64),
        cells=np.zeros((0, 3), np.int64),
        input_scalars={"a": a, "c": 5.0},
        output_fields={"u": np.asarray(u, np.float64), "v": np.full(n, 2.0)},
        output_scalars={"s": 1.0},
    )


@pytest.fixture
def two():
    a = _s(0, [[0, 0], [2, 0]], [1.0, 1.0], a=1.0)
    b = _s(1, [[0, 2], [2, 2], [4, 2], [6, 2]], [0.0, 0.0, 0.0, 0.0], a=3.0)
    return a, b


def test_fit_pools_nodes_for_fields_and_coords_but_samples_for_scalars(two):
    norm = Normalizer.fit(two, ["a", "c"], ["u", "v"])
    # u: 2 of 6 nodes are 1 -> mean 1/3 (not the per-sample average 1/2)
    assert norm.field_mean[0] == pytest.approx(1 / 3)
    assert norm.field_std[0] == pytest.approx(np.std([1, 1, 0, 0, 0, 0]))
    assert norm.coord_mean == pytest.approx([7 / 3, 4 / 3])  # x: 0,2,0,2,4,6; y: 0,0,2,2,2,2
    # input scalar a: one value per sample -> mean 2, std 1 (not node-weighted)
    assert norm.in_mean[0] == pytest.approx(2.0) and norm.in_std[0] == pytest.approx(1.0)
    assert norm.constant_channels() == ["input:c", "field:v"]


def test_encode_decode_round_trip(two):
    norm = Normalizer.fit(two, ["a", "c"], ["u", "v"])
    for s in two:
        y = norm.encode_fields(s)
        assert y.shape == (s.n_nodes, 2)
        back = norm.decode_fields(y)
        for f in ("u", "v"):
            np.testing.assert_allclose(back[f], s.output_fields[f], rtol=0, atol=1e-12)


def test_constant_channel_is_centred_not_divided_by_zero(two):
    norm = Normalizer.fit(two, ["a", "c"], ["u", "v"])
    assert np.all(norm.encode_fields(two[0])[:, 1] == 0.0)
    assert np.isfinite(norm.encode_inputs(two[0])).all()


def test_json_round_trip(tmp_path, two):
    norm = Normalizer.fit(two, ["a", "c"], ["u", "v"])
    norm.save(tmp_path / "n.json")
    back = Normalizer.load(tmp_path / "n.json")
    assert back.input_names == ("a", "c") and back.field_names == ("u", "v")
    for k in ("in_mean", "in_std", "coord_mean", "coord_std", "field_mean", "field_std"):
        np.testing.assert_array_equal(getattr(back, k), getattr(norm, k))


def test_fit_rejects_samples_without_outputs(two):
    bare = CachedSample(9, np.zeros((1, 2)), np.zeros((0, 3), np.int64), {"a": 1.0, "c": 5.0})
    with pytest.raises(ValueError, match="without outputs"):
        Normalizer.fit([two[0], bare], ["a", "c"], ["u", "v"])


def test_features_broadcast_scalars_and_keep_node_order(two):
    norm = Normalizer.fit(two, ["a", "c"], ["u", "v"])
    s = two[1]
    fx, coords = build_inputs(s, norm)
    assert feature_names(norm) == ["scalar:a", "scalar:c"]
    assert fx.shape == (4, 2) and coords.shape == (4, 2)
    assert fx.dtype == np.float32 and coords.dtype == np.float32
    assert np.all(fx == fx[0]) and fx[0, 0] == pytest.approx(1.0)  # (3 - 2) / 1
    # node k of the output is node k of the input
    np.testing.assert_allclose(coords, norm.encode_coords(s.points), rtol=1e-6)
    assert np.all(np.diff(coords[:, 0]) > 0)

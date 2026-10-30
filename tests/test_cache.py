import json

import numpy as np
import pytest

from plaid_surrogates.data.cache import (
    CachedSample,
    load_index,
    load_sample,
    load_split,
    save_sample,
    split_ids,
    validate_sample,
    write_index,
)


def make_sample(sample_id=0, n=7, outputs=True, seed=0):
    rng = np.random.default_rng(seed)
    return CachedSample(
        sample_id=sample_id,
        points=rng.random((n, 2)),
        cells=np.array([[0, 1, 2], [2, 3, 4], [4, 5, 6]], dtype=np.int64),
        input_scalars={"P": -43.8, "p1": 11.0},
        output_fields={"U1": rng.random(n), "sig12": rng.standard_normal(n)} if outputs else {},
        output_scalars={"max_q": 1.9e-4} if outputs else {},
        nodal_tags={"Top": np.array([4, 0, 3], dtype=np.int64), "BottomLeft": np.array([6])},
    )


def test_round_trip_is_exact(tmp_path):
    s = make_sample(sample_id=42)
    load = load_sample(save_sample(tmp_path, s))
    assert load.sample_id == 42
    np.testing.assert_array_equal(load.points, s.points)  # exact, node order preserved
    np.testing.assert_array_equal(load.cells, s.cells)
    assert load.points.dtype == np.float64 and load.cells.dtype == np.int64
    assert load.input_scalars == s.input_scalars
    assert load.output_scalars == s.output_scalars
    assert load.output_fields.keys() == s.output_fields.keys()
    for k in s.output_fields:
        np.testing.assert_array_equal(load.output_fields[k], s.output_fields[k])
    for k in s.nodal_tags:
        np.testing.assert_array_equal(load.nodal_tags[k], s.nodal_tags[k])  # tag order kept too


def test_sample_without_outputs_has_no_output_keys(tmp_path):
    path = save_sample(tmp_path, make_sample(sample_id=500, outputs=False))
    with np.load(path) as z:
        assert not [k for k in z.files if k.startswith("out_")]
    loaded = load_sample(path)
    assert not loaded.has_outputs and loaded.output_fields == {} and loaded.output_scalars == {}


def test_split_order_is_preserved_not_sorted(tmp_path):
    # IDs whose sorted order differs from their stored order (the real Tensile2d splits are
    # already sorted, so they can't catch a sorting bug).
    order = [30, 4, 17, 2, 25]
    info = {}
    for i in order:
        save_sample(tmp_path, make_sample(sample_id=i, seed=i))
        info[i] = {"file": f"samples/{i:06d}.npz", "n_nodes": 7, "n_cells": 3, "has_outputs": True}
    write_index(tmp_path, {"dataset": "fake"}, {"weird": order, "rev": order[::-1]}, info)

    index = load_index(tmp_path)
    assert split_ids(index, "weird") == order
    assert split_ids(index, "rev") == order[::-1]
    assert [s.sample_id for s in load_split(tmp_path, "weird")] == order
    # and the raw JSON keeps it too
    assert json.loads((tmp_path / "index.json").read_text())["splits"]["weird"] == order


def test_index_rejects_duplicates_and_uncached_ids(tmp_path):
    info = {1: {"file": "samples/000001.npz"}}
    with pytest.raises(ValueError, match="duplicate"):
        write_index(tmp_path, {}, {"a": [1, 1]}, info)
    with pytest.raises(ValueError, match="uncached"):
        write_index(tmp_path, {}, {"a": [1, 2]}, info)


def test_unknown_split_raises(tmp_path):
    save_sample(tmp_path, make_sample(sample_id=1))
    write_index(tmp_path, {}, {"a": [1]}, {1: {"file": "samples/000001.npz"}})
    with pytest.raises(KeyError, match="available"):
        split_ids(load_index(tmp_path), "b")


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (lambda s: setattr(s, "points", s.points.astype(np.float32)), "points"),
        (lambda s: setattr(s, "points", s.points[:, :1]), "points"),
        (lambda s: s.cells.__setitem__((0, 0), 7), "cell indices"),
        (lambda s: s.cells.__setitem__((0, 0), -1), "cell indices"),
        (lambda s: s.output_fields.__setitem__("U1", np.zeros(6)), "field U1"),
        (lambda s: s.output_fields["U1"].__setitem__(0, np.nan), "non-finite"),
        (lambda s: s.input_scalars.__setitem__("P", float("inf")), "not finite"),
        (lambda s: s.nodal_tags.__setitem__("Top", np.array([9])), "tag Top"),
    ],
)
def test_validation_rejects_bad_samples(mutate, msg):
    s = make_sample()
    mutate(s)
    with pytest.raises(ValueError, match=msg):
        validate_sample(s)


def test_save_validates(tmp_path):
    s = make_sample()
    s.cells[0, 0] = 99
    with pytest.raises(ValueError):
        save_sample(tmp_path, s)
    assert not list(tmp_path.rglob("*.npz"))

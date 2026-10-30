import numpy as np

from plaid_surrogates.data.cache import CachedSample
from plaid_surrogates.data.compare_zenodo import compare_sample, find_plaid_root, summarize


def _s(points, fields=None, tags=None, scalars=None):
    return CachedSample(
        sample_id=7,
        points=np.asarray(points, np.float64),
        cells=np.array([[0, 1, 2]], dtype=np.int64),
        input_scalars={"C11": 1.0},
        output_fields=fields or {},
        output_scalars=scalars or {},
        nodal_tags=tags if tags is not None else {"Holes": np.array([0], dtype=np.int64)},
    )


PTS = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]


def test_identical_samples_agree():
    a = _s(PTS, {"u1": np.array([1.0, 2.0, 4.0])}, scalars={"e": 2.0})
    r = compare_sample(a, _s(PTS, {"u1": np.array([1.0, 2.0, 4.0])}, scalars={"e": 2.0}))
    assert r["same_n_nodes"] and r["same_node_order"] and r["same_cells"] and r["same_tags"]
    assert r["field_rel_diff"] == {"u1": 0.0} and r["scalar_rel_diff"] == {"e": 0.0}


def test_permuted_nodes_detected_as_order_mismatch_not_count():
    r = compare_sample(_s(PTS), _s([PTS[1], PTS[0], PTS[2]]))
    assert r["same_n_nodes"] and not r["same_node_order"]
    assert r["max_coord_diff_sorted"] == 0.0  # same node set, different order


def test_count_mismatch_and_field_diff():
    r = compare_sample(_s(PTS), _s(PTS + [[1.0, 1.0]]))
    assert not r["same_n_nodes"] and "max_coord_diff" not in r
    r = compare_sample(
        _s(PTS, {"u1": np.array([1.0, 2.0, 4.0])}), _s(PTS, {"u1": np.array([1.0, 2.0, 5.0])})
    )
    assert r["field_rel_diff"]["u1"] == 0.25


def test_summary_verdict():
    splits = {"DOE_test": {"same_ids_same_order": True}}
    ok = compare_sample(_s(PTS), _s(PTS))
    bad = compare_sample(_s(PTS), _s(PTS + [[1.0, 1.0]]))
    assert summarize(splits, {"test": [ok]})["verdict"] == "copies agree"
    s = summarize(splits, {"test": [ok, bad]})
    assert s["verdict"] == "copies DIFFER" and s["test"]["first_count_mismatches"] == [[7, 3, 4]]


def test_find_plaid_root(tmp_path):
    root = tmp_path / "x" / "2D_Multiscale_Hyperelasticity"
    (root / "dataset").mkdir(parents=True)
    (root / "problem_definition").mkdir()
    assert find_plaid_root(tmp_path) == root

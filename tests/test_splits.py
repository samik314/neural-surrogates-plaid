import json

import numpy as np
import pytest

from plaid_surrogates.data import splits as sp
from plaid_surrogates.data.cache import save_sample, write_index
from tests.test_cache import make_sample

# Unsorted source order, 5 strata of 20 each.
IDS = [int(i) for i in np.random.default_rng(1).permutation(np.arange(100, 200))]
STRATA = [k % 5 for k in range(100)]


def test_quantile_bins_balanced():
    labels, edges = sp.quantile_bins(np.arange(100.0), 5)
    assert np.bincount(labels).tolist() == [20] * 5 and len(edges) == 4


def test_holdout_size_strata_and_order():
    dev = sp.stratified_holdout(IDS, STRATA, 0.1, seed=0)
    assert len(dev) == 10
    pos = {i: k for k, i in enumerate(IDS)}
    assert [pos[i] for i in dev] == sorted(pos[i] for i in dev)  # source order kept
    per_stratum = np.bincount([STRATA[pos[i]] for i in dev], minlength=5)
    assert per_stratum.tolist() == [2] * 5


def test_holdout_exact_total_with_uneven_strata():
    strata = [0] * 7 + [1] * 13 + [2] * 3  # 23 samples, frac 0.3 -> 6.9 -> 7
    dev = sp.stratified_holdout(list(range(23)), strata, 0.3, seed=3)
    assert len(dev) == 7


def test_holdout_seeded():
    a = sp.stratified_holdout(IDS, STRATA, 0.1, seed=0)
    assert a == sp.stratified_holdout(IDS, STRATA, 0.1, seed=0)
    assert a != sp.stratified_holdout(IDS, STRATA, 0.1, seed=1)


def test_kfold_partitions_and_balances():
    folds = sp.stratified_kfold(IDS, STRATA, 5, seed=0)
    flat = [i for f in folds for i in f]
    assert sorted(flat) == sorted(IDS) and len(flat) == len(set(flat))
    pos = {i: k for k, i in enumerate(IDS)}
    for f in folds:
        assert len(f) == 20
        assert np.bincount([STRATA[pos[i]] for i in f], minlength=5).tolist() == [4] * 5
        assert [pos[i] for i in f] == sorted(pos[i] for i in f)
    assert folds == sp.stratified_kfold(IDS, STRATA, 5, seed=0)
    assert folds != sp.stratified_kfold(IDS, STRATA, 5, seed=1)


def test_bad_args():
    with pytest.raises(ValueError):
        sp.stratified_holdout(IDS, STRATA, 1.5, seed=0)
    with pytest.raises(ValueError):
        sp.stratified_kfold(IDS, STRATA, 1, seed=0)


def test_ks_statistic():
    a = np.arange(10.0)
    assert sp.ks_statistic(a, a) == 0.0
    assert sp.ks_statistic(a, a + 100) == 1.0


@pytest.fixture
def fake_cache(tmp_path, monkeypatch):
    """A cache laid out like Tensile2d (train_500 name, unsorted IDs, varying node counts)."""
    cache = tmp_path / "cache" / "Tensile2d"
    info = {}
    for k, i in enumerate(IDS):
        s = make_sample(sample_id=i, n=7 + k % 10, seed=i)
        s.cells = np.array([[0, 1, 2]], dtype=np.int64)
        s.input_scalars = {n: float(k) for n in ("P", "p1", "p2", "p3", "p4", "p5")}
        save_sample(cache, s)
        info[i] = {"file": f"samples/{i:06d}.npz", "n_nodes": s.n_nodes}
    write_index(cache, {"hf_revision": "abc"}, {"train_500": IDS}, info)
    monkeypatch.setenv("PLAID_DATA_DIR", str(tmp_path))
    return cache


def test_make_and_load_splits(fake_cache):
    s = sp.make_splits(fake_cache, "Tensile2d", 0.1, 5, seed=0, n_bins=5)
    assert len(s["dev"]) == 10 and len(s["train"]) == 90
    assert set(s["dev"]).isdisjoint(s["train"]) and set(s["dev"]) | set(s["train"]) == set(IDS)
    assert sorted(i for f in s["folds"] for i in f) == sorted(IDS)
    st = s["stratification"]
    assert st["variables"] == ["n_nodes", "P"] and len(st["cell_counts"]) == 25
    assert sum(st["dev_cell_counts"]) == 10 and sum(st["cell_counts"]) == 100
    assert set(s["balance_ks_dev_vs_train"]) == {"P", "p1", "p2", "p3", "p4", "p5"}

    assert sp.main(["--dataset", "Tensile2d"]) == 0
    assert sp.main(["--dataset", "Tensile2d"]) == 1  # no silent overwrite
    loaded = sp.load_splits("Tensile2d")
    assert loaded["dev"] == s["dev"] and loaded["folds"] == s["folds"]


def test_load_splits_rejects_changed_cache(fake_cache):
    assert sp.main(["--dataset", "Tensile2d"]) == 0
    index = json.loads((fake_cache / "index.json").read_text())
    index["hf_revision"] = "different"
    (fake_cache / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="different cache"):
        sp.load_splits("Tensile2d")


def test_joint_strata_codes():
    a = np.repeat(np.arange(4.0), 4)  # 4 bins
    b = np.tile(np.arange(4.0), 4)
    labels, edges, cats, n_cells = sp.joint_strata({"a": a, "b": b}, 4)
    assert sorted(labels.tolist()) == list(range(16))  # every (a, b) cell distinct
    assert set(edges) == {"a", "b"} and cats == {} and n_cells == 16


def test_joint_strata_categorical_times_quantile():
    n_inc = np.repeat(np.array([5, 6, 8]), 6)  # integer -> categorical, 3 levels (7 absent)
    load = np.tile(np.arange(6.0), 3)  # float -> 3 quantile bins
    labels, edges, cats, n_cells = sp.joint_strata({"n_inc": n_inc, "load": load}, 3)
    assert cats == {"n_inc": [5, 6, 8]} and set(edges) == {"load"} and n_cells == 9
    assert np.bincount(labels, minlength=9).tolist() == [2] * 9


def test_inclusion_counts_from_split_names():
    index = {"splits": {"DOE_train": [1, 2, 3], "DOE_train_nb_inc_5": [1, 3], "x_nb_inc_7": [2]}}
    assert sp.inclusion_counts(index) == {1: 5, 3: 5, 2: 7}
    assert sp.inclusion_counts({"splits": {"train_500": [1]}}) == {}
    with pytest.raises(ValueError, match="two inclusion-count splits"):
        sp.inclusion_counts({"splits": {"a_nb_inc_5": [1], "b_nb_inc_6": [1]}})


def test_multischypel_stratifier():
    index = {"splits": {"DOE_train_nb_inc_5": [10], "DOE_train_nb_inc_6": [11]}}
    scal = {10: {"C11": 1.0, "C12": 0.0, "C22": 1.0}, 11: {"C11": 1.3, "C12": 0.1, "C22": 0.9}}
    v = sp.STRATIFIERS["2D_MultiScHypEl"](index, [10, 11], scal)
    assert v["n_inclusions"].tolist() == [5, 6] and v["n_inclusions"].dtype.kind == "i"
    assert v["C_minus_I_norm"] == pytest.approx([0.0, np.sqrt(0.09 + 0.02 + 0.01)])
    with pytest.raises(ValueError, match="without an inclusion-count split"):
        sp.STRATIFIERS["2D_MultiScHypEl"](index, [10, 12], {**scal, 12: scal[10]})


def test_train_val_ids_dev_and_folds():
    s = {"train": [5, 3, 9], "dev": [1, 7], "folds": [[3, 1], [9, 7], [5]]}
    assert sp.train_val_ids(s, None) == ([5, 3, 9], [1, 7])
    train, val = sp.train_val_ids(s, 1)
    assert val == [9, 7] and train == [3, 1, 5]
    assert not set(train) & set(val) and set(train) | set(val) == {1, 3, 5, 7, 9}
    with pytest.raises(ValueError, match="fold"):
        sp.train_val_ids(s, 3)


def test_full_train_ids_cover_everything_once():
    s = {"train": [5, 3, 9], "dev": [1, 7], "folds": [[3, 1], [9, 7], [5]]}
    assert sp.full_train_ids(s) == [3, 1, 9, 7, 5]
    with pytest.raises(ValueError, match="partition"):
        sp.full_train_ids({**s, "folds": [[3, 1], [9]]})  # 5 and 7 missing

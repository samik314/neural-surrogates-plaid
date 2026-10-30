import hashlib
import json

import numpy as np
import pytest

from plaid_surrogates.data.cache import load_split, save_sample, write_index
from plaid_surrogates.metrics import evaluate, reference_records
from plaid_surrogates.submission import (
    build_records,
    read_submission,
    write_split_submission,
    write_submission,
)
from tests.test_cache import make_sample

FIELDS, SCALARS = ("U1", "sig12"), ("smax",)
IDS = [7, 3, 11]  # stored order differs from sorted order
NN = {7: 4, 3: 6, 11: 5}


def preds():
    # Inserted in SORTED order on purpose: the writer must not follow dict order.
    return {
        i: {"U1": np.full(NN[i], float(i)), "sig12": np.zeros(NN[i]), "smax": float(i)}
        for i in sorted(IDS)
    }


def test_records_follow_split_order_not_dict_order():
    recs = build_records(preds(), IDS, NN, FIELDS, SCALARS)
    assert [r["smax"] for r in recs] == [7.0, 3.0, 11.0]
    assert [len(r["U1"]) for r in recs] == [4, 6, 5]
    assert all(r["U1"][0] == r["smax"] for r in recs)


def test_record_types():
    p = preds()
    p[7]["U1"] = p[7]["U1"].astype(np.float32)
    p[7]["smax"] = np.float32(7.0)
    rec = build_records(p, IDS, NN, FIELDS, SCALARS)[0]
    assert rec["U1"].dtype == np.float64 and rec["U1"].ndim == 1
    assert type(rec["smax"]) is float
    assert list(rec) == ["U1", "sig12", "smax"]


def test_records_do_not_alias_inputs():
    p = preds()
    rec = build_records(p, IDS, NN, FIELDS, SCALARS)[0]
    p[7]["U1"][0] = -1.0
    assert rec["U1"][0] == 7.0


def _drop(d, k):
    d.pop(k)
    return d


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (lambda p: _drop(p, 3), "missing \\[3\\]"),
        (lambda p: p.update({99: p[3]}) or p, "extra \\[99\\]"),
        (lambda p: _drop(p[3], "smax") and p, "keys"),
        (lambda p: p[3].update(q=np.zeros(6)) or p, "keys"),
        (lambda p: p[3].update(U1=np.zeros(5)) or p, "mesh has 6 nodes"),
        (lambda p: p[3].update(U1=np.zeros((6, 1))) or p, "shape"),
        (lambda p: p[3]["U1"].__setitem__(2, np.nan) or p, "non-finite"),
        (lambda p: p[3].update(smax=np.array([1.0, 2.0])) or p, "one finite number"),
        (lambda p: p[3].update(smax=float("inf")) or p, "one finite number"),
    ],
)
def test_validation(mutate, msg):
    with pytest.raises(ValueError, match=msg):
        build_records(mutate(preds()), IDS, NN, FIELDS, SCALARS)


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        build_records(preds(), [7, 3, 7], NN, FIELDS, SCALARS)


def test_write_read_round_trip(tmp_path):
    recs = build_records(preds(), IDS, NN, FIELDS, SCALARS)
    out = tmp_path / "sub" / "prediction_fake.pkl"
    meta = write_submission(recs, out, {"run_id": "r1"})
    raw = out.read_bytes()
    assert raw[:2] == b"\x80\x04"  # pickle protocol 4
    assert meta["sha256"] == hashlib.sha256(raw).hexdigest() and meta["n_samples"] == 3
    assert json.loads((out.parent / "submission_meta.json").read_text())["run_id"] == "r1"
    back = read_submission(out)
    assert [list(r) for r in back] == [list(r) for r in recs]
    for a, b in zip(back, recs, strict=True):
        np.testing.assert_array_equal(a["U1"], b["U1"])
        assert a["smax"] == b["smax"]
    assert not list(out.parent.glob("*.tmp"))


def test_dev_file_scores_exactly_like_in_memory(tmp_path):
    """Round trip (docs/methods.md §3): a dev-split file scores exactly like in-memory preds."""
    cache = tmp_path / "cache"
    order = [30, 4, 17]
    info = {}
    for i in order:
        s = make_sample(sample_id=i, n=7, seed=i)
        save_sample(cache, s)
        info[i] = {"file": f"samples/{i:06d}.npz", "n_nodes": s.n_nodes}
    write_index(cache, {"dataset": "fake"}, {"train": order}, info)

    dev_ids = [17, 30]  # an explicit, unsorted dev list
    samples = [s for s in load_split(cache, "train") if s.sample_id in dev_ids]
    samples.sort(key=lambda s: dev_ids.index(s.sample_id))
    rng = np.random.default_rng(0)
    p = {
        s.sample_id: {
            "U1": s.output_fields["U1"] + rng.normal(0, 0.1, s.n_nodes),
            "sig12": s.output_fields["sig12"] * 1.1,
            "max_q": s.output_scalars["max_q"] * 0.9,
        }
        for s in samples
    }
    fields, scalars = ("U1", "sig12"), ("max_q",)
    refs = reference_records(samples, fields, scalars)
    recs, meta = write_split_submission(
        p, cache, dev_ids, fields, scalars, tmp_path / "dev.pkl", {}, tmp_path / "dev.meta.json"
    )
    in_memory = evaluate(refs, [p[i] for i in dev_ids], fields, scalars)
    from_file = evaluate(refs, read_submission(tmp_path / "dev.pkl"), fields, scalars)
    assert from_file == in_memory and from_file["total_error"] > 0
    assert meta["split"] == "explicit id list" and meta["n_samples"] == 2


def test_split_name_uses_stored_order(tmp_path):
    cache = tmp_path / "cache"
    order = [30, 4, 17]
    info = {}
    for i in order:
        s = make_sample(sample_id=i, n=5 + i % 4, outputs=False, seed=i)
        s.cells = np.array([[0, 1, 2]], dtype=np.int64)  # make_sample's cells need >= 7 nodes
        s.nodal_tags = {}
        save_sample(cache, s)
        info[i] = {"file": f"samples/{i:06d}.npz", "n_nodes": 5 + i % 4}
    write_index(cache, {"dataset": "fake"}, {"test": order}, info)
    p = {i: {"U1": np.full(5 + i % 4, float(i))} for i in sorted(order)}
    recs, _ = write_split_submission(p, cache, "test", ("U1",), (), tmp_path / "t.pkl", {})
    assert [r["U1"][0] for r in recs] == [30.0, 4.0, 17.0]

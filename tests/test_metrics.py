import math

import numpy as np
import pytest

from plaid_surrogates.metrics import (
    evaluate,
    evaluate_maxcomponent_diagnostic,
    field_rrmse,
    field_rrmse_maxcomponent_diagnostic,
    scalar_rrmse,
)

rng = np.random.default_rng(0)


def test_perfect_prediction_is_zero():
    refs = [rng.standard_normal(n) for n in (5, 9, 13)]
    assert field_rrmse(refs, [r.copy() for r in refs]) == 0.0
    assert scalar_rrmse([1.5, -2.0], [1.5, -2.0]) == 0.0
    recs = [{"u": r, "s": float(i + 1)} for i, r in enumerate(refs)]
    assert evaluate(recs, recs, ["u"], ["s"])["total_error"] == 0.0


def test_constant_offset_hand_computed():
    # ref has max|f| = 4 in sample 0 and 2 in sample 1; offset c = 0.1 everywhere.
    refs = [np.array([1.0, -4.0, 2.0]), np.array([2.0, 0.5])]
    preds = [r + 0.1 for r in refs]
    expected = math.sqrt(0.5 * (0.1**2 / 16 + 0.1**2 / 4))
    assert field_rrmse(refs, preds) == pytest.approx(expected, rel=1e-14)


def test_normalization_is_per_sample_not_dataset_wide():
    # Same relative error pattern at scales 1 and 1e6 -> identical RRMSE to either alone.
    base = np.array([0.2, -1.0, 0.7, 0.4])
    err = np.array([0.01, -0.02, 0.0, 0.03])
    one = field_rrmse([base], [base + err])
    both = field_rrmse([base, 1e6 * base], [base + err, 1e6 * (base + err)])
    assert both == pytest.approx(one, rel=1e-12)
    # A dataset-wide max would make the small-scale sample's error vanish instead.
    pooled_scale = 1e6
    pooled = math.sqrt(0.5 * (np.mean(err**2) + np.mean((1e6 * err) ** 2)) / pooled_scale**2)
    assert not math.isclose(both, pooled, rel_tol=1e-3)


def test_nodes_averaged_within_each_sample_before_averaging_samples():
    # Sample A: 2 nodes, one wrong by 1 (max|ref| = 1). Sample B: 8 nodes, all exact.
    refs = [np.array([1.0, 1.0]), np.ones(8)]
    preds = [np.array([0.0, 1.0]), np.ones(8)]
    per_sample = math.sqrt(0.5 * (0.5 + 0.0))  # mean over samples of per-sample node means
    pooled_nodes = math.sqrt(1.0 / 10)  # pooling all 10 nodes would give this instead
    got = field_rrmse(refs, preds)
    assert got == pytest.approx(per_sample, rel=1e-14)
    assert not math.isclose(got, pooled_nodes, rel_tol=1e-3)


def test_scalar_hand_computed():
    assert scalar_rrmse([2.0, 4.0], [3.0, 4.0]) == pytest.approx(math.sqrt(0.125), rel=1e-14)
    assert scalar_rrmse([-2.0], [-1.0]) == pytest.approx(0.5, rel=1e-14)  # sign-safe


def test_total_error_is_mean_of_per_output_rrmse():
    refs = [{"a": np.array([1.0, 2.0]), "b": np.array([10.0, -5.0]), "s": 2.0}]
    preds = [{"a": np.array([1.5, 2.0]), "b": np.array([10.0, -5.0]), "s": 3.0}]
    res = evaluate(refs, preds, ["a", "b"], ["s"])
    ra, rb, rs = math.sqrt(0.25 / 2 / 4), 0.0, 0.5
    assert res["rrmse"] == pytest.approx({"a": ra, "b": rb, "s": rs}, rel=1e-14)
    assert res["total_error"] == pytest.approx((ra + rb + rs) / 3, rel=1e-14)
    pooled = math.sqrt((ra**2 + rb**2 + rs**2) / 3)
    assert not math.isclose(res["total_error"], pooled, rel_tol=1e-3)
    assert res["n_samples"] == 1


def test_accepts_numpy_and_float32_types():
    refs = [{"u": np.array([1.0, 2.0], dtype=np.float32), "s": np.float32(2.0)}]
    preds = [{"u": [1.0, 2.0], "s": np.array(2.0)}]
    assert evaluate(refs, preds, ["u"], ["s"])["total_error"] == 0.0


@pytest.mark.parametrize(
    "refs, preds, err, msg",
    [
        ([{"u": [1.0, 2.0]}], [], ValueError, "predictions"),
        ([{"u": [1.0, 2.0]}], [{}], KeyError, "'u'"),
        ([{"u": [1.0, 2.0]}], [{"u": [1.0]}], ValueError, "mesh has 2 nodes"),
        ([{"u": [1.0, 2.0]}], [{"u": [1.0, np.nan]}], ValueError, "non-finite"),
        ([{"u": [1.0, 2.0]}], [{"u": [[1.0, 2.0]]}], ValueError, "1-D"),
        ([{"u": [0.0, 0.0]}], [{"u": [0.0, 0.0]}], ValueError, "normalizer"),
    ],
)
def test_field_input_errors(refs, preds, err, msg):
    with pytest.raises(err, match=msg):
        evaluate(refs, preds, ["u"], [])


@pytest.mark.parametrize(
    "ref, pred, msg",
    [
        ([0.0], [1.0], "reference is 0"),
        ([1.0], [np.inf], "not finite"),
        ([1.0], [np.array([1.0, 2.0])], "single number"),
    ],
)
def test_scalar_input_errors(ref, pred, msg):
    with pytest.raises(ValueError, match=msg):
        scalar_rrmse(ref, pred)


def test_empty_and_nothing_to_score():
    with pytest.raises(ValueError, match="no samples"):
        evaluate([], [], ["u"], [])
    with pytest.raises(ValueError, match="nothing"):
        evaluate([{"u": [1.0]}], [{"u": [1.0]}], [], [])


def test_maxcomponent_diagnostic_vs_maxabs():
    pos = [np.array([0.5, 2.0, 1.0])]
    pos_pred = [np.array([0.6, 2.0, 1.0])]
    assert field_rrmse_maxcomponent_diagnostic(pos, pos_pred) == pytest.approx(
        field_rrmse(pos, pos_pred), rel=1e-14
    )
    neg = [np.array([-4.0, -2.0, 1.0])]  # max = 1, max|.| = 4
    neg_pred = [np.array([-4.0, -2.0, 1.5])]
    assert field_rrmse_maxcomponent_diagnostic(neg, neg_pred) == pytest.approx(
        4 * field_rrmse(neg, neg_pred), rel=1e-14
    )
    with pytest.raises(ValueError, match="normalizer"):
        field_rrmse_maxcomponent_diagnostic([np.array([-1.0, -2.0])], [np.zeros(2)])
    recs, precs = [{"u": neg[0], "s": 2.0}], [{"u": neg_pred[0], "s": 3.0}]
    d = evaluate_maxcomponent_diagnostic(recs, precs, ["u"], ["s"])
    assert d["rrmse"]["s"] == evaluate(recs, precs, ["u"], ["s"])["rrmse"]["s"]
    assert d["total_error"] == pytest.approx((d["rrmse"]["u"] + 0.5) / 2, rel=1e-14)

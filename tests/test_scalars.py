import numpy as np
import pytest

from plaid_surrogates.data.cache import CachedSample
from plaid_surrogates.scalars import TRAINING_MEAN, ScalarPredictor, options

SCALARS = ("max_von_mises", "max_U2_top", "max_sig22_top")


def _sample(i, p, u2, sig, out=None):
    n = len(u2)
    return CachedSample(
        sample_id=i,
        points=np.zeros((n, 2)),
        cells=np.zeros((0, 3), np.int64),
        input_scalars={"P": p},
        output_fields={"U2": np.asarray(u2, float), **sig},
        output_scalars=out or {},
        nodal_tags={"Top": np.array([0, 1], dtype=np.int64)},
    )


def _sig(n, s22):
    return {"sig11": np.zeros(n), "sig22": np.asarray(s22, float), "sig12": np.zeros(n)}


def test_options_listing():
    assert options("Tensile2d", "max_U2_top")[0] == TRAINING_MEAN
    assert "-P" in options("Tensile2d", "max_sig22_top")
    vm = options("Tensile2d", "max_von_mises")
    assert not any("fitted" in o for o in vm)  # fitted candidates are not selectable
    assert options("2D_MultiScHypEl", "effective_energy") == [TRAINING_MEAN, "RVE average of psi"]
    assert options("2D_MultiScHypEl", "no_such_scalar") == [TRAINING_MEAN]


def test_fit_and_predict_mixed_methods():
    train = [
        _sample(0, -40.0, [1, 2, 9], _sig(3, [40, 41, 0]),
                {"max_von_mises": 10.0, "max_U2_top": 2.0, "max_sig22_top": 41.0}),
        _sample(1, -50.0, [3, 1, 0], _sig(3, [50, 49, 0]),
                {"max_von_mises": 20.0, "max_U2_top": 3.0, "max_sig22_top": 50.0}),
    ]
    methods = {
        "max_von_mises": TRAINING_MEAN,
        "max_U2_top": "max(U2) over Top",
        "max_sig22_top": "-P",
    }
    sp = ScalarPredictor.fit("Tensile2d", methods, train, SCALARS)
    assert sp.training_means == {"max_von_mises": 15.0, "max_U2_top": 2.5, "max_sig22_top": 45.5}
    new = _sample(7, -42.0, [5.0, 6.0, 100.0], _sig(3, [1, 2, 3]))
    pred = sp.predict(new.output_fields, new)
    # U2 identity uses Top nodes only (node 2 = 100 is not on Top); -P uses the input
    assert pred == {"max_von_mises": 15.0, "max_U2_top": 6.0, "max_sig22_top": 42.0}

    rr = sp.option_rrmse({s.sample_id: s.output_fields for s in train}, train)
    assert rr["max_U2_top"]["max(U2) over Top"] == 0.0
    assert rr["max_sig22_top"]["max(sig22) over Top"] == 0.0
    assert rr["max_sig22_top"]["-P"] == pytest.approx(np.sqrt(((1 / 41) ** 2 + 0) / 2))
    assert set(rr["max_von_mises"]) == set(options("Tensile2d", "max_von_mises"))


def test_option_rrmse_skips_options_that_need_missing_tags():
    s = _sample(0, -40.0, [1, 2], _sig(2, [40, 41]),
                {"max_von_mises": 1.0, "max_U2_top": 2.0, "max_sig22_top": 41.0})
    s.nodal_tags = {}
    sp = ScalarPredictor(
        "Tensile2d", dict.fromkeys(SCALARS, TRAINING_MEAN), dict.fromkeys(SCALARS, 1.0)
    )
    rr = sp.option_rrmse({0: s.output_fields}, [s])
    assert rr["max_U2_top"]["max(U2) over Top"] is None
    assert rr["max_U2_top"]["max(U2) over all nodes"] == 0.0


def test_validation_errors():
    train = [_sample(0, -40.0, [1], _sig(1, [1]), dict.fromkeys(SCALARS, 1.0))]
    with pytest.raises(ValueError, match="unknown predictor"):
        ScalarPredictor.fit("Tensile2d", {**dict.fromkeys(SCALARS, TRAINING_MEAN),
                                          "max_U2_top": "typo"}, train, SCALARS)
    with pytest.raises(ValueError, match="missing"):
        ScalarPredictor.fit("Tensile2d", {"max_U2_top": TRAINING_MEAN}, train, SCALARS)


def test_round_trip_and_legacy_format():
    sp = ScalarPredictor("Tensile2d", {"max_sig22_top": "-P"}, {"max_sig22_top": 45.0})
    back = ScalarPredictor.from_dict(sp.to_dict(), "Tensile2d")
    assert back.methods == sp.methods and back.training_means == sp.training_means
    legacy = {"kind": "training mean", "fit_on_n_samples": 450, "values": {"max_q": 3.0}}
    old = ScalarPredictor.from_dict(legacy, "Tensile2d")
    assert old.methods == {"max_q": TRAINING_MEAN} and old.training_means == {"max_q": 3.0}
    with pytest.raises(ValueError, match="unknown scalar_predictor format"):
        ScalarPredictor.from_dict({"something": 1}, "Tensile2d")

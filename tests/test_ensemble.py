"""ensemble.py on two tiny CPU runs over the fake Tensile2d cache (fixture from test_train)."""

import json

import numpy as np
import pytest

from plaid_surrogates import ensemble, predict, train
from plaid_surrogates.data import splits
from plaid_surrogates.data.dataset import CacheDataset
from plaid_surrogates.data.registry import get_spec
from plaid_surrogates.metrics import evaluate, reference_records
from plaid_surrogates.submission import read_submission
from plaid_surrogates.utils.paths import data_dir
from tests.test_train import env  # noqa: F401  (pytest fixture)

T2D = get_spec("Tensile2d")


def test_mean_records_and_checks():
    a = [{"f": np.array([1.0, 3.0]), "s": 2.0}]
    b = [{"f": np.array([3.0, 5.0]), "s": 4.0}]
    m = ensemble.mean_records([a, b])
    np.testing.assert_array_equal(m[0]["f"], [2.0, 4.0])
    assert m[0]["s"] == 3.0 and isinstance(m[0]["s"], float)
    with pytest.raises(ValueError, match="at least 2"):
        ensemble.mean_records([a])
    with pytest.raises(ValueError, match="shapes differ"):
        ensemble.mean_records([a, [{"f": np.array([1.0]), "s": 1.0}]])
    with pytest.raises(ValueError, match="keys"):
        ensemble.mean_records([a, [{"g": np.array([1.0, 1.0]), "s": 1.0}]])


@pytest.fixture
def two_runs(env):  # noqa: F811
    *_, log, cfg_path = env
    out = []
    for seed in (0, 1):
        run = train.train_run(cfg_path, seed=seed)
        predict.predict_run(run, "dev")
        predict.predict_run(run, "test")
        out.append(run)
    return out, log, env


def test_dev_ensemble_scores_the_mean(two_runs):
    (r0, r1), log, _ = two_runs
    out = ensemble.ensemble_run([r0, r1], "dev", "best")
    ens_dir = out.parent
    assert "_ensemble_" in ens_dir.name and ens_dir.name.endswith("_k2_dev")
    p0 = read_submission(r0 / "predictions_dev_best.pkl")
    p1 = read_submission(r1 / "predictions_dev_best.pkl")
    got = read_submission(out)
    for a, b, g in zip(p0, p1, got, strict=True):
        for k in g:
            np.testing.assert_allclose(g[k], (np.asarray(a[k]) + np.asarray(b[k])) / 2, rtol=1e-15)
    ids = splits.load_splits("Tensile2d")["dev"]
    samples = list(
        CacheDataset(data_dir() / "cache" / "Tensile2d", T2D.train_split, ids=ids, in_memory=True)
    )
    refs = reference_records(samples, T2D.scored_fields, T2D.scored_scalars)
    want = evaluate(refs, got, T2D.scored_fields, T2D.scored_scalars)["total_error"]
    m = json.loads((ens_dir / "metrics_dev.json").read_text())
    assert m["ensemble"]["total_error"] == pytest.approx(want, rel=1e-12)
    assert set(m["members"]) == {r0.name, r1.name}
    row = next(line for line in log.read_text().splitlines() if ens_dir.name in line)
    assert "| ok |" in row and "ensemble-" in row


def test_test_ensemble_writes_a_submission(two_runs):
    (r0, r1), _, (cache, test_ids, *_) = two_runs
    out = ensemble.ensemble_run([str(r0), r1.name], "test", "best")
    sub = read_submission(out)
    assert len(sub) == len(test_ids) and set(sub[0]) == set(T2D.scored_fields + T2D.scored_scalars)
    meta = json.loads((out.parent / "submission_meta.json").read_text())
    assert [m["run_id"] for m in meta["members"]] == [r0.name, r1.name]
    assert meta["leaderboard"] == T2D.leaderboard
    assert (out.parent / "LOG_ROW.md").read_text().startswith("UPLOAD TO: " + T2D.leaderboard)


def test_ensemble_refuses_bad_members(two_runs):
    (r0, r1), _, (*_, cfg_path) = two_runs
    with pytest.raises(ValueError, match="twice"):
        ensemble.ensemble_run([r0, r0], "test", "best")
    with pytest.raises(ValueError, match="best.pt, not last.pt"):
        ensemble.ensemble_run([r0, r1], "test", "last")
    fold_run = train.train_run(cfg_path, seed=2, fold=1)
    predict.predict_run(fold_run, "dev")
    with pytest.raises(ValueError, match="dev-protocol"):
        ensemble.ensemble_run([r0, fold_run], "dev", "best")

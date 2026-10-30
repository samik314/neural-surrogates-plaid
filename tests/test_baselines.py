"""Baselines end to end on a small fake Tensile2d-shaped cache (never touches the real logs)."""

import json

import numpy as np
import pytest

from plaid_surrogates import baselines
from plaid_surrogates.data import splits
from plaid_surrogates.data.cache import (
    CachedSample,
    load_index,
    sample_filename,
    save_sample,
    write_index,
)
from plaid_surrogates.data.registry import get_spec
from plaid_surrogates.submission import read_submission
from plaid_surrogates.utils import runs
from tests.test_runs import LOG

T2D = get_spec("Tensile2d")


def _sample(i, n, rng, has_outputs):
    return CachedSample(
        sample_id=i,
        points=rng.random((n, 2)),
        cells=np.array([[0, 1, 2]], dtype=np.int64),
        input_scalars={p: float(rng.uniform(1, 2)) for p in T2D.input_scalars},
        output_fields=(
            {f: rng.standard_normal(n) + 3.0 for f in T2D.output_fields} if has_outputs else {}
        ),
        output_scalars=(
            {o: float(rng.uniform(1, 2)) for o in T2D.output_scalars} if has_outputs else {}
        ),
    )


@pytest.fixture
def env(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    cache = tmp_path / "data" / "cache" / "Tensile2d"
    train_ids = [int(i) for i in rng.permutation(40)]  # unsorted, like a real split could be
    test_ids = list(range(40, 46))[::-1]  # stored in reverse order
    info = {}
    for i in train_ids + test_ids:
        n = 5 + i % 7  # consecutive test IDs get distinct node counts
        save_sample(cache, _sample(i, n, rng, has_outputs=i < 40))
        info[i] = {"file": sample_filename(i), "n_nodes": n, "n_cells": 1, "has_outputs": i < 40}
    write_index(cache, {"hf_revision": "fake"}, {"train_500": train_ids, "test": test_ids}, info)

    log = tmp_path / "LOG.md"
    log.write_text(LOG)
    monkeypatch.setenv("PLAID_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PLAID_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runs, "experiments_log_path", lambda: log)
    assert splits.main(["--dataset", "Tensile2d"]) == 0
    return cache, test_ids, log


def _tiny(i, u, s):
    n = len(u)
    return CachedSample(i, np.zeros((n, 2)), np.zeros((0, 3), np.int64), {}, {"u": u}, {"s": s})


def test_mean_fit_is_pooled_over_nodes():
    a, b = _tiny(0, np.ones(2), 1.0), _tiny(1, np.zeros(8), 3.0)
    m = baselines.MeanBaseline.fit([a, b], ["u"], ["s"])
    assert m.field_means["u"] == pytest.approx(0.2)  # 2 of 10 nodes are 1 (not (1 + 0) / 2)
    assert m.scalar_means["s"] == pytest.approx(2.0)
    pred = m.predict(b)
    assert pred["u"].shape == (8,) and np.all(pred["u"] == 0.2) and pred["s"] == 2.0


def test_mean_baseline_end_to_end(env):
    cache, test_ids, log = env
    run = baselines.run_baseline("Tensile2d", "mean", submission=True)

    expected_files = (
        "config.json",
        "provenance.json",
        "pip_freeze.txt",
        "baseline_params.json",
        "predictions_dev.pkl",
        "predictions_dev.meta.json",
        "metrics_dev.json",
    )
    for f in expected_files:
        assert (run / f).is_file(), f
    m = json.loads((run / "metrics_dev.json").read_text())
    scored = set(T2D.scored_fields) | set(T2D.scored_scalars)
    assert set(m["maxabs"]["rrmse"]) == scored and m["maxabs"]["n_samples"] == 4
    assert set(m["maxcomponent_diagnostic"]["rrmse"]) == scored

    sub = read_submission(run / "submission" / "prediction_tensile2d.pkl")
    index = load_index(cache)
    assert len(sub) == len(test_ids)
    assert [len(r["U1"]) for r in sub] == [index["samples"][str(i)]["n_nodes"] for i in test_ids]
    assert all(set(r) == scored for r in sub)
    meta = json.loads((run / "submission" / "submission_meta.json").read_text())
    assert meta["split"] == "test" and meta["n_samples"] == len(test_ids)
    assert meta["dev_total_error"] == m["maxabs"]["total_error"]
    assert (run / "submission" / "LOG_ROW.md").read_text().startswith(
        "UPLOAD TO: https://huggingface.co/spaces/PLAIDcompetitions/Tensile2dBenchmark"
    )

    lines = log.read_text().splitlines()
    row = next(k for k, line in enumerate(lines) if run.name in line)
    assert "| ok |" in lines[row] and row < lines.index("## Storage checkpoints")


def test_zero_baseline_scalar_rrmse_is_one(env):
    run = baselines.run_baseline("Tensile2d", "zero")
    m = json.loads((run / "metrics_dev.json").read_text())["maxabs"]["rrmse"]
    for s in T2D.scored_scalars:
        assert m[s] == 1.0
    assert not (run / "submission").exists()


def test_failed_run_is_logged(env, monkeypatch):
    _, _, log = env

    def boom(*a, **k):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(baselines.MeanBaseline, "fit", boom)
    with pytest.raises(RuntimeError, match="synthetic"):
        baselines.run_baseline("Tensile2d", "mean")
    failed = [line for line in log.read_text().splitlines() if "| failed |" in line]
    assert len(failed) == 1 and "synthetic failure" in failed[0]
    run_dirs = list((log.parent / "runs").iterdir())
    assert len(run_dirs) == 1 and (run_dirs[0] / "error.txt").is_file()

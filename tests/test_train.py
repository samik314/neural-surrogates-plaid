"""train -> predict end to end on a small fake Tensile2d-shaped cache, on CPU (never touches the
real logs, data, or runs)."""

import json

import numpy as np
import pytest
import torch
import yaml

from plaid_surrogates import predict, train
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
from plaid_surrogates.utils.paths import repo_root
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
    train_ids = [int(i) for i in rng.permutation(40)]
    test_ids = list(range(40, 46))[::-1]  # stored in reverse order
    info = {}
    for i in train_ids + test_ids:
        n = 6 + i % 9
        save_sample(cache, _sample(i, n, rng, has_outputs=i < 40))
        info[i] = {"file": sample_filename(i), "n_nodes": n, "n_cells": 1, "has_outputs": i < 40}
    write_index(cache, {"hf_revision": "fake"}, {"train_500": train_ids, "test": test_ids}, info)

    log = tmp_path / "LOG.md"
    log.write_text(LOG)
    monkeypatch.setenv("PLAID_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PLAID_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.delenv("PLAID_DEVICE", raising=False)
    monkeypatch.setattr(runs, "experiments_log_path", lambda: log)
    assert splits.main(["--dataset", "Tensile2d"]) == 0

    cfg = yaml.safe_load((repo_root() / "configs" / "tensile2d_transolver_tiny.yaml").read_text())
    cfg["device"] = "cpu"
    cfg["model"].update(n_layers=1, n_hidden=16, n_head=2, slice_num=4, mlp_ratio=2)
    cfg["optim"].update(epochs=2, grad_accum=4)
    cfg_path = tmp_path / "tiny_cpu.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cache, test_ids, log, cfg_path


def test_train_then_predict_end_to_end(env):
    cache, test_ids, log, cfg_path = env
    run = train.train_run(cfg_path, seed=0)

    for f in (
        "config.json",
        "provenance.json",
        "pip_freeze.txt",
        "normalization.json",
        "scalar_predictor.json",
        "curves.csv",
        "metrics_dev.json",
        "best.pt",
        "last.pt",
    ):
        assert (run / f).is_file(), f
    m = json.loads((run / "metrics_dev.json").read_text())
    scored = set(T2D.scored_fields) | set(T2D.scored_scalars)
    assert set(m["best"]["rrmse"]) == scored and m["best"]["n_samples"] == 4
    assert m["last"]["epoch"] == 2 and m["best"]["total_error"] <= m["last"]["total_error"]
    curves = (run / "curves.csv").read_text().splitlines()
    assert len(curves) == 3 and curves[0].startswith("epoch,updates,lr,train_loss,dev_total_error")

    row = next(line for line in log.read_text().splitlines() if run.name in line)
    assert "| ok |" in row and "max_von_mises=training mean" in row  # key order is not fixed
    opts = m["best"]["scalar_options_rrmse"]
    assert opts["max_U2_top"]["training mean"] > 0
    assert opts["max_U2_top"]["max(U2) over Top"] is None  # the fake cache has no Top tag
    assert opts["max_sig22_top"]["-P"] is not None

    # dev round trip: predict.py re-scores its own file and must match the training-time number
    predict.predict_run(run, "dev")
    d = json.loads((run / "metrics_dev_predict_best.json").read_text())
    assert d["maxabs"]["total_error"] == pytest.approx(m["best"]["total_error"], rel=1e-12)
    with pytest.raises(FileExistsError):
        predict.predict_run(run, "dev")
    predict.predict_run(run, "dev", checkpoint="last")

    # test submission: benchmark order, own node counts, scored keys only
    sub = read_submission(predict.predict_run(run, "test"))
    index = load_index(cache)
    assert len(sub) == len(test_ids)
    assert [len(r["U1"]) for r in sub] == [index["samples"][str(i)]["n_nodes"] for i in test_ids]
    assert all(set(r) == scored for r in sub)
    log_row = (run / "submission" / "LOG_ROW.md").read_text()
    assert log_row.startswith("UPLOAD TO: https://huggingface.co/spaces/PLAIDcompetitions/Tensile2d")
    meta = json.loads((run / "submission" / "submission_meta.json").read_text())
    assert meta["leaderboard"] == T2D.leaderboard


def test_fold_run_validates_on_its_fold(env):
    *_, log, cfg_path = env
    run = train.train_run(cfg_path, seed=0, fold=1)
    s = splits.load_splits("Tensile2d")
    assert run.name.endswith("_s0_f1")
    c = json.loads((run / "config.json").read_text())
    assert c["fold"] == 1 and c["val_ids"] == s["folds"][1]
    m = json.loads((run / "metrics_dev.json").read_text())
    assert m["best"]["n_samples"] == len(s["folds"][1])
    norm = json.loads((run / "normalization.json").read_text())
    assert norm["fit_on"]["n_samples"] == 40 - len(s["folds"][1])  # fold k never in the fit
    row = next(line for line in log.read_text().splitlines() if run.name in line)
    assert "| ok |" in row and "| s0/f1 |" in row
    predict.predict_run(run, "dev")  # round trip on the fold's own validation set
    d = json.loads((run / "metrics_dev_predict_best.json").read_text())
    assert d["maxabs"]["n_samples"] == len(s["folds"][1])


def test_full_retrain(env):
    cache, test_ids, log, cfg_path = env
    cfg = yaml.safe_load(cfg_path.read_text())
    s = splits.load_splits("Tensile2d")
    run = train.train_run(cfg_path, seed=0, full=True)
    assert run.name.endswith("_s0_full")
    c = json.loads((run / "config.json").read_text())
    o = cfg["optim"]
    want = train.full_epochs(o["epochs"], len(s["train"]), 40, o["grad_accum"])
    assert c["full"] is True and c["val_ids"] == [] and c["epochs"] == want
    norm = json.loads((run / "normalization.json").read_text())
    assert norm["fit_on"]["n_samples"] == 40  # every training sample, dev included
    assert (run / "last.pt").is_file() and not (run / "best.pt").exists()
    assert json.loads((run / "train_summary.json").read_text())["epochs"] == want
    row = next(line for line in log.read_text().splitlines() if run.name in line)
    assert "| ok |" in row and "| s0/full |" in row and "no validation" in row
    with pytest.raises(ValueError, match="no validation set"):
        predict.predict_run(run, "dev", checkpoint="last")
    sub = read_submission(predict.predict_run(run, "test", checkpoint="last"))
    assert len(sub) == len(test_ids)
    with pytest.raises(ValueError, match="exclusive"):
        train.train_run(cfg_path, seed=0, fold=1, full=True)


def test_full_epochs_freeze_the_number_of_updates():
    # the 2D_MultiScHypEl case: 200 epochs on 688 samples -> 180 epochs on 764 (grad_accum 4)
    assert train.full_epochs(200, 688, 764, 4) == 180
    assert train.full_epochs(200, 688, 688, 4) == 200
    assert train.full_epochs(1, 10, 1000, 4) == 1


def test_metric_loss_run(env):
    *_, log, cfg_path = env
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["loss"] = "metric"
    cfg_path.write_text(yaml.safe_dump(cfg))
    run = train.train_run(cfg_path, seed=0)
    curves = (run / "curves.csv").read_text().splitlines()
    header, last = curves[0].split(","), curves[-1].split(",")
    ema = [k for k, c in enumerate(header) if c.startswith("ema_rrmse_")]
    assert len(ema) == len(T2D.scored_fields) and all(float(last[k]) > 0 for k in ema)
    assert json.loads((run / "config.json").read_text())["loss"] == "metric"
    row = next(line for line in log.read_text().splitlines() if run.name in line)
    assert "| ok |" in row and "loss metric" in row
    predict.predict_run(run, "dev")  # reloads the saved config, loss key included


def test_same_seed_same_result(env):
    *_, cfg_path = env
    a = train.train_run(cfg_path, seed=3)
    b = train.train_run(cfg_path, seed=3)
    assert a != b  # never reuses a run folder
    ma = json.loads((a / "metrics_dev.json").read_text())
    mb = json.loads((b / "metrics_dev.json").read_text())
    assert ma["best"]["total_error"] == mb["best"]["total_error"]


def test_failed_run_is_logged(env, monkeypatch):
    *_, log, cfg_path = env

    def boom(*a, **k):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(train, "build_transolver", boom)
    with pytest.raises(RuntimeError, match="synthetic"):
        train.train_run(cfg_path, seed=0)
    failed = [line for line in log.read_text().splitlines() if "| failed |" in line]
    assert len(failed) == 1 and "synthetic failure" in failed[0]
    run_dirs = list((log.parent / "runs").iterdir())
    assert len(run_dirs) == 1 and (run_dirs[0] / "error.txt").is_file()


def test_bf16_on_cpu_is_refused(env):
    *_, log, cfg_path = env
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["precision"] = "bf16"
    cfg_path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="bf16"):
        train.train_run(cfg_path, seed=0)
    assert "| failed |" in log.read_text()


def test_lr_schedule():
    f = train.lr_factor("cosine", 10)
    assert f(0) == 1.0 and f(10) == pytest.approx(0.0) and f(5) == pytest.approx(0.5)
    assert train.lr_factor("constant", 10)(7) == 1.0
    assert torch.isfinite(torch.tensor(f(20)))

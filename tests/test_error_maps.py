import json

import numpy as np
import pytest

from plaid_surrogates import error_maps as em
from plaid_surrogates.data import splits
from plaid_surrogates.data.cache import (
    CachedSample,
    load_index,
    load_sample,
    sample_filename,
    save_sample,
    write_index,
)
from plaid_surrogates.data.registry import get_spec
from plaid_surrogates.submission import write_split_submission
from tests.test_inspect import grid

T2D = get_spec("Tensile2d")


@pytest.fixture
def run(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    cache = tmp_path / "data" / "cache" / "Tensile2d"
    pts, cells = grid(3)
    top = np.flatnonzero(pts[:, 1] == 1.0).astype(np.int64)
    bottom = np.flatnonzero(pts[:, 1] == 0.0).astype(np.int64)
    train_ids, test_ids, info = list(range(40)), list(range(40, 43)), {}
    for i in train_ids + test_ids:
        s = CachedSample(
            i,
            pts,
            cells,
            {p: float(rng.uniform(1, 2)) for p in T2D.input_scalars},
            nodal_tags={"Top": top, "Bottom": bottom},
        )
        if i < 40:
            s.output_fields = {f: rng.normal(5, 1, len(pts)) for f in T2D.output_fields}
            s.output_scalars = dict.fromkeys(T2D.output_scalars, 1.0)
        save_sample(cache, s)
        info[i] = {"file": sample_filename(i), "n_nodes": len(pts), "has_outputs": i < 40}
    write_index(cache, {"hf_revision": "fake"}, {"train_500": train_ids, "test": test_ids}, info)
    monkeypatch.setenv("PLAID_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PLAID_RUNS_DIR", str(tmp_path / "runs"))
    assert splits.main(["--dataset", "Tensile2d"]) == 0

    dev = splits.load_splits("Tensile2d", cache)["dev"]
    index = load_index(cache)
    preds = {}
    for k, i in enumerate(dev):
        s = load_sample(cache / index["samples"][str(i)]["file"])
        noise = 1e-4 * 10.0**k  # each later dev sample is 10x worse
        preds[i] = {
            **{f: s.output_fields[f] + noise for f in T2D.scored_fields},
            **dict.fromkeys(T2D.scored_scalars, 1.0),
        }
    run_dir = tmp_path / "runs" / "fake_run"
    write_split_submission(
        preds,
        cache,
        dev,
        T2D.scored_fields,
        T2D.scored_scalars,
        run_dir / "predictions_dev_best.pkl",
        {},
    )
    (run_dir / "config.json").write_text(json.dumps({"config": {"dataset": "Tensile2d"}}))
    return run_dir, dev


def test_ranking_figures_and_distance_profile(run):
    run_dir, dev = run
    out = em.run_error_maps(run_dir, "best", n_worst=2)
    ids = [r["id"] for r in out["ranking"]]
    assert ids == list(reversed(dev))  # noise grows along the dev order
    scores = [r["field_score"] for r in out["ranking"]]
    assert scores == sorted(scores, reverse=True)
    for i in out["worst_plotted"]:
        assert (run_dir / "figures" / f"error_map_{i}.png").stat().st_size > 1000
    prof = out["mean_rel_error_by_distance"]
    assert set(prof) == {"top", "bottom"}
    assert sum(out["nodes_per_bin"]["top"]) == len(dev) * 16
    assert json.loads((run_dir / "figures" / "error_maps.json").read_text())["run_id"] == "fake_run"


def test_cli_rejects_unknown_run(run):
    with pytest.raises(FileNotFoundError):
        em.main(["--run", "no_such_run"])

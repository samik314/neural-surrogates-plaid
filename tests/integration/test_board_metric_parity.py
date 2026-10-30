"""Field + scalar metric parity with the 2D_MultiScHypEl board's scoring code (docs/methods.md §2).

The board's "Submission information" page (PLAIDcompetitions/2DMultiscaleHyperelasticityBenchmark,
read 2026-10-08) publishes how the reference file is built and the `_metric` that scores it. Both
are copied below verbatim. We build a board-style reference for our dev samples from the Zenodo
copy and score our own dev prediction file with it:
  A. board `_metric` total and per-output errors == our metrics.py (predict.py round trip);
  B. the board-style reference has the same keys, lengths and kinds as our files;
  C. every test-submission pickle in the run goes through `_metric` (against itself) cleanly.
Needs PLAID_BOARD_RUN (a 2D_MultiScHypEl run folder with predictions_dev_best.pkl and
submission/*.pkl) and PLAID_ZENODO_ROOT (the extracted Zenodo directory); skipped otherwise.
"""

import json
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest

from plaid_surrogates.data.compare_zenodo import find_plaid_root
from plaid_surrogates.data.splits import load_splits
from plaid_surrogates.utils.paths import data_dir

RUN = os.environ.get("PLAID_BOARD_RUN")
ZENODO = os.environ.get("PLAID_ZENODO_ROOT")
pytestmark = pytest.mark.skipif(
    not (RUN and ZENODO), reason="needs PLAID_BOARD_RUN and PLAID_ZENODO_ROOT"
)

# ---- verbatim from the board's submission page ------------------------------------------------
out_fields_names = ["u1", "u2", "P11", "P12", "P22", "P21", "psi"]
out_scalars_names = ["effective_energy"]


def _metric(reference_split, prediction_split):
    assert len(reference_split) == len(prediction_split)
    errors = {name:0. for name in out_fields_names + out_scalars_names}  # noqa: E231
    n_samples = len(reference_split)
    for i in range(n_samples):
        for fn in out_fields_names:
            errors[fn] += (np.linalg.norm(prediction_split[i][fn] - reference_split[i][fn])**2)/(reference_split[i][fn].shape[0]*np.linalg.norm(reference_split[i][fn], ord = np.inf)**2)  # noqa: E501, E251
        for sn in out_scalars_names:
            errors[sn] += ((prediction_split[i][sn] - reference_split[i][sn])**2)/(reference_split[i][sn]**2)  # noqa: E501
    for fn in out_fields_names:
        errors[fn] = np.sqrt(errors[fn]/n_samples)
    for sn in out_scalars_names:
        errors[sn] = np.sqrt(errors[sn]/n_samples)

    return errors
# -----------------------------------------------------------------------------------------------


def _board_reference(ids):
    """The board's reference construction, on the Zenodo copy, for the given sample IDs."""
    from plaid.containers.dataset import Dataset

    dataset = Dataset()
    dataset._load_from_dir_(find_plaid_root(Path(ZENODO)) / "dataset", ids=ids)
    reference = []
    for i, id in enumerate(ids):  # noqa: A001 - as on the board page
        reference.append({})
        sample = dataset[id]
        for fn in out_fields_names:
            reference[i][fn] = sample.get_field(fn)
        for sn in out_scalars_names:
            reference[i][sn] = sample.get_scalar(sn)
    return reference


def _kind(v):
    if isinstance(v, np.ndarray):
        return ("ndarray", v.ndim)
    return ("number", None) if np.isscalar(v) else (type(v).__name__, None)


@pytest.fixture(scope="module")
def dev():
    run = Path(RUN)
    ids = load_splits("2D_MultiScHypEl", data_dir() / "cache" / "2D_MultiScHypEl")["dev"]
    if not (run / "predictions_dev_best.pkl").exists():
        pytest.skip("no dev predictions in this run (e.g. a full retrain); A and B need them")
    with open(run / "predictions_dev_best.pkl", "rb") as fh:
        preds = pickle.load(fh)
    assert len(preds) == len(ids)
    ours = json.loads((run / "metrics_dev_predict_best.json").read_text())["maxabs"]
    return _board_reference(ids), preds, ours


def test_A_board_metric_equals_ours(dev):
    reference, preds, ours = dev
    board = _metric(reference, preds)
    total = float(np.mean(list(board.values())))
    print(f"\nboard _metric total {total:.12f}   ours {ours['total_error']:.12f}")
    for k, v in board.items():
        print(f"  {k:18s} board {float(v):.12f}   ours {ours['rrmse'][k]:.12f}")
    assert total == pytest.approx(ours["total_error"], rel=1e-12)
    for k, v in board.items():
        assert float(v) == pytest.approx(ours["rrmse"][k], rel=1e-12), k


def test_B_reference_matches_our_file_layout(dev):
    reference, preds, _ = dev
    r0, p0 = reference[0], preds[0]
    print("\nreference types:", {k: (type(v).__name__, getattr(v, "dtype", None)) for k, v in r0.items()})  # noqa: E501
    print("our types:      ", {k: (type(v).__name__, getattr(v, "dtype", None)) for k, v in p0.items()})  # noqa: E501
    assert list(r0) == list(p0)
    for r, p in zip(reference, preds, strict=True):
        for k in r:
            assert _kind(r[k]) == _kind(p[k]), k
            if isinstance(r[k], np.ndarray):
                assert r[k].shape == p[k].shape, k


def test_C_every_test_submission_runs_through_board_metric():
    files = sorted((Path(RUN) / "submission").glob("*.pkl"))
    assert files, "no test submission pickles in the run"
    for f in files:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)  # numpy.core path in _np1 files
            with open(f, "rb") as fh:
                sub = pickle.load(fh)
        errors = _metric(sub, sub)
        print(f"\n{f.name}: {len(sub)} samples, errors all zero: {all(v == 0 for v in errors.values())}")  # noqa: E501
        assert len(sub) == 376 and all(v == 0 for v in errors.values()), f.name

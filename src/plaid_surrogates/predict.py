"""Predictions from a trained run, on each sample's own nodes in original order (docs/methods.md
§3).

    python -m plaid_surrogates.predict --run <run folder or id> --split dev|test [--checkpoint best]

dev:  writes predictions_dev_<ckpt>.pkl (+ meta) through the submission writer, re-scores that
      file with metrics.py, and checks the score against the training-time evaluation of the
      same checkpoint (round trip of the whole training path). Mismatch -> error.
test: writes submission/prediction_<dataset>.pkl + submission_meta.json + LOG_ROW.md.
      The upload is manual (Rule 1).
Existing outputs are never overwritten unless --overwrite is given.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from plaid_surrogates.config import config_from_dict
from plaid_surrogates.data.cache import CachedSample
from plaid_surrogates.data.dataset import CacheDataset
from plaid_surrogates.data.features import build_inputs
from plaid_surrogates.data.normalize import Normalizer
from plaid_surrogates.data.registry import get_spec
from plaid_surrogates.data.splits import load_splits, train_val_ids
from plaid_surrogates.metrics import evaluate, reference_records
from plaid_surrogates.models.build import build_transolver
from plaid_surrogates.scalars import ScalarPredictor
from plaid_surrogates.submission import read_submission, write_split_submission
from plaid_surrogates.utils import runs
from plaid_surrogates.utils.device import get_device
from plaid_surrogates.utils.paths import data_dir, runs_dir

CHECKPOINTS = ("best", "last")
SPLITS = ("dev", "test")
# predict.py re-runs the same weights on the same inputs, usually on the same GPU model, so the
# scores should agree to round-off. Anything larger means the training path and this one differ.
ROUNDTRIP_RTOL = 1e-6


def autocast(device: torch.device, precision: str) -> torch.autocast:
    """bf16 autocast when precision == 'bf16', otherwise a disabled (fp32) context."""
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16")


@torch.no_grad()
def predict_fields(
    model: nn.Module,
    samples: Sequence[CachedSample],
    norm: Normalizer,
    device: torch.device,
    precision: str,
    feature_set: str = "input_scalars",
    dataset: str = "",
) -> dict[int, dict[str, np.ndarray]]:
    """Full-mesh forward pass per sample -> {sample_id: {field: (N,) float64, physical units}}."""
    model.eval()
    out: dict[int, dict[str, np.ndarray]] = {}
    for s in samples:
        fx, coords = build_inputs(s, norm, feature_set, dataset)
        with autocast(device, precision):
            y = model(
                torch.from_numpy(fx).to(device).unsqueeze(0),
                embedding=torch.from_numpy(coords).to(device).unsqueeze(0),
            )
        out[s.sample_id] = norm.decode_fields(y[0].float().cpu().numpy())
    return out


def add_scalars(
    fields_by_id: Mapping[int, Mapping[str, np.ndarray]],
    samples: Sequence[CachedSample],
    scalar_predictor: ScalarPredictor,
) -> dict[int, dict[str, Any]]:
    """Complete each sample's record with the configured scalar predictions."""
    return {
        s.sample_id: {
            **fields_by_id[s.sample_id],
            **scalar_predictor.predict(fields_by_id[s.sample_id], s),
        }
        for s in samples
    }


def resolve_run(run: str | Path) -> Path:
    """A run folder path, or a run id under PLAID_RUNS_DIR."""
    for p in (Path(run), runs_dir() / str(run)):
        if p.is_dir():
            return p.resolve()
    raise FileNotFoundError(f"no run folder {run!r} (also looked in {runs_dir()})")


def load_trained(
    run_dir: Path, checkpoint: str, device: torch.device
) -> tuple[nn.Module, Normalizer, ScalarPredictor, dict[str, Any]]:
    """Rebuild the model from a checkpoint; returns (model, normalizer, scalar predictor, ckpt)."""
    if checkpoint not in CHECKPOINTS:
        raise ValueError(f"checkpoint must be one of {CHECKPOINTS}")
    ckpt = torch.load(run_dir / f"{checkpoint}.pt", map_location=device, weights_only=True)
    cfg = config_from_dict(ckpt["config"])
    model = build_transolver(cfg.model, ckpt["functional_dim"], ckpt["out_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    norm = Normalizer.load(run_dir / "normalization.json")
    scalar_predictor = ScalarPredictor.from_dict(
        json.loads((run_dir / "scalar_predictor.json").read_text()), cfg.dataset
    )
    return model, norm, scalar_predictor, ckpt


def _refuse_overwrite(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [p for p in paths if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{existing[0]} exists; pass --overwrite to replace it")


def predict_run(
    run: str | Path,
    split: str,
    checkpoint: str = "best",
    device: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Predict `split` with a run's checkpoint; returns the written prediction file."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    run_dir = resolve_run(run)
    run_config = json.loads((run_dir / "config.json").read_text())
    cfg = config_from_dict(run_config["config"])
    spec = get_spec(cfg.dataset)
    fields, scalars = spec.scored_fields, spec.scored_scalars
    cache_dir = data_dir() / "cache" / cfg.dataset
    dev_ = get_device(device or cfg.device)
    model, norm, scalar_predictor, ckpt = load_trained(run_dir, checkpoint, dev_)
    prov = json.loads((run_dir / "provenance.json").read_text())
    meta = {
        "run_id": run_dir.name,
        "checkpoint": checkpoint,
        "epoch": ckpt["epoch"],
        "git_commit": prov["git_commit"],
        "git_dirty": prov["git_dirty"],
    }

    if split == "dev" and run_config.get("full"):
        raise ValueError(f"{run_dir.name} is a full retrain: it has no validation set")
    if split == "dev":
        splits = load_splits(cfg.dataset, cache_dir)
        if splits["source_hash"] != run_config["splits_source_hash"]:
            raise ValueError("the splits file changed since this run was trained")
        # "dev" = the run's own validation set: dev, or CV fold k for a --fold run
        fold = run_config.get("fold")
        ids = train_val_ids(splits, fold)[1]
        if fold is not None and ids != run_config["val_ids"]:
            raise ValueError(f"fold {fold} IDs differ from the ones this run validated on")
        out = run_dir / f"predictions_dev_{checkpoint}.pkl"
        meta_path = run_dir / f"predictions_dev_{checkpoint}.meta.json"
        metrics_path = run_dir / f"metrics_dev_predict_{checkpoint}.json"
        _refuse_overwrite([out, meta_path, metrics_path], overwrite)
        samples = list(CacheDataset(cache_dir, spec.train_split, ids=ids, in_memory=True))
        fields_by_id = predict_fields(
            model, samples, norm, dev_, cfg.precision, cfg.features, cfg.dataset
        )
        preds = add_scalars(fields_by_id, samples, scalar_predictor)
        write_split_submission(preds, cache_dir, ids, fields, scalars, out, meta, meta_path)
        refs = reference_records(samples, fields, scalars)
        res = evaluate(refs, read_submission(out), fields, scalars)
        train_time = float(ckpt["dev_total_error"])
        rel = abs(res["total_error"] - train_time) / max(abs(train_time), 1e-300)
        metrics_path.write_text(
            json.dumps(
                {
                    "split": "dev",
                    **meta,
                    "maxabs": res,
                    "training_time_total_error": train_time,
                    "relative_difference": rel,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"dev total_error {res['total_error']:.6f} from {out.name}")
        print(f"training-time evaluation of {checkpoint}.pt: {train_time:.6f} (rel diff {rel:.2e})")
        for name, v in res["rrmse"].items():
            print(f"  {name:16s} {v:.4f}")
        if rel > ROUNDTRIP_RTOL:
            raise RuntimeError(
                f"round trip failed: predict.py dev score {res['total_error']!r} != training-time "
                f"{train_time!r} (rel {rel:.2e} > {ROUNDTRIP_RTOL})"
            )
        return out

    sub_dir = run_dir / "submission"
    out = sub_dir / f"prediction_{cfg.dataset.lower()}.pkl"
    _refuse_overwrite([out], overwrite)
    samples = list(CacheDataset(cache_dir, spec.test_split, in_memory=True))
    fields_by_id = predict_fields(
        model, samples, norm, dev_, cfg.precision, cfg.features, cfg.dataset
    )
    preds = add_scalars(fields_by_id, samples, scalar_predictor)
    dev_total = ckpt.get("dev_total_error")  # None for a full retrain (no validation set)
    write_split_submission(
        preds,
        cache_dir,
        spec.test_split,
        fields,
        scalars,
        out,
        {
            **meta,
            "config": run_config,
            "dev_total_error": dev_total,
            "leaderboard": spec.leaderboard,
            "environment": prov["environment"],
        },
    )
    row = {
        "date": "<upload date>",
        "run id": run_dir.name,
        "dataset": cfg.dataset,
        "dev total_error (mean ± std if seeds/CV)": (
            f"{float(dev_total):.4f} ({checkpoint}.pt, epoch {ckpt['epoch']})"
            if dev_total is not None
            else f"— (full retrain, {checkpoint}.pt, epoch {ckpt['epoch']}; quote its CV score)"
        ),
        "notes": f"Transolver {cfg.tag}",
    }
    (sub_dir / "LOG_ROW.md").write_text(runs.log_row_text(row, cfg.dataset, spec.leaderboard))
    print(f"Submission: {out}  ({len(samples)} samples; upload it yourself; see LOG_ROW.md)")
    print(f"Upload to: {spec.leaderboard}")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--run", required=True, help="run folder, or run id under PLAID_RUNS_DIR")
    p.add_argument("--split", required=True, choices=SPLITS)
    p.add_argument("--checkpoint", default="best", choices=CHECKPOINTS)
    p.add_argument("--device", default=None, help="default: the run config's device")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    predict_run(args.run, args.split, args.checkpoint, args.device, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

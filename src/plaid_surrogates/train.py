"""Train a model from a config. Every run gets a run folder and one experiments/LOG.md row,
including failed, killed and OOM runs (docs/methods.md §5).

    python -m plaid_surrogates.train --config configs/tensile2d_transolver_tiny.yaml --seed 0
    python -m plaid_surrogates.train --config <yaml> --seed 0 --fold 2      # 5-fold CV run
    python -m plaid_surrogates.train --config <yaml> --seed 0 --full        # final retrain (§6)

Normalization is fit on the run's training IDs only: the dev protocol's `train` IDs, or, with
`--fold k`, every training sample outside CV fold k (validated on fold k instead of dev).
Training uses full meshes (batch size 1) with gradient accumulation. The exact benchmark metric
(metrics.py, physical units, float64) is computed on the validation set every `eval_every`
epochs and always after the last epoch; two checkpoints are kept: best (lowest validation
total_error, never training loss) and last. Each scalar's predictor (training mean or a checked
identity) is set in the config; every option is also scored on the validation set at each
evaluation, so the choice is made there (docs/methods.md §4).

`--full` is the final retrain (docs/methods.md §2): every official training sample, normalization
and scalar predictors refit on all of them, no validation set (so no evaluation and no best.pt), and
the number of optimizer updates frozen at the dev protocol's (`full_epochs`); only last.pt is kept.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import random
import socket
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from plaid_surrogates.config import TrainConfig, config_to_dict, load_config
from plaid_surrogates.data.dataset import CacheDataset
from plaid_surrogates.data.features import build_inputs, feature_names
from plaid_surrogates.data.normalize import Normalizer
from plaid_surrogates.data.registry import get_spec
from plaid_surrogates.data.splits import (
    full_train_ids,
    load_splits,
    splits_path,
    train_val_ids,
)
from plaid_surrogates.losses import MetricLoss, metric_scale, standardized_mse
from plaid_surrogates.metrics import evaluate, reference_records
from plaid_surrogates.models.build import build_transolver, count_params
from plaid_surrogates.predict import add_scalars, autocast, predict_fields
from plaid_surrogates.scalars import ScalarPredictor
from plaid_surrogates.utils import runs
from plaid_surrogates.utils.device import device_description, get_device
from plaid_surrogates.utils.paths import data_dir, repo_root
from plaid_surrogates.utils.provenance import write_provenance

MODEL_NAME = "transolver"


def seed_everything(seed: int) -> None:
    """Seed python, numpy (legacy global RNG, for any library using it) and torch (all devices)."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - we use explicit Generators; this covers third-party code
    torch.manual_seed(seed)


def lr_factor(schedule: str, total_updates: int):
    """LR multiplier per optimizer update: cosine from 1 to 0 over the run, or constant."""
    if schedule == "constant":
        return lambda step: 1.0
    return lambda step: 0.5 * (1.0 + math.cos(math.pi * min(step, total_updates) / total_updates))


def check_device(cfg: TrainConfig, device: torch.device) -> None:
    """Refuse configs the device can't honour, instead of silently changing them."""
    if cfg.model.use_te and device.type != "cuda":
        raise ValueError(
            "use_te=True needs an NVIDIA GPU with transformer-engine "
            "(docs/design/DSG-001-environment.md)"
        )
    if cfg.precision == "bf16" and not (device.type == "cuda" and torch.cuda.is_bf16_supported()):
        raise ValueError(f"bf16 requested but {device} does not support it")


def save_checkpoint(path: Path, model: nn.Module, payload: dict[str, Any]) -> None:
    """Atomic torch.save of the state dict plus everything needed to rebuild the model."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"model_state": model.state_dict(), **payload}, tmp)
    os.replace(tmp, path)


def full_epochs(epochs: int, n_ref: int, n_full: int, grad_accum: int) -> int:
    """Epochs on `n_full` samples giving the same number of optimizer updates as `epochs` on
    `n_ref` (docs/methods.md §2: freeze the number of updates, not the number of epochs)."""
    per_ref, per_full = math.ceil(n_ref / grad_accum), math.ceil(n_full / grad_accum)
    return max(1, round(epochs * per_ref / per_full))


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def train_run(
    config_path: Path, seed: int, notes: str = "", fold: int | None = None, full: bool = False
) -> Path:
    """Run one training job end to end; returns the run folder. Logs the run even on failure."""
    if full and fold is not None:
        raise ValueError("--full and --fold are exclusive")
    cfg = load_config(config_path)
    spec = get_spec(cfg.dataset)
    fields, scalars = spec.scored_fields, spec.scored_scalars
    cache_dir = data_dir() / "cache" / cfg.dataset
    splits = load_splits(cfg.dataset, cache_dir)
    o = cfg.optim
    if full:
        train_ids, val_ids = full_train_ids(splits), []
        val_name = "none (full retrain)"
        epochs = full_epochs(o.epochs, len(splits["train"]), len(train_ids), o.grad_accum)
    else:
        train_ids, val_ids = train_val_ids(splits, fold)  # validation: dev, or CV fold `fold`
        val_name = "dev" if fold is None else f"fold {fold}"
        epochs = o.epochs
    ref_updates = o.epochs * math.ceil(len(splits["train"]) / o.grad_accum)

    run_dir = runs.make_run_dir(cfg.dataset, MODEL_NAME, cfg.tag, seed, fold, full=full)
    t0 = time.time()
    log_row: dict[str, Any] = {
        "date": dt.date.today().isoformat(),
        "run id": run_dir.name,
        "dataset": cfg.dataset,
        "model": f"{MODEL_NAME}-{cfg.tag}",
        "parent run": "—",
        "key change": f"config {Path(config_path).name}",
        "seed/fold": (
            f"s{seed}/full" if full else f"s{seed}" if fold is None else f"s{seed}/f{fold}"
        ),
        "machine / GPU": socket.gethostname(),
    }
    try:
        device = get_device(cfg.device)
        log_row["machine / GPU"] = f"{socket.gethostname()} / {device_description(device)}"
        check_device(cfg, device)
        _write_json(
            run_dir / "config.json",
            {
                "config": config_to_dict(cfg),
                "config_file": str(config_path),
                "seed": seed,
                "fold": fold,
                "full": full,
                "epochs": epochs,
                "updates_matched_to": (
                    f"dev protocol: {o.epochs} epochs x {len(splits['train'])} samples "
                    f"= {ref_updates} updates"
                )
                if full
                else None,
                "val_ids": val_ids,
                "model": MODEL_NAME,
                "feature_set": cfg.features,
                "loss": cfg.loss,
                "scored_fields": list(fields),
                "scored_scalars": list(scalars),
                "scalar_predictors": cfg.scalars,
                "splits_file": str(splits_path(cfg.dataset)),
                "splits_source_hash": splits["source_hash"],
                "fit_on": (
                    f"all of {spec.train_split} ({len(train_ids)} samples)"
                    if full
                    else f"{spec.train_split} minus {val_name} ({len(train_ids)} samples)"
                ),
                "evaluated_on": f"{val_name} ({len(val_ids)} samples)",
            },
        )
        write_provenance(
            run_dir,
            repo_root(),
            extra={
                "run_id": run_dir.name,
                "seed": seed,
                "device": device_description(device),
                "precision": cfg.precision,
                "config_file": str(config_path),
            },
        )
        seed_everything(seed)

        train_samples = list(
            CacheDataset(cache_dir, spec.train_split, ids=train_ids, in_memory=True)
        )
        dev_samples = list(
            CacheDataset(cache_dir, spec.train_split, ids=val_ids, in_memory=True)
        )
        norm = Normalizer.fit(train_samples, spec.input_scalars, fields)
        norm.save(run_dir / "normalization.json")
        scalar_predictor = ScalarPredictor.fit(cfg.dataset, cfg.scalars, train_samples, scalars)
        _write_json(run_dir / "scalar_predictor.json", scalar_predictor.to_dict())

        names = feature_names(norm, cfg.features, cfg.dataset)
        functional_dim, out_dim = len(names), len(fields)
        model = build_transolver(cfg.model, functional_dim, out_dim).to(device)
        n_params = count_params(model)
        log_row["params"] = n_params
        print(
            f"{MODEL_NAME}: {n_params} params, fx dim {functional_dim}, out dim {out_dim}, "
            f"{device_description(device)}"
        )

        # Whole training portion on the device once (Tensile2d: ~0.2 GB in float32). The metric
        # loss also needs each sample's fixed rescaling c_f = std_f^2 / max|ref_f|^2 (losses.py).
        data, scales = [], []
        for s in train_samples:
            fx, coords = build_inputs(s, norm, cfg.features, cfg.dataset)
            y = norm.encode_fields(s).astype(np.float32)
            data.append(tuple(torch.from_numpy(a).to(device).unsqueeze(0) for a in (fx, coords, y)))
            if cfg.loss == "metric":
                ref = np.stack([s.output_fields[f] for f in fields], axis=1)
                c = metric_scale(norm.field_scale(), ref)
                scales.append(torch.from_numpy(c.astype(np.float32)).to(device))
        metric_loss = MetricLoss(len(fields)) if cfg.loss == "metric" else None
        refs = reference_records(dev_samples, fields, scalars) if dev_samples else []

        n = len(data)
        total_updates = epochs * math.ceil(n / o.grad_accum)
        opt = torch.optim.AdamW(model.parameters(), lr=o.lr, weight_decay=o.weight_decay)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor(o.schedule, total_updates))
        order_rng = np.random.default_rng(seed)
        ckpt_payload = {
            "config": config_to_dict(cfg),
            "functional_dim": functional_dim,
            "feature_names": names,
            "out_dim": out_dim,
            "fields": list(fields),
            "seed": seed,
        }
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        ema_cols = [f"ema_rrmse_{f}" for f in fields] if metric_loss else []
        curve_cols = [
            "epoch", "updates", "lr", "train_loss", "dev_total_error", *fields, *scalars,
            *ema_cols, "epoch_s",
        ]
        curves_path = run_dir / "curves.csv"
        with curves_path.open("w", newline="") as fh:
            csv.writer(fh).writerow(curve_cols)

        best: dict[str, Any] = {"total_error": math.inf}
        last: dict[str, Any] = {}
        updates = 0
        for epoch in range(1, epochs + 1):
            te = time.time()
            model.train()
            opt.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for k, j in enumerate(order_rng.permutation(n)):
                fx_t, coords_t, y_t = data[j]
                with autocast(device, cfg.precision):
                    pred = model(fx_t, embedding=coords_t)
                if metric_loss is None:
                    loss = standardized_mse(pred.float(), y_t)
                else:
                    loss = metric_loss(pred.float(), y_t, scales[j])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch}, step {k}")
                group = min(o.grad_accum, n - (k // o.grad_accum) * o.grad_accum)
                (loss / group).backward()
                loss_sum += loss.item()
                if (k + 1) % o.grad_accum == 0 or k == n - 1:
                    if o.grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), o.grad_clip)
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                    updates += 1

            row: dict[str, Any] = {
                "epoch": epoch,
                "updates": updates,
                "lr": sched.get_last_lr()[0],
                "train_loss": loss_sum / n,
            }
            if metric_loss is not None:
                row.update(zip(ema_cols, metric_loss.ema_rrmse().tolist(), strict=True))
            if dev_samples and (epoch % cfg.eval_every == 0 or epoch == epochs):
                fields_by_id = predict_fields(
                    model, dev_samples, norm, device, cfg.precision, cfg.features, cfg.dataset
                )
                preds = add_scalars(fields_by_id, dev_samples, scalar_predictor)
                res = evaluate(refs, [preds[s.sample_id] for s in dev_samples], fields, scalars)
                res["scalar_options_rrmse"] = scalar_predictor.option_rrmse(
                    fields_by_id, dev_samples
                )
                last = {"epoch": epoch, **res}
                row.update({"dev_total_error": res["total_error"], **res["rrmse"]})
                if res["total_error"] < best["total_error"]:
                    best = {"epoch": epoch, **res}
                    save_checkpoint(
                        run_dir / "best.pt",
                        model,
                        {**ckpt_payload, "epoch": epoch, "dev_total_error": res["total_error"]},
                    )
            row["epoch_s"] = round(time.time() - te, 2)
            with curves_path.open("a", newline="") as fh:
                csv.writer(fh).writerow([row.get(c, "") for c in curve_cols])
            print(
                f"epoch {epoch}/{epochs}  loss {row['train_loss']:.4e}  "
                f"dev {row.get('dev_total_error', float('nan')):.4f}  ({row['epoch_s']} s)",
                flush=True,
            )

        if full:
            last = {"epoch": epochs, "total_error": None}
        save_checkpoint(
            run_dir / "last.pt",
            model,
            {**ckpt_payload, "epoch": last["epoch"], "dev_total_error": last["total_error"]},
        )
        peak_mib = None
        if device.type == "cuda":
            peak_mib = torch.cuda.max_memory_allocated(device) / 2**20
        if full:
            final_row = dict(row)
            _write_json(
                run_dir / "train_summary.json",
                {
                    "full": True,
                    "epochs": epochs,
                    "updates": updates,
                    "updates_matched_to": ref_updates,
                    "n_train": n,
                    "final_train_loss": final_row["train_loss"],
                    "final_ema_rrmse": {c[10:]: final_row[c] for c in ema_cols},
                    "peak_gpu_mib": peak_mib,
                },
            )
        else:
            _write_json(
                run_dir / "metrics_dev.json",
                {"split": "dev", "best": best, "last": last, "peak_gpu_mib": peak_mib},
            )
    except BaseException as e:
        (run_dir / "error.txt").write_text(traceback.format_exc())
        if isinstance(e, KeyboardInterrupt):
            log_row["status"] = "killed"
        elif isinstance(e, torch.OutOfMemoryError):
            log_row["status"] = "OOM"
        else:
            log_row["status"] = "failed"
        log_row["wall time"] = f"{time.time() - t0:.1f} s"
        log_row["notes"] = f"{type(e).__name__}: {e}"[:200]
        runs.append_table_row(runs.experiments_log_path(), runs.EXPERIMENT_COLUMNS, log_row)
        raise

    if full:
        note_parts = [
            f"FULL RETRAIN on all {len(train_ids)} {spec.train_split} samples, no validation",
            f"{epochs} epochs = {updates} updates (dev protocol: {ref_updates})",
            "final checkpoint last.pt only",
            f"{cfg.precision}",
            f"features {cfg.features}",
            f"loss {cfg.loss}",
            f"peak {peak_mib:.0f} MiB" if peak_mib is not None else "",
            notes,
        ]
        log_row.update(
            {
                "status": "ok",
                "wall time": f"{time.time() - t0:.1f} s",
                "dev total_error": "— (no validation)",
                "worst output (RRMSE)": "—",
                "notes": "; ".join(x for x in note_parts if x),
            }
        )
        runs.append_table_row(runs.experiments_log_path(), runs.EXPERIMENT_COLUMNS, log_row)
        print(f"Run: {run_dir}")
        print(f"full retrain: {epochs} epochs, {updates} updates (dev protocol {ref_updates})")
        return run_dir

    worst = max(best["rrmse"], key=best["rrmse"].get)
    note_parts = [
        f"best epoch {best['epoch']}/{cfg.optim.epochs}",
        f"last {last['total_error']:.4f}",
        f"{cfg.precision}",
        f"features {cfg.features}",
        f"loss {cfg.loss}",
        "scalars: " + ", ".join(f"{k}={v}" for k, v in cfg.scalars.items()),
        f"peak {peak_mib:.0f} MiB" if peak_mib is not None else "",
        notes,
    ]
    log_row.update(
        {
            "status": "ok",
            "wall time": f"{time.time() - t0:.1f} s",
            "dev total_error": f"{best['total_error']:.4f}",
            "worst output (RRMSE)": f"{worst} ({best['rrmse'][worst]:.4f})",
            "notes": "; ".join(x for x in note_parts if x),
        }
    )
    runs.append_table_row(runs.experiments_log_path(), runs.EXPERIMENT_COLUMNS, log_row)
    print(f"Run: {run_dir}")
    print(f"best dev total_error {best['total_error']:.4f} (epoch {best['epoch']})")
    for name, v in best["rrmse"].items():
        print(f"  {name:16s} {v:.4f}")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--notes", default="")
    p.add_argument("--fold", type=int, default=None, help="CV fold k: validate on folds[k]")
    p.add_argument("--full", action="store_true", help="final retrain on all training data")
    args = p.parse_args(argv)
    train_run(args.config, args.seed, args.notes, args.fold, args.full)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

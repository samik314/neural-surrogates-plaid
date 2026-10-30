"""Dumb baselines: zero fields, and one training mean per output broadcast to every node.

    python -m plaid_surrogates.baselines --dataset Tensile2d --kind mean|zero [--submission]

Fit on the dev protocol's `train` IDs and scored on `dev` (data/splits/<dataset>.json). With
--submission, the SAME fitted predictor also writes the test-split submission (the leaderboard
calibration file), so the local->leaderboard gap measures dev-vs-test and the metric definition
only. CPU and float64 numpy only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import socket
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from plaid_surrogates.data.cache import CachedSample
from plaid_surrogates.data.dataset import CacheDataset
from plaid_surrogates.data.registry import get_spec
from plaid_surrogates.data.splits import load_splits, splits_path
from plaid_surrogates.metrics import (
    evaluate,
    evaluate_maxcomponent_diagnostic,
    reference_records,
)
from plaid_surrogates.submission import read_submission, write_split_submission
from plaid_surrogates.utils import runs
from plaid_surrogates.utils.paths import data_dir, repo_root
from plaid_surrogates.utils.provenance import write_provenance

KINDS = ("zero", "mean")


@dataclass
class ZeroBaseline:
    """Predicts 0 for every field value and every scalar (scalar RRMSE is then exactly 1)."""

    fields: tuple[str, ...]
    scalars: tuple[str, ...]

    def predict(self, s: CachedSample) -> dict[str, Any]:
        out: dict[str, Any] = {f: np.zeros(s.n_nodes) for f in self.fields}
        out.update({k: 0.0 for k in self.scalars})
        return out

    def params(self) -> dict[str, Any]:
        return {}


@dataclass
class MeanBaseline:
    """One training mean per field (pooled over all training nodes) and per scalar."""

    field_means: dict[str, float]
    scalar_means: dict[str, float]

    @classmethod
    def fit(
        cls, samples: Sequence[CachedSample], fields: Sequence[str], scalars: Sequence[str]
    ) -> MeanBaseline:
        if not samples:
            raise ValueError("no training samples")
        n_total = sum(s.n_nodes for s in samples)
        field_means = {
            f: float(sum(float(np.sum(s.output_fields[f])) for s in samples) / n_total)
            for f in fields
        }
        scalar_means = {k: float(np.mean([s.output_scalars[k] for s in samples])) for k in scalars}
        return cls(field_means, scalar_means)

    def predict(self, s: CachedSample) -> dict[str, Any]:
        out: dict[str, Any] = {f: np.full(s.n_nodes, m) for f, m in self.field_means.items()}
        out.update(self.scalar_means)
        return out

    def params(self) -> dict[str, Any]:
        return {"field_means": self.field_means, "scalar_means": self.scalar_means}


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def run_baseline(
    dataset: str, kind: str, seed: int = 0, submission: bool = False, notes: str = ""
) -> Path:
    """Fit, score on dev, optionally write the test submission; logs the run either way."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    spec = get_spec(dataset)
    fields, scalars = spec.scored_fields, spec.scored_scalars
    cache_dir = data_dir() / "cache" / dataset
    splits = load_splits(dataset, cache_dir)

    run_dir = runs.make_run_dir(dataset, "baseline", kind, seed)
    t0 = time.time()
    config = {
        "dataset": dataset,
        "model": f"baseline-{kind}",
        "seed": seed,
        "cache_dir": str(cache_dir),
        "splits_file": str(splits_path(dataset)),
        "splits_source_hash": splits["source_hash"],
        "fit_on": f"{spec.train_split} minus dev ({len(splits['train'])} samples)",
        "scored_on": f"dev ({len(splits['dev'])} samples)",
        "scored_fields": list(fields),
        "scored_scalars": list(scalars),
        "write_test_submission": submission,
    }
    log_row: dict[str, Any] = {
        "date": dt.date.today().isoformat(),
        "run id": run_dir.name,
        "dataset": dataset,
        "model": f"baseline-{kind}",
        "parent run": "—",
        "key change": "dumb baseline",
        "seed/fold": f"s{seed}",
        "machine / GPU": f"{socket.gethostname()} / CPU",
    }
    try:
        _write_json(run_dir / "config.json", config)
        prov = write_provenance(
            run_dir,
            repo_root(),
            extra={"run_id": run_dir.name, "seed": seed, "device": "cpu", "precision": "float64"},
        )

        train = CacheDataset(cache_dir, spec.train_split, ids=splits["train"], in_memory=True)
        dev = CacheDataset(cache_dir, spec.train_split, ids=splits["dev"], in_memory=True)
        train_samples, dev_samples = list(train), list(dev)
        if kind == "zero":
            model: ZeroBaseline | MeanBaseline = ZeroBaseline(fields, scalars)
        else:
            model = MeanBaseline.fit(train_samples, fields, scalars)
        _write_json(run_dir / "baseline_params.json", model.params())

        dev_file = run_dir / "predictions_dev.pkl"
        dev_records, _ = write_split_submission(
            {s.sample_id: model.predict(s) for s in dev_samples},
            cache_dir,
            splits["dev"],
            fields,
            scalars,
            dev_file,
            {"run_id": run_dir.name},
            meta_path=run_dir / "predictions_dev.meta.json",
        )
        refs = reference_records(dev_samples, fields, scalars)
        res = evaluate(refs, dev_records, fields, scalars)
        diag = evaluate_maxcomponent_diagnostic(refs, dev_records, fields, scalars)
        if evaluate(refs, read_submission(dev_file), fields, scalars) != res:
            raise RuntimeError("dev prediction file does not score like the in-memory records")
        _write_json(
            run_dir / "metrics_dev.json",
            {"split": "dev", "maxabs": res, "maxcomponent_diagnostic": diag},
        )

        if submission:
            test = CacheDataset(cache_dir, spec.test_split)
            sub_dir = run_dir / "submission"
            write_split_submission(
                {s.sample_id: model.predict(s) for s in test},
                cache_dir,
                spec.test_split,
                fields,
                scalars,
                sub_dir / f"prediction_{dataset.lower()}.pkl",
                {
                    "run_id": run_dir.name,
                    "git_commit": prov["git_commit"],
                    "git_dirty": prov["git_dirty"],
                    "config": config,
                    "dev_total_error": res["total_error"],
                    "dev_total_error_maxcomponent_diagnostic": diag["total_error"],
                    "leaderboard": spec.leaderboard,
                    "environment": prov["environment"],
                },
            )
            row = {
                "date": "<upload date>",
                "run id": run_dir.name,
                "dataset": dataset,
                "dev total_error (mean ± std if seeds/CV)": (
                    f"{res['total_error']:.4f} (max|f|); "
                    f"{diag['total_error']:.4f} if leaderboard uses max f"
                ),
                "notes": "calibration: broadcast-mean baseline",
            }
            (sub_dir / "LOG_ROW.md").write_text(runs.log_row_text(row, dataset, spec.leaderboard))
    except BaseException as e:
        (run_dir / "error.txt").write_text(traceback.format_exc())
        log_row["status"] = "killed" if isinstance(e, KeyboardInterrupt) else "failed"
        log_row["wall time"] = f"{time.time() - t0:.1f} s"
        log_row["notes"] = f"{type(e).__name__}: {e}"[:200]
        runs.append_table_row(runs.experiments_log_path(), runs.EXPERIMENT_COLUMNS, log_row)
        raise

    worst = max(res["rrmse"], key=res["rrmse"].get)
    log_row.update(
        {
            "status": "ok",
            "params": len(model.params().get("field_means", {}))
            + len(model.params().get("scalar_means", {})),
            "wall time": f"{time.time() - t0:.1f} s",
            "dev total_error": f"{res['total_error']:.4f}",
            "worst output (RRMSE)": f"{worst} ({res['rrmse'][worst]:.4f})",
            "notes": "; ".join(
                x
                for x in (
                    f"max-f diagnostic total {diag['total_error']:.4f}",
                    "test submission written" if submission else "",
                    notes,
                )
                if x
            ),
        }
    )
    runs.append_table_row(runs.experiments_log_path(), runs.EXPERIMENT_COLUMNS, log_row)

    print(f"Run: {run_dir}")
    total, total_diag = res["total_error"], diag["total_error"]
    print(f"dev total_error {total:.4f}  (max-f diagnostic {total_diag:.4f})")
    for name, v in res["rrmse"].items():
        print(f"  {name:16s} {v:.4f}   (max-f: {diag['rrmse'][name]:.4f})")
    if submission:
        print(f"Submission: {run_dir / 'submission'}  (upload it yourself; see LOG_ROW.md)")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--kind", required=True, choices=KINDS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--submission", action="store_true", help="also write the test submission")
    p.add_argument("--notes", default="")
    args = p.parse_args(argv)
    run_baseline(args.dataset, args.kind, args.seed, args.submission, args.notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

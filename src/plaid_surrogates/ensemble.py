"""Equal-weight ensemble of several runs' predictions, written as one submission (or dev file).

    python -m plaid_surrogates.ensemble --runs <run> <run> <run> --split test --checkpoint last
    python -m plaid_surrogates.ensemble --runs <run> <run> <run> --split dev  --checkpoint best

Every field and scalar is the plain mean over the members. For 2D_MultiScHypEl this keeps the
scalar consistent with the fields: effective_energy is the RVE average of psi in every member,
and an average of RVE averages is the RVE average of the averaged psi.

Members must agree on dataset, scored outputs, and sample IDs (same split, same order: checked via
the sample-ID hash each prediction file's metadata carries); the submission writer then re-checks
keys, node counts, and finiteness. `test` averages each member's submission/ file and writes a new
submission folder (its LOG_ROW.md names the target board). `dev` averages each member's dev
predictions and scores the mean with metrics.py: members must be plain dev-protocol runs (no
--fold, no --full), so they validated on the same samples. Outputs go to a fresh folder
<PLAID_RUNS_DIR>/<stamp>_<dataset>_ensemble_<tag>_k<n>_<split>/, plus one experiments/LOG.md row.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import socket
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from plaid_surrogates.config import config_from_dict
from plaid_surrogates.data.cache import load_index, split_ids
from plaid_surrogates.data.dataset import CacheDataset
from plaid_surrogates.data.registry import get_spec
from plaid_surrogates.data.splits import load_splits
from plaid_surrogates.metrics import evaluate, reference_records
from plaid_surrogates.predict import resolve_run
from plaid_surrogates.submission import read_submission, write_split_submission
from plaid_surrogates.utils import runs
from plaid_surrogates.utils.paths import data_dir, repo_root, runs_dir
from plaid_surrogates.utils.provenance import write_provenance

SPLITS = ("dev", "test")


def _ids_hash(ids: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps([int(i) for i in ids]).encode()).hexdigest()


def mean_records(members: Sequence[Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Element-wise mean of aligned record lists (same length, same keys, same array shapes)."""
    if len(members) < 2:
        raise ValueError("an ensemble needs at least 2 members")
    n = len(members[0])
    if any(len(m) != n for m in members):
        raise ValueError(f"members have different numbers of samples: {[len(m) for m in members]}")
    out = []
    for i in range(n):
        keys = list(members[0][i])
        if any(list(m[i]) != keys for m in members):
            raise ValueError(f"sample {i}: members disagree on keys")
        rec: dict[str, Any] = {}
        for k in keys:
            vals = [np.asarray(m[i][k], dtype=np.float64) for m in members]
            if any(v.shape != vals[0].shape for v in vals):
                raise ValueError(f"sample {i}: {k} shapes differ: {[v.shape for v in vals]}")
            mean = np.mean(vals, axis=0)
            rec[k] = float(mean) if mean.ndim == 0 else mean
        out.append(rec)
    return out


def _member(run: str | Path, split: str, checkpoint: str) -> dict[str, Any]:
    run_dir = resolve_run(run)
    rc = json.loads((run_dir / "config.json").read_text())
    cfg = config_from_dict(rc["config"])
    if split == "dev":
        if rc.get("fold") is not None or rc.get("full"):
            raise ValueError(f"{run_dir.name}: dev ensembles need plain dev-protocol runs")
        pred = run_dir / f"predictions_dev_{checkpoint}.pkl"
        meta = json.loads((run_dir / f"predictions_dev_{checkpoint}.meta.json").read_text())
    else:
        pred = run_dir / "submission" / f"prediction_{cfg.dataset.lower()}.pkl"
        meta = json.loads((run_dir / "submission" / "submission_meta.json").read_text())
    if not pred.is_file():
        raise FileNotFoundError(f"{pred} missing: run predict --split {split} first")
    if meta.get("checkpoint", checkpoint) != checkpoint:
        got = meta.get("checkpoint")
        raise ValueError(f"{pred} was written from {got}.pt, not {checkpoint}.pt")
    data = pred.read_bytes()
    return {
        "run_dir": run_dir,
        "dataset": cfg.dataset,
        "tag": cfg.tag,
        "file": pred,
        "sha256": hashlib.sha256(data).hexdigest(),
        "ids_sha256": meta["sample_ids_sha256"],
        "records": read_submission(pred),
    }


def ensemble_run(
    run_list: Sequence[str | Path], split: str, checkpoint: str, name: str | None = None
) -> Path:
    """Build the ensemble; returns the written prediction file."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    members = [_member(r, split, checkpoint) for r in run_list]
    dirs = [m["run_dir"] for m in members]
    if len(set(dirs)) != len(dirs):
        raise ValueError("the same run is listed twice")
    dataset = members[0]["dataset"]
    if any(m["dataset"] != dataset for m in members):
        raise ValueError("members come from different datasets")
    spec = get_spec(dataset)
    fields, scalars = spec.scored_fields, spec.scored_scalars
    cache_dir = data_dir() / "cache" / dataset
    ids = load_splits(dataset, cache_dir)["dev"] if split == "dev" else None
    target = spec.test_split if split == "test" else ids
    want_hash = _ids_hash(ids) if split == "dev" else None
    hashes = {m["ids_sha256"] for m in members}
    if len(hashes) != 1 or (want_hash is not None and hashes != {want_hash}):
        raise ValueError("members predicted different samples or orders (sample-ID hashes differ)")
    tags = {m["tag"] for m in members}
    tag = name or (tags.pop() if len(tags) == 1 else None)
    if not tag or "_" in tag or "/" in tag:
        raise ValueError(f"members have tags {sorted(tags)}: pass --name (no '_' or '/')")

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = runs_dir() / f"{stamp}_{dataset}_ensemble_{tag}_k{len(members)}_{split}"
    out_dir.mkdir(parents=True, exist_ok=False)
    write_provenance(
        out_dir,
        repo_root(),
        extra={"run_id": out_dir.name, "members": [m["run_dir"].name for m in members]},
    )

    mean = mean_records([m["records"] for m in members])
    order = split_ids(load_index(cache_dir), target) if isinstance(target, str) else target
    preds_by_id = dict(zip(order, mean, strict=True))
    member_info = [
        {"run_id": m["run_dir"].name, "file": m["file"].name, "sha256": m["sha256"]}
        for m in members
    ]
    meta = {
        "run_id": out_dir.name,
        "method": "equal-weight mean of members' predictions",
        "checkpoint": checkpoint,
        "members": member_info,
        "leaderboard": spec.leaderboard,
    }
    log_row: dict[str, Any] = {
        "date": dt.date.today().isoformat(),
        "run id": out_dir.name,
        "status": "ok",
        "dataset": dataset,
        "model": f"ensemble-{tag}",
        "parent run": ", ".join(m["run_dir"].name for m in members),
        "key change": f"equal-weight mean of {len(members)} runs ({checkpoint}.pt)",
        "seed/fold": "—",
        "machine / GPU": f"{socket.gethostname()} / CPU",
    }
    if split == "test":
        out = out_dir / "submission" / f"prediction_{dataset.lower()}.pkl"
        write_split_submission(preds_by_id, cache_dir, target, fields, scalars, out, meta)
        row = {
            "date": "<upload date>",
            "run id": out_dir.name,
            "dataset": dataset,
            "dev total_error (mean ± std if seeds/CV)": f"— (ensemble of {len(members)} runs)",
            "notes": f"ensemble of {', '.join(m['run_dir'].name for m in members)}",
        }
        (out_dir / "submission" / "LOG_ROW.md").write_text(
            runs.log_row_text(row, dataset, spec.leaderboard)
        )
        log_row.update(
            {"dev total_error": "— (test-set ensemble)", "notes": "submission file written"}
        )
        print(f"Submission: {out}  ({len(mean)} samples)\nUpload to: {spec.leaderboard}")
    else:
        out = out_dir / f"predictions_dev_{checkpoint}.pkl"
        write_split_submission(
            preds_by_id,
            cache_dir,
            ids,
            fields,
            scalars,
            out,
            meta,
            out_dir / f"predictions_dev_{checkpoint}.meta.json",
        )
        samples = list(CacheDataset(cache_dir, spec.train_split, ids=ids, in_memory=True))
        refs = reference_records(samples, fields, scalars)
        res = evaluate(refs, read_submission(out), fields, scalars)
        singles = {
            m["run_dir"].name: evaluate(refs, m["records"], fields, scalars)["total_error"]
            for m in members
        }
        (out_dir / "metrics_dev.json").write_text(
            json.dumps({"split": "dev", "ensemble": res, "members": singles}, indent=2) + "\n"
        )
        worst = max(res["rrmse"], key=res["rrmse"].get)
        log_row.update(
            {
                "dev total_error": f"{res['total_error']:.4f}",
                "worst output (RRMSE)": f"{worst} ({res['rrmse'][worst]:.4f})",
                "notes": "members: " + ", ".join(f"{v:.4f}" for v in singles.values()),
            }
        )
        print(f"dev ensemble total_error {res['total_error']:.6f}; members {singles}")
        for k, v in res["rrmse"].items():
            print(f"  {k:16s} {v:.4f}")
    runs.append_table_row(runs.experiments_log_path(), runs.EXPERIMENT_COLUMNS, log_row)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="member run folders or run ids")
    p.add_argument("--split", choices=SPLITS, required=True)
    p.add_argument("--checkpoint", choices=("best", "last"), required=True)
    p.add_argument("--name", default=None, help="tag for the output folder (default: members' tag)")
    args = p.parse_args(argv)
    ensemble_run(args.runs, args.split, args.checkpoint, args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Scalar physics identities, checked on the training data before any use (docs/methods.md §4).

    python -m plaid_surrogates.identities --dataset Tensile2d [--run <run dir|id>] [--overwrite]

For each scored scalar, several candidate formulas compute it from the reference fields, nodal
tags and input scalars. For every candidate the report gives the per-sample relative error on all
of `train_500` (an exact identity has nothing to fit, so this leaks nothing) and the scalar RRMSE
on the dev split, next to the training-mean predictor. A candidate passes if its largest relative
error is <= 1e-3. The plane-strain von Mises candidate has one parameter, Poisson's ratio, fitted
on the dev protocol's `train` IDs only (never on dev).

With --run, the candidates are also applied to that run's predicted dev fields
(predictions_dev_best.pkl), which shows the scalar error to expect with imperfect fields.
Reads only the cache; writes <PLAID_DATA_DIR>/audit/<dataset>/identities.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from plaid_surrogates.data.cache import CachedSample, load_index, load_sample, split_ids
from plaid_surrogates.data.inspect import integrate_p1
from plaid_surrogates.data.registry import get_spec
from plaid_surrogates.data.splits import load_splits
from plaid_surrogates.metrics import scalar_rrmse
from plaid_surrogates.submission import read_submission
from plaid_surrogates.utils.paths import data_dir, runs_dir

PASS_REL = 1e-3


def von_mises_plane_stress(s11: np.ndarray, s22: np.ndarray, s12: np.ndarray) -> np.ndarray:
    """In-plane von Mises stress with sigma33 = 0."""
    return np.sqrt(s11**2 - s11 * s22 + s22**2 + 3.0 * s12**2)


def von_mises_plane_strain(
    s11: np.ndarray, s22: np.ndarray, s12: np.ndarray, nu: float
) -> np.ndarray:
    """Von Mises stress with the linear-elastic plane-strain sigma33 = nu (sigma11 + sigma22)."""
    s33 = nu * (s11 + s22)
    return np.sqrt(0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2) + 3.0 * s12**2)


# Candidate(fields, sample, nu) -> scalar. `fields` may be predicted; everything else (mesh, tags,
# input scalars) comes from the cached sample. `nu` is only used by fitted candidates.
Candidate = Callable[[Mapping[str, np.ndarray], CachedSample, float], float]


def on_top(f: Mapping[str, np.ndarray], s: CachedSample, name: str) -> np.ndarray:
    """Values of field `name` on the sample's `Top` tag nodes."""
    return f[name][s.nodal_tags["Top"]]


def rve_average(field: np.ndarray, s: CachedSample) -> float:
    """Integral of the P1 field over the mesh divided by the bounding-box (RVE) area."""
    lo, hi = s.points.min(axis=0), s.points.max(axis=0)
    return integrate_p1(field, s.points, s.cells) / float(np.prod(hi - lo))


CANDIDATES: dict[str, dict[str, dict[str, Candidate]]] = {
    "Tensile2d": {
        "max_U2_top": {
            "max(U2) over Top": lambda f, s, nu: float(np.max(on_top(f, s, "U2"))),
            "max(U2) over all nodes": lambda f, s, nu: float(np.max(f["U2"])),
            "max(|U2|) over Top": lambda f, s, nu: float(np.max(np.abs(on_top(f, s, "U2")))),
        },
        "max_sig22_top": {
            "max(sig22) over Top": lambda f, s, nu: float(np.max(on_top(f, s, "sig22"))),
            "-P": lambda f, s, nu: -float(s.input_scalars["P"]),
            "mean(sig22) over Top": lambda f, s, nu: float(np.mean(on_top(f, s, "sig22"))),
        },
        "max_von_mises": {
            "max VM over nodes, plane stress": lambda f, s, nu: float(
                np.max(von_mises_plane_stress(f["sig11"], f["sig22"], f["sig12"]))
            ),
            "max VM over nodes, plane strain (nu fitted)": lambda f, s, nu: float(
                np.max(von_mises_plane_strain(f["sig11"], f["sig22"], f["sig12"], nu))
            ),
        },
    },
    "2D_MultiScHypEl": {
        "effective_energy": {
            # psi = 0 in the voids, so the homogenized energy is the average over the whole RVE
            "RVE average of psi": lambda f, s, nu: rve_average(f["psi"], s),
        },
    },
}
FITTED = {"max VM over nodes, plane strain (nu fitted)"}


def _apply(cand: Candidate, items: Sequence[tuple], nu: float) -> np.ndarray:
    """items: (fields, sample) pairs."""
    return np.array([cand(f, s, nu) for f, s in items])


def fit_nu(cand: Candidate, items: Sequence[tuple], refs: np.ndarray) -> float:
    """Poisson's ratio in [0, 0.5] minimizing the scalar RRMSE: 0.01 grid, then 0.0005 around it."""

    def loss(nu: float) -> float:
        return scalar_rrmse(list(refs), list(_apply(cand, items, nu)))

    coarse = np.linspace(0.0, 0.5, 51)
    best = float(coarse[np.argmin([loss(nu) for nu in coarse])])
    fine = np.clip(np.arange(best - 0.01, best + 0.01 + 1e-12, 0.0005), 0.0, 0.5)
    return float(fine[np.argmin([loss(nu) for nu in fine])])


def _stats(rel: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(rel)),
        "p95": float(np.quantile(rel, 0.95)),
        "max": float(rel.max()),
    }


def run_identity_check(cache_dir: Path, dataset: str, run_dir: Path | None = None) -> dict:
    """Score every candidate on the reference fields (and, optionally, a run's predicted fields)."""
    if dataset not in CANDIDATES:
        raise ValueError(f"no identity candidates for {dataset}; known: {sorted(CANDIDATES)}")
    spec = get_spec(dataset)
    index = load_index(cache_dir)
    splits = load_splits(dataset, cache_dir)
    all_ids = split_ids(index, spec.train_split)
    samples = {i: load_sample(cache_dir / index["samples"][str(i)]["file"]) for i in all_ids}
    items = {i: (s.output_fields, s) for i, s in samples.items()}
    dev, train = splits["dev"], splits["train"]

    out: dict[str, Any] = {
        "dataset": dataset,
        "hf_revision": index.get("hf_revision"),
        "n_all": len(all_ids),
        "n_fit": len(train),
        "n_dev": len(dev),
        "pass_rel": PASS_REL,
        "scalars": {},
    }
    nus: dict[str, float] = {}
    for scalar, cands in CANDIDATES[dataset].items():
        ref = {i: samples[i].output_scalars[scalar] for i in all_ids}
        ref_dev = [ref[i] for i in dev]
        train_mean = float(np.mean([ref[i] for i in train]))
        block: dict[str, Any] = {
            "training_mean_dev_rrmse": scalar_rrmse(ref_dev, [train_mean] * len(dev)),
            "candidates": {},
        }
        for name, cand in cands.items():
            nu = 0.0
            if name in FITTED:
                nu = fit_nu(cand, [items[i] for i in train], np.array([ref[i] for i in train]))
                nus[name] = nu
            pred = dict(zip(all_ids, _apply(cand, [items[i] for i in all_ids], nu), strict=True))
            rel = np.array([abs(pred[i] - ref[i]) / abs(ref[i]) for i in all_ids])
            block["candidates"][name] = {
                "rel_error_all": _stats(rel),
                "passes": bool(rel.max() <= PASS_REL),
                "dev_rrmse": scalar_rrmse(ref_dev, [pred[i] for i in dev]),
                **({"nu": nu} if name in FITTED else {}),
            }
        out["scalars"][scalar] = block

    if run_dir is not None:
        recs = read_submission(run_dir / "predictions_dev_best.pkl")
        if len(recs) != len(dev):
            raise ValueError(f"{len(recs)} predicted records, but the dev split has {len(dev)}")
        pred_items = [(r, samples[i]) for r, i in zip(recs, dev, strict=True)]
        run_block: dict[str, Any] = {"run_id": run_dir.name, "scalars": {}}
        for scalar, cands in CANDIDATES[dataset].items():
            ref_dev = [samples[i].output_scalars[scalar] for i in dev]
            run_block["scalars"][scalar] = {
                "as_predicted_by_run": scalar_rrmse(ref_dev, [r[scalar] for r in recs]),
                "from_predicted_fields": {
                    name: scalar_rrmse(ref_dev, list(_apply(cand, pred_items, nus.get(name, 0.0))))
                    for name, cand in cands.items()
                },
            }
        out["run"] = run_block
    return out


def render_markdown(r: dict) -> str:
    """Readable summary of `run_identity_check`."""
    lines = [
        f"# Scalar identities: {r['dataset']} (HF {str(r['hf_revision'])[:10]})",
        "",
        f"Relative error over all {r['n_all']} training samples (median / p95 / max); pass if "
        f"max <= {r['pass_rel']:g}. Dev RRMSE over {r['n_dev']} dev samples. Fitted parameters "
        f"use the {r['n_fit']} dev-protocol training samples only.",
    ]
    for scalar, b in r["scalars"].items():
        lines += [
            "",
            f"## {scalar}",
            "",
            f"Training mean, dev RRMSE: {b['training_mean_dev_rrmse']:.4f}",
            "",
            "| candidate | rel error median / p95 / max | passes | dev RRMSE |",
            "|---|---|---|---|",
        ]
        for name, c in b["candidates"].items():
            e = c["rel_error_all"]
            label = f"{name} (nu = {c['nu']:.4f})" if "nu" in c else name
            lines.append(
                f"| {label} | {e['median']:.3g} / {e['p95']:.3g} / {e['max']:.3g} | "
                f"{'yes' if c['passes'] else 'no'} | {c['dev_rrmse']:.4f} |"
            )
    if "run" in r:
        lines += ["", f"## Applied to predicted dev fields of {r['run']['run_id']}", ""]
        lines += ["| scalar | candidate | dev RRMSE |", "|---|---|---|"]
        for scalar, b in r["run"]["scalars"].items():
            lines.append(f"| {scalar} | as predicted by the run | {b['as_predicted_by_run']:.4f} |")
            for name, v in b["from_predicted_fields"].items():
                lines.append(f"| {scalar} | {name} | {v:.4f} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(CANDIDATES))
    p.add_argument("--run", default=None, help="run folder or id whose predicted dev fields to use")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    run_dir = None
    if args.run:
        run_dir = next((q for q in (Path(args.run), runs_dir() / args.run) if q.is_dir()), None)
        if run_dir is None:
            print(f"no run folder {args.run!r}", file=sys.stderr)
            return 1
    out = data_dir() / "audit" / args.dataset
    if (out / "identities.json").exists() and not args.overwrite:
        print(f"{out / 'identities.json'} exists; pass --overwrite to replace it.", file=sys.stderr)
        return 1
    report = run_identity_check(data_dir() / "cache" / args.dataset, args.dataset, run_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "identities.json").write_text(json.dumps(report, indent=1) + "\n")
    md = render_markdown(report)
    (out / "identities.md").write_text(md)
    print(md)
    print(f"Wrote {out / 'identities.json'} and identities.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

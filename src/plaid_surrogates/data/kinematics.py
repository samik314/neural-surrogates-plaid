"""Displacement kinematics check: is u the imposed affine deformation plus a periodic
fluctuation, and what is its offset?

    python -m plaid_surrogates.data.kinematics --dataset 2D_MultiScHypEl [--overwrite]

The data audit found max|u| ~ 1 in every sample on a unit RVE whose imposed stretch is <= ~10 %.
Per sample this fits u ~ A X + c by least squares (A: 2x2 displacement gradient, c: offset) and
reports: the offset |c|; the affine part's size max|A X| over the RVE; the residual (fluctuation)
max|u - A X - c|; whether (I + A)^T (I + A) reproduces the input C (so the affine part IS the
imposed macroscopic deformation); and the nodes with |u| ~ 0 (a pinned node fixes the offset).
Boundary check (KUBC: u = (F - I) X on the outer boundary): the same fit on the outer-boundary
nodes only (tag `Ext_bound`), its boundary residual, and the scale s for which
(I + A/s)^T (I + A/s) (or (I + A/s)(I + A/s)^T) best reproduces the input C, testing whether u is
stored in other units than X.
Also: effective_energy dev RRMSE of the training mean and of the RVE-area average of psi.
Reads only the cache; writes <PLAID_DATA_DIR>/audit/<dataset>/kinematics.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from plaid_surrogates.data.cache import CachedSample, load_index, load_sample, split_ids
from plaid_surrogates.data.inspect import integrate_p1
from plaid_surrogates.data.registry import get_spec
from plaid_surrogates.data.splits import load_splits
from plaid_surrogates.metrics import scalar_rrmse
from plaid_surrogates.utils.paths import data_dir


def fit_affine(points: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares u ~ A X + c. Returns (A (2, 2), c (2,), residual (N, 2))."""
    design = np.column_stack([points, np.ones(len(points))])
    coef, *_ = np.linalg.lstsq(design, u, rcond=None)  # (3, 2): rows x, y, 1
    a = coef[:2].T  # u_i = sum_j A_ij X_j + c_i
    c = coef[2]
    return a, c, u - design @ coef


def cauchy_green(a: np.ndarray) -> np.ndarray:
    """(C11, C12, C22) of F = I + A."""
    f = np.eye(2) + a
    cg = f.T @ f
    return np.array([cg[0, 0], cg[0, 1], cg[1, 1]])


SCALES = np.geomspace(0.1, 100.0, 301)


def best_scale(a: np.ndarray, c_in: np.ndarray) -> dict[str, float]:
    """Scale s minimizing max|C(I + A/s) - C_in| for C = F^T F and for B = F F^T (grid search)."""
    out: dict[str, float] = {}
    for name, fn in (("FtF", cauchy_green), ("FFt", lambda m: cauchy_green(m.T))):
        errs = [float(np.abs(fn(a / s) - c_in).max()) for s in SCALES]
        k = int(np.argmin(errs))
        out[f"{name}_scale"] = float(SCALES[k])
        out[f"{name}_err"] = errs[k]
    return out


def sample_kinematics(s: CachedSample, boundary_tag: str = "Ext_bound") -> dict[str, Any]:
    """Affine fit of one sample's displacement field."""
    u = np.column_stack([s.output_fields["u1"], s.output_fields["u2"]])
    a, c, res = fit_affine(s.points, u)
    c_in = np.array([s.input_scalars[k] for k in ("C11", "C12", "C22")])
    norm_u = np.linalg.norm(u, axis=1)
    k0 = int(np.argmin(norm_u))
    bnd: dict[str, Any] = {}
    if boundary_tag in s.nodal_tags:
        idx = s.nodal_tags[boundary_tag]
        a_b, c_b, res_b = fit_affine(s.points[idx], u[idx])
        bnd = {
            "bnd_fluct_max": float(np.linalg.norm(res_b, axis=1).max()),
            "bnd_offset_norm": float(np.linalg.norm(c_b)),
            "bnd_C_fit_err": float(np.abs(cauchy_green(a_b) - c_in).max()),
            **{f"bnd_{k}": v for k, v in best_scale(a_b, c_in).items()},
        }
    return {
        **bnd,
        "max_abs_u": float(np.abs(u).max()),
        "offset_norm": float(np.linalg.norm(c)),
        "offset": [float(c[0]), float(c[1])],
        "affine_max": float(np.linalg.norm(s.points @ a.T, axis=1).max()),
        "fluct_max": float(np.linalg.norm(res, axis=1).max()),
        "fluct_rms": float(np.sqrt(np.mean(np.sum(res**2, axis=1)))),
        "C_fit_err": float(np.abs(cauchy_green(a) - c_in).max()),
        "min_norm_u": float(norm_u[k0]),
        "min_norm_u_at": [float(s.points[k0, 0]), float(s.points[k0, 1])],
        "n_nodes_u_zero": int((norm_u <= 1e-12 * max(norm_u.max(), 1e-300)).sum()),
    }


def _q(v: list[float]) -> dict[str, float]:
    a = np.asarray(v, float)
    return {
        "min": float(a.min()),
        "median": float(np.median(a)),
        "p95": float(np.quantile(a, 0.95)),
        "max": float(a.max()),
    }


def run_check(cache_dir: Path, dataset: str) -> dict[str, Any]:
    """Kinematics over every training sample, plus the effective_energy reference numbers."""
    spec = get_spec(dataset)
    index = load_index(cache_dir)
    ids = split_ids(index, spec.train_split)
    samples = {i: load_sample(cache_dir / index["samples"][str(i)]["file"]) for i in ids}
    recs = [sample_kinematics(samples[i]) for i in ids]
    keys = ["max_abs_u", "offset_norm", "affine_max", "fluct_max", "fluct_rms", "C_fit_err",
            "min_norm_u"]
    keys += [k for k in recs[0] if k.startswith("bnd_")]
    out: dict[str, Any] = {
        "dataset": dataset,
        "n_samples": len(ids),
        "quantiles": {k: _q([r[k] for r in recs]) for k in keys},
        "offset_components": {
            "c1": _q([r["offset"][0] for r in recs]),
            "c2": _q([r["offset"][1] for r in recs]),
        },
        "samples_with_a_node_at_u_zero": sum(r["n_nodes_u_zero"] > 0 for r in recs),
        "min_norm_u_location_examples": [r["min_norm_u_at"] for r in recs[:5]],
    }
    splits = load_splits(dataset, cache_dir)
    e = {i: samples[i].output_scalars["effective_energy"] for i in ids}
    ref_dev = [e[i] for i in splits["dev"]]
    mean_train = float(np.mean([e[i] for i in splits["train"]]))
    psi_avg = {}
    for i in splits["dev"]:
        s = samples[i]
        lo, hi = s.points.min(axis=0), s.points.max(axis=0)
        rve_area = float(np.prod(hi - lo))
        psi_avg[i] = integrate_p1(s.output_fields["psi"], s.points, s.cells) / rve_area
    out["effective_energy_dev_rrmse"] = {
        "training mean": scalar_rrmse(ref_dev, [mean_train] * len(ref_dev)),
        "RVE average of psi (reference psi)": scalar_rrmse(
            ref_dev, [psi_avg[i] for i in splits["dev"]]
        ),
    }
    return out


def render_markdown(r: dict[str, Any]) -> str:
    lines = [
        f"# Displacement kinematics: {r['dataset']} ({r['n_samples']} training samples)",
        "",
        "Fit per sample: u ~ A X + c. Quantiles are min / median / p95 / max.",
        "",
        "| quantity | min | median | p95 | max |",
        "|---|---|---|---|---|",
    ]
    labels = {
        "max_abs_u": "max |u| (component)",
        "offset_norm": "|c| (offset)",
        "affine_max": "max |A X| over the RVE",
        "fluct_max": "max |u - A X - c| (fluctuation)",
        "fluct_rms": "rms fluctuation",
        "C_fit_err": "max |C((I+A)) - C_input|",
        "min_norm_u": "min over nodes of |u|",
        "bnd_fluct_max": "boundary-only fit: max residual on the boundary",
        "bnd_offset_norm": "boundary-only fit: |c|",
        "bnd_C_fit_err": "boundary-only fit: max |C(I+A) - C_input|",
        "bnd_FtF_scale": "best s for C = (I+A/s)^T (I+A/s)",
        "bnd_FtF_err": "  ... max |C - C_input| at that s",
        "bnd_FFt_scale": "best s for C = (I+A/s) (I+A/s)^T",
        "bnd_FFt_err": "  ... max |C - C_input| at that s",
    }
    def row(label: str, q: dict[str, float]) -> str:
        cells = " | ".join(f"{q[x]:.4g}" for x in ("min", "median", "p95", "max"))
        return f"| {label} | {cells} |"

    for k, label in labels.items():
        if k in r["quantiles"]:
            lines.append(row(label, r["quantiles"][k]))
    for k, q in r["offset_components"].items():
        lines.append(row(f"offset {k}", q))
    lines += [
        "",
        f"Samples with a node at u = 0: {r['samples_with_a_node_at_u_zero']} of {r['n_samples']}",
        f"Node of smallest |u|, first 5 samples: {r['min_norm_u_location_examples']}",
        "",
        "## effective_energy, dev RRMSE",
        "",
    ]
    lines += [f"- {k}: {v:.4f}" for k, v in r["effective_energy_dev_rrmse"].items()]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    out = data_dir() / "audit" / args.dataset
    if (out / "kinematics.json").exists() and not args.overwrite:
        print(f"{out / 'kinematics.json'} exists; pass --overwrite to replace it.", file=sys.stderr)
        return 1
    r = run_check(data_dir() / "cache" / args.dataset, args.dataset)
    out.mkdir(parents=True, exist_ok=True)
    (out / "kinematics.json").write_text(json.dumps(r, indent=1) + "\n")
    md = render_markdown(r)
    (out / "kinematics.md").write_text(md)
    print(md)
    print(f"Wrote {out / 'kinematics.json'} and kinematics.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

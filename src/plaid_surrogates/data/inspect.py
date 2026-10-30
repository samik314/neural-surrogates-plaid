"""Data audit (docs/methods.md §1): what the converted cache actually contains, before modeling.

    python -m plaid_surrogates.data.inspect --dataset 2D_MultiScHypEl [--overwrite]

Reads only the cache (never imports plaid). Writes <PLAID_DATA_DIR>/audit/<dataset>/audit.json
and audit.md (outside git) and prints audit.md. Per dataset it reports:
- mesh sizes, element orientation, the RVE bounding box (same for every sample or not);
- free-edge loops: interior loops = holes when the holes are voids, checked against the
  inclusion count from the `*_nb_inc_<k>` splits; meshed area vs bounding-box area;
- periodicity: do opposite bounding-box edges carry matching nodes;
- nodal tags: names, presence, and where their nodes sit (outer edges, hole loops, interior);
- per-sample max|f| of every scored field and |s| of every scored scalar (near-zero flags);
- train vs test distributions (input scalars, node counts, inclusion counts; KS statistic);
- when psi and effective_energy exist: effective_energy vs the area average of psi over the
  RVE and over the solid (diagnostic only; nothing here is used by the model).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from plaid_surrogates.data.cache import CachedSample, load_index, load_sample, split_ids
from plaid_surrogates.data.registry import get_spec
from plaid_surrogates.data.splits import inclusion_counts, ks_statistic
from plaid_surrogates.utils.paths import data_dir

NEAR_ZERO_REL = 1e-2  # flag a per-sample max below 1% of the median over samples


# --- geometry helpers (pure, unit-tested)


def triangle_areas(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Signed triangle areas (positive = counter-clockwise)."""
    p0, p1, p2 = points[cells[:, 0]], points[cells[:, 1]], points[cells[:, 2]]
    return 0.5 * (
        (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
        - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    )


def integrate_p1(field: np.ndarray, points: np.ndarray, cells: np.ndarray) -> float:
    """Exact integral of the piecewise-linear interpolant of a nodal field over the mesh."""
    areas = np.abs(triangle_areas(points, cells))
    return float(np.sum(areas * field[cells].mean(axis=1)))


def boundary_edges(cells: np.ndarray) -> np.ndarray:
    """Edges used by exactly one triangle, as sorted node pairs (E, 2)."""
    e = np.sort(np.concatenate([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    if (counts > 2).any():
        raise ValueError("non-manifold mesh: an edge is shared by more than two triangles")
    return uniq[counts == 1]


def edge_loops(edges: np.ndarray) -> list[np.ndarray]:
    """Node sets of the connected components of a set of edges (union-find)."""
    parent: dict[int, int] = {}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges.tolist():
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups: dict[int, list[int]] = defaultdict(list)
    for node in parent:
        groups[find(node)].append(node)
    return [np.array(sorted(g), dtype=np.int64) for g in groups.values()]


def on_bbox(points: np.ndarray, tol: float) -> dict[str, np.ndarray]:
    """Boolean masks of nodes on each side of the bounding box."""
    lo, hi = points.min(axis=0), points.max(axis=0)
    return {
        "left": np.abs(points[:, 0] - lo[0]) <= tol,
        "right": np.abs(points[:, 0] - hi[0]) <= tol,
        "bottom": np.abs(points[:, 1] - lo[1]) <= tol,
        "top": np.abs(points[:, 1] - hi[1]) <= tol,
    }


def periodic_match(points: np.ndarray, tol: float) -> dict[str, bool]:
    """Do left/right (and bottom/top) edges carry nodes at the same y (x) positions?"""
    side = on_bbox(points, tol)

    def same(a: np.ndarray, b: np.ndarray) -> bool:
        return len(a) == len(b) and bool(np.all(np.abs(np.sort(a) - np.sort(b)) <= tol))

    return {
        "x": same(points[side["left"], 1], points[side["right"], 1]),
        "y": same(points[side["bottom"], 0], points[side["top"], 0]),
    }


# --- per-sample record


def sample_record(s: CachedSample, fields: Sequence[str], scalars: Sequence[str]) -> dict:
    """Everything the audit needs from one sample (small, JSON-friendly)."""
    pts, cells = s.points, s.cells
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    size = float(np.max(hi - lo))
    tol = 1e-8 * size
    areas = triangle_areas(pts, cells)
    bbox_area = float(np.prod(hi - lo))
    meshed = float(np.abs(areas).sum())
    side = on_bbox(pts, tol)
    on_outer = side["left"] | side["right"] | side["bottom"] | side["top"]
    loops = edge_loops(boundary_edges(cells))
    interior = [lp for lp in loops if not on_outer[lp].any()]
    hole_nodes = np.zeros(len(pts), dtype=bool)
    for lp in interior:
        hole_nodes[lp] = True

    tags = {}
    for name, idx in s.nodal_tags.items():
        n = max(len(idx), 1)
        tags[name] = {
            "n": int(len(idx)),
            **{k: float(m[idx].sum() / n) for k, m in side.items()},
            "hole": float(hole_nodes[idx].sum() / n),
            "interior": float((~on_outer[idx] & ~hole_nodes[idx]).sum() / n),
        }

    rec: dict[str, Any] = {
        "id": s.sample_id,
        "n_nodes": s.n_nodes,
        "n_cells": int(len(cells)),
        "bbox": [float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])],
        "n_inverted": int((areas < 0).sum()),
        "n_degenerate": int((np.abs(areas) <= 1e-14 * bbox_area).sum()),
        "n_loops": len(loops),
        "n_interior_loops": len(interior),
        "void_fraction": 1.0 - meshed / bbox_area,
        "periodic": periodic_match(pts, tol),
        "tags": tags,
        "inputs": dict(s.input_scalars),
    }
    if s.has_outputs:
        rec["field_maxabs"] = {f: float(np.max(np.abs(s.output_fields[f]))) for f in fields}
        rec["scalar_abs"] = {k: float(abs(s.output_scalars[k])) for k in scalars}
        rec["scalar"] = {k: float(s.output_scalars[k]) for k in scalars}
        if "psi" in s.output_fields and "effective_energy" in s.output_scalars:
            integral = integrate_p1(s.output_fields["psi"], pts, cells)
            rec["psi_integral"] = integral
            rec["psi_avg_rve"] = integral / bbox_area
            rec["psi_avg_solid"] = integral / meshed
    return rec


# --- aggregation


def _q(values: Sequence[float]) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {}
    return {
        "min": float(v.min()),
        "p05": float(np.quantile(v, 0.05)),
        "median": float(np.median(v)),
        "max": float(v.max()),
    }


def run_audit(cache_dir: Path, dataset: str) -> dict[str, Any]:
    """Audit every sample in the cache; returns a JSON-friendly summary."""
    spec = get_spec(dataset)
    index = load_index(cache_dir)
    train_ids = split_ids(index, spec.train_split)
    test_ids = split_ids(index, spec.test_split)
    assigned = set(train_ids) | set(test_ids)
    all_ids = [int(i) for i in index["samples"]]
    n_inc = inclusion_counts(index)
    fields, scalars = spec.scored_fields, spec.scored_scalars

    recs = {}
    for i in all_ids:
        s = load_sample(cache_dir / index["samples"][str(i)]["file"])
        recs[i] = sample_record(s, fields, scalars)
    tr = [recs[i] for i in train_ids]
    te = [recs[i] for i in test_ids]
    out_recs = [r for r in recs.values() if "field_maxabs" in r]

    bboxes = np.array([r["bbox"] for r in recs.values()])
    audit: dict[str, Any] = {
        "dataset": dataset,
        "hf_revision": index.get("hf_revision"),
        "n_samples": len(all_ids),
        "splits": {name: len(ids) for name, ids in index["splits"].items()},
        "unassigned_samples": [i for i in all_ids if i not in assigned],
        "mesh": {
            "n_nodes": {
                "train": _q([r["n_nodes"] for r in tr]),
                "test": _q([r["n_nodes"] for r in te]),
            },
            "n_cells": _q([r["n_cells"] for r in recs.values()]),
            "bbox_min_max": {
                "xmin": _q(bboxes[:, 0]), "xmax": _q(bboxes[:, 1]),
                "ymin": _q(bboxes[:, 2]), "ymax": _q(bboxes[:, 3]),
            },
            "same_bbox_all_samples": bool(np.ptp(bboxes, axis=0).max() <= 1e-8),
            "samples_with_inverted_cells": sum(r["n_inverted"] > 0 for r in recs.values()),
            "samples_with_degenerate_cells": sum(r["n_degenerate"] > 0 for r in recs.values()),
        },
        "topology": {
            "n_loops": _q([r["n_loops"] for r in recs.values()]),
            "n_interior_loops": _q([r["n_interior_loops"] for r in recs.values()]),
            "void_fraction": _q([r["void_fraction"] for r in recs.values()]),
        },
        "periodic_fraction": {
            ax: float(np.mean([r["periodic"][ax] for r in recs.values()])) for ax in ("x", "y")
        },
    }

    if n_inc:
        pairs: dict[str, int] = defaultdict(int)
        for i, k in n_inc.items():
            pairs[f"n_inc={k}, interior_loops={recs[i]['n_interior_loops']}"] += 1
        audit["topology"]["inclusions_vs_interior_loops"] = dict(sorted(pairs.items()))
        audit["topology"]["fraction_interior_loops_equal_n_inc"] = float(
            np.mean([recs[i]["n_interior_loops"] == k for i, k in n_inc.items()])
        )
        audit["topology"]["n_inc_counts"] = {
            "train": dict(sorted(_count([n_inc[i] for i in train_ids if i in n_inc]).items())),
            "test": dict(sorted(_count([n_inc[i] for i in test_ids if i in n_inc]).items())),
        }

    tag_names = sorted({t for r in recs.values() for t in r["tags"]})
    audit["tags"] = {
        t: {
            "present_fraction": float(np.mean([t in r["tags"] for r in recs.values()])),
            "n_nodes": _q([r["tags"][t]["n"] for r in recs.values() if t in r["tags"]]),
            "mean_location_fraction": {
                loc: float(np.mean([r["tags"][t][loc] for r in recs.values() if t in r["tags"]]))
                for loc in ("left", "right", "bottom", "top", "hole", "interior")
            },
        }
        for t in tag_names
    }

    flags: list[str] = []
    audit["fields_maxabs"] = {}
    for f in fields:
        q = _q([r["field_maxabs"][f] for r in out_recs])
        audit["fields_maxabs"][f] = q
        if q and q["min"] < NEAR_ZERO_REL * q["median"]:
            flags.append(f"field {f}: smallest per-sample max|f| {q['min']:.3g} < 1% of median")
    audit["scalars_abs"] = {}
    for k in scalars:
        q = _q([r["scalar_abs"][k] for r in out_recs])
        audit["scalars_abs"][k] = q
        if q and q["min"] < NEAR_ZERO_REL * q["median"]:
            flags.append(f"scalar {k}: smallest |s| {q['min']:.3g} < 1% of median")
        neg = sum(r["scalar"][k] < 0 for r in out_recs)
        if neg:
            flags.append(f"scalar {k}: {neg} negative values")
    audit["near_zero_flags"] = flags

    if any("psi_integral" in r for r in out_recs):
        e = np.array([r["scalar"]["effective_energy"] for r in out_recs])
        for key in ("psi_avg_rve", "psi_avg_solid"):
            est = np.array([r[key] for r in out_recs])
            audit.setdefault("energy_identity_rel_error", {})[key] = _q(np.abs(est - e) / np.abs(e))

    audit["train_vs_test"] = {
        "inputs": {
            name: {
                "train": _q([r["inputs"][name] for r in tr]),
                "test": _q([r["inputs"][name] for r in te]),
                "ks": ks_statistic(
                    np.array([r["inputs"][name] for r in tr]),
                    np.array([r["inputs"][name] for r in te]),
                ),
            }
            for name in spec.input_scalars
        },
        "n_nodes_ks": ks_statistic(
            np.array([r["n_nodes"] for r in tr], float), np.array([r["n_nodes"] for r in te], float)
        ),
    }
    if n_inc:
        audit["train_vs_test"]["n_inc_ks"] = ks_statistic(
            np.array([n_inc[i] for i in train_ids if i in n_inc], float),
            np.array([n_inc[i] for i in test_ids if i in n_inc], float),
        )
    return audit


def _count(values: Sequence[int]) -> dict[int, int]:
    out: dict[int, int] = defaultdict(int)
    for v in values:
        out[int(v)] += 1
    return dict(out)


def _fmt(q: dict[str, float]) -> str:
    if not q:
        return "–"
    return " / ".join(f"{q[k]:.4g}" for k in ("min", "p05", "median", "max"))


def render_markdown(a: dict[str, Any]) -> str:
    """Human-readable summary of `run_audit` output."""
    m, t = a["mesh"], a["topology"]
    lines = [
        f"# Data audit: {a['dataset']} (HF {str(a['hf_revision'])[:10]})",
        "",
        f"Samples: {a['n_samples']}; splits: {a['splits']}",
        f"Samples in neither train nor test split: {a['unassigned_samples'] or 'none'}",
        "",
        "Quantiles are min / p05 / median / max over samples.",
        "",
        "## Mesh and topology",
        "",
        f"- nodes, train: {_fmt(m['n_nodes']['train'])}; test: {_fmt(m['n_nodes']['test'])}",
        f"- cells: {_fmt(m['n_cells'])}",
        f"- same bounding box in every sample: {m['same_bbox_all_samples']}; "
        f"x [{_fmt(m['bbox_min_max']['xmin'])}] to [{_fmt(m['bbox_min_max']['xmax'])}], "
        f"y [{_fmt(m['bbox_min_max']['ymin'])}] to [{_fmt(m['bbox_min_max']['ymax'])}]",
        f"- samples with inverted / degenerate cells: {m['samples_with_inverted_cells']} / "
        f"{m['samples_with_degenerate_cells']}",
        f"- free-edge loops: {_fmt(t['n_loops'])}; "
        f"interior loops (holes): {_fmt(t['n_interior_loops'])}",
        f"- void fraction (1 - meshed area / bbox area): {_fmt(t['void_fraction'])}",
        f"- periodic node layout, fraction of samples: x {a['periodic_fraction']['x']:.3f}, "
        f"y {a['periodic_fraction']['y']:.3f}",
    ]
    if "inclusions_vs_interior_loops" in t:
        lines += [
            "- interior loops == inclusion count in "
            f"{t['fraction_interior_loops_equal_n_inc']:.3f} of samples",
            f"- (inclusions, interior loops) pairs: {t['inclusions_vs_interior_loops']}",
            f"- inclusion counts: train {t['n_inc_counts']['train']}, "
            f"test {t['n_inc_counts']['test']}",
        ]
    locs = ("left", "right", "bottom", "top", "hole", "interior")
    lines += ["", "## Nodal tags", "", "| tag | present | nodes | " + " | ".join(locs) + " |"]
    lines += ["|---" * (3 + len(locs)) + "|"]
    for name, d in a["tags"].items():
        loc = d["mean_location_fraction"]
        lines.append(
            f"| {name} | {d['present_fraction']:.3f} | {_fmt(d['n_nodes'])} | "
            + " | ".join(f"{loc[k]:.2f}" for k in locs)
            + " |"
        )
    lines += ["", "## Output ranges (samples with outputs)", ""]
    lines += ["| output | per-sample max abs |", "|---|---|"]
    for f, q in a["fields_maxabs"].items():
        lines.append(f"| field {f} | {_fmt(q)} |")
    for k, q in a["scalars_abs"].items():
        lines.append(f"| scalar {k} | {_fmt(q)} |")
    lines += ["", "Near-zero flags: " + ("; ".join(a["near_zero_flags"]) or "none")]
    if "energy_identity_rel_error" in a:
        lines += ["", "## effective_energy vs area average of psi (relative error)", ""]
        for key, q in a["energy_identity_rel_error"].items():
            lines.append(f"- {key}: {_fmt(q)}")
    lines += ["", "## Train vs test", "", "| input | train | test | KS |", "|---|---|---|---|"]
    for name, d in a["train_vs_test"]["inputs"].items():
        lines.append(f"| {name} | {_fmt(d['train'])} | {_fmt(d['test'])} | {d['ks']:.3f} |")
    lines.append(f"\nKS node count: {a['train_vs_test']['n_nodes_ks']:.3f}")
    if "n_inc_ks" in a["train_vs_test"]:
        lines.append(f"KS inclusion count: {a['train_vs_test']['n_inc_ks']:.3f}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", type=Path, default=None, help="default: <PLAID_DATA_DIR>/audit/<name>")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    cache_dir = data_dir() / "cache" / args.dataset
    out = args.out or data_dir() / "audit" / args.dataset
    if (out / "audit.json").exists() and not args.overwrite:
        print(f"{out / 'audit.json'} exists; pass --overwrite to replace it.", file=sys.stderr)
        return 1
    audit = run_audit(cache_dir, args.dataset)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(audit, indent=1) + "\n")
    md = render_markdown(audit)
    (out / "audit.md").write_text(md)
    print(md)
    print(f"Wrote {out / 'audit.json'} and audit.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

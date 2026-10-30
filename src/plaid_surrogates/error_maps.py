"""Where does a run's error sit? (docs/methods.md §4; docs/design/DSG-002-data-2d-multischypel.md)

    python -m plaid_surrogates.error_maps --run <run dir|id> [--checkpoint best] [--n-worst 3]

Reads the run's dev predictions (predictions_dev_<ckpt>.pkl, written by `predict --split dev`)
and the cache. Writes into <run>/figures/:
- error_map_<sample id>.png for the n worst dev samples, ranked by their summed field
  contribution to the metric, sum_f (1/N) ||pred - ref||^2 / max|ref|^2; one panel per scored
  field showing |pred - ref| / max|ref| on the mesh, with the mesh boundary drawn;
- error_maps.json: that ranking, plus the error profile against the distance to each boundary
  category (data/features.py): the mean of |pred - ref| / max|ref| over all dev nodes and scored
  fields, binned by the node's distance (/ RVE size) to that category's boundary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from plaid_surrogates.data.cache import CachedSample, load_index, load_sample  # noqa: E402
from plaid_surrogates.data.features import (  # noqa: E402
    TAG_CATEGORIES,
    boundary_distances,
    free_edges,
)
from plaid_surrogates.data.registry import get_spec  # noqa: E402
from plaid_surrogates.data.splits import load_splits  # noqa: E402
from plaid_surrogates.submission import read_submission  # noqa: E402
from plaid_surrogates.utils.paths import data_dir, runs_dir  # noqa: E402

DIST_BINS = np.array([0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 1.5])


def relative_errors(
    rec: dict[str, Any], s: CachedSample, fields: tuple[str, ...]
) -> dict[str, np.ndarray]:
    """|pred - ref| / max|ref| per node, per field (the metric's per-sample normalization)."""
    return {
        f: np.abs(np.asarray(rec[f]) - s.output_fields[f]) / np.max(np.abs(s.output_fields[f]))
        for f in fields
    }


def sample_score(rel: dict[str, np.ndarray]) -> float:
    """Summed field contribution to the metric: sum_f mean_nodes(rel_f^2)."""
    return float(sum(np.mean(r**2) for r in rel.values()))


def plot_sample(s: CachedSample, rel: dict[str, np.ndarray], title: str, path: Path) -> None:
    """One panel per field: nodal relative error on the mesh, boundary edges in black."""
    n = len(rel)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.6 * nrows), squeeze=False)
    edges = free_edges(s.cells)
    x, y = s.points[:, 0], s.points[:, 1]
    for ax, (name, r) in zip(axes.flat, rel.items(), strict=False):
        tc = ax.tripcolor(x, y, s.cells, r, shading="gouraud", cmap="magma")
        for a, b in edges:
            ax.plot(x[[a, b]], y[[a, b]], "k-", lw=0.4)
        ax.set_title(f"{name}: max {r.max():.2e}")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(tc, ax=ax, shrink=0.8)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_error_maps(run_dir: Path, checkpoint: str = "best", n_worst: int = 3) -> dict[str, Any]:
    """Rank dev samples, draw the worst, and profile error vs boundary distance."""
    run_cfg = json.loads((run_dir / "config.json").read_text())
    dataset = run_cfg["config"]["dataset"]
    spec = get_spec(dataset)
    cache_dir = data_dir() / "cache" / dataset
    index = load_index(cache_dir)
    dev = load_splits(dataset, cache_dir)["dev"]
    recs = read_submission(run_dir / f"predictions_dev_{checkpoint}.pkl")
    if len(recs) != len(dev):
        raise ValueError(f"{len(recs)} predicted records, but the dev split has {len(dev)}")

    fields = spec.scored_fields
    cats = list(TAG_CATEGORIES.get(dataset, {}))
    scores: dict[int, float] = {}
    rels: dict[int, dict[str, np.ndarray]] = {}
    sums = {c: np.zeros(len(DIST_BINS) - 1) for c in cats}
    counts = {c: np.zeros(len(DIST_BINS) - 1) for c in cats}
    samples: dict[int, CachedSample] = {}
    for rec, i in zip(recs, dev, strict=True):
        s = load_sample(cache_dir / index["samples"][str(i)]["file"])
        rel = relative_errors(rec, s, fields)
        scores[i], rels[i], samples[i] = sample_score(rel), rel, s
        if cats:
            d = boundary_distances(s, dataset)
            node_err = np.mean(np.stack(list(rel.values()), axis=1), axis=1)
            for k, c in enumerate(cats):
                b = np.clip(np.digitize(d[:, k], DIST_BINS) - 1, 0, len(DIST_BINS) - 2)
                sums[c] += np.bincount(b, weights=node_err, minlength=len(DIST_BINS) - 1)
                counts[c] += np.bincount(b, minlength=len(DIST_BINS) - 1)

    ranking = sorted(scores, key=scores.get, reverse=True)
    for i in ranking[:n_worst]:
        plot_sample(
            samples[i],
            rels[i],
            f"{run_dir.name} ({checkpoint}.pt), dev sample {i}: field score {scores[i]:.2e}",
            run_dir / "figures" / f"error_map_{i}.png",
        )
    out = {
        "run_id": run_dir.name,
        "checkpoint": checkpoint,
        "ranking": [{"id": i, "field_score": scores[i]} for i in ranking],
        "worst_plotted": ranking[:n_worst],
        "distance_bins": DIST_BINS.tolist(),
        "mean_rel_error_by_distance": {
            c: [float(tot / n) if n else None for tot, n in zip(sums[c], counts[c], strict=True)]
            for c in cats
        },
        "nodes_per_bin": {c: counts[c].astype(int).tolist() for c in cats},
    }
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures" / "error_maps.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--run", required=True, help="run folder, or run id under PLAID_RUNS_DIR")
    p.add_argument("--checkpoint", default="best", choices=("best", "last"))
    p.add_argument("--n-worst", type=int, default=3)
    args = p.parse_args(argv)
    run_dir = next((q for q in (Path(args.run), runs_dir() / args.run) if q.is_dir()), None)
    if run_dir is None:
        raise FileNotFoundError(f"no run folder {args.run!r}")
    out = run_error_maps(run_dir, args.checkpoint, args.n_worst)
    print(f"worst dev samples: {out['ranking'][: args.n_worst]}")
    for c, v in out["mean_rel_error_by_distance"].items():
        cells = ", ".join("–" if x is None else f"{x:.2e}" for x in v)
        print(f"mean |e|/max|ref| by distance to {c} (bins {out['distance_bins']}): {cells}")
    print(f"figures: {run_dir / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

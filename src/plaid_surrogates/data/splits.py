"""Seeded, stratified dev split and CV folds over the official training split (docs/methods.md §2).

    python -m plaid_surrogates.data.splits --dataset Tensile2d [--val-frac 0.1 --folds 5 --seed 0]

Strata: joint bins of per-dataset variables. Float variables are cut into quantile bins; integer
variables are categorical (one stratum per value). Tensile2d: node count x load P (quantile bins);
2D_MultiScHypEl: number of inclusions (categorical) x macroscopic deformation magnitude.
Writes <PLAID_DATA_DIR>/splits/<dataset>.json with:
    dev    ~val_frac of the training split   (day-to-day model selection)
    train  the rest                          (what dev experiments train on)
    folds  k stratified folds over the WHOLE training split (independent of dev; final candidates)
All ID lists keep the source (benchmark) order; they're subsets, never reshuffled lists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from plaid_surrogates.data.cache import load_index, load_sample, split_ids
from plaid_surrogates.data.registry import get_spec
from plaid_surrogates.utils.paths import data_dir

FORMAT_VERSION = 1


# --- stratification variables, per dataset


def _tensile2d_vars(
    index: dict[str, Any], ids: Sequence[int], scalars: dict[int, dict[str, float]]
) -> dict[str, np.ndarray]:
    # Node count ~ geometry (p1..p5); P = applied load, which sets the stress scale directly.
    return {
        "n_nodes": np.array([index["samples"][str(i)]["n_nodes"] for i in ids], dtype=float),
        "P": np.array([scalars[i]["P"] for i in ids]),
    }


def inclusion_counts(index: dict[str, Any]) -> dict[int, int]:
    """Sample ID -> number of inclusions, from the dataset's `*_nb_inc_<k>` splits ({} if none)."""
    out: dict[int, int] = {}
    for name, ids in index["splits"].items():
        m = re.search(r"_nb_inc_(\d+)$", name)
        if m is None:
            continue
        k = int(m.group(1))
        for i in ids:
            if out.setdefault(int(i), k) != k:
                raise ValueError(f"sample {i} is in two inclusion-count splits")
    return out


def _multischypel_vars(
    index: dict[str, Any], ids: Sequence[int], scalars: dict[int, dict[str, float]]
) -> dict[str, np.ndarray]:
    # Number of inclusions (categorical: the geometry family) x ||C - I||_F, the size of the
    # imposed macroscopic deformation (C11, C12, C22 = right Cauchy-Green tensor, identity at
    # rest), which sets the stress and energy scale. The data audit checks both readings.
    n_inc = inclusion_counts(index)
    missing = [i for i in ids if i not in n_inc]
    if missing:
        raise ValueError(f"samples without an inclusion-count split, e.g. {missing[:5]}")
    c = np.array([[scalars[i]["C11"], scalars[i]["C12"], scalars[i]["C22"]] for i in ids])
    load = np.sqrt((c[:, 0] - 1.0) ** 2 + 2.0 * c[:, 1] ** 2 + (c[:, 2] - 1.0) ** 2)
    return {
        "n_inclusions": np.array([n_inc[i] for i in ids], dtype=np.int64),
        "C_minus_I_norm": load,
    }


# dataset -> function(index, ids, input_scalars_by_id) -> {variable: per-sample values}.
# Float variables are cut into n_bins quantile bins; integer variables are categorical.
# Strata are the joint bins.
StratFn = Callable[
    [dict[str, Any], Sequence[int], dict[int, dict[str, float]]], dict[str, np.ndarray]
]
STRATIFIERS: dict[str, StratFn] = {
    "Tensile2d": _tensile2d_vars,
    "2D_MultiScHypEl": _multischypel_vars,
}


def quantile_bins(values: np.ndarray, n_bins: int) -> tuple[np.ndarray, list[float]]:
    """Label each value with its quantile bin (0..n_bins-1); returns labels and interior edges."""
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(edges, values, side="right"), [float(e) for e in edges]


def joint_strata(
    variables: dict[str, np.ndarray], n_bins: int
) -> tuple[np.ndarray, dict[str, list[float]], dict[str, list[int]], int]:
    """Joint stratum label per sample (mixed-radix code).

    Float variables get `n_bins` quantile bins; integer variables are categorical (one level per
    distinct value). Returns (labels, quantile edges by variable, categories by variable, number
    of cells).
    """
    labels = None
    edges: dict[str, list[float]] = {}
    categories: dict[str, list[int]] = {}
    n_cells = 1
    for name, values in variables.items():
        values = np.asarray(values)
        if np.issubdtype(values.dtype, np.integer):
            levels, lab = np.unique(values, return_inverse=True)
            categories[name] = [int(v) for v in levels]
            radix = len(levels)
        else:
            lab, edges[name] = quantile_bins(values, n_bins)
            radix = n_bins
        labels = lab if labels is None else labels * radix + lab
        n_cells *= radix
    return labels, edges, categories, n_cells


# --- split construction (pure functions, unit-tested)


def stratified_holdout(
    ids: Sequence[int], strata: Sequence[int], frac: float, seed: int
) -> list[int]:
    """Pick round(frac * len(ids)) IDs, allocated across strata proportionally.

    Largest-remainder allocation hits the exact total; within a stratum the pick is random
    (seeded). Returned in source order.
    """
    if not 0 < frac < 1:
        raise ValueError(f"frac must be in (0, 1), got {frac}")
    ids, strata = list(ids), np.asarray(strata)
    rng = np.random.default_rng(seed)
    labels = np.unique(strata)
    sizes = np.array([(strata == s).sum() for s in labels])
    target = round(frac * len(ids))
    quota = frac * sizes
    alloc = np.floor(quota).astype(int)
    remainder = quota - alloc
    order = np.lexsort((rng.random(len(labels)), -remainder))  # ties broken randomly
    alloc[order[: target - alloc.sum()]] += 1

    chosen: set[int] = set()
    for s, n in zip(labels, alloc, strict=True):
        members = [i for i, st in zip(ids, strata, strict=True) if st == s]
        chosen.update(rng.choice(members, size=n, replace=False).tolist())
    return [i for i in ids if i in chosen]


def stratified_kfold(
    ids: Sequence[int], strata: Sequence[int], k: int, seed: int
) -> list[list[int]]:
    """k disjoint folds covering `ids`, each with ~equal share of every stratum. Source order."""
    if k < 2:
        raise ValueError("need at least 2 folds")
    ids, strata = list(ids), np.asarray(strata)
    rng = np.random.default_rng(seed)
    dealt: list[int] = []
    for s in np.unique(strata):
        members = [i for i, st in zip(ids, strata, strict=True) if st == s]
        dealt.extend(rng.permutation(members).tolist())
    fold_of = {i: n % k for n, i in enumerate(dealt)}  # round-robin across the stratum-ordered list
    return [[i for i in ids if fold_of[i] == f] for f in range(k)]


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic max|F_a - F_b| (balance diagnostic)."""
    grid = np.sort(np.concatenate([a, b]))
    fa = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    fb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))


def source_hash(index: dict[str, Any], split: str) -> str:
    """Fingerprint of the source split + data revision; splits files are invalid if it changes."""
    payload = json.dumps([index.get("hf_revision"), split, split_ids(index, split)])
    return hashlib.sha256(payload.encode()).hexdigest()


# --- file-level API


def make_splits(
    cache_dir: Path, dataset: str, val_frac: float, n_folds: int, seed: int, n_bins: int
) -> dict[str, Any]:
    """Build the splits dict for `dataset` from its cache."""
    spec = get_spec(dataset)
    index = load_index(cache_dir)
    source = split_ids(index, spec.train_split)
    scal = {
        i: load_sample(cache_dir / index["samples"][str(i)]["file"]).input_scalars for i in source
    }
    variables = STRATIFIERS[dataset](index, source, scal)
    strata, edges, categories, n_cells = joint_strata(variables, n_bins)

    dev = stratified_holdout(source, strata, val_frac, seed)
    dev_set = set(dev)
    train = [i for i in source if i not in dev_set]
    folds = stratified_kfold(source, strata, n_folds, seed)

    # Balance report: dev vs train distributions of every input scalar (KS statistic).
    balance = {
        name: ks_statistic(
            np.array([scal[i][name] for i in dev]), np.array([scal[i][name] for i in train])
        )
        for name in spec.input_scalars
    }
    dev_strata = strata[[k for k, i in enumerate(source) if i in dev_set]]
    return {
        "format_version": FORMAT_VERSION,
        "dataset": dataset,
        "source_split": spec.train_split,
        "source_hash": source_hash(index, spec.train_split),
        "seed": seed,
        "val_frac": val_frac,
        "stratification": {
            "variables": list(variables),
            "n_bins_per_variable": n_bins,
            "interior_edges": edges,
            "categories": categories,
            "cell_counts": np.bincount(strata, minlength=n_cells).tolist(),
            "dev_cell_counts": np.bincount(dev_strata, minlength=n_cells).tolist(),
        },
        "balance_ks_dev_vs_train": balance,
        "dev": dev,
        "train": train,
        "folds": folds,
    }


def splits_path(dataset: str) -> Path:
    return data_dir() / "splits" / f"{dataset}.json"


def load_splits(dataset: str, cache_dir: Path | None = None) -> dict[str, Any]:
    """Read the splits file and verify it still matches the cache's training split."""
    splits = json.loads(splits_path(dataset).read_text())
    cache_dir = cache_dir or data_dir() / "cache" / dataset
    index = load_index(cache_dir)
    if splits["source_hash"] != source_hash(index, splits["source_split"]):
        raise ValueError(f"{splits_path(dataset)} was made from a different cache; regenerate it")
    return splits


def train_val_ids(splits: dict[str, Any], fold: int | None) -> tuple[list[int], list[int]]:
    """(training IDs, validation IDs) for a run: train/dev without a fold; with fold k, every
    sample outside folds[k] (the other folds, in fold order) and folds[k]."""
    if fold is None:
        return list(splits["train"]), list(splits["dev"])
    folds = splits["folds"]
    if not 0 <= fold < len(folds):
        raise ValueError(f"fold must be in 0..{len(folds) - 1}, got {fold}")
    train = [i for k, f in enumerate(folds) if k != fold for i in f]
    return train, list(folds[fold])


def full_train_ids(splits: dict[str, Any]) -> list[int]:
    """Every official training sample (the folds cover them all, disjointly), in fold order:
    the training set of the final retrain (docs/methods.md §2)."""
    ids = [i for f in splits["folds"] for i in f]
    if len(set(ids)) != len(ids) or set(ids) != set(splits["train"]) | set(splits["dev"]):
        raise ValueError("folds do not partition train + dev; regenerate the splits file")
    return ids


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bins", type=int, default=5)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    out = splits_path(args.dataset)
    if out.exists() and not args.overwrite:
        print(f"{out} exists; pass --overwrite to replace it.", file=sys.stderr)
        return 1
    splits = make_splits(
        data_dir() / "cache" / args.dataset,
        args.dataset,
        args.val_frac,
        args.folds,
        args.seed,
        args.bins,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(splits, indent=1) + "\n")
    st = splits["stratification"]
    print(
        f"Wrote {out}: dev {len(splits['dev'])}, train {len(splits['train'])}, "
        f"folds {[len(f) for f in splits['folds']]}"
    )
    print(
        f"  strata over {st['variables']}: {st['n_bins_per_variable']}-quantile bins for floats, "
        f"categories {st['categories']}"
    )
    print(f"  cell sizes (sorted) {sorted(st['cell_counts'])}")
    print(f"  dev per cell {st['dev_cell_counts']}")
    print(
        "  KS dev vs train:", {k: round(v, 3) for k, v in splits["balance_ks_dev_vs_train"].items()}
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

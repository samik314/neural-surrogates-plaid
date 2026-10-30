"""Compare our cache (built from the Hugging Face dataset) with the Zenodo copy of a PLAID dataset.

    python -m plaid_surrogates.data.compare_zenodo --dataset 2D_MultiScHypEl \
        --zenodo-root <extracted dir> [--n-train 20] [--overwrite]

Why: every 2D_MultiScHypEl leaderboard upload fails while other teams score. The board's scorer
computes `prediction[i][fn] - reference[i][fn]`, which only fails when our arrays and its
reference arrays differ in length. The board builds the reference from the HF or the Zenodo copy
(the official FNO baseline uses Zenodo), so we check that the two copies agree on what a
submission depends on:
- the DOE_test / DOE_train ID lists and their order;
- for every DOE_test sample: node count, node coordinates (and, if they differ, whether the nodes
  match up to a permutation), TRI_3 connectivity, nodal tags;
- the same plus output fields and scalars for the first `n_train` DOE_train samples.
Zenodo samples are read by `convert.load_plaid_dir` (the only module allowed to import plaid,
docs/methods.md §6) into cache form; this module compares cache-form samples only.
Writes <PLAID_DATA_DIR>/audit/<dataset>/zenodo_compare.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from plaid_surrogates.data.cache import CachedSample, load_index, load_sample, split_ids
from plaid_surrogates.data.registry import get_spec
from plaid_surrogates.utils.paths import data_dir

COORD_TOL = 1e-12  # both copies store node coordinates in float64


def _sorted_rows(points: np.ndarray) -> np.ndarray:
    return points[np.lexsort(points.T[::-1])]


def compare_sample(ours: CachedSample, theirs: CachedSample) -> dict[str, Any]:
    """One sample, ours vs theirs. Outputs are compared only when both copies have them."""
    r: dict[str, Any] = {
        "id": ours.sample_id,
        "n_nodes_ours": ours.n_nodes,
        "n_nodes_zenodo": theirs.n_nodes,
    }
    r["same_n_nodes"] = ours.n_nodes == theirs.n_nodes
    if r["same_n_nodes"]:
        r["max_coord_diff"] = float(np.max(np.abs(ours.points - theirs.points)))
        r["same_node_order"] = r["max_coord_diff"] <= COORD_TOL
        if not r["same_node_order"]:
            r["max_coord_diff_sorted"] = float(
                np.max(np.abs(_sorted_rows(ours.points) - _sorted_rows(theirs.points)))
            )
    r["same_cells"] = bool(np.array_equal(ours.cells, theirs.cells))
    r["same_tags"] = set(ours.nodal_tags) == set(theirs.nodal_tags) and all(
        np.array_equal(ours.nodal_tags[k], theirs.nodal_tags[k]) for k in ours.nodal_tags
    )
    if ours.output_fields and theirs.output_fields and r["same_n_nodes"]:
        r["field_rel_diff"] = {
            f: float(np.max(np.abs(theirs.output_fields[f] - v)) / np.max(np.abs(v)))
            for f, v in ours.output_fields.items()
        }
    if ours.output_scalars and theirs.output_scalars:
        r["scalar_rel_diff"] = {
            s: abs(theirs.output_scalars[s] - v) / abs(v) for s, v in ours.output_scalars.items()
        }
    return r


def find_plaid_root(root: Path) -> Path:
    """The directory holding `dataset/` and `problem_definition/` inside the extracted archive."""
    hits = sorted(p.parent for p in Path(root).rglob("problem_definition") if p.is_dir())
    hits = [h for h in hits if (h / "dataset").is_dir()]
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one PLAID root under {root}, found {hits}")
    return hits[0]


def summarize(splits: dict[str, Any], parts: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    s: dict[str, Any] = {}
    for part, rows in parts.items():
        s[part] = {
            "n": len(rows),
            "n_count_mismatch": sum(not r["same_n_nodes"] for r in rows),
            "n_order_mismatch": sum(r.get("same_node_order") is False for r in rows),
            "n_cells_mismatch": sum(not r["same_cells"] for r in rows),
            "n_tags_mismatch": sum(not r["same_tags"] for r in rows),
            "first_count_mismatches": [
                [r["id"], r["n_nodes_ours"], r["n_nodes_zenodo"]]
                for r in rows
                if not r["same_n_nodes"]
            ][:10],
        }
        fr = [v for r in rows for v in r.get("field_rel_diff", {}).values()]
        sr = [v for r in rows for v in r.get("scalar_rel_diff", {}).values()]
        if fr:
            s[part]["max_field_rel_diff"] = max(fr)
        if sr:
            s[part]["max_scalar_rel_diff"] = max(sr)
    agree = all(v["same_ids_same_order"] for v in splits.values()) and all(
        v["n_count_mismatch"] + v["n_order_mismatch"] + v["n_cells_mismatch"] == 0
        for v in s.values()
    )
    s["verdict"] = "copies agree" if agree else "copies DIFFER"
    return s


def run_compare(dataset: str, zenodo_root: Path, n_train: int) -> dict[str, Any]:
    from plaid_surrogates.data.convert import load_plaid_dir

    spec = get_spec(dataset)
    cache_dir = data_dir() / "cache" / dataset
    index = load_index(cache_dir)
    plaid_root = find_plaid_root(zenodo_root)
    test_ids = split_ids(index, spec.test_split)
    train_ids = split_ids(index, spec.train_split)[:n_train]
    their_splits, theirs = load_plaid_dir(plaid_root, spec, test_ids + train_ids)

    splits = {}
    for split in (spec.test_split, spec.train_split):
        ours_ids = split_ids(index, split)
        splits[split] = {
            "n_ours": len(ours_ids),
            "n_zenodo": len(their_splits[split]),
            "same_ids_same_order": ours_ids == their_splits[split],
            "same_id_set": set(ours_ids) == set(their_splits[split]),
        }

    def rows(ids: list[int]) -> list[dict[str, Any]]:
        return [
            compare_sample(load_sample(cache_dir / index["samples"][str(i)]["file"]), theirs[i])
            for i in ids
        ]

    parts = {"test": rows(test_ids), "train": rows(train_ids)}
    return {
        "dataset": dataset,
        "zenodo_root": str(plaid_root),
        "splits": splits,
        **parts,
        "summary": summarize(splits, parts),
    }


def render_markdown(out: dict[str, Any]) -> str:
    s = out["summary"]
    lines = [
        f"# {out['dataset']}: our cache (Hugging Face) vs the Zenodo copy",
        "",
        f"Zenodo root: `{out['zenodo_root']}`. **Verdict: {s['verdict']}.**",
        "",
        "| split | ours | Zenodo | same IDs, same order | same ID set |",
        "|---|---|---|---|---|",
    ]
    for k, v in out["splits"].items():
        lines.append(
            f"| {k} | {v['n_ours']} | {v['n_zenodo']} | {v['same_ids_same_order']} | "
            f"{v['same_id_set']} |"
        )
    lines += [
        "",
        "| samples | n | node-count mismatches | node-order mismatches | cell mismatches | "
        "tag mismatches | max field rel. diff | max scalar rel. diff |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in ("test", "train"):
        v = s[p]
        lines.append(
            f"| {p} | {v['n']} | {v['n_count_mismatch']} | {v['n_order_mismatch']} | "
            f"{v['n_cells_mismatch']} | {v['n_tags_mismatch']} | "
            f"{v.get('max_field_rel_diff', '—')} | {v.get('max_scalar_rel_diff', '—')} |"
        )
    for p in ("test", "train"):
        if s[p]["first_count_mismatches"]:
            lines += ["", f"First {p} node-count mismatches [id, ours, Zenodo]:"]
            lines.append(str(s[p]["first_count_mismatches"]))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--zenodo-root", required=True, type=Path)
    p.add_argument("--n-train", type=int, default=20)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    out_dir = data_dir() / "audit" / args.dataset
    if (out_dir / "zenodo_compare.json").exists() and not args.overwrite:
        print(f"{out_dir / 'zenodo_compare.json'} exists; pass --overwrite.", file=sys.stderr)
        return 1
    r = run_compare(args.dataset, args.zenodo_root, args.n_train)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "zenodo_compare.json").write_text(json.dumps(r, indent=1) + "\n")
    md = render_markdown(r)
    (out_dir / "zenodo_compare.md").write_text(md)
    print(md)
    print(f"Wrote {out_dir / 'zenodo_compare.json'} and zenodo_compare.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""HF -> PLAID -> cache conversion. The ONLY runtime module that imports plaid (docs/methods.md §6).

    python -m plaid_surrogates.data.convert --dataset Tensile2d [--processes 4] [--overwrite]

plaid/datasets are imported inside `main()`, after HF_HOME is set, so downloads land in the
project's cache and this module stays importable (and unit-testable) without PLAID.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from plaid_surrogates.data.cache import (
    INDEX_FILE,
    SAMPLES_DIR,
    CachedSample,
    sample_filename,
    save_sample,
    write_index,
)
from plaid_surrogates.data.registry import DatasetSpec, get_spec
from plaid_surrogates.utils.paths import data_dir, set_hf_home_env


def check_problem_matches_spec(problem: Any, spec: DatasetSpec) -> None:
    """Stop if the PLAID problem definition disagrees with the registry (docs/methods.md §1)."""
    got = {
        "input_scalars": set(problem.get_input_scalars_names()),
        "output_fields": set(problem.get_output_fields_names()),
        "output_scalars": set(problem.get_output_scalars_names()),
    }
    for key, names in got.items():
        want = set(getattr(spec, key))
        if names != want:
            raise ValueError(
                f"{spec.name}: problem {key} {sorted(names)} != registry {sorted(want)}"
            )
    splits = problem.get_split()
    for s in (spec.train_split, spec.test_split):
        if s not in splits:
            raise ValueError(f"{spec.name}: split {s!r} missing; have {list(splits)}")


def plaid_sample_to_cached(sample: Any, sample_id: int, spec: DatasetSpec) -> CachedSample:
    """Extract one PLAID Sample into plain arrays, keeping node order exactly as stored."""
    elements = sample.get_elements()
    if set(elements) != {"TRI_3"}:
        raise ValueError(f"sample {sample_id}: expected only TRI_3 elements, got {list(elements)}")

    field_names = {str(n) for n in sample.get_field_names()}
    scalar_names = {str(n) for n in sample.get_scalar_names()}
    unexpected = field_names - set(spec.output_fields)
    if unexpected:
        raise ValueError(f"sample {sample_id}: unexpected fields {sorted(unexpected)}")
    missing_inputs = set(spec.input_scalars) - scalar_names
    if missing_inputs:
        raise ValueError(f"sample {sample_id}: missing input scalars {sorted(missing_inputs)}")

    # Outputs are all-or-nothing: train samples have every output, test samples none.
    present = (field_names & set(spec.output_fields)) | (scalar_names & set(spec.output_scalars))
    if present and present != set(spec.output_fields) | set(spec.output_scalars):
        raise ValueError(f"sample {sample_id}: partial outputs {sorted(present)}")

    return CachedSample(
        sample_id=int(sample_id),
        points=np.asarray(sample.get_nodes(), dtype=np.float64),
        cells=np.asarray(elements["TRI_3"], dtype=np.int64),
        input_scalars={n: float(sample.get_scalar(n)) for n in spec.input_scalars},
        output_fields=(
            {n: np.asarray(sample.get_field(n), dtype=np.float64) for n in spec.output_fields}
            if present
            else {}
        ),
        output_scalars=(
            {n: float(sample.get_scalar(n)) for n in spec.output_scalars} if present else {}
        ),
        nodal_tags={
            str(k): np.asarray(v, dtype=np.int64) for k, v in sample.get_nodal_tags().items()
        },
    )


def check_outputs_vs_splits(
    has_outputs: dict[int, bool], splits: dict[str, list[int]], spec: DatasetSpec
) -> None:
    """Train samples must have outputs; test samples must not (outputs are withheld)."""
    no_out = [i for i in splits[spec.train_split] if not has_outputs[i]]
    if no_out:
        raise ValueError(f"{spec.train_split}: samples without outputs, e.g. {no_out[:5]}")
    leaked = [i for i in splits[spec.test_split] if has_outputs[i]]
    if leaked:
        raise ValueError(f"{spec.test_split}: samples WITH outputs, e.g. {leaked[:5]}; investigate")


def _chunks(ids: list[int], size: int):
    for k in range(0, len(ids), size):
        yield ids[k : k + size]


def convert(spec: DatasetSpec, out: Path, processes: int, chunk_size: int) -> Path:
    """Convert every row of the HF dataset into `out`; returns the index.json path."""
    from datasets import load_dataset
    from plaid.bridges.huggingface_bridge import huggingface_dataset_to_plaid

    hf = load_dataset(spec.hf_repo, split="all_samples", revision=spec.revision)
    # huggingface_dataset_to_plaid gives each sample its HF row index as ID.
    _, problem = huggingface_dataset_to_plaid(hf, ids=[0], processes_number=1, verbose=False)
    check_problem_matches_spec(problem, spec)
    splits = {str(k): [int(i) for i in v] for k, v in problem.get_split().items()}

    all_ids = list(range(len(hf)))
    samples_info: dict[int, dict[str, Any]] = {}
    t0 = time.time()
    for chunk in _chunks(all_ids, chunk_size):
        ds, _ = huggingface_dataset_to_plaid(
            hf, ids=chunk, processes_number=min(processes, len(chunk)), verbose=False
        )
        for i in chunk:
            s = plaid_sample_to_cached(ds[i], i, spec)
            save_sample(out, s)
            samples_info[i] = {
                "file": sample_filename(i),
                "n_nodes": s.n_nodes,
                "n_cells": int(s.cells.shape[0]),
                "has_outputs": s.has_outputs,
            }
        print(
            f"  {len(samples_info)}/{len(all_ids)} samples  ({time.time() - t0:.0f} s)", flush=True
        )

    check_outputs_vs_splits({i: v["has_outputs"] for i, v in samples_info.items()}, splits, spec)
    meta = {
        "dataset": spec.name,
        "hf_repo": spec.hf_repo,
        "hf_revision": spec.revision,
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "versions": {p: version(p) for p in ("pyplaid", "datasets", "numpy")},
        "train_split": spec.train_split,
        "test_split": spec.test_split,
        "input_scalars": list(spec.input_scalars),
        "output_fields": list(spec.output_fields),
        "output_scalars": list(spec.output_scalars),
    }
    return write_index(out, meta, splits, samples_info)


def load_plaid_dir(
    plaid_root: Path, spec: DatasetSpec, ids: list[int], processes: int = 2
) -> tuple[dict[str, list[int]], dict[int, CachedSample]]:
    """Read a PLAID directory (e.g. the Zenodo copy: `dataset/` + `problem_definition/`).

    Returns the train/test split ID lists (as stored) and the requested samples in cache form,
    through the same `plaid_sample_to_cached` the HF conversion uses. Used by compare_zenodo.py.
    """
    from plaid.containers.dataset import Dataset
    from plaid.problem_definition import ProblemDefinition

    problem = ProblemDefinition()
    problem._load_from_dir_(Path(plaid_root) / "problem_definition")
    check_problem_matches_spec(problem, spec)
    names = (spec.train_split, spec.test_split)
    splits = {s: [int(i) for i in problem.get_split(s)] for s in names}
    ds = Dataset()
    ds._load_from_dir_(Path(plaid_root) / "dataset", ids=ids, processes_number=processes)
    return splits, {int(i): plaid_sample_to_cached(ds[i], int(i), spec) for i in ids}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", type=Path, default=None, help="default: <PLAID_DATA_DIR>/cache/<name>")
    p.add_argument("--processes", type=int, default=4)
    p.add_argument("--chunk-size", type=int, default=64, help="samples held in memory at once")
    p.add_argument("--overwrite", action="store_true", help="delete an existing cache first")
    args = p.parse_args(argv)

    spec = get_spec(args.dataset)
    out = (args.out or data_dir() / "cache" / spec.name).resolve()
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            print(
                f"{out} exists and is not empty; pass --overwrite to replace it.", file=sys.stderr
            )
            return 1
        foreign = {p.name for p in out.iterdir()} - {INDEX_FILE, SAMPLES_DIR}
        if foreign:
            print(
                f"Refusing to delete {out}: not a cache dir (has {sorted(foreign)[:5]}).",
                file=sys.stderr,
            )
            return 1
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"HF_HOME = {set_hf_home_env()}")
    print(f"Converting {spec.name} ({spec.hf_repo}@{spec.revision[:10]}) -> {out}")
    index_path = convert(spec, out, args.processes, args.chunk_size)
    print(f"Done: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

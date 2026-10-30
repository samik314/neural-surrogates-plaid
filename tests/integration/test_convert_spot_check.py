"""Cache vs PLAID on a few samples of every converted dataset. Needs pyplaid, the HF cache, and
the converted cache; a dataset whose cache is missing is skipped.

Runs offline (HF_HUB_OFFLINE) so it never triggers a download.
"""

import os

import numpy as np
import pytest

from plaid_surrogates.data.cache import load_index, load_sample, split_ids
from plaid_surrogates.data.convert import plaid_sample_to_cached
from plaid_surrogates.data.registry import REGISTRY, get_spec
from plaid_surrogates.utils.paths import data_dir, set_hf_home_env


def _cache(name):
    return data_dir() / "cache" / name


def spot_ids(index, spec):
    """Start, middle and end of the train and test splits, plus up to two samples in neither."""
    ids = []
    for split in (spec.train_split, spec.test_split):
        s = split_ids(index, split)
        ids += [s[0], s[len(s) // 2], s[-1]]
    assigned = set(split_ids(index, spec.train_split)) | set(split_ids(index, spec.test_split))
    ids += [int(i) for i in index["samples"] if int(i) not in assigned][:2]
    return list(dict.fromkeys(int(i) for i in ids))


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(
            name,
            marks=pytest.mark.skipif(
                not (_cache(name) / "index.json").is_file(), reason=f"no converted cache for {name}"
            ),
        )
        for name in REGISTRY
    ],
)
def plaid_data(request):
    spec = get_spec(request.param)
    cache = _cache(spec.name)
    index = load_index(cache)
    ids = spot_ids(index, spec)
    set_hf_home_env()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from datasets import load_dataset
    from plaid.bridges.huggingface_bridge import huggingface_dataset_to_plaid

    hf = load_dataset(spec.hf_repo, split="all_samples", revision=spec.revision)
    dataset, problem = huggingface_dataset_to_plaid(hf, ids=ids, processes_number=1, verbose=False)
    return spec, cache, index, ids, dataset, problem


def test_splits_identical_to_plaid_in_order(plaid_data):
    *_, problem = plaid_data
    index = plaid_data[2]
    plaid_splits = problem.get_split()
    assert list(index["splits"]) == [str(k) for k in plaid_splits]
    for name, ids in plaid_splits.items():
        assert split_ids(index, str(name)) == [int(i) for i in ids]


def test_cached_samples_equal_plaid(plaid_data):
    spec, cache, index, ids, dataset, _ = plaid_data
    train, test = set(split_ids(index, spec.train_split)), set(split_ids(index, spec.test_split))
    for sid in ids:
        ref = dataset[sid]
        cached = load_sample(cache / index["samples"][str(sid)]["file"])
        np.testing.assert_array_equal(cached.points, ref.get_nodes())
        np.testing.assert_array_equal(cached.cells, ref.get_elements()["TRI_3"])
        for name in spec.input_scalars:
            assert cached.input_scalars[name] == float(ref.get_scalar(name))
        for name, idx in ref.get_nodal_tags().items():
            np.testing.assert_array_equal(cached.nodal_tags[str(name)], idx)
        if cached.has_outputs:
            assert sid not in test, f"test sample {sid} has outputs"
            for name in spec.output_fields:
                np.testing.assert_array_equal(cached.output_fields[name], ref.get_field(name))
            for name in spec.output_scalars:
                assert cached.output_scalars[name] == float(ref.get_scalar(name))
        else:
            assert sid not in train, f"train sample {sid} has no outputs"
        # and the converter itself reproduces the cached sample
        again = plaid_sample_to_cached(ref, sid, spec)
        np.testing.assert_array_equal(again.points, cached.points)

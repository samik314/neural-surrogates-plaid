import subprocess
import sys
import textwrap

import numpy as np
import pytest
from torch.utils.data import DataLoader

from plaid_surrogates.data.cache import save_sample, write_index
from plaid_surrogates.data.dataset import CacheDataset, collate_single
from tests.test_cache import make_sample

ORDER = [30, 4, 17, 2]  # sorted order differs from stored order


@pytest.fixture
def cache(tmp_path):
    info = {}
    for i in ORDER + [99]:
        save_sample(tmp_path, make_sample(sample_id=i, n=7, outputs=(i != 99), seed=i))
        info[i] = {"file": f"samples/{i:06d}.npz", "has_outputs": i != 99}
    write_index(tmp_path, {"dataset": "fake"}, {"train": ORDER, "test": [99]}, info)
    return tmp_path


def test_items_follow_stored_order(cache):
    ds = CacheDataset(cache, "train")
    assert len(ds) == 4 and ds.ids == ORDER
    assert [ds[k].sample_id for k in range(len(ds))] == ORDER


def test_in_memory_matches_lazy(cache):
    lazy, mem = CacheDataset(cache, "train"), CacheDataset(cache, "train", in_memory=True)
    for k in range(len(lazy)):
        np.testing.assert_array_equal(lazy[k].points, mem[k].points)
        np.testing.assert_array_equal(lazy[k].output_fields["U1"], mem[k].output_fields["U1"])


def test_explicit_ids_kept_as_given_and_checked(cache):
    assert CacheDataset(cache, "train", ids=[17, 30]).ids == [17, 30]
    with pytest.raises(ValueError, match="not in split"):
        CacheDataset(cache, "train", ids=[99])


def test_test_split_has_no_outputs(cache):
    assert not CacheDataset(cache, "test")[0].has_outputs


def test_dataloader_batch_one(cache):
    loader = DataLoader(CacheDataset(cache, "train"), batch_size=1, collate_fn=collate_single)
    assert [s.sample_id for s in loader] == ORDER
    with pytest.raises(ValueError, match="batch_size=1"):
        next(
            iter(DataLoader(CacheDataset(cache, "train"), batch_size=2, collate_fn=collate_single))
        )


def test_cache_loading_works_with_plaid_blocked(cache):
    """docs/methods.md §6: everything except convert.py runs without PLAID installed."""
    code = textwrap.dedent(f"""
        import sys
        sys.modules["plaid"] = None  # makes any `import plaid` raise ImportError
        import pkgutil, importlib, plaid_surrogates
        for m in pkgutil.walk_packages(plaid_surrogates.__path__, "plaid_surrogates."):
            importlib.import_module(m.name)  # every module, convert.py included (lazy imports)
        from plaid_surrogates.data.dataset import CacheDataset
        ds = CacheDataset({str(cache)!r}, "train")
        assert [ds[k].sample_id for k in range(len(ds))] == {ORDER!r}
        try:
            import plaid
        except ImportError:
            print("OK")
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "OK"

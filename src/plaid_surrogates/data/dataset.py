"""Torch Dataset over the converted cache. Reads only the cache; never imports plaid.

Items are raw `CachedSample`s (numpy, physical units, original node order). Features and
normalization are built on top of this in features.py / normalize.py.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from torch.utils.data import Dataset

from plaid_surrogates.data.cache import CachedSample, load_index, load_sample, split_ids


class CacheDataset(Dataset):
    """Samples of one split (or an explicit ID list) in stored order.

    Args:
        cache_dir: converted cache directory (holds index.json).
        split: split name from index.json, e.g. "train_500" or "test".
        ids: optional subset/reordering (e.g. a dev fold); every ID must belong to `split`.
            Used as given, never sorted.
        in_memory: load every sample once up front (the Tensile2d cache is ~0.6 GB).
    """

    def __init__(
        self,
        cache_dir: Path,
        split: str,
        ids: Sequence[int] | None = None,
        in_memory: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.index = load_index(self.cache_dir)
        self.split = split
        split_list = split_ids(self.index, split)
        if ids is None:
            self.ids = split_list
        else:
            outside = set(ids) - set(split_list)
            if outside:
                raise ValueError(f"IDs not in split {split!r}: {sorted(outside)[:5]}")
            self.ids = [int(i) for i in ids]
        self._cache: list[CachedSample] | None = None
        if in_memory:
            self._cache = [self._load(i) for i in self.ids]

    def _load(self, sample_id: int) -> CachedSample:
        return load_sample(self.cache_dir / self.index["samples"][str(sample_id)]["file"])

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, k: int) -> CachedSample:
        if self._cache is not None:
            return self._cache[k]
        return self._load(self.ids[k])


def collate_single(batch: list[CachedSample]) -> CachedSample:
    """DataLoader collate for batch_size=1 (one full mesh per step; meshes differ in size)."""
    if len(batch) != 1:
        raise ValueError("variable-size meshes: use batch_size=1 with gradient accumulation")
    return batch[0]

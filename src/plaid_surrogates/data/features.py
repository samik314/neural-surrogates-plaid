"""Per-node model inputs from a cached sample (fixed-dimensional schema, docs/methods.md §4).

Feature sets (chosen in the config; each step adds one group, as a logged ablation):
- "input_scalars": the sample's standardized input scalars broadcast to every node.
- "input_scalars+tags": + a one-hot of the node's semantic boundary category, from a fixed
  per-dataset schema (data audit): one column per category plus "interior". Raw tag names never
  become features directly.
- "input_scalars+tags+dist": + the distance from each node to the nearest edge of each boundary
  category (exact point-to-segment distance over the category's free edges), divided by the
  RVE (bounding-box) size so it is dimensionless and needs no fitted statistics.
- "input_scalars+tags+dist+lig": + the distance to the SECOND-nearest hole (each hole = one
  connected loop of hole-boundary edges), same scaling. With the nearest-hole distance it
  locates the narrow ligaments between clustered holes, where the geometry diagnostic found the
  largest errors. Datasets with a "hole" category only.
Normalized coordinates go ONLY in the Transolver embedding, never in fx.
"""

from __future__ import annotations

import numpy as np

from plaid_surrogates.data.cache import CachedSample
from plaid_surrogates.data.normalize import Normalizer

FEATURE_SETS = (
    "input_scalars",
    "input_scalars+tags",
    "input_scalars+tags+dist",
    "input_scalars+tags+dist+lig",
)

# dataset -> semantic boundary category -> raw tag names (data audit). Order = column order.
TAG_CATEGORIES: dict[str, dict[str, tuple[str, ...]]] = {
    "Tensile2d": {"top": ("Top",), "bottom": ("Bottom", "BottomLeft")},
    "2D_MultiScHypEl": {"outer": ("Ext_bound",), "hole": ("Holes",)},
}


def _check(feature_set: str, dataset: str) -> None:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}; known: {FEATURE_SETS}")
    if feature_set != "input_scalars" and dataset not in TAG_CATEGORIES:
        raise ValueError(f"no tag categories defined for {dataset}")
    if "lig" in feature_set and "hole" not in TAG_CATEGORIES[dataset]:
        raise ValueError(f"feature set {feature_set!r} needs a 'hole' category; {dataset} has none")


def feature_names(
    norm: Normalizer, feature_set: str = "input_scalars", dataset: str = ""
) -> list[str]:
    """Column names of fx, in order."""
    _check(feature_set, dataset)
    names = [f"scalar:{n}" for n in norm.input_names]
    if "tags" in feature_set:
        names += [f"tag:{c}" for c in TAG_CATEGORIES[dataset]] + ["tag:interior"]
    if "dist" in feature_set:
        names += [f"dist:{c}" for c in TAG_CATEGORIES[dataset]]
    if "lig" in feature_set:
        names += ["dist:hole2"]
    return names


def category_masks(s: CachedSample, dataset: str) -> dict[str, np.ndarray]:
    """Boolean node mask per semantic category (a raw tag missing from a sample is just empty)."""
    masks = {}
    for cat, raw in TAG_CATEGORIES[dataset].items():
        m = np.zeros(s.n_nodes, dtype=bool)
        for name in raw:
            if name in s.nodal_tags:
                m[s.nodal_tags[name]] = True
        masks[cat] = m
    return masks


def free_edges(cells: np.ndarray) -> np.ndarray:
    """Edges used by exactly one triangle (the mesh boundary), as node pairs (E, 2)."""
    e = np.sort(np.concatenate([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    return uniq[counts == 1]


def point_segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point (N, 2) to the nearest of the segments a[k]-b[k] (E, 2); (N,)."""
    if len(a) == 0:
        raise ValueError("no segments")
    out = np.full(len(points), np.inf)
    ab = b - a
    ab2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-300)
    for start in range(0, len(points), 2048):  # chunks keep memory at ~2048 x E
        p = points[start : start + 2048, None, :]
        t = np.clip(np.einsum("nej,ej->ne", p - a[None], ab) / ab2, 0.0, 1.0)
        closest = a[None] + t[..., None] * ab[None]
        out[start : start + 2048] = np.sqrt(((p - closest) ** 2).sum(-1)).min(axis=1)
    return out


# (dataset, sample_id, n_nodes) -> distances. A sample's mesh never changes, and computing the
# distances costs ~0.35 s per 2D_MultiScHypEl sample, so dev evaluations reuse them in-process.
_DIST_CACHE: dict[tuple[str, int, int], np.ndarray] = {}


def boundary_distances(s: CachedSample, dataset: str) -> np.ndarray:
    """(N, n_categories): distance to each category's boundary edges / RVE size (cached)."""
    key = (dataset, s.sample_id, s.n_nodes)
    if key not in _DIST_CACHE:
        _DIST_CACHE[key] = _boundary_distances(s, dataset)
    return _DIST_CACHE[key]


def _boundary_distances(s: CachedSample, dataset: str) -> np.ndarray:
    masks = category_masks(s, dataset)
    edges = free_edges(s.cells)
    size = float(np.max(s.points.max(axis=0) - s.points.min(axis=0)))
    cols = []
    for cat, m in masks.items():
        sel = edges[m[edges[:, 0]] & m[edges[:, 1]]]  # free edges with both ends in the category
        if len(sel) == 0:
            raise ValueError(f"sample {s.sample_id}: no boundary edges in category {cat!r}")
        d = point_segment_distance(s.points, s.points[sel[:, 0]], s.points[sel[:, 1]])
        cols.append(d / size)
    return np.stack(cols, axis=1)


def edge_components(edges: np.ndarray) -> list[np.ndarray]:
    """Group edges into connected components (union-find over their nodes); edge index arrays."""
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
    roots = np.array([find(a) for a in edges[:, 0].tolist()])
    return [np.flatnonzero(roots == r) for r in np.unique(roots)]


_HOLE2_CACHE: dict[tuple[str, int, int], np.ndarray] = {}


def second_hole_distance(s: CachedSample, dataset: str) -> np.ndarray:
    """(N,): distance to the second-nearest hole / RVE size (cached; needs >= 2 holes)."""
    key = (dataset, s.sample_id, s.n_nodes)
    if key not in _HOLE2_CACHE:
        hole = category_masks(s, dataset)["hole"]
        edges = free_edges(s.cells)
        edges = edges[hole[edges[:, 0]] & hole[edges[:, 1]]]
        loops = edge_components(edges)
        if len(loops) < 2:
            raise ValueError(f"sample {s.sample_id}: {len(loops)} hole(s); need at least 2")
        size = float(np.max(s.points.max(axis=0) - s.points.min(axis=0)))
        per_hole = np.stack(
            [
                point_segment_distance(s.points, s.points[edges[k, 0]], s.points[edges[k, 1]])
                for k in loops
            ],
            axis=1,
        )
        _HOLE2_CACHE[key] = np.sort(per_hole, axis=1)[:, 1] / size
    return _HOLE2_CACHE[key]


def build_inputs(
    s: CachedSample, norm: Normalizer, feature_set: str = "input_scalars", dataset: str = ""
) -> tuple[np.ndarray, np.ndarray]:
    """Return (fx (N, C_in), coords (N, 2)), float32, rows in the sample's original node order."""
    _check(feature_set, dataset)
    cols = [np.broadcast_to(norm.encode_inputs(s), (s.n_nodes, len(norm.input_names)))]
    if "tags" in feature_set:
        masks = category_masks(s, dataset)
        onehot = np.stack(list(masks.values()), axis=1).astype(np.float64)
        interior = ~onehot.any(axis=1, keepdims=True)
        cols += [onehot, interior.astype(np.float64)]
    if "dist" in feature_set:
        cols.append(boundary_distances(s, dataset))
    if "lig" in feature_set:
        cols.append(second_hole_distance(s, dataset)[:, None])
    fx = np.concatenate(cols, axis=1)
    coords = norm.encode_coords(s.points)
    return np.ascontiguousarray(fx, np.float32), np.ascontiguousarray(coords, np.float32)

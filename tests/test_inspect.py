"""Data audit on toy meshes whose topology is known exactly."""

import json

import numpy as np
import pytest

from plaid_surrogates.data import inspect as ins
from plaid_surrogates.data.cache import CachedSample, sample_filename, save_sample, write_index
from plaid_surrogates.data.registry import get_spec

MS = get_spec("2D_MultiScHypEl")


def grid(n: int, removed=()):
    """Unit square, n x n cells, two CCW triangles per cell; `removed` cells (i, j) become holes."""
    xs = np.linspace(0.0, 1.0, n + 1)
    points = np.array([[x, y] for y in xs for x in xs])
    cells = []
    for j in range(n):
        for i in range(n):
            if (i, j) in removed:
                continue
            a, b = j * (n + 1) + i, j * (n + 1) + i + 1
            c, d = b + n + 1, a + n + 1
            cells += [[a, b, c], [a, c, d]]
    used = np.unique(cells)
    remap = -np.ones(len(points), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return points[used], remap[np.array(cells, dtype=np.int64)]


def test_areas_and_p1_integral():
    pts, cells = grid(3)
    a = ins.triangle_areas(pts, cells)
    assert np.all(a > 0) and a.sum() == pytest.approx(1.0)
    assert ins.integrate_p1(pts[:, 0], pts, cells) == pytest.approx(0.5)  # exact for linear f
    pts, cells = grid(3, removed={(1, 1)})
    assert ins.integrate_p1(pts[:, 0], pts, cells) == pytest.approx(0.5 - 0.5 / 9)


@pytest.mark.parametrize(
    ("n", "removed", "loops", "interior"),
    [
        (3, set(), 1, 0),
        (3, {(1, 1)}, 2, 1),
        (5, {(1, 1), (3, 3)}, 3, 2),
        (3, {(0, 1)}, 1, 0),  # a hole cut by the outer edge is a notch, not an interior loop
    ],
)
def test_free_edge_loops(n, removed, loops, interior):
    pts, cells = grid(n, removed)
    s = CachedSample(0, pts, cells, {})
    rec = ins.sample_record(s, (), ())
    assert rec["n_loops"] == loops and rec["n_interior_loops"] == interior
    assert rec["void_fraction"] == pytest.approx(len(removed) / n**2)
    assert rec["n_inverted"] == 0 and rec["n_degenerate"] == 0


def test_periodicity_detection():
    pts, _ = grid(3)
    assert ins.periodic_match(pts, 1e-9) == {"x": True, "y": True}
    shifted = pts.copy()
    shifted[(pts[:, 0] == 1.0) & (pts[:, 1] > 0) & (pts[:, 1] < 1), 1] += 0.01
    assert ins.periodic_match(shifted, 1e-9) == {"x": False, "y": True}


def test_non_manifold_mesh_rejected():
    cells = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]], dtype=np.int64)
    with pytest.raises(ValueError, match="non-manifold"):
        ins.boundary_edges(cells)


def _ms_sample(i, removed, rng, outputs, p12_scale=1.0):
    pts, cells = grid(3, removed)
    n = len(pts)
    top = np.flatnonzero(pts[:, 1] == 1.0).astype(np.int64)
    s = CachedSample(
        sample_id=i,
        points=pts,
        cells=cells,
        input_scalars={"C11": 1 + rng.uniform(0, 0.2), "C12": rng.uniform(-0.1, 0.1), "C22": 1.0},
        nodal_tags={"Top": top},
    )
    if outputs:
        s.output_fields = {f: rng.standard_normal(n) + 2.0 for f in MS.output_fields}
        s.output_fields["P12"] = s.output_fields["P12"] * p12_scale
        s.output_fields["psi"] = np.full(n, 2.0)
        s.output_scalars = {"effective_energy": 2.0}  # = average of psi over the solid
    return s


@pytest.fixture
def ms_cache(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    cache = tmp_path / "data" / "cache" / "2D_MultiScHypEl"
    layout = {0: {(1, 1)}, 1: set(), 2: {(1, 1)}, 3: {(1, 1)}, 4: set()}  # 0-2 train, 3-4 test
    info = {}
    for i, removed in layout.items():
        s = _ms_sample(i, removed, rng, outputs=i < 3, p12_scale=1e-6 if i == 0 else 1.0)
        save_sample(cache, s)
        info[i] = {"file": sample_filename(i), "n_nodes": s.n_nodes, "has_outputs": i < 3}
    splits = {
        "DOE_train": [2, 0, 1],
        "DOE_test": [4, 3],
        "DOE_train_nb_inc_1": [0, 2],
        "DOE_train_nb_inc_0": [1],
        "DOE_test_nb_inc_1": [3],
        "DOE_test_nb_inc_0": [4],
    }
    write_index(cache, {"hf_revision": "fake"}, splits, info)
    monkeypatch.setenv("PLAID_DATA_DIR", str(tmp_path / "data"))
    return cache


def test_audit_end_to_end(ms_cache):
    a = ins.run_audit(ms_cache, "2D_MultiScHypEl")
    assert a["n_samples"] == 5 and a["unassigned_samples"] == []
    t = a["topology"]
    assert t["fraction_interior_loops_equal_n_inc"] == 1.0
    assert t["n_inc_counts"] == {"train": {0: 1, 1: 2}, "test": {0: 1, 1: 1}}
    assert a["mesh"]["same_bbox_all_samples"] is True
    assert a["periodic_fraction"] == {"x": 1.0, "y": 1.0}
    assert a["tags"]["Top"]["mean_location_fraction"]["top"] == 1.0
    assert any(f.startswith("field P12") for f in a["near_zero_flags"])
    assert not any("effective_energy" in f for f in a["near_zero_flags"])
    err = a["energy_identity_rel_error"]
    assert err["psi_avg_solid"]["max"] == pytest.approx(0.0, abs=1e-12)
    assert err["psi_avg_rve"]["median"] == pytest.approx(1 / 9)  # holed samples: area 8/9
    assert set(a["train_vs_test"]["inputs"]) == {"C11", "C12", "C22"}
    assert "Near-zero flags" in ins.render_markdown(a)


def test_audit_cli_writes_and_refuses_overwrite(ms_cache):
    assert ins.main(["--dataset", "2D_MultiScHypEl"]) == 0
    out = ms_cache.parent.parent / "audit" / "2D_MultiScHypEl"
    assert json.loads((out / "audit.json").read_text())["dataset"] == "2D_MultiScHypEl"
    assert (out / "audit.md").is_file()
    assert ins.main(["--dataset", "2D_MultiScHypEl"]) == 1
    assert ins.main(["--dataset", "2D_MultiScHypEl", "--overwrite"]) == 0

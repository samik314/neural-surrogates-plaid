"""Scalar identity check on a fake Tensile2d cache built so that known identities hold exactly."""

import json

import numpy as np
import pytest

from plaid_surrogates import identities as idn
from plaid_surrogates.data import splits
from plaid_surrogates.data.cache import CachedSample, sample_filename, save_sample, write_index
from plaid_surrogates.data.registry import get_spec
from plaid_surrogates.submission import write_split_submission

T2D = get_spec("Tensile2d")
NU = 0.3


def test_von_mises_forms():
    z = np.zeros(1)
    assert idn.von_mises_plane_stress(np.array([100.0]), z, z) == pytest.approx(100.0)
    assert idn.von_mises_plane_stress(z, z, np.array([10.0])) == pytest.approx(10 * np.sqrt(3))
    a = np.array([50.0])
    assert idn.von_mises_plane_stress(a, a, z) == pytest.approx(50.0)  # equibiaxial
    rng = np.random.default_rng(0)
    s = rng.normal(size=(3, 20))
    np.testing.assert_allclose(
        idn.von_mises_plane_strain(*s, nu=0.0), idn.von_mises_plane_stress(*s), rtol=1e-14
    )
    # nu = 0.5 with equal in-plane normal stresses and no shear is hydrostatic: zero
    assert idn.von_mises_plane_strain(a, a, z, nu=0.5) == pytest.approx(0.0, abs=1e-12)


def _sample(i, rng, outputs):
    n = 12
    top = np.array([0, 1, 2], dtype=np.int64)
    inputs = {p: float(rng.uniform(1, 2)) for p in T2D.input_scalars}
    inputs["P"] = float(rng.uniform(-50, -40))
    s = CachedSample(i, rng.random((n, 2)), np.array([[0, 1, 2]], np.int64), inputs)
    s.nodal_tags = {"Top": top, "Bottom": np.array([9, 10, 11], dtype=np.int64)}
    if outputs:
        f = {name: rng.normal(0, 100, n) for name in T2D.output_fields}
        f["U2"] = rng.uniform(0.001, 0.01, n)
        f["U2"][5] = f["U2"][top].max() + 1.0  # global max differs from the max over Top
        p = inputs["P"]
        f["sig22"][top] = [-p, -p - 1.0, -p - 2.0]  # max over Top == -P, mean over Top != -P
        s.output_fields = f
        s.output_scalars = {
            "max_U2_top": float(f["U2"][top].max()),
            "max_sig22_top": -p,
            "max_von_mises": float(
                idn.von_mises_plane_strain(f["sig11"], f["sig22"], f["sig12"], NU).max()
            ),
            "max_q": 1.0,
        }
    return s


@pytest.fixture
def cache(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    cache = tmp_path / "data" / "cache" / "Tensile2d"
    train_ids = [int(i) for i in rng.permutation(40)]
    test_ids = list(range(40, 46))
    info = {}
    for i in train_ids + test_ids:
        s = _sample(i, rng, outputs=i < 40)
        save_sample(cache, s)
        info[i] = {"file": sample_filename(i), "n_nodes": s.n_nodes, "has_outputs": i < 40}
    write_index(cache, {"hf_revision": "fake"}, {"train_500": train_ids, "test": test_ids}, info)
    monkeypatch.setenv("PLAID_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PLAID_RUNS_DIR", str(tmp_path / "runs"))
    assert splits.main(["--dataset", "Tensile2d"]) == 0
    return cache


def test_candidates_on_reference_fields(cache):
    r = idn.run_identity_check(cache, "Tensile2d")
    assert r["n_all"] == 40 and r["n_dev"] == 4 and r["n_fit"] == 36
    u2 = r["scalars"]["max_U2_top"]["candidates"]
    assert u2["max(U2) over Top"]["passes"] and u2["max(U2) over Top"]["dev_rrmse"] == 0.0
    assert not u2["max(U2) over all nodes"]["passes"]
    s22 = r["scalars"]["max_sig22_top"]["candidates"]
    assert s22["-P"]["passes"] and s22["max(sig22) over Top"]["passes"]
    assert not s22["mean(sig22) over Top"]["passes"]
    vm = r["scalars"]["max_von_mises"]["candidates"]
    strain = vm["max VM over nodes, plane strain (nu fitted)"]
    assert strain["nu"] == pytest.approx(NU, abs=1e-3) and strain["passes"]
    assert not vm["max VM over nodes, plane stress"]["passes"]
    for b in r["scalars"].values():
        assert b["training_mean_dev_rrmse"] > 0
    assert "| -P |" in idn.render_markdown(r)


def test_candidates_on_predicted_fields(cache, tmp_path):
    sp = splits.load_splits("Tensile2d", cache)
    fields, scalars = T2D.scored_fields, T2D.scored_scalars
    run = tmp_path / "runs" / "fake_run"
    preds = {}
    for i in sp["dev"]:
        s = _load(cache, i)
        preds[i] = {**{f: s.output_fields[f] for f in fields}, **{k: 1.0 for k in scalars}}
    write_split_submission(
        preds, cache, sp["dev"], fields, scalars, run / "predictions_dev_best.pkl", {}
    )
    r = idn.run_identity_check(cache, "Tensile2d", run)
    b = r["run"]["scalars"]
    assert b["max_U2_top"]["from_predicted_fields"]["max(U2) over Top"] == 0.0
    assert b["max_sig22_top"]["from_predicted_fields"]["-P"] == 0.0
    assert b["max_U2_top"]["as_predicted_by_run"] > 0.5  # constant 1.0 vs ~0.01


def _load(cache, i):
    from plaid_surrogates.data.cache import load_index, load_sample

    return load_sample(cache / load_index(cache)["samples"][str(i)]["file"])


def test_cli_writes_and_refuses_overwrite(cache):
    assert idn.main(["--dataset", "Tensile2d"]) == 0
    out = cache.parent.parent / "audit" / "Tensile2d"
    assert json.loads((out / "identities.json").read_text())["n_all"] == 40
    assert (out / "identities.md").is_file()
    assert idn.main(["--dataset", "Tensile2d"]) == 1
    assert idn.main(["--dataset", "Tensile2d", "--run", "no_such_run"]) == 1


def test_rve_average_of_psi_counts_voids_as_zero():
    from tests.test_inspect import grid

    pts, cells = grid(3, removed={(1, 1)})
    s = CachedSample(0, pts, cells, {})
    cand = idn.CANDIDATES["2D_MultiScHypEl"]["effective_energy"]["RVE average of psi"]
    assert cand({"psi": np.full(len(pts), 2.0)}, s, 0.0) == pytest.approx(2.0 * 8 / 9)

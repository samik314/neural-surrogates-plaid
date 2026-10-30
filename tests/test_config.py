from pathlib import Path

import pytest
import yaml

from plaid_surrogates.config import config_from_dict, config_to_dict, load_config
from plaid_surrogates.utils.paths import repo_root

TINY = repo_root() / "configs" / "tensile2d_transolver_tiny.yaml"


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_repo_config_loads_and_round_trips():
    cfg = load_config(TINY)
    assert cfg.dataset == "Tensile2d" and cfg.model.use_te is False
    assert isinstance(cfg.optim.lr, float) and cfg.optim.lr == pytest.approx(1e-3)
    assert config_from_dict(config_to_dict(cfg)) == cfg


@pytest.mark.parametrize("where", ["top", "model", "optim"])
def test_unknown_key_rejected(tmp_path, where):
    d = yaml.safe_load(TINY.read_text())
    (d if where == "top" else d[where])["typo_key"] = 1
    with pytest.raises(KeyError, match="typo_key"):
        load_config(_write(tmp_path, d))


def test_missing_key_rejected(tmp_path):
    d = yaml.safe_load(TINY.read_text())
    del d["model"]["use_te"]
    with pytest.raises(KeyError, match="use_te"):
        load_config(_write(tmp_path, d))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("precision",), "fp16"),
        (("optim", "schedule"), "linear"),
        (("model", "n_head"), 3),  # 64 % 3 != 0
        (("optim", "epochs"), 0),
        (("tag",), "has_underscore"),
        (("loss",), "l1"),
    ],
)
def test_invalid_values_rejected(tmp_path, path, value):
    d = yaml.safe_load(TINY.read_text())
    target = d
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, d))


def test_wrong_type_rejected(tmp_path):
    d = yaml.safe_load(TINY.read_text())
    d["model"]["n_layers"] = "two"
    with pytest.raises(TypeError, match="n_layers"):
        load_config(_write(tmp_path, d))


def test_scalars_mapping_must_be_str_to_str(tmp_path):
    d = yaml.safe_load(TINY.read_text())
    d["scalars"]["max_U2_top"] = 3
    with pytest.raises(TypeError, match="scalars"):
        load_config(_write(tmp_path, d))


def test_legacy_run_config_without_scalars_loads():
    d = config_to_dict(load_config(TINY))
    del d["scalars"], d["features"], d["loss"]  # runs saved before these keys existed
    cfg = config_from_dict(d)
    assert cfg.scalars == {} and cfg.features == "input_scalars"
    assert cfg.loss == "standardized_mse"


def test_int_accepted_for_float(tmp_path):
    d = yaml.safe_load(TINY.read_text())
    d["optim"]["grad_clip"] = 1
    cfg = load_config(_write(tmp_path, d))
    assert isinstance(cfg.optim.grad_clip, float)


def test_unknown_feature_set_rejected(tmp_path):
    d = yaml.safe_load(TINY.read_text())
    d["features"] = "input_scalars+magic"
    with pytest.raises(ValueError, match="features"):
        load_config(_write(tmp_path, d))


def test_geoligml_differs_from_parent_only_in_tag_and_loss():
    cfgs = repo_root() / "configs"
    a = config_to_dict(load_config(cfgs / "2dmultischypel_transolver_geolig.yaml"))
    b = config_to_dict(load_config(cfgs / "2dmultischypel_transolver_geoligml.yaml"))
    assert {k for k in a if a[k] != b[k]} == {"tag", "loss"} and b["loss"] == "metric"


@pytest.mark.parametrize(
    ("arm", "parent", "changed"),
    [
        ("geoligmls64", "geoligml", {"slice_num": 64}),
        ("geoligmlw256", "geoligml", {"n_hidden": 256, "n_head": 8}),
        ("geoligmls128", "geoligmls64", {"slice_num": 128}),
        ("geoligmls256", "geoligmls128", {"slice_num": 256}),
    ],
)
def test_ablation_configs_change_only_the_model_knob(arm, parent, changed):
    cfgs = repo_root() / "configs"
    a = config_to_dict(load_config(cfgs / f"2dmultischypel_transolver_{parent}.yaml"))
    b = config_to_dict(load_config(cfgs / f"2dmultischypel_transolver_{arm}.yaml"))
    assert {k for k in a if a[k] != b[k]} == {"tag", "model"} and b["tag"] == arm
    diff = {k: b["model"][k] for k in a["model"] if a["model"][k] != b["model"][k]}
    assert diff == changed


@pytest.mark.parametrize(
    ("arm", "section", "changed"),
    [
        ("geoligmls256e400", "optim", {"epochs": 400}),
        ("geoligmls256l8", "model", {"n_layers": 8}),
        ("geoligmls256pp", "model", {"plus": True}),
    ],
)
def test_probe_configs_change_one_setting(arm, section, changed):
    cfgs = repo_root() / "configs"
    a = config_to_dict(load_config(cfgs / "2dmultischypel_transolver_geoligmls256.yaml"))
    b = config_to_dict(load_config(cfgs / f"2dmultischypel_transolver_{arm}.yaml"))
    assert {k for k in a if a[k] != b[k]} == {"tag", section} and b["tag"] == arm
    assert {k: b[section][k] for k in a[section] if a[section][k] != b[section][k]} == changed

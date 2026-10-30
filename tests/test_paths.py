import os
import shutil
import subprocess
import sys

import pytest

from plaid_surrogates.utils import paths


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("PLAID_DATA_DIR", "PLAID_RUNS_DIR", "HF_HOME"):
        monkeypatch.delenv(var, raising=False)


def test_repo_root_has_pyproject():
    assert (paths.repo_root() / "pyproject.toml").is_file()


def test_defaults_are_inside_repo():
    root = paths.repo_root()
    assert paths.data_dir() == root / "data"
    assert paths.runs_dir() == root / "runs"
    assert paths.hf_home() == root / "data" / "hf_cache"


def test_env_vars_override(monkeypatch, tmp_path):
    monkeypatch.setenv("PLAID_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("PLAID_RUNS_DIR", str(tmp_path / "r"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert paths.data_dir() == tmp_path / "d"
    assert paths.runs_dir() == tmp_path / "r"
    assert paths.hf_home() == tmp_path / "hf"


def test_paths_with_spaces(monkeypatch, tmp_path):
    spaced = tmp_path / "Brown Dropbox" / "data"
    monkeypatch.setenv("PLAID_DATA_DIR", str(spaced))
    assert paths.data_dir() == spaced


def test_set_hf_home_env_exports_default(monkeypatch):
    monkeypatch.delitem(sys.modules, "datasets", raising=False)
    monkeypatch.delitem(sys.modules, "huggingface_hub", raising=False)
    path = paths.set_hf_home_env()

    assert os.environ["HF_HOME"] == str(path) == str(paths.repo_root() / "data" / "hf_cache")


def test_gitignore_keeps_code_and_drops_data():
    root = paths.repo_root()
    if shutil.which("git") is None or not (root / ".git").exists():
        pytest.skip("not a git checkout")

    def ignored(rel: str) -> bool:
        # check-ignore exits 0 if the path is ignored, 1 if not; --no-index works for new files.
        out = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "-q", "--no-index", rel], check=False
        )
        return out.returncode == 0

    for code in ("src/plaid_surrogates/data/cache.py", "src/plaid_surrogates/data/__init__.py"):
        assert not ignored(code), f"{code} is gitignored"
    for junk in ("data/cache/Tensile2d/samples/000000.npz", "data/hf_cache/x", "runs/r1/ckpt"):
        assert ignored(junk), f"{junk} is NOT gitignored"


def test_set_hf_home_env_refuses_after_hf_import(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", object())
    with pytest.raises(RuntimeError):
        paths.set_hf_home_env()

import json
import shutil
import subprocess

import pytest

from plaid_surrogates.utils.provenance import compare_to_lock, git_info, write_provenance

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git(repo, *args):
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", "a.py")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_git_info_outside_repo(tmp_path):
    info = git_info(tmp_path)
    assert info == {"commit": None, "dirty": None, "diff": None, "untracked": None}


@needs_git
def test_git_info_clean_then_dirty(repo):
    clean = git_info(repo)
    assert len(clean["commit"]) == 40 and clean["dirty"] is False and clean["diff"] == ""

    (repo / "a.py").write_text("x = 2\n")
    (repo / "new.py").write_text("")
    dirty = git_info(repo)
    assert dirty["dirty"] is True
    assert "+x = 2" in dirty["diff"]
    assert dirty["untracked"] == ["new.py"]


@needs_git
def test_write_provenance_saves_patch_only_when_dirty(repo, tmp_path_factory):
    run_clean = tmp_path_factory.mktemp("run_clean")
    prov = write_provenance(run_clean, repo, extra={"seed": 0})
    assert not (run_clean / "git_diff.patch").exists()
    assert (run_clean / "pip_freeze.txt").stat().st_size > 0
    saved = json.loads((run_clean / "provenance.json").read_text())
    assert saved["seed"] == 0 and saved["git_commit"] == prov["git_commit"]
    assert saved["environment"]["packages"]["torch"] is not None

    (repo / "a.py").write_text("x = 3\n")
    run_dirty = tmp_path_factory.mktemp("run_dirty")
    write_provenance(run_dirty, repo)
    assert "+x = 3" in (run_dirty / "git_diff.patch").read_text()


def test_compare_to_lock_ignores_comments_options_and_case():
    lock = ["# header", "--extra-index-url https://x", "", "Torch==2.14.0", "numpy==2.5.3"]
    assert compare_to_lock(["torch==2.14.0", "numpy==2.5.3"], lock)["matches"]

    res = compare_to_lock(["torch==2.14.0", "numpy==2.6.0", "extra==1"], lock)
    assert not res["matches"]
    assert res["not_in_lock"] == ["extra==1", "numpy==2.6.0"]
    assert res["missing_from_env"] == ["numpy==2.5.3"]

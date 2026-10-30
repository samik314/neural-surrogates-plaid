import datetime as dt

import pytest

from plaid_surrogates.utils import runs

NOW = dt.datetime(2026, 9, 29, 18, 5)

LOG = """# Experiment log

| date | run id | status | dataset | model | parent run | key change | seed/fold | params | wall time | machine / GPU | dev total_error | worst output (RRMSE) | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

## Storage checkpoints

| date | milestone | data/ | runs/ | HF cache | env | total |
|---|---|---|---|---|---|---|
| 2026-09-29 | M0a | 1.6G | 0 | 1G | - | 2.6G |
"""  # noqa: E501


def test_run_name():
    assert runs.run_name("Tensile2d", "transolver", "base", 0, now=NOW) == (
        "20260929-1805_Tensile2d_transolver_base_s0"
    )
    assert runs.run_name("Tensile2d", "transolver", "base", 1, fold=2, now=NOW).endswith("_s1_f2")


def test_make_run_dir_never_reuses(tmp_path):
    a = runs.make_run_dir("T", "m", "t", 0, root=tmp_path, now=NOW)
    b = runs.make_run_dir("T", "m", "t", 0, root=tmp_path, now=NOW)
    assert a.is_dir() and b.is_dir() and a != b
    assert b.name == a.name + "-1"


def test_rows_go_into_the_run_table_in_order(tmp_path):
    log = tmp_path / "LOG.md"
    log.write_text(LOG)
    runs.append_table_row(log, runs.EXPERIMENT_COLUMNS, {"run id": "first", "status": "ok"})
    runs.append_table_row(log, runs.EXPERIMENT_COLUMNS, {"run id": "second", "notes": "a|b"})
    lines = log.read_text().splitlines()
    k1 = next(k for k, line in enumerate(lines) if "| first |" in line)
    k2 = next(k for k, line in enumerate(lines) if "| second |" in line)
    storage = lines.index("## Storage checkpoints")
    assert k1 < k2 < storage
    assert "a\\|b" in lines[k2]
    assert lines[k2].count(" | ") == len(runs.EXPERIMENT_COLUMNS) - 1  # escaped pipe isn't a cell
    assert "| 2026-09-29 | M0a |" in lines[-1]  # storage table untouched


def test_unknown_column_and_missing_header(tmp_path):
    log = tmp_path / "LOG.md"
    log.write_text(LOG)
    with pytest.raises(ValueError, match="unknown"):
        runs.append_table_row(log, runs.EXPERIMENT_COLUMNS, {"bogus": 1})
    with pytest.raises(ValueError, match="header"):
        runs.append_table_row(log, runs.SUBMISSION_COLUMNS, {"run id": "x"})


def test_real_logs_have_the_expected_headers():
    """The column tuples must match the headers in the committed LOG.md files exactly."""
    root = runs.repo_root()
    for path, cols in (
        (root / "experiments" / "LOG.md", runs.EXPERIMENT_COLUMNS),
        (root / "submissions" / "LOG.md", runs.SUBMISSION_COLUMNS),
    ):
        assert "| " + " | ".join(cols) + " |" in path.read_text().splitlines()


def _append_rows(path, worker, n):
    for k in range(n):
        runs.append_table_row(
            path, runs.EXPERIMENT_COLUMNS, {"run id": f"w{worker}-{k}", "status": "ok"}
        )


def test_parallel_appends_keep_every_row(tmp_path):
    import multiprocessing as mp

    log = tmp_path / "LOG.md"
    log.write_text(LOG)
    procs = [mp.Process(target=_append_rows, args=(log, w, 10)) for w in range(6)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        assert p.exitcode == 0
    rows = [line for line in log.read_text().splitlines() if "| ok |" in line]
    assert len(rows) == 60
    expected = {f"w{w}-{k}" for w in range(6) for k in range(10)}
    assert expected == {r.split("|")[2].strip() for r in rows}
    assert log.read_text().index("## Storage checkpoints") > log.read_text().index(rows[-1])


def test_every_dataset_names_its_leaderboard():
    from plaid_surrogates.data.registry import REGISTRY

    for name, spec in REGISTRY.items():
        assert spec.leaderboard.startswith("https://huggingface.co/spaces/PLAIDcompetitions/"), name
    assert len({s.leaderboard for s in REGISTRY.values()}) == len(REGISTRY)  # no shared board


def test_run_name_full_and_fold_are_exclusive():
    import datetime as dt

    now = dt.datetime(2026, 10, 15, 9, 5)
    assert runs.run_name("D", "m", "t", 0, now=now, full=True) == "20261015-0905_D_m_t_s0_full"
    assert runs.run_name("D", "m", "t", 0, 2, now=now) == "20261015-0905_D_m_t_s0_f2"
    with pytest.raises(ValueError, match="either"):
        runs.run_name("D", "m", "t", 0, 2, now=now, full=True)

# Experiment log

`train.py`, `baselines.py` and `ensemble.py` append one row per run here,
including failed and killed runs (docs/methods.md §5). Each run's folder under
`runs/` holds the full record.

This log starts empty in this repository. The runs behind the reported results
are summarized in
[DSG-002](../docs/design/DSG-002-data-2d-multischypel.md), and each has its
config in `configs/`.

| date | run id | status | dataset | model | parent run | key change | seed/fold | params | wall time | machine / GPU | dev total_error | worst output (RRMSE) | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

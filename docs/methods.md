# Methods

How the data is handled, how models are scored and selected, and the rules the
code follows. Code comments point here as `docs/methods.md §N`.

## §1 Data and cache

Two PLAID benchmarks, loaded from Hugging Face with `pyplaid` 0.1.15:

| Dataset | Train / test | Mesh | Inputs | Scored outputs |
|---|---|---|---|---|
| `Tensile2d` | `train_500` (500) / `test` (200) | ~9.4k nodes, triangles | scalars `P, p1..p5` + geometry | fields `U1, U2, sig11, sig22, sig12`; scalars `max_von_mises, max_U2_top, max_sig22_top` |
| `2D_MultiScHypEl` | `DOE_train` (764) / `DOE_test` (376) | ~5.7k nodes, triangles, unit-square RVE with 5–8 voids, periodic mesh | scalars `C11, C12, C22` + geometry | fields `u1, u2, P11, P12, P22, P21, psi`; scalar `effective_energy` |

Test outputs are withheld; scores come from the public leaderboards.

`data/convert.py` converts each dataset once into a cache: one `.npz` per sample
(`points`, `cells`, `input_scalars`, `output_fields`, `output_scalars`,
`nodal_tags`) plus `index.json`. Invariants, each covered by a test:

- **Split order:** `index.json` stores each split's IDs exactly in
  `problem.get_split()` order; nothing downstream sorts them.
- **Node order:** a sample's nodes keep their original order from cache to
  submission file.
- **No test outputs:** test samples are cached without outputs.

Tags are collapsed into fixed semantic categories (2D_MultiScHypEl: outer
boundary `Ext_bound`, hole boundary `Holes`, interior; Tensile2d: `Top`,
`Bottom`), never one feature per raw tag. Findings of the data audit
(`data/inspect.py`) are in `docs/design/DSG-002-data-2d-multischypel.md`.

## §2 Metric and validation

**Metric** (identical to the leaderboards'). For each scored field f over n
samples, sample i having N_i nodes:

    RRMSE_f = sqrt( (1/n) Σ_i ||f_ref,i − f_pred,i||² / (N_i · ||f_ref,i||∞²) )

and for each scored scalar s: `RRMSE_s = sqrt( (1/n) Σ_i (s_ref,i − s_pred,i)² /
s_ref,i² )`. `total_error` is the mean of the 8 RRMSEs. It is computed in
float64, in physical units, with each sample normalized by its own max |f|.

Checks: unit tests on hand-computed cases; the scalar part against
`plaid.post.metrics`; and both parts against the 2D leaderboard's own published
scoring function, which reproduces our dev scores to a relative 1e-12
(`tests/integration/test_board_metric_parity.py`).

**Validation.**
- *Dev split:* 10 % of the official training split, seeded and stratified (2D:
  number of holes × load magnitude), saved in `data/splits/<dataset>.json`. Used
  for everyday decisions.
- *Noise floor:* configs are compared on 3 seeds. A difference counts if it
  exceeds twice the larger seed standard deviation of the two arms, and at least
  0.001.
- *5-fold CV:* stratified folds over the whole training split, only for the
  final candidates; compared on the last checkpoint so the validation fold plays
  no part in checkpoint choice.
- *Checkpoints:* the best checkpoint is the one with the lowest exact dev
  `total_error`, never the lowest training loss.
- *Final retrain* (`train.py --full`): all official training samples,
  normalization refit, no validation set, the number of optimizer updates frozen
  at the validated runs' (so more samples per epoch means fewer epochs), and the
  final checkpoint.
- Leaderboard scores are reported, never used to choose anything.

## §3 Submission format

A pickle of a list with one dict per test sample, in the split's stored order.
Each dict maps every scored field to a 1-D float64 array on that sample's nodes
(original order) and every scored scalar to a float. `submission.py` validates
sample count, keys, array lengths and finiteness before writing, and writes
`submission_meta.json` (run, commit, config, checksum) and `LOG_ROW.md`, which
starts with the URL of the leaderboard the file belongs to. A round-trip test
writes the dev split with the same writer and requires `metrics.py` to score it
exactly as during training.

## §4 Model, features and losses

**Model:** PhysicsNeMo's `Transolver` (2.1.1), built with every constructor
argument explicit (`use_te=False` unless Transformer Engine is installed).
Coordinates go into the embedding only; everything else is a per-node input
feature. Training uses whole meshes (batch size 1) with gradient accumulation;
inference is always on the whole mesh.

**Input features** (fixed width, added one at a time as logged ablations): input
scalars broadcast to every node; the one-hot tag category; distances to the
outer and hole boundaries, divided by the RVE size; the distance to the
second-nearest hole. Statistics come from the training portion only.

**Targets:** each field standardized with training-portion statistics;
predictions are mapped back to physical units before any metric.

**Losses** (`losses.py`):
1. Standardized MSE (baseline).
2. *Metric-consistent loss:* per sample and field, the squared error divided by
   that sample's max |f|² (computed exactly from the standardized error),
   averaged over fields with weights 1 / (running RRMSE), detached. This follows
   the gradient of the metric itself.

**Scalars:** each scored scalar comes from the training mean or from a formula
applied to the predicted fields. A formula counts as an exact identity if, on
the reference fields, its largest relative error over the training split is at
most 1e-3 (`identities.py`). Tensile2d: `max_U2_top`, `max_sig22_top` and
`max_von_mises` (plane-stress form) are exact nodal identities. 2D_MultiScHypEl:
`effective_energy` = RVE-area average of predicted `psi` (P1 integration over
the triangles). It is not exact (relative error up to 5.9 % with reference
`psi`) but is used: with reference `psi` its dev RRMSE is 0.0177, against 0.1117
for the training mean.

## §5 Run records

Every run gets `runs/<YYYYMMDD-HHMM>_<dataset>_<model>_<tag>_s<seed>[_f<fold> |
_full]/` with the resolved config, seed, split or fold, git commit (and
`git_diff.patch` if the tree was dirty), environment snapshot (package versions,
GPU, precision, `pip freeze`), Slurm job ID, normalization statistics, training
curves (CSV), best and last checkpoints, predictions, and per-output metrics.
Every run, including failed and killed ones, is appended to a run log. Seeds fix
Python, NumPy and PyTorch.

## §6 Design rules

- **Only `data/convert.py` imports `plaid`.** Everything downstream reads the
  cache, so training, prediction and scoring run on machines without PLAID.
  Integration tests may import it.
- **Configs are strict:** YAML loaded into frozen dataclasses; an unknown or
  missing key is an error, so no value hides in code.
- **float32 first;** bf16 only after measuring that it helps (at this model size
  it didn't).
- **On a cluster, every heavy step is a Slurm job** (`scripts/slurm/`).

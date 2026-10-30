# DEC-001 — Use PhysicsNeMo 2.1.1 (not 2.2.x) in a single env

- **Date:** 2026-09-29
- **Status:** accepted

## Context

The plan pinned `nvidia-physicsnemo==2.2.2` and `pyplaid==0.1.15`. They can't be
installed together:

- `nvidia-physicsnemo` 2.2.0–2.2.2 requires `fsspec>=2026.4.0`.
- `pyplaid` 0.1.15 requires `datasets<5`, and every `datasets` 4.x (latest
  4.8.5) requires `fsspec<=2026.2.0`.

pip then backtracks `datasets` to 2.14.4, which needs a pyarrow with no Python
3.12 wheel, so the build fails. This is a metadata conflict, so it happens on
every platform, the cluster included. PhysicsNeMo 2.1.x has no `fsspec` floor.

## Options

1. **PhysicsNeMo 2.1.1, one env** (chosen).
2. Two envs: `plaid_convert` (pyplaid + datasets) and `plaid_env` (PhysicsNeMo
   2.2.2). This is clean, because Rule 5 already confines `plaid` to
   `convert.py`, but there are two envs to maintain and switch between.
3. Keep 2.2.2 and force `fsspec` past one package's declared bounds. This leaves
   an env that `pip check` reports as broken. Rejected.

## What 2.1.1 → 2.2.2 changes (checked by diffing the two wheels)

- `models/transolver/transolver.py`: 24 changed lines. Docstrings, plus a
  reshape fix for embeddings on **structured** grids only. We use
  `structured_shape=None`, so there's no effect.
- `nn/module/physics_attention.py`: an import move, and a dropout-placement
  change in the Transformer Engine path only. No effect with `use_te=False`.
- `models/meshgraphnet/`: identical.
- Only in 2.2: **GeoTransolver, FLARE**, several experimental models, and extra
  `physicsnemo.mesh` utilities (SDF, sampling, deformation/remeshing).
- A hardened archive-path check in `core/module.py` (`.mdlus` loading). Only
  relevant for untrusted model files; we only load our own checkpoints.

## Consequences

- Transolver (incl. `plus=True`) and MeshGraphNet are available unchanged for
  M0–M3.
- GeoTransolver/FLARE aren't available. If we want them later, choose then
  between option 2 and newer `datasets`/`pyplaid` releases, if the conflict has
  been fixed upstream by that time.
- `fsspec` and `s3fs` are pinned to 2026.2.0.

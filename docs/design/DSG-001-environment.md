# DSG-001 — Environment

One mamba env, `plaid_env`, for everything: conversion (PLAID), training,
prediction, metrics. Mamba supplies only Python and pip; every package comes
from pip. Mixing conda-forge torch with pip-installed PhysicsNeMo tends to
break.

Last verified: 2026-09-29, Linux laptop (RTX 2000 Ada Laptop GPU, 8 GB, driver
550.163.01); 2026-09-30, a Slurm GPU cluster (RTX A6000 / A5000, driver
575.57.08; see Per-machine notes).

## Pins and why

| Package | Version | Why |
|---|---|---|
| Python | 3.12 | pyplaid 0.1.x needs >=3.11,<3.14; PhysicsNeMo needs >=3.11,<3.15 |
| `torch` / `torchvision` | 2.14.0+cu126 / 0.29.0+cu126 | cu126 runs on drivers >= 525. Both **must** come from the same index (see Pitfalls) |
| `nvidia-physicsnemo` | 2.1.1 | 2.2.x can't coexist with pyplaid 0.1.15: `docs/decisions/DEC-001-physicsnemo-2.1.1.md` |
| `pyplaid` | 0.1.15 | Has the HF loader and `plaid.post.metrics` (both removed in 1.0) |
| `datasets` | 4.8.5 (`<5`) | Required by pyplaid 0.1.15 |
| `fsspec`, `s3fs` | 2026.2.0 | `datasets` 4.x caps `fsspec<=2026.2.0`; pinning both stops resolver backtracking |

## Recipe

Option A: step by step (the order that was actually run and verified):

```bash
mamba create -n plaid_env python=3.12 pip -y
mamba activate plaid_env

# 1. torch + torchvision together, from the CUDA index matching the driver
pip install "torch==2.14.0" "torchvision==0.29.0" \
    --index-url https://download.pytorch.org/whl/cu126

# 2. everything else
pip install "pyplaid==0.1.15" "nvidia-physicsnemo==2.1.1" "datasets<5" \
            "fsspec==2026.2.0" "s3fs==2026.2.0" pytest ruff pyyaml matplotlib
```

Option B: one shot from the repo files. Verified from scratch on the cluster
(2026-09-30): in a fresh `python=3.12 pip` env, `pip install -r
requirements.lock` reproduces the lock exactly.

```bash
mamba env create -f environment.yml            # top-level pins only
# or, for the exact package set:
mamba create -n plaid_env python=3.12 pip -y && mamba activate plaid_env
pip install -r requirements.lock
```

## Verify

```bash
python -W ignore -c "
from importlib.metadata import version as v
import torch, torchvision
from physicsnemo.models.transolver import Transolver
from plaid.bridges.huggingface_bridge import huggingface_dataset_to_plaid
from plaid.post.metrics import compute_rRMSE_RMSE
print(torch.__version__, torchvision.__version__,
      v('nvidia-physicsnemo'), v('pyplaid'), v('datasets'))
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''
print('cuda', torch.cuda.is_available(), gpu)"
pip check
```

Expected on the laptop: `2.14.0+cu126 0.29.0+cu126 2.1.1 0.1.15 4.8.5`, `cuda
True ...`, and `No broken requirements found.`

Smoke-test result (Transolver, n_layers=4, n_hidden=128, n_head=4, slice_num=32,
`use_te=False`, ~0.78M params, random 9.4k-node input, batch 1, AdamW step):
**fp32 26 ms/step, 409 MiB peak; bf16 autocast 11 ms/step, 270 MiB peak.**

## Pitfalls hit (and fixes)

1. **pip tries to build `pyarrow` from source → fails (`Cannot import
   'setuptools.build_meta'`).** Cause: PhysicsNeMo 2.2.x needs
   `fsspec>=2026.4.0`, `datasets<5` needs `fsspec<=2026.2.0`; pip backtracks
   `datasets` to 2.14 (2023), whose old pyarrow has no cp312 wheel. Fix:
   PhysicsNeMo 2.1.1 and pin `fsspec`/`s3fs` 2026.2.0. General rule: **if pip
   starts compiling something, the resolver has backtracked — stop and read the
   pins.**
2. **`RuntimeError: operator torchvision::nms does not exist`** when importing
   PhysicsNeMo models. Cause: torchvision pulled in from plain PyPI (a different
   CUDA build) on top of a cu126 torch. Fix: `pip install --force-reinstall
   --no-deps "torchvision==0.29.0" --index-url
   https://download.pytorch.org/whl/cu126`. Prevention: install torch and
   torchvision in the same command from the same index (step 1).
3. **pyplaid 0.1.15 deprecation warnings** (not an error):
   `get_input_scalars_names`, `get_output_fields_names` etc. warn "use
   `get_in_features_identifiers` … will be removed in v0.2.0" (seen during
   conversion and in the integration tests). We are pinned to 0.1.15, so nothing
   to change; revisit only if pyplaid is ever upgraded (the 1.0 API change is
   the bigger issue there).
4. **Transolver's constructor defaults** in 2.1.1 are `use_te=True`,
   `n_hidden=256`, `n_head=8`. Always pass them explicitly; `use_te=False`
   unless Transformer Engine is installed on an NVIDIA GPU.

## Per-machine notes

- **Linux laptop:** as above.
- **Slurm cluster:** verified 2026-09-30. Every heavy step runs as a batch job;
  the install is `scripts/slurm/setup_env.sh` (CPU partition, 4 CPUs, 16 GB,
  about 11 min). Recipe inside it:

  ```bash
  module load miniforge3          # or any conda/mamba install
  source "${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh"
  # explicit prefix, e.g. ~/.conda/envs/plaid_env
  mamba create -y -p "${PLAID_ENV}" python=3.12 pip
  conda activate "${PLAID_ENV}"
  cd neural-surrogates-plaid
  # --no-cache-dir keeps ~3-4 GB of wheels out of ~/.cache/pip
  pip install --no-cache-dir -r requirements.lock
  pip install --no-cache-dir -e . --no-deps
  pip check && pytest -q
  ```

  - Result: `pip freeze` is **identical to `requirements.lock`** (all 128 pins;
    `packaging` 26.3 arrives with pip from conda-forge and prints as `@
    file://`), so there is no separate cluster lock. Env size **7.7 GB**.
    Compute nodes have outbound internet (PyPI and download.pytorch.org at ~117
    MB/s). Only `antlr4-python3-runtime` builds locally (pure Python, no wheel
    on PyPI); anything else compiling means the resolver backtracked (Pitfall
    1).
  - GPU nodes run **driver 575.57.08 (CUDA ≤ 12.9)**, so the cu126 wheels work
    unchanged. GPU check on an RTX A6000 (48 GB, cc 8.6): torch 2.14.0+cu126,
    CUDA runtime 12.6, cuDNN 9.10.2, `is_bf16_supported()` True. Smoke test as
    above (0.78M params, 9.4k nodes, batch 1): **fp32 17.7 ms/step, 409 MiB
    peak; bf16 autocast 22.4 ms/step, 282 MiB peak.** bf16 is *slower* than fp32
    here: at this model size the step on an A6000 is bound by kernel-launch
    overhead, not arithmetic, and autocast's extra cast kernels cost more than
    the tensor cores save (the weaker laptop GPU is compute-bound on the same
    kernels, hence its 2.4× bf16 speedup). Re-measure bf16 at the real model
    size before assuming a speedup; if steps stay launch-bound, CUDA graphs /
    `torch.compile` matter more than precision.
  - Harmless stderr: `libmamba Could not open lockfile '.../pkgs/pkgs.lock'` (a
    read-only shared package cache; mamba falls back to `~/.conda/pkgs`), and on
    CPU nodes `Warp CUDA warning: Could not find or load the NVIDIA CUDA
    driver`.
- **Mac:** not set up. `pycgns` has no macOS arm64 wheel; there, install without
  pyplaid and read the converted cache (docs/methods.md §6).

## Updating the environment

Any package change: install it, run the Verify block and `pytest -q`, then
regenerate `requirements.lock` (`pip freeze`, keeping the header and the
`--extra-index-url` line, and replacing any `@ file://` lines with
`==<version>`) and `environment.yml`, and note the change here.

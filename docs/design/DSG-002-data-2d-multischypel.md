# DSG-002 — 2D_MultiScHypEl: data audit and experiments

Audit: `python -m plaid_surrogates.data.inspect --dataset 2D_MultiScHypEl`
(2026-10-05), cache at HF revision `68ae0593`. Full report:
`<PLAID_DATA_DIR>/audit/2D_MultiScHypEl/ audit.md` (outside git). Split:
`data/splits/2D_MultiScHypEl.json`.

## Facts

| Item | Finding |
|---|---|
| Samples | 1140: `DOE_train` 764, `DOE_test` 376 (outputs withheld); none unassigned |
| Domain | unit square RVE in every sample |
| Mesh | linear triangles; 4.1k-7.1k nodes (median 5.7k); no inverted or degenerate cells |
| Holes | **voids**: interior free-edge loops == inclusion count (5-8) in all 1140 samples; no hole crosses the RVE edge; void fraction 3.9-6.2 % |
| Periodicity | node positions match on opposite edges (x and y) in every sample |
| Tags | `Ext_bound` (outer edge, 160 nodes, every sample) and `Holes` (hole boundaries, 160-256 nodes) |
| Inputs | C11, C22 in [0.8, 1.2], C12 in [-0.2, 0.2]: right Cauchy-Green tensor around identity, tension and compression |
| Output maxima | per-sample max abs: u1, u2 0.87-1.30; P11, P22 2.5e6-7.8e6; P12, P21 1.2e6-3.7e6; psi 8.8e6-4.5e7; effective_energy 9.0e5-1.6e6. No near-zero values |
| Train vs test | same design of experiments: KS <= 0.053 for C11, C12, C22, node count, inclusion count |

## What it means for modeling

- **Tag schema is fixed:** three semantic categories, outer boundary / hole
  boundary / interior, as a one-hot per node. The raw tag set is identical
  across samples, so no per-sample tag remapping is needed.
- **Geometry encoding:** the holes are traction-free voids, so the network must
  see where they are. Candidates, one ablation each: the hole-boundary one-hot,
  distance to the nearest hole boundary, and distance to the outer boundary.
- **Periodic mesh:** consistent with periodic boundary conditions. Periodic
  coordinate features (sin/cos of 2πx, 2πy) are a natural ablation alongside raw
  coordinates.
- **Metric-consistent loss is safe:** no field or scalar has near-zero
  per-sample maxima, so the δ stabilizer in the loss can stay tiny.
- **effective_energy vs area average of psi is not exact:** relative error
  median 1.4 %, max 5.9 % when dividing by the RVE area (psi = 0 in the voids),
  and median 5.8 % when dividing by the solid area. RVE-area averaging is the
  right one physically. The gap is likely because nodal psi is a recovered
  quadrature quantity, so its P1 integral is inexact near the holes. This fails
  the 1e-3 bar for an exact identity (docs/methods.md §4), so it is used as an
  approximate predictor, not an identity (dev RRMSE below).
- **Displacements (`python -m plaid_surrogates.data.kinematics`):**
  - u = 0 exactly at the origin corner in all 764 training samples, so there is
    no free rigid offset to learn.
  - **KUBC confirmed:** on the outer boundary u is exactly affine, u = A X (fit
    residual ≤ 2.7e-8, offset ≤ 8.6e-12). Inside, the fluctuation is large:
    max |u − A X − c| 0.24–0.56, rms 0.10–0.23, against an affine part of max |A
    X| 1.1–2.2.
  - **A does not match the input C** as described in the paper (C = Fᵀ F with
    F = I + A): the boundary A gives max |C(I + A) − C_input| ≈ 3. No single
    unit scale fixes it: the best s for (I + A/s)ᵀ(I + A/s) ≈ C ranges from 7 to
    the grid edge at 100, with a 0.14 median residual even then. So u is not
    stored as the displacement of F with Fᵀ F = C, in any one unit. Training is
    unaffected (standardization absorbs scale), but the map C → A is unknown.
  - **Lever for later:** A is exact on the boundary and fixed per sample. If A
    is a deterministic function of (C11, C12, C22), the affine part can be
    computed (or learned from 3 inputs) and the network trained on the
    fluctuation u − A X only. Next diagnostic: regress the 4 boundary entries of
    A on C.
- **effective_energy:** training mean dev RRMSE 0.1117; RVE-area average of the
  reference psi 0.0177 (relative error over all 764 training samples: median 1.4
  %, p95 4.1 %, max 5.9 %, so not exact). Used as the scalar predictor in the 2D
  configs; a learned correction is the obvious follow-up if it dominates.
- **Feature cost:** the distance features take ~0.35 s per sample (exact
  point-to-segment over ~5.7k nodes × the boundary edges), so they are computed
  once per sample and cached. On a typical sample 3.7 % of nodes are
  outer-boundary, 3.7 % hole-boundary, 92.7 % interior.

## Split

Strata: inclusion count (categorical, 5-8) × 3 quantile bins of ||C − I||_F
(size of the imposed deformation, which sets the stress and energy scale). 12
cells of 53-75 samples; dev 76 (5-7 per cell), train 688, five folds of 152-153
over all 764. KS dev vs train: C11 0.104, C12 0.116, C22 0.099 (5 % critical
value ≈ 0.16).

## Geometry-encoding diagnostic (2026-10-07)

Two arms that differ only in `features`, 3 seeds each, RTX A5000, 200 epochs.

| Arm | Features | Dev total_error (s0 / s1 / s2) | Mean ± std |
|---|---|---|---|
| base | input scalars | 0.1037 / 0.1057 / 0.1027 | 0.1040 ± 0.0015 |
| geo | + one-hot (outer / hole / interior) + distance to outer and hole boundaries | 0.0334 / 0.0322 / 0.0364 | 0.0340 ± 0.0022 |

- Without geometry the model underfits (final train loss ~0.3 vs ~0.03) and its
  dev error peaks mid-run: it cannot tell where the traction-free holes are.
  Geometry is the central input.
- Error vs distance to the nearest hole (mean |pred − ref| / max|ref| over dev
  nodes and fields): base 0.19 within 0.01 of a hole vs 0.022 beyond 0.32; geo
  0.058 vs 0.007. The features lower the error everywhere by ~3× but leave the
  ~8× concentration at the holes.
- Error maps (`<run>/figures/`): the largest errors sit in narrow ligaments
  between closely spaced holes (stress concentrations); the worst dev samples
  (1130, 758, 1110) are clustered layouts in every seed.
- Geo arm per output: P12, P21 0.053–0.062 (worst); P11, P22, psi ≈ 0.033; u1,
  u2 ≈ 0.017; effective_energy (RVE average of predicted psi) 0.022–0.025, near
  the 0.0177 the identity reaches with reference psi.
- Next, one change at a time against the geo arm (bar ≈ 0.004): a ligament
  feature (distance to the second-nearest hole), the metric-consistent loss, the
  fluctuation target u − A X (if A is a function of C), resolution (slices,
  Transolver++).

### Ablation (a): ligament feature (2026-10-07)

The geo features plus the distance to the second-nearest hole
(`input_scalars+tags+dist+lig`).

| Arm | Dev total_error (s0 / s1 / s2) | Mean ± std |
|---|---|---|
| geo | 0.0334 / 0.0322 / 0.0364 | 0.0340 ± 0.0021 |
| geolig | 0.0321 / 0.0312 / 0.0337 | 0.0323 ± 0.0013 |

- −0.0017, below the 0.004 bar. Every output and every same-seed pair improves
  slightly (P12, P21 0.057 → 0.054; paired t ≈ 3.3, 2 dof), but the result is
  not significant.
- The near-hole error barely moves (0.060 → 0.058 within 0.01 of a hole), and
  the worst dev samples are unchanged (1130, 758, 1110, 1003). Distance to the
  second hole is not what the model is missing in clustered ligaments.
- Kept as the parent for the next ablations: free (+256 params, +15 MiB peak)
  and never worse.

### Ablation (b): metric-consistent loss (2026-10-07)

geolig with `loss: metric` (losses.py: per-sample max|ref| normalization, fields
weighted by a detached 1 / running RRMSE).

| Arm | Dev total_error (s0 / s1 / s2) | Mean ± std |
|---|---|---|
| geolig | 0.0321 / 0.0312 / 0.0337 | 0.0323 ± 0.0013 |
| geoligml | 0.0301 / 0.0289 / 0.0294 | 0.0295 ± 0.0006 |

- −0.0028: below the 0.004 bar, but every seed and every output improves and the
  seed spread halves (Welch t ≈ 3.4). Cumulative from geo: −0.0045. Adopted as
  the parent.
- Largest gains where standardized MSE misweighted the metric. Standardized MSE
  weights a field's squared error by 1/std_f² (node spread); the metric by
  1/max|ref|². Typical max/std is 3.4 for u but 9.5 for P11, P22, 7.4 for psi
  and 5.2 for P12, P21, so relative to the metric the old loss under-weighted u
  by ~8× against P11 (and the 1/RRMSE weights push u further, since its RRMSE is
  the smallest). Result: u1, u2 0.016 → 0.011; effective_energy (RVE average of
  predicted psi) 0.022 → 0.017. Stresses move 2–5 %.
- End-of-training EMA (training) RRMSE vs dev: u 0.006 vs 0.011, P12/P21 0.036
  vs 0.051, P11/P22 0.025 vs 0.032. Stresses are limited both by fit (P12 0.036
  even on training data) and by generalization. Near-hole error 0.058 → 0.053
  within 0.01 of a hole; worst dev samples still the clustered layouts (758,
  1130, 1003, 868).

### Ablation (d): resolution vs capacity (2026-10-08)

Two arms on top of geoligml, one model knob each.

| Arm | Change | Dev total_error (s0 / s1 / s2) | Mean ± std | Train P12 (EMA, end) | Params / time |
|---|---|---|---|---|---|
| geoligml | — | 0.0301 / 0.0289 / 0.0294 | 0.0295 ± 0.0006 | 0.036 | 0.78M / 40 min |
| geoligmls64 | slice_num 32 → 64 | 0.0271 / 0.0271 / 0.0267 | **0.0270 ± 0.0002** | 0.035 | 0.78M / 41 min |
| geoligmlw256 | n_hidden 256, n_head 8 | 0.0275 / 0.0276 / 0.0274 | 0.0275 ± 0.0001 | 0.028 | 3.05M / 58 min |

- **More slices help generalization, not fit.** s64 leaves the training error
  almost unchanged (P12 0.036 → 0.035) but cuts dev P12/P21 0.051 → 0.045 and u
  0.011 → 0.009. Finer physics slices group fewer, more local nodes, which seems
  to transfer better to unseen hole layouts. Near-hole error 0.053 → 0.049
  within 0.01 of a hole.
- **Width fits better and overfits.** w256 drops training P12 to 0.028, but dev
  P12 only to 0.047; best epochs 85–150 with last > best. Not adopted without
  regularization.
- s64 is adopted as the parent (−0.0025, ≈ 7× the parent's spread). The seed
  spread is now 0.0002, so the old fixed 0.004 bar (2× geo's 0.002 spread) is
  replaced by 2× the larger arm std with a 0.001 floor (docs/methods.md §2).

### Ablations (e), (f): slice_num 128 and 256 (2026-10-08/09)

| Arm | Change | Dev total_error (s0 / s1 / s2) | Mean ± std | P12 / P21 | Time per run |
|---|---|---|---|---|---|
| geoligmls64 | — | 0.0271 / 0.0271 / 0.0267 | 0.0270 ± 0.0002 | 0.045 / 0.046 | ~41 min |
| geoligmls128 | slice_num 64 → 128 | 0.0251 / 0.0247 / 0.0253 | **0.0251 ± 0.0003** | 0.042 / 0.042 | ~46–60 min |
| geoligmls256 | slice_num 128 → 256 | 0.0238 / 0.0246 / 0.0245 | 0.0243 ± 0.0004 | 0.041 / 0.041 | ~58–62 min |

- 64 → 128 clears the bar (−0.0019, about 2× the 0.001 floor). Same pattern as
  32 → 64: training error barely moves (P12 EMA 0.035 → 0.035), validation error
  falls, so finer slicing helps generalization.
- 128 → 256 gains −0.0008, below the bar; seed 0 (0.0238) is the luckiest seed.
  Settled by CV below.
- Across the resolution steps 32 → 128, dev went 0.0295 → 0.0251, about as much
  as the features and the loss together contributed after the geometry
  diagnostic (0.0340 → 0.0295).

### 5-fold cross-validation: geoligmls128 vs geoligmls256 (2026-10-14)

`train.py --fold k` (d3e3261): train on the ~611 DOE_train samples outside
folds[k], validate on folds[k] (153, 153, 153, 153, 152 samples), normalization
fit without the fold, seed 0. Compared on the **last** checkpoint (no selection
on the validation fold; last and best differ by ≤ 0.0001).

| | geoligmls128 | geoligmls256 |
|---|---|---|
| **5-fold CV total_error** | **0.0256 ± 0.0006** | **0.0248 ± 0.0002** |
| per fold (0–4) | 0.0252 / 0.0258 / 0.0263 / 0.0258 / 0.0248 | 0.0246 / 0.0247 / 0.0250 / 0.0247 / 0.0249 |
| u1 / u2 | 0.0082 / 0.0083 | 0.0076 / 0.0077 |
| P11 / P22 | 0.0283 / 0.0283 | 0.0278 / 0.0277 |
| P12 / P21 | 0.0439 / 0.0430 | 0.0419 / 0.0413 |
| psi / effective_energy | 0.0282 / 0.0164 | 0.0280 / 0.0162 |

- Paired by fold, 256 − 128 = −0.0006, −0.0012, −0.0013, −0.0010, +0.0001: mean
  −0.0008 ± 0.0006 (t ≈ 3.3, 4 of 5 folds better), the same −0.0008 as the
  3-seed dev estimate, with a 3× smaller fold-to-fold spread.
- **geoligmls256 adopted as the final configuration** (2026-10-15). Strictly
  0.0002 short of the bar, adopted because two independent estimates agree and
  the cost is ~20 % more time per run.

### First leaderboard result (2026-10-15)

geoligmls128 seed 1 (dev 0.0247), standard writer output, on the 2D board:
**0.0252, rank 16/29** (public subset). Board vs dev per output: u1 0.0076 vs
0.0075, u2 0.0082 vs 0.0076, P11 0.0281 vs 0.0283, P12 0.0442 vs 0.0415, P22
0.0283 vs 0.0278, P21 0.0429 vs 0.0418, psi 0.0283 vs 0.0281, effective_energy
0.0140 vs 0.0153. The board score falls inside this config's CV estimate (0.0256
± 0.0006): local validation predicts the test score. Published baselines on the
same board: Augur 0.0221 (better), FNO 0.0302, Vi-Transformer 0.0325, MARIO
0.0573.

### Final model, ensemble, and last probes (2026-10-15..22)

- **Final retrain** (`train.py --full`, docs/methods.md §2): geoligmls256 on all
  764 training samples, 180 epochs = 34,380 updates (dev protocol 34,400), last
  checkpoint. Seeds 0, 1, 2 (fixed in advance). Board, seed 0 alone
  (2026-10-17): **0.0242**.
- **3-model ensemble** (`ensemble.py`, equal-weight mean of the three full
  retrains): board **0.0219** (2026-10-20, rank 13), lower than every published
  baseline on the board (Augur 0.0221). The same ensemble of the three
  dev-protocol seeds scores 0.0222 on dev (members 0.0238 / 0.0246 / 0.0245), so
  the dev estimate predicted the board within 0.0003. Gains are largest for u
  (−18 %) and effective_energy (−25 %).
- **Probes vs geoligmls256 (one seed, dev):** 400 epochs: best 0.0244 / last
  0.0252; 8 layers: 0.0243 / 0.0249; Transolver++: 0.0251 / 0.0252. None helps.
  With more epochs or depth the training error keeps falling (P11 0.024 → 0.020,
  P12 0.035 → 0.029) while dev doesn't move, and the last checkpoint is worse
  than the best: the model is **generalization-limited** (688-764 meshes), not
  capacity-limited. That is also why the ensemble helps. Next levers, not run:
  input features that add information (direction to the nearest hole + its
  radius) and symmetry-based data augmentation (mirrored / rotated RVEs with C,
  u, P transformed).
- **Transolver++ caveat:** in PhysicsNeMo 2.1.1 its slice attention samples
  Gumbel noise in eval mode too (`physicsnemo/nn/module/gumbel_softmax.py`), so
  predictions are not deterministic and our exact dev round trip fails (rel
  5.7e-4).

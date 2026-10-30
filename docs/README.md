# Documentation

Documents carry an ID: `DEC-NNN` for decision records (why something was
chosen) and `DSG-NNN` for design notes (how something works, what was found).
Each prefix is numbered on its own, in the order the documents were written.

| ID | Document | What it covers |
|---|---|---|
| — | [`methods.md`](methods.md) | Data and cache, metric and validation, submission format, model and loss, run records, design rules. Code comments point here as `docs/methods.md §N`. |
| DEC-001 | [PhysicsNeMo 2.1.1](decisions/DEC-001-physicsnemo-2.1.1.md) | Why PhysicsNeMo 2.1.1 and not 2.2.x: a dependency conflict with `pyplaid` 0.1.15. |
| DSG-001 | [Environment](design/DSG-001-environment.md) | Package pins and why, the install recipe, how to verify it, known pitfalls. |
| DSG-002 | [2D_MultiScHypEl data and experiments](design/DSG-002-data-2d-multischypel.md) | Data audit, kinematics, the geometry diagnostic, every ablation, 5-fold CV, the final model and leaderboard results. |

Figures for the top-level README are in [`figures/`](figures/). They are drawn
by `scripts/make_figures.py` from
[`figures/results_2d.csv`](figures/results_2d.csv), except `error_map_2d.png`,
which `python -m plaid_surrogates.error_maps` wrote for the final config.

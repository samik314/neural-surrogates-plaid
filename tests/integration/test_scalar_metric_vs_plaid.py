"""Our scalar RRMSE vs plaid.post.metrics on perturbed predictions (docs/methods.md §2).

pyplaid rounds rRMSE to 6 significant digits ("{:#.6g}"), hence rtol=1e-5.
Fields can't be parity-tested: pyplaid has no field metric.
"""

import numpy as np
import pytest
from plaid.post.metrics import compute_rRMSE_RMSE, prepare_metrics_for_split

from plaid_surrogates.metrics import scalar_rrmse


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_scalar_rrmse_matches_plaid(seed):
    rng = np.random.default_rng(seed)
    n = 57
    # Mixed signs and scales, like max_sig22_top (~40) vs max_U2_top (~1e-2).
    refs = {
        "big": rng.uniform(20, 60, n),
        "small": rng.uniform(1e-3, 2e-2, n),
        "signed": rng.choice([-1.0, 1.0], n) * rng.uniform(0.5, 3.0, n),
    }
    preds = {
        k: v * (1 + rng.normal(0, 0.05, n)) + rng.normal(0, 0.01 * v.std(), n)
        for k, v in refs.items()
    }
    split = {"dev": list(rng.permutation(n))}  # an unsorted ID list, like a real split could be

    rel, ab = {"dev": {}}, {"dev": {}}
    for k in refs:
        rel["dev"][k], ab["dev"][k] = np.empty(n), np.empty(n)
        prepare_metrics_for_split(refs[k], preds[k], split["dev"], rel["dev"][k], ab["dev"][k])
    metrics = {}
    compute_rRMSE_RMSE(metrics, rel, ab, split, list(refs))

    for k in refs:
        ours = scalar_rrmse(refs[k][split["dev"]], preds[k][split["dev"]], k)
        assert ours == pytest.approx(metrics["rRMSE for scalars"]["dev"][k], rel=1e-5)

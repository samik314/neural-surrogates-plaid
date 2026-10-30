"""Per-dataset facts shared by convert, metrics, submission: HF source, splits, scored outputs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    """Static description of one PLAID benchmark dataset."""

    name: str
    hf_repo: str
    revision: str  # pinned HF commit, so the cache is reproducible
    train_split: str
    test_split: str
    input_scalars: tuple[str, ...]
    output_fields: tuple[str, ...]  # everything the dataset provides
    output_scalars: tuple[str, ...]
    scored_fields: tuple[str, ...]  # what the leaderboard scores (subset of output_fields)
    scored_scalars: tuple[str, ...]
    leaderboard: str = ""  # the HF competition Space this dataset's submission files go to

    def __post_init__(self) -> None:
        if not set(self.scored_fields) <= set(self.output_fields):
            raise ValueError(f"{self.name}: scored fields not a subset of output fields")
        if not set(self.scored_scalars) <= set(self.output_scalars):
            raise ValueError(f"{self.name}: scored scalars not a subset of output scalars")


REGISTRY: dict[str, DatasetSpec] = {
    "Tensile2d": DatasetSpec(
        name="Tensile2d",
        hf_repo="PLAID-datasets/Tensile2d",
        revision="c3caf2136b2a7d1aba3e5585ea992d2c26a5ffbb",  # 2025-11-04
        train_split="train_500",
        test_split="test",
        input_scalars=("P", "p1", "p2", "p3", "p4", "p5"),
        output_fields=("U1", "U2", "q", "sig11", "sig12", "sig22"),
        output_scalars=("max_von_mises", "max_q", "max_U2_top", "max_sig22_top"),
        # Not in the problem definition. Confirmed by the official FNO and MMGP submission writers
        # (plaid-benchmarks@db1422de4e), the PLAID paper's per-output table, and the leaderboard's
        # per-output columns (checked 2026-10).
        scored_fields=("U1", "U2", "sig11", "sig22", "sig12"),
        scored_scalars=("max_von_mises", "max_U2_top", "max_sig22_top"),
        leaderboard="https://huggingface.co/spaces/PLAIDcompetitions/Tensile2dBenchmark",
    ),
    "2D_MultiScHypEl": DatasetSpec(
        name="2D_MultiScHypEl",
        hf_repo="PLAID-datasets/2D_Multiscale_Hyperelasticity",
        revision="68ae0593401e3507eeb2253cb0db2b2d6c6397bf",  # 2025-11-04
        train_split="DOE_train",
        test_split="DOE_test",
        input_scalars=("C11", "C12", "C22"),
        output_fields=("u1", "u2", "P11", "P12", "P22", "P21", "psi"),
        output_scalars=("effective_energy",),
        # Every output is scored (dataset card, PLAID paper, docs/methods.md §1). Note P12 != P21
        # (first Piola-Kirchhoff stress is not symmetric).
        scored_fields=("u1", "u2", "P11", "P12", "P22", "P21", "psi"),
        scored_scalars=("effective_energy",),
        leaderboard=(
            "https://huggingface.co/spaces/PLAIDcompetitions/2DMultiscaleHyperelasticityBenchmark"
        ),
    ),
}


def get_spec(name: str) -> DatasetSpec:
    """Look up a dataset by name; raises KeyError listing the known names."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown dataset {name!r}; known: {sorted(REGISTRY)}") from None

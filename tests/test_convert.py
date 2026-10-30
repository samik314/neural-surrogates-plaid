"""convert.py logic with a duck-typed fake PLAID sample (no PLAID needed)."""

import numpy as np
import pytest

from plaid_surrogates.data import convert
from plaid_surrogates.data.registry import DatasetSpec, get_spec

SPEC = DatasetSpec(
    name="Fake",
    hf_repo="x/y",
    revision="0" * 40,
    train_split="train",
    test_split="test",
    input_scalars=("P",),
    output_fields=("U1", "U2"),
    output_scalars=("smax",),
    scored_fields=("U1",),
    scored_scalars=("smax",),
)


class FakeSample:
    def __init__(self, n=5, outputs=True, fields=None, scalars=None, elements=None):
        self.nodes = np.arange(2 * n, dtype=float).reshape(n, 2)
        self.elements = elements or {"TRI_3": np.array([[0, 1, 2], [2, 3, 4]])}
        self.fields = (
            fields
            if fields is not None
            else ({"U1": np.linspace(0, 1, n), "U2": -np.linspace(0, 1, n)} if outputs else {})
        )
        self.scalars = (
            scalars if scalars is not None else ({"P": 2.0, "smax": 3.0} if outputs else {"P": 2.0})
        )
        self.tags = {"Top": np.array([4, 3])}

    def get_nodes(self):
        return self.nodes

    def get_elements(self):
        return self.elements

    def get_field_names(self):
        return list(self.fields)

    def get_field(self, n):
        return self.fields[n]

    def get_scalar_names(self):
        return [np.str_(k) for k in self.scalars]  # PLAID returns np.str_

    def get_scalar(self, n):
        return np.float64(self.scalars[n])

    def get_nodal_tags(self):
        return self.tags


def test_train_sample_extracted_in_node_order():
    fake = FakeSample()
    s = convert.plaid_sample_to_cached(fake, 3, SPEC)
    assert s.sample_id == 3 and s.has_outputs
    np.testing.assert_array_equal(s.points, fake.nodes)
    np.testing.assert_array_equal(s.output_fields["U2"], fake.fields["U2"])
    assert s.input_scalars == {"P": 2.0} and s.output_scalars == {"smax": 3.0}
    assert type(s.input_scalars["P"]) is float
    np.testing.assert_array_equal(s.nodal_tags["Top"], [4, 3])
    assert s.cells.dtype == np.int64


def test_test_sample_has_no_outputs():
    s = convert.plaid_sample_to_cached(FakeSample(outputs=False), 9, SPEC)
    assert not s.has_outputs


@pytest.mark.parametrize(
    "kwargs, msg",
    [
        ({"elements": {"QUAD_4": np.zeros((1, 4), int)}}, "TRI_3"),
        ({"fields": {"U1": np.zeros(5)}}, "partial outputs"),
        ({"fields": {"U1": np.zeros(5), "U2": np.zeros(5), "bogus": np.zeros(5)}}, "unexpected"),
        ({"scalars": {"smax": 1.0}}, "missing input"),
    ],
)
def test_malformed_samples_rejected(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        convert.plaid_sample_to_cached(FakeSample(**kwargs), 0, SPEC)


def test_outputs_vs_splits():
    splits = {"train": [0, 1], "test": [2]}
    convert.check_outputs_vs_splits({0: True, 1: True, 2: False}, splits, SPEC)
    with pytest.raises(ValueError, match="without outputs"):
        convert.check_outputs_vs_splits({0: True, 1: False, 2: False}, splits, SPEC)
    with pytest.raises(ValueError, match="WITH outputs"):
        convert.check_outputs_vs_splits({0: True, 1: True, 2: True}, splits, SPEC)


class FakeProblem:
    def __init__(self, fields=("U1", "U2")):
        self.fields = list(fields)

    def get_input_scalars_names(self):
        return ["P"]

    def get_output_fields_names(self):
        return self.fields

    def get_output_scalars_names(self):
        return ["smax"]

    def get_split(self):
        return {"train": [0], "test": [1]}


def test_problem_mismatch_stops():
    convert.check_problem_matches_spec(FakeProblem(), SPEC)
    with pytest.raises(ValueError, match="output_fields"):
        convert.check_problem_matches_spec(FakeProblem(fields=("U1",)), SPEC)


def test_overwrite_refuses_non_cache_dir(tmp_path):
    (tmp_path / "thesis.tex").write_text("precious")
    assert convert.main(["--dataset", "Tensile2d", "--out", str(tmp_path), "--overwrite"]) == 1
    assert (tmp_path / "thesis.tex").exists()


def test_existing_cache_needs_overwrite(tmp_path):
    (tmp_path / "index.json").write_text("{}")
    assert convert.main(["--dataset", "Tensile2d", "--out", str(tmp_path)]) == 1


def test_registry_scored_subset():
    spec = get_spec("Tensile2d")
    assert len(spec.scored_fields) + len(spec.scored_scalars) == 8
    with pytest.raises(KeyError, match="known"):
        get_spec("nope")

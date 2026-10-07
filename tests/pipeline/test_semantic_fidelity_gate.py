"""Provides helper functions: test_semantic_contract_contains_release_fields, test_semantic_compare_fails_on_dropped_variable, test_semantic_compare_fails_on_lost_edge, test_semantic_compare_fails_on_changed_dimension, and 3 more.

Public functions: test_semantic_contract_contains_release_fields, test_semantic_compare_fails_on_dropped_variable, test_semantic_compare_fails_on_lost_edge, test_semantic_compare_fails_on_changed_dimension, test_semantic_compare_fails_on_changed_matrix_shape, test_semantic_gate_writes_passed_ledger, test_semantic_gate_fails_unsupported_success_claim
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from gnn.pipeline.semantic_fidelity import (
    build_semantic_contract,
    compare_semantic_contracts,
    run_semantic_fidelity_gate,
)

SAMPLE = Path("input/gnn_files/basics/dynamic_perception.md")


def test_semantic_contract_contains_release_fields() -> None:
    contract = build_semantic_contract(SAMPLE)

    assert contract["schema"] == "gnn_semantic_contract_v2"
    assert contract["model_identity"]["model_name"]
    assert {variable["name"] for variable in contract["variables"]} >= {"A", "B"}
    assert contract["edges"]
    assert "A" in contract["parameter_shapes"]
    assert contract["contract_hash"]


def test_semantic_compare_fails_on_dropped_variable() -> None:
    original = build_semantic_contract(SAMPLE)
    changed = deepcopy(original)
    changed["variables"] = changed["variables"][1:]

    differences = compare_semantic_contracts(original, changed)

    assert any(diff.field == "variables" for diff in differences)


def test_semantic_compare_fails_on_lost_edge() -> None:
    original = build_semantic_contract(SAMPLE)
    changed = deepcopy(original)
    changed["edges"] = changed["edges"][1:]

    differences = compare_semantic_contracts(original, changed)

    assert any(diff.field == "edges" for diff in differences)


def test_semantic_compare_fails_on_changed_dimension() -> None:
    original = build_semantic_contract(SAMPLE)
    changed = deepcopy(original)
    changed["variables"][0]["dimensions"] = [999]

    differences = compare_semantic_contracts(original, changed)

    assert any(diff.field == "variables" for diff in differences)


def test_semantic_compare_fails_on_changed_matrix_shape() -> None:
    original = build_semantic_contract(SAMPLE)
    changed = deepcopy(original)
    changed["parameter_shapes"]["A"] = [1, 1, 1]

    differences = compare_semantic_contracts(original, changed)

    assert any(diff.field == "parameter_shapes" for diff in differences)


def test_semantic_gate_writes_passed_ledger(tmp_path: Path) -> None:
    ledger = run_semantic_fidelity_gate(
        Path("input/model_family_manifest.json"),
        tmp_path,
        family_names=["basics"],
        strict=True,
    )

    assert ledger["status"] == "passed"
    assert (tmp_path / "semantic_fidelity_ledger.json").exists()
    assert (tmp_path / "semantic_fidelity_ledger.md").exists()


def test_semantic_gate_fails_unsupported_success_claim(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Semantic fidelity failed"):
        run_semantic_fidelity_gate(
            Path("input/model_family_manifest.json"),
            tmp_path,
            family_names=["basics"],
            formats=("pnml",),
            strict=True,
        )


def test_semantic_compare_rejects_same_shape_changed_values() -> None:
    original = build_semantic_contract(SAMPLE)
    changed = deepcopy(original)
    changed["parameter_values"]["A"][0][0] += 0.01
    assert any(
        d.field == "parameter_values"
        for d in compare_semantic_contracts(original, changed)
    )


def test_semantic_compare_rejects_lost_parameter_provenance() -> None:
    original = build_semantic_contract(SAMPLE)
    changed = deepcopy(original)
    changed["parameter_metadata"] = {"A": {"derived": True}}
    assert any(
        d.field == "parameter_metadata"
        for d in compare_semantic_contracts(original, changed)
    )

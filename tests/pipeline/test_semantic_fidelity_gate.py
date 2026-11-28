"""Provides helper functions: test_semantic_contract_contains_release_fields, test_semantic_compare_fails_on_dropped_variable, test_semantic_compare_fails_on_lost_edge, test_semantic_compare_fails_on_changed_dimension, and 3 more.

Public functions: test_semantic_contract_contains_release_fields, test_semantic_compare_fails_on_dropped_variable, test_semantic_compare_fails_on_lost_edge, test_semantic_compare_fails_on_changed_dimension, test_semantic_compare_fails_on_changed_matrix_shape, test_semantic_gate_writes_passed_ledger, test_semantic_gate_fails_unsupported_success_claim
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from gnn.parsers import GNNParsingSystem
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


def test_semantic_compare_rejects_numeric_substitution_for_boolean() -> None:
    original = {"parameter_values": {"mask": [True, False]}}
    changed = {"parameter_values": {"mask": [1, 0]}}
    assert [d.field for d in compare_semantic_contracts(original, changed)] == [
        "parameter_values"
    ]


def test_semantic_compare_rejects_lost_edge_annotation() -> None:
    original = build_semantic_contract(SAMPLE)
    original["edges"][0]["annotation"] = "likelihood:α"
    changed = deepcopy(original)
    changed["edges"][0]["annotation"] = None
    assert [d.field for d in compare_semantic_contracts(original, changed)] == ["edges"]


def test_semantic_gate_round_trips_saved_yaml_and_markdown(tmp_path: Path) -> None:
    ledger = run_semantic_fidelity_gate(
        Path("input/model_family_manifest.json"),
        tmp_path,
        family_names=["basics"],
        formats=("json", "yaml", "markdown"),
        strict=True,
    )
    assert ledger["status"] == "passed"
    for model in ledger["families"][0]["models"]:
        for result in model["round_trips"]:
            assert Path(result["artifact"]).is_file()


def test_semantic_gate_rejects_changes_visible_only_in_the_saved_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_file = GNNParsingSystem.serialize_to_file

    def altered_write(system, model, file_path, format=None):
        write_file(system, model, file_path, format)
        path = Path(file_path)
        saved = json.loads(path.read_text(encoding="utf-8"))
        saved["extensions"]["gnn_section"] = "ChangedAfterSerialization"
        path.write_text(json.dumps(saved), encoding="utf-8")

    monkeypatch.setattr(GNNParsingSystem, "serialize_to_file", altered_write)
    with pytest.raises(RuntimeError, match="Semantic fidelity failed: basics"):
        run_semantic_fidelity_gate(
            Path("input/model_family_manifest.json"),
            tmp_path,
            family_names=["basics"],
            strict=True,
        )
    ledger = json.loads((tmp_path / "semantic_fidelity_ledger.json").read_text())
    differences = ledger["families"][0]["models"][0]["round_trips"][0]["differences"]
    assert [difference["field"] for difference in differences] == ["model_identity"]

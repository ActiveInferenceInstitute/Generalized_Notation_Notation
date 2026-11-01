"""Tests for the pinned ontology vocabulary bridge (GNN consumer side).

The bridge is the Active Inference Ontology repository's generated projection
of the labels GNN declares (``GNN/act_inf_ontology_terms.json``, schema
``active-inference-ontology/gnn-terms/v1``). GNN keeps a byte-exact copy beside
its pin record; these tests exercise the verification, the resolution rule and
the invariant that ``obo:`` values never reach identifier output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gnn.ontology import (
    DECLARED_STATUS_NOT_DECLARED,
    DECLARED_STATUS_RESOLVED,
    DECLARED_STATUS_UNRESOLVED,
    declared_label_index,
    load_defined_ontology_terms,
    load_ontology_bridge,
    load_ontology_pin,
    ontology_bridge_path,
    ontology_bridge_status,
    ontology_pin_path,
    pinned_declared_index,
    resolve_declared_label,
    sha256_file,
    validate_annotations,
    verify_ontology_pin,
)
from tests.helpers import load_module_from_path

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_SCRIPT = REPO_ROOT / "scripts" / "check_ontology_pin.py"
BUNDLED_VOCABULARY = (
    REPO_ROOT / "src" / "gnn" / "ontology" / "act_inf_ontology_terms.json"
)

#: The audited hash of the bundled 64-label vocabulary, recorded by the
#: ontology repository's ``GNN/declared-labels.json``.
BUNDLED_VOCABULARY_SHA256 = (
    "a9639fbabc2e227adcf00450aad23638ceaaa384caea29824ae2d063a4559d1c"
)


def _gate_module():
    return load_module_from_path("check_ontology_pin", GATE_SCRIPT)


def test_pinned_copy_verifies_against_its_pin() -> None:
    summary = verify_ontology_pin()
    assert summary["verified"], summary["errors"]
    assert summary["sha256"] == summary["pin"]["sha256"]
    assert summary["bridge"]["release"] == "v5"
    counts = summary["bridge"]["counts"]
    assert counts["declaredLabels"] == 64
    assert counts["terms"] == 429
    assert counts["resolvedDeclaredLabels"] == 12
    assert counts["unresolvedDeclaredLabels"] == 52


def test_pin_records_the_embedded_source_and_export_digests() -> None:
    pin = json.loads(ontology_pin_path().read_text(encoding="utf-8"))
    bridge = json.loads(ontology_bridge_path().read_text(encoding="utf-8"))
    assert pin["repository"] == "ActiveInferenceInstitute/Active_Inference_Ontology"
    assert pin["path"] == "GNN/act_inf_ontology_terms.json"
    assert pin["release"] == "v5"
    assert pin["schema"] == bridge["schema"]
    assert pin["sourceSha256"] == bridge["source"]["sha256"]
    assert pin["exportSha256"] == bridge["export"]["sha256"]


def test_bundled_vocabulary_is_the_audited_surface() -> None:
    assert sha256_file(BUNDLED_VOCABULARY) == BUNDLED_VOCABULARY_SHA256


def test_tampered_copy_fails_verification(tmp_path: Path) -> None:
    copy = tmp_path / "copy.json"
    copy.write_text(
        ontology_bridge_path().read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    pin = tmp_path / "pin.json"
    pin.write_text(ontology_pin_path().read_text(encoding="utf-8"), encoding="utf-8")

    summary = verify_ontology_pin(bridge_path=copy, pin_path=pin)

    assert not summary["verified"]
    assert any("SHA-256" in error for error in summary["errors"])


def test_missing_copy_fails_verification(tmp_path: Path) -> None:
    summary = verify_ontology_pin(bridge_path=tmp_path / "absent.json")
    assert not summary["verified"]
    assert any("missing" in error for error in summary["errors"])


def test_incomplete_pin_is_rejected(tmp_path: Path) -> None:
    pin = tmp_path / "pin.json"
    pin.write_text(
        json.dumps(
            {"repository": "ActiveInferenceInstitute/Active_Inference_Ontology"}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required fields"):
        load_ontology_pin(str(pin))

    summary = verify_ontology_pin(pin_path=pin)
    assert not summary["verified"]
    assert any("pin record unreadable" in error for error in summary["errors"])


def test_resolution_rule_covers_all_declared_outcomes() -> None:
    declared = declared_label_index(load_ontology_bridge())

    resolved = resolve_declared_label("Action", declared)
    assert resolved.status == DECLARED_STATUS_RESOLVED
    assert resolved.identifier == "term-action"
    assert resolved.ontology_term == "Action"
    assert resolved.reason == ""

    unresolved = resolve_declared_label("LikelihoodMatrix", declared)
    assert unresolved.status == DECLARED_STATUS_UNRESOLVED
    assert unresolved.identifier is None
    assert unresolved.reason == "not in ontology release v5"

    undeclared = resolve_declared_label("NotARealLabel", declared)
    assert undeclared.status == DECLARED_STATUS_NOT_DECLARED
    assert undeclared.identifier is None


def test_resolution_is_case_insensitive() -> None:
    declared = declared_label_index(load_ontology_bridge())
    assert resolve_declared_label("hiddenstate", declared).identifier == (
        "term-hidden-state"
    )


def test_unavailable_bridge_reports_without_inventing_identifiers() -> None:
    resolution = resolve_declared_label("Action", None)
    assert resolution.status == "bridge_unavailable"
    assert resolution.identifier is None
    assert resolution.reason == "pinned ontology bridge unavailable"


def test_pinned_index_loads_the_audited_declared_set() -> None:
    index, error = pinned_declared_index()
    assert error is None
    assert index is not None and len(index) == 64


def test_validate_annotations_resolves_declared_labels() -> None:
    result = validate_annotations(
        [
            "s=HiddenState",
            "o=Observation",
            "A=LikelihoodMatrix",
            "u=UnknownLabel",
        ]
    )

    resolved = result["matched_terms"]["s"]
    assert resolved["ontology_id"] == "term-hidden-state"
    assert resolved["ontology_status"] == DECLARED_STATUS_RESOLVED
    assert resolved["ontology_term"] == "Hidden State"

    unresolved = result["matched_terms"]["A"]
    assert unresolved["term_name"] == "LikelihoodMatrix"
    assert unresolved["ontology_id"] is None
    assert unresolved["ontology_status"] == DECLARED_STATUS_UNRESOLVED
    assert unresolved["ontology_reason"] == "not in ontology release v5"

    assert result["invalid_annotations"] == ["u=UnknownLabel"]
    assert result["ontology_identifier_counts"]["resolved"] == 2
    assert result["ontology_identifier_counts"]["unresolved"] == 1


def test_all_twelve_resolved_labels_map_to_term_ids() -> None:
    bridge = load_ontology_bridge()
    resolved_labels = sorted(
        label
        for label, entry in bridge["declared"].items()
        if entry["status"] == DECLARED_STATUS_RESOLVED
    )
    assert len(resolved_labels) == 12

    result = validate_annotations(
        [f"x{index}={label}" for index, label in enumerate(resolved_labels)]
    )
    assert result["invalid_annotations"] == []
    identifiers = {entry["ontology_id"] for entry in result["matched_terms"].values()}
    assert all(
        isinstance(identifier, str) and identifier.startswith("term-")
        for identifier in identifiers
    )
    assert len(identifiers) == 12


def test_validation_output_never_carries_obo_identifiers() -> None:
    result = validate_annotations(
        ["s=HiddenState", "A=LikelihoodMatrix", "x=NotDeclaredAtAll"]
    )
    serialised = json.dumps(result)
    assert "obo:" not in serialised
    assert "uri" not in result["matched_terms"]["s"]
    # The bundled vocabulary still carries the values as declared metadata.
    vocabulary = load_defined_ontology_terms()
    assert vocabulary["HiddenState"]["uri"].startswith("obo:")


def test_builtin_default_terms_carry_no_uri() -> None:
    defaults = load_defined_ontology_terms(search_paths=[])
    assert len(defaults) == 7
    assert all("uri" not in entry for entry in defaults.values())


def test_bridge_status_is_recorded_for_run_artifacts() -> None:
    status = ontology_bridge_status()
    assert status["available"] is True
    assert status["verified"] is True
    assert status["schema"] == "active-inference-ontology/gnn-terms/v1"
    assert status["release"] == "v5"
    assert status["sha256"] == verify_ontology_pin()["sha256"]


def test_gate_script_verifies_the_live_tree(capsys: pytest.CaptureFixture[str]) -> None:
    module = _gate_module()
    assert module.main([]) == 0
    output = capsys.readouterr().out
    assert "verified" in output
    assert "12 resolved, 52 unresolved" in output


def test_gate_script_fails_when_verification_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import gnn.ontology.bridge as bridge_module

    module = _gate_module()
    monkeypatch.setattr(
        bridge_module,
        "verify_ontology_pin",
        lambda *args, **kwargs: {
            "verified": False,
            "errors": ["pinned bridge copy SHA-256 deadbeef does not match the pin"],
            "sha256": None,
            "pin": None,
            "bridge": None,
        },
    )

    assert module.main([]) == 1
    output = capsys.readouterr().out
    assert "1 failure(s)" in output
    assert "does not match the pin" in output

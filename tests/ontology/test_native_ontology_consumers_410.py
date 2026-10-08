"""Saved vocabulary/source consumers with independent report JSON checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gnn.ontology import (
    OntologyTermIndex,
    generate_ontology_report_for_file,
    load_defined_ontology_terms,
    process_ontology,
)

SOURCE = """# Saved semantic sensor model
## Ontology
Concept: hidden state
Concepts: sensory observation
Relation: hidden state produces sensory observation
Property: inference is local
Annotation: s=HiddenState
Annotations: o=Observation
## ModelParameters
Annotation: outside=UnknownTerm
"""


def vocabulary(root: Path) -> Path:
    path = root / "terms.json"
    path.write_text(
        json.dumps(
            {
                "states": [
                    {
                        "name": "HiddenState",
                        "description": "unobserved state",
                        "uri": "urn:local:hidden",
                    }
                ],
                "sensors": [
                    {
                        "term": "Observation",
                        "description": "sensor μ",
                        "uri": "urn:local:observation",
                    }
                ],
                "controls": ["Action"],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_saved_source_and_vocabulary_have_consistent_public_semantics(
    tmp_path: Path,
) -> None:
    terms_file = vocabulary(tmp_path)
    index = OntologyTermIndex.from_file(terms_file)
    assert index.lookup("HIDDENSTATE")["name"] == "HiddenState"
    assert index.lookup("observation")["uri"] == "urn:local:observation"
    assert index.lookup("action")["category"] == "controls"
    target = tmp_path / "authored"
    target.mkdir()
    source = target / "sensor.md"
    source.write_text(SOURCE, encoding="utf-8")
    output = tmp_path / "reports"
    assert (
        process_ontology(
            target, output, strict_validation=True, ontology_terms_file=terms_file
        )
        is True
    )
    summary = json.loads((output / "ontology_results.json").read_text())
    assert summary["success"] is True and summary["processed_files"] == 1
    assert summary["errors"] == [] and len(summary["reports"]) == 1
    report = json.loads(Path(summary["reports"][0]).read_text())
    assert report["ontology_data"] == {
        "concepts": ["hidden state", "sensory observation"],
        "relations": ["hidden state produces sensory observation"],
        "properties": ["inference is local"],
        "annotations": ["s=HiddenState", "o=Observation"],
    }
    assert report["validation_result"]["valid_annotations"] == [
        "s=HiddenState",
        "o=Observation",
    ]
    assert report["validation_result"]["invalid_annotations"] == []
    assert report["summary"]["total_concepts"] == 2
    assert report["summary"]["total_relations"] == 1
    assert report["summary"]["total_properties"] == 1
    assert report["summary"]["total_annotations"] == 2
    assert source.read_text() == SOURCE

    controls = """## ActInfOntologyAnnotation
s=HiddenState
Annotation: Action
Concept: hidden = state
Relations: s=o
Properties: observation = local
opaque: x=HiddenState
opaque: omitted
## ModelParameters
s=UnknownTerm
"""
    control_source = target / "controls.md"
    control_source.write_text(controls, encoding="utf-8")
    result = generate_ontology_report_for_file(
        control_source, output, ontology_terms=index.terms
    )
    assert result["success"] is True
    control_report = json.loads(Path(result["report_file"]).read_text())
    assert control_report["ontology_data"] == {
        "concepts": ["hidden = state"],
        "relations": ["s=o"],
        "properties": ["observation = local"],
        "annotations": ["s=HiddenState", "Action", "opaque: x=HiddenState"],
    }
    assert control_report["validation_result"]["valid_annotations"] == [
        "s=HiddenState",
        "Action",
        "opaque: x=HiddenState",
    ]
    assert control_report["validation_result"]["invalid_annotations"] == []
    assert control_source.read_text() == controls
    assert source.read_text() == SOURCE


@pytest.mark.parametrize(
    "payload",
    [
        "{ malformed json",
        '{"HiddenState": {}, "hiddenstate": {}}',
        '{"HiddenState": 42}',
    ],
)
def test_authoritative_bad_vocabulary_refuses_reports_without_builtin_fallback(
    tmp_path: Path, payload: str
) -> None:
    terms = tmp_path / "authoritative.json"
    terms.write_text(payload)
    with pytest.raises(ValueError):
        OntologyTermIndex.from_file(terms)
    target = tmp_path / "source"
    target.mkdir()
    (target / "sensor.md").write_text(SOURCE)
    output = tmp_path / "refused"
    assert process_ontology(target, output, ontology_terms_file=terms) is False
    receipt = json.loads((output / "ontology_results.json").read_text())
    assert receipt["success"] is False
    assert receipt["reports"] == [] and receipt["processed_files"] == 0
    assert receipt["errors"][0]["error_type"] == "ontology_terms_load_error"
    assert not list(output.rglob("*_ontology_report.json"))
    assert terms.read_text() == payload


def test_missing_authoritative_vocabulary_is_not_replaced_by_defaults(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "absent.json"
    with pytest.raises(FileNotFoundError):
        OntologyTermIndex.from_file(missing)
    assert not missing.exists()


def test_explicit_candidate_search_continues_to_a_real_valid_vocabulary(
    tmp_path: Path,
) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("not json")
    valid = vocabulary(tmp_path)
    terms = load_defined_ontology_terms(search_paths=[broken, valid])
    assert set(terms) == {"Action", "HiddenState", "Observation"}
    assert terms["Observation"]["description"] == "sensor μ"
    assert broken.read_text() == "not json"


def test_report_writer_refuses_a_file_as_output_directory_without_overwriting_it(
    tmp_path: Path,
) -> None:
    terms = vocabulary(tmp_path)
    source = tmp_path / "sensor.md"
    source.write_text(SOURCE)
    blocked = tmp_path / "caller-file"
    blocked.write_bytes(b"preserve ontology neighbor")
    result = generate_ontology_report_for_file(
        source, blocked, ontology_terms=OntologyTermIndex.from_file(terms).terms
    )
    assert result["success"] is False
    assert result["error"]
    assert blocked.read_bytes() == b"preserve ontology neighbor"
    assert not list(tmp_path.rglob("*_ontology_report.json"))

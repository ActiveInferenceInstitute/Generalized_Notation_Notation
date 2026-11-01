"""Regression tests for the canonical ``FEP.GnnDocument`` Lean serializer.

The serializer must emit the frozen 13-section inventory in canonical rank
order (``fep_formal/src/fep_formal/formal/gnn_document.lean``) and a round-trip
``-- MODEL_DATA:`` payload carrying the canonical typed keys.
"""

from __future__ import annotations

import json
import re

from gnn.parsers.common import (
    Connection,
    ConnectionType,
    DataType,
    GNNInternalRepresentation,
    OntologyMapping,
    Parameter,
    Variable,
    VariableType,
)
from gnn.parsers.lean_parser import LeanGNNParser
from gnn.parsers.lean_serializer import LeanSerializer

#: Frozen section kinds in canonical rank order.
SECTION_KINDS = (
    "gnnSection",
    "gnnVersionAndFlags",
    "modelName",
    "modelAnnotation",
    "stateSpaceBlock",
    "connections",
    "initialParameterization",
    "equations",
    "time",
    "actInfOntologyAnnotation",
    "modelParameters",
    "footer",
    "signature",
)


def _finite_fixture() -> GNNInternalRepresentation:
    model = GNNInternalRepresentation(
        model_name="Test Model", annotation="fixture", version="1.0"
    )
    model.variables = [
        Variable(
            name="s",
            var_type=VariableType.HIDDEN_STATE,
            dimensions=[2, 1],
            data_type=DataType.FLOAT,
        ),
        Variable(
            name="A",
            var_type=VariableType.LIKELIHOOD_MATRIX,
            dimensions=[2, 2],
            data_type=DataType.FLOAT,
        ),
    ]
    model.connections = [
        Connection(
            source_variables=["A"],
            target_variables=["s"],
            connection_type=ConnectionType.DIRECTED,
            annotation="uses",
        )
    ]
    model.parameters = [
        Parameter(name="A", value="{\n  (0.5, 0.5),\n  (0.5, 0.5)\n}"),
        Parameter(name="num_hidden_states", value=2),
    ]
    return model


def test_lean_serializer_emits_canonical_section_inventory() -> None:
    document = LeanSerializer().serialize(_finite_fixture())

    markers = [f"-- GnnSectionKind.{kind}" for kind in SECTION_KINDS]
    positions = [document.index(marker) for marker in markers]
    assert positions == sorted(positions), "sections out of canonical rank order"

    # Required kinds (GNN-E001) plus typed values for every section.
    for kind in (
        "gnnSection",
        "gnnVersionAndFlags",
        "modelName",
        "stateSpaceBlock",
        "connections",
    ):
        assert f"def _{kind} : GnnSection :=" in document

    # GnnParamEntry-shaped payloads with verbatim brace strings.
    assert '{ varName := "A", payload := "' in document
    assert "GnnDim.lit 2" in document
    assert "GnnDim.lit 1" in document
    assert "ConnKind.directed" in document

    # The aggregate document assembles every section.
    assert "def document : GnnDocument :=" in document
    for kind in SECTION_KINDS:
        assert f"_{kind}" in document


def test_lean_serializer_model_data_canonical_schema() -> None:
    match = re.search(
        r"-- MODEL_DATA: (\{.+\})", LeanSerializer().serialize(_finite_fixture())
    )
    assert match is not None, "missing -- MODEL_DATA payload"
    data = json.loads(match.group(1))

    assert data["schema_version"] == 1
    assert data["model_family"] == "finite"
    assert [entry["decl"] for entry in data["state_spaces"]] == ["A", "s"]
    assert data["state_spaces"][0]["dims"] == [2, 2]
    assert data["state_spaces"][0]["value_type"] == "float"
    assert data["parameterizations"] == [
        {"var_name": "A", "payload": "{\n  (0.5, 0.5),\n  (0.5, 0.5)\n}"}
    ]
    assert data["ontology_bindings"] == []

    # Legacy keys survive for the shared strict round-trip path.
    assert data["model_name"] == "Test Model"
    assert len(data["variables"]) == 2
    assert data["connections"][0]["connection_type"] == "directed"


def test_lean_serializer_ontology_bindings_payload() -> None:
    model = _finite_fixture()
    model.ontology_mappings = [
        OntologyMapping(variable_name="A", ontology_term="LikelihoodMatrix")
    ]
    match = re.search(r"-- MODEL_DATA: (\{.+\})", LeanSerializer().serialize(model))
    assert match is not None
    data = json.loads(match.group(1))
    assert data["ontology_bindings"] == [{"var_name": "A", "term": "LikelihoodMatrix"}]


def test_lean_serializer_continuous_family_detection() -> None:
    model = _finite_fixture()
    model.variables.append(
        Variable(
            name="H",
            var_type=VariableType.OBSERVATION,
            dimensions=[2, 2],
            data_type=DataType.FLOAT,
        )
    )
    document = LeanSerializer().serialize(model)
    match = re.search(r"-- MODEL_DATA: (\{.+\})", document)
    assert match is not None
    assert json.loads(match.group(1))["model_family"] == "continuous"


def test_lean_parser_round_trips_model_data() -> None:
    document = LeanSerializer().serialize(_finite_fixture())
    result = LeanGNNParser().parse_string(document)

    assert result.success, result.errors
    assert result.model.model_name == "Test Model"
    assert [variable.name for variable in result.model.variables] == ["A", "s"]
    assert result.model.connections[0].source_variables == ["A"]
    assert result.model.parameters[0].name == "A"

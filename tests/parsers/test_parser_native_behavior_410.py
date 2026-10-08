"""Native format consumers preserve model evidence without executing input."""

from pathlib import Path

import pytest
import yaml

from gnn.parsers import GNNFormat, GNNParsingSystem
from gnn.parsers.common import DataType, VariableType


def test_python_source_is_inspected_without_running_it(tmp_path: Path) -> None:
    sentinel = tmp_path / "must-not-exist"
    source = tmp_path / "agent.py"
    source.write_text(
        f'''"""A scientific model supplied by an untrusted author."""
from pathlib import Path
import numpy as np
Path({str(sentinel)!r}).write_text("executed")

class EvidenceModel:
    prior = np.ones((2,))

    def update(self):
        state = np.zeros((2, 3))
        observation = 4
        action = True
        likelihood = np.ones((3, 2))
        state = np.zeros((99,))

def infer():
    policy = np.arange(3)
    preference = 0.25
''',
        encoding="utf-8",
    )
    result = GNNParsingSystem().parse_file(source, GNNFormat.PYTHON)

    assert result.success, result.errors
    assert not sentinel.exists()
    assert result.model.model_name == "EvidenceModel"
    variables = {variable.name: variable for variable in result.model.variables}
    assert variables["state"].dimensions == [2, 3]
    assert variables["state"].var_type is VariableType.HIDDEN_STATE
    assert variables["observation"].data_type is DataType.INTEGER
    assert variables["action"].data_type is DataType.BINARY
    assert variables["action"].var_type is VariableType.ACTION
    assert variables["likelihood"].dimensions == [3, 2]
    assert variables["likelihood"].var_type is VariableType.LIKELIHOOD_MATRIX
    assert variables["policy"].data_type is DataType.INTEGER
    assert variables["preference"].var_type is VariableType.PREFERENCE_VECTOR
    assert variables["preference"].data_type is DataType.CONTINUOUS
    assert result.source_file == str(source)


def test_authored_protobuf_preserves_scalar_types_and_scientific_annotations(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sensor.proto"
    source.write_text(
        """syntax = "proto3";
package sensor.agent;
message Sensor {
  int32 state = 1;
  double gain = 2;
  bool policy = 3;
  string observation = 4;
}
// Connection: state --directed--> observation
// Parameter: steps = 3
// Parameter: temperature = -0.125
// Parameter: enabled = true
""",
        encoding="utf-8",
    )

    result = GNNParsingSystem().parse_file(source)

    assert result.success, result.errors
    assert result.source_file == str(source)
    assert result.model.model_name == "sensor_agent"
    variables = {variable.name: variable for variable in result.model.variables}
    assert set(variables) == {"state", "gain", "policy", "observation"}
    assert variables["state"].data_type is DataType.INTEGER
    assert variables["gain"].data_type is DataType.FLOAT
    assert variables["policy"].data_type is DataType.BINARY
    assert variables["observation"].data_type is DataType.CATEGORICAL
    assert len(result.model.connections) == 1
    assert result.model.connections[0].source_variables == ["state"]
    assert result.model.connections[0].target_variables == ["observation"]
    assert result.model.connections[0].connection_type.value == "directed"
    assert {
        parameter.name: parameter.value for parameter in result.model.parameters
    } == {"steps": 3, "temperature": -0.125, "enabled": True}


def test_protobuf_metadata_without_model_fields_cannot_report_success(
    tmp_path: Path,
) -> None:
    source = tmp_path / "empty.proto"
    source.write_text(
        'syntax = "proto3";\npackage empty.sensor;\nmessage Empty {}\n',
        encoding="utf-8",
    )

    result = GNNParsingSystem().parse_file(source)

    assert not result.success
    assert result.model.variables == []
    assert any("No variables" in error for error in result.errors)


def test_yaml_authored_mapping_preserves_scientific_values(tmp_path: Path) -> None:
    source = tmp_path / "agent.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "name": "Sensor agent",
                "description": "An asymmetric observation model",
                "state_space": {
                    "state": {
                        "type": "hidden_state",
                        "data_type": "categorical",
                        "dimensions": [2],
                    },
                    "observation": {
                        "type": "observation",
                        "data_type": "integer",
                        "dimensions": [3],
                        "description": "sensor symbol",
                    },
                },
                "connections": [
                    {"from": "state", "to": "observation", "weight": 0.75},
                    "state-observation",
                ],
                "initial_parameterization": {"D": [0.3, 0.7], "gain": 0.125},
                "equations": [
                    "o = A s",
                    {"label": "prediction", "equation": "A s", "format": "text"},
                ],
                "time_specification": {
                    "time_type": "Dynamic",
                    "discretization": "Discrete",
                    "model_time_horizon": 4,
                },
                "actinf_ontology_annotation": {"state": "HiddenState"},
                "created_at": "2026-01-01T00:00:00",
                "modified_at": "2026-01-02T00:00:00",
                "extensions": {"sensor_units": "symbol"},
                "raw_sections": {"provenance": "measured sensor"},
                "checksum": "author-receipt",
            }
        ),
        encoding="utf-8",
    )
    result = GNNParsingSystem().parse_file(source)

    assert result.success, result.errors
    assert result.model.model_name == "Sensor agent"
    assert result.model.annotation == "An asymmetric observation model"
    variables = {variable.name: variable for variable in result.model.variables}
    assert variables["state"].dimensions == [2]
    assert variables["observation"].dimensions == [3]
    assert variables["observation"].var_type is VariableType.OBSERVATION
    assert result.model.connections[0].source_variables == ["state"]
    assert result.model.connections[0].target_variables == ["observation"]
    assert result.model.connections[0].weight == 0.75
    assert result.model.connections[1].connection_type.value == "undirected"
    parameters = {
        parameter.name: parameter.value for parameter in result.model.parameters
    }
    assert parameters == {"D": [0.3, 0.7], "gain": 0.125}
    assert result.model.equations[0].content == "o = A s"
    assert result.model.equations[1].label == "prediction"
    assert result.model.time_specification.horizon == 4
    assert result.model.time_specification.time_type == "Dynamic"
    assert result.model.ontology_mappings[0].ontology_term == "HiddenState"
    assert result.model.created_at.isoformat() == "2026-01-01T00:00:00"
    assert result.model.modified_at.isoformat() == "2026-01-02T00:00:00"
    assert result.model.extensions == {"sensor_units": "symbol"}
    assert result.model.raw_sections == {"provenance": "measured sensor"}
    assert result.model.checksum == "author-receipt"


@pytest.mark.parametrize("extension", [".xml", ".xsd", ".pnml"])
def test_xml_based_formats_refuse_external_entity_expansion(
    tmp_path: Path, extension: str
) -> None:
    private_fixture = tmp_path / "private-fixture.txt"
    private_fixture.write_text("must-not-be-expanded-into-the-model", encoding="utf-8")
    source = tmp_path / f"entity{extension}"
    source.write_text(
        f'''<?xml version="1.0"?>
        <!DOCTYPE gnn [<!ENTITY leak SYSTEM "{private_fixture.as_uri()}">]>
        <gnn name="&leak;"/>
        ''',
        encoding="utf-8",
    )

    result = GNNParsingSystem().parse_file(source)

    assert not result.success
    assert result.errors
    assert "must-not-be-expanded" not in result.model.model_name
    assert result.model.variables == []


@pytest.mark.parametrize(
    "extension,content",
    [
        (
            ".bnf",
            """# Grammar for Sensor Model
            <state> ::= "integer"
            <observation> ::= <state> | "float"
            <action> ::= "boolean"
            <policy> ::= <action> <state> <policy>
            """,
        ),
        (
            ".ebnf",
            """# Grammar for Sensor Model
            state = "integer";
            observation = [state] | "float";
            action = "boolean";
            policy = {action}, (state), policy;
            """,
        ),
    ],
)
def test_authored_grammar_preserves_dependencies_without_self_edges(
    tmp_path: Path, extension: str, content: str
) -> None:
    source = tmp_path / f"sensor{extension}"
    source.write_text(content, encoding="utf-8")

    result = GNNParsingSystem().parse_file(source)

    assert result.success, result.errors
    assert result.model.model_name == "Sensor Model"
    assert {variable.name for variable in result.model.variables} == {
        "state",
        "observation",
        "action",
        "policy",
    }
    assert {
        (tuple(connection.source_variables), tuple(connection.target_variables))
        for connection in result.model.connections
    } == {
        (("state",), ("observation",)),
        (("action",), ("policy",)),
        (("state",), ("policy",)),
    }


@pytest.mark.parametrize(
    "extension,content,expected",
    [
        (
            ".xsd",
            """<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                 targetNamespace="https://example.org/SensorModel">
              <xs:element name="state" type="xs:double"/>
              <xs:element name="observation" type="xs:int"/>
              <xs:element name="action" type="xs:boolean"/>
              <xs:element name="label" type="xs:string"/>
            </xs:schema>""",
            {
                "state": "float",
                "observation": "integer",
                "action": "binary",
                "label": "categorical",
            },
        ),
        (
            ".asn1",
            """Sensor DEFINITIONS ::= BEGIN
            Agent ::= SEQUENCE {
              state REAL,
              observation INTEGER,
              action BOOLEAN,
              label UTF8String
            }
            END""",
            {
                "state": "float",
                "observation": "integer",
                "action": "binary",
                "label": "categorical",
            },
        ),
        (
            ".pkl",
            """class SensorModel {
              state: Float
              observation: Int
              action: Boolean
              policy: List<String>
              version: String = "1.0"
              gain: Float = 0.25
            }""",
            {"state": "float", "observation": "integer", "action": "binary"},
        ),
        (
            ".als",
            """sig Sensor { state: one State, observation: one Observation }
               sig State {}
               sig Observation {}""",
            {"Sensor_state": "categorical", "Sensor_observation": "categorical"},
        ),
        (
            ".zed",
            """┌─ Sensor ──┐
            state : ℝ
            observation : ℕ
            action : 𝔹
            label : SYMBOL
            └──┘""",
            {
                "state": "float",
                "observation": "integer",
                "action": "binary",
                "label": "categorical",
            },
        ),
    ],
    ids=["xsd", "asn1", "pkl", "alloy", "z-notation"],
)
def test_authored_schema_is_parsed_without_embedded_model_data(
    tmp_path: Path, extension: str, content: str, expected: dict[str, str]
) -> None:
    source = tmp_path / f"sensor{extension}"
    source.write_text(content, encoding="utf-8")

    result = GNNParsingSystem().parse_file(source)

    assert result.success, result.errors
    actual = {
        variable.name: variable.data_type.value for variable in result.model.variables
    }
    for name, data_type in expected.items():
        assert actual[name] == data_type
    observation = next(
        variable
        for variable in result.model.variables
        if "observation" in variable.name
    )
    assert observation.var_type is VariableType.OBSERVATION
    assert result.source_file == str(source)


def test_authored_xml_preserves_each_declaration_once(tmp_path: Path) -> None:
    source = tmp_path / "sensor.xml"
    source.write_text(
        """<gnn name="Sensor" version="2.0">
        <metadata><annotation>Measured sensor calibration</annotation></metadata>
        <variables>
          <variable name="state" type="hidden_state" dimensions="2" data_type="categorical"/>
          <variable name="observation" type="observation" dimensions="3" data_type="integer">
            <description>zero-based symbol</description>
          </variable>
        </variables>
        <connections><connection from="state" to="observation" weight="0.75"/></connections>
        <parameters><parameter name="gain" value="0.125" type="float"/></parameters>
        <equations><equation label="predict" format="text">o = A s</equation></equations>
        <time_specification type="Dynamic" discretization="Discrete" horizon="4"/>
        <ontology_mappings><mapping variable="state" term="HiddenState"/></ontology_mappings>
        </gnn>""",
        encoding="utf-8",
    )

    result = GNNParsingSystem().parse_file(source)

    assert result.success, result.errors
    assert result.model.model_name == "Sensor"
    assert result.model.version == "2.0"
    assert result.model.annotation == "Measured sensor calibration"
    assert [variable.name for variable in result.model.variables] == [
        "state",
        "observation",
    ]
    assert result.model.variables[1].dimensions == [3]
    assert result.model.variables[1].description == "zero-based symbol"
    assert [
        (parameter.name, parameter.value) for parameter in result.model.parameters
    ] == [("gain", "0.125")]
    assert [
        (equation.label, equation.content) for equation in result.model.equations
    ] == [("predict", "o = A s")]
    assert [
        (mapping.variable_name, mapping.ontology_term)
        for mapping in result.model.ontology_mappings
    ] == [("state", "HiddenState")]
    assert result.model.time_specification.horizon == 4
    assert result.model.connections[0].weight == 0.75


def test_authored_xml_retains_distinct_physical_same_name_declarations(
    tmp_path: Path,
) -> None:
    source = tmp_path / "duplicate_names.xml"
    source.write_text(
        """<gnn name="Ambiguous author declarations"><variables>
        <variable name="state" dimensions="2"/>
        <variable name="state" dimensions="3"/>
        </variables></gnn>""",
        encoding="utf-8",
    )

    result = GNNParsingSystem().parse_file(source)

    # Parsing keeps source declarations available for downstream validation;
    # XPath overlap must not manufacture duplicates or normalize author input.
    assert result.success, result.errors
    assert [
        (variable.name, variable.dimensions) for variable in result.model.variables
    ] == [("state", [2]), ("state", [3])]


def test_symbolic_maxima_source_preserves_model_axes_constants_and_dependencies(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sensor.mac"
    source.write_text(
        """/* Model: SymbolicSensor */
likelihood: matrix([0.8,0.2],[0.3,0.7],[0.1,0.9]);
state: matrix([0.25],[0.75]);
gain: -0.125;
steps: 3;
phase: %pi / 2;
decay: %e ^ -gain;
predict(state, gain) := gain * state;
solve(gain * state = 1, state);
""",
        encoding="utf-8",
    )

    result = GNNParsingSystem().parse_file(source, GNNFormat.MAXIMA)

    assert result.success, result.errors
    assert result.model.model_name == "SymbolicSensor"
    variables = {variable.name: variable for variable in result.model.variables}
    assert variables["likelihood"].dimensions == [3, 2]
    assert variables["likelihood"].var_type is VariableType.LIKELIHOOD_MATRIX
    assert variables["state"].dimensions == [2, 1]
    parameters = {
        parameter.name: parameter.value for parameter in result.model.parameters
    }
    assert parameters["gain"] == -0.125
    assert parameters["steps"] == 3
    assert "%pi / 2" in parameters["phase"]
    assert "%e ^ -gain" in parameters["decay"]
    equations = {equation.label: equation for equation in result.model.equations}
    assert equations["predict"].content == "predict(state, gain) := gain * state"
    assert equations["solve_1"].content == "solve(gain * state = 1, state)"
    assert {
        (connection.source_variables[0], connection.target_variables[0])
        for connection in result.model.connections
    } == {("state", "predict"), ("gain", "predict")}


def test_coq_authored_declarations_retain_types_and_unverified_theorem_text(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sensor.v"
    source.write_text(
        """Require Import Reals.
Require Import Lists.
Module SensorEvidence.
Parameter hidden_state : nat.
Parameter observation : R.
Variable control_action : bool.
Definition preference : R := 2.
Theorem natural_identity : forall n : nat, n = n.
Proof. intros. reflexivity. Qed.
End SensorEvidence.
""",
        encoding="utf-8",
    )

    result = GNNParsingSystem().parse_file(source, GNNFormat.COQ)

    assert result.success, result.errors
    assert result.model.model_name == "SensorEvidence"
    variables = {variable.name: variable for variable in result.model.variables}
    assert variables["hidden_state"].var_type is VariableType.HIDDEN_STATE
    assert variables["hidden_state"].data_type is DataType.INTEGER
    assert variables["observation"].var_type is VariableType.OBSERVATION
    assert variables["observation"].data_type is DataType.CONTINUOUS
    assert variables["control_action"].var_type is VariableType.ACTION
    assert variables["control_action"].data_type is DataType.BINARY
    assert variables["preference"].data_type is DataType.CONTINUOUS
    assert result.model.extensions["coq_requires"] == ["Reals", "Lists"]
    assert result.model.extensions["uses_reals"]
    assert result.model.extensions["uses_lists"]
    # This consumer extracts declarations, not a compiler/proof acceptance receipt.
    theorem = next(
        parameter
        for parameter in result.model.parameters
        if parameter.name == "theorem_natural_identity"
    )
    assert theorem.value == "forall n : nat, n = n"
    assert theorem.type_hint == "theorem"

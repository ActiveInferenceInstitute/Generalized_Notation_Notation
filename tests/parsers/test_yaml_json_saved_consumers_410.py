"""Native public saved-configuration consumers retain authored data and failures."""

import json
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from gnn.parsers import GNNParsingSystem
from gnn.parsers.common import (
    ConnectionType,
    DataType,
    GNNInternalRepresentation,
    ParseError,
    TimeSpecification,
)
from gnn.schema_validator import GNNParser

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def authored_values() -> dict:
    return {
        "calibration": [[-0.25, 0.0, 1.5], [0.0, -2.0, 3.0]],
        "offset": 0,
        "enabled": False,
        "settings": {"bias": 0, "enabled": False, "labels": ["α", "zero"]},
    }


def canonical_document() -> dict:
    return {
        "model_name": "Authored calibration α",
        "version": "1.1",
        "annotation": "Authored saved configuration; no inference provenance.",
        "variables": [
            {
                "name": "calibration",
                "var_type": "hidden_state",
                "data_type": "float",
                "dimensions": [2, 3],
                "description": "signed and zero calibration",
                "constraints": {"min": -2, "enabled": False},
            },
            {
                "name": "observation",
                "var_type": "observation",
                "data_type": "integer",
                "dimensions": [2],
                "description": "reported count",
            },
        ],
        "connections": [
            {
                "source_variables": ["calibration"],
                "target_variables": ["observation"],
                "connection_type": "directed",
                "weight": 0.0,
                "annotation": "authored: α",
                "description": "reported relation",
            }
        ],
        "parameters": [
            {"name": name, "value": value, "description": "supplied " + name}
            for name, value in authored_values().items()
        ],
        "equations": [
            {
                "label": "identity",
                "content": "x = x",
                "format": "latex",
                "description": "authored relation",
            }
        ],
        "time_specification": {
            "time_type": "Dynamic",
            "discretization": "DiscreteTime",
            "horizon": 0,
        },
        "ontology_mappings": [
            {
                "variable_name": "calibration",
                "ontology_term": "HiddenState",
                "description": "authored identity",
            }
        ],
        "created_at": "2026-10-01T12:00:00+00:00",
        "modified_at": "2026-10-02T13:30:00+00:00",
        "checksum": "authored-checksum-identity",
        "extensions": {"reviewed": False, "offset": 0, "note": "manual α"},
        "raw_sections": {
            "Footer": "manual footer  \n",
            "Signature": "authored signature",
        },
    }


def assert_model_values(model) -> None:
    assert model.model_name == "Authored calibration α"
    assert model.version == "1.1"
    assert model.annotation == "Authored saved configuration; no inference provenance."
    variables = {variable.name: variable for variable in model.variables}
    assert set(variables) == {"calibration", "observation"}
    assert variables["calibration"].dimensions == [2, 3]
    assert variables["calibration"].data_type is DataType.FLOAT
    assert variables["calibration"].description == "signed and zero calibration"
    assert variables["observation"].dimensions == [2]
    assert variables["observation"].data_type is DataType.INTEGER
    assert {
        parameter.name: parameter.value for parameter in model.parameters
    } == authored_values()
    assert (
        type(
            {parameter.name: parameter.value for parameter in model.parameters}[
                "offset"
            ]
        )
        is int
    )
    assert {parameter.name: parameter.value for parameter in model.parameters}[
        "enabled"
    ] is False
    settings = {parameter.name: parameter.value for parameter in model.parameters}[
        "settings"
    ]
    assert settings["enabled"] is False
    assert type(settings["bias"]) is int
    assert len(model.connections) == 1
    connection = model.connections[0]
    assert connection.source_variables == ["calibration"]
    assert connection.target_variables == ["observation"]
    assert connection.connection_type is ConnectionType.DIRECTED
    assert connection.weight == 0.0
    assert type(connection.weight) is float
    assert connection.annotation == "authored: α"
    assert model.time_specification.time_type == "Dynamic"
    assert model.time_specification.discretization == "DiscreteTime"
    assert model.time_specification.horizon == 0
    assert model.created_at == datetime.fromisoformat("2026-10-01T12:00:00+00:00")
    assert model.modified_at == datetime.fromisoformat("2026-10-02T13:30:00+00:00")
    assert model.checksum == "authored-checksum-identity"
    assert model.extensions == {"reviewed": False, "offset": 0, "note": "manual α"}


def test_authored_saved_json_public_registry_and_fresh_reopen_preserve_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "authored.json"
    source.write_text(
        json.dumps(canonical_document(), ensure_ascii=False), encoding="utf-8"
    )
    original = source.read_bytes()
    system = GNNParsingSystem()
    parsed = system.parse_file(source)
    assert parsed.success, parsed.errors
    assert_model_values(parsed.model)
    assert parsed.source_file == str(source)
    assert parsed.model.variables[0].constraints == {"min": -2, "enabled": False}
    assert parsed.model.raw_sections == canonical_document()["raw_sections"]
    assert parsed.model.equations[0].content == "x = x"
    assert parsed.model.equations[0].format == "latex"
    assert parsed.model.ontology_mappings[0].ontology_term == "HiddenState"
    parsed.model.parameters[0].value[0][0] = 99
    parsed.model.extensions["reviewed"] = True
    reopened = system.parse_file(source)
    assert reopened.success, reopened.errors
    assert reopened.source_file == str(source)
    assert_model_values(reopened.model)
    assert source.read_bytes() == original


@pytest.mark.parametrize("layout", ["compact_list", "named_scalar_specs"])
def test_authored_yaml_compact_config_saved_json_keeps_declarations_and_scalars(
    tmp_path: Path, layout: str
) -> None:
    variables = (
        ["state[2],float", "observation[1],integer"]
        if layout == "compact_list"
        else {"state": "float", "observation": "integer"}
    )
    document = {
        "name": "Authored compact YAML",
        "variables": variables,
        "parameters": [
            {
                "name": "offset",
                "value": 0,
                "type": "integer",
                "description": "exact zero",
            },
            {
                "name": "enabled",
                "value": False,
                "type": "binary",
                "description": "authored switch",
            },
        ],
        "connections": ["state>observation", "state-observation"],
        "equations": ["x = x"],
        "time": "Static",
        "ontology_mappings": [{"variable": "state", "term": "HiddenState"}],
        "created_at": datetime.fromisoformat("2026-10-01T12:00:00+00:00"),
        "modified_at": datetime.fromisoformat("2026-10-02T13:30:00+00:00"),
    }
    source = tmp_path / "compact.yaml"
    source.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    original = source.read_bytes()
    system = GNNParsingSystem()
    destination = tmp_path / "saved.json"
    system.convert_file(source, destination)
    saved = destination.read_bytes()
    reopened = system.parse_file(destination)
    assert reopened.success, reopened.errors
    assert reopened.source_file == str(destination)
    model = reopened.model
    assert model.model_name == "Authored compact YAML"
    assert [(variable.name, variable.data_type) for variable in model.variables] == [
        ("state", DataType.FLOAT),
        ("observation", DataType.INTEGER),
    ]
    assert [variable.dimensions for variable in model.variables] == (
        [[2], [1]] if layout == "compact_list" else [[], []]
    )
    parameters = {parameter.name: parameter for parameter in model.parameters}
    assert parameters["offset"].value == 0
    assert type(parameters["offset"].value) is int
    assert parameters["offset"].type_hint == "integer"
    assert parameters["offset"].description == "exact zero"
    assert parameters["enabled"].value is False
    assert parameters["enabled"].type_hint == "binary"
    assert parameters["enabled"].description == "authored switch"
    assert [
        (
            connection.source_variables,
            connection.target_variables,
            connection.connection_type,
        )
        for connection in model.connections
    ] == [
        (["state"], ["observation"], ConnectionType.DIRECTED),
        (["state"], ["observation"], ConnectionType.UNDIRECTED),
    ]
    assert model.equations[0].content == "x = x"
    assert model.time_specification.time_type == "Static"
    assert model.time_specification.horizon is None
    assert model.ontology_mappings[0].variable_name == "state"
    assert model.ontology_mappings[0].ontology_term == "HiddenState"
    assert model.created_at == datetime.fromisoformat("2026-10-01T12:00:00+00:00")
    assert model.modified_at == datetime.fromisoformat("2026-10-02T13:30:00+00:00")
    assert source.read_bytes() == original
    assert destination.read_bytes() == saved


def test_saved_json_schema_facade_preserves_declared_datatype(tmp_path: Path) -> None:
    source = tmp_path / "authored.json"
    source.write_text(
        json.dumps(canonical_document(), ensure_ascii=False), encoding="utf-8"
    )
    original = source.read_bytes()
    facade = GNNParser().parse_file(source)
    assert facade.model_name == "Authored calibration α"
    assert facade.parameters == authored_values()
    assert facade.variables["calibration"].dimensions == [2, 3]
    assert facade.variables["observation"].data_type == "integer"
    assert source.read_bytes() == original


@pytest.mark.parametrize("style", ["canonical_list", "authored_aliases"])
@pytest.mark.parametrize("suffix", [".yaml", ".yml"])
def test_authored_yaml_saved_config_public_json_consumer_preserves_values(
    tmp_path: Path, style: str, suffix: str
) -> None:
    document = canonical_document()
    if style == "authored_aliases":
        document["name"] = document.pop("model_name")
        document["description"] = document.pop("annotation")
        document["state_space"] = {
            variable["name"]: {
                "type": variable["var_type"],
                "data_type": variable["data_type"],
                "dimensions": variable["dimensions"],
                "description": variable["description"],
            }
            for variable in document.pop("variables")
        }
        document["initial_parameterization"] = authored_values()
        del document["parameters"]
        document["time"] = {
            "type": "Dynamic",
            "discretization": "DiscreteTime",
            "model_time_horizon": 0,
        }
        del document["time_specification"]
        document["actinf_ontology_annotation"] = {"calibration": "HiddenState"}
        del document["ontology_mappings"]
        document["connections"] = [
            {
                "from": "calibration",
                "to": "observation",
                "type": "directed",
                "weight": 0.0,
                "annotation": "authored: α",
                "description": "reported relation",
            }
        ]
        document["equations"] = [
            "x = x",
            {
                "label": "identity",
                "equation": "x = x",
                "description": "authored relation",
            },
        ]
    source = tmp_path / ("authored" + suffix)
    source.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    original = source.read_bytes()
    system = GNNParsingSystem()
    parsed = system.parse_file(source)
    assert parsed.success, parsed.errors
    assert parsed.source_file == str(source)
    assert_model_values(parsed.model)
    assert parsed.model.raw_sections == canonical_document()["raw_sections"]
    assert all(equation.content == "x = x" for equation in parsed.model.equations)
    destination = tmp_path / "converted.json"
    assert system.convert_file(source, destination) == destination
    converted = destination.read_bytes()
    reopened = system.parse_file(destination)
    assert reopened.success, reopened.errors
    assert reopened.source_file == str(destination)
    assert_model_values(reopened.model)
    assert destination.read_bytes() == converted
    assert source.read_bytes() == original


@pytest.mark.parametrize("payload", [b"{not json", b"[1,2]", b'"scalar"', b"\xff"])
def test_saved_json_refusal_preserves_source_and_prior_destination(
    tmp_path: Path, payload: bytes
) -> None:
    source = tmp_path / "invalid.json"
    source.write_bytes(payload)
    destination = tmp_path / "prior.yaml"
    destination.write_bytes(b"manual prior destination\r\n")
    system = GNNParsingSystem()
    parsed = system.parse_file(source)
    assert not parsed.success
    assert parsed.source_file == str(source)
    assert parsed.errors
    with pytest.raises(ParseError, match=str(source)):
        system.convert_file(source, destination)
    assert source.read_bytes() == payload
    assert destination.read_bytes() == b"manual prior destination\r\n"


def test_missing_saved_source_and_directory_refuse_prior_destination_replacement(
    tmp_path: Path,
) -> None:
    system = GNNParsingSystem()
    destination = tmp_path / "prior.json"
    destination.write_bytes(b"prior artifact")
    missing = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError, match=str(missing)):
        system.convert_file(missing, destination)
    source_directory = tmp_path / "directory.json"
    source_directory.mkdir()
    with pytest.raises(ParseError, match=str(source_directory)):
        system.convert_file(source_directory, destination)
    assert destination.read_bytes() == b"prior artifact"
    assert source_directory.is_dir()


@pytest.mark.parametrize("payload", ["- one\n- two\n", "scalar\n"])
def test_saved_yaml_nonobject_refuses_prior_destination_replacement(
    tmp_path: Path, payload: str
) -> None:
    source = tmp_path / "not_a_model.yaml"
    source.write_text(payload, encoding="utf-8")
    destination = tmp_path / "prior.json"
    destination.write_bytes(b"manual prior artifact")
    system = GNNParsingSystem()
    parsed = system.parse_file(source)
    assert not parsed.success
    assert parsed.source_file == str(source)
    assert parsed.errors
    with pytest.raises(ParseError, match=str(source)):
        system.convert_file(source, destination)
    assert source.read_text(encoding="utf-8") == payload
    assert destination.read_bytes() == b"manual prior artifact"


def test_authored_yaml_leading_comment_is_not_misidentified_as_markdown(
    tmp_path: Path,
) -> None:
    source = tmp_path / "commented.yaml"
    source.write_text(
        "# Authored YAML configuration\n" + yaml.safe_dump(canonical_document()),
        encoding="utf-8",
    )
    before = source.read_bytes()
    parsed = GNNParsingSystem().parse_file(source)
    assert parsed.success, parsed.errors
    assert_model_values(parsed.model)
    assert source.read_bytes() == before


@pytest.mark.parametrize(
    "payload",
    [
        "model_name: broken\nvariables: [\n",
        "!!python/object/apply:builtins.str [authored-scalar]\n",
    ],
)
def test_malformed_or_unsupported_yaml_does_not_appear_as_successful_empty_model(
    tmp_path: Path, payload: str
) -> None:
    source = tmp_path / "invalid.yaml"
    source.write_text(payload, encoding="utf-8")
    destination = tmp_path / "prior.json"
    destination.write_bytes(b"manual prior artifact")
    system = GNNParsingSystem()
    parsed = system.parse_file(source)
    assert not parsed.success, parsed.model
    assert parsed.errors
    with pytest.raises(ParseError, match=str(source)):
        system.convert_file(source, destination)
    assert source.read_text(encoding="utf-8") == payload
    assert destination.read_bytes() == b"manual prior artifact"


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_actual_canonical_saved_time_specification_retains_declared_step_size(
    tmp_path: Path, suffix: str
) -> None:
    model = GNNInternalRepresentation(
        model_name="Authored discrete time metadata",
        time_specification=TimeSpecification(
            time_type="Dynamic",
            discretization="DiscreteTime",
            horizon=4,
            step_size=0.125,
        ),
    )
    system = GNNParsingSystem()
    artifact = tmp_path / ("saved_time" + suffix)
    system.serialize_to_file(model, artifact)
    saved = artifact.read_bytes()
    document = json.loads(saved) if suffix == ".json" else yaml.safe_load(saved)
    assert document["time_specification"]["step_size"] == 0.125
    reopened = system.parse_file(artifact)
    assert reopened.success, reopened.errors
    assert reopened.model.time_specification.step_size == 0.125
    assert reopened.model.time_specification.horizon == 4
    assert artifact.read_bytes() == saved
    assert model.time_specification.step_size == 0.125


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
@pytest.mark.parametrize(
    "datatype, value, expected_enum",
    [("integer", 0, DataType.INTEGER), ("binary", False, DataType.BINARY)],
)
def test_saved_config_schema_facade_exposes_canonical_datatype_and_exact_value(
    tmp_path: Path, suffix: str, datatype: str, value: object, expected_enum: DataType
) -> None:
    document = {
        "model_name": "Authored reported value",
        "variables": [
            {
                "name": "reported",
                "var_type": "observation",
                "data_type": datatype,
                "dimensions": [1],
                "description": "authored declaration",
            }
        ],
        "parameters": [{"name": "reported", "value": value}],
    }
    source = tmp_path / ("reported" + suffix)
    source.write_text(
        json.dumps(document) if suffix == ".json" else yaml.safe_dump(document),
        encoding="utf-8",
    )
    before = source.read_bytes()
    native = GNNParsingSystem().parse_file(source)
    assert native.success, native.errors
    assert native.model.variables[0].data_type is expected_enum
    assert native.model.variables[0].dimensions == [1]
    assert native.model.parameters[0].value == value
    assert type(native.model.parameters[0].value) is type(value)
    facade = GNNParser().parse_file(source)
    assert facade.variables["reported"].data_type == datatype
    assert facade.variables["reported"].dimensions == [1]
    assert facade.variables["reported"].description == "authored declaration"
    assert facade.parameters["reported"] == value
    assert type(facade.parameters["reported"]) is type(value)
    assert source.read_bytes() == before


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
@pytest.mark.parametrize("step_size", [0.0, None])
def test_canonical_saved_time_zero_or_absence_is_not_inferred(
    tmp_path: Path, suffix: str, step_size: float | None
) -> None:
    model = GNNInternalRepresentation(
        model_name="Authored time field",
        time_specification=TimeSpecification(
            time_type="Dynamic", horizon=0, step_size=step_size
        ),
    )
    artifact = tmp_path / ("saved_time" + suffix)
    system = GNNParsingSystem()
    system.serialize_to_file(model, artifact)
    before = artifact.read_bytes()
    saved = json.loads(before) if suffix == ".json" else yaml.safe_load(before)
    assert saved["time_specification"]["step_size"] == step_size
    reopened = system.parse_file(artifact)
    assert reopened.success, reopened.errors
    assert reopened.model.time_specification.step_size == step_size
    assert type(reopened.model.time_specification.step_size) is type(step_size)
    assert reopened.model.time_specification.horizon == 0
    assert artifact.read_bytes() == before
    assert model.time_specification.step_size == step_size

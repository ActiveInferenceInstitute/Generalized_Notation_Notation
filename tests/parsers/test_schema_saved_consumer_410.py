"""Authored saved schemas retain interchange values and causal input failures.

These file-based consumers use the public registry. They do not execute schema
languages or invoke PKL's separate native-evaluator helper.
"""

import hashlib
import json
from pathlib import Path

import pytest

from gnn.parsers import GNNFormat, GNNParsingSystem
from gnn.parsers.common import DataType, ParseError, VariableType

SCHEMAS = [
    (
        GNNFormat.XSD,
        ".xsd",
        '<!-- MODEL_DATA: {data} -->\n<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"><xs:element name="decoy" type="xs:string"/></xs:schema>',
    ),
    (
        GNNFormat.ASN1,
        ".asn1",
        "-- MODEL_DATA: {data}\nDecoy DEFINITIONS ::= BEGIN\nIgnored ::= SEQUENCE {{ decoy INTEGER }}\nEND",
    ),
    (
        GNNFormat.PKL,
        ".pkl",
        "/* MODEL_DATA: {data} */\nclass Decoy {{ state: String }}",
    ),
    (GNNFormat.ALLOY, ".als", "/* MODEL_DATA: {data} */\nmodule decoy\nsig Decoy {{}}"),
    (
        GNNFormat.Z_NOTATION,
        ".zed",
        "% MODEL_DATA: {data}\n\\begin{{schema}}{{Decoy}}\ndecoy : \\nat\n\\end{{schema}}",
    ),
]


def saved_model_data() -> dict:
    return {
        "model_name": "Saved sensor α",
        "annotation": "Signed calibration with authored label metadata",
        "version": "1.0",
        "variables": [
            {
                "name": "state",
                "var_type": "hidden_state",
                "data_type": "float",
                "dimensions": [2, 3],
                "description": "sensor state",
            },
            {
                "name": "observation",
                "var_type": "observation",
                "data_type": "integer",
                "dimensions": [2],
                "description": "sensor count",
            },
        ],
        "connections": [
            {
                "source_variables": ["state"],
                "target_variables": ["observation"],
                "connection_type": "directed",
                "annotation": "measurement α",
                "description": "reported relation",
            }
        ],
        "parameters": [
            {
                "name": "calibration",
                "value": [[-0.125, 0.0, 0.25], [0.0, 1.0, 0.0]],
                "description": "authored coefficients",
            },
            {"name": "offset", "value": 0, "description": "zero offset"},
            {"name": "enabled", "value": False, "description": "explicit false"},
            {
                "name": "labels",
                "value": {"sensor": "α", "indices": [0, 1]},
                "description": "label metadata",
            },
        ],
        "time_specification": {
            "time_type": "Dynamic",
            "discretization": "Discrete",
            "horizon": 5,
            "step_size": 0.25,
        },
        "ontology_mappings": [
            {
                "variable_name": "state",
                "ontology_term": "HiddenState",
                "description": "supplied mapping",
            }
        ],
    }


@pytest.mark.parametrize(
    "format_hint,suffix,template", SCHEMAS, ids=[item[0].value for item in SCHEMAS]
)
def test_saved_schema_interchange_is_authoritative_and_preserves_values(
    tmp_path: Path, format_hint: GNNFormat, suffix: str, template: str
) -> None:
    source = tmp_path / f"sensor{suffix}"
    payload = saved_model_data()
    source.write_text(
        template.format(data=json.dumps(payload, ensure_ascii=False)), encoding="utf-8"
    )
    before = hashlib.sha256(source.read_bytes()).hexdigest()

    result = GNNParsingSystem().parse_file(source)

    assert result.success, result.errors
    assert result.source_file == str(source)
    assert result.model.source_format is format_hint
    assert result.model.model_name == "Saved sensor α"
    assert result.model.annotation == payload["annotation"]
    variables = {variable.name: variable for variable in result.model.variables}
    assert set(variables) == {"state", "observation"}
    assert variables["state"].dimensions == [2, 3]
    assert variables["state"].data_type is DataType.FLOAT
    assert variables["observation"].var_type is VariableType.OBSERVATION
    assert variables["observation"].data_type is DataType.INTEGER
    assert {
        parameter.name: parameter.value for parameter in result.model.parameters
    } == {
        "calibration": [[-0.125, 0.0, 0.25], [0.0, 1.0, 0.0]],
        "offset": 0,
        "enabled": False,
        "labels": {"sensor": "α", "indices": [0, 1]},
    }
    assert [parameter.description for parameter in result.model.parameters] == [
        parameter["description"] for parameter in payload["parameters"]
    ]
    assert len(result.model.connections) == 1
    assert result.model.connections[0].source_variables == ["state"]
    assert result.model.connections[0].target_variables == ["observation"]
    assert result.model.connections[0].annotation == "measurement α"
    assert result.model.time_specification.horizon == 5
    assert result.model.time_specification.step_size == 0.25
    assert result.model.time_specification.discretization == "Discrete"
    assert [
        (mapping.variable_name, mapping.ontology_term, mapping.description)
        for mapping in result.model.ontology_mappings
    ] == [("state", "HiddenState", "supplied mapping")]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


@pytest.mark.parametrize(
    "format_hint,suffix,template", SCHEMAS, ids=[item[0].value for item in SCHEMAS]
)
@pytest.mark.parametrize("defect", ["variable_name", "parameter_value"])
def test_malformed_saved_schema_is_failed_with_actual_file_identity(
    tmp_path: Path, format_hint: GNNFormat, suffix: str, template: str, defect: str
) -> None:
    source = tmp_path / f"malformed{suffix}"
    payload = saved_model_data()
    if defect == "variable_name":
        del payload["variables"][0]["name"]
        missing_field = "name"
    else:
        del payload["parameters"][0]["value"]
        missing_field = "value"
    source.write_text(template.format(data=json.dumps(payload)), encoding="utf-8")
    original = source.read_bytes()

    result = GNNParsingSystem().parse_file(source)

    assert not result.success
    assert result.source_file == str(source)
    assert result.model.source_format is format_hint
    assert any(
        "Failed to parse embedded data" in error and missing_field in error
        for error in result.errors
    ), result.errors
    assert source.read_bytes() == original


@pytest.mark.parametrize(
    "format_hint,suffix,template", SCHEMAS, ids=[item[0].value for item in SCHEMAS]
)
def test_saved_schema_invalid_utf8_is_a_read_failure(
    tmp_path: Path, format_hint: GNNFormat, suffix: str, template: str
) -> None:
    source = tmp_path / f"invalid-encoding{suffix}"
    source.write_bytes(b"\xff\xfe not UTF-8 schema")

    result = GNNParsingSystem().parse_file(source)

    assert not result.success
    assert result.source_file == str(source)
    assert result.model.source_format is format_hint
    assert any("utf-8" in error and "decode" in error for error in result.errors), (
        result.errors
    )
    assert result.model.variables == []
    assert result.model.parameters == []
    assert source.read_bytes() == b"\xff\xfe not UTF-8 schema"


def test_authored_pkl_mapping_admission_preserves_names_and_literal_text(
    tmp_path: Path,
) -> None:
    source = tmp_path / "mapping.pkl"
    source.write_text(
        """class Variable {}
class Sensor {
    state: List<Float>
    observation: List<Int>
    action: List<Boolean>
    gain: Float = -0.125
}
variables: Mapping<String, Variable> = new Mapping {
    ["reported_state"] = new Variable {}
}
parameters: Mapping<String, Float> = new Mapping {
    ["zero"] = 0
    ["signed_gain"] = -0.125
}
""",
        encoding="utf-8",
    )
    before = source.read_bytes()

    result = GNNParsingSystem().parse_file(source)

    assert result.success, result.errors
    assert result.source_file == str(source)
    variables = {variable.name: variable for variable in result.model.variables}
    assert set(variables) == {"state", "observation", "action", "reported_state"}
    assert variables["state"].data_type is DataType.FLOAT
    assert variables["observation"].data_type is DataType.INTEGER
    assert variables["action"].data_type is DataType.BINARY
    assert all(
        variables[name].dimensions == [1] for name in ("state", "observation", "action")
    )
    assert variables["reported_state"].var_type is VariableType.HIDDEN_STATE
    assert {
        parameter.name: parameter.value for parameter in result.model.parameters
    } == {"gain": "-0.125", "zero": "0", "signed_gain": "-0.125"}
    assert (
        next(
            parameter
            for parameter in result.model.parameters
            if parameter.name == "gain"
        ).type_hint
        == "Float"
    )
    assert source.read_bytes() == before


@pytest.mark.parametrize(
    "format_hint,suffix,prefix,native",
    [
        (
            GNNFormat.ASN1,
            ".asn1",
            "--",
            "Decoy DEFINITIONS ::= BEGIN\nIgnored ::= SEQUENCE { decoy INTEGER }\nEND",
        ),
        (
            GNNFormat.Z_NOTATION,
            ".zed",
            "%",
            "┌─ Decoy ──┐\nstate : ℝ\nstate ∈ {0, 1}\n└──┘",
        ),
    ],
    ids=["asn1", "z-notation"],
)
@pytest.mark.parametrize("position", ["trailing", "inline"])
def test_line_metadata_retains_saved_identity_in_supported_positions(
    tmp_path: Path,
    format_hint: GNNFormat,
    suffix: str,
    prefix: str,
    native: str,
    position: str,
) -> None:
    source = tmp_path / f"position{suffix}"
    metadata = f"{prefix} MODEL_DATA: {json.dumps(saved_model_data())}"
    if position == "trailing":
        content = f"{native}\n{metadata}\n"
    else:
        content = f"{native.splitlines()[0]} {metadata}\n" + "\n".join(
            native.splitlines()[1:]
        )
    source.write_text(content, encoding="utf-8")

    result = GNNParsingSystem().parse_file(source)

    assert result.success, result.errors
    assert result.model.model_name == "Saved sensor α"
    assert result.source_file == str(source)
    assert result.model.source_format is format_hint
    assert {parameter.name: parameter.value for parameter in result.model.parameters}[
        "enabled"
    ] is False
    assert {
        variable.name: variable.dimensions for variable in result.model.variables
    } == {"state": [2, 3], "observation": [2]}


@pytest.mark.parametrize(
    "suffix,prefix,native",
    [
        (
            ".asn1",
            "--",
            "Decoy DEFINITIONS ::= BEGIN\nIgnored ::= SEQUENCE { decoy INTEGER }\nEND",
        ),
        (".zed", "%", "┌─ Decoy ──┐\nstate : ℝ\nstate ∈ {0, 1}\n└──┘"),
    ],
    ids=["asn1", "z-notation"],
)
@pytest.mark.parametrize(
    "payload", ['{"model_name": "incomplete"', "[]", "null", '"wrong object type"']
)
def test_present_invalid_line_payload_refuses_native_fallback_and_file_conversion(
    tmp_path: Path, suffix: str, prefix: str, native: str, payload: str
) -> None:
    source = tmp_path / f"malformed-comment{suffix}"
    source.write_text(f"{prefix} MODEL_DATA: {payload}\n{native}\n", encoding="utf-8")
    original = source.read_bytes()
    output = tmp_path / "previous-result.json"
    output.write_bytes(b"caller-owned previous result")
    system = GNNParsingSystem()

    result = system.parse_file(source)

    assert not result.success
    assert result.source_file == str(source)
    assert any("MODEL_DATA" in error for error in result.errors), result.errors
    assert result.model.variables == []
    assert result.model.parameters == []
    with pytest.raises(ParseError, match="MODEL_DATA") as error:
        system.convert_file(source, output)
    assert str(source) in str(error.value)
    assert output.read_bytes() == b"caller-owned previous result"
    assert source.read_bytes() == original


def test_asn1_existing_block_interchange_preserves_supplied_description(
    tmp_path: Path,
) -> None:
    source = tmp_path / "block.asn1"
    source.write_text(
        "/* MODEL_DATA: "
        + json.dumps(saved_model_data())
        + " */\nDecoy DEFINITIONS ::= BEGIN\nIgnored ::= SEQUENCE { decoy INTEGER }\nEND\n",
        encoding="utf-8",
    )

    result = GNNParsingSystem().parse_file(source)

    assert result.success, result.errors
    assert result.model.model_name == "Saved sensor α"
    assert (
        next(
            parameter
            for parameter in result.model.parameters
            if parameter.name == "calibration"
        ).description
        == "authored coefficients"
    )
    assert {variable.name: variable.dimensions for variable in result.model.variables}[
        "state"
    ] == [2, 3]


def test_z_legacy_double_percent_comment_is_still_admitted(tmp_path: Path) -> None:
    source = tmp_path / "double-percent.zed"
    source.write_text(
        "%% MODEL_DATA: " + json.dumps(saved_model_data()) + "\n", encoding="utf-8"
    )

    result = GNNParsingSystem().parse_file(source)

    assert result.success, result.errors
    assert result.model.model_name == "Saved sensor α"
    assert {parameter.name: parameter.value for parameter in result.model.parameters}[
        "offset"
    ] == 0


@pytest.mark.parametrize(
    "format_hint,suffix,template", SCHEMAS, ids=[item[0].value for item in SCHEMAS]
)
def test_public_file_conversion_reopens_each_saved_schema_with_original_values(
    tmp_path: Path, format_hint: GNNFormat, suffix: str, template: str
) -> None:
    source = tmp_path / "authored.xsd"
    source.write_text(
        SCHEMAS[0][2].format(data=json.dumps(saved_model_data())), encoding="utf-8"
    )
    source_bytes = source.read_bytes()
    destination = tmp_path / "published" / f"converted{suffix}"
    system = GNNParsingSystem()

    result_path = system.convert_file(source, destination)
    reopened = system.parse_file(destination)

    assert result_path == destination
    assert destination.is_file()
    assert reopened.success, reopened.errors
    assert reopened.source_file == str(destination)
    assert reopened.model.source_format is format_hint
    assert reopened.model.model_name == "Saved sensor α"
    assert reopened.model.annotation == saved_model_data()["annotation"]
    assert {
        parameter.name: parameter.value for parameter in reopened.model.parameters
    } == {
        parameter["name"]: parameter["value"]
        for parameter in saved_model_data()["parameters"]
    }
    assert {parameter.name: parameter.value for parameter in reopened.model.parameters}[
        "enabled"
    ] is False
    assert {
        variable.name: variable.dimensions for variable in reopened.model.variables
    } == {"state": [2, 3], "observation": [2]}
    assert reopened.model.time_specification.step_size == 0.25
    assert reopened.model.connections[0].annotation == "measurement α"
    assert source.read_bytes() == source_bytes

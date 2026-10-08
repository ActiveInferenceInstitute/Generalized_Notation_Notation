"""Actual saved files through public registry, unified and schema consumers.

Native schema declaration extraction is not compiler or formal-proof acceptance.
No native PKL compiler or dependency-recovery route is invoked by these cases.
"""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from gnn.parsers import GNNFormat, GNNParsingSystem, UnifiedGNNParser
from gnn.parsers.common import ParseError
from gnn.schema_validator import CrossFormatValidator

NATIVE_SCHEMAS = [
    (
        ".xsd",
        GNNFormat.XSD,
        """<?xml version="1.0"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
 targetNamespace="https://example.invalid/saved/Sensor">
  <xs:element name="state" type="xs:double"/>
  <xs:element name="observation" type="xs:integer"/>
  <xs:element name="enabled" type="xs:boolean"/>
  <xs:element name="label" type="xs:string"/>
</xs:schema>
""",
        "Sensor",
        {
            "state": ([1], "float", "hidden_state"),
            "observation": ([1], "integer", "observation"),
            "enabled": ([1], "binary", "hidden_state"),
            "label": ([1], "categorical", "hidden_state"),
        },
        {},
    ),
    (
        ".asn1",
        GNNFormat.ASN1,
        """Sensor DEFINITIONS ::= BEGIN
Reading ::= SEQUENCE {
 state REAL,
 observation INTEGER,
 enabled BOOLEAN,
 label UTF8STRING,
 samples SEQUENCE OF REAL
}
END
""",
        "SensorModel",
        {
            "state": ([], "float", "hidden_state"),
            "observation": ([], "integer", "observation"),
            "enabled": ([], "binary", "hidden_state"),
            "label": ([], "categorical", "hidden_state"),
            # This native reader gives sequence declarations its documented
            # one-slot structural dimension; it does not infer data length.
            "samples": ([1], "float", "hidden_state"),
        },
        {},
    ),
    (
        ".pkl",
        GNNFormat.PKL,
        """class SensorModel {
 state: Double
 observation: Int
 action: Boolean
 policy: List<Float>
 labels: Mapping<String, String>
 gain: Float = -0.125
 enabled: Boolean = false
 offset: Int = 0
}
""",
        "SensorModel",
        {
            "state": ([], "float", "hidden_state"),
            "observation": ([], "integer", "observation"),
            "action": ([], "binary", "action"),
        },
        {
            "policy": "<List<Float>>",
            "labels": "<Mapping<String, String>>",
            "gain": "-0.125",
            "enabled": "false",
            "offset": "0",
        },
    ),
]


@pytest.mark.parametrize(
    "suffix,format_hint,content,name,variables,parameters",
    NATIVE_SCHEMAS,
    ids=[case[1].value for case in NATIVE_SCHEMAS],
)
def test_public_native_schema_file_conversion_preserves_declared_structure(
    tmp_path, suffix, format_hint, content, name, variables, parameters
):
    source = tmp_path / f"authored{suffix}"
    source.write_text(content, encoding="utf-8")
    before = source.read_bytes()
    registry = GNNParsingSystem()
    parsed = registry.parse_file(source)
    assert parsed.success, parsed.errors
    assert parsed.source_file == str(source)
    assert parsed.model.source_format is format_hint
    assert parsed.model.model_name == name
    assert {
        v.name: (v.dimensions, v.data_type.value, v.var_type.value)
        for v in parsed.model.variables
    } == variables
    assert {p.name: p.value for p in parsed.model.parameters} == parameters
    descriptions = {v.name: v.description for v in parsed.model.variables}
    assert all(descriptions.values())
    alternate = UnifiedGNNParser().parse_file(source)
    assert alternate.success, alternate.errors
    assert alternate.source_file == str(source)
    assert alternate.model.source_format is format_hint
    assert alternate.model.model_name == name
    assert {
        v.name: (v.dimensions, v.data_type.value, v.var_type.value)
        for v in alternate.model.variables
    } == variables
    assert {p.name: p.value for p in alternate.model.parameters} == parameters
    # Native PKL defaults above remain literal text by the existing registry
    # contract; do not relabel them as evaluated numeric/Boolean values.
    for extension in ("json", "yaml", "xml"):
        destination = tmp_path / "converted" / f"model.{extension}"
        returned = registry.convert_file(source, destination)
        assert returned == destination
        reopened = GNNParsingSystem().parse_file(destination)
        assert reopened.success, reopened.errors
        assert reopened.model.model_name == name
        assert reopened.model.annotation == parsed.model.annotation
        assert {
            v.name: (v.dimensions, v.data_type.value, v.var_type.value)
            for v in reopened.model.variables
        } == variables
        assert {v.name: v.description for v in reopened.model.variables} == descriptions
        assert {p.name: p.value for p in reopened.model.parameters} == parameters
    assert source.read_bytes() == before


def saved_document():
    return {
        "model_name": "Byte identity α",
        "version": "1.1",
        "annotation": "Authored saved configuration with independent provenance.",
        "variables": [
            {
                "name": "state",
                "dimensions": [2, 3],
                "data_type": "float",
                "var_type": "hidden_state",
                "description": "signed calibration",
            },
            {
                "name": "observation",
                "dimensions": [2],
                "data_type": "integer",
                "var_type": "observation",
                "description": "count",
            },
        ],
        "connections": [
            {
                "source_variables": ["state"],
                "target_variables": ["observation"],
                "connection_type": "directed",
                "annotation": "reading α",
            }
        ],
        "parameters": [
            {"name": "calibration", "value": [[-0.5, 0.0, 1.5], [0.0, 2.0, -3.0]]},
            {"name": "offset", "value": 0},
            {"name": "enabled", "value": False},
        ],
        "time_specification": {
            "time_type": "Dynamic",
            "discretization": "Discrete",
            "horizon": 6,
            "step_size": 0.125,
        },
        "checksum": "saved-metadata-is-not-a-file-digest",
    }


@pytest.mark.parametrize("syntax", ["json", "yaml"])
def test_public_unified_saved_files_bind_actual_source_digest_and_independent_values(
    tmp_path, syntax
):
    supplied = saved_document()
    content = (
        json.dumps(supplied, ensure_ascii=False, indent=2)
        if syntax == "json"
        else "---\n# syntax: YAML saved configuration\n"
        + yaml.safe_dump(supplied, allow_unicode=True, sort_keys=False)
    )
    source = tmp_path / "model.saved"
    source.write_text(content, encoding="utf-8")
    before = source.read_bytes()
    parser = UnifiedGNNParser()
    result = parser.parse_file(source)
    assert result.success, result.errors
    assert result.source_file == str(source)
    assert result.model.source_format.value == syntax
    assert (
        result.model.checksum == hashlib.md5(before, usedforsecurity=False).hexdigest()
    )
    assert result.model.checksum != supplied["checksum"]
    assert result.model.model_name == "Byte identity α"
    assert result.model.version == "1.1"
    assert result.model.annotation == supplied["annotation"]
    assert {p.name: p.value for p in result.model.parameters} == {
        "calibration": [[-0.5, 0.0, 1.5], [0.0, 2.0, -3.0]],
        "offset": 0,
        "enabled": False,
    }
    assert (
        next(p.value for p in result.model.parameters if p.name == "enabled") is False
    )
    assert {v.name: v.dimensions for v in result.model.variables} == {
        "state": [2, 3],
        "observation": [2],
    }
    assert result.model.connections[0].annotation == "reading α"
    assert result.model.time_specification.step_size == 0.125
    assert result.model.time_specification.horizon == 6
    # The same public parser instance must reopen changed bytes from the same
    # pathname. Its concrete-parser cache must not cache a previous file result.
    changed = content.replace("Byte identity α", "Second saved identity β")
    source.write_text(changed, encoding="utf-8")
    second = parser.parse_file(source)
    assert second.success, second.errors
    assert second.model.model_name == "Second saved identity β"
    assert (
        second.model.checksum
        == hashlib.md5(changed.encode("utf-8"), usedforsecurity=False).hexdigest()
    )
    assert second.model.checksum != result.model.checksum
    assert result.model.model_name == "Byte identity α"
    assert source.read_text(encoding="utf-8") == changed


@pytest.mark.parametrize("suffix", [".xml", ".unrecognized"])
def test_public_unified_ambiguous_xml_reopens_native_xsd_with_correct_identity(
    tmp_path, suffix
):
    content = NATIVE_SCHEMAS[0][2]
    source = tmp_path / f"schema{suffix}"
    source.write_text(content, encoding="utf-8")
    result = UnifiedGNNParser().parse_file(source)
    assert result.success, result.errors
    assert result.source_file == str(source)
    assert result.model.source_format is GNNFormat.XSD
    assert result.model.model_name == "Sensor"
    assert {v.name: v.data_type.value for v in result.model.variables} == {
        "state": "float",
        "observation": "integer",
        "enabled": "binary",
        "label": "categorical",
    }
    assert source.read_text(encoding="utf-8") == content


@pytest.mark.parametrize("syntax", [GNNFormat.JSON, GNNFormat.YAML, GNNFormat.XSD])
def test_public_unified_string_refusal_is_not_an_apparent_success(syntax):
    content = {
        GNNFormat.JSON: '{"model_name":',
        GNNFormat.YAML: "variables: [",
        GNNFormat.XSD: "<xs:schema>",
    }[syntax]
    result = UnifiedGNNParser().parse_string(content, syntax)
    assert result.success is False
    assert result.errors
    assert result.source_file is None
    assert result.model.source_format is syntax
    assert (
        result.model.checksum
        == hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()
    )


@pytest.mark.parametrize("condition", ["missing", "directory", "invalid_utf8"])
def test_public_unified_file_refusal_and_conversion_preserve_destination(
    tmp_path, condition
):
    source = tmp_path / "refused.xsd"
    if condition == "directory":
        source.mkdir()
    elif condition == "invalid_utf8":
        source.write_bytes(b"\xff\xfe\x00")
    destination = tmp_path / "previous.json"
    destination.write_bytes(b'{"previous":0,"enabled":false}')
    before = destination.read_bytes()
    if condition == "missing":
        with pytest.raises(FileNotFoundError, match="refused.xsd"):
            UnifiedGNNParser().parse_file(source)
        with pytest.raises(FileNotFoundError, match="refused.xsd"):
            GNNParsingSystem().convert_file(source, destination)
    else:
        result = UnifiedGNNParser().parse_file(source)
        assert result.success is False
        assert result.source_file == str(source)
        assert any(str(source) in error or "decode" in error for error in result.errors)
        with pytest.raises(ParseError):
            GNNParsingSystem().convert_file(source, destination)
    assert destination.read_bytes() == before
    if condition == "directory":
        assert source.is_dir()
        assert list(source.iterdir()) == []
    elif condition == "invalid_utf8":
        assert source.read_bytes() == b"\xff\xfe\x00"


def schema_files(root, json_sections, yaml_sections):
    directory = root / "schemas"
    directory.mkdir(parents=True)
    (directory / "json.json").write_text(
        json.dumps(
            {"type": "object", "properties": {section: {} for section in json_sections}}
        ),
        encoding="utf-8",
    )
    (directory / "yaml.yaml").write_text(
        yaml.safe_dump(
            {
                "required_sections": yaml_sections,
            }
        ),
        encoding="utf-8",
    )
    return {path: path.read_bytes() for path in directory.iterdir()}


def test_public_schema_definition_consumer_compares_authored_files_without_mutating(
    tmp_path,
):
    expected = [
        "ModelName",
        "StateSpaceBlock",
        "Connections",
        "InitialParameterization",
    ]
    originals = schema_files(tmp_path, expected, expected)
    result = CrossFormatValidator(
        gnn_module_path=tmp_path
    ).validate_schema_definitions_consistency()
    assert result.is_consistent is True, result.inconsistencies
    assert set(result.schema_formats) == {"json", "yaml"}
    assert result.inconsistencies == []
    assert any("Good structural consistency" in warning for warning in result.warnings)
    for suffix in ("xsd.xsd", "proto.proto"):
        assert any(
            str(tmp_path / "schemas" / suffix) in warning for warning in result.warnings
        )
    assert {path: path.read_bytes() for path in originals} == originals


def test_public_schema_definition_consumer_reports_real_different_sections(tmp_path):
    originals = schema_files(
        tmp_path,
        ["ModelName", "StateSpaceBlock", "Connections", "InitialParameterization"],
        ["ModelName", "AuthoredOtherSection"],
    )
    result = CrossFormatValidator(
        gnn_module_path=tmp_path
    ).validate_schema_definitions_consistency()
    assert result.is_consistent is False
    assert (
        "Significant structural differences between JSON and YAML schemas"
        in result.inconsistencies
    )
    assert any("AuthoredOtherSection" in warning for warning in result.warnings)
    assert any("StateSpaceBlock" in warning for warning in result.warnings)
    assert any(
        "yaml schema has poor coverage" in error for error in result.inconsistencies
    )
    assert {path: path.read_bytes() for path in originals} == originals


@pytest.mark.parametrize("syntax,content", [("json", "{"), ("yaml", "sections: [")])
def test_public_schema_definition_corruption_retains_causal_format_failure(
    tmp_path, syntax, content
):
    expected = [
        "ModelName",
        "StateSpaceBlock",
        "Connections",
        "InitialParameterization",
    ]
    schema_files(tmp_path, expected, expected)
    source = tmp_path / "schemas" / f"{syntax}.{syntax}"
    source.write_text(content, encoding="utf-8")
    before = source.read_bytes()
    result = CrossFormatValidator(
        gnn_module_path=tmp_path
    ).validate_schema_definitions_consistency()
    assert result.is_consistent is False
    assert syntax not in result.schema_formats
    assert any(
        f"Failed to load {syntax} schema:" in error for error in result.inconsistencies
    )
    assert source.read_bytes() == before


def test_public_cross_format_file_list_reports_read_failures_without_changing_sources(
    tmp_path, caplog
):
    source = tmp_path / "source.md"
    content = "## GNNSection\nActInfPOMDP\n## ModelName\nSaved source\n"
    source.write_text(content, encoding="utf-8")
    corrupt = tmp_path / "corrupt.md"
    corrupt.write_bytes(b"\xff")
    absent = tmp_path / "absent.md"
    validator = CrossFormatValidator(gnn_module_path=tmp_path)
    validator.configure(output_dir=tmp_path / "validation-output")
    report = validator.validate([source, corrupt, absent])
    assert report["success"] is False
    assert report["files_validated"] == 3
    assert set(report["results"]) == {str(source)}
    assert report["results"][str(source)].metadata["source_format"] == "markdown"
    assert str(corrupt) in caplog.text
    assert str(absent) in caplog.text
    assert source.read_text(encoding="utf-8") == content
    assert corrupt.read_bytes() == b"\xff"
    assert not absent.exists()

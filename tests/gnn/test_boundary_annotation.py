"""Boundary declaration round trips and rejection cases."""

import json
from pathlib import Path

import pytest

from gnn.parsers.json_parser import JSONGNNParser
from gnn.parsers.json_serializer import JSONSerializer
from gnn.parsers.markdown_parser import MarkdownGNNParser
from gnn.parsers.markdown_serializer import MarkdownSerializer
from gnn.parsers.yaml_parser import YAMLGNNParser
from gnn.parsers.yaml_serializer import YAMLSerializer

_MODEL = """## ModelName
Synthetic boundary model

## StateSpaceBlock
temperature[1,float]
flux[1,float]
"""

_DECLARATIONS = """## BoundaryAnnotation
temperature: unit=K; support=cell x step; form=point
flux: unit=?; convention.reference=as received; form=interval; example.axis=face
"""


@pytest.mark.parametrize(
    ("serializer", "parser"),
    [
        (MarkdownSerializer, MarkdownGNNParser),
        (JSONSerializer, JSONGNNParser),
        (YAMLSerializer, YAMLGNNParser),
    ],
)
def test_boundary_declarations_round_trip(serializer, parser):
    parsed = MarkdownGNNParser().parse_string(_DECLARATIONS + "\n" + _MODEL)
    assert parsed.success, parsed.errors
    assert any("example.axis" in message for message in parsed.warnings)
    assert parsed.model.boundary_annotations["flux"]["unit"] == "?"
    exported = serializer().serialize(parsed.model)

    restored = parser().parse_string(exported)
    assert restored.success, restored.errors
    assert restored.model.boundary_annotations == parsed.model.boundary_annotations
    assert any("example.axis" in message for message in restored.warnings)


def test_boundary_section_is_optional():
    parsed = MarkdownGNNParser().parse_string(_MODEL)
    assert parsed.success, parsed.errors
    assert parsed.model.boundary_annotations == {}
    assert "BoundaryAnnotation" not in MarkdownSerializer().serialize(parsed.model)
    assert "boundary_annotation" not in json.loads(
        JSONSerializer().serialize(parsed.model)
    )
    assert "boundary_annotation" not in YAMLSerializer().serialize(parsed.model)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ("flux: form=point\nflux: form=interval", "Duplicate boundary declaration"),
        ("flux: form=point; form=range", "Duplicate boundary field"),
        ("unknown: form=point", "unknown variable"),
        ("flux: form=estimate", "Unsupported boundary value form"),
        ("flux: unit=", "Invalid text"),
        ("flux: unit", "Expected key=value"),
        ("", "at least one declaration"),
        (
            "flux: form=point\n\n## Boundary Annotation\ntemperature: form=point",
            "Duplicate BoundaryAnnotation section",
        ),
    ],
)
def test_reject_invalid_boundary_sections(rows, message):
    parsed = MarkdownGNNParser().parse_string(
        _MODEL + "\n## BoundaryAnnotation\n" + rows
    )
    assert not parsed.success
    assert any(message in error for error in parsed.errors)


def test_json_schema_boundary_fields():
    schema_dir = Path(__file__).resolve().parents[2] / "src/gnn/schemas"
    schema = json.loads((schema_dir / "json.json").read_text())
    assert "boundary_annotation" not in schema["required"]
    import jsonschema

    validator = jsonschema.Draft7Validator(schema["properties"]["boundary_annotation"])
    assert validator.is_valid(
        {"flux": {"unit": "?", "form": "interval", "example.axis": "face"}}
    )
    assert not validator.is_valid({"flux": {"form": "estimate"}})
    assert not validator.is_valid({"flux": {"unit": 12}})


@pytest.mark.parametrize(
    ("parser", "text"),
    [
        (JSONGNNParser, '{"boundary_annotation":{"flux":{"unit":"K","unit":"C"}}}'),
        (JSONGNNParser, '{"boundary_annotation":{"flux":{},"flux":{}}}'),
        (JSONGNNParser, '{"boundary_annotation":{},"boundary_annotation":{}}'),
        (YAMLGNNParser, "boundary_annotation: {flux: {unit: K, unit: C}}"),
        (YAMLGNNParser, "boundary_annotation: {flux: {}, flux: {}}"),
        (YAMLGNNParser, "boundary_annotation: {flux: bad, flux: {form: point}}"),
        (YAMLGNNParser, "boundary_annotation: {}\nboundary_annotation: {}"),
    ],
)
def test_reject_duplicate_boundary_keys_in_structured_formats(parser, text):
    parsed = parser().parse_string(text)
    assert not parsed.success
    assert any("duplicate boundary" in error.lower() for error in parsed.errors)


@pytest.mark.parametrize("parser", [JSONGNNParser, YAMLGNNParser])
def test_boundary_references_formal_state_space_block(parser):
    content = json.dumps(
        {
            "state_space_block": {"flux": {"dimensions": [1], "type": "float"}},
            "boundary_annotation": {"flux": {"form": "point"}},
        }
    )
    parsed = parser().parse_string(content)
    assert parsed.success, parsed.errors
    assert parsed.model.boundary_annotations == {"flux": {"form": "point"}}


def test_formal_markdown_parser_checks_boundary_declarations():
    from gnn.schema_validator.syntax import GNNParser

    parsed = GNNParser().parse_content(_DECLARATIONS + "\n" + _MODEL)
    assert parsed.metadata["boundary_annotation"]["flux"]["form"] == "interval"
    broken = _DECLARATIONS.replace("flux:", "other:") + "\n" + _MODEL
    with pytest.raises(ValueError, match="unknown variable"):
        GNNParser().parse_content(broken)


@pytest.mark.parametrize("spelling", ["pi", "π"])
def test_boundary_uses_each_markdown_parsers_declared_variable_key(spelling):
    from gnn.schema_validator.syntax import GNNParser

    content = f"## StateSpaceBlock\npi[1,float]\n\n## BoundaryAnnotation\n{spelling}: form=point\n"
    registered = MarkdownGNNParser().parse_string(content)
    assert registered.success, registered.errors
    assert registered.model.boundary_annotations == {"π": {"form": "point"}}

    formal = GNNParser().parse_content(content)
    assert set(formal.variables) == {"pi"}
    assert formal.metadata["boundary_annotation"] == {"pi": {"form": "point"}}


def test_boundary_rejects_two_spellings_of_the_same_variable():
    content = "## StateSpaceBlock\npi[1,float]\n## BoundaryAnnotation\npi: form=point\nπ: form=range"
    parsed = MarkdownGNNParser().parse_string(content)
    assert not parsed.success
    assert any("Duplicate boundary declaration" in error for error in parsed.errors)


@pytest.mark.parametrize("extension", ["json", "yaml"])
def test_validator_does_not_recover_invalid_structured_boundary_to_markdown(
    tmp_path, extension
):
    from gnn.schema_validator.validator import GNNValidator
    from gnn.types import ValidationLevel

    path = tmp_path / f"boundary.{extension}"
    path.write_text(
        json.dumps(
            {
                "model_name": "Boundary fixture",
                "variables": [{"name": "flux"}],
                "boundary_annotation": {"flux": {"form": "invalid"}},
            }
        )
    )
    result = GNNValidator().validate_file(path, validation_level=ValidationLevel.BASIC)
    assert not result.is_valid
    assert any("Unsupported boundary value form" in error for error in result.errors)


def test_registered_export_can_round_trip_without_silently_losing_boundary():
    from gnn.parsers.common import GNNFormat, ParseError
    from gnn.parsers.system import GNNParsingSystem

    system = GNNParsingSystem()
    parsed = system.parse_string(_DECLARATIONS + "\n" + _MODEL, GNNFormat.MARKDOWN)
    assert parsed.success, parsed.errors
    exported = system.serialize(parsed.model, GNNFormat.JSON)
    recovered = system.parse_string(exported, GNNFormat.JSON)
    assert recovered.model.boundary_annotations == parsed.model.boundary_annotations

    with pytest.raises(ParseError, match="would drop BoundaryAnnotation"):
        system.serialize(parsed.model, GNNFormat.XML)
    legacy = system.parse_string(_MODEL, GNNFormat.MARKDOWN)
    assert system.serialize(legacy.model, GNNFormat.XML)

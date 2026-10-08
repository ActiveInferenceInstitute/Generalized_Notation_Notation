"""Authored Markdown datatype fields survive public readers and saved artifacts."""

import hashlib
from pathlib import Path

import pytest

from gnn.parsers import GNNParsingSystem
from gnn.parsers.common import DataType
from gnn.schema_validator import GNNParser, GNNValidator, ValidationLevel
from gnn.utils.runtime_safety.safe_eval import safe_literal_eval

pytestmark = [pytest.mark.unit, pytest.mark.fast]

TYPE_CASES = [
    ("int", DataType.INTEGER, [[0], [-2]]),
    ("bool", DataType.BINARY, [[False], [True]]),
    ("binary", DataType.BINARY, [[0], [1]]),
]


def write_authored_source(
    tmp_path: Path, declared_type: str, separator: str, values: list
) -> Path:
    import json

    source = tmp_path / "authored_datatype.md"
    source.write_text(
        "## GNNSection\nAuthoredSavedTypeWitness\n"
        "## GNNVersionAndFlags\nGNN v1\n"
        "## ModelName\nAuthored datatype persistence\n"
        "## ModelAnnotation\nSaved metadata witness; no inference execution.\n"
        "## StateSpaceBlock\n"
        f"reported[2{separator}1{separator}type={declared_type}] # authored type\n"
        "## Connections\nreported>reported\n"
        "## InitialParameterization\nreported=" + json.dumps(values) + "\n"
        "## Equations\nreported = reported\n"
        "## Time\nStatic\n"
        "## ActInfOntologyAnnotation\nreported=Observation\n"
        "## ModelParameters\nnum_timesteps=1\n"
        "## Footer\nAuthored saved type witness.\n",
        encoding="utf-8",
    )
    return source


@pytest.mark.parametrize("separator", [",", ", "])
@pytest.mark.parametrize("declared_type,canonical,values", TYPE_CASES)
def test_native_file_readers_preserve_authored_bracket_datatype(
    tmp_path: Path,
    declared_type: str,
    canonical: DataType,
    values: list,
    separator: str,
) -> None:
    source = write_authored_source(tmp_path, declared_type, separator, values)
    before = source.read_bytes()
    syntax = GNNParser().parse_file(source)
    registry = GNNParsingSystem().parse_file(source)
    assert registry.success, registry.errors
    assert syntax.variables["reported"].data_type == declared_type
    assert syntax.variables["reported"].dimensions == [2, 1]
    assert syntax.variables["reported"].description == "authored type"
    assert syntax.parameters["reported"] == values
    assert len(registry.model.variables) == 1
    variable = registry.model.variables[0]
    assert variable.data_type is canonical
    assert variable.dimensions == [2, 1]
    assert variable.description == "authored type"
    assert {p.name: p.value for p in registry.model.parameters} == {
        "reported": values,
        "num_timesteps": 1,
    }
    assert source.read_bytes() == before


@pytest.mark.parametrize("suffix", [".json", ".xml", ".yaml"])
def test_nested_json_array_public_conversion_preserves_quoted_scalars(
    tmp_path: Path, suffix: str
) -> None:
    values = [[False, "true"], [True, "falsehood and truefalse"]]
    source = write_authored_source(tmp_path, "categorical", ",", values)
    source.write_text(
        source.read_text().replace("reported[2,1,", "reported[2,2,"),
        encoding="utf-8",
    )
    before = source.read_bytes()
    system = GNNParsingSystem()
    syntax = GNNParser().parse_file(source)
    assert syntax.parameters["reported"] == values
    destination = tmp_path / ("saved" + suffix)
    assert system.convert_file(source, destination) == destination
    saved_before = destination.read_bytes()
    reopened = system.parse_file(destination)
    assert reopened.success, reopened.errors
    assert reopened.model.variables[0].dimensions == [2, 2]
    assert reopened.model.variables[0].data_type is DataType.CATEGORICAL
    reopened_values = {p.name: p.value for p in reopened.model.parameters}
    assert reopened_values["reported"] == values
    assert type(reopened_values["reported"][0][0]) is bool
    assert source.read_bytes() == before
    assert destination.read_bytes() == saved_before


def write_array_literal(tmp_path: Path, literal: str) -> Path:
    source = write_authored_source(tmp_path, "categorical", ",", [[0], [1]])
    source.write_text(
        source.read_text().replace("reported=[[0], [1]]", "reported=" + literal),
        encoding="utf-8",
    )
    return source


@pytest.mark.parametrize("depth", [10, 11])
def test_native_json_array_keeps_existing_depth_boundary(
    tmp_path: Path, depth: int
) -> None:
    literal = "[" * depth + "false" + "]" * depth
    source = write_array_literal(tmp_path, literal)
    before = source.read_bytes()
    parsed = GNNParsingSystem().parse_file(source)
    assert parsed.success, parsed.errors
    value = {p.name: p.value for p in parsed.model.parameters}["reported"]
    if depth == 10:
        expected = False
        for _ in range(depth):
            expected = [expected]
        assert value == expected
    else:
        with pytest.raises(ValueError, match="nesting depth 10"):
            safe_literal_eval(literal.replace("false", "False"))
        # The existing manual recovery retains an uninterpreted inner fragment.
        assert value == ["[" * (depth - 1) + "false" + "]" * (depth - 1)]
    assert source.read_bytes() == before


@pytest.mark.parametrize("length", [10_000, 10_001])
def test_native_json_array_keeps_existing_literal_length_boundary(
    tmp_path: Path, length: int
) -> None:
    prefix, suffix = '[[false,"', '"]]'
    label = "x" * (length - len(prefix) - len(suffix))
    literal = prefix + label + suffix
    assert len(literal) == length
    source = write_array_literal(tmp_path, literal)
    before = source.read_bytes()
    parsed = GNNParsingSystem().parse_file(source)
    assert parsed.success, parsed.errors
    value = {p.name: p.value for p in parsed.model.parameters}["reported"]
    if length == 10_000:
        assert value == [[False, label]]
    else:
        with pytest.raises(ValueError, match="10000 characters"):
            safe_literal_eval(literal.replace("false", "False"))
        assert value == ["[false", '"' + label + '"]']
    assert source.read_bytes() == before


@pytest.mark.parametrize(
    "literal,legacy",
    [
        ("[[false],]", ["[false]"]),
        ('[{"enabled":false}]', ['{"enabled":false}']),
        ("[[null]]", ["[null]"]),
    ],
)
def test_rejected_json_array_retains_existing_manual_recovery(
    tmp_path: Path, literal: str, legacy: list
) -> None:
    source = write_array_literal(tmp_path, literal)
    before = source.read_bytes()
    parsed = GNNParsingSystem().parse_file(source)
    assert parsed.success, parsed.errors
    assert {p.name: p.value for p in parsed.model.parameters}["reported"] == legacy
    assert source.read_bytes() == before


def test_native_python_tuple_array_keeps_existing_literal_grammar(
    tmp_path: Path,
) -> None:
    source = write_array_literal(tmp_path, "[(False, True)]")
    before = source.read_bytes()
    parsed = GNNParsingSystem().parse_file(source)
    assert parsed.success, parsed.errors
    value = {p.name: p.value for p in parsed.model.parameters}["reported"]
    assert value == [(False, True)]
    assert type(value[0]) is tuple
    assert source.read_bytes() == before


@pytest.mark.parametrize("separator", [",", ", "])
@pytest.mark.parametrize("declared_type,canonical,values", TYPE_CASES)
def test_validator_saved_artifacts_preserve_declared_datatype_and_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    declared_type: str,
    canonical: DataType,
    values: list,
    separator: str,
) -> None:
    from gnn.testing import round_trip_config

    source = write_authored_source(tmp_path, declared_type, separator, values)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = tmp_path / "saved"
    destination.mkdir()
    sentinel = destination / "existing_manual_note.txt"
    sentinel.write_bytes(b"manual destination bytes\r\n")
    monkeypatch.setitem(round_trip_config.OUTPUT_CONFIG, "save_test_artifacts", True)
    validator = GNNValidator(
        enable_round_trip_testing=True,
        enable_cross_validation=False,
        validation_level=ValidationLevel.ROUND_TRIP,
    )
    validator.round_trip_tester.temp_dir = destination
    result = validator.validate_file(source)
    assert len(result.round_trip_results) == 3, result.warnings
    assert {item.target_format.value for item in result.round_trip_results} == {
        "json",
        "xml",
        "yaml",
    }
    for item in result.round_trip_results:
        assert item.success, (item.errors, item.differences)
        for model in (item.original_model, item.parsed_back_model):
            assert model.variables[0].name == "reported"
            assert model.variables[0].data_type is canonical
            assert model.variables[0].dimensions == [2, 1]
            assert model.variables[0].description == "authored type"
            assert {p.name: p.value for p in model.parameters} == {"reported": values}
            assert type(model.parameters[0].value[0][0]) is type(values[0][0])
    artifacts = sorted(destination.glob("test_model.*"))
    assert {artifact.suffix for artifact in artifacts} == {".json", ".xml", ".yaml"}
    for artifact in artifacts:
        saved_bytes = artifact.read_bytes()
        reopened = GNNParsingSystem().parse_file(artifact)
        assert reopened.success, (artifact, reopened.errors)
        assert reopened.model.variables[0].data_type is canonical
        assert reopened.model.variables[0].dimensions == [2, 1]
        assert reopened.model.variables[0].description == "authored type"
        assert {p.name: p.value for p in reopened.model.parameters} == {
            "reported": values
        }
        assert type(reopened.model.parameters[0].value[0][0]) is type(values[0][0])
        assert artifact.read_bytes() == saved_bytes
    assert sentinel.read_bytes() == b"manual destination bytes\r\n"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash


@pytest.mark.parametrize(
    "declaration,raw_type,canonical",
    [
        ("reported[2,1]", "float", DataType.FLOAT),
        ("reported[2,1,type=float]", "float", DataType.FLOAT),
        ("reported[2,1,type=custom]", "custom", DataType.CATEGORICAL),
    ],
)
def test_saved_file_preserves_default_float_and_existing_unknown_type_policy(
    tmp_path: Path, declaration: str, raw_type: str, canonical: DataType
) -> None:
    source = write_authored_source(tmp_path, "float", ",", [[0], [-2]])
    source.write_text(
        source.read_text().replace("reported[2,1,type=float]", declaration),
        encoding="utf-8",
    )
    before = source.read_bytes()
    syntax = GNNParser().parse_file(source)
    registry = GNNParsingSystem().parse_file(source)
    assert registry.success, registry.errors
    assert syntax.variables["reported"].data_type == raw_type
    assert registry.model.variables[0].data_type is canonical
    assert syntax.variables["reported"].dimensions == [2, 1]
    assert registry.model.variables[0].dimensions == [2, 1]
    assert syntax.parameters["reported"] == [[0], [-2]]
    assert source.read_bytes() == before


def test_native_schema_reader_keeps_legacy_explicit_suffix_precedence(
    tmp_path: Path,
) -> None:
    source = write_authored_source(tmp_path, "bool", ",", [[False], [True]])
    source.write_text(
        source.read_text().replace(
            "reported[2,1,type=bool]", "reported[2,1,type=bool],type=int"
        ),
        encoding="utf-8",
    )
    before = source.read_bytes()
    parsed = GNNParser().parse_file(source)
    assert parsed.variables["reported"].data_type == "int"
    assert parsed.variables["reported"].dimensions == [2, 1]
    assert parsed.variables["reported"].description == "authored type"
    assert source.read_bytes() == before


def test_bracket_default_hint_remains_authored_metadata_not_a_datatype(
    tmp_path: Path,
) -> None:
    source = write_authored_source(tmp_path, "float", ", ", [[0.0], [0.0]])
    source.write_text(
        source.read_text().replace(
            "reported[2, 1, type=float]",
            "reported[2, 1, type=float, default=zeros]",
        ),
        encoding="utf-8",
    )
    before = source.read_bytes()
    syntax = GNNParser().parse_file(source)
    registry = GNNParsingSystem().parse_file(source)
    assert registry.success, registry.errors
    assert syntax.variables["reported"].data_type == "float"
    # Preserve the syntax reader's existing nonnumeric dimension-token policy.
    assert syntax.variables["reported"].dimensions == [2, 1, "default=zeros"]
    assert registry.model.variables[0].data_type is DataType.FLOAT
    assert registry.model.variables[0].dimensions == [2, 1]
    assert "type=float, default=zeros" in registry.model.raw_sections["StateSpaceBlock"]
    assert source.read_bytes() == before

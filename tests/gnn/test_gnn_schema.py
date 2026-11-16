#!/usr/bin/env python3
"""
Tests for gnn/schema.py — parse_connections, parse_state_space,
validate_matrix_dimensions, validate_gnn_object, GNNParseError.
"""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

GNN_SAMPLE = """
## GNNSection
ActInfPOMDP

## ModelName
Test Model

## StateSpaceBlock
A[3,3,type=float]
B[3,3,3,type=float]
s[3,1,type=float]
o[2,1,type=int]

## Connections
A>s
s-o
B>s:transition

## InitialParameterization
A={(0.9,0.1,0.0),(0.0,0.9,0.1),(0.1,0.0,0.9)}
"""


class TestGNNParseError:
    def test_str_no_location(self) -> Any:
        from gnn.schema import GNNParseError

        err = GNNParseError(code="GNN-E001", message="Bad input")
        s = str(err)
        assert "GNN-E001" in s
        assert "Bad input" in s

    def test_str_with_line_and_file(self) -> Any:
        from gnn.schema import GNNParseError

        err = GNNParseError(
            code="GNN-E001", message="Bad input", line=42, file="model.md"
        )
        assert ":42" in str(err)

    def test_str_with_file_shows_filename(self) -> Any:
        from gnn.schema import GNNParseError

        err = GNNParseError(code="GNN-E001", message="Msg", line=5, file="model.md")
        s = str(err)
        assert "model.md" in s
        assert ":5" in s

    def test_default_severity_is_error(self) -> Any:
        from gnn.schema import GNNParseError

        err = GNNParseError(code="GNN-E001", message="x")
        assert err.severity == "error"


class TestParseConnections:
    def test_directed_connection(self) -> Any:
        from gnn.schema import parse_connections

        content = "## Connections\nA>B\n"
        edges, errors = parse_connections(content)
        assert len(edges) == 1
        assert edges[0].source == "A"
        assert edges[0].target == "B"
        assert edges[0].directed is True
        assert not errors

    def test_undirected_connection(self) -> Any:
        from gnn.schema import parse_connections

        content = "## Connections\nA-B\n"
        edges, errors = parse_connections(content)
        assert len(edges) == 1
        assert edges[0].directed is False

    def test_labeled_connection(self) -> Any:
        from gnn.schema import parse_connections

        content = "## Connections\nA>B:myLabel\n"
        edges, errors = parse_connections(content)
        assert len(edges) == 1
        assert edges[0].label == "myLabel"

    def test_multiple_connections_parsed(self) -> Any:
        from gnn.schema import parse_connections

        edges, errors = parse_connections(GNN_SAMPLE)
        assert len(edges) == 3

    def test_invalid_connection_produces_error(self) -> Any:
        from gnn.schema import parse_connections

        content = "## Connections\nNOT_A_CONNECTION\n"
        edges, errors = parse_connections(content)
        assert len(edges) == 0
        assert any("GNN-E005" in e.code for e in errors)

    def test_unknown_variable_produces_warning(self) -> Any:
        from gnn.schema import parse_connections

        content = "## Connections\nX>Y\n"
        _, errors = parse_connections(content, known_variables={"A", "B"})
        assert any("GNN-W002" in e.code for e in errors)

    def test_known_variable_no_warning(self) -> Any:
        from gnn.schema import parse_connections

        content = "## Connections\nA>B\n"
        _, errors = parse_connections(content, known_variables={"A", "B"})
        warning_errors = [e for e in errors if e.code == "GNN-W002"]
        assert len(warning_errors) == 0

    def test_empty_content_returns_empty(self) -> Any:
        from gnn.schema import parse_connections

        edges, errors = parse_connections("")
        assert edges == []
        assert errors == []

    def test_section_outside_connections_ignored(self) -> Any:
        from gnn.schema import parse_connections

        content = "## StateSpaceBlock\nA>B\n## Connections\nC>D\n"
        edges, _ = parse_connections(content)
        assert len(edges) == 1
        assert edges[0].source == "C"


class TestParseStateSpace:
    def test_basic_variable_parsed(self) -> Any:
        from gnn.schema import parse_state_space

        content = "## StateSpaceBlock\nA[3,3,type=float]\n"
        variables, errors = parse_state_space(content)
        assert len(variables) == 1
        assert variables[0].name == "A"
        assert variables[0].dtype == "float"

    def test_multiple_variables_parsed(self) -> Any:
        from gnn.schema import parse_state_space

        variables, errors = parse_state_space(GNN_SAMPLE)
        names = {v.name for v in variables}
        assert "A" in names
        assert "B" in names
        assert "s" in names
        assert "o" in names

    def test_dtype_extracted(self) -> Any:
        from gnn.schema import parse_state_space

        content = "## StateSpaceBlock\nX[2,type=int]\n"
        variables, _ = parse_state_space(content)
        assert variables[0].dtype == "int"

    def test_default_dtype_is_float(self) -> Any:
        from gnn.schema import parse_state_space

        content = "## StateSpaceBlock\nX[3]\n"
        variables, _ = parse_state_space(content)
        assert variables[0].dtype == "float"

    def test_inline_comments_stripped(self) -> Any:
        from gnn.schema import parse_state_space

        content = "## StateSpaceBlock\nA[3,3,type=float]  # likelihood\n"
        variables, errors = parse_state_space(content)
        assert len(variables) == 1
        assert variables[0].name == "A"

    def test_empty_section_returns_empty(self) -> Any:
        from gnn.schema import parse_state_space

        variables, errors = parse_state_space("## StateSpaceBlock\n")
        assert variables == []
        assert errors == []


class TestValidateMatrixDimensions:
    def _validate(self, content: str) -> list[Any]:
        from gnn.schema import parse_state_space, validate_matrix_dimensions

        variables, parse_errors = parse_state_space(content)
        assert parse_errors == []
        return validate_matrix_dimensions(content, variables)

    def test_valid_one_line_vector_assignment_passes(self) -> None:
        content = """
## StateSpaceBlock
C[3,type=float]

## InitialParameterization
C={(0.1, 0.2, 0.7)}
"""

        assert self._validate(content) == []

    def test_one_line_vector_assignment_dimension_mismatch_fails(self) -> None:
        content = """
## StateSpaceBlock
C[4,type=float]

## InitialParameterization
C={(0.1, 0.2, 0.7)}
"""

        errors = self._validate(content)

        assert len(errors) == 1
        assert errors[0].code == "GNN-E002"
        assert "declared shape 4" in errors[0].message
        assert "parameterization has shape 3" in errors[0].message

    def test_valid_3d_tensor_assignment_passes(self) -> None:
        content = """
## StateSpaceBlock
B[3,3,3,type=float]

## InitialParameterization
B={
  ( (1.0,0.0,0.0), (0.0,1.0,0.0), (0.0,0.0,1.0) ),
  ( (0.0,1.0,0.0), (1.0,0.0,0.0), (0.0,0.0,1.0) ),
  ( (0.0,0.0,1.0), (0.0,1.0,0.0), (1.0,0.0,0.0) )
}
"""

        assert self._validate(content) == []

    def test_action_major_3d_tensor_assignment_passes(self) -> None:
        content = """
## StateSpaceBlock
B[4,4,3,type=float]

## InitialParameterization
B={
  ( (0.9,0.1,0.0,0.0), (0.0,0.9,0.1,0.0), (0.0,0.0,0.9,0.1), (0.1,0.0,0.0,0.9) ),
  ( (0.9,0.0,0.0,0.1), (0.1,0.9,0.0,0.0), (0.0,0.1,0.9,0.0), (0.0,0.0,0.1,0.9) ),
  ( (0.8,0.1,0.1,0.0), (0.1,0.8,0.0,0.1), (0.1,0.0,0.8,0.1), (0.0,0.1,0.1,0.8) )
}
"""

        assert self._validate(content) == []

    def test_valid_9x9x5_action_major_tensor_assignment_passes(self) -> None:
        planes = []
        for action_index in range(5):
            rows = []
            for row_index in range(9):
                row = ["0.0"] * 9
                row[(row_index + action_index) % 9] = "1.0"
                rows.append("(" + ",".join(row) + ")")
            planes.append("(" + ",".join(rows) + ")")
        tensor = "{\n" + ",\n".join(planes) + "\n}"
        content = f"""
## StateSpaceBlock
B[9,9,5,type=float]

## InitialParameterization
B={tensor}
"""

        assert self._validate(content) == []

    def test_invalid_3d_tensor_assignment_fails(self) -> None:
        content = """
## StateSpaceBlock
B[2,2,2,type=float]

## InitialParameterization
B={
  ( (1.0,0.0), (0.0,1.0) ),
  ( (0.0,1.0), (1.0,0.0) ),
  ( (0.5,0.5), (0.5,0.5) )
}
"""

        errors = self._validate(content)

        assert len(errors) == 1
        assert errors[0].code == "GNN-E002"
        assert "declared shape 2x2x2" in errors[0].message
        assert "parameterization has shape 3x2x2" in errors[0].message

    def test_ragged_tensor_assignment_fails(self) -> None:
        content = """
## StateSpaceBlock
A[2,2,type=float]

## InitialParameterization
A={(1.0, 0.0), (0.5, 0.25, 0.25)}
"""

        errors = self._validate(content)

        assert len(errors) == 1
        assert errors[0].code == "GNN-E002"
        assert "ragged" in errors[0].message

    def test_packaged_gridworld_template_has_valid_tensor_shape(self) -> None:
        from gnn.schema import parse_state_space, validate_matrix_dimensions

        template = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "gnn"
            / "cli"
            / "template_assets"
            / "pomdp_gridworld_3x3.md"
        )
        content = template.read_text(encoding="utf-8")
        variables, parse_errors = parse_state_space(content, file_path=str(template))

        assert parse_errors == []
        assert (
            validate_matrix_dimensions(content, variables, file_path=str(template))
            == []
        )


class TestValidateGnnObject:
    def test_valid_object_returns_no_errors(self) -> Any:
        from gnn.schema import validate_gnn_object

        obj: dict[str, Any] = {
            "model_name": "Test",
            "state_space": {"A": {"dimensions": [3, 3], "type": "float"}},
            "connections": [],
            "parameters": {},
        }
        errors = validate_gnn_object(obj)
        assert isinstance(errors, list)

    def test_empty_dict_returns_errors(self) -> Any:
        from gnn.schema import validate_gnn_object

        errors = validate_gnn_object({})
        assert isinstance(errors, list)

    def test_schema_reports_the_runtime_validation_mode(self) -> None:
        from importlib.util import find_spec

        from gnn.schema import check_gnn_object

        result = check_gnn_object(
            {
                "gnn_section": "ActInfPOMDP",
                "gnn_version": "1",
                "model_name": 42,
                "state_space": {},
                "connections": [],
            }
        )
        if find_spec("jsonschema") is not None:
            assert result.mode == "draft-2020-12"
            assert any("42 is not of type" in error for error in result.errors)
        else:
            assert result.mode == "required-keys"
            assert not result.errors

    def test_schema_reports_required_keys_when_jsonschema_is_unavailable(self) -> None:
        from unittest.mock import patch

        from gnn.schema import check_gnn_object, validate_gnn_object

        invalid_schema = {
            "gnn_section": "ActInfPOMDP",
            "gnn_version": "1",
            "model_name": 42,
            "state_space": {},
            "connections": [],
        }
        with patch.dict(sys.modules, {"jsonschema": None}):
            result = check_gnn_object(invalid_schema)
            assert result.mode == "required-keys"
            assert not result.errors
            assert validate_gnn_object(invalid_schema) == result.errors
            missing = check_gnn_object({})
            assert missing.mode == result.mode
            assert "Missing required key: 'model_name'" in missing.errors

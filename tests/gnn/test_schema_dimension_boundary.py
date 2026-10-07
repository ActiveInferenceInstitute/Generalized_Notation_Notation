"""Compatibility and diagnostic parity across the schema dimension boundary."""

from dataclasses import asdict

import pytest

import gnn.schema as schema
from gnn.schema import parameter_dimensions, parser, types


@pytest.mark.parametrize(
    ("declaration", "assignment", "expected"),
    [
        ("A[2,2]", "A={{0.8,0.1},{0.2,0.9}}", []),
        ("D[2,1]", "D={0.4,0.6}", []),
        ("B[2,2,3]", "B={{{1,0},{0,1}},{{0,1},{1,0}},{{0.5,0.5},{0.5,0.5}}}", []),
        ("A[2,2]", "A={{0.8,0.1},{0.2}}", ["GNN-E002"]),
        ("A[n,n]", "A={{1}}", []),
        ("A[2,2]", "C={1,2}", ["GNN-W003"]),
    ],
)
def test_dimension_boundary_preserves_layouts_and_diagnostics(
    declaration: str, assignment: str, expected: list[str]
) -> None:
    content = (
        f"## StateSpaceBlock\n{declaration}\n## InitialParameterization\n{assignment}\n"
    )
    variables, errors = schema.parse_state_space(content, file_path="model.md")
    assert not errors
    public = schema.validate_matrix_dimensions(content, variables, file_path="model.md")
    direct = parameter_dimensions.validate_matrix_dimensions(
        content, variables, file_path="model.md"
    )
    assert [asdict(error) for error in public] == [asdict(error) for error in direct]
    assert [error.code for error in public] == expected
    assert all(error.file == "model.md" and error.line == 4 for error in public)


def test_public_schema_exports_preserve_compatibility_identity() -> None:
    assert (
        schema.validate_matrix_dimensions
        is parser.validate_matrix_dimensions
        is parameter_dimensions.validate_matrix_dimensions
    )
    assert schema.GNNVariable is parser.GNNVariable is types.GNNVariable
    assert (
        schema.GNNConnectionEdge is parser.GNNConnectionEdge is types.GNNConnectionEdge
    )
    assert schema.GNNParseError is parser.GNNParseError is types.GNNParseError

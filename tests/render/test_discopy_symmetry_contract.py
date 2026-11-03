"""Provides helper functions: test_discopy_matrix_permutation_metadata_uses_parsed_parameters, test_discopy_matrix_permutation_rejects_mismatched_dimension, test_discopy_matrix_permutation_rejects_missing_matrix, test_discopy_matrix_permutation_metadata_runtime_validator, and 2 more.

Public functions: test_discopy_matrix_permutation_metadata_uses_parsed_parameters, test_discopy_matrix_permutation_rejects_mismatched_dimension, test_discopy_matrix_permutation_rejects_missing_matrix, test_discopy_matrix_permutation_metadata_runtime_validator, test_discopy_generated_script_does_not_install_dependencies, test_discopy_generated_script_exports_permutation_metadata
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from gnn.render.discopy.discopy_renderer import DisCoPyRenderer
from gnn.render.discopy.symmetry import (
    build_matrix_permutation_metadata,
    validate_matrix_permutation_metadata,
)


def test_discopy_matrix_permutation_metadata_uses_parsed_parameters() -> None:
    spec = {
        "initial_parameterization": {
            "A": [[0.9, 0.1], [0.2, 0.8]],
            "matrix_permutations": {"A": [1, 0]},
        }
    }
    metadata = build_matrix_permutation_metadata(spec)
    assert metadata["A"]["shape"] == [2, 2]
    assert metadata["A"]["permutation"] == [1, 0]


def test_discopy_matrix_permutation_rejects_mismatched_dimension() -> None:
    spec = {
        "initial_parameterization": {
            "A": [[0.9, 0.1], [0.2, 0.8]],
            "matrix_permutations": {"A": [0, 1, 2]},
        }
    }
    with pytest.raises(ValueError, match="does not match"):
        build_matrix_permutation_metadata(spec)


def test_discopy_matrix_permutation_rejects_missing_matrix() -> None:
    spec = {"initial_parameterization": {"matrix_permutations": {"B": [0, 1]}}}

    with pytest.raises(ValueError, match="missing matrix"):
        build_matrix_permutation_metadata(spec)


def test_discopy_matrix_permutation_metadata_runtime_validator() -> None:
    validate_matrix_permutation_metadata(
        {"A": {"axis": "rows", "shape": [2, 2], "permutation": [1, 0]}}
    )

    with pytest.raises(ValueError, match="Permutation length"):
        validate_matrix_permutation_metadata(
            {"A": {"axis": "rows", "shape": [2, 2], "permutation": [0]}}
        )


def test_discopy_generated_script_does_not_install_dependencies() -> None:
    renderer = DisCoPyRenderer(options={})
    code = renderer._generate_discopy_diagram_code(
        {"model_name": "demo", "initial_parameterization": {}}, "demo"
    )
    assert "pip install" not in code
    assert "subprocess" not in code
    assert "MATRIX_PERMUTATION_METADATA" in code
    assert "MATRIX_PERMUTATION_APPLIED_TO_DIAGRAM = False" in code
    assert "validate_matrix_permutation_metadata(MATRIX_PERMUTATION_METADATA)" in code
    assert '"matrix_permutation_metadata": MATRIX_PERMUTATION_METADATA' in code
    assert (
        '"matrix_permutation_applied_to_diagram": MATRIX_PERMUTATION_APPLIED_TO_DIAGRAM'
        in code
    )


def test_discopy_generated_script_exports_permutation_metadata(tmp_path: Path) -> None:
    if importlib.util.find_spec("discopy") is None:
        raise AssertionError("DisCoPy optional dependency is not installed")
    renderer = DisCoPyRenderer(options={})
    code = renderer._generate_discopy_diagram_code(
        {
            "model_name": "demo",
            "initial_parameterization": {
                "A": [[0.9, 0.1], [0.2, 0.8]],
                "matrix_permutations": {"A": [1, 0]},
            },
        },
        "demo",
    )
    script_path = tmp_path / "demo_discopy.py"
    script_path.write_text(code, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, script_path.name],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    circuit_info = json.loads(
        (tmp_path / "discopy_diagrams" / "circuit_info.json").read_text(
            encoding="utf-8"
        )
    )
    assert circuit_info["matrix_permutation_metadata"]["A"]["shape"] == [2, 2]
    assert circuit_info["matrix_permutation_metadata"]["A"]["permutation"] == [1, 0]
    assert circuit_info["matrix_permutation_applied_to_diagram"] is False

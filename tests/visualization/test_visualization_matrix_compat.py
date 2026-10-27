"""Tests for gnn/visualization/matrix_compat.py facade.

The facade is a thin re-export forwarding to gnn.visualization.matrix.compat.
These tests verify the forwarding surface resolves and behaves correctly
without duplicating the matrix module's own coverage.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestMatrixCompatFacade:
    """Test the facade re-exports resolve to the real implementations."""

    def test_parse_matrix_data_is_callable(self) -> None:
        from gnn.visualization.matrix_compat import parse_matrix_data

        assert callable(parse_matrix_data)

    def test_parse_matrix_data_parses_string(self) -> None:
        import numpy as np

        from gnn.visualization.matrix_compat import parse_matrix_data

        # Real string-matrix parse forwards to numpy.
        result = parse_matrix_data("[[1 2]\n [3 4]]")
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.size == 4

    def test_generate_matrix_visualizations_is_callable(self) -> None:
        from gnn.visualization.matrix_compat import generate_matrix_visualizations

        assert callable(generate_matrix_visualizations)

    def test_all_matches_public_surface(self) -> None:
        import gnn.visualization.matrix_compat as facade

        assert facade.__all__ == [
            "parse_matrix_data",
            "generate_matrix_visualizations",
        ]
        for name in facade.__all__:
            assert hasattr(facade, name)
            assert callable(getattr(facade, name))

    def test_forwarding_identity(self) -> None:
        from gnn.visualization.matrix import compat as real
        from gnn.visualization.matrix_compat import (
            generate_matrix_visualizations,
            parse_matrix_data,
        )

        # The facade must forward the exact same callables.
        assert parse_matrix_data is real.parse_matrix_data
        assert generate_matrix_visualizations is real.generate_matrix_visualizations


def test_analysis_matrix_parser_keeps_shared_public_surface() -> None:
    from gnn.analysis.analyzer import parse_matrix_data as analyzer_parse
    from gnn.analysis.simulation_visualizations import (
        parse_matrix_data as analysis_parse,
    )
    from gnn.visualization.matrix.compat import parse_matrix_data

    assert analysis_parse is parse_matrix_data
    assert analyzer_parse is parse_matrix_data
    assert parse_matrix_data("no numeric data") is None
    assert parse_matrix_data("[[1 2] [3 4]]").tolist() == [1.0, 2.0, 3.0, 4.0]


@pytest.mark.parametrize(
    "module_name",
    [
        "gnn.visualization.matrix_compat",
        "gnn.analysis.simulation_visualizations",
    ],
)
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("-3", [-3.0]),
        ("[[-3, +2], [.5, -.25]]", [-3.0, 2.0, 0.5, -0.25]),
        ("[-1e-3, +2E2, 3.e-2, -.5e+1]", [-0.001, 200.0, 0.03, -5.0]),
        ("[1, 1e309]", None),
        ("[-1E999, 2]", None),
        ("[NaN, 1]", None),
        ("[-Infinity, 2]", None),
        ("[+inf, -3]", None),
        ("", None),
        ("[]", None),
        ("no numeric data", None),
        (None, None),
        ([[1, 2]], None),
    ],
)
def test_shared_advisory_parser_preserves_numeric_values_or_refuses(
    module_name: str,
    value: object,
    expected: list[float] | None,
) -> None:
    from importlib import import_module

    import numpy as np

    parse = import_module(module_name).parse_matrix_data
    result = parse(value)
    if expected is None:
        assert result is None
    else:
        assert result.ndim == 1
        np.testing.assert_array_equal(result, expected)

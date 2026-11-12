"""Outcome judgement for the MCP ``validate_gnn_files`` surface.

``GNNTypeChecker.validate_gnn_files`` returns the pipeline exit sentinel:
``True`` (exit 0), ``2`` (exit 2, SUCCESS_WITH_WARNINGS), ``False`` (exit 1).
Only literal ``True`` is a pass. ``False == 0`` is true in Python, so any
comparison against zero turns a hard failure into a reported success.
"""

from pathlib import Path

import pytest

from gnn.type_checker.mcp import validate_gnn_files_mcp

# B orientation contradictions are hard failures under strict mode; this
# fixture is the strict-mode refusal path the type checker reports as False.
_STRICT_HARD_FAILURE_GNN = """\
## GNNSection
ActInfPOMDP

## ModelName
BOrientation

## StateSpaceBlock
B[2,2,2]  # B[next_state, prev_state, action]
s[2,1,type=float]

## InitialParameterization
# rows are previous states, columns are next states
B={
  ( (0.95, 0.05), (0.05, 0.95) ),
  ( (0.05, 0.95), (0.95, 0.05) )
}

## Connections
s-s

## Footer
BOrientation
"""


@pytest.mark.parametrize(
    ("returned", "expected_success", "expected_exit_code"),
    [
        (True, True, 0),
        (2, False, 2),
        (False, False, 1),
        (None, False, 1),
        (0, False, 1),
    ],
)
def test_success_requires_literal_true(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returned: object,
    expected_success: bool,
    expected_exit_code: int,
) -> None:
    """Every non-``True`` sentinel must be reported as a refusal."""

    def _returned(*_args: object, **_kwargs: object) -> object:
        return returned

    monkeypatch.setattr("gnn.type_checker.GNNTypeChecker.validate_gnn_files", _returned)
    result = validate_gnn_files_mcp(str(tmp_path), str(tmp_path / "out"))

    assert result["success"] is expected_success
    if expected_success:
        assert "completed successfully" in result["message"]
    else:
        assert "completed successfully" not in result["message"]
    assert result["exit_code"] == expected_exit_code


def test_invalid_file_is_not_reported_as_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real exit-2 run (invalid GNN document) must not claim success."""
    monkeypatch.setattr(
        "gnn.type_checker.visualizer.generate_all_visualizations",
        lambda *_args, **_kwargs: [],
    )
    (tmp_path / "invalid.gnn").write_text("", encoding="utf-8")

    result = validate_gnn_files_mcp(str(tmp_path), str(tmp_path / "out"))

    assert result["success"] is False
    assert "completed successfully" not in result["message"]
    assert result["exit_code"] == 2


def test_strict_hard_failure_is_not_reported_as_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real hard failure (``False``) must not claim success via ``False == 0``."""
    monkeypatch.setattr(
        "gnn.type_checker.visualizer.generate_all_visualizations",
        lambda *_args, **_kwargs: [],
    )
    (tmp_path / "b.gnn").write_text(_STRICT_HARD_FAILURE_GNN, encoding="utf-8")

    result = validate_gnn_files_mcp(str(tmp_path), str(tmp_path / "out"), strict=True)

    assert result["success"] is False
    assert "completed successfully" not in result["message"]
    assert result["exit_code"] == 1

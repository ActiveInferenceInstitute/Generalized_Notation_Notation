"""B-tensor orientation diagnostic and --transpose-b contract tests.

Pins the Step 6 orientation stage (``gnn.validation.orientation``): the
canonical contract (``B[next_state, previous_state, action]`` with
column-stochastic per-action slices), the textbook row-stochastic warning
with state-factor and slice naming, the ambiguous and non-stochastic
deferrals, the opt-in transposition round-trip on a hand-computable 2x2
example, the ``process_validation`` kwarg wiring, and the zero-warnings
guarantee over the gold exemplar corpus.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import gnn.validation as validation
from gnn.validation import (
    StageServices,
    check_b_orientation,
    process_validation,
    validate_directory,
)
from gnn.validation.orientation import scan_b_orientation, transpose_b_to_canonical

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "input" / "gnn_files"

_DECL = (
    "B[2,2,2,type=float]  # Transition matrix: B[next_state, previous_state, actions]"
)

# Textbook layout, action-outer: values[a][p][n], every row sums to 1.
_TEXTBOOK_3D_LITERAL = """B={
  ( (0.9, 0.1), (0.2, 0.8) ),
  ( (0.3, 0.7), (0.6, 0.4) )
}"""

# Canonical layout, next-outer: values[n][p][a]; per-action slices
# (values[:,:,a]) are column-stochastic over next states.
_CANONICAL_3D_LITERAL = """B={
  ( (0.9, 0.3), (0.2, 0.6) ),
  ( (0.1, 0.7), (0.8, 0.4) )
}"""


def _model_md(b_literal: str, with_state_var: bool = True) -> str:
    """Wrap a B literal in a minimal well-formed GNN model."""
    state = "s[2,1,type=float]    # Hidden state\n" if with_state_var else ""
    return (
        "# GNN Example: Orientation Fixture\n\n"
        "# GNN Version: 1.0\n\n"
        "## ModelName\n\n"
        "Orientation Fixture\n\n"
        "## StateSpaceBlock\n\n"
        f"{_DECL}\n"
        f"{state}"
        "\n## Connections\n\n"
        "s-B\n\n"
        "## InitialParameterization\n\n"
        f"{b_literal}\n"
    )


def _write_manifest(base_output: Path, parsed_file: Path) -> None:
    """Write a minimal step-3 manifest pointing at one parsed model."""
    gnn_output = base_output / "3_gnn_output"
    gnn_output.mkdir(parents=True, exist_ok=True)
    (gnn_output / "gnn_processing_results.json").write_text(
        json.dumps(
            {
                "processed_files": [
                    {
                        "file_name": parsed_file.name,
                        "file_path": str(parsed_file),
                        "parse_success": True,
                        "parsed_model_file": str(parsed_file),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _capture_orientation_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    """Record the kwargs process_validation forwards to the orientation stage."""
    real_stage = validation.check_b_orientation
    captured: list[dict[str, Any]] = []

    def recorder(
        model_data: str | Path | dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        captured.append(dict(kwargs))
        return real_stage(model_data, **kwargs)

    monkeypatch.setattr(validation, "check_b_orientation", recorder)
    return captured


class TestCanonicalSilent:
    """Canonical orientations must pass with no warnings and no notes."""

    def test_action_inner_column_stochastic_is_canonical(self) -> None:
        result = scan_b_orientation(_model_md(_CANONICAL_3D_LITERAL))
        tensor = result["tensors"][0]
        assert result["warnings"] == []
        assert result["notes"] == []
        assert tensor["orientation"] == "canonical"
        assert tensor["action_axis"] == "inner"
        assert result["valid"] is True
        assert result["orientation_score"] == 1.0

    def test_action_outer_column_stochastic_is_canonical(self) -> None:
        # Corpus pattern: (action, next, prev) storage, slices column-stochastic.
        literal = "B={\n  ( (0.9, 0.2), (0.1, 0.8) ),\n  ( (0.3, 0.6), (0.7, 0.4) )\n}"
        result = scan_b_orientation(_model_md(literal))
        tensor = result["tensors"][0]
        assert result["warnings"] == []
        assert tensor["orientation"] == "canonical"

    def test_2d_column_stochastic_is_canonical(self) -> None:
        result = scan_b_orientation(_model_md("B={\n  (0.9, 0.2),\n  (0.1, 0.8)\n}"))
        assert result["warnings"] == []
        assert result["tensors"][0]["orientation"] == "canonical"

    def test_no_b_literal_is_silent(self) -> None:
        result = scan_b_orientation("# no B here\n## ModelName\n\nEmpty\n")
        assert result["warnings"] == []
        assert result["notes"] == []
        assert result["tensors"] == []


class TestTextbookWarning:
    """Row-stochastic (textbook) tensors warn with actionable naming."""

    def test_warning_names_factor_slices_and_fix(self) -> None:
        result = scan_b_orientation(_model_md(_TEXTBOOK_3D_LITERAL))
        tensor = result["tensors"][0]
        assert tensor["orientation"] == "row_stochastic"
        assert tensor["action_axis"] == "outer"
        assert tensor["flipped_slices"] == [0, 1]
        assert len(result["warnings"]) == 1
        warning = result["warnings"][0]
        assert warning.startswith("[B-orientation]")
        assert "B[2,2,2]" in warning
        assert "state factor s (2 states)" in warning
        assert "action slices [0, 1]" in warning
        assert "row-stochastic" in warning
        assert "previous state s_t" in warning
        assert "--transpose-b" in warning
        assert "gnn_syntax.md" in warning

    def test_warning_without_state_variable_omits_name(self) -> None:
        result = scan_b_orientation(
            _model_md(_TEXTBOOK_3D_LITERAL, with_state_var=False)
        )
        warning = result["warnings"][0]
        assert "state factor (2 states)" in warning
        assert "state factor s" not in warning

    def test_2d_textbook_warns_without_action_axis(self) -> None:
        result = scan_b_orientation(_model_md("B={\n  (0.8, 0.2),\n  (0.6, 0.4)\n}"))
        tensor = result["tensors"][0]
        assert tensor["orientation"] == "row_stochastic"
        assert tensor["action_axis"] is None
        assert tensor["flipped_slices"] == [0]
        assert len(result["warnings"]) == 1

    def test_warning_does_not_invalidate(self) -> None:
        result = scan_b_orientation(_model_md(_TEXTBOOK_3D_LITERAL))
        assert result["valid"] is True
        assert result["orientation_score"] == pytest.approx(0.95)


class TestAmbiguousAndNonStochastic:
    """Doubly-stochastic tensors note only; non-stochastic stay silent."""

    def test_doubly_stochastic_notes_only(self) -> None:
        result = scan_b_orientation(_model_md("B={\n  (0.5, 0.5),\n  (0.5, 0.5)\n}"))
        tensor = result["tensors"][0]
        assert tensor["orientation"] == "ambiguous"
        assert result["warnings"] == []
        assert len(result["notes"]) == 1
        assert "doubly stochastic" in result["notes"][0]

    def test_non_stochastic_is_silent(self) -> None:
        result = scan_b_orientation(_model_md("B={\n  (0.5, 0.6),\n  (0.5, 0.5)\n}"))
        tensor = result["tensors"][0]
        assert tensor["orientation"] == "non_stochastic"
        assert result["warnings"] == []
        assert result["notes"] == []


class TestTransposeOption:
    """The opt-in transposition proves the canonical fix per tensor."""

    def test_hand_computed_transform_action_outer(self) -> None:
        old = [[[0.9, 0.1], [0.2, 0.8]], [[0.3, 0.7], [0.6, 0.4]]]
        transposed = transpose_b_to_canonical(old, "outer")
        # canonical[n][p][a] = old[a][p][n] (canonicalize_pomdp mapping)
        expected = [
            # next=0
            [[0.9, 0.3], [0.2, 0.6]],  # [n=0][p][a]
            # next=1
            [[0.1, 0.7], [0.8, 0.4]],
        ]
        assert transposed == expected
        # Hand-checked transition probabilities:
        # P(next=0 | prev=0, a=0) = 0.9, P(next=1 | prev=0, a=0) = 0.1
        assert transposed[0][0][0] == 0.9
        assert transposed[1][0][0] == 0.1
        assert transposed[0][1][1] == 0.6

    def test_hand_computed_transform_2d(self) -> None:
        old = [[0.8, 0.2], [0.6, 0.4]]
        transposed = transpose_b_to_canonical(old, None)
        assert transposed == [[0.8, 0.6], [0.2, 0.4]]

    def test_option_records_transposition_and_reverifies(self) -> None:
        result = scan_b_orientation(_model_md(_TEXTBOOK_3D_LITERAL), transpose_b=True)
        tensor = result["tensors"][0]
        assert tensor["transposed"] is True
        assert tensor["previous_orientation"] == "row_stochastic"
        assert tensor["canonical_after_transpose"] is True
        notes = "\n".join(result["notes"])
        assert "transposed to canonical order" in notes
        assert "previous orientation was row_stochastic" in notes
        assert "row-stochastic" in result["warnings"][0]

    def test_option_transposes_2d_textbook(self) -> None:
        result = scan_b_orientation(
            _model_md("B={\n  (0.8, 0.2),\n  (0.6, 0.4)\n}"), transpose_b=True
        )
        tensor = result["tensors"][0]
        assert tensor["transposed"] is True
        assert tensor["canonical_after_transpose"] is True

    def test_option_never_transposes_canonical_tensors(self) -> None:
        result = scan_b_orientation(_model_md(_CANONICAL_3D_LITERAL), transpose_b=True)
        tensor = result["tensors"][0]
        assert tensor["transposed"] is False
        assert result["warnings"] == []

    def test_option_does_not_modify_source_content(self) -> None:
        content = _model_md(_TEXTBOOK_3D_LITERAL)
        before = scan_b_orientation(content)
        scan_b_orientation(content, transpose_b=True)
        after = scan_b_orientation(content)
        assert before == after


class TestStageContract:
    """check_b_orientation mirrors the sibling stage input/output contract."""

    def test_accepts_path(self, tmp_path: Path) -> None:
        path = tmp_path / "model.gnn"
        path.write_text(_model_md(_TEXTBOOK_3D_LITERAL), encoding="utf-8")
        result = check_b_orientation(path)
        assert result["file_path"] == str(path)
        assert result["file_name"] == "model.gnn"
        assert len(result["warnings"]) == 1
        assert result["valid"] is True
        assert result["recovery"] is False

    def test_accepts_parsed_model_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "model.json"
        path.write_text(
            json.dumps(
                {
                    "file_path": str(path),
                    "raw_sections": {
                        "StateSpaceBlock": _DECL,
                        "InitialParameterization": _TEXTBOOK_3D_LITERAL,
                    },
                }
            ),
            encoding="utf-8",
        )
        result = check_b_orientation(json.loads(path.read_text(encoding="utf-8")))
        assert len(result["warnings"]) == 1
        assert result["tensors"][0]["orientation"] == "row_stochastic"

    def test_bad_input_returns_error_receipt(self) -> None:
        result = check_b_orientation(123)  # type: ignore[arg-type]
        assert result["status"] == "error"
        assert result["valid"] is False
        assert result["recovery"] is True


class TestProcessValidationWiring:
    """The orchestrator forwards transpose_b to the orientation stage."""

    def test_transpose_b_kwarg_is_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _capture_orientation_kwargs(monkeypatch)
        parsed = tmp_path / "model.json"
        parsed.write_text(
            json.dumps(
                {
                    "file_path": str(parsed),
                    "raw_sections": {
                        "StateSpaceBlock": _DECL,
                        "InitialParameterization": _TEXTBOOK_3D_LITERAL,
                    },
                }
            ),
            encoding="utf-8",
        )
        _write_manifest(tmp_path / "run", parsed)
        output = tmp_path / "run" / "6_validation_output"

        success = process_validation(tmp_path / "models", output, transpose_b=True)

        assert success is True
        assert captured == [{"transpose_b": True}]
        receipt = json.loads(
            (output / "validation_results.json").read_text(encoding="utf-8")
        )
        assert receipt["context"]["transpose_b"] is True
        orientation = receipt["files_validated"][0]["validations"]["orientation"]
        assert orientation["tensors"][0]["transposed"] is True

    def test_default_run_has_no_transpose_kwarg(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _capture_orientation_kwargs(monkeypatch)
        parsed = tmp_path / "model.json"
        parsed.write_text(
            json.dumps(
                {
                    "file_path": str(parsed),
                    "raw_sections": {
                        "StateSpaceBlock": _DECL,
                        "InitialParameterization": _CANONICAL_3D_LITERAL,
                    },
                }
            ),
            encoding="utf-8",
        )
        _write_manifest(tmp_path / "run", parsed)
        output = tmp_path / "run" / "6_validation_output"

        assert process_validation(tmp_path / "models", output) is True
        assert captured == [{"transpose_b": False}]

    def test_validate_directory_skips_orientation_when_not_injected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Alternative pipelines without the orientation stage run three stages."""
        parsed = tmp_path / "model.json"
        parsed.write_text(
            json.dumps(
                {
                    "file_path": str(parsed),
                    "raw_sections": {
                        "StateSpaceBlock": _DECL,
                        "InitialParameterization": _TEXTBOOK_3D_LITERAL,
                    },
                }
            ),
            encoding="utf-8",
        )
        _write_manifest(tmp_path / "run", parsed)
        output = tmp_path / "run" / "6_validation_output"

        def _always_valid(
            model_data: str | Path | dict[str, Any], **kwargs: Any
        ) -> dict[str, Any]:
            return {"valid": True, "score": 1.0}

        success = validate_directory(
            tmp_path / "models",
            output,
            services=StageServices(
                semantic=_always_valid,
                performance=_always_valid,
                consistency=_always_valid,
            ),
        )

        assert success is True
        receipt = json.loads(
            (output / "validation_results.json").read_text(encoding="utf-8")
        )
        file_result = receipt["files_validated"][0]
        assert "orientation" not in file_result["validations"]
        assert file_result["success"] is True

    def test_process_validation_mcp_signature_exposes_transpose_b(self) -> None:
        import inspect

        from gnn.validation.mcp import process_validation_mcp

        assert "transpose_b" in inspect.signature(process_validation_mcp).parameters


class TestGoldCorpusRegression:
    """The diagnostic must never warn on the canonical exemplar corpus."""

    def test_zero_warnings_across_corpus(self) -> None:
        assert CORPUS_DIR.is_dir(), "gold corpus missing"
        warnings: list[str] = []
        for path in sorted(CORPUS_DIR.rglob("*.md")):
            result = scan_b_orientation(path.read_text(encoding="utf-8"))
            warnings.extend(
                f"{path.relative_to(CORPUS_DIR)}: {message}"
                for message in result["warnings"]
            )
        assert warnings == []

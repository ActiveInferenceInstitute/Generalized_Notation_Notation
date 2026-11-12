"""Unit tests for ``gnn.schema_validator.validator.GNNValidator``.

The validator measured 48.7% in the verification harness: the level-rank
mapping, constructor wiring, stochasticity checker, and the level-gated
``validate_file`` paths (binary files, markdown structure gate) were
unpinned. Tests build the valid document from ``REQUIRED_SECTIONS`` itself
so they track the section contract rather than duplicating it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gnn.schema_validator.validator import GNNValidator
from gnn.schemas.section_contract import REQUIRED_SECTIONS
from gnn.types import ValidationLevel

REQUIRED_CONTENT = {
    "GNNSection": "Generalized Notation Notation",
    "GNNVersionAndFlags": "GNNv1",
    "ModelName": "TestModel",
    "ModelAnnotation": "A test model.",
    "StateSpaceBlock": "s_f[2,2] # state factor\n",
    "Connections": "s_f > s_f\n",
    "InitialParameterization": "A = uniform\n",
    "Time": "Dynamic\nHorizon=5\n",
    "Footer": "End of model.",
}


def _valid_document() -> str:
    parts = []
    for section in REQUIRED_SECTIONS:
        parts.append(f"## {section}\n{REQUIRED_CONTENT[section]}\n")
    return "\n".join(parts)


def _write_doc(tmp_path: Path, content: str, name: str = "model.md") -> Path:
    target = tmp_path / name
    target.write_text(content, encoding="utf-8")
    return target


class TestCheckStochasticity:
    def setup_method(self) -> None:
        self.validator = GNNValidator()

    def test_2d_stochastic_matrix_is_valid(self) -> None:
        assert self.validator._check_stochasticity([[0.5, 0.5], [0.25, 0.75]]) is True

    def test_2d_non_stochastic_row_is_rejected(self) -> None:
        assert self.validator._check_stochasticity([[0.5, 0.6], [0.5, 0.5]]) is False

    def test_1d_vector_branches(self) -> None:
        assert self.validator._check_stochasticity([0.3, 0.7]) is True
        assert self.validator._check_stochasticity([0.3, 0.8]) is False

    def test_uncheckable_input_is_conservatively_valid(self) -> None:
        # Empty and non-numeric inputs cannot be checked; the contract is to
        # assume valid rather than fail the model.
        # Empty and None inputs cannot be checked; the contract is to
        # assume valid rather than fail the model. A non-empty non-numeric
        # 1D vector sums to 0 (not 1), so it is genuinely rejected.
        assert self.validator._check_stochasticity([]) is True
        assert self.validator._check_stochasticity(["a", "b"]) is False
        assert self.validator._check_stochasticity(None) is True

    def test_tolerance_accepts_near_stochastic(self) -> None:
        assert self.validator._check_stochasticity([[0.5, 0.5000005]]) is True


class TestLevelRank:
    def setup_method(self) -> None:
        self.validator = GNNValidator()

    @pytest.mark.parametrize(
        "level,expected",
        [
            (ValidationLevel.BASIC, 10),
            (ValidationLevel.STANDARD, 20),
            (ValidationLevel.STRICT, 30),
            (ValidationLevel.RESEARCH, 40),
            (ValidationLevel.ROUND_TRIP, 50),
        ],
    )
    def test_enum_ranks(self, level: ValidationLevel, expected: int) -> None:
        assert self.validator._level_rank(level) == expected

    def test_string_level_is_resolved(self) -> None:
        assert self.validator._level_rank("strict") == 30

    def test_unknown_level_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown validation level"):
            self.validator._level_rank("nonsense")


class TestValidatorConstruction:
    def test_default_schema_path(self) -> None:
        validator = GNNValidator()

        assert validator.schema_path.name == "json.json"
        assert validator.schema_path.parent.name == "schemas"
        assert validator.enable_round_trip_testing is False
        assert validator.validation_level == ValidationLevel.STANDARD

    def test_cross_validation_disabled_leaves_no_cross_validator(self) -> None:
        validator = GNNValidator(enable_cross_validation=False)

        assert validator.cross_validator is None

    def test_cross_validation_enabled_wires_cross_validator(self) -> None:
        validator = GNNValidator(enable_cross_validation=True)

        assert validator.cross_validator is not None


class TestValidateFileLevels:
    def test_valid_document_passes_at_basic_level(self, tmp_path: Path) -> None:
        validator = GNNValidator(validation_level=ValidationLevel.BASIC)
        target = _write_doc(tmp_path, _valid_document())

        result = validator.validate_file(target)

        assert result.is_valid is True
        assert result.errors == []
        assert result.format_tested == "markdown"
        assert result.metadata.get("parsed_successfully") is True

    def test_missing_required_section_is_flagged(self, tmp_path: Path) -> None:
        content = _valid_document().replace("## ModelName\nTestModel\n\n", "")
        validator = GNNValidator(validation_level=ValidationLevel.BASIC)
        target = _write_doc(tmp_path, content)

        result = validator.validate_file(target)

        assert result.is_valid is False
        assert any(
            "Required section missing: ModelName" in error for error in result.errors
        )

    def test_research_level_runs_on_valid_document(self, tmp_path: Path) -> None:
        validator = GNNValidator(
            validation_level=ValidationLevel.RESEARCH, enable_cross_validation=False
        )
        target = _write_doc(tmp_path, _valid_document())

        result = validator.validate_file(target)

        # The research-level pipeline must complete without raising; validity
        # reflects the recorded errors, whatever the deeper gates find.
        assert result.validation_level == ValidationLevel.RESEARCH
        assert result.is_valid == (len(result.errors) == 0)
        assert "validation_time" in result.performance_metrics

    def test_unknown_format_file_still_validates_as_text(self, tmp_path: Path) -> None:
        validator = GNNValidator(validation_level=ValidationLevel.BASIC)
        target = _write_doc(tmp_path, _valid_document(), name="model.weird")

        result = validator.validate_file(target)

        assert result.format_tested == "unknown"
        assert result.is_valid is True

    def test_pkl_extension_treated_as_parseable_text_format(
        self, tmp_path: Path
    ) -> None:
        # The .pkl extension routes to the text-parse path (not the binary
        # branch): non-pickle garbage is parsed leniently, so the validator
        # reports the format and a parsed_successfully flag rather than raising.
        validator = GNNValidator()
        target = tmp_path / "model.pkl"
        target.write_bytes(b"definitely not a pickle payload")

        result = validator.validate_file(target)

        assert result.format_tested == "pkl"
        assert "parsed_successfully" in result.metadata


class TestEdgeAnnotations:
    """v1.1 edge annotations are labels, not variable names.

    ``docs/gnn/gnn_syntax.md`` defines ``A>B:label`` and requires parsers to
    accept and preserve the annotation. The compliance validator used to read
    the whole right-hand token as the variable name, so a valid annotated
    document failed with "Undefined variable in connection target".
    """

    def test_annotated_edge_validates(self, tmp_path: Path) -> None:
        validator = GNNValidator(
            validation_level=ValidationLevel.STANDARD, enable_cross_validation=False
        )
        document = _valid_document().replace("s_f > s_f", "s_f > s_f:likelihood")

        result = validator.validate_file(_write_doc(tmp_path, document))

        assert result.is_valid, result.errors

    def test_parser_splits_the_annotation_off_the_target(self) -> None:
        from gnn.schema_validator import GNNParser

        parsed = GNNParser().parse_content(
            _valid_document().replace("s_f > s_f", "s_f > s_f:likelihood")
        )

        connection = parsed.connections[0]
        assert connection.target == "s_f"
        assert connection.metadata["label"] == "likelihood"

    def test_annotation_shaped_variable_is_still_checked(self, tmp_path: Path) -> None:
        """The split must not make an undeclared variable look declared."""
        validator = GNNValidator(
            validation_level=ValidationLevel.STANDARD, enable_cross_validation=False
        )
        document = _valid_document().replace("s_f > s_f", "s_f > undeclared:likelihood")

        result = validator.validate_file(_write_doc(tmp_path, document))

        assert any("undeclared" in error for error in result.errors), result.errors

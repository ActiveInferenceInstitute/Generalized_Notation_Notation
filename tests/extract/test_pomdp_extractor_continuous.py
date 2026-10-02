#!/usr/bin/env python3
"""
Unit tests for the POMDP extractor's per-factor continuous branch.

Covers ``_is_continuous_model`` classification, per-factor vs plain routing
in ``_extract_continuous_dimensions`` (with honest provenance labels), and
per-factor key collection + scalar coercion in
``_collect_continuous_parameters``. Stdlib-only plain dict/list fixtures:
numpy-free, zero skips.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.extract.pomdp_extractor import POMDPExtractor


def _factor_block() -> dict:
    """A minimal two-factor per-factor LGSSM block (factor 1 sized)."""
    return {
        "F_f1": [[1.0, 0.1], [0.0, 1.0]],
        "H_f1": [[1.0, 0.0], [0.0, 1.0]],
        "Q_f1": [[0.01, 0.0], [0.0, 0.01]],
        "R_f1": [[0.1, 0.0], [0.0, 0.1]],
        "prior_mean_f1": [0.0, 0.0],
        "prior_cov_f1": [[1.0, 0.0], [0.0, 1.0]],
        "F_f2": [[0.9]],
        "H_f2": [[1.0]],
        "Q_f2": [[0.01]],
        "R_f2": [[0.1]],
        "prior_mean_f2": [0.0],
        "prior_cov_f2": [[1.0]],
    }


class TestIsContinuousModel:
    """Per-factor classification in ``_is_continuous_model``."""

    def test_per_factor_keys_only_classifies_continuous(self) -> None:
        extractor = POMDPExtractor()
        assert extractor._is_continuous_model(None, _factor_block()) is True

    def test_section_with_plain_block_regression(self) -> None:
        extractor = POMDPExtractor()
        params = {"F": [[1.0]], "H": [[1.0]], "Q": [[0.1]], "R": [[0.1]]}
        assert extractor._is_continuous_model("## ContinuousStateSpace", params) is True

    def test_discrete_only_is_not_continuous(self) -> None:
        extractor = POMDPExtractor()
        params = {"A": [[0.7, 0.3]], "B": [[[0.8, 0.2], [0.4, 0.6]]]}
        assert extractor._is_continuous_model(None, params) is False


class TestExtractContinuousDimensions:
    """Per-factor vs plain routing in ``_extract_continuous_dimensions``."""

    def test_per_factor_block_dims_and_provenance(self) -> None:
        extractor = POMDPExtractor()
        dims = extractor._extract_continuous_dimensions(
            {}, _factor_block(), {"num_timesteps": 15, "num_factors": 2}
        )
        assert dims == (2, 2, 0, 15)
        assert extractor._dimension_sources == {
            "num_states": "per_factor_block",
            "num_observations": "per_factor_block",
            "num_actions": "default",
            "num_timesteps": "ModelParameters",
        }

    def test_no_continuous_block_raises(self) -> None:
        extractor = POMDPExtractor()
        with pytest.raises(ValueError, match="missing linear-Gaussian"):
            extractor._extract_continuous_dimensions({}, {"B": [[[1.0]]]}, {})

    def test_plain_only_regression(self) -> None:
        extractor = POMDPExtractor()
        params = {
            "F": [[1.0, 0.1], [0.0, 1.0]],
            "H": [[1.0, 0.0]],
            "Q": [[0.01, 0.0], [0.0, 0.01]],
            "R": [[0.1]],
            "prior_mean": [0.0, 0.0],
            "prior_cov": [[1.0, 0.0], [0.0, 1.0]],
        }
        dims = extractor._extract_continuous_dimensions({}, params, {})
        assert dims == (2, 1, 0, None)
        assert extractor._dimension_sources["num_states"] == "variable_dimensions"

    def test_mixed_plain_and_per_factor_plain_wins(self) -> None:
        extractor = POMDPExtractor()
        params = dict(_factor_block())
        params.update(
            {
                "F": [[1.0, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "H": [[1.0, 0.0, 0.0]],
                "Q": [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]],
                "R": [[0.1]],
                "prior_mean": [0.0, 0.0, 0.0],
                "prior_cov": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            }
        )
        dims = extractor._extract_continuous_dimensions({}, params, {})
        assert dims == (3, 1, 0, None)
        assert extractor._dimension_sources["num_states"] == "variable_dimensions"


class TestCollectContinuousParameters:
    """Per-factor collection and scalar coercion."""

    def test_collects_factor_keys_and_coerces_gain(self) -> None:
        extractor = POMDPExtractor()
        params = dict(_factor_block())
        params["goal_mean_f1"] = [1.0, 2.0]
        params["control_gain_f1"] = [0.5]
        out = extractor._collect_continuous_parameters(params)
        assert out["F_f1"] == [[1.0, 0.1], [0.0, 1.0]]
        assert out["prior_cov_f1"] == [[1.0, 0.0], [0.0, 1.0]]
        assert out["goal_mean_f1"] == [1.0, 2.0]
        assert out["control_gain_f1"] == 0.5
        assert isinstance(out["control_gain_f1"], float)

    def test_plain_only_regression_unchanged(self) -> None:
        extractor = POMDPExtractor()
        params = {
            "F": [[1.0]],
            "H": [[1.0]],
            "Q": [[0.1]],
            "R": [[0.1]],
            "prior_mean": [0.0],
            "prior_cov": [[1.0]],
            "control_gain": [0.25],
        }
        out = extractor._collect_continuous_parameters(params)
        assert out["control_gain"] == 0.25
        assert out["F"] == [[1.0]]
        assert "control_gain_f1" not in out

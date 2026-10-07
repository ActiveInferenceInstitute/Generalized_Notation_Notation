"""Pins for ``type_checker.resource_estimator`` (previously 27%)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import gnn.type_checker.resource_estimator as resource_estimator
from gnn.type_checker.estimation.estimator import GNNResourceEstimator
from gnn.type_checker.estimation.report_html import _generate_visualizations_for_html
from tests.helpers.bar_labels import (
    assert_bar_labels_offset_in_points,
    assert_png_bounded,
    figures_held_open,
    open_bar_figures,
)

_SPEC = """## GNNSection
ActInfPOMDP

## ModelName
EstimatorProbe

## StateSpaceBlock
s[3,1,type=float]
o[3,1,type=int]

## Connections
s-s
s-o

## Footer
EstimatorProbe
"""


def test_facade_exposes_the_canonical_estimator() -> None:
    assert resource_estimator.__all__ == ["GNNResourceEstimator"]
    assert resource_estimator.GNNResourceEstimator is GNNResourceEstimator


def test_main_estimates_single_file_and_prints_report(
    tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = tmp_path / "probe.gnn"
    spec.write_text(_SPEC, encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["resource_estimator", str(spec), "-o", str(tmp_path)]
    )

    exit_code = resource_estimator.main()

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "EstimatorProbe" in captured.out or "probe" in captured.out


def test_main_estimates_directory_recursive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "specs"
    target.mkdir()
    (target / "probe.gnn").write_text(_SPEC, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["resource_estimator", str(target), "--recursive", "-o", str(tmp_path)],
    )

    exit_code = resource_estimator.main()

    assert exit_code == 0


# Degenerate: near-zero estimates, where the old +0.01 data-unit label offset
# was ~100 axes-heights above the bars; large: estimates in the thousands.
_ESTIMATE_SCALES = {"degenerate": 1e-4, "large": 5e3}


@pytest.mark.parametrize("case", sorted(_ESTIMATE_SCALES))
def test_html_estimate_bar_labels_are_offset_in_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    scale = _ESTIMATE_SCALES[case]
    results = {
        f"model_{i}.md": {
            "memory_estimate": scale * (i + 1),
            "inference_estimate": scale * (i + 2),
            "storage_estimate": scale * (i + 3),
        }
        for i in range(3)
    }

    with figures_held_open(monkeypatch):
        _generate_visualizations_for_html(results, tmp_path)
        for name in (
            "memory_usage_html.png",
            "inference_time_html.png",
            "storage_requirements_html.png",
        ):
            assert_png_bounded(tmp_path / name)
        figures = open_bar_figures()
        assert len(figures) == 3
        for fig in figures:
            assert_bar_labels_offset_in_points(fig)

"""Unit tests for the Step-13 cross-framework injection seam.

``_collect_cross_framework_summary`` reads a prior cross-framework
comparison HTML plus the sibling per-framework ``simulation_results.json``
files (light identity fields only, never executing anything) so the LLM
prompt can be extended with per-framework outcome metadata. These tests pin
the found / absent / unreadable contracts and the bounded scan caps.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from gnn.llm.processor import _collect_cross_framework_summary


def _make_comparison(
    output_root: Path,
    model_stem: str,
    framework_payloads: dict[str, dict[str, object]],
    *,
    nested: bool = False,
) -> Path:
    execute_root = output_root.parent / "12_execute_output"
    base = execute_root / model_stem if nested else execute_root
    for framework, payload in framework_payloads.items():
        fw_dir = base / framework
        fw_dir.mkdir(parents=True, exist_ok=True)
        (fw_dir / "simulation_results.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    html = base / f"{model_stem}_comparison.html"
    html.write_text("<html><body>comparison</body></html>", encoding="utf-8")
    return html


def _payload(framework: str, all_valid: bool | None) -> dict[str, object]:
    payload: dict[str, object] = {"framework": framework}
    if all_valid is not None:
        payload["validation"] = {"all_valid": all_valid}
    return payload


class TestCollectCrossFrameworkSummary:
    @pytest.mark.unit
    def test_flat_comparison_is_summarized(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "13_llm_output"
        output_dir.mkdir()
        _make_comparison(
            output_dir,
            "pomdp_model",
            {
                "jax": _payload("jax", True),
                "pymdp": _payload("PyMDP", False),
                "rxinfer": _payload("rxinfer", None),
            },
        )

        summary = _collect_cross_framework_summary(output_dir, "pomdp_model")

        assert summary is not None
        assert summary["model"] == "pomdp_model"
        assert summary["comparison_html"] == (
            "12_execute_output/pomdp_model_comparison.html"
        )
        assert summary["frameworks"] == [
            {"framework": "jax", "status": "success"},
            {"framework": "PyMDP", "status": "validation_failed"},
            {"framework": "rxinfer", "status": "unknown"},
        ]

    @pytest.mark.unit
    def test_nested_comparison_is_summarized(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "13_llm_output"
        output_dir.mkdir()
        _make_comparison(
            output_dir,
            "nested_model",
            {"pymdp": _payload("PyMDP", True)},
            nested=True,
        )

        summary = _collect_cross_framework_summary(output_dir, "nested_model")

        assert summary is not None
        assert summary["comparison_html"] == (
            "12_execute_output/nested_model/nested_model_comparison.html"
        )
        assert summary["frameworks"] == [{"framework": "PyMDP", "status": "success"}]

    @pytest.mark.unit
    def test_absent_artifact_returns_none(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "13_llm_output"
        output_dir.mkdir()

        assert _collect_cross_framework_summary(output_dir, "ghost") is None

    @pytest.mark.unit
    def test_absent_execute_dir_returns_none(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "13_llm_output"
        output_dir.mkdir()

        # No 12_execute_output sibling at all: quiet absence.
        assert _collect_cross_framework_summary(output_dir, "ghost") is None

    @pytest.mark.unit
    def test_unreadable_results_return_empty_dict_with_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        output_dir = tmp_path / "13_llm_output"
        output_dir.mkdir()
        execute_root = output_dir.parent / "12_execute_output"
        (execute_root / "pymdp").mkdir(parents=True)
        (execute_root / "pymdp" / "simulation_results.json").write_text(
            "{not json", encoding="utf-8"
        )
        (execute_root / "model_comparison.html").write_text(
            "<html></html>", encoding="utf-8"
        )

        with caplog.at_level(logging.WARNING):
            summary = _collect_cross_framework_summary(output_dir, "model")

        assert summary == {}
        assert any(
            "Could not read cross-framework results" in record.message
            for record in caplog.records
        )

    @pytest.mark.unit
    def test_framework_results_scan_is_capped(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "13_llm_output"
        output_dir.mkdir()
        payloads = {
            f"framework_{index:02d}": _payload(f"framework_{index:02d}", True)
            for index in range(12)
        }
        _make_comparison(output_dir, "big_model", payloads)

        summary = _collect_cross_framework_summary(output_dir, "big_model")

        assert summary is not None
        assert len(summary["frameworks"]) == 8

    @pytest.mark.unit
    def test_html_candidates_scan_is_bounded(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "13_llm_output"
        output_dir.mkdir()
        execute_root = output_dir.parent / "12_execute_output"
        for index in range(7):
            nested = execute_root / f"lane_{index:02d}"
            nested.mkdir(parents=True)
            (nested / "many_model_comparison.html").write_text(
                "<html></html>", encoding="utf-8"
            )

        summary = _collect_cross_framework_summary(output_dir, "many_model")

        assert summary is not None
        assert summary["comparison_html"] == (
            "12_execute_output/lane_00/many_model_comparison.html"
        )
        assert summary["frameworks"] == []

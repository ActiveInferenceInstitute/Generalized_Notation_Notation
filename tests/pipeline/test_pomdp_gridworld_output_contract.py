"""Tests for the public POMDP GridWorld output contract checker."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = REPO_ROOT / "scripts" / "check_pomdp_gridworld_outputs.py"


def _load_checker() -> Any:
    spec = importlib.util.spec_from_file_location(
        "check_pomdp_gridworld_outputs", CHECKER_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, content: str = "POMDP GridWorld output") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _minimal_valid_output_tree(root: Path) -> None:
    checker = _load_checker()

    _write_json(
        root / "00_pipeline_summary" / "pipeline_execution_summary.json",
        {
            "arguments": {"target_dir": "input/gnn_files/pomdp_gridworld"},
            "steps": [
                {"script_name": f"{idx}_step.py", "status": "success", "exit_code": 0}
                for idx in range(25)
            ],
        },
    )
    _write_json(
        root / "3_gnn_output" / "gnn_processing_summary.json",
        {"total_files": 1, "successful_parses": 1},
    )
    _write_json(
        root
        / "3_gnn_output"
        / "pomdp_gridworld_3x3"
        / "pomdp_gridworld_3x3_parsed.json",
        {"model_name": "POMDP GridWorld 3x3"},
    )

    framework_results = {
        name: {"success": True, "message": "ok"} for name in checker.ALL_RENDER_TARGETS
    }
    _write_json(
        root / "11_render_output" / "render_processing_summary.json",
        {
            "total_files": 1,
            "successful_files": 1,
            "failed_framework_renderings": [],
            "file_results": {
                "input/gnn_files/pomdp_gridworld/pomdp_gridworld_3x3.md": {
                    "framework_results": framework_results
                }
            },
        },
    )
    for framework in checker.ALL_RENDER_TARGETS:
        suffix = (
            ".stan"
            if framework == "stan"
            else ".jl"
            if framework in {"rxinfer", "activeinference_jl"}
            else ".py"
        )
        _write_text(
            root
            / "11_render_output"
            / "pomdp_gridworld_3x3"
            / framework
            / f"pomdp_gridworld_3x3_{framework}{suffix}",
            "rendered",
        )

    _write_json(
        root / "12_execute_output" / "summaries" / "execution_summary.json",
        {"succeeded": 3, "failed": 0, "skipped": 6},
    )
    for framework in checker.STRICT_EXECUTION_TARGETS:
        _write_json(
            root
            / "12_execute_output"
            / "pomdp_gridworld_3x3"
            / framework
            / "simulation_data"
            / "simulation_results.json",
            {
                "success": True,
                "num_timesteps": 15,
                "model_parameters": {"B_shape": [9, 9, 5]},
                "validation": {"all_valid": True},
            },
        )

    _write_json(
        root
        / "16_analysis_output"
        / "cross_framework"
        / "gridworld_analysis_manifest.json",
        {
            "frameworks": list(checker.STRICT_EXECUTION_TARGETS),
            "matrix_provenance_equal": True,
            "outputs": {
                "png": ["cross_framework/cross_framework_comparison.png"],
                "gif": [f"gif_{idx}.gif" for idx in range(7)],
                "statistics": ["statistics.json"],
            },
        },
    )
    _write_text(
        root
        / "16_analysis_output"
        / "cross_framework"
        / "cross_framework_comparison.png",
        "png",
    )
    for idx in range(7):
        _write_text(
            root / "16_analysis_output" / "cross_framework" / f"gif_{idx}.gif", "gif"
        )

    _write_text(root / "23_report_output" / "report.md", "POMDP GridWorld report")
    for page in (
        "index.html",
        "pipeline.html",
        "reports.html",
        "analysis.html",
        "visualization.html",
    ):
        _write_text(root / "20_website_output" / page, "POMDP GridWorld website")


def test_pomdp_gridworld_output_contract_accepts_valid_tree(tmp_path: Path) -> None:
    checker = _load_checker()
    _minimal_valid_output_tree(tmp_path)

    report = checker.validate_output_tree(tmp_path)

    assert report.ok, report.errors
    assert "render outputs" in report.checked
    assert "analysis outputs" in report.checked


def test_pomdp_gridworld_output_contract_rejects_missing_render_target(
    tmp_path: Path,
) -> None:
    checker = _load_checker()
    _minimal_valid_output_tree(tmp_path)
    summary_path = tmp_path / "11_render_output" / "render_processing_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    file_result = next(iter(summary["file_results"].values()))
    file_result["framework_results"].pop("stan")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    report = checker.validate_output_tree(tmp_path)

    assert not report.ok
    assert any("all render targets" in error for error in report.errors)


def _mutate_analysis_manifest(root: Path, mutate: Any) -> None:
    manifest_path = (
        root
        / "16_analysis_output"
        / "cross_framework"
        / "gridworld_analysis_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_pomdp_gridworld_output_contract_accepts_strict_targets_plus_extra(
    tmp_path: Path,
) -> None:
    checker = _load_checker()
    _minimal_valid_output_tree(tmp_path)
    _mutate_analysis_manifest(
        tmp_path,
        lambda manifest: manifest.update(
            frameworks=sorted([*checker.STRICT_EXECUTION_TARGETS, "numpyro"])
        ),
    )

    report = checker.validate_output_tree(tmp_path)

    assert report.ok, report.errors
    assert "analysis outputs" in report.checked


def test_pomdp_gridworld_output_contract_rejects_missing_strict_target(
    tmp_path: Path,
) -> None:
    checker = _load_checker()
    _minimal_valid_output_tree(tmp_path)
    _mutate_analysis_manifest(
        tmp_path,
        lambda manifest: manifest.update(
            frameworks=sorted(
                target
                for target in [*checker.STRICT_EXECUTION_TARGETS, "numpyro"]
                if target != "rxinfer"
            ),
        ),
    )
    report = checker.validate_output_tree(tmp_path)

    assert not report.ok
    assert any("rxinfer" in error for error in report.errors)

"""The pipeline executes the cpomdp scripts it renders."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

from gnn.execute import processor as execute_processor
from gnn.execute.detection import parse_frameworks_parameter
from gnn.execute.processor import execute_single_script, process_execute
from gnn.extract.pomdp_extractor import extract_pomdp_from_file
from gnn.render.pomdp_contract import detect_pomdp_space_model_kinds
from gnn.render.processor import process_render
from gnn.utils.runtime_safety.framework_availability import FRAMEWORK_IMPORT_CHECK

REPO = Path(__file__).resolve().parents[2]
CONTINUOUS_DIR = REPO / "input" / "gnn_files" / "continuous"
# Composed exemplars (factored, hybrid, multi-agent) get an unsupported receipt
# at render, so only the single-kind linear-Gaussian models reach execution.
MODELS = sorted(
    p.stem
    for p in CONTINUOUS_DIR.glob("*.md")
    if len(detect_pomdp_space_model_kinds(extract_pomdp_from_file(p))) == 1
)
LOGGER = logging.getLogger("test_cpomdp_execution")


def test_all_preset_and_explicit_selection_include_cpomdp() -> None:
    assert "cpomdp" in parse_frameworks_parameter("all", LOGGER)
    assert parse_frameworks_parameter("cpomdp", LOGGER) == ["cpomdp"]


def test_missing_extra_is_skipped_with_its_install_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "render" / "model" / "cpomdp" / "model_cpomdp.py"
    script.parent.mkdir(parents=True)
    script.write_text("import cpomdp\n")
    monkeypatch.setattr(
        execute_processor,
        "_is_python_framework_dependency_available",
        lambda *args, **kwargs: False,
    )
    info: dict[str, Any] = {
        "path": script,
        "name": script.name,
        "framework": "cpomdp",
        "executor": sys.executable,
    }
    result = execute_single_script(info, tmp_path / "results", False, LOGGER)
    assert result["skipped"] is True
    assert result["error"] == "Dependency not installed: cpomdp"
    assert FRAMEWORK_IMPORT_CHECK["cpomdp"] == ("cpomdp", "uv sync --extra cpomdp")


@pytest.mark.needs_cpomdp
def test_pipeline_executes_every_rendered_cpomdp_script(tmp_path: Path) -> None:
    render_dir = tmp_path / "11_render_output"
    execute_dir = tmp_path / "12_execute_output"
    assert process_render(CONTINUOUS_DIR, render_dir, frameworks=["cpomdp"]) is True

    process_execute(
        CONTINUOUS_DIR,
        execute_dir,
        frameworks="cpomdp",
        render_output_dir=render_dir,
    )

    summary = json.loads(
        (execute_dir / "summaries" / "execution_summary.json").read_text()
    )
    status = summary["framework_status"]["cpomdp"]
    assert status["status"] == "success"
    assert status["successful"] == len(MODELS)
    for model in MODELS:
        results = execute_dir / model / "cpomdp" / "simulation_data"
        payload = json.loads((results / "simulation_results.json").read_text())
        assert payload["framework"] == "cpomdp"
        assert payload["validation"]["all_valid"] is True

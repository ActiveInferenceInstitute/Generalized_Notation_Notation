"""Public local meta-analysis of byte-preserved historical native results.

Fixtures retain their original numbers, sign conventions, producer metadata
and hashes. These witnesses exercise saved-data consumption, not execution,
backend readiness, inference correctness or a newly measured scaling study.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
from pathlib import Path
from typing import Any

import pytest

from gnn.integration import SweepDataCollector, run_meta_analysis
from gnn.processing import process_gnn_multi_format

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests/fixtures/native_meta_analysis_410"
LOGGER = logging.getLogger(__name__)
pytestmark = pytest.mark.integration


def _hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _saved_tree(tmp_path: Path, summary_mode: str = "aggregate") -> dict[str, Any]:
    root = tmp_path / "historical-saved-data"
    execute = root / "12_execute_output"
    provenance = json.loads((FIXTURES / "provenance.json").read_text())
    paths: dict[str, Path] = {}
    for name, filename in (
        ("actinf_pomdp_agent", "pymdp_signed_simulation_results.json"),
        ("tmaze_epistemic", "static_perception_simulation_results.json"),
    ):
        path = execute / name / "pymdp/simulation_data/simulation_results.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(FIXTURES / filename, path)
        paths[name] = path
    summary = execute / "summaries/execution_summary.json"
    summary.parent.mkdir()
    actual = provenance["static_producer"]["execution_summary"]
    if summary_mode == "aggregate":
        summary.write_text(json.dumps(actual), encoding="utf-8")
    elif summary_mode == "unreadable":
        summary.mkdir()
    elif summary_mode == "invalid":
        summary.write_text("{ invalid saved timing summary", encoding="utf-8")
    elif summary_mode != "absent":
        raise ValueError(summary_mode)
    log = execute / "tmaze_epistemic/pymdp/execution_logs/native_results.json"
    log.parent.mkdir()
    log.write_text(json.dumps(actual["execution_details"][0]), encoding="utf-8")
    render = root / "11_render_output"
    render.mkdir()
    (render / "render_processing_summary.json").write_text(
        json.dumps(provenance["static_producer"]["render_summary"]), encoding="utf-8"
    )
    return {
        "root": root,
        "execute": execute,
        "render": render,
        "paths": paths,
        "provenance": provenance,
    }


@pytest.mark.parametrize(
    "filename",
    [
        "pymdp_signed_simulation_results.json",
        "static_perception_simulation_results.json",
    ],
)
def test_historical_native_fixture_identity_and_producer_declarations(
    filename: str,
) -> None:
    provenance = json.loads((FIXTURES / "provenance.json").read_text())
    data = (FIXTURES / filename).read_bytes()
    expected = provenance["fixtures"][filename]
    assert len(data) == expected["bytes"]
    assert hashlib.sha256(data).hexdigest() == expected["sha256"]
    result = json.loads(data)
    assert result["model_name"] == expected["historical_model_name"]
    assert result["success"] is True and result["pymdp_version"] == "1.0.0"
    assert len(result["observations"]) == result["num_timesteps"]
    assert len(result["beliefs"]) == result["num_timesteps"]
    assert len(result["true_states"]) == result["num_timesteps"] + 1
    assert result["metrics"]["expected_free_energy"] == result["expected_free_energy"]
    assert "neg_efe = -EFE" in result["expected_free_energy_convention"]
    assert "historical" in provenance["scope"].lower()


def test_public_meta_analysis_reopens_saved_numeric_reports_without_invented_sweep(
    tmp_path: Path,
) -> None:
    fixture_before = _hashes(FIXTURES)
    saved = _saved_tree(tmp_path)
    # Produce the optional footprint through the maintained public parser;
    # this is separately captured serialization, not part of the historical run.
    source = tmp_path / "serialization-source"
    source.mkdir()
    shutil.copyfile(
        REPO / "input/gnn_files/basics/static_perception.md", source / "model.md"
    )
    source_before = _hashes(source)
    assert process_gnn_multi_format(
        source, saved["root"], LOGGER, serialize_preset="minimal"
    )
    before = _hashes(saved["root"])
    destination = tmp_path / "meta-report"
    result = run_meta_analysis(saved["execute"], destination, saved["render"])
    assert result is not None and result["records"] == 2
    # The archived model names do not declare a sweep grid. No renaming of
    # measured models to N/T slots, fabricated fits or plotting claims is allowed.
    assert result["plots"] == []
    report = Path(result["report"]).read_text()
    assert "Simulation Quality Metrics" in report
    assert "Scaling Analysis" not in report and "JIT compilation overhead" not in report
    records = {
        record.model_name: record
        for record in SweepDataCollector(saved["execute"], saved["render"]).collect()
    }
    assert set(records) == {"actinf_pomdp_agent", "tmaze_epistemic"}
    for name, path in saved["paths"].items():
        data = json.loads(path.read_text())
        record = records[name]
        assert record.num_states is None
        assert record.num_timesteps == data["num_timesteps"]
        assert record.model_params == data["model_parameters"]
        assert record.vfe_trace == data["metrics"]["variational_free_energy"]
        assert record.efe_trace == [
            max(row) for row in data["metrics"]["expected_free_energy"]
        ]
        # Public collector contract compares the saved paired observations and
        # states. This does not certify observation/state scientific equivalence.
        paired = list(zip(data["observations"], data["true_states"]))
        accuracy = sum(observation == state for observation, state in paired) / len(
            paired
        )
        assert record.final_accuracy == pytest.approx(accuracy)
        assert f"{accuracy:.3f}" in report
        final_belief = data["beliefs"][-1]
        entropy = -math.fsum(
            probability * math.log(probability)
            for probability in final_belief
            if probability > 0
        )
        assert record.mean_belief_entropy == pytest.approx(entropy, abs=1e-6)
        assert f"{record.mean_belief_entropy:.4f}" in report
    measured = saved["provenance"]["static_producer"]["execution_summary"][
        "execution_details"
    ][0]
    static = records["tmaze_epistemic"]
    assert static.execution_time == measured["execution_time"]
    assert static.execution_time_samples == measured["execution_time_samples"]
    assert static.execution_time_std == measured["execution_time_std"]
    assert records["actinf_pomdp_agent"].execution_time == 0.0
    metrics = next(
        iter(
            saved["provenance"]["static_producer"]["render_summary"][
                "file_results"
            ].values()
        )
    )["framework_results"]["pymdp"]["code_metrics"]
    assert static.lines_of_code == metrics["lines_of_code"]
    assert static.total_lines == metrics["total_lines"]
    statistics = json.loads(Path(result["statistics_json"]).read_text())
    runtime = statistics["per_framework"]["pymdp"]
    assert runtime["records_total"] == 2 and runtime["successful_runs"] == 1
    assert runtime["success_rate"] == 0.5
    assert runtime["runtime_median_s"] == measured["execution_time"]
    assert runtime["runtime_mean_s"] == measured["execution_time"]
    assert statistics["per_cell_best_framework"] == []
    assert statistics["loglog_runtime_vs_n_by_T"] == {}
    validation = json.loads(Path(result["validation_json"]).read_text())
    assert validation["summary"]["error"] == 0
    footprint = json.loads(
        (saved["root"] / "3_gnn_output/format_statistics.json").read_text()
    )
    assert "Step 3 serialization footprint" in report
    for fmt in ("markdown", "python", "json"):
        assert f"| {fmt} | {footprint[fmt]['total_size'] / 1e6:.2f} |" in report
    assert _hashes(source) == source_before
    assert _hashes(saved["root"]) == before
    assert _hashes(FIXTURES) == fixture_before


@pytest.mark.parametrize("summary_mode", ["absent", "invalid", "unreadable"])
def test_native_saved_log_fallback_preserves_actual_timing_and_partial_metrics(
    tmp_path: Path,
    summary_mode: str,
) -> None:
    saved = _saved_tree(tmp_path, summary_mode)
    before = _hashes(saved["root"])
    result = run_meta_analysis(saved["execute"], tmp_path / "fallback", saved["render"])
    assert result is not None and result["records"] == 2
    statistics = json.loads(Path(result["statistics_json"]).read_text())
    original = saved["provenance"]["static_producer"]["execution_summary"][
        "execution_details"
    ][0]
    assert (
        statistics["per_framework"]["pymdp"]["runtime_median_s"]
        == original["execution_time"]
    )
    records = SweepDataCollector(saved["execute"], saved["render"]).collect()
    measured = next(
        record for record in records if record.model_name == "tmaze_epistemic"
    )
    assert measured.execution_time_mean == original["execution_time_mean"]
    assert measured.execution_benchmark_repeats == 2
    assert measured.execution_time_samples == original["execution_time_samples"]
    assert measured.execution_time_std == original["execution_time_std"]
    assert measured.lines_of_code == 460 and measured.total_lines == 493
    assert _hashes(saved["root"]) == before


@pytest.mark.parametrize(
    "failure", ["missing", "empty", "invalid-json", "blocked-summary"]
)
def test_unavailable_saved_inputs_never_publish_a_successful_meta_report(
    tmp_path: Path,
    failure: str,
) -> None:
    execute = tmp_path / "unavailable-execution"
    if failure != "missing":
        execute.mkdir()
    if failure == "invalid-json":
        simulation = execute / "broken/pymdp/simulation_data/simulation_results.json"
        simulation.parent.mkdir(parents=True)
        simulation.write_text("{ malformed native result copy", encoding="utf-8")
    elif failure == "blocked-summary":
        (execute / "summaries/execution_summary.json").mkdir(parents=True)
    before = _hashes(execute)
    output = tmp_path / "no-report"
    assert run_meta_analysis(execute, output) is None
    assert not list(output.rglob("*.json")) and not list(output.rglob("*.md"))
    assert _hashes(execute) == before


def test_native_output_refusal_retains_saved_inputs_and_caller_file(
    tmp_path: Path,
) -> None:
    saved = _saved_tree(tmp_path)
    before = _hashes(saved["root"])
    output = tmp_path / "occupied-output"
    output.write_text("Retain the caller's original file.\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_meta_analysis(saved["execute"], output, saved["render"])
    assert output.read_text() == "Retain the caller's original file.\n"
    assert _hashes(saved["root"]) == before

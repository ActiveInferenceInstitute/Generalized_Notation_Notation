"""Bounded public benchmark consumers with genuine local execution/artifacts.

The static perception exemplar is explicitly aliased into a fixed corpus slot
to keep native execution small. These are receipt/behavior witnesses, not
throughput, scaling, exact peak-RSS, or backend convergence claims.
"""

from __future__ import annotations

import hashlib
import json
import math
import pstats
import shutil
import statistics
from pathlib import Path

import pytest
from PIL import Image

from gnn.analysis.complexity.benchmark import (
    estimate_path_complexity,
    run_complexity_benchmark,
)
from gnn.analysis.performance_benchmark import main, run_visualization_benchmark

REPO = Path(__file__).resolve().parents[2]
FIXED_SLOT = Path("discrete/tmaze_epistemic.md")


def _corpus(tmp_path: Path, relative: str) -> tuple[Path, Path, str]:
    target = tmp_path / "selected"
    original = REPO / "input/gnn_files" / relative
    destination = target / FIXED_SLOT
    destination.parent.mkdir(parents=True)
    shutil.copyfile(original, destination)
    return target, destination, hashlib.sha256(original.read_bytes()).hexdigest()


@pytest.mark.integration
def test_native_complexity_receipts_join_actual_repeated_execution_by_source(
    tmp_path: Path,
) -> None:
    target, source, source_sha = _corpus(tmp_path, "basics/static_perception.md")
    output = tmp_path / "empirical"
    result = run_complexity_benchmark(target, output, frameworks="pymdp", repeats=2)
    assert result["status"] == "success"
    assert result["models"] == ["Static Perception Model"]
    benchmark = json.loads(Path(result["benchmark_receipt"]).read_text())
    calibration = json.loads(Path(result["calibration_receipt"]).read_text())
    assert benchmark["receipt_type"] == "gnn.complexity_benchmark/v1"
    assert calibration["receipt_type"] == "gnn.complexity_calibration/v1"
    assert benchmark["environment"] == calibration["environment"]
    assert benchmark["environment"]["repeats"] == 2
    assert set(benchmark["environment"]["backend_versions"]) == {"pymdp"}
    assert (
        source.read_bytes() == (output / "benchmark_corpus" / FIXED_SLOT).read_bytes()
    )
    assert len(benchmark["rows"]) == 1
    row = benchmark["rows"][0]
    assert row == result["rows"][0]
    assert row["source_sha256"] == source_sha
    assert row["available"] is True and row["success"] is True
    assert row["return_code"] == 0 and row["error"] is None and not row["cancelled"]
    samples = row["execution_time_samples"]
    assert len(samples) == 2 and all(
        math.isfinite(value) and value > 0 for value in samples
    )
    assert row["execution_time"] == pytest.approx(statistics.median(samples))
    assert row["execution_time_mean"] == pytest.approx(statistics.mean(samples))
    assert row["execution_time_std"] == pytest.approx(statistics.pstdev(samples))
    assert row["child_peak_rss_mb"] > 0 and row["rss_samples_count"] > 0
    assert row["rss_sample_interval_seconds"] > 0

    summary = json.loads(
        (
            output
            / "benchmark/12_execute_output/summaries/execution_summary_detail.json"
        ).read_text()
    )
    assert len(summary["execution_details"]) == 1
    detail = summary["execution_details"][0]
    assert detail["source_sha256"] == source_sha and detail["attempts_started"] == 2
    assert detail["cleanup_verified"] is True
    assert detail["execution_time_samples"] == samples
    assert detail["child_peak_rss_mb"] == row["child_peak_rss_mb"]
    script = Path(detail["script_identity"]["path"])
    assert (
        hashlib.sha256(script.read_bytes()).hexdigest()
        == detail["script_identity"]["sha256"]
    )
    structured_files = list(
        (output / "benchmark/12_execute_output").glob(
            "*/*/execution_logs/*_results.json"
        )
    )
    assert len(structured_files) == 1
    structured = json.loads(structured_files[0].read_text())
    assert structured["success"] is True
    assert len(structured["simulation_data"]["observations"]) == 5
    assert len(structured["simulation_data"]["actions"]) == 5

    measured = [item for item in calibration["rows"] if item["framework"] == "pymdp"]
    assert len(measured) == 1 and measured[0]["source_sha256"] == source_sha
    assert measured[0]["wall_median_seconds"] == row["execution_time"]
    assert measured[0]["peak_rss_mb"] == row["child_peak_rss_mb"]
    assert measured[0]["calibration_note"].startswith("measured K=2 repeats")
    assert "ESTIMATE" in measured[0]["asymptotic"]
    unmeasured = [item for item in calibration["rows"] if item["framework"] != "pymdp"]
    assert len(unmeasured) == 10
    assert all(
        item["wall_median_seconds"] is None and item["peak_rss_mb"] is None
        for item in unmeasured
    )


@pytest.mark.integration
def test_actual_unsupported_execution_contract_never_becomes_a_measurement(
    tmp_path: Path,
) -> None:
    target, _, source_sha = _corpus(tmp_path, "discrete/tmaze_epistemic.md")
    output = tmp_path / "unsupported"
    result = run_complexity_benchmark(target, output, frameworks="pymdp", repeats=1)
    assert result["status"] == "failed"
    (row,) = result["rows"]
    assert row["source_sha256"] == source_sha and row["success"] is None
    for key in (
        "execution_time",
        "execution_time_mean",
        "execution_time_std",
        "execution_time_samples",
        "child_peak_rss_mb",
        "rss_samples_count",
        "rss_sample_interval_seconds",
        "return_code",
    ):
        assert row[key] is None
    render = json.loads(
        (
            output / "benchmark/11_render_output/render_processing_summary.json"
        ).read_text()
    )
    (refusal,) = render["unsupported_framework_renderings"]
    assert refusal["framework"] == "pymdp"
    assert "unsupported-execution-contract" in refusal["message"]
    assert not list((output / "benchmark/11_render_output").glob("*/*/*.py"))
    calibration = json.loads(Path(result["calibration_receipt"]).read_text())
    (backend,) = [item for item in calibration["rows"] if item["framework"] == "pymdp"]
    assert backend["wall_median_seconds"] is None and backend["peak_rss_mb"] is None
    assert "no measurement recorded" in backend["calibration_note"]


def test_static_file_and_directory_receipts_preserve_actual_byte_identities(
    tmp_path: Path,
) -> None:
    target, source, source_sha = _corpus(tmp_path, "basics/static_perception.md")
    other = target / "nested/other.md"
    other.parent.mkdir()
    other.write_bytes(source.read_bytes())
    single = estimate_path_complexity(source)
    directory = estimate_path_complexity(target)
    assert len(single) == 1 and len(directory) == 2
    assert single[0]["model"]["source_sha256"] == source_sha
    assert [item["model"]["path"] for item in directory] == [str(source), str(other)]
    assert all(item["model"]["source_sha256"] == source_sha for item in directory)
    assert all(
        item["receipt_type"] == "gnn.complexity_estimate/v1" for item in directory
    )
    for item in directory:
        assert "execution_time" not in item
        (backend,) = [row for row in item["per_backend"] if row["framework"] == "pymdp"]
        assert "ESTIMATE" in backend["asymptotic"]
        assert "execution_time" not in backend
    with pytest.raises(FileNotFoundError):
        estimate_path_complexity(tmp_path / "missing.md")


def test_missing_fixed_corpus_and_unknown_backend_refuse_execution(
    tmp_path: Path,
) -> None:
    target = tmp_path / "empty"
    target.mkdir()
    with pytest.raises(ValueError, match="no corpus models staged"):
        run_complexity_benchmark(target, tmp_path / "empty-output", frameworks="pymdp")
    with pytest.raises(ValueError, match="unknown framework"):
        run_complexity_benchmark(
            target, tmp_path / "invalid-output", frameworks="provider-name"
        )
    assert not list(tmp_path.rglob("complexity_benchmark.json"))
    assert not list(tmp_path.rglob("*_results.json"))


@pytest.mark.integration
def test_public_profile_cli_records_real_step8_artifacts_and_a_readable_profile(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "source"
    target.mkdir()
    source = target / "model.md"
    source.write_bytes(
        (REPO / "input/gnn_files/basics/static_perception.md").read_bytes()
    )
    output = tmp_path / "profiled"
    assert (
        main(
            [
                "--target-dir",
                str(target),
                "--output-dir",
                str(output),
                "--repetitions",
                "1",
                "--concurrency",
                "1",
                "--timeout-seconds",
                "90",
                "--max-source-files",
                "1",
                "--profile",
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    receipt = json.loads((output / "benchmark.json").read_text())
    assert printed["success"] is receipt["success"] is True
    assert receipt["configuration"]["profile"] is True
    assert receipt["configuration"]["llm_mode"] == "disabled"
    assert receipt["code_identity_unchanged"] is True
    assert (
        receipt["corpus"][0]["sha256"]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    stats = pstats.Stats(str(output / "run_00/phase.prof"))
    assert stats.total_calls > 0
    (row,) = receipt["rows"]
    assert row["phase"]["returncode"] == 0 and row["phase"]["process_cpu_seconds"] > 0
    assert row["envelope"]["cleanup_verified"] is True
    pngs = [item for item in row["artifacts"] if item["path"].endswith(".png")]
    assert len(pngs) > 1 and row["png_artifact_count"] == len(pngs)
    assert row["artifact_count"] == len(row["artifacts"])
    assert row["artifact_bytes"] == sum(item["bytes"] for item in row["artifacts"])
    for artifact in row["artifacts"]:
        path = output / "run_00/output" / artifact["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
    for artifact in pngs:
        with Image.open(output / "run_00/output" / artifact["path"]) as image:
            assert image.size == (artifact["width"], artifact["height"])
            assert (
                hashlib.sha256(image.convert("RGBA").tobytes()).hexdigest()
                == artifact["rgba_sha256"]
            )


@pytest.mark.parametrize(
    "case, message",
    [
        ("missing", "must be a directory"),
        ("empty", "at least one selected source"),
        ("inside", "outside the selected source tree"),
        ("symlink", "ordinary files confined"),
    ],
)
def test_public_visualization_benchmark_refuses_invalid_source_boundaries(
    tmp_path: Path, case: str, message: str
) -> None:
    target, output = tmp_path / "selected", tmp_path / "unused"
    if case != "missing":
        target.mkdir()
    if case in {"inside", "symlink"}:
        source = target / "model.gnn"
        if case == "symlink":
            outside = tmp_path / "outside.gnn"
            outside.write_text("caller-owned source")
            source.symlink_to(outside)
        else:
            source.write_text("caller-owned source")
            output = target / "inside"
    with pytest.raises(ValueError, match=message):
        run_visualization_benchmark(target, output, repetitions=1)
    assert not output.exists()

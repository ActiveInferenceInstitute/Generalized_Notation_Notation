"""Native benchmark admission, measurement truth and lossless encoder behavior."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from gnn.analysis.analyzer import run_performance_benchmarks
from gnn.analysis.performance_benchmark import run_visualization_benchmark
from gnn.visualization.plotting.utils import save_figure


def test_faster_png_encoding_preserves_pixels_metadata_and_explicit_options(
    tmp_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1, 2], [0.2, -0.3, 0.8], "o--", label="Reported values")
    ax.legend()
    baseline, candidate, requested = (
        tmp_path / name for name in ("baseline.png", "candidate.png", "requested.png")
    )
    try:
        plt.savefig(
            baseline,
            dpi=90,
            metadata={"Description": "Bound plot"},
            pil_kwargs={"compress_level": 6},
        )
        save_figure(candidate, dpi=90, metadata={"Description": "Bound plot"})
        save_figure(
            requested,
            dpi=90,
            metadata={"Description": "Bound plot"},
            pil_kwargs={"compress_level": 6},
        )
        with Image.open(baseline) as first, Image.open(candidate) as second:
            assert first.size == second.size
            assert first.info == second.info
            np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
        assert requested.read_bytes() == baseline.read_bytes()
        assert candidate.read_bytes() != baseline.read_bytes()
        assert candidate.stat().st_size > baseline.stat().st_size
    finally:
        plt.close(fig)


def test_non_png_export_retains_format_defaults(tmp_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    try:
        path = tmp_path / "native.svg"
        save_figure(path)
        assert "<svg" in path.read_text()
    finally:
        plt.close(fig)


def test_parser_benchmark_distinguishes_estimates_from_process_measurement(
    tmp_path: Path,
) -> None:
    source = tmp_path / "model.gnn"
    source.write_text(
        "## StateSpaceBlock\ns[2,1,type=Categorical]\n## Connections\ns > s\n"
    )
    receipt = run_performance_benchmarks(source)
    assert receipt["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert receipt["parse_time"] >= 0 and receipt["parse_cpu_time"] >= 0
    assert "not measured RSS" in receipt["memory_measurement"]
    assert "heuristic" in receipt["runtime_estimate_method"]


@pytest.mark.parametrize(
    "options",
    [
        {"repetitions": 0},
        {"repetitions": 6},
        {"concurrency": True},
        {"concurrency": 3},
        {"timeout_seconds": float("inf")},
        {"timeout_seconds": 1801},
        {"max_input_bytes": 0},
    ],
)
def test_invalid_benchmark_budgets_refuse_before_output_or_spawn(
    tmp_path: Path, options: dict
) -> None:
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    (source / "model.gnn").write_text("source")
    with pytest.raises(ValueError):
        run_visualization_benchmark(source, output, **options)
    assert not output.exists()


def test_input_byte_limit_is_independent_of_rss_estimates(tmp_path: Path) -> None:
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    (source / "model.gnn").write_bytes(b"x" * 1024)
    with pytest.raises(ValueError, match="source bytes"):
        run_visualization_benchmark(source, output, max_input_bytes=1023)
    assert not output.exists()


def test_existing_output_cannot_serve_as_a_fresh_benchmark(tmp_path: Path) -> None:
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    inherited = output / "inherited.png"
    inherited.write_bytes(b"prior-run")
    (source / "model.gnn").write_text("source")
    with pytest.raises(FileExistsError):
        run_visualization_benchmark(source, output, repetitions=1)
    assert inherited.read_bytes() == b"prior-run"


@pytest.mark.integration
def test_native_benchmark_measures_nested_selection_and_real_artifacts(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[2]
    source = tmp_path / "source" / "nested"
    source.mkdir(parents=True)
    original = repo / "input/gnn_files/basics/static_perception.md"
    shutil.copyfile(original, source / original.name)
    receipt = run_visualization_benchmark(
        source.parent, tmp_path / "fresh", repetitions=1, timeout_seconds=90
    )
    assert receipt["success"]
    assert receipt["corpus"][0]["path"] == "nested/static_perception.md"
    assert (
        receipt["corpus"][0]["sha256"]
        == hashlib.sha256(original.read_bytes()).hexdigest()
    )
    assert receipt["configuration"]["llm_mode"] == "disabled"
    row = receipt["rows"][0]
    assert row["phase"]["process_cpu_seconds"] > 0
    assert row["envelope"]["rss_samples_count"] > 0
    assert row["envelope"]["child_peak_rss_mb"] > 0
    pngs = [
        artifact for artifact in row["artifacts"] if artifact["path"].endswith(".png")
    ]
    assert len(pngs) > 1
    assert all(artifact["rgba_sha256"] and artifact["width"] > 100 for artifact in pngs)

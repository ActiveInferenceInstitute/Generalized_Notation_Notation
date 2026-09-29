#!/usr/bin/env python3
"""Step 8 must not fail figures silently.

Covers the matrix visualizer's error handling (a figure-creation error is
logged; a ``TypeError`` from an oversized Agg canvas reaches the smaller save
fallbacks) and the step exit code (any failed figure turns the step into the
pipeline's ``2`` = success-with-warnings, while a clean corpus stays ``0``).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gnn.utils.errors.error_handling import coerce_step_exit_code
from gnn.visualization.core.process import process_visualization
from gnn.visualization.matrix import visualizer as visualizer_module
from gnn.visualization.matrix.visualizer import MatrixVisualizer

REPO_ROOT = Path(__file__).resolve().parents[2]
GNN_INPUT = REPO_ROOT / "input" / "gnn_files"
STEP_8 = REPO_ROOT / "src" / "gnn" / "8_visualization.py"
_TEST_LOGGER = logging.getLogger("test_visualization_figure_failures")


def _transition_tensor() -> np.ndarray:
    """A valid 3x3x2 POMDP transition tensor (columns sum to 1)."""
    tensor = np.zeros((3, 3, 2))
    tensor[:, :, 0] = np.eye(3)
    tensor[:, :, 1] = np.full((3, 3), 1.0 / 3.0)
    return tensor


def _step_exit_code(result: Any) -> int:
    return coerce_step_exit_code(
        result, step_name="8_visualization.py", logger=_TEST_LOGGER
    )


@pytest.mark.unit
def test_pomdp_analysis_figure_creation_error_is_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def _broken_figure(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("figure backend unavailable")

    monkeypatch.setattr(visualizer_module.plt, "figure", _broken_figure)
    output_path = tmp_path / "analysis.png"

    with caplog.at_level(logging.ERROR, logger=visualizer_module.logger.name):
        ok = MatrixVisualizer().generate_pomdp_transition_analysis(
            _transition_tensor(), output_path
        )

    assert ok is False
    assert not output_path.exists()
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "figure-creation failure was swallowed without a log record"
    record = errors[0]
    assert "analysis.png" in record.getMessage()
    assert record.exc_info is not None
    assert "figure backend unavailable" in str(record.exc_info[1])


@pytest.mark.unit
def test_pomdp_analysis_type_error_on_save_reaches_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_savefig = visualizer_module.plt.savefig
    calls: list[dict[str, Any]] = []

    def _savefig(*args: Any, **kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        if len(calls) == 1:
            # What RendererAgg raises for an oversized tight-bbox canvas.
            raise TypeError("width and height must each be below 32768")
        return real_savefig(*args, **kwargs)

    monkeypatch.setattr(visualizer_module.plt, "savefig", _savefig)
    output_path = tmp_path / "analysis.png"

    ok = MatrixVisualizer().generate_pomdp_transition_analysis(
        _transition_tensor(), output_path
    )

    assert ok is True
    assert output_path.is_file()
    assert len(calls) == 2
    assert calls[1] == {"dpi": 72}, (
        "second attempt must be the (8, 6) @ 72 dpi fallback"
    )


def _write_gnn_input(target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    source = GNN_INPUT / "discrete" / "simple_mdp.md"
    dest = target / source.name
    dest.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


@pytest.mark.unit
def test_step8_reports_failed_figure_as_warning_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "in"
    _write_gnn_input(target)
    out = tmp_path / "8_visualization_output"

    def _failing_analysis(
        self: MatrixVisualizer, tensor: Any, output_path: Path
    ) -> bool:
        return False

    monkeypatch.setattr(
        MatrixVisualizer, "generate_pomdp_transition_analysis", _failing_analysis
    )

    result = process_visualization(target, out)

    assert result == 2
    assert _step_exit_code(result) == 2
    summary = json.loads((out / "visualization_summary.json").read_text("utf-8"))
    failures = summary["errors"]
    assert any("POMDP transition analysis" in f for f in failures), failures

    manifest = json.loads(
        (out / "simple_mdp" / "simple_mdp_viz_manifest.json").read_text("utf-8")
    )
    assert manifest["figure_failures"] == failures

    # The partial cache must not pass as clean: the next run retries the
    # failed figure instead of reusing the cached PNGs.
    monkeypatch.undo()
    assert process_visualization(target, out) is True
    assert (out / "simple_mdp" / "simple_mdp_B_analysis.png").is_file()


@pytest.mark.unit
def test_step8_clean_file_and_rank4_tensor_exit_zero(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    target = tmp_path / "in"
    target.mkdir()
    for name in ("simple_mdp.md", "time_varying_dynamics.md"):
        source = GNN_INPUT / "discrete" / name
        (target / name).write_text(source.read_text(encoding="utf-8"), "utf-8")
    out = tmp_path / "8_visualization_output"

    with caplog.at_level(logging.WARNING):
        result = process_visualization(target, out)

    # A rank-4 tensor (time_varying_dynamics.B_t) has no renderer: it must be
    # skipped as unsupported, not attempted and logged as a figure ERROR.
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors == []
    summary = json.loads((out / "visualization_summary.json").read_text("utf-8"))
    assert summary["errors"] == []
    assert result is True
    assert _step_exit_code(result) == 0


@pytest.mark.pipeline
@pytest.mark.slow
def test_step8_script_clean_corpus_exits_zero(tmp_path: Path) -> None:
    env = {**os.environ, "MPLBACKEND": "Agg"}
    proc = subprocess.run(
        [
            sys.executable,
            str(STEP_8),
            "--target-dir",
            str(GNN_INPUT),
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=900,
    )
    log = proc.stdout + proc.stderr
    assert proc.returncode == 0, log[-4000:]
    assert " - ERROR - " not in log, log[-4000:]

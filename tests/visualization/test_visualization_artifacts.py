#!/usr/bin/env python3
"""Tests for step-8 sidecars and figure robustness.

Covers network stats orientation, ontology legend and viz manifest; the
matrix visualizer's error handling; the Step 8 exit code when a figure fails
(``2`` = success-with-warnings, while a clean corpus stays ``0``); and bar value
labels offset in points rather than data units.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.analysis.jax.analyzer import (
    create_jax_visualizations,
    create_visualizations_from_structured_data,
)
from gnn.analysis.viz_plots import generate_action_analysis
from gnn.integration.meta_analysis.collector import SweepRecord
from gnn.integration.meta_analysis.visualizer import SweepVisualizer
from gnn.utils.errors.error_handling import coerce_step_exit_code
from gnn.visualization.core.process import (
    discover_visualization_files,
    process_single_gnn_file,
    process_visualization,
)
from gnn.visualization.graph.network_visualizations import (
    generate_network_visualizations,
)
from gnn.visualization.matrix import visualizer as visualizer_module
from gnn.visualization.matrix.visualizer import MatrixVisualizer
from tests.helpers.bar_labels import (
    assert_bar_labels_offset_in_points,
    assert_png_bounded,
    figures_held_open,
    open_bar_figures,
)


@pytest.mark.unit
def test_network_stats_gnn_edge_orientation_and_ontology_legend(tmp_path: Path) -> None:
    out = tmp_path / "viz"
    out.mkdir()
    parsed: dict[str, Any] = {
        "variables": [
            {"name": "a", "var_type": "hidden_state"},
            {"name": "b", "var_type": "hidden_state"},
            {"name": "c", "var_type": "observation"},
        ],
        "connections": [
            {
                "source_variables": ["a"],
                "target_variables": ["b"],
                "connection_type": "directed",
            },
            {
                "source_variables": ["b"],
                "target_variables": ["c"],
                "connection_type": "undirected",
            },
        ],
        "ontology_labels": {"a": "HiddenState", "c": "Observation"},
    }
    paths = generate_network_visualizations(parsed, out, "mini")
    if not paths:
        raise AssertionError("networkx/matplotlib not available for graph generation")

    stats_path = out / "mini_network_stats.json"
    assert stats_path.is_file()
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert "gnn_edge_orientation" in stats
    orient = stats["gnn_edge_orientation"]
    assert orient["directed_variable_pairs"] == 1
    assert orient["undirected_variable_pairs"] == 1

    leg = out / "mini_ontology_legend.txt"
    assert leg.is_file()
    text = leg.read_text(encoding="utf-8")
    assert "variable\tontology_term" in text
    assert "a\tHiddenState" in text
    assert "c\tObservation" in text


@pytest.mark.unit
def test_viz_manifest_json_after_process_single_gnn_file(tmp_path: Path) -> None:
    base = tmp_path
    gnn_in = base / "in"
    gnn_in.mkdir()
    gnn_file = gnn_in / "tiny.md"
    gnn_file.write_text("# tiny\n", encoding="utf-8")

    step3_model = base / "3_gnn_output" / "tiny"
    step3_model.mkdir(parents=True)
    parsed: dict[str, Any] = {
        "model_name": "tiny",
        "variables": [
            {"name": "x", "var_type": "hidden_state", "dimensions": [2]},
            {"name": "y", "var_type": "observation", "dimensions": [2]},
        ],
        "connections": [
            {
                "source_variables": ["x"],
                "target_variables": ["y"],
                "connection_type": "directed",
            },
        ],
        "parameters": [
            {
                "name": "A",
                "type": "matrix",
                "shape": [2, 2],
                "values": [1.0, 0.0, 0.0, 1.0],
            }
        ],
        "ontology_mappings": [
            {"variable_name": "x", "ontology_term": "HiddenState"},
        ],
        "raw_sections": {},
    }
    parsed_path = step3_model / "tiny_parsed.json"
    parsed_path.write_text(json.dumps(parsed), encoding="utf-8")

    results_dir = base / "8_visualization_output"
    results_dir.mkdir(parents=True)

    paths = process_single_gnn_file(gnn_file, results_dir, verbose=False)
    assert paths, "expected at least one visualization artifact"

    model_dir = results_dir / "tiny"
    manifest_path = model_dir / "tiny_viz_manifest.json"
    assert manifest_path.is_file(), f"missing manifest; got paths: {paths[:5]}..."

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_name"] == "tiny"
    assert manifest["ontology_label_count"] == 1
    assert "viz_meta" in manifest
    assert manifest["viz_meta"].get("source") == "parsed_json"
    assert isinstance(manifest["artifacts"], list)
    assert manifest["artifact_count"] == len(manifest["artifacts"])

    leg = model_dir / "tiny_ontology_legend.txt"
    assert leg.is_file()

    stats_path = model_dir / "tiny_network_stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        assert "gnn_edge_orientation" in stats


@pytest.mark.unit
def test_visualization_discovery_honors_recursive_flag(tmp_path: Path) -> None:
    source = tmp_path / "models"
    nested = source / "nested"
    nested.mkdir(parents=True)
    root_file = source / "root_model.gnn"
    nested_file = nested / "nested_model.gnn"
    root_file.write_text("root", encoding="utf-8")
    nested_file.write_text("nested", encoding="utf-8")

    non_recursive = discover_visualization_files(source, recursive=False)
    recursive = discover_visualization_files(source, recursive=True)

    assert non_recursive == [root_file]
    assert recursive == sorted(
        [root_file, nested_file], key=lambda path: path.relative_to(source).as_posix()
    )


@pytest.mark.unit
def test_process_visualization_empty_input_returns_warning_code(tmp_path: Path) -> None:
    source = tmp_path / "empty"
    source.mkdir()
    output = tmp_path / "viz_output"

    result = process_visualization(source, output, recursive=False)

    assert result == 2
    summary = json.loads((output / "visualization_summary.json").read_text())
    assert summary["processed_files"] == 0
    assert summary["warnings"]


# --- matplotlib 3.11 regressions --------------------------------------------
# uv.lock resolves matplotlib 3.10 on Python 3.11 and 3.11 on Python >= 3.12,
# so these run on every CI leg. Each test asserts the figure file is actually
# written: the plotting code catches and logs its own exceptions, so a return
# value alone does not prove the figure exists. Covered: boxplot(labels=)
# removed in 3.11 (tick_labels= since 3.9); Axes.pie returning a PieContainer
# with no len() in 3.11; and a deterministic B tensor making the POMDP
# transition analysis request a ~1081 x 18e9 px Agg canvas (every version).


def _assert_png_written(path: Path) -> None:
    assert path.is_file(), f"figure not written: {path}"
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


class TestPomdpTransitionAnalysis:
    """generate_pomdp_transition_analysis must write a bounded-size PNG."""

    @pytest.mark.parametrize(
        "tensor",
        [
            # Identity-like permutation matrices: zero entropy per action.
            np.stack(
                [
                    np.eye(3),
                    np.eye(3)[[1, 0, 2]],
                    np.eye(3)[[2, 1, 0]],
                ],
                axis=2,
            ),
            # Single-row deterministic tensor (tmaze_epistemic shape).
            np.array([[[1.0, 0.0], [0.0, 1.0]]]),
            # Ten actions: past the 3x3 grid, add_subplot(3, 3, 10) raised before the cap.
            np.full((3, 3, 10), 1 / 3),
        ],
        ids=["deterministic_3x3x3", "deterministic_1x2x2", "ten_actions"],
    )
    def test_writes_png_of_sane_size(self, tmp_path: Path, tensor) -> None:
        from PIL import Image

        from gnn.visualization.matrix.visualizer import MatrixVisualizer

        out = tmp_path / "pomdp_transition_analysis.png"
        assert MatrixVisualizer().generate_pomdp_transition_analysis(tensor, out)
        _assert_png_written(out)
        with Image.open(out) as img:
            width, height = img.size
        assert width < 5000 and height < 5000, (width, height)

    def test_deterministic_transitions_have_zero_entropy_labels(
        self, tmp_path: Path
    ) -> None:
        """0·log(0) is 0, so deterministic transitions must not go negative."""

        from gnn.visualization.matrix.visualizer import MatrixVisualizer

        tensor = np.stack([np.eye(2), np.eye(2)[[1, 0]]], axis=2)
        captured: list[str] = []
        original_savefig = plt.savefig

        def capture_then_save(*args, **kwargs):
            fig = plt.gcf()
            entropy_ax = next(
                ax
                for ax in fig.axes
                if ax.get_title() == "Transition Entropy by Action"
            )
            captured.extend(t.get_text() for t in entropy_ax.texts)
            return original_savefig(*args, **kwargs)

        plt.savefig = capture_then_save
        try:
            out = tmp_path / "pomdp.png"
            assert MatrixVisualizer().generate_pomdp_transition_analysis(tensor, out)
        finally:
            plt.savefig = original_savefig
        _assert_png_written(out)
        assert captured == ["0.000", "0.000"]


class TestTypeCategoryPieChart:
    """generate_type_category_pie_chart must handle tuple and PieContainer."""

    def test_writes_png(self, tmp_path: Path) -> None:
        from gnn.type_checker.visualizer import generate_type_category_pie_chart

        results = {
            "type_analysis": [
                {"type_distribution": {"float": 6, "int": 3, "categorical": 2}},
                {"type_distribution": {"float": 2, "bool": 1}},
            ]
        }
        path = generate_type_category_pie_chart(results, tmp_path)
        assert path is not None
        _assert_png_written(path)


class TestBoxplotTickLabels:
    """Box plots must use ``tick_labels=`` (``labels=`` removed in 3.11)."""

    def test_unified_dashboard_entropy_comparison_written(self, tmp_path: Path) -> None:
        from gnn.analysis.viz_dashboard import generate_unified_framework_dashboard

        rng = np.random.default_rng(0)

        def beliefs(steps: int = 6, states: int = 3) -> list[list[float]]:
            raw = rng.random((steps, states)) + 0.1
            return (raw / raw.sum(axis=1, keepdims=True)).tolist()

        framework_data = {
            "pymdp": {
                "framework": "pymdp",
                "simulation_data": {"beliefs": beliefs(), "actions": [0, 1, 0]},
            },
            "jax": {
                "framework": "jax",
                "simulation_data": {"beliefs": beliefs(), "actions": [1, 1, 0]},
            },
        }
        generate_unified_framework_dashboard(framework_data, tmp_path, "Test Model")
        _assert_png_written(tmp_path / "unified_entropy_comparison.png")

    def test_render_statistical_plots_written(self, tmp_path: Path) -> None:
        from gnn.render.visualization_suite import VisualizationSuite

        suite = VisualizationSuite(tmp_path, "mpl_compat")
        traces = {
            "belief": [0.1, 0.4, 0.35, 0.6, 0.55],
            "reward": [0.0, 1.0, 0.0, 1.0, 1.0],
        }
        files = suite._create_statistical_plots(traces)
        assert files, "no statistical plot produced"
        for path in files:
            _assert_png_written(Path(path))


# --- Step 8: no silent figure failures ------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
GNN_INPUT = REPO_ROOT / "input" / "gnn_files"
STEP_8 = REPO_ROOT / "src" / "gnn" / "8_visualization.py"
_TEST_LOGGER = logging.getLogger("test_visualization_artifacts")


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


# --- Bar value labels: offset in points, not data units --------------------
#
# A data-unit offset (``bar.get_height() + 0.5``) is unbounded in pixels: its
# screen size scales with 1 / y-range (same bug class as the 18e9-px tight-bbox
# canvas). Each site runs with degenerate and large-range data; the pixel gap
# between bar top and label must be the fixed 3 pt, and the PNG bounded.


@pytest.fixture
def kept_figures(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep figures open past the code-under-test's ``plt.close`` calls."""
    with figures_held_open(monkeypatch):
        yield


def _bar_figure() -> Figure:
    """The single open figure that contains bars."""
    (fig,) = open_bar_figures()
    return fig


# Degenerate: a single action taken once (count range [0, 1]); large: counts
# in the thousands. A data-unit offset is ~half the axes in the first case and
# sub-pixel in the second; a point offset is the same in both.
ACTION_SEQUENCES = {
    "degenerate": [0],
    "large": [0] * 1500 + [1] * 2500,
}


@pytest.mark.unit
@pytest.mark.parametrize("case", sorted(ACTION_SEQUENCES))
def test_viz_plots_action_counts_label_offset(
    tmp_path: Path, kept_figures: None, case: str
) -> None:
    out = tmp_path / "actions.png"
    generate_action_analysis(ACTION_SEQUENCES[case], out)
    assert_png_bounded(out)
    assert_bar_labels_offset_in_points(_bar_figure())


@pytest.mark.unit
@pytest.mark.parametrize("case", sorted(ACTION_SEQUENCES))
def test_jax_structured_action_counts_label_offset(
    tmp_path: Path, kept_figures: None, case: str
) -> None:
    paths = create_visualizations_from_structured_data(
        {"actions": ACTION_SEQUENCES[case]}, tmp_path, "m"
    )
    pngs = [Path(p) for p in paths if Path(p).suffix == ".png"]
    assert pngs, paths
    for png in pngs:
        assert_png_bounded(png)
    assert_bar_labels_offset_in_points(_bar_figure())


@pytest.mark.unit
@pytest.mark.parametrize("case", sorted(ACTION_SEQUENCES))
def test_jax_action_distribution_label_offset(
    tmp_path: Path, kept_figures: None, case: str
) -> None:
    paths = create_jax_visualizations(
        {"simulation_data": {"actions": ACTION_SEQUENCES[case]}}, tmp_path, "m"
    )
    dist = tmp_path / "m_jax_action_dist.png"
    assert str(dist) in paths, paths
    assert_png_bounded(dist)
    assert_bar_labels_offset_in_points(_bar_figure())


def _sweep(time_of: Callable[[int, int], float]) -> list[SweepRecord]:
    return [
        SweepRecord(
            model_name=f"pymdp_scaling_N{n}_T{t}",
            framework="pymdp",
            num_states=n,
            num_timesteps=t,
            execution_time=time_of(n, t),
            success=True,
        )
        for n in (2, 4, 8)
        for t in (10, 20, 40)
    ]


# Degenerate: flat timings give exponents of ~0 (the y-range collapses onto
# the alpha=1 reference line); large: steep power laws give exponents ~40.
SWEEPS: dict[str, Callable[[int, int], float]] = {
    "degenerate": lambda n, t: 0.25,
    "large": lambda n, t: 1e-6 * float(n) ** 40 * float(t) ** 40,
}


@pytest.mark.unit
@pytest.mark.parametrize("case", sorted(SWEEPS))
def test_meta_analysis_scaling_exponent_label_offset(
    tmp_path: Path, kept_figures: None, case: str
) -> None:
    records = _sweep(SWEEPS[case])
    viz = SweepVisualizer(records, tmp_path)
    path = viz._plot_scaling_exponent_summary(records, ["pymdp"])
    assert path is not None
    assert_png_bounded(Path(path))
    assert_bar_labels_offset_in_points(_bar_figure())

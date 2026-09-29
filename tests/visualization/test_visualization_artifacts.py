#!/usr/bin/env python3
"""Tests for step-8 sidecars: network stats orientation, ontology legend, viz manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.visualization.core.process import (
    discover_visualization_files,
    process_single_gnn_file,
    process_visualization,
)
from gnn.visualization.graph.network_visualizations import (
    generate_network_visualizations,
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
            # More actions than the grid's top-row cells.
            np.full((4, 4, 5), 0.25),
        ],
        ids=["deterministic_3x3x3", "deterministic_1x2x2", "five_actions"],
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
        import matplotlib.pyplot as plt

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

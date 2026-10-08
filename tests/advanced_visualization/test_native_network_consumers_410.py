"""Saved Step 3 models reach native public network artifacts without data loss."""

import json
from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure
from PIL import Image

from gnn.advanced_visualization import process_advanced_viz


def saved_model(tmp_path, edges, *, legacy=False, variables=None):
    names = ["sensor", "belief", "prior"]
    model = {
        "variables": variables
        if variables is not None
        else [
            {"name": name, "var_type": kind, "dimensions": [3, 2]}
            for name, kind in zip(
                names, ["observation", "hidden_state", "prior_vector"], strict=True
            )
        ],
        "connections": [
            {"source": source, "target": target}
            if legacy
            else {"source_variables": [source], "target_variables": [target]}
            for source, target in edges
        ],
    }
    step3 = tmp_path / "3_gnn_output"
    step3.mkdir()
    parsed = step3 / "authored_network_parsed.json"
    parsed.write_text(json.dumps(model))
    inventory = step3 / "gnn_processing_results.json"
    inventory.write_text(
        json.dumps(
            {
                "processed_files": [
                    {"parse_success": True, "parsed_model_file": str(parsed)}
                ],
                "output_directory": str(step3),
            }
        )
    )
    return model, parsed, inventory


@pytest.fixture
def figures(monkeypatch):
    """Observe native savefig without replacing rendering or the producer."""
    saved = {}
    original = Figure.savefig

    def record(figure, filename, *args, **kwargs):
        saved[Path(filename).name] = figure
        return original(figure, filename, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", record)
    return saved


def run_saved_model(tmp_path, parsed, inventory, viz_type):
    before = {path: path.read_bytes() for path in [parsed, inventory]}
    output = tmp_path / "9_advanced_viz_output"
    result = process_advanced_viz(tmp_path, output, viz_type=viz_type)
    assert {path: path.read_bytes() for path in before} == before
    summary = json.loads((output / "advanced_viz_summary.json").read_text())
    return result, output, summary


def assert_native_artifact(summary, output):
    assert (summary["successful"], summary["failed"], summary["skipped"]) == (1, 0, 0)
    attempt = summary["attempts"][0]
    assert attempt["status"] == "success"
    assert attempt["fallback_used"] is False
    assert attempt["output_files"] == summary["output_files"]
    path = Path(summary["output_files"][0])
    assert path.parent == output
    if path.suffix == ".png":
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.width > 100 and image.height > 100
            image.verify()
    else:
        assert path.is_file() and path.stat().st_size > 100
    return path


@pytest.mark.parametrize(
    ("edges", "expected"),
    [
        (
            [("sensor", "belief"), ("belief", "prior")],
            {"nodes": 3, "edges": 2, "density": 1 / 3, "avg_clustering": 0},
        ),
        (
            [("sensor", "belief"), ("belief", "prior"), ("prior", "sensor")],
            {"nodes": 3, "edges": 3, "density": 0.5, "avg_clustering": 1},
        ),
    ],
)
def test_public_network_metrics_retain_directed_graph_and_hand_computed_values(
    tmp_path, figures, edges, expected
):
    _, parsed, inventory = saved_model(tmp_path, edges)
    result, output, summary = run_saved_model(tmp_path, parsed, inventory, "network")
    assert result is True
    path = assert_native_artifact(summary, output)
    graph, metrics = figures[path.name].axes
    assert set(text.get_text() for text in graph.texts) == {"sensor", "belief", "prior"}
    actual = dict(
        zip(
            [text.get_text() for text in metrics.get_xticklabels()],
            [bar.get_height() for bar in metrics.patches],
            strict=True,
        )
    )
    assert actual == pytest.approx({**expected, "max_degree_centrality": 1})


@pytest.mark.parametrize("legacy", [False, True])
def test_public_3d_network_edges_connect_the_actual_authored_nodes(
    tmp_path, figures, legacy
):
    edges = [("sensor", "belief"), ("belief", "prior")]
    _, parsed, inventory = saved_model(tmp_path, edges, legacy=legacy)
    result, output, summary = run_saved_model(tmp_path, parsed, inventory, "3d")
    assert result is True
    path = assert_native_artifact(summary, output)
    axis = figures[path.name].axes[0]
    points = {
        text.get_text(): np.array([text.get_position_3d()]).reshape(3)
        for text in axis.texts
    }
    assert set(points) == {"sensor", "belief", "prior"}
    assert len(axis.lines) == len(edges)
    for line, (source, target) in zip(axis.lines, edges, strict=True):
        np.testing.assert_array_equal(
            np.asarray(line.get_data_3d()).T, [points[source], points[target]]
        )
    assert (axis.get_xlabel(), axis.get_ylabel(), axis.get_zlabel()) == (
        "X Dimension",
        "Y Dimension",
        "Z Dimension",
    )


@pytest.mark.parametrize("legacy", [False, True])
def test_public_interactive_dashboard_preserves_direction_counts_and_node_identity(
    tmp_path, monkeypatch, legacy
):
    go = pytest.importorskip("plotly.graph_objects")
    figures = {}
    original = go.Figure.to_html

    def observe(figure, *args, **kwargs):
        figures[figure.layout.title.text] = figure
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(go.Figure, "to_html", observe)
    edges = [("sensor", "belief"), ("belief", "prior")]
    _, parsed, inventory = saved_model(tmp_path, edges, legacy=legacy)
    result, output, summary = run_saved_model(tmp_path, parsed, inventory, "dashboard")
    assert result is True
    path = assert_native_artifact(summary, output)
    html = path.read_text()
    assert "GNN Model Dashboard: authored_network" in html
    assert "Adjacency Matrix" in html and "3D Network Structure" in html
    pie = figures["Variable Type Distribution"].data[0]
    assert dict(zip(pie.labels, pie.values, strict=True)) == {
        "observation": 1,
        "hidden_state": 1,
        "prior_vector": 1,
    }
    adjacency = figures["Adjacency Matrix"].data[0]
    assert list(adjacency.x) == list(adjacency.y) == ["sensor", "belief", "prior"]
    np.testing.assert_array_equal(adjacency.z, [[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    traces = figures["Interactive 3D Structure"].data
    nodes = traces[0]
    assert list(nodes.text) == ["sensor", "belief", "prior"]
    coords = np.column_stack([nodes.x, nodes.y, nodes.z])
    assert len(traces) == 3
    for trace, indices in zip(traces[1:], [[0, 1], [1, 2]], strict=True):
        np.testing.assert_array_equal(
            np.column_stack([trace.x, trace.y, trace.z]), coords[indices]
        )


def test_public_network_with_no_edges_reports_no_artifact_instead_of_success(tmp_path):
    _, parsed, inventory = saved_model(tmp_path, [])
    result, output, summary = run_saved_model(tmp_path, parsed, inventory, "network")
    assert result == 2 and result is not True
    assert summary["output_files"] == []
    assert summary["skipped"] == 1 and summary["successful"] == 0
    assert summary["attempts"][0]["error_message"] == "Insufficient network data"
    assert not list(output.glob("*.png"))


def test_public_3d_network_rejects_unusable_saved_variables_without_artifact(tmp_path):
    _, parsed, inventory = saved_model(tmp_path, [], variables=[])
    result, output, summary = run_saved_model(tmp_path, parsed, inventory, "3d")
    assert result is False
    assert summary["failed"] == 1 and summary["successful"] == 0
    assert "Data validation failed" in summary["attempts"][0]["error_message"]
    assert summary["output_files"] == [] and not list(output.glob("*.png"))

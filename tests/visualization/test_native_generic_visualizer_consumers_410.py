"""Public generic visualization saves actual graphs, matrix CSVs and reader summaries."""

import csv
import logging
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from gnn.visualization import (
    GNNVisualizer,
    generate_graph_visualization,
    generate_matrix_visualization,
    generate_visualizations,
)
from tests.visualization.test_native_matrix_views_410 import (
    figures as figures,
)
from tests.visualization.test_native_matrix_views_410 import (
    native_png,
)


@pytest.fixture
def owned_cwd(tmp_path, monkeypatch):
    # One-shot wrappers use cwd.parent for their default workspace output.
    directory = tmp_path / "workspace" / "src"
    directory.mkdir(parents=True)
    monkeypatch.chdir(directory)
    return directory


def test_public_connection_graph_saves_actual_directed_and_bidirectional_edges(
    tmp_path, owned_cwd, monkeypatch, figures
):
    data = {
        "Edges": [
            {
                "source": "observation",
                "target": "state",
                "directed": True,
                "constraint": "likelihood",
                "comment": "sensor Ω",
            },
            {"source": "state", "target": "policy", "directed": False},
            {"source": "", "target": "ignored", "directed": True},
        ]
    }
    before = pickle.dumps(data, protocol=5)
    captured = []
    original = nx.draw_networkx_edges

    def record(graph, *args, **kwargs):
        captured.append(graph.copy())
        return original(graph, *args, **kwargs)

    monkeypatch.setattr(nx, "draw_networkx_edges", record)
    output = tmp_path / "graph"
    output.mkdir()
    assert generate_graph_visualization(data, str(output / "requested.png"))
    path = output / "connections.png"
    native_png(path)
    assert len(captured) == 1
    graph = captured[0]
    assert set(graph.nodes) == {"observation", "state", "policy"}
    assert set(graph.edges) == {
        ("observation", "state"),
        ("state", "policy"),
        ("policy", "state"),
    }
    assert graph["observation"]["state"] == {
        "constraint": "likelihood",
        "comment": "sensor Ω",
    }
    text = {item.get_text() for item in figures[path].axes[0].texts}
    assert {"observation", "state", "policy", "likelihood"} <= text
    assert pickle.dumps(data, protocol=5) == before


@pytest.mark.parametrize("data", [{}, {"Edges": []}])
def test_public_graph_empty_input_preserves_no_graph_artifact_contract(
    tmp_path, owned_cwd, data
):
    assert generate_graph_visualization(data, str(tmp_path / "requested.png"))
    assert not (tmp_path / "connections.png").exists()
    assert not (tmp_path / "connections_error.png").exists()


def test_public_graph_invalid_endpoints_save_explicit_empty_graph(
    tmp_path, owned_cwd, figures
):
    data = {"Edges": [{"source": "", "target": "state"}]}
    assert generate_graph_visualization(data, str(tmp_path / "requested.png"))
    path = tmp_path / "connections.png"
    native_png(path)
    assert [item.get_text() for item in figures[path].axes[0].texts] == [
        "No connections found"
    ]


@pytest.mark.parametrize(
    "method", ["generate_graph_visualization", "create_network_diagram"]
)
def test_public_node_grid_retains_labels_and_declared_connection_count(
    tmp_path, method, figures
):
    data = {
        "variables": [{"name": "observation Ω"}, "state", {"name": "policy"}],
        "connections": [{"source": "observation Ω", "target": "state"}],
    }
    before = pickle.dumps(data, protocol=5)
    result = getattr(GNNVisualizer(output_dir=str(tmp_path)), method)(data)
    assert result == {"status": "SUCCESS", "output_dir": str(tmp_path / "graph")}
    path = tmp_path / "graph" / "graph.png"
    native_png(path)
    axis = figures[path].axes[0]
    assert {item.get_text() for item in axis.texts} == {
        "observation Ω",
        "state",
        "policy",
    }
    assert axis.get_title() == "Graph Visualization (3 variables, 1 connections)"
    assert pickle.dumps(data, protocol=5) == before


def test_public_generic_matrix_wrapper_exports_exact_authored_values_and_statistics(
    tmp_path, owned_cwd, figures
):
    data = {"InitialParameterization": "A={(0.8,0.1),(0.2,0.9)}\nC={-2.0,0.75}"}
    before = pickle.dumps(data, protocol=5)
    assert generate_matrix_visualization(data, str(tmp_path / "requested.png"))
    native_png(tmp_path / "matrix_analysis.png")
    native_png(tmp_path / "matrix_statistics.png")
    csvs = sorted(tmp_path.glob("matrix_analysis_matrix_*.csv"))
    assert len(csvs) == 2
    expected = {"A": [[0.8, 0.1], [0.2, 0.9]], "C": [-2.0, 0.75]}
    for path in csvs:
        with path.open(newline="") as stream:
            rows = list(csv.reader(stream))
        name = rows[0][0].removeprefix("Matrix: ")
        values = (
            [[float(value) for value in row[1:]] for row in rows[5:]]
            if name == "A"
            else [float(value) for value in rows[5]]
        )
        np.testing.assert_allclose(values, expected[name])
    axes = {
        axis.get_title(): axis
        for axis in figures[tmp_path / "matrix_statistics.png"].axes
    }
    assert [bar.get_height() for bar in axes["Matrix Sizes"].patches] == [4, 2]
    assert [bar.get_height() for bar in axes["Matrix Means"].patches] == pytest.approx(
        [0.5, -0.625]
    )
    assert pickle.dumps(data, protocol=5) == before


@pytest.mark.parametrize("extension", ["md", "csv"])
def test_public_file_visualization_summary_comes_from_actual_markdown_and_csv_reader(
    tmp_path, extension
):
    if extension == "md":
        content = "# Authored sensor Ω\n## ModelName\nAuthored sensor Ω\n## StateSpaceBlock\nstate[2,type=float]\nobservation[3,type=float]\n## Connections\nobservation > state\n## InitialParameterization\nA={(0.7,0.1),(0.2,0.3),(0.1,0.6)}\n"
    else:
        content = '# Authored sensor Ω\nGNNSection,Content\nStateSpaceBlock,"state[2,type=float]\nobservation[3,type=float]"\nConnections,"observation > state"\nInitialParameterization,"A={(0.7,0.1),(0.2,0.3),(0.1,0.6)}"\n'
    path = tmp_path / f"authored.{extension}"
    path.write_text(content, encoding="utf-8")
    before = path.read_bytes()
    visualizer = GNNVisualizer(output_dir=str(tmp_path / "outputs"))
    output = Path(visualizer.visualize_file(str(path)))
    assert output == tmp_path / "outputs" / "authored"
    summary = (output / "visualization_summary.txt").read_text()
    assert "Visualization Summary for authored" in summary
    assert (
        "Authored sensor Ω" in summary
        and "'state'" in summary
        and "'observation'" in summary
    )
    assert "'source': 'observation'" in summary and "'target': 'state'" in summary
    assert "'dimensions': [2]" in summary and "'dimensions': [3]" in summary
    assert (
        "parser: ✓ Available" in (output / "visualization_capabilities.txt").read_text()
    )
    assert path.read_bytes() == before


def test_public_missing_file_summary_records_actual_reader_failure(tmp_path):
    source = tmp_path / "missing.md"
    output = Path(
        GNNVisualizer(output_dir=str(tmp_path / "outputs")).visualize_file(str(source))
    )
    summary = (output / "visualization_summary.txt").read_text()
    assert "Parser failed" in summary and "missing.md" in summary
    assert "error:" in summary and "No such file" in summary
    assert not source.exists()


def test_public_directory_summary_keeps_each_authored_source_separate(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    for name in ["east", "west"]:
        (sources / f"{name}.md").write_text(
            f"## ModelName\n{name} sensor\n## StateSpaceBlock\nstate[2,type=float]\n"
        )
    before = {path: path.read_bytes() for path in sources.iterdir()}
    output = tmp_path / "reports"
    assert GNNVisualizer(output_dir=str(output)).visualize_directory(
        str(sources)
    ) == str(output)
    for name in ["east", "west"]:
        text = (output / name / "visualization_summary.txt").read_text()
        assert f"{name} sensor" in text
    assert {path: path.read_bytes() for path in before} == before


def test_public_batch_empty_directory_returns_false_with_native_output_layout(tmp_path):
    source = tmp_path / "sources"
    source.mkdir()
    output = tmp_path / "outputs"
    assert not generate_visualizations(
        logging.getLogger("native-generic-consumer"), source, output
    )
    assert (output / "8_visualization_output").is_dir()


def test_public_empty_node_grid_saves_visible_absence_message(tmp_path, figures):
    assert (
        GNNVisualizer(output_dir=str(tmp_path)).generate_graph_visualization()["status"]
        == "SUCCESS"
    )
    path = tmp_path / "graph" / "graph.png"
    native_png(path)
    axis = figures[path].axes[0]
    assert axis.get_title() == "Graph Visualization (empty)"
    assert [item.get_text() for item in axis.texts] == ["No graph data provided"]


def test_public_file_report_has_real_occupied_destination_error_context(tmp_path):
    source = tmp_path / "authored.md"
    source.write_text("## ModelName\nAuthored source Ω\n")
    before = source.read_bytes()
    output = tmp_path / "reports"
    output.mkdir()
    occupied = output / "authored"
    occupied.write_bytes(b"occupied original artifact\n")
    result = GNNVisualizer(output_dir=str(output)).visualize_file(str(source))
    assert result == str(output / "authored_error")
    text = (Path(result) / "visualization_error.txt").read_text()
    assert str(source) in text and str(occupied) in text and "File exists" in text
    assert occupied.read_bytes() == b"occupied original artifact\n"
    assert source.read_bytes() == before


@pytest.mark.parametrize("recursive", [False, True])
def test_public_batch_writes_selected_source_summaries_without_rewriting_sources(
    tmp_path, recursive
):
    source = tmp_path / "sources"
    nested = source / "nested"
    nested.mkdir(parents=True)
    paths = [source / "east.md", nested / "west.md"]
    for path in paths:
        path.write_text(
            f"## ModelName\n{path.stem} authored sensor Ω\n## StateSpaceBlock\nstate[2,type=float]\n"
        )
    before = {path: path.read_bytes() for path in paths}
    output = tmp_path / "outputs"
    assert generate_visualizations(
        logging.getLogger("native-generic-consumer"),
        source,
        output,
        recursive=recursive,
    )
    reports = output / "8_visualization_output"
    assert (
        "east authored sensor Ω"
        in (reports / "east" / "visualization_summary.txt").read_text()
    )
    if recursive:
        assert (
            "west authored sensor Ω"
            in (reports / "west" / "visualization_summary.txt").read_text()
        )
    else:
        assert not (reports / "west").exists()
    assert {path: path.read_bytes() for path in before} == before

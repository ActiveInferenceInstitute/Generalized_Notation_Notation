"""Native raw-GNN consumers, using the default parser and real saved artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure
from PIL import Image

from gnn.advanced_visualization import AdvancedVisualizer, VisualizationDataExtractor

RAW_GNN = """# Finite signed diagnostic
## GNNSection
ActInfPOMDP
## GNNVersionAndFlags
GNN v1
## ModelName
Finite signed diagnostic
## StateSpaceBlock
s[2,type=float]
W[2,2,type=float]
o[2,type=int]
## Connections
s>W
W>o
## InitialParameterization
W = [[-2.0, 0.0], [3.0, 7.0]]
## Time
Static
## ActInfOntologyAnnotation
s=HiddenState
o=Observation
"""


@pytest.fixture
def native_figures(monkeypatch):
    observed = {}
    original = Figure.savefig

    def forward(figure, filename, *args, **kwargs):
        result = original(figure, filename, *args, **kwargs)
        observed[Path(filename).name] = figure
        return result

    monkeypatch.setattr(Figure, "savefig", forward)
    return observed


def _png(path):
    with Image.open(path) as image:
        image.load()
        assert image.format == "PNG"
        assert image.width > 100 and image.height > 100
        assert image.getbbox() is not None


def test_default_raw_gnn_consumer_preserves_counts_edges_signed_values_and_links(
    tmp_path, native_figures
):
    source = tmp_path / "authored.md"
    source.write_text(RAW_GNN)
    original = source.read_bytes()
    output = tmp_path / "artifacts"
    files = AdvancedVisualizer().generate_visualizations(
        source.read_text(), "finite", output, interactive=False
    )
    model = output / "finite"
    expected = {
        "finite_statistics.png",
        "finite_network.png",
        "finite_heatmap.png",
        "finite_advanced_summary.html",
        "finite_advanced_viz_manifest.json",
    }
    assert {Path(item).name for item in files} == expected
    for name in expected:
        assert (model / name).is_file()
    for name in expected:
        if name.endswith(".png"):
            _png(model / name)

    statistics = native_figures["finite_statistics.png"].axes[0]
    assert [tick.get_text() for tick in statistics.get_xticklabels()] == [
        "hidden_state",
        "observation",
    ]
    assert [bar.get_height() for bar in statistics.patches] == [2, 1]

    network = native_figures["finite_network.png"].axes[0]
    labels = [label.get_text() for label in network.texts]
    assert labels == ["s", "W", "o"]
    points = {
        name: np.asarray(artist.get_offsets())[0]
        for name, artist in zip(labels, network.collections)
    }
    assert len(network.lines) == 2
    for line, (start, end) in zip(network.lines, [("s", "W"), ("W", "o")]):
        np.testing.assert_array_equal(
            np.column_stack(line.get_data()), np.stack([points[start], points[end]])
        )

    heatmap = native_figures["finite_heatmap.png"].axes[0]
    np.testing.assert_array_equal(
        heatmap.images[0].get_array(), [[-2.0, 0.0], [3.0, 7.0]]
    )
    with (model / "finite_heatmap_data.csv").open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[1] == ["Shape: (2, 2)"]
    assert rows[-2:] == [["Row 0", "-2.0", "0.0"], ["Row 1", "3.0", "7.0"]]
    html = (model / "finite_advanced_summary.html").read_text()
    for name in ["finite_statistics.png", "finite_network.png", "finite_heatmap.png"]:
        assert f"src='{name}'" in html
    manifest = json.loads((model / "finite_advanced_viz_manifest.json").read_text())
    assert manifest["model"] == "finite"
    assert {Path(item).name for item in manifest["generated"]} == expected - {
        "finite_advanced_viz_manifest.json"
    }
    assert all(Path(item).is_file() for item in manifest["generated"])
    assert source.read_bytes() == original


def test_raw_gnn_native_export_selection_omits_html_and_manifest(tmp_path):
    output = tmp_path / "artifacts"
    files = AdvancedVisualizer().generate_visualizations(
        RAW_GNN, "finite", output, export_formats=["png"], interactive=False
    )
    assert {Path(item).suffix for item in files} == {".png"}
    assert len(files) == 3
    for item in files:
        _png(Path(item))
    assert not list(output.rglob("*.html"))
    assert not list(output.rglob("*.json"))


@pytest.mark.parametrize("content", ["", '{"model_name":'])
def test_default_parser_rejection_produces_explicit_recovery_without_model_plots(
    tmp_path, content
):
    parsed = VisualizationDataExtractor().extract_from_content(content)
    assert parsed["success"] is False
    assert parsed["errors"]
    output = tmp_path / "artifacts"
    files = AdvancedVisualizer().generate_visualizations(content, "invalid", output)
    assert files == [str(output / "invalid" / "invalid_fallback_summary.html")]
    html = Path(files[0]).read_text()
    assert "invalid - Recovery Visualization" in html
    assert "Model Content Summary" in html
    assert not list(output.rglob("*.png"))
    assert not list(output.rglob("*.csv"))
    assert not list(output.rglob("*.json"))

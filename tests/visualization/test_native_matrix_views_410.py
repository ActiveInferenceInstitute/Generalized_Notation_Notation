"""Public matrix plotting consumers preserve signed values and scientific axes."""

import csv
from pathlib import Path

import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from PIL import Image

from gnn.visualization.matrix import MatrixVisualizer


@pytest.fixture
def figures(monkeypatch):
    saved = {}
    original = Figure.savefig

    def record(figure, filename, *args, **kwargs):
        saved[Path(filename)] = figure
        return original(figure, filename, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", record)
    with plt.rc_context():
        yield saved


def native_png(path):
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert image.width > 100 and image.height > 100
        image.verify()


def plot_values(axis):
    arrays = [np.asarray(image.get_array()) for image in axis.images]
    arrays.extend(
        np.asarray(collection.get_array())
        for collection in axis.collections
        if collection.get_array() is not None
    )
    assert len(arrays) == 1
    return arrays[0]


def test_public_single_heatmap_retains_signed_nondefault_matrix(tmp_path, figures):
    matrix = np.array([[-2.0, 0.25, 3.0], [1.5, -0.75, 4.0]])
    before = matrix.copy()
    path = tmp_path / "signed.png"
    assert MatrixVisualizer().generate_single_matrix_heatmap(
        matrix, "Calibration", path
    )
    native_png(path)
    axis = figures[path].axes[0]
    np.testing.assert_array_equal(plot_values(axis).reshape(matrix.shape), matrix)
    assert axis.get_title() == "Matrix: Calibration"
    np.testing.assert_array_equal(matrix, before)


def test_public_correlation_plot_has_hand_computed_column_correlations(
    tmp_path, figures
):
    # Column 1 is twice column 0; column 2 decreases linearly as column 0 grows.
    matrix = np.tile([[1.0, 2.0, 7.0], [2.0, 4.0, 5.0], [3.0, 6.0, 3.0]], (1, 4))
    before = matrix.copy()
    path = tmp_path / "correlation.png"
    assert MatrixVisualizer().generate_matrix_correlation_plot(matrix, "Sensors", path)
    native_png(path)
    axis = figures[path].axes[0]
    np.testing.assert_allclose(
        plot_values(axis).reshape(12, 12),
        np.tile([[1, 1, -1], [1, 1, -1], [-1, -1, 1]], (4, 4)),
        rtol=0,
        atol=1e-14,
    )
    assert axis.get_xlabel() == axis.get_ylabel() == "Variables"
    np.testing.assert_array_equal(matrix, before)


def test_public_histogram_preserves_empirical_mass_and_signed_range(tmp_path, figures):
    matrix = np.array([[-2.0, -2.0, 1.0], [1.0, 1.0, 4.0]])
    before = matrix.copy()
    path = tmp_path / "distribution.png"
    assert MatrixVisualizer().generate_matrix_histogram(matrix, "Signed Payoffs", path)
    native_png(path)
    axis = figures[path].axes[0]
    occupied = [bar for bar in axis.patches if bar.get_height() > 0]
    assert len(occupied) == 3
    assert [bar.get_height() * bar.get_width() for bar in occupied] == pytest.approx(
        [1 / 3, 1 / 2, 1 / 6]
    )
    assert min(bar.get_x() for bar in axis.patches) == pytest.approx(-2)
    assert max(bar.get_x() + bar.get_width() for bar in axis.patches) == pytest.approx(
        4
    )
    assert axis.get_ylabel() == "Density"
    np.testing.assert_array_equal(matrix, before)


@pytest.mark.parametrize("count", [1, 4])
def test_public_composed_view_keeps_each_matrix_identity_and_values(
    tmp_path, figures, count
):
    matrices = {
        f"sensor_{index}": np.arange(6).reshape(2, 3) + index * 10
        for index in range(count)
    }
    before = {name: matrix.copy() for name, matrix in matrices.items()}
    path = tmp_path / "composed.png"
    assert MatrixVisualizer().generate_matrix_composed_view(matrices, path)
    native_png(path)
    visible = [axis for axis in figures[path].axes if axis.get_visible()]
    assert len(visible) == count
    for axis, (name, matrix) in zip(visible, matrices.items(), strict=True):
        assert axis.get_title() == f"{name}\n(2×3)"
        np.testing.assert_array_equal(plot_values(axis).reshape(2, 3), matrix)
        np.testing.assert_array_equal(matrix, before[name])


def test_public_large_heatmap_preserves_displayed_source_values_without_mutation(
    tmp_path, figures
):
    matrix = np.arange(40 * 40, dtype=float).reshape(40, 40)
    before = matrix.copy()
    path = tmp_path / "sampled.png"
    assert MatrixVisualizer().generate_single_matrix_heatmap(
        matrix, "Large Sensor", path
    )
    native_png(path)
    shown = plot_values(figures[path].axes[0])
    assert shown.shape == (20, 20)
    np.testing.assert_array_equal(shown[[0, -1]][:, [0, -1]], [[0, 38], [1520, 1558]])
    np.testing.assert_array_equal(matrix, before)


def test_public_matrix_directory_exports_authored_values_and_separate_model_artifacts(
    tmp_path, figures
):
    source = tmp_path / "sources"
    source.mkdir()
    content = "## InitialParameterization\nA={(0.8,0.1),(0.2,0.9)}\nC={-2.0,0.75}\n## Footer\nend\n"
    authored = {
        "east": ([[0.8, 0.1], [0.2, 0.9]], [-2.0, 0.75]),
        "west": ([[0.3, 0.9], [0.7, 0.1]], [-0.75, 2.0]),
    }
    for name in authored:
        if name == "west":
            content = content.replace(
                "(0.8,0.1),(0.2,0.9)", "(0.3,0.9),(0.7,0.1)"
            ).replace("-2.0,0.75", "-0.75,2.0")
        (source / f"{name}.md").write_text(content)
    before = {path: path.read_bytes() for path in source.iterdir()}
    output = tmp_path / "artifacts"
    paths = MatrixVisualizer().visualize_directory(source, output)
    assert set(paths) == {
        str(output / name / artifact)
        for name in ["east", "west"]
        for artifact in ["matrix_analysis.png", "matrix_statistics.png"]
    }
    for path in paths:
        native_png(path)
    for name, (likelihood, preferences) in authored.items():
        csvs = sorted((output / name).glob("matrix_analysis_matrix_*.csv"))
        assert len(csvs) == 2
        with csvs[0].open(newline="") as stream:
            rows = list(csv.reader(stream))
        assert rows[0] == ["Matrix: A"]
        np.testing.assert_allclose(
            [[float(value) for value in row[1:]] for row in rows[5:]],
            likelihood,
        )
        with csvs[1].open(newline="") as stream:
            rows = list(csv.reader(stream))
        assert rows[0] == ["Matrix: C"]
        np.testing.assert_allclose([float(value) for value in rows[5]], preferences)
        stats = figures[output / name / "matrix_statistics.png"].axes
        by_title = {axis.get_title(): axis for axis in stats}
        assert [bar.get_height() for bar in by_title["Matrix Sizes"].patches] == [4, 2]
        assert [
            bar.get_height() for bar in by_title["Matrix Means"].patches
        ] == pytest.approx([0.5, sum(preferences) / 2])
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize(
    "method", ["generate_single_matrix_heatmap", "generate_matrix_histogram"]
)
def test_public_matrix_views_refuse_unrenderable_data_without_artifact(
    tmp_path, method, caplog
):
    matrix = np.ones((2, 2, 2)) if "heatmap" in method else np.array([[np.nan, np.inf]])
    path = tmp_path / "refused.png"
    result = getattr(MatrixVisualizer(), method)(
        matrix, "Invalid Authored Values", path
    )
    assert result is False and not path.exists()
    assert (
        "Error generating" in caplog.text and "Invalid Authored Values" in caplog.text
    )


def test_public_empty_composed_view_refuses_without_artifact(tmp_path):
    path = tmp_path / "empty.png"
    assert MatrixVisualizer().generate_matrix_composed_view({}, path) is False
    assert not path.exists()

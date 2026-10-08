"""Actual signed correlation heatmaps under the unchanged strict warning gate."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from PIL import Image

from gnn.visualization.matrix import MatrixVisualizer


@pytest.fixture
def saved_artists(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Inspect final artists at the external save seam, then save the real PNG."""
    saved = []
    savefig = Figure.savefig

    def capture(figure: Figure, *args, **kwargs):
        axis = figure.axes[0]
        meshes = [item for item in axis.collections if isinstance(item, QuadMesh)]
        artist = meshes[0] if meshes else axis.images[0]
        saved.append(
            {
                "array": np.asarray(artist.get_array()).copy(),
                "kind": "mesh" if meshes else "image",
                "clim": artist.get_clim(),
                "zero_position": artist.norm(0.0),
                "zero_color": artist.cmap(artist.norm(0.0)),
                "xlabel": axis.get_xlabel(),
                "ylabel": axis.get_ylabel(),
                "title": axis.get_title(),
                "annotations": [item.get_text() for item in axis.texts],
                "xticks": [item.get_text() for item in axis.get_xticklabels()],
                "yticks": [item.get_text() for item in axis.get_yticklabels()],
                "aspect": axis.get_aspect(),
            }
        )
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    return saved


@pytest.mark.parametrize(
    "matrix, expected",
    [
        (
            [[1.0, 2.0, 7.0], [2.0, 4.0, 5.0], [3.0, 6.0, 3.0]],
            [[1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]],
        ),
        (
            [[-1.0, -1.0, -2.0], [-1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 2.0]],
            [
                [1.0, 0.0, 1 / np.sqrt(2)],
                [0.0, 1.0, 1 / np.sqrt(2)],
                [1 / np.sqrt(2), 1 / np.sqrt(2), 1.0],
            ],
        ),
        (
            [[5.0, -2.0], [5.0, -2.0], [5.0, -2.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ),
        (
            [[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]],
            [[1.0, 0.0], [0.0, 0.0]],
        ),
        ([[0.25]], [[0.25]]),
        ([[-1.0], [0.0], [1.0]], [[-1.0], [0.0], [1.0]]),
    ],
    ids=[
        "signed-hand-correlations",
        "positive-and-zero-hand-correlations",
        "all-constant-columns",
        "one-constant-column",
        "existing-one-cell-passthrough",
        "existing-one-column-passthrough",
    ],
)
def test_native_small_correlation_preserves_data_center_annotations_and_source(
    tmp_path: Path, saved_artists: list[dict], matrix: list, expected: list
) -> None:
    data = np.array(matrix, dtype=float)
    original = data.copy()
    data.setflags(write=False)
    path = tmp_path / "correlation.png"
    assert MatrixVisualizer().generate_matrix_correlation_plot(
        data, "Authored Sensors", path
    )
    assert path.is_file() and path.stat().st_size > 0
    with Image.open(path) as image:
        assert image.format == "PNG" and image.width > 300 and image.height > 300
        image.load()
    np.testing.assert_array_equal(data, original)
    assert not data.flags.writeable
    (rendered,) = saved_artists
    expected_array = np.array(expected)
    np.testing.assert_allclose(
        rendered["array"].reshape(expected_array.shape), expected_array, atol=1e-14
    )
    assert rendered["kind"] == "mesh"
    limit = float(np.max(np.abs(expected_array))) or 1.0
    assert rendered["clim"] == (-limit, limit)
    assert rendered["zero_position"] == 0.5
    np.testing.assert_allclose(
        rendered["zero_color"], matplotlib.colormaps["coolwarm"](0.5), atol=1e-14
    )
    assert rendered["annotations"] == [f"{value:.2f}" for value in expected_array.flat]
    assert rendered["xlabel"] == rendered["ylabel"] == "Variables"
    assert rendered["title"] == "Correlation Matrix: Authored Sensors"
    assert rendered["xticks"] == [
        str(index) for index in range(expected_array.shape[1])
    ]
    assert rendered["yticks"] == [
        str(index) for index in range(expected_array.shape[0])
    ]
    assert rendered["aspect"] == 1.0


def test_native_large_correlation_keeps_existing_image_branch_and_signed_data(
    tmp_path: Path, saved_artists: list[dict]
) -> None:
    signs = np.array([1.0, -1.0] * 6)
    data = np.array([1.0, 2.0, 3.0])[:, None] * signs[None, :]
    original = data.copy()
    path = tmp_path / "large-correlation.png"
    assert MatrixVisualizer().generate_matrix_correlation_plot(
        data, "Large Sensors", path
    )
    (rendered,) = saved_artists
    assert rendered["kind"] == "image"
    np.testing.assert_allclose(rendered["array"], np.outer(signs, signs))
    assert rendered["clim"] == (-1.0, 1.0)
    assert rendered["annotations"] == []
    assert rendered["aspect"] == "auto"
    assert rendered["xlabel"] == rendered["ylabel"] == "Variables"
    np.testing.assert_array_equal(data, original)
    with Image.open(path) as image:
        image.load()
        assert image.width > 300 and image.height > 300

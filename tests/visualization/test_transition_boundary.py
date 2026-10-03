"""Numerical and image parity at the canonical transition-figure boundary."""

from pathlib import Path

import matplotlib.axes
import numpy as np
import pytest

from gnn.visualization.matrix.transition_analysis import (
    generate_pomdp_transition_analysis,
)
from gnn.visualization.matrix.visualizer import MatrixVisualizer, logger
from gnn.visualization.matrix_visualizer import (
    MatrixVisualizer as LegacyMatrixVisualizer,
)


def test_transition_boundary_preserves_canonical_axes_and_figure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Distinct columns/actions make an axis permutation observable.
    tensor = np.array([[[1.0, 0.2], [0.3, 0.6]], [[0.0, 0.8], [0.7, 0.4]]])
    bars = []
    original = matplotlib.axes.Axes.bar

    def record_bar(self, x, height, *args, **kwargs):
        bars.append(np.asarray(height).copy())
        return original(self, x, height, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "bar", record_bar)
    public_path, direct_path = tmp_path / "public.png", tmp_path / "direct.png"
    assert LegacyMatrixVisualizer is MatrixVisualizer
    assert MatrixVisualizer().generate_pomdp_transition_analysis(tensor, public_path)
    assert generate_pomdp_transition_analysis(tensor, direct_path, logger=logger)
    expected_entropy = [
        -(0.3 * np.log(0.3) + 0.7 * np.log(0.7)) / 2,
        -(0.2 * np.log(0.2) + 0.8 * np.log(0.8) + 0.6 * np.log(0.6) + 0.4 * np.log(0.4))
        / 2,
    ]
    for offset in (0, 3):
        np.testing.assert_allclose(bars[offset], expected_entropy)
        np.testing.assert_allclose(bars[offset + 1], [0.85, 0.7])
        np.testing.assert_allclose(bars[offset + 2], [1.5, 2.0])
    assert public_path.read_bytes() == direct_path.read_bytes()

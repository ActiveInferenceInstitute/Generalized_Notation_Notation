"""Shared checks that bar value labels are offset in points, not data units.

A label placed at ``bar.get_height() + 0.01`` (a data-unit offset) is
unbounded in pixels: its screen size scales with 1 / y-range, so near-zero
data pushes the label -- and a ``bbox_inches="tight"`` canvas -- arbitrarily
far away. A point offset (``xytext=(0, 3), textcoords="offset points"``) is a
fixed pixel gap at every data scale. These helpers measure that gap on live
figures, so a test can drive plotting code with degenerate and large-range
data and assert the same few-pixel gap in both.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.figure import Figure

OFFSET_POINTS = 3.0
MAX_PNG_SIDE = 20_000


@contextmanager
def figures_held_open(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep figures open past the code-under-test's ``plt.close`` calls."""
    real_close = plt.close
    real_close("all")
    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(plt, "close", lambda *args, **kwargs: None)
            yield
    finally:
        real_close("all")


def bar_label_gaps_px(fig: Figure) -> list[float]:
    """Pixel gap between each labelled bar's top and the bottom of its label."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    gaps: list[float] = []
    for ax in fig.axes:
        assert isinstance(ax, Axes)
        bars = [
            patch
            for container in ax.containers
            if isinstance(container, BarContainer)
            for patch in container
        ]
        texts = [t for t in ax.texts if t.get_text().strip()]
        if not texts:
            continue
        for bar in bars:
            extent = bar.get_window_extent(renderer)
            center_x = extent.x0 + extent.width / 2
            label = min(
                texts,
                key=lambda t: abs(
                    (
                        t.get_window_extent(renderer).x0
                        + t.get_window_extent(renderer).x1
                    )
                    / 2
                    - center_x
                ),
            )
            gaps.append(label.get_window_extent(renderer).y0 - extent.y1)
    return gaps


def assert_bar_labels_offset_in_points(fig: Figure) -> None:
    """Every bar label sits ``OFFSET_POINTS`` above its bar, in pixels."""
    expected = OFFSET_POINTS * fig.dpi / 72.0
    gaps = bar_label_gaps_px(fig)
    assert gaps, "no labelled bars found"
    for gap in gaps:
        assert gap == pytest.approx(expected, abs=0.75), (
            f"label gap {gap:.2f}px != {expected:.2f}px (3 pt): offset is in data units"
        )


def assert_png_bounded(path: Path) -> None:
    """The PNG exists and neither side exceeds ``MAX_PNG_SIDE`` pixels."""
    assert path.is_file(), f"{path} not written"
    height, width = mpimg.imread(path).shape[:2]
    assert 0 < width <= MAX_PNG_SIDE and 0 < height <= MAX_PNG_SIDE, (width, height)


def open_bar_figures() -> list[Figure]:
    """Every open figure that contains at least one bar container."""
    figures = [plt.figure(num) for num in plt.get_fignums()]
    found = [
        fig
        for fig in figures
        if any(isinstance(c, BarContainer) for ax in fig.axes for c in ax.containers)
    ]
    assert found, "no figure with bars was produced"
    return found

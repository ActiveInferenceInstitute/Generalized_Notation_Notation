#!/usr/bin/env python3
"""Bar value labels must be offset in points, not data units.

A data-unit offset (``bar.get_height() + 0.5``) is unbounded in pixels: its
screen size scales with 1 / y-range, which is the same bug class as the
18e9-px tight-bbox canvas fixed in the POMDP transition analysis. Each site is
driven with degenerate (minimal-range) and large-range data; the pixel gap
between the bar top and its label must be the fixed 3 pt in both, and the
saved PNG must exist with bounded dimensions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.figure import Figure

from gnn.analysis.jax.analyzer import (
    create_jax_visualizations,
    create_visualizations_from_structured_data,
)
from gnn.analysis.viz_plots import generate_action_analysis
from gnn.integration.meta_analysis.collector import SweepRecord
from gnn.integration.meta_analysis.visualizer import SweepVisualizer

OFFSET_POINTS = 3.0
MAX_PNG_SIDE = 20_000


@pytest.fixture
def kept_figures(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep figures open past the code-under-test's ``plt.close`` calls."""
    real_close = plt.close
    real_close("all")
    monkeypatch.setattr(plt, "close", lambda *args, **kwargs: None)
    yield
    monkeypatch.undo()
    real_close("all")


def _label_gaps_px(fig: Figure) -> list[float]:
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
        for bar in bars:
            top = bar.get_window_extent(renderer).y1
            center_x = bar.get_window_extent(renderer).x0 + (
                bar.get_window_extent(renderer).width / 2
            )
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
            gaps.append(label.get_window_extent(renderer).y0 - top)
    return gaps


def _assert_point_offset(fig: Figure) -> None:
    expected = OFFSET_POINTS * fig.dpi / 72.0
    gaps = _label_gaps_px(fig)
    assert gaps, "no labelled bars found"
    for gap in gaps:
        assert gap == pytest.approx(expected, abs=0.75), (
            f"label gap {gap:.2f}px != {expected:.2f}px (3 pt): offset is in data units"
        )


def _assert_bounded_png(path: Path) -> None:
    assert path.is_file(), f"{path} not written"
    height, width = mpimg.imread(path).shape[:2]
    assert 0 < width <= MAX_PNG_SIDE and 0 < height <= MAX_PNG_SIDE, (width, height)


def _bar_figure() -> Figure:
    """The open figure that contains bar containers."""
    for num in plt.get_fignums():
        fig = plt.figure(num)
        if any(isinstance(c, BarContainer) for ax in fig.axes for c in ax.containers):
            return fig
    raise AssertionError("no figure with bars was produced")


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
    _assert_bounded_png(out)
    _assert_point_offset(_bar_figure())


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
        _assert_bounded_png(png)
    _assert_point_offset(_bar_figure())


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
    _assert_bounded_png(dist)
    _assert_point_offset(_bar_figure())


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
    _assert_bounded_png(Path(path))
    _assert_point_offset(_bar_figure())

#!/usr/bin/env python3
"""
Matplotlib bootstrap, style constants, and shared plot helpers for GNN
meta-analysis sweep visualizations.

Extracted from ``integration.meta_analysis.visualizer``.
"""

from __future__ import annotations

import logging
from typing import Any, cast

# Deferred matplotlib import to avoid import-time side effects
_MPL_AVAILABLE = False
try:
    import os

    from gnn.utils.system_env.matplotlib_setup import apply_env_backend_if_set

    apply_env_backend_if_set()
    import matplotlib

    if not os.environ.get("MPLBACKEND"):
        matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    _MPL_AVAILABLE = True
except ImportError as exc:
    logging.getLogger(__name__).debug(
        "Matplotlib unavailable for meta-analysis visualizations: %s", exc
    )


# Style Constants — Curated for ultra-bold high-contrast scientific reports
_STYLE: dict[str, Any] = {
    "bg_color": "white",
    "axis_bg": "#FFFFFF",
    "text_color": "#212529",
    "grid_color": "#DEE2E6",
    "title_size": 22,
    "label_size": 18,
    "tick_size": 14,
    "line_width": 3.0,
    "marker_size": 10,
    "legend_size": 14,
    "annotation_size": 13,
    "watermark_size": 10,
}


# Color palette — Adjusted for white background visibility
_FRAMEWORK_COLORS: dict[str, Any] = {
    "pymdp": "#E63946",  # Vivid Red
    "jax": "#457B9D",  # Deep Blue-Gray
    "numpyro": "#1D3557",  # Navy
    "rxinfer": "#2A9D8F",  # Teal/Green
    "activeinference_jl": "#B8860B",  # Dark Goldenrod (visible on white)
    "discopy": "#9B59B6",  # Amethyst
    "bnlearn": "#D35400",  # Pumpkin
    "pytorch": "#EE4C2C",  # PyTorch Red
}


def _get_color(framework: str) -> str:
    """Return color."""
    return cast("str", _FRAMEWORK_COLORS.get(framework, "#AAAAAA"))


def _fmt_time(val: float) -> str:
    """Format a runtime value as human-readable string."""
    if val < 0.001:
        return "—"
    if val < 1.0:
        return f"{val * 1000:.0f}ms"
    if val < 60:
        return f"{val:.1f}s"
    if val < 3600:
        return f"{val / 60:.1f}m"
    return f"{val / 3600:.1f}h"


def _add_watermark(ax: plt.Axes) -> Any:
    """Add a small watermark to the plot for traceability."""
    import datetime

    from gnn import __version__

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    watermark = f"GNN Scaling Analysis | {timestamp} | v{__version__}"

    # Use figure-level text to avoid 2D/3D coordinate issues
    figure = ax.get_figure()
    if figure is None:
        return
    figure.text(
        0.99,
        0.01,
        watermark,
        color="#CED4DA",
        fontsize=_STYLE["watermark_size"],
        ha="right",
        va="bottom",
        alpha=0.6,
    )

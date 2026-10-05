#!/usr/bin/env python3
"""Deterministic thin-orchestrator generator for the GNN "Triple Play" figure.

Renders a conceptual diagram: a single GNN text specification fanning out to
three renderings — (1) human-readable text/Markdown, (2) graphical model
visualization, (3) executable cognitive model across simulation backends.

Real data (counts shown on the executable node, families/steps on captions) is
read from the deterministic producer's output at
``output/data/manuscript_variables.json``. No counts are hard-coded.

Headless (MPLBACKEND=Agg), deterministic (no timestamps/random state), saves a
>=150 DPI PNG to ``output/figures/gnn_triple_play.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, deterministic backend
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.manuscript_figure_tokens import load_tokens  # noqa: E402

OUT_PATH = PROJECT_ROOT / "output" / "figures" / "gnn_triple_play.png"


def load_variables() -> dict:
    """Load the producer output through the recorder that pins figure freshness.

    Every key read here is written into ``output/figures/figure_registry.json``
    by the figure build, so a committed PNG whose counts have fallen behind the
    token map fails the suite rather than shipping.
    """
    return load_tokens()


def draw_box(
    ax,
    xy,
    w,
    h,
    text,
    facecolor,
    edgecolor,
    fontsize=11,
    fontweight="bold",
    textcolor="white",
):
    """Draw a rounded box centered at xy with wrapped text."""
    x, y = xy
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.8,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color=textcolor,
        zorder=3,
        wrap=True,
    )


def draw_arrow(ax, start, end, color="#334155"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=20,
        linewidth=2.0,
        color=color,
        shrinkA=4,
        shrinkB=4,
        zorder=1,
    )
    ax.add_patch(arrow)


def main() -> Path:
    vars_ = load_variables()
    backend_count = vars_["GNN_BACKEND_COUNT"]
    backend_list = vars_["GNN_BACKEND_LIST"]
    family_count = vars_["GNN_FAMILY_COUNT"]
    step_range = vars_["GNN_STEP_RANGE"]
    step_count = vars_["GNN_STEP_COUNT"]
    exec_count = vars_["GNN_EXECUTABLE_BACKEND_COUNT"]
    exec_step = vars_["GNN_STEP_EXECUTE"]

    # Show a representative subset of backends on the executable node so the
    # text stays legible; full count comes from real data.
    backends = [b.strip() for b in backend_list.split(",")]
    backend_preview = ", ".join(backends[:3]) + f", +{len(backends) - 3} more"

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Source node: the single GNN text specification
    src_xy = (1.9, 3.5)
    draw_box(
        ax,
        src_xy,
        2.6,
        2.0,
        "GNN\nText Specification\n(single source of truth)",
        facecolor="#1e3a8a",
        edgecolor="#0f172a",
        fontsize=12,
    )

    # Three rendering targets
    text_xy = (7.6, 5.6)
    graph_xy = (7.6, 3.5)
    exec_xy = (7.6, 1.2)

    draw_box(
        ax,
        text_xy,
        4.0,
        1.5,
        "1. Human-Readable\nText / Markdown\nrendered documentation",
        facecolor="#0f766e",
        edgecolor="#0f172a",
        fontsize=11,
    )
    draw_box(
        ax,
        graph_xy,
        4.0,
        1.5,
        f"2. Graphical Model\nfactor-graph & matrix\nvisualizations ({family_count} families)",
        facecolor="#7c3aed",
        edgecolor="#0f172a",
        fontsize=11,
    )
    draw_box(
        ax,
        exec_xy,
        4.0,
        1.5,
        f"3. Executable Cognitive Model\n{exec_count}/{backend_count} targets: Step {exec_step} adapters\n{backend_preview}",
        facecolor="#b45309",
        edgecolor="#0f172a",
        fontsize=10,
    )

    # Fan-out arrows from source to each target
    draw_arrow(
        ax, (src_xy[0] + 1.35, src_xy[1] + 0.3), (text_xy[0] - 2.05, text_xy[1] - 0.3)
    )
    draw_arrow(ax, (src_xy[0] + 1.35, src_xy[1]), (graph_xy[0] - 2.05, graph_xy[1]))
    draw_arrow(
        ax, (src_xy[0] + 1.35, src_xy[1] - 0.3), (exec_xy[0] - 2.05, exec_xy[1] + 0.3)
    )

    ax.set_title(
        "The GNN Triple Play", fontsize=18, fontweight="bold", color="#0f172a", pad=14
    )
    ax.text(
        5.0,
        0.15,
        f"One specification, three renderings — driven through the {step_count}-step pipeline (steps {step_range}).",
        ha="center",
        va="center",
        fontsize=10,
        style="italic",
        color="#334155",
    )

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return OUT_PATH


if __name__ == "__main__":
    out = main()
    print(str(out))

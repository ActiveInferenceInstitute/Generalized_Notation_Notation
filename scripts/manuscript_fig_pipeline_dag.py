#!/usr/bin/env python3
"""Deterministic generator for the GNN 25-step pipeline DAG figure.

Thin orchestrator: reads the real step roster, phases, and required orchestration
prerequisites from ``src/gnn/STEP_INDEX.md`` (the master step table plus its
"Data Dependency Graph" mermaid block) and renders a directed graph in rowwise
topological order. No counts, names, or edges are hard-coded — everything is
parsed from the source-of-truth file.

Readability design: each step is a wide rounded-rectangle box (not a circle),
sized so the full ``N module_name`` label fits without truncation. Nodes are
wrapped across five columns in a deterministic topological order. Multiline
labels preserve every module name at a consistent font size in print.

Output: output/figures/gnn_pipeline_dag.png (>=150 DPI, headless).
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, deterministic
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
STEP_INDEX = REPO_ROOT / "src" / "gnn" / "STEP_INDEX.md"
OUT_PNG = REPO_ROOT / "output" / "figures" / "gnn_pipeline_dag.png"

# Phase -> color (stable, ordered for the legend).
PHASE_COLORS: dict[str, str] = {
    "Global": "#9CA3AF",  # gray
    "Core": "#2563EB",  # blue
    "Analysis": "#7C3AED",  # purple
    "Simulation": "#DC2626",  # red
    "Output": "#059669",  # green
}

# Box geometry in data coordinates (half-width / half-height).
BOX_HALF_W = 1.35
BOX_HALF_H = 0.65

# Character-based size hint; the final renderer bounds check rejects overflow.
LABEL_FIT_CHARS = 16
BASE_FONT = 11.0


def parse_steps(text: str) -> dict[int, dict[str, str]]:
    """Parse the master step table: step number -> {name, phase}."""
    steps: dict[int, dict[str, str]] = {}
    # Table rows look like: | 0 | [`0_template.py`](...) | [`template/`](...) | Global | ...
    row_re = re.compile(r"^\|\s*(\d+)\s*\|(.+)$")
    for line in text.splitlines():
        m = row_re.match(line.strip())
        if not m:
            continue
        step = int(m.group(1))
        cells = [c.strip() for c in m.group(2).split("|")]
        # cells: [Script, Module Dir, Phase, Purpose, ...]
        if len(cells) < 3:
            continue
        phase = cells[2]
        if phase not in PHASE_COLORS:
            continue  # skip legend/other tables that happen to start with a digit
        # Short label from the module dir cell, e.g. [`template/`](...) -> template
        mod = cells[1]
        mm = re.search(r"`([^`/]+)/?`", mod)
        name = mm.group(1).rstrip("/") if mm else f"step{step}"
        steps[step] = {"name": name, "phase": phase}
    return steps


def parse_dependencies(text: str, valid: set[int]) -> list[tuple[int, int]]:
    """Parse required step edges from the Data Dependency Graph mermaid block."""
    # Isolate the dependency-graph mermaid block (graph TD ... ).
    start = text.find("## Data Dependency Graph")
    block = text[start:] if start != -1 else text
    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    # Edges like: S3[3 GNN Parse] --> S5[5 Type Check]  or  S3 --> S4[4 Registry]
    # Node labels (the [...] block) are optional on either endpoint, and an
    # edge label |"optional enrichment"| may sit between the arrow and target.
    edge_re = re.compile(
        r"S(\d+)(?:\[[^\]]*\])?\s*-->(?:\s*\|[^|]*\|)?\s*S(\d+)(?:\[[^\]]*\])?"
    )
    for src, dst in edge_re.findall(block):
        a, b = int(src), int(dst)
        if a in valid and b in valid and (a, b) not in seen:
            edges.append((a, b))
            seen.add((a, b))
    return edges


def compute_layout(
    g: nx.DiGraph, steps: dict[int, dict[str, str]]
) -> dict[int, tuple[float, float]]:
    """Arrange a deterministic topological order in five columns for print."""
    order = (
        list(nx.lexicographical_topological_sort(g))
        if nx.is_directed_acyclic_graph(g)
        else sorted(steps)
    )
    return {
        node: ((index % 5) * 3.4, -(index // 5) * 2.0)
        for index, node in enumerate(order)
    }


def draw_box(ax: Axes, x: float, y: float, label: str, color: str) -> None:
    """Draw a rounded step node with a centered multiline module label."""
    box = FancyBboxPatch(
        (x - BOX_HALF_W, y - BOX_HALF_H),
        2 * BOX_HALF_W,
        2 * BOX_HALF_H,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=color,
        edgecolor="#1E293B",
        linewidth=1.1,
        zorder=3,
    )
    ax.add_patch(box)
    # Keep a readable font floor; the renderer rejects labels that still overflow.
    font = BASE_FONT
    longest_line = max(len(line) for line in label.splitlines())
    if longest_line > LABEL_FIT_CHARS:
        font = max(9.0, BASE_FONT * LABEL_FIT_CHARS / longest_line)
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=font,
        fontweight="bold",
        color="white",
        zorder=4,
    )


def main() -> None:
    text = STEP_INDEX.read_text(encoding="utf-8")
    steps = parse_steps(text)
    if not steps:
        raise SystemExit("No steps parsed from STEP_INDEX.md")
    edges = parse_dependencies(text, set(steps))

    g: nx.DiGraph = nx.DiGraph()
    for s, meta in steps.items():
        g.add_node(s, **meta)
    g.add_edges_from(edges)

    pos = compute_layout(g, steps)

    # Five-column rows keep full module labels readable in a manuscript column.
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    unit_in = 0.52
    fig_w = (max(xs) - min(xs)) * unit_in + 2.6
    fig_h = (max(ys) - min(ys)) * unit_in + 1.9
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # --- edges first (under the boxes) ------------------------------------
    for a, b in edges:
        xa, ya = pos[a]
        xb, yb = pos[b]
        # Connect same-row peers horizontally and later rows from bottom to top.
        if ya == yb:
            start = (xa + BOX_HALF_W, ya)
            end = (xb - BOX_HALF_W, yb)
        else:
            start = (xa, ya - BOX_HALF_H)
            end = (xb, yb + BOX_HALF_H)
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            connectionstyle="arc3,rad=0.06",
            color="#94A3B8",
            linewidth=1.1,
            zorder=1,
        )
        ax.add_patch(arrow)

    # --- boxes ------------------------------------------------------------
    for n in sorted(steps):
        x, y = pos[n]
        name = steps[n]["name"].replace("_", "_\n")
        label = f"{n}\n{name}"
        draw_box(ax, x, y, label, PHASE_COLORS[steps[n]["phase"]])

    # --- frame, legend, title ---------------------------------------------
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - BOX_HALF_W - 0.6, max(xs) + BOX_HALF_W + 0.6)
    ax.set_ylim(min(ys) - BOX_HALF_H - 0.8, max(ys) + BOX_HALF_H + 1.0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    legend_handles = [mpatches.Patch(color=c, label=p) for p, c in PHASE_COLORS.items()]
    ax.legend(
        handles=legend_handles,
        title="Phase",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(PHASE_COLORS),
        frameon=False,
        fontsize=10,
        title_fontsize=10.5,
    )

    n_steps = len(steps)
    ax.set_title(
        f"GNN {n_steps}-Step Processing Pipeline",
        fontsize=20,
        fontweight="bold",
        pad=65,
    )
    ax.text(
        0.5,
        -0.02,
        f"{n_steps} steps ({min(steps)}–{max(steps)}) · "
        f"{len(edges)} required step prerequisites · "
        "rowwise topological order; colored by execution phase",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color="#475569",
    )

    # Measure the actual renderer instead of assuming character counts imply fit.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for box, label in zip(ax.patches[-n_steps:], ax.texts[:n_steps], strict=True):
        bounds = box.get_window_extent(renderer)
        text_bounds = label.get_window_extent(renderer)
        if not (
            bounds.x0 <= text_bounds.x0 <= text_bounds.x1 <= bounds.x1
            and bounds.y0 <= text_bounds.y0 <= text_bounds.y1 <= bounds.y1
        ):
            plt.close(fig)
            raise ValueError(f"Pipeline label overflows its node: {label.get_text()!r}")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(str(OUT_PNG))


if __name__ == "__main__":
    main()

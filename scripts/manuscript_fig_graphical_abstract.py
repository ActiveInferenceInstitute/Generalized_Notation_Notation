#!/usr/bin/env python3
"""Deterministic generator for the manuscript's graphical abstract (cover figure).

One comprehensive panel summarizing GNN end-to-end: a plain-text specification
enters the numbered pipeline (parse, type-check + validate with the B-tensor
orientation diagnostic, render, execute, analyze) and the resulting artifacts
flow into the cross-repository interchange checks. Every count, step number and
the version string are read from the deterministic producer's output through
``scripts.lib.manuscript_figure_tokens.load_tokens``, so the panel is a second
renderer of the same token map the prose renders and the freshness suite can
fail it when a count moves. No counts, step numbers, or backend facts are
hard-coded here; the stage structure (which step a stage maps to) is authored,
exactly like the DAG figure authors its phase coloring.

Layout is measured, not guessed. The axes span the figure exactly
(``subplots_adjust(0, 0, 1, 1)``), so one data unit is one figure inch and
every wrap/shrink threshold below compares like with like; every string is
wrapped or shrunk against the actual renderer's text extents, and a final
no-clip pass asserts that no text artist reaches within the padding of its
card before the PNG is written — the generator raises instead of shipping a
clipped or colliding label.

Headless (matplotlib "Agg" before pyplot import), deterministic (fixed layout,
no timestamps or random state), saves a 300 DPI PNG to
``output/figures/gnn_graphical_abstract.png`` and prints exactly that path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, deterministic backend
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.manuscript_figure_tokens import load_tokens  # noqa: E402

OUT_PATH = PROJECT_ROOT / "output" / "figures" / "gnn_graphical_abstract.png"

# Stage palette — the same hue families the DAG and Triple Play figures use, so
# the cover panel reads as part of one figure system: navy for the document,
# the DAG's Core blue for parse/validate, Simulation red for render/execute,
# Analysis purple, Output green for interchange.
_NAVY = "#1E3A8A"
_BLUE = "#2563EB"
_RED = "#DC2626"
_PURPLE = "#7C3AED"
_GREEN = "#059669"
_INK = "#0F172A"
_SLATE = "#334155"
_EDGE = "#1E293B"

# (title, color, pill token key, body text). The pill key selects which
# producer step token supplies the number; body text is wrapped to the card
# width at draw time.
_STAGE_SPEC: list[dict[str, object]] = [
    {
        "title": "GNN document",
        "color": _NAVY,
        "pill": "the source",
        "body": "plain-text spec declaring A, B, C, D, E",
    },
    {
        "title": "Parse",
        "color": _BLUE,
        "pill": "GNN_STEP_GNN",
        "body": "model discovery; typed export",
    },
    {
        "title": "Validate",
        "color": _BLUE,
        "pill": "GNN_STEP_TYPE_CHECKER:GNN_STEP_VALIDATION",
        "body": "type, schema, and B-orientation checks",
    },
    {
        "title": "Render",
        "color": _RED,
        "pill": "GNN_STEP_RENDER",
        "body": "backend-specific code generation",
    },
    {
        "title": "Execute",
        "color": _RED,
        "pill": "GNN_STEP_EXECUTE",
        "body": "supervised backend dispatch",
    },
    {
        "title": "Analyze",
        "color": _PURPLE,
        "pill": "GNN_STEP_ANALYSIS",
        "body": "aggregation; reports",
    },
    {
        "title": "Interchange",
        "color": _GREEN,
        "pill": "cross-repo",
        "body": "GEO-INFER, fep_lean pinned-pair checks",
    },
]

# Relative card widths (sum + gaps + margins = the x span below). Data units
# are figure inches (the axes span the figure exactly), so these are inches.
_CARD_WIDTHS = [1.35, 1.15, 1.45, 1.42, 1.42, 1.18, 1.82]
_GAP = 0.24
_X_MARGIN = 0.03
_X_SPAN = sum(_CARD_WIDTHS) + _GAP * (len(_CARD_WIDTHS) - 1) + 2 * _X_MARGIN

# Vertical bands: title / subtitle / cards / callout strip / count strip.
_Y_SPAN = 4.9
_TITLE_Y = 4.58
_SUBTITLE_Y = 4.24
_CARD_TOP = 3.66
_CARD_BOTTOM = 2.10
_HEADER_H = 0.40
_CALLOUT_TOP_Y = 1.72
_CALLOUT_DY = 0.27
_CALLOUT_MAX_LINES = 3
_RULE_Y = 0.94
_COUNTS_Y = 0.55

_BODY_FONT = 8.6
_HEADER_FONT = 10.5
_HEADER_MIN_FONT = 8.5
_PILL_FONT = 8.2

# Minimum clearance (inches) the no-clip pass demands between any text artist
# and the horizontal bounds it was placed inside.
_CLIP_PAD = 0.045


class _ClipCheck:
    """Registry of (artist, xmin, xmax) bounds each text must stay within."""

    def __init__(self) -> None:
        self._items: list[tuple[object, float, float, str]] = []

    def add(self, artist: object, xmin: float, xmax: float, what: str) -> None:
        self._items.append((artist, xmin, xmax, what))

    def verify(self, fig: Figure, ax: Axes) -> None:
        """Raise unless every registered text clears its bounds by _CLIP_PAD."""
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        problems: list[str] = []
        for artist, xmin, xmax, what in self._items:
            ext = artist.get_window_extent(renderer=renderer)  # type: ignore[union-attr]
            left = ax.transData.transform((xmin, 0.0))[0]
            right = ax.transData.transform((xmax, 0.0))[0]
            pad_px = _CLIP_PAD * fig.dpi
            if ext.x0 < left + pad_px or ext.x1 > right - pad_px:
                problems.append(
                    f"{what}: text spans {ext.x0 / fig.dpi:.3f}-{ext.x1 / fig.dpi:.3f}in, "
                    f"bounds {xmin:.3f}-{xmax:.3f}in (pad {_CLIP_PAD}in)"
                )
        if problems:
            raise ValueError(
                "graphical abstract would ship clipped text:\n" + "\n".join(problems)
            )


def _text_width_in(fig: Figure, ax: Axes, s: str, fontsize: float) -> float:
    """Measured width of *s* at *fontsize* in figure inches (renderer-based)."""
    probe = ax.text(0, 0, s, fontsize=fontsize, fontweight="bold")
    fig.canvas.draw()
    try:
        renderer = fig.canvas.get_renderer()
        width = probe.get_window_extent(renderer=renderer).width / fig.dpi
    finally:
        probe.remove()
    return width


def _wrap_to_width(
    fig: Figure, ax: Axes, text: str, fontsize: float, max_width_in: float
) -> list[str]:
    """Greedy word wrap against measured text extents.

    Raises when a single word cannot fit — the layout must be adjusted rather
    than shipping a clipped label.
    """
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if _text_width_in(fig, ax, candidate, fontsize) <= max_width_in:
            current = candidate
            continue
        if not current:
            raise ValueError(
                f"word {word!r} at {fontsize}pt is wider "
                f"({_text_width_in(fig, ax, word, fontsize):.2f}in) than the "
                f"{max_width_in:.2f}in box it must sit in"
            )
        lines.append(current)
        current = word
    if current:
        lines.append(current)
    return lines


def _pill_text(tokens: dict[str, str], key: object) -> str | None:
    """Step pill for a stage: 'Step N' / 'Steps N–M', or an authored label."""
    if not key:
        return None
    keys = str(key).split(":")
    if len(keys) == 1 and keys[0].startswith("GNN_STEP"):
        return f"Step {tokens[keys[0]]}"
    if len(keys) == 2:
        return f"Steps {tokens[keys[0]]}\u2013{tokens[keys[1]]}"
    return keys[0]


def _draw_card(
    fig: Figure,
    ax: Axes,
    clip: _ClipCheck,
    x: float,
    w: float,
    title: str,
    color: str,
    pill: str | None,
    body: str,
) -> None:
    """One stage card: colored header strip, step pill, wrapped centered body."""
    body_h = _CARD_TOP - _CARD_BOTTOM
    ax.add_patch(
        FancyBboxPatch(
            (x, _CARD_BOTTOM),
            w,
            body_h,
            boxstyle="round,pad=0.0,rounding_size=0.07",
            facecolor="white",
            edgecolor=_EDGE,
            linewidth=1.2,
            zorder=2,
        )
    )
    ax.add_patch(
        Rectangle(
            (x, _CARD_TOP - _HEADER_H),
            w,
            _HEADER_H,
            facecolor=color,
            edgecolor=_EDGE,
            linewidth=1.2,
            zorder=3,
        )
    )
    # Shrink the header font until the title fits with side padding.
    header_fs = _HEADER_FONT
    while (
        _text_width_in(fig, ax, title, header_fs) > w - 0.30
        and header_fs > _HEADER_MIN_FONT
    ):
        header_fs -= 0.5
    header_text = ax.text(
        x + w / 2,
        _CARD_TOP - _HEADER_H / 2,
        title,
        ha="center",
        va="center",
        fontsize=header_fs,
        fontweight="bold",
        color="white",
        zorder=4,
    )
    clip.add(header_text, x, x + w, f"header {title!r}")
    inner_w = w - 0.24
    body_top = _CARD_TOP - _HEADER_H - 0.28
    if pill:
        pill_w = _text_width_in(fig, ax, pill, _PILL_FONT) + 0.18
        ax.add_patch(
            FancyBboxPatch(
                (x + w / 2 - pill_w / 2, _CARD_TOP - _HEADER_H - 0.32),
                pill_w,
                0.26,
                boxstyle="round,pad=0.0,rounding_size=0.11",
                facecolor="#EEF2F7",
                edgecolor="#CBD5E1",
                linewidth=0.8,
                zorder=4,
            )
        )
        pill_text = ax.text(
            x + w / 2,
            _CARD_TOP - _HEADER_H - 0.19,
            pill,
            ha="center",
            va="center",
            fontsize=_PILL_FONT,
            fontweight="bold",
            color=color,
            zorder=5,
        )
        clip.add(pill_text, x, x + w, f"pill {title!r}")
        body_top = _CARD_TOP - _HEADER_H - 0.46
    lines = _wrap_to_width(fig, ax, body, _BODY_FONT, inner_w)
    if len(lines) > 3:
        raise ValueError(f"card {title!r} body wraps to {len(lines)} lines; max 3")
    first_y = body_top - 0.10
    for i, line in enumerate(lines):
        body_text = ax.text(
            x + w / 2,
            first_y - i * 0.235,
            line,
            ha="center",
            va="center",
            fontsize=_BODY_FONT,
            color=_SLATE,
            zorder=4,
        )
        clip.add(body_text, x, x + w, f"body {title!r} line {i}")


def main() -> Path:
    tokens = load_tokens()

    step_count = tokens["GNN_STEP_COUNT"]
    step_range = tokens["GNN_STEP_RANGE"]
    backend_count = tokens["GNN_BACKEND_COUNT"]
    exec_count = tokens["GNN_EXECUTABLE_BACKEND_COUNT"]
    family_count = tokens["GNN_FAMILY_COUNT"]
    mcp_count = tokens["GNN_MCP_TOOL_COUNT"]
    version = tokens["GNN_VERSION"]
    exec_step = tokens["GNN_STEP_EXECUTE"]

    fig, ax = plt.subplots(figsize=(11, 4.9))
    # Span the figure exactly: one data unit == one figure inch, so the width
    # budgets above compare like with like against measured text extents.
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, _X_SPAN)
    ax.set_ylim(0, _Y_SPAN)
    ax.axis("off")
    clip = _ClipCheck()

    title_text = ax.text(
        _X_SPAN / 2,
        _TITLE_Y,
        "Generalized Notation Notation at a Glance",
        ha="center",
        va="center",
        fontsize=19,
        fontweight="bold",
        color=_INK,
    )
    clip.add(title_text, 0.1, _X_SPAN - 0.1, "title")
    subtitle_text = ax.text(
        _X_SPAN / 2,
        _SUBTITLE_Y,
        f"GNN v{version} \u2014 one text specification, many traceable realizations",
        ha="center",
        va="center",
        fontsize=11,
        color="#475569",
        style="italic",
    )
    clip.add(subtitle_text, 0.1, _X_SPAN - 0.1, "subtitle")

    x = _X_MARGIN
    spans: list[tuple[float, float]] = []
    for spec, w in zip(_STAGE_SPEC, _CARD_WIDTHS):
        pill = _pill_text(tokens, spec["pill"])
        _draw_card(
            fig,
            ax,
            clip,
            x,
            w,
            str(spec["title"]),
            str(spec["color"]),
            pill,
            str(spec["body"]),
        )
        spans.append((x, x + w))
        x += w + _GAP

    # Left-to-right flow arrows between adjacent cards at header mid-height.
    arrow_y = _CARD_TOP - _HEADER_H / 2
    for i in range(len(spans) - 1):
        ax.add_patch(
            FancyArrowPatch(
                (spans[i][1] + 0.02, arrow_y),
                (spans[i + 1][0] - 0.02, arrow_y),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.8,
                color="#475569",
                shrinkA=0,
                shrinkB=0,
                zorder=1,
            )
        )

    # Callout strip: the two wave-2 capabilities, stated as one wrapped note.
    callout = (
        "New in validation \u2014 the B-tensor orientation check reads every B literal "
        "as column-stochastic B[s\u2032, s, a] (the pymdp convention); --transpose-b maps "
        "row-stochastic textbook literals onto the canonical order instead of failing "
        "them. New in execution \u2014 the bnlearn lane generates runnable Python from "
        "the same parsed model and skips with a recorded status when the bnlearn "
        "runtime is absent."
    )
    callout_lines = _wrap_to_width(fig, ax, callout, 9.3, _X_SPAN - 1.2)
    if len(callout_lines) > _CALLOUT_MAX_LINES:
        raise ValueError(
            f"callout wraps to {len(callout_lines)} lines; max {_CALLOUT_MAX_LINES}"
        )
    for i, line in enumerate(callout_lines):
        callout_text = ax.text(
            0.55,
            _CALLOUT_TOP_Y - i * _CALLOUT_DY,
            line,
            ha="left",
            va="center",
            fontsize=9.3,
            color=_SLATE,
        )
        clip.add(callout_text, 0.45, _X_SPAN - 0.05, f"callout line {i}")

    ax.add_patch(
        Rectangle(
            (_X_MARGIN, _RULE_Y),
            _X_SPAN - 2 * _X_MARGIN,
            0.012,
            facecolor="#CBD5E1",
            edgecolor="none",
            zorder=1,
        )
    )

    if exec_count == backend_count:
        exec_phrase = f"{exec_count} render-target adapters at Step {exec_step}"
    else:
        exec_phrase = (
            f"{exec_count}/{backend_count} render-target adapters at Step {exec_step}"
        )
    counts = (
        f"{step_count} pipeline steps ({step_range}) \u00b7 {family_count} model families "
        f"\u00b7 {backend_count} registered render backends, {exec_phrase} \u00b7 "
        f"{mcp_count} MCP tools"
    )
    counts_fs = 10.5
    while (
        _text_width_in(fig, ax, counts, counts_fs) > _X_SPAN - 0.4 and counts_fs > 8.0
    ):
        counts_fs -= 0.5
    counts_text = ax.text(
        _X_SPAN / 2,
        _COUNTS_Y,
        counts,
        ha="center",
        va="center",
        fontsize=counts_fs,
        fontweight="bold",
        color=_INK,
    )
    clip.add(counts_text, 0.1, _X_SPAN - 0.1, "counts strip")

    # Mechanical no-clip gate: nothing ships half a character outside its box.
    clip.verify(fig, ax)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, facecolor="white")
    plt.close(fig)
    print(str(OUT_PATH))
    return OUT_PATH


if __name__ == "__main__":
    print(str(main()))

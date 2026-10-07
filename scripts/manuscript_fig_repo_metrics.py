#!/usr/bin/env python3
"""Deterministic generator for the GNN repository-scale metrics figure.

Thin orchestrator: reads repository-scale counts STRICTLY from the deterministic
producer output (output/data/manuscript_variables.json) and renders a horizontal
bar chart. No counts are hard-coded; only the metric keys and human-readable
labels live here.

The token map is loaded through
``scripts.lib.manuscript_figure_tokens.load_tokens`` so that every value this
figure prints is recorded in ``output/figures/figure_registry.json``. That
record is what lets the suite fail a committed PNG whose numbers have fallen
behind the token map, instead of shipping a bar chart that contradicts the
prose beside it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Repo root = parent of this scripts/ directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lib.manuscript_figure_tokens import load_tokens  # noqa: E402

OUTPUT_PATH = REPO_ROOT / "output" / "figures" / "gnn_repo_metrics.png"

# (json_key, human-readable label) — order is rendering order (top to bottom).
#
# The label must name the set the token counts. GNN_BACKEND_COUNT is
# len(FRAMEWORK_REGISTRY) — the *registered* render targets;
# GNN_EXECUTABLE_BACKEND_COUNT is the subset with ``supports_execution``
# as declared by the current render registry; Lean is an execution-only
# bridge outside that counted set, exactly as the same
# document's tbl:backend_registry reports. Labelling both "Registered
# backends" would erase the registry/executor distinction; keep the two
# labels distinct so the sets stay comparable.
METRICS: list[tuple[str, str]] = [
    ("GNN_STEP_COUNT", "Pipeline steps"),
    ("GNN_FAMILY_COUNT", "Model families"),
    ("GNN_BACKEND_COUNT", "Registered backends"),
    ("GNN_EXECUTABLE_BACKEND_COUNT", "Render-target adapters"),
    ("GNN_MCP_TOOL_COUNT", "MCP tools"),
    ("GNN_SRC_PACKAGE_COUNT", "Source packages"),
    ("GNN_TEST_FILE_COUNT", "Test files"),
    ("GNN_EXAMPLE_COUNT", "Examples"),
    ("GNN_DOC_FILE_COUNT", "Documentation files"),
]


def main() -> None:
    data = load_tokens()

    labels = [label for _, label in METRICS]
    values = [int(data[key]) for key, _ in METRICS]
    version = str(data["GNN_VERSION"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Render top-to-bottom in declared order by reversing for the y-axis.
    y_positions = list(range(len(labels)))[::-1]

    fig, ax = plt.subplots(figsize=(9, 5.4))
    bars = ax.barh(y_positions, values, color="#2c7fb8", edgecolor="#0f3d5c")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=12.5)
    ax.set_xlabel("Count (log scale)", fontsize=12)
    ax.set_title(
        f"GeneralizedNotationNotation at a Glance (v{version})",
        fontsize=16,
        fontweight="bold",
    )

    # Log scale keeps the small bars (pipeline steps) and the large ones
    # (documentation files) legible in one frame.
    ax.set_xscale("log")
    ax.set_xlim(1, max(values) * 1.6)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() * 1.05,
            bar.get_y() + bar.get_height() / 2,
            f"{value}",
            va="center",
            ha="left",
            fontsize=12,
            fontweight="bold",
            color="#0f3d5c",
        )

    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200)
    plt.close(fig)

    print(str(OUTPUT_PATH))


if __name__ == "__main__":
    main()

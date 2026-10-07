#!/usr/bin/env python3
"""GIF animation generator for RxInfer simulation results.

Produces animated GIF files using matplotlib FuncAnimation showing
belief evolution, state tracking, actions, VFE convergence, and the
bayesian graphical model with node colors dynamically updating.

Publication-quality white minimal style. Uses the GNN Connections
section to draw the graphical model structure (nodes and edges).
"""

import base64
import colorsys
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
from matplotlib.patches import Circle

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize

from .analyzer import compute_per_factor_beliefs
from .animator import (
    _normalize_actions,
    _normalize_beliefs,
    _normalize_observations,
    _normalize_true_states,
    _normalize_vfe,
)

logger = logging.getLogger(__name__)


def _hue_palette(count: int) -> List[tuple[float, float, float]]:
    """Evenly spaced publication-style RGB colors for `count` categories."""
    span = max(count, 1)
    return [colorsys.hls_to_rgb(i / span, 0.5, 0.7) for i in range(span)]


def _normalize_efe_per_action(data: Dict[str, Any]) -> List[List[float]]:
    """Extract EFE-per-action as a list of rows (timestep x action)."""
    eaa = data.get("efe_per_action")
    if eaa is None:
        metrics = data.get("metrics")
        if isinstance(metrics, dict):
            eaa = metrics.get("efe_per_action")
    if not eaa:
        return []
    rows: List[List[float]] = []
    for row in eaa:
        if isinstance(row, (list, tuple)):
            rows.append([float(x) for x in row])
    return rows


def _normalize_policy_posterior(data: Dict[str, Any]) -> List[List[float]]:
    """Extract policy posterior as a list of rows (timestep x action prob)."""
    pp = data.get("policy_posterior")
    if pp is None:
        metrics = data.get("metrics")
        if isinstance(metrics, dict):
            pp = metrics.get("policy_posterior")
    if not pp:
        return []
    rows: List[List[float]] = []
    for row in pp:
        if isinstance(row, (list, tuple)):
            rows.append([float(x) for x in row])
    return rows


def _parse_gnn_connections(
    data: Dict[str, Any],
) -> tuple[dict[str, tuple[float, float]], list[tuple[str, str]]]:
    """Parse GNN connections into graph node positions and edges.

    Extracts the ``connections`` list from the embedded ``gnn_spec`` (a dict or
    JSON string), falling back to the base64-encoded spec in
    ``runtime_metadata.gnn_spec_b64`` when the inline field is absent. Nodes
    are laid out in a left-to-right temporal chain grouped by role (priors,
    states, observations, policy). If no connections parse, the standard POMDP
    graphical-model structure (D -> s -> s' -> o with A, B, C, u, G) is used.

    Args:
        data: The ``rxinfer_simulation_v1`` results dict. May carry the GNN
            spec as a dict under ``gnn_spec``, a JSON string under ``gnn_spec``,
            or base64-encoded under ``runtime_metadata.gnn_spec_b64``.

    Returns:
        A tuple of ``(node_positions, edges)`` where ``node_positions`` maps
        each node name to its ``(x, y)`` layout coordinate and ``edges`` is a
        list of ``(source, target)`` node-name pairs.
    """
    spec = data.get("gnn_spec", {})
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except Exception as exc:
            logger.warning("Failed to parse gnn_spec JSON: %s", exc)
            spec = {}

    connections_raw = spec.get("connections", [])
    if not connections_raw:
        # Fallback: use the connections from the GNN file content if embedded
        spec_b64 = data.get("runtime_metadata", {}).get("gnn_spec_b64", "")
        if not spec_b64:
            return {}, []
        try:
            spec = json.loads(base64.b64decode(spec_b64).decode("utf-8"))
            connections_raw = spec.get("connections", [])
        except Exception as exc:
            logger.warning("Failed to decode/parse base64 gnn_spec: %s", exc)
            return {}, []

    # Parse connections like "D>s", "s-A", "s>s_prime"
    edges: list[tuple[str, str]] = []
    nodes: set[str] = set()
    for conn in connections_raw:
        conn_str = str(conn).strip()
        # GNN uses > for directed, - for undirected
        for sep in [">", "-"]:
            if sep in conn_str:
                parts = conn_str.split(sep, 1)
                if len(parts) == 2:
                    src, tgt = parts[0].strip(), parts[1].strip()
                    edges.append((src, tgt))
                    nodes.add(src)
                    nodes.add(tgt)
                    break

    # If no connections parsed, try to build from standard POMDP structure
    if not nodes:
        # Standard POMDP nodes: D -> s -> s' -> o, A -> o, B -> s', C -> G, u -> s'
        nodes = {"D", "s", "s'", "o", "A", "B", "C", "u", "G"}
        edges = [
            ("D", "s"),
            ("s", "o"),
            ("A", "o"),
            ("s", "s'"),
            ("B", "s'"),
            ("u", "s'"),
            ("C", "G"),
            ("G", "u"),
        ]

    # Layout: left-to-right temporal chain
    # Group nodes by temporal role
    col0 = []  # priors/params
    col1 = []  # states
    col2 = []  # observations
    col3 = []  # policy/action

    for n in sorted(nodes):
        if n in ("D", "A", "B", "C", "E"):
            col0.append(n)
        elif n in ("s", "s'", "s_f0", "s_f1", "s_prime"):
            col1.append(n)
        elif n in ("o", "o_m0", "y"):
            col2.append(n)
        elif n in ("u", "pi", "G", "F"):
            col3.append(n)
        else:
            col1.append(n)

    positions: dict[str, tuple[float, float]] = {}
    for i, n in enumerate(col0):
        positions[n] = (0.0, 1.0 - i * 0.3)
    for i, n in enumerate(col1):
        positions[n] = (0.33, 1.0 - i * 0.3)
    for i, n in enumerate(col2):
        positions[n] = (0.66, 1.0 - i * 0.3)
    for i, n in enumerate(col3):
        positions[n] = (1.0, 1.0 - i * 0.3)

    return positions, edges


def _resolve_strategy_layout(data: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the graph-panel node layout via the render-side strategy (FP-8).

    Reads ``runtime_metadata.model_kind`` (defaulting to ``"flat"`` when
    absent, keeping old behavior for payloads written before the field
    existed) and asks the registered ``ModelStrategy`` for its
    ``generate_graph_layout(gnn_spec)``. Every registered strategy provides
    a layout natively.

    Raises:
        ValueError: When ``model_kind`` is not a known ``ModelKind`` value —
            loud, never silently absorbed.
    """
    from gnn.render.pomdp_contract import ModelKind
    from gnn.render.rxinfer.model_strategies import get_model_strategy

    kind_value = str((data.get("runtime_metadata") or {}).get("model_kind", "flat"))
    try:
        kind = ModelKind(kind_value)
    except ValueError as exc:
        raise ValueError(
            f"unknown model_kind {kind_value!r} in runtime_metadata; "
            f"expected one of {[member.value for member in ModelKind]}"
        ) from exc

    gnn_spec = data.get("gnn_spec", {})
    if isinstance(gnn_spec, str):
        try:
            gnn_spec = json.loads(gnn_spec)
        except (json.JSONDecodeError, ValueError):
            gnn_spec = {}
    if not isinstance(gnn_spec, dict):
        gnn_spec = {}

    layout = get_model_strategy(kind).generate_graph_layout(gnn_spec)
    return layout if isinstance(layout, dict) else {}


def _node_value(
    node_name: str,
    step: int,
    beliefs: List[List[float]],
    observations: List[int],
    actions: List[int],
    vfe: List[float],
) -> float:
    """Get the current 'value' (probability or intensity) for a node at a step."""
    if step < 0:
        return 0.0
    if node_name in ("s", "s_f0", "s_f1"):
        if step < len(beliefs):
            return float(max(beliefs[step]))  # max belief = confidence
        return 0.0
    elif node_name in ("s'", "s_prime"):
        if step > 0 and step - 1 < len(beliefs):
            return float(max(beliefs[step - 1]))
        return 0.0
    elif node_name in ("o", "o_m0", "y"):
        if step < len(observations):
            return float(observations[step]) / max(max(observations) + 1, 1)
        return 0.0
    elif node_name in ("u",):
        if step < len(actions):
            return float(actions[step]) / max(max(actions) + 1, 1)
        return 0.0
    elif node_name in ("D", "A", "B", "C", "E"):
        return 0.5  # static parameters
    elif node_name in ("G", "F", "pi"):
        if step < len(vfe):
            return min(float(vfe[step]) / 10.0, 1.0)
        return 0.0
    return 0.0


def _draw_graph_model(
    ax: Any,
    positions: Dict[str, tuple[float, float]],
    edges: List[tuple[str, str]],
    step: int,
    beliefs: List[List[float]],
    observations: List[int],
    actions: List[int],
    vfe: List[float],
    state_colors: List[tuple[float, float, float]],
) -> None:
    """Draw the bayesian graphical model on the given axes."""
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.15, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw edges
    for src, tgt in edges:
        if src in positions and tgt in positions:
            x1, y1 = positions[src]
            x2, y2 = positions[tgt]
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="->", color="#888", lw=1.2, shrinkA=18, shrinkB=18
                ),
            )

    # Draw nodes
    for name, (x, y) in positions.items():
        val = _node_value(name, step, beliefs, observations, actions, vfe)

        # Color: use belief confidence for state nodes, gray for params
        if name in ("s", "s'", "s_f0", "s_f1", "s_prime"):
            # Color by argmax state
            if step < len(beliefs) and name in ("s", "s_f0", "s_f1"):
                argmax = int(np.argmax(beliefs[step]))
                color = state_colors[argmax % len(state_colors)]
                alpha = 0.3 + 0.7 * val  # brighter when more confident
            elif step > 0 and step - 1 < len(beliefs) and name in ("s'", "s_prime"):
                argmax = int(np.argmax(beliefs[step - 1]))
                color = state_colors[argmax % len(state_colors)]
                alpha = 0.3 + 0.7 * val
            else:
                color = (0.7, 0.7, 0.7)
                alpha = 0.5
        elif name in ("o", "o_m0", "y"):
            # Color by observation value
            if step < len(observations):
                obs_idx = observations[step]
                color = state_colors[obs_idx % len(state_colors)]
                alpha = 0.8
            else:
                color = (0.7, 0.7, 0.7)
                alpha = 0.5
        elif name in ("u",):
            if step < len(actions):
                act_idx = actions[step]
                color = state_colors[act_idx % len(state_colors)]
                alpha = 0.8
            else:
                color = (0.7, 0.7, 0.7)
                alpha = 0.5
        elif name in ("G", "F"):
            # VFE: red intensity
            color = (1.0, 0.3, 0.3)
            alpha = 0.3 + 0.7 * val
        else:
            # Parameters: light gray
            color = (0.85, 0.85, 0.85)
            alpha = 0.9

        circle = Circle(
            (x, y), 0.05, color=color, alpha=alpha, ec="#333", lw=1.5, zorder=5
        )
        ax.add_patch(circle)
        ax.text(
            x,
            y - 0.09,
            name,
            ha="center",
            va="top",
            fontsize=9,
            color="#333",
            fontweight="bold",
            zorder=6,
        )


def generate_gif_animation(
    data: Dict[str, Any],
    output_path: Path,
    model_name: str = "model",
    fps: int = 4,
    dpi: int = 100,
) -> str:
    """Generate an animated GIF from RxInfer simulation results.

    The GIF shows a 2x3 panel in publication-quality white style:
    - Top-left: Belief bar chart (colors per state, heights = probability). For
      multi-factor models (``model_parameters.state_factors`` describing more
      than one factor of size > 1) this panel is subdivided into one bar
      sub-panel per factor showing that factor's marginal, since a joint bar
      chart over 256 or 729 states is unreadable. Joint entropy is annotated on
      the first sub-panel.
    - Top-centre: State tracking heatmap (proper discrete colors)
    - Top-right: VFE convergence line
    - Bottom-left: Bayesian graphical model (nodes/edges, node colors dynamic)
    - Bottom-centre: EFE per action heatmap (per-timestep EFE landscape)
    - Bottom-right: Policy posterior (stacked action distribution over time)

    Args:
        data: Simulation results dict (rxinfer_simulation_v1)
        output_path: Where to write the GIF file
        model_name: Model name for labeling
        fps: Frames per second (default 4)
        dpi: DPI for the figure (default 100)

    Returns:
        Path to the generated GIF file, or "" if no beliefs
    """
    from gnn.analysis.result_adapter import model_family, result_views

    from .family_visuals import continuous_gif, marginal_gif

    if fps <= 0 or dpi <= 0:
        raise ValueError("Animation fps and dpi must be positive")
    views = result_views(data)
    if model_family(data) == "continuous":
        return continuous_gif(data, output_path, model_name, fps, dpi)
    if len(views) > 1:
        return marginal_gif(
            data, views, output_path, model_name, fps, dpi, generate_gif_animation
        )
    if views:
        data = next(iter(views.values()))
    beliefs = _normalize_beliefs(data)
    if not beliefs:
        logger.warning("No beliefs found — cannot generate GIF")
        return ""

    observations = _normalize_observations(data)
    true_states = _normalize_true_states(data)
    actions = _normalize_actions(data)
    vfe = _normalize_vfe(data)
    efe_per_action = _normalize_efe_per_action(data)
    policy_posterior = _normalize_policy_posterior(data)

    n_steps = len(beliefs)
    n_states = len(beliefs[0]) if beliefs else 0
    n_actions = (
        len(efe_per_action[0])
        if efe_per_action
        else (
            len(policy_posterior[0])
            if policy_posterior
            else (max(actions) + 1 if actions else 0)
        )
    )

    beliefs_arr = np.array(beliefs)

    # Per-factor marginals (D4): empty for flat / single-factor models, in which
    # case the top-left panel stays the joint belief bar chart.
    per_factor = compute_per_factor_beliefs(data)
    factor_names = list(per_factor)
    factor_traces = {
        name: np.asarray(per_factor[name], dtype=float) for name in factor_names
    }
    factor_palettes = {
        name: _hue_palette(factor_traces[name].shape[1]) for name in factor_names
    }

    # Joint entropy per timestep, annotated on the factor panels.
    belief_clipped = np.clip(beliefs_arr, 1e-10, 1.0)
    joint_entropy = -np.sum(belief_clipped * np.log2(belief_clipped), axis=1)

    # Parse graphical model structure. Node positions are then overridden by
    # the render-side model strategy's native layout (FP-8) wherever the
    # strategy declares a coordinate for a parsed node; the GNN-derived node
    # set and edges stay authoritative for graph structure.
    positions, edges = _parse_gnn_connections(data)
    strategy_layout = _resolve_strategy_layout(data)
    for node, coord in strategy_layout.items():
        if node in positions and isinstance(coord, (tuple, list)) and len(coord) == 2:
            positions[node] = (float(coord[0]), float(coord[1]))

    # Color palette — distinct colors for each state (publication style)
    state_colors = _hue_palette(max(n_states, n_actions))

    # Create figure — white publication style (2x3 grid: D6/D8 panels)
    plt.style.use("default")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor="white")
    fig.suptitle(
        f"RxInfer — {model_name}", fontsize=13, fontweight="bold", color="#222"
    )
    fig.subplots_adjust(hspace=0.45, wspace=0.32)

    # Subdivide the top-left cell into one sub-panel per factor (D4).
    if factor_names:
        top_left_spec = axes[0, 0].get_subplotspec()
        axes[0, 0].remove()
        inner_grid = top_left_spec.subgridspec(1, len(factor_names), wspace=0.45)
        factor_axes = [fig.add_subplot(inner_grid[i]) for i in range(len(factor_names))]
        animated_axes = factor_axes + list(axes.flat)[1:]
    else:
        factor_axes = []
        animated_axes = list(axes.flat)

    # Pre-compute VFE range
    vfe_arr = np.array(vfe) if vfe else np.array([0.0])
    vfe_min = float(vfe_arr.min()) if len(vfe_arr) > 0 else 0.0
    vfe_max = float(vfe_arr.max()) if len(vfe_arr) > 0 else 1.0
    vfe_range = vfe_max - vfe_min if vfe_max != vfe_min else 1.0

    # Pre-compute EFE range + a single colorbar (D6) so frames stay comparable
    # and no colorbar accumulates per frame.
    if efe_per_action and n_actions > 0:
        eaa_full = np.asarray(
            [row[:n_actions] for row in efe_per_action], dtype=float
        ).T
        efe_vmin = float(np.nanmin(eaa_full)) if eaa_full.size else 0.0
        efe_vmax = float(np.nanmax(eaa_full)) if eaa_full.size else 1.0
        if efe_vmin == efe_vmax:
            efe_vmax = efe_vmin + 1e-6
        efe_sm = plt.cm.ScalarMappable(
            cmap="RdBu_r", norm=Normalize(vmin=efe_vmin, vmax=efe_vmax)
        )
        efe_sm.set_array([])
        efe_cb = fig.colorbar(efe_sm, ax=axes[1, 1], fraction=0.046, pad=0.04)
        efe_cb.set_label("EFE", fontsize=7, color="#444")
        efe_cb.ax.tick_params(labelsize=6, colors="#444")
    else:
        eaa_full = np.empty((0, 0))
        efe_vmin, efe_vmax = 0.0, 1.0

    def animate(frame: int) -> List[Any]:
        """Update function for each animation frame."""
        for ax in animated_axes:
            ax.clear()
        step = min(frame, n_steps - 1)

        # === Top-left: per-factor marginals (D4) or joint belief bar chart ===
        if factor_axes:
            for panel, name in zip(factor_axes, factor_names):
                marginal = factor_traces[name][step]
                factor_size = marginal.shape[0]
                panel.set_facecolor("white")
                panel.bar(
                    np.arange(factor_size),
                    marginal,
                    color=factor_palettes[name],
                    edgecolor="#333",
                    linewidth=0.4,
                )
                panel.set_ylim(0, 1.0)
                panel.set_xlim(-0.5, factor_size - 0.5)
                panel.set_title(name, fontsize=9, color="#222")
                panel.set_xlabel("State", fontsize=7, color="#444")
                panel.tick_params(colors="#444", labelsize=6)
                panel.spines["top"].set_visible(False)
                panel.spines["right"].set_visible(False)
                # Only the leftmost panel carries the shared probability scale.
                if panel is not factor_axes[0]:
                    panel.set_yticklabels([])
            factor_axes[0].set_ylabel("P(factor state)", fontsize=8, color="#444")
            factor_axes[0].text(
                0.02,
                0.98,
                f"H(joint)={joint_entropy[step]:.2f} bits",
                transform=factor_axes[0].transAxes,
                ha="left",
                va="top",
                color="#444",
                fontsize=7,
            )
        else:
            ax1 = axes[0, 0]
            ax1.set_facecolor("white")
            belief = beliefs_arr[step]
            x_pos = np.arange(n_states)
            ax1.bar(x_pos, belief, color=state_colors, edgecolor="#333", linewidth=0.5)
            ax1.set_ylim(0, 1.0)
            ax1.set_xlim(-0.5, n_states - 0.5)
            ax1.set_title(f"Belief (t={step + 1})", fontsize=10, color="#222")
            ax1.set_xlabel("State", fontsize=9, color="#444")
            ax1.set_ylabel("P(state)", fontsize=9, color="#444")
            ax1.tick_params(colors="#444", labelsize=8)
            ax1.spines["top"].set_visible(False)
            ax1.spines["right"].set_visible(False)
            for j in range(n_states):
                ax1.text(
                    j,
                    belief[j] + 0.02,
                    f"{belief[j]:.2f}",
                    ha="center",
                    va="bottom",
                    color="#444",
                    fontsize=7,
                )

        # === Top-right: State tracking heatmap ===
        ax2 = axes[0, 1]
        ax2.set_facecolor("white")
        # Discrete colormap for state indices
        if n_states > 1:
            cmap = ListedColormap(state_colors[:n_states])
            bounds = np.arange(n_states + 1) - 0.5
            norm = BoundaryNorm(bounds, cmap.N)
        else:
            cmap = ListedColormap([state_colors[0]])
            norm = None

        if step >= 0:
            argmax_beliefs = np.argmax(beliefs_arr[: step + 1], axis=1)
            rows = [argmax_beliefs]
            row_labels = ["MAP belief"]
            if true_states:
                rows.append(np.array(true_states[: step + 1]))
                row_labels.append("True state")
            if observations:
                rows.append(np.array(observations[: step + 1]))
                row_labels.append("Observation")
            heatmap_data = np.vstack(rows)
            if norm:
                ax2.imshow(
                    heatmap_data,
                    aspect="auto",
                    cmap=cmap,
                    norm=norm,
                    interpolation="nearest",
                )
            else:
                ax2.imshow(
                    heatmap_data, aspect="auto", cmap=cmap, interpolation="nearest"
                )

            ax2.set_yticks(list(range(len(row_labels))))
            ax2.set_yticklabels(row_labels, fontsize=8, color="#444")
            ax2.set_title("State Tracking", fontsize=10, color="#222")
            ax2.set_xlabel("Timestep", fontsize=9, color="#444")
            ax2.tick_params(colors="#444", labelsize=7)

        # === Bottom-left: Bayesian graphical model ===
        ax3 = axes[1, 0]
        ax3.set_facecolor("white")
        _draw_graph_model(
            ax3,
            positions,
            edges,
            step,
            beliefs,
            observations,
            actions,
            vfe,
            state_colors,
        )
        ax3.set_title("Graphical Model", fontsize=10, color="#222")

        # === Top-right: VFE convergence ===
        ax4 = axes[0, 2]
        ax4.set_facecolor("white")
        if vfe and len(vfe) > 0:
            vfe_step = min(int(step * len(vfe) / max(n_steps, 1)), len(vfe) - 1)
            vfe_x = np.arange(vfe_step + 1)
            vfe_y = vfe_arr[: vfe_step + 1]
            ax4.fill_between(vfe_x, vfe_y, vfe_min, alpha=0.2, color="#c0392b")
            ax4.plot(vfe_x, vfe_y, color="#c0392b", linewidth=1.5)
            if len(vfe_y) > 0:
                ax4.scatter([vfe_step], [vfe_y[-1]], color="#c0392b", s=25, zorder=5)
            ax4.set_ylim(vfe_min - 0.1 * vfe_range, vfe_max + 0.1 * vfe_range)
        ax4.set_title("VFE Convergence", fontsize=10, color="#222")
        ax4.set_xlabel("Iteration", fontsize=9, color="#444")
        ax4.set_ylabel("VFE", fontsize=9, color="#444")
        ax4.tick_params(colors="#444", labelsize=7)
        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)

        # === Bottom-centre: EFE per action heatmap (D6) ===
        ax5 = axes[1, 1]
        ax5.set_facecolor("white")
        if efe_per_action and n_actions > 0 and step >= 0:
            n_eaa_steps = min(step + 1, eaa_full.shape[1])
            if n_eaa_steps > 0:
                ax5.imshow(
                    eaa_full[:, :n_eaa_steps],
                    aspect="auto",
                    cmap="RdBu_r",
                    origin="lower",
                    interpolation="nearest",
                    vmin=efe_vmin,
                    vmax=efe_vmax,
                )
                ax5.set_yticks(range(n_actions))
                ax5.set_yticklabels(
                    [f"A{i + 1}" for i in range(n_actions)], fontsize=7, color="#444"
                )
                ax5.set_title("EFE per Action", fontsize=10, color="#222")
                ax5.set_xlabel("Timestep", fontsize=9, color="#444")
                ax5.tick_params(colors="#444", labelsize=7)
            else:
                ax5.set_title("EFE per Action", fontsize=10, color="#222")
                ax5.text(
                    0.5,
                    0.5,
                    "No EFE data",
                    ha="center",
                    va="center",
                    color="#888",
                    fontsize=9,
                )
        else:
            ax5.set_title("EFE per Action", fontsize=10, color="#222")
            ax5.text(
                0.5,
                0.5,
                "No EFE data",
                ha="center",
                va="center",
                color="#888",
                fontsize=9,
            )

        # === Bottom-right: Policy posterior (D8) ===
        ax6 = axes[1, 2]
        ax6.set_facecolor("white")
        if policy_posterior and step >= 0 and n_actions > 0:
            n_pp_steps = min(step + 1, len(policy_posterior))
            if n_pp_steps > 0:
                pp_arr = np.asarray(
                    [policy_posterior[i][:n_actions] for i in range(n_pp_steps)],
                    dtype=float,
                )  # (timestep x action)
                pp_x = np.arange(n_pp_steps)
                pp_colors = [
                    state_colors[i] for i in range(min(n_actions, len(state_colors)))
                ]
                ax6.stackplot(
                    pp_x,
                    pp_arr.T,
                    labels=[f"A{i + 1}" for i in range(n_actions)],
                    colors=pp_colors,
                    baseline="zero",
                )
                ax6.set_xlim(0, max(n_pp_steps - 1, 1))
                ax6.set_ylim(0, 1.0)
                ax6.set_title("Policy Posterior", fontsize=10, color="#222")
                ax6.set_xlabel("Timestep", fontsize=9, color="#444")
                ax6.set_ylabel("P(action)", fontsize=9, color="#444")
                ax6.tick_params(colors="#444", labelsize=7)
                ax6.spines["top"].set_visible(False)
                ax6.spines["right"].set_visible(False)
                if n_actions <= 6:
                    ax6.legend(fontsize=6, loc="center left", frameon=False)
            else:
                ax6.text(
                    0.5,
                    0.5,
                    "No policy data",
                    ha="center",
                    va="center",
                    color="#888",
                    fontsize=9,
                )
                ax6.set_title("Policy Posterior", fontsize=10, color="#222")
        else:
            ax6.text(
                0.5,
                0.5,
                "No policy data",
                ha="center",
                va="center",
                color="#888",
                fontsize=9,
            )
            ax6.set_title("Policy Posterior", fontsize=10, color="#222")

        # Step indicator
        fig.text(
            0.5,
            0.02,
            f"Step {step + 1}/{n_steps}",
            ha="center",
            color="#222",
            fontsize=9,
            fontweight="bold",
        )
        return []

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=n_steps, interval=1000 // fps, repeat=True
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.PillowWriter(fps=fps)
    anim.save(str(output_path), writer=writer)
    plt.close(fig)

    # Write reproducibility manifest sidecar
    rt = data.get("runtime_metadata", {})
    spec_str = json.dumps(data.get("gnn_spec", {}), sort_keys=True)
    manifest = {
        "gnn_spec_sha256": hashlib.sha256(spec_str.encode()).hexdigest(),
        "julia_version": rt.get("julia_version", "unknown"),
        "rxinfer_version": rt.get("rxinfer_version", "unknown"),
        "seed": rt.get("random_seed", "unknown"),
        "timesteps": data.get("num_timesteps", "unknown"),
        "inference_iterations": data.get("model_parameters", {}).get(
            "inference_iterations", "unknown"
        ),
        "belief_accuracy": data.get("validation", {}).get("belief_accuracy"),
        "inference_converged": data.get("validation", {}).get("inference_converged"),
        "uses_real_rxinfer": rt.get("uses_real_rxinfer"),
        "model_kind": rt.get("model_kind", "unknown"),
        "num_states": data.get("model_parameters", {}).get("num_states"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "gif_animator.py",
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    logger.info("Generated GIF animation: %s", output_path)
    return str(output_path)


__all__: list[Any] = ["generate_gif_animation"]

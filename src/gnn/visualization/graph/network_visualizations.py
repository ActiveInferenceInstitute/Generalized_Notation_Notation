"""
Network graph generation for GNN models (directed vs undirected edges, ontology labels).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from gnn.visualization.connection_format import normalize_connection_format

try:
    from gnn.parsers.common import VariableType

    _HID = VariableType.HIDDEN_STATE.value
    _OBS = VariableType.OBSERVATION.value
    _ACT = VariableType.ACTION.value
    _POL = VariableType.POLICY.value
    _PRI = VariableType.PRIOR_VECTOR.value
except ImportError:
    _HID, _OBS, _ACT, _POL, _PRI = (
        "hidden_state",
        "observation",
        "action",
        "policy",
        "prior_vector",
    )

from ..compat.viz_compat import MATPLOTLIB_AVAILABLE, plt
from ..compat.viz_compat import viz_var_type as _var_type
from ..theme import VAR_TYPE_COLORS
from ..theme import get_edge_style as _get_edge_style

try:
    import os
    import sys

    if sys.version_info >= (3, 13):
        os.environ.pop("NETWORKX_AUTOMATIC_BACKENDS", None)
        os.environ["NETWORKX_CACHE_CONVERTED_GRAPHS"] = "1"
    import networkx as nx

    NETWORKX_AVAILABLE = True
except (ImportError, RecursionError, AttributeError, ValueError):
    nx = cast(Any, None)
    NETWORKX_AVAILABLE = False

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    go = cast(Any, None)
    PLOTLY_AVAILABLE = False

from ..plotting.utils import safe_tight_layout, save_figure

logger = logging.getLogger(__name__)


def _connection_is_undirected(conn_info: Dict[str, Any]) -> bool:
    """Handle connection is undirected for internal callers."""
    ct = str(conn_info.get("connection_type", "directed")).lower().strip()
    return ct == "undirected"


def generate_network_visualizations(
    parsed_data: Dict[str, Any], output_dir: Path, model_name: str
) -> List[str]:
    """Generate network visualizations."""
    visualizations: List[str] = []

    if not NETWORKX_AVAILABLE or not MATPLOTLIB_AVAILABLE or plt is None or nx is None:
        return visualizations

    try:
        variables = parsed_data.get("variables", [])
        if not isinstance(variables, list):
            variables = []

        connections = parsed_data.get("connections", [])
        if not isinstance(connections, list):
            connections = []

        ontology_labels: Dict[str, str] = parsed_data.get("ontology_labels") or {}

        G_layout = nx.Graph()
        directed_edges: List[Tuple[str, str, str]] = []
        undirected_edges: List[Tuple[str, str, str]] = []

        for var_info in variables:
            if isinstance(var_info, dict):
                var_name = var_info.get("name", "unknown")
                var_type = _var_type(var_info)
                dimensions = var_info.get("dimensions", [])
                description = var_info.get("description", "")
                G_layout.add_node(
                    var_name,
                    type=var_type,
                    dimensions=dimensions,
                    description=description,
                    size=max(1, min(10, len(dimensions) * 2)),
                )

        # Name → type lookup built once (the per-edge rescan over all
        # variables was O(variables × edges)).
        var_types: Dict[str, str] = {
            str(v["name"]): _var_type(v)
            for v in variables
            if isinstance(v, dict) and v.get("name")
        }

        for conn_info in connections:
            if not isinstance(conn_info, dict):
                continue
            normalized_conn = normalize_connection_format(conn_info)
            source_vars = normalized_conn.get("source_variables", [])
            target_vars = normalized_conn.get("target_variables", [])
            undir = _connection_is_undirected(conn_info)

            for source_var in source_vars:
                for target_var in target_vars:
                    if not source_var or not target_var or source_var == target_var:
                        continue
                    conn_type = _determine_connection_type(
                        source_var,
                        target_var,
                        var_types.get(source_var),
                        var_types.get(target_var),
                    )
                    G_layout.add_edge(source_var, target_var)
                    pair = (source_var, target_var, conn_type)
                    if undir:
                        undirected_edges.append(pair)
                    else:
                        directed_edges.append(pair)

        if len(G_layout.nodes()) == 0:
            logger.warning(
                "No valid nodes found for network visualization of %s", model_name
            )
            return visualizations

        plt.figure(figsize=(14, 12))
        pos = nx.spring_layout(G_layout, k=2, iterations=100, seed=42)

        node_sizes = [
            G_layout.nodes[node].get("size", 5) * 100 for node in G_layout.nodes()
        ]
        node_types = [
            G_layout.nodes[node].get("type", "unknown") for node in G_layout.nodes()
        ]

        node_colors = [
            VAR_TYPE_COLORS.get(node_type, "gray") for node_type in node_types
        ]

        edge_groups_dir: Dict[str, List[Tuple[str, str]]] = {}
        for u, v, conn_type in directed_edges:
            edge_groups_dir.setdefault(conn_type, []).append((u, v))

        edge_groups_undir: Dict[str, List[Tuple[str, str]]] = {}
        for u, v, conn_type in undirected_edges:
            edge_groups_undir.setdefault(conn_type, []).append((u, v))

        G_dir = nx.DiGraph()
        G_dir.add_nodes_from(G_layout.nodes(data=True))
        for u, v, _d in directed_edges:
            G_dir.add_edge(u, v)

        G_undir = nx.Graph()
        G_undir.add_nodes_from(G_layout.nodes(data=True))
        for u, v, _d in undirected_edges:
            G_undir.add_edge(u, v)

        for conn_type, edges in edge_groups_undir.items():
            style = _get_edge_style(conn_type)
            if edges:
                nx.draw_networkx_edges(
                    G_undir,
                    pos,
                    edgelist=edges,
                    edge_color=style["color"],
                    width=style["width"],
                    alpha=style["alpha"],
                    arrows=False,
                    style=style["style"],
                )

        for conn_type, edges in edge_groups_dir.items():
            style = _get_edge_style(conn_type)
            if edges:
                nx.draw_networkx_edges(
                    G_dir,
                    pos,
                    edgelist=edges,
                    edge_color=style["color"],
                    width=style["width"],
                    alpha=style["alpha"],
                    arrows=True,
                    arrowsize=20,
                    style=style["style"],
                )

        nx.draw_networkx_nodes(
            G_layout, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8
        )

        label_map: Dict[str, str] = {}
        for node in G_layout.nodes():
            term = ontology_labels.get(node)
            if term:
                label_map[node] = f"{node}\n({term})"
            else:
                label_map[node] = str(node)
        nx.draw_networkx_labels(
            G_layout, pos, labels=label_map, font_size=9, font_weight="bold"
        )

        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, fc=color, label=var_type)
            for var_type, color in VAR_TYPE_COLORS.items()
        ]
        plt.legend(
            handles=legend_elements, loc="upper right", bbox_to_anchor=(1.0, 1.0)
        )

        plt.title(
            f"Bayesian Graphical Model: {model_name}\nGNN network (directed vs undirected)",
            fontsize=16,
            fontweight="bold",
        )
        plt.axis("off")
        safe_tight_layout()

        network_path = output_dir / f"{model_name}_network_graph.png"
        save_figure(network_path, dpi=300, bbox_inches="tight")
        plt.close()
        visualizations.append(str(network_path))

        stats = _compute_graph_metrics(variables, connections)
        stats_path = output_dir / f"{model_name}_network_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        visualizations.append(str(stats_path))

        if ontology_labels:
            leg_path = output_dir / f"{model_name}_ontology_legend.txt"
            try:
                rows = [
                    f"{var}\t{term}" for var, term in sorted(ontology_labels.items())
                ]
                leg_path.write_text(
                    "variable\tontology_term\n" + "\n".join(rows) + "\n",
                    encoding="utf-8",
                )
                visualizations.append(str(leg_path))
            except OSError as e:
                logger.debug("Could not write ontology legend %s: %s", leg_path, e)

        G_combined = nx.DiGraph()
        G_combined.add_nodes_from(G_layout.nodes(data=True))
        for u, v, _d in directed_edges:
            G_combined.add_edge(u, v)
        for u, v, _d in undirected_edges:
            G_combined.add_edge(u, v)

        if PLOTLY_AVAILABLE:
            try:
                interactive_path = _generate_interactive_network(
                    G_combined, output_dir / f"{model_name}_network_interactive.html"
                )
                if interactive_path:
                    visualizations.append(str(interactive_path))
            except Exception as e:
                logger.warning("Failed to generate interactive network: %s", e)

    except Exception as e:
        logger.exception(
            "Error generating network visualizations for %s: %s", model_name, e
        )

    return visualizations


def _determine_connection_type(
    source_var: str,
    target_var: str,
    source_type: Optional[str] = None,
    target_type: Optional[str] = None,
) -> str:
    """Handle determine connection type for internal callers."""
    if source_type and target_type:
        if source_type == _HID and target_type == _HID:
            return "state_transition"
        if source_type == _HID and target_type == _OBS:
            return "observation_generation"
        if source_type == _HID and "transition" in target_type:
            return "state_action_influence"
        if source_type == _ACT and target_type == _HID:
            return "action_effect"
        if source_type == _POL and target_type == _ACT:
            return "policy_selection"
        if source_type == _PRI and target_type == _HID:
            return "prior_influence"
        if source_type == _HID and "likelihood" in target_type:
            return "likelihood_influence"
        if "free_energy" in source_type or "free_energy" in target_type:
            return "energy_flow"

    if source_var == "s" and target_var in ["A", "o"]:
        return "state_observation"
    if source_var in ["s", "s_prime"] and target_var == "B":
        return "state_transition_matrix"
    if source_var == "C" and target_var == "G":
        return "preference_energy"
    if source_var == "E" and target_var == "\u03c0":
        return "habit_policy"
    if source_var == "\u03c0" and target_var == "u":
        return "policy_action"

    return "generic_causal"


def _compute_graph_metrics(
    variables: List[Dict[str, Any]], connections: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compute network metrics: counts, type tallies, orientation, topology."""
    stats: Dict[str, Any] = {
        "total_variables": len(variables),
        "total_connections": len(connections),
        "variable_types": {},
        "connection_types": {},
        "gnn_edge_orientation": {
            "directed_variable_pairs": 0,
            "undirected_variable_pairs": 0,
        },
        "network_properties": {},
    }

    for var_info in variables:
        if isinstance(var_info, dict):
            var_type = _var_type(var_info)
            stats["variable_types"][var_type] = (
                stats["variable_types"].get(var_type, 0) + 1
            )

    orient = stats["gnn_edge_orientation"]
    for conn_info in connections:
        if isinstance(conn_info, dict):
            normalized_conn = normalize_connection_format(conn_info)
            source_vars = normalized_conn.get("source_variables", [])
            target_vars = normalized_conn.get("target_variables", [])
            undir = _connection_is_undirected(conn_info)
            for source_var in source_vars:
                for target_var in target_vars:
                    if source_var != target_var:
                        conn_type = f"{source_var}->{target_var}"
                        stats["connection_types"][conn_type] = (
                            stats["connection_types"].get(conn_type, 0) + 1
                        )
                        if undir:
                            orient["undirected_variable_pairs"] += 1
                        else:
                            orient["directed_variable_pairs"] += 1

    if NETWORKX_AVAILABLE and nx is not None:
        try:
            simple_G = nx.DiGraph()
            for conn_info in connections:
                normalized_conn = normalize_connection_format(conn_info)
                source_vars = normalized_conn.get("source_variables", [])
                target_vars = normalized_conn.get("target_variables", [])
                for source_var in source_vars:
                    for target_var in target_vars:
                        if source_var != target_var:
                            simple_G.add_edge(source_var, target_var)

            if len(simple_G.nodes()) > 0:
                stats["network_properties"] = {
                    "num_nodes": simple_G.number_of_nodes(),
                    "num_edges": simple_G.number_of_edges(),
                    "density": nx.density(simple_G),
                    "is_strongly_connected": nx.is_strongly_connected(simple_G)
                    if len(simple_G.nodes()) > 1
                    else True,
                    "is_weakly_connected": nx.is_weakly_connected(simple_G)
                    if len(simple_G.nodes()) > 1
                    else True,
                    "average_clustering": nx.average_clustering(simple_G),
                }
        except Exception as e:
            logger.warning("Error calculating network properties: %s", e)

    return stats


def _generate_interactive_network(G: Any, output_path: Path) -> bool:
    """Generate interactive network."""
    if not PLOTLY_AVAILABLE or not go or nx is None:
        return False

    try:
        pos = nx.spring_layout(G, k=2, iterations=100, seed=42)

        edge_x: List[Any] = []
        edge_y: List[Any] = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line={"width": 1, "color": "#888"},
            hoverinfo="none",
            mode="lines",
        )

        node_x: list[Any] = []
        node_y: list[Any] = []
        node_info: list[Any] = []
        node_types = nx.get_node_attributes(G, "type")

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_info.append(
                f"{node}<br>Type: {node_types.get(node, 'unknown')}<br>Connections: {G.degree(node)}"
            )

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=list(G.nodes()),
            textposition="top center",
            hovertext=node_info,
            marker={
                "size": [G.degree(node) * 10 + 20 for node in G.nodes()],
                "color": [
                    "lightblue"
                    if node_types.get(node, "unknown") == "unknown"
                    else "orange"
                    for node in G.nodes()
                ],
                "line": {"width": 2},
            },
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=f"Interactive Network Graph ({len(G.nodes())} nodes, {len(G.edges())} edges)",
                title_font_size=16,
                showlegend=False,
                hovermode="closest",
                margin={"b": 20, "l": 5, "r": 5, "t": 40},
                annotations=[
                    {
                        "text": "Hover over nodes and edges for details",
                        "showarrow": False,
                        "xref": "paper",
                        "yref": "paper",
                        "x": 0.005,
                        "y": -0.002,
                    }
                ],
                xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
                yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            ),
        )

        fig.write_html(str(output_path))
        return True

    except Exception as e:
        logger.warning("Error generating interactive network: %s", e)
        return False

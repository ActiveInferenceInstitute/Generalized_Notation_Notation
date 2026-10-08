"""
Network and POMDP visualization sub-module.

Provides 3D network visualization, interactive dashboards, POMDP transition
analysis, policy visualization, network metrics, and D2 diagram generation.

Extracted from processor.py for maintainability.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, cast

from gnn.render.naming import safe_output_stem

from ._shared import (
    MATPLOTLIB_AVAILABLE,
    NUMPY_AVAILABLE,
    SEABORN_AVAILABLE,
    VAR_TYPE_COLORS,
    VAR_TYPE_UNKNOWN_COLOR,
    AdvancedVisualizationAttempt,
    _calculate_semantic_positions,
    _conn_endpoints,
    _generate_fallback_report,
    _MatrixVisualizer,
    get_sns,
    np,
    plt,
    validate_visualization_data,
)


def _generate_3d_visualization(
    model_name: str,
    model_data: Dict[str, Any],
    output_dir: Path,
    export_formats: List[str],
    dependencies: Dict[str, bool],
    logger: logging.Logger,
) -> AdvancedVisualizationAttempt:
    model_name = safe_output_stem(model_name)
    attempt = AdvancedVisualizationAttempt(
        viz_type="3d", model_name=model_name, status="skipped"
    )

    start_time = time.time()

    try:
        if MATPLOTLIB_AVAILABLE and plt and np:
            fig = plt.figure(figsize=(10, 8))
            ax = cast(Any, fig.add_subplot(111, projection="3d"))

            variables = model_data.get("variables", [])
            connections = model_data.get("connections", [])

            validation_results = validate_visualization_data(model_data, logger)

            if not validation_results["overall_valid"]:
                logger.warning(
                    f"Poor data quality for 3D visualization of {model_name}"
                )
                if validation_results["errors"]:
                    logger.error(f"Errors: {validation_results['errors']}")
                if validation_results["warnings"]:
                    logger.warning(f"Warnings: {validation_results['warnings']}")
                attempt.status = "failed"
                attempt.error_message = f"Data validation failed: {len(validation_results['errors'])} errors, {len(validation_results['warnings'])} warnings"
                return attempt

            if variables:
                positions = _calculate_semantic_positions(variables, connections)

                name_to_idx = {var.get("name"): i for i, var in enumerate(variables)}

                node_sizes: list[Any] = []
                node_colors: list[Any] = []

                for var in variables:
                    var_type = var.get("var_type", "unknown")
                    dimensions = var.get("dimensions", [])

                    base_size = 50.0
                    if isinstance(dimensions, list) and len(dimensions) > 0:
                        dim_product = 1.0
                        for dim in dimensions[:2]:
                            if isinstance(dim, (int, float)):
                                dim_product *= dim
                        size_multiplier = min(3, max(0.5, dim_product / 10))
                        base_size *= size_multiplier

                    node_sizes.append(base_size)
                    node_colors.append(
                        VAR_TYPE_COLORS.get(var_type, VAR_TYPE_UNKNOWN_COLOR)
                    )

                for i, (pos, color, size) in enumerate(
                    zip(positions, node_colors, node_sizes)
                ):
                    ax.scatter(
                        pos[0],
                        pos[1],
                        pos[2],
                        c=color,
                        s=size,
                        alpha=0.8,
                        edgecolors="black",
                    )
                    ax.text(
                        pos[0],
                        pos[1],
                        pos[2],
                        variables[i].get("name", f"Var{i}"),
                        fontsize=8,
                        ha="center",
                        va="center",
                        fontweight="bold",
                    )

                if connections:
                    for conn_info in connections:
                        source_vars, target_vars = _conn_endpoints(conn_info)

                        for source_var in source_vars:
                            for target_var in target_vars:
                                if source_var != target_var:
                                    source_idx = name_to_idx.get(source_var)
                                    target_idx = name_to_idx.get(target_var)

                                    if (
                                        source_idx is not None
                                        and target_idx is not None
                                    ):
                                        source_pos = positions[source_idx]
                                        target_pos = positions[target_idx]

                                        ax.plot(
                                            [source_pos[0], target_pos[0]],
                                            [source_pos[1], target_pos[1]],
                                            [source_pos[2], target_pos[2]],
                                            "gray",
                                            alpha=0.5,
                                            linewidth=2,
                                        )

                                        ax.scatter(
                                            target_pos[0],
                                            target_pos[1],
                                            target_pos[2],
                                            c="red",
                                            s=30,
                                            marker=">",
                                            alpha=0.7,
                                        )

                ax.set_xlabel("X Dimension")
                ax.set_ylabel("Y Dimension")
                ax.set_zlabel("Z Dimension")
                ax.set_title(f"3D Model Structure: {model_name}")

                plt.tight_layout()

                output_file = output_dir / f"{model_name}_3d_visualization.png"
                plt.savefig(output_file, dpi=150, bbox_inches="tight")
                plt.close()

                attempt.status = "success"
                attempt.output_files.append(str(output_file))
                logger.info(f"Generated 3D visualization: {output_file}")
            else:
                logger.info(f"No variables found for 3D visualization of {model_name}")
                attempt.status = "skipped"

        elif not dependencies.get("plotly"):
            logger.info(
                f"Skipping 3D visualization for {model_name} (plotly not available)"
            )
            attempt.fallback_used = True
            _generate_fallback_report(model_name, "3d", output_dir, model_data, logger)
            attempt.status = "success"
            attempt.output_files.append(
                str(output_dir / f"{model_name}_3d_fallback.html")
            )
        else:
            logger.info(
                f"3D visualization for {model_name} - using matplotlib recovery"
            )
            attempt.fallback_used = True
            _generate_fallback_report(model_name, "3d", output_dir, model_data, logger)
            attempt.status = "success"
            attempt.output_files.append(
                str(output_dir / f"{model_name}_3d_fallback.html")
            )

    except Exception as e:
        logger.error(f"Failed to generate 3D visualization for {model_name}: {e}")
        attempt.status = "failed"
        attempt.error_message = str(e)
    finally:
        attempt.duration_ms = (time.time() - start_time) * 1000

    return attempt


def _generate_interactive_dashboard(
    model_name: str,
    model_data: Dict[str, Any],
    output_dir: Path,
    export_formats: List[str],
    dependencies: Dict[str, bool],
    logger: logging.Logger,
) -> AdvancedVisualizationAttempt:
    """Generate interactive dashboard"""
    model_name = safe_output_stem(model_name)
    attempt = AdvancedVisualizationAttempt(
        viz_type="dashboard", model_name=model_name, status="skipped"
    )

    start_time = time.time()

    try:
        if not (dependencies.get("plotly") or dependencies.get("bokeh")):
            logger.info(
                f"Skipping dashboard for {model_name} (no interactive libraries available)"
            )
            attempt.fallback_used = True
            _generate_fallback_report(
                model_name, "dashboard", output_dir, model_data, logger
            )
            attempt.status = "success"
            attempt.output_files.append(
                str(output_dir / f"{model_name}_dashboard_fallback.html")
            )
        else:
            import plotly.express as px
            import plotly.graph_objects as go

            variables = model_data.get("variables", [])
            connections = model_data.get("connections", [])

            num_vars = len(variables)
            var_types: dict[Any, Any] = {}
            for v in variables:
                v_type = v.get("var_type", "unknown")
                var_types[v_type] = var_types.get(v_type, 0) + 1

            fig_types = px.pie(
                values=list(var_types.values()),
                names=list(var_types.keys()),
                title="Variable Type Distribution",
            )

            adj_matrix: Any = np.zeros((num_vars, num_vars)) if np is not None else []
            var_names = [v.get("name", f"v{i}") for i, v in enumerate(variables)]
            name_to_idx = {name: i for i, name in enumerate(var_names)}

            if NUMPY_AVAILABLE:
                for conn in connections:
                    source_vars, target_vars = _conn_endpoints(conn)
                    for s in source_vars:
                        for t in target_vars:
                            s_idx = name_to_idx.get(s)
                            t_idx = name_to_idx.get(t)
                            if s_idx is not None and t_idx is not None:
                                adj_matrix[s_idx, t_idx] = 1

                fig_adj = px.imshow(
                    adj_matrix,
                    x=var_names,
                    y=var_names,
                    title="Adjacency Matrix",
                    color_continuous_scale="Viridis",
                )
            else:
                fig_adj = go.Figure().add_annotation(text="NumPy not available")

            positions = _calculate_semantic_positions(variables, connections)

            if np is not None and len(positions) > 0:
                x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

                colors: list[Any] = []
                type_color_map: dict[str, Any] = {
                    "likelihood_matrix": 1,
                    "transition_matrix": 2,
                    "preference_vector": 3,
                    "prior_vector": 4,
                    "hidden_state": 5,
                    "observation": 6,
                    "policy": 7,
                    "action": 8,
                }
                for v in variables:
                    colors.append(type_color_map.get(v.get("var_type"), 0))

                fig_3d = go.Figure(
                    data=[
                        go.Scatter3d(
                            x=x,
                            y=y,
                            z=z,
                            mode="markers+text",
                            marker={
                                "size": 8,
                                "color": colors,
                                "colorscale": "Viridis",
                                "opacity": 0.8,
                            },
                            text=var_names,
                            hoverinfo="text",
                        )
                    ]
                )

                for conn in connections:
                    source_vars, target_vars = _conn_endpoints(conn)
                    for s in source_vars:
                        for t in target_vars:
                            s_idx = name_to_idx.get(s)
                            t_idx = name_to_idx.get(t)
                            if s_idx is not None and t_idx is not None:
                                fig_3d.add_trace(
                                    go.Scatter3d(
                                        x=[positions[s_idx, 0], positions[t_idx, 0]],
                                        y=[positions[s_idx, 1], positions[t_idx, 1]],
                                        z=[positions[s_idx, 2], positions[t_idx, 2]],
                                        mode="lines",
                                        line={"color": "gray", "width": 1},
                                        opacity=0.5,
                                        showlegend=False,
                                    )
                                )
                fig_3d.update_layout(title="Interactive 3D Structure")
            else:
                fig_3d = go.Figure().add_annotation(text="Cannot generate 3D plot")

            output_file = output_dir / f"{model_name}_interactive_dashboard.html"

            with open(output_file, "w") as f:
                f.write(
                    f"<html><head><title>GNN Dashboard: {model_name}</title></head><body>"
                )
                f.write(f"<h1>GNN Model Dashboard: {model_name}</h1>")
                f.write("<hr>")
                f.write("<h2>Variable Distribution</h2>")
                f.write(fig_types.to_html(full_html=False, include_plotlyjs="cdn"))
                f.write("<h2>Adjacency Matrix</h2>")
                f.write(fig_adj.to_html(full_html=False, include_plotlyjs=False))
                f.write("<h2>3D Network Structure</h2>")
                f.write(fig_3d.to_html(full_html=False, include_plotlyjs=False))
                f.write("</body></html>")

            attempt.status = "success"
            attempt.output_files.append(str(output_file))
            logger.info(f"Generated interactive dashboard: {output_file}")

    except Exception as e:
        logger.error(f"Failed to generate dashboard for {model_name}: {e}")
        attempt.status = "failed"
        attempt.error_message = str(e)
    finally:
        attempt.duration_ms = (time.time() - start_time) * 1000

    return attempt


def _generate_pomdp_transition_analysis(
    model_name: str,
    model_data: Dict[str, Any],
    output_dir: Path,
    dependencies: Dict[str, bool],
    logger: logging.Logger,
) -> AdvancedVisualizationAttempt:
    """Generate POMDP transition matrix (B matrix) analysis"""
    model_name = safe_output_stem(model_name)
    attempt = AdvancedVisualizationAttempt(
        viz_type="pomdp_transitions", model_name=model_name, status="in_progress"
    )

    start_time = time.time()

    try:
        if not MATPLOTLIB_AVAILABLE or not np:
            attempt.status = "skipped"
            attempt.error_message = "matplotlib/numpy not available"
            return attempt

        mv = _MatrixVisualizer()
        if mv is None:
            attempt.status = "failed"
            attempt.error_message = "MatrixVisualizer not available"
            return attempt

        parameters = model_data.get("parameters", [])
        matrices = mv.extract_matrix_data_from_parameters(parameters)

        if "B" not in matrices:
            attempt.status = "skipped"
            attempt.error_message = (
                "Not applicable to this model type (B matrix not found)"
            )
            return attempt

        B_matrix = matrices["B"]

        if B_matrix.ndim == 3:
            num_actions = B_matrix.shape[2]
            fig, axes = plt.subplots(1, num_actions, figsize=(5 * num_actions, 5))
            if num_actions == 1:
                axes = [axes]

            for action_idx in range(num_actions):
                transition_slice = B_matrix[:, :, action_idx]

                sns = get_sns()
                if SEABORN_AVAILABLE and sns:
                    sns.heatmap(
                        transition_slice,
                        annot=True,
                        fmt=".2f",
                        cmap="Blues",
                        ax=axes[action_idx],
                        cbar=True,
                    )
                else:
                    im = axes[action_idx].imshow(
                        transition_slice, cmap="Blues", aspect="auto"
                    )
                    plt.colorbar(im, ax=axes[action_idx])
                    for i in range(transition_slice.shape[0]):
                        for j in range(transition_slice.shape[1]):
                            axes[action_idx].text(
                                j,
                                i,
                                f"{transition_slice[i, j]:.2f}",
                                ha="center",
                                va="center",
                                color="white"
                                if transition_slice[i, j] > 0.5
                                else "black",
                            )

                axes[action_idx].set_title(f"Transition Matrix (Action {action_idx})")
                axes[action_idx].set_xlabel("Previous State")
                axes[action_idx].set_ylabel("Next State")
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns = get_sns()
            if SEABORN_AVAILABLE and sns:
                sns.heatmap(B_matrix, annot=True, fmt=".2f", cmap="Blues", ax=ax)
            else:
                im = ax.imshow(B_matrix, cmap="Blues", aspect="auto")
                plt.colorbar(im, ax=ax)
            ax.set_title("Transition Matrix (B)")
            ax.set_xlabel("Previous State")
            ax.set_ylabel("Next State")

        plt.suptitle(
            f"POMDP Transition Analysis: {model_name}", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()

        output_file = output_dir / f"{model_name}_pomdp_transitions.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        attempt.status = "success"
        attempt.output_files = [str(output_file)]

    except Exception as e:
        logger.error(
            f"Failed to generate POMDP transition analysis for {model_name}: {e}"
        )
        attempt.status = "failed"
        attempt.error_message = str(e)
    finally:
        attempt.duration_ms = (time.time() - start_time) * 1000

    return attempt


def _generate_policy_visualization(
    model_name: str,
    model_data: Dict[str, Any],
    output_dir: Path,
    dependencies: Dict[str, bool],
    logger: logging.Logger,
) -> AdvancedVisualizationAttempt:
    """Generate policy distribution visualizations"""
    model_name = safe_output_stem(model_name)
    attempt = AdvancedVisualizationAttempt(
        viz_type="policy", model_name=model_name, status="in_progress"
    )

    start_time = time.time()

    try:
        if not MATPLOTLIB_AVAILABLE or not np:
            attempt.status = "skipped"
            attempt.error_message = "matplotlib/numpy not available"
            return attempt

        variables = model_data.get("variables", [])
        parameters = model_data.get("parameters", [])

        policy_data: dict[Any, Any] = {}

        for var in variables:
            if isinstance(var, dict):
                name = var.get("name", "")
                var_type = var.get("var_type", "")
                if "policy" in var_type.lower() or name == "\u03c0" or name == "pi":
                    policy_data[name] = var

        mv = _MatrixVisualizer()
        if mv is None:
            attempt.status = "failed"
            attempt.error_message = "MatrixVisualizer not available"
            return attempt
        matrices = mv.extract_matrix_data_from_parameters(parameters)

        if "E" in matrices:
            policy_data["E"] = matrices["E"]

        if not policy_data:
            attempt.status = "skipped"
            attempt.error_message = "No policy data found"
            return attempt

        num_policies = len(policy_data)
        fig, axes = plt.subplots(1, num_policies, figsize=(5 * num_policies, 5))
        if num_policies == 1:
            axes = [axes]

        for idx, (name, data) in enumerate(policy_data.items()):
            if isinstance(data, np.ndarray):
                policy_vec = data.flatten()
            elif isinstance(data, dict):
                dims = data.get("dimensions", [3])
                policy_vec = np.ones(dims[0]) / dims[0]
            else:
                continue

            axes[idx].bar(range(len(policy_vec)), policy_vec, alpha=0.7)
            axes[idx].set_title(f"Policy Distribution: {name}")
            axes[idx].set_xlabel("Action Index")
            axes[idx].set_ylabel("Probability")
            axes[idx].set_ylim(0, 1)
            axes[idx].grid(True, alpha=0.3)

        plt.suptitle(
            f"Policy Visualization: {model_name}", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()

        output_file = output_dir / f"{model_name}_policy_visualization.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        attempt.status = "success"
        attempt.output_files = [str(output_file)]

    except Exception as e:
        logger.error(f"Failed to generate policy visualization for {model_name}: {e}")
        attempt.status = "failed"
        attempt.error_message = str(e)
    finally:
        attempt.duration_ms = (time.time() - start_time) * 1000

    return attempt


def _generate_network_metrics(
    model_name: str,
    model_data: Dict[str, Any],
    output_dir: Path,
    dependencies: Dict[str, bool],
    logger: logging.Logger,
) -> AdvancedVisualizationAttempt:
    """Generate network analysis metrics and visualizations"""
    model_name = safe_output_stem(model_name)
    attempt = AdvancedVisualizationAttempt(
        viz_type="network_metrics", model_name=model_name, status="in_progress"
    )

    start_time = time.time()

    try:
        if not MATPLOTLIB_AVAILABLE or not np:
            attempt.status = "skipped"
            attempt.error_message = "matplotlib/numpy not available"
            return attempt

        try:
            import networkx as nx

            nx_available = True
        except ImportError:
            nx_available = False

        variables = model_data.get("variables", [])
        connections = model_data.get("connections", [])

        if not variables or not connections:
            attempt.status = "skipped"
            attempt.error_message = "Insufficient network data"
            return attempt

        output_files: list[Any] = []

        if nx_available:
            G = nx.DiGraph()

            for var in variables:
                if isinstance(var, dict):
                    G.add_node(var.get("name", "unknown"))

            for conn in connections:
                if isinstance(conn, dict):
                    sources, targets = _conn_endpoints(conn)
                    for source in sources:
                        for target in targets:
                            if source and target:
                                G.add_edge(source, target)

            metrics: dict[str, Any] = {
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "density": nx.density(G) if G.number_of_nodes() > 1 else 0,
                "avg_clustering": nx.average_clustering(G.to_undirected())
                if G.number_of_nodes() > 1
                else 0,
            }

            if G.number_of_nodes() > 0:
                try:
                    degree_centrality = nx.degree_centrality(G)
                    metrics["max_degree_centrality"] = (
                        max(degree_centrality.values()) if degree_centrality else 0
                    )
                except (nx.NetworkXError, ValueError, ZeroDivisionError) as e:
                    logger.debug("Centrality metrics failed on degenerate graph: %s", e)

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
            nx.draw(
                G,
                pos,
                ax=axes[0],
                with_labels=True,
                node_color="lightblue",
                node_size=500,
                font_size=8,
                arrows=True,
                edge_color="gray",
            )
            axes[0].set_title("Network Graph")

            metric_names = list(metrics.keys())
            metric_values = [metrics[k] for k in metric_names]
            axes[1].bar(range(len(metric_names)), metric_values, alpha=0.7)
            axes[1].set_xticks(range(len(metric_names)))
            axes[1].set_xticklabels(metric_names, rotation=45, ha="right")
            axes[1].set_title("Network Metrics")
            axes[1].set_ylabel("Value")
            axes[1].grid(True, alpha=0.3)

            plt.suptitle(
                f"Network Analysis: {model_name}", fontsize=14, fontweight="bold"
            )
            plt.tight_layout()

            output_file = output_dir / f"{model_name}_network_metrics.png"
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()
            output_files.append(str(output_file))
        else:
            fig, ax = plt.subplots(figsize=(8, 6))

            stats: dict[str, Any] = {
                "Variables": len(variables),
                "Connections": len(connections),
                "Avg Connections/Node": len(connections) / len(variables)
                if variables
                else 0,
            }

            ax.bar(range(len(stats)), list(stats.values()), alpha=0.7)
            ax.set_xticks(range(len(stats)))
            ax.set_xticklabels(list(stats.keys()), rotation=45, ha="right")
            ax.set_title(f"Network Statistics: {model_name}")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            output_file = output_dir / f"{model_name}_network_metrics.png"
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()
            output_files.append(str(output_file))

        attempt.status = "success"
        attempt.output_files = output_files

    except Exception as e:
        logger.error(f"Failed to generate network metrics for {model_name}: {e}")
        attempt.status = "failed"
        attempt.error_message = str(e)
    finally:
        attempt.duration_ms = (time.time() - start_time) * 1000

    return attempt


def _generate_d2_visualizations_safe(
    model_data: Dict[str, Any], output_dir: Path, logger: logging.Logger
) -> AdvancedVisualizationAttempt:
    """
    Generate D2 diagram visualizations for GNN models.

    Args:
        model_data: Parsed GNN model data
        output_dir: Output directory for visualizations
        logger: Logger instance

    Returns:
        AdvancedVisualizationAttempt tracking the generation
    """
    model_name = safe_output_stem(model_data.get("model_name", "unknown_model"))
    attempt = AdvancedVisualizationAttempt(
        viz_type="d2_diagrams", model_name=model_name, status="in_progress"
    )

    start_time = time.time()

    try:
        try:
            from .d2_visualizer import D2Visualizer
        except ImportError:
            logger.warning("D2 visualizer module not available")
            attempt.status = "skipped"
            attempt.error_message = "D2 visualizer not available"
            return attempt

        logger.info(f"Generating D2 diagrams for {model_name}...")

        visualizer = D2Visualizer(logger=logger)

        if not visualizer.d2_available:
            logger.warning("D2 CLI not available. Install from https://d2lang.com")
            attempt.status = "skipped"
            attempt.error_message = "D2 CLI not installed"
            attempt.fallback_used = True
            return attempt

        d2_output_dir = output_dir / "d2_diagrams" / model_name
        d2_output_dir.mkdir(parents=True, exist_ok=True)

        results = visualizer.generate_all_diagrams_for_model(
            model_data, d2_output_dir, formats=["svg", "png"]
        )

        successful = 0
        for result in results:
            if result.success:
                successful += 1
                for output_file in result.output_files:
                    attempt.output_files.append(str(output_file))
                logger.info(f"Generated D2 diagram: {result.diagram_name}")
            else:
                logger.warning(
                    f"Failed D2 diagram {result.diagram_name}: {result.error_message}"
                )

        if successful > 0:
            attempt.status = "success"
            logger.info(f"Generated {successful} D2 diagrams for {model_name}")
        else:
            attempt.status = "failed"
            attempt.error_message = "No D2 diagrams generated successfully"

    except Exception as e:
        logger.error(f"Failed to generate D2 visualizations for {model_name}: {e}")
        attempt.status = "failed"
        attempt.error_message = str(e)
    finally:
        attempt.duration_ms = (time.time() - start_time) * 1000

    return attempt


def _generate_pipeline_d2_diagrams_safe(
    output_dir: Path, logger: logging.Logger
) -> AdvancedVisualizationAttempt:
    """
    Generate D2 diagrams for GNN pipeline architecture.

    Args:
        output_dir: Output directory for visualizations
        logger: Logger instance

    Returns:
        AdvancedVisualizationAttempt tracking the generation
    """
    attempt = AdvancedVisualizationAttempt(
        viz_type="d2_pipeline_diagrams", model_name="gnn_pipeline", status="in_progress"
    )

    start_time = time.time()

    try:
        try:
            from .d2_visualizer import D2Visualizer
        except ImportError:
            logger.warning("D2 visualizer module not available")
            attempt.status = "skipped"
            attempt.error_message = "D2 visualizer not available"
            return attempt

        logger.info("Generating pipeline D2 diagrams...")

        visualizer = D2Visualizer(logger=logger)

        if not visualizer.d2_available:
            logger.warning("D2 CLI not available. Install from https://d2lang.com")
            attempt.status = "skipped"
            attempt.error_message = "D2 CLI not installed"
            attempt.fallback_used = True
            return attempt

        d2_output_dir = output_dir / "d2_diagrams" / "pipeline"
        d2_output_dir.mkdir(parents=True, exist_ok=True)

        flow_spec = visualizer.generate_pipeline_flow_diagram(include_frameworks=True)
        flow_result = visualizer.compile_d2_diagram(
            flow_spec, d2_output_dir, formats=["svg", "png"]
        )

        if flow_result.success:
            for output_file in flow_result.output_files:
                attempt.output_files.append(str(output_file))
            logger.info("Generated pipeline flow diagram")

        framework_spec = visualizer.generate_framework_mapping_diagram()
        framework_result = visualizer.compile_d2_diagram(
            framework_spec, d2_output_dir, formats=["svg", "png"]
        )

        if framework_result.success:
            for output_file in framework_result.output_files:
                attempt.output_files.append(str(output_file))
            logger.info("Generated framework mapping diagram")

        concepts_spec = visualizer.generate_active_inference_concepts_diagram()
        concepts_result = visualizer.compile_d2_diagram(
            concepts_spec, d2_output_dir, formats=["svg", "png"]
        )

        if concepts_result.success:
            for output_file in concepts_result.output_files:
                attempt.output_files.append(str(output_file))
            logger.info("Generated Active Inference concepts diagram")

        total_results: list[Any] = [flow_result, framework_result, concepts_result]
        successful = sum(1 for r in total_results if r.success)

        if successful > 0:
            attempt.status = "success"
            logger.info(f"Generated {successful} pipeline D2 diagrams")
        else:
            attempt.status = "failed"
            attempt.error_message = "No pipeline D2 diagrams generated successfully"

    except Exception as e:
        logger.error(f"Failed to generate pipeline D2 diagrams: {e}")
        attempt.status = "failed"
        attempt.error_message = str(e)
    finally:
        attempt.duration_ms = (time.time() - start_time) * 1000

    return attempt

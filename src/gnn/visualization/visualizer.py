"""
GNN Visualizer Module

This module provides the main visualization functionality for GNN models.
It generates comprehensive state-space visualizations of GNN files and models.
"""

from __future__ import annotations

import datetime
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

from gnn.pipeline.config import get_output_dir_for_script
from gnn.processing.discovery import is_model_source_path
from gnn.utils import performance_tracker

from .matrix import MatrixVisualizer
from .ontology import OntologyVisualizer
from .parse.gnn_file_parser import GNNParser

# Optional dependency imports
# numpy is a required core dependency (pyproject); ``np`` is not used in this
# module, but the capability flag is reported in the capabilities summary.
NUMPY_AVAILABLE = True

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = cast(Any, None)
    MATPLOTLIB_AVAILABLE = False

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    nx = cast(Any, None)
    NETWORKX_AVAILABLE = False

PARSER_AVAILABLE = True
MATRIX_VISUALIZER_AVAILABLE = True
ONTOLOGY_VISUALIZER_AVAILABLE = True

logger = logging.getLogger(__name__)


class GNNVisualizer:
    """
    Visualizer for GNN models.

    This class provides methods to visualize GNN models from parsed GNN files.
    It generates various visualizations of the model's state space, connections,
    and other properties.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        project_root: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Initialize the GNN visualizer.

        Args:
            output_dir: Directory where output visualizations will be saved.
                        If None, creates a timestamped directory in the current working directory.
            project_root: Optional path to the project root for making file paths relative.
        """
        self.parser = GNNParser()
        self.matrix_visualizer = MatrixVisualizer()
        self.ontology_visualizer = OntologyVisualizer()

        # Track what functionality is available
        self.capabilities = {
            "parser": PARSER_AVAILABLE and self.parser is not None,
            "matrix_visualizer": MATRIX_VISUALIZER_AVAILABLE
            and self.matrix_visualizer is not None,
            "ontology_visualizer": ONTOLOGY_VISUALIZER_AVAILABLE
            and self.ontology_visualizer is not None,
            "matplotlib": MATPLOTLIB_AVAILABLE,
            "networkx": NETWORKX_AVAILABLE,
            "numpy": NUMPY_AVAILABLE,
        }

        # Create timestamped output directory if not provided
        # Prefer centralized, numbered step output folder under project `output/`.
        # If no explicit `output_dir` is provided, we place results under
        # `<project_root>/output/8_visualization_output/gnn_visualization_<timestamp>`
        if output_dir is None:
            # Determine project root (assume src/ is current working directory when running steps)
            project_root_output_dir = Path.cwd().parent / "output"
            viz_step_output = get_output_dir_for_script(
                "8_visualization.py", project_root_output_dir
            )
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # Create a timestamped subdirectory inside the standardized step folder
            resolved_output_dir: str | Path = (
                viz_step_output / f"gnn_visualization_{timestamp}"
            )
        else:
            resolved_output_dir = output_dir

        self.output_dir = Path(resolved_output_dir)
        # Ensure parent numeric step directory exists (e.g., 8_visualization_output)
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Best-effort create; if this fails, raise so callers are made aware
            raise

        self.project_root = Path(project_root).resolve() if project_root else None

    def visualize_file(self, file_path: str) -> str:
        """
        Generate visualizations for a GNN file.

        Args:
            file_path: Path to the GNN file to visualize

        Returns:
            Path to the directory containing generated visualizations
        """
        try:
            # Create subdirectory for this file
            file_name = Path(file_path).stem
            file_output_dir = self.output_dir / file_name
            file_output_dir.mkdir(exist_ok=True)

            # Create a capabilities report first
            capabilities_file = file_output_dir / "visualization_capabilities.txt"
            with open(capabilities_file, "w") as f:
                f.write("GNN Visualization Capabilities Report\n")
                f.write("====================================\n\n")
                for capability, available in self.capabilities.items():
                    status = "✓ Available" if available else "✗ Missing"
                    f.write(f"{capability}: {status}\n")
                f.write("\n")

            # Try to parse the GNN file
            parsed_data = None
            if self.capabilities["parser"]:
                try:
                    parsed_data = self.parser.parse_file(file_path)
                except Exception as e:
                    # Fall back to basic file reading
                    parsed_data = {"error": f"Parser failed: {e}"}

            if parsed_data is None:
                # Recovery: basic file analysis
                try:
                    with open(file_path) as f:
                        content = f.read()

                    # Simple text analysis
                    parsed_data = {
                        "file_size": len(content),
                        "line_count": len(content.split("\n")),
                        "variables_found": len(
                            [
                                line
                                for line in content.split("\n")
                                if "var" in line.lower()
                            ]
                        ),
                        "parameters_found": len(
                            [
                                line
                                for line in content.split("\n")
                                if "param" in line.lower()
                            ]
                        ),
                        "content_preview": content[:500] + "..."
                        if len(content) > 500
                        else content,
                    }
                except Exception as e:
                    parsed_data = {"error": f"Failed to read file: {e}"}

            # Generate basic visualizations based on available capabilities
            visualization_count = 0

            # Try matrix visualizations
            if self.capabilities["matrix_visualizer"] and parsed_data.get("parameters"):
                try:
                    matrix_output = file_output_dir / "matrix_analysis.png"
                    if self.matrix_visualizer.generate_matrix_analysis(
                        parsed_data["parameters"], matrix_output
                    ):
                        visualization_count += 1
                except Exception as e:
                    logger.debug(f"Matrix visualization skipped (non-fatal): {e}")

            # Generate basic text summary even if visualizations fail
            summary_file = file_output_dir / "visualization_summary.txt"
            with open(summary_file, "w") as f:
                f.write(f"Visualization Summary for {file_name}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated visualizations: {visualization_count}\n")
                f.write(
                    f"Capabilities available: {sum(self.capabilities.values())}/{len(self.capabilities)}\n\n"
                )

                if isinstance(parsed_data, dict):
                    f.write("Parsed Data Summary:\n")
                    for key, value in parsed_data.items():
                        if key != "content_preview":
                            f.write(f"  {key}: {value}\n")
                f.write("\n")

                f.write("Missing Dependencies:\n")
                for capability, available in self.capabilities.items():
                    if not available:
                        f.write(f"  - {capability}\n")

            return str(file_output_dir)

        except Exception as e:
            # Create error report even if everything fails
            error_dir = self.output_dir / f"{Path(file_path).stem}_error"
            error_dir.mkdir(exist_ok=True)

            error_file = error_dir / "visualization_error.txt"
            with open(error_file, "w") as f:
                f.write("Visualization Error Report\n")
                f.write("=========================\n\n")
                f.write(f"File: {file_path}\n")
                f.write(f"Error: {str(e)}\n")
                f.write(f"Capabilities: {self.capabilities}\n")

            return str(error_dir)

    def visualize_directory(self, dir_path: str) -> str:
        """
        Generate visualizations for all GNN files in a directory.

        Args:
            dir_path: Path to directory containing GNN files

        Returns:
            Path to the directory containing all generated visualizations
        """
        directory_path = Path(dir_path)

        # Process all markdown files in the directory
        for file_path in directory_path.glob("*.md"):
            try:
                self.visualize_file(str(file_path))
            except Exception as e:
                logger.warning("Error processing %s: %s", file_path, e)

        return str(self.output_dir)

    def generate_graph_visualization(
        self, graph_data: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Generate graph visualization."""
        output_dir = self.output_dir / "graph"
        output_dir.mkdir(parents=True, exist_ok=True)
        if MATPLOTLIB_AVAILABLE:
            variables = (graph_data or {}).get("variables", [])
            connections = (graph_data or {}).get("connections", [])
            plt.figure(figsize=(10, 8))
            if variables:
                # Render variables as labeled nodes in a grid
                n = len(variables)
                cols = max(1, int(n**0.5) + 1)
                for idx, var in enumerate(variables):
                    name = (
                        var.get("name", f"v{idx}")
                        if isinstance(var, dict)
                        else str(var)
                    )
                    x, y = idx % cols, -(idx // cols)
                    plt.scatter(
                        x, y, s=400, zorder=3, color="#5B9BD5", edgecolors="black"
                    )
                    plt.annotate(
                        name,
                        (x, y),
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                    )
                plt.title(
                    f"Graph Visualization ({n} variables, {len(connections)} connections)"
                )
            else:
                plt.text(
                    0.5,
                    0.5,
                    "No graph data provided",
                    ha="center",
                    va="center",
                    transform=plt.gca().transAxes,
                    fontsize=12,
                )
                plt.title("Graph Visualization (empty)")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(output_dir / "graph.png", dpi=150, bbox_inches="tight")
            plt.close()
        return {"status": "SUCCESS", "output_dir": str(output_dir)}

    def create_network_diagram(
        self, graph_data: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Create network diagram."""
        return self.generate_graph_visualization(graph_data)

    def _visualize_connections(
        self, parsed_data: Dict[str, Any], output_dir: Path
    ) -> None:
        """Generate visualization of the connections/edges in the model."""
        if "Edges" not in parsed_data or not parsed_data["Edges"]:
            return

        edges = parsed_data["Edges"]

        # Create directed graph
        G = nx.DiGraph()

        try:
            # Add nodes and edges
            for edge in edges:
                source = edge.get("source", "")
                target = edge.get("target", "")
                if not source or not target:
                    continue

                directed = edge.get("directed", True)
                constraint = edge.get("constraint", None)
                comment = edge.get("comment", None)

                G.add_node(source)
                G.add_node(target)

                if directed:
                    G.add_edge(source, target, constraint=constraint, comment=comment)
                else:
                    # For undirected edges in a directed graph, add edges in both directions
                    G.add_edge(source, target, constraint=constraint, comment=comment)
                    G.add_edge(target, source, constraint=constraint, comment=comment)

            # Create figure
            plt.figure(figsize=(12, 10))

            if G.number_of_nodes() > 0:
                # Set node positions using spring layout
                pos = nx.spring_layout(G, seed=42)

                # Draw nodes
                nx.draw_networkx_nodes(
                    G, pos, node_size=700, node_color="lightblue", alpha=0.8
                )

                # Draw edges
                nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7, arrowsize=20)

                # Draw labels
                nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")

                # Add edge labels for constraints
                edge_labels = {
                    (edge.get("source", ""), edge.get("target", "")): edge.get(
                        "constraint", ""
                    )
                    for edge in edges
                    if edge.get("constraint")
                }
                if edge_labels:
                    nx.draw_networkx_edge_labels(
                        G, pos, edge_labels=edge_labels, font_size=10
                    )
            else:
                plt.text(
                    0.5,
                    0.5,
                    "No connections found",
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=14,
                )

            # Set title
            plt.title("Model Connections", fontsize=14, fontweight="bold")

            # Remove axis
            plt.axis("off")

            # Save figure
            plt.tight_layout()
            plt.savefig(output_dir / "connections.png", dpi=150, bbox_inches="tight")
            plt.close()

            logger.info(
                "Connections visualization saved to %s", output_dir / "connections.png"
            )
        except Exception as e:
            # Create error text figure if visualization fails
            plt.figure(figsize=(10, 5))
            plt.text(
                0.5,
                0.5,
                f"Error generating connections visualization: {str(e)}",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=12,
                wrap=True,
            )
            plt.axis("off")
            plt.savefig(output_dir / "connections_error.png", dpi=150)
            plt.close()

    def _extract_parameters_from_parsed_data(
        self, parsed_data: Dict[str, Any]
    ) -> List[Dict]:
        """
        Extract parameters from parsed data for matrix visualization.

        Args:
            parsed_data: Parsed GNN data

        Returns:
            List of parameter dictionaries
        """
        parameters: list[Any] = []

        # Extract from InitialParameterization section
        if "InitialParameterization" in parsed_data:
            init_content = parsed_data["InitialParameterization"]

            # Parse matrix definitions (A, B, C, D, E matrices)
            matrix_pattern = r"([A-Z])\s*=\s*\{([^}]+)\}"
            for match in re.finditer(matrix_pattern, init_content):
                matrix_name = match.group(1)
                matrix_data = match.group(2)

                try:
                    # Convert matrix data to list format
                    matrix_list = self._parse_matrix_string(matrix_data)
                    parameters.append({"name": matrix_name, "value": matrix_list})
                except Exception as e:
                    logger.debug(
                        f"Skipping matrix {matrix_name} due to parse error: {e}"
                    )
                    continue

        return parameters

    def _parse_matrix_string(self, matrix_str: str) -> List[List[float]]:
        """
        Parse matrix string into list format.

        Args:
            matrix_str: Matrix data as string

        Returns:
            List representation of matrix
        """
        # Remove extra whitespace and newlines
        matrix_str = re.sub(r"\s+", " ", matrix_str.strip())

        # Parse nested tuples
        matrix_str = matrix_str.replace("(", "[").replace(")", "]")

        # Convert to Python list structure
        matrix_str = matrix_str.replace("[", "[").replace("]", "]")

        # Evaluate as Python expression
        from gnn.utils.runtime_safety.safe_eval import (
            MATRIX_MAX_LEN,
            safe_literal_eval,
        )

        matrix_data = safe_literal_eval(matrix_str, max_len=MATRIX_MAX_LEN)

        return cast("list[list[float]]", matrix_data)


def generate_graph_visualization(gnn_data: Dict[str, Any], output_path: str) -> bool:
    """
    Generate a graph visualization from GNN data.

    Args:
        gnn_data: Parsed GNN data dictionary
        output_path: Path where the visualization should be saved

    Returns:
        True if successful, False otherwise
    """
    try:
        visualizer = GNNVisualizer()
        visualizer._visualize_connections(gnn_data, Path(output_path).parent)
        return True
    except Exception as e:
        logger.warning("Error generating graph visualization: %s", e)
        return False


def generate_matrix_visualization(gnn_data: Dict[str, Any], output_path: str) -> bool:
    """
    Generate matrix visualizations from GNN data.

    Args:
        gnn_data: Parsed GNN data dictionary
        output_path: Path where the visualization should be saved

    Returns:
        True if successful, False otherwise
    """
    try:
        visualizer = GNNVisualizer()
        # Extract parameters and generate matrix visualizations
        parameters = visualizer._extract_parameters_from_parsed_data(gnn_data)
        if parameters:
            output_dir = Path(output_path).parent
            visualizer.matrix_visualizer.generate_matrix_analysis(
                parameters, output_dir / "matrix_analysis.png"
            )
            visualizer.matrix_visualizer.generate_matrix_statistics(
                parameters, output_dir / "matrix_statistics.png"
            )
        return True
    except Exception as e:
        logger.warning("Error generating matrix visualization: %s", e)
        return False


def create_visualization_report(gnn_file_path: str, output_dir: str) -> str:
    """
    Create a comprehensive visualization report for a GNN file.

    Args:
        gnn_file_path: Path to the GNN file
        output_dir: Output directory for visualizations

    Returns:
        Path to the generated report
    """
    try:
        visualizer = GNNVisualizer(output_dir=output_dir)
        result_path = visualizer.visualize_file(gnn_file_path)
        return result_path
    except Exception as e:
        logger.warning("Error creating visualization report: %s", e)
        return ""


def visualize_gnn_model(gnn_content: str, model_name: str, output_dir: str) -> dict:
    """
    Visualize a GNN model from content string.

    Args:
        gnn_content: GNN model content as string
        model_name: Name of the model
        output_dir: Output directory for visualizations

    Returns:
        Dictionary with visualization result information
    """
    import tempfile

    try:
        # Create temporary file for parsing
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(gnn_content)
            temp_path = f.name

        # Create visualizations
        visualizer = GNNVisualizer(output_dir=output_dir)
        result_path = visualizer.visualize_file(temp_path)

        return {
            "success": True,
            "model_name": model_name,
            "output_directory": result_path,
            "message": "Visualization generated successfully",
        }

    except Exception as e:
        return {
            "success": False,
            "model_name": model_name,
            "error": str(e),
            "error_type": type(e).__name__,
        }
    finally:
        # Clean up temporary file
        if "temp_path" in locals():
            os.unlink(temp_path)


def generate_visualizations(
    logger: logging.Logger,
    target_dir: Path,
    output_dir: Path,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> bool:
    """
    Generate visualizations for GNN models.

    Args:
        target_dir: Directory containing GNN files to visualize
        output_dir: Output directory for results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional visualization options

    Returns:
        True if visualization succeeded, False otherwise
    """
    from contextlib import contextmanager

    from gnn.utils.logging_utils import (
        log_step_error,
        log_step_start,
        log_step_success,
        log_step_warning,
    )

    @contextmanager
    def _noop_context() -> Any:
        """Handle noop context for internal callers."""
        yield

    log_step_start(logger, f"Generating visualizations for GNN files in: {target_dir}")

    # Use centralized output directory configuration
    viz_output_dir = get_output_dir_for_script("8_visualization.py", output_dir)

    try:
        # Create GNN visualizer instance
        gnn_visualizer = GNNVisualizer(output_dir=str(viz_output_dir))

        # Initialize results dictionary
        results: dict[str, Any] = {"success": False, "files_processed": 0}

        # Use performance tracking for visualization generation
        ctx = (
            performance_tracker.track_operation("generate_all_visualizations")
            if performance_tracker
            else _noop_context()
        )
        with ctx:
            # Find GNN files
            if recursive:
                gnn_files = list(target_dir.rglob("*.md"))
            else:
                gnn_files = list(target_dir.glob("*.md"))
            gnn_files = [path for path in gnn_files if is_model_source_path(path)]

            log_step_success(logger, f"Found {len(gnn_files)} GNN files to visualize")

            # Process each file
            processed_count = 0
            for gnn_file in gnn_files:
                try:
                    output_path = gnn_visualizer.visualize_file(str(gnn_file))
                    log_step_success(
                        logger,
                        f"Generated visualization for {gnn_file.name}: {output_path}",
                    )
                    processed_count += 1
                except Exception as e:
                    log_step_warning(
                        logger, f"Failed to visualize {gnn_file.name}: {e}"
                    )

            results["files_processed"] = processed_count
            results["success"] = processed_count > 0

        # Generate matrix visualizations if available
        if MATRIX_VISUALIZER_AVAILABLE and MatrixVisualizer is not None:
            try:
                ctx2 = (
                    performance_tracker.track_operation(
                        "generate_matrix_visualizations"
                    )
                    if performance_tracker
                    else _noop_context()
                )
                with ctx2:
                    matrix_viz = MatrixVisualizer()
                    matrix_viz.visualize_directory(
                        input_dir=target_dir, output_dir=viz_output_dir / "matrices"
                    )
                log_step_success(logger, "Matrix visualizations completed")
            except Exception as e:
                log_step_warning(logger, f"Matrix visualization failed: {e}")

        # Generate ontology visualizations if available
        if ONTOLOGY_VISUALIZER_AVAILABLE and OntologyVisualizer is not None:
            try:
                ctx3 = (
                    performance_tracker.track_operation(
                        "generate_ontology_visualizations"
                    )
                    if performance_tracker
                    else _noop_context()
                )
                with ctx3:
                    ontology_viz = OntologyVisualizer()
                    ontology_viz.visualize_directory(
                        input_dir=target_dir, output_dir=viz_output_dir / "ontology"
                    )
                log_step_success(logger, "Ontology visualizations completed")
            except Exception as e:
                log_step_warning(logger, f"Ontology visualization failed: {e}")

        # Log results summary
        if results.get("success", False):
            log_step_success(
                logger,
                f"Visualization generation completed successfully. Files processed: {results.get('files_processed', 0)}",
            )
        else:
            log_step_warning(
                logger,
                f"Visualization generation completed with issues. Files processed: {results.get('files_processed', 0)}",
            )

        return cast("bool", results.get("success", False))

    except Exception as e:
        log_step_error(logger, f"Visualization generation failed: {e}")
        return False

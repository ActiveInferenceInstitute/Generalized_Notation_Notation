"""
DisCoPy Analysis Module

Per-framework analysis and visualization for DisCoPy categorical diagrams.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Import shared visualization utilities (centralized matplotlib setup)
from ..viz_base import MATPLOTLIB_AVAILABLE, np, plt


def generate_analysis_from_logs(
    execution_dir: Path, output_dir: Path, verbose: bool = False
) -> List[str]:
    """
    Generate analysis and visualizations from DisCoPy execution logs.

    Args:
        execution_dir: Directory containing execution results
        output_dir: Directory to save visualizations
        verbose: Enable verbose logging

    Returns:
        List of generated visualization file paths
    """
    visualizations: list[Any] = []

    try:
        # Find DisCoPy execution results in model subdirectories
        # Pattern: execution_dir/*/discopy/ (e.g., actinf_pomdp_agent/discopy)
        discopy_dirs = list(execution_dir.glob("*/discopy"))

        if verbose:
            logger.info(f"Searching for DisCoPy results in {execution_dir}")
            logger.info(f"Found {len(discopy_dirs)} DisCoPy directories")

        for discopy_dir in discopy_dirs:
            model_name = discopy_dir.parent.name
            sim_data_dir = discopy_dir / "simulation_data"

            if sim_data_dir.exists():
                # Load circuit analysis results - look for any JSON files
                analysis_files = list(sim_data_dir.glob("*circuit*.json"))
                if verbose:
                    logger.info(
                        f"Found {len(analysis_files)} circuit files in {sim_data_dir}"
                    )

                for analysis_file in analysis_files:
                    try:
                        with open(analysis_file) as f:
                            data = json.load(f)

                        # Add metadata from file
                        data["source_file"] = str(analysis_file)

                        viz_files = create_discopy_visualizations(
                            data, output_dir, model_name, verbose
                        )
                        visualizations.extend(viz_files)
                        logger.info(f"Processed {analysis_file.name} for {model_name}")

                    except Exception as e:
                        logger.warning(f"Failed to process {analysis_file}: {e}")
            else:
                if verbose:
                    logger.debug(f"No simulation_data directory in {discopy_dir}")

            # Also collect existing diagram images from execution output
            viz_dir = discopy_dir / "visualizations"
            if viz_dir.exists():
                for img_file in viz_dir.glob("*.png"):
                    visualizations.append(str(img_file))

            # Check for discopy_diagrams folder (created by execution)
            diagrams_dir = discopy_dir / "discopy_diagrams"
            if diagrams_dir.exists():
                for img_file in diagrams_dir.glob("*.png"):
                    visualizations.append(str(img_file))

    except Exception as e:
        logger.error(f"DisCoPy analysis failed: {e}")

    return visualizations


def create_discopy_visualizations(
    data: Dict[str, Any], output_dir: Path, model_name: str, verbose: bool = False
) -> List[str]:
    """
    Create visualizations from DisCoPy circuit analysis data.

    Handles multiple data formats:
    - circuit_info.json: has 'components', 'analysis', 'parameters'
    - circuit_analysis.json: has 'num_components', 'loop_domain', etc.

    Args:
        data: Circuit analysis dictionary
        output_dir: Output directory
        model_name: Name of the model
        verbose: Enable verbose logging

    Returns:
        List of generated file paths
    """
    visualizations: list[Any] = []

    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available, skipping DisCoPy visualizations")
        return visualizations

    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract circuit info from various possible data formats
    components = data.get("components", data.get("boxes", []))
    analysis = data.get("analysis", {})
    parameters = data.get("parameters", {})

    # Get counts from either format
    num_components = analysis.get(
        "num_components", data.get("num_components", len(components))
    )

    # Skip if no meaningful data
    if not components and num_components == 0:
        if verbose:
            logger.debug(
                f"No components found in data from {data.get('source_file', 'unknown')}"
            )
        return visualizations

    # 1. Circuit Summary (existing visualization)
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Component list
        ax1 = axes[0]
        if components:
            # Components are strings in actual data
            y_pos = range(len(components))
            ax1.barh(y_pos, [1] * len(components), color="steelblue", alpha=0.7)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(components)
            ax1.set_xlabel("Present", fontweight="bold")
            ax1.set_title(
                f"DisCoPy Components ({len(components)} total)", fontweight="bold"
            )
        else:
            ax1.text(
                0.5,
                0.5,
                f"Components: {num_components}",
                ha="center",
                va="center",
                fontsize=14,
            )
            ax1.set_title("Components", fontweight="bold")

        # Right: Circuit structure info
        ax2 = axes[1]

        # Build info text from analysis and parameters
        info_lines: list[Any] = []

        if parameters:
            info_lines.append(f"States: {parameters.get('num_states', 'N/A')}")
            info_lines.append(
                f"Observations: {parameters.get('num_observations', 'N/A')}"
            )
            info_lines.append(f"Actions: {parameters.get('num_actions', 'N/A')}")

        if analysis:
            info_lines.append("")
            info_lines.append(
                f"Loop: {analysis.get('loop_domain', '?')} → {analysis.get('loop_codomain', '?')}"
            )
            info_lines.append(
                f"Model: {analysis.get('model_domain', '?')} → {analysis.get('model_codomain', '?')}"
            )

        if data.get("model_name"):
            info_lines.insert(0, f"Model: {data.get('model_name')}")
            info_lines.insert(1, "")

        info_text = "\n".join(info_lines) if info_lines else "No additional info"
        ax2.text(
            0.5,
            0.5,
            info_text,
            ha="center",
            va="center",
            fontsize=12,
            fontfamily="monospace",
            wrap=True,
        )
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis("off")
        ax2.set_title("Circuit Structure", fontweight="bold")

        plt.suptitle(
            f"DisCoPy Categorical Analysis - {model_name}",
            fontweight="bold",
            fontsize=14,
        )
        plt.tight_layout()

        viz_file = output_dir / f"{model_name}_discopy_circuit_summary.png"
        plt.savefig(viz_file, dpi=300, bbox_inches="tight")
        plt.close()
        visualizations.append(str(viz_file))
        logger.info(f"Generated circuit summary: {viz_file.name}")
    except Exception as e:
        logger.warning(f"Failed to create circuit summary: {e}")

    # 2. Component Flow Network Diagram
    if components and len(components) >= 2:
        try:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Define component categories for layout
            [c for c in components if "matrix" in c.lower()]
            [c for c in components if "vector" in c.lower()]
            [
                c
                for c in components
                if "inference" in c.lower() or "selection" in c.lower()
            ]

            # All components in flow order
            all_comps = components if components else []
            n_comps = len(all_comps)

            if n_comps > 0:
                # Create flow layout - curved path
                angles = np.linspace(np.pi, 0, n_comps)
                radius = 0.35
                center_x, center_y = 0.5, 0.4

                # Colors by component type
                colors: list[Any] = []
                for comp in all_comps:
                    if "matrix" in comp.lower():
                        colors.append("#4CAF50")  # Green
                    elif "vector" in comp.lower():
                        colors.append("#2196F3")  # Blue
                    elif "inference" in comp.lower():
                        colors.append("#FF9800")  # Orange
                    elif "selection" in comp.lower():
                        colors.append("#E91E63")  # Pink
                    else:
                        colors.append("#9C27B0")  # Purple

                # Draw nodes
                positions: list[Any] = []
                for _, (angle, comp, color) in enumerate(
                    zip(angles, all_comps, colors)
                ):
                    x = center_x + radius * np.cos(angle)
                    y = center_y + radius * np.sin(angle)
                    positions.append((x, y))

                    # Draw node circle
                    circle = plt.Circle(
                        (x, y), 0.06, color=color, alpha=0.8, ec="black", lw=2
                    )
                    ax.add_patch(circle)

                    # Label
                    label_y_offset = 0.12 if y > center_y else -0.12
                    ax.text(
                        x,
                        y + label_y_offset,
                        comp.replace("_", "\n"),
                        ha="center",
                        va="center" if y > center_y else "top",
                        fontsize=8,
                        fontweight="bold",
                    )

                # Draw flow arrows
                for i in range(len(positions) - 1):
                    x1, y1 = positions[i]
                    x2, y2 = positions[i + 1]

                    # Shorten arrow to not overlap circles
                    dx, dy = x2 - x1, y2 - y1
                    length = np.sqrt(dx**2 + dy**2)
                    if length > 0:
                        dx, dy = dx / length * 0.08, dy / length * 0.08
                        ax.annotate(
                            "",
                            xy=(x2 - dx, y2 - dy),
                            xytext=(x1 + dx, y1 + dy),
                            arrowprops={
                                "arrowstyle": "->",
                                "color": "gray",
                                "lw": 2,
                                "connectionstyle": "arc3,rad=0.1",
                            },
                        )

                # Add legend
                legend_items: list[Any] = [
                    ("Matrix", "#4CAF50"),
                    ("Vector", "#2196F3"),
                    ("Inference", "#FF9800"),
                    ("Selection", "#E91E63"),
                ]
                for _, (label, color) in enumerate(legend_items):
                    ax.scatter([], [], c=color, s=100, label=label)
                ax.legend(loc="lower right", fontsize=9)

                # Add domain/codomain if available
                if analysis:
                    domain = analysis.get("model_domain", "O")
                    codomain = analysis.get("model_codomain", "A")
                    ax.text(
                        0.1,
                        0.4,
                        f"Input:\n{domain}",
                        ha="center",
                        va="center",
                        fontsize=12,
                        fontweight="bold",
                        bbox={
                            "boxstyle": "round",
                            "facecolor": "lightgreen",
                            "alpha": 0.5,
                        },
                    )
                    ax.text(
                        0.9,
                        0.4,
                        f"Output:\n{codomain}",
                        ha="center",
                        va="center",
                        fontsize=12,
                        fontweight="bold",
                        bbox={
                            "boxstyle": "round",
                            "facecolor": "lightcoral",
                            "alpha": 0.5,
                        },
                    )

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            ax.set_title(
                f"DisCoPy Component Flow - {model_name}",
                fontweight="bold",
                fontsize=14,
                pad=20,
            )

            viz_file = output_dir / f"{model_name}_discopy_component_flow.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated component flow: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create component flow: {e}")

    # 3. Categorical Structure Summary
    if analysis or parameters:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.axis("off")

            # Create summary card
            summary_lines: list[Any] = [
                "╔══════════════════════════════════════════╗",
                "║     CATEGORICAL DIAGRAM STRUCTURE        ║",
                "╠══════════════════════════════════════════╣",
            ]

            if data.get("model_name"):
                summary_lines.append(f"║ Model: {data.get('model_name'):<32} ║")

            summary_lines.append("╠══════════════════════════════════════════╣")

            if parameters:
                summary_lines.append(
                    f"║ States:       {str(parameters.get('num_states', 'N/A')):<27} ║"
                )
                summary_lines.append(
                    f"║ Observations: {str(parameters.get('num_observations', 'N/A')):<27} ║"
                )
                summary_lines.append(
                    f"║ Actions:      {str(parameters.get('num_actions', 'N/A')):<27} ║"
                )

            if components:
                summary_lines.append("╠══════════════════════════════════════════╣")
                summary_lines.append(f"║ Components:   {len(components):<27} ║")

            if analysis:
                summary_lines.append("╠══════════════════════════════════════════╣")
                loop_str = f"{analysis.get('loop_domain', '?')} → {analysis.get('loop_codomain', '?')}"
                model_str = f"{analysis.get('model_domain', '?')} → {analysis.get('model_codomain', '?')}"
                summary_lines.append(f"║ Loop:         {loop_str:<27} ║")
                summary_lines.append(f"║ Model:        {model_str:<27} ║")

            summary_lines.append("╚══════════════════════════════════════════╝")

            summary_text = "\n".join(summary_lines)
            ax.text(
                0.5,
                0.5,
                summary_text,
                ha="center",
                va="center",
                fontsize=11,
                fontfamily="monospace",
                transform=ax.transAxes,
                bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.8},
            )

            viz_file = output_dir / f"{model_name}_discopy_structure_card.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated structure card: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create structure card: {e}")

    return visualizations


def extract_circuit_data(
    execution_dir: Path, logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Extract DisCoPy circuit data from execution outputs.

    Args:
        execution_dir: Directory containing execution results
        logger: Logger instance

    Returns:
        Dictionary with extracted circuit data
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    data: dict[str, Any] = {
        "boxes": [],
        "wires": [],
        "types": [],
        "n_boxes": 0,
        "n_wires": 0,
        "model_name": "",
        "framework": "discopy",
    }

    try:
        sim_data_dir = execution_dir / "simulation_data"
        if sim_data_dir.exists():
            analysis_files = list(sim_data_dir.glob("*circuit*.json"))
            if analysis_files:
                with open(analysis_files[0]) as f:
                    results = json.load(f)
                data.update(results)

    except Exception as e:
        logger.warning(f"Failed to extract DisCoPy data: {e}")

    return data


def analyze_diagram_structure(diagrams: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the structure of DisCoPy diagrams.

    Args:
        diagrams: List of diagram dictionaries

    Returns:
        Structure analysis summary
    """
    analysis: dict[str, Any] = {
        "total_diagrams": len(diagrams),
        "total_boxes": 0,
        "total_wires": 0,
        "box_types": {},
        "wire_types": {},
        "composition_depth": 0,
    }

    for diagram in diagrams:
        boxes = diagram.get("boxes", [])
        wires = diagram.get("wires", [])

        analysis["total_boxes"] += len(boxes)
        analysis["total_wires"] += len(wires)

        for box in boxes:
            box_type = box if isinstance(box, str) else box.get("type", "unknown")
            analysis["box_types"][box_type] = analysis["box_types"].get(box_type, 0) + 1

    return analysis


__all__: list[Any] = [
    "generate_analysis_from_logs",
    "create_discopy_visualizations",
    "extract_circuit_data",
    "analyze_diagram_structure",
]

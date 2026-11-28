"""
Interactive visualization sub-module.

Provides interactive Plotly dashboards, D2 diagram generation,
and pipeline D2 diagram generation.

Extracted from processor.py for maintainability.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from ._shared import AdvancedVisualizationAttempt, _MatrixVisualizer

logger = logging.getLogger(__name__)


def _generate_interactive_plotly_dashboard(
    model_name: str,
    model_data: Dict[str, Any],
    output_dir: Path,
    export_formats: List[str],
    dependencies: Dict[str, bool],
    logger: logging.Logger,
    *,
    html_suffix: str = "interactive_dashboard",
) -> AdvancedVisualizationAttempt:
    """Generate full interactive Plotly dashboard"""
    attempt = AdvancedVisualizationAttempt(
        viz_type="interactive_dashboard", model_name=model_name, status="in_progress"
    )

    start_time = time.time()

    try:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            plotly_available = True
        except ImportError:
            plotly_available = False

        if not plotly_available:
            attempt.status = "skipped"
            attempt.error_message = "plotly not available"
            return attempt

        variables = model_data.get("variables", [])
        parameters = model_data.get("parameters", [])
        connections = model_data.get("connections", [])

        mv = _MatrixVisualizer()
        if mv is None:
            attempt.status = "skipped"
            attempt.error_message = "MatrixVisualizer not available"
            return attempt
        matrices = mv.extract_matrix_data_from_parameters(parameters)

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Variable Types",
                "Matrix Overview",
                "Network Graph",
                "Model Statistics",
            ),
            specs=[
                [{"type": "pie"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "table"}],
            ],
        )

        var_types: dict[Any, Any] = {}
        for var in variables:
            if isinstance(var, dict):
                vtype = var.get("var_type", "unknown")
                var_types[vtype] = var_types.get(vtype, 0) + 1

        if var_types:
            fig.add_trace(
                go.Pie(
                    labels=list(var_types.keys()),
                    values=list(var_types.values()),
                    name="Types",
                ),
                row=1,
                col=1,
            )

        matrix_names = list(matrices.keys())
        matrix_sizes = [matrices[name].size for name in matrix_names]

        if matrix_names:
            fig.add_trace(
                go.Bar(x=matrix_names, y=matrix_sizes, name="Matrix Sizes"),
                row=1,
                col=2,
            )

        if connections:
            x_coords: list[Any] = []
            y_coords: list[Any] = []
            labels: list[Any] = []

            for i, conn in enumerate(connections[:20]):
                if isinstance(conn, dict):
                    source = str(
                        conn.get("source_variables", [conn.get("source", "")])[0]
                        if conn.get("source_variables")
                        else conn.get("source", "")
                    )
                    target = str(
                        conn.get("target_variables", [conn.get("target", "")])[0]
                        if conn.get("target_variables")
                        else conn.get("target", "")
                    )
                    x_coords.append(i % 5)
                    y_coords.append(i // 5)
                    labels.append(f"{source}\u2192{target}")

            if x_coords:
                fig.add_trace(
                    go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode="markers+text",
                        text=labels,
                        textposition="middle center",
                        name="Connections",
                    ),
                    row=2,
                    col=1,
                )

        stats_data: dict[str, Any] = {
            "Metric": ["Variables", "Parameters", "Connections", "Matrices"],
            "Count": [len(variables), len(parameters), len(connections), len(matrices)],
        }

        fig.add_trace(
            go.Table(
                header={"values": list(stats_data.keys())},
                cells={"values": [stats_data[k] for k in stats_data.keys()]},
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title_text=f"Interactive Dashboard: {model_name}",
            height=800,
            showlegend=True,
        )

        if "html" in export_formats:
            output_file = output_dir / f"{model_name}_{html_suffix}.html"
            fig.write_html(str(output_file))
            attempt.output_files.append(str(output_file))

        if "json" in export_formats:
            output_file = output_dir / f"{model_name}_interactive_dashboard.json"
            fig.write_json(str(output_file))
            attempt.output_files.append(str(output_file))

        if "png" in export_formats:
            output_file = output_dir / f"{model_name}_interactive_dashboard.png"
            fig.write_image(str(output_file), width=1200, height=800)
            attempt.output_files.append(str(output_file))

        if attempt.output_files:
            attempt.status = "success"
        else:
            attempt.status = "failed"
            attempt.error_message = (
                "No supported interactive export format was requested"
            )

    except Exception as e:
        logger.error(f"Failed to generate interactive dashboard for {model_name}: {e}")
        attempt.status = "failed"
        attempt.error_message = str(e)
    finally:
        attempt.duration_ms = (time.time() - start_time) * 1000

    return attempt


# D2 functions live in network_viz.py — do not duplicate here.

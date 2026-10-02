"""
HTML report generation for the resource estimator.
"""

import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

MATPLOTLIB_AVAILABLE = False
try:
    from gnn.utils.system_env.matplotlib_setup import apply_env_backend_if_set

    apply_env_backend_if_set()
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError as exc:
    logger.debug("Matplotlib unavailable for estimation HTML plots: %s", exc)


def generate_html_report(
    results: Dict[str, Any], detailed_metrics: Dict[str, Any], output_dir: str
) -> str:
    """Generate a detailed HTML report with visualizations."""

    if not results:
        return "No results to report."

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if we have any valid results
    valid_results = [
        r
        for r in results.values()
        if "error" not in r and r["memory_estimate"] is not None
    ]
    if not valid_results:
        return "No valid results to report."

    html_report_path = output_path / "resource_report_detailed.html"

    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GNN Resource Estimation Report</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; background-color: #f8f9fa; }
            h1, h2, h3 { color: #2c3e50; }
            h1 { border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            .summary-cards { display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 30px; }
            .card { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); flex: 1; min-width: 250px; border-left: 5px solid #3498db; }
            .card-title { font-size: 0.9em; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
            .card-value { font-size: 1.8em; font-weight: bold; color: #2c3e50; }
            .visualizations { display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 30px; }
            .viz-container { background: white; border-radius: 8px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); flex: 1; min-width: 300px; text-align: center; }
            .viz-container img { max-width: 100%; height: auto; border-radius: 4px; }
            .model-section { background: white; border-radius: 8px; padding: 25px; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
            th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; font-weight: 600; color: #333; }
            tr:hover { background-color: #f5f5f5; }
            .metric-definition { background: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 4px solid #3498db; }
            .error-message { color: #e74c3c; font-weight: bold; padding: 10px; background: #fadbd8; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>GNN Resource Estimation Report</h1>
    """

    # Add summary statistics
    avg_memory = sum(r["memory_estimate"] for r in valid_results) / len(valid_results)
    avg_inference = sum(r["inference_estimate"] for r in valid_results) / len(
        valid_results
    )
    avg_storage = sum(r["storage_estimate"] for r in valid_results) / len(valid_results)

    html_content += f"""
        <div class="summary-cards">
            <div class="card">
                <div class="card-title">Models Analyzed</div>
                <div class="card-value">{len(results)}</div>
            </div>
            <div class="card" style="border-left-color: #3498db;">
                <div class="card-title">Avg Memory Usage</div>
                <div class="card-value">{avg_memory:.2f} KB</div>
            </div>
            <div class="card" style="border-left-color: #2ecc71;">
                <div class="card-title">Avg Inference Time</div>
                <div class="card-value">{avg_inference:.2f} units</div>
            </div>
            <div class="card" style="border-left-color: #e74c3c;">
                <div class="card-title">Avg Storage Needed</div>
                <div class="card-value">{avg_storage:.2f} KB</div>
            </div>
        </div>
    """

    # Add visualizations if available
    _generate_visualizations_for_html(results, output_path)

    html_content += """
        <h2>Comparative Analysis</h2>
        <div class="visualizations">
    """

    viz_files: list[Any] = [
        ("memory_usage_html.png", "Memory Usage"),
        ("inference_time_html.png", "Inference Time"),
        ("storage_requirements_html.png", "Storage Requirements"),
    ]

    for viz_file, viz_title in viz_files:
        if (output_path / viz_file).exists():
            html_content += f"""
            <div class="viz-container">
                <h3>{viz_title}</h3>
                <img src="{viz_file}" alt="{viz_title}">
            </div>
            """

    html_content += "</div>"

    # Detailed section for each model
    html_content += "<h2>Detailed Model Analysis</h2>"

    for file_path, res in results.items():
        model_name = Path(file_path).name

        if "error" in res:
            html_content += f"""
            <div class="model-section">
                <h3>{model_name}</h3>
                <p><strong>Path:</strong> {file_path}</p>
                <div class="error-message">Error: {res["error"]}</div>
            </div>
            """
            continue

        metrics = detailed_metrics.get(file_path, {})

        html_content += f"""
        <div class="model-section">
            <h3>{model_name}</h3>
            <p><strong>Path:</strong> {file_path}</p>
            <p><strong>Model Type:</strong> {metrics.get("model_type", "Unknown")}</p>
            
            <h4>Core Resource Estimates</h4>
            <table>
                <tr><th>Metric</th><th>Estimate</th><th>Description</th></tr>
                <tr><td>Memory Footprint</td><td>{res["memory_estimate"]:.2f} KB</td><td>RAM needed for variables</td></tr>
                <tr><td>Inference Cost</td><td>{res["inference_estimate"]:.2f} units</td><td>Relative computational cost</td></tr>
                <tr><td>Storage Required</td><td>{res["storage_estimate"]:.2f} KB</td><td>Disk space for model file</td></tr>
                <tr><td>FLOPS Estimate</td><td>{_metric_scalar(metrics, "flops_estimate", "total_flops"):.2e}</td><td>Floating-point operations per inference</td></tr>
                <tr><td>Est. Inference Time</td><td>{_metric_scalar(metrics, "inference_time_estimate", "cpu_time_seconds") * 1000:.4f} ms</td><td>Approximate time on typical CPU</td></tr>
            </table>
        </div>
        """

    html_content += """
    </body>
    </html>
    """

    with open(html_report_path, "w") as f:
        f.write(html_content)

    return str(html_report_path)


def _generate_visualizations_for_html(
    results: Dict[str, Any], output_dir: Path
) -> None:
    """Generate visualizations specifically for HTML embedding."""
    if not results or not MATPLOTLIB_AVAILABLE:
        if not MATPLOTLIB_AVAILABLE:
            logger.debug("Skipping estimation HTML plots: matplotlib unavailable")
        return

    # Extract data for plots
    files = [Path(file_path).name for file_path in results.keys()]
    memory_values = [
        result["memory_estimate"]
        for result in results.values()
        if "error" not in result and result["memory_estimate"] is not None
    ]
    inference_values = [
        result["inference_estimate"]
        for result in results.values()
        if "error" not in result and result["inference_estimate"] is not None
    ]
    storage_values = [
        result["storage_estimate"]
        for result in results.values()
        if "error" not in result and result["storage_estimate"] is not None
    ]

    if not memory_values or not inference_values or not storage_values:
        return

    # Short file names for better display
    short_files = [
        f[:20] + "..." if len(f) > 20 else f for f in files[: len(memory_values)]
    ]

    # Memory usage plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(short_files, memory_values, color="skyblue")
    for bar in bars:
        height = bar.get_height()
        # Offset in points, not data units: a +0.01 data offset is unbounded
        # in pixels when the estimates (and so the y-range) are near zero.
        plt.annotate(
            f"{height:.2f}",
            (bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.title("Memory Usage Estimates", fontsize=14, fontweight="bold")
    plt.xlabel("Model File", fontsize=12)
    plt.ylabel("Memory Usage (KB)", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "memory_usage_html.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Inference time plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(short_files, inference_values, color="lightgreen")
    for bar in bars:
        height = bar.get_height()
        # Offset in points, not data units: a +0.01 data offset is unbounded
        # in pixels when the estimates (and so the y-range) are near zero.
        plt.annotate(
            f"{height:.2f}",
            (bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.title("Inference Time Estimates", fontsize=14, fontweight="bold")
    plt.xlabel("Model File", fontsize=12)
    plt.ylabel("Inference Time (units)", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "inference_time_html.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Storage requirements plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(short_files, storage_values, color="salmon")
    for bar in bars:
        height = bar.get_height()
        # Offset in points, not data units: a +0.01 data offset is unbounded
        # in pixels when the estimates (and so the y-range) are near zero.
        plt.annotate(
            f"{height:.2f}",
            (bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.title("Storage Requirements Estimates", fontsize=14, fontweight="bold")
    plt.xlabel("Model File", fontsize=12)
    plt.ylabel("Storage (KB)", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(
        output_dir / "storage_requirements_html.png", dpi=120, bbox_inches="tight"
    )
    plt.close()


def _metric_scalar(
    metrics: Dict[str, Any], key: str, subkey: str, default: float = 0.0
) -> float:
    """Return a numeric metric, reaching into a dict when needed.

    ``detailed_metrics`` stores ``flops_estimate`` and
    ``inference_time_estimate`` as structured dicts (from
    :func:`estimate_flops` / :func:`estimate_inference_time`); the HTML
    template formats a scalar. This helper tolerates either shape so the
    report renders regardless of how a caller populated the metrics.
    """
    value = metrics.get(key, default)
    if isinstance(value, dict):
        value = value.get(subkey, default)
    if isinstance(value, (int, float)):
        return float(value)
    return default

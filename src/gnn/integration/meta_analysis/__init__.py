"""
Meta-Analysis: Parameter Sweep Runtime & Simulation Analysis.

This submodule provides comprehensive analysis and visualization of GNN pipeline
parameter sweep outputs, including:

- Runtime scaling analysis across (N, T) parameter grids
- Cross-framework performance comparison heatmaps
- Simulation metric extraction (VFE, EFE, belief accuracy)
- Scaling law regression and visualization
- Markdown report generation

Architecture:
    integration/
    ├── meta_analysis/
    │   ├── __init__.py      ← this file
    │   ├── collector.py     ← data collection from execution outputs
    │   ├── statistics.py  ← aggregate JSON summaries (meta_statistics.json)
    │   ├── validator.py   ← sweep grid / benchmark coherence checks
    │   ├── visualizer.py    ← matplotlib visualization generation
    │   └── reporter.py      ← markdown report generation
"""

from typing import Any

from .collector import SweepDataCollector, SweepRecord
from .reporter import SweepReporter
from .statistics import compute_meta_statistics
from .validator import validate_sweep_records
from .visualizer import SweepVisualizer

__all__: list[Any] = [
    "SweepDataCollector",
    "SweepRecord",
    "SweepVisualizer",
    "SweepReporter",
    "compute_meta_statistics",
    "validate_sweep_records",
    "run_meta_analysis",
]


def run_meta_analysis(
    execute_output_dir: Any,
    output_dir: Any,
    render_output_dir: Any = None,
    logger: Any = None,
    verbose: Any = False,
) -> Any:
    """Run the full meta-analysis pipeline on execution outputs.

    Args:
        execute_output_dir: Path to 12_execute_output directory.
        output_dir: Directory to write meta-analysis results.
        render_output_dir: Optional path to 11_render_output directory.
        logger: Optional logger instance.
        verbose: Enable verbose logging.

    Returns:
        dict with keys ``records``, ``plots``, ``report``, ``validation_json``, ``statistics_json``, or None on failure.
    """
    import json
    import logging
    from pathlib import Path

    from .statistics import compute_meta_statistics
    from .validator import validate_sweep_records

    if logger is None:
        logger = logging.getLogger("integration.meta_analysis")

    execute_output_dir = Path(execute_output_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect sweep data
    collector = SweepDataCollector(
        execute_output_dir, render_output_dir=render_output_dir, logger=logger
    )
    records = collector.collect()

    if collector.handoff_refusal is not None:
        # Fail closed: stale or mixed-run execution results must not feed
        # sweep statistics. Leave a durable refusal receipt for review.
        (output_dir / "execution_handoff.json").write_text(
            json.dumps(
                {"status": "refused", "reason": collector.handoff_refusal}, indent=2
            ),
            encoding="utf-8",
        )
        logger.error(
            "Meta-analysis refused execution handoff: %s", collector.handoff_refusal
        )
        return None

    if not records:
        logger.warning("No sweep records found — skipping meta-analysis")
        return None

    logger.info(f"Collected {len(records)} sweep records from execution outputs")

    pipeline_root = execute_output_dir.parent
    gnn_stats_path = pipeline_root / "3_gnn_output" / "format_statistics.json"
    gnn_format_statistics = None
    if gnn_stats_path.is_file():
        try:
            gnn_format_statistics = json.loads(
                gnn_stats_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("Could not load format_statistics.json: %s", exc)

    validation_payload = validate_sweep_records(records)
    validation_path = output_dir / "sweep_validation.json"
    validation_path.write_text(
        json.dumps(validation_payload, indent=2), encoding="utf-8"
    )

    statistics_payload = compute_meta_statistics(records)
    statistics_path = output_dir / "meta_statistics.json"
    statistics_path.write_text(
        json.dumps(statistics_payload, indent=2), encoding="utf-8"
    )

    # 2. Generate visualizations
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    visualizer = SweepVisualizer(
        records,
        viz_dir,
        logger=logger,
        gnn_format_statistics=gnn_format_statistics,
    )
    generated_plots = visualizer.generate_all()
    logger.info(f"Generated {len(generated_plots)} meta-analysis visualizations")

    # 3. Generate report
    reporter = SweepReporter(
        records,
        generated_plots,
        output_dir,
        logger=logger,
        validation_payload=validation_payload,
        statistics_payload=statistics_payload,
        gnn_format_statistics=gnn_format_statistics,
    )
    report_path = reporter.generate()
    logger.info(f"Generated meta-analysis report: {report_path}")

    return {
        "records": len(records),
        "plots": generated_plots,
        "report": str(report_path),
        "validation_json": str(validation_path),
        "statistics_json": str(statistics_path),
        "execution_handoff": collector.handoff,
    }

"""
Sweep Visualizer — matplotlib-based plotting for parameter sweep meta-analysis.

Generates publication-quality figures:
- Runtime heatmaps (N × T grids per framework) with proper NaN handling
- Runtime scaling curves (log-log plots)
- Cross-framework comparison bar charts
- Per-step timing comparison
- Simulation metric summaries (accuracy, entropy, EFE convergence)
- Plaintext CSV data export

Fixes and improvements over v1:
- Heatmap grids use the FULL N×T space (from all records), not just the framework's data
- NaN cells shown as hatched gray instead of silently dropped
- Color scale uses log normalization for large dynamic ranges
- CSV plaintext export alongside every visualization

Mechanical split facade: plotting methods live in ``visualizer_*`` sibling
modules as mixins of ``SweepVisualizer``. Shared style state lives in
``visualizer_style``; ``_MPL_AVAILABLE`` stays importable here as the
monkeypatch target.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .collector import SweepRecord
from .visualizer_export import SweepExportMixin
from .visualizer_metric_plots import SweepMetricPlotMixin
from .visualizer_runtime_plots import SweepRuntimePlotMixin
from .visualizer_style import (
    _MPL_AVAILABLE,
)
from .visualizer_summary_plots import SweepSummaryPlotMixin


class SweepVisualizer(
    SweepRuntimePlotMixin,
    SweepMetricPlotMixin,
    SweepSummaryPlotMixin,
    SweepExportMixin,
):
    """Generate visualizations from sweep records."""

    def __init__(
        self,
        records: List[SweepRecord],
        output_dir: Path,
        logger: Optional[logging.Logger] = None,
        gnn_format_statistics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the instance."""
        self.records = records
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        self._gnn_format_statistics = gnn_format_statistics

    def generate_all(self) -> List[str]:
        """Generate all available plots. Returns list of generated file paths."""
        if not _MPL_AVAILABLE:
            self.logger.warning("matplotlib not available — skipping visualizations")
            return []

        generated: List[str] = []

        # Filter to records with valid sweep parameters
        sweep_records = [
            r
            for r in self.records
            if r.num_states is not None and r.num_timesteps is not None
        ]

        if not sweep_records:
            self.logger.info(
                "No sweep-parameterized records found; skipping sweep visualizations"
            )
            return generated

        # Global N and T values across ALL frameworks
        all_n_values = sorted(
            {r.num_states for r in sweep_records if r.num_states is not None}
        )
        all_t_values = sorted(
            {r.num_timesteps for r in sweep_records if r.num_timesteps is not None}
        )

        # Extract unique frameworks with actual runtime data
        runtime_frameworks = sorted(
            {r.framework for r in sweep_records if r.execution_time > 0 and r.success}
        )

        # Each plot is best-effort: a single plot (e.g. a log-scale axis with
        # only non-positive runtime data) must not abort the whole meta-analysis.
        def _try_plot(name: str, fn: Any) -> None:
            try:
                path = fn()
                if path:
                    generated.append(str(path))
            except Exception as exc:  # noqa: BLE001 - best-effort plotting
                self.logger.warning(
                    "Meta-analysis plot '%s' failed (non-fatal): %s", name, exc
                )

        # 0. Export plaintext CSV of all data
        _try_plot("sweep_data_csv", lambda: self._export_csv(sweep_records))

        # 1. Runtime heatmaps per framework (using global grid)
        for fw in runtime_frameworks:
            _try_plot(
                "runtime_heatmap",
                lambda fw=fw: self._plot_runtime_heatmap(
                    sweep_records, fw, all_n_values, all_t_values
                ),
            )

        # 2. Runtime scaling curves
        _try_plot(
            "runtime_scaling",
            lambda: self._plot_runtime_scaling(sweep_records, runtime_frameworks),
        )

        # 3. Cross-framework bar chart
        _try_plot(
            "framework_comparison",
            lambda: self._plot_framework_comparison(sweep_records, runtime_frameworks),
        )

        # 4. Time-per-step comparison
        _try_plot(
            "time_per_step",
            lambda: self._plot_time_per_step(sweep_records, runtime_frameworks),
        )

        # 5. Accuracy comparison
        _try_plot(
            "accuracy_comparison",
            lambda: self._plot_accuracy_comparison(sweep_records, runtime_frameworks),
        )

        # 6. Belief entropy comparison
        _try_plot(
            "entropy_comparison",
            lambda: self._plot_entropy_comparison(sweep_records, runtime_frameworks),
        )

        # 7. Accuracy Heatmap
        for fw in runtime_frameworks:
            _try_plot(
                "accuracy_heatmap",
                lambda fw=fw: self._plot_accuracy_heatmap(
                    sweep_records, fw, all_n_values, all_t_values
                ),
            )

        # 8. Entropy Heatmap
        for fw in runtime_frameworks:
            _try_plot(
                "entropy_heatmap",
                lambda fw=fw: self._plot_entropy_heatmap(
                    sweep_records, fw, all_n_values, all_t_values
                ),
            )

        # 9. 3D Runtime Surface
        for fw in runtime_frameworks:
            _try_plot(
                "3d_runtime_surface",
                lambda fw=fw: self._plot_3d_runtime_surface(
                    sweep_records, fw, all_n_values, all_t_values
                ),
            )

        # 10. Compute Efficiency
        _try_plot(
            "compute_efficiency",
            lambda: self._plot_compute_efficiency(sweep_records, runtime_frameworks),
        )

        # 11. Resource Scaling (LOC)
        _try_plot(
            "resource_scaling",
            lambda: self._plot_resource_scaling(sweep_records, runtime_frameworks),
        )

        # 12. Accuracy-Entropy Correlation
        _try_plot(
            "accuracy_entropy_correlation",
            lambda: self._plot_accuracy_entropy_correlation(
                sweep_records, runtime_frameworks
            ),
        )

        # 13. Inference Throughput vs State Space
        _try_plot(
            "throughput_vs_n",
            lambda: self._plot_throughput_vs_n(sweep_records, runtime_frameworks),
        )

        # 14. Runtime Distribution (violin/box plot per N)
        _try_plot(
            "runtime_distribution",
            lambda: self._plot_runtime_distribution(sweep_records, runtime_frameworks),
        )

        # 15. Scaling Exponent Summary Bar Chart
        _try_plot(
            "scaling_exponent_summary",
            lambda: self._plot_scaling_exponent_summary(
                sweep_records, runtime_frameworks
            ),
        )

        # 16. Code Efficiency (LOC per N² vs N)
        _try_plot(
            "code_efficiency",
            lambda: self._plot_code_efficiency(sweep_records, runtime_frameworks),
        )

        # 17. Comprehensive Dashboard (multi-panel summary)
        _try_plot(
            "comprehensive_dashboard",
            lambda: self._plot_comprehensive_dashboard(
                sweep_records, runtime_frameworks, all_n_values, all_t_values
            ),
        )

        # 18. Accuracy vs Timesteps (convergence)
        _try_plot(
            "accuracy_vs_timesteps",
            lambda: self._plot_accuracy_vs_timesteps(sweep_records, runtime_frameworks),
        )

        # 19. Step 3 serialization footprint (when format_statistics.json provided)
        _try_plot(
            "gnn_serialization_footprint",
            lambda: self._plot_gnn_serialization_footprint(),
        )

        # 20. Benchmark uncertainty bars (when σ > 0)
        _try_plot(
            "runtime_uncertainty",
            lambda: self._plot_runtime_uncertainty(sweep_records, runtime_frameworks),
        )

        return generated

    @staticmethod
    def _safe_log_scale(
        ax: Any, *, x: bool = False, y: bool = False, base: int = 10
    ) -> None:
        """Apply log scale only when the axis data contains positive values.

        A log-scaled axis whose data has no positive values (an empty panel or
        zero/negative runtimes) raises at render time with "Data has no
        positive values". Falling back to linear keeps the plot useful.
        """
        try:
            if x:
                xmin, xmax = ax.get_xlim()
                if xmin > 0 and xmax > 0:
                    ax.set_xscale("log", base=base)
            if y:
                ymin, ymax = ax.get_ylim()
                if ymin > 0 and ymax > 0:
                    ax.set_yscale("log", base=base)
        except (ValueError, TypeError, AttributeError):
            pass

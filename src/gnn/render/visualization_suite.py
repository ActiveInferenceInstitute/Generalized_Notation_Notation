#!/usr/bin/env python3
"""
Visualization Suite and Data Exports
Provides comprehensive visualization and data export utilities for all frameworks
"""

from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np

try:
    import pandas as pd

    _PANDAS_AVAILABLE = True
except ImportError:
    pd = cast(Any, None)
    _PANDAS_AVAILABLE = False
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, cast

try:
    import seaborn as sns

    _SNS_AVAILABLE = True
except ImportError:
    sns = cast(Any, None)
    _SNS_AVAILABLE = False

try:
    import h5py

    _H5PY_AVAILABLE = True
except ImportError:
    h5py = cast(Any, None)
    _H5PY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Set style for consistent, professional plots
plt.style.use("seaborn-v0_8")
if _SNS_AVAILABLE:
    sns.set_palette("husl")


class ComprehensiveDataExporter:
    """Exports simulation data in multiple formats for reproducibility"""

    def __init__(self, output_dir: Path, simulation_name: str) -> None:
        """Initialize the instance."""
        self.output_dir = Path(output_dir)
        self.simulation_name = simulation_name
        self.data_dir = self.output_dir / "data_exports"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def export_all_formats(self, data_dict: Dict[str, Any]) -> List[Path]:
        """Export data in JSON, CSV, HDF5, and pickle formats"""

        exported_files: list[Any] = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. JSON Export (human-readable)
        json_file = self.data_dir / f"{self.simulation_name}_data_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(self._make_json_serializable(data_dict), f, indent=2)
        exported_files.append(json_file)

        # 2. CSV Export (for spreadsheet analysis)
        if "traces" in data_dict:
            csv_file = self.data_dir / f"{self.simulation_name}_traces_{timestamp}.csv"
            self._export_traces_to_csv(data_dict["traces"], csv_file)
            exported_files.append(csv_file)

        # 3. HDF5 Export (for large datasets)
        if _H5PY_AVAILABLE:
            try:
                hdf5_file = (
                    self.data_dir / f"{self.simulation_name}_data_{timestamp}.h5"
                )
                with h5py.File(hdf5_file, "w") as f:
                    self._write_dict_to_hdf5(data_dict, f)
                exported_files.append(hdf5_file)
            except Exception as exc:
                logger.warning("HDF5 export failed, skipping: %s", exc)
        else:
            logger.debug("h5py not available, skipping HDF5 export")

        # 4. Metadata file
        meta_file = self.data_dir / f"{self.simulation_name}_metadata_{timestamp}.json"
        metadata: dict[str, Any] = {
            "simulation_name": self.simulation_name,
            "export_timestamp": datetime.now().isoformat(),
            "exported_files": [str(f.name) for f in exported_files],
            "data_keys": list(data_dict.keys()),
            "export_formats": ["JSON", "CSV", "HDF5", "Metadata"],
        }
        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=2)
        exported_files.append(meta_file)

        return exported_files

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-serializable objects"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        else:
            return obj

    def _export_traces_to_csv(self, traces_dict: Dict, csv_file: Path) -> Any:
        """Export trace data to CSV format"""

        # Flatten traces into tabular format
        rows: list[Any] = []
        max_length = max(
            len(v) if isinstance(v, list) else 1 for v in traces_dict.values()
        )

        for i in range(max_length):
            row: dict[str, Any] = {"step": i + 1}
            for key, values in traces_dict.items():
                if isinstance(values, list) and len(values) > i:
                    if isinstance(values[i], list):
                        # Handle multi-dimensional data
                        for j, val in enumerate(values[i]):
                            row[f"{key}_{j}"] = val
                    else:
                        row[key] = values[i]
                else:
                    row[key] = None
            rows.append(row)

        with open(csv_file, "w", newline="") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

    def _write_dict_to_hdf5(self, data_dict: Dict, hdf5_group: Any) -> Any:
        """Recursively write dictionary to HDF5 group"""
        for key, value in data_dict.items():
            if isinstance(value, dict):
                subgroup = hdf5_group.create_group(key)
                self._write_dict_to_hdf5(value, subgroup)
            elif isinstance(value, list):
                if value and isinstance(value[0], (int, float, complex)):
                    hdf5_group.create_dataset(key, data=np.array(value))
                elif value and isinstance(value[0], list):
                    hdf5_group.create_dataset(key, data=np.array(value))
                else:
                    # Store as string for complex objects
                    hdf5_group.create_dataset(key, data=str(value))
            elif isinstance(value, np.ndarray):
                hdf5_group.create_dataset(key, data=value)
            elif isinstance(value, (int, float, complex)):
                hdf5_group.create_dataset(key, data=value)
            else:
                hdf5_group.create_dataset(key, data=str(value))


class VisualizationSuite:
    """Comprehensive visualization suite with multiple chart types"""

    def __init__(self, output_dir: Path, simulation_name: str) -> None:
        """Initialize the instance."""
        self.output_dir = Path(output_dir)
        self.simulation_name = simulation_name
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # Set up matplotlib for high-quality output
        plt.rcParams["figure.dpi"] = 300
        plt.rcParams["savefig.dpi"] = 300
        plt.rcParams["savefig.bbox"] = "tight"
        plt.rcParams["font.size"] = 10

    def create_comprehensive_suite(
        self, simulation_results: Dict[str, Any]
    ) -> List[Path]:
        """Create comprehensive visualization suite"""

        viz_files: list[Any] = []
        traces = simulation_results.get("traces", {})
        summary = simulation_results.get("summary", {})

        # 1. Time Series Analysis
        if traces:
            viz_files.extend(self._create_time_series_plots(traces))

        # 2. Statistical Analysis
        if traces:
            viz_files.extend(self._create_statistical_plots(traces))

        # 3. Distribution Analysis
        if traces:
            viz_files.extend(self._create_distribution_plots(traces))

        # 4. Correlation Analysis
        if traces:
            viz_files.extend(self._create_correlation_plots(traces))

        # 5. Performance Dashboard
        if traces and summary:
            viz_files.extend(self._create_performance_dashboard(traces, summary))

        # 6. Comparative Analysis
        if traces:
            viz_files.extend(self._create_comparative_plots(traces))

        return viz_files

    def _create_time_series_plots(self, traces: Dict) -> List[Path]:
        """Create comprehensive time series visualizations"""
        viz_files: list[Any] = []

        # Multi-panel time series
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            f"Time Series Analysis - {self.simulation_name}",
            fontsize=16,
            fontweight="bold",
        )

        # Plot 1: Primary traces
        ax1 = axes[0, 0]
        for key, values in traces.items():
            if (
                isinstance(values, list)
                and len(values) > 0
                and isinstance(values[0], (int, float))
            ):
                ax1.plot(values, label=key, linewidth=2, alpha=0.8)
        ax1.set_title("Primary Time Series")
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Value")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Cumulative traces
        ax2 = axes[0, 1]
        for key, values in traces.items():
            if (
                isinstance(values, list)
                and len(values) > 0
                and isinstance(values[0], (int, float))
            ):
                ax2.plot(
                    np.cumsum(values),
                    label=f"{key} (cumulative)",
                    linewidth=2,
                    alpha=0.8,
                )
        ax2.set_title("Cumulative Evolution")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Cumulative Value")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Moving averages
        ax3 = axes[1, 0]
        window = min(5, max(1, len(list(traces.values())[0]) // 3))
        for key, values in traces.items():
            if (
                isinstance(values, list)
                and len(values) > 0
                and isinstance(values[0], (int, float))
            ):
                if _PANDAS_AVAILABLE:
                    moving_avg = (
                        pd.Series(values).rolling(window=window, min_periods=1).mean()
                    )
                else:
                    arr = np.array(values, dtype=float)
                    moving_avg = np.convolve(arr, np.ones(window) / window, mode="same")
                ax3.plot(
                    moving_avg, label=f"{key} (MA-{window})", linewidth=2, alpha=0.8
                )
        ax3.set_title(f"Moving Averages (window={window})")
        ax3.set_xlabel("Time Step")
        ax3.set_ylabel("Moving Average")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Derivatives/Changes
        ax4 = axes[1, 1]
        for key, values in traces.items():
            if (
                isinstance(values, list)
                and len(values) > 1
                and isinstance(values[0], (int, float))
            ):
                changes = np.diff(values)
                ax4.plot(changes, label=f"{key} (Δ)", linewidth=2, alpha=0.8)
        ax4.set_title("Rate of Change")
        ax4.set_xlabel("Time Step")
        ax4.set_ylabel("Change (Δ)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        ts_file = self.viz_dir / f"{self.simulation_name}_COMPREHENSIVE_time_series.png"
        plt.savefig(ts_file)
        plt.close()
        viz_files.append(ts_file)

        return viz_files

    def _create_statistical_plots(self, traces: Dict) -> List[Path]:
        """Create statistical analysis plots"""
        viz_files: list[Any] = []

        # Statistical summary plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            f"Statistical Analysis - {self.simulation_name}",
            fontsize=16,
            fontweight="bold",
        )

        numeric_traces = {
            k: v
            for k, v in traces.items()
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float))
        }

        if numeric_traces:
            # Box plots
            ax1 = axes[0, 0]
            data_for_box = list(numeric_traces.values())
            labels = list(numeric_traces.keys())
            ax1.boxplot(data_for_box, tick_labels=labels, patch_artist=True)
            ax1.set_title("Distribution Summary (Box Plots)")
            ax1.set_ylabel("Value")
            plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

            # Violin plots
            ax2 = axes[0, 1]
            positions = range(1, len(numeric_traces) + 1)
            ax2.violinplot(data_for_box, positions=positions)
            ax2.set_xticks(positions)
            ax2.set_xticklabels(labels, rotation=45, ha="right")
            ax2.set_title("Distribution Shape (Violin Plots)")
            ax2.set_ylabel("Value")

            # Q-Q plots for normality
            ax3 = axes[1, 0]
            from scipy import stats

            for _, (_, values) in enumerate(numeric_traces.items()):
                stats.probplot(values, dist="norm", plot=ax3)
                break  # Just show one for space
            ax3.set_title("Normality Check (Q-Q Plot)")

            # Autocorrelation
            ax4 = axes[1, 1]
            for key, values in list(numeric_traces.items())[
                :3
            ]:  # Limit to 3 for clarity
                if _PANDAS_AVAILABLE:
                    autocorr = (
                        pd.Series(values).autocorr(lag=1) if len(values) > 1 else 0
                    )
                    lags = (
                        range(min(20, len(values) // 2))
                        if len(values) > 20
                        else range(len(values) - 1)
                    )
                    autocorrs = (
                        [pd.Series(values).autocorr(lag=lag) for lag in lags]
                        if lags
                        else [0]
                    )
                else:
                    autocorr = 0
                    lags = range(0)
                    autocorrs = []
                ax4.plot(
                    lags,
                    autocorrs,
                    label=f"{key} (r={autocorr:.3f})",
                    marker="o",
                    alpha=0.8,
                )
            ax4.set_title("Autocorrelation Analysis")
            ax4.set_xlabel("Lag")
            ax4.set_ylabel("Autocorrelation")
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        stat_file = self.viz_dir / f"{self.simulation_name}_STATISTICAL_analysis.png"
        plt.savefig(stat_file)
        plt.close()
        viz_files.append(stat_file)

        return viz_files

    def _create_distribution_plots(self, traces: Dict) -> List[Path]:
        """Create distribution analysis plots"""
        viz_files: list[Any] = []

        # Distribution analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            f"Distribution Analysis - {self.simulation_name}",
            fontsize=16,
            fontweight="bold",
        )

        numeric_traces = {
            k: v
            for k, v in traces.items()
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float))
        }

        if numeric_traces:
            # Histograms
            ax1 = axes[0, 0]
            for key, values in list(numeric_traces.items())[:4]:  # Limit for clarity
                ax1.hist(values, alpha=0.6, label=key, bins=20, density=True)
            ax1.set_title("Probability Distributions")
            ax1.set_xlabel("Value")
            ax1.set_ylabel("Density")
            ax1.legend()

            # Kernel Density Estimation
            ax2 = axes[0, 1]
            for key, values in list(numeric_traces.items())[:4]:
                try:
                    sns.kdeplot(data=values, ax=ax2, label=key, alpha=0.8)
                except Exception as e:
                    logger.debug(f"KDE plot skipped for {key}: {e}")
            ax2.set_title("Kernel Density Estimation")
            ax2.set_xlabel("Value")
            ax2.set_ylabel("Density")
            ax2.legend()

            # Empirical CDF
            ax3 = axes[1, 0]
            for key, values in list(numeric_traces.items())[:4]:
                sorted_values = np.sort(values)
                y = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
                ax3.plot(sorted_values, y, label=key, linewidth=2, alpha=0.8)
            ax3.set_title("Empirical Cumulative Distribution")
            ax3.set_xlabel("Value")
            ax3.set_ylabel("Cumulative Probability")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Distribution comparison
            ax4 = axes[1, 1]
            if len(numeric_traces) >= 2:
                keys = list(numeric_traces.keys())[:2]
                values1, values2 = numeric_traces[keys[0]], numeric_traces[keys[1]]
                ax4.scatter(values1, values2, alpha=0.6, s=30)
                ax4.set_xlabel(keys[0])
                ax4.set_ylabel(keys[1])
                ax4.set_title(f"{keys[0]} vs {keys[1]}")

                # Add correlation coefficient
                if len(values1) == len(values2):
                    corr = np.corrcoef(values1, values2)[0, 1]
                    ax4.text(
                        0.05,
                        0.95,
                        f"r = {corr:.3f}",
                        transform=ax4.transAxes,
                        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
                    )

        plt.tight_layout()
        dist_file = self.viz_dir / f"{self.simulation_name}_DISTRIBUTIONS.png"
        plt.savefig(dist_file)
        plt.close()
        viz_files.append(dist_file)

        return viz_files

    def _create_correlation_plots(self, traces: Dict) -> List[Path]:
        """Create correlation analysis plots"""
        viz_files: list[Any] = []

        # Prepare correlation data
        numeric_traces = {
            k: v
            for k, v in traces.items()
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float))
        }

        if len(numeric_traces) < 2:
            return viz_files

        if not _PANDAS_AVAILABLE:
            return viz_files

        # Create correlation matrix
        trace_df = pd.DataFrame(numeric_traces)
        corr_matrix = trace_df.corr()

        # Correlation heatmap
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(
            f"Correlation Analysis - {self.simulation_name}",
            fontsize=16,
            fontweight="bold",
        )

        # Heatmap
        ax1 = axes[0]
        im = ax1.imshow(corr_matrix, cmap="RdYlBu_r", aspect="auto", vmin=-1, vmax=1)
        ax1.set_xticks(range(len(corr_matrix.columns)))
        ax1.set_yticks(range(len(corr_matrix.index)))
        ax1.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
        ax1.set_yticklabels(corr_matrix.index)
        ax1.set_title("Correlation Matrix")

        # Add correlation values to heatmap
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                ax1.text(
                    j,
                    i,
                    f"{corr_matrix.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                )

        plt.colorbar(im, ax=ax1)

        # Pairwise scatter plots
        ax2 = axes[1]
        if len(numeric_traces) >= 2:
            keys = list(numeric_traces.keys())
            colors = plt.get_cmap("Set3")(np.linspace(0, 1, len(keys)))

            for i, key1 in enumerate(keys):
                for _, key2 in enumerate(keys[i + 1 :], i + 1):
                    if key1 != key2:
                        values1, values2 = numeric_traces[key1], numeric_traces[key2]
                        if len(values1) == len(values2):
                            ax2.scatter(
                                values1,
                                values2,
                                label=f"{key1} vs {key2}",
                                alpha=0.6,
                                s=30,
                                color=colors[i],
                            )

            ax2.set_title("Pairwise Relationships")
            ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        corr_file = self.viz_dir / f"{self.simulation_name}_CORRELATIONS.png"
        plt.savefig(corr_file)
        plt.close()
        viz_files.append(corr_file)

        return viz_files

    def _create_performance_dashboard(self, traces: Dict, summary: Dict) -> List[Path]:
        """Create comprehensive performance dashboard"""
        viz_files: list[Any] = []

        # Performance dashboard
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[2, 1, 1])
        fig.suptitle(
            f"Performance Dashboard - {self.simulation_name}",
            fontsize=18,
            fontweight="bold",
        )

        # Main performance plot (spans 2 columns)
        ax_main = fig.add_subplot(gs[0, :2])
        numeric_traces = {
            k: v
            for k, v in traces.items()
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float))
        }

        for key, values in list(numeric_traces.items())[:3]:  # Show top 3
            ax_main.plot(
                values, label=key, linewidth=3, alpha=0.8, marker="o", markersize=4
            )
        ax_main.set_title("Primary Performance Metrics", fontsize=14, fontweight="bold")
        ax_main.set_xlabel("Time Step")
        ax_main.set_ylabel("Value")
        ax_main.legend(loc="best")
        ax_main.grid(True, alpha=0.3)

        # Summary statistics
        ax_summary = fig.add_subplot(gs[0, 2])
        summary_text: list[Any] = []
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                summary_text.append(f"{key}: {value:.3f}")
            else:
                summary_text.append(f"{key}: {str(value)[:20]}...")

        ax_summary.text(
            0.1,
            0.9,
            "\n".join(summary_text[:8]),
            transform=ax_summary.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.7},
        )
        ax_summary.set_title("Summary Statistics")
        ax_summary.axis("off")

        # Performance trends
        ax_trend = fig.add_subplot(gs[1, :])
        if numeric_traces:
            # Calculate trends (simple linear regression slope)
            trends: dict[Any, Any] = {}
            for key, values in numeric_traces.items():
                if len(values) > 1:
                    x = np.arange(len(values))
                    slope, _ = np.polyfit(x, values, 1)
                    trends[key] = slope

            if trends:
                keys, slopes = zip(*trends.items())
                colors = ["green" if s > 0 else "red" for s in slopes]
                bars = ax_trend.bar(keys, slopes, color=colors, alpha=0.7)
                ax_trend.set_title("Performance Trends (Slope Analysis)")
                ax_trend.set_ylabel("Trend (slope)")
                ax_trend.axhline(y=0, color="black", linestyle="--", alpha=0.5)
                plt.setp(ax_trend.get_xticklabels(), rotation=45, ha="right")

                # Add value labels on bars
                for bar, slope in zip(bars, slopes):
                    height = bar.get_height()
                    ax_trend.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.01 * max(abs(s) for s in slopes),
                        f"{slope:.4f}",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

        # Efficiency analysis
        ax_efficiency = fig.add_subplot(gs[2, :])
        if numeric_traces:
            # Calculate efficiency metrics
            efficiency_data: list[Any] = []
            for key, values in list(numeric_traces.items())[:5]:
                if len(values) > 0:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    cv = (
                        std_val / abs(mean_val) if mean_val != 0 else float("inf")
                    )  # Coefficient of variation
                    efficiency_data.append((key, mean_val, std_val, cv))

            if efficiency_data:
                keys, means, stds, cvs = zip(*efficiency_data)
                x_pos = np.arange(len(keys))

                # Create efficiency plot with error bars
                bars = ax_efficiency.bar(x_pos, means, yerr=stds, alpha=0.7, capsize=5)
                ax_efficiency.set_title("Efficiency Analysis (Mean ± Std, CV in text)")
                ax_efficiency.set_ylabel("Mean Value")
                ax_efficiency.set_xticks(x_pos)
                ax_efficiency.set_xticklabels(keys, rotation=45, ha="right")

                # Add CV as text labels
                for i, (bar, cv) in enumerate(zip(bars, cvs)):
                    height = bar.get_height()
                    ax_efficiency.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + stds[i] + 0.1,
                        f"CV:{cv:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        plt.tight_layout()
        dashboard_file = (
            self.viz_dir / f"{self.simulation_name}_PERFORMANCE_dashboard.png"
        )
        plt.savefig(dashboard_file)
        plt.close()
        viz_files.append(dashboard_file)

        return viz_files

    def _create_comparative_plots(self, traces: Dict) -> List[Path]:
        """Create comparative analysis plots"""
        viz_files: list[Any] = []

        # Comparative analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            f"Comparative Analysis - {self.simulation_name}",
            fontsize=16,
            fontweight="bold",
        )

        numeric_traces = {
            k: v
            for k, v in traces.items()
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float))
        }

        if len(numeric_traces) >= 2:
            # Radar plot for multi-dimensional comparison
            ax1 = axes[0, 0]
            # Calculate summary statistics for radar plot
            stats: dict[Any, Any] = {}
            for key, values in list(numeric_traces.items())[
                :5
            ]:  # Limit to 5 for clarity
                stats[key] = {
                    "mean": np.mean(values),
                    "max": np.max(values),
                    "min": np.min(values),
                    "std": np.std(values),
                }

            if stats:
                categories = list(stats.keys())
                means = [stats[cat]["mean"] for cat in categories]

                # Normalize for radar plot
                if max(means) > 0:
                    means_norm = [m / max(means) for m in means]

                    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
                    means_norm += means_norm[:1]  # Complete the circle
                    angles = np.concatenate((angles, [angles[0]]))

                    ax1.plot(angles, means_norm, "o-", linewidth=2, alpha=0.8)
                    ax1.fill(angles, means_norm, alpha=0.25)
                    ax1.set_xticks(angles[:-1])
                    ax1.set_xticklabels(categories)
                    ax1.set_ylim(0, 1)
                    ax1.set_title("Multi-dimensional Profile\n(Normalized Means)")
                    ax1.grid(True)

            # Range comparison
            ax2 = axes[0, 1]
            ranges = [
                (np.max(values) - np.min(values)) for values in numeric_traces.values()
            ]
            keys = list(numeric_traces.keys())
            bars = ax2.bar(keys, ranges, alpha=0.7, color="skyblue")
            ax2.set_title("Value Ranges")
            ax2.set_ylabel("Range (Max - Min)")
            plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

            # Add range values on bars
            for bar, range_val in zip(bars, ranges):
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01 * max(ranges),
                    f"{range_val:.2f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            # Variance comparison
            ax3 = axes[1, 0]
            variances = [np.var(values) for values in numeric_traces.values()]
            bars = ax3.bar(keys, variances, alpha=0.7, color="lightcoral")
            ax3.set_title("Variance Comparison")
            ax3.set_ylabel("Variance")
            plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")

            # Percentile comparison
            ax4 = axes[1, 1]
            percentiles: list[Any] = [25, 50, 75]
            width = 0.2
            x_pos = np.arange(len(keys))

            for i, p in enumerate(percentiles):
                p_values = [
                    np.percentile(values, p) for values in numeric_traces.values()
                ]
                ax4.bar(
                    x_pos + i * width,
                    p_values,
                    width,
                    label=f"{p}th percentile",
                    alpha=0.7,
                )

            ax4.set_title("Percentile Comparison")
            ax4.set_ylabel("Value")
            ax4.set_xticks(x_pos + width)
            ax4.set_xticklabels(keys, rotation=45, ha="right")
            ax4.legend()

        plt.tight_layout()
        comp_file = self.viz_dir / f"{self.simulation_name}_COMPARATIVE_analysis.png"
        plt.savefig(comp_file)
        plt.close()
        viz_files.append(comp_file)

        return viz_files

#!/usr/bin/env python3
"""
Runtime/scaling/throughput plot mixins for GNN meta-analysis sweep visualizations.

Extracted from ``integration.meta_analysis.visualizer``.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

from .collector import SweepRecord
from .visualizer_style import (
    _MPL_AVAILABLE,
    _STYLE,
    _add_watermark,
    _fmt_time,
    _get_color,
    mcolors,
    np,
    plt,
)


class SweepRuntimePlotMixin:
    """Verbatim plot methods moved from ``SweepVisualizer``."""

    if TYPE_CHECKING:
        output_dir: Path
        logger: logging.Logger

        @staticmethod
        def _safe_log_scale(
            ax: Any, *, x: bool = False, y: bool = False, base: int = 10
        ) -> None: ...
    # ─── Heatmap ───────────────────────────────────────────────────────────

    def _plot_runtime_heatmap(
        self,
        records: List[SweepRecord],
        framework: str,
        all_n_values: List[int],
        all_t_values: List[int],
    ) -> Optional[Path]:
        """Generate an N×T runtime heatmap for a single framework.

        Uses the GLOBAL N and T grid so all heatmaps share the same dimensions.
        NaN cells (missing data) are shown as dark gray with a hatch pattern.
        """
        if len(all_n_values) < 2 or len(all_t_values) < 2:
            return None

        # Build grid from all sweep records for this framework
        fw_records = {
            (r.num_states, r.num_timesteps): r
            for r in records
            if r.framework == framework
            and r.num_states is not None
            and r.num_timesteps is not None
        }

        grid = np.full((len(all_n_values), len(all_t_values)), np.nan)
        for (n, t), r in fw_records.items():
            if (
                n in all_n_values
                and t in all_t_values
                and r.execution_time > 0
                and r.success
            ):
                i = all_n_values.index(n)
                j = all_t_values.index(t)
                grid[i, j] = r.execution_time

        # Skip if no data at all
        valid_count = np.count_nonzero(~np.isnan(grid))
        if valid_count == 0:
            return None

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor(_STYLE["bg_color"])
        ax.set_facecolor(_STYLE["axis_bg"])

        # Use log normalization if the dynamic range exceeds 10x
        vmin = np.nanmin(grid)
        vmax = np.nanmax(grid)
        if vmax > 0 and vmax / max(vmin, 0.001) > 10:
            norm = mcolors.LogNorm(vmin=max(vmin, 0.01), vmax=vmax)
        else:
            norm = None

        # Draw the heatmap — NaN cells will be transparent
        im = ax.imshow(grid, cmap="YlOrRd", aspect="auto", origin="lower", norm=norm)

        # Fill NaN cells with a light gray hatched pattern
        for i in range(len(all_n_values)):
            for j in range(len(all_t_values)):
                if np.isnan(grid[i, j]):
                    ax.add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5),
                            1,
                            1,
                            fill=True,
                            facecolor="#F1F3F5",
                            edgecolor="#DEE2E6",
                            hatch="xxx",
                            linewidth=0.5,
                        )
                    )
                    ax.text(
                        j,
                        i,
                        "N/A",
                        ha="center",
                        va="center",
                        fontsize=_STYLE["tick_size"] - 2,
                        color="#ADB5BD",
                        fontstyle="italic",
                    )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(
            "Runtime (s)", size=_STYLE["label_size"], color=_STYLE["text_color"]
        )
        cbar.ax.tick_params(labelsize=_STYLE["tick_size"], colors=_STYLE["text_color"])

        # Tick labels: use actual N and T values
        ax.set_xticks(range(len(all_t_values)))
        ax.set_xticklabels(
            [f"{t:,}" for t in all_t_values],
            color=_STYLE["text_color"],
            fontsize=_STYLE["tick_size"],
        )
        ax.set_yticks(range(len(all_n_values)))
        ax.set_yticklabels(
            [str(n) for n in all_n_values],
            color=_STYLE["text_color"],
            fontsize=_STYLE["tick_size"],
        )

        ax.set_xlabel(
            "Timesteps (T)",
            color=_STYLE["text_color"],
            fontsize=_STYLE["label_size"],
            fontweight="bold",
        )
        ax.set_ylabel(
            "State Space Size (N)",
            color=_STYLE["text_color"],
            fontsize=_STYLE["label_size"],
            fontweight="bold",
        )

        # Enhanced Title with Stats
        avg_runtime = np.nanmean(grid)
        ax.set_title(
            f"Wall-Clock Runtime: {framework}",
            color=_STYLE["text_color"],
            fontsize=_STYLE["title_size"],
            fontweight="bold",
            pad=30,
        )
        subtitle = f"N ∈ [{min(all_n_values)}, {max(all_n_values)}] · T ∈ [{min(all_t_values):,}, {max(all_t_values):,}] · Mean: {_fmt_time(avg_runtime)} · Range: {_fmt_time(np.nanmin(grid))}–{_fmt_time(np.nanmax(grid))}"
        ax.text(
            0.5,
            1.02,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            fontsize=_STYLE["tick_size"] - 1,
            color="#495057",
        )
        _add_watermark(ax)
        ax.tick_params(colors=_STYLE["text_color"])

        # Annotate cells with values
        for i in range(len(all_n_values)):
            for j in range(len(all_t_values)):
                val = grid[i, j]
                if not np.isnan(val):
                    label = _fmt_time(val)
                    # Use contrast color based on intensity
                    text_color = "white" if val > vmax * 0.7 else "black"
                    ax.text(
                        j,
                        i,
                        label,
                        ha="center",
                        va="center",
                        fontsize=_STYLE["annotation_size"],
                        fontweight="bold",
                        color=text_color,
                    )

        fig.tight_layout()
        fw_dir = self.output_dir / framework / "heatmaps"
        fw_dir.mkdir(parents=True, exist_ok=True)
        path = fw_dir / f"runtime_heatmap_{framework}.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

        csv_path = fw_dir / f"runtime_heatmap_{framework}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["N \\ T"] + [str(t) for t in all_t_values])
            for i, n in enumerate(all_n_values):
                row = [str(n)] + [
                    f"{grid[i, j]:.4f}" if not np.isnan(grid[i, j]) else "N/A"
                    for j in range(len(all_t_values))
                ]
                writer.writerow(row)

        self.logger.info(f"Generated runtime heatmap: {path.name} and {csv_path.name}")
        return path

    # ─── Scaling curves ────────────────────────────────────────────────────

    def _plot_runtime_scaling(
        self, records: List[SweepRecord], frameworks: List[str]
    ) -> Optional[Path]:
        """Log-log scaling curves: total runtime vs N for each T tier with power-law fits."""
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.patch.set_facecolor(_STYLE["bg_color"])

        t_values = sorted(
            {r.num_timesteps for r in records if r.num_timesteps is not None}
        )
        n_values = sorted({r.num_states for r in records if r.num_states is not None})

        for ax_idx, (
            x_param,
            x_label,
            group_param,
            group_label,
            group_values,
        ) in enumerate(
            [
                ("num_states", "State Space Size (N)", "num_timesteps", "T", t_values),
                ("num_timesteps", "Timesteps (T)", "num_states", "N", n_values),
            ]
        ):
            ax = axes[ax_idx]
            ax.set_facecolor(_STYLE["axis_bg"])

            for fw in frameworks:
                fw_records = [
                    r
                    for r in records
                    if r.framework == fw and r.execution_time > 0 and r.success
                ]
                if not fw_records:
                    continue

                for gval in group_values:
                    subset = [r for r in fw_records if getattr(r, group_param) == gval]
                    if len(subset) < 2:
                        continue

                    subset.sort(key=lambda r: getattr(r, x_param) or 0)
                    xs = np.array([getattr(r, x_param) for r in subset], dtype=float)
                    ys = np.array([r.execution_time for r in subset], dtype=float)
                    yerr = np.array(
                        [
                            (getattr(r, "execution_time_std", None) or 0.0)
                            for r in subset
                        ],
                        dtype=float,
                    )
                    show_err = bool(np.any(yerr > 0))

                    color = _get_color(fw)

                    if show_err:
                        ax.errorbar(
                            xs,
                            ys,
                            yerr=yerr,
                            fmt="none",
                            ecolor=color,
                            elinewidth=1.2,
                            capsize=4,
                            alpha=0.55,
                            zorder=2,
                        )

                    ax.plot(
                        xs,
                        ys,
                        "o",
                        color=color,
                        alpha=0.7,
                        markersize=_STYLE["marker_size"],
                        markeredgecolor="white",
                        markeredgewidth=0.5,
                        label=f"{fw} ({group_label}={gval})",
                        zorder=3,
                    )

                    # Fit power law: y = a * x^b => log(y) = log(a) + b*log(x)
                    try:
                        valid = (xs > 0) & (ys > 0)
                        if np.sum(valid) >= 2:
                            log_x = np.log(xs[valid])
                            log_y = np.log(ys[valid])

                            # Log-linear fit
                            coeffs, residuals, rank, singular_values, rcond = (
                                np.polyfit(log_x, log_y, 1, full=True)
                            )
                            exponent = coeffs[0]

                            # R^2 calculation
                            y_mean = np.mean(log_y)
                            ss_tot = np.sum((log_y - y_mean) ** 2)
                            ss_res = residuals[0] if len(residuals) > 0 else 0
                            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1

                            # RMSE calculation
                            y_pred = coeffs[0] * log_x + coeffs[1]
                            rmse = np.sqrt(np.mean((log_y - y_pred) ** 2))

                            # Label with exponents and metrics
                            label = f"{fw} ({group_label}={gval}): α={exponent:.2f} (R²={r_squared:.3f}, RMSE={rmse:.3f})"

                            fit_fn = np.exp(coeffs[1]) * (xs**exponent)
                            ax.plot(
                                xs,
                                fit_fn,
                                "-",
                                color=color,
                                alpha=0.6,
                                linewidth=_STYLE["line_width"],
                                label=label,
                            )

                            # Annotate exponent near the end of the line
                            ax.text(
                                xs[-1],
                                ys[-1],
                                f" α={exponent:.2f}",
                                color=color,
                                fontsize=_STYLE["tick_size"] - 4,
                                fontweight="bold",
                                va="center",
                            )
                    except Exception as e:
                        self.logger.debug(
                            "Skipped scaling fit for %s=%s in %s: %s",
                            group_label,
                            gval,
                            fw,
                            e,
                        )

            ax.set_xlabel(
                x_label,
                color=_STYLE["text_color"],
                fontsize=_STYLE["label_size"],
                fontweight="bold",
            )
            ax.set_ylabel(
                "Wall-clock Runtime (s)",
                color=_STYLE["text_color"],
                fontsize=_STYLE["label_size"],
                fontweight="bold",
            )
            ax.set_title(
                f"Runtime Scaling vs {x_label.split('(')[0].strip()}",
                color=_STYLE["text_color"],
                fontsize=_STYLE["title_size"],
                fontweight="bold",
            )
            self._safe_log_scale(ax, x=True, y=True)
            ax.tick_params(colors=_STYLE["text_color"], labelsize=_STYLE["tick_size"])
            ax.grid(
                True,
                which="both",
                alpha=0.4,
                color=_STYLE["grid_color"],
                linestyle="--",
            )
            _add_watermark(ax)

            # Legend
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                by_label = dict(zip(labels, handles))
                ax.legend(
                    by_label.values(),
                    by_label.keys(),
                    fontsize=_STYLE["legend_size"],
                    facecolor="white",
                    edgecolor=_STYLE["grid_color"],
                    labelcolor="black",
                    loc="best",
                    framealpha=0.9,
                    title="Framework (Params)",
                )

        # Global Title
        fig.suptitle(
            "Descriptive fits to saved runtime observations",
            color=_STYLE["text_color"],
            fontsize=_STYLE["title_size"] + 2,
            fontweight="bold",
            y=1.02,
        )
        fig.tight_layout()

        cf_dir = self.output_dir / "cross_framework" / "scaling"
        cf_dir.mkdir(parents=True, exist_ok=True)
        path = cf_dir / "runtime_scaling_curves.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

        csv_path = cf_dir / "runtime_scaling_curves.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Framework", "N", "T", "Runtime (s)", "Runtime_std (s)"])
            for fw in frameworks:
                fw_records = [
                    r
                    for r in records
                    if r.framework == fw and r.execution_time > 0 and r.success
                ]
                for r in sorted(
                    fw_records, key=lambda x: (x.num_states or 0, x.num_timesteps or 0)
                ):
                    std_val = ""
                    if r.execution_time_std is not None and r.execution_time_std > 0:
                        std_val = f"{r.execution_time_std:.4f}"
                    writer.writerow(
                        [
                            fw,
                            r.num_states,
                            r.num_timesteps,
                            f"{r.execution_time:.4f}",
                            std_val,
                        ]
                    )

        self.logger.info(f"Generated scaling curves: {path.name} and {csv_path.name}")
        return path

    # ─── Framework comparison ──────────────────────────────────────────────

    def _plot_framework_comparison(
        self, records: List[SweepRecord], frameworks: List[str]
    ) -> Optional[Path]:
        """Grouped bar chart: runtime per model, grouped by framework."""
        models = sorted({r.model_name for r in records if r.num_states is not None})
        if not models or not frameworks:
            return None

        fig, ax = plt.subplots(figsize=(max(14, len(models) * 1.2), 8))
        fig.patch.set_facecolor(_STYLE["bg_color"])
        ax.set_facecolor(_STYLE["axis_bg"])

        x = np.arange(len(models))
        bar_width = 0.8 / max(len(frameworks), 1)

        all_times: list[Any] = []
        for i, fw in enumerate(frameworks):
            times: list[Any] = []
            for model in models:
                match = [
                    r
                    for r in records
                    if r.model_name == model
                    and r.framework == fw
                    and r.success
                    and r.execution_time > 0
                ]
                times.append(match[0].execution_time if match else 0)

            all_times.extend([t for t in times if t > 0])
            offset = (i - len(frameworks) / 2 + 0.5) * bar_width
            bars = ax.bar(
                x + offset,
                times,
                bar_width * 0.9,
                color=_get_color(fw),
                alpha=0.8,
                label=fw,
                edgecolor="black",
                linewidth=0.5,
            )

            # Annotate non-zero bars
            for bar, t in zip(bars, times):
                if t > 0:
                    label = _fmt_time(t)
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.05,
                        label,
                        ha="center",
                        va="bottom",
                        fontsize=_STYLE["tick_size"] - 4,
                        color=_STYLE["text_color"],
                        fontweight="bold",
                        rotation=45,
                    )

        # Global Median Line
        if all_times:
            median_val = np.median(all_times)
            ax.axhline(
                median_val,
                color="#E63946",
                linestyle="--",
                alpha=0.6,
                linewidth=1.5,
                label=f"Global Median ({_fmt_time(median_val)})",
            )

        # Clean up model labels
        import re as _re

        display_labels: list[Any] = []
        for m in models:
            label_match = _re.search(r"N(\d+).*T(\d+)", m)
            if label_match:
                display_labels.append(
                    f"N={label_match.group(1)}\nT={label_match.group(2)}"
                )
            else:
                display_labels.append(m[:20])

        ax.set_xticks(x)
        ax.set_xticklabels(
            display_labels, color=_STYLE["text_color"], fontsize=_STYLE["tick_size"] - 2
        )
        ax.set_ylabel(
            "Runtime (s)",
            color=_STYLE["text_color"],
            fontsize=_STYLE["label_size"],
            fontweight="bold",
        )
        ax.set_title(
            "Cross-Framework Runtime Comparison",
            color=_STYLE["text_color"],
            fontsize=_STYLE["title_size"],
            fontweight="bold",
        )
        self._safe_log_scale(ax, y=True)
        ax.tick_params(colors=_STYLE["text_color"], labelsize=_STYLE["tick_size"])
        ax.grid(axis="y", alpha=0.4, color=_STYLE["grid_color"], linestyle="--")

        ax.legend(
            fontsize=_STYLE["legend_size"],
            facecolor="white",
            edgecolor=_STYLE["grid_color"],
            labelcolor="black",
            loc="upper left",
            ncol=min(len(frameworks) + 1, 4),
            framealpha=0.9,
        )

        _add_watermark(ax)

        fig.tight_layout()
        cf_dir = self.output_dir / "cross_framework" / "comparisons"
        cf_dir.mkdir(parents=True, exist_ok=True)
        path = cf_dir / "framework_runtime_comparison.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

        csv_path = cf_dir / "framework_runtime_comparison.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Model"] + frameworks)
            for model in models:
                row: list[Any] = [model]
                for fw in frameworks:
                    match = [
                        r
                        for r in records
                        if r.model_name == model
                        and r.framework == fw
                        and r.success
                        and r.execution_time > 0
                    ]
                    row.append(f"{match[0].execution_time:.4f}" if match else "N/A")
                writer.writerow(row)

        self.logger.info(
            f"Generated framework comparison: {path.name} and {csv_path.name}"
        )
        return path

    # ─── Time per step ─────────────────────────────────────────────────────

    def _plot_time_per_step(
        self, records: List[SweepRecord], frameworks: List[str]
    ) -> Optional[Path]:
        """Per-step timing comparison grouped by framework and N."""
        # Build grouped data: for each (framework, N), compute median ms/step
        groups: list[Any] = []
        for fw in frameworks:
            fw_records = [
                r
                for r in records
                if r.framework == fw
                and r.success
                and r.time_per_step is not None
                and r.time_per_step > 0
            ]
            if not fw_records:
                continue
            n_values = sorted(
                {r.num_states for r in fw_records if r.num_states is not None}
            )
            for n in n_values:
                subset = [r for r in fw_records if r.num_states == n]
                times = [r.time_per_step for r in subset if r.time_per_step is not None]
                if times:
                    median_t = sorted(times)[len(times) // 2]
                    groups.append((fw, n, median_t))

        if not groups:
            return None

        fig, ax = plt.subplots(figsize=(max(12, len(groups) * 0.8), 7))
        fig.patch.set_facecolor(_STYLE["bg_color"])
        ax.set_facecolor(_STYLE["axis_bg"])

        labels = [f"N={n}\n{fw}" for fw, n, _ in groups]
        values = [t for _, _, t in groups]
        colors = [_get_color(fw) for fw, _, _ in groups]

        bars = ax.bar(
            range(len(groups)),
            values,
            color=colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

        # Annotate bars — place inside bar for tall values to avoid title collision
        for bar, val in zip(bars, values):
            if val > 300:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 0.85,
                    f"{val:.1f}",
                    ha="center",
                    va="top",
                    fontsize=_STYLE["tick_size"] - 3,
                    color="white",
                    fontweight="bold",
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.05,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=_STYLE["tick_size"] - 3,
                    color=_STYLE["text_color"],
                    fontweight="bold",
                )

        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(
            labels,
            color=_STYLE["text_color"],
            fontsize=_STYLE["tick_size"] - 3,
            rotation=45,
            ha="right",
        )
        ax.set_ylabel(
            "Time per Timestep (ms)",
            color=_STYLE["text_color"],
            fontsize=_STYLE["label_size"],
            fontweight="bold",
        )
        ax.set_title(
            "Per-Timestep Execution Cost",
            color=_STYLE["text_color"],
            fontsize=_STYLE["title_size"],
            fontweight="bold",
            pad=15,
        )
        ax.tick_params(colors=_STYLE["text_color"], labelsize=_STYLE["tick_size"] - 2)
        self._safe_log_scale(ax, y=True)
        ax.grid(axis="y", alpha=0.4, color=_STYLE["grid_color"], linestyle="--")

        _add_watermark(ax)

        fig.tight_layout()
        cf_dir = self.output_dir / "cross_framework" / "comparisons"
        cf_dir.mkdir(parents=True, exist_ok=True)
        path = cf_dir / "time_per_step.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

        csv_path = cf_dir / "time_per_step.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Framework", "N", "Median Time per Step (ms)"])
            for fw, n, t in groups:
                writer.writerow([fw, n, f"{t:.4f}"])

        self.logger.info(f"Generated time-per-step: {path.name} and {csv_path.name}")
        return path

    # ─── New Visualizations ────────────────────────────────────────────────

    def _plot_3d_runtime_surface(
        self,
        records: List[SweepRecord],
        framework: str,
        all_n_values: List[int],
        all_t_values: List[int],
    ) -> Optional[Path]:
        """Generate a 3D surface plot of runtime vs N and T."""
        if len(all_n_values) < 2 or len(all_t_values) < 2:
            return None

        # Build grid
        fw_records = {
            (r.num_states, r.num_timesteps): r
            for r in records
            if r.framework == framework
            and r.num_states is not None
            and r.num_timesteps is not None
        }

        X, Y = np.meshgrid(np.log10(all_t_values), np.log10(all_n_values))
        Z = np.full(X.shape, np.nan)

        for i, n in enumerate(all_n_values):
            for j, t in enumerate(all_t_values):
                r = fw_records.get((n, t))
                if r and r.execution_time > 0 and r.success:
                    Z[i, j] = np.log10(r.execution_time)

        if np.all(np.isnan(Z)):
            return None

        fig = plt.figure(figsize=(12, 10))
        fig.patch.set_facecolor(_STYLE["bg_color"])
        ax = cast(Any, fig.add_subplot(111, projection="3d"))
        ax.set_facecolor(_STYLE["bg_color"])

        # Mask NaN for surface plot
        Zm = np.ma.masked_invalid(Z)

        ax.plot_surface(
            X, Y, Zm, cmap="viridis", edgecolor="black", linewidth=0.1, alpha=0.8
        )

        ax.set_xlabel(
            "log10(Timesteps T)",
            color=_STYLE["text_color"],
            fontsize=_STYLE["label_size"],
            labelpad=10,
        )
        ax.set_ylabel(
            "log10(State Size N)",
            color=_STYLE["text_color"],
            fontsize=_STYLE["label_size"],
            labelpad=10,
        )
        ax.set_zlabel(
            "log10(Runtime [s])",
            color=_STYLE["text_color"],
            fontsize=_STYLE["label_size"],
            labelpad=10,
        )
        ax.set_title(
            f"3D Runtime Response Surface: {framework}",
            color=_STYLE["text_color"],
            fontsize=_STYLE["title_size"],
            fontweight="bold",
            pad=20,
        )

        _add_watermark(ax)

        # Style the panes for white background
        ax.xaxis.pane.set_facecolor("#F8F9FA")
        ax.yaxis.pane.set_facecolor("#F8F9FA")
        ax.zaxis.pane.set_facecolor("#F8F9FA")
        ax.xaxis.pane.set_edgecolor("#DEE2E6")
        ax.yaxis.pane.set_edgecolor("#DEE2E6")
        ax.zaxis.pane.set_edgecolor("#DEE2E6")

        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.label.set_color(_STYLE["text_color"])
            axis.set_tick_params(
                colors=_STYLE["text_color"], labelsize=_STYLE["tick_size"] - 2
            )

        fig.tight_layout()
        fw_dir = self.output_dir / framework / "surfaces"
        fw_dir.mkdir(parents=True, exist_ok=True)
        path = fw_dir / f"runtime_surface_3d_{framework}.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_compute_efficiency(
        self, records: List[SweepRecord], frameworks: List[str]
    ) -> Optional[Path]:
        """Plot 'Compute Efficiency' (N^3 * T / runtime) to show scaling bottlenecks."""
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor(_STYLE["bg_color"])
        ax.set_facecolor(_STYLE["axis_bg"])

        for fw in frameworks:
            fw_records = [
                r
                for r in records
                if r.framework == fw and r.success and r.execution_time > 0
            ]
            if not fw_records:
                continue

            # Group by N, take average across T
            n_values = sorted(
                {r.num_states for r in fw_records if r.num_states is not None}
            )
            efficiencies: list[Any] = []
            for n in n_values:
                subset = [r for r in fw_records if r.num_states == n]
                # Approximation of ops: N^3 * T
                scores = [
                    ((r.num_states**3) * (r.num_timesteps or 1)) / r.execution_time
                    for r in subset
                ]
                efficiencies.append(np.mean(scores))

            # Normalize to max efficiency across all frameworks
            ax.plot(
                n_values,
                efficiencies,
                "o-",
                color=_get_color(fw),
                label=fw,
                linewidth=_STYLE["line_width"],
                markersize=_STYLE["marker_size"],
                markeredgecolor="white",
            )

        self._safe_log_scale(ax, x=True, y=True)
        ax.set_xlabel(
            "State Space Size (N)",
            color=_STYLE["text_color"],
            fontsize=_STYLE["label_size"],
            fontweight="bold",
        )
        ax.set_ylabel(
            "Efficiency Score (Ops/sec proxy)",
            color=_STYLE["text_color"],
            fontsize=_STYLE["label_size"],
            fontweight="bold",
        )
        ax.set_title(
            "Compute Efficiency Benchmark",
            color=_STYLE["text_color"],
            fontsize=_STYLE["title_size"],
            fontweight="bold",
        )

        _add_watermark(ax)
        ax.tick_params(colors=_STYLE["text_color"], labelsize=_STYLE["tick_size"])
        ax.grid(
            True, which="both", alpha=0.4, color=_STYLE["grid_color"], linestyle="--"
        )
        ax.legend(
            facecolor="white",
            edgecolor=_STYLE["grid_color"],
            labelcolor="black",
            fontsize=_STYLE["legend_size"],
        )

        fig.tight_layout()
        cf_dir = self.output_dir / "cross_framework" / "comparisons"
        cf_dir.mkdir(parents=True, exist_ok=True)
        path = cf_dir / "compute_efficiency.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        return path

    # ─── Throughput vs N ───────────────────────────────────────────────────

    def _plot_throughput_vs_n(
        self, records: List[SweepRecord], frameworks: List[str]
    ) -> Optional[Path]:
        """Inference throughput (timesteps/second) vs state space size."""
        valid = [
            r
            for r in records
            if r.execution_time > 0 and r.success and r.num_states and r.num_timesteps
        ]
        if not valid:
            return None
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(_STYLE["bg_color"])
        ax.set_facecolor(_STYLE["axis_bg"])
        for fw in frameworks:
            fw_recs = [r for r in valid if r.framework == fw]
            if not fw_recs:
                continue
            ns = sorted({r.num_states for r in fw_recs if r.num_states is not None})
            throughputs: list[Any] = []
            for n in ns:
                subset = [r for r in fw_recs if r.num_states == n]
                throughputs.append(
                    np.mean([r.num_timesteps / r.execution_time for r in subset])
                )
            ax.plot(
                ns,
                throughputs,
                "o-",
                color=_get_color(fw),
                label=fw,
                linewidth=_STYLE["line_width"],
                markersize=_STYLE["marker_size"],
                alpha=0.8,
                markeredgecolor="white",
            )
            for n, tp in zip(ns, throughputs):
                ax.annotate(
                    f"{tp:.0f}",
                    (n, tp),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=_STYLE["tick_size"] - 3,
                    color=_STYLE["text_color"],
                    fontweight="bold",
                )
        self._safe_log_scale(ax, x=True, y=True, base=2)
        ax.set_xlabel(
            "State Space Size (N)",
            color=_STYLE["text_color"],
            fontsize=_STYLE["label_size"],
            fontweight="bold",
        )
        ax.set_ylabel(
            "Throughput (timesteps/s)",
            color=_STYLE["text_color"],
            fontsize=_STYLE["label_size"],
            fontweight="bold",
        )
        ax.set_title(
            "Inference Throughput vs Model Complexity",
            color=_STYLE["text_color"],
            fontsize=_STYLE["title_size"],
            fontweight="bold",
        )
        ax.tick_params(colors=_STYLE["text_color"], labelsize=_STYLE["tick_size"])
        ax.grid(
            True, which="both", alpha=0.4, color=_STYLE["grid_color"], linestyle="--"
        )
        ax.legend(
            fontsize=_STYLE["legend_size"],
            facecolor="white",
            edgecolor=_STYLE["grid_color"],
            labelcolor="black",
        )
        _add_watermark(ax)
        fig.tight_layout()
        cf_dir = self.output_dir / "cross_framework" / "scaling"
        cf_dir.mkdir(parents=True, exist_ok=True)
        path = cf_dir / "throughput_vs_n.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        csv_path = cf_dir / "throughput_vs_n.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Framework", "N", "Avg_Throughput_timesteps_per_s"])
            for fw in frameworks:
                fw_recs = [r for r in valid if r.framework == fw]
                for n in sorted(
                    {r.num_states for r in fw_recs if r.num_states is not None}
                ):
                    subset = [r for r in fw_recs if r.num_states == n]
                    w.writerow(
                        [
                            fw,
                            n,
                            f"{np.mean([r.num_timesteps / r.execution_time for r in subset]):.2f}",
                        ]
                    )
        self.logger.info(f"Generated throughput plot: {path.name}")
        return path

    # ─── Runtime Distribution ──────────────────────────────────────────────

    def _plot_runtime_distribution(
        self, records: List[SweepRecord], frameworks: List[str]
    ) -> Optional[Path]:
        """Box plot of runtime distribution for each N value across all T."""
        valid = [
            r for r in records if r.execution_time > 0 and r.success and r.num_states
        ]
        if len(valid) < 4:
            return None
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor(_STYLE["bg_color"])
        ax.set_facecolor(_STYLE["axis_bg"])
        ns = sorted({r.num_states for r in valid if r.num_states is not None})
        data_by_n: list[Any] = []
        labels: list[Any] = []
        for n in ns:
            runtimes = [r.execution_time for r in valid if r.num_states == n]
            if runtimes:
                data_by_n.append(runtimes)
                labels.append(f"N={n}")
        if not data_by_n:
            plt.close(fig)
            return None
        bp = ax.boxplot(
            data_by_n,
            patch_artist=True,
            tick_labels=labels,
            widths=0.6,
            medianprops=dict(color="black", linewidth=2),
        )
        base_color = _get_color(frameworks[0]) if frameworks else "#E63946"
        for patch in bp["boxes"]:
            patch.set_facecolor(base_color)
            patch.set_alpha(0.7)
        for i, d in enumerate(data_by_n):
            x = np.random.normal(i + 1, 0.04, size=len(d))
            ax.scatter(
                x,
                d,
                alpha=0.5,
                s=30,
                color="black",
                zorder=3,
                edgecolors="white",
                linewidths=0.5,
            )
        ax.set_ylabel(
            "Wall-clock Runtime (s)",
            color=_STYLE["text_color"],
            fontsize=_STYLE["label_size"],
            fontweight="bold",
        )
        ax.set_xlabel(
            "State Space Size",
            color=_STYLE["text_color"],
            fontsize=_STYLE["label_size"],
            fontweight="bold",
        )
        ax.set_title(
            "Runtime Distribution by State Space Size",
            color=_STYLE["text_color"],
            fontsize=_STYLE["title_size"],
            fontweight="bold",
        )
        ax.tick_params(colors=_STYLE["text_color"], labelsize=_STYLE["tick_size"])
        self._safe_log_scale(ax, y=True)
        ax.grid(axis="y", alpha=0.4, color=_STYLE["grid_color"], linestyle="--")
        medians = [np.median(d) for d in data_by_n]
        stats = f"Median range: {_fmt_time(min(medians))}–{_fmt_time(max(medians))}"
        ax.text(
            0.02,
            0.98,
            stats,
            transform=ax.transAxes,
            fontsize=_STYLE["tick_size"] - 2,
            va="top",
            color="#495057",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="#F8F9FA",
                edgecolor="#DEE2E6",
                alpha=0.9,
            ),
        )
        _add_watermark(ax)
        fig.tight_layout()
        cf_dir = self.output_dir / "cross_framework" / "comparisons"
        cf_dir.mkdir(parents=True, exist_ok=True)
        path = cf_dir / "runtime_distribution.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Generated runtime distribution: {path.name}")
        return path

    def _plot_runtime_uncertainty(
        self,
        records: List[SweepRecord],
        frameworks: List[str],
    ) -> Optional[Path]:
        """Bar chart of median runtime with ±σ when benchmark repeats recorded σ > 0."""
        if not _MPL_AVAILABLE:
            return None
        uncertain = [
            r
            for r in records
            if r.framework in frameworks
            and r.execution_time_std is not None
            and r.execution_time_std > 0
            and r.success
            and r.num_states is not None
            and r.num_timesteps is not None
        ]
        if not uncertain:
            return None
        by_t: Dict[int, List[SweepRecord]] = {}
        for r in uncertain:
            by_t.setdefault(int(r.num_timesteps or 0), []).append(r)

        n_panels = len(by_t)
        fig, axes_arr = plt.subplots(n_panels, 1, figsize=(12, 4 * max(1, n_panels)))
        if n_panels == 1:
            axes_list: List[Any] = [axes_arr]
        else:
            axes_list = list(axes_arr)

        for ax, (t_val, group) in zip(axes_list, sorted(by_t.items())):
            group.sort(key=lambda r: (r.num_states or 0, r.framework))
            xs = np.arange(len(group))
            heights = [r.execution_time for r in group]
            yerr = [float(r.execution_time_std or 0) for r in group]
            colors_b = [_get_color(r.framework) for r in group]
            ax.bar(
                xs,
                heights,
                yerr=yerr,
                color=colors_b,
                capsize=4,
                edgecolor=_STYLE["grid_color"],
            )
            ax.set_xticks(xs)
            ax.set_xticklabels(
                [f"N={r.num_states}\n({r.framework})" for r in group], fontsize=10
            )
            ax.set_ylabel("Time (s)", color=_STYLE["text_color"])
            ax.set_title(
                f"Runtime ± σ (T={t_val})",
                color=_STYLE["text_color"],
                fontweight="bold",
            )
            ax.tick_params(colors=_STYLE["text_color"])
            ax.grid(
                True, axis="y", alpha=0.35, color=_STYLE["grid_color"], linestyle="--"
            )
            _add_watermark(ax)
        fig.suptitle(
            "Benchmark repeat uncertainty",
            fontsize=_STYLE["title_size"],
            fontweight="bold",
            color=_STYLE["text_color"],
        )
        fig.tight_layout()
        cf_dir = self.output_dir / "cross_framework" / "comparisons"
        cf_dir.mkdir(parents=True, exist_ok=True)
        path = cf_dir / "runtime_uncertainty.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Generated runtime uncertainty plot: {path.name}")
        return path

#!/usr/bin/env python3
"""
Resource, dashboard, and serialization-footprint plot mixins for GNN meta-analysis sweep visualizations.

Extracted from ``integration.meta_analysis.visualizer``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .collector import SweepRecord
from .visualizer_style import (
    _MPL_AVAILABLE,
    _STYLE,
    _add_watermark,
    _fmt_time,
    _get_color,
    np,
    plt,
)


class SweepSummaryPlotMixin:
    """Verbatim plot methods moved from ``SweepVisualizer``."""

    if TYPE_CHECKING:
        output_dir: Path
        logger: logging.Logger
        _gnn_format_statistics: Optional[Dict[str, Any]]

        @staticmethod
        def _safe_log_scale(
            ax: Any, *, x: bool = False, y: bool = False, base: int = 10
        ) -> None: ...

    def _plot_resource_scaling(
        self, records: List[SweepRecord], frameworks: List[str]
    ) -> Optional[Path]:
        """Plot lines of code (LOC) vs N."""
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor(_STYLE["bg_color"])
        ax.set_facecolor(_STYLE["axis_bg"])

        for fw in frameworks:
            fw_records = [
                r for r in records if r.framework == fw and r.lines_of_code is not None
            ]
            if not fw_records:
                continue

            # Group by N (T doesn't affect LOC usually)
            n_values = sorted(
                {r.num_states for r in fw_records if r.num_states is not None}
            )
            loc_values: list[Any] = []
            for n in n_values:
                subset = [r for r in fw_records if r.num_states == n]
                loc_values.append(np.mean([r.lines_of_code for r in subset]))

            ax.plot(
                n_values,
                loc_values,
                "D-",
                color=_get_color(fw),
                label=fw,
                linewidth=_STYLE["line_width"],
                markersize=_STYLE["marker_size"],
                alpha=0.8,
                markeredgecolor="white",
            )

            # Power-law fit for LOC scaling
            try:
                if len(n_values) >= 3:
                    log_n = np.log(np.array(n_values, dtype=float))
                    log_loc = np.log(np.array(loc_values, dtype=float))
                    coeffs = np.polyfit(log_n, log_loc, 1)
                    exponent = coeffs[0]
                    fit_vals = (
                        np.exp(coeffs[1]) * np.array(n_values, dtype=float) ** exponent
                    )
                    ax.plot(
                        n_values,
                        fit_vals,
                        "--",
                        color=_get_color(fw),
                        alpha=0.5,
                        linewidth=1.5,
                        label=f"{fw} fit: O(N^{exponent:.2f})",
                    )
            except Exception as e:
                self.logger.debug("Skipped LOC fit for %s: %s", fw, e)

        self._safe_log_scale(ax, x=True, y=True)
        ax.set_xlabel(
            "State Space Size (N)",
            color=_STYLE["text_color"],
            fontsize=_STYLE["label_size"],
            fontweight="bold",
        )
        ax.set_ylabel(
            "Lines of Code (Rendered)",
            color=_STYLE["text_color"],
            fontsize=_STYLE["label_size"],
            fontweight="bold",
        )
        ax.set_title(
            "Generated Code Complexity Scaling",
            color=_STYLE["text_color"],
            fontsize=_STYLE["title_size"],
            fontweight="bold",
        )
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
        _add_watermark(ax)

        fig.tight_layout()
        cf_dir = self.output_dir / "cross_framework" / "scaling"
        cf_dir.mkdir(parents=True, exist_ok=True)
        path = cf_dir / "resource_scaling_loc.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        return path

    # ─── Scaling Exponent Summary ──────────────────────────────────────────

    def _plot_scaling_exponent_summary(
        self, records: List[SweepRecord], frameworks: List[str]
    ) -> Optional[Path]:
        """Bar chart of empirical scaling exponents by parameter grouping."""
        valid = [
            r
            for r in records
            if r.execution_time > 0 and r.success and r.num_states and r.num_timesteps
        ]
        if len(valid) < 6:
            return None
        t_values = sorted({r.num_timesteps for r in valid})
        n_values = sorted({r.num_states for r in valid if r.num_states is not None})
        exponents: list[Any] = []  # (label, exponent, r_squared)
        for fw in frameworks:
            fw_recs = [r for r in valid if r.framework == fw]
            for t in t_values:
                subset = sorted(
                    [r for r in fw_recs if r.num_timesteps == t],
                    key=lambda r: r.num_states,
                )
                if len(subset) < 3:
                    continue
                xs, ys = (
                    np.array([r.num_states for r in subset], dtype=float),
                    np.array([r.execution_time for r in subset], dtype=float),
                )
                try:
                    lx, ly = np.log(xs), np.log(ys)
                    c = np.polyfit(lx, ly, 1)
                    ss_res = np.sum((ly - (c[0] * lx + c[1])) ** 2)
                    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
                    exponents.append(
                        (
                            f"N-scaling\n(T={t})",
                            c[0],
                            1 - ss_res / ss_tot if ss_tot > 0 else 0,
                        )
                    )
                except Exception as e:
                    self.logger.debug(
                        "Skipped N-scaling exponent for %s at T=%s: %s", fw, t, e
                    )
            for n in n_values:
                subset = sorted(
                    [r for r in fw_recs if r.num_states == n],
                    key=lambda r: r.num_timesteps,
                )
                if len(subset) < 3:
                    continue
                xs, ys = (
                    np.array([r.num_timesteps for r in subset], dtype=float),
                    np.array([r.execution_time for r in subset], dtype=float),
                )
                try:
                    lx, ly = np.log(xs), np.log(ys)
                    c = np.polyfit(lx, ly, 1)
                    ss_res = np.sum((ly - (c[0] * lx + c[1])) ** 2)
                    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
                    exponents.append(
                        (
                            f"T-scaling\n(N={n})",
                            c[0],
                            1 - ss_res / ss_tot if ss_tot > 0 else 0,
                        )
                    )
                except Exception as e:
                    self.logger.debug(
                        "Skipped T-scaling exponent for %s at N=%s: %s", fw, n, e
                    )
        if not exponents:
            return None
        fig, ax = plt.subplots(figsize=(max(14, len(exponents) * 1.2), 8))
        fig.patch.set_facecolor(_STYLE["bg_color"])
        ax.set_facecolor(_STYLE["axis_bg"])
        labels = [e[0] for e in exponents]
        vals = [e[1] for e in exponents]
        r2s = [e[2] for e in exponents]
        colors = ["#E63946" if "N-scaling" in l else "#457B9D" for l in labels]
        bars = ax.bar(
            range(len(vals)),
            vals,
            color=colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )
        for bar, v, r2 in zip(bars, vals, r2s):
            # Offset in points, not data units, so the label gap does not
            # depend on the exponent scale (a data-unit offset is unbounded
            # in pixels when the y-range is near zero).
            ax.annotate(
                f"α={v:.2f}\nR²={r2:.2f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=_STYLE["tick_size"] - 3,
                fontweight="bold",
                color=_STYLE["text_color"],
            )
        ax.axhline(
            1.0,
            color="#ADB5BD",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label="Linear scaling (α=1)",
        )
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(
            labels, fontsize=_STYLE["tick_size"] - 3, color=_STYLE["text_color"]
        )
        ax.set_ylabel(
            "Scaling Exponent (α)",
            color=_STYLE["text_color"],
            fontsize=_STYLE["label_size"],
            fontweight="bold",
        )
        ax.set_title(
            "Empirical Scaling Exponents by Parameter Group",
            color=_STYLE["text_color"],
            fontsize=_STYLE["title_size"],
            fontweight="bold",
        )
        ax.tick_params(colors=_STYLE["text_color"], labelsize=_STYLE["tick_size"])
        ax.grid(axis="y", alpha=0.4, color=_STYLE["grid_color"], linestyle="--")
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
        path = cf_dir / "scaling_exponent_summary.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Generated scaling exponent summary: {path.name}")
        return path

    # ─── Code Efficiency ───────────────────────────────────────────────────

    def _plot_code_efficiency(
        self, records: List[SweepRecord], frameworks: List[str]
    ) -> Optional[Path]:
        """LOC per state-squared (LOC/N²) vs N — measures code generation efficiency."""
        valid = [
            r
            for r in records
            if r.lines_of_code and r.num_states and r.lines_of_code > 0
        ]
        if len(valid) < 3:
            return None
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(_STYLE["bg_color"])
        ax.set_facecolor(_STYLE["axis_bg"])
        for fw in frameworks:
            fw_recs = [r for r in valid if r.framework == fw]
            if not fw_recs:
                continue
            ns = sorted({r.num_states for r in fw_recs if r.num_states is not None})
            efficiencies: list[Any] = []
            for n in ns:
                subset = [r for r in fw_recs if r.num_states == n]
                avg_loc = np.mean([r.lines_of_code for r in subset])
                efficiencies.append(avg_loc / (n * n))
            ax.plot(
                ns,
                efficiencies,
                "s-",
                color=_get_color(fw),
                label=fw,
                linewidth=_STYLE["line_width"],
                markersize=_STYLE["marker_size"],
                alpha=0.8,
                markeredgecolor="white",
            )
        self._safe_log_scale(ax, x=True, base=2)
        ax.set_xlabel(
            "State Space Size (N)",
            color=_STYLE["text_color"],
            fontsize=_STYLE["label_size"],
            fontweight="bold",
        )
        ax.set_ylabel(
            "LOC / N²",
            color=_STYLE["text_color"],
            fontsize=_STYLE["label_size"],
            fontweight="bold",
        )
        ax.set_title(
            "Code Generation Efficiency (LOC per State²)",
            color=_STYLE["text_color"],
            fontsize=_STYLE["title_size"],
            fontweight="bold",
        )
        ax.tick_params(colors=_STYLE["text_color"], labelsize=_STYLE["tick_size"])
        ax.grid(True, alpha=0.4, color=_STYLE["grid_color"], linestyle="--")
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
        path = cf_dir / "code_efficiency.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Generated code efficiency: {path.name}")
        return path

    # ─── Comprehensive Dashboard ───────────────────────────────────────────

    def _plot_comprehensive_dashboard(
        self,
        records: List[SweepRecord],
        frameworks: List[str],
        all_n: List[int],
        all_t: List[int],
    ) -> Optional[Path]:
        """Multi-panel summary dashboard combining key metrics."""
        valid = [r for r in records if r.execution_time > 0 and r.success]
        if len(valid) < 4:
            return None
        fig, axes = plt.subplots(2, 3, figsize=(24, 14))
        fig.patch.set_facecolor(_STYLE["bg_color"])
        fig.suptitle(
            "PyMDP Scaling Analysis — Executive Dashboard",
            fontsize=_STYLE["title_size"] + 4,
            fontweight="bold",
            color=_STYLE["text_color"],
            y=1.01,
        )

        # Panel 1: Runtime vs N (median across T)
        ax = axes[0, 0]
        ax.set_facecolor(_STYLE["axis_bg"])
        for fw in frameworks:
            fw_recs = [r for r in valid if r.framework == fw]
            ns = sorted({r.num_states for r in fw_recs if r.num_states})
            meds = [
                np.median([r.execution_time for r in fw_recs if r.num_states == n])
                for n in ns
            ]
            ax.plot(
                ns,
                meds,
                "o-",
                color=_get_color(fw),
                label=fw,
                linewidth=2.5,
                markersize=8,
                markeredgecolor="white",
            )
        self._safe_log_scale(ax, x=True, y=True, base=2)
        ax.set_title(
            "Median Runtime vs N",
            fontsize=_STYLE["label_size"],
            fontweight="bold",
            color=_STYLE["text_color"],
        )
        ax.set_xlabel("N")
        ax.set_ylabel("Runtime (s)")
        ax.grid(True, alpha=0.3, color=_STYLE["grid_color"], linestyle="--")

        # Panel 2: Throughput vs N
        ax = axes[0, 1]
        ax.set_facecolor(_STYLE["axis_bg"])
        for fw in frameworks:
            fw_recs = [
                r
                for r in valid
                if r.framework == fw and r.num_states and r.num_timesteps
            ]
            ns = sorted({r.num_states for r in fw_recs if r.num_states is not None})
            tps = [
                np.mean(
                    [
                        r.num_timesteps / r.execution_time
                        for r in fw_recs
                        if r.num_states == n
                    ]
                )
                for n in ns
            ]
            ax.plot(
                ns,
                tps,
                "o-",
                color=_get_color(fw),
                label=fw,
                linewidth=2.5,
                markersize=8,
                markeredgecolor="white",
            )
        self._safe_log_scale(ax, x=True, y=True, base=2)
        ax.set_title(
            "Throughput vs N",
            fontsize=_STYLE["label_size"],
            fontweight="bold",
            color=_STYLE["text_color"],
        )
        ax.set_xlabel("N")
        ax.set_ylabel("Timesteps/s")
        ax.grid(True, alpha=0.3, color=_STYLE["grid_color"], linestyle="--")

        # Panel 3: Accuracy vs N
        ax = axes[0, 2]
        ax.set_facecolor(_STYLE["axis_bg"])
        acc_recs = [r for r in valid if r.final_accuracy is not None and r.num_states]
        for fw in frameworks:
            fw_a = [r for r in acc_recs if r.framework == fw]
            ns = sorted({r.num_states for r in fw_a if r.num_states is not None})
            accs = [
                np.mean([r.final_accuracy for r in fw_a if r.num_states == n])
                for n in ns
            ]
            ax.plot(
                ns,
                accs,
                "o-",
                color=_get_color(fw),
                label=fw,
                linewidth=2.5,
                markersize=8,
                markeredgecolor="white",
            )
        self._safe_log_scale(ax, x=True, base=2)
        ax.set_ylim(0.55, 1.02)
        ax.set_title(
            "Accuracy vs N",
            fontsize=_STYLE["label_size"],
            fontweight="bold",
            color=_STYLE["text_color"],
        )
        ax.set_xlabel("N")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.3, color=_STYLE["grid_color"], linestyle="--")

        # Panel 4: LOC vs N
        ax = axes[1, 0]
        ax.set_facecolor(_STYLE["axis_bg"])
        loc_recs = [r for r in records if r.lines_of_code and r.num_states]
        for fw in frameworks:
            fw_l = [r for r in loc_recs if r.framework == fw]
            ns = sorted({r.num_states for r in fw_l if r.num_states is not None})
            locs = [
                np.mean([r.lines_of_code for r in fw_l if r.num_states == n])
                for n in ns
            ]
            ax.plot(
                ns,
                locs,
                "D-",
                color=_get_color(fw),
                label=fw,
                linewidth=2.5,
                markersize=8,
                markeredgecolor="white",
            )
        self._safe_log_scale(ax, x=True, y=True, base=2)
        ax.set_title(
            "Generated LOC vs N",
            fontsize=_STYLE["label_size"],
            fontweight="bold",
            color=_STYLE["text_color"],
        )
        ax.set_xlabel("N")
        ax.set_ylabel("Lines of Code")
        ax.grid(True, alpha=0.3, color=_STYLE["grid_color"], linestyle="--")

        # Panel 5: Entropy vs N
        ax = axes[1, 1]
        ax.set_facecolor(_STYLE["axis_bg"])
        ent_recs = [
            r for r in valid if r.mean_belief_entropy is not None and r.num_states
        ]
        for fw in frameworks:
            fw_e = [r for r in ent_recs if r.framework == fw]
            ns = sorted({r.num_states for r in fw_e if r.num_states is not None})
            ents = [
                np.mean([r.mean_belief_entropy for r in fw_e if r.num_states == n])
                for n in ns
            ]
            ax.plot(
                ns,
                ents,
                "s-",
                color=_get_color(fw),
                label=fw,
                linewidth=2.5,
                markersize=8,
                markeredgecolor="white",
            )
        self._safe_log_scale(ax, x=True, base=2)
        ax.set_title(
            "Belief Entropy vs N",
            fontsize=_STYLE["label_size"],
            fontweight="bold",
            color=_STYLE["text_color"],
        )
        ax.set_xlabel("N")
        ax.set_ylabel("Entropy (nats)")
        ax.grid(True, alpha=0.3, color=_STYLE["grid_color"], linestyle="--")

        # Panel 6: Summary stats table
        ax = axes[1, 2]
        ax.set_facecolor(_STYLE["axis_bg"])
        ax.axis("off")
        total_models = len(valid)
        total_runtime = sum(r.execution_time for r in valid)
        avg_acc = (
            np.mean([r.final_accuracy for r in valid if r.final_accuracy is not None])
            if any(r.final_accuracy for r in valid)
            else 0
        )
        n_range = f"[{min(all_n)}, {max(all_n)}]"
        t_range = f"[{min(all_t):,}, {max(all_t):,}]"
        table_data: list[Any] = [
            ["Total Models", str(total_models)],
            ["N Range", n_range],
            ["T Range", t_range],
            ["Total Runtime", _fmt_time(total_runtime)],
            ["Mean Accuracy", f"{avg_acc:.3f}"],
            ["Frameworks", ", ".join(frameworks)],
        ]
        table = ax.table(
            cellText=table_data,
            colLabels=["Metric", "Value"],
            loc="center",
            cellLoc="left",
            colWidths=[0.5, 0.5],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(_STYLE["tick_size"])
        table.scale(1, 2)
        for key, cell in table.get_celld().items():
            cell.set_edgecolor(_STYLE["grid_color"])
            if key[0] == 0:
                cell.set_facecolor("#E63946")
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor("#F8F9FA" if key[0] % 2 else "white")
        ax.set_title(
            "Summary Statistics",
            fontsize=_STYLE["label_size"],
            fontweight="bold",
            color=_STYLE["text_color"],
            pad=20,
        )

        _add_watermark(axes[0, 0])
        fig.tight_layout()
        cf_dir = self.output_dir / "cross_framework"
        cf_dir.mkdir(parents=True, exist_ok=True)
        path = cf_dir / "comprehensive_dashboard.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Generated comprehensive dashboard: {path.name}")
        return path

    def _plot_gnn_serialization_footprint(self) -> Optional[Path]:
        """Bar chart of total bytes per emitted format from Step 3 format_statistics.json."""
        if not _MPL_AVAILABLE:
            return None
        stats = self._gnn_format_statistics
        if not isinstance(stats, dict):
            return None
        labels: List[str] = []
        sizes_mb: List[float] = []
        for key in ("markdown", "python", "json"):
            block = stats.get(key)
            if not isinstance(block, dict):
                continue
            sz = block.get("total_size")
            if sz is None:
                continue
            labels.append(key)
            sizes_mb.append(float(sz) / 1e6)
        if not labels:
            return None
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor(_STYLE["bg_color"])
        ax.set_facecolor(_STYLE["axis_bg"])
        palette: list[Any] = ["#457B9D", "#E63946", "#2A9D8F"]
        ax.bar(
            labels,
            sizes_mb,
            color=palette[: len(labels)],
            edgecolor=_STYLE["grid_color"],
        )
        ax.set_ylabel(
            "Total size (MB)", color=_STYLE["text_color"], fontsize=_STYLE["label_size"]
        )
        ax.set_title(
            "Step 3 multi-format serialization footprint",
            color=_STYLE["text_color"],
            fontsize=_STYLE["title_size"],
            fontweight="bold",
        )
        ax.tick_params(colors=_STYLE["text_color"], labelsize=_STYLE["tick_size"])
        ax.grid(True, axis="y", alpha=0.35, color=_STYLE["grid_color"], linestyle="--")
        _add_watermark(ax)
        fig.tight_layout()
        data_dir = self.output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / "gnn_serialization_footprint.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Generated GNN serialization footprint: {path.name}")
        return path

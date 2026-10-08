"""
Sweep Reporter — generates markdown meta-analysis reports from sweep data.

Produces a comprehensive report with:
- Sweep parameter grid overview
- Runtime statistics tables
- Framework comparison summary
- Simulation quality metrics
- JIT overhead discussion
- Theoretical complexity comparison
- Throughput results
- Embedded visualization references
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .collector import SweepRecord


class SweepReporter:
    """Generate markdown reports from sweep analysis data."""

    def __init__(
        self,
        records: list[SweepRecord],
        plot_paths: list[str],
        output_dir: Path,
        logger: logging.Logger | None = None,
        *,
        validation_payload: dict[str, Any] | None = None,
        statistics_payload: dict[str, Any] | None = None,
        validation_json_name: str = "sweep_validation.json",
        statistics_json_name: str = "meta_statistics.json",
        gnn_format_statistics: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the instance."""
        self.records = records
        self.plot_paths = plot_paths
        self.output_dir = Path(output_dir)
        self.logger = logger or logging.getLogger(__name__)
        self._validation_payload = validation_payload
        self._statistics_payload = statistics_payload
        self._validation_json_name = validation_json_name
        self._statistics_json_name = statistics_json_name
        self._gnn_format_statistics = gnn_format_statistics

    def generate(self) -> Path:
        """Generate the meta-analysis report. Returns the report path."""
        report_path = self.output_dir / "meta_analysis_report.md"

        sections: list[Any] = [
            self._header(),
            self._sweep_overview(),
            self._runtime_table(),
            self._framework_summary(),
            self._simulation_metrics(),
            self._scaling_analysis(),
            self._resource_efficiency(),
            self._validation_section(),
            self._meta_statistics_section(),
            self._gnn_footprint_section(),
            self._visualizations_section(),
        ]

        report = "\n\n".join(s for s in sections if s)
        report_path.write_text(report)
        return report_path

    def _header(self) -> str:
        """Report header."""
        n_models = len({r.model_name for r in self.records})
        n_frameworks = len({r.framework for r in self.records})
        return (
            "# Parameter Sweep Meta-Analysis Report\n\n"
            f"**Models**: {n_models} | "
            f"**Frameworks**: {n_frameworks} | "
            f"**Total Records**: {len(self.records)}\n\n"
            "---"
        )

    def _sweep_overview(self) -> str:
        """Parameter grid overview."""
        n_values = sorted(
            {r.num_states for r in self.records if r.num_states is not None}
        )
        t_values = sorted(
            {r.num_timesteps for r in self.records if r.num_timesteps is not None}
        )
        frameworks = sorted({r.framework for r in self.records})

        if not n_values or not t_values:
            return ""

        lines: list[Any] = [
            "## Sweep Configuration",
            "",
            f"- **State space sizes (N)**: {', '.join(str(n) for n in n_values)}",
            f"- **Timestep counts (T)**: {', '.join(f'{t:,}' for t in t_values)}",
            f"- **Grid dimensions**: {len(n_values)} × {len(t_values)} = {len(n_values) * len(t_values)} cells",
            f"- **Frameworks**: {', '.join(frameworks)}",
        ]
        repeat_vals = sorted(
            {
                r.execution_benchmark_repeats
                for r in self.records
                if r.execution_benchmark_repeats > 1
            }
        )
        if repeat_vals:
            mx = max(r.execution_benchmark_repeats for r in self.records)
            lines.append(
                f"- **Execution benchmark repeats**: up to **{mx}** per script "
                "(reported runtimes are **median** seconds across repeats)"
            )
        return "\n".join(lines)

    def _runtime_table(self) -> str:
        """Runtime comparison table."""
        # Build a table: rows = (N, T), columns = frameworks
        frameworks = sorted(
            {r.framework for r in self.records if r.execution_time > 0 and r.success}
        )
        if not frameworks:
            return ""

        # Sorted sweep cells
        cells = sorted(
            {
                (r.num_states, r.num_timesteps)
                for r in self.records
                if r.num_states is not None and r.num_timesteps is not None
            }
        )
        if not cells:
            return ""

        lines: list[Any] = ["## Runtime Comparison", ""]

        # Header
        header = "| N | T | " + " | ".join(frameworks) + " |"
        sep = "|---|---|" + "|".join(["---"] * len(frameworks)) + "|"
        lines.extend([header, sep])

        for n, t in cells:
            row: list[Any] = [str(n), f"{t:,}"]
            for fw in frameworks:
                match = [
                    r
                    for r in self.records
                    if r.num_states == n
                    and r.num_timesteps == t
                    and r.framework == fw
                    and r.success
                ]
                if match and match[0].execution_time > 0:
                    et = match[0].execution_time
                    row.append(f"{et:.1f}s" if et < 60 else f"{et / 60:.1f}m")
                elif match and match[0].timed_out:
                    row.append("⏰ timeout")
                else:
                    row.append("—")
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def _framework_summary(self) -> str:
        """Per-framework aggregate statistics."""
        frameworks = sorted(
            {r.framework for r in self.records if r.success and r.execution_time > 0}
        )
        if not frameworks:
            return ""

        lines: list[Any] = ["## Framework Summary", ""]
        header = "| Framework | Runs | Avg Runtime | Min | Max | Avg ms/step |"
        sep = "|---|---|---|---|---|---|"
        lines.extend([header, sep])

        for fw in frameworks:
            fw_recs = [
                r
                for r in self.records
                if r.framework == fw and r.success and r.execution_time > 0
            ]
            if not fw_recs:
                continue

            times = [r.execution_time for r in fw_recs]
            avg_t = sum(times) / len(times)
            min_t = min(times)
            max_t = max(times)

            per_step = [r.time_per_step for r in fw_recs if r.time_per_step is not None]
            avg_ps = f"{sum(per_step) / len(per_step):.2f}" if per_step else "—"

            def _fmt(v: Any) -> Any:
                """Handle fmt for internal callers."""
                return f"{v:.1f}s" if v < 60 else f"{v / 60:.1f}m"

            lines.append(
                f"| {fw} | {len(fw_recs)} | {_fmt(avg_t)} | {_fmt(min_t)} | {_fmt(max_t)} | {avg_ps} |"
            )

        return "\n".join(lines)

    def _simulation_metrics(self) -> str:
        """Simulation quality metrics summary."""
        acc_records = [r for r in self.records if r.final_accuracy is not None]
        ent_records = [r for r in self.records if r.mean_belief_entropy is not None]

        if not acc_records and not ent_records:
            return ""

        lines: list[Any] = ["## Simulation Quality Metrics", ""]

        if acc_records or ent_records:
            lines.append("### Quality-Certainty Correlation")
            lines.append("")
            # Compute correlation if both metrics exist for enough models
            common = [
                r
                for r in self.records
                if r.final_accuracy is not None and r.mean_belief_entropy is not None
            ]
            if len(common) >= 2:
                import numpy as np

                accs = np.array([r.final_accuracy for r in common])
                ents = np.array([r.mean_belief_entropy for r in common])
                r_corr = np.corrcoef(ents, accs)[0, 1]
                lines.append(
                    f"Across **{len(common)}** models, the Pearson correlation between belief entropy and accuracy is **r = {r_corr:.3f}**."
                )
            lines.append("")

        if acc_records:
            lines.append("### Observation Accuracy")
            lines.append("")
            header = "| Model | Framework | Accuracy |"
            sep = "|---|---|---|"
            lines.extend([header, sep])

            for r in sorted(acc_records, key=lambda x: (x.model_name, x.framework)):
                lines.append(
                    f"| {r.sweep_label} | {r.framework} | {r.final_accuracy:.3f} |"
                )
            lines.append("")

        if ent_records:
            lines.append("### Belief Entropy (Final Window)")
            lines.append("")
            header = "| Model | Framework | Mean Entropy (nats) |"
            sep = "|---|---|---|"
            lines.extend([header, sep])

            for r in sorted(ent_records, key=lambda x: (x.model_name, x.framework)):
                lines.append(
                    f"| {r.sweep_label} | {r.framework} | {r.mean_belief_entropy:.4f} |"
                )

        return "\n".join(lines)

    def _scaling_analysis(self) -> str:
        """Scaling law analysis text."""
        import numpy as np

        pymdp_recs = [
            r
            for r in self.records
            if r.framework == "pymdp"
            and r.success
            and r.execution_time > 0
            and r.num_states is not None
            and r.num_timesteps is not None
        ]

        if len(pymdp_recs) < 3:
            return ""

        lines: list[Any] = ["## Scaling Analysis", ""]

        # Group by T, fit N scaling
        t_values = sorted({r.num_timesteps for r in pymdp_recs})
        for t in t_values:
            subset = [r for r in pymdp_recs if r.num_timesteps == t]
            if len(subset) < 2:
                continue

            subset.sort(key=lambda r: r.num_states)
            n_vals = np.array([r.num_states for r in subset], dtype=float)
            rt_vals = np.array([r.execution_time for r in subset], dtype=float)

            try:
                valid = (n_vals > 0) & (rt_vals > 0)
                if np.sum(valid) >= 2:
                    log_x = np.log(n_vals[valid])
                    log_y = np.log(rt_vals[valid])
                    coeffs, residuals, rank, s, rcond = np.polyfit(
                        log_x, log_y, 1, full=True
                    )
                    exponent = coeffs[0]

                    # R^2 and RMSE
                    y_pred = coeffs[0] * log_x + coeffs[1]
                    rmse = np.sqrt(np.mean((log_y - y_pred) ** 2))
                    y_mean = np.mean(log_y)
                    ss_tot = np.sum((log_y - y_mean) ** 2)
                    ss_res = residuals[0] if len(residuals) > 0 else 0
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1

                    lines.append(
                        f"- **T={t:,}**: Runtime scales as **O(N^{exponent:.2f})** ($R^2$={r2:.3f}, RMSE={rmse:.3f}) "
                        f"(N={n_vals[0]}→{n_vals[-1]}: {rt_vals[0]:.1f}s → {rt_vals[-1]:.1f}s)"
                    )
            except Exception as e:
                self.logger.debug("Skipped N scaling fit for T=%s: %s", t, e)

        # Group by N, fit T scaling
        n_values = sorted({r.num_states for r in pymdp_recs})
        for n in n_values:
            subset = [r for r in pymdp_recs if r.num_states == n]
            if len(subset) < 2:
                continue

            subset.sort(key=lambda r: r.num_timesteps)
            t_vals_dim = np.array([r.num_timesteps for r in subset], dtype=float)
            runtime_vals = np.array([r.execution_time for r in subset], dtype=float)

            try:
                valid = (t_vals_dim > 0) & (runtime_vals > 0)
                if np.sum(valid) >= 2:
                    log_x = np.log(t_vals_dim[valid])
                    log_y = np.log(runtime_vals[valid])
                    coeffs, residuals, rank, s, rcond = np.polyfit(
                        log_x, log_y, 1, full=True
                    )
                    exponent = coeffs[0]

                    # R^2 and RMSE
                    y_pred = coeffs[0] * log_x + coeffs[1]
                    rmse = np.sqrt(np.mean((log_y - y_pred) ** 2))
                    y_mean = np.mean(log_y)
                    ss_tot = np.sum((log_y - y_mean) ** 2)
                    ss_res = residuals[0] if len(residuals) > 0 else 0
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1

                    lines.append(
                        f"- **N={n}**: Runtime scales as **O(T^{exponent:.2f})** ($R^2$={r2:.3f}, RMSE={rmse:.3f}) "
                        f"(T={t_vals_dim[0]:,}→{t_vals_dim[-1]:,}: {runtime_vals[0]:.1f}s → {runtime_vals[-1]:.1f}s)"
                    )
            except Exception as e:
                self.logger.debug("Skipped T scaling fit for N=%s: %s", n, e)

        unc = [
            r
            for r in pymdp_recs
            if r.execution_benchmark_repeats > 1
            and r.execution_time_std is not None
            and r.execution_time_std > 0
        ]
        if unc:
            med_std = float(np.median([float(r.execution_time_std) for r in unc]))
            lines.append("")
            lines.append(
                f"**Repeat uncertainty**: {len(unc)} sweep cells used Step 12 benchmark repeats "
                f"(median within-cell σ ≈ {med_std:.3f}s on linear runtime). "
                "Fits use **median** seconds; scaling plots show ±σ bars when σ is recorded."
            )

        lines.append("")
        lines.append("### Interpretation of saved runtime fits")
        lines.append("")
        lines.append(
            "These log-log fits are descriptive summaries of the saved runtime "
            "observations for the listed parameter ranges. They do not establish "
            "asymptotic algorithmic complexity or performance outside those ranges."
        )
        lines.append(
            "Interpretation depends on the recorded models, execution configuration "
            "and environment. Runtime values alone do not support causal attribution "
            "to compilation, initialization or inference work."
        )

        return "\n".join(lines) if len(lines) > 2 else ""

    def _resource_efficiency(self) -> str:
        """Summary of code complexity and resource scaling."""
        loc_recs = [r for r in self.records if r.lines_of_code is not None]
        if not loc_recs:
            return ""

        lines: list[Any] = ["## Recorded Source Sizes & Throughput", ""]
        lines.append(
            "Descriptive source-size and timestep-throughput summaries from saved records."
        )

        # Throughput Table
        lines.append("### Inference Throughput")
        lines.append("")
        lines.append("| Model | Framework | Throughput (Steps/sec) |")
        lines.append("|---|---|---|")
        for r in sorted(
            self.records, key=lambda x: (x.num_states or 0, x.num_timesteps or 0)
        ):
            if r.execution_time > 0 and r.num_timesteps:
                throughput = r.num_timesteps / r.execution_time
                lines.append(f"| {r.sweep_label} | {r.framework} | {throughput:.2f} |")
        lines.append("")

        lines.append("### Recorded Source Sizes by State-Space Size")
        lines.append("")

        # Table of LOC vs N. LOC records come from the render summary, which
        # does not always carry sweep parameters, so num_states may be None —
        # bucket those under 0 instead of raising on mixed None/int sorting.
        n_values = sorted({r.num_states or 0 for r in loc_recs})
        frameworks = sorted({r.framework for r in loc_recs})

        header = "| N | " + " | ".join(f"{fw} LOC" for fw in frameworks) + " |"
        sep = "|---|" + "|".join(["---"] * len(frameworks)) + "|"
        lines.extend([header, sep])

        for n in n_values:
            row: list[Any] = [str(n)]
            for fw in frameworks:
                match = [
                    r
                    for r in loc_recs
                    if (r.num_states or 0) == n and r.framework == fw
                ]
                if match:
                    row.append(f"{match[0].lines_of_code:,}")
                else:
                    row.append("—")
            lines.append("| " + " | ".join(row) + " |")

        lines.append("\n> [!NOTE]")
        lines.append(
            "> Recorded source size describes these files and configurations only. "
            "LOC counts do not establish asymptotic source-size growth or computational complexity."
        )

        return "\n".join(lines)

    def _validation_section(self) -> str:
        """Structured validation summary (non-fatal checks)."""
        payload = self._validation_payload
        if not payload:
            return ""
        summary = payload.get("summary") or {}
        lines: list[Any] = [
            "## Sweep validation",
            "",
            f"- Machine-readable report: [{self._validation_json_name}]({self._validation_json_name})",
            f"- Issues: **info**={summary.get('info', 0)}, **warning**={summary.get('warning', 0)}, "
            f"**error**={summary.get('error', 0)}",
            "",
        ]
        grid = payload.get("grid") or {}
        if grid.get("expected_cells"):
            lines.append(
                f"- Expected grid cells (N×T): **{grid['expected_cells']}**, "
                f"records with (N,T): **{grid.get('record_cells', 0)}**"
            )
            lines.append("")
        issues = payload.get("issues") or []
        if issues:
            lines.append("### Sample issues")
            lines.append("")
            for item in issues[:12]:
                sev = item.get("severity", "?")
                code = item.get("code", "")
                msg = item.get("message", "")
                lines.append(f"- **{sev}** `{code}`: {msg}")
            if len(issues) > 12:
                lines.append(f"- … and **{len(issues) - 12}** more (see JSON)")
            lines.append("")
        return "\n".join(lines)

    def _meta_statistics_section(self) -> str:
        """Link aggregate statistics and show per-framework medians."""
        payload = self._statistics_payload
        if not payload or payload.get("error"):
            return ""
        lines: list[Any] = [
            "## Aggregate statistics",
            "",
            f"- Full export: [{self._statistics_json_name}]({self._statistics_json_name})",
            "",
        ]
        per_fw = payload.get("per_framework") or {}
        if per_fw:
            lines.append("### Per-framework runtime (successful runs)")
            lines.append("")
            lines.append("| Framework | Runs OK | Median runtime (s) | Mean (s) |")
            lines.append("|---|---|---|---|")
            for fw, block in sorted(per_fw.items()):
                med = block.get("runtime_median_s")
                mean_v = block.get("runtime_mean_s")
                med_s = f"{float(med):.4f}" if isinstance(med, (int, float)) else "—"
                mean_s = (
                    f"{float(mean_v):.4f}" if isinstance(mean_v, (int, float)) else "—"
                )
                lines.append(
                    f"| {fw} | {block.get('successful_runs', 0)} | {med_s} | {mean_s} |"
                )
            lines.append("")
        slopes = payload.get("loglog_runtime_vs_n_by_T") or {}
        if slopes:
            lines.append("### Log-log runtime vs N (PyMDP slopes by T)")
            lines.append("")
            for t_key, coef in sorted(slopes.items(), key=lambda kv: int(kv[0])):
                slope = coef.get("slope")
                r2 = coef.get("r_squared")
                n_pts = coef.get("n_points")
                slope_s = (
                    f"{float(slope):.3f}" if isinstance(slope, (int, float)) else "—"
                )
                r2_s = f"{float(r2):.3f}" if isinstance(r2, (int, float)) else "—"
                lines.append(f"- **T={t_key}**: slope={slope_s}, R²={r2_s}, n={n_pts}")
            lines.append("")
        return "\n".join(lines)

    def _gnn_footprint_section(self) -> str:
        """Step 3 serialization totals when format_statistics.json was ingested."""
        stats = self._gnn_format_statistics
        if not isinstance(stats, dict):
            return ""
        lines: list[Any] = ["## Step 3 serialization footprint", ""]
        rows: list[Any] = []
        for key in ("markdown", "python", "json"):
            block = stats.get(key)
            if not isinstance(block, dict):
                continue
            sz = block.get("total_size")
            if sz is None:
                continue
            rows.append((key, int(sz)))
        if not rows:
            return ""
        lines.append("| Format | Total size (MB) | Files OK |")
        lines.append("|---|---|---|")
        for key, sz in rows:
            block = stats[key]
            ok = block.get("successful", block.get("total_files", "—"))
            lines.append(f"| {key} | {sz / 1e6:.2f} | {ok} |")
        lines.append("")
        return "\n".join(lines)

    def _visualizations_section(self) -> str:
        """Reference generated plots and their plaintext data."""
        if not self.plot_paths:
            return ""

        lines: list[Any] = ["## Visualizations & Data Reports", ""]
        for p in self.plot_paths:
            path_obj = Path(p)
            name = path_obj.stem.replace("_", " ").title()
            fname = path_obj.name

            try:
                rel_path = path_obj.relative_to(self.output_dir)
            except ValueError:
                rel_path = Path(fname)

            csv_path = path_obj.with_suffix(".csv")
            try:
                rel_csv_path = csv_path.relative_to(self.output_dir)
            except ValueError:
                rel_csv_path = Path(csv_path.name)

            if fname.endswith(".csv"):
                lines.append(f"- **{name}**: Data Report: [{fname}]({rel_path})")
            elif csv_path.exists():
                lines.append(
                    f"- **{name}**: Image: [{fname}]({rel_path}) | Data Report: [{csv_path.name}]({rel_csv_path})"
                )
            else:
                lines.append(f"- **{name}**: [{fname}]({rel_path})")

        return "\n".join(lines)

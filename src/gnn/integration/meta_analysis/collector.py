"""
Sweep Data Collector — harvests runtime and simulation data from execution outputs.

Scans the 12_execute_output directory tree for execution_summary.json and
per-model simulation_results.json files, producing a list of SweepRecord
dataclass instances ready for analysis and visualization.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gnn.analysis.math_utils import compute_shannon_entropy


@dataclass
class SweepRecord:
    """One cell in a parameter sweep grid."""

    model_name: str
    framework: str

    # Sweep parameters (extracted from model_name, e.g. "pymdp_scaling_N3_T100")
    num_states: Any = None  # N dimension
    num_timesteps: Any = None  # T dimension

    # Execution metrics
    execution_time: float = 0.0  # seconds (median when benchmark repeats > 1)
    execution_time_std: Any = None
    execution_time_mean: Any = None
    execution_benchmark_repeats: int = 1
    execution_time_samples: list[float] | None = None
    success: bool = False
    timed_out: bool = False

    # Render metrics (from 11_render_output)
    lines_of_code: Any = None
    total_lines: Any = None

    # Simulation metrics (from simulation_results.json)
    final_accuracy: Any = None
    mean_belief_entropy: Any = None
    efe_trace: list[float] = field(default_factory=list)
    vfe_trace: list[float] = field(default_factory=list)

    # Raw model parameters
    model_params: dict[str, Any] = field(default_factory=dict)

    # Source paths
    simulation_results_path: str | None = None

    # Admitted gnn_execution_handoff_v1 identity (None for unversioned summaries)
    provenance: dict[str, Any] | None = None

    @property
    def sweep_label(self) -> str:
        """Human-readable label for this sweep cell."""
        parts: list[Any] = []
        if self.num_states is not None:
            parts.append(f"N={self.num_states}")
        if self.num_timesteps is not None:
            parts.append(f"T={self.num_timesteps}")
        return ", ".join(parts) if parts else self.model_name

    @property
    def time_per_step(self) -> float | None:
        """Execution time per simulation timestep (ms)."""
        if self.execution_time > 0 and self.num_timesteps and self.num_timesteps > 0:
            return float((self.execution_time / self.num_timesteps) * 1000.0)
        return None


# Regex to extract N and T from model names like "pymdp_scaling_N27_T10000"
_SWEEP_PARAM_RE = re.compile(r"N(\d+).*?T(\d+)", re.IGNORECASE)


def _parse_sweep_params(model_name: str) -> tuple[int | None, int | None]:
    """Extract (num_states, num_timesteps) from a model name."""
    m = _SWEEP_PARAM_RE.search(model_name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


class SweepDataCollector:
    """Collect parameter sweep data from the 12_execute_output tree."""

    def __init__(
        self,
        execute_output_dir: Path,
        render_output_dir: Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the instance."""
        self.execute_output_dir = Path(execute_output_dir)
        self.render_output_dir = Path(render_output_dir) if render_output_dir else None
        self.logger = logger or logging.getLogger(__name__)
        # Admission receipt of a versioned Step 12 handoff (None when unversioned)
        # and the refusal reason when versioned input was stale or mixed-run.
        self.handoff: dict[str, Any] | None = None
        self.handoff_refusal: str | None = None

    def collect(self) -> list[SweepRecord]:
        """Scan execution outputs and return sweep records.

        Strategy:
        1. Load execution_summary.json for runtime data per (model, framework).
        2. Walk model directories for simulation_results.json for metric data.
        3. Merge by (model_name, framework).

        A versioned (``gnn_execution_handoff_v1``) summary governs the whole
        collection: it must admit against the current run, historical
        execution logs are not mined, and simulation results are restricted
        to admitted (model, framework) cells. A refused handoff collects
        nothing. Unversioned summaries keep their historical behaviour when no run
        context governs them.
        """
        from gnn.pipeline.run_context import current_run_context

        records: dict[tuple[str, str], SweepRecord] = {}
        self.handoff = None
        self.handoff_refusal = None

        # Phase 1: execution summary (aggregate timing data)
        summary_path = self.execute_output_dir / "summaries" / "execution_summary.json"
        if summary_path.exists():
            self._collect_from_summary(summary_path, records)
        elif current_run_context() is not None:
            self.handoff_refusal = "Current run has no Step 12 execution summary"
        if self.handoff_refusal is not None:
            self.logger.error(f"Execution handoff refused: {self.handoff_refusal}")
            return []
        admitted_cells = set(records) if self.handoff is not None else None

        # Phase 2: per-script execution logs (fills in runtime gaps). Unversioned
        # only: the logs outlive their run and cannot be bound to a handoff.
        if admitted_cells is None:
            self._collect_from_execution_logs(records)

        # Phase 3: render metrics (LOC, etc.)
        if self.render_output_dir:
            self._collect_render_metrics(records)

        # Phase 4: simulation results (metric data)
        self._collect_simulation_results(records, admitted_cells)

        result = list(records.values())
        self.logger.info(
            f"Collected {len(result)} records "
            f"({len({r.model_name for r in result})} models, "
            f"{len({r.framework for r in result})} frameworks)"
        )
        return result

    def _collect_from_summary(
        self, summary_path: Path, records: dict[tuple[str, str], SweepRecord]
    ) -> None:
        """Extract runtime data from the execution summary."""
        import hashlib

        from gnn.pipeline.artifact_ownership import (
            admit_execution_summary,
            owned_artifact_checksums,
        )
        from gnn.pipeline.run_context import current_run_context

        try:
            raw = summary_path.read_bytes()
            data = json.loads(raw)
        except (json.JSONDecodeError, OSError) as e:
            if current_run_context() is not None:
                self.handoff_refusal = f"Unreadable execution summary: {e}"
            self.logger.warning(f"Failed to read execution summary: {e}")
            return
        try:
            if not isinstance(data, dict):
                raise ValueError("Execution summary must be a JSON object")
            context = current_run_context()
            if context is not None:
                root = Path(context.output_root).resolve()
                checksums = owned_artifact_checksums(root) or {}
                try:
                    relative = summary_path.resolve().relative_to(root).as_posix()
                except ValueError:
                    relative = None
                if checksums and checksums.get(relative or "") != (
                    hashlib.sha256(raw).hexdigest()
                ):
                    raise ValueError(
                        "Execution summary is not the current-run inventory artifact"
                    )
            self.handoff = admit_execution_summary(data)
        except (KeyError, OSError, TypeError, ValueError) as e:
            self.handoff_refusal = str(e)
            return

        for detail in data.get("execution_details", []):
            model_name = detail.get("model_name", "unknown")
            framework = detail.get("framework", "unknown")
            key = (model_name, framework)

            n, t = _parse_sweep_params(model_name)

            record = records.setdefault(
                key,
                SweepRecord(
                    model_name=model_name,
                    framework=framework,
                    num_states=n,
                    num_timesteps=t,
                ),
            )
            if self.handoff is not None:
                handoff = detail["_gnn_provenance"]
                record.provenance = {
                    field: handoff.get(field)
                    for field in (
                        "run_id",
                        "model_id",
                        "artifact_stem",
                        "source_relative_path",
                        "source_sha256",
                        "rendered_artifact_sha256",
                    )
                }

            record.execution_time = float(detail.get("execution_time", 0.0))
            record.execution_time_std = detail.get("execution_time_std")
            if record.execution_time_std is not None:
                record.execution_time_std = float(record.execution_time_std)
            mean_v = detail.get("execution_time_mean")
            record.execution_time_mean = float(mean_v) if mean_v is not None else None
            record.execution_benchmark_repeats = int(
                detail.get("execution_benchmark_repeats", 1)
            )
            samples = detail.get("execution_time_samples")
            if isinstance(samples, list):
                record.execution_time_samples = [float(x) for x in samples]
            record.success = detail.get("success", False) and not detail.get(
                "skipped", False
            )
            record.timed_out = "timed out" in detail.get("error", "").lower()

    def _collect_from_execution_logs(
        self, records: dict[tuple[str, str], SweepRecord]
    ) -> None:
        """Mine per-script execution_logs/*_results.json for runtime data.

        This catches timing data from prior runs that may have been overwritten
        in the aggregate execution_summary.json by a later partial re-run.
        """
        for log_file in self.execute_output_dir.rglob("execution_logs/*_results.json"):
            try:
                data = json.loads(log_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            model_name = data.get("model_name", "unknown")
            framework = data.get("framework", "unknown")
            key = (model_name, framework)

            exec_time = float(data.get("execution_time", 0.0))
            success = data.get("success", False)

            if exec_time <= 0 or not success:
                continue

            n, t = _parse_sweep_params(model_name)

            record = records.setdefault(
                key,
                SweepRecord(
                    model_name=model_name,
                    framework=framework,
                    num_states=n,
                    num_timesteps=t,
                ),
            )

            # Only update if we don't already have valid runtime data
            if record.execution_time <= 0:
                record.execution_time = exec_time
                record.success = True
                std_v = data.get("execution_time_std")
                if std_v is not None:
                    record.execution_time_std = float(std_v)
                mean_v = data.get("execution_time_mean")
                if mean_v is not None:
                    record.execution_time_mean = float(mean_v)
                record.execution_benchmark_repeats = int(
                    data.get("execution_benchmark_repeats", 1)
                )
                samples = data.get("execution_time_samples")
                if isinstance(samples, list):
                    record.execution_time_samples = [float(x) for x in samples]

    def _collect_render_metrics(
        self, records: dict[tuple[str, str], SweepRecord]
    ) -> None:
        """Harvest lines of code and other render-time metrics."""
        if self.render_output_dir is None:
            return
        summary_path = self.render_output_dir / "render_processing_summary.json"
        if not summary_path.exists():
            return

        try:
            data = json.loads(summary_path.read_text())
        except (json.JSONDecodeError, OSError):
            return

        file_results = data.get("file_results", {})
        for gnn_path, res in file_results.items():
            fw_results = res.get("framework_results", {})
            for fw, fw_res in fw_results.items():
                # We need to find the right record. GNN path usually contains the model name.
                # Or we can use the output file name if we can match it.
                # For scaling study, model_name is usually the folder name or from file_results keys.

                # Extract model name from GNN path (filename without extension)
                model_name = Path(gnn_path).stem
                key = (model_name, fw)

                if key in records:
                    metrics = fw_res.get("code_metrics", {})
                    records[key].lines_of_code = metrics.get("lines_of_code")
                    records[key].total_lines = metrics.get("total_lines")
                else:
                    # Try fuzzy match if exact model_name doesn't match
                    # (sometimes model_name in execute summary is slightly different)
                    for (r_model, r_fw), record in records.items():
                        if r_fw == fw and (
                            r_model in model_name or model_name in r_model
                        ):
                            metrics = fw_res.get("code_metrics", {})
                            record.lines_of_code = metrics.get("lines_of_code")
                            record.total_lines = metrics.get("total_lines")
                            break

    def _collect_simulation_results(
        self,
        records: dict[tuple[str, str], SweepRecord],
        admitted_cells: set[tuple[str, str]] | None = None,
    ) -> None:
        """Walk the output tree for *simulation_results.json files.

        Matches both the exact name ``simulation_results.json`` and the
        prefixed variant ``<ModelName>_simulation_results.json`` that the
        execute step may produce when multiple results coexist. With a
        versioned handoff only admitted (model, framework) cells are read, and
        under a run context only current-run inventory files.
        """
        from gnn.pipeline.artifact_ownership import current_artifact_files
        from gnn.pipeline.run_context import current_run_context

        if admitted_cells is not None and current_run_context() is not None:
            sim_files = current_artifact_files(
                self.execute_output_dir, "*simulation_results.json"
            )
        else:
            sim_files = sorted(
                self.execute_output_dir.rglob("*simulation_results.json")
            )
        for sim_file in sim_files:
            try:
                data = json.loads(sim_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            # Determine model_name and framework from path
            # Expected: .../12_execute_output/model_name/framework/simulation_data/simulation_results.json
            try:
                parts = sim_file.relative_to(self.execute_output_dir).parts
            except ValueError:
                parts = (
                    sim_file.resolve()
                    .relative_to(self.execute_output_dir.resolve())
                    .parts
                )
            if len(parts) >= 3:
                model_name = parts[0]
                framework = parts[1]
            else:
                model_name = data.get("model_name", sim_file.parent.parent.name)
                framework = data.get("framework", sim_file.parent.name)

            key = (model_name, framework)
            if admitted_cells is not None and key not in admitted_cells:
                continue
            n, t = _parse_sweep_params(model_name)

            record = records.setdefault(
                key,
                SweepRecord(
                    model_name=model_name,
                    framework=framework,
                    num_states=n,
                    num_timesteps=t or data.get("num_timesteps"),
                ),
            )

            record.simulation_results_path = str(sim_file)
            record.success = record.success or data.get("success", False)

            # Extract timesteps from data if not parsed from name
            if record.num_timesteps is None:
                record.num_timesteps = data.get("num_timesteps")

            # Model parameters
            record.model_params = data.get("model_parameters", {})

            # Metrics
            metrics = data.get("metrics", {})
            if "expected_free_energy" in metrics:
                efe_raw = metrics["expected_free_energy"]
                # EFE may be list-of-lists (one per timestep); take max per step
                if efe_raw and isinstance(efe_raw[0], list):
                    record.efe_trace = [max(step) if step else 0.0 for step in efe_raw]
                else:
                    record.efe_trace = efe_raw

            if "variational_free_energy" in metrics:
                record.vfe_trace = metrics["variational_free_energy"]

            # Compute final accuracy: fraction of correct observations
            observations = data.get("observations", [])
            true_states = data.get("true_states", [])
            if observations and true_states:
                matched = min(len(observations), len(true_states))
                correct = sum(
                    1
                    for o, s in zip(observations[:matched], true_states[:matched])
                    if o == s
                )
                record.final_accuracy = correct / matched if matched > 0 else None

            # Belief entropy (mean of last 10% of steps)
            beliefs = data.get("beliefs", [])
            if beliefs:
                window = max(1, len(beliefs) // 10)
                entropies: list[float] = []
                for belief in beliefs[-window:]:
                    if isinstance(belief, list):
                        entropies.append(compute_shannon_entropy(belief))
                if entropies:
                    record.mean_belief_entropy = sum(entropies) / len(entropies)

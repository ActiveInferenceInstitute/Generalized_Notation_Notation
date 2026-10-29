#!/usr/bin/env python3
"""
Thin orchestrator for PyMDP GNN Scaling Analysis.

This script end-to-end manages the PyMDP scaling study by:
1. Loading defaults from a plaintext config file (scripts/experiments/pymdp_scaling_config.yaml, located next to this runner).
2. Generating stochastic (non-trivial) GNN specifications across an N×T parameter grid.
3. Clearing previous generated files by default (configurable).
4. Invoking the core GNN pipeline (src/gnn/main.py) to parse, render, execute, and analyze them.

Dense matrix text for B is O(n^3) in output size; use ``max_n`` and ``max_file_size_mb`` in
config to avoid multi-gigabyte files and OSError: [Errno 28] when the volume is full.
Preflight messages align with Pipeline Step 5 (type checker) storage/resource estimation
(``src/gnn/type_checker/resource_estimator.py``); for per-file reports after generation, use
``src/gnn/5_type_checker.py --estimate-resources``.

Usage:
    uv run python scripts/experiments/run_pymdp_gnn_scaling_analysis.py
    uv run python scripts/experiments/run_pymdp_gnn_scaling_analysis.py --no-clear
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import yaml

# Repository root (parent of `scripts/`) for path hints and optional imports.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "scripts"))

from pymdp_spec_generator import (  # type: ignore[import-not-found]
    estimate_gnn_file_bytes,
    generate_gnn_file,
)

from gnn.utils.observability.visual_logging import create_visual_logger

# Default sweep grid
DEFAULT_N_VALUES = [2, 4, 8, 16]
DEFAULT_T_VALUES = [10, 100, 500, 1000]

# Default noise parameters
DEFAULT_A_SIGNAL = 0.85  # probability of correct observation
DEFAULT_B_SIGNAL = 0.80  # probability of intended transition

# Default pipeline timeout (seconds)
DEFAULT_TIMEOUT = 1200
DEFAULT_FRAMEWORKS = "pymdp"
DEFAULT_STRICT_FRAMEWORK_SUCCESS = True
DEFAULT_PIPELINE_OUTPUT_DIR = "output/pymdp_scaling_pipeline"
DEFAULT_PIPELINE_STEPS = "3,11,12,17"
DEFAULT_RUN_INTEGRATION_ON_FAILURE = False
DEFAULT_EXECUTION_WORKERS = 1
DEFAULT_DISTRIBUTED = False
DEFAULT_BACKEND = "ray"
DEFAULT_MATPLOTLIB_HEADLESS = True
DEFAULT_GNN_SERIALIZE_PRESET = "minimal"
DEFAULT_EXECUTION_BENCHMARK_REPEATS = 1

# Canonical sweep config lives next to the runner; never repoint to scripts/pymdp_scaling_config.yaml.
CONFIG_FILE = Path(__file__).resolve().parent / "pymdp_scaling_config.yaml"

# Margin added to total estimated spec bytes before comparing to free space (post-mkdir check).
MARGIN_BYTES = 50 * 1024 * 1024

# Defaults for output safety (B tensor expansion is O(n^3) in bytes).
DEFAULT_MAX_N = 256
# Single-file cap; B dominates (~24 * n^3 bytes). n=128 is ~50 MiB; n=256 needs ~500 MiB.
DEFAULT_MAX_FILE_SIZE_MB = 100
DEFAULT_MIN_FREE_DISK_MB = 200

# Step 5 (type checker) is the pipeline’s resource / storage estimation step; this script’s
# preflight is conceptually adjacent (headroom before writing dense InitialParameterization).
_STEP5_REF_SHORT = "Same policy role as Pipeline Step 5 (type_checker: resource_estimator / estimate_storage)."
# Last written preflight payload (pretty JSON; avoids multi-KiB lines on narrow terminals).
LAST_RESOURCE_GATE_JSON = CONFIG_FILE.parent / "pymdp_scaling_last_resource_gate.json"
RUN_MANIFEST_FILENAME = "pymdp_scaling_run_manifest.json"

# Initialize Visual Logger
logger = create_visual_logger("scaling_orchestrator")


class PipelineInvocation(NamedTuple):
    """A pipeline command phase used by the scaling orchestrator."""

    label: str
    cmd: list[str]
    cwd: Path


def _write_resource_gate_file(payload: dict[str, object]) -> Path | None:
    """Write full gate payload for tooling; return None if the volume is already full."""
    path = LAST_RESOURCE_GATE_JSON
    try:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
    except OSError as e:
        if e.errno == errno.ENOSPC:
            return None
        raise
    return path


def _print_resource_gate_path_or_compact(
    payload: dict[str, object], path_or_none: Path | None
) -> None:
    """Print the payload file path, or compact JSON if the file could not be written."""
    if path_or_none is not None:
        print(
            f"  resource_gate JSON: {path_or_none} (full payload; avoids wrap on small terminals)",
            file=sys.stderr,
        )
        return

    volume = payload.get("volume")
    free_bytes: object = None
    if isinstance(volume, dict):
        free_bytes = volume.get("free_bytes")
    compact: dict[str, object] = {
        "kind": payload.get("kind"),
        "check_path": payload.get("check_path"),
        "free_bytes": free_bytes,
        "policy_min_free_mib": payload.get("policy_min_free_mib"),
        "shortfall_bytes": payload.get("shortfall_bytes"),
        "total_estimated_spec_bytes": payload.get("total_estimated_spec_bytes"),
    }
    if "required_bytes_with_margin" in payload:
        compact["required_bytes_with_margin"] = payload.get(
            "required_bytes_with_margin"
        )
    print(
        f"  resource_gate (compact; could not write {LAST_RESOURCE_GATE_JSON.name}): "
        f"{json.dumps(compact, separators=(',', ':'))}",
        file=sys.stderr,
    )


# from pymdp_spec_generator import generate_gnn_file, estimate_gnn_file_bytes


def _skip_reason(
    n: int,
    t: int,
    *,
    max_n: int,
    max_file_bytes: int | None,
    a_signal: float,
    b_signal: float,
    timeout_n: int = 16,
    timeout_t: int = 3000,
) -> str | None:
    if n < 1 or t < 1:
        return "invalid"
    if n > max_n:
        return "max_n"
    est = estimate_gnn_file_bytes(n, t, a_signal, b_signal)
    if max_file_bytes is not None and est > max_file_bytes:
        return "max_size"
    if n >= timeout_n and t >= timeout_t:
        return "timeout_bounds"
    return None


class SweepPlan(NamedTuple):
    """Planned (N,T) jobs after max_n / max_size / timeout_bounds filters."""

    planned: list[tuple[int, int]]
    total_estimated_bytes: int
    skip_lines: list[str]
    grid_pairs: int


def build_sweep_plan(
    n_values: list[int],
    t_values: list[int],
    *,
    max_n: int,
    max_file_bytes: int | None,
    a_signal: float,
    b_signal: float,
    timeout_n: int = 16,
    timeout_t: int = 3000,
) -> SweepPlan:
    """Compute which grid cells are written and total estimated UTF-8 bytes (conservative)."""
    skip_lines: list[str] = []
    planned: list[tuple[int, int]] = []
    total_est = 0
    for n in n_values:
        for t in t_values:
            reason = _skip_reason(
                n,
                t,
                max_n=max_n,
                max_file_bytes=max_file_bytes,
                a_signal=a_signal,
                b_signal=b_signal,
                timeout_n=timeout_n,
                timeout_t=timeout_t,
            )
            if reason is None:
                total_est += estimate_gnn_file_bytes(n, t, a_signal, b_signal)
                planned.append((n, t))
            elif reason == "max_n":
                skip_lines.append(
                    f"  [SKIP] N={n}, T={t} (N > max_n={max_n}; dense B is O(n^3) bytes)"
                )
            elif reason == "max_size":
                est = estimate_gnn_file_bytes(n, t, a_signal, b_signal)
                skip_lines.append(
                    f"  [SKIP] N={n}, T={t} (estimated {est / (1024**2):.1f} MiB > --max-file-size-mb)"
                )
            elif reason == "timeout_bounds":
                skip_lines.append(
                    f"  [SKIP] N={n}, T={t} (exceeds reasonable execution bounds >= N{timeout_n} T{timeout_t})"
                )
    return SweepPlan(
        planned=planned,
        total_estimated_bytes=total_est,
        skip_lines=skip_lines,
        grid_pairs=len(n_values) * len(t_values),
    )


def _resolve_disk_check_path(out_dir: Path) -> Path:
    """Path used for statvfs: prefer existing out_dir, else its parent (volume still correct)."""
    if out_dir.exists():
        return out_dir
    return out_dir.parent


def _resolve_output_dir(output_dir: str) -> Path:
    """Resolve output paths from the repository root, regardless of the caller's cwd."""
    path = Path(output_dir).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _build_pipeline_invocation(
    out_dir: Path,
    pipeline_output_dir: Path,
    timeout: int,
    frameworks: str,
    strict_framework_success: bool = DEFAULT_STRICT_FRAMEWORK_SUCCESS,
    pipeline_steps: str = DEFAULT_PIPELINE_STEPS,
    execution_workers: int = DEFAULT_EXECUTION_WORKERS,
    distributed: bool = DEFAULT_DISTRIBUTED,
    backend: str = DEFAULT_BACKEND,
    *,
    serialize_preset: str = "full",
    execution_benchmark_repeats: int = 1,
) -> tuple[list[str], Path]:
    """Build the uv command and cwd used for invoking the core pipeline."""
    cmd = [
        "uv",
        "run",
        "python",
        "src/gnn/main.py",
        "--target-dir",
        str(out_dir),
        "--output-dir",
        str(pipeline_output_dir),
        "--render-output-dir",
        str(pipeline_output_dir / "11_render_output"),
        "--only-steps",
        pipeline_steps,
        "--timeout",
        str(timeout),
        "--frameworks",
        frameworks,
        "--execution-workers",
        str(max(1, int(execution_workers))),
        "--verbose",
    ]
    if strict_framework_success:
        cmd.append("--strict-framework-success")
    if distributed:
        cmd.append("--distributed")
        cmd.extend(["--backend", backend])
    preset_norm = (serialize_preset or "full").strip().lower()
    if preset_norm == "minimal":
        cmd.extend(["--serialize-preset", "minimal"])
    bench_rep = max(1, int(execution_benchmark_repeats))
    if bench_rep > 1:
        cmd.extend(["--execution-benchmark-repeats", str(bench_rep)])
    return cmd, PROJECT_ROOT


def _parse_pipeline_steps(pipeline_steps: str) -> list[str]:
    """Parse a comma-separated pipeline step list into normalized step numbers."""
    return [step.strip() for step in pipeline_steps.split(",") if step.strip()]


def _format_pipeline_steps(steps: list[str]) -> str:
    """Format normalized step numbers for the core pipeline CLI."""
    return ",".join(steps)


def _build_pipeline_invocations(
    out_dir: Path,
    pipeline_output_dir: Path,
    timeout: int,
    frameworks: str,
    strict_framework_success: bool = DEFAULT_STRICT_FRAMEWORK_SUCCESS,
    pipeline_steps: str = DEFAULT_PIPELINE_STEPS,
    run_integration_on_failure: bool = DEFAULT_RUN_INTEGRATION_ON_FAILURE,
    execution_workers: int = DEFAULT_EXECUTION_WORKERS,
    distributed: bool = DEFAULT_DISTRIBUTED,
    backend: str = DEFAULT_BACKEND,
    *,
    serialize_preset: str = "full",
    execution_benchmark_repeats: int = 1,
) -> list[PipelineInvocation]:
    """Build composable pipeline phases for a scaling run.

    Step 17 consumes execution outputs, so the default behavior runs it as a
    dependent phase only after earlier phases succeed. This avoids publishing
    partial meta-analysis when Step 12 is interrupted or fails.
    """
    steps = _parse_pipeline_steps(pipeline_steps)
    if not steps:
        raise ValueError("pipeline_steps must include at least one step")

    if run_integration_on_failure or "17" not in steps or len(steps) == 1:
        cmd, cwd = _build_pipeline_invocation(
            out_dir,
            pipeline_output_dir,
            timeout,
            frameworks,
            strict_framework_success=strict_framework_success,
            pipeline_steps=_format_pipeline_steps(steps),
            execution_workers=execution_workers,
            distributed=distributed,
            backend=backend,
            serialize_preset=serialize_preset,
            execution_benchmark_repeats=execution_benchmark_repeats,
        )
        label = "integration" if steps == ["17"] else "main"
        return [PipelineInvocation(label=label, cmd=cmd, cwd=cwd)]

    pre_integration_steps = [step for step in steps if step != "17"]
    phases: list[PipelineInvocation] = []
    if pre_integration_steps:
        cmd, cwd = _build_pipeline_invocation(
            out_dir,
            pipeline_output_dir,
            timeout,
            frameworks,
            strict_framework_success=strict_framework_success,
            pipeline_steps=_format_pipeline_steps(pre_integration_steps),
            execution_workers=execution_workers,
            distributed=distributed,
            backend=backend,
            serialize_preset=serialize_preset,
            execution_benchmark_repeats=execution_benchmark_repeats,
        )
        phases.append(PipelineInvocation(label="main", cmd=cmd, cwd=cwd))

    cmd, cwd = _build_pipeline_invocation(
        out_dir,
        pipeline_output_dir,
        timeout,
        frameworks,
        strict_framework_success=strict_framework_success,
        pipeline_steps="17",
        execution_workers=execution_workers,
        distributed=distributed,
        backend=backend,
        serialize_preset=serialize_preset,
        execution_benchmark_repeats=execution_benchmark_repeats,
    )
    phases.append(PipelineInvocation(label="integration", cmd=cmd, cwd=cwd))
    return phases


def _build_run_manifest(
    *,
    status: str,
    plan: SweepPlan,
    out_dir: Path,
    pipeline_output_dir: Path,
    config: dict[str, object],
    phases: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Build a reproducibility manifest for a scaling experiment."""
    return {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "configuration": config,
        "paths": {
            "generated_specs": str(out_dir),
            "pipeline_output": str(pipeline_output_dir),
            "render_output": str(pipeline_output_dir / "11_render_output"),
            "execution_output": str(pipeline_output_dir / "12_execute_output"),
            "integration_output": str(pipeline_output_dir / "17_integration_output"),
        },
        "grid_pairs": plan.grid_pairs,
        "planned_count": len(plan.planned),
        "planned_pairs": [[n, t] for n, t in plan.planned],
        "skip_lines": plan.skip_lines,
        "estimated_spec_bytes": plan.total_estimated_bytes,
        "estimated_spec_mib": round(plan.total_estimated_bytes / (1024**2), 3),
        "phases": phases or [],
    }


def _write_run_manifest(pipeline_output_dir: Path, manifest: dict[str, object]) -> Path:
    """Write the scaling run manifest into the isolated pipeline output."""
    pipeline_output_dir.mkdir(parents=True, exist_ok=True)
    path = pipeline_output_dir / RUN_MANIFEST_FILENAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    return path


def _usage_snapshot(path: Path) -> dict[str, Any]:
    u = shutil.disk_usage(path)
    total = u.total
    used = u.total - u.free
    used_pct = (100.0 * used / total) if total else 0.0
    return {
        "path": str(path.resolve()),
        "total_bytes": total,
        "used_bytes": used,
        "free_bytes": u.free,
        "used_percent": used_pct,
    }


def _path_cwd_note(out_dir: Path) -> str | None:
    """Warn when cwd makes output land under `scripts/` or outside the repo."""
    try:
        out_r = out_dir.resolve()
        pr = PROJECT_ROOT.resolve()
    except OSError:
        return None
    scripts_dir = pr / "scripts"
    try:
        out_r.relative_to(scripts_dir)
        return (
            f"Output is under {scripts_dir} (relative output_dir from running inside scripts/). "
            f"To write next to the repo’s input tree, `cd {pr}` then run via "
            "`uv run python scripts/experiments/run_pymdp_gnn_scaling_analysis.py`, or set `output_dir` to "
            f"`{pr / 'input' / 'gnn_files' / 'pymdp_scaling_study'}`."
        )
    except ValueError:
        return None
    if pr not in (out_r, *out_r.parents):
        return (
            f"Output resolves to {out_r}, outside the repository root {pr}. "
            "Use --output-dir under the project if that was unintended."
        )
    return None


def _resource_gate_dict(
    *,
    kind: str,
    check_path: Path,
    plan: SweepPlan,
    policy_min_free_bytes: int,
    margin_bytes: int,
    a_signal: float,
    b_signal: float,
) -> dict[str, object]:
    snap = _usage_snapshot(check_path)
    need_b = policy_min_free_bytes
    free_b = int(snap["free_bytes"])
    shortfall = max(0, need_b - free_b)
    largest = max(plan.planned, key=lambda nt: (nt[0], nt[1]), default=(0, 0))
    return {
        "kind": kind,
        "preflight": "pymdp_scaling",
        "pipeline_alignment": {
            "step": 5,
            "module": "type_checker",
            "role": "storage_and_resource_estimation",
            "see": "src/gnn/type_checker/resource_estimator.py; estimation_strategies.estimate_storage",
        },
        "volume": {k: v for k, v in snap.items() if k != "path"},
        "check_path": snap["path"],
        "policy_min_free_bytes": need_b,
        "policy_min_free_mib": round(need_b / (1024 * 1024), 3),
        "shortfall_bytes": shortfall,
        "planned_spec_count": len(plan.planned),
        "grid_pairs": plan.grid_pairs,
        "total_estimated_spec_bytes": plan.total_estimated_bytes,
        "total_estimated_spec_gib": round(plan.total_estimated_bytes / (1024**3), 6),
        "largest_planned_n": largest[0],
        "largest_planned_t": largest[1],
        "write_margin_bytes": margin_bytes,
        "signal": {"a": a_signal, "b": b_signal},
    }


def _print_min_free_violation(
    plan: SweepPlan,
    out_dir: Path,
    check_path: Path,
    need_m: int,
    *,
    a_signal: float,
    b_signal: float,
) -> None:
    need_b = int(need_m * 1024 * 1024)
    snap = _usage_snapshot(check_path)
    free_b = int(snap["free_bytes"])
    shortfall = max(0, need_b - free_b)
    total_b = int(snap["total_bytes"])
    used_pct = float(snap["used_percent"])
    free_gib = free_b / (1024**3)
    total_gib = total_b / (1024**3)

    logger.print_status(
        "Preflight resource gate: insufficient free space for policy.", "error"
    )

    data = {
        "Volume": str(check_path.resolve()),
        "Space": f"{free_gib:.3f} GiB free of {total_gib:.3f} GiB total ({used_pct:.1f}% used)",
        "Policy": f"Require ≥ {need_m} MiB free",
        "Shortfall": f"≈ {shortfall / (1024**2):.1f} MiB",
        "Sweep": f"{plan.grid_pairs} grid pairs; {len(plan.planned)} spec(s) planned",
        "Estimated Text": f"≈ {plan.total_estimated_bytes / (1024**2):.1f} MiB",
    }
    logger.print_summary("Resource Shortfall", data)

    note = _path_cwd_note(out_dir)
    payload = _resource_gate_dict(
        kind="min_free_policy_violation",
        check_path=check_path,
        plan=plan,
        policy_min_free_bytes=need_b,
        margin_bytes=0,
        a_signal=a_signal,
        b_signal=b_signal,
    )
    gate_path = _write_resource_gate_file(payload)
    _print_resource_gate_path_or_compact(payload, gate_path)

    if note:
        logger.print_status(note, "warning")

    logger.print_error_with_recovery(
        "Insufficient disk space for policy.",
        [
            "Free disk space on the target volume",
            f"Lower min_free_disk_mb in {CONFIG_FILE.name}",
            "Use --min-free-disk-mb 0 to skip this check (risky)",
        ],
    )


def _print_aggregate_violation(
    plan: SweepPlan,
    out_dir: Path,
    free_b: int,
    *,
    a_signal: float,
    b_signal: float,
) -> None:
    margin = MARGIN_BYTES
    stat_path = out_dir if out_dir.exists() else out_dir.parent
    snap = _usage_snapshot(stat_path)

    logger.print_status(
        "Preflight resource gate: planned GNN file bytes exceed free space.", "error"
    )

    data = {
        "Estimated Size": f"{plan.total_estimated_bytes / (1024**3):.3f} GiB",
        "Margin": f"{margin // (1024**2)} MiB",
        "Total Needed": f"{(plan.total_estimated_bytes + margin) / (1024**3):.3f} GiB",
        "Free on Volume": f"{free_b / (1024**3):.3f} GiB",
        "Usage": f"{float(snap['used_percent']):.1f}% used",
    }
    logger.print_summary("Capacity Violation", data)

    payload = _resource_gate_dict(
        kind="aggregate_planned_size_violation",
        check_path=stat_path,
        plan=plan,
        policy_min_free_bytes=0,
        margin_bytes=margin,
        a_signal=a_signal,
        b_signal=b_signal,
    )
    payload["required_bytes_with_margin"] = plan.total_estimated_bytes + margin
    gate_path = _write_resource_gate_file(payload)
    _print_resource_gate_path_or_compact(payload, gate_path)

    logger.print_error_with_recovery(
        "Estimated specs exceed available space.",
        [
            "Shrink n_values in configuration",
            "Increase max_n or max_file_size skips",
            "Free disk space on the target volume",
        ],
    )


def _load_and_validate_config() -> dict:
    """Load configuration from YAML, creating it with defaults if missing."""
    if not CONFIG_FILE.exists():
        default_config = {
            "n_values": DEFAULT_N_VALUES,
            "t_values": DEFAULT_T_VALUES,
            "a_signal": DEFAULT_A_SIGNAL,
            "b_signal": DEFAULT_B_SIGNAL,
            "timeout": DEFAULT_TIMEOUT,
            "output_dir": "input/gnn_files/pymdp_scaling_study",
            "pipeline_output_dir": DEFAULT_PIPELINE_OUTPUT_DIR,
            "clear_outputs_before_run": True,
            "frameworks": DEFAULT_FRAMEWORKS,
            "strict_framework_success": DEFAULT_STRICT_FRAMEWORK_SUCCESS,
            "pipeline_steps": DEFAULT_PIPELINE_STEPS,
            "run_integration_on_failure": DEFAULT_RUN_INTEGRATION_ON_FAILURE,
            "max_n": DEFAULT_MAX_N,
            "max_file_size_mb": DEFAULT_MAX_FILE_SIZE_MB,
            "min_free_disk_mb": DEFAULT_MIN_FREE_DISK_MB,
            "type_check": True,
            "skip_timeout_bounds_n": 16,
            "skip_timeout_bounds_t": 3000,
            "execution_workers": 1,
            "distributed": False,
            "backend": "ray",
            "matplotlib_headless": DEFAULT_MATPLOTLIB_HEADLESS,
            "gnn_serialize_preset": DEFAULT_GNN_SERIALIZE_PRESET,
            "execution_benchmark_repeats": DEFAULT_EXECUTION_BENCHMARK_REPEATS,
        }
        with open(CONFIG_FILE, "w") as f:
            yaml.dump(default_config, f, sort_keys=False)
        logger.print_status(
            f"Created default config file at {CONFIG_FILE.relative_to(PROJECT_ROOT)}",
            "success",
        )
        return default_config

    try:
        with open(CONFIG_FILE, "r") as f:
            config = yaml.safe_load(f)
            if not isinstance(config, dict):
                config = {}
            return config
    except yaml.YAMLError as e:
        logger.print_status(f"Error parsing {CONFIG_FILE.name}: {e}", "error")
        sys.exit(1)


def _get_parser(config: dict) -> argparse.ArgumentParser:
    """Create the argument parser for the scaling orchestrator."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end PyMDP GNN scaling analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n-values",
        type=str,
        default=",".join(str(n) for n in config.get("n_values", DEFAULT_N_VALUES)),
        help="Comma-separated N values",
    )
    parser.add_argument(
        "--t-values",
        type=str,
        default=",".join(str(t) for t in config.get("t_values", DEFAULT_T_VALUES)),
        help="Comma-separated T values",
    )
    parser.add_argument(
        "--a-signal",
        type=float,
        default=config.get("a_signal", DEFAULT_A_SIGNAL),
        help="A-matrix signal strength",
    )
    parser.add_argument(
        "--b-signal",
        type=float,
        default=config.get("b_signal", DEFAULT_B_SIGNAL),
        help="B-matrix signal strength",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.get("output_dir", "input/gnn_files/pymdp_scaling_study"),
        help="Output directory for generated GNN files",
    )
    parser.add_argument(
        "--pipeline-output-dir",
        type=str,
        default=config.get("pipeline_output_dir", DEFAULT_PIPELINE_OUTPUT_DIR),
        help="Pipeline output directory for this scaling run",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=config.get("timeout", DEFAULT_TIMEOUT),
        help="Pipeline execution timeout in seconds",
    )
    parser.add_argument(
        "--frameworks",
        type=str,
        default=config.get("frameworks", DEFAULT_FRAMEWORKS),
        help="Frameworks to render and execute (default: pymdp)",
    )
    parser.add_argument(
        "--execution-workers",
        type=int,
        default=config.get("execution_workers", DEFAULT_EXECUTION_WORKERS),
        help="Number of local or distributed workers for Step 12 execution",
    )
    parser.add_argument(
        "--distributed",
        action=argparse.BooleanOptionalAction,
        default=config.get("distributed", DEFAULT_DISTRIBUTED),
        help="Route Step 12 script execution through the Ray/Dask distributed dispatcher",
    )
    parser.add_argument(
        "--backend",
        choices=["ray", "dask"],
        default=config.get("backend", DEFAULT_BACKEND),
        help="Distributed backend for Step 12 when --distributed is enabled",
    )
    parser.add_argument(
        "--strict-framework-success",
        action=argparse.BooleanOptionalAction,
        default=config.get(
            "strict_framework_success", DEFAULT_STRICT_FRAMEWORK_SUCCESS
        ),
        help="Fail Step 11 if any requested framework render fails",
    )
    parser.add_argument(
        "--pipeline-steps",
        type=str,
        default=config.get("pipeline_steps", DEFAULT_PIPELINE_STEPS),
        help="Pipeline steps for the study (default: 3,11,12,17)",
    )
    parser.add_argument(
        "--run-integration-on-failure",
        action=argparse.BooleanOptionalAction,
        default=config.get(
            "run_integration_on_failure", DEFAULT_RUN_INTEGRATION_ON_FAILURE
        ),
        help="Allow Step 17 to run even if an earlier step fails",
    )

    clear_default = config.get("clear_outputs_before_run", True)
    if clear_default:
        parser.add_argument(
            "--no-clear",
            action="store_false",
            dest="clear",
            help="Do not clear generated output dir before run",
        )
        parser.set_defaults(clear=True)
    else:
        parser.add_argument(
            "--clear",
            action="store_true",
            dest="clear",
            help="Clear generated output dir before run",
        )
        parser.set_defaults(clear=False)

    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Only generate the files, do not run the pipeline",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=config.get("max_n", DEFAULT_MAX_N),
        help="Skip N larger than this",
    )
    parser.add_argument(
        "--max-file-size-mb",
        type=float,
        default=config.get("max_file_size_mb", DEFAULT_MAX_FILE_SIZE_MB),
        help="Skip a spec if estimated file size would exceed this",
    )
    parser.add_argument(
        "--min-free-disk-mb",
        type=float,
        default=config.get("min_free_disk_mb", DEFAULT_MIN_FREE_DISK_MB),
        help="Abort if free space on the output volume is below this",
    )
    parser.add_argument(
        "--type-check",
        action=argparse.BooleanOptionalAction,
        default=config.get("type_check", True),
        help="Include step 5 (type_check) in pipeline steps",
    )
    parser.add_argument(
        "--matplotlib-headless",
        action=argparse.BooleanOptionalAction,
        default=config.get("matplotlib_headless", DEFAULT_MATPLOTLIB_HEADLESS),
        help="Set MPLBACKEND=Agg for pipeline subprocesses (recommended on macOS batch runs)",
    )
    parser.add_argument(
        "--gnn-serialize-preset",
        choices=["full", "minimal"],
        default=config.get("gnn_serialize_preset", DEFAULT_GNN_SERIALIZE_PRESET),
        help="Forwarded to Step 3: multi-format serialization subset",
    )
    parser.add_argument(
        "--execution-benchmark-repeats",
        type=int,
        default=int(
            config.get(
                "execution_benchmark_repeats", DEFAULT_EXECUTION_BENCHMARK_REPEATS
            )
        ),
        help="Forwarded to Step 12: sequential repeats per script (median timing when >1)",
    )
    parser.add_argument(
        "--factorized",
        action=argparse.BooleanOptionalAction,
        default=config.get("factorized", False),
        help="Run the sparse Kronecker-factorized sweep (roadmap MAJ-02) instead of "
        "the dense pipeline sweep: per-factor matrices are executed with mean-field "
        "active inference in JAX and the joint state space is never materialised",
    )
    parser.add_argument(
        "--factors",
        type=str,
        default=config.get("factors", "2,2,2,2,2,2"),
        help="Comma-separated per-factor state sizes for the factorized sweep "
        "(joint state space = product; default 2x2x2x2x2x2 = 64 states)",
    )
    parser.add_argument(
        "--factor-timesteps",
        type=str,
        default=config.get("factor_timesteps", "10,100"),
        help="Comma-separated T values for the factorized sweep",
    )
    parser.add_argument(
        "--factor-output-dir",
        type=str,
        default=config.get(
            "factor_output_dir", "input/gnn_files/pymdp_kronecker_study"
        ),
        help="Output directory for generated factorized GNN spec files",
    )
    return parser


def _run_factorized_sweep(args: argparse.Namespace) -> int:
    """Run the sparse Kronecker-factorized sweep (roadmap MAJ-02).

    Builds factor-separable models whose joint state space is the product of
    per-factor sizes and executes them through the Step 12 public entry
    ``execute.jax.kronecker_executor.execute_kronecker_factorized`` — the
    joint is never materialised, and each run writes the standard executor
    artifacts (``simulation_data/simulation_results.json`` carrying the
    ``jax_kronecker_factorized_v1`` schema plus ``kronecker_execution_summary.json``)
    under ``<pipeline_output_dir>/pymdp_kronecker_pipeline/T<t>``. Also writes
    the equivalent multi-factor GNN spec files (per-factor A/B/C/D) for
    provenance, and a JSON manifest with the joint size, artifact paths, wall
    time and validation for every (factors, T) pair.

    Returns 0 on success, 1 on failure.
    """
    from pymdp_spec_generator import generate_factorized_gnn_file

    from gnn.execute.jax.kronecker_executor import execute_kronecker_factorized

    try:
        factor_sizes = [int(x.strip()) for x in args.factors.split(",")]
    except ValueError:
        logger.print_status(f"Invalid --factors value: {args.factors!r}", "error")
        return 1
    if not factor_sizes or any(n < 2 for n in factor_sizes):
        logger.print_status(
            "--factors must be a non-empty list of integers >= 2", "error"
        )
        return 1
    try:
        t_values = [int(x.strip()) for x in args.factor_timesteps.split(",")]
    except ValueError:
        logger.print_status(
            f"Invalid --factor-timesteps value: {args.factor_timesteps!r}", "error"
        )
        return 1

    joint_size = 1
    for n in factor_sizes:
        joint_size *= n
    logger.print_header(
        "Kronecker-Factorized Sweep",
        f"factors={factor_sizes} joint={joint_size} states (never materialised)",
    )

    spec_dir = _resolve_output_dir(args.factor_output_dir)
    pipeline_output_dir = _resolve_output_dir(args.pipeline_output_dir)
    manifest_path = pipeline_output_dir / "pymdp_kronecker_scaling_manifest.json"
    pipeline_output_dir.mkdir(parents=True, exist_ok=True)

    def _rel(path: Path) -> str:
        try:
            return str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)

    runs: list[dict[str, object]] = []
    for t in t_values:
        spec_text = generate_factorized_gnn_file(factor_sizes, t)
        spec_path = spec_dir / f"kronecker_N{joint_size}_T{t}.md"
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(spec_text, encoding="utf-8")

        run_output_dir = pipeline_output_dir / "pymdp_kronecker_pipeline" / f"T{t}"
        started = time.monotonic()
        envelope = execute_kronecker_factorized(
            {
                "factor_sizes": factor_sizes,
                "t": t,
                "seed": 42,
                "model_name": f"kronecker_N{joint_size}_T{t}",
            },
            run_output_dir,
        )
        elapsed = time.monotonic() - started
        simulation = envelope["simulation"]
        all_valid = bool(simulation["validation"]["all_valid"])
        runs.append(
            {
                "factor_sizes": factor_sizes,
                "joint_state_space_size": joint_size,
                "timesteps": t,
                "spec_file": _rel(spec_path),
                "output_dir": _rel(run_output_dir),
                "simulation_results_file": _rel(
                    run_output_dir / "simulation_data" / "simulation_results.json"
                ),
                "execution_summary_file": _rel(
                    run_output_dir / "kronecker_execution_summary.json"
                ),
                "elapsed_seconds": round(elapsed, 4),
                "validation_all_valid": all_valid,
                "joint_materialized": simulation["model_parameters"][
                    "joint_materialized"
                ],
            }
        )
        logger.print_status(
            f"N={joint_size} T={t}: {elapsed:.3f}s all_valid={all_valid} "
            "(joint not materialised)",
            "success" if all_valid else "error",
        )

    manifest = {
        "schema_version": "pymdp_kronecker_scaling_manifest_v1",
        "factor_sizes": factor_sizes,
        "joint_state_space_size": joint_size,
        "backend": "jax-mean-field",
        "executor": "gnn.execute.jax.kronecker_executor.execute_kronecker_factorized",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.print_status(
        f"Kronecker manifest: {manifest_path} ({len(runs)} runs)", "success"
    )
    return 0


def main() -> int:
    start_time = datetime.now(timezone.utc)
    config = _load_and_validate_config()
    parser = _get_parser(config)
    args = parser.parse_args()

    logger.print_header("PyMDP GNN Scaling Analysis", "Exponential Performance Sweep")

    if args.factorized:
        return _run_factorized_sweep(args)

    n_values = [int(x.strip()) for x in args.n_values.split(",")]
    t_values = [int(x.strip()) for x in args.t_values.split(",")]

    out_dir = _resolve_output_dir(args.output_dir)
    pipeline_output_dir = _resolve_output_dir(args.pipeline_output_dir)
    max_file_bytes = (
        int(args.max_file_size_mb * 1024 * 1024) if args.max_file_size_mb > 0 else None
    )

    plan = build_sweep_plan(
        n_values,
        t_values,
        max_n=args.max_n,
        max_file_bytes=max_file_bytes,
        a_signal=args.a_signal,
        b_signal=args.b_signal,
        timeout_n=config.get("skip_timeout_bounds_n", 16),
        timeout_t=config.get("skip_timeout_bounds_t", 3000),
    )

    for line in plan.skip_lines:
        logger.print_status(line, "warning")

    if not plan.planned:
        logger.print_status(
            "No GNN specs to write (entire grid skipped or empty).", "error"
        )
        return 1

    # Preflight Check
    check_path = _resolve_disk_check_path(out_dir)
    if args.min_free_disk_mb > 0:
        need_b = int(args.min_free_disk_mb * 1024 * 1024)
        if shutil.disk_usage(check_path).free < need_b:
            _print_min_free_violation(
                plan,
                out_dir,
                check_path,
                int(args.min_free_disk_mb),
                a_signal=args.a_signal,
                b_signal=args.b_signal,
            )
            return 1

    if plan.total_estimated_bytes + MARGIN_BYTES > shutil.disk_usage(check_path).free:
        _print_aggregate_violation(
            plan,
            out_dir,
            shutil.disk_usage(check_path).free,
            a_signal=args.a_signal,
            b_signal=args.b_signal,
        )
        return 1

    # Final Step Setup
    pipeline_steps_list = args.pipeline_steps.split(",")
    if args.type_check and "5" not in pipeline_steps_list:
        if "3" in pipeline_steps_list:
            pipeline_steps_list.insert(pipeline_steps_list.index("3") + 1, "5")
        else:
            pipeline_steps_list.insert(0, "5")
    args.pipeline_steps = ",".join(pipeline_steps_list)

    # Execution Flow
    try:
        # 1. Cleanup
        if args.clear:
            for d in [out_dir, pipeline_output_dir]:
                if d.exists():
                    logger.print_status(
                        f"Clearing existing outputs in {d.relative_to(PROJECT_ROOT)}...",
                        "info",
                    )
                    shutil.rmtree(d)

        out_dir.mkdir(parents=True, exist_ok=True)
        pipeline_output_dir.mkdir(parents=True, exist_ok=True)

        run_config = vars(args)
        manifest_path = _write_run_manifest(
            pipeline_output_dir,
            _build_run_manifest(
                status="planned",
                plan=plan,
                out_dir=out_dir,
                pipeline_output_dir=pipeline_output_dir,
                config=run_config,
            ),
        )
        logger.print_status(
            f"Scaling run manifest: {manifest_path.relative_to(PROJECT_ROOT)}", "info"
        )

        # 2. Generation
        logger.print_status(
            f"Generating {len(plan.planned)} GNN specs ({plan.total_estimated_bytes / (1024**2):.1f} MiB)...",
            "progress",
        )
        count = 0
        for n, t in plan.planned:
            filepath = out_dir / f"pymdp_scaling_N{n}_T{t}.md"
            filepath.write_text(generate_gnn_file(n, t, args.a_signal, args.b_signal))
            count += 1
        logger.print_status(
            f"Generated {count} GNN spec files in {args.output_dir}", "success"
        )

        if args.skip_pipeline:
            logger.print_status(
                "Skipping pipeline execution (--skip-pipeline passed).", "info"
            )
            return 0

        # 3. Pipeline Invocation
        phases = _build_pipeline_invocations(
            out_dir,
            pipeline_output_dir,
            args.timeout,
            args.frameworks,
            strict_framework_success=args.strict_framework_success,
            pipeline_steps=args.pipeline_steps,
            run_integration_on_failure=args.run_integration_on_failure,
            execution_workers=max(1, args.execution_workers),
            distributed=args.distributed,
            backend=args.backend,
            serialize_preset=args.gnn_serialize_preset,
            execution_benchmark_repeats=max(1, int(args.execution_benchmark_repeats)),
        )

        phase_results = []
        for phase in phases:
            logger.print_status(
                f"Pipeline phase '{phase.label}': Steps {phase.cmd[phase.cmd.index('--only-steps') + 1]}",
                "rocket",
            )
            try:
                phase_env = os.environ.copy()
                if args.matplotlib_headless:
                    phase_env.setdefault("MPLBACKEND", "Agg")
                subprocess.run(phase.cmd, check=True, cwd=phase.cwd, env=phase_env)
                phase_results.append(
                    {"label": phase.label, "status": "success", "exit_code": 0}
                )
            except subprocess.CalledProcessError as e:
                phase_results.append(
                    {
                        "label": phase.label,
                        "status": "failed",
                        "exit_code": e.returncode,
                    }
                )
                _write_run_manifest(
                    pipeline_output_dir,
                    _build_run_manifest(
                        status="failed",
                        plan=plan,
                        out_dir=out_dir,
                        pipeline_output_dir=pipeline_output_dir,
                        config=run_config,
                        phases=phase_results,
                    ),
                )
                logger.print_status(
                    f"Pipeline phase '{phase.label}' failed with exit code {e.returncode}",
                    "error",
                )
                return e.returncode

        # 4. Completion
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        _write_run_manifest(
            pipeline_output_dir,
            _build_run_manifest(
                status="success",
                plan=plan,
                out_dir=out_dir,
                pipeline_output_dir=pipeline_output_dir,
                config=run_config,
                phases=phase_results,
            ),
        )

        logger.print_completion_banner(
            True,
            duration,
            {
                "Total Files": count,
                "Grid Pairs": plan.grid_pairs,
                "Output Dir": args.pipeline_output_dir,
                "Report Location": str(
                    pipeline_output_dir
                    / "17_integration_output"
                    / "integration_results"
                    / "meta_analysis"
                ),
            },
        )
        return 0

    except KeyboardInterrupt:
        logger.print_status("Scaling analysis interrupted by user.", "warning")
        _write_run_manifest(
            pipeline_output_dir,
            _build_run_manifest(
                status="interrupted",
                plan=plan,
                out_dir=out_dir,
                pipeline_output_dir=pipeline_output_dir,
                config=run_config,
            ),
        )
        return 130
    except Exception as e:
        logger.print_status(f"Unexpected error: {e}", "error")
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""Empirical complexity benchmark harness: measured execution vs static bounds.

Runs the fixed six-exemplar corpus through the existing Step 11 render step
(``gnn.render.process_render``) and the Step 12 executor
(``gnn.execute.processor.process_execute``), collects per-script execution
details + structured results, and emits two receipt types:

- ``gnn.complexity_benchmark/v1`` — measurement rows per (model, framework)
  with the K-repeat timing trio, envelope RSS keys (nulls where unmeasured),
  availability, and error/return-code evidence.
- ``gnn.complexity_calibration/v1`` — the same rows joined with the static
  per-backend bounds from :mod:`gnn.analysis.complexity.estimator` on
  ``(model.source_sha256, framework)``, one ``calibration_note`` per row.

Honesty rules: estimates stay ESTIMATE-labeled in the static receipt; measured
rows always carry the environment block; nothing is fabricated — nulls where
unmeasured, ``available: false`` recorded explicitly, and failures carried
with their error/return-code instead of being skipped silently.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gnn.analysis.complexity import estimate_model_complexity, to_json_text
from gnn.utils.runtime_safety.framework_availability import is_framework_available

HARNESS_VERSION = "1"
RECEIPT_TYPE_BENCHMARK = "gnn.complexity_benchmark/v1"
RECEIPT_TYPE_CALIBRATION = "gnn.complexity_calibration/v1"
BENCHMARK_RECEIPT_FILENAME = "complexity_benchmark.json"
CALIBRATION_RECEIPT_FILENAME = "complexity_calibration.json"

#: Pinned calibration row key set (``gnn.complexity_calibration/v1``).
CALIBRATION_ROW_KEYS: frozenset[str] = frozenset(
    {
        "model_name",
        "source_sha256",
        "framework",
        "applicable",
        "asymptotic",
        "complexity_class",
        "wall_median_seconds",
        "peak_rss_mb",
        "repeats",
        "calibration_note",
    }
)

#: Pinned receipt-level environment block key set (both receipt types).
ENVIRONMENT_KEYS: frozenset[str] = frozenset(
    {
        "python_version",
        "platform",
        "accelerator_type",
        "backend_versions",
        "repeats",
        "sandbox_mode",
        "generated_at",
    }
)

RXINFER_CALIBRATION_NOTE = (
    "Julia first-run includes startup; steady-state vs cold-start"
)
SAMPLING_CALIBRATION_NOTE = "sampling bound per-sample; samples knob dominates"

#: Fixed corpus: six committed exemplars relative to the input corpus dir.
#: Coverage rationale: discrete (canonical POMDP gridworld), discrete
#: (epistemic-value active inference), continuous x multi_agent (LGSSM
#: composition), hybrid (discrete + continuous composed), nonstationary
#: (regime switching), and scaling (state-space growth at fixed horizon).
CORPUS_MODELS: tuple[str, ...] = (
    "pomdp_gridworld/pomdp_gridworld_3x3.md",
    "discrete/tmaze_epistemic.md",
    "continuous/multi_agent_lgssm.md",
    "continuous/hybrid_discrete_continuous.md",
    "discrete/regime_switched_dynamics.md",
    "pymdp_scaling_study/pymdp_scaling_N8_T100.md",
)


def estimate_path_complexity(path: Path) -> list[dict[str, Any]]:
    """Estimate static complexity for a spec file or every spec under a dir.

    A file yields ``[estimate_model_complexity(file)]``; a directory yields
    one receipt per ``*.md`` under it (recursive, sorted order), reusing the
    estimator unchanged.
    """
    path = Path(path)
    if path.is_file():
        return [estimate_model_complexity(path)]
    if path.is_dir():
        return [
            estimate_model_complexity(spec)
            for spec in sorted(path.rglob("*.md"))
            if spec.is_file()
        ]
    raise FileNotFoundError(f"no such file or directory: {path}")


def run_complexity_benchmark(
    target_dir: Path,
    output_dir: Path,
    *,
    frameworks: str = "all",
    repeats: int = 3,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the fixed corpus through render + execute and publish both receipts.

    Args:
        target_dir: Corpus directory containing the ``CORPUS_MODELS`` files.
        output_dir: Benchmark artifact root; the two receipts land at its
            top level as ``complexity_benchmark.json`` and
            ``complexity_calibration.json`` (stable key order; repeated runs
            overwrite them), with staged/render/execute trees underneath.
        frameworks: ``"all"``, ``"lite"``, or a comma-separated backend list.
        repeats: K benchmark repeats per rendered script (>= 1).
        verbose: Enable verbose step logging.

    Returns:
        ``{"status", "benchmark_receipt", "calibration_receipt", "rows",
        "models"}`` where the receipt values are file-path strings.
    """
    target_dir = Path(target_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    repeat_k = max(1, int(repeats))
    requested = _requested_frameworks(frameworks)
    _configure_logging(verbose)

    stage_dir = output_dir / "benchmark_corpus"
    render_dir = output_dir / "benchmark" / "11_render_output"
    execute_dir = output_dir / "benchmark" / "12_execute_output"

    staged = _stage_corpus(target_dir, stage_dir)
    if not staged:
        missing = [rel for rel in CORPUS_MODELS if not (target_dir / rel).is_file()]
        raise ValueError(
            f"no corpus models staged from {target_dir}; missing: {missing}"
        )

    run_id = uuid.uuid4().hex
    _render_corpus(stage_dir, render_dir, frameworks, verbose, run_id)
    _execute_corpus(
        stage_dir, render_dir, execute_dir, frameworks, repeat_k, verbose, run_id
    )

    details = _load_execution_details(execute_dir)
    structured_results = _load_structured_results(execute_dir)
    receipts = estimate_path_complexity(stage_dir)
    sha_by_stem = {
        Path(receipt["model"]["path"]).stem: receipt["model"]["source_sha256"]
        for receipt in receipts
    }
    _inject_source_sha256(execute_dir, sha_by_stem)

    rows = _build_measurement_rows(receipts, requested, details)
    calibration_rows = _build_calibration_rows(receipts, rows, repeat_k)
    environment = _environment_block(structured_results, requested, repeat_k)

    benchmark_receipt = {
        "receipt_type": RECEIPT_TYPE_BENCHMARK,
        "harness_version": HARNESS_VERSION,
        "environment": environment,
        "rows": rows,
    }
    calibration_receipt = {
        "receipt_type": RECEIPT_TYPE_CALIBRATION,
        "harness_version": HARNESS_VERSION,
        "environment": environment,
        "rows": calibration_rows,
    }
    benchmark_path = output_dir / BENCHMARK_RECEIPT_FILENAME
    calibration_path = output_dir / CALIBRATION_RECEIPT_FILENAME
    benchmark_path.write_text(to_json_text(benchmark_receipt), encoding="utf-8")
    calibration_path.write_text(to_json_text(calibration_receipt), encoding="utf-8")

    return {
        "status": _status(rows),
        "benchmark_receipt": str(benchmark_path),
        "calibration_receipt": str(calibration_path),
        "rows": rows,
        "models": sorted({row["model_name"] for row in rows}),
    }


def _configure_logging(verbose: bool) -> None:
    """Attach a root handler once when verbose; never double-configure."""
    if not verbose:
        return
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
        )


def _requested_frameworks(frameworks: str) -> list[str]:
    """Resolve the frameworks argument into the fixed backend row order."""
    from gnn.analysis.complexity.bounds import BACKEND_ORDER
    from gnn.frameworks import ALL_FRAMEWORKS, LITE_FRAMEWORKS

    text = (frameworks or "all").strip()
    lowered = text.lower()
    if lowered == "all":
        return list(BACKEND_ORDER)
    if lowered == "lite":
        lite = set(LITE_FRAMEWORKS)
        return [name for name in BACKEND_ORDER if name in lite]
    names = [part.strip() for part in text.split(",") if part.strip()]
    unknown = sorted(set(names) - set(ALL_FRAMEWORKS))
    if unknown:
        raise ValueError(
            f"unknown framework(s): {unknown}; valid: {list(ALL_FRAMEWORKS)}"
        )
    return names


def _stage_corpus(corpus_dir: Path, stage_dir: Path) -> list[Path]:
    """Copy the corpus exemplars into a fresh staging tree (byte-identical)."""
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    for relative in CORPUS_MODELS:
        source = corpus_dir / relative
        if not source.is_file():
            logging.getLogger(__name__).warning(
                "corpus model missing, skipped: %s", source
            )
            continue
        destination = stage_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        staged.append(destination)
    return staged


def _render_corpus(
    stage_dir: Path,
    render_dir: Path,
    frameworks: str,
    verbose: bool,
    run_id: str,
) -> None:
    """Render the staged corpus via the Step 11 processor into render_dir."""
    from gnn.render import process_render
    from gnn.render.framework_registry import get_lite_frameworks

    text = (frameworks or "all").strip()
    if text.lower() == "all":
        frameworks_arg: list[str] | None = None
    elif text.lower() == "lite":
        frameworks_arg = list(get_lite_frameworks())
    else:
        frameworks_arg = [part.strip() for part in text.split(",") if part.strip()]
    if render_dir.exists():
        shutil.rmtree(render_dir)
    render_dir.mkdir(parents=True, exist_ok=True)
    process_render(
        stage_dir,
        render_dir,
        verbose=verbose,
        frameworks=frameworks_arg,
        run_id=run_id,
    )


def _execute_corpus(
    stage_dir: Path,
    render_dir: Path,
    execute_dir: Path,
    frameworks: str,
    repeats: int,
    verbose: bool,
    run_id: str,
) -> Any:
    """Execute the rendered scripts via the Step 12 processor."""
    from gnn.execute.processor import process_execute

    if execute_dir.exists():
        shutil.rmtree(execute_dir)
    execute_dir.mkdir(parents=True, exist_ok=True)
    # ``target_dir`` is a stable sentinel (the staged corpus — nothing writes
    # into it during execution) so the Step 12 post-run input-identity guard
    # never fires; the rendered scripts live under ``render_output_dir``, whose
    # cwd-relative outputs legitimately mutate while scripts run.
    return process_execute(
        stage_dir,
        execute_dir,
        verbose=verbose,
        frameworks=frameworks,
        execution_benchmark_repeats=repeats,
        execution_summary_detail=True,
        require_render_summary=True,
        render_output_dir=render_dir,
        run_id=run_id,
    )


def _load_execution_details(execute_dir: Path) -> list[dict[str, Any]]:
    """Collect per-script execution details from the Step 12 summary files."""
    candidates = (
        execute_dir / "summaries" / "execution_summary_detail.json",
        execute_dir / "summaries" / "execution_summary.json",
    )
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        details = (
            payload.get("execution_details") if isinstance(payload, dict) else None
        )
        if isinstance(details, list):
            return [detail for detail in details if isinstance(detail, dict)]
    return []


def _load_structured_results(execute_dir: Path) -> list[dict[str, Any]]:
    """Collect per-script structured result JSONs (executor environment keys)."""
    results: list[dict[str, Any]] = []
    if not execute_dir.is_dir():
        return results
    for path in sorted(execute_dir.glob("*/*/execution_logs/*_results.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            results.append(payload)
    return results


def _stem_from_script_path(script_path: str) -> str:
    """Derive the model-directory stem from a rendered script path."""
    parts = Path(script_path).parts
    return parts[-3] if len(parts) >= 3 else ""


def _inject_source_sha256(execute_dir: Path, sha_by_stem: dict[str, str]) -> None:
    """Enrich per-script execution details with their source ``sha256``.

    Best-effort: only adds the key (never removes or reorders), and rewrites
    the summary files with stable key order so later slimming keeps the join
    key (see ``execute.metadata._slim_execution_detail`` keys_keep).
    """
    for name in ("execution_summary.json", "execution_summary_detail.json"):
        path = execute_dir / "summaries" / name
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        details = (
            payload.get("execution_details") if isinstance(payload, dict) else None
        )
        if not isinstance(details, list):
            continue
        changed = False
        for detail in details:
            if not isinstance(detail, dict) or "source_sha256" in detail:
                continue
            stem = str(
                detail.get("model_name")
                or _stem_from_script_path(str(detail.get("script_path") or ""))
            )
            sha256 = sha_by_stem.get(stem)
            if sha256:
                detail["source_sha256"] = sha256
                changed = True
        if changed:
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8"
            )
            os.replace(temporary, path)


def _detail_by_key(
    details: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Index execution details by (model stem, framework), preferring success."""
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for detail in details:
        framework = str(detail.get("framework") or "")
        stem = str(
            detail.get("model_name")
            or _stem_from_script_path(str(detail.get("script_path") or ""))
        )
        if not framework or not stem:
            continue
        current = indexed.get((stem, framework))
        if current is None or (
            bool(detail.get("success")) and not bool(current.get("success"))
        ):
            indexed[(stem, framework)] = detail
    return indexed


def _build_measurement_rows(
    receipts: list[dict[str, Any]],
    requested: list[str],
    details: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One measurement row per (model, framework); nulls where unmeasured."""
    indexed = _detail_by_key(details)
    rows: list[dict[str, Any]] = []
    for receipt in receipts:
        model = receipt["model"]
        stem = Path(model["path"]).stem
        static_rows = {
            str(backend.get("framework")): backend
            for backend in receipt.get("per_backend", [])
            if isinstance(backend, dict)
        }
        for framework in requested:
            detail = indexed.get((stem, framework))
            skipped = bool(detail.get("skipped")) if detail else False
            available = is_framework_available(framework) and not skipped
            if detail is None:
                success: bool | None = None
            elif skipped:
                success = None
            else:
                success = bool(detail.get("success"))
            samples = detail.get("execution_time_samples") if detail else None
            rows.append(
                {
                    "model_name": model["name"],
                    "source_sha256": model["source_sha256"],
                    "framework": framework,
                    "available": available,
                    "success": success,
                    "applicable": bool(static_rows[framework]["applicable"])
                    if framework in static_rows
                    else None,
                    "execution_time": _measured(detail, "execution_time", float),
                    "execution_time_mean": _measured(
                        detail, "execution_time_mean", float
                    ),
                    "execution_time_std": _measured(
                        detail, "execution_time_std", float
                    ),
                    "execution_time_samples": [float(s) for s in samples]
                    if isinstance(samples, list)
                    else None,
                    "child_peak_rss_mb": _measured(detail, "child_peak_rss_mb", float),
                    "rss_sample_interval_seconds": _measured(
                        detail, "rss_sample_interval_seconds", float
                    ),
                    "rss_samples_count": _measured(detail, "rss_samples_count", int),
                    "cancelled": bool(detail.get("cancelled", False))
                    if detail
                    else False,
                    "return_code": detail.get("return_code") if detail else None,
                    "error": detail.get("error") if detail else None,
                }
            )
    return rows


def _measured(detail: dict[str, Any] | None, key: str, caster: type) -> Any:
    """Carry a measured key from the detail, or None when unmeasured."""
    if not detail:
        return None
    value = detail.get(key)
    if value is None:
        return None
    try:
        return caster(value)
    except (TypeError, ValueError):
        return None


def _build_calibration_rows(
    receipts: list[dict[str, Any]],
    measurement_rows: list[dict[str, Any]],
    repeats: int,
) -> list[dict[str, Any]]:
    """Join measurement rows with the static bounds; one note per row."""
    measured = {
        (row["source_sha256"], row["framework"]): row for row in measurement_rows
    }
    calibration: list[dict[str, Any]] = []
    for receipt in receipts:
        static_rows = {
            str(backend.get("framework")): backend
            for backend in receipt.get("per_backend", [])
            if isinstance(backend, dict)
        }
        for framework in static_rows:
            row = measured.get((receipt["model"]["source_sha256"], framework))
            static = static_rows[framework]
            calibration.append(
                {
                    "model_name": receipt["model"]["name"],
                    "source_sha256": receipt["model"]["source_sha256"],
                    "framework": framework,
                    "applicable": bool(static.get("applicable")),
                    "asymptotic": static.get("asymptotic"),
                    "complexity_class": static.get("complexity_class"),
                    "wall_median_seconds": (row or {}).get("execution_time"),
                    "peak_rss_mb": (row or {}).get("child_peak_rss_mb"),
                    "repeats": repeats,
                    "calibration_note": _calibration_note(row, static),
                }
            )

    for row in calibration:
        if row.keys() != CALIBRATION_ROW_KEYS:
            raise ValueError(f"calibration row key drift: {sorted(row)}")
    return calibration


def _calibration_note(row: dict[str, Any] | None, static: dict[str, Any]) -> str:
    """Non-empty note per row: framework/model-specific, else row-state."""
    framework = str(static.get("framework"))
    if framework == "rxinfer":
        return RXINFER_CALIBRATION_NOTE
    if str(static.get("complexity_class")) == "sampling":
        return SAMPLING_CALIBRATION_NOTE
    if row is None or not row.get("available"):
        return "backend unavailable; no measurement"
    if row.get("success") is True:
        samples = row.get("execution_time_samples") or []
        return (
            f"measured K={len(samples)} repeats; wall median of execution_time samples"
        )
    if row.get("success") is False:
        return "execution failed; no measurement recorded"
    return "no measurement recorded for this backend in this run"


def _environment_block(
    structured_results: list[dict[str, Any]],
    requested: list[str],
    repeats: int,
) -> dict[str, Any]:
    """Receipt-level environment: measured values, nulls where unobserved."""
    backend_versions: dict[str, Any] = dict.fromkeys(requested, None)
    accelerator_type: str | None = None
    python_version = platform.python_version()
    sandbox_mode = os.environ.get("GNN_SANDBOX", "off")
    for result in structured_results:
        metadata = result.get("execution_metadata")
        if not isinstance(metadata, dict):
            continue
        framework = str(result.get("framework") or "")
        if (
            framework in backend_versions
            and backend_versions[framework] is None
            and metadata.get("backend_version")
        ):
            backend_versions[framework] = str(metadata["backend_version"])
        if accelerator_type is None and metadata.get("accelerator_type"):
            accelerator_type = str(metadata["accelerator_type"])
        if metadata.get("python_version"):
            python_version = str(metadata["python_version"])
        if metadata.get("sandbox_mode"):
            sandbox_mode = str(metadata["sandbox_mode"])
    if accelerator_type is None:
        try:
            from gnn.execute.detection import _detect_accelerator_type

            accelerator_type = _detect_accelerator_type()
        except Exception:  # noqa: BLE001 — environment probe is best-effort
            accelerator_type = None
    block = {
        "python_version": python_version,
        "platform": platform.platform(),
        "accelerator_type": accelerator_type,
        "backend_versions": backend_versions,
        "repeats": repeats,
        "sandbox_mode": sandbox_mode,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    if block.keys() != ENVIRONMENT_KEYS:
        raise ValueError(f"environment block key drift: {sorted(block)}")
    return block


def _status(rows: list[dict[str, Any]]) -> str:
    """Exit-code-style status over the measurement rows."""
    if not rows:
        return "failed"
    measured = sum(1 for row in rows if row.get("success") is True)
    failed = sum(1 for row in rows if row.get("success") is False)
    unmeasured = sum(
        1 for row in rows if row.get("success") is None and row.get("available")
    )
    unavailable = sum(1 for row in rows if not row.get("available"))
    if measured == 0:
        return "failed" if (failed or unmeasured) else "skipped"
    if failed:
        return "success_with_failures"
    if unmeasured or unavailable:
        return "success_with_skips"
    return "success"


__all__ = [
    "BENCHMARK_RECEIPT_FILENAME",
    "CALIBRATION_RECEIPT_FILENAME",
    "CALIBRATION_ROW_KEYS",
    "CORPUS_MODELS",
    "HARNESS_VERSION",
    "ENVIRONMENT_KEYS",
    "RECEIPT_TYPE_BENCHMARK",
    "RECEIPT_TYPE_CALIBRATION",
    "estimate_path_complexity",
    "run_complexity_benchmark",
]

"""Cross-framework reliability gates for maintained GNN model families.

Contract: the ledger compares *structural metadata* between backends that a
model-family profile declares compatible — recorded ``random_seed`` presence,
``num_timesteps``, matrix shapes and trace lengths (see
:func:`compare_framework_metrics`). It deliberately does not compare trajectory
values across backends: the maintained frameworks reach the same model through
different inference numerics (message passing, sampling, JAX/likelihood
parameterisation, string diagrams), so cross-backend value equality is not a
defined contract. Trajectory assertions belong to the per-framework contract
tests under ``tests/execute``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, cast

from gnn.pipeline.model_family_acceptance import (
    ModelFamily,
    load_model_family_manifest,
    run_model_family_acceptance,
)
from gnn.report.cross_framework_reliability import (
    render_cross_framework_reliability_markdown,
)

MAINTAINED_FRAMEWORKS = (
    "pymdp",
    "rxinfer",
    "jax",
    "numpyro",
    "pytorch",
    "activeinference_jl",
    "discopy",
)
FRAMEWORK_RESULT_KEYS = {
    "pymdp": "pymdp_executions",
    "rxinfer": "rxinfer_executions",
    "jax": "jax_executions",
    "numpyro": "numpyro_executions",
    "pytorch": "pytorch_executions",
    "activeinference_jl": "activeinference_executions",
    "discopy": "discopy_executions",
}
SCHEMA_VERSION = "gnn_cross_framework_reliability_ledger_v1"
AcceptanceRunner = Callable[[ModelFamily, Path], dict[str, Any]]


@dataclass(frozen=True)
class FrameworkComparisonIssue:
    """One cross-framework comparison mismatch."""

    field: str
    message: str

    def to_dict(self) -> dict[str, str]:
        """Return a serializable issue record."""
        return {"field": self.field, "message": self.message}


def run_cross_framework_reliability(
    manifest_path: Path,
    output_dir: Path,
    *,
    family_names: Iterable[str] | None = None,
    frameworks: Sequence[str] = MAINTAINED_FRAMEWORKS,
    strict: bool = False,
    acceptance_runner: AcceptanceRunner | None = None,
) -> dict[str, Any]:
    """Run profiled cross-framework reliability checks."""
    _validate_frameworks(frameworks)
    output_dir.mkdir(parents=True, exist_ok=True)
    families = _select_families(manifest_path, family_names)
    ledger: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(),
        "manifest": str(manifest_path),
        "strict": strict,
        "frameworks": list(frameworks),
        "family_count": len(families),
        "families": [],
    }

    failures: list[str] = []
    runner = acceptance_runner or _default_acceptance_runner(manifest_path)
    for family in families:
        family_dir = output_dir / family.name
        family_dir.mkdir(parents=True, exist_ok=True)
        acceptance_ledger = runner(family, family_dir / "acceptance")
        family_result = _first_family_result(acceptance_ledger, family.name)
        reliability = _build_family_reliability(
            family,
            family_result,
            frameworks,
        )
        ledger["families"].append(reliability)
        if reliability["status"] == "failed":
            failures.append(family.name)

    ledger["status"] = "failed" if failures else "passed"
    ledger["failed_families"] = failures
    _write_ledger(ledger, output_dir)
    if strict and failures:
        raise RuntimeError(f"Cross-framework reliability failed: {', '.join(failures)}")
    return ledger


def compare_framework_metrics(
    framework_metrics: dict[str, dict[str, Any]],
) -> tuple[list[FrameworkComparisonIssue], list[dict[str, str]]]:
    """Compare normalized metrics from two or more compatible frameworks.

    Returns ``(issues, skipped)``. ``issues`` are hard mismatches that fail the
    gate; ``skipped`` records metrics that could not be compared because they
    are not reported by all compared frameworks.
    """
    available = {
        framework: metrics
        for framework, metrics in sorted(framework_metrics.items())
        if metrics.get("available")
    }
    if len(available) < 2:
        return [], []
    issues: list[FrameworkComparisonIssue] = []
    skipped: list[dict[str, str]] = []
    reference_name, reference_metrics = next(iter(available.items()))

    seed_missing = sorted(
        framework
        for framework, metrics in available.items()
        if metrics.get("random_seed") is None
    )
    if seed_missing:
        skipped.append(
            {
                "field": "random_seed",
                "reason": "not reported by all compared frameworks",
                "detail": "missing from: " + ", ".join(seed_missing),
            }
        )

    shared_keys: dict[str, set[str]] = {}
    for field in ("matrix_shapes", "trace_lengths"):
        tables = {
            framework: metrics.get(field) or {}
            for framework, metrics in available.items()
        }
        key_sets = [set(table) for table in tables.values()]
        shared = set.intersection(*key_sets)
        shared_keys[field] = shared
        missing_by_key = {
            key: sorted(name for name, table in tables.items() if key not in table)
            for key in sorted(set.union(*key_sets) - shared)
        }
        if missing_by_key:
            detail = "; ".join(
                f"{key} missing from {', '.join(frameworks_list)}"
                for key, frameworks_list in missing_by_key.items()
            )
            skipped.append(
                {
                    "field": field,
                    "reason": "key not reported by all compared frameworks",
                    "detail": detail,
                }
            )

    for framework, metrics in list(available.items())[1:]:
        if metrics.get("num_timesteps") != reference_metrics.get("num_timesteps"):
            issues.append(
                FrameworkComparisonIssue(
                    field="num_timesteps",
                    message=(
                        f"{framework} differs from {reference_name}: "
                        f"{metrics.get('num_timesteps')!r} != "
                        f"{reference_metrics.get('num_timesteps')!r}"
                    ),
                )
            )
        if not seed_missing and metrics.get("random_seed") != reference_metrics.get(
            "random_seed"
        ):
            issues.append(
                FrameworkComparisonIssue(
                    field="random_seed",
                    message=(
                        f"{framework} differs from {reference_name}: "
                        f"{metrics.get('random_seed')!r} != "
                        f"{reference_metrics.get('random_seed')!r}"
                    ),
                )
            )
        for field in ("matrix_shapes", "trace_lengths"):
            table = metrics.get(field) or {}
            reference_table = reference_metrics.get(field) or {}
            for key in sorted(shared_keys[field]):
                if table.get(key) != reference_table.get(key):
                    issues.append(
                        FrameworkComparisonIssue(
                            field=field,
                            message=(
                                f"{framework} differs from {reference_name} for "
                                f"{field} key {key!r}: {table.get(key)!r} != "
                                f"{reference_table.get(key)!r}"
                            ),
                        )
                    )
    return issues, skipped


def collect_framework_metrics(pipeline_output: Path, framework: str) -> dict[str, Any]:
    """Collect normalized comparable metrics for one framework output."""
    execution_details = _framework_execution_details(pipeline_output, framework)
    if not execution_details:
        return {
            "available": False,
            "framework": framework,
            "reason": "execution_detail_missing_or_invalid",
        }
    payload_path = _find_framework_simulation_payload(pipeline_output, framework)
    if payload_path is None:
        return {
            "available": False,
            "framework": framework,
            "reason": "simulation_payload_missing",
        }
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "framework": framework,
            "reason": f"simulation_payload_unreadable: {exc}",
            "source": str(payload_path),
        }
    model_parameters = payload.get("model_parameters", {}) or {}
    runtime_metadata = payload.get("runtime_metadata", {}) or {}
    return {
        "available": True,
        "framework": framework,
        "source": str(payload_path),
        "execution_details": [
            {
                "script_name": detail.get("script_name"),
                "model_name": detail.get("model_name"),
                "return_code": detail.get("return_code"),
                "structured_result_file": detail.get("structured_result_file"),
                "implementation_directory": detail.get("implementation_directory"),
            }
            for detail in execution_details
        ],
        "schema_version": payload.get("schema_version"),
        "success": payload.get("success"),
        "random_seed": _first_present(
            payload,
            model_parameters,
            runtime_metadata,
            ("random_seed", "seed"),
        ),
        "num_timesteps": _first_present(
            payload,
            model_parameters,
            runtime_metadata,
            ("num_timesteps", "timesteps", "T"),
        ),
        "matrix_shapes": _matrix_shapes(payload, model_parameters),
        "trace_lengths": _trace_lengths(payload),
    }


def _default_acceptance_runner(manifest_path: Path) -> AcceptanceRunner:
    def _run(family: ModelFamily, output_dir: Path) -> dict[str, Any]:
        return run_model_family_acceptance(
            manifest_path,
            output_dir,
            family_names=[family.name],
            frameworks=family.frameworks,
            strict=True,
        )

    return _run


def _select_families(
    manifest_path: Path, family_names: Iterable[str] | None
) -> list[ModelFamily]:
    requested = {name.strip() for name in family_names or [] if name.strip()}
    families = load_model_family_manifest(manifest_path)
    selected = [
        family for family in families if not requested or family.name in requested
    ]
    missing = sorted(requested - {family.name for family in families})
    if missing:
        raise KeyError(f"Unknown model families: {', '.join(missing)}")
    return selected


def _build_family_reliability(
    family: ModelFamily,
    family_result: dict[str, Any],
    frameworks: Sequence[str],
) -> dict[str, Any]:
    profile = _compatibility_profile(family, frameworks)
    framework_results: dict[str, dict[str, Any]] = {}
    pipeline_output = _pipeline_output_from_family_result(family_result)
    for framework in frameworks:
        framework_results[framework] = _framework_reliability_result(
            framework,
            profile[framework],
            family_result,
            pipeline_output,
        )

    comparable_metrics = {
        framework: result.get("metrics", {})
        for framework, result in framework_results.items()
        if result["status"] == "passed" and result.get("metrics", {}).get("available")
    }
    comparison_issues, comparison_skipped = compare_framework_metrics(
        comparable_metrics
    )
    comparison_status = (
        "failed"
        if comparison_issues
        else "passed"
        if len(comparable_metrics) >= 2
        else "skipped"
    )
    required_failures = [
        framework
        for framework, result in framework_results.items()
        if result["profile"] == "required" and result["status"] != "passed"
    ]
    status = "failed" if required_failures or comparison_issues else "passed"
    return {
        "name": family.name,
        "description": family.description,
        "status": status,
        "acceptance_status": family_result.get("status"),
        "frameworks": framework_results,
        "comparison": {
            "status": comparison_status,
            "reason": (
                "fewer_than_two_compatible_outputs"
                if len(comparable_metrics) < 2 and not comparison_issues
                else None
            ),
            "compared_frameworks": sorted(comparable_metrics),
            "issues": [issue.to_dict() for issue in comparison_issues],
            "skipped": comparison_skipped,
        },
        "required_framework_failures": required_failures,
        "artifact_links": family_result.get("artifact_links", []),
    }


def _compatibility_profile(
    family: ModelFamily, frameworks: Sequence[str]
) -> dict[str, dict[str, str]]:
    required = set(_parse_framework_list(family.frameworks))
    profile: dict[str, dict[str, str]] = {}
    family_unsupported_steps = set(
        (family.acceptance_profile or {}).get("allow_unsupported_steps", [])
    )
    render_execute_unsupported = {11, 12}.issubset(family_unsupported_steps)
    for framework in frameworks:
        if framework in required and not render_execute_unsupported:
            profile[framework] = {
                "status": "required",
                "reason": "family manifest compatible backend",
            }
        else:
            reason = (
                "family profile declares Step 11/12 unsupported"
                if render_execute_unsupported and framework in required
                else "not declared compatible for this model family"
            )
            profile[framework] = {"status": "unsupported", "reason": reason}
    return profile


def _framework_reliability_result(
    framework: str,
    profile: dict[str, str],
    family_result: dict[str, Any],
    pipeline_output: Path | None,
) -> dict[str, Any]:
    if profile["status"] == "unsupported":
        return {
            "profile": "unsupported",
            "status": "unsupported",
            "reason": profile["reason"],
            "metrics": {"available": False, "reason": profile["reason"]},
        }

    step_evidence = family_result.get("step_evidence", {})
    step_11 = step_evidence.get("11", {})
    step_12 = step_evidence.get("12", {})
    if family_result.get("status") != "passed":
        return {
            "profile": "required",
            "status": "failed",
            "reason": "family_acceptance_failed",
            "metrics": {"available": False},
        }
    for step, evidence in (("11", step_11), ("12", step_12)):
        if evidence.get("status") != "passed":
            return {
                "profile": "required",
                "status": "failed",
                "reason": f"step_{step}_evidence_not_passed",
                "metrics": {"available": False},
            }
    if pipeline_output is None:
        return {
            "profile": "required",
            "status": "failed",
            "reason": "pipeline_output_unknown",
            "metrics": {"available": False},
        }
    metrics = collect_framework_metrics(pipeline_output, framework)
    return {
        "profile": "required",
        "status": "passed" if metrics.get("available") else "failed",
        "reason": None if metrics.get("available") else metrics.get("reason"),
        "metrics": metrics,
    }


def _pipeline_output_from_family_result(family_result: dict[str, Any]) -> Path | None:
    command = family_result.get("command", [])
    if isinstance(command, list) and "--output-dir" in command:
        index = command.index("--output-dir") + 1
        if index < len(command):
            return Path(str(command[index]))
    staged = family_result.get("staged_target_dir")
    if staged:
        staged_path = Path(str(staged))
        candidate = staged_path.parent / "pipeline_output"
        if candidate.exists():
            return candidate
    return None


def _first_family_result(ledger: dict[str, Any], family_name: str) -> dict[str, Any]:
    for family in ledger.get("families", []):
        if family.get("name") == family_name:
            return cast("dict[str, Any]", family)
    raise ValueError(f"Acceptance ledger did not include {family_name}")


def _find_framework_simulation_payload(
    pipeline_output: Path, framework: str
) -> Path | None:
    execute_dir = pipeline_output / "12_execute_output"
    candidates = [
        path
        for path in execute_dir.rglob("*simulation_results.json")
        if framework in path.parts
    ]
    if not candidates:
        candidates = [
            path
            for path in execute_dir.rglob("simulation_results.json")
            if framework in path.parts
        ]
    return sorted(candidates)[0] if candidates else None


def _framework_execution_details(
    pipeline_output: Path, framework: str
) -> list[dict[str, Any]]:
    summary_path = (
        pipeline_output / "12_execute_output" / "summaries" / "execution_summary.json"
    )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    details = [
        detail
        for detail in summary.get("execution_details", [])
        if isinstance(detail, dict) and detail.get("framework") == framework
    ]
    if not details:
        return []
    valid_details: list[dict[str, Any]] = []
    for detail in details:
        structured_result = detail.get("structured_result_file")
        if (
            detail.get("success") is not True
            or detail.get("skipped") is True
            or detail.get("return_code") != 0
            or not structured_result
            or not Path(str(structured_result)).exists()
        ):
            return []
        valid_details.append(detail)
    return valid_details


def _matrix_shapes(
    payload: dict[str, Any], model_parameters: dict[str, Any]
) -> dict[str, Any]:
    shapes: dict[str, Any] = {}
    provenance = payload.get("matrix_provenance")
    if isinstance(provenance, dict):
        for matrix_name, matrix_info in provenance.items():
            if isinstance(matrix_info, dict) and "shape" in matrix_info:
                shapes[f"{matrix_name}_shape"] = matrix_info["shape"]
        shapes["matrix_provenance_keys"] = sorted(provenance)
        return dict(sorted(shapes.items()))
    for key, value in model_parameters.items():
        if key.endswith("_shape"):
            shapes[key] = value
    return dict(sorted(shapes.items()))


def _trace_lengths(payload: dict[str, Any]) -> dict[str, int]:
    keys = (
        "observations",
        "actions",
        "beliefs",
        "expected_free_energy",
        "free_energy",
        "policy_posterior",
    )
    lengths: dict[str, int] = {}
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            lengths[key] = len(value)
    return dict(sorted(lengths.items()))


def _first_present(
    payload: dict[str, Any],
    model_parameters: dict[str, Any],
    runtime_metadata: dict[str, Any],
    keys: tuple[str, ...],
) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
        if key in model_parameters:
            return model_parameters[key]
        if key in runtime_metadata:
            return runtime_metadata[key]
    return None


def _parse_framework_list(frameworks: str | None) -> list[str]:
    if not frameworks:
        return []
    return [item.strip() for item in frameworks.split(",") if item.strip()]


def _validate_frameworks(frameworks: Sequence[str]) -> None:
    unknown = sorted(set(frameworks) - set(MAINTAINED_FRAMEWORKS))
    if unknown:
        raise ValueError(f"Unprofiled frameworks: {', '.join(unknown)}")


def _write_ledger(ledger: dict[str, Any], output_dir: Path) -> None:
    (output_dir / "cross_framework_reliability_ledger.json").write_text(
        json.dumps(ledger, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "cross_framework_reliability_ledger.md").write_text(
        render_cross_framework_reliability_markdown(ledger),
        encoding="utf-8",
    )

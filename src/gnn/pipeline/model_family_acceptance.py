"""Manifest-driven model-family acceptance harness for v1.9 evidence."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

from gnn.analysis.interpretability import (
    build_family_interpretability_summary,
    render_family_interpretability_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVIDENCE_STEPS = "3,5,6,11,12,15,16,23"
PIPELINE_STEPS = tuple(range(25))
PASSING_STEP_STATUSES = {"SUCCESS", "PASSED", "PASS", "OK"}
SKIPPED_STEP_STATUSES = {"SKIPPED", "SKIP", "NOT_RUN", "NOT RUN"}
ACCEPTABLE_SUMMARY_STATUSES = {
    "SUCCESS",
    "SUCCESS_WITH_WARNINGS",
    "PARTIAL_SUCCESS",
    "COMPLETED_WITH_WARNINGS",
    "WARNING",
    "WARNINGS",
}
STEP_ARTIFACT_REQUIREMENTS = {
    "3": (
        "3_gnn_output/gnn_processing_summary.json",
        "3_gnn_output/gnn_processing_results.json",
    ),
    "5": ("5_type_checker_output/type_check_results.json",),
    "6": (
        "6_validation_output/validation_summary.json",
        "6_validation_output/validation_results.json",
    ),
    "11": ("11_render_output/render_processing_summary.json",),
    "12": ("12_execute_output/summaries/execution_summary.json",),
    "15": ("15_audio_output/audio_results.json",),
    "16": ("16_analysis_output/analysis_results.json",),
    "23": ("23_report_output/report_processing_summary.json",),
}
DEFAULT_ACCEPTANCE_PROFILE = {
    "required_steps": [3, 5, 6, 15, 16, 23],
    "evidence_steps": [11, 12],
    "allow_unsupported_steps": [],
    "unsupported_step_reasons": {},
}
# A full model-family acceptance renders and executes each maintained framework
# (PyMDP, RxInfer.jl, ActiveInference.jl — the latter two under Julia), then runs
# Step 16 analysis and Step 23 reporting. A too-tight default (historically 180s)
# spuriously fails the real cross-framework gate (run_cross_framework_reliability
# --strict) once the pipeline reaches its heavier execute/analyse stages, so the
# default is sized generously for a single-family run while remaining bounded.
# Override with GNN_MODEL_FAMILY_TIMEOUT_SECONDS for very heavy families.
DEFAULT_FAMILY_TIMEOUT_SECONDS = int(
    os.environ.get("GNN_MODEL_FAMILY_TIMEOUT_SECONDS", "600")
)


@dataclass(frozen=True)
class ModelFamily:
    """One family entry from the maintained model-family manifest."""

    name: str
    description: str
    target_dir: Path
    representative_files: tuple[str, ...]
    frameworks: str | None = None
    acceptance_profile: Dict[str, Any] | None = None


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def load_model_family_manifest(manifest_path: Path) -> List[ModelFamily]:
    """Load and validate the maintained model-family manifest."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "gnn_model_family_manifest_v1":
        raise ValueError("Unsupported model-family manifest schema")
    families = payload.get("families")
    if not isinstance(families, list):
        raise ValueError("Model-family manifest requires a families list")
    defaults = payload.get("acceptance_profile_defaults", DEFAULT_ACCEPTANCE_PROFILE)
    if not isinstance(defaults, dict):
        raise ValueError("Model-family manifest acceptance defaults must be an object")
    return [_parse_family(raw, defaults) for raw in families]


def select_model_families(
    families: Sequence[ModelFamily],
    family_names: Iterable[str] | None,
) -> List[ModelFamily]:
    """Filter ``families`` by explicitly requested names (single shared rule).

    Args:
        families: The full manifest-ordered family list.
        family_names: Requested family names; ``None``/empty selects all.
            Surrounding whitespace on each name is ignored.

    Returns:
        Families in manifest order, restricted to the requested subset.

    Raises:
        KeyError: If any requested family name is absent from the manifest.
    """
    requested = {name.strip() for name in family_names or [] if name.strip()}
    if not requested:
        return list(families)
    missing = sorted(requested - {family.name for family in families})
    if missing:
        raise KeyError(f"Unknown model families: {', '.join(missing)}")
    return [family for family in families if family.name in requested]


def run_model_family_acceptance(
    manifest_path: Path,
    output_dir: Path,
    *,
    family_names: Iterable[str] | None = None,
    only_steps: str | None = DEFAULT_EVIDENCE_STEPS,
    frameworks: str | None = None,
    strict: bool = False,
    runner: Runner | None = None,
) -> Dict[str, Any]:
    """Run representative model families and write acceptance ledgers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    families = load_model_family_manifest(manifest_path)
    selected = select_model_families(families, family_names)

    ledger: Dict[str, Any] = {
        "schema": "gnn_model_family_acceptance_ledger_v1",
        "created_at": datetime.now().isoformat(),
        "manifest": str(manifest_path),
        "strict": strict,
        "only_steps": only_steps,
        "frameworks": frameworks,
        "family_count": len(selected),
        "families": [],
    }
    failures: list[str] = []
    for family in selected:
        _validate_requested_steps_cover_profile(family, only_steps)
        family_result = _run_one_family(
            family,
            output_dir,
            only_steps=only_steps,
            frameworks=frameworks or family.frameworks,
            runner=runner or _subprocess_runner,
        )
        ledger["families"].append(family_result)
        if family_result["status"] == "failed":
            failures.append(family.name)
    ledger["status"] = "failed" if failures else "passed"
    ledger["failed_families"] = failures
    _write_ledger_artifacts(ledger, output_dir)
    if strict and failures:
        raise RuntimeError(f"Model-family acceptance failed: {', '.join(failures)}")
    return ledger


def _parse_family(raw: Any, acceptance_defaults: Dict[str, Any]) -> ModelFamily:
    if not isinstance(raw, dict):
        raise ValueError("Model-family entries must be objects")
    target_dir = REPO_ROOT / str(raw["target_dir"])
    representatives = tuple(str(item) for item in raw.get("representative_files", []))
    if not representatives:
        raise ValueError(f"Model family {raw.get('name')} has no representative files")
    raw_profile = raw.get("acceptance_profile", {})
    if raw_profile and not isinstance(raw_profile, dict):
        raise ValueError(
            f"Model family {raw.get('name')} acceptance_profile must be an object"
        )
    return ModelFamily(
        name=str(raw["name"]),
        description=str(raw["description"]),
        target_dir=target_dir,
        representative_files=representatives,
        frameworks=str(raw["frameworks"]) if raw.get("frameworks") else None,
        acceptance_profile=_normalize_acceptance_profile(
            acceptance_defaults, raw_profile if isinstance(raw_profile, dict) else {}
        ),
    )


def _normalize_acceptance_profile(
    defaults: Dict[str, Any], override: Dict[str, Any]
) -> Dict[str, Any]:
    profile = dict(DEFAULT_ACCEPTANCE_PROFILE)
    profile.update(defaults)
    profile.update(override)
    unsupported_step_reasons_raw: Any = profile.get("unsupported_step_reasons", {})
    if unsupported_step_reasons_raw is None:
        unsupported_step_reasons_raw = {}
    if not isinstance(unsupported_step_reasons_raw, dict):
        raise ValueError(
            "Acceptance profile unsupported_step_reasons must be an object"
        )
    unsupported_step_reasons: Dict[str, str] = {}
    for step_raw, reason_raw in unsupported_step_reasons_raw.items():
        reason = str(reason_raw).strip()
        if reason:
            unsupported_step_reasons[str(step_raw)] = reason
    return {
        "required_steps": _coerce_step_list(profile.get("required_steps")),
        "evidence_steps": _coerce_step_list(profile.get("evidence_steps")),
        "allow_unsupported_steps": _coerce_step_list(
            profile.get("allow_unsupported_steps")
        ),
        "unsupported_step_reasons": unsupported_step_reasons,
    }


def _coerce_step_list(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, str):
        return _parse_step_list(value)
    if not isinstance(value, list):
        raise ValueError("Acceptance profile step lists must be arrays or strings")
    return [int(item) for item in value]


def _run_one_family(
    family: ModelFamily,
    output_dir: Path,
    *,
    only_steps: str | None,
    frameworks: str | None,
    runner: Runner,
) -> Dict[str, Any]:
    family_dir = output_dir / family.name
    pipeline_output = family_dir / "pipeline_output"
    staged_input = family_dir / "input"
    _reset_family_dir(family_dir)
    family_dir.mkdir(parents=True, exist_ok=True)
    staged_input.mkdir(parents=True, exist_ok=True)
    copied_files = _stage_representative_files(family, staged_input)
    acceptance_profile = family.acceptance_profile or dict(DEFAULT_ACCEPTANCE_PROFILE)
    profiled_unsupported_steps = _profiled_unsupported_steps(
        only_steps, acceptance_profile
    )
    command = _pipeline_command(
        staged_input,
        pipeline_output,
        only_steps,
        frameworks,
        skip_steps=profiled_unsupported_steps,
    )
    completed = runner(command)
    return_code = int(completed.returncode)
    pipeline_summary = _load_pipeline_summary(pipeline_output)
    raw_step_status = _build_step_status(
        only_steps,
        return_code,
        pipeline_summary,
        profiled_unsupported_steps=profiled_unsupported_steps,
    )
    missing_summary_steps = _missing_summary_steps(
        only_steps, pipeline_summary, profiled_unsupported_steps
    )
    step_evidence = _build_step_evidence(
        raw_step_status,
        pipeline_output,
        missing_summary_steps,
        profiled_unsupported_steps=profiled_unsupported_steps,
        acceptance_profile=acceptance_profile,
    )
    step_status = _apply_acceptance_profile(
        raw_step_status, step_evidence, acceptance_profile
    )
    if pipeline_summary is not None:
        pipeline_passed = _selected_steps_passed(
            step_status, only_steps, acceptance_profile, step_evidence
        ) and _pipeline_run_outcome_acceptable(
            return_code, pipeline_summary, step_evidence
        )
    else:
        pipeline_passed = False
    interpretability_summary = build_family_interpretability_summary(
        family.name, staged_input, pipeline_output
    )
    (family_dir / "interpretability_summary.json").write_text(
        json.dumps(interpretability_summary, indent=2),
        encoding="utf-8",
    )
    (family_dir / "interpretability_summary.md").write_text(
        render_family_interpretability_markdown(interpretability_summary),
        encoding="utf-8",
    )
    return {
        "name": family.name,
        "description": family.description,
        "source_target_dir": str(family.target_dir),
        "staged_target_dir": str(staged_input),
        "representative_files": [str(path) for path in copied_files],
        "command": command,
        "return_code": return_code,
        "status": "passed" if pipeline_passed else "failed",
        "acceptance_profile": acceptance_profile,
        "profiled_unsupported_steps": [
            str(step) for step in sorted(profiled_unsupported_steps)
        ],
        "pipeline_summary": _summarize_pipeline(pipeline_summary),
        "stdout_tail": _tail_text(completed.stdout),
        "stderr_tail": _tail_text(completed.stderr),
        "steps": step_status,
        "raw_steps": raw_step_status,
        "step_evidence": step_evidence,
        "step_status_counts": _count_step_statuses(step_status),
        "artifact_links": _collect_family_artifacts(pipeline_output),
        "interpretability_summary": interpretability_summary,
    }


def _reset_family_dir(family_dir: Path) -> None:
    """Create a fresh per-family output boundary before each acceptance run."""
    if family_dir.is_symlink() or family_dir.is_file():
        family_dir.unlink()
        return
    if family_dir.exists():
        shutil.rmtree(family_dir)


def _stage_representative_files(family: ModelFamily, staged_input: Path) -> List[Path]:
    copied: list[Path] = []
    for relative_name in family.representative_files:
        source = family.target_dir / relative_name
        if not source.exists():
            raise FileNotFoundError(f"Representative fixture not found: {source}")
        destination = staged_input / source.name
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def _pipeline_command(
    target_dir: Path,
    output_dir: Path,
    only_steps: str | None,
    frameworks: str | None,
    *,
    skip_steps: set[int] | None = None,
) -> List[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "src" / "gnn" / "main.py"),
        "--target-dir",
        str(target_dir),
        "--output-dir",
        str(output_dir),
    ]
    if only_steps:
        command.extend(["--only-steps", only_steps])
    if skip_steps:
        command.extend(
            ["--skip-steps", ",".join(str(step) for step in sorted(skip_steps))]
        )
    if frameworks:
        command.extend(["--frameworks", frameworks])
    command.append("--skip-llm")
    return command


def _subprocess_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=DEFAULT_FAMILY_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=124,
            stdout=_coerce_timeout_output(exc.stdout),
            stderr=(
                _coerce_timeout_output(exc.stderr)
                + f"\nTimed out after {DEFAULT_FAMILY_TIMEOUT_SECONDS}s"
            ).strip(),
        )


def _build_step_status(
    only_steps: str | None,
    return_code: int,
    pipeline_summary: Dict[str, Any] | None = None,
    *,
    profiled_unsupported_steps: set[int] | None = None,
) -> Dict[str, str]:
    selected = set(PIPELINE_STEPS if not only_steps else _parse_step_list(only_steps))
    summary_statuses = _extract_summary_step_statuses(pipeline_summary)
    profiled_unsupported_steps = profiled_unsupported_steps or set()
    status: Dict[str, str] = {}
    for step in PIPELINE_STEPS:
        if step not in selected:
            status[str(step)] = "skipped"
        elif str(step) in summary_statuses:
            status[str(step)] = summary_statuses[str(step)]
        elif step in profiled_unsupported_steps:
            status[str(step)] = "skipped"
        elif pipeline_summary is not None:
            status[str(step)] = "failed"
        else:
            status[str(step)] = "passed" if return_code == 0 else "failed"
    return status


def _validate_requested_steps_cover_profile(
    family: ModelFamily, only_steps: str | None
) -> None:
    """Reject acceptance runs that omit profile-required evidence steps."""
    if only_steps is None:
        return
    selected = set(_parse_step_list(only_steps))
    profile = family.acceptance_profile or dict(DEFAULT_ACCEPTANCE_PROFILE)
    required = _profile_required_steps(profile)
    missing = sorted(required - selected)
    if missing:
        raise ValueError(
            f"Model family {family.name} acceptance profile requires steps "
            f"{','.join(str(step) for step in sorted(required))}; "
            f"--only-steps omitted {','.join(str(step) for step in missing)}"
        )


def _profile_required_steps(acceptance_profile: Dict[str, Any]) -> set[int]:
    """Return all steps that must be selected to produce release evidence."""
    required = set(_coerce_step_list(acceptance_profile["required_steps"]))
    required.update(_coerce_step_list(acceptance_profile["evidence_steps"]))
    required.update(_coerce_step_list(acceptance_profile["allow_unsupported_steps"]))
    return required


def _missing_summary_steps(
    only_steps: str | None,
    pipeline_summary: Dict[str, Any] | None,
    profiled_unsupported_steps: set[int] | None = None,
) -> set[str]:
    """Return selected step ids absent from an available pipeline summary."""
    if pipeline_summary is None:
        return set()
    selected = set(PIPELINE_STEPS if not only_steps else _parse_step_list(only_steps))
    selected -= profiled_unsupported_steps or set()
    summary_statuses = _extract_summary_step_statuses(pipeline_summary)
    return {str(step) for step in selected if str(step) not in summary_statuses}


def _load_pipeline_summary(pipeline_output: Path) -> Dict[str, Any] | None:
    summary_path = (
        pipeline_output / "00_pipeline_summary" / "pipeline_execution_summary.json"
    )
    if not summary_path.exists():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _summarize_pipeline(pipeline_summary: Dict[str, Any] | None) -> Dict[str, Any]:
    if not pipeline_summary:
        return {"available": False}
    return {
        "available": True,
        "overall_status": pipeline_summary.get("overall_status"),
        "total_duration_seconds": pipeline_summary.get("total_duration_seconds"),
        "performance_summary": pipeline_summary.get("performance_summary", {}),
    }


def _extract_summary_step_statuses(
    pipeline_summary: Dict[str, Any] | None,
) -> Dict[str, str]:
    if not pipeline_summary:
        return {}
    statuses: Dict[str, str] = {}
    for raw_step in pipeline_summary.get("steps", []):
        if not isinstance(raw_step, dict):
            continue
        step_number = _step_number_from_summary(raw_step)
        if step_number is None:
            continue
        statuses[str(step_number)] = _normalize_step_status(
            str(raw_step.get("status", ""))
        )
    return statuses


def _step_number_from_summary(step: Dict[str, Any]) -> int | None:
    script_name = str(step.get("script_name", ""))
    match = re.match(r"(?P<number>\d+)_", script_name)
    if match:
        return int(match.group("number"))
    raw_step = step.get("step_number")
    if isinstance(raw_step, int) and raw_step in PIPELINE_STEPS:
        return int(raw_step)
    return None


def _normalize_step_status(status: str) -> str:
    normalized = status.strip().upper().replace("-", "_")
    if normalized in PASSING_STEP_STATUSES:
        return "passed"
    if normalized in SKIPPED_STEP_STATUSES:
        return "skipped"
    if "SKIP" in normalized:
        return "skipped"
    if "SUCCESS" in normalized and "PARTIAL" not in normalized:
        return "passed"
    return "failed"


def _build_step_evidence(
    raw_step_status: Dict[str, str],
    pipeline_output: Path,
    missing_summary_steps: set[str] | None = None,
    *,
    profiled_unsupported_steps: set[int] | None = None,
    acceptance_profile: Dict[str, Any] | None = None,
) -> Dict[str, Dict[str, Any]]:
    evidence: Dict[str, Dict[str, Any]] = {}
    missing_summary_steps = missing_summary_steps or set()
    profiled_unsupported_steps = profiled_unsupported_steps or set()
    unsupported_reasons = (acceptance_profile or dict(DEFAULT_ACCEPTANCE_PROFILE)).get(
        "unsupported_step_reasons", {}
    )
    render_reason = _render_skip_or_failure_reason(pipeline_output)
    execution_reason = _execution_skip_or_failure_reason(pipeline_output)
    for step, status in raw_step_status.items():
        reason = None
        evidence_status = status
        if int(step) in profiled_unsupported_steps and status == "skipped":
            reason = str(
                unsupported_reasons.get(
                    step,
                    "profiled unsupported model family step",
                )
            )
            evidence_status = "skipped"
        elif step in missing_summary_steps:
            reason = "missing_summary_evidence"
            evidence_status = "failed"
        elif step == "11":
            reason = render_reason
        elif step == "12":
            reason = execution_reason
        artifact_links = _step_artifact_links(step, pipeline_output)
        if (
            status == "passed"
            and step in STEP_ARTIFACT_REQUIREMENTS
            and not artifact_links
        ):
            reason = "missing_artifact_evidence"
            evidence_status = "failed"
        evidence[step] = {
            "raw_status": status,
            "status": evidence_status,
            "acceptance": (
                "profiled_unsupported_skip"
                if int(step) in profiled_unsupported_steps and status == "skipped"
                else "required"
            ),
            "reason": reason,
            "artifact_links": artifact_links,
        }
    return evidence


def _apply_acceptance_profile(
    raw_step_status: Dict[str, str],
    step_evidence: Dict[str, Dict[str, Any]],
    acceptance_profile: Dict[str, Any],
) -> Dict[str, str]:
    effective = {
        step: str(step_evidence.get(step, {}).get("status", raw_status))
        for step, raw_status in raw_step_status.items()
    }
    allowed_steps = set(
        _coerce_step_list(acceptance_profile["allow_unsupported_steps"])
    )
    for step in allowed_steps:
        key = str(step)
        evidence = step_evidence.get(key, {})
        if evidence.get("acceptance") == "profiled_unsupported_skip":
            effective[key] = "skipped"
    return effective


def _profiled_unsupported_steps(
    only_steps: str | None, acceptance_profile: Dict[str, Any]
) -> set[int]:
    """Return profile-declared unsupported evidence steps selected for this run."""
    selected = set(PIPELINE_STEPS if not only_steps else _parse_step_list(only_steps))
    profiled = set(_coerce_step_list(acceptance_profile["allow_unsupported_steps"]))
    return selected & profiled


def _selected_steps_passed(
    step_status: Dict[str, str],
    only_steps: str | None,
    acceptance_profile: Dict[str, Any],
    step_evidence: Dict[str, Dict[str, Any]],
) -> bool:
    selected = set(PIPELINE_STEPS if not only_steps else _parse_step_list(only_steps))
    allowed_steps = set(
        _coerce_step_list(acceptance_profile["allow_unsupported_steps"])
    )
    for step in selected:
        key = str(step)
        status = step_status.get(key)
        if status == "passed":
            continue
        if (
            status == "skipped"
            and step in allowed_steps
            and step_evidence.get(key, {}).get("acceptance")
            == "profiled_unsupported_skip"
        ):
            continue
        return False
    return True


def _pipeline_run_outcome_acceptable(
    return_code: int,
    pipeline_summary: Dict[str, Any],
    step_evidence: Dict[str, Dict[str, Any]],
) -> bool:
    """Reject contradictory run outcomes unless all failures are profiled skips."""
    if return_code in {0, 2} and _summary_status_is_acceptable(pipeline_summary):
        return True
    if return_code == 0 and any(
        evidence.get("acceptance") == "profiled_unsupported_skip"
        for evidence in step_evidence.values()
    ):
        return True
    return False


def _summary_status_is_acceptable(pipeline_summary: Dict[str, Any]) -> bool:
    status = str(pipeline_summary.get("overall_status", "")).strip().upper()
    if not status:
        return False
    return status in ACCEPTABLE_SUMMARY_STATUSES


def _render_skip_or_failure_reason(pipeline_output: Path) -> str | None:
    summary = _load_first_json(
        pipeline_output / "11_render_output", ("render_processing_summary.json",)
    )
    successful = summary.get("successful_framework_renderings")
    messages = [
        str(item.get("message", ""))
        for item in summary.get("failed_framework_renderings", [])
        if isinstance(item, dict) and item.get("message")
    ]
    if successful == 0 and messages:
        return "; ".join(messages)
    if successful == 0:
        # The render receipt schema has no top-level ``message`` key; when no
        # framework diagnostics exist there is nothing more specific to say.
        return "no compatible renderings"
    if messages:
        return "partial_render_failure"
    return None


def _execution_skip_or_failure_reason(pipeline_output: Path) -> str | None:
    summary = _load_first_json(
        pipeline_output / "12_execute_output", ("execution_summary.json",)
    )
    parts: list[str] = []
    for key in ("skipped_reason", "message", "failure_reason", "error"):
        if summary.get(key):
            parts.append(str(summary[key]))
    for item in summary.get("render_failures", []):
        if isinstance(item, dict) and item.get("message"):
            parts.append(str(item["message"]))
    return "; ".join(parts) if parts else None


def _load_first_json(root: Path, names: Sequence[str]) -> Dict[str, Any]:
    if not root.exists():
        return {}
    for path in sorted(root.rglob("*.json")):
        if path.name in set(names):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {}
            return payload if isinstance(payload, dict) else {}
    return {}


def _step_artifact_links(step: str, pipeline_output: Path) -> List[str]:
    requirements = STEP_ARTIFACT_REQUIREMENTS.get(step, ())
    links = []
    for relative in requirements:
        path = pipeline_output / relative
        if not path.is_file():
            return []
        links.append(str(path))
    return links


def _parse_step_list(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _count_step_statuses(step_status: Dict[str, str]) -> Dict[str, int]:
    counts = {"available": 0, "passed": 0, "skipped": 0, "failed": 0}
    for status in step_status.values():
        counts[status] = counts.get(status, 0) + 1
    return counts


def _collect_family_artifacts(pipeline_output: Path) -> List[str]:
    if not pipeline_output.exists():
        return []
    return [str(path) for path in sorted(pipeline_output.rglob("*")) if path.is_file()][
        :50
    ]


def render_model_family_acceptance_markdown(ledger: Dict[str, Any]) -> str:
    """Render a compact Markdown report from a model-family acceptance ledger."""
    lines = [
        "# GNN Model Family Acceptance Ledger",
        "",
        f"- Schema: {ledger['schema']}",
        f"- Families: {ledger['family_count']}",
        f"- Strict: {str(ledger['strict']).lower()}",
        f"- Only steps: {ledger['only_steps'] or 'full pipeline'}",
        f"- Frameworks: {ledger.get('frameworks') or 'pipeline default'}",
        "",
        "| Family | Status | Models | Passed Steps | Failed Steps | Skipped Steps | Raw Failed Steps | Profiled Unsupported Skips |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family in ledger["families"]:
        counts = family["step_status_counts"]
        raw_failed = sum(
            1 for status in family.get("raw_steps", {}).values() if status == "failed"
        )
        profiled_unsupported = sum(
            1
            for evidence in family.get("step_evidence", {}).values()
            if isinstance(evidence, dict)
            and evidence.get("acceptance") == "profiled_unsupported_skip"
        )
        lines.append(
            "| {name} | {status} | {models} | {passed} | {failed} | {skipped} | {raw_failed} | {profiled} |".format(
                name=family["name"],
                status=family["status"],
                models=family["interpretability_summary"]["model_count"],
                passed=counts.get("passed", 0),
                failed=counts.get("failed", 0),
                skipped=counts.get("skipped", 0),
                raw_failed=raw_failed,
                profiled=profiled_unsupported,
            )
        )
    return "\n".join(lines) + "\n"


def _write_ledger_artifacts(ledger: Dict[str, Any], output_dir: Path) -> None:
    """Write the machine-readable ledger and its Markdown report."""
    (output_dir / "model_family_acceptance_ledger.json").write_text(
        json.dumps(ledger, indent=2),
        encoding="utf-8",
    )
    (output_dir / "model_family_acceptance_ledger.md").write_text(
        render_model_family_acceptance_markdown(ledger),
        encoding="utf-8",
    )


def _tail_text(text: str | None, max_chars: int = 4000) -> str:
    if not text:
        return ""
    return text[-max_chars:]


def _coerce_timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value

"""Admit the summary and artifact inventory before rendering a report."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gnn.pipeline.artifact_ownership import verify_owned_artifacts
from gnn.pipeline.run_manifest import DEFAULT_MANIFEST_SUBDIR, verify_run_manifests


@dataclass(frozen=True)
class ReportEvidence:
    summary: dict[str, Any]
    files: frozenset[Path] | None
    description: str


def load_report_evidence(
    output_dir: Path,
    summary_paths: tuple[Path, ...],
    *,
    active_run_id: str | None = None,
) -> ReportEvidence:
    """Use active snapshots or verify terminal bundles before projecting files.

    Pre-contract reports may still inspect historical directories. A malformed
    active snapshot or a broken terminal bundle cannot select that fallback.
    """
    manifests = output_dir / DEFAULT_MANIFEST_SUBDIR
    terminal_bundle = active_run_id is None and (manifests / "index.json").exists()
    for path in summary_paths:
        if (
            terminal_bundle or active_run_id is not None
        ) and not path.resolve().is_relative_to(output_dir.resolve()):
            raise ValueError("Report summary escapes its output directory")
        try:
            payload = path.read_bytes()
        except FileNotFoundError:
            continue
        try:
            summary = json.loads(payload)
            if not isinstance(summary, dict):
                raise ValueError("Report summary must be a JSON object")
        except (ValueError, UnicodeError):
            if active_run_id is not None or terminal_bundle:
                raise
            continue

        if active_run_id is not None and summary.get("run_id") != active_run_id:
            raise ValueError("Report summary belongs to a different run")
        integrity = summary.get("evidence_integrity", {})
        if not isinstance(integrity, dict):
            raise ValueError("Invalid report evidence status")
        if (
            integrity.get("status") == "failed"
            and summary.get("overall_status") == "FAILED"
        ):
            return ReportEvidence(
                summary, frozenset(), "Terminal evidence failed verification"
            )
        if "artifact_inventory_contract" not in summary:
            if active_run_id is not None or "evidence_integrity" in summary:
                raise ValueError(
                    "Report summary is missing its artifact inventory contract"
                )
            if terminal_bundle:
                _verify_terminal_report(output_dir, path, manifests, payload)
            return ReportEvidence(summary, None, "Historical directory inventory")
        if summary["artifact_inventory_contract"] != "current-run-v1":
            raise ValueError("Unsupported report artifact inventory contract")
        if not path.resolve().is_relative_to(output_dir.resolve()):
            raise ValueError("Report summary escapes its output directory")
        if not isinstance(summary.get("run_id"), str) or not summary["run_id"]:
            raise ValueError("Report summary is missing its run identity")

        if active_run_id is None:
            if integrity.get("status") != "verified" or not summary.get(
                "run_session_sha256"
            ):
                raise ValueError("Report requires finalized evidence for this run")
            _verify_terminal_report(output_dir, path, manifests, payload)
            description = "Verified terminal run inventory"
        else:
            try:
                verify_owned_artifacts(output_dir, summary)
            except (KeyError, TypeError, AttributeError) as error:
                raise ValueError("Invalid report artifact inventory") from error
            description = "Current invocation inventory"

        files = frozenset(
            output_dir / artifact["path"]
            for step in summary.get("steps", [])
            for artifact in step.get("artifacts", [])
        )
        return ReportEvidence(summary, files, description)

    if active_run_id is not None or terminal_bundle:
        raise ValueError("Report summary is unavailable")
    return ReportEvidence({}, None, "Historical directory inventory")


def _verify_terminal_report(
    output_dir: Path, path: Path, manifests: Path, payload: bytes
) -> None:
    if not (manifests / "index.json").resolve().is_relative_to(output_dir.resolve()):
        raise ValueError("Report manifest directory escapes its output directory")
    problems = verify_run_manifests(manifests, output_dir, summary_path=path)
    if problems:
        raise ValueError("Report evidence verification failed: " + "; ".join(problems))
    if path.read_bytes() != payload:
        raise ValueError("Report summary changed during verification")

#!/usr/bin/env python3
"""
Execution metadata and summary artifacts for GNN Step 12.

Owns the RxInfer execution-metadata sidecar/TOML loaders, the Step 11
render-summary contract, execution-summary merging/slimming, and the
markdown execution report. Extracted from ``execute.processor``.
"""

import hashlib
import json
import logging
import os
import tempfile
import tomllib
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .types import _EXECUTABLE_SUFFIXES

logger = logging.getLogger(__name__)


def _load_rxinfer_execution_metadata_sidecar(script_path: Path) -> Dict[str, Any]:
    """Load declared RxInfer execution metadata from JSON sidecar artifacts."""
    candidates = [
        script_path.with_suffix(".metadata.json"),
        script_path.with_name(f"{script_path.stem}_metadata.json"),
    ]
    seen: set[Path] = set()
    probe: Dict[str, Any] = {}
    for metadata_path in candidates:
        if metadata_path in seen or not metadata_path.exists():
            continue
        seen.add(metadata_path)
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            probe = {
                "metadata_probe": "unreadable",
                "probe_source": str(metadata_path),
                "probe_reason": str(exc),
            }
            continue
        if not isinstance(data, dict):
            probe = {
                "metadata_probe": "not_a_mapping",
                "probe_source": str(metadata_path),
                "probe_reason": f"JSON top-level value is {type(data).__name__}",
            }
            continue
        if data.get("schema") != "gnn_rxinfer_execution_metadata_v1":
            probe = {
                "metadata_probe": "schema_mismatch",
                "probe_source": str(metadata_path),
                "probe_reason": f"schema is {data.get('schema')!r}, expected "
                "'gnn_rxinfer_execution_metadata_v1'",
            }
            continue
        if data.get("script_sha256") != _sha256_file(script_path):
            probe = {
                "metadata_probe": "sha_mismatch",
                "probe_source": str(metadata_path),
                "probe_reason": "sidecar script_sha256 does not match script",
            }
            continue
        if "agent_count" not in data and "topology" not in data:
            probe = {
                "metadata_probe": "missing_topology_fields",
                "probe_source": str(metadata_path),
                "probe_reason": "sidecar has neither agent_count nor topology",
            }
            continue
        topology = data.get("topology")
        if not isinstance(topology, dict):
            topology = {}
        topology.setdefault("source", str(metadata_path))
        data["topology"] = topology
        data["agent_count"] = int(data.get("agent_count") or 0)
        data["metadata_provenance"] = data.get(
            "metadata_provenance", "rendered_rxinfer_sidecar"
        )
        data["metadata_verification"] = "script_sha256_match"
        return data
    return probe if seen else {}


def _load_rxinfer_execution_metadata_from_script(script_path: Path) -> Dict[str, Any]:
    """Load agent population metadata from rendered RxInfer metadata artifacts."""
    if not script_path.exists():
        return {}
    sidecar_metadata = _load_rxinfer_execution_metadata_sidecar(script_path)
    sidecar_probe: Dict[str, Any] = {}
    if sidecar_metadata and "metadata_probe" not in sidecar_metadata:
        return sidecar_metadata
    if "metadata_probe" in sidecar_metadata:
        # Sidecars that parse but fail validation (wrong/stale schema, sha
        # mismatch after a re-render, non-mapping, missing fields) are
        # benign: the render step rewrites them alongside the script, so a
        # leftover mismatch carries no signal for this script and must be
        # ignored exactly like an absent sidecar. Only a sidecar that
        # cannot be read at all (corrupt JSON / I/O failure) surfaces a
        # probe receipt here; the low-level sidecar loader keeps probes
        # for every failure class.
        if sidecar_metadata["metadata_probe"] == "unreadable":
            sidecar_probe = sidecar_metadata
    toml_candidates = [script_path.with_suffix(".toml")]
    seen_toml: set[Path] = set()
    toml_probe: Dict[str, Any] = {}
    for toml_path in toml_candidates:
        if toml_path in seen_toml or not toml_path.exists():
            continue
        seen_toml.add(toml_path)
        try:
            data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            toml_probe = {
                "metadata_probe": "unreadable",
                "probe_source": str(toml_path),
                "probe_reason": str(exc),
            }
            continue
        agents = data.get("agents", [])
        model = data.get("model", {})
        agent_count = (
            len(agents) if isinstance(agents, list) else model.get("nr_agents")
        )
        agent_ids = [
            agent.get("id")
            for agent in agents
            if isinstance(agent, dict) and "id" in agent
        ]
        topology_data = data.get("topology", {})
        topology: Dict[str, Any] = {
            "type": "agent_population",
            "agent_ids": agent_ids,
            "source": str(toml_path),
        }
        if isinstance(topology_data, dict):
            topology["type"] = str(topology_data.get("type") or topology["type"])
            topology["agent_ids"] = topology_data.get("agent_ids") or agent_ids
            if "edges" in topology_data:
                topology["edges"] = topology_data["edges"]
            if "clusters" in topology_data:
                topology["clusters"] = topology_data["clusters"]
            if "message_passing" in topology_data:
                topology["message_passing"] = topology_data["message_passing"]
        return {
            "agent_count": int(agent_count or 0),
            "topology": topology,
            "metadata_provenance": "rxinfer_toml_sidecar",
            "metadata_verification": "exact_stem_toml",
        }
    if sidecar_probe:
        return sidecar_probe
    return toml_probe


def _sha256_file(path: Path) -> str:
    """Return the SHA256 digest for an executable script."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_render_summary_contract(
    render_output_dir: Path,
    requested_frameworks: List[str],
    logger: logging.Logger,
    target_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
) -> Tuple[Optional[set[Path]], List[Dict[str, str]], List[Dict[str, str]]]:
    """Load the latest Step 11 render contract for script filtering and failures.

    Returns a 3-tuple ``(allowed_scripts, render_failures, unsupported_receipts)``:

    * ``allowed_scripts`` — executable scripts permitted by the contract
      (``None`` when the contract is missing or unreadable).
    * ``render_failures`` — identity-verification / render-failure receipts.
    * ``unsupported_receipts`` — receipts for frameworks the renderer declared
      cannot represent the model (nothing failed; there is nothing to execute).

    ``target_dir`` (when provided) scopes the contract to the current pipeline
    invocation. The pipeline runs Step 12 once per top-level input folder, so a
    folder-scoped invocation must only discover scripts rendered from that
    folder's source files; a global invocation (``target_dir`` is the base
    input dir) naturally matches every source file.
    """
    summary_file = render_output_dir / "render_processing_summary.json"
    if not summary_file.exists():
        return None, [], []

    try:
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read render summary %s: %s", summary_file, exc)
        return None, [], []

    if not isinstance(summary, dict):
        return None, [], []

    expected_run = run_id or os.environ.get("GNN_RUN_ID")
    receipt_identity = summary.get("receipt_identity")
    if expected_run and (
        not isinstance(receipt_identity, dict)
        or receipt_identity.get("run_id") != expected_run
    ):
        return (
            set(),
            [
                {
                    "file": str(summary_file),
                    "framework": "all",
                    "message": "Render receipt belongs to a different run",
                }
            ],
            [],
        )

    def _in_scope(source_file: str) -> bool:
        from gnn.pipeline.run_context import current_run_context

        context = current_run_context()
        if context is not None:
            return Path(source_file).stem in {
                model.artifact_stem for model in context.selected_models(12)
            }
        if target_dir is None:
            return True
        try:
            return Path(source_file).resolve().is_relative_to(target_dir.resolve())
        except (OSError, ValueError):
            return True

    requested = set(requested_frameworks)
    allowed_scripts: set[Path] = set()
    render_failures: List[Dict[str, str]] = []
    unsupported_receipts: List[Dict[str, str]] = []
    file_results = summary.get("file_results")
    if not isinstance(file_results, dict):
        return None, [], []

    for source_file, file_result in file_results.items():
        if not _in_scope(source_file):
            continue
        framework_results = (
            file_result.get("framework_results", {})
            if isinstance(file_result, dict)
            else {}
        )
        if not isinstance(framework_results, dict):
            continue
        for framework, framework_result in framework_results.items():
            if framework not in requested or not isinstance(framework_result, dict):
                continue
            if isinstance(receipt_identity, dict):
                try:
                    source = Path(source_file)
                    source_identity = {
                        "path": str(source.resolve()),
                        "sha256": _sha256_file(source),
                    }
                    artifacts = framework_result.get("artifact_identities")
                    output_files = framework_result.get("output_files", [])
                    if file_result.get("source_identity") != source_identity:
                        raise ValueError(
                            "Source content no longer matches render receipt"
                        )
                    if framework_result.get("success"):
                        if not isinstance(artifacts, list) or not isinstance(
                            output_files, list
                        ):
                            raise ValueError("Missing artifact identities")
                        expected = {str(Path(item).resolve()) for item in output_files}
                        actual = {item["path"] for item in artifacts}
                        if expected != actual:
                            raise ValueError(
                                "Artifact inventory no longer matches render receipt"
                            )
                        for artifact in artifacts:
                            if (
                                _sha256_file(Path(artifact["path"]))
                                != artifact["sha256"]
                            ):
                                raise ValueError(
                                    "Artifact content no longer matches render receipt"
                                )
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    render_failures.append(
                        {
                            "file": str(source_file),
                            "framework": str(framework),
                            "message": str(exc),
                        }
                    )
                    continue
            if framework_result.get("unsupported"):
                # The renderer declared this framework cannot represent the
                # model (categorical backend, continuous model): nothing to
                # execute and nothing failed. Surface a receipt instead of
                # silently dropping the reason.
                unsupported_receipts.append(
                    {
                        "file": str(source_file),
                        "framework": str(framework),
                        "reason": str(
                            framework_result.get("message")
                            or framework_result.get("reason")
                            or "renderer declared this framework unsupported for this model"
                        ),
                        "status": "unsupported",
                    }
                )
                continue
            if framework_result.get("success"):
                for output_file in framework_result.get("output_files") or []:
                    # Only executable artifacts are execution candidates. Stan
                    # renders a companion ``.stan`` program beside its Python
                    # driver; data/config side files are never "missing scripts".
                    if Path(output_file).suffix.lower() not in _EXECUTABLE_SUFFIXES:
                        continue
                    allowed_scripts.add(Path(output_file).resolve())
            else:
                render_failures.append(
                    {
                        "file": Path(str(source_file)).name,
                        "framework": str(framework),
                        "message": str(framework_result.get("message", "")),
                    }
                )

    return allowed_scripts, render_failures, unsupported_receipts


def _summarize_collected_outputs(coll: Any) -> Any:
    """Replace bulky collected_outputs with counts safe for aggregate JSON."""
    if coll is None:
        return None
    if isinstance(coll, dict):
        out: Dict[str, Any] = {}
        for k, v in coll.items():
            if isinstance(v, list):
                out[str(k)] = {"count": len(v)}
            elif isinstance(v, dict):
                out[str(k)] = {"n_keys": len(v)}
            else:
                out[str(k)] = v
        return out
    if isinstance(coll, list):
        return {"count": len(coll)}
    return coll


def _slim_execution_detail(detail: Dict[str, Any]) -> Dict[str, Any]:
    """Strip heavy fields from a per-script execution result for aggregate summaries."""
    keys_keep = (
        "script_path",
        "script_name",
        "framework",
        "model_name",
        "model_id",
        "source_path",
        "source_relative_path",
        "source_sha256",
        "executor",
        "success",
        "skipped",
        "status",
        "attempts_started",
        "return_code",
        "cancelled",
        "containment",
        "cleanup_verified",
        "streams_drained",
        "cleanup_timeout_seconds",
        "cleanup_error",
        "execution_error_type",
        "error",
        "error_type",
        "execution_time",
        "timestamp",
        "execution_benchmark_repeats",
        "execution_time_mean",
        "execution_time_std",
        "execution_time_samples",
        "child_peak_rss_mb",
        "rss_sample_interval_seconds",
        "rss_samples_count",
        "structured_result_file",
        "output_file",
        "implementation_directory",
        "execution_metadata",
        "script_identity",
        "source_scope",
    )
    slim: Dict[str, Any] = {}
    for k in keys_keep:
        if k in detail:
            slim[k] = detail[k]
    if isinstance(detail.get("stdout"), str):
        slim["stdout_length"] = len(detail["stdout"])
    if isinstance(detail.get("stderr"), str):
        slim["stderr_length"] = len(detail["stderr"])
    if "collected_outputs" in detail:
        slim["collected_outputs_summary"] = _summarize_collected_outputs(
            detail["collected_outputs"]
        )
    return slim


def _execution_detail_key(detail: Dict[str, Any]) -> str:
    """Stable identity for one executed script across folder invocations."""
    path = detail.get("script_path") or detail.get("path")
    if path:
        return str(Path(path).resolve())
    return str(detail.get("script_name") or detail.get("name") or "")


_STATUS_SEVERITY = {
    "failed": 3,
    "skipped": 2,
    "success_with_render_failures": 1,
    "success_with_skips": 1,
    "success": 0,
}


def _atomic_execution_json(path: Path, payload: Dict[str, Any]) -> None:
    """Publish a complete receipt using an atomic same-filesystem replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _execution_input_identity(target: Path) -> Dict[str, Any]:
    """Fingerprint the current input scope without relying on file names alone."""
    files = {}
    if target.is_dir():
        for path in sorted(target.rglob("*")):
            if path.is_file() and path.suffix.lower() in {
                ".md",
                ".json",
                ".yaml",
                ".yml",
                ".py",
                ".jl",
            }:
                files[str(path.resolve())] = _sha256_file(path)
    return {"path": str(target.resolve()), "files": files}


def _execution_script_identity(detail: Dict[str, Any]) -> Dict[str, Any]:
    path = Path(_execution_detail_key(detail)).resolve()
    return {"path": str(path), "sha256": _sha256_file(path) if path.is_file() else None}


def _merge_prior_execution_summary(
    execution_results: Dict[str, Any], results_file: Path, logger: Any
) -> None:
    """Replace this scope and recount verified records from the same run/config.

    Receipts from earlier runs and unbound receipts are archived but cannot
    authorize current
    execution claims. A fresh standalone invocation receives a fresh run ID.
    """
    prior: Dict[str, Any] = {}
    if results_file.is_file():
        try:
            data = json.loads(results_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                prior = data
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Could not read prior execution summary %s: %s", results_file, exc
            )
    if prior:
        digest = hashlib.sha256(
            json.dumps(prior, sort_keys=True, default=str).encode()
        ).hexdigest()
        archive = results_file.parent / "history" / f"execution-{digest}.json"
        if not archive.exists():
            _atomic_execution_json(archive, prior)
    config = execution_results.get("configuration", {})
    identity = {
        "run_id": str(
            execution_results.get("run_id")
            or os.environ.get("GNN_RUN_ID")
            or uuid.uuid4().hex
        ),
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode()
        ).hexdigest(),
    }
    scope = str(Path(execution_results.get("target_directory") or ".").resolve())
    current_details = [
        dict(d)
        for d in execution_results.get("execution_details", [])
        if isinstance(d, dict)
    ]
    for detail in current_details:
        detail["script_identity"] = detail.get(
            "script_identity"
        ) or _execution_script_identity(detail)
        detail["source_scope"] = scope
    current = {
        key: execution_results.get(key)
        for key in (
            "status",
            "success",
            "exit_code",
            "outcome_reason",
            "render_failures",
            "missing_render_scripts",
            "missing_render_summary",
        )
    }
    current["input_identity"] = execution_results.get(
        "input_identity"
    ) or _execution_input_identity(Path(scope))
    current["execution_details"] = [_slim_execution_detail(d) for d in current_details]
    invocations: Dict[str, Any] = {}
    from gnn.pipeline.run_context import current_run_context

    context = current_run_context()
    prior_invocations = prior.get("invocation_receipts", {})
    if (
        context is None
        and prior.get("receipt_identity") == identity
        and isinstance(prior_invocations, dict)
    ):
        for old_scope, record in prior_invocations.items():
            if (
                old_scope == scope
                or Path(old_scope).is_relative_to(Path(scope))
                or Path(scope).is_relative_to(Path(old_scope))
                or not isinstance(record, dict)
            ):
                continue
            try:
                if record.get("input_identity") != _execution_input_identity(
                    Path(old_scope)
                ):
                    continue
                if not all(
                    d.get("script_identity", {}).get("sha256")
                    and d.get("script_identity") == _execution_script_identity(d)
                    for d in record.get("execution_details", [])
                ):
                    continue
            except OSError as e:
                logger.debug(
                    "Prior invocation %s excluded from merge: identity probe "
                    "failed: %s",
                    old_scope,
                    e,
                )
                continue
            invocations[old_scope] = record
    invocations[scope] = current
    execution_results["receipt_identity"] = identity
    execution_results["run_id"] = identity["run_id"]
    execution_results["input_identity"] = current["input_identity"]
    execution_results["invocation_receipts"] = invocations
    execution_results["current_invocation"] = {**current, "target_directory": scope}
    execution_results["merged_prior_folder_runs"] = len(invocations) - 1
    merged_by_key = {}
    for old_scope, record in invocations.items():
        for detail in (
            current_details if old_scope == scope else record["execution_details"]
        ):
            merged_by_key[(_execution_detail_key(detail), detail.get("framework"))] = (
                detail
            )
    merged = list(merged_by_key.values())
    execution_results["execution_details"] = merged
    successful = failed = skipped = 0
    framework_status: Dict[str, Dict[str, Any]] = {}
    for detail in merged:
        fw = str(detail.get("framework", "unknown"))
        counts = framework_status.setdefault(
            fw, {"executions": 0, "successful": 0, "failed": 0, "skipped": 0}
        )
        counts["executions"] += 1
        if detail.get("skipped"):
            skipped += 1
            counts["skipped"] += 1
        elif detail.get("success"):
            successful += 1
            counts["successful"] += 1
        else:
            failed += 1
            counts["failed"] += 1
        if detail.get("error") and not detail.get("success"):
            counts["error"] = detail["error"]
    for counts in framework_status.values():
        counts["status"] = (
            "failed"
            if counts["failed"]
            else "success_with_skips"
            if counts["successful"] and counts["skipped"]
            else "success"
            if counts["successful"]
            else "skipped"
        )
    attempted = len(merged) - skipped
    execution_results.update(
        {
            "framework_status": framework_status,
            "total_scripts_found": len(merged),
            "total_scripts": len(merged),
            "successful_executions": successful,
            "failed_executions": failed,
            "skipped_executions": skipped,
            "attempted_scripts": attempted,
            "success_rate": round(successful / attempted * 100, 2)
            if attempted
            else 0.0,
        }
    )
    for key in ("render_failures", "missing_render_scripts"):
        items = {}
        for record in invocations.values():
            for item in record.get(key) or []:
                items[json.dumps(item, sort_keys=True, default=str)] = item
        execution_results[key] = list(items.values())
    failed_scopes = [
        record for record in invocations.values() if record.get("status") == "failed"
    ]
    skipped_scopes = any(
        record.get("status") == "skipped" for record in invocations.values()
    )
    if failed or failed_scopes:
        # Preserve the classifier's specific failure (for example a strict
        # requested-framework failure) when recomputing aggregate counters.
        reason = next(
            (
                record["outcome_reason"]
                for record in failed_scopes
                if record.get("outcome_reason")
            ),
            "script_execution_failure",
        )
        execution_results.update(
            status="failed", success=False, exit_code=1, outcome_reason=reason
        )
    elif (
        successful
        or any(record.get("success") is True for record in invocations.values())
    ) and (skipped or skipped_scopes):
        execution_results.update(
            status="success_with_skips",
            success=True,
            exit_code=0,
            outcome_reason="optional_dependencies_unavailable",
        )
    elif successful and execution_results["render_failures"]:
        execution_results.update(
            status="success_with_render_failures",
            success=True,
            exit_code=0,
            outcome_reason="best_effort_render_subset_executed",
        )
    elif successful:
        execution_results.update(
            status="success",
            success=True,
            exit_code=0,
            outcome_reason="all_scripts_succeeded",
        )
    else:
        execution_results.update(
            status="skipped",
            success=False,
            exit_code=2,
            outcome_reason="no_executable_scripts",
        )


def generate_execution_report(
    execution_results: Dict[str, Any], results_dir: Path, logger: logging.Logger
) -> None:
    """
    Generate a comprehensive execution report.

    Args:
        execution_results: Dictionary with execution results
        results_dir: Directory to save the report
        logger: Logger instance
    """
    summaries_dir = results_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    report_file = summaries_dir / "execution_report.md"

    try:
        with open(report_file, "w") as f:
            f.write("# GNN Script Execution Report\n\n")
            f.write(f"**Generated:** {execution_results['timestamp']}\n")
            f.write(f"**Target Directory:** {execution_results['target_directory']}\n")
            f.write(
                f"**Output Directory:** {execution_results['output_directory']}\n\n"
            )

            f.write("## Summary\n\n")
            f.write(
                f"- **Total Scripts Found:** {execution_results['total_scripts_found']}\n"
            )
            f.write(
                f"- **Successful Executions:** {execution_results['successful_executions']}\n"
            )
            f.write(
                f"- **Failed Executions:** {execution_results['failed_executions']}\n"
            )
            skipped = execution_results.get("skipped_executions", 0)
            if skipped:
                f.write(f"- **Skipped (dependency not installed):** {skipped}\n")
            f.write("\n")

            if execution_results["execution_details"]:
                f.write("## Execution Details\n\n")

                for detail in execution_results["execution_details"]:
                    if detail.get("skipped"):
                        status = "⏭️ SKIPPED"
                    elif detail["success"]:
                        status = "✅ SUCCESS"
                    else:
                        status = "❌ FAILED"
                    f.write(f"### {detail['script_name']} - {status}\n\n")
                    f.write(f"- **Framework:** {detail['framework']}\n")
                    f.write(f"- **Executor:** {detail['executor']}\n")
                    f.write(f"- **Path:** `{detail['script_path']}`\n")
                    if detail.get("skipped"):
                        f.write(
                            f"- **Reason:** {detail.get('error', 'Dependency not installed')}\n"
                        )
                    else:
                        f.write(
                            f"- **Return Code:** {detail.get('return_code', 'N/A')}\n"
                        )
                        f.write(
                            f"- **Execution Time:** {detail.get('execution_time', 0):.2f} seconds\n"
                        )

                        if not detail["success"] and "error" in detail:
                            f.write(f"- **Error:** {detail['error']}\n")

                        if "output_file" in detail:
                            f.write(f"- **Detailed Output:** {detail['output_file']}\n")

                    f.write("\n")

            f.write("## Next Steps\n\n")
            if execution_results["failed_executions"] > 0:
                f.write("1. Review failed executions above\n")
                f.write(
                    "2. Check individual output files for detailed error information\n"
                )
                f.write("3. Ensure required dependencies are installed\n")
                f.write("4. Verify script syntax and functionality\n\n")
            elif skipped:
                f.write(
                    "Skipped scripts are due to missing optional dependencies or unavailable system runtimes. Run `uv sync` for core Python backends; add `uv sync --extra ml-ai --extra graphs` for optional Python extension groups, and install Julia/D2 system tools as needed.\n\n"
                )
            else:
                f.write(
                    "All scripts executed successfully! Check individual output files for results.\n\n"
                )

        logger.info(f"Generated execution report: {report_file}")

    except Exception as e:
        logger.error(f"Failed to generate execution report: {e}")

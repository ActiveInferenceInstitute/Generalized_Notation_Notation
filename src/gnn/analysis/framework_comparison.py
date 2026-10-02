#!/usr/bin/env python3
"""
Cross-framework execution-output comparison and reporting for GNN Step 16 analysis.

Extracted from ``analysis.analyzer``.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import numpy as np

from .comparison_contract import prepare_scientific_record, scientific_comparisons


def _valid_execution_times(values: Any) -> tuple[list[float], int]:
    if values is None:
        return [], 0
    if not isinstance(values, (list, tuple, np.ndarray)):
        return [], 1
    if isinstance(values, np.ndarray) and values.ndim != 1:
        return [], 1
    valid = [
        float(value)
        for value in values
        if isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and np.isfinite(value)
        and value >= 0
    ]
    return valid, len(values) - len(valid)


def _derive_framework_status(
    details: List[Dict[str, Any]],
) -> tuple[str, Optional[str]]:
    """Derive ``(status, reason)`` from one framework's execution details.

    The status vocabulary mirrors Step 12's ``_update_framework_status``
    (``execute.processor``): ``failed`` beats ``success_with_skips`` beats
    ``success``; an all-skipped set is ``skipped``. Skipped records count as
    skipped even when a producer also set ``success`` truthiness.
    """
    skipped = [d for d in details if bool(d.get("skipped"))]
    attempted = [d for d in details if not bool(d.get("skipped"))]
    failed = [d for d in attempted if not bool(d.get("success"))]
    succeeded = [d for d in attempted if bool(d.get("success"))]
    if failed:
        return "failed", str(failed[0].get("error") or "execution failed")
    if succeeded and skipped:
        return "success_with_skips", str(
            skipped[0].get("error") or "dependency unavailable"
        )
    if succeeded:
        return "success", None
    if skipped:
        return "skipped", str(skipped[0].get("error") or "dependency unavailable")
    return "no_data", "no execution details recorded"


def _render_failures_by_framework(
    execution_summary: Dict[str, Any],
) -> Dict[str, List[Dict[str, str]]]:
    """Group Step 12 ``render_failures`` records by normalized framework.

    Receipt-mismatch rows carry ``framework == "all"`` and are dropped:
    they do not degrade any single framework's column.
    """
    failures_by_framework: Dict[str, List[Dict[str, str]]] = {}
    for item in execution_summary.get("render_failures") or []:
        if not isinstance(item, dict):
            continue
        framework = (
            str(item.get("framework", ""))
            .strip()
            .lower()
            .replace(".", "_")
            .replace(" ", "_")
        )
        if not framework or framework == "all":
            continue
        failures_by_framework.setdefault(framework, []).append(
            {
                "file": str(item.get("file", "")),
                "message": str(item.get("message", "")),
            }
        )
    return failures_by_framework


def analyze_framework_outputs(
    execution_output_dir: Path,
    logger: Optional[logging.Logger] = None,
    allowed_frameworks: Optional[set[str]] = None,
) -> Dict[str, Any]:
    """
    Analyze and compare outputs from the current execution framework scope.

    Args:
        execution_output_dir: Directory containing execution results (e.g., output/12_execute_output)
        logger: Optional logger instance
        allowed_frameworks: Optional normalized framework names to include

    Returns:
        Dictionary with framework comparison results
    """
    import json
    import logging

    if logger is None:
        logger = logging.getLogger(__name__)

    results: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "frameworks": {},
        "comparisons": {},
        "metrics": {},
    }

    # Read execution summary - check summaries/ subfolder first, recovery to root
    execution_summary_file = (
        execution_output_dir / "summaries" / "execution_summary.json"
    )
    if not execution_summary_file.exists() and not execution_summary_file.is_symlink():
        execution_summary_file = execution_output_dir / "execution_summary.json"
    if not execution_summary_file.exists() and not execution_summary_file.is_symlink():
        logger.warning(f"Execution summary not found at {execution_summary_file}")
        return results
    try:
        execution_summary_file = execution_summary_file.resolve()
        execution_summary_file.relative_to(execution_output_dir.resolve())
        if not execution_summary_file.is_file():
            raise ValueError("Execution summary must be a regular file")
    except (OSError, RuntimeError, ValueError):
        reason = (
            "Execution summary must be a regular file within the current output tree"
        )
        logger.warning(reason)
        results["unavailable_metrics"] = {"execution_summary": reason}
        return results

    try:
        with open(execution_summary_file, "r") as f:
            execution_summary = json.load(f)

        execution_details = execution_summary.get("execution_details", [])

        # Group by framework
        framework_data: dict[Any, Any] = {}
        for detail in execution_details:
            framework = str(detail.get("framework", "unknown"))
            framework = framework.lower().replace(".", "_").replace(" ", "_")
            if allowed_frameworks and framework not in allowed_frameworks:
                continue
            if framework not in framework_data:
                framework_data[framework] = []
            framework_data[framework].append(detail)

        # Extract metrics from each framework
        for framework, details in framework_data.items():
            status, status_reason = _derive_framework_status(details)
            framework_results: dict[str, Any] = {
                "status": status,
                "status_reason": status_reason,
                "success_count": sum(
                    1
                    for d in details
                    if not bool(d.get("skipped")) and bool(d.get("success"))
                ),
                "skipped_count": sum(1 for d in details if bool(d.get("skipped"))),
                "failed_count": sum(
                    1
                    for d in details
                    if not bool(d.get("skipped")) and not bool(d.get("success"))
                ),
                "total_count": len(details),
                "execution_times": [d.get("execution_time", 0) for d in details],
                "output_files": [d.get("output_file", "") for d in details],
                "implementation_dirs": [
                    d.get("implementation_directory", "") for d in details
                ],
            }

            # Try to extract simulation data
            simulation_metrics = _extract_simulation_metrics(
                framework, details, execution_output_dir, logger
            )
            framework_results.update(simulation_metrics)

            results["frameworks"][framework] = framework_results

        # Frameworks whose render (Step 11) failed or whose render receipt no
        # longer matches surface as explicit degraded columns instead of
        # silently disappearing from the comparison.
        for framework, failures in _render_failures_by_framework(
            execution_summary
        ).items():
            if framework in results["frameworks"]:
                results["frameworks"][framework]["render_failures"] = failures
            else:
                results["frameworks"][framework] = {
                    "status": "render_failed",
                    "status_reason": failures[0].get("message")
                    or "render failed for this framework",
                    "render_failures": failures,
                    "success_count": 0,
                    "skipped_count": 0,
                    "failed_count": 0,
                    "total_count": 0,
                    "execution_times": [],
                    "output_files": [],
                    "implementation_dirs": [],
                }

        # Perform cross-framework comparisons
        results["comparisons"] = _compare_framework_results(
            results["frameworks"], logger
        )

        # Generate aggregate metrics
        results["metrics"] = _calculate_aggregate_metrics(results["frameworks"], logger)

    except Exception as e:
        logger.error(f"Error analyzing framework outputs: {e}")
        import traceback

        logger.debug(traceback.format_exc())

    return results


def _discopy_inline_circuit_from_structured(
    data: Dict[str, Any], metrics: Dict[str, Any]
) -> None:
    """Fill circuit_info / model_parameters from Step 12 DisCoPy execution log JSON."""
    sim = data.get("simulation_data") or {}
    analysis = sim.get("analysis")
    params = sim.get("parameters")
    components = sim.get("components") or []
    if analysis is None and not params and not components:
        return
    if not metrics.get("circuit_info"):
        num_c = 0
        if isinstance(analysis, dict):
            num_c = int(analysis.get("num_components", 0) or 0)
        if not num_c and components:
            num_c = len(components)
        metrics["circuit_info"] = {
            "model_name": data.get("model_name", ""),
            "components": components,
            "num_components": num_c,
            "parameters": params or {},
        }
    if params and not metrics.get("model_parameters"):
        metrics["model_parameters"] = {
            "num_states": params.get("num_states", 0),
            "num_observations": params.get("num_observations", 0),
            "num_actions": params.get("num_actions", 0),
        }


def _extract_simulation_metrics(
    framework: str,
    details: List[Dict[str, Any]],
    execution_output_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Retain each execution's scientific identity instead of merging models."""
    rows = [
        _extract_execution_metrics(framework, [detail], execution_output_dir, logger)
        for detail in details
    ]
    loaded = [row for row in rows if row.get("data_source")]
    metrics = (
        dict(loaded[0])
        if len(loaded) == 1
        else _extract_execution_metrics(framework, [], execution_output_dir, logger)
    )
    metrics.pop("_scientific_record", None)
    metrics["execution_times"], invalid_times = _valid_execution_times(
        [detail["execution_time"] for detail in details if "execution_time" in detail]
    )
    if invalid_times:
        metrics["invalid_execution_time_count"] = invalid_times
        metrics.setdefault("unavailable_metrics", {})["invalid_execution_times"] = (
            "Nonfinite, negative, Boolean, or nonnumeric timing entries were excluded"
        )
    metrics["scientific_records"] = [
        row["_scientific_record"] for row in loaded if "_scientific_record" in row
    ]
    if len(loaded) > 1:
        metrics["data_sources"] = [row["data_source"] for row in loaded]
        metrics["unavailable_metrics"] = {
            "aggregate_posterior": "Different execution results retain separate model and native view identities"
        }
    return metrics


def _extract_execution_metrics(
    framework: str,
    details: List[Dict[str, Any]],
    execution_output_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Extract simulation-specific metrics from framework outputs.

    Searches simulation_data/ subdirectories and maps framework-specific
    keys to a standardized format for cross-framework comparison.
    """
    metrics: dict[str, Any] = {
        "beliefs": [],
        "actions": [],
        "observations": [],
        "free_energy": [],
        "execution_times": [],
        "num_timesteps": None,
        "validation": {},
        "model_parameters": {},
        "data_source": None,
    }
    execution_root = execution_output_dir.resolve()

    def _resolve_current_output_path(path_value: Any) -> Path | None:
        """Resolve an execution artifact path, rejecting paths outside this run."""
        if not path_value:
            return None
        raw_path = Path(str(path_value))
        candidates = [raw_path]
        if not raw_path.is_absolute():
            candidates.extend(
                [
                    execution_output_dir / raw_path,
                    execution_output_dir.parent / raw_path,
                ]
            )
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
                resolved.relative_to(execution_root)
            except (OSError, ValueError):
                continue
            return resolved
        logger.debug(
            "  [%s] Ignoring artifact outside current output tree: %s",
            framework,
            path_value,
        )
        return None

    def _derive_framework_output_dir(detail: Dict[str, Any]) -> Path | None:
        """Derive the current-run model/framework output directory from Step 12 details."""
        impl_dir = _resolve_current_output_path(detail.get("implementation_directory"))
        if impl_dir and impl_dir.exists():
            return impl_dir

        for path_key in ("structured_result_file", "output_file", "simulation_data"):
            artifact_path = _resolve_current_output_path(detail.get(path_key))
            if not artifact_path:
                continue
            candidate = (
                artifact_path if artifact_path.is_dir() else artifact_path.parent
            )
            if candidate.name == "execution_logs":
                candidate = candidate.parent
            if candidate.exists():
                return candidate

        model_name = detail.get("model_name")
        if model_name:
            model_candidate = _resolve_current_output_path(
                execution_output_dir
                / str(model_name)
                / framework.lower().replace(".", "_").replace(" ", "_")
            )
            if model_candidate is not None and model_candidate.exists():
                return model_candidate
        return None

    for detail in details:
        framework_key = framework.lower().replace(".", "_").replace(" ", "_")
        # Collect execution time from the detail record
        exec_time = detail.get("execution_time", 0)
        if exec_time:
            metrics["execution_times"].append(exec_time)
        if detail.get("success") is False or detail.get("skipped") is True:
            logger.debug("  [%s] Skipping non-successful execution detail", framework)
            continue

        impl_dir = _derive_framework_output_dir(detail)
        if not impl_dir or not impl_dir.exists():
            logger.debug(
                "  [%s] implementation directory missing or outside current output tree",
                framework,
            )
            continue

        # Search for simulation data files. PyMDP is schema-locked to
        # pymdp_simulation_v1; other frameworks still have heterogeneous outputs.
        candidate_files: List[Path] = []

        def _add_unique(paths: List[Path], p: Path) -> None:
            """Handle add unique for internal callers."""
            if p not in paths:
                paths.append(p)

        if framework_key == "pymdp":
            sr = detail.get("structured_result_file")
            if sr:
                sp = _resolve_current_output_path(sr)
                if sp and sp.is_file():
                    try:
                        with open(sp, "r", encoding="utf-8") as handle:
                            structured = json.load(handle)
                        for collected_path in structured.get(
                            "collected_outputs", {}
                        ).get("simulation_data", []):
                            cp = _resolve_current_output_path(collected_path)
                            if cp and cp.is_file():
                                _add_unique(candidate_files, cp)
                    except (json.JSONDecodeError, OSError) as exc:
                        logger.debug(
                            "  [pymdp] Unable to inspect structured result %s: %s",
                            sp,
                            exc,
                        )

            sim_dirs: list[Any] = [impl_dir / "simulation_data"]
            if impl_dir.name == "simulation_data":
                sim_dirs.append(impl_dir)
            for sim_dir in sim_dirs:
                if not sim_dir.exists():
                    continue
                for named_result in sorted(sim_dir.glob("*_simulation_results.json")):
                    _add_unique(candidate_files, named_result)
                bare_result = sim_dir / "simulation_results.json"
                if bare_result.exists():
                    _add_unique(candidate_files, bare_result)
            direct_result = impl_dir / "simulation_results.json"
            if direct_result.exists():
                _add_unique(candidate_files, direct_result)
        else:
            for p in (
                impl_dir / "simulation_data" / "simulation_results.json",
                impl_dir / "simulation_results.json",
                impl_dir / "results.json",
                impl_dir / "output.json",
                impl_dir / "traces.json",
            ):
                _add_unique(candidate_files, p)

            try:
                for found_file in impl_dir.rglob("simulation_results.json"):
                    _add_unique(candidate_files, found_file)
            except Exception as e:
                logger.debug(f"Error during recursive search in {impl_dir}: {e}")

            sr = detail.get("structured_result_file")
            if sr:
                sp = _resolve_current_output_path(sr)
                if sp and sp.is_file():
                    _add_unique(candidate_files, sp)
            exec_logs = impl_dir / "execution_logs"
            if exec_logs.is_dir():
                try:
                    for rj in sorted(exec_logs.glob("*_results.json")):
                        _add_unique(candidate_files, rj)
                except OSError as e:
                    logger.debug(f"  [{framework}] execution_logs glob failed: {e}")

        for output_file in candidate_files:
            resolved_file = _resolve_current_output_path(output_file)
            if resolved_file is None:
                continue
            output_file = resolved_file
            if output_file.is_file():
                try:
                    with open(output_file, "r") as f:
                        data = json.load(f)

                    if framework_key == "pymdp":
                        payload = data
                        nested_payload = data.get("simulation_data")
                        if payload.get("schema_version") != "pymdp_simulation_v1":
                            payload = (
                                nested_payload
                                if isinstance(nested_payload, dict)
                                else {}
                            )
                        if payload.get("schema_version") != "pymdp_simulation_v1":
                            logger.debug(
                                "  [pymdp] Skipping non-v1 result: %s", output_file
                            )
                            continue

                        metrics["data_source"] = str(output_file)
                        metrics["_scientific_record"] = prepare_scientific_record(
                            payload, detail
                        )
                        logger.info(
                            f"  [{framework}] Loaded pymdp_simulation_v1 data from: {output_file}"
                        )
                        beliefs = (payload.get("beliefs_by_factor", {}) or {}).get(
                            "joint_state", []
                        )
                        actions = (
                            payload.get("actions_by_control_factor", {}) or {}
                        ).get("joint_action", [])
                        observations = (
                            payload.get("observations_by_modality", {}) or {}
                        ).get("joint_observation", [])
                        free_energy = payload.get("expected_free_energy", [])
                        if not free_energy:
                            free_energy = payload.get("variational_free_energy", [])

                        metrics["beliefs"] = beliefs
                        metrics["actions"] = actions
                        metrics["observations"] = observations
                        metrics["free_energy"] = free_energy
                        metrics["num_timesteps"] = len(beliefs) if beliefs else None
                        metrics["validation"] = payload.get("validation", {})
                        metrics["model_parameters"] = payload.get(
                            "model_parameters", {}
                        )
                        confidence = (payload.get("metrics", {}) or {}).get(
                            "belief_confidence", []
                        )
                        if confidence:
                            metrics["belief_confidence"] = confidence
                        break

                    metrics["data_source"] = str(output_file)
                    metrics["_scientific_record"] = prepare_scientific_record(
                        data, detail
                    )
                    logger.info(
                        f"  [{framework}] Loaded simulation data from: {output_file}"
                    )

                    # Extract beliefs - top-level, simulation_trace, or Step 12 structured simulation_data
                    beliefs = data.get("beliefs", [])
                    if not beliefs:
                        beliefs = data.get("simulation_trace", {}).get("beliefs", [])
                    if not beliefs:
                        beliefs = data.get("simulation_data", {}).get("beliefs", [])
                    if beliefs:
                        metrics["beliefs"] = beliefs

                    # Extract actions
                    actions = data.get("actions", [])
                    if not actions:
                        actions = data.get("simulation_trace", {}).get("actions", [])
                    if not actions:
                        actions = data.get("simulation_data", {}).get("actions", [])
                    if actions:
                        metrics["actions"] = actions

                    # Extract observations
                    observations = data.get("observations", [])
                    if not observations:
                        observations = data.get("simulation_trace", {}).get(
                            "observations", []
                        )
                    if not observations:
                        observations = data.get("simulation_data", {}).get(
                            "observations", []
                        )
                    if observations:
                        metrics["observations"] = observations

                    # Extract free energy - multiple possible key locations
                    free_energy = data.get("free_energy", [])
                    if not free_energy:
                        free_energy = data.get("efe_history", [])
                    if not free_energy:
                        free_energy = data.get("metrics", {}).get(
                            "expected_free_energy", []
                        )
                    if not free_energy:
                        free_energy = data.get("simulation_trace", {}).get(
                            "efe_history", []
                        )
                    if not free_energy:
                        free_energy = data.get("simulation_data", {}).get(
                            "free_energy", []
                        )
                    if free_energy:
                        metrics["free_energy"] = free_energy

                    # Extract additional metadata
                    metrics["num_timesteps"] = data.get(
                        "num_timesteps",
                        data.get("time_steps", len(beliefs) if beliefs else None),
                    )
                    metrics["validation"] = data.get("validation", {})
                    metrics["model_parameters"] = data.get("model_parameters", {})
                    if (
                        framework == "bnlearn"
                        and data.get("success") is not None
                        and not metrics["model_parameters"]
                    ):
                        metrics["model_parameters"] = {
                            "model_name": data.get("model_name"),
                            "bnlearn_completed": bool(data.get("success")),
                        }

                    if framework == "discopy":
                        _discopy_inline_circuit_from_structured(data, metrics)

                    # Extract belief confidence if available
                    confidence = data.get("metrics", {}).get("belief_confidence", [])
                    if not confidence:
                        confidence = data.get("simulation_trace", {}).get(
                            "belief_confidence", []
                        )
                    if confidence:
                        metrics["belief_confidence"] = confidence

                    break  # Found data, stop searching
                except Exception as e:
                    logger.debug(f"  [{framework}] Error reading {output_file}: {e}")
                    continue

        # Recovery: CSV simulation data (ActiveInference.jl outputs CSV, not JSON)
        if framework_key != "pymdp" and not metrics["data_source"]:
            csv_candidates: list[Any] = [
                impl_dir / "simulation_data" / "simulation_results.csv",
            ]
            # Also search for timestamped CSV files
            sim_data_dir = impl_dir / "simulation_data"
            if sim_data_dir.exists():
                try:
                    for csv_file in sorted(
                        sim_data_dir.glob("*_simulation_results.csv")
                    ):
                        if csv_file not in csv_candidates:
                            csv_candidates.append(csv_file)
                except Exception as e:
                    logger.debug(
                        f"Error searching for timestamped CSV files in {sim_data_dir}: {e}"
                    )

            for csv_file in csv_candidates:
                resolved_csv = _resolve_current_output_path(csv_file)
                if resolved_csv is None:
                    continue
                csv_file = resolved_csv
                if csv_file.is_file():
                    try:
                        import csv as csv_module

                        beliefs = []
                        actions = []
                        observations = []

                        with open(csv_file, "r") as f:
                            # Skip comment lines (lines starting with #)
                            lines = [line for line in f if not line.startswith("#")]

                        if lines:
                            reader = csv_module.reader(lines)
                            for row in reader:
                                if len(row) >= 3:
                                    try:
                                        observations.append(int(float(row[1])))
                                        actions.append(int(float(row[2])))
                                        # Beliefs are remaining columns
                                        if len(row) > 3:
                                            belief = [float(x) for x in row[3:]]
                                            beliefs.append(belief)
                                    except ValueError as e:
                                        logger.debug(
                                            "Skipping non-numeric CSV row: %s", e
                                        )
                                        continue

                        if beliefs or actions or observations:
                            metrics["data_source"] = str(csv_file)
                            metrics["_scientific_record"] = prepare_scientific_record(
                                {
                                    "beliefs": beliefs,
                                    "actions": actions,
                                    "observations": observations,
                                },
                                detail,
                            )
                            logger.info(
                                f"  [{framework}] Loaded CSV simulation data from: {csv_file}"
                            )
                            if beliefs:
                                metrics["beliefs"] = beliefs
                            if actions:
                                metrics["actions"] = actions
                            if observations:
                                metrics["observations"] = observations
                            metrics["num_timesteps"] = (
                                len(beliefs) if beliefs else len(actions)
                            )

                            # Infer validation from data
                            if beliefs:
                                sums_valid = all(
                                    abs(sum(b) - 1.0) < 0.01 for b in beliefs
                                )
                                metrics["validation"] = {
                                    "all_beliefs_valid": True,
                                    "beliefs_sum_to_one": sums_valid,
                                    "actions_in_range": True,
                                }
                            break
                    except Exception as e:
                        logger.debug(
                            f"  [{framework}] Error reading CSV {csv_file}: {e}"
                        )
                        continue

        # Supplement: model_parameters.json (separate from simulation data)
        if metrics["data_source"] and not metrics["model_parameters"]:
            params_candidates: list[Any] = [
                impl_dir / "simulation_data" / "model_parameters.json",
            ]
            # Also search for timestamped parameter files
            params_dir = impl_dir / "simulation_data"
            if params_dir.exists():
                try:
                    for pf in sorted(params_dir.glob("*_model_parameters.json")):
                        if pf not in params_candidates:
                            params_candidates.append(pf)
                except Exception as e:
                    logger.debug(
                        f"Error searching for model parameter files in {params_dir}: {e}"
                    )

            for params_file in params_candidates:
                resolved_params = _resolve_current_output_path(params_file)
                if resolved_params is None:
                    continue
                params_file = resolved_params
                if params_file.is_file():
                    try:
                        with open(params_file, "r") as f:
                            params_data = json.load(f)
                        metrics["model_parameters"] = {
                            "num_states": params_data.get(
                                "n_states", params_data.get("num_states", 0)
                            ),
                            "num_observations": params_data.get(
                                "n_observations", params_data.get("num_observations", 0)
                            ),
                            "num_actions": params_data.get(
                                "n_actions", params_data.get("num_actions", 0)
                            ),
                        }
                        logger.info(
                            f"  [{framework}] Loaded model parameters from: {params_file}"
                        )
                        break
                    except Exception as e:
                        logger.debug(
                            f"  [{framework}] Error reading params {params_file}: {e}"
                        )

        # Supplement: circuit_info.json for DisCoPy (after execution log may lack vectors)
        if not metrics.get("circuit_info"):
            circuit_candidates: list[Any] = [
                impl_dir / "simulation_data" / "circuit_info.json",
            ]
            sim_data_dir = impl_dir / "simulation_data"
            if sim_data_dir.exists():
                try:
                    for cf in sorted(sim_data_dir.glob("*_circuit_info.json")):
                        if cf not in circuit_candidates:
                            circuit_candidates.append(cf)
                except Exception as e:
                    logger.debug(
                        f"Error searching for circuit info files in {sim_data_dir}: {e}"
                    )

            for circuit_file in circuit_candidates:
                resolved_circuit = _resolve_current_output_path(circuit_file)
                if resolved_circuit is None:
                    continue
                circuit_file = resolved_circuit
                if circuit_file.is_file():
                    try:
                        with open(circuit_file, "r") as f:
                            circuit_data = json.load(f)
                        if not metrics.get("data_source"):
                            metrics["data_source"] = str(circuit_file)
                        metrics["circuit_info"] = {
                            "model_name": circuit_data.get("model_name", ""),
                            "components": circuit_data.get("components", []),
                            "num_components": circuit_data.get("analysis", {}).get(
                                "num_components",
                                len(circuit_data.get("components", [])),
                            ),
                            "parameters": circuit_data.get("parameters", {}),
                        }
                        # Map circuit parameters to standardized model_parameters
                        params = circuit_data.get("parameters", {})
                        if params and not metrics.get("model_parameters"):
                            metrics["model_parameters"] = {
                                "num_states": params.get("num_states", 0),
                                "num_observations": params.get("num_observations", 0),
                                "num_actions": params.get("num_actions", 0),
                            }
                        logger.info(
                            f"  [{framework}] Loaded circuit info from: {circuit_file}"
                        )
                        break
                    except Exception as e:
                        logger.debug(
                            f"  [{framework}] Error reading circuit {circuit_file}: {e}"
                        )

    # Log summary of what was extracted
    data_found: list[Any] = []
    if metrics["beliefs"]:
        data_found.append(f"beliefs({len(metrics['beliefs'])} steps)")
    if metrics["actions"]:
        data_found.append(f"actions({len(metrics['actions'])})")
    if metrics["observations"]:
        data_found.append(f"observations({len(metrics['observations'])})")
    if metrics["free_energy"]:
        data_found.append(f"free_energy({len(metrics['free_energy'])})")
    if metrics.get("circuit_info"):
        data_found.append(
            f"circuit({metrics['circuit_info'].get('num_components', 0)} components)"
        )
    if metrics["model_parameters"]:
        data_found.append("model_parameters")

    if data_found:
        logger.info(f"  [{framework}] Extracted: {', '.join(data_found)}")
    else:
        if (
            framework == "bnlearn"
            and details
            and all(d.get("skipped", False) for d in details)
        ):
            logger.info(f"  [{framework}] No simulation data (all runs skipped)")
        else:
            logger.warning(f"  [{framework}] No simulation data found")

    return metrics


def _compare_framework_results(
    framework_data: Dict[str, Dict[str, Any]], logger: logging.Logger
) -> Dict[str, Any]:
    """Compare results across frameworks using real simulation data."""
    comparisons: dict[str, Any] = {
        "success_rates": {},
        "performance_comparison": {},
        "metric_agreement": {},
        "data_coverage": {},
        "simulation_statistics": {},
    }

    # Compare success rates
    for framework, data in framework_data.items():
        success_count = data.get("success_count", 0)
        total_count = data.get("total_count", 0)
        if total_count > 0:
            comparisons["success_rates"][framework] = success_count / total_count

    # Compare execution times
    for framework, data in framework_data.items():
        valid_times, invalid_times = _valid_execution_times(
            data.get("execution_times", [])
        )
        if invalid_times or data.get("invalid_execution_time_count"):
            comparisons.setdefault("unavailable_metrics", {}).setdefault(framework, {})[
                "execution_time"
            ] = {
                "reason": "Invalid timing entries were excluded",
                "invalid_count": invalid_times
                + data.get("invalid_execution_time_count", 0),
            }
        if valid_times:
            comparisons["performance_comparison"][framework] = {
                "mean": float(np.mean(valid_times)),
                "std": float(np.std(valid_times)) if len(valid_times) > 1 else 0.0,
                "min": float(np.min(valid_times)),
                "max": float(np.max(valid_times)),
            }

    # Compare data coverage — which frameworks produced which data
    for framework, data in framework_data.items():
        coverage: dict[str, Any] = {
            "status": data.get("status"),
            "has_beliefs": bool(data.get("beliefs")),
            "has_actions": bool(data.get("actions")),
            "has_observations": bool(data.get("observations")),
            "has_free_energy": bool(data.get("free_energy")),
            "has_circuit_info": bool(data.get("circuit_info")),
            "num_timesteps": data.get("num_timesteps"),
            "validation_passed": all(data.get("validation", {}).values())
            if data.get("validation")
            else None,
            "data_source": data.get("data_source"),
        }
        comparisons["data_coverage"][framework] = coverage

    records = {
        framework: data["scientific_records"]
        if "scientific_records" in data
        else [prepare_scientific_record(data)]
        for framework, data in framework_data.items()
    }
    comparisons.update(scientific_comparisons(records))

    return comparisons


def _calculate_aggregate_metrics(
    framework_data: Dict[str, Dict[str, Any]], logger: logging.Logger
) -> Dict[str, Any]:
    """Calculate aggregate metrics across all frameworks."""
    metrics: dict[str, Any] = {
        "total_frameworks": len(framework_data),
        "total_successful": sum(
            data.get("success_count", 0) for data in framework_data.values()
        ),
        "total_skipped": sum(
            data.get("skipped_count", 0) for data in framework_data.values()
        ),
        "total_failed": sum(
            data.get("failed_count", 0) for data in framework_data.values()
        ),
        "total_executions": sum(
            data.get("total_count", 0) for data in framework_data.values()
        ),
        "overall_success_rate": 0.0,
    }

    if metrics["total_executions"] > 0:
        metrics["overall_success_rate"] = (
            metrics["total_successful"] / metrics["total_executions"]
        )

    return metrics


def generate_framework_comparison_report(
    comparison_data: Dict[str, Any],
    output_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Generate a comprehensive comparison report across frameworks,
    grounded in real simulation data.

    Args:
        comparison_data: Output from analyze_framework_outputs()
        output_dir: Directory to save report
        logger: Optional logger instance

    Returns:
        Path to generated report file
    """
    import json
    import logging

    if logger is None:
        logger = logging.getLogger(__name__)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate Markdown report
    report_lines: list[Any] = [
        "# Framework Execution Comparison Report",
        "",
        f"Generated: {comparison_data.get('timestamp', 'Unknown')}",
        "",
        "## Summary",
        "",
    ]

    metrics = comparison_data.get("metrics", {})
    report_lines.extend(
        [
            f"- Total Frameworks: {metrics.get('total_frameworks', 0)}",
            f"- Total Executions: {metrics.get('total_executions', 0)}",
            f"- Successful Executions: {metrics.get('total_successful', 0)}",
            f"- Skipped Executions: {metrics.get('total_skipped', 0)}",
            f"- Failed Executions: {metrics.get('total_failed', 0)}",
            f"- Overall Success Rate: {metrics.get('overall_success_rate', 0.0):.2%}",
            "",
        ]
    )

    # Framework details with simulation data
    report_lines.extend(["## Framework Details", ""])

    for framework, data in comparison_data.get("frameworks", {}).items():
        success_count = data.get("success_count", 0)
        total_count = data.get("total_count", 0)
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0
        status = data.get("status") or "no_data"

        report_lines.extend(
            [
                f"### {framework.upper()}",
                "",
                f"- Status: {status}",
                f"- Success Rate: {success_rate:.1f}% ({success_count}/{total_count})",
            ]
        )
        status_reason = data.get("status_reason")
        if status_reason:
            report_lines.append(f"- Status Reason: {status_reason}")

        # Add execution time if available
        times, _ = _valid_execution_times(data.get("execution_times", []))
        if times:
            report_lines.append(f"- Execution Time: {times[0]:.2f}s")

        # Add simulation data coverage
        n_beliefs = len(data.get("beliefs", []))
        n_actions = len(data.get("actions", []))
        n_obs = len(data.get("observations", []))
        n_efe = len(data.get("free_energy", []))
        timesteps = data.get("num_timesteps")

        if any([n_beliefs, n_actions, n_obs, n_efe]):
            report_lines.append(f"- Timesteps: {timesteps}")
            report_lines.append(
                f"- Data: beliefs={n_beliefs}, actions={n_actions}, observations={n_obs}, free_energy={n_efe}"
            )

        # Add validation status
        validation = data.get("validation", {})
        if validation:
            passed = all(validation.values())
            checks = ", ".join(
                f"{k}={'✅' if v else '❌'}" for k, v in validation.items()
            )
            report_lines.append(
                f"- Validation: {'✅ ALL PASSED' if passed else '⚠️ ISSUES'} ({checks})"
            )

        # Add data source
        source = data.get("data_source")
        if source:
            report_lines.append(f"- Data Source: `{source}`")

        report_lines.append("")

    # Scientific statistics retain model and native view identity.
    comparisons = comparison_data.get("comparisons", {})
    sim_stats = comparisons.get("simulation_statistics", {})
    if sim_stats:
        report_lines.extend(
            [
                "## Validated Simulation Statistics",
                "",
                "| Framework | Model ID | View | Kind | Timesteps | Categorical Confidence | Posterior Std |",
                "|-----------|----------|------|------|-----------|------------------------|---------------|",
            ]
        )
        for framework, summary in sim_stats.items():
            rows = summary.get("models", {"single": summary})
            for stats in rows.values():
                views = stats.get("views", {"joint": stats})
                for name, view in views.items():
                    dims = view.get("belief_dims", [])
                    steps = view.get("num_timesteps", dims[0] if dims else "N/A")
                    confidence = (
                        f"{view['mean_confidence']:.4f}"
                        if "mean_confidence" in view
                        else "N/A"
                    )
                    std = (
                        f"{view['mean_posterior_std']:.4f}"
                        if "mean_posterior_std" in view
                        else "N/A"
                    )
                    report_lines.append(
                        f"| {framework} | {stats.get('model_id') or 'unbound'} | {name} | {stats.get('model_kind', 'unknown')} | {steps} | {confidence} | {std} |"
                    )
        report_lines.append("")
    exclusions = comparisons.get("scientific_exclusions", {})
    if exclusions:
        report_lines.extend(["## Unavailable Scientific Comparisons", ""])
        for label, reasons in exclusions.items():
            report_lines.append(f"- **{label}**: {'; '.join(reasons)}")
        report_lines.append("")

    # Data Coverage
    data_coverage = comparisons.get("data_coverage", {})
    if data_coverage:
        report_lines.extend(
            [
                "## Data Coverage",
                "",
                "| Framework | Beliefs | Actions | Observations | Free Energy | Validation |",
                "|-----------|---------|---------|--------------|-------------|------------|",
            ]
        )
        for framework, cov in data_coverage.items():
            report_lines.append(
                f"| {framework} "
                f"| {'✅' if cov.get('has_beliefs') else '❌'} "
                f"| {'✅' if cov.get('has_actions') else '❌'} "
                f"| {'✅' if cov.get('has_observations') else '❌'} "
                f"| {'✅' if cov.get('has_free_energy') else '❌'} "
                f"| {'✅' if cov.get('validation_passed') else ('❌' if cov.get('validation_passed') is False else '—')} |"
            )
        report_lines.append("")

    # Metric Agreement
    metric_agreement = comparisons.get("metric_agreement", {})
    if metric_agreement:
        report_lines.extend(["## Cross-Framework Metric Agreement", ""])
        for pair, agreement in metric_agreement.items():
            if agreement.get("status") != "comparable":
                report_lines.append(
                    f"- **{pair}**: unavailable — {'; '.join(agreement.get('reasons', []))}"
                )
                continue
            report_lines.append(
                f"- **{pair}**: model `{agreement['model_id']}`, source `{agreement['source_sha256']}`, inference `{agreement['inference_mode']}`, tolerances `{agreement['tolerance']}`"
            )
            for name, witness in agreement.get("views", {}).items():
                report_lines.append(
                    f"  - View `{name}`: maximum posterior difference {witness['max_absolute_difference']:.6g}; within tolerance: {witness['within_tolerance']}"
                )
                if "covariance_max_absolute_difference" in witness:
                    report_lines.append(
                        f"  - Covariance difference {witness['covariance_max_absolute_difference']:.6g}; within tolerance: {witness['covariance_within_tolerance']}"
                    )
        report_lines.append("")

    # Performance comparison
    if comparisons.get("performance_comparison"):
        report_lines.extend(
            [
                "## Operational Timing Aggregates",
                "",
                "Descriptive timings may cover different model sets and environments; they do not establish relative performance.",
                "",
                "| Framework | Mean Time (s) | Std Dev | Min | Max |",
                "|-----------|---------------|---------|-----|-----|",
            ]
        )

        for framework, perf in comparisons["performance_comparison"].items():
            report_lines.append(
                f"| {framework} | {perf['mean']:.3f} | {perf['std']:.3f} | {perf['min']:.3f} | {perf['max']:.3f} |"
            )
        report_lines.append("")

    # Save report to cross_framework subfolder
    cross_fw_dir = output_dir / "cross_framework"
    cross_fw_dir.mkdir(parents=True, exist_ok=True)
    report_file = cross_fw_dir / "framework_comparison_report.md"
    with open(report_file, "w") as f:
        f.write("\n".join(report_lines))

    # Also save JSON version
    json_file = cross_fw_dir / "framework_comparison_data.json"
    with open(json_file, "w") as f:
        json.dump(comparison_data, f, indent=2, default=str)

    logger.info(f"Generated framework comparison report: {report_file}")

    return str(report_file)

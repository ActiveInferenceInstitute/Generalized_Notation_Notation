"""
Framework-specific data extractors for post-simulation analysis.

Provides extract_*_data() functions for PyMDP, RxInfer.jl, ActiveInference.jl,
JAX, DisCoPy, PyTorch, and NumPyro execution results. PyTorch and NumPyro are
schema-aware (``*_simulation_v1`` dispatch) but ungated: heterogeneous payloads
are accepted as-is. bnlearn uses its source-bound fixed-action filtering contract; metadata-only
smoke demonstrations receive no posterior credit. Stan retains its native schema.

Extracted from post_simulation.py for maintainability.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, cast

logger = logging.getLogger(__name__)

CURRENT_SIMULATION_SCHEMAS = {
    "pytorch": "pytorch_simulation_v1",
    "numpyro": "numpyro_simulation_v1",
    "pymdp": "pymdp_simulation_v1",
    "rxinfer": "rxinfer_simulation_v1",
    "activeinference_jl": "activeinference_jl_simulation_v1",
}


def _normalise_current_simulation_payload(
    payload: Dict[str, Any], *, fallback_top_level: bool = True
) -> Dict[str, Any]:
    """Map current framework simulation schemas into analysis fields.

    ``fallback_top_level`` controls whether a missing ``*_by_factor`` /
    ``by_modality`` map falls back to the payload's top-level arrays. The
    rxinfer and activeinference_jl schemas carry top-level arrays (default);
    the pymdp schema stores data exclusively in the by-factor maps, so its
    extractor disables the fallback — pinned by
    ``tests/render/test_jax_factorized_pipeline.py``.
    """

    def _fallback(top_key: str) -> Any:
        return payload.get(top_key, []) if fallback_top_level else []

    beliefs_by_factor = payload.get("beliefs_by_factor", {}) or {}
    observations_by_modality = payload.get("observations_by_modality", {}) or {}
    actions_by_control_factor = payload.get("actions_by_control_factor", {}) or {}
    hidden_states_by_factor = payload.get("hidden_states_by_factor", {}) or {}
    metrics = payload.get("metrics", {}) or {}
    return {
        "traces": payload.get("simulation_trace", {}),
        "free_energy": payload.get("expected_free_energy", []),
        "states": hidden_states_by_factor.get("joint_state", _fallback("true_states")),
        "observations": observations_by_modality.get(
            "joint_observation", _fallback("observations")
        ),
        "actions": actions_by_control_factor.get("joint_action", _fallback("actions")),
        "policy": payload.get("policy_posterior", []),
        "beliefs": beliefs_by_factor.get("joint_state", _fallback("beliefs")),
        "belief_confidence": metrics.get("belief_confidence", []),
        "action_probabilities": payload.get(
            "policy_posterior", payload.get("action_probabilities", [])
        ),
        "validation": payload.get("validation", {}),
        "model_parameters": payload.get("model_parameters", {}),
        "schema_version": payload.get("schema_version"),
    }


def _load_current_schema_from_impl_dir(
    implementation_dir: Any, expected_schema: str
) -> Dict[str, Any] | None:
    """Load current schema from impl dir."""
    if not implementation_dir:
        return None
    impl_path = Path(str(implementation_dir))
    candidate_paths: list[Path] = []
    sim_data_dir = impl_path / "simulation_data"
    if sim_data_dir.exists():
        candidate_paths.extend(sorted(sim_data_dir.glob("*simulation_results.json")))
    candidate_paths.append(impl_path / "simulation_results.json")
    candidate_paths.extend(sorted(impl_path.rglob("simulation_results.json")))
    for path in candidate_paths:
        if not path.exists() or not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("schema_version") == expected_schema:
            logger.info("Reading current simulation data from %s", path.name)
            return cast(Dict[str, Any], payload)
    return None


def extract_pymdp_data(execution_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract PyMDP-specific data from execution result.

    Args:
        execution_result: Execution result dictionary

    Returns:
        Extracted simulation data
    """
    simulation_data: Dict[str, Any] = {}
    payload = (
        execution_result
        if execution_result.get("schema_version") == "pymdp_simulation_v1"
        else None
    )
    nested_payload = execution_result.get("simulation_data")
    if (
        payload is None
        and isinstance(nested_payload, dict)
        and nested_payload.get("schema_version") == "pymdp_simulation_v1"
    ):
        payload = nested_payload

    implementation_dir = execution_result.get("implementation_directory")
    if payload is None and implementation_dir:
        try:
            impl_path = Path(implementation_dir)
            sim_data_dir = impl_path / "simulation_data"
            if sim_data_dir.exists():
                results_files = list(sim_data_dir.glob("*simulation_results.json"))
                if results_files:
                    logger.info(
                        f"Reading PyMDP simulation data from {results_files[0].name}"
                    )
                    with open(results_files[0]) as f:
                        payload = json.load(f)

            # Count visualizations
            viz_dir = impl_path / "visualizations"
            if viz_dir.exists():
                viz_files = list(viz_dir.glob("*.png")) + list(viz_dir.glob("*.svg"))
                if viz_files:
                    simulation_data["visualization_count"] = len(viz_files)
                    simulation_data["visualization_files"] = [
                        str(f.name) for f in viz_files
                    ]

        except Exception as e:
            logger.warning(f"Error reading PyMDP files: {e}")

    if payload is None:
        simulation_data["extraction_error"] = "No pymdp_simulation_v1 payload found"
    elif payload.get("schema_version") != "pymdp_simulation_v1":
        simulation_data["extraction_error"] = (
            f"Unsupported PyMDP schema: {payload.get('schema_version')!r}"
        )
    else:
        # Single canonical field mapping (same normalizer the rxinfer and
        # activeinference_jl extractors use) so the three extractors cannot
        # drift. The pymdp schema carries data only in the by-factor maps,
        # so the normalizer's top-level fallback is disabled here.
        normalised = _normalise_current_simulation_payload(
            payload, fallback_top_level=False
        )
        simulation_data["beliefs"] = normalised["beliefs"]
        simulation_data["observations"] = normalised["observations"]
        simulation_data["actions"] = normalised["actions"]
        simulation_data["states"] = normalised["states"]
        simulation_data["free_energy"] = normalised["free_energy"]
        simulation_data["policy"] = normalised["policy"]
        simulation_data["belief_confidence"] = normalised["belief_confidence"]
        simulation_data["traces"] = normalised["traces"]

    result: dict[str, Any] = {
        "traces": simulation_data.get("traces", []),
        "free_energy": simulation_data.get("free_energy", []),
        "states": simulation_data.get("states", []),
        "observations": simulation_data.get("observations", []),
        "actions": simulation_data.get("actions", []),
        "policy": simulation_data.get("policy", []),
        "beliefs": simulation_data.get("beliefs", []),
        "belief_confidence": simulation_data.get("belief_confidence", []),
        "visualization_count": simulation_data.get("visualization_count", 0),
        "visualization_files": simulation_data.get("visualization_files", []),
    }
    if "extraction_error" in simulation_data:
        result["extraction_error"] = simulation_data["extraction_error"]
    return result


def extract_rxinfer_data(execution_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract RxInfer.jl-specific data from execution result.
    Enhanced to read from collected files if available.
    RxInfer stores data at top-level: beliefs, actions, observations,
    efe_history, true_states, action_probabilities, validation.
    """
    simulation_data = execution_result.get("simulation_data", {})
    payload = (
        execution_result
        if execution_result.get("schema_version")
        == CURRENT_SIMULATION_SCHEMAS["rxinfer"]
        else None
    )
    if payload is None and isinstance(simulation_data, dict):
        if (
            simulation_data.get("schema_version")
            == CURRENT_SIMULATION_SCHEMAS["rxinfer"]
        ):
            payload = simulation_data
    if payload is None:
        payload = _load_current_schema_from_impl_dir(
            execution_result.get("implementation_directory"),
            CURRENT_SIMULATION_SCHEMAS["rxinfer"],
        )
    if payload is not None:
        return _normalise_current_simulation_payload(payload)

    # Try to read from collected files if available
    implementation_dir = execution_result.get("implementation_directory")
    if implementation_dir:
        try:
            impl_path = Path(implementation_dir)
            # Check simulation_data subdirectory first, then root
            sim_data_dir = impl_path / "simulation_data"
            results_file = None
            if sim_data_dir.exists():
                results_files = list(sim_data_dir.glob("*simulation_results.json"))
                if results_files:
                    results_file = results_files[0]
            if not results_file:
                rf = impl_path / "simulation_results.json"
                if rf.exists():
                    results_file = rf

            if results_file:
                logger.info(f"Reading RxInfer simulation data from {results_file.name}")
                with open(results_file) as f:
                    file_data = json.load(f)
                    if "beliefs" in file_data:
                        simulation_data["beliefs"] = file_data["beliefs"]
                    if "true_states" in file_data:
                        simulation_data["true_states"] = file_data["true_states"]
                    if "observations" in file_data:
                        simulation_data["observations"] = file_data["observations"]
                    if "actions" in file_data:
                        simulation_data["actions"] = file_data["actions"]
                    # Extract EFE from efe_history (top-level in RxInfer output)
                    if "efe_history" in file_data:
                        simulation_data["free_energy"] = file_data["efe_history"]
                        logger.debug(
                            f"Extracted RxInfer efe_history with {len(file_data['efe_history'])} entries"
                        )
                    if "action_probabilities" in file_data:
                        simulation_data["action_probabilities"] = file_data[
                            "action_probabilities"
                        ]
        except (OSError, ValueError) as e:
            logger.warning(f"Error reading RxInfer files: {e}")
            simulation_data["_file_read_error"] = str(e)

    result: dict[str, Any] = {
        "beliefs": simulation_data.get("beliefs", []),
        "true_states": simulation_data.get("true_states", []),
        "observations": simulation_data.get("observations", []),
        "actions": simulation_data.get("actions", []),
        "free_energy": simulation_data.get("free_energy", []),
        "action_probabilities": simulation_data.get("action_probabilities", []),
        "posterior": simulation_data.get("posterior", []),
        "inference_data": simulation_data.get("inference_data", []),
    }
    if "_file_read_error" in simulation_data:
        result["extraction_error"] = simulation_data["_file_read_error"]
    return result


def extract_activeinference_jl_data(execution_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract ActiveInference.jl-specific data from execution result.
    Enhanced to read from simulation_results.csv if available.

    Args:
        execution_result: Execution result dictionary

    Returns:
        Extracted simulation data with full Active Inference fields
    """
    simulation_data = execution_result.get("simulation_data", {})
    payload = (
        execution_result
        if execution_result.get("schema_version")
        == CURRENT_SIMULATION_SCHEMAS["activeinference_jl"]
        else None
    )
    if payload is None and isinstance(simulation_data, dict):
        if (
            simulation_data.get("schema_version")
            == CURRENT_SIMULATION_SCHEMAS["activeinference_jl"]
        ):
            payload = simulation_data
    if payload is None:
        payload = _load_current_schema_from_impl_dir(
            execution_result.get("implementation_directory"),
            CURRENT_SIMULATION_SCHEMAS["activeinference_jl"],
        )
    if payload is not None:
        return _normalise_current_simulation_payload(payload)

    model_parameters = simulation_data.get("model_parameters", {})

    # Try to read from collected files if available
    implementation_dir = execution_result.get("implementation_directory")
    if implementation_dir:
        try:
            impl_path = Path(implementation_dir)

            # Look for ActiveInference.jl output directories (timestamped)
            # or directly in the implementation directory (if flattened)
            possible_dirs = [impl_path] + list(
                impl_path.glob("activeinference_outputs_*")
            )

            csv_found = False
            for search_dir in possible_dirs:
                results_file = search_dir / "simulation_results.csv"
                if results_file.exists():
                    logger.info(
                        f"Reading ActiveInference.jl simulation data from {results_file.name}"
                    )
                    import csv

                    traces: list[Any] = []
                    observations: list[Any] = []
                    actions: list[Any] = []
                    beliefs: list[Any] = []

                    with open(results_file) as f:
                        # Skip comments
                        lines = [line for line in f if not line.startswith("#")]

                        if lines:
                            reader = csv.reader(lines)
                            for row in reader:
                                if len(row) >= 3:
                                    # step, observation, action, belief...
                                    try:
                                        # Parse basic data
                                        observations.append(float(row[1]))
                                        actions.append(float(row[2]))

                                        # Beliefs are the rest of the columns
                                        if len(row) > 3:
                                            belief = [float(x) for x in row[3:]]
                                            beliefs.append(belief)

                                        # Add to generic traces for step counting
                                        traces.append(
                                            {
                                                "step": int(float(row[0])),
                                                "observation": float(row[1]),
                                                "action": float(row[2]),
                                            }
                                        )
                                    except ValueError as e:
                                        logger.debug(
                                            "Skipping non-numeric CSV row: %s", e
                                        )
                                        continue

                    if traces:
                        simulation_data["traces"] = traces
                        simulation_data["observations"] = observations
                        simulation_data["actions"] = actions
                        simulation_data["beliefs"] = beliefs
                        simulation_data["num_timesteps"] = len(traces)
                        csv_found = True
                        logger.info(
                            f"Extracted {len(traces)} steps from ActiveInference.jl results"
                        )
                        break

            if not csv_found:
                # Also try JSON simulation_results.json (new format)
                sim_data_dir = impl_path / "simulation_data"
                if sim_data_dir.exists():
                    results_files = list(sim_data_dir.glob("*simulation_results.json"))
                    for rf in results_files:
                        try:
                            with open(rf) as f:
                                file_data = json.load(f)
                            if "beliefs" in file_data:
                                simulation_data["beliefs"] = file_data["beliefs"]
                            if "actions" in file_data:
                                simulation_data["actions"] = file_data["actions"]
                            if "observations" in file_data:
                                simulation_data["observations"] = file_data[
                                    "observations"
                                ]
                            # Extract EFE from efe_history (top-level in ActiveInference.jl output)
                            if "efe_history" in file_data:
                                simulation_data["free_energy"] = file_data[
                                    "efe_history"
                                ]
                                logger.debug(
                                    f"Extracted ActiveInference.jl efe_history with {len(file_data['efe_history'])} entries"
                                )
                            logger.info(
                                f"Read ActiveInference.jl JSON results from {rf.name}"
                            )
                            break
                        except Exception as e:
                            logger.debug(f"Failed to parse {rf.name}: {e}")
                if not simulation_data.get("beliefs"):
                    logger.debug(
                        "No simulation_results.csv or JSON found for ActiveInference.jl"
                    )

        except Exception as e:
            logger.warning(f"Error reading ActiveInference.jl files: {e}")

    # Extract all Active Inference-relevant fields
    extracted: dict[str, Any] = {
        # Core state/observation/action traces
        "traces": simulation_data.get("traces", []),
        "free_energy": simulation_data.get("free_energy", []),
        "beliefs": simulation_data.get("beliefs", []),
        "states": simulation_data.get("states", []),
        "observations": simulation_data.get("observations", []),
        "actions": simulation_data.get("actions", []),
        # Expected free energy components
        "expected_free_energy": simulation_data.get("expected_free_energy", []),
        "expected_energy": simulation_data.get("expected_energy", []),
        "expected_entropy": simulation_data.get("expected_entropy", []),
        # Model parameters (A, B, C, D matrices)
        "num_states": model_parameters.get("num_states", 0),
        "num_observations": model_parameters.get("num_observations", 0),
        "num_actions": model_parameters.get("num_actions", 0),
        "A_matrix": model_parameters.get("A", []),
        "B_matrix": model_parameters.get("B", []),
        "C_vector": model_parameters.get("C", []),
        "D_vector": model_parameters.get("D", []),
        # Precision parameters
        "action_precision": simulation_data.get("action_precision", 1.0),
        "parameter_precision": simulation_data.get("parameter_precision", 1.0),
        "policy_precision": simulation_data.get("policy_precision", 1.0),
        # Inference metadata
        "num_timesteps": simulation_data.get("num_timesteps", 0),
        "inference_iterations": simulation_data.get("inference_iterations", 0),
        "convergence_threshold": simulation_data.get("convergence_threshold", 0.0),
        # Performance metrics
        "execution_time": simulation_data.get("execution_time", 0.0),
        "memory_usage": simulation_data.get("memory_usage", 0.0),
        # Additional Active Inference specific metrics
        "variational_free_energy": simulation_data.get("variational_free_energy", []),
        "information_gain": simulation_data.get("information_gain", []),
        "pragmatic_value": simulation_data.get("pragmatic_value", []),
    }

    return extracted


KRONECKER_FACTORIZED_SCHEMA = "jax_kronecker_factorized_v1"


def extract_jax_kronecker_data(execution_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract ``jax_kronecker_factorized_v1`` results into analysis fields.

    The factorized schema (roadmap MAJ-02) stores per-factor traces
    (``beliefs_by_factor`` etc.) keyed by factor name, per-factor EFE, a
    factorised policy, and model parameters that report the (never
    materialised) joint state space size.
    """
    simulation_data = execution_result.get("simulation_data", {})
    payload = (
        execution_result
        if execution_result.get("schema_version") == KRONECKER_FACTORIZED_SCHEMA
        else None
    )
    if payload is None and isinstance(simulation_data, dict):
        if simulation_data.get("schema_version") == KRONECKER_FACTORIZED_SCHEMA:
            payload = simulation_data
    if payload is None:
        payload = _load_current_schema_from_impl_dir(
            execution_result.get("implementation_directory"),
            KRONECKER_FACTORIZED_SCHEMA,
        )
    if payload is None:
        return {"extraction_error": f"No {KRONECKER_FACTORIZED_SCHEMA} payload found"}

    beliefs_by_factor = payload.get("beliefs_by_factor", {}) or {}
    states_by_factor = payload.get("true_states_by_factor", {}) or {}
    observations_by_factor = payload.get("observations_by_factor", {}) or {}
    actions_by_factor = payload.get("actions_by_factor", {}) or {}
    efe_by_factor = payload.get("efe_per_factor", {}) or {}
    policy_by_factor = payload.get("policy_by_factor", {}) or {}
    factors = list(payload.get("factors", [])) or list(beliefs_by_factor.keys())

    # Per-step total expected free energy = sum over factors of per-factor EFE.
    total_efe: list[float] = []
    if efe_by_factor and factors:
        factor_efes = [list(efe_by_factor.get(factor, [])) for factor in factors]
        if factor_efes and all(factor_efes):
            total_efe = [sum(step) for step in zip(*factor_efes)]

    model_parameters = payload.get("model_parameters", {}) or {}
    return {
        "schema_version": payload.get("schema_version"),
        "model_kind": payload.get("model_kind"),
        "beliefs": [beliefs_by_factor.get(factor, []) for factor in factors],
        "beliefs_by_factor": beliefs_by_factor,
        "states": [states_by_factor.get(factor, []) for factor in factors],
        "true_states_by_factor": states_by_factor,
        "observations": [observations_by_factor.get(factor, []) for factor in factors],
        "observations_by_factor": observations_by_factor,
        "actions": [actions_by_factor.get(factor, []) for factor in factors],
        "actions_by_factor": actions_by_factor,
        "free_energy": total_efe,
        "efe_per_factor": efe_by_factor,
        "policy": [policy_by_factor.get(factor, []) for factor in factors],
        "policy_by_factor": policy_by_factor,
        "num_timesteps": payload.get("num_timesteps"),
        "num_factors": payload.get("num_factors"),
        "factors": factors,
        "model_parameters": model_parameters,
        "validation": payload.get("validation", {}),
    }


def extract_jax_data(execution_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract JAX results, dispatching on the result schema.

    Kronecker-factorized runs (``jax_kronecker_factorized_v1``) use the
    per-factor extractor; everything else follows the pymdp-compatible
    schema (JAX's historical result shape).
    """
    if execution_result.get("schema_version") == KRONECKER_FACTORIZED_SCHEMA:
        return extract_jax_kronecker_data(execution_result)
    simulation_data = execution_result.get("simulation_data")
    if (
        isinstance(simulation_data, dict)
        and simulation_data.get("schema_version") == KRONECKER_FACTORIZED_SCHEMA
    ):
        return extract_jax_kronecker_data(execution_result)
    if execution_result.get("implementation_directory"):
        payload = _load_current_schema_from_impl_dir(
            execution_result.get("implementation_directory"),
            KRONECKER_FACTORIZED_SCHEMA,
        )
        if payload is not None:
            return extract_jax_kronecker_data(execution_result)
    return extract_pymdp_data(execution_result)


def extract_discopy_data(execution_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract DisCoPy-specific data from execution result.
    Enhanced to count executions/diagrams as 'steps' if needed.

    Args:
        execution_result: Execution result dictionary

    Returns:
        Extracted simulation data
    """
    simulation_data = execution_result.get("simulation_data", {})

    # Try to read from collected files if available
    implementation_dir = execution_result.get("implementation_directory")
    if implementation_dir:
        try:
            impl_path = Path(implementation_dir)

            # Read execution report
            possible_reports: list[Any] = [
                impl_path / "discopy_execution_report.json",
                impl_path / "discopy_results" / "discopy_execution_report.json",
                impl_path
                / "execution_results"
                / "discopy_results"
                / "discopy_execution_report.json",
            ]

            for report_file in possible_reports:
                if report_file.exists():
                    with open(report_file) as f:
                        report_data = json.load(f)

                        # Use analysis summary to populate data
                        summary = report_data.get("analysis_summary", {})
                        executions = report_data.get("executions", [])

                        total_processed = summary.get("total_files_processed", 0)
                        jax_analyzed = summary.get("jax_outputs_analyzed", 0)

                        # Populate diagrams list
                        diagrams: list[Any] = []
                        for exec_rec in executions:
                            if (
                                exec_rec.get("type") == "diagram_validation"
                                and exec_rec.get("status") == "SUCCESS"
                            ):
                                diagrams.append(exec_rec.get("file_path"))

                        simulation_data["diagrams"] = diagrams
                        simulation_data["visualization_count"] = len(diagrams)

                        if jax_analyzed > 0:
                            simulation_data["traces"] = [
                                {"step": 1, "type": "jax_eval"}
                                for _ in range(jax_analyzed)
                            ]
                        elif total_processed > 0:
                            simulation_data["traces"] = [
                                {"step": 1, "type": "diagram"}
                                for _ in range(total_processed)
                            ]

                        logger.info(
                            f"Extracted DisCoPy data: {len(diagrams)} diagrams, {jax_analyzed} JAX evals"
                        )
                        break
        except Exception as e:
            logger.warning(f"Error reading DisCoPy files: {e}")

    return {
        "diagrams": simulation_data.get("diagrams", []),
        "circuits": simulation_data.get("circuits", []),
        "traces": simulation_data.get("traces", []),
        "visualization_count": simulation_data.get("visualization_count", 0),
    }


def _extract_schema_aware_data(
    execution_result: Dict[str, Any], schema_id: str
) -> Dict[str, Any]:
    """Shared extractor for the pytorch and numpyro result schemas.

    Dispatch order: a payload stamped with ``schema_id`` is used as-is; a
    nested ``simulation_data`` dict carrying ``schema_id`` is unwrapped;
    otherwise the implementation directory is probed for the current schema.
    Schema-less payloads keep the ungated contract: a nested
    ``simulation_data`` holding beliefs/actions/observations, or the
    execution result itself, flows through the normaliser's top-level
    fallback. The pytorch/numpyro runner schemas record expected free energy
    under ``efe_history``, which is mapped onto ``expected_free_energy``
    before normalisation.
    """
    payload: Dict[str, Any] | None = None
    if execution_result.get("schema_version") == schema_id:
        payload = execution_result
    nested_payload = execution_result.get("simulation_data")
    if (
        payload is None
        and isinstance(nested_payload, dict)
        and nested_payload.get("schema_version") == schema_id
    ):
        payload = nested_payload
    if payload is None and execution_result.get("implementation_directory"):
        payload = _load_current_schema_from_impl_dir(
            execution_result.get("implementation_directory"), schema_id
        )
    if payload is None:
        schemaless = nested_payload if isinstance(nested_payload, dict) else {}
        if any(key in schemaless for key in ("beliefs", "actions", "observations")):
            payload = schemaless
        else:
            payload = execution_result
    adapted = dict(payload)
    if "efe_history" in adapted and "expected_free_energy" not in adapted:
        adapted["expected_free_energy"] = adapted["efe_history"]
    return _normalise_current_simulation_payload(adapted, fallback_top_level=True)


def extract_pytorch_data(execution_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract PyTorch results, dispatching on the result schema."""
    return _extract_schema_aware_data(
        execution_result, CURRENT_SIMULATION_SCHEMAS["pytorch"]
    )


def extract_numpyro_data(execution_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract NumPyro results, dispatching on the result schema."""
    return _extract_schema_aware_data(
        execution_result, CURRENT_SIMULATION_SCHEMAS["numpyro"]
    )


def extract_bnlearn_data(execution_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract source-bound fixed-action filtering, preserving transition timing.

    Synthetic compatibility demonstrations carry no posterior analysis credit.
    """
    schema = "bnlearn_fixed_action_filter_v1"
    payload = (
        execution_result if execution_result.get("schema_version") == schema else None
    )
    nested = execution_result.get("simulation_data")
    if (
        payload is None
        and isinstance(nested, dict)
        and nested.get("schema_version") == schema
    ):
        payload = nested
    if payload is None:
        payload = _load_current_schema_from_impl_dir(
            execution_result.get("implementation_directory"), schema
        )
    if payload is None:
        return {
            "backend_smoke_only": True,
            "beliefs": [],
            "actions": [],
            "observations": [],
        }
    from gnn.analysis.result_adapter import structured_result_data

    payload = structured_result_data(payload)
    extracted = _normalise_current_simulation_payload(payload)
    for key in (
        "transition_actions",
        "action_semantics",
        "inference_mode",
        "control_mode",
        "scientific_limits",
        "source_identity",
        "model_contract_sha256",
        "source_parameter_fidelity",
    ):
        if key in payload:
            extracted[key] = payload[key]
    return extracted

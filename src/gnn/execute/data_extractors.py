#!/usr/bin/env python3
"""
Data extraction functions for simulation output parsing.

Extracts simulation data from both file-based outputs and stdout/stderr
for all supported frameworks: PyMDP, RxInfer.jl, ActiveInference.jl, JAX,
DisCoPy, PyTorch, NumPyro, ngc-learn, Stan, and bnlearn.
"""

import json
import logging
import re
from pathlib import Path
from shutil import copy2, rmtree
from typing import TYPE_CHECKING, Any, Dict, List

_module_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from gnn.execute import FrameworkName


# ---------------------------------------------------------------------------
# Path normalization and output collection
# ---------------------------------------------------------------------------


def normalize_and_deduplicate_paths(
    found_files: List[Path], logger: logging.Logger
) -> List[Path]:
    """
    Normalize paths and remove duplicates/nested paths.

    Args:
        found_files: List of file paths to normalize
        logger: Logger instance for logging

    Returns:
        Deduplicated list of normalized paths
    """
    if not found_files:
        return []

    normalized: dict[Any, Any] = {}
    for file_path in found_files:
        try:
            abs_path = file_path.resolve()
            if abs_path not in normalized:
                normalized[abs_path] = file_path
        except (OSError, RuntimeError) as e:
            logger.debug(f"Skipping invalid path {file_path}: {e}")
            continue

    sorted_paths = sorted(normalized.values(), key=lambda p: len(p.parts))
    deduplicated: list[Any] = []

    for file_path in sorted_paths:
        file_name = file_path.name
        file_parent = file_path.parent

        is_nested_duplicate = False
        for seen_path in deduplicated:
            seen_parent = seen_path.parent
            if file_name != seen_path.name:
                continue
            try:
                is_nested = file_parent.is_relative_to(seen_parent)
            except (ValueError, AttributeError):
                is_nested = str(file_parent).startswith(str(seen_parent))
            if is_nested:
                is_nested_duplicate = True
                logger.debug(
                    f"Skipping nested duplicate: {file_path} (already have {seen_path})"
                )
                break

        if not is_nested_duplicate:
            deduplicated.append(file_path)

    if len(found_files) != len(deduplicated):
        logger.info(
            f"Deduplicated paths: {len(found_files)} -> {len(deduplicated)} files"
        )

    return deduplicated


def _clear_collected_output_dirs(output_dir: Path, logger: logging.Logger) -> None:
    """Clear per-run collected data directories before copying current outputs."""
    for child_name in ("simulation_data", "traces", "other"):
        child_dir = output_dir / child_name
        if not child_dir.exists():
            continue
        for child in child_dir.iterdir():
            try:
                if child.is_dir():
                    rmtree(child)
                else:
                    child.unlink()
            except OSError as exc:
                logger.debug(
                    "Could not remove previous collected output %s: %s", child, exc
                )


def collect_execution_outputs(
    script_path: Path,
    output_dir: Path,
    framework: "FrameworkName",
    logger: logging.Logger,
) -> Dict[str, List[str]]:
    """
    Collect all outputs from executed script and copy to execute output directory.

    Args:
        script_path: Path to the executed script
        output_dir: Execute output directory for this model/framework
        framework: Framework name
        logger: Logger instance

    Returns:
        Dictionary with lists of copied file paths by category
    """
    collected: dict[str, Any] = {
        "visualizations": [],
        "simulation_data": [],
        "traces": [],
        "other": [],
    }

    if framework == "thrml":
        # The supervised invocation has already bound this result to its token
        # and script. Preserve its owned destination; never scan old scripts.
        artifact = output_dir / "simulation_data" / "simulation_results.json"
        if artifact.is_file() and not artifact.is_symlink():
            collected["simulation_data"] = [str(artifact)]
        return collected

    try:
        script_dir = script_path.parent

        found_files: list[Any] = []

        if framework == "pymdp":
            pymdp_output = script_dir / "output" / "pymdp_simulations"
            if pymdp_output.exists():
                found_files.extend(pymdp_output.rglob("*.png"))
                found_files.extend(pymdp_output.rglob("*.svg"))
                found_files.extend(pymdp_output.rglob("*.json"))
                found_files.extend(pymdp_output.rglob("*.pkl"))
        elif framework == "discopy":
            discopy_dir = script_dir / "discopy_diagrams"
            if discopy_dir.exists():
                found_files.extend(discopy_dir.rglob("*.png"))
                found_files.extend(discopy_dir.rglob("*.svg"))
                found_files.extend(discopy_dir.rglob("*.json"))
        elif framework == "activeinference_jl":
            found_files.extend(script_dir.glob("simulation_results.json"))
            for out_dir in script_dir.glob("activeinference_outputs_*"):
                if out_dir.is_dir():
                    found_files.extend(out_dir.glob("simulation_results.json"))
                    viz_dir = out_dir / "visualizations"
                    if viz_dir.exists() and viz_dir.is_dir():
                        found_files.extend(viz_dir.glob("*.png"))
                        found_files.extend(viz_dir.glob("*.svg"))
                    sim_data_dir = out_dir / "simulation_data"
                    if sim_data_dir.exists() and sim_data_dir.is_dir():
                        found_files.extend(sim_data_dir.glob("*.json"))
                        found_files.extend(sim_data_dir.glob("*.csv"))
                    traces_dir = out_dir / "free_energy_traces"
                    if traces_dir.exists() and traces_dir.is_dir():
                        found_files.extend(traces_dir.glob("*.json"))
                        found_files.extend(traces_dir.glob("*.csv"))
        elif framework == "rxinfer":
            found_files.extend(script_dir.glob("simulation_results.json"))
            rxinfer_dir = script_dir / "rxinfer_outputs"
            if rxinfer_dir.exists():
                found_files.extend(rxinfer_dir.rglob("*.png"))
                found_files.extend(rxinfer_dir.rglob("*.json"))
                found_files.extend(rxinfer_dir.rglob("*.csv"))
        elif framework == "jax":
            jax_dir = script_dir / "jax_outputs"
            if jax_dir.exists():
                found_files.extend(jax_dir.rglob("*.png"))
                found_files.extend(jax_dir.rglob("*.json"))
        elif framework == "numpyro":
            # NumPyro writes simulation_results.json under NUMPYRO_OUTPUT_DIR (default: script cwd)
            found_files.extend(script_dir.glob("simulation_results.json"))
            numpyro_out = script_dir / "numpyro_outputs"
            if numpyro_out.exists():
                found_files.extend(numpyro_out.rglob("*.json"))
                found_files.extend(numpyro_out.rglob("*.csv"))
        elif framework == "pytorch":
            # PyTorch writes simulation_results.json under PYTORCH_OUTPUT_DIR (default: script cwd)
            found_files.extend(script_dir.glob("simulation_results.json"))
            pytorch_out = script_dir / "pytorch_outputs"
            if pytorch_out.exists():
                found_files.extend(pytorch_out.rglob("*.json"))
                found_files.extend(pytorch_out.rglob("*.csv"))
        elif framework == "ngclearn":
            # ngc-learn writes simulation_results.json under NGCLEARN_OUTPUT_DIR (default: script cwd)
            found_files.extend(script_dir.glob("simulation_results.json"))
            ngclearn_out = script_dir / "ngclearn_outputs"
            if ngclearn_out.exists():
                found_files.extend(ngclearn_out.rglob("*.json"))
                found_files.extend(ngclearn_out.rglob("*.csv"))

        if not found_files:
            found_files.extend(script_dir.rglob("*.png"))
            found_files.extend(script_dir.rglob("*.svg"))
            found_files.extend(script_dir.rglob("*.json"))
            found_files.extend(script_dir.rglob("*.pkl"))
            found_files.extend(script_dir.rglob("*.csv"))

        found_files = [
            f for f in found_files if f != script_path and f.exists() and f.is_file()
        ]
        found_files = normalize_and_deduplicate_paths(found_files, logger)

        if not found_files:
            logger.debug(
                f"No output files found for {framework} script {script_path.name}"
            )
            return collected

        _clear_collected_output_dirs(output_dir, logger)
        logger.info(f"Found {len(found_files)} output files to collect for {framework}")

        for source_file in found_files:
            try:
                ext = source_file.suffix.lower()

                if ext in [".png", ".svg", ".jpg", ".jpeg"]:
                    logger.debug(
                        f"Skipping visualization {source_file.name} (will be collected by analysis step)"
                    )
                    continue

                if ext in [".json", ".pkl", ".csv"]:
                    if (
                        "trace" in source_file.name.lower()
                        or "posterior" in source_file.name.lower()
                    ):
                        dest_dir = output_dir / "traces"
                        category = "traces"
                    else:
                        dest_dir = output_dir / "simulation_data"
                        category = "simulation_data"
                else:
                    dest_dir = output_dir / "other"
                    category = "other"

                dest_dir.mkdir(parents=True, exist_ok=True)

                dest_file = dest_dir / source_file.name

                if dest_file.exists():
                    try:
                        source_stat = source_file.stat()
                        dest_stat = dest_file.stat()
                        if (
                            source_stat.st_size == dest_stat.st_size
                            and abs(source_stat.st_mtime - dest_stat.st_mtime) < 1.0
                        ):
                            logger.debug(
                                f"Skipping duplicate: {dest_file.name} already exists"
                            )
                            collected[category].append(str(dest_file))
                            continue
                    except OSError as e:
                        logger.debug("Stat failed, treating as new file: %s", e)

                    parent_name = source_file.parent.name
                    if not source_file.name.startswith(f"{parent_name}_"):
                        dest_file = dest_dir / f"{parent_name}_{source_file.name}"
                    else:
                        counter = 1
                        base_name = source_file.stem
                        ext = source_file.suffix
                        while dest_file.exists():
                            dest_file = dest_dir / f"{base_name}_{counter}{ext}"
                            counter += 1

                copy2(source_file, dest_file)

                collected[category].append(str(dest_file))
                logger.info(
                    f"Copied {source_file.name} -> {dest_file.relative_to(output_dir)}"
                )

            except Exception as e:
                logger.warning(f"Failed to copy {source_file}: {e}")

        total_copied = sum(len(files) for files in collected.values())
        if total_copied > 0:
            logger.info(
                f"Collected {total_copied} output files: "
                f"{len(collected['visualizations'])} visualizations, "
                f"{len(collected['simulation_data'])} data files, "
                f"{len(collected['traces'])} traces"
            )

    except Exception as e:
        logger.error(f"Error collecting execution outputs: {e}")
        import traceback

        logger.debug(traceback.format_exc())

    return collected


# ---------------------------------------------------------------------------
# File-based extraction (reads saved simulation artifacts)
# ---------------------------------------------------------------------------


def extract_simulation_data_from_files(
    output_dir: Path, framework: "FrameworkName", logger: Any
) -> Dict[str, Any]:
    """
    Extract simulation data from collected files (not just stdout/stderr).

    Args:
        output_dir: Directory containing collected output files
        framework: Framework name
        logger: Logger instance

    Returns:
        Dictionary with extracted simulation data
    """
    enhanced_data: dict[Any, Any] = {}

    try:
        if framework == "thrml":
            from gnn.analysis.thrml import adapt_result

            artifact = output_dir / "simulation_data" / "simulation_results.json"
            if artifact.is_symlink() or artifact.stat().st_size > 16 * 1024 * 1024:
                raise ValueError("THRML native result is unsafe or exceeds 16 MiB")
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            adapt_result(payload)
            enhanced_data = payload
        elif framework == "pymdp":
            enhanced_data = extract_pymdp_data_from_files(output_dir, logger)
        elif framework == "rxinfer":
            enhanced_data = extract_rxinfer_data_from_files(output_dir, logger)
        elif framework == "activeinference_jl":
            enhanced_data = extract_activeinference_jl_data_from_files(
                output_dir, logger
            )
        elif framework == "discopy":
            enhanced_data = extract_discopy_data_from_files(output_dir, logger)
        elif framework == "jax":
            enhanced_data = extract_jax_data_from_files(output_dir, logger)
        elif framework == "numpyro":
            enhanced_data = extract_pymdp_like_data_from_files(
                output_dir, logger, "numpyro"
            )
        elif framework == "pytorch":
            enhanced_data = extract_pymdp_like_data_from_files(
                output_dir, logger, "pytorch"
            )
        elif framework == "ngclearn":
            enhanced_data = extract_pymdp_like_data_from_files(
                output_dir, logger, "ngclearn"
            )

    except Exception as e:
        logger.warning(
            f"Failed to extract simulation data from files for {framework}: {e}"
        )
        import traceback

        logger.debug(traceback.format_exc())

    return enhanced_data


def extract_pymdp_data_from_files(
    output_dir: Path, logger: logging.Logger
) -> Dict[str, Any]:
    """Extract PyMDP simulation data from saved files."""
    data: dict[Any, Any] = {}

    try:
        # Look for simulation_results.json
        sim_data_dir = output_dir / "simulation_data"
        if sim_data_dir.exists():
            results_files = list(sim_data_dir.glob("*simulation_results.json"))
            if results_files:
                results_file = results_files[0]
                try:
                    with open(results_file, "r") as f:
                        results = json.load(f)

                    # Extract beliefs, actions, observations
                    if "beliefs" in results:
                        data["beliefs"] = results["beliefs"]
                    if "actions" in results:
                        data["actions"] = results["actions"]
                    if "observations" in results:
                        data["observations"] = results["observations"]
                    if "expected_free_energy" in results:
                        data["expected_free_energy"] = results["expected_free_energy"]
                    if "expected_free_energy_convention" in results:
                        data["expected_free_energy_convention"] = results[
                            "expected_free_energy_convention"
                        ]
                    if "num_timesteps" in results:
                        data["num_timesteps"] = results["num_timesteps"]
                except Exception as e:
                    logger.warning(f"Failed to parse {results_file}: {e}")

        # Count visualizations
        viz_dir = output_dir / "visualizations"
        if viz_dir.exists():
            viz_files = list(viz_dir.glob("*.png")) + list(viz_dir.glob("*.svg"))
            if viz_files:
                data["visualization_count"] = len(viz_files)
                data["visualization_files"] = [str(f.name) for f in viz_files]

    except Exception as e:
        logger.warning(f"Error extracting PyMDP data from files: {e}")

    return data


def extract_pymdp_like_data_from_files(
    output_dir: Path, logger: logging.Logger, label: str = "pymdp_like"
) -> Dict[str, Any]:
    """Extract simulation data from simulation_data/simulation_results.json (NumPyro, PyTorch, etc.)."""
    data: dict[Any, Any] = {}
    try:
        sim_data_dir = output_dir / "simulation_data"
        if sim_data_dir.exists():
            results_files = list(sim_data_dir.glob("*simulation_results.json"))
            if results_files:
                results_file = results_files[0]
                try:
                    with open(results_file, "r") as f:
                        results = json.load(f)
                    for key in (
                        "schema_version",
                        "beliefs",
                        "actions",
                        "observations",
                        "free_energy",
                        "expected_free_energy",
                        "expected_free_energy_convention",
                        "num_timesteps",
                        "model_parameters",
                    ):
                        if key in results:
                            data[key] = results[key]
                    logger.info(f"Extracted {label} data from {results_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to parse {results_file}: {e}")
    except Exception as e:
        logger.warning(f"Error extracting {label} data from files: {e}")
    return data


def extract_rxinfer_data_from_files(
    output_dir: Path, logger: logging.Logger
) -> Dict[str, Any]:
    """Extract RxInfer.jl simulation data from saved files."""
    data: dict[Any, Any] = {}

    try:
        sim_data_dir = output_dir / "simulation_data"
        if sim_data_dir.exists():
            results_files = list(sim_data_dir.glob("*simulation_results.json"))
            if results_files:
                results_file = results_files[0]
                with open(results_file, "r", encoding="utf-8") as f:
                    results = json.load(f)
                from gnn.analysis.result_adapter import structured_result_data

                data.update(structured_result_data(results))
                if "vfe_per_iteration" in results:
                    data["free_energy"] = results["vfe_per_iteration"]
                logger.info(f"Extracted RxInfer data from {results_file.name}")
                return data

        # Look for inference data JSON files
        data_dir = output_dir / "inference_data"
        if data_dir.exists():
            json_files = list(data_dir.glob("*.json"))
            for json_file in json_files:
                try:
                    with open(json_file, "r") as f:
                        inference_data = json.load(f)
                        if "posterior" in inference_data:
                            data["posterior"] = inference_data["posterior"]
                except Exception as e:
                    logger.debug(f"Error reading inference data from {json_file}: {e}")

        # Look for trace files
        trace_dir = output_dir / "posterior_traces"
        if trace_dir.exists():
            trace_files = list(trace_dir.glob("*.csv"))
            if trace_files:
                data["trace_files"] = [str(f.name) for f in trace_files]

    except Exception as e:
        logger.warning(f"Error extracting RxInfer data from files: {e}")

    return data


def extract_activeinference_jl_data_from_files(
    output_dir: Path, logger: logging.Logger
) -> Dict[str, Any]:
    """Extract ActiveInference.jl simulation data from saved files."""
    data: dict[Any, Any] = {}

    try:
        # Look for activeinference_outputs_* directories (timestamped output dirs)
        output_dirs = list(output_dir.glob("activeinference_outputs_*"))
        if not output_dirs:
            # Also check parent directories
            output_dirs = list(output_dir.parent.glob("**/activeinference_outputs_*"))

        # Get most recent output directory
        if output_dirs:
            output_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest_output = output_dirs[0]
            logger.debug(f"Found ActiveInference.jl output directory: {latest_output}")

            # Parse model_parameters.json
            params_file = latest_output / "model_parameters.json"
            if params_file.exists():
                try:
                    with open(params_file, "r") as f:
                        params = json.load(f)
                        data["model_name"] = params.get("model_name")
                        data["n_states"] = params.get("n_states")
                        data["n_observations"] = params.get("n_observations")
                        data["n_actions"] = params.get("n_actions")
                        data["timestamp"] = params.get("timestamp")
                        logger.debug(f"Loaded model parameters from {params_file}")
                except Exception as e:
                    logger.warning(f"Error reading model_parameters.json: {e}")

            # Parse simulation_results.csv
            results_csv = latest_output / "simulation_results.csv"
            if results_csv.exists():
                try:
                    import csv

                    with open(results_csv, "r") as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                        if rows:
                            data["timesteps"] = len(rows)
                            data["observations"] = [
                                int(r.get("observation", 0)) for r in rows
                            ]
                            data["actions"] = [int(r.get("action", 0)) for r in rows]
                            # Extract beliefs if available
                            belief_keys = [
                                k for k in rows[0].keys() if k.startswith("belief")
                            ]
                            if belief_keys:
                                data["beliefs"] = [
                                    [float(r.get(k, 0)) for k in belief_keys]
                                    for r in rows
                                ]
                            logger.debug(
                                f"Loaded {len(rows)} timesteps from simulation_results.csv"
                            )
                except Exception as e:
                    logger.warning(f"Error reading simulation_results.csv: {e}")

            # Parse summary.txt for validation status
            summary_file = latest_output / "summary.txt"
            if summary_file.exists():
                try:
                    with open(summary_file, "r") as f:
                        summary_text = f.read()
                        data["validation_passed"] = "PASSED" in summary_text
                        logger.debug(f"Read summary from {summary_file}")
                except Exception as e:
                    logger.warning(f"Error reading summary.txt: {e}")

            # Count visualizations
            viz_files = list(latest_output.glob("*.png"))
            if viz_files:
                data["visualization_count"] = len(viz_files)
                data["visualization_files"] = [f.name for f in viz_files]

        # Also check the shared simulation_data location.
        data_dir = output_dir / "simulation_data"
        if data_dir.exists():
            json_files = list(data_dir.glob("*simulation_results.json")) + list(
                data_dir.glob("*.json")
            )
            for json_file in json_files:
                try:
                    with open(json_file, "r") as f:
                        sim_data = json.load(f)
                        for key in (
                            "schema_version",
                            "beliefs",
                            "true_states",
                            "observations",
                            "actions",
                            "expected_free_energy",
                            "expected_free_energy_convention",
                            "policy_posterior",
                            "validation",
                            "model_parameters",
                        ):
                            if key in sim_data and key not in data:
                                data[key] = sim_data[key]
                        if (
                            "expected_free_energy" in sim_data
                            and "free_energy" not in data
                        ):
                            data["free_energy"] = sim_data["expected_free_energy"]
                        if "free_energy" in sim_data and "free_energy" not in data:
                            data["free_energy"] = sim_data["free_energy"]
                        if "beliefs" in sim_data and "beliefs" not in data:
                            data["beliefs"] = sim_data["beliefs"]
                except Exception as e:
                    logger.debug(f"Error reading simulation data from {json_file}: {e}")

        # Look for free energy traces
        fe_dir = output_dir / "free_energy_traces"
        if fe_dir.exists():
            trace_files = list(fe_dir.glob("*.csv"))
            if trace_files:
                data["free_energy_trace_files"] = [str(f.name) for f in trace_files]

    except Exception as e:
        logger.warning(f"Error extracting ActiveInference.jl data from files: {e}")

    return data


def extract_discopy_data_from_files(
    output_dir: Path, logger: logging.Logger
) -> Dict[str, Any]:
    """Extract DisCoPy simulation data from saved files."""
    data: dict[Any, Any] = {}

    try:
        # Look for circuit analysis JSON in multiple possible locations
        search_dirs: list[Any] = [
            output_dir / "simulation_data",
            output_dir / "discopy_diagrams",
            output_dir / "analysis",
            output_dir,
        ]

        for search_dir in search_dirs:
            if search_dir.exists():
                json_files = list(search_dir.glob("*circuit*.json")) + list(
                    search_dir.glob("*analysis*.json")
                )
                for json_file in json_files:
                    try:
                        with open(json_file, "r") as f:
                            circuit_data = json.load(f)
                            if "circuit" in circuit_data:
                                data["circuit"] = circuit_data["circuit"]
                            if "components" in circuit_data:
                                data["components"] = circuit_data["components"]
                            if "analysis" in circuit_data:
                                data["analysis"] = circuit_data["analysis"]
                            if "parameters" in circuit_data:
                                data["parameters"] = circuit_data["parameters"]
                            logger.debug(f"Loaded DisCoPy data from {json_file}")
                    except Exception as e:
                        logger.debug(
                            f"Error reading DisCoPy data from {json_file}: {e}"
                        )

        # Count diagram outputs in multiple possible locations
        diagram_dirs: list[Any] = [
            output_dir / "discopy_diagrams",
            output_dir / "diagram_outputs",
            output_dir / "simulation_data",
        ]

        for diagram_dir in diagram_dirs:
            if diagram_dir.exists():
                diagram_files = list(diagram_dir.glob("*.png"))
                if diagram_files:
                    data["diagram_count"] = len(diagram_files)
                    data["diagram_files"] = [str(f.name) for f in diagram_files]
                    logger.debug(
                        f"Found {len(diagram_files)} DisCoPy diagrams in {diagram_dir}"
                    )
                    break

    except Exception as e:
        logger.warning(f"Error extracting DisCoPy data from files: {e}")

    return data


# JAX scripts write to the same shared contract as PyMDP:
# simulation_data/simulation_results.json with keys beliefs, actions,
# observations, num_timesteps; and visualizations/*.{png,svg}.
extract_jax_data_from_files = extract_pymdp_data_from_files


# ---------------------------------------------------------------------------
# Stdout/stderr-based extraction (parses execution output text)
# ---------------------------------------------------------------------------


def extract_simulation_data(
    stdout: str, stderr: str, framework: "FrameworkName", logger: logging.Logger
) -> Dict[str, Any]:
    """
    Extract simulation data from execution output.

    Args:
        stdout: Standard output from script execution
        stderr: Standard error from script execution
        framework: Framework name
        logger: Logger instance

    Returns:
        Dictionary with extracted simulation data
    """
    simulation_data: dict[str, Any] = {
        "traces": [],
        "free_energy": [],
        "states": [],
        "observations": [],
        "actions": [],
        "policy": [],
        "raw_output": stdout[:10000] if stdout else "",  # Limit size
        "raw_error": stderr[:10000] if stderr else "",
    }

    try:
        # Framework-specific extraction
        if framework == "pymdp":
            simulation_data.update(extract_pymdp_data(stdout, stderr))
        elif framework == "rxinfer":
            simulation_data.update(extract_rxinfer_data(stdout, stderr))
        elif framework == "activeinference_jl":
            simulation_data.update(extract_activeinference_jl_data(stdout, stderr))
        elif framework == "jax":
            simulation_data.update(extract_jax_data(stdout, stderr))
        elif framework == "discopy":
            simulation_data.update(extract_discopy_data(stdout, stderr))
        else:
            # Generic extraction - try to find common patterns
            simulation_data.update(extract_generic_data(stdout, stderr))

    except Exception as e:
        logger.warning(f"Failed to extract simulation data for {framework}: {e}")

    return simulation_data


def extract_pymdp_data(stdout: str, stderr: str) -> Dict[str, Any]:
    """Extract PyMDP-specific simulation data from stdout/stderr."""
    data: dict[Any, Any] = {}

    # Combine stdout and stderr for parsing
    combined_output = stdout + "\n" + stderr

    def _first_matches(text: str, *patterns: str) -> list:
        """Return matches from the first pattern that produces any results."""
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches
        return []

    # Observations: structured log line first, then bare keyword
    obs_matches = _first_matches(
        combined_output,
        r"Step\s+\d+:\s+obs=(\d+)",
        r"observation[:\s]+(\d+)",
        r"obs[:\s]+(\d+)",
    )
    if obs_matches:
        data["observations"] = [
            int(m if isinstance(m, str) else (m[0] or m[1])) for m in obs_matches
        ]

    # Actions: structured log line first, then bare keyword
    action_matches = _first_matches(
        combined_output,
        r"Step\s+\d+:\s+obs=\d+,\s+belief=[^\]]+,\s+action=([\d.]+)",
        r"action[:\s]+(\d+)",
        r"action_taken[:\s]+(\d+)",
    )
    if action_matches:
        data["actions"] = [
            int(float(m if isinstance(m, str) else (m[0] or m[1])))
            for m in action_matches
        ]

    # Beliefs
    belief_matches = re.findall(r"belief=\[([^\]]+)\]", combined_output, re.IGNORECASE)
    if belief_matches:
        try:
            beliefs: list[Any] = []
            for match in belief_matches:
                values = re.findall(r"[\d.]+", match)
                if values:
                    beliefs.append([float(v) for v in values])
            if beliefs:
                data["beliefs"] = beliefs
        except Exception as e:
            logging.getLogger(__name__).debug(f"Error parsing belief data: {e}")

    # States
    state_matches = re.findall(
        r"state[:\s]+\[([^\]]+)\]|states[:\s]+\[([^\]]+)\]",
        combined_output,
        re.IGNORECASE,
    )
    if state_matches and "states" not in data:
        data["states"] = [match[0] or match[1] for match in state_matches]

    # Free energy
    fe_matches = re.findall(
        r"free[_\s]?energy[:\s]+([\d.]+)|FE[:\s]+([\d.]+)",
        combined_output,
        re.IGNORECASE,
    )
    if fe_matches:
        data["free_energy"] = [float(match[0] or match[1]) for match in fe_matches]

    return data


def extract_rxinfer_data(stdout: str, stderr: str) -> Dict[str, Any]:
    """Extract RxInfer.jl-specific simulation data."""
    data: dict[Any, Any] = {}

    # Try to find posterior distributions
    posterior_pattern = r"posterior[:\s]+\[([^\]]+)\]"
    posterior_matches = re.findall(posterior_pattern, stdout, re.IGNORECASE)
    if posterior_matches:
        data["posterior"] = posterior_matches

    return data


def extract_activeinference_jl_data(stdout: str, stderr: str) -> Dict[str, Any]:
    """Extract ActiveInference.jl-specific simulation data."""
    data: dict[Any, Any] = {}

    # Try to find free energy traces
    fe_pattern = r"free[_\s]?energy[:\s]+([\d.]+)|FE[:\s]+([\d.]+)"
    fe_matches = re.findall(fe_pattern, stdout, re.IGNORECASE)
    if fe_matches:
        data["free_energy"] = [float(match[0] or match[1]) for match in fe_matches]

    # Try to find state beliefs
    belief_pattern = r"belief[:\s]+\[([^\]]+)\]|q\(s\)[:\s]+\[([^\]]+)\]"
    belief_matches = re.findall(belief_pattern, stdout, re.IGNORECASE)
    if belief_matches:
        data["beliefs"] = [match[0] or match[1] for match in belief_matches]

    return data


# JAX scripts emit the same structured output as PyMDP (JSON lines with
# beliefs/actions/observations).
extract_jax_data = extract_pymdp_data


def extract_discopy_data(stdout: str, stderr: str) -> Dict[str, Any]:
    """Extract DisCoPy-specific simulation data."""
    data: dict[Any, Any] = {}

    # Try to find diagram information
    diagram_pattern = r"diagram[:\s]+(\w+)|circuit[:\s]+(\w+)"
    diagram_matches = re.findall(diagram_pattern, stdout, re.IGNORECASE)
    if diagram_matches:
        data["diagrams"] = [match[0] or match[1] for match in diagram_matches]

    return data


def extract_generic_data(stdout: str, stderr: str) -> Dict[str, Any]:
    """Generic extraction for unknown frameworks."""
    data: dict[Any, Any] = {}

    # Try to find any numeric arrays or lists
    array_pattern = r"\[([\d.,\s]+)\]"
    array_matches = re.findall(array_pattern, stdout)
    if array_matches:
        data["arrays"] = array_matches[:10]  # Limit to first 10

    return data

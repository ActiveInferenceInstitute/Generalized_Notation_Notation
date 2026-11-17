"""
Core Type Checker Processor.

Provides the ``GNNTypeChecker`` class which orchestrates type checking
of GNN files, validating syntax, dimensions, and type consistency.
"""

import hashlib
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, TypedDict

from gnn.utils.observability.structured_logging import (
    log_step_error,
    log_step_start,
    log_step_success,
    log_step_warning,
)

from ..estimation.strategies import VariableMap, calculate_complexity
from .dimensions import (
    extract_b_matrix_evidence,
    extract_gnn_dimensions_with_diagnostics,
    validate_dimension_compatibility,
)
from .rules import (
    check_type_consistency,
    extract_types_from_content,
    get_validation_rules,
    validate_type,
)
from .sections import (
    classify_time_spec,
    detect_time_dynamics,
    extract_markdown_section,
    parse_resource_connections,
    section_presence,
)
from .summary import summarize_type_check_results

_module_logger = logging.getLogger(__name__)


class ResourceEstimate(TypedDict):
    """Stable resource-estimation result emitted by the type checker."""

    complexity_tier: str
    estimated_memory_bytes: int
    total_parameters: int
    variables: int
    connections: int
    flops_estimate: float
    complexity_score: float
    diagnostics: list[str]


def estimate_file_resources(content: str) -> ResourceEstimate:
    """Estimate computational resources needed for a GNN file using core framework logic.

    This function bridges the type checker to the estimation subsystem
    for generation of Baseball Cards during the standard validation pass.
    """
    from ..estimation.estimator import GNNResourceEstimator

    variables_with_dims, diagnostics = extract_gnn_dimensions_with_diagnostics(content)
    variable_types = {
        str(type_info["name"]): str(type_info["type"])
        for type_info in extract_types_from_content(content)
    }

    variables: VariableMap = {
        name: {
            "dimensions": dimensions,
            "type": variable_types.get(name, "float"),
        }
        for name, dimensions in variables_with_dims.items()
    }

    edges, connection_diagnostics = parse_resource_connections(content, set(variables))
    diagnostics.extend(connection_diagnostics)
    equations = extract_markdown_section(content, "Equations")

    total_parameters = 0
    memory_bytes = 0
    for name, dimensions in variables_with_dims.items():
        parameter_count = math.prod(dimensions)
        total_parameters += parameter_count
        variable_type = variable_types.get(name, "float")
        bytes_per_element = GNNResourceEstimator.MEMORY_FACTORS.get(
            variable_type,
            GNNResourceEstimator.MEMORY_FACTORS["float"],
        )
        memory_bytes += parameter_count * bytes_per_element

    complexity_metrics = calculate_complexity(variables, edges, equations)
    score = complexity_metrics["overall_complexity"]
    if score > 8.0:
        complexity_tier = "large"
    elif score > 5.0:
        complexity_tier = "medium"
    elif score > 2.0:
        complexity_tier = "small"
    else:
        complexity_tier = "minimal"

    return {
        "complexity_tier": complexity_tier,
        "estimated_memory_bytes": memory_bytes,
        "total_parameters": total_parameters,
        "variables": len(variables),
        "connections": len(edges),
        "flops_estimate": score * 500.0,
        "complexity_score": score,
        "diagnostics": diagnostics,
    }


def _type_check_file_job(job: tuple[Path, bool, bool, str | None]) -> dict[str, Any]:
    """Compute one validation/analysis pair in a worker, without output writes."""
    path, verbose, strict, expected_hash = job
    return GNNTypeChecker(strict_mode=strict)._compute_file(
        path, verbose, strict, expected_hash
    )


class GNNTypeChecker:
    """Type checker for GNN files.

    Result contracts (three sibling surfaces, deliberately distinct):

    - :meth:`validate_content` / :meth:`validate_single_gnn_file` return a
      per-file result dict whose canonical truth key is ``valid`` (bool),
      plus additive ``errors``/``warnings``/``type_issues`` lists and
      structured metadata (``variables``, ``connections``, ``sections``,
      ``time_dynamics``, ``model_complexity``, ...). The CLI artifact layer
      re-keys the same verdict as ``is_valid``; renderers accept either.
    - :meth:`check_file` returns the CLI 4-tuple
      ``(valid, errors, warnings, result_dict)`` derived from the dict.
    - :meth:`validate_gnn_files` returns the pipeline exit sentinel:
      ``True`` (exit 0), ``2`` (SUCCESS_WITH_WARNINGS), ``False`` (exit 1).
    """

    def __init__(self, strict_mode: bool = False) -> None:
        """Initialize the GNN type checker.

        Args:
            strict_mode: When True, promote recoverable warnings (notably
                B-orientation contradictions ``[GNN-E002]``) to errors. This
                is the default applied by :meth:`validate_single_gnn_file`
                and :meth:`validate_gnn_files` unless the caller overrides
                their ``strict`` argument explicitly.
        """
        self.strict_mode = strict_mode
        self.validation_rules = get_validation_rules()

    def check_file(
        self, file_path: str | Path
    ) -> tuple[bool, list[str], list[str], Dict[str, Any]]:
        """Validate one GNN file and return the CLI result tuple."""
        result = self.validate_single_gnn_file(Path(file_path))
        return (
            bool(result.get("valid", False)),
            list(result.get("errors", [])),
            list(result.get("warnings", [])),
            result,
        )

    def generate_report(
        self,
        results: Dict[str, Any],
        output_dir: Path,
        report_md_filename: str = "TYPE_CHECK_REPORT.md",
        project_root_path: str | None = None,
    ) -> str:
        """Generate the Markdown report consumed by the type-checker CLI."""
        output_dir.mkdir(parents=True, exist_ok=True)
        validation_results = [
            {"file_path": path, **data} for path, data in results.items()
        ]
        summary_data: Dict[str, Any] = {
            "processed_files": len(results),
            "success": all(
                item.get("valid", item.get("is_valid", False))
                for item in results.values()
            ),
            "errors": [
                error for item in results.values() for error in item.get("errors", [])
            ],
            "validation_results": validation_results,
            "type_analysis": [],
            "project_root": project_root_path,
        }
        report_text = self._generate_type_check_summary(summary_data)
        (output_dir / report_md_filename).write_text(report_text, encoding="utf-8")
        return report_text

    def generate_json_data(self, results: Dict[str, Any], output_path: Path) -> None:
        """Write type-check results as JSON for downstream analysis."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    def _discover_gnn_files(self, target_dir: Path) -> list[Path]:
        """Discover GNN specs across every registered non-binary extension.

        The parser stack registers more than just ``.md`` (notably ``.gnn``);
        a directory holding only a non-markdown spec must still be found.
        Binary pickle specs are excluded — they are not type-checked. Sorted
        for deterministic, reproducible discovery.
        """
        from gnn.parsers.common import get_supported_gnn_extensions
        from gnn.processing.discovery import is_model_source_path

        return sorted(
            path
            for ext in get_supported_gnn_extensions(include_binary_pickle=False)
            for path in target_dir.rglob(f"*{ext}")
            if is_model_source_path(path)
        )

    def validate_gnn_files(
        self,
        target_dir: Path,
        output_dir: Path,
        verbose: bool = False,
        cancel_token: Any = None,
        **kwargs: Any,
    ) -> bool | int:
        """Validate every GNN file in a directory.

        Args:
            target_dir: Directory containing GNN files to validate.
            output_dir: Directory to save validation results.
            verbose: Enable verbose output.
            **kwargs: ``strict`` overrides the instance default;
                ``estimate_resources`` triggers a resource-estimation pass.

        Returns:
            ``True`` (pipeline exit 0) when every file validates;
            ``2`` (``SUCCESS_WITH_WARNINGS``) when the run completed but one
            or more files had type/consistency problems, **or when no GNN
            files were found** — per the Phase 1.1 widened contract,
            "nothing to do" is a warning, not a hard error (matching
            Steps 12/16 and the render step);
            ``False`` (exit 1) on a hard failure (exception while
            processing, or strict-mode B-orientation contradictions).
        """
        logger = logging.getLogger("type_checker")

        try:
            from gnn.pipeline.run_context import (
                current_run_context,
                selected_model_sources,
            )
            from gnn.utils.file_workers import (
                check_file_work,
                ordered_file_map,
                resolve_file_workers,
            )

            context = current_run_context()
            deadline = context.deadline_monotonic if context else None
            requested_workers, executor_mode = resolve_file_workers(
                kwargs.get("file_workers"), "process"
            )
            check_file_work(deadline, cancel_token)
            log_step_start(logger, "Processing type checker")

            output_dir.mkdir(parents=True, exist_ok=True)

            results: dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "processed_files": 0,
                "success": True,
                "errors": [],
                "validation_results": [],
                "type_analysis": [],
            }
            hard_failure = False
            b_orientation_failed = False
            strict = bool(kwargs.get("strict", self.strict_mode))
            estimate_resources = bool(kwargs.get("estimate_resources", False))

            if context is not None:
                from gnn.parsers.common import get_supported_gnn_extensions

                gnn_files = selected_model_sources(
                    target_dir,
                    5,
                    extensions=get_supported_gnn_extensions(
                        include_binary_pickle=False
                    ),
                )
            else:
                gnn_files = self._discover_gnn_files(target_dir)
            source_hashes = (
                {m.artifact_stem: m.sha256 for m in context.selected_models(5)}
                if context
                else {}
            )
            worker_metadata: dict[str, Any] = {
                "requested_workers": requested_workers,
                "effective_workers": min(requested_workers, len(gnn_files)),
                "executor": "serial",
                "worker_pids": [],
            }
            results["worker_execution"] = worker_metadata
            if not gnn_files:
                logger.warning("No GNN files found for type checking")
                results["success"] = False
                results["errors"].append("No GNN files found")
            else:
                results["processed_files"] = len(gnn_files)

                jobs = (
                    (path, verbose, strict, source_hashes.get(path.stem))
                    for path in gnn_files
                )
                # The serial path preserves instance overrides; spawned workers own fresh checkers.
                from functools import partial

                compute = (
                    _type_check_file_job
                    if worker_metadata["effective_workers"] > 1
                    else partial(_type_check_with_instance, self)
                )
                for gnn_file, computed in zip(
                    gnn_files,
                    ordered_file_map(
                        compute,
                        jobs,
                        workers=min(requested_workers, len(gnn_files)),
                        executor=executor_mode,
                        deadline=deadline,
                        cancel_token=cancel_token,
                        metadata=worker_metadata,
                    ),
                    strict=True,
                ):
                    check_file_work(deadline, cancel_token)
                    if "error" in computed:
                        results["success"] = False
                        hard_failure = True
                        results["errors"].append(computed["error"])
                        logger.error(
                            "Error processing %s: %s",
                            gnn_file,
                            computed["error"]["error"],
                        )
                        continue
                    validation_result = computed["validation"]
                    results["validation_results"].append(validation_result)
                    results["type_analysis"].append(computed["analysis"])
                    if not validation_result.get("valid", False):
                        results["success"] = False
                    if any(
                        "[GNN-E002]" in str(error)
                        for error in validation_result.get("errors", [])
                    ):
                        b_orientation_failed = True

            check_file_work(deadline, cancel_token)
            results_file = output_dir / "type_check_results.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)

            from ..visualizer import generate_all_visualizations

            visual_embeddings = generate_all_visualizations(results, output_dir)
            if visual_embeddings:
                results["visual_embeddings"] = visual_embeddings

            check_file_work(deadline, cancel_token)
            summary = self._generate_type_check_summary(results)
            summary_file = output_dir / "type_check_summary.md"
            with open(summary_file, "w") as f:
                f.write(summary)

            summary_json = summarize_type_check_results(results)
            summary_json_file = output_dir / "type_check_summary.json"
            with open(summary_json_file, "w") as f:
                json.dump(summary_json, f, indent=2)

            check_file_work(deadline, cancel_token)
            if estimate_resources and (context is None or gnn_files):
                # Resource reports must use this step's immutable selection too.
                estimate_target = (
                    gnn_files[0].parent if context and gnn_files else target_dir
                )
                self._write_resource_estimates(estimate_target, output_dir, logger)

            check_file_work(deadline, cancel_token)
            if results["success"]:
                log_step_success(logger, "Type checking completed successfully")
                return True
            if hard_failure:
                log_step_error(logger, "Type checking failed")
                return False
            if strict and b_orientation_failed:
                log_step_error(
                    logger,
                    "Type checking failed (strict mode: B orientation "
                    "contradictions [GNN-E002] are errors)",
                )
                return False
            log_step_warning(
                logger,
                "Type checking completed with warnings (no GNN files found,"
                " or some files invalid)",
            )
            return 2

        except Exception as e:
            log_step_error(logger, "Type checking failed", error=str(e))
            return False

    def _compute_file(
        self, path: Path, verbose: bool, strict: bool, expected_hash: str | None
    ) -> dict[str, Any]:
        """Read once and compute both per-file products; never publish artifacts."""
        try:
            try:
                raw = path.read_bytes()
                if expected_hash and hashlib.sha256(raw).hexdigest() != expected_hash:
                    raise ValueError("Model source changed after run planning")
                # Match read_text's universal-newline behavior after hashing raw bytes.
                content = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
            except (OSError, UnicodeError) as exc:
                content = None
                validation = self._invalid_file_result(path, str(exc))
            else:
                validation = self.validate_single_gnn_file(
                    path, verbose, strict=strict, content=content
                )
            return {
                "validation": validation,
                "analysis": self._analyze_types(path, verbose, content=content),
            }
        except Exception as exc:
            return {
                "error": {
                    "file": str(path),
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            }

    def _write_resource_estimates(
        self, target_dir: Path, output_dir: Path, logger: logging.Logger
    ) -> None:
        """Run the resource estimator over ``target_dir`` and persist reports.

        Activated by the ``estimate_resources`` flag (the documented
        ``--estimate-resources`` Step 5 option). Estimation never turns a
        successful type-check run into a hard failure: its own exceptions
        are logged and swallowed so type checking results stay authoritative.
        """
        from ..estimation.estimator import GNNResourceEstimator

        resource_dir = output_dir / "resource_estimates"
        try:
            resource_dir.mkdir(parents=True, exist_ok=True)
            estimator = GNNResourceEstimator()
            estimator.estimate_from_directory(str(target_dir), recursive=True)
            estimator.generate_report(str(resource_dir))
        except Exception as e:
            logger.warning(f"Resource estimation failed: {e}")

    @staticmethod
    def _invalid_file_result(file_path: Path, error: str) -> Dict[str, Any]:
        """Canonical invalid-file dict for unreadable/unvalidatable specs."""
        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "valid": False,
            "errors": [error],
            "warnings": [],
            "type_issues": [],
            "validation_timestamp": datetime.now().isoformat(),
        }

    def validate_single_gnn_file(
        self,
        file_path: Path,
        verbose: bool = False,
        strict: bool | None = None,
        content: str | None = None,
    ) -> Dict[str, Any]:
        """Validate one GNN file on disk.

        Args:
            file_path: Path to the GNN spec.
            verbose: Enable verbose per-file logging.
            strict: Override :attr:`strict_mode` for this call (``None``
                means "use the instance default").
            content: Pre-read spec content; supplying it skips the file
                re-read (callers that already hold the content, e.g. the
                directory loop, pass it through to avoid a second read).
        """
        if content is None:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                return self._invalid_file_result(file_path, str(e))
        try:
            return self.validate_content(
                content, source_name=str(file_path), strict=strict
            )
        except Exception as e:
            return self._invalid_file_result(file_path, str(e))

    def validate_content(
        self,
        content: str,
        *,
        source_name: str = "<content>",
        strict: bool | None = None,
    ) -> Dict[str, Any]:
        """Validate GNN spec content without touching the filesystem.

        Pure entry point over a spec string — useful for MCP callers, in
        memory pipelines, and tests that do not want to materialise a file.
        ``validate_single_gnn_file`` delegates here after reading the file.
        The returned dict carries the canonical ``valid`` verdict key plus
        additive ``variables``/``connections``/``sections`` metadata so
        downstream report renderers have structured data to work with.
        """
        effective_strict = self.strict_mode if strict is None else strict
        validation_result: dict[str, Any] = {
            "file_path": source_name,
            "file_name": Path(source_name).name,
            "valid": True,
            "errors": [],
            "warnings": [],
            "type_issues": [],
            "validation_timestamp": datetime.now().isoformat(),
        }

        found_types = extract_types_from_content(content)

        for type_info in found_types:
            type_validation = validate_type(type_info)
            if not type_validation["valid"]:
                validation_result["type_issues"].append(type_validation)
                validation_result["errors"].append(
                    f"Type issue: {type_validation['message']}"
                )
                validation_result["valid"] = False

        consistency_check = check_type_consistency(found_types)
        if not consistency_check["consistent"]:
            validation_result["errors"].append(consistency_check["message"])
            validation_result["valid"] = False

        gnn_dims, dimension_diagnostics = extract_gnn_dimensions_with_diagnostics(
            content
        )
        if dimension_diagnostics:
            validation_result["errors"].extend(dimension_diagnostics)
            validation_result["valid"] = False
        if gnn_dims:
            b_evidence = extract_b_matrix_evidence(content)
            dim_check = validate_dimension_compatibility(
                gnn_dims, b_evidence=b_evidence, strict=effective_strict
            )
            validation_result["dimension_compatibility"] = dim_check
            if not dim_check["compatible"]:
                for issue in dim_check["issues"]:
                    validation_result["errors"].append(issue)
                validation_result["valid"] = False
            for warning in dim_check["warnings"]:
                validation_result["warnings"].append(warning)

        resources = estimate_file_resources(content)
        validation_result["resource_estimation"] = resources

        # Additive structured metadata for report renderers and tooling.
        validation_result["variables"] = [
            {
                "name": type_info["name"],
                "type": type_info["type"],
                "dimensions": gnn_dims.get(type_info["name"], []),
                "total_elements": int(math.prod(gnn_dims.get(type_info["name"], [1]))),
            }
            for type_info in found_types
        ]
        validation_result["variable_count"] = len(found_types)
        edges, _connection_diagnostics = parse_resource_connections(
            content, set(gnn_dims)
        )
        annotated_edges = [
            {
                **edge,
                "is_temporal": "+" in edge["source"] or "+" in edge["target"],
            }
            for edge in edges
        ]
        validation_result["connections"] = annotated_edges
        validation_result["connection_count"] = len(annotated_edges)
        validation_result["connection_types"] = {
            "directed": sum(1 for e in annotated_edges if e["type"] == "directed"),
            "undirected": sum(1 for e in annotated_edges if e["type"] == "undirected"),
            "temporal": sum(1 for e in annotated_edges if e["is_temporal"]),
        }
        validation_result["sections"] = section_presence(content)
        vars_map: VariableMap = {
            type_info["name"]: {"dimensions": gnn_dims.get(type_info["name"], [])}
            for type_info in found_types
        }
        equations_text = extract_markdown_section(content, "Equations")
        validation_result["model_complexity"] = calculate_complexity(
            vars_map, edges, equations_text
        )
        validation_result["model_type"] = classify_time_spec(content)
        validation_result["type_distribution"] = {
            type_info["type"]: sum(
                1 for other in found_types if other["type"] == type_info["type"]
            )
            for type_info in found_types
        }
        validation_result["time_dynamics"] = {
            "is_dynamic": detect_time_dynamics(content)
        }
        return validation_result

    def _analyze_types(
        self,
        file_path: Path,
        verbose: bool = False,
        content: str | None = None,
    ) -> Dict[str, Any]:
        """Analyze types in a GNN file.

        ``content`` (pre-read spec text) skips the file re-read; ``None``
        reads the file (a read failure yields the error dict).
        """
        try:
            if content is None:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

            types_found = extract_types_from_content(content)

            type_counts: dict[str, int] = {}
            for type_info in types_found:
                var_type = type_info["type"]
                type_counts[var_type] = type_counts.get(var_type, 0) + 1

            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "types_found": types_found,
                "type_distribution": type_counts,
                "total_variables": len(types_found),
                "analysis_timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "error": str(e),
                "analysis_timestamp": datetime.now().isoformat(),
            }

    def _generate_type_check_summary(self, results: Dict[str, Any]) -> str:
        """Generate a Markdown summary of type checking results."""
        summary = f"""# Type Check Summary

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Processing Results
- **Files Processed**: {results.get("processed_files", 0)}
- **Success**: {results.get("success", False)}
- **Errors**: {len(results.get("errors", []))}

## Validation Results
- **Files Validated**: {len(results.get("validation_results", []))}
- **Valid Files**: {sum(1 for r in results.get("validation_results", []) if r.get("valid", False))}
- **Invalid Files**: {sum(1 for r in results.get("validation_results", []) if not r.get("valid", False))}

## Type Analysis
- **Type Analyses**: {len(results.get("type_analysis", []))}
- **Total Variables**: {sum(a.get("total_variables", 0) for a in results.get("type_analysis", []))}

## Graphical Abstracts
"""
        visual_embeddings = results.get("visual_embeddings", [])
        if visual_embeddings:
            for embedding in visual_embeddings:
                summary += f"\n{embedding}\n"
        else:
            summary += "\n*No visual summaries could be generated.*\n"

        summary += """
## Error Summary
"""
        errors = results.get("errors", [])
        if errors:
            for error in errors:
                if isinstance(error, dict):
                    summary += f"- **{error.get('file', 'Unknown')}**: {error.get('error', 'Unknown error')}\n"
                else:
                    summary += f"- {error}\n"
        else:
            summary += "- No errors encountered\n"

        return summary


def _type_check_with_instance(
    checker: GNNTypeChecker, job: tuple[Path, bool, bool, str | None]
) -> dict[str, Any]:
    return checker._compute_file(*job)

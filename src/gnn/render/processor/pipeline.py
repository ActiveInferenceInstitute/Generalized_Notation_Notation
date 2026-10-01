#!/usr/bin/env python3
"""Render pipeline orchestration for GNN specifications.

``process_render`` walks a target directory, routes each GNN file through
the POMDP-aware processor (or the basic generator fallback), and publishes
the aggregated receipt plus overview documentation.
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from gnn.render.processor.parsing import (
    _render_succeeded,
    normalize_matrices,
    validate_pomdp_for_rendering,
)
from gnn.render.processor.receipts import (
    _render_file_identity,
    _write_render_receipt,
    parse_frameworks_selection,
)

logger = logging.getLogger(__name__)


def process_render(
    target_dir: Path,
    output_dir: Path,
    verbose: bool = False,
    frameworks: Optional[List[str]] = None,
    strict_validation: bool = True,
    strict_framework_success: bool = False,
    **kwargs: Any,
) -> Union[bool, int]:
    """
    Process render for GNN specifications with POMDP-aware processing.

    This enhanced processor:
    1. Extracts POMDP state spaces from GNN files
    2. Modularly injects them into framework-specific renderers
    3. Creates implementation-specific output subfolders
    4. Provides structured documentation and results

    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Output directory for rendered files
        verbose: Enable verbose logging
        frameworks: List of frameworks to render for (default: all)
        strict_validation: Enable strict POMDP validation

    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        logger.info(f"Processing GNN files in: {target_dir}")
        logger.info(f"Output directory: {output_dir}")

        frameworks, explicit_framework_request = parse_frameworks_selection(frameworks)
        strict_framework_success = (
            strict_framework_success or explicit_framework_request
        )

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Import POMDP processing capabilities
        try:
            from gnn.extract.pomdp_extractor import extract_pomdp_from_file
            from gnn.render.pomdp_processor import POMDPRenderProcessor

            pomdp_available = True
        except ImportError as e:
            logger.warning(f"POMDP processing modules not available: {e}")
            logger.info("Falling back to basic rendering")
            pomdp_available = False

        # Find GNN files
        from gnn.processing.discovery import is_model_source_path

        # Reuse discovery (apply default target dir like the named args), honoring
        # recursion so nested exemplar folders (discrete/, basics/, pomdp_gridworld/,
        # etc.) are found — not just top-level "*.md".
        recursive = bool(kwargs.get("recursive", True))
        gnn_files: list[Any] = []
        if recursive:
            for pattern in ["*.md", "*.json", "*.yaml", "*.yml"]:
                gnn_files.extend(target_dir.rglob(pattern))
        else:
            for pattern in ["*.md", "*.json", "*.yaml", "*.yml"]:
                gnn_files.extend(target_dir.glob(pattern))
        gnn_files = [path for path in gnn_files if is_model_source_path(path)]

        configuration = {
            "frameworks": sorted(frameworks) if frameworks else "all",
            "strict_validation": strict_validation,
            "strict_framework_success": strict_framework_success,
            "verbose": verbose,
            "pomdp_processing_available": pomdp_available,
            "options": {
                key: value
                for key, value in kwargs.items()
                if key not in {"run_id", "logger"}
            },
        }
        run_id = str(
            kwargs.get("run_id") or os.environ.get("GNN_RUN_ID") or uuid.uuid4().hex
        )
        if not gnn_files:
            logger.warning(f"No GNN files found in {target_dir}")
            if (output_dir / "render_processing_summary.json").exists():
                _write_render_receipt(
                    output_dir,
                    target_dir,
                    {},
                    configuration,
                    run_id,
                    "POMDP-aware rendering" if pomdp_available else "Basic rendering",
                )
            return 2
        source_identities = {
            str(path): _render_file_identity(path) for path in gnn_files
        }

        logger.info(f"Found {len(gnn_files)} GNN files to process")

        # Framework selection already normalized by parse_frameworks_selection.
        # frameworks=None means all registered frameworks.

        if frameworks:
            logger.info(f"Target frameworks: {frameworks}")
        else:
            logger.info("Target frameworks: all available")

        results: dict[Any, Any] = {}
        success_count = 0
        total_framework_successes = 0
        total_framework_attempts = 0
        failed_framework_renderings: list[dict[str, str]] = []
        unsupported_framework_renderings: list[dict[str, str]] = []

        if pomdp_available:
            # Use POMDP-aware processing
            for gnn_file in gnn_files:
                try:
                    logger.info(f"Processing: {gnn_file}")

                    # Extract POMDP state space from GNN file
                    pomdp_space = extract_pomdp_from_file(
                        gnn_file, strict_validation=strict_validation
                    )

                    if pomdp_space is None:
                        logger.warning(
                            f"Could not extract POMDP from {gnn_file}, trying basic rendering"
                        )
                        # Fall back to basic processing for this file
                        file_result = _process_single_gnn_file_basic(
                            gnn_file, output_dir, verbose, **kwargs
                        )
                        results[str(gnn_file)] = file_result
                        if file_result["success"]:
                            success_count += 1
                        continue

                    logger.info(
                        f"Extracted POMDP '{pomdp_space.model_name}' with {pomdp_space.num_states} states, {pomdp_space.num_observations} observations, {pomdp_space.num_actions} actions"
                    )

                    # Validate POMDP space
                    is_valid, validation_errors = validate_pomdp_for_rendering(
                        pomdp_space
                    )
                    if not is_valid:
                        if strict_validation:
                            err_msg = (
                                f"POMDP validation failed (strict); skipping render for {gnn_file.name}: "
                                f"{validation_errors}"
                            )
                            logger.error(err_msg)
                            results[str(gnn_file)] = {
                                "overall_success": False,
                                "framework_results": {},
                                "error": err_msg,
                                "validation_failed_strict": True,
                            }
                            continue
                        logger.warning(
                            f"POMDP validation failed for {gnn_file.name}: {validation_errors}; "
                            "continuing because strict_validation=False"
                        )

                    # Normalize matrices
                    pomdp_space = normalize_matrices(pomdp_space, logger)

                    # Create file-specific output directory
                    file_output_dir = output_dir / gnn_file.stem

                    # Create processor with file-specific directory
                    file_processor = POMDPRenderProcessor(file_output_dir)

                    # Process POMDP for all frameworks
                    processing_result = file_processor.process_pomdp_for_all_frameworks(
                        pomdp_space,
                        gnn_file_path=gnn_file,
                        frameworks=frameworks,
                        **kwargs,
                    )
                    processing_result["base_output_dir"] = str(file_output_dir)

                    results[str(gnn_file)] = processing_result

                    if processing_result["overall_success"]:
                        success_count += 1
                        logger.info(f"Successfully processed {gnn_file.name}")
                    else:
                        logger.error(f"Failed to process {gnn_file.name}")

                    # Count framework-level successes
                    for framework_name, result in processing_result[
                        "framework_results"
                    ].items():
                        if result.get("unsupported"):
                            unsupported_framework_renderings.append(
                                {
                                    "file": str(gnn_file),
                                    "framework": framework_name,
                                    "message": str(result.get("message", "")),
                                }
                            )
                            continue
                        total_framework_attempts += 1
                        if result["success"]:
                            total_framework_successes += 1
                        else:
                            failed_framework_renderings.append(
                                {
                                    "file": str(gnn_file),
                                    "framework": framework_name,
                                    "message": str(result.get("message", "")),
                                }
                            )

                except Exception as e:
                    error_msg = f"Error processing {gnn_file}: {e}"
                    logger.error(error_msg)
                    results[str(gnn_file)] = {
                        "overall_success": False,
                        "framework_results": {},
                        "error": error_msg,
                    }
        else:
            # Use basic rendering
            for gnn_file in gnn_files:
                try:
                    logger.info(f"Processing (basic): {gnn_file}")
                    file_result = _process_single_gnn_file_basic(
                        gnn_file, output_dir, verbose, **kwargs
                    )
                    results[str(gnn_file)] = file_result
                    if file_result["success"]:
                        success_count += 1
                        logger.info(f"Processed {gnn_file.name}")
                    else:
                        logger.error(f"Failed to process {gnn_file.name}")

                except Exception as e:
                    error_msg = f"Error processing {gnn_file}: {e}"
                    logger.error(error_msg)
                    results[str(gnn_file)] = {"success": False, "error": error_msg}

        for source, record in results.items():
            record["source_identity"] = source_identities[source]
        summary_file = output_dir / "render_processing_summary.json"
        summary = _write_render_receipt(
            output_dir,
            target_dir,
            results,
            configuration,
            run_id,
            "POMDP-aware rendering" if pomdp_available else "Basic rendering",
        )

        # Create overview documentation
        _create_overview_documentation(output_dir, summary)

        logger.info("Render processing completed")
        logger.info(f"Files: {success_count}/{len(gnn_files)} successful")
        if pomdp_available:
            logger.info(
                f"Framework renderings: {total_framework_successes}/{total_framework_attempts} successful ({summary['framework_success_rate']:.1f}%)"
            )
            if failed_framework_renderings:
                logger.warning(
                    "Framework render failures: %s",
                    "; ".join(
                        f"{Path(item['file']).name}:{item['framework']}"
                        for item in failed_framework_renderings[:10]
                    ),
                )
                if len(failed_framework_renderings) > 10:
                    logger.warning(
                        "Additional framework render failures omitted from log: %d",
                        len(failed_framework_renderings) - 10,
                    )
        logger.info(f"Summary saved to: {summary_file}")

        return _render_succeeded(
            success_count=success_count,
            total_files=len(gnn_files),
            total_framework_successes=total_framework_successes,
            total_framework_attempts=total_framework_attempts,
            strict_framework_success=strict_framework_success,
        )

    except Exception as e:
        logger.error(f"Render processing failed: {e}")
        return False


def _process_single_gnn_file_basic(
    gnn_file: Path, output_dir: Path, verbose: bool, **kwargs: Any
) -> Dict[str, Any]:
    """
    Basic processing for a single GNN file without POMDP extraction.

    Args:
        gnn_file: GNN file to process
        output_dir: Output directory
        verbose: Enable verbose logging
        **kwargs: Additional processing options

    Returns:
        Processing result dictionary
    """
    try:
        # Import basic generators
        from gnn.render.bnlearn import generate_bnlearn_code
        from gnn.render.generators import generate_discopy_code, generate_pymdp_code

        # Create basic model data from filename
        model_data: dict[str, Any] = {
            "model_name": gnn_file.stem,
            "variables": [],
            "connections": [],
        }

        # Create file-specific output directory
        file_output_dir = output_dir / gnn_file.stem
        file_output_dir.mkdir(parents=True, exist_ok=True)

        generated_files: list[Any] = []

        # Generate code for each framework
        frameworks: dict[str, Any] = {
            "pymdp": (generate_pymdp_code, ".py"),
            "discopy": (generate_discopy_code, ".py"),
            "bnlearn": (generate_bnlearn_code, ".py"),
        }

        for framework_name, (generator_func, extension) in frameworks.items():
            try:
                # Create framework subdirectory
                framework_dir = file_output_dir / framework_name
                framework_dir.mkdir(parents=True, exist_ok=True)

                # Generate code
                code = generator_func(model_data)
                if code:
                    output_file = (
                        framework_dir / f"{gnn_file.stem}_{framework_name}{extension}"
                    )
                    with open(output_file, "w") as f:
                        f.write(code)
                    generated_files.append(str(output_file))

            except Exception as e:
                logger.warning(
                    f"Failed to generate {framework_name} code for {gnn_file}: {e}"
                )

        return {
            "success": len(generated_files) > 0,
            "message": f"Generated {len(generated_files)} files"
            if generated_files
            else "No files generated",
            "generated_files": generated_files,
            "output_directory": str(file_output_dir),
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Basic processing failed: {e}",
            "generated_files": [],
        }


def _create_overview_documentation(output_dir: Path, summary: Dict[str, Any]) -> None:
    """
    Create overview documentation for the rendering results.

    Args:
        output_dir: Output directory
        summary: Processing summary data
    """
    try:
        doc_content = f"""# GNN Rendering Results

Generated: {summary["timestamp"]}
Processing Type: **{summary["processing_type"]}**

## Summary

- **Total Files**: {summary["total_files"]}
- **Successfully Processed**: {summary["successful_files"]}
- **Failed**: {summary["failed_files"]}
"""

        if summary["total_framework_attempts"] > 0:
            doc_content += f"""- **Framework Renderings**: {summary["successful_framework_renderings"]}/{summary["total_framework_attempts"]} ({summary["framework_success_rate"]:.1f}% success rate)
"""

        doc_content += f"""
## Configuration

- **Frameworks**: {summary["configuration"]["frameworks"]}
- **Strict Validation**: {summary["configuration"]["strict_validation"]}
- **Verbose**: {summary["configuration"]["verbose"]}
- **POMDP Processing**: {"✅ Available" if summary["configuration"].get("pomdp_processing_available", False) else "❌ Not Available"}

## File Results

"""

        for file_path, result in summary["file_results"].items():
            file_name = Path(file_path).name
            if result.get("overall_success", result.get("success", False)):
                doc_content += f"- ✅ **{file_name}** - Successfully processed\n"

                # Add framework details if available
                if "framework_results" in result:
                    for framework, framework_result in result[
                        "framework_results"
                    ].items():
                        status = "✅" if framework_result["success"] else "❌"
                        doc_content += f"  - {status} {framework}: {framework_result.get('message', 'N/A')}\n"
            else:
                error_msg = result.get("error", result.get("message", "Unknown error"))
                doc_content += f"- ❌ **{file_name}** - {error_msg}\n"

        doc_content += f"""

## Output Structure

The rendered files are organized in implementation-specific subfolders:

```
{output_dir}/
├── [model_name]/
│   ├── pymdp/              # PyMDP Python simulations
│   ├── rxinfer/            # RxInfer.jl Julia simulations
│   ├── activeinference_jl/ # ActiveInference.jl Julia simulations
│   ├── jax/                # JAX Python simulations
│   ├── discopy/            # DisCoPy categorical diagrams
│   ├── pytorch/            # PyTorch simulations
│   ├── numpyro/            # NumPyro simulations
│   ├── stan/               # Stan models
│   ├── bnlearn/            # Bayesian network scripts
│   ├── ngclearn/           # ngc-learn simulations
│   └── cpomdp/             # cpomdp continuous simulations (continuous models only)
└── render_processing_summary.json  # Detailed results
```

## Generated Files

Each framework subdirectory contains:
- Main simulation/diagram script
- Framework-specific README.md with model details
- Configuration files (if applicable)

## Next Steps

1. Navigate to specific framework directories to find generated code
2. Follow framework-specific READMEs for execution instructions  
3. Check the processing summary JSON for detailed results and any warnings

---

*Generated by GNN Render Processor v1.0*
"""

        doc_file = output_dir / "README.md"
        with open(doc_file, "w") as f:
            f.write(doc_content)

        logger.info(f"Created overview documentation: {doc_file}")

    except Exception as e:
        logger.warning(f"Failed to create overview documentation: {e}")

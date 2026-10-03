#!/usr/bin/env python3
"""
Pipeline file-processing subcommand handlers: run, validate, parse,
extract, render, and graph.

Each handler receives the parsed argparse namespace and returns a process
exit code per the CLI contract (0 success, 1 error, 2 warnings). Extracted
from ``cli.__init__``.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from .commands import (
    EXTRACT_DEFECT_EMPTY_SKELETON,
    EXTRACT_DEFECT_ERROR_STATUS,
    build_parse_payload,
    extract_payload_defect,
    find_render_artifact,
    render_processing_succeeded,
    run_validation_checks,
)
from .helpers import (
    EXIT_ERROR,
    EXIT_SUCCESS,
    EXIT_WARNING,
    _emit_json,
    _guard_input_file,
    _print_envelope,
    _print_extract_error,
    _render_yaml,
)

logger = logging.getLogger(__name__)


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute full pipeline."""
    original_argv = sys.argv
    try:
        from gnn.main import main as pipeline_main

        sys.argv = ["gnn"]
        extra_args: list[Any] = [
            "--target-dir",
            str(args.target_dir),
            "--output-dir",
            str(args.output_dir),
        ]
        if args.verbose:
            extra_args.append("--verbose")
        if args.log_format == "json":
            extra_args.extend(["--log-format", "json"])
        skipped_steps: set[int] = set(args.skip_steps or ())
        if args.skip_llm:
            skipped_steps.add(13)
        only_steps = set(getattr(args, "only_steps", None) or ())
        overlap = skipped_steps & only_steps
        if overlap:
            logger.error(
                "Steps cannot be both selected and skipped: %s", sorted(overlap)
            )
            return EXIT_ERROR
        if only_steps:
            extra_args.extend(
                ["--only-steps", ",".join(str(step) for step in sorted(only_steps))]
            )
        if skipped_steps:
            extra_args.extend(
                ["--skip-steps", ",".join(str(step) for step in sorted(skipped_steps))]
            )
        sys.argv.extend(extra_args)
        return pipeline_main()
    except ImportError as e:
        logger.error(f"Could not import pipeline: {e}")
        return EXIT_ERROR
    except Exception as e:
        logger.error("Pipeline execution failed: %s", e, exc_info=True)
        return EXIT_ERROR
    finally:
        sys.argv = original_argv


def _cmd_validate(args: argparse.Namespace) -> int:
    """Validate a GNN file."""
    is_json = getattr(args, "json", False)
    if not _guard_input_file(args.file, json_output=is_json, command="validate"):
        return EXIT_ERROR

    content = args.file.read_text(encoding="utf-8")
    file_name = str(args.file)

    outcome = run_validation_checks(content, file_name)
    errors = outcome.errors
    variables = outcome.variables
    connections = outcome.connections
    semantic = outcome.semantic

    # Output
    if errors:
        if is_json:
            _print_envelope(
                "warning" if not args.strict else "error",
                data={
                    "valid": False,
                    "semantic": semantic,
                    "errors": [
                        {
                            "code": e.code,
                            "message": e.message,
                            "line": e.line,
                            "file": e.file,
                        }
                        for e in errors
                    ],
                    "variables_count": len(variables),
                    "connections_count": len(connections),
                },
                error=f"{len(errors)} error(s) found",
                command="validate",
            )
        else:
            for e in errors:
                print(f"  {e}")
            if args.strict:
                print(f"\n❌ {len(errors)} error(s) found")
            else:
                print(f"\n⚠️ {len(errors)} warning(s) — pass --strict to fail")
        if args.strict:
            return EXIT_ERROR
        return EXIT_WARNING
    else:
        if is_json:
            _print_envelope(
                "success",
                data={
                    "valid": True,
                    "semantic": semantic,
                    "file": file_name,
                    "variables_count": len(variables),
                    "connections_count": len(connections),
                },
                command="validate",
            )
        else:
            print(
                f"✅ {file_name}: valid ({len(variables)} variables, {len(connections)} connections)"
            )
        return EXIT_SUCCESS


def _cmd_parse(args: argparse.Namespace) -> int:
    """Parse a GNN file and output JSON, YAML, or a summary."""
    is_json = getattr(args, "json", False)
    if not _guard_input_file(args.file, json_output=is_json, command="parse"):
        return EXIT_ERROR

    result, parse_errors = build_parse_payload(args.file)
    variables_count = len(result["variables"])
    connections_count = len(result["connections"])
    metadata = result["metadata"]

    if is_json:
        _print_envelope(
            "warning" if parse_errors else "success",
            data=result,
            error=f"{len(parse_errors)} parse warning(s)" if parse_errors else None,
            command="parse",
        )
    elif args.format == "summary":
        print(f"File: {args.file.name}")
        print(f"Variables: {variables_count}")
        print(f"Connections: {connections_count}")
        if metadata:
            print(f"Metadata: {', '.join(metadata.keys())}")
    elif args.format == "yaml":
        yaml_text = _render_yaml(result)
        if yaml_text is None:
            # PyYAML is an optional dependency: degrade to JSON, never crash.
            logger.warning("PyYAML not installed; emitting JSON instead")
            _emit_json(result)
        else:
            print(yaml_text.rstrip("\n"))
    else:
        _emit_json(result)
    if parse_errors and not is_json:
        logger.warning("Parsed %s with %d warning(s)", args.file, len(parse_errors))
    return EXIT_WARNING if parse_errors else EXIT_SUCCESS


def _cmd_extract(args: argparse.Namespace) -> int:
    """Extract the POMDP state space from a GNN file and print JSON."""
    is_json = getattr(args, "json", False)
    file_name = str(args.file)
    if not args.file.is_file():
        logger.error("GNN file not found or not a regular file: %s", file_name)
        if is_json:
            _print_envelope(
                "error",
                error={
                    "code": "GNN-CLI-001",
                    "message": f"GNN file not found or not a regular file: {file_name}",
                },
                command="extract",
            )
        else:
            _print_extract_error(
                "GNN-CLI-001",
                f"GNN file not found or not a regular file: {file_name}",
            )
        return EXIT_ERROR

    try:
        from gnn.extract import extract_to_json
    except ImportError as exc:
        logger.error("POMDP extractor unavailable: %s", exc)
        if is_json:
            _print_envelope(
                "error",
                error={
                    "code": "GNN-CLI-002",
                    "message": f"extractor unavailable: {exc}",
                },
                command="extract",
            )
        else:
            _print_extract_error("GNN-CLI-002", f"extractor unavailable: {exc}")
        return EXIT_ERROR

    try:
        payload = extract_to_json(
            args.file, strict_validation=args.strict, compact=args.compact
        )
    except Exception as exc:  # contract is non-raising; guard anyway
        logger.error("POMDP extraction failed: %s", exc)
        if is_json:
            _print_envelope(
                "error",
                error={
                    "code": "GNN-CLI-003",
                    "message": f"extraction failed: {exc}",
                },
                command="extract",
            )
        else:
            _print_extract_error("GNN-CLI-003", f"extraction failed: {exc}")
        return EXIT_ERROR

    try:
        payload_obj: Any = json.loads(payload)
    except (TypeError, ValueError):
        payload_obj = None
    if is_json and payload_obj is None:
        logger.error("Extractor output was not valid JSON: %s", file_name)
        _print_envelope(
            "error",
            error={
                "code": "extract_error",
                "message": "extractor output was not valid JSON",
            },
            command="extract",
        )
        return EXIT_ERROR
    defect = extract_payload_defect(payload_obj)
    if defect == EXTRACT_DEFECT_ERROR_STATUS:
        if is_json:
            detail = payload_obj.get("error") if isinstance(payload_obj, dict) else None
            code = "extract_error"
            message = "extraction payload reported error status"
            if isinstance(detail, dict):
                code = str(detail.get("code", code))
                message = str(detail.get("message", message))
            _print_envelope(
                "error",
                error={"code": code, "message": message},
                command="extract",
            )
        else:
            print(payload)
        return EXIT_ERROR
    if defect == EXTRACT_DEFECT_EMPTY_SKELETON:
        # Lenient extraction fabricates a default skeleton for files with
        # no GNN state-space content at all; surface that as an error.
        message = f"no POMDP state-space content found in {file_name}"
        if is_json:
            _print_envelope(
                "error",
                error={"code": "GNN-E000", "message": message},
                command="extract",
            )
        else:
            _print_extract_error("GNN-E000", message)
        return EXIT_ERROR
    if is_json:
        _print_envelope("success", data=payload_obj, command="extract")
    else:
        print(payload)
    return EXIT_SUCCESS


def _cmd_render(args: argparse.Namespace) -> int:
    """Render a GNN file to framework code."""
    is_json = getattr(args, "json", False)
    if not args.file.is_file():
        logger.error("GNN file not found or not a regular file: %s", args.file)
        if is_json:
            _print_envelope(
                "error",
                error={
                    "code": "render_error",
                    "message": f"GNN file not found or not a regular file: {args.file}",
                },
                command="render",
            )
        return EXIT_ERROR

    from gnn.render import process_render

    framework = str(args.framework)
    options = json.loads(getattr(args, "options", "{}"))
    if not isinstance(options, dict):
        raise ValueError("--options requires a JSON object")
    with tempfile.TemporaryDirectory(prefix="gnn-render-") as td:
        tmp_root = Path(td)
        input_dir = tmp_root / "input"
        input_dir.mkdir()
        shutil.copy2(args.file, input_dir / args.file.name)

        render_dir = (
            tmp_root / "render_output"
            if args.output
            else Path("output") / "11_render_output" / args.file.stem
        )
        ok = process_render(
            target_dir=input_dir,
            output_dir=render_dir,
            verbose=getattr(args, "verbose", False),
            frameworks=[framework],
            strict_validation=False,
            strict_framework_success=True,
            backend_options={framework: options},
        )

        if not render_processing_succeeded(ok):
            logger.error(
                "Render failed for %s using framework %s", args.file, framework
            )
            if is_json:
                _print_envelope(
                    "error",
                    error={
                        "code": "render_error",
                        "message": (
                            f"Render failed for {args.file} using framework {framework}"
                        ),
                    },
                    command="render",
                )
            return EXIT_ERROR

        artifact = find_render_artifact(render_dir, framework, require_current=True)
        if artifact is None:
            logger.error(
                "Render completed but no %s artifact was found in %s",
                framework,
                render_dir,
            )
            if is_json:
                _print_envelope(
                    "error",
                    error={
                        "code": "render_error",
                        "message": (
                            f"Render completed but no {framework} artifact "
                            f"was found in {render_dir}"
                        ),
                    },
                    command="render",
                )
            return EXIT_ERROR
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(artifact, args.output)

        info: dict[str, Any] = {
            "file": str(args.file),
            "framework": framework,
            "output": str(args.output) if args.output is not None else None,
            "render_dir": str(render_dir),
            "artifact": str(artifact) if artifact is not None else None,
        }
        if is_json:
            _print_envelope("success", data=info, command="render")
        else:
            destination = args.output if args.output is not None else render_dir
            print(f"Rendered {args.file} → {framework}: {destination}")
    return EXIT_SUCCESS


def _cmd_graph(args: argparse.Namespace) -> int:
    """Generate dependency graph from multi-model files."""
    is_json = getattr(args, "json", False)
    if not _guard_input_file(args.file, json_output=is_json, command="graph"):
        return EXIT_ERROR

    try:
        from gnn.multimodel.dep_graph import render_graph_from_file

        output = render_graph_from_file(str(args.file), output_format=args.format)
        if is_json:
            _print_envelope(
                "success",
                data={
                    "file": str(args.file),
                    "graph": output,
                    "format": args.format,
                },
                command="graph",
            )
        else:
            print(output)
    except ImportError as e:
        logger.error("Could not import graph generator: %s", e)
        if is_json:
            _print_envelope(
                "error",
                error=f"Could not import graph generator: {e}",
                command="graph",
            )
        return EXIT_ERROR
    return EXIT_SUCCESS

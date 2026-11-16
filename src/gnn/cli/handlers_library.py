#!/usr/bin/env python3
"""
Maintained-library subcommand handlers: templates, models, and pull.

Each handler receives the parsed argparse namespace and returns a process
exit code per the CLI contract (0 success, 1 error, 2 warnings). Extracted
from ``cli.__init__``.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

from .helpers import EXIT_ERROR, EXIT_SUCCESS, _emit_json, _print_envelope

logger = logging.getLogger(__name__)


def _cmd_templates(args: argparse.Namespace) -> int:
    """Inspect the maintained template library."""
    from gnn.operations.templates import list_templates, show_template

    is_json = getattr(args, "json", False)

    if getattr(args, "templates_command", None) in {None, "list"}:
        templates = list_templates()
        if is_json:
            _print_envelope(
                "success", data={"templates": templates}, command="templates"
            )
        else:
            _emit_json({"templates": templates})
        return EXIT_SUCCESS
    if args.templates_command == "show":
        try:
            tmpl = show_template(args.name)
            if is_json:
                _print_envelope("success", data={"template": tmpl}, command="templates")
            else:
                _emit_json({"template": tmpl})
        except KeyError as exc:
            logger.error("%s", exc)
            if is_json:
                _print_envelope("error", error=str(exc), command="templates")
            return EXIT_ERROR
        return EXIT_SUCCESS
    return EXIT_ERROR


def _cmd_models(args: argparse.Namespace) -> int:
    """Query and inspect model registry."""
    target_dir = getattr(args, "target_dir", Path("input/gnn_files"))
    query_ontology = getattr(args, "query_ontology", None)

    from gnn.model_registry import process_model_registry

    temp_out = Path(tempfile.gettempdir()) / "gnn_cli_models_registry"
    temp_out.mkdir(parents=True, exist_ok=True)
    res = process_model_registry(
        target_dir=target_dir,
        output_dir=temp_out,
        query_ontology=query_ontology,
    )
    matching = res.get("matching_models", [])

    if getattr(args, "json", False):
        _print_envelope(
            "success",
            data={
                "total_models": res.get("total_models", 0),
                "query_ontology": query_ontology,
                "matching_models": matching,
            },
            command="models",
        )
    else:
        print(f"📦 Found {len(matching)} matching model(s):")
        for m in matching:
            print(f"  - {m}")
    return EXIT_SUCCESS


def _cmd_pull(args: argparse.Namespace) -> int:
    """Copy a maintained template into an input directory."""
    from gnn.operations.templates import pull_template

    is_json = getattr(args, "json", False)
    try:
        result = pull_template(
            args.name,
            Path(args.output_dir),
            dry_run=bool(args.dry_run),
            overwrite=bool(args.overwrite),
        )
    except (KeyError, FileExistsError, FileNotFoundError, OSError) as exc:
        logger.error("%s", exc)
        if is_json:
            _print_envelope("error", error=str(exc), command="pull")
        return EXIT_ERROR

    if is_json:
        _print_envelope("success", data=result, command="pull")
    else:
        _emit_json(result)
    return EXIT_SUCCESS

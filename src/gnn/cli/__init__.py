#!/usr/bin/env python3
"""
GNN CLI — Unified command-line interface with subcommands.

Provides:
  gnn run        — Execute the full pipeline
  gnn validate   — Validate a GNN file
  gnn parse      — Parse and output JSON / YAML / summary
  gnn extract    — Extract POMDP state space as JSON
  gnn render     — Render a GNN file to a specific framework
  gnn report     — Generate pipeline report
  gnn reproduce  — Re-run from a previous run hash
  gnn preflight  — Run environment & config checks
  gnn health     — Show renderer & dependency status
  gnn serve      — Start long-running services (API surfaces or website)
  gnn templates  — Inspect maintained GNN templates
  gnn models     — Query and inspect the model registry
  gnn complexity — Static per-backend complexity bounds
  gnn benchmark  — Empirical cross-framework complexity benchmark
  gnn pull       — Copy a maintained template into an input directory
  gnn watch      — Monitor a directory and live-reparse on change
  gnn gui        — Run GUI processing (Step 22 artifacts or interactive servers)
  gnn mcp        — Inspect the MCP tool surface
  gnn lsp        — Launch Language Server

Exit-code contract: 0 = success, 1 = error, 2 = completed with warnings.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Final, List, Optional, cast

from gnn import __version__
from gnn.cli.handlers_complexity import (
    _cmd_benchmark,
    _cmd_complexity,
)
from gnn.cli.handlers_library import (
    _cmd_models,
    _cmd_pull,
    _cmd_templates,
)
from gnn.cli.handlers_ops import (
    _cmd_health,
    _cmd_preflight,
    _cmd_report,
    _cmd_reproduce,
)
from gnn.cli.handlers_pipeline import (
    _cmd_extract,
    _cmd_graph,
    _cmd_parse,
    _cmd_render,
    _cmd_run,
    _cmd_validate,
)
from gnn.cli.handlers_service import (
    _cmd_gui,
    _cmd_lsp,
    _cmd_mcp,
    _cmd_serve,
    _cmd_watch,
)
from gnn.cli.helpers import (
    EXIT_ERROR,
    EXIT_SUCCESS,
    EXIT_WARNING,
    CommandHandler,
    _emit_json,
    _ensure_src_on_path,
    _envelope,
    _guard_input_file,
    _print_envelope,
    _print_extract_error,
    _render_yaml,
    _setup_logging,
    _utf8_output_streams,
)
from gnn.cli.parser import build_parser

FEATURES: dict[str, Any] = {
    "subcommands": True,
    "pipeline_execution": True,
    "file_validation": True,
    "file_parsing": True,
    "render_dispatch": True,
    "lsp_launch": True,
}


logger = logging.getLogger(__name__)


#: Command name → handler attribute name in this module. Kept as a
#: module-level data table so tooling and tests can introspect the CLI
#: surface without importing handler objects, and so dispatch resolves
#: the attribute at call time (module-level monkeypatching still works).
COMMAND_HANDLERS: Final[dict[str, str]] = {
    "run": "_cmd_run",
    "validate": "_cmd_validate",
    "parse": "_cmd_parse",
    "extract": "_cmd_extract",
    "render": "_cmd_render",
    "report": "_cmd_report",
    "reproduce": "_cmd_reproduce",
    "preflight": "_cmd_preflight",
    "health": "_cmd_health",
    "serve": "_cmd_serve",
    "templates": "_cmd_templates",
    "models": "_cmd_models",
    "pull": "_cmd_pull",
    "lsp": "_cmd_lsp",
    "watch": "_cmd_watch",
    "graph": "_cmd_graph",
    "gui": "_cmd_gui",
    "mcp": "_cmd_mcp",
    "benchmark": "_cmd_benchmark",
    "complexity": "_cmd_complexity",
}


#: Sorted subcommand names exposed by this CLI (drives ``--help`` parity
#: checks and the ``cli.mcp`` tool listing).
SUBCOMMANDS: Final[tuple[str, ...]] = tuple(sorted(COMMAND_HANDLERS))


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint."""
    with _utf8_output_streams():
        parser = build_parser()
        args = parser.parse_args(argv)

        # Setup logging
        _setup_logging(verbose=bool(getattr(args, "verbose", False)))

        # Ensure src/ is on sys.path for all subcommands
        _ensure_src_on_path()

        if not args.command:
            parser.print_help()
            return EXIT_WARNING

        # Dispatch — resolve the handler attribute at call time so the table
        # stays introspectable while module-level monkeypatching still applies.
        handler_name = COMMAND_HANDLERS.get(args.command)
        handler = globals().get(handler_name) if handler_name else None
        if handler is None:
            parser.print_help()
            return EXIT_ERROR

        try:
            return cast("int", handler(args))
        except KeyboardInterrupt:
            logger.error("%s command interrupted", args.command)
            return EXIT_ERROR
        except Exception as exc:
            logger.error(
                "%s command failed: %s",
                args.command,
                exc,
                exc_info=bool(getattr(args, "verbose", False)),
            )
            return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())


def get_module_info() -> dict:
    """Return module metadata for composability and MCP discovery."""
    return {
        "name": "cli",
        "version": __version__,
        "description": "Unified command-line interface with subcommands",
        "features": FEATURES,
    }


__all__ = [
    "COMMAND_HANDLERS",
    "SUBCOMMANDS",
    "CommandHandler",
    "FEATURES",
    "__version__",
    "build_parser",
    "get_module_info",
    "main",
]

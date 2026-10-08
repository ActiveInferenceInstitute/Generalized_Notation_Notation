#!/usr/bin/env python3
"""
Argument parser construction for the GNN ``gnn`` command.

``build_parser`` assembles all 20 subcommands with their flags and choices;
the argparse type parsers validate step numbers and TCP ports. Extracted
from ``cli.__init__``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gnn.frameworks import RENDER_FRAMEWORKS


def _pipeline_step(value: str) -> int:
    """Parse one pipeline step number for argparse."""
    from gnn.pipeline.admission import validate_steps

    try:
        step = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "step must be an integer from 0 to 24"
        ) from exc
    try:
        validate_steps([step], field_name="steps")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return step


def _tcp_port(value: str) -> int:
    """Parse a valid TCP port for argparse."""
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``gnn`` argument parser with all 20 subcommands.

    Pure construction — no parsing side effects — so programmatic callers
    can introspect flags and choices without dispatching.
    """
    parser = argparse.ArgumentParser(
        prog="gnn",
        description="GNN Processing Pipeline — Command-line interface",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── gnn run ──────────────────────────────────────────────────────────────
    run_p = subparsers.add_parser("run", help="Execute the full pipeline")
    run_p.add_argument(
        "--target-dir", "-t", default="input/gnn_files", help="Input directory"
    )
    run_p.add_argument("--output-dir", "-o", default="output", help="Output directory")
    run_p.add_argument(
        "--skip-steps", nargs="*", type=_pipeline_step, help="Step numbers to skip"
    )
    run_p.add_argument(
        "--only-steps", nargs="*", type=_pipeline_step, help="Only run these steps"
    )
    run_p.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM step (alias for --skip-steps 13)",
    )
    run_p.add_argument(
        "--log-format",
        choices=["human", "json"],
        default="human",
        help="Output format for pipeline logs",
    )
    run_p.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    run_p.add_argument(
        "--parallel",
        action="store_true",
        help="Execute independent pipeline tiers concurrently",
    )
    run_p.add_argument(
        "--consolidated-steps",
        action="store_true",
        help="Run eligible steps in process under the same run contract",
    )

    # ── gnn validate ─────────────────────────────────────────────────────────
    validate_p = subparsers.add_parser("validate", help="Validate a GNN file")
    validate_p.add_argument("file", type=Path, help="GNN file to validate")
    validate_p.add_argument("--strict", action="store_true", help="Fail on warnings")
    validate_p.add_argument(
        "--json", action="store_true", help="Output standard JSON envelope"
    )

    # ── gnn parse ────────────────────────────────────────────────────────────
    parse_p = subparsers.add_parser("parse", help="Parse a GNN file and output JSON")
    parse_p.add_argument("file", type=Path, help="GNN file to parse")
    parse_p.add_argument(
        "--format", choices=["json", "yaml", "summary"], default="json"
    )
    parse_p.add_argument(
        "--json", action="store_true", help="Output standard JSON envelope"
    )

    # ── gnn extract ──────────────────────────────────────────────────────────
    extract_p = subparsers.add_parser(
        "extract", help="Extract POMDP state space from a GNN file as JSON"
    )
    extract_p.add_argument("file", type=Path, help="GNN file to extract")
    extract_strict_group = extract_p.add_mutually_exclusive_group()
    extract_strict_group.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        help="Fail on structural validation errors (default)",
    )
    extract_strict_group.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Lenient structural validation",
    )
    extract_p.set_defaults(strict=True)
    extract_p.add_argument(
        "--compact", action="store_true", help="Emit compact (single-line) JSON"
    )
    extract_p.add_argument(
        "--json", action="store_true", help="Output standard JSON envelope"
    )
    # ── gnn render ───────────────────────────────────────────────────────────
    render_p = subparsers.add_parser(
        "render", help="Render a GNN file to framework code"
    )
    render_p.add_argument("file", type=Path, help="GNN file to render")
    render_p.add_argument(
        "--framework",
        "-f",
        default="pymdp",
        choices=RENDER_FRAMEWORKS,
        help="Target framework",
    )
    render_p.add_argument("--output", "-o", type=Path, help="Output file path")
    render_p.add_argument(
        "--options",
        default="{}",
        help="JSON object of validated framework render options",
    )
    render_p.add_argument(
        "--json", action="store_true", help="Output standard JSON envelope"
    )

    # ── gnn report ───────────────────────────────────────────────────────────
    report_p = subparsers.add_parser("report", help="Generate pipeline report")
    report_p.add_argument(
        "--output-dir", "-o", default="output", help="Pipeline output directory"
    )
    report_p.add_argument(
        "--json", action="store_true", help="Output standard JSON envelope"
    )

    # ── gnn reproduce ────────────────────────────────────────────────────────
    reproduce_p = subparsers.add_parser(
        "reproduce", help="Re-run from a previous run hash"
    )
    reproduce_p.add_argument("run_hash", help="Run hash (12-char hex prefix)")
    reproduce_p.add_argument(
        "--history-dir",
        type=Path,
        default=Path("output/00_pipeline_summary/.history"),
        help="Directory containing index.json",
    )

    # ── gnn preflight ────────────────────────────────────────────────────────
    preflight_p = subparsers.add_parser(
        "preflight", help="Run environment & config checks"
    )
    preflight_p.add_argument(
        "--config", type=Path, default=None, help="Config file path"
    )
    preflight_p.add_argument(
        "--json", action="store_true", help="Output standard JSON envelope"
    )

    # ── gnn health ───────────────────────────────────────────────────────────
    health_p = subparsers.add_parser("health", help="Show renderer & dependency status")
    health_p.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when environment preflight reports errors",
    )
    health_p.add_argument(
        "--json", action="store_true", help="Output standard JSON envelope"
    )

    # ── gnn serve ────────────────────────────────────────────────────────────
    serve_p = subparsers.add_parser(
        "serve",
        help="Start long-running services (API surfaces or the generated website)",
    )
    serve_p.add_argument("--host", default="127.0.0.1", help="Bind host")
    serve_p.add_argument(
        "--port",
        type=_tcp_port,
        default=None,
        help="Bind port (default: 8000 for API surfaces, 8090 for the website surface)",
    )
    serve_p.add_argument(
        "--surface",
        choices=["runs", "jobs", "both", "website"],
        default="runs",
        help=(
            "Surface to start: runs (gnn.api.app), jobs (gnn.api.server), both "
            "(jobs on port+1), or website (static server over the generated "
            "pipeline output tree)"
        ),
    )
    serve_p.add_argument(
        "--root",
        default=None,
        help="Website output root directory (default: ./output)",
    )
    serve_p.add_argument(
        "--live-reload",
        action="store_true",
        help="Inject a live-reload poller into served HTML pages",
    )

    # ── gnn templates ───────────────────────────────────────────────────────
    templates_p = subparsers.add_parser("templates", help="Inspect template library")
    templates_sub = templates_p.add_subparsers(
        dest="templates_command", help="Template commands"
    )
    templates_list_p = templates_sub.add_parser("list", help="List available templates")
    templates_list_p.add_argument(
        "--json", action="store_true", help="Output standard JSON envelope"
    )
    templates_show_p = templates_sub.add_parser("show", help="Show one template")
    templates_show_p.add_argument("name", help="Template name")
    templates_show_p.add_argument(
        "--json", action="store_true", help="Output standard JSON envelope"
    )
    templates_p.add_argument(
        "--json", action="store_true", help="Output standard JSON envelope"
    )

    # ── gnn models ────────────────────────────────────────────────────────────
    models_p = subparsers.add_parser("models", help="Query and inspect model registry")
    models_sub = models_p.add_subparsers(
        dest="models_command", help="Model registry commands"
    )

    def _add_models_arguments(target: argparse.ArgumentParser) -> None:
        """Attach the shared model-registry query flags to a parser."""
        target.add_argument(
            "--target-dir",
            "-t",
            type=Path,
            default=Path("input/gnn_files"),
            help="Target directory containing GNN models",
        )
        target.add_argument(
            "--query-ontology",
            "-q",
            type=str,
            default=None,
            help="Filter registered models by ontology concept substring",
        )
        target.add_argument(
            "--json", action="store_true", help="Output standard JSON envelope"
        )

    models_list_p = models_sub.add_parser("list", help="List registered models")
    _add_models_arguments(models_list_p)
    _add_models_arguments(models_p)

    # ── gnn pull ────────────────────────────────────────────────────────────
    pull_p = subparsers.add_parser("pull", help="Copy a maintained GNN template")
    pull_p.add_argument("name", help="Template name")
    pull_p.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("input/gnn_files"),
        help="Directory to receive the template",
    )
    pull_p.add_argument("--dry-run", action="store_true", help="Report without copying")
    pull_p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace destination on checksum mismatch",
    )
    pull_p.add_argument(
        "--json", action="store_true", help="Output standard JSON envelope"
    )

    # ── gnn watch ────────────────────────────────────────────────────────────
    watch_p = subparsers.add_parser(
        "watch", help="Monitor directory and live-reparse on change"
    )
    watch_p.add_argument(
        "dir", type=Path, help="Directory to monitor (e.g. input/gnn_files/)"
    )

    # ── gnn graph ────────────────────────────────────────────────────────────
    graph_p = subparsers.add_parser(
        "graph", help="Generate dependency graph from multi-model files"
    )
    graph_p.add_argument("file", type=Path, help="GNN file to render")
    graph_p.add_argument(
        "--format", choices=["mermaid", "text"], default="mermaid", help="Output format"
    )
    graph_p.add_argument(
        "--json", action="store_true", help="Output standard JSON envelope"
    )

    # ── gnn gui ──────────────────────────────────────────────────────────────
    gui_p = subparsers.add_parser(
        "gui",
        help="Run Step 22 GUI processing (headless artifacts or interactive servers)",
    )
    gui_p.add_argument(
        "--target-dir", "-t", default="input/gnn_files", help="Input directory"
    )
    gui_p.add_argument("--output-dir", "-o", default="output", help="Output directory")
    gui_p.add_argument(
        "--gui-types",
        default="gui_1,gui_2",
        help="Comma-separated GUI types (gui_1, gui_2, gui_3, oxdraw)",
    )
    gui_p.add_argument(
        "--interactive", action="store_true", help="Launch interactive GUI servers"
    )
    gui_p.add_argument(
        "--open-browser", action="store_true", help="Open browser for interactive GUIs"
    )
    gui_p.add_argument(
        "--launch-editor",
        action="store_true",
        help="Launch oxdraw editor (interactive oxdraw GUI type)",
    )

    # ── gnn mcp ──────────────────────────────────────────────────────────────
    mcp_p = subparsers.add_parser("mcp", help="Inspect the MCP tool surface")
    mcp_sub = mcp_p.add_subparsers(dest="mcp_command", help="MCP commands")
    mcp_list_p = mcp_sub.add_parser("list", help="List registered MCP tools")
    mcp_list_p.add_argument(
        "--json", action="store_true", help="Output standard JSON envelope"
    )
    mcp_info_p = mcp_sub.add_parser("info", help="Inspect one MCP tool")
    mcp_info_p.add_argument("name", help="Registered MCP tool name")
    mcp_info_p.add_argument(
        "--json", action="store_true", help="Output standard JSON envelope"
    )
    mcp_p.add_argument(
        "--json", action="store_true", help="Output standard JSON envelope"
    )

    # ── gnn lsp ──────────────────────────────────────────────────────────────
    subparsers.add_parser("lsp", help="Launch GNN Language Server")

    # ── gnn complexity ───────────────────────────────────────────────────────
    complexity_p = subparsers.add_parser(
        "complexity", help="Static per-backend complexity bounds for GNN models"
    )
    complexity_p.add_argument(
        "path", type=Path, help="GNN model file or directory of models"
    )
    complexity_p.add_argument(
        "--json", action="store_true", help="Output standard JSON envelope"
    )
    complexity_p.add_argument(
        "--output", type=Path, help="Write the receipt JSON to this file"
    )

    # ── gnn benchmark ────────────────────────────────────────────────────────
    benchmark_p = subparsers.add_parser(
        "benchmark", help="Empirical cross-framework complexity benchmark"
    )
    benchmark_p.add_argument(
        "target_dir", type=Path, help="Directory containing the corpus to benchmark"
    )
    benchmark_p.add_argument(
        "--frameworks",
        default="all",
        help="Comma-separated frameworks to benchmark (default: all)",
    )
    benchmark_p.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Execution benchmark repeats per backend",
    )
    benchmark_p.add_argument(
        "--json", action="store_true", help="Output standard JSON envelope"
    )
    benchmark_p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/cross_framework"),
        help="Directory for benchmark and calibration receipts",
    )

    # Accept global verbosity after the selected command too, matching the
    # ordering used by the public CLI examples.
    for command_parser in [
        *subparsers.choices.values(),
        *templates_sub.choices.values(),
        *models_sub.choices.values(),
        *mcp_sub.choices.values(),
    ]:
        command_parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Enable verbose output",
        )

    return parser

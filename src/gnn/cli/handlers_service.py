#!/usr/bin/env python3
"""
Long-running service subcommand handlers: serve, watch, lsp, gui, and mcp.

Each handler receives the parsed argparse namespace and returns a process
exit code per the CLI contract (0 success, 1 error, 2 warnings). Extracted
from ``cli.__init__``.
"""

from __future__ import annotations

import argparse
import logging
from contextlib import ExitStack
from pathlib import Path
from threading import Event, Thread
from typing import Any

from .helpers import EXIT_ERROR, EXIT_SUCCESS, _print_envelope

logger = logging.getLogger(__name__)


def _serve_both(host: str, port: int) -> None:
    """Bind both API listeners and stop the peer when either server exits."""
    if not 1 <= port < 65535:
        raise RuntimeError("The both surface requires a base port from 1 to 65534")

    import uvicorn

    from gnn.api.app import create_app as create_runs_app
    from gnn.api.server import create_app as create_jobs_app

    runs = uvicorn.Server(uvicorn.Config(create_runs_app(), host=host, port=port))
    jobs = uvicorn.Server(uvicorn.Config(create_jobs_app(), host=host, port=port + 1))
    stopping = Event()
    jobs_errors: list[BaseException] = []
    with ExitStack() as stack:
        try:
            runs_socket = stack.enter_context(runs.config.bind_socket())
            jobs_socket = stack.enter_context(jobs.config.bind_socket())
        except SystemExit as exc:
            raise RuntimeError(
                f"Could not bind both API ports {port} and {port + 1}"
            ) from exc

        def run_jobs() -> None:
            try:
                jobs.run(sockets=[jobs_socket])
                if not stopping.is_set():
                    raise RuntimeError("Jobs API listener stopped unexpectedly")
            except BaseException as exc:
                jobs_errors.append(exc)
            finally:
                runs.should_exit = True

        worker = Thread(target=run_jobs, name="gnn-jobs-api")
        worker.start()
        try:
            runs.run(sockets=[runs_socket])
        except SystemExit as exc:
            raise RuntimeError("Runs API listener failed") from exc
        finally:
            stopping.set()
            jobs.should_exit = True
            worker.join()
        if jobs_errors:
            raise RuntimeError(
                f"Jobs API listener failed: {jobs_errors[0]}"
            ) from jobs_errors[0]
        if not runs.started or not jobs.started:
            raise RuntimeError("Both API listeners did not start")


def _cmd_serve(args: argparse.Namespace) -> int:
    """Start long-running services (API surfaces or the generated website)."""
    surface = str(getattr(args, "surface", "runs") or "runs")
    if surface == "website":
        from gnn.pipeline.config import DEFAULT_OUTPUT_DIR
        from gnn.website.serve import (
            DEFAULT_WEBSITE_PORT,
            WebsiteServerError,
            serve_website,
        )

        root = (
            Path(args.root) if getattr(args, "root", None) else Path(DEFAULT_OUTPUT_DIR)
        )
        port = args.port if args.port is not None else DEFAULT_WEBSITE_PORT
        try:
            serve_website(
                root,
                port=port,
                open_browser=False,
                live_reload=bool(getattr(args, "live_reload", False)),
                host=args.host,
            )
        except (WebsiteServerError, OSError) as exc:
            print(f"❌ Could not start website server: {exc}")
            return EXIT_ERROR
        return EXIT_SUCCESS
    website_only = [
        name
        for name, value in (
            ("live-reload", bool(getattr(args, "live_reload", False))),
            ("root", getattr(args, "root", None)),
        )
        if value
    ]
    if website_only:
        flags = ", ".join(f"--{name}" for name in website_only)
        print(f"❌ {flags} only valid with --surface website")
        return EXIT_ERROR
    port = args.port if args.port is not None else 8000
    try:
        from gnn.api.auth import require_secure_bind

        if not require_secure_bind(args.host):
            raise RuntimeError(
                f"Refusing to bind API server to non-loopback address {args.host!r} "
                "without authentication. Set GNN_API_KEY to enable API-key auth, "
                "or GNN_ALLOW_INSECURE_BIND=1 to explicitly accept the risk."
            )
        if surface == "jobs":
            from gnn.api.server import run_server

            run_server(host=args.host, port=port)
        elif surface == "both":
            _serve_both(args.host, port)
        else:
            from gnn.api.app import start_server

            start_server(host=args.host, port=port)
    except ImportError:
        print("❌ FastAPI not installed. Run: uv sync --extra api")
        return EXIT_ERROR
    except (OSError, RuntimeError) as exc:
        logger.error("Could not start API server: %s", exc)
        return EXIT_ERROR
    return EXIT_SUCCESS


def _cmd_lsp(args: argparse.Namespace) -> int:
    """Launch GNN Language Server."""
    try:
        from gnn.cli.lsp import start_lsp

        start_lsp()
    except ImportError as e:
        print(f"❌ Could not start LSP server: {e}")
        return EXIT_ERROR
    return EXIT_SUCCESS


def _cmd_watch(args: argparse.Namespace) -> int:
    """Monitor directory and live-reparse on change."""
    try:
        from gnn.cli.watcher import GNNWatcher

        watcher = GNNWatcher(watch_dir=args.dir)
        watcher.start()
    except ImportError as e:
        logger.error(f"Could not import watcher: {e}")
        return EXIT_ERROR
    return EXIT_SUCCESS


def _cmd_gui(args: argparse.Namespace) -> int:
    """Run Step 22 GUI processing (headless artifacts or interactive servers)."""
    try:
        from gnn.gui import process_gui

        success = process_gui(
            target_dir=Path(args.target_dir),
            output_dir=Path(args.output_dir),
            verbose=bool(getattr(args, "verbose", False)),
            gui_types=args.gui_types,
            interactive=args.interactive,
            open_browser=args.open_browser,
            launch_editor=args.launch_editor,
        )
    except ImportError as e:
        logger.error("Could not import GUI module: %s", e)
        return EXIT_ERROR
    return EXIT_SUCCESS if success else EXIT_ERROR


def _cmd_mcp(args: argparse.Namespace) -> int:
    """Inspect the MCP tool surface (list tools or show one tool)."""
    is_json = getattr(args, "json", False)
    try:
        from gnn.mcp import get_mcp_instance, initialize

        initialize()
        instance = get_mcp_instance()
    except ImportError as exc:
        logger.error("MCP tool surface unavailable: %s", exc)
        if is_json:
            _print_envelope(
                "error",
                error={"code": "mcp_unavailable", "message": str(exc)},
                command="mcp",
            )
        return EXIT_ERROR

    if getattr(args, "mcp_command", None) == "info":
        tool_info = instance.get_tool_info(str(args.name))
        if tool_info is None:
            message = f"Unknown MCP tool: {args.name}"
            logger.error("%s", message)
            if is_json:
                _print_envelope(
                    "error",
                    error={"code": "unknown_tool", "message": message},
                    command="mcp",
                )
            return EXIT_ERROR
        if is_json:
            _print_envelope("success", data=tool_info, command="mcp")
        else:
            logger.info(
                "%s — %s (%s)",
                tool_info.get("name", ""),
                tool_info.get("module", ""),
                tool_info.get("category", ""),
            )
            logger.info("%s", tool_info.get("description", ""))
        return EXIT_SUCCESS

    listed: list[dict[str, Any]] = []
    for entry in instance.list_available_tools():
        if isinstance(entry, dict):
            listed.append(
                {
                    "name": str(entry.get("name", "")),
                    "module": str(entry.get("module", "")),
                    "category": str(entry.get("category", "")),
                    "description": str(entry.get("description", "")),
                }
            )
        else:
            listed.append(
                {"name": str(entry), "module": "", "category": "", "description": ""}
            )
    listed.sort(key=lambda tool: str(tool["name"]))
    if is_json:
        _print_envelope(
            "success",
            data={"tools": listed, "total": len(listed)},
            command="mcp",
        )
    else:
        for tool in listed:
            logger.info("%s — %s (%s)", tool["name"], tool["module"], tool["category"])
        logger.info("Total: %d tools", len(listed))
    return EXIT_SUCCESS

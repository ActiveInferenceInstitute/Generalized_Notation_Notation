#!/usr/bin/env python3
"""
Shared CLI helpers for the GNN command-line interface.

Owns the standard JSON envelope builders, the shared missing-input guard,
root logging setup, the ``sys.path`` bootstrap for lazy subcommand imports,
the YAML rendering helper, and the exit-code contract constants. Extracted
from ``cli.__init__``.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Final, Optional

from gnn import __version__

logger = logging.getLogger(__name__)

EXIT_SUCCESS: Final = 0
EXIT_ERROR: Final = 1
EXIT_WARNING: Final = 2


#: Handler signature shared by every subcommand implementation.
CommandHandler = Callable[[argparse.Namespace], int]


@contextmanager
def _utf8_output_streams() -> Iterator[None]:
    """Use UTF-8 for one CLI call, retaining caller-owned stream settings."""
    originals: list[tuple[io.TextIOWrapper, str, str | None]] = []
    try:
        for stream in (sys.stdout, sys.stderr):
            if isinstance(stream, io.TextIOWrapper) and not stream.closed:
                encoding, errors = stream.encoding, stream.errors
                stream.reconfigure(encoding="utf-8", errors=errors)
                originals.append((stream, encoding, errors))
        yield
    finally:
        for stream, encoding, errors in reversed(originals):
            if not stream.closed:
                stream.reconfigure(encoding=encoding, errors=errors)


def _envelope(
    status: str,
    data: Any = None,
    error: Any = None,
    meta: Optional[dict[str, Any]] = None,
    command: Optional[str] = None,
) -> dict[str, Any]:
    """Format output matching the standard CLI JSON envelope schema.

    ``meta`` always carries ``version``; ``command`` (when known) and any
    explicit ``meta`` entries are merged additively on top.
    """
    resolved_meta: dict[str, Any] = {"version": __version__}
    if command:
        resolved_meta["command"] = command
    if meta:
        resolved_meta.update(meta)
    return {
        "status": status,
        "data": data if data is not None else {},
        "error": error,
        "meta": resolved_meta,
    }


def _emit_json(payload: dict[str, Any]) -> None:
    """Print a JSON payload with the CLI's standard indentation."""
    print(json.dumps(payload, indent=2))


def _print_envelope(
    status: str,
    data: Any = None,
    error: Any = None,
    command: Optional[str] = None,
) -> None:
    """Build and print one standard CLI envelope."""
    _emit_json(_envelope(status, data=data, error=error, command=command))


def _print_extract_error(code: str, message: str) -> None:
    """Emit the structured ``gnn extract`` error envelope.

    The extract contract pins ``error`` as an object with
    ``code``/``message``/``line``/``section`` keys.
    """
    _print_envelope(
        "error",
        error={"code": code, "message": message, "line": None, "section": None},
        command="extract",
    )


def _guard_input_file(path: Path, *, json_output: bool, command: str) -> bool:
    """Return False (after logging/reporting) when ``path`` is unreadable.

    Emits the standard string-error envelope when ``json_output`` is set.
    """
    if path.is_file():
        return True
    message = f"GNN file not found or not a regular file: {path}"
    logger.error("%s", message)
    if json_output:
        _print_envelope("error", error=message, command=command)
    return False


def _setup_logging(verbose: bool) -> None:
    """Configure root logging for one CLI invocation."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _ensure_src_on_path() -> None:
    """Ensure ``src/`` is on ``sys.path`` for lazy subcommand imports."""
    src_dir = Path(__file__).resolve().parents[3] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _render_yaml(payload: dict[str, Any]) -> Optional[str]:
    """Serialize ``payload`` to YAML text, or None when PyYAML is absent.

    PyYAML is an optional dependency (repo extra); the CLI degrades to
    JSON at the call site rather than failing the command.
    """
    try:
        import yaml
    except ImportError:
        return None
    return str(yaml.safe_dump(payload, default_flow_style=False, sort_keys=False))

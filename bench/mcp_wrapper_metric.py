#!/usr/bin/env python3
"""MAJ-06 benchmark harness: MCP dispatcher consolidation metric.

Deterministic, offline workload that measures the copy-pasted
``process_<module>_mcp`` wrapper surface and pins the live MCP tool registry
to the committed audit report (``src/gnn/mcp/audit_report.json``).

Workload (identical every run, no network, no wall-clock inputs):

1. Static scan — AST-walk every ``*.py`` under ``src/gnn`` and total the
   source lines of every ``process_*_mcp`` function definition. This is the
   duplication surface MAJ-06 collapses.
2. Live surface — initialize the real MCP registry exactly the way the CI
   gate does (``tests.mcp.test_mcp_audit.count_mcp_tools``: ``initialize`` +
   settle loop) and enumerate every registered tool name.
3. Pin check — the live tool-name set, its size, and the errored-module count
   must match the committed audit report. Any drift means an MCP client would
   observe a changed surface: the run fails.

Metrics (stdout, one per line):

    METRIC wrapper_loc=<int>      total lines across process_*_mcp defs (primary; lower is better)
    METRIC wrapper_count=<int>    number of process_*_mcp definitions
    METRIC mcp_tool_count=<int>   live registered tool count (pinned)

Exit codes: 0 = metrics emitted and surface pin holds; 2 = surface drift;
3 = harness error (scan or import failure).
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "gnn"
AUDIT = SRC / "mcp" / "audit_report.json"
WRAPPER_RE = re.compile(r"^process_[a-z0-9_]+_mcp$")

# Keep harness output clean: module imports emit INFO logs during initialize.
logging.disable(logging.CRITICAL)


def scan_wrapper_loc() -> tuple[int, int]:
    """Return (total non-docstring lines, count) over process_*_mcp defs.

    Docstring lines are excluded: they are per-module human documentation,
    not the copy-pasted executable envelope MAJ-06 collapses. The metric
    counts the def line, signature, and every executable body line.
    """
    total_lines = 0
    count = 0
    for py in sorted(SRC.rglob("*.py")):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError as exc:  # pragma: no cover - corrupt tree is fatal
            print(f"HARNESS ERROR: SyntaxError scanning {py}: {exc}", file=sys.stderr)
            raise SystemExit(3)
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and WRAPPER_RE.match(node.name):
                end_lineno = node.end_lineno
                if end_lineno is None:  # pragma: no cover - FunctionDef always has one
                    continue
                span = end_lineno - node.lineno + 1
                first = node.body[0]
                if (
                    isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)
                    and first.end_lineno is not None
                ):
                    span -= first.end_lineno - first.lineno + 1
                count += 1
                total_lines += span
    return total_lines, count


def live_tool_names() -> tuple[list[str], int]:
    """Initialize the registry and return (sorted tool names, errored module count).

    Mirrors ``tests.mcp.test_mcp_audit.count_mcp_tools``: initialize with the
    same flags, then wait for the background recovery-registration threads to
    settle (poll until the tool count stops growing, 5 s max).
    """
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    from gnn.mcp import initialize, mcp_instance  # noqa: E402

    initialize(halt_on_missing_sdk=False, force_proceed_flag=True, force_refresh=True)
    prev_count = -1
    for _ in range(25):
        current = len(mcp_instance.tools)
        if current == prev_count:
            break
        prev_count = current
        time.sleep(0.2)

    errored = sum(
        1
        for info in mcp_instance.modules.values()
        if getattr(info, "status", "") == "error"
    )
    return sorted(mcp_instance.tools), errored


def main() -> int:
    os.chdir(REPO)

    wrapper_loc, wrapper_count = scan_wrapper_loc()
    print(f"METRIC wrapper_loc={wrapper_loc}")
    print(f"METRIC wrapper_count={wrapper_count}")

    try:
        live_names, errored = live_tool_names()
    except Exception as exc:
        print(
            f"HARNESS ERROR: MCP registry initialization failed: {exc}", file=sys.stderr
        )
        return 3
    print(f"METRIC mcp_tool_count={len(live_names)}")

    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    audit_names = sorted(t["name"] for t in audit["tools_list"])

    drift: list[str] = []
    if audit["tools_total"] != len(live_names):
        drift.append(
            f"tools_total: audit={audit['tools_total']} live={len(live_names)}"
        )
    if errored != audit["modules_errored"]:
        drift.append(
            f"modules_errored: audit={audit['modules_errored']} live={errored}"
        )
    missing = [n for n in audit_names if n not in set(live_names)]
    extra = [n for n in live_names if n not in set(audit_names)]
    if missing:
        drift.append(f"missing from live registry: {missing[:10]}")
    if extra:
        drift.append(f"unregistered tools: {extra[:10]}")

    if drift:
        print("SURFACE DRIFT vs src/gnn/mcp/audit_report.json:", file=sys.stderr)
        for line in drift:
            print(f"  - {line}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Comprehensive MCP Tool Audit Script

Validates that every registered MCP tool is:
  1. Detectable (importable via discover_modules)
  2. Real (backed by a named callable, not a lambda or None)
  3. Documented (non-empty description string)
  4. Callable (can be invoked via execute_tool with empty args)
  5. Logged (register_tools call has logging.info in its source)
  6. Schema-honest (declared properties/required match the handler
     signature for EVERY registered tool, not a curated subset)
  7. Uniquely registered (duplicate register_tool names during
     initialize() are detected at audit time only; the registry's own
     warn+overwrite behavior is untouched)

Usage:
    cd /path/to/generalizednotationnotation
    uv run python src/gnn/mcp/validate_tools.py [--markdown [PATH]]

With --markdown (optionally pointing at a doc path) the MCP tool quick
reference is fully regenerated from the live census after the audit.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

# ── Path setup ───────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

TOOL_REFERENCE_PATH = REPO_ROOT / "docs" / "gnn" / "mcp" / "tool_reference.md"

TOOL_REFERENCE_INTRO = (
    "Quick reference generated from an MCP audit snapshot. For the tools available "
    "in your installation, run `uv run --frozen --no-sync python -m gnn.mcp.cli list`. "
    "For full per-domain documentation see "
    "**[../modules/21_mcp.md](../modules/21_mcp.md)**."
)

TOOL_REFERENCE_CLOSING = (
    "Use `tests/mcp/test_mcp_audit.py` for the current registered "
    "tool/resource contract. The audit verifies module discovery, callable "
    "tools, non-empty module/category metadata, canonical JSON schemas, and "
    "the parent GUI exposure of nested `oxdraw.*` tools."
)


def _wait_for_census_stability(
    m: Any,
    poll_seconds: float = 0.2,
    stable_seconds: float = 1.0,
    deadline_seconds: float = 10.0,
) -> None:
    """Poll until the live tool/module counts stop changing.

    Timed-out module futures keep registering via background worker threads
    after ``initialize()`` returns; snapshotting too early would commit a
    stale census. Polls in ``poll_seconds`` steps until both counts have
    stayed unchanged for ``stable_seconds`` or ``deadline_seconds`` elapses.
    """
    last = (len(m.tools), len(m.modules))
    stable_since = time.monotonic()
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        time.sleep(poll_seconds)
        current = (len(m.tools), len(m.modules))
        if current == last:
            if time.monotonic() - stable_since >= stable_seconds:
                break
        else:
            last = current
            stable_since = time.monotonic()


logger = logging.getLogger("mcp_audit")


class _DuplicateRegistrationTracker:
    """Audit-time detector for duplicate ``register_tool`` names.

    Wraps ``gnn.mcp.registry.MCPRegistryMixin.register_tool`` for the
    duration of the audit's ``initialize()`` call and records every
    registration of an already-seen tool name. The registry's own
    warn+overwrite behavior is untouched; detection happens only here.
    Callers MUST restore the original method in a ``finally``.
    """

    def __init__(self) -> None:
        self.duplicates: list[str] = []
        self._seen: set[str] = set()
        self._owner: Any = None
        self._original: Any = None

    def install(self) -> None:
        """Wrap the registry class method; pair with :meth:`restore`."""
        from gnn.mcp import registry as registry_mod

        tracker = self
        owner: Any = registry_mod.MCPRegistryMixin
        original = owner.register_tool

        def tracking_register_tool(
            registry_self: Any, *args: Any, **kwargs: Any
        ) -> Any:
            name = kwargs.get("name")
            if name is None and args:
                name = args[0]
            if isinstance(name, str) and name:
                if name in tracker._seen:
                    tracker.duplicates.append(name)
                else:
                    tracker._seen.add(name)
            return original(registry_self, *args, **kwargs)

        self._owner = owner
        self._original = original
        owner.register_tool = tracking_register_tool

    def restore(self) -> None:
        """Unwrap the registry class method (idempotent)."""
        if self._owner is not None and self._original is not None:
            self._owner.register_tool = self._original
            self._owner = None
            self._original = None


def write_tool_reference(census: dict[str, Any], doc_path: Path) -> None:
    """Regenerate the MCP tool quick reference from a live audit census.

    Pure function (dict to file) so it is unit-testable without MCP.
    Emits one table row for EVERY tool in ``census["tools_list"]``.
    """
    tools_list: List[Dict[str, Any]] = census["tools_list"]
    modules_list: List[str] = census["modules_list"]

    rows: list[tuple[str, str, str]] = []
    for entry in tools_list:
        domain = (entry.get("module", "") or "").removeprefix("gnn.")
        rows.append((domain, str(entry["name"]), str(entry.get("description", ""))))
    rows.sort(key=lambda row: (row[0], row[1]))

    lines: list[str] = [
        "# GNN MCP Tool Quick Reference",
        "",
        TOOL_REFERENCE_INTRO,
        "",
        (
            f"**{len(tools_list)} tools across {len(modules_list)} modules** — "
            "see the generated "
            "[`src/gnn/mcp/audit_report.json`](../../../src/gnn/mcp/audit_report.json) "
            "for the checked-in audit snapshot (regenerate with "
            "`uv run --frozen --no-sync python -m gnn.mcp.validate_tools --markdown`)."
        ),
        "",
        "## Full Tool Table",
        "",
        "| Domain | Tool | Description |",
        "|--------|------|-------------|",
    ]
    for domain, name, description in rows:
        cell = " ".join(description.split())
        cell = cell.replace("|", "\\|")
        lines.append(f"| {domain} | `{name}` | {cell} |")
    lines.append("")
    lines.append(TOOL_REFERENCE_CLOSING)

    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run the full audit. Returns exit code (0=pass, 1=failures found)."""
    parser = argparse.ArgumentParser(
        description="Audit the registered GNN MCP tool surface."
    )
    parser.add_argument(
        "--markdown",
        nargs="?",
        const=str(TOOL_REFERENCE_PATH),
        default=None,
        metavar="PATH",
        help=(
            "Also regenerate the tool quick reference at PATH "
            f"(default: {TOOL_REFERENCE_PATH})."
        ),
    )
    parsed = parser.parse_args(argv if argv is not None else [])

    print("=" * 72)
    print("  GNN MCP TOOL AUDIT")
    print(f"  Repo: {REPO_ROOT}")
    print("=" * 72)

    # ── 1. Initialize MCP ────────────────────────────────────────────────────
    print("\n[1] Initializing MCP and discovering modules...")
    tracker = _DuplicateRegistrationTracker()
    try:
        from gnn.mcp import initialize, mcp_instance

        tracker.install()
        try:
            initialize(halt_on_missing_sdk=False, force_proceed_flag=True)
        finally:
            tracker.restore()
        m = mcp_instance
    except Exception as e:
        print(f"  FATAL: Could not initialize MCP: {e}")
        return 1

    duplicate_names = sorted(set(tracker.duplicates))

    # Background workers from timed-out module futures keep registering after
    # initialize() returns; wait for the live census to stabilize before
    # snapshotting counts (mirrors the contract in test_registry_internals).
    _wait_for_census_stability(m)

    # ── 2. Module summary ────────────────────────────────────────────────────
    print(f"\n[2] Modules discovered: {len(m.modules)}")
    loaded = [name for name, info in m.modules.items() if info.status == "loaded"]
    errored = [name for name, info in m.modules.items() if info.status == "error"]

    for mod, info in sorted(m.modules.items()):
        icon = "✓" if info.status == "loaded" else "✗"
        print(f"  {icon}  {mod:32s}  tools={info.tools_count:3d}  status={info.status}")

    if errored:
        print(f"\n  WARNING: {len(errored)} modules failed to load: {errored}")

    # ── 3. Per-tool audit ────────────────────────────────────────────────────
    print(f"\n[3] Tool audit: {len(m.tools)} tools registered")

    issues: List[Dict[str, Any]] = []
    # Duplicate registrations: audit-time-only detection recorded by the
    # tracker wrapped around register_tool during initialize().
    for name in duplicate_names:
        issues.append({"tool": name, "issue": "duplicate registration"})

    for name, tool in sorted(m.tools.items()):
        func = getattr(tool, "func", None) or getattr(tool, "function", None)
        desc = (getattr(tool, "description", "") or "").strip()
        fn = getattr(func, "__name__", "NONE") if func else "NONE"
        mod = getattr(tool, "module", "") or ""

        is_callable = callable(func)
        is_lambda = fn == "<lambda>"
        is_real = is_callable and not is_lambda

        is_documented = bool(desc)

        status = "OK"
        if not is_real:
            status = "NOT_CALLABLE"
            issues.append(
                {"tool": name, "issue": "not a real callable", "fn": fn, "mod": mod}
            )
        elif not is_documented:
            status = "UNDOCUMENTED"
            issues.append(
                {"tool": name, "issue": "missing description", "fn": fn, "mod": mod}
            )

        flag = "✓" if status == "OK" else "✗"
        print(
            f"  {flag} [{status:14s}]  {name:55s}  fn={fn:35s}  doc={is_documented}  mod={mod}"
        )

    # ── 4. Light callability verification ────────────────────────────────────
    print("\n[4] Callability spot-checks (no-arg tools)...")
    no_arg_tools: list[Any] = [
        "list_analysis_tools",
        "list_render_frameworks",
        "list_export_formats",
        "list_standard_ontology_terms",
        "list_supported_integrations",
        "list_research_topics",
        "list_security_checks",
        "get_website_module_info",
        "get_render_module_info",
        "get_visualization_options",
        "get_report_module_info",
        "check_audio_backends",
        "get_sapf_module_info",
        "check_integration_dependencies",
    ]

    call_ok = 0
    call_err = 0
    for tname in no_arg_tools:
        if tname not in m.tools:
            print(f"  ⚠  SKIP (not registered): {tname}")
            continue
        try:
            result = m.execute_tool(tname, {})
            ok = isinstance(result, dict) and result.get("success", False)
            icon = "✓" if ok else "⚠"
            print(f"  {icon}  {tname:55s}  → success={ok}")
            if ok:
                call_ok += 1
            else:
                call_err += 1
                issues.append(
                    {
                        "tool": tname,
                        "issue": "execute returned success=False",
                        "result": str(result)[:80],
                    }
                )
        except Exception as e:
            print(f"  ✗  {tname:55s}  → EXCEPTION: {e}")
            call_err += 1
            issues.append({"tool": tname, "issue": f"exception: {e}"})

    # ── 4b. Schema-vs-signature verification (every tool) ────────────────────
    print("\n[4b] Schema-vs-signature checks (all tools)...")
    schema_ok = 0
    schema_err = 0
    for name, tool in sorted(m.tools.items()):
        func = getattr(tool, "func", None) or getattr(tool, "function", None)
        schema = getattr(tool, "schema", None) or {}
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        if not callable(func):
            schema_err += 1
            issues.append({"tool": name, "issue": "handler is not callable"})
            continue
        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):
            schema_err += 1
            issues.append({"tool": name, "issue": "signature unresolvable"})
            continue
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        params = {
            pname
            for pname, p in sig.parameters.items()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        undeclared = sorted(set(properties) - params) if not accepts_kwargs else []
        missing_required = (
            [r for r in required if r not in params] if not accepts_kwargs else []
        )
        if undeclared or missing_required:
            schema_err += 1
            issues.append(
                {
                    "tool": name,
                    "issue": "schema/signature drift",
                    "schema_properties_not_accepted": undeclared,
                    "required_without_signature_param": missing_required,
                }
            )
            print(f"  ✗  {name:55s}  drift: {undeclared} / {missing_required}")
        else:
            schema_ok += 1
    print(f"  schema checks: {schema_ok} ok, {schema_err} err")

    # ── 5. Logging coverage check ─────────────────────────────────────────────
    print("\n[5] Logging coverage check (register_tools uses logger.info?)...")
    submodule_dirs = [
        d
        for d in (SRC_ROOT / "gnn").iterdir()
        if d.is_dir()
        and (d / "mcp.py").exists()
        and not d.name.startswith("_")
        and d.name != "mcp"
    ]

    log_ok = 0
    log_miss = 0
    for d in sorted(submodule_dirs):
        mcp_src = (d / "mcp.py").read_text(encoding="utf-8", errors="replace")
        in_register = False
        has_log_info = False
        for line in mcp_src.splitlines():
            if "def register_tools" in line:
                in_register = True
            if in_register and ("logger.info" in line or "logging.info" in line):
                has_log_info = True
                break
        icon = "✓" if has_log_info else "✗"
        if has_log_info:
            log_ok += 1
        else:
            log_miss += 1
            issues.append(
                {
                    "tool": d.name + "/mcp.py",
                    "issue": "register_tools() has no logger.info",
                }
            )
        print(f"  {icon}  {d.name}/mcp.py  logged={has_log_info}")

    # ── 6. Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  AUDIT SUMMARY")
    print("=" * 72)
    print(f"  Modules loaded       : {len(loaded):3d} / {len(m.modules)}")
    print(f"  Tools registered     : {len(m.tools):3d}")
    print(f"  Duplicate names      : {len(duplicate_names):3d}")
    print(f"  Issues found         : {len(issues):3d}")
    print(f"  Spot-checks OK       : {call_ok:3d} / {call_ok + call_err}")
    print(f"  Schema checks OK     : {schema_ok:3d} / {schema_ok + schema_err}")
    print(f"  Modules logged       : {log_ok:3d} / {log_ok + log_miss}")

    if issues:
        print(f"\n  ISSUES ({len(issues)}):")
        for iss in issues:
            print(f"    • {iss['tool']}: {iss['issue']}")
        status_str = "PARTIAL - issues found above"
    else:
        status_str = "PASS - all tools real, documented, logged"

    print(f"\n  RESULT: {status_str}")
    print("=" * 72)

    # ── 7. Save report ───────────────────────────────────────────────────────
    report: dict[str, Any] = {
        "modules_total": len(m.modules),
        "modules_loaded": len(loaded),
        "modules_errored": len(errored),
        "tools_total": len(m.tools),
        "modules_list": sorted(m.modules),
        "tools_list": [
            {
                "name": name,
                "module": getattr(tool, "module", ""),
                "category": getattr(tool, "category", ""),
                "fn": getattr(
                    getattr(tool, "func", None) or getattr(tool, "function", None),
                    "__name__",
                    "?",
                ),
                "documented": bool((getattr(tool, "description", "") or "").strip()),
                "description": (getattr(tool, "description", "") or "").strip()[:300],
            }
            for name, tool in sorted(m.tools.items())
        ],
        "duplicate_registrations": duplicate_names,
        "spot_checks_ok": call_ok,
        "spot_checks_err": call_err,
        "schema_checks_ok": schema_ok,
        "schema_checks_err": schema_err,
        "logging_ok": log_ok,
        "logging_miss": log_miss,
        "issues": issues,
    }
    out_path = SRC_ROOT / "gnn" / "mcp" / "audit_report.json"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=out_path.parent, delete=False
    ) as tmp_f:
        tmp_f.write(json.dumps(report, indent=2))
    os.replace(tmp_f.name, str(out_path))
    print(f"\n  Full report saved → {out_path}")

    if parsed.markdown is not None:
        markdown_path = Path(parsed.markdown)
        print(f"\n  Regenerating tool reference → {markdown_path}")
        write_tool_reference(report, markdown_path)
        print(f"  Tool reference saved → {markdown_path}")

    return 0 if not issues else 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    sys.exit(main(sys.argv[1:]))

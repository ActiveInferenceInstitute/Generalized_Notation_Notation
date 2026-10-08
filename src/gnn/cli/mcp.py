"""
MCP registration for CLI module.

Provides tools for pipeline execution and health checking via MCP.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def cli_health_check(params: (Dict[str, Any]) | None = None) -> Dict[str, Any]:
    """Return CLI module health and available subcommands.

    The subcommand list is derived from the dispatcher's
    ``COMMAND_HANDLERS`` table in ``cli.__init__`` — one source of truth.
    """
    from gnn.cli import SUBCOMMANDS

    subcommands: list[str] = list(SUBCOMMANDS)
    return {
        "success": True,
        "module": "cli",
        "subcommands": subcommands,
        "subcommand_count": len(subcommands),
    }


def cli_preflight(params: (Dict[str, Any]) | None = None) -> Dict[str, Any]:
    """Run preflight checks for the pipeline environment."""
    try:
        from gnn.pipeline.preflight import run_preflight

        report = run_preflight()
        return {
            "success": report.is_ok,
            "checks_passed": report.checks_passed,
            "checks_failed": report.checks_failed,
            "optional_groups": getattr(report, "optional_groups", {}),
            "issues": [
                {
                    "category": issue.category,
                    "severity": issue.severity,
                    "message": issue.message,
                    "fix": issue.fix,
                }
                for issue in report.issues
            ],
        }
    except Exception as e:
        logger.error("CLI preflight MCP tool failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}


def register_tools(mcp_instance: Any) -> None:
    """Register CLI tools with the MCP instance."""
    mcp_instance.register_tool(
        "cli.health",
        cli_health_check,
        {"type": "object", "properties": {}},
        "Return CLI module health and list of available subcommands",
        module="cli",
        category="cli",
    )

    mcp_instance.register_tool(
        "cli.preflight",
        cli_preflight,
        {"type": "object", "properties": {}},
        "Run pipeline preflight checks and return explicit readiness diagnostics",
        module="cli",
        category="cli",
    )

    logger.info("cli module MCP tools registered (2 tools).")

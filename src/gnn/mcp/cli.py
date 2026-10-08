#!/usr/bin/env python3
"""
MCP Command Line Interface

Command-line access to GNN Model Context Protocol tools and resources.
Supports listing capabilities, executing tools, retrieving resources,
querying server status, and starting stdio/HTTP servers.
"""

import argparse
import importlib.util
import json
import logging
import time
from pathlib import Path
from typing import Any

from gnn.utils.logging_utils import setup_step_logging

# Configure logging
logger = logging.getLogger("mcp.cli")


def import_mcp() -> Any:
    """Import the MCP module dynamically."""
    try:
        from . import MCPError, initialize, mcp_instance

        return mcp_instance, initialize, MCPError
    except ImportError:
        # Try to import from path
        mcp_path = Path(__file__).parent / "mcp.py"
        if not mcp_path.exists():
            raise ImportError("MCP module not found") from None

        spec = importlib.util.spec_from_file_location("mcp", mcp_path)
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Could not load MCP module spec from {mcp_path}"
            ) from None
        mcp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mcp_module)

        return (
            mcp_module.mcp_instance,
            mcp_module.initialize,
            getattr(mcp_module, "MCPError", Exception),
        )


def _get_mcp() -> Any:
    """Import and initialize the MCP module, returning (mcp_instance, MCPError)."""
    mcp_instance, initialize, MCPError = import_mcp()
    initialize()
    return mcp_instance, MCPError


def _emit_json_machine(payload: Any) -> None:
    """Emit exactly one pure JSON machine document to stdout (json mode)."""
    print(json.dumps(payload, indent=2, default=str))


def _emit_json_error(operation: str, message: str) -> None:
    """Emit the standardized JSON error document to stdout (json mode)."""
    _emit_json_machine({"error": {"operation": operation, "message": message}})


def _cli_error(
    operation: str, e: Exception, args: Any, suggestions: bool = False
) -> None:
    """Shared CLI error handler — json mode prints one error document; human mode logs, then exits 1."""
    if getattr(args, "format", None) == "json":
        # json stdout contract: exactly one machine document (the error
        # document) on stdout with a nonzero exit code; no emoji, no log frames.
        _emit_json_error(operation, str(e))
        raise SystemExit(1)
    logger.error(f"Error {operation}: {e}")
    if getattr(args, "verbose", False):
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")

    # Use logger for user-facing error message
    logger.error(f"❌ Error: {e}")
    if suggestions:
        logger.info("💡 Suggestions:")
        logger.info("  - Check if MCP server is running")
        logger.info("  - Verify GNN modules are properly installed")
    raise SystemExit(1)


def list_capabilities(args: Any) -> Any:
    """Enhanced listing of all available MCP capabilities with better formatting."""
    try:
        mcp_instance, MCPError = _get_mcp()

        capabilities = mcp_instance.get_capabilities()

        if args.format == "json":
            # json stdout contract: exactly one machine document on stdout.
            _emit_json_machine(capabilities)
            return

        # Enhanced human-readable format
        logger.info("🚀 GNN MCP Server Capabilities")
        logger.info("=" * 50)

        server_info = capabilities.get("server", {})
        logger.info("\n📋 Server Information:")
        logger.info(f"  Name: {server_info.get('name', 'Unknown GNN MCP Server')}")
        logger.info(f"  Version: {server_info.get('version', 'Unknown')}")
        logger.info(
            f"  Description: {server_info.get('description', 'No description available')}"
        )

        # Get enhanced server status for more details
        try:
            status = mcp_instance.get_enhanced_server_status()
            health = status.get("health", {})
            logger.info(
                f"  Health: {health.get('status', 'unknown').upper()} (Score: {health.get('score', 0)}/100)"
            )
            logger.info(
                f"  Uptime: {status.get('server_info', {}).get('uptime_formatted', 'Unknown')}"
            )
        except (AttributeError, KeyError, TypeError):
            logger.info("  Health: Unable to determine")

        # Tools section
        tools = capabilities.get("tools", [])
        if tools:
            logger.info(f"\n🔧 Available Tools ({len(tools)}):")
            logger.info("-" * 30)

            # Group tools by category
            tools_by_category: dict[Any, Any] = {}
            for tool in tools:
                category = tool.get("category", "General")
                if category not in tools_by_category:
                    tools_by_category[category] = []
                tools_by_category[category].append(tool)

            for category, category_tools in tools_by_category.items():
                logger.info(f"\n  📂 {category} ({len(category_tools)} tools):")
                for tool in sorted(category_tools, key=lambda t: t["name"]):
                    experimental = (
                        " 🧪 EXPERIMENTAL" if tool.get("experimental") else ""
                    )
                    logger.info(f"    • {tool['name']}{experimental}")
                    if args.verbose:
                        logger.info(
                            f"      Description: {tool.get('description', 'No description')}"
                        )
                        logger.info(f"      Module: {tool.get('module', 'Unknown')}")
                        logger.info(f"      Version: {tool.get('version', '1.0.0')}")

        # Resources section
        resources = capabilities.get("resources", [])
        if resources:
            logger.info(f"\n📚 Available Resources ({len(resources)}):")
            logger.info("-" * 30)
            for resource in sorted(resources, key=lambda r: r["uri_template"]):
                logger.info(f"  • {resource['uri_template']}")
                if args.verbose:
                    logger.info(
                        f"    Description: {resource.get('description', 'No description')}"
                    )
                    logger.info(f"    Module: {resource.get('module', 'Unknown')}")

        # Summary
        total_tools = len(tools)
        total_resources = len(resources)
        logger.info(f"\n📊 Summary: {total_tools} tools, {total_resources} resources")

        if args.verbose:
            try:
                # Show performance summary
                status = mcp_instance.get_enhanced_server_status()
                perf = status.get("performance", {})
                logger.info("\n⚡ Performance:")
                logger.info(f"  Success Rate: {perf.get('success_rate', 0):.1%}")
                logger.info(
                    f"  Avg Execution Time: {perf.get('average_execution_time', 0):.3f}s"
                )
                logger.info(f"  Cache Hit Ratio: {perf.get('cache_hit_ratio', 0):.1%}")
            except (AttributeError, KeyError, TypeError) as e:
                logger.debug(f"Could not retrieve performance stats: {e}")

    except Exception as e:
        _cli_error("listing capabilities", e, args, suggestions=True)


def execute_tool(args: Any) -> Any:
    """Execute an MCP tool with enhanced parameter validation and error reporting."""
    try:
        mcp_instance, MCPError = _get_mcp()

        if args.tool_name not in mcp_instance.tools:
            if args.format == "json":
                # json stdout contract: failure as one error document, exit 1.
                _emit_json_error("executing tool", f"Tool '{args.tool_name}' not found")
                raise SystemExit(1)
            available_tools = list(mcp_instance.tools.keys())
            logger.error(f"Tool '{args.tool_name}' not found")
            logger.info("Available tools:")
            for tool in sorted(available_tools):
                logger.info(f"  • {tool}")
            raise SystemExit(1)

        tool = mcp_instance.tools[args.tool_name]

        params: Any = {}
        if args.params:
            try:
                params = json.loads(args.params)
            except json.JSONDecodeError as e:
                if args.format == "json":
                    _emit_json_error("executing tool", f"Invalid JSON parameters: {e}")
                    raise SystemExit(1) from e
                logger.error(f"Invalid JSON parameters: {e}")
                logger.info('Expected format: --params \'{"key": "value"}\'')
                raise SystemExit(1) from e

        if not isinstance(params, dict):
            if args.format == "json":
                _emit_json_error(
                    "executing tool",
                    f"Parameters must be a JSON object, got {type(params)}",
                )
                raise SystemExit(1)
            logger.error(f"Parameters must be a JSON object, got {type(params)}")
            raise SystemExit(1)

        if tool.schema and args.validate:
            try:
                required = tool.schema.get("required", [])
                for req in required:
                    if req not in params:
                        if args.format == "json":
                            _emit_json_error(
                                "executing tool",
                                f"Missing required parameter '{req}'",
                            )
                            raise SystemExit(1)
                        logger.error(f"Missing required parameter '{req}'")
                        logger.info(f"Required parameters: {required}")
                        raise SystemExit(1)
            except Exception as e:
                logger.warning(f"Schema validation failed: {e}")

        logger.info(f"🔧 Executing tool: {args.tool_name}")
        if args.verbose:
            logger.info(f"📝 Parameters: {json.dumps(params, indent=2)}")
            logger.info(f"🏷️  Module: {tool.module}")
            logger.info(f"📂 Category: {tool.category}")
            logger.info(f"📋 Description: {tool.description}")

        start_time = time.time()
        result = mcp_instance.execute_tool(args.tool_name, params)
        execution_time = time.time() - start_time

        if args.format == "json":
            _emit_json_machine(result)
        else:
            logger.info(f"\n✅ Tool executed successfully in {execution_time:.3f}s")
            logger.info("📊 Result:")
            logger.info(json.dumps(result, indent=2))

            if args.verbose:
                try:
                    stats = mcp_instance.get_tool_performance_stats(args.tool_name)
                    if stats:
                        logger.info("\n📈 Tool Statistics:")
                        logger.info(f"  Uses: {stats.get('execution_count', 0)}")
                        logger.info(
                            f"  Avg Time: {stats.get('average_execution_time', 0):.3f}s"
                        )
                        logger.info(
                            f"  Success Rate: {stats.get('success_rate', 0):.1%}"
                        )
                except (AttributeError, KeyError, TypeError) as e:
                    logger.debug(f"Could not retrieve tool stats: {e}")

    except MCPError as e:
        if args.format == "json":
            _emit_json_error("executing tool", str(e))
            raise SystemExit(1) from e
        logger.error(f"MCP Error: {e}")
        if args.verbose:
            logger.error(f"Error Code: {e.code}")
            logger.error(f"Error Data: {e.data}")
        raise SystemExit(1) from e
    except Exception as e:
        _cli_error("executing tool", e, args)


def get_resource(args: Any) -> Any:
    """Retrieve an MCP resource."""
    try:
        mcp_instance, MCPError = _get_mcp()

        result = mcp_instance.get_resource(args.uri)

        if args.format == "json":
            _emit_json_machine(result)
        else:
            # Human-readable format
            logger.info(f"Resource '{args.uri}' retrieved successfully:")
            logger.info(json.dumps(result, indent=2))

    except MCPError as e:
        if args.format == "json":
            _emit_json_error("retrieving resource", str(e))
            raise SystemExit(1) from e
        logger.error(f"MCP Error: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        _cli_error("retrieving resource", e, args)


def get_server_status(args: Any) -> Any:
    """Get detailed server status information."""
    try:
        mcp_instance, MCPError = _get_mcp()

        status = mcp_instance.get_server_status()

        if args.format == "json":
            _emit_json_machine(status)
        else:
            # Human-readable format
            logger.info("=== GNN MCP Server Status ===\n")
            logger.info(f"Status: {status.get('status', 'Unknown')}")
            logger.info(f"Uptime: {status.get('uptime_formatted', 'Unknown')}")
            logger.info(f"Request Count: {status.get('request_count', 0)}")
            logger.info(f"Error Count: {status.get('error_count', 0)}")
            logger.info(f"Error Rate: {status.get('error_rate', 0):.2%}")
            logger.info(f"Tools: {status.get('tools_count', 0)}")
            logger.info(f"Resources: {status.get('resources_count', 0)}")
            logger.info(
                f"Modules: {status.get('modules_count', 0)} loaded, {status.get('modules_failed', 0)} failed"
            )

            # Average execution times
            avg_times = status.get("avg_execution_times", {})
            if avg_times:
                logger.info("\nAverage Execution Times:")
                for tool, time_avg in avg_times.items():
                    logger.info(f"  {tool}: {time_avg:.3f}s")

    except Exception as e:
        _cli_error("getting server status", e, args)


def get_tool_info(args: Any) -> Any:
    """Get detailed information about a specific tool."""
    try:
        mcp_instance, MCPError = _get_mcp()

        tool_info = mcp_instance.tools.get(args.tool_name)
        if not tool_info:
            if args.format == "json":
                # json stdout contract: failure as one error document, exit 1.
                _emit_json_error(
                    "getting tool info", f"Tool '{args.tool_name}' not found"
                )
                raise SystemExit(1)
            available_tools = list(mcp_instance.tools.keys())
            logger.error(f"Tool '{args.tool_name}' not found")
            logger.info("Available tools:")
            for tool in sorted(available_tools):
                logger.info(f"  • {tool}")
            raise SystemExit(1)

        # Get enhanced tool information
        detailed_info = mcp_instance.get_tool_info(args.tool_name)
        if not detailed_info:
            detailed_info = {
                "name": tool_info.name,
                "description": tool_info.description,
                "schema": tool_info.schema,
                "module": tool_info.module,
                "category": tool_info.category,
                "version": tool_info.version,
                "experimental": tool_info.experimental,
            }

        if args.format == "json":
            _emit_json_machine(detailed_info)
        else:
            # Enhanced human-readable format
            logger.info(f"🔍 Tool Information: {detailed_info['name']}")
            logger.info("=" * 50)

            logger.info("\n📋 Basic Info:")
            logger.info(f"  Description: {detailed_info['description']}")
            logger.info(f"  Module: {detailed_info['module']}")
            logger.info(f"  Category: {detailed_info['category']}")
            logger.info(f"  Version: {detailed_info['version']}")

            if detailed_info.get("usage_count", 0) > 0:
                logger.info("\n📈 Usage Statistics:")
                logger.info(f"  Times Used: {detailed_info.get('usage_count', 0)}")
                logger.info(
                    f"  Avg Execution Time: {detailed_info.get('average_execution_time', 0):.3f}s"
                )
                logger.info(
                    f"  Success Rate: {detailed_info.get('success_rate', 0):.1%}"
                )

            logger.info("\n⚙️ Configuration:")
            logger.info(
                f"  Input Validation: {'Enabled' if detailed_info.get('input_validation', True) else 'Disabled'}"
            )
            logger.info(
                f"  Output Validation: {'Enabled' if detailed_info.get('output_validation', True) else 'Disabled'}"
            )
            logger.info(f"  Timeout: {detailed_info.get('timeout', 'None')}s")
            logger.info(f"  Max Concurrent: {detailed_info.get('max_concurrent', 1)}")
            logger.info(
                f"  Rate Limit: {detailed_info.get('rate_limit', 'None')} req/s"
            )
            logger.info(f"  Cache TTL: {detailed_info.get('cache_ttl', 'None')}s")

            if detailed_info.get("experimental"):
                logger.info("\n🧪 Status: EXPERIMENTAL")

            logger.info("\n📋 Schema:")
            logger.info(json.dumps(detailed_info["schema"], indent=2))

    except Exception as e:
        _cli_error("getting tool info", e, args)


def get_diagnostics(args: Any) -> Any:
    """Get comprehensive diagnostic information."""
    try:
        mcp_instance, MCPError = _get_mcp()

        # Get diagnostics using the new meta-tool
        try:
            result = mcp_instance.execute_tool("get_mcp_diagnostics", {})
            diagnostics = result.get("diagnostics", {})
            overall_health = result.get("overall_health", "unknown")
        except (AttributeError, KeyError, TypeError):
            # Recovery to basic diagnostics
            diagnostics = {
                "issues": [],
                "warnings": [],
                "recommendations": [],
            }
            overall_health = "unknown"
            # Keep the json payload defined even on the recovery path.
            result = {"diagnostics": diagnostics, "overall_health": overall_health}

        if args.format == "json":
            _emit_json_machine(result)
            return

        # Enhanced human-readable format
        logger.info("🔍 GNN MCP Server Diagnostics")
        logger.info("=" * 50)

        logger.info(f"\n🏥 Overall Health: {overall_health.upper()}")

        # Show issues
        issues = diagnostics.get("issues", [])
        if issues:
            logger.info(f"\n❌ Issues Found ({len(issues)}):")
            for issue in issues:
                logger.info(f"  • {issue}")
        else:
            logger.info("\n✅ No critical issues found")

        # Show warnings
        warnings = diagnostics.get("warnings", [])
        if warnings:
            logger.info(f"\n⚠️  Warnings ({len(warnings)}):")
            for warning in warnings:
                logger.info(f"  • {warning}")

        # Show recommendations
        recommendations = diagnostics.get("recommendations", [])
        if recommendations:
            logger.info(f"\n💡 Recommendations ({len(recommendations)}):")
            for rec in recommendations:
                logger.info(f"  • {rec}")

        # Show health checks
        health_checks = diagnostics.get("health_checks", {})
        if health_checks:
            logger.info("\n🔍 Health Checks:")
            for check_name, check_result in health_checks.items():
                status = "✅ PASS" if check_result else "❌ FAIL"
                logger.info(f"  • {check_name}: {status}")

        if args.verbose:
            # Show additional server stats
            try:
                status = mcp_instance.get_enhanced_server_status()
                perf = status.get("performance", {})

                logger.info("\n📊 Detailed Performance:")
                logger.info(f"  Total Requests: {perf.get('total_requests', 0)}")
                logger.info(f"  Success Rate: {perf.get('success_rate', 0):.1%}")
                logger.info(
                    f"  Avg Execution Time: {perf.get('average_execution_time', 0):.3f}s"
                )
                logger.info(f"  Cache Hit Ratio: {perf.get('cache_hit_ratio', 0):.1%}")
                logger.info(
                    f"  Active Connections: {perf.get('concurrent_requests', 0)}"
                )

                # Show module status
                modules = status.get("modules", {})
                if modules:
                    logger.info("\n📦 Module Status:")
                    for name, info in modules.items():
                        status_icon = "✅" if info.get("status") == "loaded" else "❌"
                        logger.info(
                            f"  {status_icon} {name}: {info.get('status', 'unknown')}"
                        )

            except Exception as e:
                logger.warning(f"Could not get detailed status: {e}")

    except Exception as e:
        _cli_error("getting diagnostics", e, args, suggestions=True)


def start_server(args: Any) -> Any:
    """Start the MCP server."""
    try:
        # Import specific server implementation based on transport type
        if args.transport == "stdio":
            from .server_stdio import start_stdio_server

            logger.info("Starting MCP stdio server...")
            start_stdio_server()
        elif args.transport == "http":
            from .server_http import start_http_server

            logger.info(f"Starting MCP HTTP server on {args.host}:{args.port}...")
            start_http_server(args.host, args.port)
        else:
            logger.error(f"Unsupported transport: {args.transport}")
            raise SystemExit(1)
    except ImportError as e:
        _cli_error("importing server implementation", e, args)
    except Exception as e:
        _cli_error("starting server", e, args)


def main() -> Any:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Model Context Protocol CLI for GNN",
        epilog="Example: python -m gnn.mcp.cli list --format human",
    )

    # Global options
    parser.add_argument(
        "--format",
        choices=["json", "human"],
        default="human",
        help="Output format (default: human)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    list_parser = subparsers.add_parser("list", help="List available capabilities")
    list_parser.set_defaults(func=list_capabilities)

    execute_parser = subparsers.add_parser(
        "execute", help="Execute a tool with enhanced validation"
    )
    execute_parser.add_argument("tool_name", help="Name of the tool to execute")
    execute_parser.add_argument("--params", help="JSON parameters for the tool")
    execute_parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate parameters against tool schema",
    )
    execute_parser.set_defaults(func=execute_tool)

    resource_parser = subparsers.add_parser("resource", help="Get a resource")
    resource_parser.add_argument("uri", help="URI of the resource to retrieve")
    resource_parser.set_defaults(func=get_resource)

    status_parser = subparsers.add_parser("status", help="Get server status")
    status_parser.set_defaults(func=get_server_status)

    info_parser = subparsers.add_parser("info", help="Get tool information")
    info_parser.add_argument("tool_name", help="Name of the tool")
    info_parser.set_defaults(func=get_tool_info)

    diagnostics_parser = subparsers.add_parser(
        "diagnostics", help="Get comprehensive diagnostic information"
    )
    diagnostics_parser.set_defaults(func=get_diagnostics)

    server_parser = subparsers.add_parser(
        "server", help="Start MCP server with health monitoring"
    )
    server_parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport mechanism to use (default: stdio)",
    )
    server_parser.add_argument(
        "--host", default="127.0.0.1", help="Host for HTTP server"
    )
    server_parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server"
    )
    server_parser.set_defaults(func=start_server)

    args = parser.parse_args()

    # Configure logging using GNN pipeline infrastructure
    log_format = "json" if args.format == "json" else "human"
    # Note: we might need a dedicated --log-format flag if they should be independent
    global logger
    logger = setup_step_logging("mcp.cli", verbose=args.verbose, log_format=log_format)

    if args.format == "json":
        # json stdout contract: stdout must carry exactly one JSON machine
        # document. PipelineLogger writes log frames to sys.stdout, so in json
        # mode the effective level is raised for this one-shot CLI process so
        # routine info/debug chatter cannot reach stdout; errors surface as
        # the {'error': {...}} document with exit 1. --verbose is a human-mode
        # concern and is ignored for stdout purity in json mode.
        # PipelineLogger pins handler levels/verbosity and installs its
        # StreamHandler(sys.stdout) on the root logger (the step logger itself
        # carries no handlers), so gate the root handlers too.
        for lg in (logger, logging.getLogger()):
            lg.setLevel(logging.ERROR)
        for handler in logging.getLogger().handlers + logger.handlers:
            handler.setLevel(logging.ERROR)

    if not hasattr(args, "func"):
        parser.print_help()
        raise SystemExit(1)

    args.func(args)


if __name__ == "__main__":
    main()

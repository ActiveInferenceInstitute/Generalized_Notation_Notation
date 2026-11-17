#!/usr/bin/env python3
"""
GNN MCP Inspector (Python Implementation)

This script provides a command-line interface to inspect and interact with
the GNN Model Context Protocol (MCP) server. It can launch the server
and send requests to it.

Inspired by the concept of npx @modelcontextprotocol/inspector.
"""

import argparse
import json
import queue
import shlex
import subprocess  # nosec B404
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, cast

from gnn.utils.logging_utils import setup_step_logging

logger = setup_step_logging("mcp_npx_inspector")

# --- Configuration ---
GNN_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON_EXECUTABLE = sys.executable  # Use the same python interpreter

# --- Helper Functions ---


def print_mcp_response(data: Any) -> Any:
    """Prints JSON data with indentation."""
    print(
        json.dumps(data, indent=2, sort_keys=True)
    )  # user-output: machine-parsed JSON on stdout


def read_server_output(process: Any, output_queue: Any, error_queue: Any) -> Any:
    """Reads stdout from the server process and puts lines into a queue."""
    if process.stdout:
        for line in iter(process.stdout.readline, ""):
            output_queue.put(line)
        process.stdout.close()


def read_server_errors(process: Any, error_queue: Any) -> Any:
    """Reads stderr from the server process and puts lines into a queue."""
    if process.stderr:
        for line in iter(process.stderr.readline, ""):
            error_queue.put(line)
        process.stderr.close()


class StdioMCPClient:
    """A simple client to interact with an MCP server over stdio."""

    def __init__(self, process: Any) -> None:
        """Initialize the instance."""
        self.process = process
        self.request_id_counter = 1
        self.response_timeout = 10  # seconds
        self.server_stdout_queue: queue.Queue[str] = queue.Queue()
        self.server_stderr_queue: queue.Queue[str] = queue.Queue()

        self.stdout_thread = threading.Thread(
            target=read_server_output,
            args=(self.process, self.server_stdout_queue, self.server_stderr_queue),
        )
        self.stderr_thread = threading.Thread(
            target=read_server_errors, args=(self.process, self.server_stderr_queue)
        )
        self.stdout_thread.daemon = True
        self.stderr_thread.daemon = True
        self.stdout_thread.start()
        self.stderr_thread.start()

    def _send_request(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle send request for internal callers."""
        if not self.process.stdin:
            raise OSError("Server stdin is not available.")

        request_id = f"inspector-{self.request_id_counter}"
        self.request_id_counter += 1

        rpc_request: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
            "id": request_id,
        }
        if params is not None:
            rpc_request["params"] = params

        request_str = json.dumps(rpc_request)
        logger.info(f"INSPECTOR -> SERVER: {request_str}")
        self.process.stdin.write(request_str + "\n")
        self.process.stdin.flush()

        # Wait for response
        start_time = time.time()
        while True:
            if time.time() - start_time > self.response_timeout:
                raise TimeoutError(
                    f"Timeout waiting for response to request ID {request_id}"
                )

            try:
                # Check stderr first for server-side issues unrelated to this request
                while not self.server_stderr_queue.empty():
                    err_line = self.server_stderr_queue.get_nowait().strip()
                    if err_line:  # Only print if it's not an empty line
                        logger.warning(f"SERVER (stderr): {err_line}")

                line = self.server_stdout_queue.get(
                    timeout=0.1
                )  # Check queue with timeout
                logger.info(f"SERVER -> INSPECTOR: {line.strip()}")
                response = json.loads(line)
                if response.get("id") == request_id:
                    return cast("dict[str, Any]", response)
                else:
                    # Might be a notification or unrelated message, log it and continue
                    logger.info(
                        f"INSPECTOR (info): Received unrelated message or notification: {response}",
                    )
            except queue.Empty:
                if self.process.poll() is not None:  # Server process terminated
                    raise ConnectionError(
                        "Server process terminated unexpectedly."
                    ) from None
                continue  # Timeout, try again
            except json.JSONDecodeError as e:
                logger.error(
                    f"INSPECTOR (error): Could not decode JSON from server: {line.strip()} - {e}",
                )
                # If it's a fatal error, we might not get a response with our ID.
                # This could be part of a multi-line error dump from the server.
                # For now, just log and continue waiting for our specific response ID.
            except Exception as e:
                logger.error(
                    f"INSPECTOR (error): Unexpected error reading server response: {e}",
                )
                raise  # Re-raise for now

    def get_capabilities(self) -> dict:
        # MCP standard often uses "mcp/discover" or similar,
        # but GNN server's meta_mcp.py registers "get_mcp_server_capabilities"
        """Return capabilities."""
        return self._send_request(method="get_mcp_server_capabilities")

    def execute_tool(self, tool_name: str, tool_params: dict) -> dict:
        # GNN MCP server expects tool name as method and params as params object
        """Execute tool."""
        return self._send_request(method=tool_name, params=tool_params)

    def get_resource(self, uri: str) -> dict:
        # Resources are read via the standard ``mcp.resource.get`` MCP method,
        # not via a raw URI as method. The earlier guess-work (sending the URI
        # as a method name) could never retrieve a resource.
        """Return resource."""
        return self._send_request(method="mcp.resource.get", params={"uri": uri})


# --- CLI Subcommands ---


def handle_list_capabilities(client: StdioMCPClient, args: Any) -> Any:
    """Handles the 'list-capabilities' command."""
    logger.info("Inspector: Requesting server capabilities...")
    try:
        response = client.get_capabilities()
        print_mcp_response(response)
    except Exception as e:
        logger.error(f"Error getting capabilities: {e}")
        if hasattr(e, "__cause__") and e.__cause__:
            logger.error(f"Cause: {e.__cause__}")


def handle_execute_tool(client: StdioMCPClient, args: Any) -> Any:
    """Handles the 'execute-tool' command."""
    tool_name = args.tool_name
    try:
        tool_params = json.loads(args.params) if args.params else {}
    except json.JSONDecodeError as e:
        logger.error(f"Error: Invalid JSON in --params: {e}")
        return

    logger.info(
        f"Inspector: Executing tool '{tool_name}' with params: {tool_params}",
    )
    try:
        response = client.execute_tool(tool_name, tool_params)
        print_mcp_response(response)
    except Exception as e:
        logger.error(f"Error executing tool '{tool_name}': {e}")


def handle_get_resource(client: StdioMCPClient, args: Any) -> Any:
    """Handles the 'get-resource' command."""
    uri = args.uri
    logger.info(f"Inspector: Attempting to get resource '{uri}'...")
    try:
        response = client.get_resource(
            uri
        )  # This might not work as expected with GNN MCP
        print_mcp_response(response)
    except Exception as e:
        logger.error(f"Error getting resource '{uri}': {e}")


# --- Main ---
def main() -> None:
    # This is the primary parser for the inspector tool itself.
    """Provide main behavior."""
    parser = argparse.ArgumentParser(
        description="GNN MCP Inspector. Launches and interacts with a GNN MCP server.",
        epilog=f'Example: python {sys.argv[0]} --server-cmd "{PYTHON_EXECUTABLE} -m gnn.mcp.cli server --transport stdio" list-capabilities',
    )
    parser.add_argument(
        "--server-cmd",
        help="Full command string to start the GNN MCP server. "
        "Example: 'python -m gnn.mcp.cli server --transport stdio'. "
        "If not provided, defaults to the stdio server via `python -m gnn.mcp.cli`.",
        default=f"{PYTHON_EXECUTABLE} -m gnn.mcp.cli server --transport stdio",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output from inspector and server.",
    )

    subparsers = parser.add_subparsers(
        dest="inspector_command", title="Inspector Commands", required=True
    )

    # List Capabilities
    list_parser = subparsers.add_parser(
        "list-capabilities", help="List all tools and resources from the server."
    )
    list_parser.set_defaults(func=handle_list_capabilities)

    # Execute Tool
    exec_parser = subparsers.add_parser(
        "execute-tool", help="Execute a specific tool on the server."
    )
    exec_parser.add_argument(
        "tool_name",
        help="The name of the tool to execute (e.g., get_mcp_server_status).",
    )
    exec_parser.add_argument(
        "--params",
        help='JSON string of parameters for the tool (e.g., \'{"key": "value"}\').',
        default="{}",
    )
    exec_parser.set_defaults(func=handle_execute_tool)

    # Get Resource (Experimental for GNN MCP) - Currently commented out as per previous structure
    # resource_parser = subparsers.add_parser("get-resource", help="Attempt to retrieve a resource by URI (experimental).")
    # resource_parser.add_argument("uri", help="The URI of the resource.")
    # resource_parser.set_defaults(func=handle_get_resource)

    args = parser.parse_args()

    server_cmd_str = args.server_cmd
    logger.info(f"Inspector: Using server command: {server_cmd_str}")

    # Prepare server command for subprocess
    # shlex.split is good for this if the command is a single string.
    # If it's already a list, use that.
    if isinstance(server_cmd_str, str):
        server_cmd_list = shlex.split(server_cmd_str)
    else:  # Assuming it could be pre-split if not default
        server_cmd_list = server_cmd_str

    # Sanity check: when the server command invokes a script file directly
    # (python <script> ...), verify the script exists before launching.
    # Module invocations (python -m gnn.mcp.cli ...) are resolved by the
    # interpreter itself, so argv[1] == "-m" skips this guard.
    if (
        len(server_cmd_list) > 1
        and server_cmd_list[0] == PYTHON_EXECUTABLE
        and server_cmd_list[1] != "-m"
        and not Path(server_cmd_list[1]).is_file()
    ):
        logger.error(
            f"Inspector Error: MCP CLI script not found at {server_cmd_list[1]}",
        )
        logger.error(
            "Please ensure GNN_PROJECT_ROOT is correct or provide full path in --server-cmd.",
        )
        sys.exit(1)

    if args.verbose:
        if (
            "--verbose" not in server_cmd_list and "server" in server_cmd_list
        ):  # Add verbose to server if not present
            try:
                server_idx = server_cmd_list.index("server")
                server_cmd_list.insert(
                    server_idx, "--verbose"
                )  # GNN MCP CLI uses -v or --verbose at main level
            except ValueError:
                # 'server' command not found, maybe it's a direct script call.
                # For simplicity, we assume the main CLI is used.
                logger.info(
                    "Inspector: --verbose not injected because the server subcommand was not found.",
                )
        logger.info(
            f"Inspector: Augmented server command for verbose: {' '.join(server_cmd_list)}",
        )

    server_process = None
    client = None
    try:
        logger.info("Inspector: Starting GNN MCP server process...")
        server_process = subprocess.Popen(  # nosec B603
            server_cmd_list,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # Capture server's stderr separately
            text=True,  # Work with text streams
            cwd=GNN_PROJECT_ROOT,  # Run from project root
        )

        # Give server a moment to start, especially if it logs to stderr/stdout on startup
        time.sleep(1 if "stdio" in server_cmd_str else 3)  # Longer for http potentially

        if server_process.poll() is not None:
            logger.error(
                f"Inspector Error: Server process terminated prematurely (exit code {server_process.returncode}).",
            )
            logger.info("--- Server stderr (if any) ---")
            if server_process.stderr:
                for line in server_process.stderr:
                    logger.warning(line.strip())
            logger.info("-----------------------------")
            sys.exit(1)

        logger.info(
            "Inspector: Server process started. Initializing client...",
        )
        if "stdio" in server_cmd_str:
            client = StdioMCPClient(server_process)
        elif "http" in server_cmd_str:
            logger.error(
                "Inspector Error: HTTP transport is unsupported by this inspector.",
            )
            logger.error(
                "Please use stdio transport for the server with this inspector version.",
            )
            sys.exit(1)
        else:
            logger.error(
                "Inspector Error: Could not determine server transport from command. Assuming stdio.",
            )
            client = StdioMCPClient(server_process)

        # Execute the inspector command
        if hasattr(args, "func"):
            args.func(client, args)

    except FileNotFoundError:
        logger.error(
            f"Inspector Error: Could not find server command '{server_cmd_list[0]}'. Is it in PATH or path correct?",
        )
    except ConnectionError as e:
        logger.error(f"Inspector Error: Connection to server failed: {e}")
    except TimeoutError as e:
        logger.error(f"Inspector Error: Timeout communicating with server: {e}")
    except Exception as e:
        logger.error(f"Inspector: An unexpected error occurred: {e}")
        logger.error(f"Details: {type(e).__name__}: {e.args}")

    finally:
        if server_process:
            logger.info("Inspector: Shutting down server process...")
            if server_process.stdin:
                server_process.stdin.close()  # Signal EOF to server if it's reading stdin

            # Give threads a chance to process remaining output
            if client and client.stdout_thread.is_alive():
                client.stdout_thread.join(timeout=0.5)
            if client and client.stderr_thread.is_alive():
                client.stderr_thread.join(timeout=0.5)

            if server_process.poll() is None:  # If still running
                server_process.terminate()
                try:
                    server_process.wait(timeout=2)  # Wait for termination
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "Inspector: Server did not terminate gracefully, killing.",
                    )
                    server_process.kill()
            logger.info("Inspector: Server process shut down.")

            # Drain any remaining output from queues (after threads might have exited)
            if client:
                logger.info("--- Remaining Server Stdout ---")
                while not client.server_stdout_queue.empty():
                    logger.info(client.server_stdout_queue.get_nowait().strip())
                logger.info("--- Remaining Server Stderr ---")
                while not client.server_stderr_queue.empty():
                    logger.info(client.server_stderr_queue.get_nowait().strip())
                logger.info("-----------------------------")


if __name__ == "__main__":
    main()

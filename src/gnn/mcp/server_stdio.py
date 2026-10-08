#!/usr/bin/env python3
"""
Enhanced MCP Stdio Server Implementation

This module provides a robust JSON-RPC 2.0 stdio server for the Model Context Protocol (MCP),
exposing all registered GNN tools and resources via standard input/output streams.

Key Features:
- Enhanced error handling with custom MCP error codes and JSON-RPC compliance
- Multi-threaded architecture for concurrent request processing
- Comprehensive logging and request/response tracking
- Graceful shutdown and resource cleanup
- Connection health monitoring and automatic recovery
- Extensible for meta-tools and future MCP extensions
- Performance monitoring and metrics collection
"""

import json
import logging
import os
import queue
import sys
import threading
import time
from typing import Any, Dict

# Configure logging
logger = logging.getLogger(__name__)

try:
    from . import MCPError, initialize, mcp_instance
except ImportError:  # pragma: no cover - direct-script fallback
    from gnn.mcp import MCPError, initialize, mcp_instance

from .jsonrpc import (
    INTERNAL_ERROR,
    INVALID_REQUEST,
    jsonrpc_error,
    jsonrpc_result,
    serialize_response,
    validate_request,
)
from .server_core import MCPServer, is_standard_method

# Bounded stdin framing: one line longer than MAX_LINE_BYTES is a protocol
# violation — emit a JSON-RPC error envelope, report on stderr, and exit
# cleanly (a length-unbounded byte stream cannot be resynced mid-frame).
MAX_LINE_BYTES = 1024 * 1024
_STDIO_READ_CHUNK_BYTES = 65536

_NOTIFICATION = object()


class StdioServer:
    """
    A Model Context Protocol server implementation using stdio transport.

    This server reads JSON-RPC 2.0 requests from stdin and writes responses to stdout,
    supporting both standard MCP methods and direct tool invocation.
    """

    def __init__(
        self, max_queue_size: int = 1000, request_timeout: float = 30.0
    ) -> None:
        """Initialize the stdio server with enhanced queue and thread management.

        Args:
            max_queue_size: Maximum size of request/response queues
            request_timeout: Reserved for future per-request enforcement.
                Not yet read by any code path; per-tool timeouts are enforced
                at the registry layer via ``MCPTool.timeout`` (wave-2 MAJ-09).
        """
        self.running = False
        self.request_queue: queue.Queue[dict[str, Any]] = queue.Queue(
            maxsize=max_queue_size
        )
        self.response_queue: queue.Queue[dict[str, Any]] = queue.Queue(
            maxsize=max_queue_size
        )
        self.request_timeout = request_timeout
        self._start_time = time.time()
        self._stdin_pending = bytearray()
        self._processor_finished = threading.Event()
        # Standalone writer use has no processor to wait for. start() clears
        # this event while the real server owns the processing thread.
        self._processor_finished.set()

        # Connection monitoring
        self._connection_errors = 0
        self._max_connection_errors = 10
        self._last_activity = time.time()

        # Performance tracking
        self._requests_processed = 0
        self._responses_sent = 0
        self._errors_encountered = 0

    def start(self) -> None:
        """Start the server with reader, processor, and writer threads."""
        self._stdin_pending.clear()
        self._processor_finished.clear()
        self.running = True

        # Initialize MCP
        initialize()
        logger.info("MCP stdio server initialized and ready")

        # Start reader and writer threads
        reader_thread = threading.Thread(target=self._reader_thread)
        writer_thread = threading.Thread(target=self._writer_thread)
        processor_thread = threading.Thread(target=self._processor_thread)

        reader_thread.daemon = True
        writer_thread.daemon = True
        processor_thread.daemon = True

        reader_thread.start()
        writer_thread.start()
        processor_thread.start()

        # Wait for threads to exit
        try:
            while self.running:
                reader_thread.join(0.1)
                if not reader_thread.is_alive():
                    self.running = False
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, stopping server")
            self.running = False

        writer_thread.join()
        processor_thread.join()
        logger.info("MCP stdio server stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Get server performance statistics."""
        return {
            "running": self.running,
            "requests_processed": self._requests_processed,
            "responses_sent": self._responses_sent,
            "errors_encountered": self._errors_encountered,
            "connection_errors": self._connection_errors,
            "last_activity": self._last_activity,
            "uptime": time.time() - self._start_time
            if hasattr(self, "_start_time")
            else 0,
            "queue_sizes": {
                "requests": self.request_queue.qsize(),
                "responses": self.response_queue.qsize(),
            },
        }

    def _readline_bounded(self) -> bytes | None:
        """Read one newline-terminated line from stdin with a hard size cap.

        Reads ``os.read(0, _STDIO_READ_CHUNK_BYTES)`` chunks until a newline
        or EOF, retaining bytes after the newline for the next frame in this
        server instance. The cap applies to one frame, including a final
        unterminated frame, rather than the size of a coalesced read.
        Returns the line without its trailing newline, ``b""`` at EOF, or
        ``None`` on overflow (the caller terminates the server).
        """
        pending = self._stdin_pending
        while True:
            newline = pending.find(b"\n")
            if newline != -1:
                if newline > MAX_LINE_BYTES:
                    pending.clear()
                    return None
                line = bytes(pending[:newline])
                del pending[: newline + 1]
                return line
            if len(pending) > MAX_LINE_BYTES:
                pending.clear()
                return None
            chunk = os.read(0, _STDIO_READ_CHUNK_BYTES)
            if not chunk:
                line = bytes(pending)
                pending.clear()
                return line
            pending.extend(chunk)

    def _reader_thread(self) -> Any:
        """Enhanced thread that reads JSON-RPC messages from stdin with connection monitoring."""
        try:
            while self.running:
                try:
                    line = self._readline_bounded()
                    if line is None:
                        # Oversized line: emit a protocol-valid envelope,
                        # report on stderr, and terminate — the stdio stream
                        # cannot be resynced mid-frame, matching the EOF
                        # lifecycle (clean thread exit, server shutdown).
                        logger.error(
                            "stdin line exceeded MAX_LINE_BYTES (%d bytes)",
                            MAX_LINE_BYTES,
                        )
                        try:
                            self.response_queue.put(
                                jsonrpc_error(
                                    None,
                                    INVALID_REQUEST,
                                    "Invalid Request: stdin line exceeded "
                                    f"{MAX_LINE_BYTES} bytes",
                                ),
                                timeout=1.0,
                            )
                        except queue.Full:
                            logger.warning(
                                "Response queue full, dropping error response"
                            )
                        sys.stderr.write(
                            f"MCP stdio: input line exceeded {MAX_LINE_BYTES} "
                            "bytes; terminating server.\n"
                        )
                        self.running = False
                        break
                    if not line:
                        logger.info("End of input detected, stopping server")
                        self.running = False
                        break

                    # Update activity timestamp
                    self._last_activity = time.time()

                    try:
                        message = json.loads(line.strip())
                        logger.debug(f"STDIO IN: {message}")
                        self._requests_processed += 1

                        # Check for connection health
                        if self._connection_errors > self._max_connection_errors:
                            logger.error("Too many connection errors, stopping server")
                            self.running = False
                            break

                        self.request_queue.put(message, timeout=1.0)

                    except json.JSONDecodeError as e:
                        logger.error(
                            f"Invalid JSON message: {line.strip().decode('utf-8', 'replace')} - {e}"
                        )
                        self._connection_errors += 1

                        # Send JSON-RPC parse error
                        error_response: dict[str, Any] = {
                            "jsonrpc": "2.0",
                            "error": {"code": -32700, "message": "Parse error"},
                            "id": None,
                        }
                        try:
                            self.response_queue.put(error_response, timeout=1.0)
                        except queue.Full:
                            logger.warning(
                                "Response queue full, dropping error response"
                            )

                    except queue.Full:
                        logger.warning("Request queue full, dropping message")
                        self._connection_errors += 1

                except (IOError, OSError) as e:
                    logger.error(f"IO error in reader thread: {e}")
                    self._connection_errors += 1
                    if self._connection_errors > self._max_connection_errors:
                        logger.error("Too many IO errors, stopping server")
                        self.running = False
                        break

                    # Brief pause before retrying
                    time.sleep(0.1)

        except Exception as e:
            logger.error(f"Unexpected error in reader thread: {str(e)}")
            self.running = False

    def _processor_thread(self) -> Any:
        """Enhanced thread that processes messages from the request queue with better error handling."""
        try:
            while self.running or not self.request_queue.empty():
                try:
                    message = self.request_queue.get(timeout=0.1)
                    self._process_message(message)
                    self.request_queue.task_done()
                except queue.Empty:
                    continue  # intentional: poll loop, no data available yet
                except Exception as e:
                    logger.error(f"Error in processor thread: {str(e)}")
                    self._errors_encountered += 1
                    # Continue processing other messages
                    try:
                        self.request_queue.task_done()
                    except ValueError as e:
                        logger.debug(
                            f"Task done notification failed (already completed): {e}"
                        )
        except Exception as e:
            logger.error(f"Fatal error in processor thread: {str(e)}")
            self.running = False
        finally:
            self._processor_finished.set()

    def _writer_thread(self) -> Any:
        """Enhanced thread that writes JSON-RPC responses to stdout with error recovery."""
        try:
            while (
                self.running
                or not self._processor_finished.is_set()
                or not self.response_queue.empty()
            ):
                try:
                    message = self.response_queue.get(timeout=0.1)
                    self._responses_sent += 1

                    try:
                        json_str = serialize_response(message, separators=(",", ":"))
                    except Exception as e:
                        # Never hang the client on an unserializable result:
                        # emit a protocol-valid -32603 envelope instead.
                        self._errors_encountered += 1
                        logger.error(f"Response serialization failed: {e}")
                        fallback_id = (
                            message.get("id") if isinstance(message, dict) else None
                        )
                        json_str = serialize_response(
                            jsonrpc_error(
                                fallback_id,
                                INTERNAL_ERROR,
                                "Internal error: response serialization failed",
                            ),
                            separators=(",", ":"),
                        )
                    try:
                        logger.debug(f"STDIO OUT: {json_str}")
                        sys.stdout.write(json_str + "\n")
                        sys.stdout.flush()
                    except (BrokenPipeError, IOError) as e:
                        logger.error(f"IO error writing response: {e}")
                        self._connection_errors += 1
                        if self._connection_errors > self._max_connection_errors:
                            logger.error("Too many write errors, stopping server")
                            self.running = False
                            break
                    except Exception as e:
                        logger.error(f"Error serializing/writing message: {str(e)}")
                        self._errors_encountered += 1

                    self.response_queue.task_done()

                except queue.Empty:
                    continue  # intentional: poll loop, no data available yet
                except Exception as e:
                    logger.error(f"Unexpected error in writer thread: {str(e)}")
                    self._errors_encountered += 1

        except Exception as e:
            logger.error(f"Fatal error in writer thread: {str(e)}")
            self.running = False

    def _process_message(self, message: Dict[str, Any]) -> Any:
        """Process an incoming JSON-RPC message with enhanced validation."""
        try:
            self._process_jsonrpc(message)

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            self._errors_encountered += 1
            try:
                self._send_error(
                    None, -32603, f"Internal error processing message: {str(e)}"
                )
            except Exception:
                logger.error("Failed to send error response")

    def _process_jsonrpc(self, message: Dict[str, Any]) -> Any:
        """
        Process a JSON-RPC message, supporting both standard MCP methods and direct tool invocation.
        """
        error = validate_request(message)
        if error is not None:
            if error["error"]["code"] != -32602 or "id" in message:
                self.response_queue.put(error)
            return
        request_id = message.get("id", _NOTIFICATION)
        method = message.get("method")
        params = message.get("params", {})

        if not method:
            self._send_error(request_id, -32600, "Invalid Request: missing method")
            return

        if is_standard_method(method):
            self._dispatch_standard(message)
            return

        try:
            # Standard MCP methods
            if method in ("mcp.capabilities", "get_mcp_server_capabilities"):
                result = mcp_instance.get_capabilities()
                self._send_result(request_id, result)
            elif method == "mcp.tool.execute":
                if not (
                    isinstance(params.get("name"), str)
                    and isinstance(params.get("params"), dict)
                ):
                    self._send_error(
                        request_id, -32602, "Invalid params for tool execution"
                    )
                    return
                tool_name = params["name"]
                tool_params = params["params"]
                result = mcp_instance.execute_tool(tool_name, tool_params)
                self._send_result(request_id, result)
            elif method == "mcp.resource.get":
                if not (isinstance(params.get("uri"), str)):
                    self._send_error(
                        request_id, -32602, "Invalid params for resource retrieval"
                    )
                    return
                uri = params["uri"]
                result = mcp_instance.get_resource(uri)
                self._send_result(request_id, result)
            # Direct tool invocation (meta-tools, registered tools, etc.)
            elif method in mcp_instance.tools:
                if not isinstance(params, dict):
                    self._send_error(
                        request_id, -32602, "Params must be an object (dictionary)"
                    )
                    return
                result = mcp_instance.execute_tool(method, params)
                self._send_result(request_id, result)
            else:
                # Not a dialect entry: route through the shared standard core
                # (unknown methods surface as -32601 from server_core).
                self._dispatch_standard(message)
        except MCPError as mcpe:
            logger.error(f"MCPError in method {method}: {mcpe}")
            self._send_error(
                request_id, mcpe.code, str(mcpe), data=getattr(mcpe, "data", None)
            )
        except Exception as e:
            logger.exception(f"Unhandled error in method {method}: {e}")
            self._send_error(request_id, -32603, f"Internal error: {str(e)}")

    def _dispatch_standard(self, message: Dict[str, Any]) -> None:
        """Route a validated message through the shared standard MCP core.

        ``MCPServer`` is constructed per call against the transport-owned
        ``mcp_instance`` global so tests can rebind the registry seam. A
        notification (no ``id``) executes and enqueues nothing; a request
        enqueues its full JSON-RPC response envelope.
        """
        response = MCPServer(mcp_instance=mcp_instance).handle_request(message)
        if response is not None:
            self.response_queue.put(response)

    def _send_result(self, request_id: Any, result: Any) -> Any:
        """Send a successful JSON-RPC result response."""
        if request_id is not _NOTIFICATION:
            self.response_queue.put(jsonrpc_result(request_id, result))

    def _send_error(
        self, request_id: Any, code: int, message: str, data: Any = None
    ) -> Any:
        """Send a JSON-RPC error response, including optional data."""
        if request_id is not _NOTIFICATION:
            self.response_queue.put(jsonrpc_error(request_id, code, message, data))


def start_stdio_server() -> Any:
    """Start an MCP server using stdio transport."""
    server = StdioServer()
    server.start()


if __name__ == "__main__":
    start_stdio_server()

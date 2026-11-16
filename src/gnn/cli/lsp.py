"""GNN Language Server helpers for the ``gnn lsp`` subcommand.

Pure JSON-RPC framing and request/response handlers plus the serve loop.
``start_lsp`` launches the canonical pygls server (``gnn.lsp.start_server``)
when pygls is importable, else the pygls-free JSON-RPC serve loop described
below. Transport streams are injectable (``read_message``/``write_message``
default to ``sys.stdin``/``sys.stdout`` at call time), so every layer is
testable without a real stdio session.

``textDocument/completion`` answers from the shared pygls-free vocabulary
module ``gnn.lsp.completions``; ``diagnose_text`` delegates to ``gnn.schema``
for the same diagnostics the pygls server publishes.

Public functions: read_message, write_message, handle_initialize,
handle_hover, handle_completion, diagnose_text, publish_diagnostics,
run_lsp_loop, start_lsp.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, BinaryIO, Callable, Optional, TextIO

from gnn.lsp.completions import completion_context, context_completions
from gnn.schema import (
    parse_connections,
    parse_state_space,
    validate_matrix_dimensions,
    validate_required_sections,
)

logger = logging.getLogger(__name__)


def read_message(stream: TextIO | BinaryIO | None = None) -> Any:
    """Read one Content-Length-framed JSON-RPC message.

    Real stdin is read as bytes. Injected text streams remain supported for
    embedded callers and tests.
    """
    if stream is None:
        stream = getattr(sys.stdin, "buffer", sys.stdin)
    line = stream.readline()
    header = line.decode("ascii") if isinstance(line, bytes) else line
    if not header.startswith("Content-Length: "):
        return None

    content_length = int(header[16:].strip())
    if content_length < 0:
        return None

    stream.readline()
    body = stream.read(content_length)
    if len(body) != content_length:
        return None
    return json.loads(body)


def write_message(msg: Any, stream: Optional[TextIO] = None) -> None:
    """Write a Content-Length-framed JSON-RPC message to ``stream``.

    Defaults to ``sys.stdout`` (resolved at call time so test harnesses
    that swap stdout are honored).
    """
    body = json.dumps(msg)
    frame = f"Content-Length: {len(body)}\r\n\r\n{body}"
    if stream is None:
        buffer = getattr(sys.stdout, "buffer", None)
        if buffer is not None:
            buffer.write(frame.encode("ascii"))
            buffer.flush()
            return
        stream = sys.stdout
    stream.write(frame)
    stream.flush()


def handle_initialize(msg_id: Any) -> dict[str, Any]:
    """Handle the initialize request."""
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": {
            "capabilities": {
                "textDocumentSync": 1,  # Full sync
                "completionProvider": {
                    "resolveProvider": False,
                    "triggerCharacters": ["."],
                },
                "hoverProvider": True,
            },
            "serverInfo": {"name": "gnn-lsp", "version": "1.0.0"},
        },
    }


def handle_hover(msg_id: Any, params: Any) -> dict[str, Any]:
    """Handle the textDocument/hover request."""
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": {
            "contents": {
                "kind": "markdown",
                "value": "**GNN Identifier**\n\nGeneralized Notation Notation construct.",
            }
        },
    }


def handle_completion(
    msg_id: Any,
    params: Any,
    documents: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Handle the textDocument/completion request.

    Completions come from the shared pygls-free vocabulary module
    ``gnn.lsp.completions``. ``documents`` is the session's opened-document
    store (URI -> text) maintained by ``run_lsp_loop``; it resolves the
    cursor line's context. Without it, completions default to section
    headers.
    """
    line_prefix = ""
    in_model_parameters = False
    in_gnn_section = False
    if documents:
        doc = (params or {}).get("textDocument") or {}
        position = (params or {}).get("position") or {}
        text = documents.get(doc.get("uri", ""), "")
        line_prefix, in_model_parameters, in_gnn_section = completion_context(
            text,
            int(position.get("line") or 0),
            int(position.get("character") or 0),
        )
    items = context_completions(
        line_prefix,
        in_model_parameters=in_model_parameters,
        in_gnn_section=in_gnn_section,
    )
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": {"isIncomplete": False, "items": items},
    }


def _schema_error_diagnostic(err: Any) -> dict[str, Any]:
    """Map a GNNParseError to the LSP diagnostic dict shape of the pygls path."""
    line = int(err.line) if getattr(err, "line", None) else 1
    severity = 2 if getattr(err, "severity", "error") == "warning" else 1
    return {
        "range": {
            "start": {"line": max(0, line - 1), "character": 0},
            "end": {"line": max(0, line - 1), "character": 100},
        },
        "severity": severity,
        "message": str(err),
        "source": "gnn",
    }


def diagnose_text(text: str) -> list[dict[str, Any]]:
    """Return LSP diagnostic dicts for GNN text issues.

    Keeps the brace sanity check and delegates semantic validation to
    ``gnn.schema`` (required sections, state space, connections, matrix
    dimension cross-validation), mapping ``GNNParseError`` objects to the
    same ``{range, severity, message, source}`` shape the pygls server path
    emits.

    Pure function: no I/O, deterministic output for a given document.
    Empty/whitespace-only text yields no diagnostics (nothing typed yet).
    """
    diagnostics: list[dict[str, Any]] = []

    if not text.strip():
        return diagnostics

    # Simple syntax check: look for missing closing braces
    if "{" in text and "}" not in text:
        diagnostics.append(
            {
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 0, "character": 100},
                },
                "severity": 1,  # Error
                "message": "Missing closing brace '}'",
                "source": "gnn",
            }
        )

    for err in validate_required_sections(text):
        diagnostics.append(_schema_error_diagnostic(err))

    variables, var_errors = parse_state_space(text)
    for err in var_errors:
        diagnostics.append(_schema_error_diagnostic(err))

    var_names = {v.name for v in variables}
    _, conn_errors = parse_connections(text, known_variables=var_names)
    for err in conn_errors:
        diagnostics.append(_schema_error_diagnostic(err))

    for err in validate_matrix_dimensions(text, variables):
        diagnostics.append(_schema_error_diagnostic(err))

    return diagnostics


def publish_diagnostics(uri: Any, text: Any, stream: Optional[TextIO] = None) -> None:
    """Run basic validation and publish a diagnostics notification."""
    write_message(
        {
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {"uri": uri, "diagnostics": diagnose_text(text)},
        },
        stream,
    )


def run_lsp_loop(
    reader: Callable[[], Any],
    writer: Callable[..., Any],
) -> None:
    """Serve one LSP session over injected transport callables.

    Reads framed requests via ``reader()`` until EOF or ``exit``, dispatches
    each method, and answers via ``writer``. Unhandled requests receive a
    ``Method not found`` error response; unhandled notifications are ignored.
    An exception while handling one message logs and terminates the loop.
    """
    documents: dict[str, str] = {}
    while True:
        try:
            msg = reader()
            if not msg:
                break

            logger.info("Received: %s", msg.get("method"))

            method = msg.get("method")
            msg_id = msg.get("id")

            if method == "initialize":
                writer(handle_initialize(msg_id))
            elif method == "initialized":
                logger.debug("Client initialized notification received")
            elif method == "textDocument/hover":
                writer(handle_hover(msg_id, msg.get("params")))
            elif method == "textDocument/completion":
                writer(handle_completion(msg_id, msg.get("params"), documents))
            elif method == "textDocument/didOpen":
                params = msg.get("params", {})
                doc = params.get("textDocument", {})
                uri = doc.get("uri", "")
                text = doc.get("text", "")
                if uri and text:
                    documents[uri] = text
                    publish_diagnostics(uri, text)
            elif method == "textDocument/didChange":
                params = msg.get("params", {})
                doc = params.get("textDocument", {})
                uri = doc.get("uri", "")
                changes = params.get("contentChanges", [])
                if uri and changes:
                    # Sync sends full text
                    text = changes[0].get("text", "")
                    documents[uri] = text
                    publish_diagnostics(uri, text)
            elif method == "shutdown":
                writer({"jsonrpc": "2.0", "id": msg_id, "result": None})
            elif method == "exit":
                break
            else:
                # Ignore unhandled notifications
                if msg_id is not None:
                    # Return method not found if it is a request
                    writer(
                        {
                            "jsonrpc": "2.0",
                            "id": msg_id,
                            "error": {"code": -32601, "message": "Method not found"},
                        }
                    )
        except Exception as e:
            logger.error("Error handling message: %s", e)
            break


def start_lsp(log_path: Optional[str] = "gnn-lsp.log") -> None:
    """Start the Language Server Protocol server on stdin/stdout.

    Launches the canonical pygls server (``gnn.lsp.start_server``, which
    blocks on stdio until the session ends) when pygls is importable;
    otherwise serves the pygls-free JSON-RPC loop. Logs to ``log_path``
    (default ``gnn-lsp.log`` in the working directory) so stdout stays a
    clean JSON-RPC transport; pass ``None`` to disable file logging.
    """
    if log_path is not None:
        logging.basicConfig(filename=log_path, level=logging.INFO)
    logger.info("Starting GNN LSP Server...")
    import gnn.lsp as lsp_server

    if lsp_server.PYGLS_AVAILABLE:
        lsp_server.start_server()
    else:
        run_lsp_loop(read_message, write_message)
    logger.info("GNN LSP Server shutting down.")

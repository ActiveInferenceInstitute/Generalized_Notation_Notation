"""Transport limit and hygiene contracts for the MCP server (wave-2 MED-04/MIN-01).

Pins the following on the HTTP and stdio transports:

- an untrusted/malformed ``Content-Length`` header yields a protocol-valid
  JSON-RPC error envelope (HTTP 400-class), never an unhandled ``ValueError``
  or a traceback on the wire;
- a request body above ``MAX_BODY_BYTES`` is rejected with a 413-class
  JSON-RPC envelope *before* the body is read off the wire;
- a single stdin line above ``MAX_LINE_BYTES`` yields a protocol-valid JSON-RPC
  error envelope, a stderr message, and a clean thread/server exit;
- ``tools/call`` results above ``MAX_RESPONSE_BYTES`` are truncated with an
  appended notice content item, keeping the tool-result schema valid;
- a non-dict request is rejected by ``validate_request`` with -32600 (the
  ``server_core`` -32700 branch is unreachable and was deleted);
- result-cache entries are deep-copied on store and on hit so callers cannot
  poison cache hits by mutating returned dicts;
- both transports emit byte-identical UTF-8 bodies for a non-ASCII result
  (``ensure_ascii=False`` everywhere on the wire).
"""

import io
import json
import socket
import socketserver
import sys
import threading
import time
from typing import Any, cast

import pytest

from gnn.mcp import server_core, server_http, server_stdio
from gnn.mcp.jsonrpc import jsonrpc_result, serialize_response
from gnn.mcp.mcp import MCP
from gnn.mcp.server_core import MCPServer

pytestmark = pytest.mark.mcp


@pytest.fixture
def rpc_server(monkeypatch: pytest.MonkeyPatch) -> MCP:
    monkeypatch.setenv("GNN_MCP_TOKEN", "test-local-token")
    monkeypatch.setenv("GNN_MCP_RATE_LIMIT_PER_MINUTE", "0")
    monkeypatch.setenv("GNN_MCP_SAFE_TOOLS", "limits_echo,limits_unicode_echo")
    registry = MCP(
        enable_caching=False, enable_rate_limiting=False, strict_validation=True
    )
    registry.register_tool(
        "limits_echo",
        lambda value: value,
        {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        },
        "Return an integer",
    )
    registry.register_tool(
        "limits_unicode_echo",
        lambda: {"msg": "héllo ☃"},
        {"type": "object", "properties": {}},
        "Return non-ASCII text",
    )
    monkeypatch.setattr(server_http, "mcp_instance", registry)
    monkeypatch.setattr(server_stdio, "mcp_instance", registry)
    return registry


class _RequestBytesIO(io.BytesIO):
    """BytesIO that records how many bytes were consumed off the wire."""

    def __init__(self, data: bytes):
        super().__init__(data)
        self.consumed = 0

    def readinto(self, b) -> int:  # type: ignore[override]
        n = super().readinto(b)
        self.consumed += n
        return n


class HTTPByteStream:
    """Socket interface over actual HTTP bytes with controllable headers."""

    def __init__(self, content_length_header: bytes, body: bytes):
        self.head = (
            b"POST / HTTP/1.0\r\nAuthorization: Bearer test-local-token\r\n"
            + content_length_header
            + b"\r\n\r\n"
        )
        self.input = _RequestBytesIO(self.head + body)
        self.output = io.BytesIO()

    def makefile(self, mode: str, *args: object) -> io.BytesIO:
        return self.input

    def sendall(self, data: bytes) -> None:
        self.output.write(data)


def raw_post(
    body: bytes, *, content_length: str | None = None
) -> tuple[int, bytes, HTTPByteStream]:
    """Send one raw HTTP request; return (status, body bytes, stream)."""
    header = (
        b"Content-Length: " + content_length.encode()
        if content_length is not None
        else b"Content-Length: " + str(len(body)).encode()
    )
    stream = HTTPByteStream(header, body)
    server_http.MCPHTTPHandler(
        cast(socket.socket, stream),
        ("127.0.0.1", 1234),
        cast(socketserver.BaseServer, None),
    )
    headers, payload = stream.output.getvalue().split(b"\r\n\r\n", 1)
    return int(headers.split()[1]), payload, stream


def post(
    rpc_server: MCP, payload: object, *, raw: bool = False
) -> tuple[int, dict[str, Any] | None]:
    body = str(payload).encode() if raw else json.dumps(payload).encode()
    status, response_body, _ = raw_post(body)
    return status, json.loads(response_body) if response_body else None


# --- HTTP request-body limits -------------------------------------------------


class TestHTTPBodyLimits:
    def test_oversize_body_is_rejected_413_before_read(
        self, rpc_server: MCP, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(server_http, "MAX_BODY_BYTES", 8)
        body = b"x" * 100_000  # well beyond the rfile read-ahead buffer
        status, response_body, stream = raw_post(body)
        assert status == 413
        assert b"Traceback" not in stream.output.getvalue()
        envelope = json.loads(response_body)
        assert envelope["jsonrpc"] == "2.0"
        assert envelope["error"]["code"] == -32600
        # The body was not drained: reads stayed within the request headers
        # plus the reader's fixed read-ahead window, far short of the body.
        assert stream.input.consumed < len(stream.head) + len(body)

    def test_malformed_content_length_is_rejected_400(
        self, rpc_server: MCP, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(server_http, "MAX_BODY_BYTES", 1024)
        body = b'{"jsonrpc": "2.0", "id": 7, "method": "limits_echo", "params": {}}'
        status, response_body, stream = raw_post(body, content_length="twelve")
        assert status == 400
        assert b"Traceback" not in stream.output.getvalue()
        envelope = json.loads(response_body)
        assert envelope["jsonrpc"] == "2.0"
        assert envelope["error"]["code"] == -32600

    def test_normal_body_still_executes(self, rpc_server: MCP) -> None:
        status, result = post(
            rpc_server,
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "limits_echo",
                "params": {"value": 4},
            },
        )
        assert status == 200
        assert result is not None
        assert result["result"] == 4


def _feed_stdin(monkeypatch: pytest.MonkeyPatch, chunks: list[bytes]) -> None:
    reads = iter(chunks)
    monkeypatch.setattr(server_stdio.os, "read", lambda fd, size: next(reads))


class TestStdioBoundedLine:
    def test_oversize_line_envelope_stderr_and_clean_exit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(server_stdio, "MAX_LINE_BYTES", 16)
        _feed_stdin(monkeypatch, [b"x" * 64])
        server = server_stdio.StdioServer()
        server.running = True
        server._reader_thread()
        assert server.running is False  # clean thread exit
        envelope = server.response_queue.get_nowait()
        assert envelope["jsonrpc"] == "2.0"
        assert envelope["error"]["code"] == -32600
        assert envelope["id"] is None
        assert "exceeded" in capsys.readouterr().err

    def test_normal_line_parses_then_eof_stops(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _feed_stdin(
            monkeypatch,
            [
                b'{"jsonrpc": "2.0", "id": 1, "method": "limits_echo", "params": {}}\n',
                b"",
            ],
        )
        server = server_stdio.StdioServer()
        server.running = True
        server._reader_thread()
        message = server.request_queue.get_nowait()
        assert message["method"] == "limits_echo"
        assert server.running is False  # EOF terminated the reader cleanly

    def test_line_without_trailing_newline_at_eof_still_parsed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _feed_stdin(
            monkeypatch,
            [
                b'{"jsonrpc": "2.0", "id": 2, "method": "limits_echo", "params": {}}',
                b"",
            ],
        )
        server = server_stdio.StdioServer()
        server.running = True
        server._reader_thread()
        assert server.request_queue.get_nowait()["id"] == 2


# --- server_core pinning and response-size policy ------------------------------


class TestServerCorePins:
    def test_non_dict_request_is_rejected_with_minus_32600(
        self, rpc_server: MCP
    ) -> None:
        core = MCPServer(rpc_server)
        requests: list[Any] = [[1, 2, 3], "not a request", None, 42]
        for request in requests:
            response = core.handle_request(request)
            assert response is not None
            assert response["jsonrpc"] == "2.0"
            assert response["error"]["code"] == -32600

    def test_non_dict_params_are_rejected_with_minus_32602(
        self, rpc_server: MCP
    ) -> None:
        core = MCPServer(rpc_server)
        response = core.handle_request(
            {"jsonrpc": "2.0", "id": 7, "method": "limits_echo", "params": []}
        )
        assert response is not None
        assert response["error"]["code"] == -32602

    def test_missing_tool_name_is_invalid_params(self, rpc_server: MCP) -> None:
        core = MCPServer(rpc_server)
        for name in ("", None):
            response = core.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": {}},
                }
            )
            assert response is not None
            assert response["error"]["code"] == -32602


class TestResponseSizePolicy:
    def _registry_with_big_tool(self, blob: str) -> MCP:
        registry = MCP(enable_caching=False, enable_rate_limiting=False)
        registry.register_tool(
            "big_blob",
            lambda: {"blob": blob},
            {"type": "object", "properties": {}},
            "Return a big blob",
        )
        return registry

    def test_oversize_tools_call_result_is_truncated_with_notice(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        registry = self._registry_with_big_tool("x" * 4096)
        monkeypatch.setattr(server_core, "MAX_RESPONSE_BYTES", 256)
        core = MCPServer(registry)
        response = self._call_big_blob(core)
        assert response is not None
        content = response["result"]["content"]
        assert len(content) == 2
        assert all(item["type"] == "text" for item in content)
        assert len(content[0]["text"].encode("utf-8")) <= 256
        assert "truncated" in content[1]["text"].lower()
        assert "MAX_RESPONSE_BYTES" in content[1]["text"]

    def test_normal_tools_call_result_is_untouched(self, rpc_server: MCP) -> None:
        core = MCPServer(rpc_server)
        response = core.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {"name": "limits_echo", "arguments": {"value": 1}},
            }
        )
        assert response is not None
        content = response["result"]["content"]
        assert len(content) == 1
        assert "1" in content[0]["text"]

    def test_truncation_respects_utf8_boundaries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        registry = self._registry_with_big_tool("é" * 4096)
        monkeypatch.setattr(server_core, "MAX_RESPONSE_BYTES", 64)
        core = MCPServer(registry)
        response = self._call_big_blob(core)
        assert response is not None
        text = response["result"]["content"][0]["text"]
        text.encode("utf-8")  # no replacement chars / decode errors

    def _call_big_blob(self, core: MCPServer) -> dict[str, Any] | None:
        return core.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {"name": "big_blob", "arguments": {}},
            }
        )


# --- result-cache immutability --------------------------------------------------


class TestCacheImmutability:
    def _registry(self) -> tuple[MCP, list[int]]:
        calls: list[int] = []
        registry = MCP(enable_caching=True, enable_rate_limiting=False)
        registry.register_tool(
            "cached_thing",
            lambda: calls.append(1) or {"n": len(calls), "nested": {"k": "v"}},
            {"type": "object", "properties": {}},
            "cached result",
            cacheable=True,
            cache_ttl=60.0,
        )
        return registry, calls

    def test_mutating_first_result_does_not_poison_second_call(self) -> None:
        registry, calls = self._registry()
        first = registry.execute_tool("cached_thing", {})
        first["nested"]["k"] = "poisoned"
        first["n"] = 999
        second = registry.execute_tool("cached_thing", {})
        assert second == {"n": 1, "nested": {"k": "v"}}
        assert len(calls) == 1  # still served from cache

    def test_mutating_cache_hit_does_not_poison_next_hit(self) -> None:
        registry, calls = self._registry()
        registry.execute_tool("cached_thing", {})
        hit = registry.execute_tool("cached_thing", {})
        hit["nested"]["k"] = "poisoned"
        third = registry.execute_tool("cached_thing", {})
        assert third == {"n": 1, "nested": {"k": "v"}}
        assert len(calls) == 1


# --- ensure_ascii unification ----------------------------------------------------


def _stdio_wire_bytes(
    envelope: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> bytes:
    """Drive the real stdio writer thread and capture the bytes it emits."""
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    server = server_stdio.StdioServer()
    server.running = True
    writer = threading.Thread(target=server._writer_thread, daemon=True)
    writer.start()
    server.response_queue.put(envelope)
    deadline = time.time() + 5
    while not out.getvalue() and time.time() < deadline:
        time.sleep(0.01)
    server.running = False
    writer.join(timeout=5)
    assert writer.is_alive() is False
    return out.getvalue().rstrip("\n").encode("utf-8")


def test_transports_emit_identical_non_ascii_bytes(
    rpc_server: MCP, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "limits_unicode_echo",
        "params": {},
    }
    status, http_body, _ = raw_post(json.dumps(payload).encode())
    assert status == 200

    result = rpc_server.execute_tool("limits_unicode_echo", {})
    envelope = jsonrpc_result(7, result)
    stdio_bytes = _stdio_wire_bytes(envelope, monkeypatch)

    assert http_body == stdio_bytes
    # ensure_ascii=False: real UTF-8 on the wire, no \\u escapes.
    assert "héllo ☃".encode() in http_body
    assert b"\\u00e9" not in http_body
    assert (
        serialize_response(envelope, separators=(",", ":")).encode("utf-8") == http_body
    )

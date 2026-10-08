"""Consume the maintained MCP stdio process through real authored wire frames.

No replacement registries, transport mocks, providers, or private dispatch calls.
The optional coverage command observes only this child in the caller's environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

pytestmark = pytest.mark.mcp
LINE_LIMIT = 1024 * 1024
ROOT = Path(__file__).resolve().parents[2]


class NativeStdio:
    """Own one ordinary server child and inspect only its public output."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        workspace.mkdir(parents=True, exist_ok=True)
        self.stderr_path = workspace / "stderr.log"
        self.stderr = self.stderr_path.open("wb")
        self.frames: queue.Queue[bytes] = queue.Queue()
        self.transcript: list[dict] = []
        self.next_id = 10
        env = dict(os.environ)
        env.update(
            PYTHONPATH=str(ROOT / "src"),
            PYTHONUNBUFFERED="1",
            MPLCONFIGDIR=env.get(
                "GNN_NATIVE_PROTOCOL_MPL_DIR",
                str(workspace.parent.parent / "native_protocol_matplotlib"),
            ),
        )
        # Instrumentation is opt-in, test-side, and uses the ordinary coverage
        # CLI rather than inserting a hook into a production registry or server.
        command = [sys.executable, "-P", "-m", "gnn.mcp.server_stdio"]
        coverage_dir = env.get("GNN_NATIVE_PROTOCOL_COVERAGE_DIR")
        if coverage_dir:
            env["COVERAGE_FILE"] = str(Path(coverage_dir) / ".coverage")
            command = [
                sys.executable,
                "-P",
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                "--source=gnn",
                "-m",
                "gnn.mcp.server_stdio",
            ]
        self.process = subprocess.Popen(
            command,
            cwd=workspace,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            start_new_session=os.name == "posix",
        )
        self.reader = threading.Thread(target=self._read_stdout, daemon=True)
        self.reader.start()
        try:
            initialized = self.call(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "authored-410-consumer", "version": "1"},
                },
                timeout=90,
            )
            assert initialized["result"]["protocolVersion"] == "2024-11-05"
        except BaseException:
            self.close()
            raise

    def _read_stdout(self):
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.frames.put(line)

    def receive(self, timeout=5):
        try:
            raw = self.frames.get(timeout=timeout)
        except queue.Empty:
            pytest.fail(
                f"No protocol response within {timeout}s; "
                f"child={self.process.poll()}, stderr={self.stderr_path.read_text()}"
            )
        response = json.loads(raw)
        assert response["jsonrpc"] == "2.0"
        self.transcript.append({"received": response})
        return response

    def write(self, raw: bytes):
        assert self.process.stdin is not None
        self.transcript.append(
            {
                "sent_bytes": len(raw),
                "sent_sha256": hashlib.sha256(raw).hexdigest(),
                "sent_utf8": raw.decode("utf-8", errors="replace")
                if len(raw) < 4096
                else None,
            }
        )
        self.process.stdin.write(raw)
        self.process.stdin.flush()

    def request(self, method, params=None, *, request_id=None):
        if request_id is None:
            request_id = self.next_id
            self.next_id += 1
        message = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        return message

    def call(self, method, params=None, timeout=5):
        message = self.request(method, params)
        self.transcript.append({"sent": message})
        self.write(json.dumps(message, ensure_ascii=False).encode() + b"\n")
        response = self.receive(timeout)
        assert response["id"] == message["id"]
        return response

    def tool(self, name, arguments):
        response = self.call("tools/call", {"name": name, "arguments": arguments})
        assert "error" not in response, response
        (content,) = response["result"]["content"]
        assert content["type"] == "text"
        return json.loads(content["text"])

    def eof(self):
        assert self.process.stdin is not None
        if not self.process.stdin.closed:
            self.process.stdin.close()

    def close(self):
        self.eof()
        forced = False
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            forced = True
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        self.reader.join(timeout=2)
        assert self.process.stdout is not None
        self.process.stdout.close()
        self.stderr.close()
        (self.workspace / "transcript.json").write_text(
            json.dumps(
                {
                    "pid": self.process.pid,
                    "returncode": self.process.returncode,
                    "forced_cleanup": forced,
                    "cleanup_verified": self.process.poll() is not None,
                    "frames": self.transcript,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        assert not forced, "Ordinary native stdio EOF did not finish within 5s"
        assert self.process.returncode == 0


@pytest.fixture(scope="module")
def native_stdio(tmp_path_factory):
    client = NativeStdio(tmp_path_factory.mktemp("native_shared") / "wire")
    try:
        yield client
    finally:
        client.close()


@pytest.fixture
def isolated_stdio(tmp_path):
    client = NativeStdio(tmp_path / "wire")
    try:
        yield client
    finally:
        client.close()


def test_native_discovery_schema_and_nonprovider_tool(native_stdio):
    response = native_stdio.call("tools/list")
    tools = {tool["name"]: tool for tool in response["result"]["tools"]}
    assert {
        "get_gnn_module_info",
        "get_gnn_documentation",
        "process_gnn_directory",
    } <= tools.keys()
    schema = tools["process_gnn_directory"]["schema"]
    assert schema["type"] == "object"
    assert schema["properties"]["recursive"]["type"] == "boolean"
    assert schema["properties"]["target_dir"]["type"] == "string"
    metadata = native_stdio.tool("get_gnn_module_info", {})
    assert metadata["name"] == "GNN Module"
    direct = native_stdio.call("get_gnn_module_info", {})
    assert direct["result"] == metadata


def test_native_resource_envelopes_and_authored_grammar(native_stdio):
    listed = native_stdio.call("resources/list")["result"]["resources"]
    assert any("documentation" in resource["uri_template"] for resource in listed)
    uri = "gnn://documentation/grammar"
    result = native_stdio.call("resources/read", {"uri": uri})["result"]
    (content,) = result["contents"]
    assert content["uri"] == uri
    document = json.loads(content["text"])
    assert document["success"] is True
    assert document["doc_name"] == "grammar"
    assert "=" in document["content"] and "GNN" in document["content"]
    direct = native_stdio.call("mcp.resource.get", {"uri": uri})["result"]
    assert direct["uri"] == uri and direct["content"] == document


@pytest.mark.parametrize("recursive,expected", [(False, 1), (True, 2)])
def test_native_directory_consumer_reports_actual_selected_files(
    native_stdio, tmp_path, recursive, expected
):
    target = tmp_path / "input"
    nested = target / "nested"
    nested.mkdir(parents=True)
    source = ROOT / "input/gnn_files/discrete/actinf_pomdp_agent.md"
    shutil.copyfile(source, target / "agent.md")
    shutil.copyfile(source, nested / "nested-agent.md")
    output = tmp_path / "result"
    result = native_stdio.tool(
        "process_gnn_directory",
        {"target_dir": str(target), "output_dir": str(output), "recursive": recursive},
    )
    assert result["success"] is True
    summary = json.loads((output / "gnn_processing_summary.json").read_text())
    assert summary["files_found"] == expected
    assert len(summary["results"]) == expected
    assert summary["validator_mode"] == "full" and summary["degraded"] is False
    assert all(entry["validator_mode"] == "full" for entry in summary["results"])
    assert native_stdio.call("ping")["result"] == {}


def test_native_missing_reference_is_a_structured_failed_tool_result(
    native_stdio, tmp_path
):
    target = tmp_path / "input"
    target.mkdir()
    output = tmp_path / "roundtrip"
    missing = target / "missing.md"
    result = native_stdio.tool(
        "run_round_trip_tests",
        {
            "target_dir": str(target),
            "output_dir": str(output),
            "reference_file": str(missing),
            "test_subset": ["markdown"],
        },
    )
    assert result["success"] is False
    summary = json.loads((output / "round_trip_results.json").read_text())
    assert (summary["files_tested"], summary["passed"], summary["failed"]) == (1, 0, 1)
    assert summary["parser_mode"] == "full" and summary["degraded"] is False
    assert str(missing) in summary["results"][0]["error"]
    assert native_stdio.call("ping")["result"] == {}


@pytest.mark.parametrize(
    "message,code",
    [
        ([], -32600),
        ({"jsonrpc": "1.0", "id": 100, "method": "ping"}, -32600),
        ({"jsonrpc": "2.0", "id": 100, "method": "ping", "params": []}, -32602),
        ({"jsonrpc": "2.0", "id": 100, "method": "tools/call", "params": {}}, -32602),
        (
            {
                "jsonrpc": "2.0",
                "id": 100,
                "method": "tools/call",
                "params": {"name": []},
            },
            -32602,
        ),
        (
            {
                "jsonrpc": "2.0",
                "id": 100,
                "method": "tools/call",
                "params": {"name": "not_a_registered_tool", "arguments": {}},
            },
            -32601,
        ),
    ],
)
def test_native_malformed_requests_preserve_protocol_and_liveness(
    native_stdio, message, code
):
    native_stdio.write(json.dumps(message).encode() + b"\n")
    result = native_stdio.receive()
    assert result["error"]["code"] == code
    assert "result" not in result
    assert native_stdio.call("ping")["result"] == {}


def test_native_notifications_and_unsupported_cancellation_are_truthful(native_stdio):
    native_stdio.write(b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
    # The next response must belong to ping; notifications emit no response.
    assert native_stdio.call("ping")["result"] == {}
    result = native_stdio.call("notifications/cancelled", {"requestId": 999})
    assert result["error"]["code"] == -32601
    assert native_stdio.call("ping")["result"] == {}


def test_native_coalesced_frames_are_not_dropped(native_stdio):
    messages = [native_stdio.request("ping") for _ in range(3)]
    native_stdio.write(
        b"".join(json.dumps(message).encode() + b"\n" for message in messages)
    )
    responses = [native_stdio.receive() for _ in messages]
    assert [response["id"] for response in responses] == [
        message["id"] for message in messages
    ]
    assert all(response["result"] == {} for response in responses)


def test_native_fragmented_utf8_and_malformed_then_valid_frames(native_stdio):
    message = native_stdio.request("ping", {"note": "雪🌱"})
    wire = json.dumps(message, ensure_ascii=False).encode() + b"\n"
    split = wire.index("雪".encode()) + 1
    native_stdio.write(wire[:split])
    with pytest.raises(queue.Empty):
        native_stdio.frames.get(timeout=0.1)
    native_stdio.write(wire[split:])
    assert native_stdio.receive()["id"] == message["id"]
    valid = native_stdio.request("ping")
    native_stdio.write(b"{invalid json}\n" + json.dumps(valid).encode() + b"\n")
    malformed, response = native_stdio.receive(), native_stdio.receive()
    assert malformed["id"] is None and malformed["error"]["code"] == -32700
    assert response["id"] == valid["id"] and response["result"] == {}


def padded_ping(client, size):
    message = client.request("ping", {"padding": ""})
    message["params"]["padding"] = "x" * (size - len(json.dumps(message).encode()))
    wire = json.dumps(message).encode()
    assert len(wire) == size
    return message["id"], wire


def test_native_line_limit_is_per_frame_not_total_batch(native_stdio):
    first_id, first = padded_ping(native_stdio, LINE_LIMIT)
    second_id, second = padded_ping(native_stdio, LINE_LIMIT)
    native_stdio.write(first + b"\n" + second + b"\n")
    assert native_stdio.receive()["id"] == first_id
    assert native_stdio.receive()["id"] == second_id


def test_native_oversize_frame_fails_closed_and_exits(isolated_stdio):
    native_stdio = isolated_stdio
    _, wire = padded_ping(native_stdio, LINE_LIMIT + 1)
    native_stdio.write(wire + b"\n")
    response = native_stdio.receive()
    assert response["id"] is None and response["error"]["code"] == -32600
    assert "exceeded" in response["error"]["message"]
    assert native_stdio.process.wait(timeout=5) == 0
    assert "exceeded" in native_stdio.stderr_path.read_text()


def test_native_final_unterminated_frame_is_processed_before_eof_exit(isolated_stdio):
    native_stdio = isolated_stdio
    message = native_stdio.request("ping")
    native_stdio.write(json.dumps(message).encode())
    native_stdio.eof()
    response = native_stdio.receive()
    assert response["id"] == message["id"] and response["result"] == {}
    assert native_stdio.process.wait(timeout=5) == 0


def test_native_coalesced_requests_are_drained_before_eof_exit(isolated_stdio):
    messages = [isolated_stdio.request("ping") for _ in range(3)]
    isolated_stdio.write(
        b"".join(json.dumps(message).encode() + b"\n" for message in messages)
    )
    isolated_stdio.eof()
    responses = [isolated_stdio.receive() for _ in messages]
    assert [response["id"] for response in responses] == [
        message["id"] for message in messages
    ]
    assert all(response["result"] == {} for response in responses)
    assert isolated_stdio.process.wait(timeout=5) == 0


@pytest.mark.skipif(os.name != "posix", reason="SIGINT lifecycle is a POSIX contract")
def test_native_operator_interrupt_stops_the_owned_server(isolated_stdio):
    native_stdio = isolated_stdio
    native_stdio.process.send_signal(signal.SIGINT)
    assert native_stdio.process.wait(timeout=5) == 0

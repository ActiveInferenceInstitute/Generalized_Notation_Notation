"""Deterministic Ollama transport regressions; never contact a real daemon."""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from gnn.llm.providers import ollama_provider as module
from gnn.llm.providers.base_provider import LLMConfig, LLMMessage
from gnn.llm.providers.ollama_provider import OllamaProvider

MODEL = "configured:exact"
DIGEST = "a" * 64


class _Requests(list[dict[str, Any]]):
    def __init__(self) -> None:
        super().__init__()
        self.get_paths: list[str] = []


def _response(**overrides: Any) -> dict[str, Any]:
    return {
        "model": MODEL,
        "message": {"role": "assistant", "content": "complete answer"},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 31,
        "eval_count": 3,
        **overrides,
    }


@pytest.fixture
def daemon() -> Iterator[
    Callable[..., tuple[str, list[dict[str, Any]], threading.Event]]
]:
    servers: list[tuple[ThreadingHTTPServer, threading.Thread]] = []

    def start(
        *,
        response: Any = None,
        status: int = 200,
        chunks: list[bytes] | None = None,
        delay: float = 0,
        header_delay: float = 0,
        header_drip: float = 0,
        version_response: Any = None,
        version_delay: float | Callable[[], float] = 0,
        tags_response: Any = None,
        tags_delay: float = 0,
        after_chat: Callable[[], None] | None = None,
        after_get: Callable[[str], None] | None = None,
        show_response: Any = None,
        ps_response: Any = None,
    ) -> tuple[str, list[dict[str, Any]], threading.Event]:
        received = _Requests()
        opened = threading.Event()
        body = json.dumps(_response() if response is None else response).encode()
        output = chunks if chunks is not None else [body]

        class Handler(BaseHTTPRequestHandler):
            # Exercise the socket lifetime of HTTP/1.0 as well as real reads.
            def do_GET(self) -> None:
                received.get_paths.append(self.path)
                if self.path.endswith("/api/tags"):
                    value = (
                        {"models": [{"name": MODEL, "digest": DIGEST}]}
                        if tags_response is None
                        else tags_response
                    )
                    wait = tags_delay
                elif self.path.endswith("/api/ps"):
                    value = ps_response
                    wait = 0
                else:
                    value = (
                        {"version": "0.35.0"}
                        if version_response is None
                        else version_response
                    )
                    wait = version_delay() if callable(version_delay) else version_delay
                body = json.dumps(value).encode()
                if after_get is not None:
                    after_get(self.path)
                time.sleep(wait)
                try:
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_POST(self) -> None:
                received.append(
                    {
                        "path": self.path,
                        "body": json.loads(
                            self.rfile.read(int(self.headers["Content-Length"]))
                        ),
                    }
                )
                if self.path.endswith("/api/show"):
                    result = json.dumps(show_response).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(result)))
                    self.end_headers()
                    self.wfile.write(result)
                    return
                if after_chat is not None:
                    after_chat()
                time.sleep(header_delay)
                try:
                    if header_drip:
                        headers = f"HTTP/1.0 {status} OK\r\nContent-Length: {sum(map(len, output))}\r\n\r\n".encode()
                        for byte in headers:
                            self.wfile.write(bytes([byte]))
                            self.wfile.flush()
                            time.sleep(header_drip)
                    else:
                        self.send_response(status)
                        self.send_header("Content-Length", str(sum(map(len, output))))
                        self.end_headers()
                    self.wfile.flush()
                    opened.set()
                    for part in output:
                        time.sleep(delay)
                        self.wfile.write(part)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass  # Expected when the client enforces its deadline.

            def log_message(self, *_args: Any) -> None:
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )
        thread.start()
        servers.append((server, thread))
        return f"http://127.0.0.1:{server.server_port}", received, opened

    yield start
    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def _provider(host: str, **kwargs: Any) -> OllamaProvider:
    provider = OllamaProvider(base_url=host, timeout=1, **kwargs)
    assert provider.initialize()
    return provider


@pytest.mark.parametrize("method", ["analyze_gnn_model", "compare_models"])
@pytest.mark.parametrize("finish_reason", ["stop", "length", None])
def test_gnn_wrapper_preserves_text_but_requires_terminal_completion(
    daemon: Callable[..., Any], method: str, finish_reason: str | None
) -> None:
    from gnn.llm.llm_processor import GNNLLMProcessor, LLMProcessor
    from gnn.llm.providers.base_provider import ProviderType

    host, received, _ = daemon(response=_response(done_reason=finish_reason))
    base = LLMProcessor(
        preferred_providers=[ProviderType.OLLAMA],
        api_keys={"ollama": "local"},
        provider_configs={
            "ollama": {"base_url": host, "default_model": MODEL, "timeout": 1}
        },
    )

    async def run() -> dict[str, Any]:
        processor = GNNLLMProcessor(base)
        try:
            assert await processor.initialize()
            if method == "compare_models":
                return await processor.compare_models("first source", "second source")
            return await processor.analyze_gnn_model("whole source")
        finally:
            await processor.close()

    result = asyncio.run(run())
    assert result["success"] is (finish_reason == "stop"), result
    assert result["response"] == "complete answer"
    assert result["finish_reason"] == finish_reason
    assert result["provider"] == "ollama" and result["model"] == MODEL
    assert isinstance(result["analysis_type"], str)
    json.dumps(result)  # Enum-based default analysis must remain serializable.
    if finish_reason != "stop":
        assert "unfinished" in result["error"]
    assert len(received) == 1


@pytest.mark.parametrize("running_loop", [False, True])
@pytest.mark.parametrize("finish_reason", ["stop", "length", None])
def test_plain_analysis_requires_terminal_completion(
    daemon: Callable[..., Any], running_loop: bool, finish_reason: str | None
) -> None:
    host, received, _ = daemon(response=_response(done_reason=finish_reason))
    provider = _provider(host, default_model=MODEL)

    async def in_loop() -> str:
        return provider.analyze("whole source", "summary")

    result = (
        asyncio.run(in_loop())
        if running_loop
        else provider.analyze("whole source", "summary")
    )
    if finish_reason == "stop":
        assert result == "complete answer"
    else:
        assert result.startswith("Analysis failed:") and "unfinished" in result
    assert len(received) == 1


def test_entire_large_conversation_and_options_are_http_body_without_sdk_or_cli(
    daemon: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    host, received, _ = daemon()
    # No CLI execution or SDK import is needed, including SDK namespace collisions.
    monkeypatch.setitem(sys.modules, "ollama", None)
    provider = _provider(host)
    large = "Ω scientific matrix\n" + "0.000001, " * 260000
    messages = [
        LLMMessage("system", "scientific analysis"),
        LLMMessage("assistant", "earlier conversation"),
        LLMMessage("user", large),
    ]
    config = LLMConfig(
        model=MODEL, max_tokens=512, temperature=0.1, top_p=0.9, timeout=0.7
    )
    result = asyncio.run(provider.generate_response(messages, config))
    assert len(received) == 1
    assert received[0]["path"] == "/api/chat"
    payload = received[0]["body"]
    assert payload["messages"] == [
        {"role": m.role, "content": m.content} for m in messages
    ]
    assert payload["model"] == MODEL
    assert payload["options"] == {"num_predict": 512, "temperature": 0.1, "top_p": 0.9}
    assert payload["truncate"] is False and payload["shift"] is False
    assert payload["stream"] is False
    assert result.model_used == MODEL
    assert result.usage == {"prompt_tokens": 31, "completion_tokens": 3}
    assert result.finish_reason == "stop"
    assert (
        result.metadata["runtime_identity"] == result.metadata["runtime_identity_after"]
    )
    assert result.metadata["runtime_identity"]["model_digest"] == DIGEST
    assert (
        result.metadata["runtime_validation_policy"]
        == module.OLLAMA_RUNTIME_VALIDATION_POLICY
    )
    assert result.metadata is not None and result.metadata["truncate"] is False


def test_host_port_environment_and_explicit_prefix(
    daemon: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    host, received, _ = daemon()
    monkeypatch.setenv("OLLAMA_HOST", host.removeprefix("http://"))
    provider = OllamaProvider(timeout=1, default_model=MODEL)
    assert provider.initialize()
    asyncio.run(provider.generate_response([LLMMessage("user", "hello")]))
    provider = _provider(host + "/prefix")
    asyncio.run(
        provider.generate_response(
            [LLMMessage("user", "hello")], LLMConfig(model=MODEL)
        )
    )
    assert [item["path"] for item in received] == ["/api/chat", "/prefix/api/chat"]


@pytest.mark.parametrize(
    "host",
    [
        "ftp://127.0.0.1",
        "http://u:p@localhost",
        "http://localhost/?secret=value",
        "http://localhost/#secret",
        "http://localhost:bad",
        "http://",
    ],
)
def test_invalid_host_never_initializes(host: str) -> None:
    provider = OllamaProvider(base_url=host)
    assert not provider.initialize()


@pytest.mark.parametrize("timeout", [True, 0, -1, float("nan"), float("inf")])
def test_invalid_constructor_timeout(timeout: Any) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        OllamaProvider(timeout=timeout)


@pytest.mark.parametrize("timeout", [True, 0, -1, float("nan"), float("inf")])
def test_invalid_request_timeout_never_dispatches(timeout: Any) -> None:
    provider = OllamaProvider()
    assert provider.initialize()
    assert not provider.validate_config(LLMConfig(timeout=timeout))
    with pytest.raises(ValueError, match="Invalid configuration"):
        asyncio.run(
            provider.generate_response(
                [LLMMessage("user", "hello")], LLMConfig(timeout=timeout)
            )
        )


@pytest.mark.parametrize("tokens", [True, 0, -1, 1.5])
def test_invalid_prediction_cap(tokens: Any) -> None:
    assert not OllamaProvider().validate_config(LLMConfig(max_tokens=tokens))


@pytest.mark.parametrize("temperature", [float("nan"), float("inf"), -0.1, 2.1])
def test_invalid_temperature(temperature: float) -> None:
    assert not OllamaProvider().validate_config(LLMConfig(temperature=temperature))


def test_server_context_overflow_is_one_failure_without_retry_or_truncation(
    daemon: Callable[..., Any],
) -> None:
    host, received, _ = daemon(
        status=400, response={"error": "input length exceeds context length"}
    )
    provider = _provider(host)
    with pytest.raises(
        RuntimeError, match="HTTP 400: input length exceeds context length"
    ):
        asyncio.run(
            provider.generate_response(
                [LLMMessage("user", "whole scientific source")], LLMConfig(model=MODEL)
            )
        )
    assert len(received) == 1
    assert received[0]["body"]["truncate"] is False


@pytest.mark.parametrize(
    "response, error",
    [
        (_response(model="replacement:wrong"), "differs"),
        (_response(done=False), "unfinished"),
        (_response(message={"content": ""}), "empty"),
        (_response(message={"content": None}), "malformed"),
        (["not a chat response"], "malformed"),
    ],
)
def test_invalid_response_is_never_success(
    daemon: Callable[..., Any], response: Any, error: str
) -> None:
    host, received, _ = daemon(response=response)
    with pytest.raises(RuntimeError, match=error):
        asyncio.run(
            _provider(host).generate_response(
                [LLMMessage("user", "hello")], LLMConfig(model=MODEL)
            )
        )
    assert len(received) == 1


def test_redirect_is_not_followed(daemon: Callable[..., Any]) -> None:
    host, received, _ = daemon(status=302)
    with pytest.raises(RuntimeError, match="HTTP 302"):
        asyncio.run(
            _provider(host).generate_response(
                [LLMMessage("user", "hello")], LLMConfig(model=MODEL)
            )
        )
    assert len(received) == 1


@pytest.mark.parametrize("mode", ["headers", "slow_body"])
def test_one_request_budget_includes_headers_and_slow_drip(
    daemon: Callable[..., Any], mode: str
) -> None:
    body = json.dumps(_response()).encode()
    options = (
        {"header_delay": 0.3}
        if mode == "headers"
        else {"chunks": [body[:1]] * 20, "delay": 0.025}
    )
    host, _, _ = daemon(**options)
    started = time.monotonic()
    with pytest.raises((TimeoutError, socket.timeout)):
        asyncio.run(
            _provider(host).generate_response(
                [LLMMessage("user", "hello")], LLMConfig(model=MODEL, timeout=0.1)
            )
        )
    assert time.monotonic() - started < 0.25


def test_stream_preserves_roles_options_and_does_not_mutate_config(
    daemon: Callable[..., Any],
) -> None:
    chunks = [
        json.dumps(_response(done=False, message={"content": "one"})).encode() + b"\n",
        json.dumps(_response(message={"content": "two"})).encode() + b"\n",
    ]
    host, received, _ = daemon(chunks=chunks)
    messages = [
        LLMMessage("system", "rules"),
        LLMMessage("assistant", "history"),
        LLMMessage("user", "whole source"),
    ]
    config = LLMConfig(model=MODEL, max_tokens=23)

    async def collect() -> list[str]:
        return [
            chunk async for chunk in _provider(host).generate_stream(messages, config)
        ]

    assert asyncio.run(collect()) == ["one", "two"]
    assert config.stream is False
    assert received[0]["body"]["messages"] == [
        {"role": m.role, "content": m.content} for m in messages
    ]
    assert received[0]["body"]["stream"] is True
    assert received[0]["body"]["options"]["num_predict"] == 23
    assert not received[0]["body"]["truncate"] and not received[0]["body"]["shift"]


def test_stream_budget_is_not_renewed_between_chunks(
    daemon: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    line = json.dumps(_response(done=False)).encode() + b"\n"
    host, _, _ = daemon(chunks=[line] * 20)
    offset = [0.0]
    # Advance only the provider clock; asyncio and the real HTTP server retain
    # their real clocks. Runner scheduling cannot stand in for deadline renewal.
    monkeypatch.setattr(
        module, "time", SimpleNamespace(monotonic=lambda: time.monotonic() + offset[0])
    )
    requests: list[tuple[Any, float]] = []
    original_open = module._ChatRequest.open

    def record_open(request: Any, payload: Any) -> None:
        original_open(request, payload)
        requests.append((request, request.deadline))

    monkeypatch.setattr(module._ChatRequest, "open", record_open)
    delivered: list[str] = []

    async def collect() -> None:
        async for chunk in _provider(host).generate_stream(
            [LLMMessage("user", "hello")], LLMConfig(model=MODEL, timeout=30)
        ):
            request, original_deadline = requests[0]
            assert request.deadline == original_deadline
            delivered.append(chunk)
            if len(delivered) == 2:
                offset[0] = original_deadline - time.monotonic() + 1

    with pytest.raises((TimeoutError, socket.timeout)):
        asyncio.run(collect())
    assert len(delivered) == 2


def test_stream_without_terminal_record_fails(daemon: Callable[..., Any]) -> None:
    host, _, _ = daemon(chunks=[json.dumps(_response(done=False)).encode() + b"\n"])

    async def collect() -> None:
        async for _ in _provider(host).generate_stream(
            [LLMMessage("user", "hello")], LLMConfig(model=MODEL)
        ):
            pass

    with pytest.raises(RuntimeError, match="without a completed response"):
        asyncio.run(collect())


def test_response_and_stream_line_memory_limits_are_explicit(
    daemon: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    host, _, _ = daemon()
    monkeypatch.setattr(module, "_MAX_RESPONSE_BYTES", 20)
    with pytest.raises(RuntimeError, match="8 MiB transport limit"):
        asyncio.run(
            _provider(host).generate_response(
                [LLMMessage("user", "hello")], LLMConfig(model=MODEL)
            )
        )
    monkeypatch.setattr(module, "_MAX_RESPONSE_BYTES", 8 * 1024 * 1024)
    monkeypatch.setattr(module, "_MAX_STREAM_LINE_BYTES", 20)

    async def collect() -> None:
        async for _ in _provider(host).generate_stream(
            [LLMMessage("user", "hello")], LLMConfig(model=MODEL)
        ):
            pass

    with pytest.raises(RuntimeError, match="1 MiB transport limit"):
        asyncio.run(collect())


def test_cancel_during_body_read_closes_request(
    daemon: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    host, _, opened = daemon(chunks=[b"{"] * 20, delay=0.025)
    closes: list[bool] = []
    original = module._ChatRequest.close

    def close(request: module._ChatRequest) -> None:
        closes.append(True)
        original(request)

    monkeypatch.setattr(module._ChatRequest, "close", close)

    async def cancel() -> None:
        task = asyncio.create_task(
            _provider(host).generate_response(
                [LLMMessage("user", "hello")], LLMConfig(model=MODEL)
            )
        )
        assert await asyncio.to_thread(opened.wait, 0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    started = time.monotonic()
    asyncio.run(cancel())
    assert closes and time.monotonic() - started < 0.3


@pytest.mark.parametrize(
    "version", ["0.34.9", "0.35.0-rc1", "nightly", None, "0.35", "v0.35.0"]
)
def test_unverified_runtime_cannot_post_whole_context(
    daemon: Callable[..., Any], version: Any
) -> None:
    host, received, _ = daemon(version_response={"version": version})
    with pytest.raises(RuntimeError, match="whole-context policy"):
        asyncio.run(
            _provider(host).generate_response(
                [LLMMessage("user", "whole source")], LLMConfig(model=MODEL)
            )
        )
    assert not received
    assert received.get_paths == ["/api/version"]


def test_runtime_check_and_chat_share_the_original_deadline(
    daemon: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[tuple[module._ChatRequest, float]] = []
    posts: list[module._ChatRequest] = []
    original_init = module._ChatRequest.__init__
    original_remaining = module._ChatRequest.remaining
    original_exchange = module._ChatRequest._exchange

    def track(request: module._ChatRequest, *args: Any) -> None:
        original_init(request, *args)
        requests.append((request, request.deadline))

    def remaining(request: module._ChatRequest) -> float:
        if requests and request is requests[0][0]:
            assert request.deadline == requests[0][1]
        return original_remaining(request)

    def exchange(
        request: module._ChatRequest, method: str, path: str, body: bytes | None = None
    ) -> None:
        if method == "POST":
            assert request is requests[0][0]
            assert request.deadline == requests[0][1]
            posts.append(request)
        original_exchange(request, method, path, body)

    def expire_after_post() -> None:
        assert posts == [requests[0][0]]
        # Invoke the real guard's expiry after actual HTTP admission, rather
        # than requiring runner scheduling and preflight to fit inside 80ms.
        requests[0][0]._expire()

    host, received, _ = daemon(after_chat=expire_after_post)
    monkeypatch.setattr(module._ChatRequest, "__init__", track)
    monkeypatch.setattr(module._ChatRequest, "remaining", remaining)
    monkeypatch.setattr(module._ChatRequest, "_exchange", exchange)
    provider = OllamaProvider(base_url=host, timeout=30)
    assert provider.initialize()
    with pytest.raises(TimeoutError, match="deadline"):
        asyncio.run(
            provider.generate_response(
                [LLMMessage("user", "hello")], LLMConfig(model=MODEL, timeout=30)
            )
        )
    assert len(received) == 1
    assert received.get_paths == ["/api/version", "/api/tags", "/api/version"]
    assert len(requests) == 2  # Generation owner and its bounded preflight.
    assert all(not request._guard.is_alive() for request, _ in requests)


def test_postflight_cannot_renew_generation_deadline(
    daemon: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    control = {"chat_received": False}

    def after_chat() -> None:
        control["chat_received"] = True

    host, received, _ = daemon(
        header_delay=0.4,
        version_delay=lambda: 0.9 if control["chat_received"] else 0,
        after_chat=after_chat,
    )
    provider = _provider(host)
    original = provider.preflight_model
    timeouts: list[float] = []

    async def track(model: str, timeout: float | None = None) -> dict[str, Any]:
        assert timeout is not None
        timeouts.append(timeout)
        return await original(model, timeout=timeout)

    monkeypatch.setattr(provider, "preflight_model", track)
    with pytest.raises(TimeoutError):
        asyncio.run(
            provider.generate_response(
                [LLMMessage("user", "source")], LLMConfig(model=MODEL, timeout=1)
            )
        )
    assert len(received) == 1
    assert received.get_paths.count("/api/version") == 3
    assert len(timeouts) == 2
    # Check the remaining allowance directly, with enough scheduling slack to
    # reach the postflight phase under concurrent CI load.
    assert timeouts[1] < 0.75 < timeouts[0] <= 1


def test_real_header_byte_drip_is_interrupted_and_deadline_guard_is_reaped(
    daemon: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    host, _, _ = daemon(header_drip=0.007)
    requests: list[module._ChatRequest] = []
    original = module._ChatRequest.__init__

    def track(request: module._ChatRequest, *args: Any) -> None:
        original(request, *args)
        requests.append(request)

    monkeypatch.setattr(module._ChatRequest, "__init__", track)
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        asyncio.run(
            _provider(host).generate_response(
                [LLMMessage("user", "hello")], LLMConfig(model=MODEL, timeout=0.08)
            )
        )
    assert time.monotonic() - start < 0.2
    assert len(requests) == 2  # Original deadline owner and bounded preflight.
    assert all(not request._guard.is_alive() for request in requests)


@pytest.mark.parametrize("running_loop", [False, True])
def test_analyze_never_retries_provider_runtime_error(
    running_loop: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = OllamaProvider(timeout=0.1)
    calls: list[LLMConfig] = []

    async def fail(_messages: Any, config: LLMConfig) -> Any:
        calls.append(config)
        raise RuntimeError("HTTP 400 context length exceeded")

    monkeypatch.setattr(provider, "generate_response", fail)

    async def in_loop() -> str:
        return provider.analyze("whole source", "summary")

    result = (
        asyncio.run(in_loop())
        if running_loop
        else provider.analyze("whole source", "summary")
    )
    assert result == "Analysis failed: HTTP 400 context length exceeded"
    assert len(calls) == 1
    assert (
        calls[0].timeout is not None
        and 0 < calls[0].timeout <= provider.default_timeout
    )


def test_http_preflight_has_no_sdk_cli_generation_or_model_replacement(
    daemon: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    monkeypatch.setattr(shutil, "which", lambda _command: None)
    monkeypatch.setitem(sys.modules, "ollama", None)
    host, received, _ = daemon()
    result = asyncio.run(
        _provider(host + "/prefix").preflight_model(MODEL, timeout=0.7)
    )
    assert result == {
        "model": MODEL,
        "server_version": "0.35.0",
        "context_policy": "whole-context-v1",
        "status": "ready",
        "model_digest": DIGEST,
        "generation_options": {},
    }
    assert received.get_paths == ["/prefix/api/version", "/prefix/api/tags"]
    assert not received


def test_step24_cache_binds_configured_provider_endpoint(
    daemon: Callable[..., Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import logging

    from gnn.intelligent_analysis.processor import _run_llm_analysis
    from gnn.llm.llm_processor import LLMProcessor
    from gnn.llm.providers.base_provider import ProviderType
    from tests.intelligent_analysis.test_intelligent_analysis_cache import (
        ANALYSIS_CONTEXT,
        VALID_REPORT,
    )

    other_report = VALID_REPORT.replace(
        "perfect health score", "healthy second backend"
    )
    host_a, requests_a, _ = daemon(
        response=_response(message={"content": VALID_REPORT})
    )
    host_b, requests_b, _ = daemon(
        response=_response(message={"content": other_report})
    )
    # Explicit provider configurations may differ from the ambient default.
    monkeypatch.setenv("OLLAMA_HOST", host_a)
    selected: dict[str, Any] = {}

    async def initialize() -> Any:
        return selected["processor"]

    monkeypatch.setattr("gnn.llm.llm_processor.initialize_global_processor", initialize)

    async def run() -> None:
        for host, expected in [(host_a, VALID_REPORT), (host_b, other_report)]:
            processor = LLMProcessor(
                preferred_providers=[ProviderType.OLLAMA],
                api_keys={"ollama": "local"},
                provider_configs={"ollama": {"base_url": host, "timeout": 2}},
            )
            try:
                assert await processor.initialize()
                selected["processor"] = processor
                content, source = await _run_llm_analysis(
                    ANALYSIS_CONTEXT,
                    [],
                    {},
                    logging.getLogger("step24-endpoint"),
                    analysis_model=MODEL,
                    cache_dir=tmp_path,
                )
                assert source == "llm"
                assert content == expected
            finally:
                await processor.close()

    asyncio.run(run())
    assert len(requests_a) == len(requests_b) == 1
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_step24_cache_reuses_frozen_provider_when_ambient_config_changes(
    daemon: Callable[..., Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import logging

    from gnn.intelligent_analysis.processor import _run_llm_analysis
    from gnn.llm.llm_processor import LLMProcessor
    from gnn.llm.providers.base_provider import ProviderType
    from tests.intelligent_analysis.test_intelligent_analysis_cache import (
        ANALYSIS_CONTEXT,
        VALID_REPORT,
    )

    host, received, _ = daemon(response=_response(message={"content": VALID_REPORT}))
    monkeypatch.setenv("OLLAMA_HOST", host)
    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")
    monkeypatch.setenv("OLLAMA_SEED", "42")
    processor = LLMProcessor(
        preferred_providers=[ProviderType.OLLAMA],
        api_keys={"ollama": "local"},
        provider_configs={"ollama": {"base_url": host, "timeout": 2}},
    )

    async def initialize() -> Any:
        return processor

    monkeypatch.setattr("gnn.llm.llm_processor.initialize_global_processor", initialize)

    async def run() -> None:
        try:
            assert await processor.initialize()
            for changed in (False, True):
                if changed:
                    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:1")
                    monkeypatch.setenv("OLLAMA_NUM_CTX", "16384")
                    monkeypatch.setenv("OLLAMA_SEED", "17")
                assert await _run_llm_analysis(
                    ANALYSIS_CONTEXT,
                    [],
                    {},
                    logging.getLogger("step24-frozen"),
                    analysis_model=MODEL,
                    cache_dir=tmp_path,
                ) == (VALID_REPORT, "llm")
        finally:
            await processor.close()

    asyncio.run(run())
    assert len(received) == 1
    assert received[0]["body"]["options"]["num_ctx"] == 8192
    assert received[0]["body"]["options"]["seed"] == 42
    assert len(list(tmp_path.glob("*.json"))) == 1


@pytest.mark.parametrize(
    "inventory",
    [
        {"models": []},
        {"models": [{"name": "configured:replacement"}]},
        {"models": "bad"},
        [],
    ],
)
def test_http_preflight_requires_an_exact_valid_inventory(
    daemon: Callable[..., Any], inventory: Any
) -> None:
    host, received, _ = daemon(tags_response=inventory)
    with pytest.raises(RuntimeError, match="model|inventory"):
        asyncio.run(_provider(host).preflight_model(MODEL))
    assert not received
    assert received.get_paths == ["/api/version", "/api/tags"]


@pytest.mark.parametrize("digest", [None, "", "short", "A" * 64, "g" * 64])
def test_missing_or_invalid_digest_cannot_generate(
    daemon: Callable[..., Any], digest: Any
) -> None:
    host, received, _ = daemon(
        tags_response={"models": [{"name": MODEL, "digest": digest}]}
    )
    with pytest.raises(RuntimeError, match="digest is unavailable"):
        asyncio.run(
            _provider(host).generate_response(
                [LLMMessage("user", "source")], LLMConfig(model=MODEL)
            )
        )
    assert not received


@pytest.mark.parametrize("change", ["version", "digest"])
def test_expected_runtime_change_refused_before_generation(
    daemon: Callable[..., Any], change: str
) -> None:
    version = {"version": "0.35.0"}
    inventory = {"models": [{"name": MODEL, "digest": DIGEST}]}
    host, received, _ = daemon(version_response=version, tags_response=inventory)
    provider = _provider(host)
    identity = asyncio.run(provider.preflight_model(MODEL))
    provider.expected_runtime_identity = identity
    if change == "version":
        version["version"] = "0.35.1"
    else:
        inventory["models"][0]["digest"] = "b" * 64
    with pytest.raises(module.RuntimeIdentityChanged) as error:
        asyncio.run(
            provider.generate_response(
                [LLMMessage("user", "source")], LLMConfig(model=MODEL)
            )
        )
    assert error.value.expected == identity
    assert error.value.observed != identity
    assert not received


@pytest.mark.parametrize("change", ["version", "digest"])
@pytest.mark.parametrize("stream", [False, True])
def test_runtime_change_during_response_cannot_complete(
    daemon: Callable[..., Any], change: str, stream: bool
) -> None:
    version = {"version": "0.35.0"}
    inventory = {"models": [{"name": MODEL, "digest": DIGEST}]}

    def update() -> None:
        if change == "version":
            version["version"] = "0.35.1"
        else:
            inventory["models"][0]["digest"] = "b" * 64

    host, received, _ = daemon(
        version_response=version,
        tags_response=inventory,
        after_chat=update,
        chunks=[json.dumps(_response()).encode() + b"\n"] if stream else None,
    )
    provider = _provider(host)
    yielded: list[str] = []

    async def run() -> None:
        if stream:
            async for part in provider.generate_stream(
                [LLMMessage("user", "source")], LLMConfig(model=MODEL)
            ):
                yielded.append(part)
        else:
            await provider.generate_response(
                [LLMMessage("user", "source")], LLMConfig(model=MODEL)
            )

    with pytest.raises(module.RuntimeIdentityChanged):
        asyncio.run(run())
    assert len(received) == 1
    if stream:
        assert yielded == ["complete answer"]  # Partial delivery is not completion.


@pytest.mark.timeout(120)
@pytest.mark.parametrize("change", ["version", "digest"])
def test_corpus_stops_after_runtime_drift_without_cache_or_checkpoint_admission(
    daemon: Callable[..., Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    from gnn.llm import process_llm

    version = {"version": "0.35.0"}
    inventory = {"models": [{"name": MODEL, "digest": DIGEST}]}

    def update() -> None:
        if change == "version":
            version["version"] = "0.35.1"
        else:
            inventory["models"][0]["digest"] = "b" * 64

    host, received, _ = daemon(
        version_response=version, tags_response=inventory, after_chat=update
    )
    monkeypatch.setenv("OLLAMA_HOST", host)
    target = tmp_path / "models"
    target.mkdir()
    (target / "model.md").write_text(
        "## GNNSection\nActInfPOMDP\n## ModelName\nDrift control\n"
    )
    output = tmp_path / "output"
    assert not process_llm(
        target,
        output,
        provider="ollama",
        model=MODEL,
        custom_prompts=[],
        llm_timeout=90,
    )
    results = json.loads((output / "llm_results.json").read_text())
    checkpoint = json.loads((output / "llm_checkpoint.json").read_text())
    assert results["status"] == "partial" and not results["success"]
    assert results["errors"][0]["stage"] == "runtime_identity"
    assert results["provider_matrix"]["ollama"]["reason"] == "runtime_changed"
    assert results["coverage"]["prompts_completed"] == 0
    assert len(results["unfinished_work"]) == 6
    assert results["cache_stats"]["writes"] == 0
    assert not checkpoint["completed_parts"] and not checkpoint["completed"]
    assert len(received) == 1


@pytest.mark.timeout(180)
def test_cache_only_resume_checks_runtime_again_before_success(
    daemon: Callable[..., Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.llm import process_llm

    version = {"version": "0.35.0"}
    control = {"armed": False}

    def update_after_startup(path: str) -> None:
        if control["armed"] and path == "/api/tags":
            version["version"] = "0.35.1"
            control["armed"] = False

    host, received, _ = daemon(version_response=version, after_get=update_after_startup)
    monkeypatch.setenv("OLLAMA_HOST", host)
    target = tmp_path / "models"
    target.mkdir()
    (target / "model.md").write_text(
        "## GNNSection\nActInfPOMDP\n## ModelName\nResume control\n"
    )
    output = tmp_path / "output"
    options = dict(provider="ollama", model=MODEL, custom_prompts=[], llm_timeout=120)
    assert process_llm(target, output, **options)
    assert len(received) == 6
    control["armed"] = True
    assert not process_llm(target, output, resume=True, **options)
    results = json.loads((output / "llm_results.json").read_text())
    assert results["status"] == "partial" and not results["success"]
    assert results["coverage"]["prompts_completed"] == 6
    assert results["coverage"]["new_provider_requests_completed"] == 0
    assert results["errors"][0]["stage"] == "runtime_identity"
    assert results["final_runtime_identity"] != results["runtime_identity"]
    assert len(received) == 6


def test_http_preflight_rejects_wrong_runtime_before_tags(
    daemon: Callable[..., Any],
) -> None:
    host, received, _ = daemon(version_response={"version": "0.34.9"})
    with pytest.raises(RuntimeError, match="whole-context policy"):
        asyncio.run(_provider(host).preflight_model(MODEL))
    assert received.get_paths == ["/api/version"]
    assert not received


def test_http_preflight_version_and_inventory_share_one_budget(
    daemon: Callable[..., Any],
) -> None:
    host, received, _ = daemon(version_delay=0.08, tags_delay=0.08)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        asyncio.run(_provider(host).preflight_model(MODEL, timeout=0.12))
    # The exception proves the combined 0.16s work did not get a fresh budget
    # for tags; allow scheduling/worker cleanup slack under concurrent CI load.
    assert time.monotonic() - started < 0.8
    assert not received


def test_completed_body_with_late_validation_cannot_be_success(
    daemon: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    host, _, _ = daemon()
    original = OllamaProvider._validate_response

    def delayed(response: Any, model: str) -> dict[str, Any]:
        result = original(response, model)
        time.sleep(0.06)
        return result

    monkeypatch.setattr(OllamaProvider, "_validate_response", staticmethod(delayed))
    with pytest.raises(TimeoutError):
        asyncio.run(
            _provider(host).generate_response(
                [LLMMessage("user", "hello")], LLMConfig(model=MODEL, timeout=0.03)
            )
        )


def test_terminal_stream_with_late_validation_cannot_yield_or_succeed(
    daemon: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    host, _, _ = daemon(chunks=[json.dumps(_response()).encode() + b"\n"])
    original = OllamaProvider._validate_response
    yielded: list[str] = []

    def delayed(response: Any, model: str) -> dict[str, Any]:
        result = original(response, model)
        time.sleep(0.06)
        return result

    monkeypatch.setattr(OllamaProvider, "_validate_response", staticmethod(delayed))

    async def collect() -> None:
        async for text in _provider(host).generate_stream(
            [LLMMessage("user", "hello")], LLMConfig(model=MODEL, timeout=0.03)
        ):
            yielded.append(text)

    with pytest.raises(TimeoutError):
        asyncio.run(collect())
    assert not yielded


def test_terminal_stream_consumer_delay_cannot_renew_completion_budget(
    daemon: Callable[..., Any],
) -> None:
    host, _, _ = daemon(chunks=[json.dumps(_response()).encode() + b"\n"])

    async def collect() -> None:
        async for _ in _provider(host).generate_stream(
            [LLMMessage("user", "hello")], LLMConfig(model=MODEL, timeout=0.03)
        ):
            await asyncio.sleep(0.06)

    with pytest.raises(TimeoutError):
        asyncio.run(collect())


def test_context_and_seed_options_reach_native_payload(
    daemon: Callable[..., Any],
) -> None:
    host, received, _ = daemon()
    provider = OllamaProvider(
        base_url=host, generation_options={"num_ctx": 32768, "seed": 42}
    )
    assert provider.initialize()
    asyncio.run(
        provider.generate_response(
            [LLMMessage(role="user", content="source")], LLMConfig(model=MODEL)
        )
    )
    assert received[0]["body"]["options"]["num_ctx"] == 32768
    assert received[0]["body"]["options"]["seed"] == 42


def test_frozen_empty_generation_options_do_not_inherit_later_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_NUM_CTX", "32768")
    assert OllamaProvider().generation_options == {"num_ctx": 32768}
    assert OllamaProvider(generation_options={}).generation_options == {}


@pytest.mark.timeout(300)
def test_model_digest_change_refuses_resume_and_invalidates_cache(
    daemon: Callable[..., Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.llm import process_llm

    inventory = {"models": [{"name": MODEL, "digest": "a" * 64}]}
    host, received, _ = daemon(tags_response=inventory)
    monkeypatch.setenv("OLLAMA_HOST", host)
    target = tmp_path / "models"
    target.mkdir()
    (target / "model.md").write_text(
        "## GNNSection\nActInfPOMDP\n## ModelName\nDigest control\n"
    )
    output = tmp_path / "output"
    # This tests digest custody, not startup speed: coverage instruments each
    # isolated request process, and six cold imports can exceed 30 seconds.
    options = dict(provider="ollama", model=MODEL, custom_prompts=[], llm_timeout=120)
    assert process_llm(target, output, **options), (
        output / "llm_results.json"
    ).read_text()
    assert len(received) == 6
    inventory["models"][0]["digest"] = "b" * 64
    assert not process_llm(target, output, resume=True, **options)
    refused = json.loads((output / "llm_results.json").read_text())
    assert refused["coverage"]["prompts_completed"] == 0
    assert len(refused["unfinished_work"]) == 6
    assert "runtime/model digest" in refused["errors"][0]["error"]
    assert len(received) == 6
    assert process_llm(target, output, **options), (
        output / "llm_results.json"
    ).read_text()
    assert len(received) == 12


@pytest.mark.parametrize("reason", [None, "length"])
def test_stream_cannot_hide_nonterminal_finish_reason(
    daemon: Callable[..., Any], reason: str | None
) -> None:
    host, _, _ = daemon(
        chunks=[
            json.dumps(
                _response(done_reason=reason, message={"content": "partial"})
            ).encode()
            + b"\n"
        ]
    )

    async def consume() -> list[str]:
        return [
            part
            async for part in _provider(host).generate_stream(
                [LLMMessage("user", "source")], LLMConfig(model=MODEL)
            )
        ]

    with pytest.raises(RuntimeError, match="unfinished"):
        asyncio.run(consume())


@pytest.mark.parametrize("case", ["complete", "missing", "boundary", "overflow"])
def test_bundled_prompts_submit_source_once_and_require_all_answers(
    daemon: Callable[..., Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    from gnn.llm import process_llm

    labels = [
        "explain_model",
        "identify_components",
        "analyze_structure",
        "extract_parameters",
        "practical_applications",
        "technical_description",
        "nontechnical_description",
        "runtime_behavior",
    ]
    answers = {label: f"answer for {label}" for label in labels}
    valid = case == "complete"
    if case == "missing":
        answers.pop(labels[-1])
    elif case in ("boundary", "overflow"):
        answers[labels[0]] = "x" * (128 if case == "boundary" else 129)
    host, received, _ = daemon(
        response=_response(message={"content": json.dumps(answers)})
    )
    monkeypatch.setenv("OLLAMA_HOST", host)
    target = tmp_path / "models"
    target.mkdir()
    source = "## GNNSection\nActInfPOMDP\n## ModelName\nUniqueBatchSource\n"
    (target / "model.md").write_text(source)
    output = tmp_path / "output"
    options = dict(
        provider="ollama",
        model=MODEL,
        batch_prompts=True,
        llm_timeout=30,
        answer_max_chars=128,
    )
    assert process_llm(target, output, **options) is valid
    assert len(received) == 2
    payload = received[1]["body"]
    assert payload["messages"][1]["content"].count(source) == 1
    assert set(payload["format"]["required"]) == set(labels)
    assert all(p["maxLength"] == 128 for p in payload["format"]["properties"].values())
    result = json.loads((output / "llm_results.json").read_text())
    assert result["coverage"]["prompts_completed"] == (9 if valid else 1)
    assert result["coverage"]["source_parts_completed"] == (9 if valid else 1)
    if valid:
        assert process_llm(target, output, resume=True, **options)
        assert len(received) == 2
        assert not process_llm(
            target, output, resume=True, **{**options, "answer_max_chars": 129}
        )
        assert len(received) == 2
    else:
        assert len(result["unfinished_work"]) == 8
        assert result["cache_stats"]["writes"] == 1
        if case in ("boundary", "overflow"):
            assert (
                result["analysis_results"][0]["prompt_outcomes"]["explain_model"][
                    "status"
                ]
                == "incomplete"
            )


@pytest.mark.parametrize("capacity, expected", [(8192, "admitted"), (4096, "refused")])
def test_s1_context_observations_bind_exact_local_http_request(
    daemon, capacity, expected
):
    """Local HTTP fixture proves transport behavior, never native model quality."""
    show = {
        "details": {"format": "gguf"},
        "model_info": {"general.architecture": "llama", "llama.context_length": 8192},
        "template": "fixture-template",
    }
    host, received, _ = daemon(
        show_response=show,
        ps_response={
            "models": [{"name": MODEL, "digest": DIGEST, "context_length": capacity}]
        },
    )
    provider = _provider(host, generation_options={"num_ctx": 8192})
    response = asyncio.run(
        provider.generate_response(
            [LLMMessage(role="user", content="α\r\nω")],
            LLMConfig(model=MODEL, max_tokens=256, context_admission=True),
        )
    )
    assert response.metadata["context_admission"]["status"] == expected
    chats = [r for r in received if r["path"].endswith("/api/chat")]
    assert len(chats) == 1
    assert chats[0]["body"]["truncate"] is False
    assert chats[0]["body"]["shift"] is False
    assert chats[0]["body"]["messages"][0]["content"] == "α\r\nω"
    assert any(p.endswith("/api/ps") for p in received.get_paths)
    if expected == "admitted":
        assert response.metadata["context_admission"]["attention_verified"] is False

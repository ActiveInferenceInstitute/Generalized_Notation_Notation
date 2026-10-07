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
        version_delay: float = 0,
        tags_response: Any = None,
        tags_delay: float = 0,
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
                        {"models": [{"name": MODEL}]}
                        if tags_response is None
                        else tags_response
                    )
                    wait = tags_delay
                else:
                    value = (
                        {"version": "0.35.0"}
                        if version_response is None
                        else version_response
                    )
                    wait = version_delay
                body = json.dumps(value).encode()
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
    daemon: Callable[..., Any],
) -> None:
    line = json.dumps(_response(done=False)).encode() + b"\n"
    host, _, _ = daemon(chunks=[line] * 20, delay=0.025)

    async def collect() -> None:
        async for _ in _provider(host).generate_stream(
            [LLMMessage("user", "hello")], LLMConfig(model=MODEL, timeout=0.1)
        ):
            pass

    started = time.monotonic()
    with pytest.raises((TimeoutError, socket.timeout)):
        asyncio.run(collect())
    assert time.monotonic() - started < 0.25


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
    daemon: Callable[..., Any],
) -> None:
    host, received, _ = daemon(version_delay=0.055, delay=0.055)
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        asyncio.run(
            _provider(host).generate_response(
                [LLMMessage("user", "hello")], LLMConfig(model=MODEL, timeout=0.08)
            )
        )
    assert len(received) == 1
    assert time.monotonic() - start < 0.2


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
    assert len(requests) == 1
    assert not requests[0]._guard.is_alive()


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
    }
    assert received.get_paths == ["/prefix/api/version", "/prefix/api/tags"]
    assert not received


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
    host, received, _ = daemon(version_delay=0.05, tags_delay=0.05)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        asyncio.run(_provider(host).preflight_model(MODEL, timeout=0.075))
    assert time.monotonic() - started < 0.2
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

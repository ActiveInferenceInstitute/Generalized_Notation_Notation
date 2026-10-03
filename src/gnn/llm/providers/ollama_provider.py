#!/usr/bin/env python3
"""Ollama chat transport with intact messages and one request budget.

Use the local runtime's HTTP API directly: older Python SDKs cannot express
``truncate=False``/``shift=False`` and the CLI cannot preserve chat options.
No Python client or CLI executable is required; an explicitly configured
Ollama daemon and model must already be available. Pipeline calls additionally
run in a killable request worker, which owns the authoritative hard deadline.
"""

from __future__ import annotations

import asyncio
import http.client
import json
import logging
import math
import os
import re
import socket
import threading
import time
from typing import Any, AsyncGenerator, List, Optional
from urllib.parse import SplitResult, urlsplit

from ..defaults import DEFAULT_OLLAMA_MODEL
from .base_provider import (
    BaseLLMProvider,
    LLMConfig,
    LLMMessage,
    LLMResponse,
    ProviderType,
)

logger = logging.getLogger(__name__)
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_MAX_STREAM_LINE_BYTES = 1024 * 1024


def _positive_timeout(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("Ollama timeout must be finite and positive")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Ollama timeout must be finite and positive")
    return timeout


def _endpoint(value: str) -> SplitResult:
    # OLLAMA_HOST commonly uses host:port without a URL scheme.
    parsed = urlsplit(value if "://" in value else f"http://{value}")
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Ollama host must be an HTTP(S) URL without credentials")
    _ = parsed.port  # Validate the port before any network operation.
    return parsed


class _ChatRequest:
    """Own a single connection; reads never renew the original deadline."""

    def __init__(self, endpoint: SplitResult, timeout: float) -> None:
        self.deadline = time.monotonic() + timeout
        connection_type = (
            http.client.HTTPSConnection
            if endpoint.scheme == "https"
            else http.client.HTTPConnection
        )
        self.connection = connection_type(
            endpoint.hostname or "localhost", endpoint.port, timeout=timeout
        )
        self.path = endpoint.path.rstrip("/") + "/api/chat"
        self.version_path = endpoint.path.rstrip("/") + "/api/version"
        self.tags_path = endpoint.path.rstrip("/") + "/api/tags"
        self.response: http.client.HTTPResponse | None = None
        self.socket: socket.socket | None = None
        self.bytes_read = 0
        self.server_version: str | None = None
        self._expired = False
        self._guard = threading.Timer(self.remaining(), self._expire)
        self._guard.daemon = True
        self._guard.start()

    def remaining(self) -> float:
        remaining = self.deadline - time.monotonic()
        if self._expired or remaining <= 0:
            raise TimeoutError("Ollama request deadline exhausted")
        return remaining

    def open(self, payload: dict[str, Any]) -> None:
        self.check_version()
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._exchange("POST", self.path, body)

    def check_version(self) -> None:
        """Verify the runtime's strict-context controls before using its API."""
        self._exchange("GET", self.version_path)
        version_response = json.loads(self.read_all())
        version = (
            version_response.get("version")
            if isinstance(version_response, dict)
            else None
        )
        if (
            not isinstance(version, str)
            or re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", version)
            is None
        ):
            raise RuntimeError(
                "Ollama whole-context policy requires a verified stable runtime >=0.35.0; malformed or prerelease version"
            )
        if tuple(map(int, version.split("."))) < (0, 35, 0):
            raise RuntimeError(
                f"Ollama {version} cannot verify the whole-context policy; stable runtime >=0.35.0 required"
            )
        self.server_version = version
        if self.response is not None:
            self.response.close()

    def _exchange(self, method: str, path: str, body: bytes | None = None) -> None:
        self.connection.timeout = self.remaining()
        self.socket = None
        try:
            self.connection.request(
                method, path, body, {"Content-Type": "application/json"}
            )
            # Retain the socket when HTTP/1.0 detaches it from HTTPConnection.
            # The response's file object still owns its descriptor.
            self.socket = self.connection.sock
            if self.socket is not None:
                self.socket.settimeout(self.remaining())
            self.response = self.connection.getresponse()
        except Exception:
            self.remaining()  # Convert deadline-interrupted headers to timeout.
            raise
        self.remaining()
        if self.response.status != 200:
            body = self.read_all()
            try:
                error = json.loads(body).get("error", "request rejected")
            except (ValueError, AttributeError):
                error = "request rejected"
            # Never redirect/retry a scientific prompt or place it in argv.
            raise RuntimeError(
                f"Ollama HTTP {self.response.status}: {str(error)[:1024]}"
            )

    def read_chunk(self) -> bytes:
        if self.response is None:
            raise RuntimeError("Ollama response not opened")
        remaining = self.remaining()
        if self.response.isclosed():
            return b""
        if self.socket is not None:
            self.socket.settimeout(remaining)
        try:
            chunk = self.response.read1(65536)
        except Exception:
            self.remaining()
            raise
        self.remaining()
        self.bytes_read += len(chunk)
        if self.bytes_read > _MAX_RESPONSE_BYTES:
            raise RuntimeError("Ollama response exceeds the 8 MiB transport limit")
        return chunk

    def read_all(self) -> bytes:
        chunks = []
        while chunk := self.read_chunk():
            chunks.append(chunk)
        return b"".join(chunks)

    def _interrupt_socket(self) -> None:
        for active_socket in (self.socket, self.connection.sock):
            if active_socket is not None:
                try:
                    active_socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def _expire(self) -> None:
        self._expired = True
        self._interrupt_socket()

    def close(self) -> None:
        # Interrupt an in-progress read on cancellation as well as closing the
        # response file. The outer worker remains the hard containment boundary.
        self._interrupt_socket()
        self._guard.cancel()
        self._guard.join()
        if self.response is not None:
            self.response.close()
        self.connection.close()


class OllamaProvider(BaseLLMProvider):
    """Ollama implementation of the unchanged LLM provider interface."""

    AVAILABLE_MODELS = (
        "smollm2:135m-instruct-q4_K_S",
        "gemma3:4b",
        "ministral-3:3b",
        "mistral:7b",
        "llama3.1:8b",
        "qwen2:7b",
        "llama3.1:70b",
    )
    DEFAULT_MODEL = DEFAULT_OLLAMA_MODEL

    def __init__(self, api_key: Optional[str] = None, **kwargs: Any) -> None:
        """Resolve explicit host/options without starting or changing a model."""
        super().__init__(api_key=api_key, **kwargs)
        self.base_url = (
            kwargs.get("base_url")
            or os.getenv("OLLAMA_HOST")
            or "http://127.0.0.1:11434"
        )
        self.default_model_override = kwargs.get("default_model")
        self.default_max_tokens = kwargs.get("default_max_tokens", 256)
        self.default_timeout = _positive_timeout(
            kwargs.get("timeout")
            if kwargs.get("timeout") is not None
            else os.getenv("OLLAMA_TIMEOUT", "60")
        )
        self._endpoint: SplitResult | None = None

    @property
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.OLLAMA

    @property
    def default_model(self) -> str:
        """Return the exact configured default model."""
        return self.default_model_override or self.DEFAULT_MODEL

    @property
    def available_models(self) -> List[str]:
        """Return the advertised models without selecting a replacement."""
        return list(self.AVAILABLE_MODELS)

    def initialize(self) -> bool:
        """Initialize transport configuration; model preflight stays explicit."""
        try:
            self._endpoint = _endpoint(self.base_url)
            self._is_initialized = True
            return True
        except (TypeError, ValueError):
            logger.error("Invalid Ollama HTTP transport configuration")
            self._is_initialized = False
            return False

    def validate_config(self, config: LLMConfig) -> bool:
        """Reject invalid generation options and budgets before dispatch."""
        tokens = (
            config.max_tokens
            if config.max_tokens is not None
            else self.default_max_tokens
        )
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            return False
        if config.temperature is not None and (
            not math.isfinite(config.temperature)
            or not 0.0 <= config.temperature <= 2.0
        ):
            return False
        if config.timeout is not None:
            try:
                _positive_timeout(config.timeout)
            except (TypeError, ValueError):
                return False
        return True

    def _request(
        self, messages: List[LLMMessage], config: LLMConfig, *, stream: bool
    ) -> tuple[_ChatRequest, dict[str, Any]]:
        if not self.is_initialized or self._endpoint is None:
            raise RuntimeError("Ollama provider not initialized")
        if not self.validate_config(config):
            raise ValueError("Invalid configuration parameters")
        timeout = min(
            self.default_timeout,
            _positive_timeout(config.timeout)
            if config.timeout is not None
            else self.default_timeout,
        )
        options = {
            "num_predict": config.max_tokens
            if config.max_tokens is not None
            else self.default_max_tokens,
            "temperature": config.temperature
            if config.temperature is not None
            else 0.2,
        }
        for key in ("top_p", "frequency_penalty", "presence_penalty"):
            value = getattr(config, key)
            if value is not None:
                options[key] = value
        payload = {
            "model": config.model or self.default_model,
            "messages": [
                {"role": msg.role, "content": msg.content} for msg in messages
            ],
            "options": options,
            "stream": stream,
            "truncate": False,
            "shift": False,
        }
        return _ChatRequest(self._endpoint, timeout), payload

    @staticmethod
    def _validate_response(response: Any, model: str) -> dict[str, Any]:
        if not isinstance(response, dict):
            raise RuntimeError("Ollama returned a malformed chat response")
        if response.get("error"):
            raise RuntimeError(f"Ollama rejected chat: {str(response['error'])[:1024]}")
        if response.get("model") != model:
            raise RuntimeError("Ollama response model differs from configured model")
        message = response.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise RuntimeError("Ollama returned a malformed chat message")
        return response

    async def preflight_model(
        self, model: str, timeout: Optional[float] = None
    ) -> dict[str, Any]:
        """Verify the exact model on the configured HTTP host without generation."""
        transport, _ = self._request(
            [], LLMConfig(model=model, timeout=timeout), stream=False
        )

        def call() -> dict[str, Any]:
            try:
                transport.check_version()
                transport._exchange("GET", transport.tags_path)
                tags = json.loads(transport.read_all())
                models = tags.get("models") if isinstance(tags, dict) else None
                if not isinstance(models, list):
                    raise RuntimeError("Ollama returned a malformed model inventory")
                if not any(
                    isinstance(entry, dict) and entry.get("name") == model
                    for entry in models
                ):
                    raise RuntimeError(
                        "Configured Ollama model is not installed on the configured HTTP host"
                    )
                transport.remaining()
                return {
                    "model": model,
                    "server_version": transport.server_version,
                    "context_policy": "whole-context-v1",
                    "status": "ready",
                }
            finally:
                transport.close()

        try:
            result = await asyncio.to_thread(call)
        finally:
            transport.close()
        transport.remaining()
        return result

    async def generate_response(
        self, messages: List[LLMMessage], config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """Generate from all messages without SDK/CLI retries or truncation."""
        transport, payload = self._request(
            messages, config or LLMConfig(), stream=False
        )

        def call() -> dict[str, Any]:
            try:
                transport.open(payload)
                response = self._validate_response(
                    json.loads(transport.read_all()), payload["model"]
                )
                transport.remaining()
                return response
            finally:
                transport.close()

        try:
            response = await asyncio.to_thread(call)
        finally:
            transport.close()
        transport.remaining()
        if (
            response.get("done") is not True
            or not response["message"]["content"].strip()
        ):
            raise RuntimeError("Ollama returned an unfinished or empty chat response")
        usage = {}
        for source, target in (
            ("prompt_eval_count", "prompt_tokens"),
            ("eval_count", "completion_tokens"),
        ):
            value = response.get(source)
            if isinstance(value, int) and not isinstance(value, bool):
                usage[target] = value
        return LLMResponse(
            content=response["message"]["content"],
            model_used=response["model"],
            provider=self.provider_type.value,
            usage=usage or None,
            finish_reason=response.get("done_reason"),
            metadata={
                "raw": {
                    key: value for key, value in response.items() if key != "message"
                },
                "transport": "ollama_http_chat",
                "truncate": False,
                "shift": False,
                "server_version": transport.server_version,
                "context_policy": "whole-context-v1",
            },
        )

    async def generate_stream(
        self, messages: List[LLMMessage], config: Optional[LLMConfig] = None
    ) -> AsyncGenerator[str, None]:
        """Stream role-preserving chat with one budget across every chunk."""
        transport, payload = self._request(messages, config or LLMConfig(), stream=True)
        buffer = b""
        done = False
        try:
            await asyncio.to_thread(transport.open, payload)
            while not done:
                chunk = await asyncio.to_thread(transport.read_chunk)
                buffer += chunk
                lines = buffer.split(b"\n")
                buffer = lines.pop()
                if not chunk and buffer:
                    lines.append(buffer)
                    buffer = b""
                if len(buffer) > _MAX_STREAM_LINE_BYTES or any(
                    len(line) > _MAX_STREAM_LINE_BYTES for line in lines
                ):
                    raise RuntimeError(
                        "Ollama stream line exceeds the 1 MiB transport limit"
                    )
                for line in lines:
                    if not line.strip():
                        continue
                    transport.remaining()
                    response = self._validate_response(
                        json.loads(line), payload["model"]
                    )
                    transport.remaining()
                    if response["message"]["content"]:
                        yield response["message"]["content"]
                    transport.remaining()
                    if response.get("done") is True:
                        done = True
                        break
                if not chunk and not done:
                    raise RuntimeError(
                        "Ollama stream ended without a completed response"
                    )
        finally:
            transport.close()
        transport.remaining()

    def analyze(self, content: str, task: str) -> str:
        """Perform analysis on GNN content."""
        import asyncio
        import concurrent.futures

        prompt = f"Analyze this GNN model for {task}: {content}"
        deadline = time.monotonic() + self.default_timeout
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            running_loop = False
        else:
            running_loop = True

        async def _run() -> LLMResponse:
            """Run operation."""
            return await self.generate_response(
                [LLMMessage(role="user", content=prompt)],
                LLMConfig(timeout=_positive_timeout(deadline - time.monotonic())),
            )

        def _extract(result: Any) -> str:
            """Extract operation."""
            return result.content if hasattr(result, "content") else str(result)

        try:
            if running_loop:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(lambda: asyncio.run(_run()))
                    result = future.result(
                        timeout=_positive_timeout(deadline - time.monotonic())
                    )
            else:
                result = asyncio.run(_run())
            return _extract(result)
        except Exception as e:
            logger.error(f"Ollama analysis failed: {e}")
            return f"Analysis failed: {e}"

    async def close(self) -> None:
        """Close the provider; each request owns and closes its connection."""
        self._is_initialized = False

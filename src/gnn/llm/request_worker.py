"""One pinned LLM request per killable process, including provider cleanup."""

from __future__ import annotations

import asyncio
import json
import math
import sys
import time
from dataclasses import asdict
from typing import Any

from gnn.execute.subprocess_envelope import CancelToken, run_subprocess_envelope

from .llm_processor import LLMProcessor
from .providers.base_provider import LLMConfig, LLMMessage, LLMResponse, ProviderType


async def isolated_request(**payload: Any) -> LLMResponse:
    """Bound initialization, request, retrieval, and cleanup by one deadline.

    The worker inherits credentials through the environment. They never enter
    the request payload or receipt. Timeout kills its entire process group,
    including any Ollama CLI child, before this coroutine returns.
    """
    timeout = payload["timeout"]
    if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("LLM request timeout must be finite and positive")
    request_deadline = time.monotonic() + timeout
    token = CancelToken()
    task = asyncio.create_task(
        asyncio.to_thread(
            run_subprocess_envelope,
            [sys.executable, "-m", "gnn.llm.request_worker"],
            input=json.dumps(payload),
            timeout=timeout,
            deadline_monotonic=request_deadline,
            sandbox=False,
            cancel_token=token,
        )
    )
    try:
        envelope = await asyncio.shield(task)
    except asyncio.CancelledError:
        token.cancel("LLM caller cancelled")
        await asyncio.shield(task)
        raise
    if not envelope["success"]:
        if envelope.get("error_type") == "TimeoutExpired":
            raise asyncio.TimeoutError
        try:
            diagnostic = json.loads(envelope.get("stdout", ""))
        except (ValueError, TypeError):
            diagnostic = {}
        category = diagnostic.get("error_category")
        if category == "authentication":
            raise RuntimeError(f"{payload['provider']}: 401 authentication failed")
        if category == "timeout":
            raise asyncio.TimeoutError
        diagnoses = {
            "context_limit": "Configured LLM model context cannot admit the complete input",
            "runtime_unsupported": "Configured Ollama runtime cannot verify the whole-context policy",
            "model_unavailable": "Configured Ollama model is not installed",
        }
        if category in diagnoses:
            raise RuntimeError(diagnoses[category])
        # Worker diagnostics can contain provider error bodies; keep the
        # receipt concise and never expose environment credentials.
        raise RuntimeError("Configured LLM provider/model request failed")

    def verify_remaining() -> None:
        if time.monotonic() >= request_deadline:
            raise asyncio.TimeoutError

    verify_remaining()
    response = json.loads(envelope["stdout"])
    verify_remaining()
    if payload.get("preflight"):
        ready = LLMResponse("ready", payload["model"], payload["provider"])
        verify_remaining()
        return ready
    if response["provider"] != payload["provider"]:
        raise RuntimeError("LLM response provider differs from configured provider")
    if response["model_used"] != payload["model"]:
        raise RuntimeError("LLM response model differs from configured model")
    validated = LLMResponse(**response)
    verify_remaining()
    return validated


async def _request(payload: dict[str, Any]) -> dict[str, Any]:
    provider = ProviderType(payload["provider"])
    processor = LLMProcessor(
        preferred_providers=[provider],
        provider_configs={
            provider.value: {
                "default_model": payload["model"],
                "timeout": payload["timeout"],
                **(
                    {"base_url": payload["endpoint_url"]}
                    if payload.get("endpoint_url")
                    else {}
                ),
            }
        },
    )
    try:
        if not await processor.initialize():
            raise RuntimeError("Configured provider could not initialize")
        if payload.get("preflight"):
            if provider == ProviderType.OLLAMA:
                from .processor import _validate_model_name
                from .providers.ollama_provider import OllamaProvider

                configured = processor.get_provider(provider)
                if not isinstance(configured, OllamaProvider):
                    raise RuntimeError("Configured Ollama provider is unavailable")
                await configured.preflight_model(
                    _validate_model_name(payload["model"]), timeout=payload["timeout"]
                )
            return {"ready": True}
        response = await processor.get_response(
            messages=[LLMMessage(**message) for message in payload["messages"]],
            provider_type=provider,
            model_name=payload["model"],
            max_tokens=payload["max_tokens"],
            temperature=0.2,
            config=LLMConfig(model=payload["model"], timeout=payload["timeout"]),
        )
        return asdict(response)
    finally:
        await processor.close()


def main() -> None:
    """Read one local request from stdin and emit a JSON response only."""
    payload = json.load(sys.stdin)
    try:
        response = asyncio.run(_request(payload))
    except Exception as exc:
        from .processor import _classify_auth_error

        message = str(exc).lower()
        if isinstance(exc, TimeoutError):
            category = "timeout"
        elif _classify_auth_error(str(exc)):
            category = "authentication"
        elif "whole-context policy" in message or "stable runtime" in message:
            category = "runtime_unsupported"
        elif "configured ollama model is not installed" in message:
            category = "model_unavailable"
        elif "context" in message and any(
            word in message for word in ("length", "window", "exceed")
        ):
            category = "context_limit"
        else:
            category = "provider_failure"
        print(
            json.dumps(
                {"error_category": category, "provider": payload.get("provider")}
            )
        )
        raise SystemExit(1) from None
    print(json.dumps(response))


if __name__ == "__main__":
    main()

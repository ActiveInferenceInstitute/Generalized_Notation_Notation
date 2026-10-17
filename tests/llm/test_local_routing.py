"""Explicit local selection must not trigger another provider's network calls."""

from __future__ import annotations

import asyncio
import socket
from typing import Any

import pytest

from gnn.llm.llm_processor import AnalysisType, LLMProcessor
from gnn.llm.providers.base_provider import LLMMessage, ProviderType


@pytest.mark.parametrize("operation", ["response", "analysis"])
def test_explicit_local_failure_never_calls_cloud(operation: str) -> None:
    with socket.socket() as closed:
        closed.bind(("127.0.0.1", 0))
        host = f"http://127.0.0.1:{closed.getsockname()[1]}"
        processor = LLMProcessor(
            preferred_providers=[ProviderType.OLLAMA],
            provider_configs={"ollama": {"base_url": host, "timeout": 0.2}},
        )
        asyncio.run(processor.initialize())

        class CloudSentinel:
            provider_type = ProviderType.OPENAI
            calls = 0

            async def generate_response(self, *args: Any, **kwargs: Any) -> Any:
                self.calls += 1
                raise AssertionError("Cloud provider must not be called")

        cloud = CloudSentinel()
        processor.providers[ProviderType.OPENAI] = cloud  # type: ignore[assignment]
        for _ in range(3):
            with pytest.raises((ConnectionError, TimeoutError, RuntimeError)):
                if operation == "response":
                    asyncio.run(
                        processor.get_response(
                            [LLMMessage(role="user", content="local only")],
                            provider_type=ProviderType.OLLAMA,
                        )
                    )
                else:
                    asyncio.run(
                        processor.analyze_gnn(
                            "local only",
                            AnalysisType.SUMMARY,
                            provider_type=ProviderType.OLLAMA,
                        )
                    )
        assert cloud.calls == 0


def test_unregistered_explicit_provider_does_not_select_another() -> None:
    processor = LLMProcessor(preferred_providers=[ProviderType.OLLAMA])
    asyncio.run(processor.initialize())
    with pytest.raises(RuntimeError, match="Requested provider openai is unavailable"):
        asyncio.run(
            processor.get_response(
                [LLMMessage(role="user", content="explicit")],
                provider_type=ProviderType.OPENAI,
            )
        )

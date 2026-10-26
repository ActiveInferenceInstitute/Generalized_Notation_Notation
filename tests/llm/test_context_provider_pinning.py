"""Offline public APIs cannot recover across an admitted context boundary."""

import asyncio

import pytest

from gnn.llm.llm_processor import AnalysisType, LLMProcessor
from gnn.llm.providers.base_provider import (
    LLMConfig,
    LLMMessage,
    LLMResponse,
    ProviderType,
)

pytestmark = pytest.mark.unit


class Provider:
    def __init__(self, kind, fails=False):
        self.provider_type, self.fails, self.calls = kind, fails, 0

    def format_gnn_analysis_prompt(self, *_):
        return [LLMMessage(role="user", content="fixture")]

    async def generate_response(self, *_):
        self.calls += 1
        if self.fails:
            raise RuntimeError("local refusal")
        return LLMResponse("cloud response", "cloud", self.provider_type.value)

    async def generate_stream(self, *_):
        self.calls += 1
        yield "unexpected cloud stream"


@pytest.mark.parametrize("operation", ["response", "analysis", "stream", "compare"])
@pytest.mark.parametrize("cloud_first", [False, True])
def test_context_profile_refuses_cloud_and_never_recovers(
    operation, cloud_first, monkeypatch
):
    local, cloud = Provider(ProviderType.OLLAMA, True), Provider(ProviderType.OPENAI)
    processor = LLMProcessor()
    processor.providers = {
        p.provider_type: p for p in ([cloud, local] if cloud_first else [local, cloud])
    }
    processor._initialized = True
    monkeypatch.setattr(
        processor,
        "get_best_provider_for_task",
        lambda _: next(iter(processor.providers.values())),
    )
    config = LLMConfig(context_admission=True)

    async def run():
        if operation == "response":
            return await processor.get_response(
                [LLMMessage(role="user", content="fixture")], config=config
            )
        if operation == "analysis":
            return await processor.analyze_gnn(
                "fixture", AnalysisType.SUMMARY, config=config
            )
        if operation == "compare":
            return await processor.compare_providers("fixture", config=config)
        return [
            text
            async for text in processor.generate_stream(
                [LLMMessage(role="user", content="fixture")], config=config
            )
        ]

    with pytest.raises((RuntimeError, ValueError)):
        asyncio.run(run())
    assert cloud.calls == 0
    assert local.calls == (
        1 if not cloud_first and operation in {"response", "analysis"} else 0
    )


def test_legacy_default_response_keeps_its_existing_recovery():
    local, cloud = Provider(ProviderType.OLLAMA, True), Provider(ProviderType.OPENAI)
    processor = LLMProcessor()
    processor.providers = {ProviderType.OLLAMA: local, ProviderType.OPENAI: cloud}
    processor._initialized = True
    response = asyncio.run(
        processor.get_response([LLMMessage(role="user", content="fixture")])
    )
    assert response.provider == "openai" and local.calls == cloud.calls == 1

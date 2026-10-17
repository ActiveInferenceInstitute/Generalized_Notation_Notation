#!/usr/bin/env python3
"""
Test LLM Ollama Provider

Unit tests cover `OllamaProvider` configuration and validation without a running
daemon. Chat/stream tests use `asyncio.run()` so they collect under `--strict-markers`
without `pytest-anyio`; they carry the `needs_ollama` marker and conftest
gates them on a reachable local daemon (tests/helpers/toolchain_probes.py).
Model name defaults follow `OLLAMA_TEST_MODEL`, then `OLLAMA_MODEL`,
then ``llm.defaults.DEFAULT_OLLAMA_MODEL`` (smollm2 instruct).
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# SC-44: any test in this file may perform real LLM inference whenever a local
# Ollama daemon is up. The default suite filters the whole file via
# ``-m "not ollama"`` (see justfile test-cov / ci.yml); run explicitly with
# ``-m ollama``.
pytestmark = pytest.mark.ollama


from gnn.llm.defaults import DEFAULT_OLLAMA_MODEL

OLLAMA_TEST_MODEL = os.getenv(
    "OLLAMA_TEST_MODEL", os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
)


@pytest.mark.unit
@pytest.mark.needs_ollama
def test_import_ollama_provider() -> Any:
    # Import failure under a reachable daemon is a loud bug, not a skip.
    from gnn.llm.providers.ollama_provider import OllamaProvider  # noqa: F401


@pytest.mark.unit
def test_ollama_provider_initialize(monkeypatch: Any) -> Any:
    from gnn.llm.providers.ollama_provider import OllamaProvider

    provider = OllamaProvider()
    ok = provider.initialize()

    # Transport initialization validates configuration without probing a daemon.
    assert ok is True
    assert provider.is_initialized is True
    assert provider.get_provider_info()["provider_type"] == "ollama"


@pytest.mark.unit
@pytest.mark.timeout(30)  # Prevent hanging if Ollama is slow
@pytest.mark.needs_ollama
def test_ollama_simple_chat(monkeypatch: Any) -> Any:

    async def _run() -> None:
        from gnn.llm.providers.base_provider import LLMConfig, LLMMessage
        from gnn.llm.providers.ollama_provider import OllamaProvider

        provider = OllamaProvider()
        assert provider.initialize() is True

        messages: list[Any] = [
            LLMMessage(role="system", content="You are a helpful assistant."),
            LLMMessage(role="user", content="Reply with the word 'ok'."),
        ]
        config = LLMConfig(model=OLLAMA_TEST_MODEL, max_tokens=32, temperature=0.0)

        result = await provider.generate_response(messages, config)
        assert isinstance(result.content, str)
        assert len(result.content) > 0
        assert result.provider == "ollama"
        assert result.finish_reason == "stop"

    asyncio.run(_run())


@pytest.mark.unit
@pytest.mark.timeout(120)  # CLI stream can approach OLLAMA_TIMEOUT on loaded hosts
@pytest.mark.needs_ollama
def test_ollama_streaming(monkeypatch: Any) -> Any:

    async def _run() -> None:
        from gnn.llm.providers.base_provider import LLMConfig, LLMMessage
        from gnn.llm.providers.ollama_provider import OllamaProvider

        provider = OllamaProvider()
        assert provider.initialize() is True

        messages: list[Any] = [
            LLMMessage(
                role="user", content="Give a 1-sentence summary of Active Inference."
            )
        ]
        config = LLMConfig(
            model=OLLAMA_TEST_MODEL, max_tokens=64, temperature=0.2, stream=True
        )

        chunks: list[str] = []
        async for chunk in provider.generate_stream(messages, config):
            chunks.append(chunk)

        text = "".join(chunks)
        assert isinstance(text, str)
        assert len(text) > 0

    asyncio.run(_run())


@pytest.mark.needs_ollama
@pytest.mark.integration
@pytest.mark.slow  # This test makes real LLM calls which can be slow
@pytest.mark.timeout(30)  # Prevent hanging during pipeline runs
def test_processor_uses_ollama_when_no_keys(monkeypatch: Any) -> Any:
    # Clear cloud keys to force local provider preference
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.setenv("DEFAULT_PROVIDER", "ollama")

    async def _run() -> None:
        from gnn.llm.llm_processor import LLMConfig, LLMProcessor
        from gnn.llm.providers.base_provider import LLMMessage, ProviderType

        processor = LLMProcessor()
        initialized = await processor.initialize()
        assert initialized is True

        messages: list[Any] = [
            LLMMessage(role="system", content="You are a helpful assistant."),
            LLMMessage(role="user", content="Reply with the word 'ok'."),
        ]
        result = await processor.get_response(
            messages,
            provider_type=ProviderType.OLLAMA,
            max_tokens=32,
            temperature=0.0,
            config=LLMConfig(model=OLLAMA_TEST_MODEL),
        )
        assert isinstance(result.content, str)
        assert len(result.content) > 0

    asyncio.run(_run())


@pytest.mark.unit
def test_llm_processor_includes_ollama_timeout_from_env(monkeypatch: Any) -> Any:
    """Bare ``LLMProcessor()`` must apply ``get_default_provider_configs`` (OLLAMA_TIMEOUT, etc.)."""
    monkeypatch.setenv("OLLAMA_TIMEOUT", "99")
    from gnn.llm.llm_processor import LLMProcessor

    proc = LLMProcessor()
    assert proc.provider_configs.get("ollama", {}).get("timeout") == 99.0


@pytest.mark.unit
class TestOllamaProviderConfig:
    """OllamaProvider behavior that does not require a running Ollama server."""

    def test_validate_config_rejects_non_positive_max_tokens(self) -> Any:
        from gnn.llm.providers.base_provider import LLMConfig
        from gnn.llm.providers.ollama_provider import OllamaProvider

        p = OllamaProvider()
        assert p.validate_config(LLMConfig(max_tokens=0)) is False
        assert p.validate_config(LLMConfig(max_tokens=-1)) is False

    def test_validate_config_rejects_temperature_out_of_range(self) -> Any:
        from gnn.llm.providers.base_provider import LLMConfig
        from gnn.llm.providers.ollama_provider import OllamaProvider

        p = OllamaProvider()
        assert p.validate_config(LLMConfig(temperature=-0.1)) is False
        assert p.validate_config(LLMConfig(temperature=2.1)) is False

    def test_validate_config_accepts_defaults_and_edges(self) -> Any:
        from gnn.llm.providers.base_provider import LLMConfig
        from gnn.llm.providers.ollama_provider import OllamaProvider

        p = OllamaProvider()
        assert p.validate_config(LLMConfig()) is True
        assert p.validate_config(LLMConfig(max_tokens=1, temperature=0.0)) is True
        assert p.validate_config(LLMConfig(max_tokens=512, temperature=2.0)) is True

    def test_default_model_and_available_models(self) -> Any:
        from gnn.llm.providers.ollama_provider import OllamaProvider

        p = OllamaProvider()
        assert p.default_model == OllamaProvider.DEFAULT_MODEL
        assert p.available_models == list(OllamaProvider.AVAILABLE_MODELS)

        q = OllamaProvider(default_model="custom:tag")
        assert q.default_model == "custom:tag"

    def test_provider_type_and_info_uninitialized(self) -> Any:
        from gnn.llm.providers.base_provider import ProviderType
        from gnn.llm.providers.ollama_provider import OllamaProvider

        p = OllamaProvider()
        assert p.provider_type == ProviderType.OLLAMA
        info = p.get_provider_info()
        assert info["provider_type"] == "ollama"
        assert info["is_initialized"] is False
        assert info["default_model"] == OllamaProvider.DEFAULT_MODEL

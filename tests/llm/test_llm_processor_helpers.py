"""Deterministic tests for llm.processor step-13 helpers and the shared
prompt executor (cache / timeout / auth fail-fast funnel)."""

import asyncio
import sys
from pathlib import Path
from typing import Any, cast

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.llm.cache import LLMCache
from gnn.llm.llm_processor import LLMProcessor, ProviderType, _merge_provider_configs
from gnn.llm.processor import (
    ModelNameValidationError,
    _classify_auth_error,
    _execute_prompt,
    _llm_file_sort_key,
    _model_is_cached,
    _optional_positive_int,
    _prompt_fallback_text,
    _resolve_llm_budget_seconds,
    _resolve_llm_max_files,
    _validate_model_name,
)

pytestmark = pytest.mark.unit


class TestValidateModelName:
    """Flag-injection guard: model names must never carry CLI metacharacters."""

    @pytest.mark.parametrize(
        "name",
        [
            "llama3",
            "llama3:8b",
            "gpt-oss:20b",
            "myorg/codellama-34b:q4",
        ],
    )
    def test_accepts_plain_identifiers(self, name: str) -> None:
        assert _validate_model_name(name) == name

    @pytest.mark.parametrize(
        "name",
        [
            "-o rider",  # space + payload
            "--help",  # leading dash = flag injection
            "-e import os",  # flag + code
            "llama3;rm -rf /",  # shell metacharacter
            "llama3 && touch /tmp/pwned",
            "llama3$(reboot)",
            "llama3\n--modelfile",
            "",  # empty
            "   ",  # whitespace-only
        ],
    )
    def test_rejects_injection_attempts(self, name: str) -> None:
        with pytest.raises(ModelNameValidationError):
            _validate_model_name(name)


class TestModelIsCachedValidation:
    def test_invalid_model_name_short_circuits_to_false(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        result = _model_is_cached("--pull ollama", logging.getLogger("llm.test"))
        assert result is False
        assert any(
            "Rejected invalid model name" in rec.message for rec in caplog.records
        )


class TestClassifyAuthError:
    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            ("Error code: 401 - invalid_api_key", "unknown"),
            ("403 Incorrect API key provided by OpenAI", "openai"),
            ("authentication error from OpenRouter: 401", "openrouter"),
            ("perplexity returned 401", "perplexity"),
            ("ollama auth rejected 403", "ollama"),
            ("401 unauthorized", "unknown"),
            ("connection refused", None),
            ("rate limit exceeded", None),
        ],
    )
    def test_classification(self, message: str, expected: str | None) -> None:
        assert _classify_auth_error(message) == expected


class TestPromptFallbackText:
    def test_structured_wording(self) -> None:
        text = _prompt_fallback_text("analyze_structure", custom=False)
        assert text == (
            "LLM analysis for analyze_structure was not available. "
            "Please ensure that Ollama is running and the required model is installed."
        )

    def test_custom_wording(self) -> None:
        text = _prompt_fallback_text("technical_description", custom=True)
        assert text.startswith(
            "LLM analysis for custom prompt technical_description was not available."
        )


class TestBudgetResolvers:
    def test_cli_kwarg_wins(self) -> None:
        assert (
            _resolve_llm_budget_seconds({"llm_timeout": 33}, {"timeout_seconds": 99})
            == 33
        )

    def test_total_budget_outranks_llm_timeout(self) -> None:
        assert (
            _resolve_llm_budget_seconds({"total_budget": 44, "llm_timeout": 33}, {})
            == 44
        )

    def test_config_then_default(self) -> None:
        assert _resolve_llm_budget_seconds({}, {"timeout_seconds": 55}) == 55
        assert _resolve_llm_budget_seconds({}, {}) == 600

    def test_max_files_precedence(self) -> None:
        assert _resolve_llm_max_files({"max_files": "7"}, {"max_files": 99}) == 7
        assert _resolve_llm_max_files({}, {"max_files": 3}) == 3
        assert _resolve_llm_max_files({}, {}) is None

    def test_optional_positive_int_normalization(self) -> None:
        assert _optional_positive_int("5") == 5
        assert _optional_positive_int(0) is None
        assert _optional_positive_int(-2) is None
        assert _optional_positive_int(None) is None
        assert _optional_positive_int("not-a-number") is None


class TestFileSortKey:
    def test_scaling_corpus_sorts_last(self) -> None:
        scaling = Path("input/gnn_files/pymdp_scaling_study/z.md")
        normal = Path("input/gnn_files/discrete/a.md")
        assert _llm_file_sort_key(normal) < _llm_file_sort_key(scaling)

    def test_deterministic_within_group(self) -> None:
        a = Path("input/gnn_files/a.md")
        b = Path("input/gnn_files/b.md")
        assert _llm_file_sort_key(a) < _llm_file_sort_key(b)


class TestMergeProviderConfigs:
    def test_none_returns_env_defaults(self) -> None:
        merged = _merge_provider_configs(None)
        assert "ollama" in merged and "openai" in merged

    def test_user_overrides_are_layered(self) -> None:
        merged = _merge_provider_configs({"ollama": {"default_model": "custom"}})
        assert merged["ollama"]["default_model"] == "custom"
        # Non-overridden keys survive.
        assert "timeout" in merged["ollama"] or "default_max_tokens" in merged["ollama"]

    def test_unknown_provider_key_accepted(self) -> None:
        merged = _merge_provider_configs({"newcloud": {"api": "x"}})
        assert merged["newcloud"] == {"api": "x"}


class _StubProvider:
    def __init__(self, provider_value: str) -> None:
        self.provider_type = type("PT", (), {"value": provider_value})()


class TestGetDefaultProvider:
    def test_empty_registry_returns_none(self) -> None:
        assert LLMProcessor().get_default_provider() is None

    def test_returns_registered_provider(self) -> None:
        proc = LLMProcessor()
        provider_double = _StubProvider(ProviderType.OLLAMA.value)
        proc.providers[ProviderType.OLLAMA] = provider_double  # type: ignore[assignment]
        assert proc.get_default_provider() is provider_double


class TestExecutePrompt:
    """The shared prompt funnel: cache, auth fail-fast, timeout, fallbacks."""

    def _run(self, processor: object, **overrides: Any) -> str:
        cache = overrides.pop("cache", None) or LLMCache(
            cache_dir=Path("/tmp") / "nonexistent-llm-test"
        )
        kwargs: dict[str, Any] = dict(
            cache_content="gnn",
            model_name="model",
            messages=[],
            prompt_text="prompt",
            label="label",
            custom=False,
            max_tokens=10,
            max_prompt_timeout=0.05,
            failed_auth_providers=set(),
            auth_errors=[],
        )
        kwargs.update(overrides)
        return asyncio.run(_execute_prompt(processor, cache, **kwargs))  # type: ignore[arg-type]

    def test_cache_hit_skips_provider(self, tmp_path: Path) -> None:
        cache = LLMCache(cache_dir=tmp_path)
        calls = []

        class P:
            def get_default_provider(self) -> None:
                return None

            async def get_response(self, **kw: object) -> object:
                calls.append(kw)
                return type("Response", (), {"content": "cached-text"})()

        assert self._run(P(), cache=cache) == "cached-text"
        assert self._run(P(), cache=cache) == "cached-text"
        assert len(calls) == 1

    def test_auth_failed_provider_short_circuits(self, tmp_path: Path) -> None:
        class P:
            def get_default_provider(self) -> object:
                return _StubProvider("openai")

            def get_response(self, **kw: object) -> None:
                raise AssertionError("provider must not be called after auth failure")

        out = self._run(
            P(),
            cache=LLMCache(cache_dir=tmp_path),
            failed_auth_providers={"openai"},
        )
        assert out == _prompt_fallback_text("label", custom=False)

    def test_successful_response_is_cached(self, tmp_path: Path) -> None:
        cache = LLMCache(cache_dir=tmp_path)

        class Resp:
            content = "hello"

        class P:
            def get_default_provider(self) -> None:
                return None

            async def get_response(self, **kw: object) -> Resp:
                return Resp()

        assert self._run(P(), cache=cache) == "hello"
        assert cache.summary()["writes"] == 1
        assert self._run(P(), cache=cache) == "hello"
        assert cache.summary()["hits"] == 1

    def test_empty_response_replaced_with_notice(self, tmp_path: Path) -> None:
        cache = LLMCache(cache_dir=tmp_path)

        class Resp:
            content = "  "

        class P:
            def get_default_provider(self) -> None:
                return None

            async def get_response(self, **kw: object) -> Resp:
                return Resp()

        out = self._run(P(), cache=cache)
        assert out == (
            "No response generated for prompt label. This may indicate that "
            "the LLM provider is not available or not responding."
        )

    def test_auth_error_recorded_once_and_fail_fast_later(self, tmp_path: Path) -> None:
        cache = LLMCache(cache_dir=tmp_path)
        failed: set[str] = set()
        auth_errors: list[dict[str, str]] = []

        class P:
            def get_default_provider(self) -> None:
                return None

            async def get_response(self, **kw: object) -> object:
                raise RuntimeError("401 invalid_api_key for OpenAI")

        def call(label: str) -> str:
            return asyncio.run(
                _execute_prompt(
                    cast(Any, P()),
                    cache,
                    cache_content="gnn",
                    model_name="model",
                    messages=[],
                    prompt_text=f"prompt-{label}",
                    label=label,
                    custom=False,
                    max_tokens=10,
                    max_prompt_timeout=0.05,
                    failed_auth_providers=failed,
                    auth_errors=auth_errors,
                )
            )

        first = call("label")
        assert first == _prompt_fallback_text("label", custom=False)
        assert failed == {"openai"}
        assert auth_errors == [
            {"provider": "openai", "error": "401 invalid_api_key for OpenAI"}
        ]

        # Second prompt routed to the failed provider: no new auth_errors entry.
        second = call("label2")
        assert second == _prompt_fallback_text("label2", custom=False)
        assert len(auth_errors) == 1

    def test_timeout_returns_timeout_message(self, tmp_path: Path) -> None:
        class P:
            def get_default_provider(self) -> None:
                return None

            async def get_response(self, **kw: object) -> None:
                await asyncio.sleep(0.5)

        out = self._run(P(), cache=LLMCache(cache_dir=tmp_path))
        assert out == "Prompt execution timed out after 0.05 seconds"

    def test_non_auth_error_yields_fallback_text(self, tmp_path: Path) -> None:
        class P:
            def get_default_provider(self) -> None:
                return None

            async def get_response(self, **kw: object) -> object:
                raise ValueError("disk exploded")

        out = self._run(P(), cache=LLMCache(cache_dir=tmp_path))
        assert out == _prompt_fallback_text("label", custom=False)

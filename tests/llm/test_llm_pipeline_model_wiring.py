#!/usr/bin/env python3
"""Unit tests for LLM pipeline model wiring (Step 14): Ollama tag + SUMMARY provider order."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, cast

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.mark.unit
def test_summary_task_prefers_ollama_when_registered() -> None:
    from gnn.llm.llm_processor import LLMProcessor
    from gnn.llm.providers.base_provider import ProviderType

    class ProviderProbe:
        def __init__(self, ptype: Any) -> None:
            self.provider_type = ptype

    proc = LLMProcessor(preferred_providers=[ProviderType.OLLAMA, ProviderType.OPENAI])
    ollama_provider = ProviderProbe(ProviderType.OLLAMA)
    openai_provider = ProviderProbe(ProviderType.OPENAI)
    proc.providers = cast(
        Any,
        {
            ProviderType.OLLAMA: ollama_provider,
            ProviderType.OPENAI: openai_provider,
        },
    )

    from gnn.llm.llm_processor import AnalysisType

    best = cast(Any, proc.get_best_provider_for_task(AnalysisType.SUMMARY))
    assert best is ollama_provider


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_summarize_with_ollama_model_uses_ollama_provider_and_config() -> (
    None
):
    from gnn.llm.llm_operations import LLMOperations
    from gnn.llm.providers.base_provider import LLMConfig, LLMResponse, ProviderType

    response = LLMResponse(
        content="summary text",
        model_used="smollm2:135m-instruct-q4_K_S",
        provider="ollama",
    )

    class RecordingProcessor:
        def __init__(self) -> None:
            self.called = False
            self.call_args: dict[str, Any] = {}

        async def analyze_gnn(self, **kwargs: Any) -> Any:
            self.called = True
            self.call_args = kwargs
            return response

    ops = LLMOperations.__new__(LLMOperations)
    ops.processor = cast(Any, RecordingProcessor())
    ops._initialized = True

    out = await LLMOperations._async_summarize_gnn(
        ops, "sample gnn", max_length=100, ollama_model="smollm2:135m-instruct-q4_K_S"
    )
    assert out == "summary text"
    processor = cast(Any, ops.processor)
    assert processor.called, "recording processor should have been called"
    call_kw = processor.call_args
    assert call_kw["provider_type"] == ProviderType.OLLAMA
    assert isinstance(call_kw["config"], LLMConfig)
    assert call_kw["config"].model == "smollm2:135m-instruct-q4_K_S"


@pytest.mark.unit
def test_structured_prompt_get_response_passes_model_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """process_llm structured prompt loop must pass model_name matching cache key (ollama_model)."""
    from gnn.llm.providers.base_provider import ProviderType

    captured: dict = {}

    async def controlled_get_response(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        from gnn.llm.providers.base_provider import LLMResponse

        return LLMResponse(content="ok", model_used="m", provider="ollama")

    target = tmp_path / "gnn_in"
    target.mkdir()
    out = tmp_path / "llm_out"
    out.mkdir()
    (target / "one.md").write_text(
        "## GNNSection\nX\n## ModelName\nM\n## StateSpaceBlock\ns[1]\n",
        encoding="utf-8",
    )

    with monkeypatch.context() as m:

        async def controlled_init(*args: Any, **kwargs: Any) -> Any:
            return True

        async def controlled_close(*args: Any, **kwargs: Any) -> Any:
            return None

        class InMemoryProcessor:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.initialize = controlled_init
                self.close = controlled_close
                self.get_response = controlled_get_response
                self.get_available_providers = lambda: [ProviderType.OLLAMA]

        m.setattr(
            "gnn.llm.processor.LLMProcessor",
            lambda *args, **kwargs: InMemoryProcessor(),
        )

        async def controlled_analyze(*args: Any, **kwargs: Any) -> Any:
            return {"status": "SUCCESS"}

        m.setattr("gnn.llm.processor.analyze_gnn_file_with_llm", controlled_analyze)

        m.setattr(
            "gnn.llm.processor.generate_model_insights", lambda *args, **kwargs: {}
        )
        m.setattr(
            "gnn.llm.processor.generate_code_suggestions", lambda *args, **kwargs: {}
        )
        m.setattr(
            "gnn.llm.processor.generate_documentation", lambda *args, **kwargs: {}
        )
        m.setattr(
            "gnn.llm.processor._start_ollama_if_needed",
            lambda *args, **kwargs: (True, ["smollm2:135m-instruct-q4_K_S"]),
        )
        m.setattr(
            "gnn.llm.processor._select_best_ollama_model",
            lambda *args, **kwargs: "smollm2:135m-instruct-q4_K_S",
        )

        class InMemoryCache:
            def get(self, *args: Any, **kwargs: Any) -> Any:
                return None

            def put(self, *args: Any, **kwargs: Any) -> Any:
                pass

            def summary(self, *args: Any, **kwargs: Any) -> Any:
                return {
                    "hits": 0,
                    "misses": 0,
                    "writes": 0,
                    "hit_ratio_pct": 0.0,
                    "cache_dir": str(tmp_path),
                    "entries_on_disk": 0,
                }

        m.setattr("gnn.llm.processor.LLMCache", lambda *args, **kwargs: InMemoryCache())
        m.setattr(
            "gnn.llm.processor.generate_llm_summary",
            lambda *args, **kwargs: "# summary\n",
        )
        m.setattr(
            "gnn.llm.processor.get_prompt",
            lambda *args, **kwargs: {
                "system_message": "s",
                "user_prompt": "u",
                "max_tokens": 512,
            },
        )

        from gnn.llm.processor import process_llm

        process_llm(target, out, verbose=False)

    assert captured.get("model_name") == "smollm2:135m-instruct-q4_K_S"


@pytest.mark.unit
def test_process_llm_limits_files_from_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit cap limits selection; unavailable required LLM work is partial."""
    target = tmp_path / "gnn_in"
    scaling = target / "pymdp_scaling_study"
    scaling.mkdir(parents=True)
    out = tmp_path / "llm_out"
    out.mkdir()
    for name in ["b_model.md", "a_model.md"]:
        (target / name).write_text("## GNNSection\nX\n## ModelName\nM\n")
    for name in ["pymdp_scaling_N64_T10.md", "pymdp_scaling_N128_T10.md"]:
        (scaling / name).write_text("## GNNSection\nX\n## ModelName\nS\n")

    analyzed: list[str] = []
    llm_attempts: list[bool] = []

    async def controlled_analyze(
        path: Path, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        analyzed.append(path.name)
        llm_attempts.append(kwargs["attempt_llm"])
        return {"file": path.name}

    class UnavailableProcessor:
        async def initialize(self) -> bool:
            return False

        async def close(self) -> None:
            return None

    with monkeypatch.context() as m:
        m.setattr(
            "gnn.llm.processor._get_llm_config",
            lambda: {"timeout_seconds": 600, "max_files": 2},
        )
        m.setattr(
            "gnn.llm.processor._start_ollama_if_needed",
            lambda *args, **kwargs: (False, []),
        )
        m.setattr(
            "gnn.llm.processor.LLMProcessor",
            lambda *args, **kwargs: UnavailableProcessor(),
        )
        m.setattr("gnn.llm.processor.analyze_gnn_file_with_llm", controlled_analyze)
        m.setattr(
            "gnn.llm.processor.generate_model_insights", lambda *args, **kwargs: {}
        )
        m.setattr(
            "gnn.llm.processor.generate_code_suggestions", lambda *args, **kwargs: {}
        )
        m.setattr(
            "gnn.llm.processor.generate_documentation", lambda *args, **kwargs: {}
        )
        m.setattr(
            "gnn.llm.processor.generate_llm_summary",
            lambda *args, **kwargs: "# summary\n",
        )

        from gnn.llm.processor import process_llm

        assert not process_llm(target, out, verbose=False)

    assert analyzed == ["a_model.md", "b_model.md"]
    assert llm_attempts == [False, False]

    import json

    results = json.loads((out / "llm_results.json").read_text())
    assert results["processed_files"] == 2
    assert results["total_files_discovered"] == 4
    assert results["selected_files"] == 2
    assert results["skipped_files"] == 2
    assert results["status"] == "partial"
    assert results["coverage"]["structural_completed"] == 2
    assert results["coverage"]["summaries_completed"] == 0


@pytest.mark.unit
def test_llm_budget_prefers_cli_timeout() -> None:
    from gnn.llm.processor import _resolve_llm_budget_seconds

    assert (
        _resolve_llm_budget_seconds({"llm_timeout": 1200}, {"timeout_seconds": 600})
        == 1200
    )


@pytest.mark.unit
def test_ollama_disabled_is_side_effect_free(monkeypatch: pytest.MonkeyPatch) -> None:
    from gnn.llm.processor import _start_ollama_if_needed

    monkeypatch.setenv("OLLAMA_DISABLED", "1")
    monkeypatch.setattr(
        "gnn.llm.processor.shutil.which",
        lambda *_args: (_ for _ in ()).throw(AssertionError("PATH probe attempted")),
    )

    assert _start_ollama_if_needed(logging.getLogger("test")) == (False, [])


@pytest.mark.unit
def test_unavailable_ollama_does_not_start_daemon_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gnn.llm.processor import _start_ollama_if_needed

    class FailedList:
        returncode = 1
        stdout = ""
        stderr = "not running"

    monkeypatch.delenv("OLLAMA_DISABLED", raising=False)
    monkeypatch.delenv("OLLAMA_AUTO_START", raising=False)
    monkeypatch.setattr("gnn.llm.processor.shutil.which", lambda *_args: "/bin/ollama")
    monkeypatch.setattr(
        "gnn.llm.processor.subprocess.run", lambda *_args, **_kwargs: FailedList()
    )
    monkeypatch.setattr(
        "gnn.llm.processor.subprocess.Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("daemon startup attempted")
        ),
    )

    assert _start_ollama_if_needed(logging.getLogger("test")) == (False, [])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_structural_analysis_skips_provider_when_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.llm.analyzer import _analyze_gnn_file_with_llm

    source = tmp_path / "model.md"
    source.write_text("## StateSpaceBlock\ns[2]\n", encoding="utf-8")
    monkeypatch.setattr(
        "gnn.llm.analyzer.LLMOperations",
        lambda: (_ for _ in ()).throw(AssertionError("provider retry attempted")),
    )

    result = await _analyze_gnn_file_with_llm(source, attempt_llm=False)

    assert result["status"] == "SUCCESS"
    assert result["analysis_method"] == "structural"
    assert result["llm_summary_status"] == "unavailable"
    assert "llm_summary" not in result

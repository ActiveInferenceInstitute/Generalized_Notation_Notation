"""Current-corpus coverage, fairness, deadline, and checkpoint regressions."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from gnn.llm import processor as api
from gnn.llm.providers.base_provider import LLMResponse

pytestmark = pytest.mark.unit


@pytest.fixture
def corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, list[str]]:
    root = tmp_path / "models"
    events: list[str] = []
    for folder in ("nested", "other", ""):
        source = root / folder / "same.md"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(
            f"## GNNSection\nActInfPOMDP\n## ModelName\n{folder or 'root'}\n"
        )
    (root / "README.md").write_text("# Instructions\n")
    original = api.analyze_gnn_file_with_llm

    async def structural(path: Path, *args: Any, **kwargs: Any) -> Any:
        assert kwargs["attempt_llm"] is False
        events.append(f"structure:{path.parent.name}")
        return await original(path, *args, **kwargs)

    class Processor:
        async def initialize(self) -> bool:
            events.append("initialize")
            return True

        async def get_response(self, **kwargs: Any) -> LLMResponse:
            assert kwargs["provider_type"].value == "ollama"
            assert kwargs["model_name"] == "configured:exact"
            events.append(kwargs["messages"][1].content)
            return LLMResponse("response", "configured:exact", "ollama")

        async def close(self) -> None:
            events.append("close")

    monkeypatch.setattr(api, "LLMProcessor", lambda **_: Processor())
    monkeypatch.setattr(api, "analyze_gnn_file_with_llm", structural)
    monkeypatch.setattr(
        api,
        "get_prompt",
        lambda prompt, content: {
            "system_message": "system",
            "user_prompt": f"{prompt.value}:{content}",
        },
    )
    return root, tmp_path / "output", events


def test_all_models_structure_then_all_summaries_and_collision_safe_outputs(
    corpus: Any,
) -> None:
    root, output, events = corpus
    assert api.process_llm(
        root, output, llm_config={"model": "configured:exact"}, custom_prompts=[]
    )
    assert all(event.startswith("structure:") for event in events[:3])
    assert events[3] == "initialize"
    assert all(event.startswith("summarize_content:") for event in events[4:7])
    results = json.loads((output / "llm_results.json").read_text())
    assert results["selected_files"] == 3
    assert results["budget_seconds"] == 1800
    assert results["coverage"]["summaries_completed"] == 3
    assert len({row["model_id"] for row in results["analysis_results"]}) == 3
    assert len(list(output.glob("prompts_*"))) == 3
    assert results["unfinished_work"] == []


def test_explicit_empty_selection_does_not_rediscover(corpus: Any) -> None:
    root, output, events = corpus
    assert api.process_llm(
        root, output, selected_files=[], llm_config={"model": "configured:exact"}
    )
    assert events == ["close"]
    assert json.loads((output / "llm_results.json").read_text())["selected_files"] == 0


def test_frozen_duplicate_stems_use_original_identity(
    corpus: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.pipeline.run_context import CONTEXT_ENV, build_run_context

    root, output, _ = corpus
    context = build_run_context(root, output, "llm-test", [13], {})
    path = context.write(root.parent / "context.json")
    monkeypatch.setenv(CONTEXT_ENV, str(path))
    monkeypatch.setenv("GNN_RUN_ID", context.run_id)
    assert api.process_llm(
        root, output, llm_config={"model": "configured:exact"}, custom_prompts=[]
    )
    results = json.loads((output / "llm_results.json").read_text())
    assert {row["model_id"] for row in results["analysis_results"]} == {
        model.model_id for model in context.models
    }
    assert {row["source_path"] for row in results["analysis_results"]} == {
        model.relative_path for model in context.models
    }


def test_explicit_zero_prompt_ceiling_is_not_replaced_with_default(corpus: Any) -> None:
    root, output, _ = corpus
    assert not api.process_llm(
        root, output, max_prompt_timeout=0, llm_config={"model": "configured:exact"}
    )


def test_resume_binds_sources_and_prompt_definitions(
    corpus: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, output, events = corpus
    kwargs = {"llm_config": {"model": "configured:exact"}, "custom_prompts": []}
    assert api.process_llm(root, output, **kwargs)
    events.clear()
    assert api.process_llm(root, output, resume=True, **kwargs)
    assert "initialize" not in events
    assert len(events) == 4  # three structural passes plus cleanup, no provider call
    (root / "same.md").write_text("## GNNSection\nActInfPOMDP\n## ModelName\nchanged\n")
    assert not api.process_llm(root, output, resume=True, **kwargs)


def test_cache_binds_system_message_not_only_user_prompt(
    corpus: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, output, events = corpus
    kwargs = {"llm_config": {"model": "configured:exact"}, "custom_prompts": []}
    assert api.process_llm(root, output, **kwargs)
    events.clear()
    assert api.process_llm(root, output, **kwargs)
    assert len(events) == 5  # structure, initialize and close; all responses cached
    monkeypatch.setattr(
        api,
        "get_prompt",
        lambda prompt, content: {
            "system_message": "changed instructions",
            "user_prompt": f"{prompt.value}:{content}",
        },
    )
    events.clear()
    assert api.process_llm(root, output, **kwargs)
    assert len(events) == 23  # every request is re-executed under the new definition


@pytest.mark.asyncio
async def test_worker_rejects_substituted_model(
    corpus: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.llm import request_worker

    monkeypatch.setattr(
        request_worker,
        "run_subprocess_envelope",
        lambda *args, **kwargs: {
            "success": True,
            "stdout": json.dumps(
                {
                    "content": "response",
                    "model_used": "substitute",
                    "provider": "ollama",
                }
            ),
        },
    )
    with pytest.raises(RuntimeError, match="model differs"):
        await request_worker.isolated_request(
            provider="ollama", model="configured:exact", timeout=1
        )


@pytest.mark.asyncio
async def test_isolated_auth_failure_is_typed_and_suppresses_retries(
    corpus: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.llm import request_worker

    calls = []

    def failed(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "success": False,
            "stdout": json.dumps({"error_category": "authentication"}),
        }

    monkeypatch.setattr(request_worker, "run_subprocess_envelope", failed)
    root, output, _ = corpus
    cache = api.LLMCache(cache_dir=output / "cache")
    failures: set[str] = set()
    errors: list[dict[str, Any]] = []
    for label in ("first", "second"):
        await api._execute_prompt(
            api._LIVE_PROCESSOR_CLASS(),
            cache,
            cache_content="content",
            model_name="configured:exact",
            messages=[],
            prompt_text=label,
            label=label,
            custom=False,
            max_tokens=10,
            max_prompt_timeout=1,
            failed_auth_providers=failures,
            auth_errors=errors,
            provider_type=api.ProviderType.OLLAMA,
            isolated=True,
        )
    assert len(calls) == 1 and failures == {"ollama"}
    assert errors == [
        {"provider": "ollama", "error": "ollama: 401 authentication failed"}
    ]


def test_exhaustion_preserves_full_structural_coverage_and_partial_receipt(
    corpus: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, output, events = corpus

    class Slow:
        async def initialize(self) -> bool:
            return True

        async def get_response(self, **_: Any) -> Any:
            await asyncio.sleep(1)

        async def close(self) -> None:
            pass

    monkeypatch.setattr(api, "LLMProcessor", lambda **_: Slow())
    assert not api.process_llm(
        root,
        output,
        llm_timeout=0.05,
        max_prompt_timeout=0.02,
        llm_config={"model": "configured:exact"},
        custom_prompts=[],
    )
    results = json.loads((output / "llm_results.json").read_text())
    assert results["status"] == "timed_out"
    assert results["coverage"]["structural_completed"] == 3
    assert results["coverage"]["summaries_completed"] == 0
    assert len(results["unfinished_work"]) == 18
    assert (output / "llm_checkpoint.json").exists()


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf")])
def test_invalid_explicit_budgets_are_rejected(value: Any) -> None:
    with pytest.raises(ValueError):
        api._resolve_llm_budget_seconds({"llm_timeout": value}, {})


@pytest.mark.asyncio
async def test_isolated_request_kills_real_worker_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import time

    import psutil

    from gnn.llm import request_worker

    receipt = tmp_path / "pids"
    original = request_worker.run_subprocess_envelope
    code = """
import pathlib, subprocess, sys, time, os
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
pathlib.Path(sys.argv[1]).write_text(str(os.getpid()) + ',' + str(child.pid))
time.sleep(60)
"""

    def envelope(command: Any, **kwargs: Any) -> Any:
        return original([sys.executable, "-c", code, str(receipt)], **kwargs)

    monkeypatch.setattr(request_worker, "run_subprocess_envelope", envelope)
    started = time.monotonic()
    with pytest.raises(asyncio.TimeoutError):
        await request_worker.isolated_request(
            provider="ollama", model="test", timeout=0.5
        )
    assert time.monotonic() - started < 3
    assert receipt.exists()
    for pid in map(int, receipt.read_text().split(",")):
        try:
            process = psutil.Process(pid)
        except psutil.NoSuchProcess:
            continue
        assert process.status() == psutil.STATUS_ZOMBIE
        assert pid != os.getpid()


@pytest.mark.parametrize(
    "changed", [{"max_prompt_timeout": 12}, {"total_budget": 351}, {"max_files": 3}]
)
def test_resume_rejects_changed_effective_limits(
    corpus: Any, changed: dict[str, Any]
) -> None:
    root, output, events = corpus
    kwargs = {
        "llm_config": {"model": "configured:exact"},
        "custom_prompts": [],
        "total_budget": 350,
    }
    assert api.process_llm(root, output, **kwargs)
    events.clear()
    kwargs.update(changed)
    assert not api.process_llm(root, output, resume=True, **kwargs)
    assert not events


def test_checkpoint_binds_complete_context_policy(corpus: Any) -> None:
    root, output, _ = corpus
    assert api.process_llm(
        root, output, llm_config={"model": "configured:exact"}, custom_prompts=[]
    )
    binding = json.loads((output / "llm_checkpoint.json").read_text())["binding"]
    assert binding["schema_version"] == 2
    assert binding["effective_options"]["input_policy"] == "complete_context_v1"
    assert binding["effective_options"]["max_tokens_ceiling"] == 512


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "category, expected",
    [
        ("context_limit", "cannot admit the complete input"),
        ("runtime_unsupported", "cannot verify the whole-context policy"),
        ("model_unavailable", "model is not installed"),
    ],
)
async def test_worker_preserves_safe_provider_failure_category(
    monkeypatch: pytest.MonkeyPatch, category: str, expected: str
) -> None:
    from gnn.llm import request_worker

    def failed_envelope(*args: object, **kwargs: object) -> dict[str, Any]:
        return {"success": False, "stdout": json.dumps({"error_category": category})}

    monkeypatch.setattr(request_worker, "run_subprocess_envelope", failed_envelope)
    with pytest.raises(RuntimeError, match=expected):
        await request_worker.isolated_request(
            provider="ollama", model="configured:exact", timeout=1
        )


@pytest.mark.asyncio
async def test_worker_transport_timeout_retains_timeout_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gnn.llm import request_worker

    monkeypatch.setattr(
        request_worker,
        "run_subprocess_envelope",
        lambda *args, **kwargs: {
            "success": False,
            "stdout": json.dumps({"error_category": "timeout"}),
        },
    )
    with pytest.raises(asyncio.TimeoutError):
        await request_worker.isolated_request(
            provider="ollama", model="configured:exact", timeout=1
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("preflight", [False, True])
async def test_worker_rejects_late_host_decode(
    monkeypatch: pytest.MonkeyPatch, preflight: bool
) -> None:
    from dataclasses import asdict

    from gnn.llm import request_worker

    body = (
        {"ready": True}
        if preflight
        else asdict(LLMResponse("content", "configured:exact", "ollama"))
    )
    monkeypatch.setattr(
        request_worker,
        "run_subprocess_envelope",
        lambda *args, **kwargs: {"success": True, "stdout": json.dumps(body)},
    )
    from types import SimpleNamespace

    clock = {"now": 0.0}
    original_loads = json.loads

    def late_decode(value: str) -> Any:
        decoded = original_loads(value)
        clock["now"] = 1.01
        return decoded

    monkeypatch.setattr(
        request_worker, "time", SimpleNamespace(monotonic=lambda: clock["now"])
    )
    monkeypatch.setattr(
        request_worker, "json", SimpleNamespace(loads=late_decode, dumps=json.dumps)
    )
    with pytest.raises(asyncio.TimeoutError):
        await request_worker.isolated_request(
            provider="ollama", model="configured:exact", timeout=1, preflight=preflight
        )


def test_changed_endpoint_rejects_resume_and_separates_cache(
    corpus: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, output, events = corpus
    kwargs = {"llm_config": {"model": "configured:exact"}, "custom_prompts": []}
    monkeypatch.setenv("OLLAMA_HOST", "http://host-a.invalid:11434")
    assert api.process_llm(root, output, **kwargs)
    first = json.loads((output / "llm_checkpoint.json").read_text())["fingerprint"]
    events.clear()
    monkeypatch.setenv("OLLAMA_HOST", "http://host-b.invalid:11434")
    assert not api.process_llm(root, output, resume=True, **kwargs)
    assert not events
    assert api.process_llm(root, output, **kwargs)
    second = json.loads((output / "llm_checkpoint.json").read_text())["fingerprint"]
    assert first != second
    assert (
        len(
            [
                event
                for event in events
                if not event.startswith("structure:")
                and event not in {"initialize", "close"}
            ]
        )
        == 18
    )
    assert (
        len(list((output / ".cache" / "ollama" / "complete_context_v1").iterdir())) == 2
    )


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://user:private@localhost",
        "http://localhost?token=private",
        "file:///tmp/model",
    ],
)
def test_endpoint_identity_excludes_credentials(
    endpoint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.llm.provider_identity import resolved_endpoint
    from gnn.llm.providers.base_provider import ProviderType

    monkeypatch.setenv("OLLAMA_HOST", endpoint)
    with pytest.raises(ValueError, match="without URL credentials"):
        resolved_endpoint(ProviderType.OLLAMA)


@pytest.mark.asyncio
async def test_late_cache_hit_is_not_recorded_as_completed_prompt() -> None:
    remaining = {"seconds": 1.0}

    class Cache:
        def get(self, *args: Any) -> str:
            remaining["seconds"] = 0.0
            return "a previously stored complete response"

    class Processor:
        async def get_response(self, **kwargs: Any) -> None:
            pytest.fail("a cache hit must not call the provider")

    outcomes: dict[str, Any] = {}
    result = await api._execute_prompt(
        Processor(),
        Cache(),
        cache_content="source",
        model_name="configured:exact",
        messages=[],
        prompt_text="prompt",
        label="summary",
        custom=False,
        max_tokens=32,
        max_prompt_timeout=1,
        failed_auth_providers=set(),
        auth_errors=[],
        outcomes=outcomes,
        remaining_budget=lambda: remaining["seconds"],
    )
    assert outcomes["summary"]["status"] == "timed_out"
    assert "timed out" in result


@pytest.mark.parametrize(
    "late_file", ["llm_checkpoint.json", "llm_results.json", "llm_summary.md"]
)
def test_final_publication_crossing_deadline_preserves_non_success(
    corpus: Any,
    monkeypatch: pytest.MonkeyPatch,
    late_file: str,
) -> None:
    from types import SimpleNamespace

    from gnn.llm import corpus_runner

    root, output, _ = corpus
    clock = {"now": 100.0, "complete_checkpoint_writes": 0, "triggered": False}
    original_write = corpus_runner.atomic_write_text

    def late(path: Path, body: str) -> None:
        original_write(path, body)
        if (
            path.name == "llm_checkpoint.json"
            and len(json.loads(body)["completed"]) == 18
        ):
            clock["complete_checkpoint_writes"] += 1
        final_checkpoint = (
            path.name == "llm_checkpoint.json"
            and clock["complete_checkpoint_writes"] == 2
        )
        final_results = (
            path.name == "llm_results.json" and json.loads(body).get("success") is True
        )
        final_summary = path.name == "llm_summary.md"
        if (
            not clock["triggered"]
            and path.name == late_file
            and (final_checkpoint or final_results or final_summary)
        ):
            clock["now"] = 103.0
            clock["triggered"] = True

    monkeypatch.setattr(
        corpus_runner, "time", SimpleNamespace(monotonic=lambda: clock["now"])
    )
    monkeypatch.setattr(corpus_runner, "atomic_write_text", late)
    assert not api.process_llm(
        root,
        output,
        llm_config={"model": "configured:exact"},
        total_budget=2,
        custom_prompts=[],
    )
    results = json.loads((output / "llm_results.json").read_text())
    assert results["success"] is False and results["status"] == "timed_out"
    assert results["finalization_status"] == "timed_out"
    assert results["coverage"]["prompts_completed"] == 18
    assert clock["triggered"]


def test_last_cache_publication_cannot_make_exhausted_corpus_success(
    corpus: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from gnn.llm import corpus_runner

    root, output, _ = corpus
    clock = {"now": 100.0, "puts": 0}
    original_cache = api.LLMCache

    class Cache(original_cache):
        def put(self, *args: Any, **kwargs: Any) -> Any:
            result = super().put(*args, **kwargs)
            clock["puts"] += 1
            if clock["puts"] == 18:
                clock["now"] = 103.0
            return result

    monkeypatch.setattr(api, "LLMCache", Cache)
    monkeypatch.setattr(
        corpus_runner, "time", SimpleNamespace(monotonic=lambda: clock["now"])
    )
    assert not api.process_llm(
        root,
        output,
        llm_config={"model": "configured:exact"},
        total_budget=2,
        custom_prompts=[],
    )
    results = json.loads((output / "llm_results.json").read_text())
    assert results["status"] == "timed_out"
    assert results["coverage"]["prompts_completed"] == 17

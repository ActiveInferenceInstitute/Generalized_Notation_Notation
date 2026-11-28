"""S1 request-plan scheduler: offline multi-window, resume and refusal controls.

Every provider reply here comes from an injected executor or a fixture
``isolated_request``. These tests exercise scheduling, checkpoint persistence
and admission only; they are not native or live provider evidence.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from gnn.llm import plan_scheduler as ps
from gnn.llm import processor as api
from gnn.llm import request_worker
from gnn.llm.context_admission import checked_model_context, context_receipt
from gnn.llm.defaults import OLLAMA_RUNTIME_VALIDATION_POLICY
from gnn.llm.providers.base_provider import LLMResponse
from gnn.llm.request_plan import (
    COMPLETION_POLICY,
    RequestPlanRefused,
    admit_request_plan,
)

pytestmark = pytest.mark.unit

ENDPOINT = hashlib.sha256(b"http://127.0.0.1:11434").hexdigest()
MODEL_DIGEST = "a" * 64
SOURCE = (
    "﻿## GNNSection\r\nActInfPOMDP\r\n## StateSpaceBlock\r\n"
    "s[2,1,type=int] # état μ ≥ 0 🧠\r\n## Notes\nmixed endings\rno final newline"
).encode()
PROMPT = ps.prompt_definition("summary", "Analyze GNN.", "Summarize the model.")
RESERVATIONS = {"system": 64, "task": 64, "prior_receipts": 256, "output": 128}


def capacity(num_ctx: int = 4096) -> dict[str, Any]:
    return checked_model_context(
        {
            "details": {"format": "gguf"},
            "model_info": {
                "general.architecture": "llama",
                "llama.context_length": 8192,
            },
            "template": "fixture-template",
        },
        {"server_version": "0.40.11", "generation_options": {"num_ctx": num_ctx}},
    )


def admit(source: bytes = SOURCE, **overrides: Any):
    arguments: dict[str, Any] = {
        "provider": "ollama",
        "endpoint_sha256": ENDPOINT,
        "model": "fixture:exact",
        "model_digest": MODEL_DIGEST,
        "prompt_definition": PROMPT,
        "capacity": capacity(),
        "reservations": RESERVATIONS,
        "mode": "part_evidence",
        "segment_max_bytes": 24,
        "deadline_seconds": 600,
        "request_timeout_seconds": 120,
    }
    arguments.update(overrides)
    return admit_request_plan(source, **arguments)


def terminal(**overrides: Any) -> dict[str, Any]:
    value = {
        "status": "success",
        "finish_reason": "stop",
        "completion_policy": COMPLETION_POLICY,
        "provider": "ollama",
        "model": "fixture:exact",
        "model_digest": MODEL_DIGEST,
        "endpoint_sha256": ENDPOINT,
        "prompt_tokens": 200,
        "completion_tokens": 20,
    }
    value.update(overrides)
    return value


class Executor:
    """Deterministic injected executor with per-request outcome overrides."""

    def __init__(self, overrides: dict[str, dict[str, Any]] | None = None) -> None:
        self.calls: list[tuple[list[dict[str, str]], int, str]] = []
        self.overrides = overrides or {}

    async def __call__(
        self, messages: list[dict[str, str]], max_tokens: int, key: str
    ) -> tuple[str, dict[str, Any]]:
        self.calls.append((copy.deepcopy(messages), max_tokens, key))
        name = key.split(":", 1)[1]
        override = self.overrides.get(name, self.overrides.get("*", {}))
        return f"answer for {name}", terminal(**override)

    @property
    def names(self) -> list[str]:
        return [key.split(":", 1)[1] for _, _, key in self.calls]


def run(plan: Any, directory: Path, execute: Any, **kwargs: Any) -> dict[str, Any]:
    source = kwargs.pop("source", SOURCE)
    prompt = kwargs.pop("prompt", PROMPT)
    return asyncio.run(
        ps.run_request_plan(
            plan, source, prompt, directory=directory, execute=execute, **kwargs
        )
    )


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


# --- multi-window scheduling and byte preservation --------------------------


def test_multi_window_plan_completes_with_exact_byte_windows(tmp_path: Path) -> None:
    plan = admit()
    assert len(plan.segments) > 3
    executor = Executor()
    receipt = run(plan, tmp_path / "plan", executor)
    ids = [segment.segment_id for segment in plan.segments]
    assert executor.names == [*ids, "synthesis"]
    assert all(tokens == RESERVATIONS["output"] for _, tokens, _ in executor.calls)
    assert receipt["status"] == "complete" and receipt["complete"] is True
    assert receipt["accepted"] == ids and receipt["pending"] == []
    assert receipt["new_requests"] == [*ids, "synthesis"]
    assert receipt["resumed_segments"] == [] and receipt["refusals"] == []
    assert receipt["full_source_claim"] is False
    assert receipt["attention_verified"] is False

    files = ps.PlanFiles(tmp_path / "plan")
    assert json.loads(files.plan.read_text())["plan_sha256"] == plan.sha256
    rebuilt = b""
    for segment in plan.segments:
        ledger = json.loads(files.segment_request(segment.segment_id).read_text())
        user = ledger["messages"][1]["content"].encode("utf-8")
        bounds = ledger["content_range"]
        window = user[bounds["start_byte"] : bounds["end_byte"]]
        assert window == SOURCE[segment.start_byte : segment.end_byte]
        assert ledger["source_range"]["start_byte"] == len(rebuilt)
        rebuilt += window
        checkpoint = json.loads(files.segment(segment.segment_id).read_text())
        assert checkpoint["request_sha256"] == segment.request_sha256
        assert checkpoint["outcome"]["finish_reason"] == "stop"
    assert rebuilt == SOURCE  # BOM, CRLF, lone CR and no final newline survive
    synthesis = json.loads(files.synthesis.read_text())
    assert [d["segment_id"] for d in synthesis["dependencies"]] == ids
    assert synthesis["full_source_claim"] is False
    synthesis_user = executor.calls[-1][0][1]["content"]
    positions = [synthesis_user.index(f"[{sid} ") for sid in ids]
    assert positions == sorted(positions)
    assert ps.SOURCE_BEGIN not in synthesis_user
    on_disk = json.loads(files.receipt.read_text())
    assert on_disk == receipt


def test_checkpoints_and_receipts_are_byte_deterministic(tmp_path: Path) -> None:
    plan = admit()
    run(plan, tmp_path / "first", Executor())
    run(plan, tmp_path / "second", Executor())
    first, second = tree_bytes(tmp_path / "first"), tree_bytes(tmp_path / "second")
    assert first == second
    assert all(value.endswith(b"\n") for value in first.values())


def test_joint_source_synthesis_carries_exact_source_range(tmp_path: Path) -> None:
    plan = admit(
        mode="joint_source",
        source_token_count={
            "origin": "native_tokenizer",
            "tokens": 60,
            "model_digest": MODEL_DIGEST,
            "template_sha256": capacity()["template_sha256"],
        },
    )
    executor = Executor()
    receipt = run(plan, tmp_path / "plan", executor)
    assert receipt["complete"] and receipt["full_source_claim"] is True
    ledger = json.loads(ps.PlanFiles(tmp_path / "plan").synthesis_request.read_text())
    user = ledger["messages"][1]["content"].encode("utf-8")
    bounds = ledger["content_range"]
    assert user[bounds["start_byte"] : bounds["end_byte"]] == SOURCE


# --- checkpoint resume -------------------------------------------------------


def test_resume_requests_only_unfinished_windows(tmp_path: Path) -> None:
    plan = admit()
    ids = [segment.segment_id for segment in plan.segments]
    directory = tmp_path / "plan"

    def cancel_after_two() -> bool:
        return ps.PlanFiles(directory).segment(ids[1]).is_file()

    first = Executor()
    stopped = run(plan, directory, first, cancelled=cancel_after_two)
    assert stopped["status"] == "cancelled" and not stopped["complete"]
    assert stopped["accepted"] == ids[:2] and stopped["pending"] == ids[2:]
    assert not ps.PlanFiles(directory).synthesis.exists()

    second = Executor()
    resumed = run(plan, directory, second, resume=True)
    assert second.names == [*ids[2:], "synthesis"]
    assert resumed["resumed_segments"] == ids[:2]
    assert resumed["complete"] is True

    third = Executor()
    again = run(plan, directory, third, resume=True)
    assert third.calls == []
    assert again["status"] == "complete"
    assert again["resumed_segments"] == ids and again["new_requests"] == []


def test_non_resume_run_forgets_prior_checkpoints(tmp_path: Path) -> None:
    plan = admit()
    directory = tmp_path / "plan"
    run(plan, directory, Executor())
    fresh = Executor()
    receipt = run(plan, directory, fresh)
    assert receipt["resumed_segments"] == []
    assert len(fresh.calls) == len(plan.segments) + 1


def test_resume_rejects_corrupt_or_tampered_checkpoints(tmp_path: Path) -> None:
    plan = admit()
    ids = [segment.segment_id for segment in plan.segments]
    directory = tmp_path / "plan"
    run(plan, directory, Executor())
    files = ps.PlanFiles(directory)
    files.segment(ids[0]).write_text("{not json")
    tampered = json.loads(files.segment(ids[1]).read_text())
    tampered["response"] = "relabeled answer"
    files.segment(ids[1]).write_text(json.dumps(tampered))
    executor = Executor()
    receipt = run(plan, directory, executor, resume=True)
    assert executor.names == [ids[0], ids[1], "synthesis"]
    assert len(receipt["rejected_checkpoints"]) == 2
    assert receipt["complete"] is True


def test_resume_refuses_a_changed_plan_without_touching_files(tmp_path: Path) -> None:
    directory = tmp_path / "plan"
    run(admit(), directory, Executor())
    before = tree_bytes(directory)
    changed = admit(segment_max_bytes=32)
    executor = Executor()
    receipt = run(changed, directory, executor, resume=True)
    assert receipt["status"] == "refused" and executor.calls == []
    assert "segments" in receipt["refusals"][0]["reason"]
    assert receipt["pending"] == [s.segment_id for s in changed.segments]
    assert tree_bytes(directory) == before


def test_resume_without_retained_plan_starts_fresh(tmp_path: Path) -> None:
    plan = admit()
    executor = Executor()
    receipt = run(plan, tmp_path / "plan", executor, resume=True)
    assert receipt["complete"] and len(executor.calls) == len(plan.segments) + 1


# --- terminal-stop completion admission --------------------------------------


@pytest.mark.parametrize(
    "override, reason",
    [
        ({"finish_reason": "length"}, "incomplete"),
        ({"status": "incomplete"}, "incomplete"),
        ({"finish_reason": None}, "incomplete"),
        ({"completion_tokens": RESERVATIONS["output"] + 1}, "exceed"),
        ({"prompt_tokens": None}, "native prompt tokens"),
        ({"model_digest": "b" * 64}, "model_digest differs"),
        ({"completion_policy": "terminal_stop_and_answer_margin_v2"}, "terminal"),
    ],
)
def test_truncated_or_unbound_window_is_refused_and_never_checkpointed(
    tmp_path: Path, override: dict[str, Any], reason: str
) -> None:
    plan = admit()
    ids = [segment.segment_id for segment in plan.segments]
    executor = Executor({ids[1]: override})
    directory = tmp_path / "plan"
    receipt = run(plan, directory, executor)
    assert receipt["status"] == "refused" and receipt["complete"] is False
    assert [r["request"] for r in receipt["refusals"]] == [ids[1]]
    assert reason in receipt["refusals"][0]["reason"]
    assert receipt["pending"] == [ids[1]]
    assert "synthesis" not in executor.names  # no synthesis over a gap
    assert executor.names.count(ids[1]) == 1  # no retry or weaker fallback
    files = ps.PlanFiles(directory)
    assert not files.segment(ids[1]).exists()
    assert files.segment_request(ids[1]).exists()
    assert not files.synthesis.exists()

    healed = Executor()
    resumed = run(plan, directory, healed, resume=True)
    assert healed.names == [ids[1], "synthesis"]
    assert resumed["complete"] is True


def test_truncated_synthesis_is_refused_then_resumed_alone(tmp_path: Path) -> None:
    plan = admit()
    directory = tmp_path / "plan"
    executor = Executor({"synthesis": {"finish_reason": "length"}})
    receipt = run(plan, directory, executor)
    assert receipt["status"] == "refused"
    assert receipt["refusals"] == [
        {
            "request": "synthesis",
            "reason": "reply is incomplete; only terminal_stop_v1 replies may be "
            "checkpointed",
        }
    ]
    assert receipt["synthesis"] == "ready" and receipt["pending"] == []
    assert not ps.PlanFiles(directory).synthesis.exists()
    healed = Executor()
    assert run(plan, directory, healed, resume=True)["complete"] is True
    assert healed.names == ["synthesis"]


def test_deadline_stops_scheduling_before_a_request_cannot_fit(tmp_path: Path) -> None:
    plan = admit(deadline_seconds=300, request_timeout_seconds=120)
    ticks = iter(range(0, 10_000, 100))
    executor = Executor()
    receipt = run(plan, tmp_path / "plan", executor, clock=lambda: next(ticks))
    assert receipt["status"] == "deadline_exceeded"
    assert 0 < len(executor.calls) < len(plan.segments)
    assert receipt["pending"]


@pytest.mark.parametrize("stop_reason", ["cancelled", "deadline_exceeded"])
@pytest.mark.parametrize("phase", ["window", "synthesis"])
def test_late_terminal_reply_never_becomes_a_checkpoint(
    tmp_path: Path, phase: str, stop_reason: str
) -> None:
    plan = admit(deadline_seconds=200, request_timeout_seconds=120)
    directory = tmp_path / "plan"
    now = 0.0
    stopped = False
    label = plan.segments[0].segment_id if phase == "window" else "synthesis"

    async def execute(messages: Any, max_tokens: int, key: str) -> Any:
        nonlocal now, stopped
        if key.endswith(f":{label}"):
            if stop_reason == "deadline_exceeded":
                now = 200.0
            else:
                stopped = True
        return "terminal-looking reply", terminal()

    receipt = run(
        plan, directory, execute, clock=lambda: now, cancelled=lambda: stopped
    )
    files = ps.PlanFiles(directory)
    reply = files.segment(label) if phase == "window" else files.synthesis
    assert receipt["status"] == stop_reason and not receipt["complete"]
    assert receipt["new_requests"][-1] == label
    assert not reply.exists()


def test_request_timeout_is_enforced_before_checkpoint(tmp_path: Path) -> None:
    plan = admit(request_timeout_seconds=0.01, segment_max_bytes=1000)

    async def execute(messages: Any, max_tokens: int, key: str) -> Any:
        await asyncio.sleep(60)
        return "never in time", terminal()

    directory = tmp_path / "plan"
    receipt = run(plan, directory, execute)
    assert receipt["status"] == "refused" and not receipt["complete"]
    assert receipt["refusals"] == [
        {"request": plan.segments[0].segment_id, "reason": "request timeout exceeded"}
    ]
    assert not ps.PlanFiles(directory).segment(plan.segments[0].segment_id).exists()


# --- request construction and settings --------------------------------------


def test_prompt_and_source_identity_are_enforced(tmp_path: Path) -> None:
    plan = admit()
    other = ps.prompt_definition("summary", "Analyze GNN.", "Another task.")
    with pytest.raises(RequestPlanRefused, match="prompt definition differs"):
        run(plan, tmp_path / "a", Executor(), prompt=other)
    with pytest.raises(RequestPlanRefused, match="unsupported template"):
        run(plan, tmp_path / "b", Executor(), prompt={**PROMPT, "schema": "x"})
    with pytest.raises(RequestPlanRefused, match="source bytes differ"):
        run(plan, tmp_path / "c", Executor(), source=SOURCE + b"\n")
    segment = plan.segments[0].segment_id
    with pytest.raises(RequestPlanRefused, match="bytes differ"):
        ps.window_request(plan, b"X" + SOURCE[1:], segment, PROMPT)
    with pytest.raises(RequestPlanRefused, match="unfinished"):
        ps.synthesis_request(plan, SOURCE, {}, PROMPT)


def test_settings_are_off_by_default_and_strictly_checked() -> None:
    assert ps.checked_settings(None) is None
    assert ps.checked_settings(False) is None
    settings = ps.checked_settings(
        {
            "segment_max_bytes": 64,
            "reservations": RESERVATIONS,
            "deadline_seconds": 60,
        }
    )
    assert settings == {
        "mode": "part_evidence",
        "segment_max_bytes": 64,
        "reservations": dict(sorted(RESERVATIONS.items())),
        "prompt": None,
        "deadline_seconds": 60,
    }
    for bad, match in [
        ("yes", "mapping"),
        ({"segment_max_bytes": 64, "reservations": {}, "extra": 1}, "mapping"),
        ({"mode": "fragments", "segment_max_bytes": 64}, "mode"),
        ({"segment_max_bytes": 3, "reservations": {}}, "segment_max_bytes"),
        ({"segment_max_bytes": True, "reservations": {}}, "segment_max_bytes"),
        ({"segment_max_bytes": 64, "reservations": []}, "reservations"),
    ]:
        with pytest.raises(ValueError, match=match):
            ps.checked_settings(bad)


def test_corpus_admission_uses_only_the_verified_runtime_identity() -> None:
    settings = ps.checked_settings(
        {"segment_max_bytes": 24, "reservations": RESERVATIONS}
    )
    assert settings is not None
    arguments: dict[str, Any] = {
        "provider": "ollama",
        "endpoint_sha256": ENDPOINT,
        "model": "fixture:exact",
        "deadline_seconds": 600,
        "request_timeout_seconds": 120,
    }
    runtime = {"model_digest": MODEL_DIGEST, "context_capability": capacity()}
    plan = ps.admit_corpus_plan(
        SOURCE, settings, PROMPT, runtime_identity=runtime, **arguments
    )
    assert plan.sha256 == admit().sha256
    with pytest.raises(RequestPlanRefused, match="verified runtime identity"):
        ps.admit_corpus_plan(SOURCE, settings, PROMPT, runtime_identity={}, **arguments)
    joint = {**settings, "mode": "joint_source"}
    with pytest.raises(RequestPlanRefused, match="native tokenizer count"):
        ps.admit_corpus_plan(
            SOURCE, joint, PROMPT, runtime_identity=runtime, **arguments
        )


def test_plan_outcome_reads_counts_only_from_native_admission() -> None:
    native = {
        "status": "success",
        "finish_reason": "stop",
        "provider": "ollama",
        "model": "fixture:exact",
        "request_sha256": "c" * 64,
        "context_admission": {
            "model_digest": MODEL_DIGEST,
            "prompt_tokens": 10,
            "completion_tokens": 5,
        },
    }
    mapped = ps.plan_outcome(native, provider="ollama", endpoint_sha256=ENDPOINT)
    assert mapped == {
        "status": "success",
        "finish_reason": "stop",
        "completion_policy": COMPLETION_POLICY,
        "provider": "ollama",
        "model": "fixture:exact",
        "model_digest": MODEL_DIGEST,
        "endpoint_sha256": ENDPOINT,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "request_sha256": "c" * 64,
    }
    bare = ps.plan_outcome(
        {"status": "success", "finish_reason": None},
        provider="ollama",
        endpoint_sha256=ENDPOINT,
    )
    assert bare["prompt_tokens"] is None and bare["model_digest"] is None
    assert bare["provider"] == "ollama"


# --- process_llm corpus wiring (fixture isolated_request) --------------------


@pytest.fixture
def plan_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root = tmp_path / "models"
    root.mkdir()
    raw = (
        "## GNNSection\r\nActInfPOMDP\r\n## ModelName\r\nFIRST=alpha é\r\n"
        "## Notes\r\nLAST=omega 🧠\r\n"
    ).encode()
    (root / "model.md").write_bytes(raw)
    runtime: dict[str, Any] = {
        "model": "fixture:exact",
        "model_digest": MODEL_DIGEST,
        "server_version": "0.40.11",
        "generation_options": {"num_ctx": 8192},
        "status": "ready",
        "context_policy": "whole-context-v1",
    }
    runtime["context_capability"] = checked_model_context(
        {
            "details": {"format": "gguf"},
            "model_info": {
                "general.architecture": "llama",
                "llama.context_length": 8192,
            },
            "template": "fixture-template",
        },
        runtime,
    )
    calls: list[dict[str, Any]] = []
    control: dict[str, Any] = {"finish": "stop", "plan_finish": "stop"}

    async def request(**kwargs: Any) -> LLMResponse:
        if kwargs.get("preflight"):
            return LLMResponse(
                "", runtime["model"], "ollama", metadata=copy.deepcopy(runtime)
            )
        calls.append(copy.deepcopy(kwargs))
        in_plan = ps.WINDOW_BEGIN in kwargs["messages"][1]["content"] or (
            "Ordered window answers" in kwargs["messages"][1]["content"]
        )
        admission = context_receipt(
            messages=kwargs["messages"],
            identity=runtime,
            loaded={
                "name": runtime["model"],
                "digest": runtime["model_digest"],
                "context_length": 8192,
            },
            usage={"prompt_tokens": 200, "completion_tokens": 12},
            max_tokens=kwargs["max_tokens"],
        )
        return LLMResponse(
            "plan answer" if in_plan else "prompt answer",
            runtime["model"],
            "ollama",
            finish_reason=control["plan_finish" if in_plan else "finish"],
            metadata={
                "runtime_identity": copy.deepcopy(runtime),
                "runtime_identity_after": copy.deepcopy(runtime),
                "runtime_validation_policy": OLLAMA_RUNTIME_VALIDATION_POLICY,
                "context_admission": admission,
            },
        )

    monkeypatch.setattr(request_worker, "isolated_request", request)
    monkeypatch.setattr(
        api,
        "get_prompt",
        lambda task, content: {
            "system_message": "fixture task",
            "user_prompt": task.value + ":" + content,
        },
    )
    options = {
        "model": runtime["model"],
        "custom_prompts": [],
        "llm_config": {"ollama_options": {"num_ctx": 8192}},
        "max_tokens": 256,
        "request_plan": {
            "segment_max_bytes": 20,
            "reservations": {
                "system": 64,
                "task": 64,
                "prior_receipts": 512,
                "output": 256,
            },
        },
    }
    return {
        "root": root,
        "output": tmp_path / "output",
        "raw": raw,
        "calls": calls,
        "control": control,
        "options": options,
    }


def plan_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        call
        for call in calls
        if ps.WINDOW_BEGIN in call["messages"][1]["content"]
        or "Ordered window answers" in call["messages"][1]["content"]
    ]


def test_process_llm_runs_the_request_plan_and_resumes_without_requests(
    plan_corpus: dict[str, Any],
) -> None:
    root, output, raw = plan_corpus["root"], plan_corpus["output"], plan_corpus["raw"]
    calls, options = plan_corpus["calls"], plan_corpus["options"]
    assert api.process_llm(root, output, **options)
    results = json.loads((output / "llm_results.json").read_text())
    [receipt] = results["request_plans"]
    windows = -(-len(raw) // 20)
    assert receipt["status"] == "complete" and receipt["complete"] is True
    assert receipt["prompt"] == "summarize_content"
    assert receipt["segments_required"] >= windows
    assert results["coverage"]["request_plans_completed"] == 1
    assert results["coverage"]["request_plans_required"] == 1
    assert len(plan_calls(calls)) == receipt["segments_required"] + 1
    directory = output / receipt["directory"]
    rebuilt = b""
    for segment_id in receipt["accepted"]:
        ledger = json.loads(
            (directory / "segments" / f"{segment_id}.request.json").read_text()
        )
        user = ledger["messages"][1]["content"].encode("utf-8")
        bounds = ledger["content_range"]
        rebuilt += user[bounds["start_byte"] : bounds["end_byte"]]
    assert rebuilt == raw
    checkpoint = json.loads((output / "llm_checkpoint.json").read_text())
    assert checkpoint["binding"]["request_plan"]["prompt"] == "summarize_content"

    calls.clear()
    assert api.process_llm(root, output, resume=True, **options)
    assert plan_calls(calls) == []
    [again] = json.loads((output / "llm_results.json").read_text())["request_plans"]
    assert again["resumed_segments"] == receipt["accepted"]
    assert again["new_requests"] == [] and again["complete"] is True


def test_corpus_requests_receive_the_tighter_plan_timeout(
    plan_corpus: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = api._execute_prompt
    timeouts = []

    async def record_timeout(*args: Any, **kwargs: Any) -> str:
        if kwargs.get("synthesis_plan_sha256"):
            timeouts.append(kwargs["max_prompt_timeout"])
        return await original(*args, **kwargs)

    monkeypatch.setattr(api, "_execute_prompt", record_timeout)
    options = copy.deepcopy(plan_corpus["options"])
    options["request_plan"]["request_timeout_seconds"] = 7.0
    assert api.process_llm(plan_corpus["root"], plan_corpus["output"], **options)
    assert timeouts and all(0.0 < value <= 7.0 for value in timeouts)


def test_process_llm_refuses_truncated_plan_windows(
    plan_corpus: dict[str, Any],
) -> None:
    plan_corpus["control"]["plan_finish"] = "length"
    output = plan_corpus["output"]
    assert not api.process_llm(plan_corpus["root"], output, **plan_corpus["options"])
    results = json.loads((output / "llm_results.json").read_text())
    [receipt] = results["request_plans"]
    assert receipt["status"] == "refused" and receipt["accepted"] == []
    assert len(receipt["refusals"]) == receipt["segments_required"]
    assert results["coverage"]["request_plans_completed"] == 0
    assert (
        results["coverage"]["prompts_completed"]
        == results["coverage"]["prompts_required"]
    )
    directory = output / receipt["directory"]
    assert list((directory / "segments").glob("segment_????.json")) == []
    assert not (directory / "synthesis.json").exists()


def test_process_llm_joint_source_refuses_without_native_count(
    plan_corpus: dict[str, Any],
) -> None:
    options = copy.deepcopy(plan_corpus["options"])
    options["request_plan"]["mode"] = "joint_source"
    output = plan_corpus["output"]
    assert not api.process_llm(plan_corpus["root"], output, **options)
    [receipt] = json.loads((output / "llm_results.json").read_text())["request_plans"]
    assert receipt["status"] == "refused"
    assert "native tokenizer count" in receipt["refusals"][0]["reason"]
    assert plan_calls(plan_corpus["calls"]) == []


def test_process_llm_without_request_plan_keeps_the_established_binding(
    plan_corpus: dict[str, Any], tmp_path: Path
) -> None:
    options = dict(plan_corpus["options"])
    options.pop("request_plan")
    assert api.process_llm(plan_corpus["root"], tmp_path / "a", **options)
    assert api.process_llm(
        plan_corpus["root"], tmp_path / "b", request_plan=None, **options
    )
    first = json.loads((tmp_path / "a" / "llm_checkpoint.json").read_text())
    second = json.loads((tmp_path / "b" / "llm_checkpoint.json").read_text())
    assert "request_plan" not in first["binding"]
    assert first["fingerprint"] == second["fingerprint"]
    results = json.loads((tmp_path / "a" / "llm_results.json").read_text())
    assert "request_plans" not in results
    assert "request_plans_required" not in results["coverage"]
    assert plan_calls(plan_corpus["calls"]) == []


@pytest.mark.parametrize(
    "change",
    [
        {"provider": "openai"},
        {"llm_config": {}},
        {"request_plan": {"segment_max_bytes": 20, "reservations": {}, "prompt": "x"}},
    ],
)
def test_process_llm_rejects_unsupported_request_plan_configuration(
    plan_corpus: dict[str, Any], change: dict[str, Any]
) -> None:
    options = {**plan_corpus["options"], **change}
    output = plan_corpus["output"]
    assert not api.process_llm(plan_corpus["root"], output, **options)
    assert not (output / "llm_results.json").exists()


def test_process_llm_plan_waits_when_provider_preflight_fails(
    plan_corpus: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    async def unavailable(**kwargs: Any) -> LLMResponse:
        raise RuntimeError("fixture provider unavailable")

    monkeypatch.setattr(request_worker, "isolated_request", unavailable)
    output = plan_corpus["output"]
    assert not api.process_llm(plan_corpus["root"], output, **plan_corpus["options"])
    [receipt] = json.loads((output / "llm_results.json").read_text())["request_plans"]
    assert receipt["status"] == "refused"
    assert "verified runtime identity" in receipt["refusals"][0]["reason"]


def test_process_llm_plan_rejects_source_changed_after_selection(
    plan_corpus: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = Path.read_bytes
    model = plan_corpus["root"] / "model.md"

    def drifting(self: Path) -> bytes:
        data = original(self)
        caller = sys._getframe(1).f_globals.get("__name__")
        if self == model and caller == ps.__name__:
            return data + b"\n"  # the edit lands after the selection froze
        return data

    monkeypatch.setattr(Path, "read_bytes", drifting)
    output = plan_corpus["output"]
    assert not api.process_llm(plan_corpus["root"], output, **plan_corpus["options"])
    [receipt] = json.loads((output / "llm_results.json").read_text())["request_plans"]
    assert receipt["status"] == "refused"
    assert receipt["refusals"][0]["reason"] == "source changed since selection"

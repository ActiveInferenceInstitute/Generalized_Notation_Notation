"""Fixture-only whole-corpus synthesis/resume controls, with no model calls."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from gnn.llm import processor as api
from gnn.llm import request_worker
from gnn.llm.context_admission import checked_model_context, context_receipt
from gnn.llm.defaults import OLLAMA_RUNTIME_VALIDATION_POLICY
from gnn.llm.providers.base_provider import LLMResponse

pytestmark = pytest.mark.unit


@pytest.fixture
def synthesis_corpus(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    raw = (
        b"## GNNSection\r\nActInfPOMDP\r\n## ModelName\r\nFIRST=alpha\r\nLAST=omega\r\n"
    )
    (root / "model.md").write_bytes(raw)
    runtime = {
        "model": "fixture:exact",
        "model_digest": "a" * 64,
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
    calls = []
    control = {"finish": "stop", "corrupt_context": False}

    async def request(**kwargs):
        if kwargs.get("preflight"):
            assert kwargs["context_admission"] is True
            return LLMResponse(
                "", runtime["model"], "ollama", metadata=copy.deepcopy(runtime)
            )
        calls.append(copy.deepcopy(kwargs))
        fields = kwargs.get("response_format")
        text = (
            json.dumps(
                {name: "distinct answer for " + name for name in fields["properties"]}
            )
            if fields
            else "fragment answer"
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
        if control["corrupt_context"]:
            admission["messages_sha256"] = "b" * 64
        return LLMResponse(
            text,
            runtime["model"],
            "ollama",
            finish_reason=control["finish"],
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
        "source_chunk_bytes": 40,
        "synthesis_mode": "joint_source",
        "max_tokens": 256,
        "synthesis_max_tokens": 256,
    }
    return root, tmp_path / "output", raw, calls, control, options


def test_corpus_joint_source_actual_request_ledger_then_zero_generation_resume(
    synthesis_corpus,
):
    root, output, raw, calls, _, options = synthesis_corpus
    assert api.process_llm(root, output, **options)
    assert len(calls) == 18
    reducers = [
        c for c in calls if "EXACT_ORIGINAL_SOURCE_BEGIN" in c["messages"][1]["content"]
    ]
    assert len(reducers) == 6
    for path in output.glob("prompts_*/synthesis/*.request.json"):
        record = json.loads(path.read_text())
        bounds = record["exact_source_request_range"]
        assert (
            record["messages"][1]["content"].encode()[
                bounds["start_byte"] : bounds["end_byte"]
            ]
            == raw
        )
    results = json.loads((output / "llm_results.json").read_text())
    assert results["coverage"]["syntheses_completed"] == 6
    assert results["coverage"]["source_parts_completed"] == 12
    calls.clear()
    assert api.process_llm(root, output, resume=True, **options)
    assert calls == []
    assert (
        json.loads((output / "llm_results.json").read_text())["coverage"][
            "new_provider_requests_completed"
        ]
        == 0
    )


@pytest.mark.parametrize(
    "control_key, value", [("finish", "length"), ("corrupt_context", True)]
)
def test_incomplete_or_unbound_native_fixture_stays_unfinished(
    synthesis_corpus, control_key, value
):
    root, output, _, _, control, options = synthesis_corpus
    control[control_key] = value
    assert not api.process_llm(root, output, **options)
    results = json.loads((output / "llm_results.json").read_text())
    assert results["coverage"]["prompts_completed"] == 0
    assert results["unfinished_work"]
    assert (
        json.loads((output / "llm_checkpoint.json").read_text())["completed_syntheses"]
        == {}
    )


def test_resume_missing_context_proof_refuses_even_with_recomputed_outcome_digest(
    synthesis_corpus,
):
    root, output, _, calls, _, options = synthesis_corpus
    assert api.process_llm(root, output, **options)
    path = output / "llm_checkpoint.json"
    record = json.loads(path.read_text())
    first = next(iter(record["completed_parts"].values()))
    first["outcome"].pop("context_admission")
    first["outcome_sha256"] = hashlib.sha256(
        json.dumps(first["outcome"], sort_keys=True).encode()
    ).hexdigest()
    path.write_text(json.dumps(record))
    calls.clear()
    assert not api.process_llm(root, output, resume=True, **options)
    assert calls == []


@pytest.mark.parametrize(
    "status,finish", [("incomplete", "stop"), ("success", "length"), ("success", None)]
)
def test_resumed_map_terminal_state_cannot_be_promoted(
    synthesis_corpus, status, finish
):
    root, output, _, calls, _, options = synthesis_corpus
    assert api.process_llm(root, output, **options)
    path = output / "llm_checkpoint.json"
    record = json.loads(path.read_text())
    first = next(iter(record["completed_parts"].values()))
    first["outcome"].update(status=status, finish_reason=finish)
    first["outcome_sha256"] = hashlib.sha256(
        json.dumps(first["outcome"], sort_keys=True).encode()
    ).hexdigest()
    path.write_text(json.dumps(record))
    calls.clear()
    assert not api.process_llm(root, output, resume=True, **options)
    assert calls == []


@pytest.mark.parametrize(
    "damage", [None, "swap", "raw", "key", "unflag", "missing_flag"]
)
def test_batched_map_answers_bind_their_native_answer_key(synthesis_corpus, damage):
    root, output, _, calls, _, options = synthesis_corpus
    options["batch_prompts"] = True
    assert api.process_llm(root, output, **options)
    path = output / "llm_checkpoint.json"
    record = json.loads(path.read_text())
    entries = record["completed_parts"]
    keys = [
        key
        for key, value in entries.items()
        if value["outcome"].get("batched") and key.endswith("part_0001")
    ]
    assert len(keys) == 5
    first, second = keys[:2]
    assert entries[first]["response"] != entries[second]["response"]
    assert entries[first]["native_response"] == entries[second]["native_response"]
    record["completed_syntheses"] = {}  # interrupted after maps, before reducers
    record["completed"] = {}
    if damage == "swap":
        entries[first], entries[second] = entries[second], entries[first]
    if damage == "raw":
        entries[first]["native_response"] = json.dumps(
            dict.fromkeys(entries[first]["outcome"]["batch_fields"], "changed")
        )
        entries[first]["native_response_sha256"] = hashlib.sha256(
            entries[first]["native_response"].encode()
        ).hexdigest()
    if damage == "key":
        entries[first]["answer_key"] = "wrong"
    if damage in {"unflag", "missing_flag"}:
        if damage == "unflag":
            entries[first]["outcome"]["batched"] = False
        else:
            entries[first]["outcome"].pop("batched")
        entries[first]["outcome_sha256"] = hashlib.sha256(
            json.dumps(entries[first]["outcome"], sort_keys=True).encode()
        ).hexdigest()
    path.write_text(json.dumps(record))
    calls.clear()
    assert api.process_llm(root, output, resume=True, **options) is (damage is None)
    assert all(
        "EXACT_ORIGINAL_SOURCE_BEGIN" in call["messages"][1]["content"]
        for call in calls
    )

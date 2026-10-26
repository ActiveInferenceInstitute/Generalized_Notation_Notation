"""Offline protocol controls; none establish native model acceptance."""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import pytest

from gnn.llm.cache import LLMCache
from gnn.llm.context_admission import (
    checked_model_context,
    context_receipt,
    validate_context_receipt,
)
from gnn.llm.corpus_synthesis import run_prompt_synthesis
from gnn.llm.cross_part_synthesis import (
    build_plan,
    completed_entry,
    digest,
    resumed_entry,
)
from gnn.llm.source_parts import partition_source

pytestmark = pytest.mark.unit
MESSAGES = [{"role": "user", "content": "first α\r\nlast β"}]


def identity(version="0.40.11"):
    value = {
        "model": "fixture:exact",
        "model_digest": "a" * 64,
        "server_version": version,
        "generation_options": {"num_ctx": 8192},
    }
    value["context_capability"] = checked_model_context(
        {
            "details": {"format": "gguf"},
            "model_info": {
                "general.architecture": "llama",
                "llama.context_length": 8192,
            },
            "template": "fixture-template",
        },
        value,
    )
    return value


def receipt(**overrides):
    args = {
        "messages": MESSAGES,
        "identity": identity(),
        "loaded": {"name": "fixture:exact", "digest": "a" * 64, "context_length": 8192},
        "usage": {"prompt_tokens": 100, "completion_tokens": 12},
        "max_tokens": 256,
        "transport": "fixture_protocol",
    }
    args.update(overrides)
    return context_receipt(**args)


@pytest.mark.parametrize("version", ["0.35.0", "0.40.11"])
def test_reviewed_context_is_bounded_contract_not_attention(version):
    r = receipt(identity=identity(version))
    validate_context_receipt(
        r, messages=MESSAGES, identity=identity(version), max_tokens=256, native=False
    )
    assert r["admission_basis"] == "reviewed_provider_contract"
    assert r["attention_verified"] is False
    assert r["token_retention_attestation"] == "provider_reported_not_independent"


@pytest.mark.parametrize("value", ["0.40.12", "0.34.0", "latest", None])
def test_unknown_provider_contract_refuses(value):
    with pytest.raises(ValueError, match="unsupported"):
        identity(value)


@pytest.mark.parametrize(
    "changes",
    [
        {"loaded": {"name": "other", "digest": "a" * 64, "context_length": 8192}},
        {
            "loaded": {
                "name": "fixture:exact",
                "digest": "b" * 64,
                "context_length": 8192,
            }
        },
        {
            "loaded": {
                "name": "fixture:exact",
                "digest": "a" * 64,
                "context_length": 4096,
            }
        },
        {"usage": {"prompt_tokens": 8000, "completion_tokens": 12}},
        {"usage": {"prompt_tokens": True, "completion_tokens": 12}},
        {"usage": {"prompt_tokens": 100, "completion_tokens": 257}},
        {"usage": {"prompt_tokens": 100, "completion_tokens": 0}},
        {"max_tokens": True},
    ],
)
def test_context_identity_usage_and_reserved_output_refuse(changes):
    with pytest.raises(ValueError):
        receipt(**changes)


@pytest.mark.parametrize(
    "field,value",
    [
        ("attention_verified", True),
        ("messages_sha256", "b" * 64),
        ("transport_evidence", "ollama_http_chat"),
        ("input_controls", {"truncate": True, "shift": False}),
    ],
)
def test_context_proof_cannot_be_relabeled(field, value):
    r = receipt()
    r[field] = value
    with pytest.raises(ValueError):
        validate_context_receipt(
            r, messages=MESSAGES, identity=identity(), max_tokens=256, native=False
        )


def plan(mode="joint_source"):
    source = "FIRST=α\r\n" + "x" * 12 + "\r\nLAST=β"
    parts = partition_source(source, 20)
    children = [
        {
            "response": f"answer{i}",
            "sha256": digest(f"answer{i}"),
            "outcome": {
                "status": "success",
                "finish_reason": "stop",
                "request_sha256": str(i + 1) * 64,
            },
        }
        for i in range(len(parts))
    ]
    return (
        build_plan(
            content=source,
            parts=parts,
            children=children,
            label="summary",
            task={"system_message": "fixture", "user_prompt": "Relate FIRST and LAST"},
            mode=mode,
        ),
        source,
        parts,
        children,
    )


def test_joint_source_includes_every_original_byte_and_all_children():
    p, source, parts, _ = plan()
    bounds = p.ledger["exact_source_request_range"]
    assert (
        p.user_prompt.encode()[bounds["start_byte"] : bounds["end_byte"]]
        == source.encode()
    )
    assert len(p.ledger["source_parts"]) == len(parts)
    assert p.ledger["source_parts"][-1]["end_byte"] == len(source.encode())
    derived, _, _, _ = plan("part_evidence")
    assert derived.ledger["exact_source_request_range"] is None
    assert source not in derived.user_prompt
    assert "Do not claim" in derived.system_message


@pytest.mark.parametrize(
    "damage", ["unfinished", "response", "request", "order", "source"]
)
def test_synthesis_dependency_corruption_refuses(damage):
    p, source, parts, children = plan()
    if damage == "unfinished":
        children[0]["outcome"]["status"] = "pending"
    if damage == "response":
        children[0]["response"] += "changed"
    if damage == "request":
        children[0]["outcome"]["request_sha256"] = 42
    if damage == "order":
        parts.reverse()
    if damage == "source":
        source += "changed"
    with pytest.raises(ValueError):
        build_plan(
            content=source,
            parts=parts,
            children=children,
            label="summary",
            task={"system_message": "fixture", "user_prompt": "task"},
            mode=p.mode,
        )


def test_reducer_requires_terminal_schema_and_exact_resume_dependencies():
    p, *_ = plan()
    outcome = {
        "status": "success",
        "finish_reason": "stop",
        "context_admission": receipt(),
    }
    entry = completed_entry(
        p, '{"summary":"FIRST and LAST are related"}', outcome, None
    )
    assert resumed_entry(p, entry, None) == entry
    for key, value in [
        ("native_response", None),
        ("response", "changed"),
        ("outcome", {}),
        ("plan_sha256", "b" * 64),
    ]:
        broken = copy.deepcopy(entry)
        broken[key] = value
        with pytest.raises(ValueError):
            resumed_entry(p, broken, None)
    with pytest.raises(ValueError):
        completed_entry(
            p, '{"summary":"answer"}', {**outcome, "finish_reason": "length"}, None
        )
    with pytest.raises(ValueError):
        completed_entry(p, '{"other":"answer"}', outcome, None)


def test_completed_reducer_resumes_without_execution_and_failed_node_is_not_admitted():
    p, *_ = plan()
    calls, saved = [], []
    state = {}

    async def execute(*_):
        calls.append(True)
        return '{"summary":"both facts"}', {
            "status": "success",
            "finish_reason": "stop",
        }

    args = dict(
        plan=p,
        key="node",
        state=state,
        execute=execute,
        validate=lambda *_: None,
        max_chars=None,
        save=lambda: saved.append(True),
    )
    assert asyncio.run(run_prompt_synthesis(**args))[0] == "both facts"
    assert asyncio.run(run_prompt_synthesis(**args))[2]["resumed"] is True
    assert len(calls) == len(saved) == 1

    async def incomplete(*_):
        return "partial", {"status": "incomplete", "finish_reason": "length"}

    args.update(key="unfinished", execute=incomplete)
    assert asyncio.run(run_prompt_synthesis(**args))[0] is None
    assert "unfinished" not in state["completed_syntheses"]


def test_verified_cache_preserves_proof_and_rejects_legacy_or_modified_metadata(
    tmp_path: Path,
):
    cache = LLMCache(cache_dir=tmp_path)
    cache.put("source", "model", "request", "legacy")
    assert cache.get_verified("source", "model", "request") is None
    cache.put_verified(
        "source", "model", "request", "answer", {"context_admission": receipt()}
    )
    assert (
        cache.get_verified("source", "model", "request")["metadata"][
            "context_admission"
        ]
        == receipt()
    )
    path = next(tmp_path.glob("*.json"))
    record = json.loads(path.read_text())
    record["metadata"]["context_admission"]["attention_verified"] = True
    path.write_text(json.dumps(record))
    assert cache.get_verified("source", "model", "request") is None


@pytest.mark.parametrize(
    "status,finish", [("incomplete", "stop"), ("success", "length"), ("success", None)]
)
def test_source_children_require_original_success_and_terminal_stop(status, finish):
    p, source, parts, children = plan()
    children[0]["outcome"].update(status=status, finish_reason=finish)
    with pytest.raises(ValueError, match="unfinished"):
        build_plan(
            content=source,
            parts=parts,
            children=children,
            label="summary",
            task={"system_message": "fixture", "user_prompt": "task"},
            mode=p.mode,
        )


def test_v2_prompt_and_full_plan_bind_distinct_child_requests(monkeypatch, tmp_path):
    """Visible text can agree while the DAG-bound cache must remain distinct."""
    from gnn.llm import processor as api
    from gnn.llm import request_worker
    from gnn.llm.defaults import OLLAMA_RUNTIME_VALIDATION_POLICY
    from gnn.llm.providers.base_provider import LLMMessage, LLMResponse, ProviderType

    a, source, parts, children = plan()
    children[0]["outcome"]["request_sha256"] = "f" * 64
    b = build_plan(
        content=source,
        parts=parts,
        children=children,
        label="summary",
        task={"system_message": "fixture", "user_prompt": "Relate FIRST and LAST"},
        mode="joint_source",
    )
    assert a.messages == b.messages and a.sha256 != b.sha256
    assert a.user_prompt.startswith("TASK INSTRUCTIONS\n")
    assert digest(source) not in a.user_prompt
    assert a.ledger["source_sha256"] == digest(source)
    runtime, calls = identity(), []

    async def request(**kwargs):
        calls.append(kwargs)
        context = context_receipt(
            messages=kwargs["messages"],
            identity=runtime,
            loaded={
                "name": runtime["model"],
                "digest": runtime["model_digest"],
                "context_length": 8192,
            },
            usage={"prompt_tokens": 200, "completion_tokens": 12},
            max_tokens=256,
        )
        return LLMResponse(
            "fixture answer " + str(len(calls)),
            runtime["model"],
            "ollama",
            finish_reason="stop",
            metadata={
                "runtime_identity": runtime,
                "runtime_identity_after": runtime,
                "runtime_validation_policy": OLLAMA_RUNTIME_VALIDATION_POLICY,
                "context_admission": context,
            },
        )

    monkeypatch.setattr(request_worker, "isolated_request", request)
    cache, outcomes = LLMCache(cache_dir=tmp_path), {}

    async def execute(p, key):
        return await api._execute_prompt(
            api._LIVE_PROCESSOR_CLASS(),
            cache,
            cache_content=source,
            model_name=runtime["model"],
            messages=[LLMMessage(**m) for m in p.messages],
            prompt_text=p.user_prompt,
            label=key,
            custom=False,
            max_tokens=256,
            max_prompt_timeout=2,
            failed_auth_providers=set(),
            auth_errors=[],
            outcomes=outcomes,
            provider_type=ProviderType.OLLAMA,
            isolated=True,
            runtime_identity=runtime,
            context_admission=True,
            synthesis_plan_sha256=p.sha256,
        )

    first = asyncio.run(execute(a, "first"))
    second = asyncio.run(execute(b, "second"))
    third = asyncio.run(execute(b, "third"))
    assert len(calls) == 2 and first != second == third
    assert outcomes["first"]["request_sha256"] != outcomes["second"]["request_sha256"]
    assert outcomes["third"]["cached"] is True


@pytest.mark.parametrize("words", [True, 0, "40"])
def test_synthesis_word_instruction_requires_an_explicit_positive_integer(words):
    _, source, parts, children = plan()
    with pytest.raises(ValueError, match="answer_words"):
        build_plan(
            content=source,
            parts=parts,
            children=children,
            label="summary",
            task={"system_message": "fixture", "user_prompt": "task"},
            mode="joint_source",
            answer_words=words,
        )


def test_word_instruction_binds_the_plan_and_stays_out_of_raw_source():
    _, source, parts, children = plan()
    p = build_plan(
        content=source,
        parts=parts,
        children=children,
        label="summary",
        task={"system_message": "fixture", "user_prompt": "task"},
        mode="joint_source",
        answer_words=40,
    )
    assert "at most 40 words" in p.system_message and p.ledger["answer_words"] == 40
    bounds = p.ledger["exact_source_request_range"]
    assert (
        p.user_prompt.encode()[bounds["start_byte"] : bounds["end_byte"]]
        == source.encode()
    )

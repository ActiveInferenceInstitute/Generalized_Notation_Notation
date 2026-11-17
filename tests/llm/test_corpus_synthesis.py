"""Fixture-only whole-corpus synthesis/resume controls, with no model calls."""

from __future__ import annotations

import asyncio
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
    raw = ("FIRST=α\r\n" + "x" * 12 + "\r\nLAST=β").encode("utf-8")
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
    control = {"finish": "stop", "corrupt_context": False, "runtime": runtime}

    async def request(**kwargs):
        identity = copy.deepcopy(runtime)
        if not kwargs.get("context_admission"):
            identity.pop("context_capability", None)
        if kwargs.get("preflight"):
            return LLMResponse("", runtime["model"], "ollama", metadata=identity)
        calls.append(copy.deepcopy(kwargs))
        fields = kwargs.get("response_format")
        text = (
            json.dumps(
                {name: "distinct answer for " + name for name in fields["properties"]}
            )
            if fields
            else "fragment answer"
        )
        admission = (
            context_receipt(
                messages=kwargs["messages"],
                identity=identity,
                loaded={
                    "name": runtime["model"],
                    "digest": runtime["model_digest"],
                    "context_length": 8192,
                },
                usage={"prompt_tokens": 200, "completion_tokens": 12},
                max_tokens=kwargs["max_tokens"],
            )
            if kwargs.get("context_admission")
            else None
        )
        if admission is not None and control["corrupt_context"]:
            admission["messages_sha256"] = "b" * 64
        return LLMResponse(
            text,
            runtime["model"],
            "ollama",
            finish_reason=control["finish"],
            metadata={
                "runtime_identity": copy.deepcopy(identity),
                "runtime_identity_after": copy.deepcopy(identity),
                "runtime_validation_policy": OLLAMA_RUNTIME_VALIDATION_POLICY,
                **({"context_admission": admission} if admission is not None else {}),
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
        "source_chunk_bytes": 20,
        "synthesis_mode": "joint_source",
        "max_tokens": 256,
        "synthesis_max_tokens": 256,
    }
    return root, tmp_path / "output", raw, calls, control, options


@pytest.mark.parametrize("mode", ["joint_source", "part_evidence"])
def test_corpus_exact_bytes_request_ledger_then_zero_generation_resume(
    synthesis_corpus, mode
):
    root, output, raw, calls, _, options = synthesis_corpus
    options["synthesis_mode"] = mode
    assert api.process_llm(root, output, **options)
    assert len(calls) == 18
    assert sum(bool(call.get("response_format")) for call in calls) == 6
    results = json.loads((output / "llm_results.json").read_text())
    row = results["analysis_results"][0]
    context = row["source_context"]
    assert context["joint_context"] is False
    assert context["source_bytes"] == 31
    assert context["parts"] == [
        {
            "start_byte": 0,
            "end_byte": 20,
            "sha256": hashlib.sha256(raw[:20]).hexdigest(),
        },
        {
            "start_byte": 20,
            "end_byte": 31,
            "sha256": hashlib.sha256(raw[20:]).hexdigest(),
        },
    ]
    source_dir = output / "source_parts" / row["artifact_stem"]
    assert (source_dir / "part_0001.txt").read_bytes() == raw[:20]
    assert (source_dir / "part_0002.txt").read_bytes() == raw[20:]
    assert (
        b"".join(path.read_bytes() for path in sorted(source_dir.glob("part_*.txt")))
        == raw
    )
    for path in output.glob("prompts_*/synthesis/*.request.json"):
        record = json.loads(path.read_text())
        bounds = record["exact_source_request_range"]
        prompt = record["messages"][1]["content"].encode("utf-8")
        if mode == "joint_source":
            assert prompt[bounds["start_byte"] : bounds["end_byte"]] == raw
        else:
            assert bounds is None
            assert b"EXACT_ORIGINAL_SOURCE_BEGIN" not in prompt
        assert record["source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert results["coverage"]["syntheses_completed"] == 6
    assert results["coverage"]["source_parts_completed"] == 12
    assert results["coverage"]["request_groups_completed"] == 18
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
    "control_key, value",
    [
        ("finish", "length"),
        ("finish", None),
        ("finish", "content_filter"),
        ("finish", "tool_calls"),
        ("corrupt_context", True),
    ],
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


@pytest.mark.parametrize("mode", ["none", "joint_source", "part_evidence"])
@pytest.mark.parametrize(
    "status,finish", [("incomplete", "stop"), ("success", "length"), ("success", None)]
)
def test_resumed_map_terminal_state_cannot_be_promoted(
    synthesis_corpus, status, finish, mode
):
    root, output, _, calls, _, options = synthesis_corpus
    options["synthesis_mode"] = mode
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
    # Exercise map checks with reducer caches retained.
    record["completed_syntheses"] = {}
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


@pytest.mark.parametrize("mode", ["joint_source", "part_evidence"])
def test_duplicate_stem_sources_keep_distinct_complete_synthesis_artifacts(
    synthesis_corpus, mode
):
    root, output, raw, calls, _, options = synthesis_corpus
    other = root / "nested" / "model.md"
    other.parent.mkdir()
    other.write_bytes(raw.replace(b"xxxxxxxxxxxx", b"yyyyyyyyyyyy"))
    options["synthesis_mode"] = mode
    assert api.process_llm(root, output, **options)
    result = json.loads((output / "llm_results.json").read_text())
    rows = result["analysis_results"]
    assert len(calls) == 36
    assert result["coverage"]["source_parts_completed"] == 24
    assert result["coverage"]["syntheses_completed"] == 12
    assert rows[0]["model_id"] != rows[1]["model_id"]
    assert rows[0]["artifact_stem"] != rows[1]["artifact_stem"]
    for row in rows:
        original = (root / row["source_path"]).read_bytes()
        assert row["source_sha256"] == hashlib.sha256(original).hexdigest()
        directory = output / "source_parts" / row["artifact_stem"]
        assert (
            b"".join(path.read_bytes() for path in sorted(directory.glob("part_*.txt")))
            == original
        )
        assert (
            len(
                list(
                    (output / f"prompts_{row['artifact_stem']}" / "synthesis").glob(
                        "*.native.json"
                    )
                )
            )
            == 6
        )


@pytest.mark.parametrize("mode", ["joint_source", "part_evidence"])
@pytest.mark.parametrize("damage", [None, "request_hash"])
def test_interrupted_reducer_reuses_only_exact_finished_dependencies(
    synthesis_corpus, monkeypatch, mode, damage
):
    root, output, _, calls, _, options = synthesis_corpus
    options["synthesis_mode"] = mode
    actual_request = request_worker.isolated_request
    control = {"interrupt": True}

    async def interrupt(**payload):
        if (
            control["interrupt"]
            and not payload.get("preflight")
            and payload.get("response_format")
        ):
            control["interrupt"] = False
            raise asyncio.CancelledError
        return await actual_request(**payload)

    monkeypatch.setattr(request_worker, "isolated_request", interrupt)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(api._process_llm_async(root, output, False, **options))
    checkpoint = output / "llm_checkpoint.json"
    record = json.loads(checkpoint.read_text())
    assert len(record["completed_parts"]) == 2
    assert record["completed_syntheses"] == {}
    assert len(calls) == 2
    if damage:
        entry = next(iter(record["completed_parts"].values()))
        entry["outcome"]["request_sha256"] = "0" * 64
        entry["outcome_sha256"] = hashlib.sha256(
            json.dumps(entry["outcome"], sort_keys=True).encode()
        ).hexdigest()
        checkpoint.write_text(json.dumps(record))
    calls.clear()
    assert api.process_llm(root, output, resume=True, **options) is (damage is None)
    if damage:
        assert calls == []
    else:
        assert len(calls) == 16
        calls.clear()
        assert api.process_llm(root, output, resume=True, **options)
        assert calls == []


def test_resumed_reducer_rejects_valid_but_wrong_request_hash(synthesis_corpus):
    root, output, _, calls, _, options = synthesis_corpus
    assert api.process_llm(root, output, **options)
    path = output / "llm_checkpoint.json"
    record = json.loads(path.read_text())
    entry = next(iter(record["completed_syntheses"].values()))
    entry["outcome"]["request_sha256"] = "0" * 64
    entry["outcome_sha256"] = hashlib.sha256(
        json.dumps(entry["outcome"], sort_keys=True).encode()
    ).hexdigest()
    path.write_text(json.dumps(record))
    calls.clear()
    assert not api.process_llm(root, output, resume=True, **options)
    assert calls == []


@pytest.mark.parametrize(
    "change",
    [
        "source",
        "model",
        "endpoint",
        "template",
        "model_tokenizer_digest",
        "prompt",
        "num_ctx",
        "seed",
        "max_tokens",
        "synthesis_max_tokens",
        "source_chunk_bytes",
        "llm_timeout",
        "max_prompt_timeout",
        "response_words",
        "answer_max_chars",
        "synthesis_mode",
        "checkpoint_json",
        "binding_part_range",
        "binding_part_order",
    ],
)
def test_resume_refuses_changed_identity_configuration_or_checkpoint(
    synthesis_corpus, monkeypatch, change
):
    root, output, raw, calls, control, options = synthesis_corpus
    assert api.process_llm(root, output, **options)
    if change == "source":
        (root / "model.md").write_bytes(raw.replace(b"x", b"y"))
    elif change == "model":
        options["model"] = "fixture:different"
    elif change == "endpoint":
        monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    elif change == "template":
        control["runtime"]["context_capability"]["template_sha256"] = "b" * 64
    elif change == "model_tokenizer_digest":
        control["runtime"]["model_digest"] = "b" * 64
    elif change == "prompt":
        actual_prompt = api.get_prompt

        def changed_prompt(task, content):
            current = actual_prompt(task, content)
            return {**current, "user_prompt": current["user_prompt"] + "\nchanged"}

        monkeypatch.setattr(api, "get_prompt", changed_prompt)
    elif change in ("num_ctx", "seed"):
        options["llm_config"]["ollama_options"][change] = (
            4096 if change == "num_ctx" else 7
        )
    elif change in ("max_tokens", "synthesis_max_tokens"):
        options[change] = 257
    elif change == "source_chunk_bytes":
        options[change] = 21
    elif change in ("llm_timeout", "max_prompt_timeout"):
        options[change] = 30
    elif change == "response_words":
        options[change] = 60
    elif change == "answer_max_chars":
        options[change] = 500
    elif change == "synthesis_mode":
        options[change] = "part_evidence"
    else:
        path = output / "llm_checkpoint.json"
        if change == "checkpoint_json":
            path.write_text("{bad")
        else:
            checkpoint = json.loads(path.read_text())
            parts = checkpoint["binding"]["source_parts"][0]
            if change == "binding_part_range":
                parts[0]["end_byte"] -= 1
            else:
                parts.reverse()
            path.write_text(json.dumps(checkpoint))
    calls.clear()
    assert not api.process_llm(root, output, resume=True, **options)
    assert calls == []


@pytest.mark.parametrize("phase", ["validation", "publication"])
def test_expired_verified_cache_publication_preserves_only_finished_prefix(
    synthesis_corpus, monkeypatch, phase
):
    from types import SimpleNamespace

    from gnn.llm import context_admission, corpus_runner

    root, output, _, calls, _, options = synthesis_corpus
    clock = {"now": 100.0, "puts": 0}
    actual_cache = api.LLMCache
    actual_validate = context_admission.validate_context_receipt

    def expire_after_validation(*args, **kwargs):
        actual_validate(*args, **kwargs)
        clock["now"] = 103.0

    if phase == "validation":
        monkeypatch.setattr(
            context_admission, "validate_context_receipt", expire_after_validation
        )

    class Cache(actual_cache):
        def put_verified(self, *args, **kwargs):
            publication = super().put_verified(*args, **kwargs)
            clock["puts"] += 1
            if phase == "publication" and clock["puts"] == 18:
                clock["now"] = 103.0
            return publication

    monkeypatch.setattr(api, "LLMCache", Cache)
    monkeypatch.setattr(
        corpus_runner, "time", SimpleNamespace(monotonic=lambda: clock["now"])
    )
    assert not api.process_llm(root, output, total_budget=2, **options)
    result = json.loads((output / "llm_results.json").read_text())
    checkpoint = json.loads((output / "llm_checkpoint.json").read_text())
    assert result["status"] == "timed_out"
    completed = 5 if phase == "publication" else 0
    assert result["coverage"]["prompts_completed"] == completed
    assert len(calls) == (18 if phase == "publication" else 1)
    assert len(checkpoint["completed_parts"]) == (12 if phase == "publication" else 0)
    assert len(checkpoint["completed_syntheses"]) == completed
    assert len(list((output / ".cache").rglob("*.json"))) == (
        17 if phase == "publication" else 0
    )

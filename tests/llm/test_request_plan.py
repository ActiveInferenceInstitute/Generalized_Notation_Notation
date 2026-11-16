"""S1 ``gnn_llm_request_plan_v1``: admission, checkpoints and resume revalidation.

Provider replies here are injected outcome records; they exercise admission
and scheduling only and are not native provider evidence.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from copy import deepcopy

import pytest

from gnn.llm.context_admission import checked_model_context
from gnn.llm.request_plan import (
    COMPLETION_POLICY,
    SCHEMA,
    RequestPlanRefused,
    admit_request_plan,
    next_requests,
    plan_from_record,
    reconstruct_source,
    revalidate_resume,
    segment_checkpoint,
    synthesis_checkpoint,
)

ENDPOINT = hashlib.sha256(b"http://127.0.0.1:11434").hexdigest()
MODEL_DIGEST = "sha256:" + "a" * 64
SOURCE = (
    "﻿# GNN model\r\n## StateSpaceBlock\r\ns[2,1,type=int] # état\r\n"
    "## Notes\nμ-precision ≥ 0 — 🧠 mixed endings\r\nno trailing newline"
).encode("utf-8")


def capacity(num_ctx: int = 4096, template: str = "{{ .Prompt }}") -> dict:
    return checked_model_context(
        {
            "details": {"format": "gguf"},
            "model_info": {
                "general.architecture": "llama",
                "llama.context_length": 8192,
            },
            "template": template,
        },
        {"server_version": "0.40.11", "generation_options": {"num_ctx": num_ctx}},
    )


def admit(source: bytes = SOURCE, **overrides):
    arguments = {
        "provider": "ollama",
        "endpoint_sha256": ENDPOINT,
        "model": "gemma3:4b",
        "model_digest": MODEL_DIGEST,
        "prompt_definition": {"label": "summary", "system": "Analyze GNN."},
        "capacity": capacity(),
        "reservations": {
            "system": 64,
            "task": 128,
            "prior_receipts": 512,
            "output": 256,
        },
        "mode": "part_evidence",
        "segment_max_bytes": 16,
        "deadline_seconds": 600,
        "request_timeout_seconds": 120,
    }
    arguments.update(overrides)
    return admit_request_plan(source, **arguments)


def outcome(**overrides) -> dict:
    value = {
        "status": "success",
        "finish_reason": "stop",
        "completion_policy": COMPLETION_POLICY,
        "provider": "ollama",
        "model": "gemma3:4b",
        "model_digest": MODEL_DIGEST,
        "endpoint_sha256": ENDPOINT,
        "prompt_tokens": 300,
        "completion_tokens": 40,
    }
    value.update(overrides)
    return value


def checkpoints(plan, skip: int = 0) -> list[dict]:
    return [
        segment_checkpoint(plan, segment.segment_id, f"answer {index}", outcome())
        for index, segment in enumerate(plan.segments[skip:])
    ]


@pytest.mark.parametrize(
    "source",
    [
        SOURCE,
        SOURCE + b"\r\n",
        b"",
        b"\n",
        "🧠🧠🧠🧠🧠".encode(),
        "a\r\nb\rc\n d".encode(),
    ],
)
def test_segments_are_contiguous_and_reconstruct_exact_bytes(source):
    plan = admit(source, segment_max_bytes=5)
    record = plan.record()
    assert record["schema"] == SCHEMA
    assert record["source"] == {
        "encoding": "utf-8",
        "sha256": hashlib.sha256(source).hexdigest(),
        "byte_length": len(source),
    }
    offset = 0
    for segment in plan.segments:
        assert segment.start_byte == offset
        piece = source[segment.start_byte : segment.end_byte]
        piece.decode("utf-8")
        assert hashlib.sha256(piece).hexdigest() == segment.sha256
        offset = segment.end_byte
    assert offset == len(source)
    assert reconstruct_source(plan, source) == source
    assert plan_from_record(record).sha256 == plan.sha256


def test_crlf_and_trailing_byte_changes_are_distinct_sources():
    lf = SOURCE.replace(b"\r\n", b"\n")
    assert admit(lf).sha256 != admit(SOURCE).sha256
    with pytest.raises(RequestPlanRefused, match="source bytes differ"):
        reconstruct_source(admit(SOURCE), SOURCE + b"\n")
    with pytest.raises(RequestPlanRefused, match="source bytes differ"):
        reconstruct_source(admit(SOURCE), lf)


def test_non_utf8_and_text_sources_refuse():
    with pytest.raises(RequestPlanRefused, match="strict UTF-8"):
        admit(b"\xff\xfe broken")
    with pytest.raises(RequestPlanRefused, match="exact file bytes"):
        admit(SOURCE.decode())  # type: ignore[arg-type]


def test_plan_is_immutable_and_deterministic():
    plan = admit()
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.sha256 = "0" * 64  # type: ignore[misc]
    record = plan.record()
    record["model"] = "other"
    record["segments"].clear()
    assert plan.record()["model"] == "gemma3:4b"
    assert plan.record()["segments"]
    assert admit().sha256 == plan.sha256
    assert plan.cache_identity() != plan.cache_identity("other_policy")
    assert plan.cache_identity() != admit(segment_max_bytes=17).cache_identity()
    policies = plan.record()["policies"]
    assert policies == {
        "completion_policy": COMPLETION_POLICY,
        "truncation": False,
        "model_substitution": False,
        "budget_increase": False,
    }


@pytest.mark.parametrize(
    "overrides,field",
    [
        ({"model": "llama3:8b"}, "model"),
        ({"model_digest": "sha256:" + "b" * 64}, "model"),
        ({"endpoint_sha256": "c" * 64}, "endpoint"),
        ({"capacity": capacity(template="{{ .System }}")}, "tokenizer"),
        ({"prompt_definition": {"label": "summary", "system": "Changed."}}, "prompt"),
        ({"provider": "openai"}, "provider"),
        (
            {
                "reservations": {
                    "system": 64,
                    "task": 128,
                    "prior_receipts": 512,
                    "output": 512,
                }
            },
            "context",
        ),
        ({"segment_max_bytes": 32}, "segments"),
        ({"deadline_seconds": 900}, "deadline"),
    ],
)
def test_changed_identities_reject_resume(overrides, field):
    retained = admit().record()
    with pytest.raises(
        RequestPlanRefused, match=f"changed since checkpointing: .*{field}"
    ):
        revalidate_resume(retained, admit(**overrides), SOURCE, [])


def test_changed_source_rejects_resume():
    retained = admit().record()
    changed = SOURCE.replace(b"\r\n", b"\n", 1)
    with pytest.raises(RequestPlanRefused, match="source"):
        revalidate_resume(retained, admit(changed), changed, [])
    with pytest.raises(RequestPlanRefused, match="source bytes differ"):
        revalidate_resume(retained, admit(), changed, [])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(plan_sha256="0" * 64),
        lambda r: r["segments"][1].update(
            start_byte=r["segments"][1]["start_byte"] + 1
        ),
        lambda r: r["segments"][0].update(sha256="e" * 64),
        lambda r: r["segments"].pop(),
        lambda r: r.update(schema="gnn_llm_request_plan_v0"),
        lambda r: r.pop("segments"),
    ],
)
def test_corrupt_retained_plan_records_refuse(mutate):
    record = admit().record()
    mutate(record)
    with pytest.raises(RequestPlanRefused):
        plan_from_record(record)


def test_unreviewed_capacity_and_impossible_budgets_refuse():
    with pytest.raises(RequestPlanRefused, match="reviewed provider-contract"):
        admit(capacity={"capacity_tokens": 4096, "origin": "len(text) / 4"})
    with pytest.raises(RequestPlanRefused, match="no admitted source budget"):
        admit(
            reservations={
                "system": 2000,
                "task": 1000,
                "prior_receipts": 800,
                "output": 296,
            }
        )
    with pytest.raises(RequestPlanRefused, match="exactly"):
        admit(reservations={"system": 1, "task": 1, "output": 1})
    with pytest.raises(RequestPlanRefused, match="output reservation"):
        admit(reservations={"system": 1, "task": 1, "prior_receipts": 0, "output": 0})
    with pytest.raises(RequestPlanRefused, match="timeout exceeds"):
        admit(deadline_seconds=60, request_timeout_seconds=120)
    with pytest.raises(RequestPlanRefused, match="segment_max_bytes"):
        admit(segment_max_bytes=3)
    with pytest.raises(RequestPlanRefused, match="mode"):
        admit(mode="none")


def native_count(tokens: int, **overrides) -> dict:
    value = {
        "origin": "native_tokenizer",
        "tokens": tokens,
        "model_digest": MODEL_DIGEST,
        "template_sha256": capacity()["template_sha256"],
    }
    value.update(overrides)
    return value


def test_joint_source_requires_native_count_that_fits():
    with pytest.raises(RequestPlanRefused, match="native tokenizer count"):
        admit(mode="joint_source")
    with pytest.raises(RequestPlanRefused, match="native tokenizer"):
        admit(
            mode="joint_source", source_token_count=native_count(10, origin="chars/4")
        )
    with pytest.raises(RequestPlanRefused, match="different tokenizer"):
        admit(
            mode="joint_source",
            source_token_count=native_count(10, model_digest="sha256:" + "d" * 64),
        )
    # 4096 capacity - (64+128+512+256+1) = 3135 tokens remain for the source.
    with pytest.raises(RequestPlanRefused, match="refusing without truncation"):
        admit(mode="joint_source", source_token_count=native_count(3136))
    plan = admit(mode="joint_source", source_token_count=native_count(3135))
    assert plan.record()["full_source"] == {
        "required": True,
        "admitted": True,
        "required_tokens": 4096,
    }


def test_incomplete_and_misidentified_replies_cannot_checkpoint():
    plan = admit()
    segment = plan.segments[0].segment_id
    for bad, match in (
        (outcome(status="incomplete", finish_reason="length"), "incomplete"),
        (outcome(finish_reason="length"), "incomplete"),
        (outcome(completion_policy="best_effort"), "terminal_stop_v1"),
        (outcome(model="llama3:8b"), "model differs"),
        (outcome(endpoint_sha256="f" * 64), "endpoint_sha256 differs"),
        (outcome(prompt_tokens=4096 - 256), "exceed the admitted context"),
        (outcome(completion_tokens=257), "exceed the admitted context"),
        (outcome(prompt_tokens=None), "native prompt tokens"),
    ):
        with pytest.raises(RequestPlanRefused, match=match):
            segment_checkpoint(plan, segment, "answer", bad)
    with pytest.raises(RequestPlanRefused, match="not in the plan"):
        segment_checkpoint(plan, "segment_9999", "answer", outcome())
    with pytest.raises(RequestPlanRefused, match="empty"):
        segment_checkpoint(plan, segment, "  ", outcome())


def test_interrupted_resume_schedules_only_missing_segments():
    plan = admit()
    count = len(plan.segments)
    assert count > 3
    retained = json.loads(json.dumps(plan.record()))
    saved = json.loads(json.dumps(checkpoints(plan)[:2]))
    state, accepted = revalidate_resume(retained, plan, SOURCE, saved)
    assert state.accepted == tuple(s.segment_id for s in plan.segments[:2])
    assert state.pending == tuple(s.segment_id for s in plan.segments[2:])
    assert state.synthesis == "blocked" and not state.complete
    decision = next_requests(plan, state, started_at=0.0, now=10.0, cancelled=False)
    assert decision["status"] == "segments"
    assert [row["segment_id"] for row in decision["requests"]] == list(state.pending)
    assert set(accepted) == set(state.accepted)


@pytest.mark.parametrize(
    "corrupt,reason",
    [
        (lambda c: c.update(response="edited"), "digest differs"),
        (lambda c: c.update(checkpoint_sha256="0" * 64), "digest differs"),
        (lambda c: c.update(schema="other"), "schema differs"),
        (lambda c: c.update(plan_sha256="1" * 64), "different request plan"),
        (lambda c: c["outcome"].update(finish_reason="length"), "digest differs"),
    ],
)
def test_corrupt_checkpoints_are_rejected_and_rescheduled(corrupt, reason):
    plan = admit()
    saved = checkpoints(plan)
    corrupt(saved[1])
    state, _ = revalidate_resume(plan.record(), plan, SOURCE, saved)
    assert plan.segments[1].segment_id in state.pending
    assert len(state.pending) == 1
    assert reason in state.rejected[0][1]


def test_resealed_incomplete_or_relabeled_checkpoints_are_rejected():
    plan = admit()
    saved = checkpoints(plan)
    for index, mutate in enumerate(
        (
            lambda c: c["outcome"].update(status="incomplete", finish_reason="length"),
            lambda c: c.update(segment_id=plan.segments[0].segment_id),
            lambda c: c.update(response_sha256="2" * 64),
        ),
        start=1,
    ):
        entry = saved[index]
        mutate(entry)
        body = {k: v for k, v in entry.items() if k != "checkpoint_sha256"}
        entry["checkpoint_sha256"] = hashlib.sha256(
            json.dumps(
                body, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            ).encode()
        ).hexdigest()
    state, _ = revalidate_resume(plan.record(), plan, SOURCE, saved)
    assert set(state.pending) >= {s.segment_id for s in plan.segments[1:4]}
    assert any("incomplete" in reason for _, reason in state.rejected)
    assert any("differs" in reason for _, reason in state.rejected)
    assert not isinstance(
        revalidate_resume(plan.record(), plan, SOURCE, [None])[0].rejected[0][1],
        type(None),
    )


def test_conflicting_duplicate_checkpoints_reject_the_segment():
    plan = admit()
    saved = checkpoints(plan)
    other = segment_checkpoint(
        plan, plan.segments[0].segment_id, "different", outcome()
    )
    state, accepted = revalidate_resume(
        plan.record(), plan, SOURCE, [*saved, saved[0], other]
    )
    assert state.pending == (plan.segments[0].segment_id,)
    assert plan.segments[0].segment_id not in accepted
    assert any("conflicting" in reason for _, reason in state.rejected)


@pytest.mark.parametrize("mode", ["part_evidence", "joint_source"])
def test_synthesis_completes_only_with_every_current_segment(mode):
    extra = {"source_token_count": native_count(100)} if mode == "joint_source" else {}
    plan = admit(mode=mode, **extra)
    saved = checkpoints(plan)
    state, accepted = revalidate_resume(plan.record(), plan, SOURCE, saved)
    assert state.synthesis == "ready" and not state.pending
    decision = next_requests(plan, state, started_at=0.0, now=1.0, cancelled=False)
    assert decision["status"] == "synthesis"
    assert decision["requests"][0]["mode"] == mode
    final = synthesis_checkpoint(plan, accepted, '{"summary": "ok"}', outcome())
    assert final["full_source_claim"] is (mode == "joint_source")
    state, _ = revalidate_resume(plan.record(), plan, SOURCE, saved, final)
    assert state.complete and state.synthesis == "complete"
    assert state.full_source_claim is (mode == "joint_source")
    assert (
        next_requests(plan, state, started_at=0.0, now=1.0, cancelled=False)["status"]
        == "complete"
    )
    # A re-run segment changes the dependency digest; the old synthesis is stale.
    rerun = deepcopy(saved)
    rerun[0] = segment_checkpoint(plan, plan.segments[0].segment_id, "rerun", outcome())
    stale, _ = revalidate_resume(plan.record(), plan, SOURCE, rerun, final)
    assert stale.synthesis == "rejected" and not stale.complete
    # Missing a dependency blocks synthesis creation and completion.
    partial, partial_accepted = revalidate_resume(
        plan.record(), plan, SOURCE, saved[1:], final
    )
    assert partial.synthesis == "rejected" and "unfinished" in partial.synthesis_reason
    with pytest.raises(RequestPlanRefused, match="unfinished"):
        synthesis_checkpoint(plan, partial_accepted, "x", outcome())


def test_incomplete_or_relabeled_synthesis_never_completes():
    plan = admit()
    saved = checkpoints(plan)
    _, accepted = revalidate_resume(plan.record(), plan, SOURCE, saved)
    with pytest.raises(RequestPlanRefused, match="incomplete"):
        synthesis_checkpoint(
            plan, accepted, "x", outcome(status="incomplete", finish_reason="length")
        )
    final = synthesis_checkpoint(plan, accepted, "answer", outcome())
    relabeled = deepcopy(final)
    relabeled["full_source_claim"] = True
    state, _ = revalidate_resume(plan.record(), plan, SOURCE, saved, relabeled)
    assert state.synthesis == "rejected" and not state.full_source_claim


def test_deadline_and_cancellation_leave_work_unfinished():
    plan = admit()
    state, _ = revalidate_resume(plan.record(), plan, SOURCE, [])
    assert (
        next_requests(plan, state, started_at=0.0, now=1.0, cancelled=True)["status"]
        == "cancelled"
    )
    late = next_requests(plan, state, started_at=0.0, now=481.0, cancelled=False)
    assert late["status"] == "deadline_exceeded" and late["requests"] == []
    other = admit(segment_max_bytes=40)
    with pytest.raises(RequestPlanRefused, match="different request plan"):
        next_requests(other, state, started_at=0.0, now=1.0, cancelled=False)

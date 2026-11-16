"""Immutable S1 request plans, segment checkpoints and resume revalidation.

``gnn_llm_request_plan_v1`` is admitted before any provider call. It binds the
exact source bytes, their contiguous segment byte offsets, the provider,
endpoint, model, tokenizer/chat-template and prompt-definition identities, the
context reservations, the synthesis task contract and the invocation deadline.
Nothing in a plan can change after admission: segment requests, checkpoints,
synthesis and cache identities all carry the full plan digest.

This module performs no provider I/O. It admits plans, creates terminal
checkpoints from already-observed replies and revalidates retained records on
resume. Context capacity comes only from the reviewed provider contract in
:mod:`gnn.llm.context_admission`; byte bounds never certify token fit, so each
reply must still carry native token counts inside the admitted budget.
A segment answer is evidence about its segment only. ``part_evidence``
synthesis never claims full-source reading; ``joint_source`` is admitted only
when a native tokenizer count shows the entire source fits one request.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .context_admission import POLICY as CONTEXT_POLICY

SCHEMA = "gnn_llm_request_plan_v1"
SEGMENT_CHECKPOINT_SCHEMA = "gnn_llm_segment_checkpoint_v1"
SYNTHESIS_CHECKPOINT_SCHEMA = "gnn_llm_synthesis_checkpoint_v1"
COMPLETION_POLICY = "terminal_stop_v1"
ENCODING = "utf-8"
TERMINAL_TOKENS = 1
TASK_CONTRACTS: dict[str, dict[str, Any]] = {
    "part_evidence": {
        "contract": "part_evidence_synthesis_v1",
        "requires_full_source": False,
        "may_conclude": "Only what the ordered segment answers jointly support; "
        "no claim of having read the original source in one context.",
    },
    "joint_source": {
        "contract": "joint_source_synthesis_v1",
        "requires_full_source": True,
        "may_conclude": "Relationships across the exact original source supplied "
        "intact in one admitted request, using segment answers as fallible aids.",
    },
}
RESERVATION_KEYS = ("system", "task", "prior_receipts", "output")
OUTCOME_IDENTITY_KEYS = ("provider", "model", "model_digest", "endpoint_sha256")
_HEX64 = re.compile("[0-9a-f]{64}")


class RequestPlanRefused(ValueError):
    """Admission, checkpoint or resume refused; no weaker fallback is chosen."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest_json(value: Any) -> str:
    """SHA-256 of canonical JSON (sorted keys, UTF-8, no NaN)."""
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hex64(value: Any, name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise RequestPlanRefused(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RequestPlanRefused(f"{name} must be a nonempty string")
    return value


def _count(value: Any, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise RequestPlanRefused(f"{name} must be an integer >= {minimum}")
    return value


def _seconds(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestPlanRefused(f"{name} must be a positive number of seconds")
    if not 0 < float(value) < float("inf"):
        raise RequestPlanRefused(f"{name} must be a positive number of seconds")
    return float(value)


@dataclass(frozen=True)
class PlanSegment:
    """One contiguous byte range of the source and its request identity."""

    segment_id: str
    start_byte: int
    end_byte: int
    sha256: str
    request_sha256: str

    def record(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
            "sha256": self.sha256,
            "request_sha256": self.request_sha256,
        }


@dataclass(frozen=True)
class RequestPlan:
    """An admitted, immutable plan; ``record()`` returns a fresh JSON copy."""

    sha256: str
    core_sha256: str
    segments: tuple[PlanSegment, ...]
    _record_json: str

    def record(self) -> dict[str, Any]:
        value: dict[str, Any] = json.loads(self._record_json)
        return value

    @property
    def mode(self) -> str:
        return str(self.record()["task_contract"]["mode"])

    def segment(self, segment_id: str) -> PlanSegment:
        for segment in self.segments:
            if segment.segment_id == segment_id:
                return segment
        raise RequestPlanRefused(f"segment {segment_id!r} is not in the plan")

    def cache_identity(self, completion_policy: str = COMPLETION_POLICY) -> str:
        """Bind cache entries to the complete plan and completion policy."""
        return digest_json(
            {"plan_sha256": self.sha256, "completion_policy": completion_policy}
        )


def _segment_bounds(source: bytes, max_bytes: int) -> list[tuple[int, int]]:
    """Cut on UTF-8 scalar boundaries, never inside a multibyte sequence."""
    bounds = []
    start = 0
    while start < len(source):
        end = min(len(source), start + max_bytes)
        while end < len(source) and source[end] & 0xC0 == 0x80:
            end -= 1
        bounds.append((start, end))
        start = end
    return bounds or [(0, 0)]


def _checked_capacity(capacity: Mapping[str, Any]) -> dict[str, Any]:
    """Accept only a reviewed provider-contract context record."""
    if not isinstance(capacity, Mapping) or capacity.get("policy") != CONTEXT_POLICY:
        raise RequestPlanRefused(
            "context capacity requires a reviewed provider-contract record; "
            "character counts or guessed tokens cannot certify admission"
        )
    maximum = _count(capacity.get("model_context_tokens"), "model_context_tokens")
    requested = _count(
        capacity.get("requested_context_tokens"), "requested_context_tokens"
    )
    if requested > maximum:
        raise RequestPlanRefused("requested context exceeds advertised model context")
    return {
        "policy": CONTEXT_POLICY,
        "architecture": _text(capacity.get("architecture"), "architecture"),
        "model_context_tokens": maximum,
        "capacity_tokens": requested,
        "template_sha256": _hex64(capacity.get("template_sha256"), "template_sha256"),
    }


def _checked_source_count(
    count: Mapping[str, Any] | None, model_digest: str, template_sha256: str
) -> dict[str, Any] | None:
    if count is None:
        return None
    if not isinstance(count, Mapping) or count.get("origin") != "native_tokenizer":
        raise RequestPlanRefused(
            "full-source token count must come from the native tokenizer"
        )
    if (
        count.get("model_digest") != model_digest
        or count.get("template_sha256") != template_sha256
    ):
        raise RequestPlanRefused(
            "full-source token count was measured under a different tokenizer identity"
        )
    return {
        "origin": "native_tokenizer",
        "tokens": _count(count.get("tokens"), "source tokens", minimum=0),
        "model_digest": model_digest,
        "template_sha256": template_sha256,
    }


def admit_request_plan(
    source: bytes,
    *,
    provider: str,
    endpoint_sha256: str,
    model: str,
    model_digest: str,
    prompt_definition: Mapping[str, Any],
    capacity: Mapping[str, Any],
    reservations: Mapping[str, int],
    mode: str,
    segment_max_bytes: int,
    deadline_seconds: float,
    request_timeout_seconds: float,
    source_token_count: Mapping[str, Any] | None = None,
) -> RequestPlan:
    """Admit one immutable plan before any provider call, or refuse.

    ``source`` is the exact file bytes; CRLF, BOM, Unicode and the final byte
    are preserved and must decode as strict UTF-8. ``reservations`` budget the
    system message, task instructions, prior segment receipts and output
    generation. Full-source tasks refuse when the native source count plus the
    reservations cannot fit the admitted capacity; no truncation, model
    substitution or budget increase is ever selected instead.
    """
    if not isinstance(source, bytes):
        raise RequestPlanRefused("source must be the exact file bytes")
    try:
        source.decode(ENCODING, errors="strict")
    except UnicodeDecodeError as exc:
        raise RequestPlanRefused(f"source is not strict UTF-8: {exc}") from exc
    if mode not in TASK_CONTRACTS:
        raise RequestPlanRefused("mode must be part_evidence or joint_source")
    _count(segment_max_bytes, "segment_max_bytes", minimum=4)
    deadline = _seconds(deadline_seconds, "deadline_seconds")
    timeout = _seconds(request_timeout_seconds, "request_timeout_seconds")
    if timeout > deadline:
        raise RequestPlanRefused("request timeout exceeds the invocation deadline")
    if not isinstance(prompt_definition, Mapping) or not prompt_definition:
        raise RequestPlanRefused("prompt definition must be a nonempty mapping")
    try:
        prompt_sha256 = digest_json(dict(prompt_definition))
    except (TypeError, ValueError) as exc:
        raise RequestPlanRefused(f"prompt definition is not canonical JSON: {exc}")
    context = _checked_capacity(capacity)
    digest = _text(model_digest, "model_digest")
    if not isinstance(reservations, Mapping) or set(reservations) != set(
        RESERVATION_KEYS
    ):
        raise RequestPlanRefused(
            "reservations must name exactly " + ", ".join(RESERVATION_KEYS)
        )
    reserved = {
        key: _count(reservations[key], f"{key} reservation", minimum=0)
        for key in RESERVATION_KEYS
    }
    _count(reserved["output"], "output reservation")
    overhead = sum(reserved.values()) + TERMINAL_TOKENS
    segment_budget = context["capacity_tokens"] - overhead
    if segment_budget < 1:
        raise RequestPlanRefused("reservations leave no admitted source budget")
    counted = _checked_source_count(
        source_token_count, digest, context["template_sha256"]
    )
    task = {"mode": mode, **TASK_CONTRACTS[mode]}
    full_source: dict[str, Any] = {"required": task["requires_full_source"]}
    if task["requires_full_source"]:
        if counted is None:
            raise RequestPlanRefused(
                "joint_source requires a native tokenizer count of the full source"
            )
        needed = counted["tokens"] + overhead
        if needed > context["capacity_tokens"]:
            raise RequestPlanRefused(
                f"full-source task needs {needed} tokens but admitted context is "
                f"{context['capacity_tokens']}; refusing without truncation"
            )
        full_source.update(admitted=True, required_tokens=needed)
    core = {
        "schema": SCHEMA,
        "source": {
            "encoding": ENCODING,
            "sha256": digest_bytes(source),
            "byte_length": len(source),
        },
        "provider": _text(provider, "provider"),
        "endpoint_sha256": _hex64(endpoint_sha256, "endpoint_sha256"),
        "model": _text(model, "model"),
        "model_digest": digest,
        "tokenizer_identity": {
            "model_digest": digest,
            "template_sha256": context["template_sha256"],
        },
        "prompt_definition_sha256": prompt_sha256,
        "context": {
            **context,
            "reservations": reserved,
            "terminal_tokens": TERMINAL_TOKENS,
            "segment_token_budget": segment_budget,
            "source_token_count": counted,
        },
        "task_contract": task,
        "full_source": full_source,
        "segment_max_bytes": segment_max_bytes,
        "deadline": {
            "deadline_seconds": deadline,
            "request_timeout_seconds": timeout,
            "clock": "monotonic",
            "cancellation": "required",
        },
        "policies": {
            "completion_policy": COMPLETION_POLICY,
            "truncation": False,
            "model_substitution": False,
            "budget_increase": False,
        },
    }
    core_sha256 = digest_json(core)
    segments = []
    for index, (start, end) in enumerate(_segment_bounds(source, segment_max_bytes)):
        segment_id = f"segment_{index + 1:04d}"
        content = digest_bytes(source[start:end])
        request = digest_json(
            {
                "plan_core_sha256": core_sha256,
                "segment_id": segment_id,
                "start_byte": start,
                "end_byte": end,
                "sha256": content,
            }
        )
        segments.append(PlanSegment(segment_id, start, end, content, request))
    return _seal(core, tuple(segments))


def _seal(core: dict[str, Any], segments: tuple[PlanSegment, ...]) -> RequestPlan:
    record = {**core, "segments": [segment.record() for segment in segments]}
    record["plan_sha256"] = digest_json(record)
    return RequestPlan(
        sha256=record["plan_sha256"],
        core_sha256=digest_json(core),
        segments=segments,
        _record_json=_canonical(record),
    )


def reconstruct_source(plan: RequestPlan, source: bytes) -> bytes:
    """Reassemble the source from plan offsets and verify every byte and digest."""
    record = plan.record()
    if (
        digest_bytes(source) != record["source"]["sha256"]
        or len(source) != record["source"]["byte_length"]
    ):
        raise RequestPlanRefused("source bytes differ from the admitted plan")
    offset = 0
    pieces = []
    for segment in plan.segments:
        if segment.start_byte != offset or segment.end_byte < segment.start_byte:
            raise RequestPlanRefused("segment offsets contain a gap or overlap")
        piece = source[segment.start_byte : segment.end_byte]
        if digest_bytes(piece) != segment.sha256:
            raise RequestPlanRefused(f"{segment.segment_id} digest differs")
        piece.decode(ENCODING, errors="strict")
        pieces.append(piece)
        offset = segment.end_byte
    rebuilt = b"".join(pieces)
    if offset != len(source) or rebuilt != source:
        raise RequestPlanRefused("segments do not reconstruct the exact source")
    return rebuilt


def plan_from_record(record: Mapping[str, Any]) -> RequestPlan:
    """Rehydrate a retained plan, refusing any internally inconsistent record."""
    if not isinstance(record, Mapping) or record.get("schema") != SCHEMA:
        raise RequestPlanRefused("retained request plan has an unsupported schema")
    try:
        body = {key: value for key, value in record.items() if key != "plan_sha256"}
        core = {key: value for key, value in body.items() if key != "segments"}
        core_sha256 = digest_json(core)
        rows = body["segments"]
        if not isinstance(rows, list) or not rows:
            raise RequestPlanRefused("retained request plan has no segments")
        segments = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or set(row) != {
                "segment_id",
                "start_byte",
                "end_byte",
                "sha256",
                "request_sha256",
            }:
                raise RequestPlanRefused("retained segment record is malformed")
            expected_id = f"segment_{index + 1:04d}"
            request = digest_json(
                {
                    "plan_core_sha256": core_sha256,
                    "segment_id": row["segment_id"],
                    "start_byte": row["start_byte"],
                    "end_byte": row["end_byte"],
                    "sha256": row["sha256"],
                }
            )
            if row["segment_id"] != expected_id or row["request_sha256"] != request:
                raise RequestPlanRefused(
                    f"retained {expected_id} request digest differs"
                )
            segments.append(
                PlanSegment(
                    row["segment_id"],
                    _count(row["start_byte"], "start_byte", minimum=0),
                    _count(row["end_byte"], "end_byte", minimum=0),
                    _hex64(row["sha256"], "segment sha256"),
                    row["request_sha256"],
                )
            )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, RequestPlanRefused):
            raise
        raise RequestPlanRefused(f"retained request plan is corrupt: {exc}") from exc
    plan = _seal(core, tuple(segments))
    if plan.sha256 != record.get("plan_sha256"):
        raise RequestPlanRefused("retained request plan digest differs")
    return plan


def _plan_differences(retained: dict[str, Any], current: dict[str, Any]) -> list[str]:
    fields = {
        "source": ("source",),
        "provider": ("provider",),
        "endpoint": ("endpoint_sha256",),
        "model": ("model", "model_digest"),
        "tokenizer": ("tokenizer_identity",),
        "prompt": ("prompt_definition_sha256",),
        "context": ("context",),
        "task_contract": ("task_contract", "full_source"),
        "segments": ("segment_max_bytes", "segments"),
        "deadline": ("deadline",),
        "policies": ("policies",),
    }
    return [
        name
        for name, keys in fields.items()
        if any(retained.get(key) != current.get(key) for key in keys)
    ]


def _checked_outcome(plan: RequestPlan, outcome: Any) -> dict[str, Any]:
    """Require a terminal stop with the plan's identity and admitted budget."""
    if not isinstance(outcome, Mapping):
        raise RequestPlanRefused("reply outcome is unavailable")
    status = outcome.get("status")
    if status == "incomplete" or outcome.get("finish_reason") != "stop":
        raise RequestPlanRefused(
            "reply is incomplete; only terminal_stop_v1 replies may be checkpointed"
        )
    if status != "success" or outcome.get("completion_policy") != COMPLETION_POLICY:
        raise RequestPlanRefused("reply did not satisfy terminal_stop_v1")
    record = plan.record()
    for key in OUTCOME_IDENTITY_KEYS:
        if outcome.get(key) != record[key]:
            raise RequestPlanRefused(f"reply {key} differs from the admitted plan")
    context = record["context"]
    output = context["reservations"]["output"]
    prompt = _count(outcome.get("prompt_tokens"), "native prompt tokens")
    completion = _count(outcome.get("completion_tokens"), "native output tokens")
    if (
        completion > output
        or prompt + output + context["terminal_tokens"] > context["capacity_tokens"]
    ):
        raise RequestPlanRefused("native token counts exceed the admitted context")
    return dict(outcome)


def _sealed(entry: dict[str, Any]) -> dict[str, Any]:
    entry["checkpoint_sha256"] = digest_json(entry)
    return entry


def segment_checkpoint(
    plan: RequestPlan, segment_id: str, response: str, outcome: Mapping[str, Any]
) -> dict[str, Any]:
    """Create a terminal segment checkpoint; incomplete replies are refused."""
    segment = plan.segment(segment_id)
    if not isinstance(response, str) or not response.strip():
        raise RequestPlanRefused("segment response is empty")
    return _sealed(
        {
            "schema": SEGMENT_CHECKPOINT_SCHEMA,
            "plan_sha256": plan.sha256,
            "segment_id": segment.segment_id,
            "segment_sha256": segment.sha256,
            "request_sha256": segment.request_sha256,
            "response": response,
            "response_sha256": digest_bytes(response.encode("utf-8")),
            "outcome": _checked_outcome(plan, outcome),
        }
    )


def _segment_dependencies(
    plan: RequestPlan, accepted: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, str]]:
    missing = [s.segment_id for s in plan.segments if s.segment_id not in accepted]
    if missing:
        raise RequestPlanRefused(
            "synthesis dependencies are unfinished: " + ", ".join(missing)
        )
    return [
        {
            "segment_id": s.segment_id,
            "checkpoint_sha256": str(accepted[s.segment_id]["checkpoint_sha256"]),
        }
        for s in plan.segments
    ]


def synthesis_checkpoint(
    plan: RequestPlan,
    segment_checkpoints: Mapping[str, Mapping[str, Any]],
    response: str,
    outcome: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a terminal synthesis to the plan and every accepted segment."""
    if not isinstance(response, str) or not response.strip():
        raise RequestPlanRefused("synthesis response is empty")
    record = plan.record()
    if record["full_source"]["required"] and not record["full_source"].get("admitted"):
        raise RequestPlanRefused("joint_source synthesis was not admitted")
    return _sealed(
        {
            "schema": SYNTHESIS_CHECKPOINT_SCHEMA,
            "plan_sha256": plan.sha256,
            "mode": record["task_contract"]["mode"],
            "task_contract": record["task_contract"]["contract"],
            "full_source_claim": record["task_contract"]["requires_full_source"],
            "dependencies": _segment_dependencies(plan, segment_checkpoints),
            "response": response,
            "response_sha256": digest_bytes(response.encode("utf-8")),
            "outcome": _checked_outcome(plan, outcome),
        }
    )


def _stored_reply(entry: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    response, outcome = entry.get("response"), entry.get("outcome")
    if not isinstance(response, str) or not isinstance(outcome, Mapping):
        raise RequestPlanRefused("checkpoint response or outcome is malformed")
    return response, outcome


def _revalidate_segment(plan: RequestPlan, entry: Any) -> dict[str, Any]:
    if not isinstance(entry, Mapping):
        raise RequestPlanRefused("checkpoint is not a mapping")
    if entry.get("schema") != SEGMENT_CHECKPOINT_SCHEMA:
        raise RequestPlanRefused("checkpoint schema differs")
    if entry.get("plan_sha256") != plan.sha256:
        raise RequestPlanRefused("checkpoint belongs to a different request plan")
    body = {key: value for key, value in entry.items() if key != "checkpoint_sha256"}
    if entry.get("checkpoint_sha256") != digest_json(body):
        raise RequestPlanRefused("checkpoint digest differs (corrupt checkpoint)")
    response, outcome = _stored_reply(entry)
    rebuilt = segment_checkpoint(plan, str(entry.get("segment_id")), response, outcome)
    if rebuilt != dict(entry):
        raise RequestPlanRefused("checkpoint segment, request or response differs")
    return rebuilt


@dataclass(frozen=True)
class ResumeState:
    """Revalidated progress; only ``complete`` may finish the prompt unit."""

    plan_sha256: str
    accepted: tuple[str, ...]
    pending: tuple[str, ...]
    rejected: tuple[tuple[str, str], ...]
    synthesis: str
    synthesis_reason: str
    complete: bool
    full_source_claim: bool


def revalidate_resume(
    retained_plan: Mapping[str, Any],
    current: RequestPlan,
    source: bytes,
    checkpoints: Iterable[Any] = (),
    synthesis: Any = None,
) -> tuple[ResumeState, dict[str, dict[str, Any]]]:
    """Revalidate a retained plan and its checkpoints before scheduling more work.

    A retained plan whose digest differs from the plan admitted from current
    inputs (source, provider, endpoint, model, tokenizer, prompt, context,
    task, segments or deadline) is refused as a whole. Individually corrupt,
    stale, conflicting or incomplete checkpoints are rejected and their
    segments return to ``pending``; nothing borrows an earlier verdict.
    """
    retained = plan_from_record(retained_plan)
    if retained.sha256 != current.sha256:
        changed = _plan_differences(retained.record(), current.record())
        raise RequestPlanRefused(
            "request plan changed since checkpointing: " + ", ".join(changed)
        )
    reconstruct_source(current, source)
    accepted: dict[str, dict[str, Any]] = {}
    rejected: list[tuple[str, str]] = []
    conflicted: set[str] = set()
    for index, entry in enumerate(checkpoints):
        label = (
            str(entry.get("segment_id"))
            if isinstance(entry, Mapping)
            else f"checkpoint[{index}]"
        )
        try:
            valid = _revalidate_segment(current, entry)
        except RequestPlanRefused as exc:
            rejected.append((label, str(exc)))
            continue
        previous = accepted.get(valid["segment_id"])
        if previous is not None and previous != valid:
            conflicted.add(valid["segment_id"])
            rejected.append((label, "conflicting checkpoints for one segment"))
        accepted.setdefault(valid["segment_id"], valid)
    for segment_id in conflicted:
        accepted.pop(segment_id, None)
    order = [segment.segment_id for segment in current.segments]
    pending = tuple(segment_id for segment_id in order if segment_id not in accepted)
    full_source_claim = bool(current.record()["task_contract"]["requires_full_source"])
    status, reason, complete = "blocked", "segment dependencies are unfinished", False
    if not pending:
        status, reason = "ready", "every segment checkpoint is terminal and current"
    if synthesis is not None:
        try:
            if not isinstance(synthesis, Mapping):
                raise RequestPlanRefused("synthesis checkpoint is not a mapping")
            body = {k: v for k, v in synthesis.items() if k != "checkpoint_sha256"}
            if synthesis.get("checkpoint_sha256") != digest_json(body):
                raise RequestPlanRefused("synthesis checkpoint digest differs")
            response, outcome = _stored_reply(synthesis)
            rebuilt = synthesis_checkpoint(current, accepted, response, outcome)
            if rebuilt != dict(synthesis):
                raise RequestPlanRefused(
                    "synthesis plan, mode or segment dependencies differ"
                )
            status, reason, complete = "complete", "", True
        except RequestPlanRefused as exc:
            status, reason = "rejected", str(exc)
    return (
        ResumeState(
            plan_sha256=current.sha256,
            accepted=tuple(
                segment_id for segment_id in order if segment_id in accepted
            ),
            pending=pending,
            rejected=tuple(rejected),
            synthesis=status,
            synthesis_reason=reason,
            complete=complete,
            full_source_claim=full_source_claim and complete,
        ),
        accepted,
    )


def next_requests(
    plan: RequestPlan,
    state: ResumeState,
    *,
    started_at: float,
    now: float,
    cancelled: bool,
) -> dict[str, Any]:
    """Return the next schedulable requests under the plan's deadline.

    ``started_at``/``now`` are monotonic clock readings from one invocation.
    Cancellation and an exhausted deadline leave remaining work unfinished; a
    request is offered only when a full per-request timeout still fits.
    """
    if state.plan_sha256 != plan.sha256:
        raise RequestPlanRefused("resume state belongs to a different request plan")
    deadline = plan.record()["deadline"]
    remaining = deadline["deadline_seconds"] - (now - started_at)
    base = {"plan_sha256": plan.sha256, "remaining_seconds": max(0.0, remaining)}
    if state.complete:
        return {**base, "status": "complete", "requests": []}
    if cancelled:
        return {**base, "status": "cancelled", "requests": []}
    if remaining < deadline["request_timeout_seconds"]:
        return {**base, "status": "deadline_exceeded", "requests": []}
    if state.pending:
        return {
            **base,
            "status": "segments",
            "requests": [
                plan.segment(segment_id).record() for segment_id in state.pending
            ],
        }
    return {
        **base,
        "status": "synthesis",
        "requests": [
            {
                "mode": plan.mode,
                "plan_sha256": plan.sha256,
                "dependencies": list(state.accepted),
            }
        ],
    }

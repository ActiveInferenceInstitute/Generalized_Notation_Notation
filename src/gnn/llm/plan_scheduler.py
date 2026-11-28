"""S1 scheduler for admitted ``gnn_llm_request_plan_v1`` plans.

:mod:`gnn.llm.request_plan` admits immutable plans and revalidates retained
checkpoints but performs no I/O. This module drives one admitted plan through
an injected request executor:

* every window request carries the exact source bytes of one plan segment
  (``source[start_byte:end_byte]``), and its request ledger records where those
  bytes sit inside the encoded user message so byte-offset preservation can be
  checked from the retained files alone;
* each terminal segment reply becomes a deterministic per-window checkpoint
  (canonical JSON, one file per segment, no timestamps) written atomically;
* replies that are truncated, incomplete, lack native token counts or carry
  another identity are refused by ``terminal_stop_v1`` admission, recorded as
  refusals and never checkpointed; the window stays pending and nothing weaker
  is substituted in this invocation;
* resume rehydrates the retained plan, refuses it whole when any admitted input
  changed and revalidates every retained checkpoint before scheduling more
  work, so completed windows are never requested again;
* synthesis runs only after every window is accepted and is bound to the plan
  and every window checkpoint.

The executor is the only provider boundary. Offline tests inject it; the corpus
runner adapts :func:`gnn.llm.processor._execute_prompt`. Nothing here is live
provider evidence: a ``complete`` receipt states that the admitted plan's
checkpoints are terminal and current, not that a model attended to anything.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gnn.pipeline._io import atomic_write_text

from .request_plan import (
    COMPLETION_POLICY,
    ENCODING,
    TASK_CONTRACTS,
    RequestPlan,
    RequestPlanRefused,
    ResumeState,
    admit_request_plan,
    digest_bytes,
    digest_json,
    next_requests,
    reconstruct_source,
    revalidate_resume,
    segment_checkpoint,
    synthesis_checkpoint,
)

RUN_SCHEMA = "gnn_llm_plan_run_v1"
PROMPT_SCHEMA = "gnn_llm_plan_prompt_v1"
WINDOW_TEMPLATE = "plan_window_messages_v1"
SYNTHESIS_TEMPLATE = "plan_synthesis_messages_v1"
WINDOW_BEGIN = "GNN_SOURCE_WINDOW_BEGIN"
WINDOW_END = "GNN_SOURCE_WINDOW_END"
SOURCE_BEGIN = "EXACT_ORIGINAL_SOURCE_BEGIN"
SOURCE_END = "EXACT_ORIGINAL_SOURCE_END"
SETTINGS_KEYS = frozenset(
    {
        "mode",
        "segment_max_bytes",
        "reservations",
        "prompt",
        "deadline_seconds",
        "request_timeout_seconds",
    }
)

#: ``execute(messages, max_tokens, request_key)`` returns ``(text, outcome)``.
Executor = Callable[
    [list[dict[str, str]], int, str], Awaitable[tuple[str, Mapping[str, Any]]]
]


def prompt_definition(label: str, system_message: str, task: str) -> dict[str, Any]:
    """The prompt identity bound into a plan; templates are versioned names."""
    return {
        "schema": PROMPT_SCHEMA,
        "label": label,
        "system_message": system_message,
        "task": task,
        "window_template": WINDOW_TEMPLATE,
        "synthesis_template": SYNTHESIS_TEMPLATE,
    }


def _checked_prompt(plan: RequestPlan, prompt: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(prompt)
    if (
        value.get("schema") != PROMPT_SCHEMA
        or value.get("window_template") != WINDOW_TEMPLATE
        or value.get("synthesis_template") != SYNTHESIS_TEMPLATE
    ):
        raise RequestPlanRefused("prompt definition uses an unsupported template")
    if digest_json(value) != plan.record()["prompt_definition_sha256"]:
        raise RequestPlanRefused("prompt definition differs from the admitted plan")
    return value


def _encoded_offset(text: str) -> int:
    return len(text.encode(ENCODING))


def window_request(
    plan: RequestPlan, source: bytes, segment_id: str, prompt: Mapping[str, Any]
) -> dict[str, Any]:
    """Build one window request and its byte-preservation ledger.

    The user message embeds ``source[start_byte:end_byte]`` decoded strictly;
    ``content_range`` gives that slice's byte offsets inside the UTF-8
    encoded user message, so ``user.encode()[a:b]`` equals the source slice.
    """
    checked = _checked_prompt(plan, prompt)
    segment = plan.segment(segment_id)
    piece = source[segment.start_byte : segment.end_byte]
    if digest_bytes(piece) != segment.sha256:
        raise RequestPlanRefused(f"{segment_id} bytes differ from the admitted plan")
    text = piece.decode(ENCODING, errors="strict")
    total = len(plan.segments)
    index = plan.segments.index(segment) + 1
    length = plan.record()["source"]["byte_length"]
    scope = (
        f"This request contains window {index} of {total}: source bytes "
        f"[{segment.start_byte}, {segment.end_byte}) of {length}. Answer only "
        "from this window; report context outside it as missing."
    )
    head = (
        f"{checked['task']}\n\n{WINDOW_BEGIN} {segment_id} "
        f"bytes [{segment.start_byte}, {segment.end_byte})\n"
    )
    user = f"{head}{text}\n{WINDOW_END}"
    start = _encoded_offset(head)
    messages = [
        {"role": "system", "content": f"{checked['system_message']}\n\n{scope}"},
        {"role": "user", "content": user},
    ]
    return {
        "plan_sha256": plan.sha256,
        "segment_id": segment_id,
        "request_sha256": segment.request_sha256,
        "source_range": {
            "start_byte": segment.start_byte,
            "end_byte": segment.end_byte,
            "sha256": segment.sha256,
        },
        "content_range": {"start_byte": start, "end_byte": start + len(piece)},
        "messages": messages,
        "messages_sha256": digest_json(messages),
    }


def synthesis_request(
    plan: RequestPlan,
    source: bytes,
    accepted: Mapping[str, Mapping[str, Any]],
    prompt: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the synthesis request from ordered, accepted window answers.

    ``part_evidence`` sees only the window answers; ``joint_source`` (admitted
    only with a native full-source count) also receives the exact source.
    """
    checked = _checked_prompt(plan, prompt)
    record = plan.record()
    mode = record["task_contract"]["mode"]
    rows = []
    for segment in plan.segments:
        entry = accepted.get(segment.segment_id)
        if entry is None:
            raise RequestPlanRefused(
                f"synthesis dependency {segment.segment_id} is unfinished"
            )
        rows.append(
            f"[{segment.segment_id} bytes [{segment.start_byte}, "
            f"{segment.end_byte}) checkpoint {entry['checkpoint_sha256']}]\n"
            f"{entry['response']}"
        )
    contract = TASK_CONTRACTS[mode]
    system = (
        f"{checked['system_message']}\n\nSynthesis contract "
        f"{contract['contract']}: {contract['may_conclude']}"
    )
    head = f"{checked['task']}\n\nOrdered window answers:\n\n" + "\n\n".join(rows)
    content_range = None
    if record["task_contract"]["requires_full_source"]:
        head += f"\n\n{SOURCE_BEGIN}\n"
        start = _encoded_offset(head)
        user = head + source.decode(ENCODING, errors="strict") + f"\n{SOURCE_END}"
        content_range = {"start_byte": start, "end_byte": start + len(source)}
    else:
        user = head
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return {
        "plan_sha256": plan.sha256,
        "mode": mode,
        "dependencies": [segment.segment_id for segment in plan.segments],
        "content_range": content_range,
        "messages": messages,
        "messages_sha256": digest_json(messages),
    }


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return {"_unreadable": f"{path.name}: {exc}"}


@dataclass(frozen=True)
class PlanFiles:
    """Deterministic layout of one plan's retained records."""

    root: Path

    @property
    def plan(self) -> Path:
        return self.root / "plan.json"

    @property
    def segments(self) -> Path:
        return self.root / "segments"

    @property
    def synthesis(self) -> Path:
        return self.root / "synthesis.json"

    @property
    def synthesis_request(self) -> Path:
        return self.root / "synthesis.request.json"

    @property
    def receipt(self) -> Path:
        return self.root / "receipt.json"

    def segment(self, segment_id: str) -> Path:
        return self.segments / f"{segment_id}.json"

    def segment_request(self, segment_id: str) -> Path:
        return self.segments / f"{segment_id}.request.json"

    def checkpoints(self) -> list[Any]:
        if not self.segments.is_dir():
            return []
        return [
            _load(path)
            for path in sorted(self.segments.glob("segment_*.json"))
            if not path.name.endswith(".request.json")
        ]

    def reset(self) -> None:
        """Forget prior checkpoints of this plan directory (non-resume runs)."""
        if self.segments.is_dir():
            shutil.rmtree(self.segments)
        for path in (self.synthesis, self.synthesis_request, self.receipt):
            path.unlink(missing_ok=True)


def _receipt(
    plan: RequestPlan,
    state: ResumeState,
    *,
    status: str,
    resumed: tuple[str, ...],
    new_requests: list[str],
    refusals: list[dict[str, str]],
    synthesis: Mapping[str, Any] | None,
    rejected: tuple[tuple[str, str], ...] = (),
) -> dict[str, Any]:
    retained_rejections = list(dict.fromkeys((*rejected, *state.rejected)))
    return {
        "schema": RUN_SCHEMA,
        "plan_sha256": plan.sha256,
        "mode": plan.mode,
        "completion_policy": COMPLETION_POLICY,
        "status": status,
        "complete": state.complete,
        "full_source_claim": state.full_source_claim,
        "segments_required": len(plan.segments),
        "segments_accepted": len(state.accepted),
        "accepted": list(state.accepted),
        "pending": list(state.pending),
        "resumed_segments": list(resumed),
        "new_requests": new_requests,
        "rejected_checkpoints": [
            {"checkpoint": label, "reason": reason}
            for label, reason in retained_rejections
        ],
        "refusals": refusals,
        "synthesis": state.synthesis,
        "synthesis_reason": state.synthesis_reason,
        "synthesis_checkpoint_sha256": synthesis.get("checkpoint_sha256")
        if synthesis
        else None,
        "synthesis_response_sha256": synthesis.get("response_sha256")
        if synthesis
        else None,
        "provider_evidence": "executor_reported_not_independent",
        "attention_verified": False,
    }


async def run_request_plan(
    plan: RequestPlan,
    source: bytes,
    prompt: Mapping[str, Any],
    *,
    directory: Path,
    execute: Executor,
    resume: bool = False,
    clock: Callable[[], float] = time.monotonic,
    cancelled: Callable[[], bool] = lambda: False,
) -> dict[str, Any]:
    """Run or resume one admitted plan and return its deterministic receipt.

    The receipt is also written to ``directory/receipt.json``. A retained plan
    whose identity differs from ``plan`` is refused whole (``status``
    ``refused``) and its files are left untouched.
    """
    _checked_prompt(plan, prompt)
    reconstruct_source(plan, source)
    files = PlanFiles(Path(directory))
    synthesis: dict[str, Any] | None = None
    if resume and files.plan.exists():
        retained = _load(files.plan)
        retained_synthesis = (
            _load(files.synthesis) if files.synthesis.exists() else None
        )
        try:
            state, accepted = revalidate_resume(
                retained, plan, source, files.checkpoints(), retained_synthesis
            )
        except RequestPlanRefused as exc:
            empty = ResumeState(
                plan_sha256=plan.sha256,
                accepted=(),
                pending=tuple(segment.segment_id for segment in plan.segments),
                rejected=(),
                synthesis="blocked",
                synthesis_reason=str(exc),
                complete=False,
                full_source_claim=False,
            )
            return _receipt(
                plan,
                empty,
                status="refused",
                resumed=(),
                new_requests=[],
                refusals=[{"request": "plan", "reason": str(exc)}],
                synthesis=None,
            )
        if state.complete:
            synthesis = retained_synthesis
    else:
        files.reset()
        atomic_write_text(files.plan, _dump(plan.record()))
        state, accepted = revalidate_resume(plan.record(), plan, source)
    resumed = state.accepted
    rejected = state.rejected
    started = clock()
    deadline = plan.record()["deadline"]
    expires_at = started + deadline["deadline_seconds"]
    request_timeout = deadline["request_timeout_seconds"]
    max_tokens = plan.record()["context"]["reservations"]["output"]
    new_requests: list[str] = []
    refusals: list[dict[str, str]] = []
    refused: set[str] = set()
    status = "blocked"
    while True:
        scheduled_at = clock()
        schedule = next_requests(
            plan, state, started_at=started, now=scheduled_at, cancelled=cancelled()
        )
        if schedule["status"] in {"complete", "cancelled", "deadline_exceeded"}:
            status = schedule["status"]
            break
        if schedule["status"] == "segments":
            waiting = [
                r for r in schedule["requests"] if r["segment_id"] not in refused
            ]
            if not waiting:
                status = "refused"
                break
            label = waiting[0]["segment_id"]
            request = window_request(plan, source, label, prompt)
            request_path = files.segment_request(label)
        else:
            if "synthesis" in refused:
                status = "refused"
                break
            label = "synthesis"
            request = synthesis_request(plan, source, accepted, prompt)
            request_path = files.synthesis_request

        atomic_write_text(request_path, _dump(request))
        new_requests.append(label)
        timeout = asyncio.timeout(request_timeout)
        try:
            async with timeout:
                text, outcome = await execute(
                    request["messages"], max_tokens, f"{plan.sha256[:16]}:{label}"
                )
        except TimeoutError:
            timed_out = True
        else:
            timed_out = timeout.expired()

        # A terminal-looking reply may arrive after cancellation or its budget.
        # Decide before creating a checkpoint or updating accepted state.
        finished_at = clock()
        if cancelled():
            status = "cancelled"
            break
        if finished_at >= expires_at:
            status = "deadline_exceeded"
            break
        if timed_out or finished_at - scheduled_at > request_timeout:
            refused.add(label)
            refusals.append({"request": label, "reason": "request timeout exceeded"})
            continue

        try:
            if label == "synthesis":
                synthesis = synthesis_checkpoint(plan, accepted, text, outcome)
                atomic_write_text(files.synthesis, _dump(synthesis))
            else:
                checkpoint = segment_checkpoint(plan, label, text, outcome)
                atomic_write_text(files.segment(label), _dump(checkpoint))
                accepted[label] = checkpoint
        except RequestPlanRefused as exc:
            refused.add(label)
            refusals.append({"request": label, "reason": str(exc)})
            continue
        state, accepted = revalidate_resume(
            plan.record(), plan, source, accepted.values(), synthesis
        )
    receipt = _receipt(
        plan,
        state,
        status=status,
        resumed=resumed,
        new_requests=new_requests,
        refusals=refusals,
        synthesis=synthesis,
        rejected=rejected,
    )
    atomic_write_text(files.receipt, _dump(receipt))
    return receipt


def checked_settings(value: Any) -> dict[str, Any] | None:
    """Normalize the opt-in ``request_plan`` corpus option, or ``None`` (off).

    ``None``/``False`` keep the established scheduler unchanged. A mapping
    must name ``mode``, ``segment_max_bytes`` and all four ``reservations``;
    ``prompt`` defaults to the summary prompt, and the deadline/timeout default
    to the corpus budget and prompt ceiling at admission time.
    """
    if value is None or value is False:
        return None
    if not isinstance(value, Mapping) or not set(value) <= SETTINGS_KEYS:
        raise ValueError(
            "request_plan must be null or a mapping of "
            + ", ".join(sorted(SETTINGS_KEYS))
        )
    mode = value.get("mode", "part_evidence")
    if mode not in TASK_CONTRACTS:
        raise ValueError("request_plan mode must be part_evidence or joint_source")
    size = value.get("segment_max_bytes")
    if type(size) is not int or size < 4:
        raise ValueError("request_plan segment_max_bytes must be an integer >= 4")
    reservations = value.get("reservations")
    if not isinstance(reservations, Mapping):
        raise ValueError("request_plan reservations must be a mapping")
    settings: dict[str, Any] = {
        "mode": mode,
        "segment_max_bytes": size,
        "reservations": {str(k): v for k, v in sorted(reservations.items())},
        "prompt": value.get("prompt"),
    }
    for key in ("deadline_seconds", "request_timeout_seconds"):
        if value.get(key) is not None:
            settings[key] = value[key]
    return settings


def admit_corpus_plan(
    source: bytes,
    settings: Mapping[str, Any],
    prompt: Mapping[str, Any],
    *,
    provider: str,
    endpoint_sha256: str,
    model: str,
    runtime_identity: Mapping[str, Any],
    deadline_seconds: float,
    request_timeout_seconds: float,
) -> RequestPlan:
    """Admit a corpus plan from the verified runtime identity, or refuse.

    Capacity comes only from the preflight's reviewed ``context_capability``
    and the model digest from that same identity; neither is configurable.
    No native full-source tokenizer count is available on this path, so
    ``joint_source`` refuses at admission instead of guessing.
    """
    capability = runtime_identity.get("context_capability")
    digest = runtime_identity.get("model_digest")
    if not isinstance(capability, Mapping) or not isinstance(digest, str):
        raise RequestPlanRefused(
            "request plan requires a verified runtime identity with reviewed "
            "context capability"
        )
    return admit_request_plan(
        source,
        provider=provider,
        endpoint_sha256=endpoint_sha256,
        model=model,
        model_digest=digest,
        prompt_definition=prompt,
        capacity=capability,
        reservations=settings["reservations"],
        mode=settings["mode"],
        segment_max_bytes=settings["segment_max_bytes"],
        deadline_seconds=settings.get("deadline_seconds", deadline_seconds),
        request_timeout_seconds=settings.get(
            "request_timeout_seconds", request_timeout_seconds
        ),
    )


def plan_outcome(
    outcome: Mapping[str, Any], *, provider: str, endpoint_sha256: str
) -> dict[str, Any]:
    """Map a corpus prompt outcome onto the plan's terminal-stop record.

    Token counts and the model digest are read only from the native context
    admission receipt; when it is absent the counts stay missing and
    ``terminal_stop_v1`` admission refuses the reply. ``finish_reason`` and
    ``status`` are copied, never inferred.
    """
    admission = outcome.get("context_admission")
    admission = admission if isinstance(admission, Mapping) else {}
    return {
        "status": outcome.get("status"),
        "finish_reason": outcome.get("finish_reason"),
        "completion_policy": COMPLETION_POLICY,
        "provider": outcome.get("provider") or provider,
        "model": outcome.get("model"),
        "model_digest": admission.get("model_digest"),
        "endpoint_sha256": endpoint_sha256,
        "prompt_tokens": admission.get("prompt_tokens"),
        "completion_tokens": admission.get("completion_tokens"),
        "request_sha256": outcome.get("request_sha256"),
    }


async def run_corpus_plans(
    api: Any,
    settings: Mapping[str, Any],
    tasks: list[tuple[str, Any, str | None]],
    records: list[tuple[Path, dict[str, Any], str, dict[str, str]]],
    *,
    output_dir: Path,
    processor: Any,
    cache: Any,
    provider: Any,
    endpoint_url: str | None,
    endpoint_sha256: str,
    model: str,
    runtime_identity: Mapping[str, Any],
    generation_options: Mapping[str, Any],
    isolated: bool,
    failed_auth: set[str],
    auth_errors: list[dict[str, Any]],
    resume: bool,
    deadline_seconds: float,
    request_timeout_seconds: float,
    remaining: Callable[[], float],
) -> list[dict[str, Any]]:
    """Admit and run one request plan per selected source (``process_llm`` S1).

    ``processor`` is ``None`` when the provider preflight did not succeed; the
    plan is then admitted (when possible) but no request is issued and every
    window stays pending. Source bytes are re-read and must match the frozen
    selection digest.
    """
    label = settings["prompt"]
    _, prompt_type, custom_text = next(task for task in tasks if task[0] == label)
    if prompt_type is not None:
        definition = api.get_prompt(
            prompt_type, "[The source window or ordered window answers follow.]"
        )
        system_message, task = definition["system_message"], definition["user_prompt"]
    else:
        system_message = "You are an expert in Active Inference and GNN."
        task = str(custom_text)
    prompt = prompt_definition(label, system_message, task)
    receipts: list[dict[str, Any]] = []
    for path, _analysis, content, identity in records:
        source = Path(path).read_bytes()
        directory = output_dir / "request_plans" / identity["artifact_stem"] / label
        base = {
            "model_id": identity["model_id"],
            "source_path": identity["source_path"],
            "prompt": label,
            "directory": directory.relative_to(output_dir).as_posix(),
        }
        if digest_bytes(source) != identity["source_sha256"]:
            receipts.append(
                {
                    **base,
                    "status": "refused",
                    "complete": False,
                    "refusals": [
                        {"request": "plan", "reason": "source changed since selection"}
                    ],
                }
            )
            continue
        try:
            plan = admit_corpus_plan(
                source,
                settings,
                prompt,
                provider=provider.value,
                endpoint_sha256=endpoint_sha256,
                model=model,
                runtime_identity=runtime_identity,
                deadline_seconds=deadline_seconds,
                request_timeout_seconds=request_timeout_seconds,
            )
        except RequestPlanRefused as exc:
            receipts.append(
                {
                    **base,
                    "status": "refused",
                    "complete": False,
                    "refusals": [{"request": "plan", "reason": str(exc)}],
                }
            )
            continue

        async def execute(
            messages: list[dict[str, str]],
            max_tokens: int,
            key: str,
            *,
            cache_content: str = content,
            plan_sha256: str = plan.sha256,
            plan_timeout: float = plan.record()["deadline"]["request_timeout_seconds"],
        ) -> tuple[str, Mapping[str, Any]]:
            outcomes: dict[str, Any] = {}
            text = await api._execute_prompt(
                processor,
                cache,
                cache_content=cache_content,
                model_name=model,
                messages=[api.LLMMessage(**message) for message in messages],
                prompt_text=messages[-1]["content"],
                label=key,
                custom=prompt_type is None,
                max_tokens=max_tokens,
                max_prompt_timeout=min(
                    plan_timeout, request_timeout_seconds, remaining()
                ),
                failed_auth_providers=failed_auth,
                auth_errors=auth_errors,
                outcomes=outcomes,
                provider_type=provider,
                isolated=isolated,
                endpoint_url=endpoint_url,
                generation_options=dict(generation_options),
                runtime_identity=dict(runtime_identity),
                remaining_budget=remaining,
                context_admission=True,
                synthesis_plan_sha256=plan_sha256,
            )
            return text, plan_outcome(
                outcomes.get(key, {}),
                provider=provider.value,
                endpoint_sha256=endpoint_sha256,
            )

        receipt = await run_request_plan(
            plan,
            source,
            prompt,
            directory=directory,
            execute=execute,
            resume=resume,
            cancelled=lambda: processor is None or remaining() <= 0,
        )
        if processor is None and receipt["status"] == "cancelled":
            receipt["status"] = "provider_unavailable"
        receipts.append({**base, **receipt})
    return receipts

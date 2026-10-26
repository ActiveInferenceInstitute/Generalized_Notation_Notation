"""Small checkpoint-aware reducer delegation for corpus scheduling."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from .cross_part_synthesis import SynthesisPlan, completed_entry, resumed_entry


async def run_prompt_synthesis(
    *,
    plan: SynthesisPlan,
    key: str,
    state: dict[str, Any],
    execute: Callable[[SynthesisPlan, str], Awaitable[tuple[str, dict[str, Any]]]],
    validate: Callable[[SynthesisPlan, dict[str, Any]], None],
    max_chars: int | None,
    save: Callable[[], None],
) -> tuple[str | None, str, dict[str, Any]]:
    """Reuse only an exact valid reducer; persist a newly completed node once."""
    entries = state.setdefault("completed_syntheses", {})
    prior = entries.get(key)
    if prior is not None:
        checked = resumed_entry(plan, prior, max_chars)
        validate(plan, checked["outcome"])
        return (
            checked["response"],
            checked["native_response"],
            {
                **checked["outcome"],
                "resumed": True,
                "plan_sha256": plan.sha256,
                "synthesis_mode": plan.mode,
            },
        )
    native_text, outcome = await execute(plan, key)
    outcome = {**outcome, "plan_sha256": plan.sha256, "synthesis_mode": plan.mode}
    if outcome.get("status") != "success":
        return None, native_text, outcome
    validate(plan, outcome)
    entry = completed_entry(plan, native_text, outcome, max_chars)
    entries[key] = entry
    save()
    return entry["response"], native_text, outcome

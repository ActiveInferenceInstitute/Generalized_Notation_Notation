"""Side-effect-free admission shared by public pipeline execution surfaces."""

from __future__ import annotations

from typing import Any

from gnn.pipeline.step_registry import STEPS

STEP_NUMBERS = frozenset(int(step.script_stem.partition("_")[0]) for step in STEPS)
_ALIASES = {
    alias: int(step.script_stem.partition("_")[0])
    for step in STEPS
    for alias in (
        step.script_stem.partition("_")[0],
        step.script_stem,
        step.script_name,
        step.script_stem.partition("_")[2],
    )
}
# Historical Python discovery alias retained for execution compatibility.
_ALIASES["advanced_visualization"] = _ALIASES["advanced_viz"]


def validate_steps(
    value: Any,
    *,
    field_name: str,
    aliases: bool = False,
    allow_empty: bool = True,
) -> list[int] | None:
    """Admit unique registered steps; omitted selection differs from empty.

    JSON surfaces accept actual integers only. Python and textual CLI adapters
    can opt into the registered number/name aliases. No invalid token is dropped.
    """
    if value is None:
        return None
    if aliases and isinstance(value, (str, int)) and not isinstance(value, bool):
        value = (
            (value.split(",") if value.strip() else [])
            if isinstance(value, str)
            else [value]
        )
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of integers")
    result: list[int] = []
    for token in value:
        number = None
        if isinstance(token, int) and not isinstance(token, bool):
            number = token
        elif aliases and isinstance(token, str):
            number = _ALIASES.get(token.strip())
        if number not in STEP_NUMBERS:
            raise ValueError(
                f"Invalid step number or name in {field_name}: {token!r}; "
                f"expected integers between 0 and {max(STEP_NUMBERS)}"
            )
        if number in result:
            raise ValueError(f"{field_name} must not contain duplicate step numbers")
        result.append(number)
    if not result and not allow_empty:
        raise ValueError(
            f"{field_name} resolved to no executable steps; omit it to run all"
        )
    return sorted(result)


def validate_boolean(value: Any, *, field_name: str) -> bool:
    """Reject implicit coercion of strings, numbers and containers into flags."""
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value

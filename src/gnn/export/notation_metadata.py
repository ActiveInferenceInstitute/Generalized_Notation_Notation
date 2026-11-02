"""Notation-derived GEO-INFER metadata with structured provenance.

The GEO-INFER interchange contract originally required every metadata field
(step seconds, units, space kind) to arrive through explicit caller options.
GNN-05 adds an opt-in derivation fallback: values may be read from explicit
declarations inside the GNN source itself -- never guessed and never silently
defaulted -- and every accepted value ships a structured derivation record
that the exporter merges into the artifact's ``provenance``.

Derivation sources:

- ``## Time`` entries ``TimeStep=60s`` / ``StepSeconds=60`` / ``dt=0.5``.
  A bare number is read as seconds and the interpretation is recorded; unit
  suffixes (``s``, ``ms``, ``min``, ``h``, ``days``) and a ``TimeUnits=``
  companion key are accepted.
- ``## ModelParameters`` entry ``dt: 0.1`` (bare number read as seconds,
  interpretation recorded).
- ``SpaceKind=categorical|h3`` inside ``## Time`` (optional; the categorical
  writer default is recorded as non-derived otherwise).
- Time-index corroboration recorded in provenance: the variable named by
  ``Time=<var>`` / ``DiscreteTime=<var>``, its ``StateSpaceBlock`` declaration
  and its ``ActInfOntologyAnnotation`` term.

Derivation never invents per-coordinate ``units`` (the notation declares
none) and never fabricates H3 state IDs. Contradictory or unparseable
declarations fail visibly instead of falling back to a default.
"""

from __future__ import annotations

import math
import re
from typing import Any, Final

from gnn.parsers.common import canonical_section_heading

__all__ = ["NotationMetadataError", "derive_geo_metadata"]


class NotationMetadataError(ValueError):
    """Raised when the notation carries no derivable metadata; no defaults."""


_TIME_UNIT_TO_SECONDS: Final[dict[str, float]] = {
    "": 1.0,
    "s": 1.0,
    "sec": 1.0,
    "secs": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "ms": 0.001,
    "millisecond": 0.001,
    "milliseconds": 0.001,
    "min": 60.0,
    "mins": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hrs": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
    "d": 86400.0,
    "day": 86400.0,
    "days": 86400.0,
}

_STEP_KEYS: Final[tuple[str, ...]] = ("timestep", "stepseconds", "dt")
_TIME_UNIT_KEYS: Final[tuple[str, ...]] = ("timeunits", "timeunit")
_SPACE_KIND_KEY: Final[str] = "spacekind"
_TIME_INDEX_KEYS: Final[tuple[str, ...]] = ("time", "discretetime")
_SPACE_KINDS: Final[frozenset[str]] = frozenset({"categorical", "h3"})

_NO_STEP_MESSAGE: Final[str] = (
    "GEO-INFER metadata derivation failed: no time-step declaration found. "
    "Searched the ## Time section for TimeStep/StepSeconds/dt (bare numbers "
    "are read as seconds; unit suffixes and a TimeUnits companion key are "
    "accepted) and ## ModelParameters for dt. Add e.g. 'TimeStep=60s' to the "
    "## Time section or supply explicit --geo-step-seconds / geo_infer options."
)

_NUMBER_UNIT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<number>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*"
    r"(?P<unit>[A-Za-z]*)$"
)


def _section_bodies(content: str, name: str) -> list[str]:
    """Return the bodies of every ``## <name>`` section in the source.

    A heading that names the section with different whitespace (for example
    ``## Model Parameters`` for ``## ModelParameters``) is the same section.
    """
    headings = list(
        re.compile(r"^##[ \t]+(?P<heading>.+?)[ \t]*$", re.M).finditer(content)
    )
    bodies: list[str] = []
    for index, match in enumerate(headings):
        if canonical_section_heading(match.group("heading")) != name:
            continue
        end = headings[index + 1].start() if index + 1 < len(headings) else len(content)
        body = content[match.end() : end]
        bodies.append(body[1:] if body.startswith("\n") else body)
    return bodies


def _time_entries(body: str) -> list[tuple[str, str | None, str]]:
    """Return ``(lowercased key, raw value or None, cleaned line)`` entries.

    Bare marker lines (``Dynamic``, ``Discrete``) carry ``None`` values;
    ``key=value`` lines are split at the first ``=`` after comment stripping.
    """
    entries: list[tuple[str, str | None, str]] = []
    for line in body.splitlines():
        cleaned = line.split("#", 1)[0].strip()
        if not cleaned:
            continue
        if "=" in cleaned:
            key, _, value = cleaned.partition("=")
            entries.append((key.strip().lower(), value.strip(), cleaned))
        else:
            entries.append((cleaned.lower(), None, cleaned))
    return entries


def _model_parameter_entries(body: str) -> list[tuple[str, str, str]]:
    """Return ``(lowercased key, raw value, cleaned line)`` YAML-style entries."""
    entries: list[tuple[str, str, str]] = []
    for line in body.splitlines():
        cleaned = line.split("#", 1)[0].strip()
        if not cleaned or ":" not in cleaned:
            continue
        key, _, value = cleaned.partition(":")
        entries.append((key.strip().lower(), value.strip(), cleaned))
    return entries


def _seconds_from_value(
    raw: str, companion_unit: str | None, context: str
) -> tuple[float, str]:
    """Convert a declared value to ``(seconds, interpretation)``.

    Raises:
        NotationMetadataError: The value is not a positive finite duration or
            uses a unit that is neither a supported suffix nor the companion
            ``TimeUnits`` key.
    """
    match = _NUMBER_UNIT_PATTERN.fullmatch(raw)
    if match is None:
        raise NotationMetadataError(
            f"{context}: cannot read time step value {raw!r}; expected a "
            "positive number with an optional unit suffix"
        )
    number = float(match.group("number"))
    unit = match.group("unit").lower()
    if unit and unit not in _TIME_UNIT_TO_SECONDS:
        supported = sorted(name for name in _TIME_UNIT_TO_SECONDS if name)
        raise NotationMetadataError(
            f"{context}: unsupported time unit {unit!r}; accepted units: {supported}"
        )
    factor = _TIME_UNIT_TO_SECONDS.get(unit, 1.0)
    if unit:
        interpretation = f"{number:g} {unit} = {number * factor:g} seconds"
    elif companion_unit is not None:
        companion = companion_unit.lower()
        companion_factor = _TIME_UNIT_TO_SECONDS.get(companion)
        if not companion_factor:
            supported = sorted(name for name in _TIME_UNIT_TO_SECONDS if name)
            raise NotationMetadataError(
                f"{context}: unsupported TimeUnits value {companion_unit!r}; "
                f"accepted units: {supported}"
            )
        factor = companion_factor
        interpretation = (
            f"bare {number:g} read as {companion} = {number * factor:g} seconds"
        )
    else:
        interpretation = f"bare {number:g} interpreted as {number:g} seconds"
    seconds = number * factor
    if not math.isfinite(seconds) or seconds <= 0:
        raise NotationMetadataError(
            f"{context}: time step must be finite and positive, got {raw!r}"
        )
    return seconds, interpretation


def _close(a: float, b: float) -> bool:
    """Compare two derived durations, tolerating float representation dust."""
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-15)


def _single_companion_unit(entries: list[tuple[str, str | None, str]]) -> str | None:
    """Return the sole ``TimeUnits`` value, rejecting conflicting repeats."""
    values = [value for key, value, _ in entries if key in _TIME_UNIT_KEYS and value]
    distinct = {value.lower() for value in values}
    if len(distinct) > 1:
        raise NotationMetadataError(
            "conflicting TimeUnits declarations in ## Time: "
            + ", ".join(sorted(distinct))
        )
    return values[0] if values else None


def _declared_space_kind(
    entries: list[tuple[str, str | None, str]], options: dict[str, Any]
) -> dict[str, Any]:
    """Resolve ``space_kind`` from an explicit ``SpaceKind`` declaration.

    Only a declared value enters ``options``; the categorical writer default
    is recorded as non-derived so the merged mapping never carries a space
    option the linear_gaussian writer would reject.
    """
    values = [value for key, value, _ in entries if key == _SPACE_KIND_KEY and value]
    distinct = {value.lower() for value in values}
    if len(distinct) > 1:
        raise NotationMetadataError(
            "conflicting SpaceKind declarations in ## Time: "
            + ", ".join(sorted(distinct))
        )
    if values:
        kind = values[0].lower()
        if kind not in _SPACE_KINDS:
            raise NotationMetadataError(
                f"## Time SpaceKind={values[0]!r} is unsupported; expected "
                "categorical or h3"
            )
        options["space_kind"] = kind
        record: dict[str, Any] = {
            "value": kind,
            "derived": True,
            "source": "## Time SpaceKind",
            "raw_value": values[0],
        }
        if kind == "h3":
            record["note"] = (
                "state_ids must still be supplied explicitly; derivation "
                "never fabricates H3 cell IDs"
            )
        return record
    return {"value": "categorical", "derived": False, "source": "writer default"}


def _time_index_record(
    content: str, entries: list[tuple[str, str | None, str]]
) -> dict[str, Any]:
    """Corroborate the time-index variable across notation sections."""
    variables = {
        value.strip() for key, value, _ in entries if key in _TIME_INDEX_KEYS and value
    }
    if len(variables) > 1:
        raise NotationMetadataError(
            "contradictory time-index declarations in ## Time: "
            + ", ".join(sorted(variables))
        )
    variable = next(iter(variables)) if variables else None
    record: dict[str, Any] = {
        "variable": variable,
        "state_space_declaration": None,
        "ontology_term": None,
    }
    if variable is None:
        return record
    prefix = f"{variable}["
    for body in _section_bodies(content, "StateSpaceBlock"):
        for line in body.splitlines():
            cleaned = line.split("#", 1)[0].strip()
            if cleaned.startswith(prefix):
                record["state_space_declaration"] = cleaned
                break
        if record["state_space_declaration"] is not None:
            break
    for body in _section_bodies(content, "ActInfOntologyAnnotation"):
        for line in body.splitlines():
            cleaned = line.split("###", 1)[0].strip()
            if "=" not in cleaned:
                continue
            name, _, term = cleaned.partition("=")
            if name.strip() == variable:
                record["ontology_term"] = term.strip()
                break
        if record["ontology_term"] is not None:
            break
    return record


def _step_candidates(
    content: str, entries: list[tuple[str, str | None, str]]
) -> list[tuple[str, str, str, float, str]]:
    """Collect every explicit step declaration with its parsed duration.

    Returns ``(section, key, raw_value, seconds, interpretation)`` tuples from
    both the ``## Time`` entries and the ``## ModelParameters`` entries so
    that contradictory cross-section declarations can be rejected upstream.
    """
    companion_unit = _single_companion_unit(entries)
    candidates: list[tuple[str, str, str, float, str]] = []
    for key, value, _ in entries:
        if key in _STEP_KEYS and value:
            context = f"## Time {key}={value!r}"
            seconds, interpretation = _seconds_from_value(
                value, companion_unit, context
            )
            candidates.append(("Time", key, value, seconds, interpretation))
    for body in _section_bodies(content, "ModelParameters"):
        for key, value, _ in _model_parameter_entries(body):
            if key in _STEP_KEYS and value:
                context = f"## ModelParameters {key}: {value!r}"
                seconds, interpretation = _seconds_from_value(value, None, context)
                candidates.append(
                    ("ModelParameters", key, value, seconds, interpretation)
                )
    return candidates


def derive_geo_metadata(content: str) -> dict[str, Any]:
    """Derive ``geo_infer`` export options from explicit notation declarations.

    Args:
        content: GNN Markdown source text.
    Returns:
        ``{"options": {...}, "metadata_derivation": {...}}`` where ``options``
        holds writer keyword arguments (``step_seconds``, ``space_kind``) and
        ``metadata_derivation`` is the JSON-safe provenance record for the
        artifact's ``provenance`` object.
    Raises:
        NotationMetadataError: The notation declares no usable time step, is
            contradictory, or declares continuous time; no defaults are
            substituted.
    """
    time_bodies = _section_bodies(content, "Time")
    if len(time_bodies) != 1:
        raise NotationMetadataError(
            "metadata derivation requires exactly one ## Time section; "
            f"found {len(time_bodies)}"
        )
    entries = _time_entries(time_bodies[0])

    tokens = " ".join(f"{key} {value or ''}" for key, value, _ in entries)
    has_continuous = "continuous" in tokens
    has_discrete = "discrete" in tokens
    if has_continuous and has_discrete:
        raise NotationMetadataError(
            "## Time declares both Discrete and Continuous; resolve the "
            "contradiction before export"
        )
    if has_continuous:
        raise NotationMetadataError(
            "notation declares Continuous time; notation-derived step_seconds "
            "requires an explicit Discrete declaration"
        )

    candidates = _step_candidates(content, entries)
    if not candidates:
        raise NotationMetadataError(_NO_STEP_MESSAGE)
    distinct: list[float] = []
    for _, _, _, seconds, _ in candidates:
        if not any(_close(seconds, seen) for seen in distinct):
            distinct.append(seconds)
    if len(distinct) > 1:
        raise NotationMetadataError(
            "ambiguous time-step declarations: "
            + ", ".join(
                f"## {section} {key}={raw!r} ({seconds} seconds)"
                for section, key, raw, seconds, _ in candidates
            )
        )
    step_seconds = candidates[0][3]

    options: dict[str, Any] = {"step_seconds": step_seconds}
    record: dict[str, Any] = {
        "step_seconds": {
            "value": step_seconds,
            "derived": True,
            "sources": [
                {
                    "section": section,
                    "key": key,
                    "raw_value": raw,
                    "interpretation": interpretation,
                }
                for section, key, raw, _, interpretation in candidates
            ],
        },
        "space_kind": _declared_space_kind(entries, options),
        "units": {
            "derived": False,
            "reason": (
                "GNN notation declares no per-coordinate units; supply units "
                "via explicit geo_infer options for linear_gaussian exports"
            ),
        },
        "time_index": _time_index_record(content, entries),
        "discrete_time_declared": has_discrete,
    }
    return {"options": options, "metadata_derivation": record}

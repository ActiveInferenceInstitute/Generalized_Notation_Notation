"""Printed marginal parsing and the daf-jev posterior sidecar contract."""

from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Final, Mapping

from .graphspec import (
    _KEY_RE,
    _ROW_SUM_TOLERANCE,
)

logger = logging.getLogger(__name__)


MARGINALS_FORMAT: Final[str] = "dafjev.bayesnet-posteriors/1"


_MARGINAL_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"^  (?P<key>[A-Za-z_][A-Za-z0-9_]*):\s+(?P<body>.+)$"
)


_MARGINAL_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<state>.+?)=(?P<num>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)


_MARGINAL_ROUND_ERROR: Final[float] = 5e-7


def parse_marginals(text: str) -> dict[str, dict[str, float]]:
    """Parse the marginal block an emitted RxInfer.jl script prints to stdout.

    Parses exactly the ``_run_inference`` println block of
    ``emit_rxinfer_jl`` — lines of the shape ``  <key>: <state>=<p>  ...``
    where ``p`` is Julia's ``round(probs[i]; digits=6)`` text (non-negative
    decimal/exponent literals such as ``0.692308``, ``1.0``, ``1.0e-7``).
    Everything else is skipped: the ``Posteriors (marginal P(key)):`` and
    ``Observed evidence:`` headers, evidence lines (``  <key> = <state>``),
    learning-path lines, and surrounding output. A line in the marginal
    shape whose tokens do not parse raises ``ValueError`` naming the
    offending line (fail-closed), as do duplicate keys and duplicate
    states within a line (the emitter prints each variable and state
    once, so a duplicate means mixed output); a literal overflowing to a
    non-finite float (``1e999``) is rejected the same way — the module
    rejects non-finite probabilities everywhere. Text without marginal
    lines parses to ``{}``.

    Printed keys are the sanitized Julia identifiers (``RAW_TO_IDENT``;
    identical to the GraphSpec key unless sanitization renamed it) and the
    pairs are split on the emitter's exact two-space separator, so state
    labels must not contain a run of two spaces.
    """
    marginals: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        match = _MARGINAL_LINE_RE.match(line)
        if match is None:
            continue
        key, body = match.group("key"), match.group("body")
        if key in marginals:
            raise ValueError(
                f"parse_marginals: duplicate marginal key {key!r} on line {line!r}"
            )
        parsed: dict[str, float] = {}
        for token in body.split("  "):
            token_match = _MARGINAL_TOKEN_RE.fullmatch(token)
            if token_match is None:
                raise ValueError(
                    f"parse_marginals: unparsable marginal token {token!r} on "
                    f"line {line!r} (expected '<state>=<probability>')"
                )
            state = token_match.group("state")
            if state in parsed:
                raise ValueError(
                    f"parse_marginals: duplicate state {state!r} on line {line!r}"
                )
            num = float(token_match.group("num"))
            if not math.isfinite(num):
                raise ValueError(
                    f"parse_marginals: marginal {state!r} on line {line!r} "
                    f"is not finite: {token!r}"
                )
            parsed[state] = num
        marginals[key] = parsed
    return marginals


def write_marginals(
    marginals: Mapping[str, Mapping[str, float]],
    path: str | os.PathLike[str],
    *,
    source_model: str | None = None,
) -> Path:
    """Serialize parsed marginals as a ``gnn.marginals/1`` JSON sidecar.

    Document shape (JSON keys and order):

        {"format": "dafjev.bayesnet-posteriors/1",
         "marginals": {"<key>": {"<state>": <p>, ...}, ...},
         "source_model": <name or null>}

    ``source_model`` records the emitting model's name (the
    ``emit_rxinfer_jl`` ``model_name``, e.g. ``asia_model``) when the
    caller knows it; omitted means ``null``. Validation is fail-closed and
    mirrors the GraphSpec probability rules (finite, >= 0, each row
    summing to 1.0) with a rounding-aware row-sum budget
    (``_ROW_SUM_TOLERANCE + n_states * _MARGINAL_ROUND_ERROR``) that
    absorbs the emitter's 6-digit rounding without false-rejecting
    legitimate parsed rows. Insertion order (the printed topological
    order) is preserved in the JSON. Returns the written ``Path``.
    """
    if not isinstance(marginals, Mapping) or not marginals:
        raise ValueError(
            "write_marginals: marginals must be a non-empty mapping of "
            f"variable key -> {{state: probability}}, got {marginals!r}"
        )
    doc_marginals: dict[str, dict[str, float]] = {}
    for key, states in marginals.items():
        if not isinstance(key, str) or not _KEY_RE.match(key):
            raise ValueError(
                f"write_marginals: invalid marginal key {key!r} "
                "(must match [A-Za-z_][A-Za-z0-9_-]*)"
            )
        if not isinstance(states, Mapping) or not states:
            raise ValueError(
                f"write_marginals: marginal {key!r} must be a non-empty "
                f"mapping of state -> probability, got {states!r}"
            )
        probs: dict[str, float] = {}
        for state, p in states.items():
            if not isinstance(state, str) or not state:
                raise ValueError(
                    f"write_marginals: marginal {key!r} state must be a "
                    f"non-empty string, got {state!r}"
                )
            if isinstance(p, bool) or not isinstance(p, (int, float)):
                raise ValueError(
                    f"write_marginals: marginal {key!r} state {state!r} "
                    f"must map to a number, got {p!r}"
                )
            if not math.isfinite(float(p)):
                raise ValueError(
                    f"write_marginals: marginal {key!r} state {state!r} is "
                    f"not finite: {p!r}"
                )
            if p < 0:
                raise ValueError(
                    f"write_marginals: marginal {key!r} state {state!r} is "
                    f"negative: {p!r}"
                )
            probs[state] = float(p)
        total = sum(probs.values())
        tolerance = _ROW_SUM_TOLERANCE + len(probs) * _MARGINAL_ROUND_ERROR
        if abs(total - 1.0) > tolerance:
            raise ValueError(
                f"write_marginals: marginal {key!r} probabilities sum to "
                f"{total!r}, expected 1.0 within {tolerance!r}"
            )
        doc_marginals[key] = probs
    if source_model is not None and (
        not isinstance(source_model, str) or not source_model
    ):
        raise ValueError(
            f"write_marginals: source_model must be a non-empty string or "
            f"None, got {source_model!r}"
        )
    doc: dict[str, Any] = {
        "format": MARGINALS_FORMAT,
        "marginals": doc_marginals,
        "source_model": source_model,
    }
    out_path = Path(path)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")
    return out_path

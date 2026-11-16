"""Headless POMDP extraction entry point (stdlib-only, versioned contract).

This module is the machine-readable, headless entry point for extracting a
POMDP state space from a GNN specification file. It deliberately imports only
the standard library at module scope; the extractor module
(``gnn.extract.pomdp_extractor`` — itself stdlib-only at module scope) is imported
lazily inside the call path, so ``python -m gnn.extract`` works without the
full pipeline stack and without heavy module-scope dependencies.

Contract (extraction envelope schema version 1.0.0)
====================================================

``extract_to_json(path, *, strict_validation=True, on_error="lenient",
compact=False, strict_probabilities=False) -> str`` always returns a JSON
string and never raises.

- **Success** (extraction produced a ``POMDPStateSpace``): the JSON object is
  ``POMDPStateSpace.to_dict()`` (no ``status`` key). ``compact=True`` emits
  ``separators=(",", ":")`` and no indentation; the default emits
  ``indent=2``.
- **Failure** (the extractor raises a structured error — e.g. under
  ``on_error="raise"`` — or returns no result): the JSON object is the error
  envelope::

      {"status": "error",
       "error": {"code": ..., "message": ..., "line": ..., "section": ...}}

  ``code``/``message``/``line``/``section`` come from the extractor's
  structured error attributes when present (e.g. ``GNN-E002`` shape
  contradictions, ``GNN-E006`` parameter-parse failures, ``GNN-E007``
  opt-in strict probability violations); a bare failure with
  no structured error uses code ``"GNN-E000"``. Parameter-parse failures are
  never silently dropped by the extractor: they are recorded in
  ``matrix_provenance`` in every ``on_error`` mode.

``main(argv=None) -> int`` wraps ``extract_to_json`` as a CLI:

- ``python -m gnn.extract FILE [--strict|--no-strict] [--strict-probabilities]
  [--compact]`` (``--strict-probabilities`` also selects ``on_error="raise"``,
  so the first probability/dimension violation becomes the error envelope)
- success: prints the payload JSON, exit code 0
- failure: prints the error envelope JSON, exit code 1

Exit codes follow the repo convention (0 = success, 1 = error).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

__all__ = ["extract_to_json", "main"]

_ENVELOPE_SCHEMA_VERSION = "1.0.0"
_GENERIC_EXTRACTION_ERROR_CODE = "GNN-E000"


def _dumps(payload: Any, *, compact: bool) -> str:
    """Serialize ``payload`` honoring the compact flag (shared JSON style)."""
    if compact:
        return json.dumps(payload, separators=(",", ":"))
    return json.dumps(payload, indent=2)


def _error_envelope(exc: BaseException) -> dict[str, Any]:
    """Build the failure envelope from a structured extractor error.

    Structured extractor errors carry ``code``/``message`` (and optionally
    ``line``/``section``); unexpected failures fall back to the generic
    ``GNN-E000`` code with the exception text as the message.
    """
    code = getattr(exc, "code", None) or _GENERIC_EXTRACTION_ERROR_CODE
    message = getattr(exc, "message", None) or str(exc)
    error: dict[str, Any] = {
        "code": code,
        "message": message,
        "line": getattr(exc, "line", None),
        "section": getattr(exc, "section", None),
    }
    return {"status": "error", "error": error}


def _none_result_envelope(path: str | Path) -> dict[str, Any]:
    """Build the failure envelope for an extraction that produced no result."""
    return {
        "status": "error",
        "error": {
            "code": _GENERIC_EXTRACTION_ERROR_CODE,
            "message": (
                f"Extraction failed: no POMDP state space could be extracted "
                f"from {path}"
            ),
            "line": None,
            "section": None,
        },
    }


def extract_to_json(
    path: str | Path,
    *,
    strict_validation: bool = True,
    on_error: str = "lenient",
    compact: bool = False,
    strict_probabilities: bool = False,
) -> str:
    """Extract a POMDP state space from ``path`` and return the payload JSON.

    Args:
        path: Path to a GNN specification file containing a POMDP.
        strict_validation: Enable strict validation in the extractor.
        on_error: Extractor error mode — ``"lenient"`` (default), ``"raise"``,
            or ``"collect"``. Passed through to
            ``gnn.extract.pomdp_extractor.extract_pomdp_from_file``.
        compact: Emit compact JSON (no indentation, ``(",", ":")``
            separators) instead of the default ``indent=2``.
        strict_probabilities: Opt in to the extractor's strict probability
            and dimension contract (requires ``strict_validation``); with
            ``on_error="raise"`` a violation becomes the error envelope.

    Returns:
        JSON string: the ``POMDPStateSpace.to_dict()`` payload on success, or
        the ``{"status": "error", "error": {...}}`` envelope on failure. Never
        raises.
    """
    # Lazy import keeps `import gnn.extract` free of any pipeline transitive
    # dependencies; gnn.extract.pomdp_extractor is stdlib-only at module scope.
    from gnn.extract.pomdp_extractor import OnErrorMode, extract_pomdp_from_file

    try:
        result: Any = extract_pomdp_from_file(
            path,
            strict_validation=strict_validation,
            on_error=cast(OnErrorMode, on_error),
            strict_probabilities=strict_probabilities,
        )
    except (
        Exception
    ) as exc:  # structured errors (on_error="raise") and unexpected failures
        return _dumps(_error_envelope(exc), compact=compact)

    # on_error="collect" returns (spec | None, list of structured errors).
    if isinstance(result, tuple):
        spec, errors = result
        if spec is not None:
            return _dumps(spec.to_dict(), compact=compact)
        if errors:
            return _dumps(_error_envelope(errors[0]), compact=compact)
        return _dumps(_none_result_envelope(path), compact=compact)

    if result is None:
        return _dumps(_none_result_envelope(path), compact=compact)
    return _dumps(result.to_dict(), compact=compact)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: print the extraction payload or error envelope.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        0 on success, 1 on extraction failure (repo exit-code convention).
    """
    parser = argparse.ArgumentParser(
        prog="gnn.extract",
        description=(
            "Extract a POMDP state space from a GNN specification file and "
            "print it as JSON (see the gnn.extract module contract, schema "
            f"version {_ENVELOPE_SCHEMA_VERSION})."
        ),
    )
    parser.add_argument("file", help="Path to a GNN (.md) specification file")
    strict_group = parser.add_mutually_exclusive_group()
    strict_group.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=True,
        help="Enable strict validation (default)",
    )
    strict_group.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Disable strict validation",
    )
    parser.add_argument(
        "--strict-probabilities",
        action="store_true",
        help=(
            "Fail on non-stochastic A/B/D tensors, negative or non-finite "
            "entries, and D/C length mismatches (requires strict validation)"
        ),
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit compact JSON instead of indented JSON",
    )
    args = parser.parse_args(argv)

    if args.strict_probabilities and not args.strict:
        parser.error("--strict-probabilities cannot be combined with --no-strict")
    if args.strict_probabilities:
        payload = extract_to_json(
            args.file,
            strict_validation=True,
            on_error="raise",
            compact=args.compact,
            strict_probabilities=True,
        )
    else:
        payload = extract_to_json(
            args.file, strict_validation=args.strict, compact=args.compact
        )
    print(payload)

    envelope: Any = json.loads(payload)
    if isinstance(envelope, dict) and envelope.get("status") == "error":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

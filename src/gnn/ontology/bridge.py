#!/usr/bin/env python3
"""Pinned consumer view of the Active Inference Ontology vocabulary bridge.

The ontology repository publishes ``GNN/act_inf_ontology_terms.json`` (schema
``active-inference-ontology/gnn-terms/v1``): every release-v5 label with a
stable ``term-*`` identifier, plus the resolution status of the 64 labels the
bundled GNN vocabulary declares. GNN keeps a byte-exact copy at
``active_inference_ontology_terms.json`` beside this module and records its
provenance in ``ontology-pin.json``; ``scripts/check_ontology_pin.py`` fails
when the copy and the pin disagree.

Resolution rule for annotations: a declared label whose ``status`` is
``resolved`` maps to its ``term-*`` identifier; a declared label whose
``status`` is ``unresolved`` has no counterpart in ontology release v5 and is
reported as such, never given an invented identifier. Labels outside the
audited declared set are reported as ``not_declared``.

The ``obo:ACTO_*`` / ``obo:TEMP_*`` values carried by the bundled GNN
vocabulary are declared metadata only. No ACTO ontology exists in the OBO
Foundry, Bioregistry, or OLS4, so those prefixes are never emitted as
identifiers by any code path in this package.

All helpers are deterministic: file reads and hashing only; no network, clock,
or randomness.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Tuple

ONTOLOGY_BRIDGE_SCHEMA = "active-inference-ontology/gnn-terms/v1"
ONTOLOGY_BRIDGE_FILENAME = "active_inference_ontology_terms.json"
ONTOLOGY_PIN_FILENAME = "ontology-pin.json"
ONTOLOGY_PIN_REQUIRED_KEYS = (
    "repository",
    "commit",
    "release",
    "path",
    "sha256",
    "sourceSha256",
    "exportSha256",
)

DECLARED_STATUS_RESOLVED = "resolved"
DECLARED_STATUS_UNRESOLVED = "unresolved"
DECLARED_STATUS_NOT_DECLARED = "not_declared"
DECLARED_STATUS_BRIDGE_UNAVAILABLE = "bridge_unavailable"
DECLARED_STATUSES = (
    DECLARED_STATUS_RESOLVED,
    DECLARED_STATUS_UNRESOLVED,
    DECLARED_STATUS_NOT_DECLARED,
    DECLARED_STATUS_BRIDGE_UNAVAILABLE,
)

RESOLVED_ID_PREFIX = "term-"
UNRESOLVED_REASON = "not in ontology release v5"
UNDECLARED_REASON = "label is not in the audited declared set"
BRIDGE_UNAVAILABLE_REASON = "pinned ontology bridge unavailable"

DeclaredIndex = Dict[str, Dict[str, Any]]
"""Case-folded declared label -> the bridge's ``declared`` entry for it."""


def ontology_dir() -> Path:
    """Directory owning the pinned bridge copy and its pin record."""
    return Path(__file__).resolve().parent


def ontology_bridge_path() -> Path:
    """Path of the pinned bridge copy inside the package."""
    return ontology_dir() / ONTOLOGY_BRIDGE_FILENAME


def ontology_pin_path() -> Path:
    """Path of the pin record for the pinned bridge copy."""
    return ontology_dir() / ONTOLOGY_PIN_FILENAME


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of ``path`` (chunked read)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


@lru_cache(maxsize=4)
def load_ontology_bridge(path: Optional[str] = None) -> Dict[str, Any]:
    """Load and structurally validate a vocabulary bridge document.

    ``path`` defaults to the pinned copy beside this module, then the
    repo-relative ``src/gnn/ontology/...`` location for source checkouts.

    Raises:
        FileNotFoundError: Neither candidate exists.
        ValueError: The document is not a bridge of the supported schema, or
            its ``declared``/``terms`` blocks are missing or malformed.

    Cached per path; call ``load_ontology_bridge.cache_clear()`` after
    replacing a copy in place.
    """
    candidates = (
        [Path(path)]
        if path is not None
        else [
            ontology_bridge_path(),
            Path("src/gnn/ontology") / ONTOLOGY_BRIDGE_FILENAME,
        ]
    )
    source = next((candidate for candidate in candidates if candidate.exists()), None)
    if source is None:
        raise FileNotFoundError(
            "ontology bridge not found: " + ", ".join(str(c) for c in candidates)
        )

    with open(source, "r", encoding="utf-8") as handle:
        document = json.load(handle)

    if not isinstance(document, dict):
        raise ValueError("ontology bridge must be a JSON object")
    if document.get("schema") != ONTOLOGY_BRIDGE_SCHEMA:
        raise ValueError(
            f"unsupported ontology bridge schema: {document.get('schema')!r}"
        )
    declared = document.get("declared")
    terms = document.get("terms")
    if not isinstance(declared, dict) or not declared:
        raise ValueError("ontology bridge declares no labels")
    if not isinstance(terms, dict) or not terms:
        raise ValueError("ontology bridge carries no terms")
    for label, entry in declared.items():
        status = entry.get("status") if isinstance(entry, dict) else None
        if status not in (DECLARED_STATUS_RESOLVED, DECLARED_STATUS_UNRESOLVED):
            raise ValueError(
                f"ontology bridge declared label {label!r} has an unknown status"
            )
    return document


def load_ontology_pin(path: Optional[str] = None) -> Dict[str, Any]:
    """Load and structurally validate the bridge copy's pin record.

    Raises:
        FileNotFoundError: The pin record does not exist.
        ValueError: The record is not an object or misses a required field.
    """
    source = Path(path) if path is not None else ontology_pin_path()
    if not source.exists():
        raise FileNotFoundError(f"ontology pin not found: {source}")
    with open(source, "r", encoding="utf-8") as handle:
        pin = json.load(handle)
    if not isinstance(pin, dict):
        raise ValueError("ontology pin must be a JSON object")
    missing = [
        key for key in ONTOLOGY_PIN_REQUIRED_KEYS if not _is_nonempty_str(pin.get(key))
    ]
    if missing:
        raise ValueError(
            "ontology pin is missing required fields: " + ", ".join(missing)
        )
    return pin


def verify_ontology_pin(
    bridge_path: Optional[Path] = None,
    pin_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Verify the pinned bridge copy against its pin record.

    Checks, each appended to ``errors`` when it fails:

    1. the copy exists and its SHA-256 equals ``pin['sha256']``;
    2. the bridge schema equals ``pin['schema']`` (when present) and the
       module's supported schema;
    3. the bridge's embedded ``source.sha256`` / ``export.sha256`` equal the
       pin's ``sourceSha256`` / ``exportSha256``;
    4. the bridge's declared/term counts agree with its own ``counts`` block;
    5. every resolved declared label carries a ``term-*`` identifier and every
       unresolved declared label carries none.

    Returns a JSON-serialisable summary (``verified``, ``errors``, ``sha256``,
    ``pin``, ``bridge``); verification failures are reported, not raised.
    """
    errors: list[str] = []
    summary: Dict[str, Any] = {
        "verified": False,
        "errors": errors,
        "sha256": None,
        "pin": None,
        "bridge": None,
    }
    copy = Path(bridge_path) if bridge_path is not None else ontology_bridge_path()
    pin_file = Path(pin_path) if pin_path is not None else ontology_pin_path()

    try:
        pin = load_ontology_pin(str(pin_file))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"pin record unreadable: {exc}")
        return summary
    summary["pin"] = {key: pin.get(key) for key in ONTOLOGY_PIN_REQUIRED_KEYS}

    if not copy.exists():
        errors.append(f"pinned bridge copy missing: {copy}")
        return summary
    try:
        actual = sha256_file(copy)
    except OSError as exc:
        errors.append(f"pinned bridge copy unreadable: {exc}")
        return summary
    summary["sha256"] = actual
    if actual != pin["sha256"]:
        errors.append(
            f"pinned bridge copy SHA-256 {actual} does not match the pin {pin['sha256']}"
        )

    try:
        bridge = load_ontology_bridge(str(copy))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"pinned bridge copy invalid: {exc}")
        return summary

    release = bridge.get("release")
    summary["bridge"] = {
        "schema": bridge.get("schema"),
        "release": release.get("version") if isinstance(release, dict) else None,
        "counts": bridge.get("counts"),
    }
    if "schema" in pin and bridge.get("schema") != pin["schema"]:
        errors.append(
            f"bridge schema {bridge.get('schema')!r} does not match the pin {pin['schema']!r}"
        )

    source = bridge.get("source") if isinstance(bridge.get("source"), dict) else {}
    export = bridge.get("export") if isinstance(bridge.get("export"), dict) else {}
    for label, pin_key, embedded in (
        ("source", "sourceSha256", source.get("sha256")),
        ("export", "exportSha256", export.get("sha256")),
    ):
        if embedded != pin[pin_key]:
            errors.append(
                f"bridge {label} digest {embedded} does not match the pin {pin[pin_key]}"
            )

    declared = bridge["declared"]
    terms = bridge["terms"]
    counts = bridge.get("counts") if isinstance(bridge.get("counts"), dict) else {}
    for label, expected in (
        ("declaredLabels", len(declared)),
        ("terms", len(terms)),
    ):
        recorded = counts.get(label)
        if recorded is not None and recorded != expected:
            errors.append(
                f"bridge counts.{label} records {recorded} but the document carries {expected}"
            )
    resolved = sum(
        1
        for entry in declared.values()
        if entry.get("status") == DECLARED_STATUS_RESOLVED
    )
    recorded_resolved = counts.get("resolvedDeclaredLabels")
    if recorded_resolved is not None and recorded_resolved != resolved:
        errors.append(
            "bridge counts.resolvedDeclaredLabels records "
            f"{recorded_resolved} but {resolved} declared labels are resolved"
        )

    for label, entry in sorted(declared.items()):
        identifier = entry.get("id")
        if entry.get("status") == DECLARED_STATUS_RESOLVED:
            if not _is_nonempty_str(identifier) or not str(identifier).startswith(
                RESOLVED_ID_PREFIX
            ):
                errors.append(
                    f"resolved declared label {label!r} lacks a {RESOLVED_ID_PREFIX}* identifier"
                )
        elif identifier is not None:
            errors.append(
                f"unresolved declared label {label!r} carries an identifier ({identifier!r})"
            )

    summary["verified"] = not errors
    return summary


def ontology_bridge_status() -> Dict[str, Any]:
    """Pin/bridge verification summary for run artifacts.

    Missing, unreadable or schema-invalid pin/copy states already come back
    from :func:`verify_ontology_pin` as ``verified: False`` with the reasons in
    ``errors``, so this wrapper adds no failure mode of its own.
    """
    verification = verify_ontology_pin()
    bridge = verification.get("bridge") or {}
    pin = verification.get("pin") or {}
    return {
        "available": bridge.get("schema") is not None,
        "verified": verification["verified"],
        "schema": bridge.get("schema"),
        "release": bridge.get("release"),
        "sha256": verification.get("sha256"),
        "pin_commit": pin.get("commit"),
        "errors": verification["errors"],
    }


def declared_label_index(bridge: Dict[str, Any]) -> DeclaredIndex:
    """Build the case-folded declared-label index for a bridge document."""
    return {
        str(label).casefold(): entry
        for label, entry in bridge.get("declared", {}).items()
        if isinstance(entry, dict)
    }


class DeclaredResolution(NamedTuple):
    """Resolution of one annotation label against the bridge's declared map."""

    status: str
    identifier: Optional[str]
    ontology_term: Optional[str]
    reason: str


def resolve_declared_label(
    label: str, declared: Optional[DeclaredIndex]
) -> DeclaredResolution:
    """Resolve one annotation label to its ontology identifier.

    ``declared`` is an index from :func:`declared_label_index`; ``None`` means
    the pinned bridge was unavailable, and every label resolves to
    ``bridge_unavailable`` rather than to an invented identifier.
    """
    if declared is None:
        return DeclaredResolution(
            DECLARED_STATUS_BRIDGE_UNAVAILABLE, None, None, BRIDGE_UNAVAILABLE_REASON
        )
    entry = declared.get(label.casefold() if isinstance(label, str) else "")
    if entry is None:
        return DeclaredResolution(
            DECLARED_STATUS_NOT_DECLARED, None, None, UNDECLARED_REASON
        )
    identifier = entry.get("id")
    if entry.get("status") == DECLARED_STATUS_RESOLVED and _is_nonempty_str(identifier):
        return DeclaredResolution(
            DECLARED_STATUS_RESOLVED,
            str(identifier),
            entry.get("ontologyTerm")
            if _is_nonempty_str(entry.get("ontologyTerm"))
            else None,
            "",
        )
    return DeclaredResolution(DECLARED_STATUS_UNRESOLVED, None, None, UNRESOLVED_REASON)


@lru_cache(maxsize=1)
def pinned_declared_index() -> Tuple[Optional[DeclaredIndex], Optional[str]]:
    """Return ``(declared index, error)`` for the pinned bridge copy.

    The index is ``None`` when the copy cannot be loaded, with the failure
    detail in ``error``; callers then report every label as
    ``bridge_unavailable``. Cached; call ``pinned_declared_index.cache_clear()``
    after replacing the copy in place.
    """
    try:
        return declared_label_index(load_ontology_bridge()), None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, f"{BRIDGE_UNAVAILABLE_REASON}: {exc}"

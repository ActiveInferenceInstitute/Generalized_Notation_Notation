"""Downstream importer conformance: read a repository's GNN pin, validate its specs.

A repository that applies Generalized Notation Notation records which upstream
revision it consumes in a pin file, and keeps one or more ``*.gnn.md``
specifications that must parse under that revision. Every importer used to
carry its own script for this; the shapes drifted (``commit`` vs ``revision``,
``tag`` vs ``release``, four different file locations), so nothing could check
the estate as a whole.

This module is the single reader for those pins. It accepts the shapes that
exist, normalizes them to :class:`ImporterPin`, and can validate the
repository's specifications with the parser that is installed — the same
parser the pin names.

Canonical pin (``schema``: :data:`PIN_SCHEMA`)::

    {
      "schema": "gnn-importer-pin/v1",
      "repository": "ActiveInferenceInstitute/Generalized_Notation_Notation",
      "tag": "v5.0.3",
      "commit": "8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1",
      "recorded": "2026-11-12",
      "spec_files": ["models/gridworld_3x3.gnn.md"]
    }

Earlier shapes are read, never rewritten: ``revision`` is accepted for
``commit``, ``release`` for ``tag``, ``repository`` may be an ``org/name`` or a
URL, and ``spec_files`` is optional (absent means "discover ``*.gnn.md``").

Usage::

    python -m gnn.importers check --repo . --json
    python -m gnn.importers check --repo . --expected-commit 8f293d9b...
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "PIN_SCHEMA",
    "ImporterPin",
    "PinError",
    "check_importer",
    "discover_specs",
    "find_pin",
    "main",
    "normalize_repository",
    "read_pin",
]

#: Schema identifier a canonical importer pin declares.
PIN_SCHEMA = "gnn-importer-pin/v1"

#: The upstream a pin is expected to name. A pin that names a fork is a real
#: finding: the consumer is tracking code this distribution does not publish.
UPSTREAM_REPOSITORY = "ActiveInferenceInstitute/Generalized_Notation_Notation"

#: Pin locations in precedence order. The first file that exists wins; the
#: order is canonical-location-first so a repository that adopts
#: ``.github/gnn-pin.json`` keeps one reader even if an older file remains.
PIN_CANDIDATES: Tuple[str, ...] = (
    ".github/gnn-pin.json",
    "specs/gnn-pin.json",
    "gnn/upstream.json",
    "models/gnn-pin.json",
    "models/gnn.pin.json",
    "NHIM/gnn-pin.json",
    ".github/gnn-pair.json",
)

#: Directories never searched for specifications.
SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "dist",
        "build",
        "site-packages",
        "output",
        "cache",
        "tmp",
    }
)


class PinError(ValueError):
    """A pin file exists but does not record a usable upstream revision."""


@dataclass(frozen=True)
class ImporterPin:
    """A normalized importer pin.

    ``commit`` is the only required identity: a tag can move, a commit cannot.
    ``schema`` and ``recorded`` are ``None`` for pins that predate the
    canonical schema; :func:`check_importer` reports that as a finding rather
    than failing, so an importer can adopt the schema deliberately.
    """

    path: Path
    commit: str
    tag: Optional[str] = None
    repository: Optional[str] = None
    schema: Optional[str] = None
    recorded: Optional[str] = None
    spec_files: Tuple[str, ...] = ()
    raw: Dict[str, Any] = field(default_factory=dict)


def normalize_repository(value: Any) -> Optional[str]:
    """Return ``org/name`` for a pin's repository field, or ``None``.

    Accepts the two forms in use: a bare ``org/name`` and a GitHub URL with or
    without a ``.git`` suffix.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.startswith(("http://", "https://")):
        text = text.split("://", 1)[1]
        if text.startswith("github.com/"):
            text = text[len("github.com/") :]
    if text.startswith("git@github.com:"):
        text = text[len("git@github.com:") :]
    if text.endswith(".git"):
        text = text[: -len(".git")]
    parts = [part for part in text.split("/") if part]
    if len(parts) < 2:
        return None
    return f"{parts[0]}/{parts[1]}"


def find_pin(repo_root: Path) -> Optional[Path]:
    """Return the first pin file that exists under ``repo_root``."""
    for relative in PIN_CANDIDATES:
        candidate = repo_root / relative
        if candidate.is_file():
            return candidate
    return None


def read_pin(path: Path) -> ImporterPin:
    """Read and normalize one pin file.

    Raises:
        PinError: the file is not JSON, or records no commit/revision.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PinError(f"{path}: not readable JSON ({exc})") from exc
    if not isinstance(payload, dict):
        raise PinError(f"{path}: pin must be a JSON object")

    commit = payload.get("commit") or payload.get("revision")
    if not isinstance(commit, str) or not commit.strip():
        raise PinError(f"{path}: pin records no commit/revision")
    tag = payload.get("tag") or payload.get("release")
    spec_files = payload.get("spec_files")
    if isinstance(spec_files, str):
        spec_files = [spec_files]
    if not isinstance(spec_files, list):
        spec_files = []
    return ImporterPin(
        path=path,
        commit=commit.strip(),
        tag=tag.strip() if isinstance(tag, str) and tag.strip() else None,
        repository=normalize_repository(payload.get("repository")),
        schema=payload.get("schema")
        if isinstance(payload.get("schema"), str)
        else None,
        recorded=payload.get("recorded")
        if isinstance(payload.get("recorded"), str)
        else None,
        spec_files=tuple(str(item) for item in spec_files),
        raw=payload,
    )


def _walk_specs(repo_root: Path) -> Iterable[Path]:
    for path in sorted(repo_root.rglob("*.gnn.md")):
        if any(part in SKIP_DIRS for part in path.relative_to(repo_root).parts):
            continue
        if path.is_file():
            yield path


def discover_specs(repo_root: Path, pin: Optional[ImporterPin] = None) -> List[Path]:
    """Return the specifications this repository declares or carries.

    A pin that lists ``spec_files`` is authoritative: a missing declared file is
    reported by :func:`check_importer`, not silently replaced by a glob.
    """
    if pin is not None and pin.spec_files:
        return [repo_root / relative for relative in pin.spec_files]
    return list(_walk_specs(repo_root))


def _validate_spec(path: Path) -> List[str]:
    """Validate one specification with the installed parser; return errors."""
    from gnn.parsers.basic import validate_gnn_syntax

    try:
        is_valid, errors = validate_gnn_syntax(path)
    except (
        OSError,
        UnicodeDecodeError,
        ValueError,
        TypeError,
        KeyError,
        RecursionError,
    ) as exc:
        # A document from another repository is untrusted input: a validator
        # failure on it is a finding about that document, not a crash here.
        return [f"validator raised {type(exc).__name__}: {exc}"]
    if is_valid:
        return []
    return [str(error) for error in errors] or ["invalid specification"]


def check_importer(
    repo_root: Path,
    *,
    expected_commit: Optional[str] = None,
    expected_tag: Optional[str] = None,
    expected_repository: Optional[str] = UPSTREAM_REPOSITORY,
    validate_specs: bool = True,
    require_pin: bool = True,
) -> Dict[str, Any]:
    """Check one importer checkout and return a JSON-ready report.

    Findings carry ``level`` (``error`` fails the check, ``warning`` does not)
    and ``code`` so a caller can filter without parsing prose.
    """
    repo_root = repo_root.resolve()
    findings: List[Dict[str, str]] = []
    report: Dict[str, Any] = {
        "repo": str(repo_root),
        "pin": None,
        "specs": {"count": 0, "files": [], "invalid": []},
        "findings": findings,
        "ok": False,
    }

    pin_path = find_pin(repo_root)
    pin: Optional[ImporterPin] = None
    if pin_path is None:
        if require_pin:
            findings.append(
                {
                    "level": "error",
                    "code": "pin-missing",
                    "message": "no GNN pin file found (looked for "
                    + ", ".join(PIN_CANDIDATES)
                    + ")",
                }
            )
        else:
            findings.append(
                {
                    "level": "warning",
                    "code": "pin-missing",
                    "message": "no GNN pin file; the specification files are still validated",
                }
            )
    else:
        try:
            pin = read_pin(pin_path)
        except PinError as exc:
            findings.append(
                {"level": "error", "code": "pin-unreadable", "message": str(exc)}
            )
        else:
            report["pin"] = {
                "path": str(pin.path.relative_to(repo_root)),
                "commit": pin.commit,
                "tag": pin.tag,
                "repository": pin.repository,
                "schema": pin.schema,
                "recorded": pin.recorded,
                "spec_files": list(pin.spec_files),
            }
            if pin.schema != PIN_SCHEMA:
                findings.append(
                    {
                        "level": "warning",
                        "code": "pin-schema-unversioned",
                        "message": f"pin declares schema {pin.schema!r}, not {PIN_SCHEMA!r}",
                    }
                )
            if expected_commit and pin.commit != expected_commit:
                findings.append(
                    {
                        "level": "error",
                        "code": "pin-commit-mismatch",
                        "message": f"pin records {pin.commit}, expected {expected_commit}",
                    }
                )
            if expected_tag and pin.tag != expected_tag:
                findings.append(
                    {
                        "level": "error",
                        "code": "pin-tag-mismatch",
                        "message": f"pin records tag {pin.tag!r}, expected {expected_tag!r}",
                    }
                )
            if (
                expected_repository
                and pin.repository
                and pin.repository != expected_repository
            ):
                findings.append(
                    {
                        "level": "error",
                        "code": "pin-repository-mismatch",
                        "message": (
                            f"pin names {pin.repository}, expected {expected_repository}; "
                            "a pin that names a fork tracks code this distribution does not publish"
                        ),
                    }
                )

    specs = discover_specs(repo_root, pin)
    report["specs"]["count"] = len(specs)
    report["specs"]["files"] = [str(path.relative_to(repo_root)) for path in specs]
    for spec in specs:
        if not spec.is_file():
            findings.append(
                {
                    "level": "error",
                    "code": "spec-missing",
                    "message": f"{spec.relative_to(repo_root)} is declared by the pin but absent",
                }
            )
            continue
        if not validate_specs:
            continue
        errors = _validate_spec(spec)
        if errors:
            report["specs"]["invalid"].append(
                {"file": str(spec.relative_to(repo_root)), "errors": errors}
            )
            findings.append(
                {
                    "level": "error",
                    "code": "spec-invalid",
                    "message": f"{spec.relative_to(repo_root)}: {errors[0]}",
                }
            )

    if pin is not None and not specs:
        findings.append(
            {
                "level": "warning",
                "code": "spec-absent",
                "message": "pin present but the repository carries no *.gnn.md specification",
            }
        )

    report["ok"] = not any(item["level"] == "error" for item in findings)
    return report


def _format_text(report: Dict[str, Any]) -> str:
    lines = [f"importer: {report['repo']}"]
    pin = report["pin"]
    if pin:
        lines.append(
            f"  pin: {pin['path']} -> {pin['tag'] or '(no tag)'} @ {pin['commit'][:10]}"
            f" (schema {pin['schema'] or 'unversioned'})"
        )
    else:
        lines.append("  pin: none")
    lines.append(f"  specs: {report['specs']['count']}")
    for item in report["findings"]:
        lines.append(f"  [{item['level']}] {item['code']}: {item['message']}")
    lines.append("  verdict: " + ("ok" if report["ok"] else "FAILED"))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m gnn.importers`` entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m gnn.importers",
        description="Check a downstream repository's GNN pin and specifications.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check", help="check one importer checkout")
    check.add_argument("--repo", default=".", help="repository root to check")
    check.add_argument(
        "--expected-commit", default=None, help="upstream commit the pin must record"
    )
    check.add_argument(
        "--expected-tag", default=None, help="upstream tag the pin must record"
    )
    check.add_argument(
        "--expected-repository",
        default=UPSTREAM_REPOSITORY,
        help="repository the pin must name (empty string disables the check)",
    )
    check.add_argument(
        "--no-validate",
        action="store_true",
        help="read the pin and list specifications without parsing them",
    )
    check.add_argument(
        "--allow-missing-pin",
        action="store_true",
        help="treat a missing pin as a warning instead of an error",
    )
    check.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = parser.parse_args(argv)

    report = check_importer(
        Path(args.repo),
        expected_commit=args.expected_commit,
        expected_tag=args.expected_tag,
        expected_repository=args.expected_repository or None,
        validate_specs=not args.no_validate,
        require_pin=not args.allow_missing_pin,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_format_text(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

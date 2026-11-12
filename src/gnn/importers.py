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

Two shapes need a hint from the caller:

* **A two-party custody pin.** ``fep_formal`` records ``fep_lean`` and ``gnn``
  as separate sub-objects of ``specs/gnn-bridge-w2-source-custody/source-pin.json``
  with per-file SHA-256 owners instead of a release tag. ``--pin`` names the
  file and ``--pin-party gnn`` names the sub-record; the reader normalizes it
  to the same :class:`ImporterPin`.
* **A differently directed pair pin.** The upstream GNN checkout pins its
  downstreams in ``.github/gnn-pair.json``; discovery would read that as a GNN
  pin and report a fork. ``--pin`` is how a caller with a non-conventional pin
  states which file it means.

``--require-pointer`` additionally checks ``.github/GNN.md``: the pointer must
name the pin file, the pinned revision and — when given — the conformance
workflow and job key. It is opt-in because a repository adopts it deliberately,
in the same change that writes the pointer.

Usage::

    python -m gnn.importers check --repo . --json
    python -m gnn.importers check --repo . --expected-commit 8f293d9b...
    python -m gnn.importers check --repo . --require-pointer \
        --workflow .github/workflows/ci.yml --job gnn-conformance
    python -m gnn.importers check --repo . \
        --pin specs/gnn-bridge-w2-source-custody/source-pin.json --pin-party gnn
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "IMPORTER_VARIABLES_SCHEMA",
    "PIN_SCHEMA",
    "POINTER_PATH",
    "ImporterPin",
    "PinError",
    "check_importer",
    "check_pointer",
    "discover_specs",
    "find_pin",
    "importer_variables",
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

#: The human-readable pointer every importer publishes beside its pin. It
#: records what the conformance job parses, so a reader can reach the pin, the
#: revision and the job from one file instead of reconstructing them from a
#: workflow definition.
POINTER_PATH = ".github/GNN.md"

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
    #: Sub-record this pin was read from, for a two-party custody pin that
    #: nests one object per repository (``fep_formal`` records ``fep_lean`` and
    #: ``gnn`` in one file). ``None`` for the ordinary flat pin.
    party: Optional[str] = None
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


def read_pin(path: Path, *, party: Optional[str] = None) -> ImporterPin:
    """Read and normalize one pin file.

    Args:
        path: The pin file.
        party: Sub-record to read, for a custody pin that nests one object per
            repository. ``None`` reads the file as a flat pin.

    Raises:
        PinError: the file is not JSON, records no commit/revision, or the
            requested ``party`` sub-record is absent.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PinError(f"{path}: not readable JSON ({exc})") from exc
    if not isinstance(payload, dict):
        raise PinError(f"{path}: pin must be a JSON object")

    source = payload
    if party is not None:
        nested = payload.get(party)
        if not isinstance(nested, dict):
            raise PinError(
                f"{path}: records no {party!r} sub-record "
                f"(file names {sorted(payload)})"
            )
        source = nested

    commit = source.get("commit") or source.get("revision")
    if not isinstance(commit, str) or not commit.strip():
        raise PinError(f"{path}: pin records no commit/revision")
    tag = source.get("tag") or source.get("release")
    spec_files = source.get("spec_files")
    if isinstance(spec_files, str):
        spec_files = [spec_files]
    if not isinstance(spec_files, list):
        spec_files = []
    return ImporterPin(
        path=path,
        commit=commit.strip(),
        tag=tag.strip() if isinstance(tag, str) and tag.strip() else None,
        repository=normalize_repository(source.get("repository")),
        schema=source.get("schema") if isinstance(source.get("schema"), str) else None,
        recorded=source.get("recorded")
        if isinstance(source.get("recorded"), str)
        else None,
        spec_files=tuple(str(item) for item in spec_files),
        party=party,
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


def _relative(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root`` when it is inside it, else the name."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def check_pointer(
    repo_root: Path,
    pin: ImporterPin,
    *,
    workflow: Optional[str] = None,
    job: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Check the importer's ``.github/GNN.md`` pointer against its pin.

    The pointer is what a human reads and what the estate's online re-read
    looks for; a pointer that names a different pin, revision or job is a
    finding, not a documentation preference. Every check is textual: the
    pointer is prose, so the reader asks whether the facts appear, not whether
    the file has a schema.
    """
    findings: List[Dict[str, str]] = []
    path = repo_root / POINTER_PATH
    if not path.is_file():
        return [
            {
                "level": "error",
                "code": "pointer-missing",
                "message": f"{POINTER_PATH} is absent; it must name {_relative(pin.path, repo_root)}, "
                f"{pin.commit[:12]} and the conformance job",
            }
        ]
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [
            {
                "level": "error",
                "code": "pointer-unreadable",
                "message": f"{POINTER_PATH} is not readable text ({exc})",
            }
        ]

    pin_name = _relative(pin.path, repo_root)
    if pin_name not in text:
        findings.append(
            {
                "level": "error",
                "code": "pointer-pin-unnamed",
                "message": f"{POINTER_PATH} does not name the pin file {pin_name}",
            }
        )
    if pin.commit not in text and pin.commit[:12] not in text:
        findings.append(
            {
                "level": "error",
                "code": "pointer-revision-unnamed",
                "message": f"{POINTER_PATH} does not name the pinned revision {pin.commit[:12]}",
            }
        )
    if workflow and Path(workflow).name not in text:
        findings.append(
            {
                "level": "error",
                "code": "pointer-workflow-unnamed",
                "message": f"{POINTER_PATH} does not name the workflow {workflow}",
            }
        )
    if job and job not in text:
        findings.append(
            {
                "level": "error",
                "code": "pointer-job-unnamed",
                "message": f"{POINTER_PATH} does not name the job key {job}",
            }
        )
    return findings


#: Schema identifier for the GNN-derived subset of a manuscript variable map.
IMPORTER_VARIABLES_SCHEMA = "gnn-importer-variables/v1"


def importer_variables(
    pin: ImporterPin,
    *,
    repo_root: Path,
    job: Optional[str] = None,
    job_name: Optional[str] = None,
) -> Dict[str, str]:
    """Return the GNN-derived manuscript variables for one importer.

    A repository whose manuscript states which GNN release it consumes needs
    the same seven facts this reader already normalizes. Producing them here
    means the prose and the conformance job cannot disagree about the pin, and
    a consumer does not re-read the JSON in its own variable script.

    ``gnn_job_key`` and ``gnn_job_name`` are the conformance job the registry
    row records; a caller that has them passes them, because the reader does
    not parse workflow YAML.
    """
    return {
        "schema": IMPORTER_VARIABLES_SCHEMA,
        "gnn_pin_path": _relative(pin.path, repo_root),
        "gnn_pin_schema": pin.schema or "",
        "gnn_repository": pin.repository or "",
        "gnn_tag": pin.tag or "",
        "gnn_commit": pin.commit,
        "gnn_job_key": job or "",
        "gnn_job_name": job_name or "",
    }


def check_importer(
    repo_root: Path,
    *,
    expected_commit: Optional[str] = None,
    expected_tag: Optional[str] = None,
    expected_repository: Optional[str] = UPSTREAM_REPOSITORY,
    validate_specs: bool = True,
    require_pin: bool = True,
    pin_path: Optional[Path] = None,
    pin_party: Optional[str] = None,
    require_pointer: bool = False,
    pointer_workflow: Optional[str] = None,
    pointer_job: Optional[str] = None,
) -> Dict[str, Any]:
    """Check one importer checkout and return a JSON-ready report.

    Findings carry ``level`` (``error`` fails the check, ``warning`` does not)
    and ``code`` so a caller can filter without parsing prose.

    Args:
        pin_path: Read this pin instead of discovering one. Needed where a
            repository keeps a differently named custody pin
            (``fep_formal``'s ``source-pin.json``) or where the conventional
            candidate list would pick a *different repository's* pin (the
            upstream GNN checkout pins its downstreams in
            ``.github/gnn-pair.json``).
        pin_party: Sub-record to read inside ``pin_path`` for a two-party pin.
        require_pointer: Also require ``.github/GNN.md`` to name the pin, the
            revision and the declared workflow and job.
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

    pin_file = find_pin(repo_root)
    if pin_path is not None:
        pin_file = pin_path if pin_path.is_absolute() else repo_root / pin_path
        if not pin_file.is_file():
            findings.append(
                {
                    "level": "error",
                    "code": "pin-missing",
                    "message": f"the declared pin {_relative(pin_file, repo_root)} is absent",
                }
            )
            pin_file = None
    pin: Optional[ImporterPin] = None
    if pin_file is None:
        if pin_path is not None:
            pass  # the absent declared pin is already the finding
        elif require_pin:
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
            pin = read_pin(pin_file, party=pin_party)
        except PinError as exc:
            findings.append(
                {"level": "error", "code": "pin-unreadable", "message": str(exc)}
            )
        else:
            report["pin"] = {
                "path": _relative(pin.path, repo_root),
                "party": pin.party,
                "commit": pin.commit,
                "tag": pin.tag,
                "repository": pin.repository,
                "schema": pin.schema,
                "recorded": pin.recorded,
                "spec_files": list(pin.spec_files),
            }
            if not re.fullmatch(r"[0-9a-f]{40}", pin.commit):
                findings.append(
                    {
                        "level": "error",
                        "code": "pin-commit-shape",
                        "message": f"pin records {pin.commit!r}, which is not a 40-character commit id",
                    }
                )
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

    if require_pointer:
        if pin is None:
            findings.append(
                {
                    "level": "error",
                    "code": "pointer-unchecked",
                    "message": f"{POINTER_PATH} cannot be checked without a readable pin",
                }
            )
        else:
            pointer_findings = check_pointer(
                repo_root,
                pin,
                workflow=pointer_workflow,
                job=pointer_job,
            )
            findings.extend(pointer_findings)
            report["pointer"] = {
                "path": POINTER_PATH,
                "ok": not pointer_findings,
            }

    report["ok"] = not any(item["level"] == "error" for item in findings)
    return report


def _format_text(report: Dict[str, Any]) -> str:
    lines = [f"importer: {report['repo']}"]
    pin = report["pin"]
    if pin:
        party = f" [{pin['party']}]" if pin.get("party") else ""
        lines.append(
            f"  pin: {pin['path']}{party} -> {pin['tag'] or '(no tag)'} @ {pin['commit'][:10]}"
            f" (schema {pin['schema'] or 'unversioned'})"
        )
    else:
        lines.append("  pin: none")
    lines.append(f"  specs: {report['specs']['count']}")
    pointer = report.get("pointer")
    if pointer:
        lines.append(
            f"  pointer: {pointer['path']} -> {'ok' if pointer['ok'] else 'FAILED'}"
        )
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
    check.add_argument(
        "--pin",
        default=None,
        help="read this pin instead of discovering one (for a differently named custody pin)",
    )
    check.add_argument(
        "--pin-party",
        default=None,
        help="sub-record to read inside --pin for a two-party custody pin",
    )
    check.add_argument(
        "--require-pointer",
        action="store_true",
        help=f"also require {POINTER_PATH} to name the pin, revision, workflow and job",
    )
    check.add_argument(
        "--workflow",
        default=None,
        help="conformance workflow the pointer must name (with --require-pointer)",
    )
    check.add_argument(
        "--job",
        default=None,
        help="conformance job key the pointer must name (with --require-pointer)",
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
        pin_path=Path(args.pin) if args.pin else None,
        pin_party=args.pin_party,
        require_pointer=args.require_pointer,
        pointer_workflow=args.workflow,
        pointer_job=args.job,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_format_text(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

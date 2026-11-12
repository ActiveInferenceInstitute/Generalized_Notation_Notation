#!/usr/bin/env python3
"""Check the downstream importer registry against the prose registry and origin.

``docs/downstream/importers.json`` is the machine-readable projection of
``docs/downstream/README.md``: one row per repository that applies GNN, the
file that records the upstream revision it consumes, how many specifications it
carries, and the workflow job that publishes its conformance context.

Two registries that can disagree are worse than one, so this gate makes them
one artifact in two forms:

- **Offline (default, deterministic, no network).** Schema and enum checks,
  one row per repository, docs <-> JSON parity in both directions, pin parity
  against the declared upstream release, and the spec-count contract per row
  kind (``spec`` rows carry at least one specification, ``pin-only`` rows
  carry none).
- **Online (``--online``).** Re-reads each public repository's pin file at its
  default branch and compares path, tag and commit with the registry, and
  confirms the named workflow exists and declares the named job. This is the
  drift detector; it needs ``gh`` and network access, so it is never part of
  the local gate.

A row is evidence of a declaration, not of a passing run: the registry says
what a repository pins and where its evidence lives, and the repository's own
workflow decides whether that evidence is green.

Usage::

    uv run --frozen --no-sync python scripts/check_importer_registry.py --strict
    uv run --frozen --no-sync python scripts/check_importer_registry.py --online
    uv run --frozen --no-sync python scripts/check_importer_registry.py --repo ../pymcp --name pymcp
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs" / "downstream" / "importers.json"
PROSE = ROOT / "docs" / "downstream" / "README.md"

SCHEMA = "gnn-importer-registry/v1"
KINDS = ("spec", "pin-only", "contract")
PIN_KINDS = ("file", "dependency", "content-hash")
VISIBILITIES = ("public", "private")

_ROW_RE = re.compile(r"^\|\s*\[([^\]]+)\]\([^)]+\)")


class Findings:
    """Collect findings with a level and a stable code."""

    def __init__(self) -> None:
        self.items: List[Dict[str, str]] = []

    def error(self, code: str, message: str) -> None:
        self.items.append({"level": "error", "code": code, "message": message})

    def warning(self, code: str, message: str) -> None:
        self.items.append({"level": "warning", "code": code, "message": message})

    @property
    def failed(self) -> bool:
        return any(item["level"] == "error" for item in self.items)


def load_registry(path: Path = REGISTRY) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: registry must be a JSON object")
    return payload


def prose_repositories(path: Path = PROSE) -> Set[str]:
    """Return every repository named by a table row in the prose registry."""
    names: Set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _ROW_RE.match(line)
        if match:
            names.add(match.group(1))
    return names


def _require(condition: bool, findings: Findings, code: str, message: str) -> None:
    if not condition:
        findings.error(code, message)


def check_schema(registry: Dict[str, Any], findings: Findings) -> None:
    _require(
        registry.get("schema") == SCHEMA,
        findings,
        "schema-id",
        f"registry schema is {registry.get('schema')!r}, expected {SCHEMA!r}",
    )
    upstream = registry.get("upstream") or {}
    _require(
        bool(upstream.get("tag")), findings, "upstream-tag", "upstream.tag is empty"
    )
    _require(
        isinstance(upstream.get("commit"), str)
        and len(upstream.get("commit", "")) == 40,
        findings,
        "upstream-commit",
        "upstream.commit must be a 40-character commit hash",
    )
    rows = registry.get("importers")
    _require(
        isinstance(rows, list) and bool(rows),
        findings,
        "rows",
        "importers must be a non-empty list",
    )
    if not isinstance(rows, list):
        return

    seen: Set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            findings.error("row-shape", f"row is not an object: {row!r}")
            continue
        repo = row.get("repo")
        if not isinstance(repo, str) or not repo:
            findings.error("row-repo", f"row has no repo name: {row!r}")
            continue
        if repo in seen:
            findings.error("row-duplicate", f"{repo}: listed more than once")
        seen.add(repo)
        _require(
            row.get("kind") in KINDS,
            findings,
            "row-kind",
            f"{repo}: kind {row.get('kind')!r} not in {KINDS}",
        )
        _require(
            row.get("visibility") in VISIBILITIES,
            findings,
            "row-visibility",
            f"{repo}: visibility {row.get('visibility')!r} not in {VISIBILITIES}",
        )
        specs = row.get("specs") or {}
        _require(
            isinstance(specs.get("count"), int) and specs.get("count", -1) >= 0,
            findings,
            "row-spec-count",
            f"{repo}: specs.count must be a non-negative integer",
        )
        pin = row.get("pin")
        if pin is None:
            _require(
                row.get("kind") == "contract",
                findings,
                "row-pin-missing",
                f"{repo}: only a contract row may omit its pin",
            )
            continue
        if not isinstance(pin, dict):
            findings.error("row-pin", f"{repo}: pin must be an object or null")
            continue
        _require(
            pin.get("kind") in PIN_KINDS,
            findings,
            "pin-kind",
            f"{repo}: pin.kind {pin.get('kind')!r} not in {PIN_KINDS}",
        )
        _require(
            isinstance(pin.get("path"), str) and bool(pin.get("path")),
            findings,
            "pin-path",
            f"{repo}: pin.path is empty",
        )


def check_docs_parity(registry: Dict[str, Any], findings: Findings) -> None:
    rows = registry.get("importers") or []
    registry_names = {
        row["repo"] for row in rows if isinstance(row, dict) and row.get("repo")
    }
    documented = prose_repositories()
    for missing in sorted(registry_names - documented):
        findings.error(
            "docs-missing-row",
            f"{missing}: in importers.json but not in docs/downstream/README.md",
        )
    for extra in sorted(documented - registry_names):
        findings.error(
            "registry-missing-row",
            f"{extra}: in docs/downstream/README.md but not in importers.json",
        )


def check_pin_parity(registry: Dict[str, Any], findings: Findings) -> None:
    upstream = registry.get("upstream") or {}
    tag, commit = upstream.get("tag"), upstream.get("commit")
    for row in registry.get("importers") or []:
        if not isinstance(row, dict):
            continue
        repo, kind, pin = row.get("repo"), row.get("kind"), row.get("pin")
        specs = (row.get("specs") or {}).get("count", 0)
        if kind == "spec":
            _require(
                specs >= 1,
                findings,
                "spec-count",
                f"{repo}: kind 'spec' must carry at least one specification",
            )
        if kind == "pin-only":
            _require(
                specs == 0,
                findings,
                "pin-only-count",
                f"{repo}: kind 'pin-only' must carry no specification (found {specs})",
            )
        if not isinstance(pin, dict):
            continue
        if pin.get("kind") in ("file", "dependency"):
            if pin.get("tag") is None:
                findings.warning(
                    "pin-tag-absent",
                    f"{repo}: pin records a commit but no tag; the tag is the human-readable identity",
                )
            else:
                _require(
                    pin.get("tag") == tag,
                    findings,
                    "pin-tag",
                    f"{repo}: pin tag {pin.get('tag')!r} != upstream {tag!r}",
                )
            _require(
                pin.get("commit") == commit,
                findings,
                "pin-commit",
                f"{repo}: pin commit {str(pin.get('commit'))[:10]} != upstream {str(commit)[:10]}",
            )
        conformance = row.get("conformance")
        if pin.get("kind") != "content-hash":
            _require(
                isinstance(conformance, dict)
                and conformance.get("workflow")
                and conformance.get("job"),
                findings,
                "conformance-missing",
                f"{repo}: no conformance workflow/job recorded",
            )


def _gh(*args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["gh", *args], capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def check_online(registry: Dict[str, Any], findings: Findings) -> None:
    """Re-read each public repository's pin and workflow at origin."""
    if _gh("auth", "status") is None:
        findings.warning(
            "online-unavailable",
            "gh is unavailable or unauthenticated; skipped origin re-read",
        )
        return
    for row in registry.get("importers") or []:
        if not isinstance(row, dict) or row.get("visibility") != "public":
            continue
        repo, pin = row.get("repo"), row.get("pin")
        full = f"ActiveInferenceInstitute/{repo}"
        default = _gh("api", f"repos/{full}", "--jq", ".default_branch")
        if not default:
            findings.error("origin-repo", f"{repo}: not readable at origin")
            continue
        default = default.strip()
        if isinstance(pin, dict) and pin.get("path"):
            raw = _gh(
                "api",
                f"repos/{full}/contents/{pin['path']}?ref={default}",
                "-H",
                "Accept: application/vnd.github.raw",
            )
            if raw is None:
                findings.error(
                    "origin-pin", f"{repo}: {pin['path']} not found at origin/{default}"
                )
                continue
            if pin.get("kind") == "dependency":
                # A dependency pin is a declaration inside a build file, not a
                # JSON document: the revision must appear in the file's text.
                if pin.get("commit") and pin["commit"] not in raw:
                    findings.error(
                        "origin-pin-drift",
                        f"{repo}: {pin['path']} does not name {str(pin.get('commit'))[:10]} at origin",
                    )
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                if pin.get("kind") != "content-hash":
                    findings.error(
                        "origin-pin-json",
                        f"{repo}: {pin['path']} is not JSON at origin",
                    )
                continue
            if pin.get("kind") in ("file", "dependency") and isinstance(payload, dict):
                commit = payload.get("commit") or payload.get("revision")
                tag = payload.get("tag") or payload.get("release")
                if commit != pin.get("commit"):
                    findings.error(
                        "origin-pin-drift",
                        f"{repo}: origin pin commit {str(commit)[:10]} != registry {str(pin.get('commit'))[:10]}",
                    )
                if tag != pin.get("tag"):
                    findings.error(
                        "origin-pin-drift",
                        f"{repo}: origin pin tag {tag!r} != registry {pin.get('tag')!r}",
                    )
        conformance = row.get("conformance")
        if isinstance(conformance, dict) and conformance.get("workflow"):
            raw = _gh(
                "api",
                f"repos/{full}/contents/{conformance['workflow']}?ref={default}",
                "-H",
                "Accept: application/vnd.github.raw",
            )
            if raw is None:
                findings.error(
                    "origin-workflow",
                    f"{repo}: {conformance['workflow']} not found at origin",
                )
            elif conformance.get("job") and f"  {conformance['job']}:" not in raw:
                findings.error(
                    "origin-job",
                    f"{repo}: job {conformance['job']!r} not declared in {conformance['workflow']}",
                )


def check_local_repo(
    registry: Dict[str, Any], name: str, path: Path, findings: Findings
) -> None:
    """Check one local checkout with the shipped reader and compare it to its row."""
    rows = {
        row.get("repo"): row
        for row in registry.get("importers") or []
        if isinstance(row, dict)
    }
    row = rows.get(name)
    if row is None:
        findings.error("local-unknown", f"{name}: not in the registry")
        return
    from gnn.importers import check_importer

    upstream = registry.get("upstream") or {}
    pin = row.get("pin") or {}
    report = check_importer(
        path,
        expected_commit=upstream.get("commit")
        if pin.get("kind") in ("file", "dependency")
        else None,
        expected_tag=upstream.get("tag")
        if pin.get("kind") in ("file", "dependency")
        else None,
        require_pin=pin.get("kind") is not None,
    )
    for item in report["findings"]:
        if item["level"] == "error":
            findings.error(f"local-{item['code']}", f"{name}: {item['message']}")
        else:
            findings.warning(f"local-{item['code']}", f"{name}: {item['message']}")
    declared = (row.get("specs") or {}).get("count")
    if declared is not None and report["specs"]["count"] != declared:
        findings.error(
            "local-spec-count",
            f"{name}: checkout carries {report['specs']['count']} specifications, registry says {declared}",
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None
    )
    parser.add_argument(
        "--strict", action="store_true", help="fail on warnings as well as errors"
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="re-read every public repository at origin",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=None,
        help="also check this local importer checkout",
    )
    parser.add_argument(
        "--name", default=None, help="registry name of the --repo checkout"
    )
    parser.add_argument("--json", action="store_true", help="emit findings as JSON")
    args = parser.parse_args(argv)

    findings = Findings()
    try:
        registry = load_registry()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"importer registry: {exc}", file=sys.stderr)
        return 1

    check_schema(registry, findings)
    check_docs_parity(registry, findings)
    check_pin_parity(registry, findings)
    if args.online:
        check_online(registry, findings)
    if args.repo is not None:
        if not args.name:
            findings.error("local-name", "--repo requires --name")
        else:
            check_local_repo(registry, args.name, args.repo, findings)

    failed = findings.failed or (
        args.strict and any(i["level"] == "warning" for i in findings.items)
    )
    if args.json:
        print(
            json.dumps(
                {"ok": not failed, "findings": findings.items}, indent=2, sort_keys=True
            )
        )
    else:
        rows = len(registry.get("importers") or [])
        print(
            f"importer registry: {rows} repositories, upstream {(registry.get('upstream') or {}).get('tag')}"
        )
        for item in findings.items:
            print(f"  [{item['level']}] {item['code']}: {item['message']}")
        print("  verdict: " + ("ok" if not failed else "FAILED"))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

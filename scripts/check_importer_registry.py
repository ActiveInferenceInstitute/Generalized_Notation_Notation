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
  between each row's tag and commit as this repository's tags resolve them,
  the registry's declared upstream release against its own tag, and the
  spec-count contract per row kind (``spec`` rows carry at least one
  specification, ``pin-only`` rows carry none). Rows may hold different
  releases: the registry records the release each member actually declares,
  not one estate-wide pin.
- **Online (``--online``).** Re-reads each public repository's pin file at its
  default branch and compares path, tag and commit with the registry, confirms
  the named workflow exists and declares the named job, and confirms the row's
  repository publishes the ``.github/GNN.md`` pointer that records what the job
  parses. This is the drift detector; it needs ``gh`` and network access, so it
  is never part of the local gate.

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

#: The one command a member runs instead of carrying its own pin reader.
CANONICAL_IMPORTER_ARM = "python -m gnn.importers check --repo ."

#: How a member reaches the canonical reader. ``direct`` is the only kind that
#: needs no member-side copy; ``wrapper`` is for a gate whose language cannot
#: import it and records the file that shells out; ``upstream`` is the reader's
#: own repository, which is not an importer of itself.
INSTALL_KINDS = ("direct", "wrapper", "upstream")

#: The documentation hub every member publishes, and the flattened indexes some
#: add beside it. A member's declared hub must be one of these, so a typo in the
#: contract cannot pass as a path.
DOCS_HUBS = ("docs/README.md",)
DOCS_FLAT_INDEXES = ("docs/INDEX.md", "docs/documentation-index.md")

#: The template's own manuscript producer: the filename the docxology render
#: pipeline invokes before a PDF render.
CANONICAL_MANUSCRIPT_PRODUCER = "scripts/z_generate_manuscript_variables.py"

#: Markers that bound the contract's member table, so prose parity is a set
#: comparison rather than a scan of every repository name in the document.
MEMBERS_START = "<!-- gnn-fep-members:start -->"
MEMBERS_END = "<!-- gnn-fep-members:end -->"

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


def _tag_commit(tag: str) -> Optional[str]:
    """Return the commit a release tag names in this repository, or ``None``.

    The registry records a pin per member, and members legitimately hold
    different releases while the estate cuts over. The offline check therefore
    resolves each pin's tag against this repository's own tags instead of
    requiring every row to equal the newest release.
    """
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "rev-parse",
                "--verify",
                f"refs/tags/{tag}^{{commit}}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


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
    if tag:
        _require(
            _tag_commit(tag) == commit,
            findings,
            "upstream-tag-commit",
            f"upstream tag {tag!r} does not resolve to declared commit {str(commit)[:10]}",
        )
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
            pinned_tag = pin.get("tag")
            if pinned_tag is None:
                findings.warning(
                    "pin-tag-absent",
                    f"{repo}: pin records a commit but no tag; the tag is the human-readable identity",
                )
            else:
                resolved = _tag_commit(pinned_tag)
                _require(
                    resolved is not None,
                    findings,
                    "pin-tag-unknown",
                    f"{repo}: pin tag {pinned_tag!r} is not a tag in this repository",
                )
                if resolved is not None:
                    _require(
                        pin.get("commit") == resolved,
                        findings,
                        "pin-commit",
                        f"{repo}: pin tag {pinned_tag!r} resolves to "
                        f"{resolved[:10]}, not {str(pin.get('commit'))[:10]}",
                    )
        conformance = row.get("conformance")
        if pin.get("kind") != "content-hash":
            hosted = (
                isinstance(conformance, dict)
                and conformance.get("workflow")
                and conformance.get("job")
            )
            local = (
                isinstance(conformance, dict)
                and conformance.get("name")
                and conformance.get("local_arm")
            )
            _require(
                bool(hosted or local),
                findings,
                "conformance-missing",
                f"{repo}: no hosted conformance workflow/job or local arm recorded",
            )


def contract_members(path: Path) -> Set[str]:
    """Return the repositories the contract's marked member table names."""
    text = path.read_text(encoding="utf-8")
    if MEMBERS_START not in text or MEMBERS_END not in text:
        raise ValueError(f"{path}: missing {MEMBERS_START} / {MEMBERS_END} markers")
    block = text.split(MEMBERS_START, 1)[1].split(MEMBERS_END, 1)[0]
    names: Set[str] = set()
    for line in block.splitlines():
        match = _ROW_RE.match(line)
        if match:
            names.add(match.group(1))
    return names


def check_workflow_groups(registry: Dict[str, Any], findings: Findings) -> None:
    """Check each workflow group against the importer rows it claims.

    A group records, for one set of repositories, how each member reaches the
    canonical pin reader. The value is that three separate hand-edited
    artifacts — the contract prose, this JSON, and each member's row — agree:
    a member the contract names but the rows do not know, a pin path that
    drifted from the row, or an ``arm`` that is not the canonical command is
    exactly the divergence this check exists to refuse.
    """
    groups = registry.get("workflow_groups")
    if groups is None:
        return
    if not isinstance(groups, dict) or not groups:
        findings.error("group-shape", "workflow_groups must be a non-empty object")
        return
    rows = {
        row.get("repo"): row
        for row in registry.get("importers") or []
        if isinstance(row, dict)
    }
    for group_id, group in sorted(groups.items()):
        if not isinstance(group, dict):
            findings.error("group-shape", f"{group_id}: must be an object")
            continue
        contract = group.get("contract")
        contract_path: Optional[Path] = None
        if not isinstance(contract, str) or not contract:
            findings.error("group-contract", f"{group_id}: contract is empty")
        else:
            contract_path = ROOT / contract
            if not contract_path.is_file():
                findings.error(
                    "group-contract", f"{group_id}: {contract} does not exist"
                )
                contract_path = None
        members = group.get("members")
        if not isinstance(members, list) or not members:
            findings.error(
                "group-members", f"{group_id}: members must be a non-empty list"
            )
            continue

        seen: Set[str] = set()
        for member in members:
            if not isinstance(member, dict):
                findings.error(
                    "group-member-shape", f"{group_id}: member is not an object"
                )
                continue
            repo = member.get("repo")
            if not isinstance(repo, str) or not repo:
                findings.error("group-member-repo", f"{group_id}: member has no repo")
                continue
            if repo in seen:
                findings.error(
                    "group-member-duplicate", f"{group_id}: {repo} listed twice"
                )
            seen.add(repo)
            row = rows.get(repo)
            role = member.get("role")
            _require(
                role in ("upstream", "importer"),
                findings,
                "group-member-role",
                f"{repo}: role {role!r} not in ('upstream', 'importer')",
            )
            if row is None:
                upstream = (registry.get("upstream") or {}).get("repository") or ""
                if role == "upstream" and repo == upstream.rsplit("/", 1)[-1]:
                    _require(
                        member.get("pin") is None and member.get("pointer") is None,
                        findings,
                        "group-upstream-pin",
                        f"{repo}: the reader's own repository carries no importer pin or pointer",
                    )
                    continue
                findings.error(
                    "group-member-unknown",
                    f"{group_id}: {repo} is not an importer row",
                )
                continue
            install = member.get("install")
            _require(
                install in INSTALL_KINDS,
                findings,
                "group-member-install",
                f"{repo}: install {install!r} not in {INSTALL_KINDS}",
            )
            _require(
                member.get("arm") == CANONICAL_IMPORTER_ARM,
                findings,
                "group-member-arm",
                f"{repo}: arm {member.get('arm')!r} is not the canonical "
                f"{CANONICAL_IMPORTER_ARM!r}",
            )
            if install == "wrapper":
                wrapper = member.get("wrapper")
                _require(
                    isinstance(wrapper, str) and bool(wrapper),
                    findings,
                    "group-member-wrapper",
                    f"{repo}: install 'wrapper' must name the wrapper file",
                )
                _require(
                    isinstance(member.get("reason"), str)
                    and bool(member.get("reason")),
                    findings,
                    "group-member-reason",
                    f"{repo}: install 'wrapper' must record why it cannot call the reader",
                )
            expected_pin = (row.get("pin") or {}).get("path")
            _require(
                member.get("pin") == expected_pin,
                findings,
                "group-member-pin",
                f"{repo}: contract pin {member.get('pin')!r} != row pin {expected_pin!r}",
            )
            hub = member.get("docs_index")
            _require(
                hub is None or hub in DOCS_HUBS,
                findings,
                "group-member-docs",
                f"{repo}: docs_index {hub!r} not in {DOCS_HUBS} or null",
            )
            flat = member.get("docs_flat_index")
            _require(
                flat is None or flat in DOCS_FLAT_INDEXES,
                findings,
                "group-member-flat-index",
                f"{repo}: docs_flat_index {flat!r} not in {DOCS_FLAT_INDEXES} or null",
            )
            manuscript = member.get("manuscript")
            _require(
                manuscript is None
                or (
                    isinstance(manuscript, str)
                    and manuscript.endswith(".py")
                    and "/" in manuscript
                ),
                findings,
                "group-member-manuscript",
                f"{repo}: manuscript {manuscript!r} must be a script path or null",
            )
            gate = member.get("gate")
            _require(
                gate is None or isinstance(gate, str),
                findings,
                "group-member-gate",
                f"{repo}: gate must be a command string or null",
            )

        if contract_path is None:
            continue
        try:
            documented = contract_members(contract_path)
        except (OSError, ValueError) as exc:
            findings.error("group-contract-members", f"{group_id}: {exc}")
            continue
        for missing in sorted(seen - documented):
            findings.error(
                "group-member-undocumented",
                f"{group_id}: {missing} is in the JSON but not the contract table",
            )
        for extra in sorted(documented - seen):
            findings.error(
                "group-member-unregistered",
                f"{group_id}: {extra} is in the contract table but not the JSON",
            )


class GhError(RuntimeError):
    """A ``gh`` call failed for a reason other than a missing path."""


def _gh(*args: str) -> Optional[str]:
    """Run one ``gh`` call. ``None`` means the path is absent at origin.

    A transient failure (network, 5xx, rate limit, timeout) must not be reported
    as a missing file: it raises :class:`GhError`, and the caller degrades that
    repository to an ``online-unavailable`` warning instead of a false drift
    error.
    """
    try:
        result = subprocess.run(
            ["gh", *args], capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GhError(f"gh could not run: {exc}") from exc
    if result.returncode == 0:
        return result.stdout
    stderr = (result.stderr or "") + (result.stdout or "")
    if "HTTP 404" in stderr or "Not Found" in stderr:
        return None
    detail = (result.stderr or result.stdout or "").strip().splitlines()
    raise GhError(detail[-1][:200] if detail else "gh failed")


def check_online(registry: Dict[str, Any], findings: Findings) -> None:
    """Re-read each public repository's pin, workflow job and pointer at origin."""
    try:
        authenticated = _gh("auth", "status") is not None
    except GhError:
        authenticated = False
    if not authenticated:
        findings.warning(
            "online-unavailable",
            "gh is unavailable or unauthenticated; skipped origin re-read",
        )
        return
    for row in registry.get("importers") or []:
        if not isinstance(row, dict) or row.get("visibility") != "public":
            continue
        try:
            _check_public_row_online(row, findings)
        except GhError as exc:
            findings.warning("online-unavailable", f"{row.get('repo')}: {exc}")


def _check_public_row_online(row: Dict[str, Any], findings: Findings) -> None:
    """Compare one public row's pin, conformance job and pointer with origin."""
    repo, pin = row.get("repo"), row.get("pin")
    full = f"ActiveInferenceInstitute/{repo}"
    default = _gh("api", f"repos/{full}", "--jq", ".default_branch")
    if not default:
        findings.error("origin-repo", f"{repo}: not readable at origin")
        return
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
        elif pin.get("kind") == "dependency":
            # A dependency pin is a declaration inside a build file, not a
            # JSON document: the revision must appear in the file's text.
            if pin.get("commit") and pin["commit"] not in raw:
                findings.error(
                    "origin-pin-drift",
                    f"{repo}: {pin['path']} does not name {str(pin.get('commit'))[:10]} at origin",
                )
        else:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                if pin.get("kind") != "content-hash":
                    findings.error(
                        "origin-pin-json",
                        f"{repo}: {pin['path']} is not JSON at origin",
                    )
            else:
                if pin.get("kind") in ("file", "dependency") and isinstance(
                    payload, dict
                ):
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
    if conformance:
        pointer = _gh(
            "api",
            f"repos/{full}/contents/.github/GNN.md?ref={default}",
            "-H",
            "Accept: application/vnd.github.raw",
        )
        if pointer is None:
            findings.error(
                "origin-pointer",
                f"{repo}: .github/GNN.md not found at origin/{default}",
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
        pin_path=Path(pin["path"]) if pin.get("path") else None,
        pin_party=pin.get("party"),
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
    check_workflow_groups(registry, findings)
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
        pins = [row.get("pin") or {} for row in registry.get("importers") or []]
        canonical = sum(1 for pin in pins if pin.get("schema") == "gnn-importer-pin/v1")
        declared = sum(1 for pin in pins if pin.get("kind") in ("file", "dependency"))
        print(
            f"  pin schema: {canonical} of {declared} pins declare gnn-importer-pin/v1"
        )
        groups = registry.get("workflow_groups") or {}
        for group_id, group in sorted(groups.items()):
            members = (group or {}).get("members") or []
            direct = sum(1 for m in members if m.get("install") == "direct")
            print(
                f"  group {group_id}: {len(members)} members, {direct} call the reader directly"
            )
        for item in findings.items:
            print(f"  [{item['level']}] {item['code']}: {item['message']}")
        print("  verdict: " + ("ok" if not failed else "FAILED"))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

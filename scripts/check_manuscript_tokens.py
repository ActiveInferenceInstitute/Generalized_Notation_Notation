#!/usr/bin/env python3
"""Manuscript integrity gate for GeneralizedNotationNotation.

Deterministic checks run before/after rendering the token-injected manuscript:

1. **Unknown tokens** — every ``{{TOKEN}}`` used in a rendered section must be a key
   emitted by ``gnn.manuscript.generate_variables``. An unknown token would
   survive substitution and render literally in the PDF.
2. **Dangling citations** — every Pandoc ``[@key]`` must resolve to an entry in
   ``manuscript/references.bib``.
3. **Hard-coded counts** — bare literals equal to a high-value producer count (e.g. the
   test-file count, MCP-tool count) are flagged: they should be ``{{TOKEN}}`` instead.
4. **Hard-coded step numbers** — a bare integer in a step phrase — "step 3",
   "Steps 5 and 6", and the plural forms where the number leads ("25 steps",
   "25-step pipeline") — that equals a ``GNN_STEP_*`` token value is flagged
   regardless of size. Step numbers are below the ``_HARDCODE_MIN`` floor the
   count scan uses, so they were invisible to check 3 while the producer
   emitted a token for every one of them.
5. **Malformed cross-references** — a pandoc-crossref marker glued to a stray prefix
   character (``+@fig:x``) still resolves, so no reference gate catches it, but the
   prefix renders literally into the PDF ("+fig. 2 depicts...").
6. **config.yaml drift** — ``manuscript/config.yaml`` is never token-substituted, so its
   ``version:`` / ``date:`` literals are compared against the producer's values.
7. **Figure accessibility registry** — every ``{#fig:...}`` label a section declares must
   have an ``output/figures/figure_registry.json`` entry with alt text and an existing
   image file. The template's own ``validate_figure_registry`` runs at the *validation*
   stage, not the render stage, so a render could (and did) ship a PDF with the registry
   absent entirely. This is the in-repo gate that does not depend on the template.
8. **Repository path claims** — every ``input/...`` literal must exist on disk, and where
   prose names a manifest family beside such a literal, the literal must match that
   family's declared ``target_dir``. Every other check compares a *number* against the
   producer; a path is not a number, so all of them stayed green while a commit moved a
   family's ``target_dir`` and three prose sites went on naming the old directory.
9. **Unresolved tokens in the shipped copy** — every ``{{TOKEN}}`` surviving in
   ``output/manuscript/**`` after hydration fails ``--strict``: that is a token
   the producer does not emit, rendered verbatim into the PDF.
10. **Cross-reference labels** — every ``@fig:...``/``@tbl:...`` reference must
   name a label a manuscript section declares (``{#fig:...}``/``{#tbl:...}``);
   a reference to a label that does not exist renders as "fig. ???".

Under ``--strict`` the gate additionally proves the committed artifacts agree
with the producer at HEAD:

* ``token_checksum(generate_variables(HEAD))`` must equal the checksum of the
  committed ``output/data/manuscript_variables.json``;
* every registry ``png_sha256`` must be the digest of the PNG on disk;
* every registry ``source_sha256`` digest must equal the file's bytes at HEAD
  on a clean tree (dirty trees compare against the working tree instead,
  since HEAD cannot describe uncommitted sources) — a figure built from
  working-tree data the prose does not describe (``fig:pipeline`` /
  ``fig:family_matrix`` / ``fig:backend_matrix`` read STEP_INDEX.md / the
  manifest / the registry directly) fails here.
undeclared cross-reference, config.yaml drift, contradicted path claim, or a
``GNN_GIT_COMMIT`` of ``unknown`` is found (hard gate). Hard-coded count and
step-number findings and the strict artifact-agreement checks are reported as
warnings unless ``--strict`` is passed.

Usage:
    python scripts/check_manuscript_tokens.py [--strict] [--base-ref origin/main]

PR base mode evaluates immutable merge-base evidence with this implementation.
Only unchanged findings with identical supporting bytes warn; new drift fails.
No base argument keeps main and scheduled gates strict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from gnn.manuscript import (  # noqa: E402
    RepositorySnapshot,
    config_metadata_drift,
    generate_variables,
    load_variables,
    token_checksum,
)
from gnn.manuscript.gate_baseline import compare_findings  # noqa: E402
from gnn.manuscript.substitution import EXCLUDED_DOC_FILENAMES, TOKEN_RE  # noqa: E402
from gnn.manuscript.variables import _families  # noqa: E402

_EXCLUDED = set(EXCLUDED_DOC_FILENAMES)
_TOKEN_RE = TOKEN_RE
_CITE_RE = re.compile(r"@([A-Za-z][\w:-]+)")
_BIB_KEY_RE = re.compile(r"^@\w+\{([^,]+),", re.MULTILINE)
# Counts small enough to appear coincidentally (step numbers, dims) are not flagged.
_HARDCODE_MIN = 10
# "step 3", "Steps 5 and 6", "steps 11 and 12" — the integers a GNN_STEP_* token owns —
# and the plural forms where the number leads: "25 steps", "25-step pipeline".
_STEP_PHRASE_RE = re.compile(
    r"(?:[Ss]teps?\s+(?P<lead>\d+(?:\s*(?:,|and|to|through|–|-)\s*\d+)*))"
    r"|(?:\b(?P<trail>\d+)[\s-]*steps?\b)"
)
_INT_RE = re.compile(r"\d+")
# A crossref marker preceded by a character that is neither "[" nor "!" nor
# whitespace: the marker resolves, so the reference gate stays green, and the
# stray prefix renders literally.
_MALFORMED_XREF_RE = re.compile(r"(?<=[^\[!\s])@(?:fig|tbl|eq|sec):[\w:-]+")


def _section_files(manuscript_dir: Path) -> list[Path]:
    return [p for p in sorted(manuscript_dir.glob("*.md")) if p.name not in _EXCLUDED]


_FIG_LABEL_RE = re.compile(r"\{#(fig:[\w:-]+)")


def _figure_registry_issues(manuscript_dir: Path, sections: list[Path]) -> list[str]:
    """Check every declared figure label against the accessibility registry."""
    declared: dict[str, str] = {}
    for path in sections:
        for label in _FIG_LABEL_RE.findall(path.read_text(encoding="utf-8")):
            declared.setdefault(label, path.name)
    if not declared:
        return []
    registry_path = _PROJECT_ROOT / "output" / "figures" / "figure_registry.json"
    if not registry_path.is_file():
        return [
            f"{registry_path.relative_to(_PROJECT_ROOT)} is missing while "
            f"{len(declared)} figure(s) are referenced — run "
            "python -m scripts.manuscript_build_figures"
        ]
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    records = payload.get("figures", payload) if isinstance(payload, dict) else payload
    by_label = {str(rec.get("label")): rec for rec in records if isinstance(rec, dict)}
    issues: list[str] = []
    for label, section in sorted(declared.items()):
        record = by_label.get(label)
        if record is None:
            issues.append(f"{section}: {label} has no figure_registry.json entry")
            continue
        if not str(record.get("alt_text", "")).strip():
            issues.append(f"figure_registry.json: {label} has no alt_text")
        filename = str(record.get("filename", ""))
        # A hand-edited registry may carry an absolute or pathy filename;
        # normalize it into output/figures/ before joining instead of letting
        # an absolute path bypass the parent directory entirely.
        image = Path(filename) if filename else Path()
        if image.is_absolute():
            try:
                image = image.relative_to(registry_path.parent)
            except ValueError:
                issues.append(
                    f"figure_registry.json: {label} image {filename!r} is "
                    "outside output/figures/"
                )
                continue
        if not filename or not (registry_path.parent / image).is_file():
            issues.append(
                f"figure_registry.json: {label} image {filename!r} is missing"
            )
    return issues


# {#fig:...}/{#tbl:...} declarations a section makes, and the @fig:/@tbl: marks
# that reference them. A reference to a label nothing declares renders as
# "fig. ???" / "table ???" in the PDF and no other rule sees it.
_DECLARED_LABEL_RE = re.compile(r"\{#((?:fig|tbl):[\w:-]+)")
_XREF_RE = re.compile(r"@(?:fig|tbl):[\w:-]+")


def _display(path: Path) -> str:
    """Repo-relative display name, or the absolute form when outside the repo
    (fixture trees in tests live in tmp_path, not under _PROJECT_ROOT)."""
    try:
        return str(path.relative_to(_PROJECT_ROOT))
    except ValueError:
        return str(path)


def _crossref_issues(sections: list[Path], variables: Mapping[str, str]) -> list[str]:
    """Every ``@fig:``/``@tbl:`` reference must name a declared label.

    Declarations live in two places: the section markdown (``{#fig:...}`` on
    an embed, ``{#tbl:...}`` on a caption) and the producer's table tokens —
    ``{{GNN_STEP_TABLE}}`` and friends arrive with their ``{#tbl:...}``
    caption attached, so the declared set is the union of both.
    """
    declared: set[str] = set()
    referenced: set[tuple[str, str]] = set()
    for path in sections:
        body = _strip_code(path.read_text(encoding="utf-8"))
        declared.update(_DECLARED_LABEL_RE.findall(body))
        referenced.update((path.name, m.group(0)) for m in _XREF_RE.finditer(body))
    # The producer's table tokens arrive with their {#tbl:...} caption
    # attached; the tables they carry are declared labels too.
    for value in variables.values():
        declared.update(_DECLARED_LABEL_RE.findall(str(value)))
    return [
        f"{name}: {mark} names no declared {{#{mark[1:]}}} label"
        for name, mark in sorted(referenced)
        if mark[1:] not in declared
    ]


def _unresolved_output_tokens(output_dir: Path) -> list[str]:
    """Unresolved ``{{TOKEN}}`` in the hydrated copies under output/manuscript/.

    The producer emits every token the sections use; one it does not emit
    survives substitution and reaches the PDF verbatim. This is the scan of
    the thing that actually ships, not the sources.
    """
    if not output_dir.is_dir():
        return []
    issues: list[str] = []
    for path in sorted(output_dir.rglob("*.md")):
        body = _strip_code(path.read_text(encoding="utf-8"))
        for tok in _TOKEN_RE.findall(body):
            issues.append(
                f"{_display(path)}: unresolved {{{{{tok}}}}} reached the rendered copy"
            )
    return issues


def _section_hardcode_issues(
    name: str,
    no_tokens: str,
    hardcode_targets: Mapping[str, str],
    step_literals: Mapping[str, str],
) -> list[str]:
    """Bare-literal and step-phrase findings for one section body.

    ``no_tokens`` must already be code-stripped and have ``{{TOKEN}}`` removed,
    so a token's own value never reads as a hard-coded literal.
    """
    issues: list[str] = []
    for value, key in hardcode_targets.items():
        if re.search(rf"(?<!\d){re.escape(value)}(?!\d)", no_tokens):
            issues.append(f"{name}: literal {value} should be {{{{{key}}}}}")
    for phrase in _STEP_PHRASE_RE.finditer(no_tokens):
        numbers = phrase.group("lead") or phrase.group("trail")
        for number in _INT_RE.findall(numbers):
            step_key = step_literals.get(number)
            if step_key:
                issues.append(
                    f"{name}: step literal {number} in {phrase.group(0)!r} "
                    f"should be {{{{{step_key}}}}}"
                )
    return issues


def _strip_code(text: str) -> str:
    """Remove fenced and inline code so we do not flag tokens/numbers in code."""
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]*`", "", text)
    return text


# --- Rule 8: repository path claims -----------------------------------------
#
# Every check above compares a *number* against the producer. A path is not a
# number, so all of them stayed green while a commit repointed the `multiagent`
# family's target_dir from input/multi_agent_models into input/gnn_files/ and
# three prose sites went on naming the old directory. That is the gap this rule
# closes, in the same shape as the rest of the gate: the manifest is the source
# of truth and the manuscript is checked against it.
#
# Two independent checks:
#   (a) existence — an `input/...` literal must name something on disk;
#   (b) family pairing — where prose names a manifest family and an `input/...`
#       literal in the same breath, the literal must be that family's declared
#       target_dir, an ancestor of it (`input/gnn_files` covering a family), or
#       a path inside it.
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_INPUT_PATH_RE = re.compile(r"^input/[\w./-]+$")
# How far from a family mention a path literal is still read as a claim about
# that family. Wide enough to span "the `x` family, whose target directory is
# `input/y`", narrow enough not to reach across a paragraph.
_CLAIM_WINDOW = 160


def _code_spans(text: str) -> list[tuple[int, str]]:
    """``(offset, content)`` for each inline code span, fenced blocks removed.

    Fenced blocks are dropped because they hold reproduction *commands*, where a
    path is an argument rather than a claim about where a family lives.
    """
    without_fences = re.sub(
        r"```.*?```", lambda m: " " * len(m.group(0)), text, flags=re.DOTALL
    )
    return [(m.start(1), m.group(1)) for m in _INLINE_CODE_RE.finditer(without_fences)]


def _path_claim_issues(sections: list[Path], families: list[dict]) -> list[str]:
    """Check `input/...` literals against the model-family manifest."""
    target_dirs = {
        str(f.get("name", "")): str(f.get("target_dir", "")).rstrip("/")
        for f in families
        if f.get("name") and f.get("target_dir")
    }
    issues: list[str] = []
    for path in sections:
        raw = path.read_text(encoding="utf-8")
        spans = _code_spans(raw)
        paths = [
            (offset, content.rstrip("/"))
            for offset, content in spans
            if _INPUT_PATH_RE.match(content.rstrip("/"))
        ]
        for offset, literal in paths:
            if not (_PROJECT_ROOT / literal).exists():
                issues.append(
                    f"{path.name}: `{literal}` does not exist in the repository"
                )
        for offset, content in spans:
            family = content.strip()
            declared = target_dirs.get(family)
            if declared is None:
                continue
            # Only a mention that actually calls it a family is a family claim;
            # `continuous` and `structured` are ordinary words otherwise.
            around = raw[max(0, offset - 40) : offset + len(content) + 40]
            if "famil" not in around.lower():
                continue
            # Only the NEAREST path literal is read as this family's location.
            # Taking every literal in the window swept in unrelated siblings
            # ("`input/recursive_models/` is a reserved directory") that happen
            # to share a sentence with a family mention.
            near = [
                (abs(p_offset - offset), literal)
                for p_offset, literal in paths
                if abs(p_offset - offset) <= _CLAIM_WINDOW
            ]
            if not near:
                continue
            _, literal = min(near)
            related = (
                literal == declared
                or literal.startswith(declared + "/")
                or declared.startswith(literal + "/")
            )
            if not related:
                issues.append(
                    f"{path.name}: the `{family}` family is named beside "
                    f"`{literal}`, but the manifest declares its target_dir "
                    f"as `{declared}`"
                )
    return sorted(set(issues))


def _git_blob_digest(rel: str) -> str | None:
    """sha256 of *rel*'s bytes at HEAD, or None when HEAD lacks the file."""
    result = subprocess.run(
        ["git", "show", f"HEAD:{rel}"],
        cwd=str(_PROJECT_ROOT),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return hashlib.sha256(result.stdout).hexdigest()


def _strip_volatile_tokens(variables: Mapping[str, str]) -> dict[str, str]:
    """Copy of a token map without commit-varying tokens (checksum input)."""
    return {k: v for k, v in variables.items() if k != "GNN_GIT_COMMIT"}


def _committed_variables_issue(live: Mapping[str, str], variables_json: Path) -> str:
    """Failure message when the committed token map is not the producer's at HEAD.

    SC-3: the render hydrates prose from the committed JSON while nothing
    compared it to a fresh ``generate_variables(HEAD)`` — a consistently-stale
    map (stale JSON, stale PNGs, stale registry) passed every gate together.
    The checksum pins the committed map to the producer.
    """
    if not variables_json.is_file():
        return (
            f"{_display(variables_json)} is missing — run "
            "scripts/z_generate_manuscript_variables.py"
        )
    try:
        committed = load_variables(variables_json)
    except (OSError, ValueError) as exc:
        return f"{_display(variables_json)} is unreadable: {exc}"
    # GNN_GIT_COMMIT cannot participate in the comparison: a committed map
    # can never record the hash of the commit that carries it, so including
    # it would make the gate fail on every commit that moves HEAD. The
    # receipt's ``counts_describe_commit`` pins the commit separately; the
    # checksum covers the commit-stable counts (the rot SC-3 guards against).
    live_sum = token_checksum(_strip_volatile_tokens(live))
    committed_sum = token_checksum(_strip_volatile_tokens(committed))
    if live_sum != committed_sum:
        return (
            "stale: committed token_checksum "
            f"{committed_sum[:12]} != producer-at-HEAD {live_sum[:12]} — run "
            "scripts/z_generate_manuscript_variables.py"
        )
    return ""


def _working_tree_is_clean() -> bool:
    """True when ``git status --porcelain`` reports nothing (or git is absent
    — the digest comparison then degrades to the working tree, which is all
    a tarball checkout has)."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    return result.returncode == 0 and not result.stdout.strip()


def _figure_artifact_issues(
    snapshot: RepositorySnapshot | None = None,
) -> tuple[list[str], list[str]]:
    """``(png_issues, source_issues)`` comparing figure_registry.json to disk/HEAD.

    ``png_sha256`` must be the digest of the PNG file on disk (the artifact a
    rebuild must reproduce). Every ``source_sha256`` digest must equal the
    data the prose describes: on a clean tree that is the file's bytes at HEAD
    (SC-21 — the three figures whose generators read STEP_INDEX.md / the
    family manifest / the framework registry from the working tree must be
    built from data the prose's HEAD snapshot also describes); on a dirty tree
    HEAD cannot describe the current sources, so the reference is the working
    tree instead, and the check degrades to "figures must match the sources
    now on disk".
    """
    clean = snapshot is not None or _working_tree_is_clean()
    registry_path = _PROJECT_ROOT / "output" / "figures" / "figure_registry.json"
    if not registry_path.is_file():
        return [
            "output/figures/figure_registry.json is missing — run "
            "python -m scripts.manuscript_build_figures"
        ], []
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    records = payload.get("figures", payload) if isinstance(payload, dict) else payload
    png_issues: list[str] = []
    source_issues: list[str] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        label = str(rec.get("label"))
        filename = str(rec.get("filename", ""))
        png = Path(filename) if filename else Path()
        if png.is_absolute():
            try:
                png = png.relative_to(registry_path.parent)
            except ValueError:
                png_issues.append(
                    f"{label}: registry image {filename!r} is outside output/figures/"
                )
                continue
        png_file = registry_path.parent / png
        digest = str(rec.get("png_sha256", ""))
        if not filename or not png_file.is_file():
            png_issues.append(
                f"{label}: {filename!r} not built — run "
                "python -m scripts.manuscript_build_figures"
            )
        elif digest and hashlib.sha256(png_file.read_bytes()).hexdigest() != digest:
            png_issues.append(
                f"{label}: committed {filename} is not the figure the registry "
                "records — rebuild (python -m scripts.manuscript_build_figures) "
                "and commit both"
            )
        for rel, recorded in sorted((rec.get("source_sha256") or {}).items()):
            if clean:
                blob = (
                    hashlib.sha256(snapshot.read_bytes(str(rel))).hexdigest()
                    if snapshot is not None and snapshot.exists(str(rel))
                    else None
                    if snapshot is not None
                    else _git_blob_digest(str(rel))
                )
            else:
                source = _PROJECT_ROOT / str(rel)
                blob = (
                    hashlib.sha256(source.read_bytes()).hexdigest()
                    if source.is_file()
                    else None
                )
            if blob is None:
                source_issues.append(
                    f"{label}: source {rel} cannot be read from audited evidence"
                )
            elif blob != str(recorded):
                source_issues.append(
                    f"{label}: source {rel} differs from the "
                    "build-time digest — rebuild the figures on a commit the "
                    "prose describes"
                )
    return png_issues, source_issues


def _strict_artifact_checks(variables: Mapping[str, str], ok: bool) -> bool:
    """The --strict artifact-agreement checks; prints findings, returns verdict."""
    committed_issue = _committed_variables_issue(
        variables, _PROJECT_ROOT / "output" / "data" / "manuscript_variables.json"
    )
    if committed_issue:
        ok = False
        print(f"\nCOMMITTED TOKEN MAP: ✗ {committed_issue}")
    png_issues, source_issues = _figure_artifact_issues()
    if png_issues:
        ok = False
        print(f"\nFIGURE PNG DIGESTS ({len(png_issues)}):")
        for p in png_issues:
            print(f"  ✗ {p}")
    if source_issues:
        ok = False
        print(f"\nFIGURE SOURCE DIGESTS vs HEAD ({len(source_issues)}):")
        for s in source_issues:
            print(f"  ✗ {s}")
    output_tokens = _unresolved_output_tokens(_PROJECT_ROOT / "output" / "manuscript")
    if output_tokens:
        ok = False
        print(f"\nUNRESOLVED TOKENS IN output/manuscript ({len(output_tokens)}):")
        for t in output_tokens:
            print(f"  ✗ {t}")
    return ok


def _provenance_issue(variables: Mapping[str, str]) -> str:
    """Failure message when the producer reports no commit (SC-23 sentinel)."""
    if variables.get("GNN_GIT_COMMIT", "unknown") == "unknown":
        return (
            "GNN_GIT_COMMIT=unknown — the token map was computed without git, "
            "so no commit describes these numbers"
        )
    return ""


def audit_issues(
    strict: bool,
    *,
    variables: dict[str, str] | None = None,
    snapshot: RepositorySnapshot | None = None,
) -> list[str]:
    """Evaluate HEAD audit semantics against explicit repository evidence."""
    manuscript_dir = _PROJECT_ROOT / "manuscript"
    variables = (
        variables if variables is not None else generate_variables(_PROJECT_ROOT)
    )
    known_tokens = set(variables)

    bib_text = (manuscript_dir / "references.bib").read_text(encoding="utf-8")
    bib_keys = set(_BIB_KEY_RE.findall(bib_text))

    # High-value counts that should always be tokens, not literals. The SC-20
    # additions are the structure/coverage counts a prose sentence most often
    # re-types by hand; small ones are filtered by the _HARDCODE_MIN floor.
    hardcode_targets = {
        variables[k]: k
        for k in (
            "GNN_MCP_TOOL_COUNT",
            "GNN_TEST_FILE_COUNT",
            "GNN_TEST_FUNCTION_COUNT",
            "GNN_DOC_FILE_COUNT",
            "GNN_SRC_PY_FILE_COUNT",
            "GNN_SRC_LOC",
            "GNN_EXAMPLE_COUNT",
            "GNN_SRC_PACKAGE_COUNT",
            "GNN_STEP_COUNT",
            "GNN_FAMILY_COUNT",
            "GNN_BACKEND_COUNT",
            "GNN_MAINTAINED_FRAMEWORK_COUNT",
            "GNN_OUTPUT_FIGURE_COUNT",
            "GNN_OUTPUT_ARTIFACT_FIGURE_COUNT",
            "GNN_MANUSCRIPT_FIGURE_COUNT",
        )
        if variables.get(k, "").isdigit() and int(variables[k]) >= _HARDCODE_MIN
    }

    # config.yaml title/subtitle are NOT token-substituted (the injector only
    # processes manuscript/*.md), so a count baked into them silently drifts.
    # Scan them against every producer count, including the small ones the body
    # scan skips, so a regression like a hard-coded "25-step" subtitle fails here.
    config_text = (manuscript_dir / "config.yaml").read_text(encoding="utf-8")
    config_counts = {
        variables[k]: k
        for k in (
            "GNN_STEP_COUNT",
            "GNN_FAMILY_COUNT",
            "GNN_BACKEND_COUNT",
            "GNN_MCP_TOOL_COUNT",
            "GNN_EXAMPLE_COUNT",
            "GNN_SRC_PACKAGE_COUNT",
        )
        if variables.get(k, "").isdigit()
    }
    # version:/date: are producer-owned. scripts/z_generate_manuscript_variables.py
    # writes them; this catches a hand-edit that put them back out of step.
    config_drift = config_metadata_drift(_PROJECT_ROOT, variables)

    config_hardcoded: list[str] = []
    for line in config_text.splitlines():
        stripped = line.strip()
        if not (stripped.startswith(("title:", "subtitle:"))):
            continue
        for value, key in config_counts.items():
            if re.search(rf"(?<!\d){re.escape(value)}(?!\d)", line):
                config_hardcoded.append(
                    f"config.yaml: {stripped.split(':')[0]} hard-codes {value} (use a description without the number; {{{{{key}}}}} does not resolve in config.yaml)"
                )

    # Every pipeline step number the producer owns, keyed by its literal value.
    # These sit below _HARDCODE_MIN, so the count scan above cannot see them.
    step_literals = {
        value: key
        for key, value in variables.items()
        if key.startswith("GNN_STEP_") and value.isdigit()
    }

    unknown_tokens: list[str] = []
    dangling_cites: list[str] = []
    malformed_xrefs: list[str] = []
    hardcoded: list[str] = list(config_hardcoded)

    for path in _section_files(manuscript_dir):
        raw = path.read_text(encoding="utf-8")
        body = _strip_code(raw)
        for tok in _TOKEN_RE.findall(raw):
            if tok not in known_tokens:
                unknown_tokens.append(f"{path.name}: {{{{{tok}}}}}")
        for key in _CITE_RE.findall(body):
            # Skip {#sec:...}/{#fig:...} anchors which are not citations.
            if key.startswith(("sec:", "fig:", "tbl:", "eq:")):
                continue
            if key not in bib_keys:
                dangling_cites.append(f"{path.name}: [@{key}]")
        for match in _MALFORMED_XREF_RE.finditer(body):
            malformed_xrefs.append(
                f"{path.name}: {match.group(0)!r} has a stray prefix"
            )
        # token-stripped body so {{COUNT}} does not count as a literal
        no_tokens = _TOKEN_RE.sub("", body)
        hardcoded.extend(
            _section_hardcode_issues(
                path.name, no_tokens, hardcode_targets, step_literals
            )
        )

    print(f"Sections checked: {len(_section_files(manuscript_dir))}")
    print(f"Known tokens: {len(known_tokens)} | Bib keys: {len(bib_keys)}")

    issues: list[str] = []
    ok = True
    if unknown_tokens:
        issues.extend(unknown_tokens)
        ok = False
        print(f"\nUNKNOWN TOKENS ({len(unknown_tokens)}) — would render literally:")
        for u in unknown_tokens:
            print(f"  ✗ {u}")
    if dangling_cites:
        issues.extend(dangling_cites)
        ok = False
        print(f"\nDANGLING CITATIONS ({len(dangling_cites)}):")
        for d in sorted(set(dangling_cites)):
            print(f"  ✗ {d}")
    if malformed_xrefs:
        issues.extend(malformed_xrefs)
        ok = False
    figure_issues = _figure_registry_issues(
        manuscript_dir, _section_files(manuscript_dir)
    )
    if figure_issues:
        issues.extend(figure_issues)
        ok = False
        print(f"\nFIGURE REGISTRY ({len(figure_issues)}):")
        for f in figure_issues:
            print(f"  ✗ {f}")
    crossrefs = _crossref_issues(_section_files(manuscript_dir), variables)
    if crossrefs:
        issues.extend(crossrefs)
        ok = False
        print(f"\nUNDECLARED CROSS-REFERENCE LABELS ({len(crossrefs)}):")
        for c in crossrefs:
            print(f"  ✗ {c}")
    path_claims = _path_claim_issues(
        _section_files(manuscript_dir),
        _families(
            snapshot if snapshot is not None else RepositorySnapshot(_PROJECT_ROOT)
        ),
    )
    if path_claims:
        issues.extend(path_claims)
        ok = False
        print(f"\nREPOSITORY PATH CLAIMS ({len(path_claims)}):")
        for c in path_claims:
            print(f"  \u2717 {c}")
    if config_drift:
        issues.extend(f"manuscript/config.yaml: {issue}" for issue in config_drift)
        ok = False
        print(f"\nCONFIG.YAML DRIFT ({len(config_drift)}):")
        for d in config_drift:
            print(f"  ✗ {d} — run scripts/z_generate_manuscript_variables.py")
    if config_hardcoded:
        issues.extend(config_hardcoded)
        # config.yaml counts cannot be tokenized away — always a hard failure.
        ok = False
    provenance_issue = _provenance_issue(variables)
    if provenance_issue:
        issues.append(provenance_issue)
        # SC-23: the degraded snapshot sentinel is a published token; numbers
        # with no commit behind them are not publishable, so this is a hard
        # failure in both modes, not a strict-only one.
        ok = False
        print(f"\nPRODUCER PROVENANCE: ✗ {provenance_issue}")
    if hardcoded:
        print(f"\nHARD-CODED COUNTS ({len(hardcoded)}) — prefer tokens:")
        for h in sorted(set(hardcoded)):
            print(f"  ! {h}")
        if strict:
            issues.extend(hardcoded)
            ok = False
    if strict:
        committed = _committed_variables_issue(
            variables, _PROJECT_ROOT / "output/data/manuscript_variables.json"
        )
        if committed:
            issues.append(f"output/data/manuscript_variables.json: {committed}")
        png, sources = _figure_artifact_issues(snapshot)
        issues.extend(png)
        issues.extend(sources)
        issues.extend(_unresolved_output_tokens(_PROJECT_ROOT / "output/manuscript"))
        ok = not issues

    if ok and not hardcoded:
        print("\n✅ Manuscript token/citation integrity: clean")
    elif ok:
        print(
            "\n✅ No unknown tokens or dangling citations (hard-coded warnings above)"
        )
    else:
        print("\n❌ Manuscript integrity gate FAILED")
    return sorted(set(issues))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manuscript token/citation integrity gate"
    )
    parser.add_argument(
        "--strict", action="store_true", help="treat hard-coded counts as failures"
    )
    parser.add_argument(
        "--base-ref",
        default=os.environ.get("GNN_MANUSCRIPT_BASE_REF") or None,
        help="PR target ref; tolerate only exact inherited findings at merge base",
    )
    args = parser.parse_args()
    import contextlib
    import io

    audit_output = io.StringIO()
    with contextlib.redirect_stdout(audit_output):
        issues = audit_issues(args.strict)
    warnings = [
        line.strip()
        for line in audit_output.getvalue().splitlines()
        if line.lstrip().startswith("! ")
    ]
    if not args.strict and warnings:
        print("HARD-CODED COUNTS — prefer tokens:")
        for warning in warnings:
            print(f"  {warning}")
    project_root = _PROJECT_ROOT

    def audit_base(root: Path, sha: str) -> list[str]:
        import contextlib
        import io

        global _PROJECT_ROOT
        snapshot = RepositorySnapshot(project_root, sha)
        variables = generate_variables(root, snapshot=snapshot)
        try:
            _PROJECT_ROOT = root
            with contextlib.redirect_stdout(io.StringIO()):
                return audit_issues(args.strict, variables=variables, snapshot=snapshot)
        finally:
            _PROJECT_ROOT = project_root

    comparison = compare_findings(project_root, issues, args.base_ref, audit_base)
    if comparison.inherited:
        print(f"WARNING: stale base {comparison.base_sha}; fix main first")
        for issue in comparison.inherited:
            print(f"  unchanged inherited drift: {issue}")
    for issue in comparison.failures:
        print(f"  FAILED: {issue}")
    if not comparison.failures:
        print(
            "No new manuscript integrity failures"
            if comparison.inherited
            else "Manuscript token/citation integrity: no failures (warnings above)"
            if warnings and not args.strict
            else "Manuscript token/citation integrity: clean"
        )
    return 1 if comparison.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Markdown documentation audit: relative links, AGENTS→SPEC footers, src/doc coverage,
AGENTS↔README pairing, SPEC coverage, prose layout drift, stale version claims,
registry-bound engine counts and model-kind coverage.

Run from repository root:
  uv run --extra dev python docs/development/docs_audit.py
  uv run --extra dev python docs/development/docs_audit.py --strict   # exit 1 if any issue
  uv run --extra dev python docs/development/docs_audit.py --check-anchors  # verify #fragments in .md links (optional)
  uv run --extra dev python docs/development/docs_audit.py --strict --check-anchors --no-write

With ``--strict`` and any findings, a **full per-issue listing** is written to stderr by default
(terminal-friendly fix loop). Use ``--quiet`` to print only counts and the one-line summary.

Writes ``docs/development/docs_audit_report.md`` unless ``--no-write`` is passed or a
custom ``--report-path`` is provided.
"""

from __future__ import annotations

import argparse
import ast
import html
import logging
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import unquote, urlsplit

REPO_ROOT = Path(__file__).resolve().parents[2]

# Do not scan these path segments
SKIP_PARTS = frozenset(
    {
        "node_modules",
        ".venv",
        "__pycache__",
        ".git",
        ".myp",
        ".herdr-project",  # herdr lane runtime dirs (synced briefs, reports)
        "dist",
        "build",
        ".eggs",
        # Pipeline-generated artifacts — not maintained documentation.
        # LLM output, render output, execute output, etc. may contain code-like
        # fragments that the link regex would otherwise mis-parse.
        "output",
    }
)

# docs/ subtrees excluded from maintained-folder checks (generated or exploratory)
DOC_MAINTAINED_SKIP_PARTS = frozenset(
    {
        "fleet-logs",
        "other",
        "__pycache__",
        ".git",
        "node_modules",
        ".venv",
    }
)

# Pairing report: skip dirs where AGENTS/README policy does not apply
PAIRING_SKIP_PARTS = frozenset(
    {
        "fleet-logs",
        "node_modules",
        ".venv",
        "__pycache__",
        ".git",
        "output",
        "other",
        "build",
        "dist",
        ".claude",
        ".desloppify",
        ".eggs",
    }
)


def should_skip(path: Path) -> bool:
    try:
        rel = path.relative_to(REPO_ROOT)
    except ValueError:
        return True
    return any(p in SKIP_PARTS for p in rel.parts) or _path_is_generated_output(rel)


def _path_is_generated_output(rel: Path) -> bool:
    """Run outputs that are intentionally excluded from maintained-doc audits."""
    parts = rel.parts
    if not parts:
        return False
    if parts[0] == "output":
        return True
    if any(part.startswith("activeinference_outputs_") for part in parts):
        return True
    if any(part.endswith("_outputs") or "_outputs_" in part for part in parts):
        return True
    return "pomdp_gridworld_outputs" in parts


def iter_markdown_files() -> list[Path]:
    out: set[Path] = set()
    for p in REPO_ROOT.rglob("*.md"):
        if should_skip(p):
            continue
        try:
            p.relative_to(REPO_ROOT)
        except ValueError:
            continue
        out.add(p)
    return sorted(out)


# Inline code spans must not turn example syntax into links.
INLINE_CODE_RE = re.compile(r"(`+)(?:(?!\1).)+\1")
DESTINATION_RE = re.compile(
    r"!?\[(?:[^\]\[]|\[[^\]\[]*\])*\]"
    r"\(\s*(<[^>\n]+>|(?:[^\s()\\]|\\.|\([^()]*\))+)"
    r"(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^()]*\)))?\s*\)"
)
REFERENCE_RE = re.compile(r"^ {0,3}\[[^]\n]+\]:\s*(<[^>]+>|\S+)", re.MULTILINE)


def _strip_fenced_code(md: str) -> str:
    """Blank fenced code while preserving line numbers, including long fences."""
    lines: list[str] = []
    fence_char = ""
    fence_size = 0
    for line in md.splitlines(keepends=True):
        match = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\r\n"))
        if fence_char:
            if (
                match
                and match[1][0] == fence_char
                and len(match[1]) >= fence_size
                and not match[2].strip()
            ):
                fence_char = ""
            lines.append("\n" if line.endswith("\n") else "")
        elif match and not (match[1][0] == "`" and "`" in match[2]):
            fence_char, fence_size = match[1][0], len(match[1])
            lines.append("\n" if line.endswith("\n") else "")
        else:
            lines.append(line)
    return "".join(lines)


def _strip_code(md: str) -> str:
    """Remove code without changing the line numbers of real links."""
    return INLINE_CODE_RE.sub(lambda m: " " * len(m.group(0)), _strip_fenced_code(md))


def extract_links(md: str) -> list[str]:
    """Extract inline destinations and reference definitions, excluding code."""
    stripped = _strip_code(md)
    return [m[1].strip("<>") for m in DESTINATION_RE.finditer(stripped)] + [
        m[1].strip("<>") for m in REFERENCE_RE.finditer(stripped)
    ]


def gfm_slug(heading_line: str) -> str:
    """GitHub-compatible heading slug, per cmark-gfm's `gfm_auto_identifiers`
    extension (github/cmark-gfm, extensions/gfm_auto_identifiers.c):

    1. Take the heading text (inline formatting markers such as backticks are
       removed, as GitHub renders them away before slugging).
    2. Lowercase (Unicode-aware).
    3. Remove every character that is not a Unicode letter, digit, underscore,
       hyphen, or space (this strips emoji and other punctuation).
    4. Replace each space with a hyphen, then strip leading/trailing hyphens
       (GitHub drops hyphens left over from removed punctuation/emoji). Interior
       space runs are NOT collapsed — one hyphen per space.

    >>> gfm_slug("## 🚀 Start here")
    'start-here'
    >>> gfm_slug("## 中文 标题!")
    '中文-标题'
    >>> gfm_slug("## C++ & Python")
    'c--python'
    >>> gfm_slug("## Foo   Bar")
    'foo---bar'
    """
    m = re.match(r"^#{1,6}\s+(.+)$", heading_line.strip())
    text = m.group(1) if m else heading_line
    text = re.sub(r"\s+#+\s*$", "", text)
    text = re.sub(r"!?\[([^]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]*>", "", text)
    text = html.unescape(text).strip().replace("`", "").lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    return text.replace(" ", "-").strip("-")


def _heading_slugs_in_markdown(md: str) -> set[str]:
    """Rendered ATX/Setext headings, duplicate suffixes and explicit HTML IDs."""
    prose = _strip_fenced_code(md)
    slugs = {
        m[1]
        for m in re.finditer(
            r"<(?:a|h[1-6])\b[^>]*\b(?:id|name)=[\"']([^\"']+)[\"']",
            prose,
            re.IGNORECASE,
        )
    }
    used: set[str] = set()
    lines = prose.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^ {0,3}#{1,6}\s+", line):
            heading = line.strip()
        elif i and lines[i - 1].strip() and re.fullmatch(r" {0,3}(?:=+|-+)\s*", line):
            heading = lines[i - 1].strip()
        else:
            continue
        base = gfm_slug(heading)
        slug = base
        suffix = 0
        while slug in used:
            suffix += 1
            slug = f"{base}-{suffix}"
        used.add(slug)
        slugs.add(slug)
    return slugs


def audit_bad_markdown_anchors(files: list[Path]) -> list[tuple[Path, int, str, str]]:
    """Check same-page and cross-file Markdown anchors against rendered headings."""
    issues: list[tuple[Path, int, str, str]] = []
    slug_cache: dict[Path, set[str]] = {}
    for src in files:
        try:
            rel_src = src.relative_to(REPO_ROOT)
        except ValueError:
            continue
        lines = _strip_code(
            src.read_text(encoding="utf-8", errors="replace")
        ).splitlines()
        for i, line in enumerate(lines, start=1):
            for href in extract_links(line):
                url = urlsplit(href)
                if url.scheme or url.netloc or not url.fragment:
                    continue
                resolved = resolve_link(src, href) if url.path else src
                if (
                    resolved is None
                    or not resolved.is_file()
                    or resolved.suffix.lower() != ".md"
                ):
                    continue
                if not resolved.is_relative_to(REPO_ROOT):
                    continue
                if resolved not in slug_cache:
                    slug_cache[resolved] = _heading_slugs_in_markdown(
                        resolved.read_text(encoding="utf-8", errors="replace")
                    )
                frag = unquote(url.fragment)
                if frag not in slug_cache[resolved]:
                    issues.append(
                        (
                            rel_src,
                            i,
                            href,
                            f"anchor #{frag} not found (headings in `{resolved.relative_to(REPO_ROOT)}`)",
                        )
                    )
    return issues


def resolve_link(source_file: Path, href: str) -> Path | None:
    url = urlsplit(href.strip("<>"))
    if url.scheme or url.netloc or not url.path:
        return None
    return (source_file.parent / unquote(url.path)).resolve()


def audit_broken_links(files: list[Path]) -> list[tuple[Path, int, str, str]]:
    """(source_file, line_no, href, reason)"""
    issues: list[tuple[Path, int, str, str]] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = _strip_code(text).splitlines()
        for i, line in enumerate(lines, start=1):
            for href in extract_links(line):
                if href.startswith("`"):
                    continue
                resolved = resolve_link(path, href)
                if resolved is None:
                    continue
                try:
                    rtarget = resolved.relative_to(REPO_ROOT)
                except ValueError:
                    issues.append(
                        (
                            path.relative_to(REPO_ROOT),
                            i,
                            href,
                            "relative link escapes repository",
                        )
                    )
                    continue
                if resolved.exists():
                    continue
                rel = path.relative_to(REPO_ROOT)
                issues.append((rel, i, href, f"missing: {rtarget}"))
    return issues


def audit_agents_spec() -> list[tuple[Path, str]]:
    """AGENTS.md that mention sibling SPEC.md link but file missing."""
    issues: list[tuple[Path, str]] = []
    for agents in REPO_ROOT.rglob("AGENTS.md"):
        if should_skip(agents):
            continue
        if "output" in agents.parts:
            continue
        text = agents.read_text(encoding="utf-8", errors="replace")
        if "[SPEC](SPEC.md)" not in text and "](SPEC.md)" not in text:
            continue
        spec = agents.parent / "SPEC.md"
        if not spec.exists():
            rel = agents.relative_to(REPO_ROOT)
            issues.append((rel, "references SPEC.md but sibling SPEC.md missing"))
    return issues


VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)

# Prose-only references to the removed "src/<module>/" layout. The canonical
# surface since v3.3.0 is ``src/gnn/<module>/`` (or ``gnn.<module>``); bare
# ``src/<module>`` and dotted ``src.<module>`` spellings are stale residue of
# the pre-3.3.0 tree. Excluded from matching: comments, fenced code blocks,
# and inline code spans — same treatment as the link/anchor audits.
PROSE_SRC_PATH_RE = re.compile(r"(?<![\w./-])src/[a-zA-Z_][\w-]*/")
PROSE_SRC_IMPORT_RE = re.compile(
    r"(?<![\w./])src\.[a-zA-Z_][\w.]*(?:\.[a-zA-Z_][\w]*)*"
)
PROSE_ALLOWED_SRC_PATHS = frozenset(
    {
        "src/gnn/",
        "src/tests/",
        "src/scripts/",
    }
)

# "### Current Version: X" / "Current Version: X" claims in module AGENTS.md
# files. The product version lives in pyproject.toml; per-tool metadata (the
# external oxdraw CLI version) is out of scope for this check.
CURRENT_VERSION_CLAIM_RE = re.compile(
    r"(?im)^\s*(?:#+\s*)?current version\s*[:=]\s*(\S+)"
)

# Claims already in canonical deferred form are exempt: any version claim
# that references pyproject.toml (the canonical source) instead of a number.
CANONICAL_REF_RE = re.compile(r"pyproject\.toml")

# Per-module metadata: a "Current Version:" block that explicitly states it
# is module-level (independent of the pipeline release) is exempt — the
# claim documents the module/tool, not the pipeline.
PER_MODULE_VERSION_RE = re.compile(
    r"(?i)module|per-module|independent of the pipeline|external tool"
)

# Version claims in docs: either the explicit "Version" line or bare
# "Pipeline Version: X" footers. Only enforced where a module/page states its
# own version, to avoid flagging legitimate historical release notes.
STALE_VERSION_CLAIM_RE = re.compile(
    r"(?im)^(?:\*\*)?\s*(?:pipeline )?version(?:\*\*)?\s*[:=]\s*\[?(\d+\.\d+\.\d+)\]?"
)

# Render-engine count claims: footer/status metadata lines ("**Renderers**:
# 9 backends") and prose phrases bound to the framework registry ("the nine
# computational engines declared in ``framework_registry.py``", "single
# declaration of the nine frameworks", "Supports nine backends: ...") name
# the maintained renderer inventory declared by
# ``src/gnn/render/framework_registry.py``. The count is judged against the
# live registry (parsed via ``ast``, never imported) so the claim cannot
# drift when a backend is added.
ENGINE_COUNT_WORDS: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

_ENGINE_COUNT_NUM = "|".join(ENGINE_COUNT_WORDS) + r"|\d{1,2}"

# "9 backends" / "nine frameworks" / "ten computational engines". A count
# directly preceded by "other" ("reports it alongside the other nine
# backends" — implies the total, not a claim of it) or by "Step N" ("Step 12
# backends remain core deps" — a step number, not a count) is not a claim.
ENGINE_COUNT_RE = re.compile(
    rf"(?i)\b(?:(?P<idiom>other|step\s+\d{{1,2}})\s+)?"
    rf"(?P<count>{_ENGINE_COUNT_NUM})\s+"
    r"(?P<noun>(?:computational\s+|rendering\s+)?(?:engines|frameworks|backends))\b"
)

# Footer/status metadata lines: the count may sit anywhere after the label.
ENGINE_COUNT_FOOTER_LABEL_RE = re.compile(r"(?i)\*{0,2}renderers?\*{0,2}\s*:\s*")

# Prose phrases that bind a count to the full maintained renderer inventory.
# Partial-surface phrasings ("renders one parsed spec to six backends —
# RxInfer.jl, ..." for the cross-framework comparison tool, "the dispatch
# covers all eleven backends" for the Step-12 execution axis) deliberately
# do not bind and are never judged against the render registry.
ENGINE_COUNT_PROSE_BINDING_RE = re.compile(
    r"(?i)declaration of the\b|declared in\b|registered in\b|wired under\b"
    r"|supports\b|renders?\s+to\b|code (?:generation|rendering) for\b"
)


def engine_count_scan_files() -> list[Path]:
    """Files scanned for render-engine count claims: the maintained-doc set
    plus the root ``SPEC.md``. The root ecosystem spec states the engine
    inventory ("the N computational engines declared in
    ``framework_registry.py``") but is not in ``MAINTAINED_ROOT_DOC_FILES``
    (whose bare-``Version`` enforcement is deliberately not extended here),
    so it is added to this check's scan set explicitly."""
    return maintained_doc_files() + [REPO_ROOT / "SPEC.md"]


def _framework_registry_count() -> int | None:
    """Number of frameworks declared by ``FRAMEWORK_REGISTRY``.

    Parsed with :mod:`ast` from the registry source so the audit stays
    dependency-free; ``None`` when the registry cannot be read or the
    assignment cannot be located (the audit then reports that directly
    instead of silently skipping the check).
    """
    literal = _registry_literal("FRAMEWORK_REGISTRY")
    if literal is None:
        return None
    return len(literal.keys)


def audit_engine_count_claims(
    files: list[Path],
) -> list[tuple[Path, int, str, str]]:
    """Flag render-engine count claims that contradict the framework registry.

    Two claim shapes are enforced against ``len(FRAMEWORK_REGISTRY)``:

    - **Footer/status lines**: ``**Renderers**: N backends`` metadata lines
      in module/domain README footers. The count is searched after the
      ``Renderers:`` label so sibling counts on the same line (``Modules``,
      ``Pipeline steps``) are never picked up.
    - **Registry-bound prose**: a count phrase on a line that also binds it
      to the full renderer inventory (``declared in …registry``,
      ``single declaration of the N frameworks``, ``wired under``,
      ``Supports N backends``, ``renders to N backends``, ``Code
      generation/rendering for``). Subset claims ("renders one parsed spec
      to six backends") and execution-axis totals ("the dispatch covers all
      eleven backends") do not bind and are exempt by construction.

    ``other N`` / ``Step N`` idioms adjacent to a count are never claims.
    """
    registry_count = _framework_registry_count()
    if registry_count is None:
        return [
            (
                Path("src/gnn/render/framework_registry.py"),
                0,
                "",
                "cannot parse FRAMEWORK_REGISTRY",
            )
        ]

    issues: list[tuple[Path, int, str, str]] = []

    def _claim(count_text: str, noun: str) -> bool:
        claimed = ENGINE_COUNT_WORDS.get(count_text.lower())
        claimed = claimed if claimed is not None else int(count_text)
        return claimed != registry_count

    def _reason(count_text: str) -> str:
        return (
            f"render-engine count claim != framework_registry "
            f"({registry_count} frameworks)"
        )

    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = path.relative_to(REPO_ROOT)
        for lineno, line in enumerate(
            _strip_code_and_links(text).splitlines(), start=1
        ):
            footer = ENGINE_COUNT_FOOTER_LABEL_RE.search(line)
            matches = list(ENGINE_COUNT_RE.finditer(line))
            if footer is not None:
                matches = [m for m in matches if m.start() >= footer.end()]
            else:
                if not ENGINE_COUNT_PROSE_BINDING_RE.search(line):
                    continue
            for match in matches:
                if match.group("idiom"):
                    continue
                count_text = match.group("count")
                if _claim(count_text, match.group("noun")):
                    issues.append(
                        (
                            rel,
                            lineno,
                            count_text,
                            _reason(count_text),
                        )
                    )
    return issues


# ---------------------------------------------------------------------------
# Registry-bound model-kind coverage and frame documentation.
#
# The exemplar docs state which render frames consume which state-space kind.
# Those statements are judged against each frame's ``state_space`` declaration
# in ``src/gnn/render/framework_registry.py`` -- the same declaration the render
# compatibility gate types its ``unsupported-<kind>-state:`` refusals against --
# so a frame that declares a kind and is missing from the docs fails the audit
# instead of disappearing silently.
# ---------------------------------------------------------------------------

#: Registry dimension that binds the model-kind claims.
MODEL_KIND_DIMENSION = "state_space"

#: The state-space values the docs describe.
MODEL_KIND_VALUES = ("discrete", "continuous")

#: Documentation outside ``maintained_doc_files()`` that carries the exemplar
#: corpus's model-kind contract (the input tables and the data contract).
MODEL_KIND_EXTRA_DOC_FILES = (
    "input/AGENTS.md",
    "input/README.md",
    "input/SPEC.md",
)

#: Files that must carry the model-kind table (kind rows -> renderers ->
#: refusals) the coverage check judges.
MODEL_KIND_TABLE_FILES = ("input/AGENTS.md", "input/README.md")

#: Registry key -> the spellings the docs use for that frame. The
#: ``activeinference_jl`` pattern requires the ``jl`` suffix so the research
#: field name ("Active Inference") is never read as the frame.
MODEL_KIND_FRAME_PATTERNS: dict[str, str] = {
    "pymdp": r"pymdp",
    "rxinfer": r"rx[\s._-]?infer(?:[\s._-]*jl)?",
    "activeinference_jl": r"active[\s_-]?inference[\s._-]*jl",
    "jax": r"jax",
    "discopy": r"discopy",
    "pytorch": r"pytorch",
    "numpyro": r"numpyro",
    "stan": r"stan",
    "bnlearn": r"bnlearn",
    "cpomdp": r"cpomdp",
    "thrml": r"thrml",
    "ngclearn": r"ngc[\s-]?learn",
}

#: Phrases that bind an enumerated frame list to a state-space kind
#: ("Continuous models render and execute on JAX, ...", "Renders + executes on
#: ...", "reach Step 12 only for the backends that render them ...").
MODEL_KIND_LIST_BINDING_RE = re.compile(
    r"(?i)\brenders?\s+(?:and\s+)?executes?\s+on\b"
    r"|\brenders?\s*\+\s*executes?\s+on\b"
    r"|\brenders?\s+on\b"
    r"|\bexecutes?\s+on\b"
    r"|\brun(?:s|ned)?\s+natively\s+on\b"
    r"|\breach\s+step\s+12\s+only\s+for\b"
)

#: Sentence/clause boundary inside one markdown block unit.
MODEL_KIND_CLAUSE_SPLIT_RE = re.compile(r"(?<=[.;])\s+")

#: Markdown block starter: a new paragraph unit begins at a blank line or at a
#: heading, list item, block quote or table row, so a claim wrapped over two
#: lines is judged whole without merging neighbouring list items.
MODEL_KIND_BLOCK_START_RE = re.compile(r"^\s*(?:[#>|*+-]|\d+[.)])\s")

#: Model-kind table wiring (both tables carry the same three columns).
MODEL_KIND_TABLE_KIND_HEADER_RE = re.compile(r"(?i)\bmodel\s+kind\b")
MODEL_KIND_TABLE_RENDER_HEADER_RE = re.compile(r"(?i)\brenders?\b[^|]*\bexecutes?\b")
MODEL_KIND_TABLE_UNSUPPORTED_HEADER_RE = re.compile(r"(?i)unsupported")
MODEL_KIND_TABLE_ROW_RE = re.compile(r"(?i)^\s*\**\s*(discrete|continuous)\b")
MODEL_KIND_TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)*\|?\s*$"
)


def model_kind_scan_files() -> list[Path]:
    """Files scanned for model-kind frame claims: the maintained-doc set plus
    the three ``input/`` contract documents. The exemplar table and the data
    contract live outside ``maintained_doc_files()`` (like the root
    ``SPEC.md`` in the engine-count check), so they are added explicitly."""
    return maintained_doc_files() + [
        REPO_ROOT / rel for rel in MODEL_KIND_EXTRA_DOC_FILES
    ]


def _registry_literal(name: str) -> ast.Dict | None:
    """The ``ast.Dict`` literal assigned to *name* in the render registry.

    Parsed with :mod:`ast` so the audit stays dependency-free; unwraps the
    ``MappingProxyType({...})`` call the registry uses for immutable mappings.
    ``None`` when the source cannot be read or the assignment is not a dict.
    """
    path = REPO_ROOT / "src" / "gnn" / "render" / "framework_registry.py"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return None
    for node in ast.walk(tree):
        value = None
        if (
            isinstance(node, ast.AnnAssign)
            and getattr(node.target, "id", None) == name
        ):
            value = node.value
        elif isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) == name for target in node.targets
        ):
            value = node.value
        if value is None:
            continue
        if isinstance(value, ast.Call) and value.args:
            value = value.args[0]
        if isinstance(value, ast.Dict):
            return value
    return None


def _dict_string_keys(literal: ast.Dict) -> dict[str, ast.expr] | None:
    """``{string key: value node}`` for a registry dict literal."""
    out: dict[str, ast.expr] = {}
    for key, value in zip(literal.keys, literal.values):
        name = getattr(key, "value", None)
        if not isinstance(name, str):
            return None
        out[name] = value
    return out


def _render_registry_facts() -> (
    tuple[dict[str, bool], dict[str, tuple[str, ...]]] | None
):
    """``(frame -> experimental, frame -> declared state spaces)``.

    ``None`` when the registry cannot be read or either declaration is
    missing; the audit reports that directly rather than skipping the check.
    """
    frames_literal = _registry_literal("FRAMEWORK_REGISTRY")
    declarations_literal = _registry_literal("RENDER_CAPABILITY_DECLARATIONS")
    if frames_literal is None or declarations_literal is None:
        return None
    frame_entries = _dict_string_keys(frames_literal)
    declaration_entries = _dict_string_keys(declarations_literal)
    if frame_entries is None or declaration_entries is None:
        return None
    frames: dict[str, bool] = {}
    for name, value in frame_entries.items():
        experimental = False
        if isinstance(value, ast.Dict):
            fields = _dict_string_keys(value) or {}
            flag = fields.get("experimental")
            experimental = bool(getattr(flag, "value", False))
        frames[name] = experimental
    declarations: dict[str, tuple[str, ...]] = {}
    for name, value in declaration_entries.items():
        values: tuple[str, ...] = ()
        if isinstance(value, ast.Dict):
            fields = _dict_string_keys(value) or {}
            dimension = fields.get(MODEL_KIND_DIMENSION)
            if dimension is None:
                values = ()
            elif isinstance(dimension, ast.Tuple):
                values = tuple(
                    element.value
                    for element in dimension.elts
                    if isinstance(element, ast.Constant)
                    and isinstance(element.value, str)
                )
            else:
                return None
        declarations[name] = values
    return frames, declarations


def _model_kind_expectations(
    frames: dict[str, bool], declarations: dict[str, tuple[str, ...]]
) -> dict[str, tuple[frozenset[str], frozenset[str]]]:
    """Default render set per model kind: ``kind -> (renders_on, refused_on)``.

    Only non-experimental frames are judged: experimental frames require
    explicit selection and are documented on their own implementation pages.
    """
    default = [name for name, experimental in frames.items() if not experimental]
    expectations: dict[str, tuple[frozenset[str], frozenset[str]]] = {}
    for kind in MODEL_KIND_VALUES:
        renders_on = frozenset(
            name for name in default if kind in declarations.get(name, ())
        )
        expectations[kind] = (renders_on, frozenset(default) - renders_on)
    return expectations


def _frames_named_in(text: str) -> set[str]:
    """Registry keys whose documented spelling appears in *text*."""
    return {
        name
        for name, pattern in MODEL_KIND_FRAME_PATTERNS.items()
        if re.search(rf"(?i)\b(?:{pattern})\b", text)
    }


def _model_kind_mentions(text: str) -> set[str]:
    """State-space values named in *text*."""
    return {
        kind
        for kind in MODEL_KIND_VALUES
        if re.search(rf"(?i)\b{kind}\b", text)
    }


def _model_kind_contradiction(
    frame: str,
    kind: str,
    frames: dict[str, bool],
    declarations: dict[str, tuple[str, ...]],
) -> str:
    """Why *frame* cannot sit in the *kind* render position."""
    if frame not in frames:
        return f"{frame} is not declared in FRAMEWORK_REGISTRY"
    if frames[frame]:
        return f"{frame} is experimental and requires explicit selection"
    declared = declarations.get(frame, ())
    if kind in declared:
        return f"{frame} declares the {kind} state space"
    if declared:
        return f"{frame} declares {', '.join(declared)} state space, not {kind}"
    return f"{frame} declares no state space"


def _table_cells(line: str) -> list[str]:
    """Cell text of one markdown table row."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _model_kind_table_issues(
    rel: Path,
    text: str,
    frames: dict[str, bool],
    declarations: dict[str, tuple[str, ...]],
    expectations: dict[str, tuple[frozenset[str], frozenset[str]]],
) -> tuple[list[tuple[Path, int, str, str]], bool]:
    """Judge every model-kind table in *text*; returns ``(issues, found)``."""
    issues: list[tuple[Path, int, str, str]] = []
    lines = text.splitlines()
    found = False
    index = 0
    while index < len(lines):
        cells = _table_cells(lines[index]) if lines[index].startswith("|") else []
        kind_col = next(
            (
                col
                for col, cell in enumerate(cells)
                if MODEL_KIND_TABLE_KIND_HEADER_RE.search(cell)
            ),
            None,
        )
        render_col = next(
            (
                col
                for col, cell in enumerate(cells)
                if MODEL_KIND_TABLE_RENDER_HEADER_RE.search(cell)
            ),
            None,
        )
        unsupported_col = next(
            (
                col
                for col, cell in enumerate(cells)
                if MODEL_KIND_TABLE_UNSUPPORTED_HEADER_RE.search(cell)
            ),
            None,
        )
        if kind_col is None or render_col is None or unsupported_col is None:
            index += 1
            continue
        found = True
        row = index + 1
        while row < len(lines) and lines[row].startswith("|"):
            row_cells = _table_cells(lines[row])
            row_kind = None
            if kind_col < len(row_cells):
                match = MODEL_KIND_TABLE_ROW_RE.match(row_cells[kind_col])
                row_kind = match.group(1).lower() if match else None
            if row_kind is not None and not MODEL_KIND_TABLE_SEPARATOR_RE.match(
                lines[row]
            ):
                renders_on, refused_on = expectations[row_kind]
                rendered = _frames_named_in(row_cells[render_col])
                for frame in sorted(renders_on - rendered):
                    issues.append(
                        (
                            rel,
                            row + 1,
                            frame,
                            f"model-kind table omits {frame} from the "
                            f"{row_kind} render row, but its registry entry "
                            f"declares the {row_kind} state space",
                        )
                    )
                for frame in sorted(rendered - renders_on):
                    issues.append(
                        (
                            rel,
                            row + 1,
                            frame,
                            f"model-kind table lists {frame} in the "
                            f"{row_kind} render row: "
                            + _model_kind_contradiction(
                                frame, row_kind, frames, declarations
                            ),
                        )
                    )
                refused = _frames_named_in(row_cells[unsupported_col])
                for frame in sorted(refused_on - refused):
                    issues.append(
                        (
                            rel,
                            row + 1,
                            frame,
                            f"model-kind table does not report {frame} as "
                            f"unsupported for {row_kind} models, but its "
                            f"registry entry does not declare the {row_kind} "
                            f"state space",
                        )
                    )
                for frame in sorted(refused - refused_on):
                    issues.append(
                        (
                            rel,
                            row + 1,
                            frame,
                            f"model-kind table reports {frame} as unsupported "
                            f"for {row_kind} models: "
                            + _model_kind_contradiction(
                                frame, row_kind, frames, declarations
                            ),
                        )
                    )
            row += 1
        index = row
    return issues, found


def _model_kind_units(text: str) -> list[tuple[int, str]]:
    """Markdown lines grouped into block units (paragraph, list item, table
    row), so a claim wrapped over two lines is judged as one statement."""
    units: list[tuple[int, str]] = []
    start: int | None = None
    parts: list[str] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        new_block = not line or bool(MODEL_KIND_BLOCK_START_RE.match(raw))
        if start is None:
            if line:
                start, parts = lineno, [line]
            continue
        if new_block:
            units.append((start, " ".join(parts)))
            start, parts = (lineno, [line]) if line else (None, [])
        else:
            parts.append(line)
    if start is not None:
        units.append((start, " ".join(parts)))
    return units


def _model_kind_prose_issues(
    rel: Path,
    text: str,
    frames: dict[str, bool],
    declarations: dict[str, tuple[str, ...]],
    expectations: dict[str, tuple[frozenset[str], frozenset[str]]],
) -> list[tuple[Path, int, str, str]]:
    """Judge registry-bound frame lists in prose against the declarations.

    A clause that binds a frame list to a state space ("Continuous models
    render and execute on JAX, ...") must name every default frame declaring
    that kind and must not name a frame that does not. A clause with no kind
    word of its own ("They render and execute on ...") inherits the nearest
    preceding kind mention in the same block, and a declaration-bound sentence
    that enumerates nothing is accepted unchanged.
    """
    issues: list[tuple[Path, int, str, str]] = []
    for lineno, unit in _model_kind_units(text):
        current_kinds: set[str] = set()
        for clause in MODEL_KIND_CLAUSE_SPLIT_RE.split(unit):
            clause_kinds = _model_kind_mentions(clause)
            if clause_kinds:
                current_kinds = clause_kinds
            if not MODEL_KIND_LIST_BINDING_RE.search(clause):
                continue
            named = _frames_named_in(clause)
            if len(named) < 2:
                continue
            kinds = clause_kinds or current_kinds
            for kind in MODEL_KIND_VALUES:
                if kind not in kinds:
                    continue
                renders_on, _refused_on = expectations[kind]
                for frame in sorted(named):
                    if kind not in declarations.get(frame, ()):
                        issues.append(
                            (
                                rel,
                                lineno,
                                frame,
                                f"registry-bound {kind} renderer list names "
                                f"{frame}: "
                                + _model_kind_contradiction(
                                    frame, kind, frames, declarations
                                ),
                            )
                        )
                for frame in sorted(renders_on - named):
                    issues.append(
                        (
                            rel,
                            lineno,
                            frame,
                            f"registry-bound {kind} renderer list omits "
                            f"{frame}, whose registry entry declares the "
                            f"{kind} state space",
                        )
                    )
    return issues


def audit_model_kind_coverage(
    files: list[Path],
) -> list[tuple[Path, int, str, str]]:
    """Flag model-kind frame claims that contradict the registry declarations.

    Two surfaces are enforced against ``RENDER_CAPABILITY_DECLARATIONS``:

    - **The model-kind tables** (``input/AGENTS.md``, ``input/README.md``):
      the render cell must name exactly the default (non-experimental) frames
      whose ``state_space`` declaration includes that kind, and the
      ``unsupported`` cell exactly the default frames whose declaration does
      not. A frame in the wrong role, an experimental frame offered as a
      default target, or a missing table is an issue.
    - **Registry-bound prose lists**: a clause that binds a frame list to a
      state-space kind must be complete for that kind. This is the recurrence
      gate: a frame that declares a kind and is missing from a maintained-doc
      list fails here instead of being silently omitted.
    """
    facts = _render_registry_facts()
    if facts is None:
        return [
            (
                Path("src/gnn/render/framework_registry.py"),
                0,
                "",
                "cannot parse FRAMEWORK_REGISTRY / "
                "RENDER_CAPABILITY_DECLARATIONS",
            )
        ]
    frames, declarations = facts
    expectations = _model_kind_expectations(frames, declarations)
    issues: list[tuple[Path, int, str, str]] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = path.relative_to(REPO_ROOT)
        stripped = _strip_fenced_code(text)
        table_issues, table_found = _model_kind_table_issues(
            rel, stripped, frames, declarations, expectations
        )
        if str(rel) in MODEL_KIND_TABLE_FILES and not table_found:
            issues.append(
                (
                    rel,
                    0,
                    "",
                    "cannot locate the model-kind table with a render column",
                )
            )
        issues.extend(table_issues)
        issues.extend(
            _model_kind_prose_issues(
                rel, stripped, frames, declarations, expectations
            )
        )
    return issues


def audit_render_doc_frame_coverage() -> list[tuple[Path, int, str, str]]:
    """Every registry frame must be visible in the maintained render docs.

    A default (non-experimental) frame is named in the Step 11 module page; an
    experimental frame carries its own implementation page, because its
    contract is not part of the default render path. A registry frame that is
    documented nowhere fails here instead of shipping undocumented.
    """
    facts = _render_registry_facts()
    if facts is None:
        return [
            (
                Path("src/gnn/render/framework_registry.py"),
                0,
                "",
                "cannot parse FRAMEWORK_REGISTRY",
            )
        ]
    frames, _declarations = facts
    module_rel = Path("docs/gnn/modules/11_render.md")
    try:
        module_text = (REPO_ROOT / module_rel).read_text(encoding="utf-8")
    except OSError:
        return [
            (
                module_rel,
                0,
                "",
                f"missing: {module_rel}",
            )
        ]
    named = _frames_named_in(module_text)
    issues: list[tuple[Path, int, str, str]] = []
    for frame in sorted(frames):
        if frames[frame]:
            page_rel = Path("docs/gnn/implementations") / f"{frame}.md"
            if not (REPO_ROOT / page_rel).is_file():
                issues.append(
                    (
                        page_rel,
                        0,
                        frame,
                        "experimental frame has no implementation page",
                    )
                )
        elif frame not in named:
            issues.append(
                (
                    module_rel,
                    0,
                    frame,
                    f"frame is not named in {module_rel}",
                )
            )
    return issues


def audit_src_spec_coverage() -> list[tuple[Path, str]]:
    """Every module directory under src/gnn/ with .py files must have SPEC.md.

    The AGENTS.md/README.md pairing checks already cover agent-facing docs;
    SPEC.md is the interface contract of a module (public API, data flow) and
    is required for the same module set. ``src/gnn`` itself counts as a module
    directory (it carries AGENTS.md/README.md/SPEC.md at its root).
    """
    missing: list[tuple[Path, str]] = []
    for d in sorted((REPO_ROOT / "src" / "gnn").iterdir()):
        if not d.is_dir():
            continue
        rel_dir = d.relative_to(REPO_ROOT)
        if rel_dir.parts[0] in SKIP_PARTS or any(
            p in SKIP_PARTS for p in rel_dir.parts
        ):
            continue
        if not any(d.glob("*.py")):
            continue
        if not (d / "SPEC.md").is_file():
            missing.append((rel_dir, "module directory has .py files but no SPEC.md"))
    return missing


def _strip_code_and_links(md: str) -> str:
    """Remove code while retaining canonical owner links used by claim checks.

    Link text/targets remain available to the canonical-reference matcher;
    fenced examples cannot contribute prose/version claims.
    """
    return _strip_code(md)


# Named repo-root docs inside the maintained set (package-version label
# scope: a bare "Version:" here claims the pipeline release).
MAINTAINED_ROOT_DOC_FILES = (
    "AGENTS.md",
    "README.md",
    "ARCHITECTURE.md",
    "TO-DO.md",
    "SETUP_GUIDE.md",
    "CLAUDE.md",
)

# Historical release-note files — exempt from stale-version/prose checks.
MAINTAINED_HISTORICAL_FILES = frozenset({"CHANGELOG.md", "VERSION_MAP.md"})

# Vendored/third-party source-quote trees whose md files quote upstream code
# (e.g. ActiveInference.jl sources with their own ``src/`` layout). Quoted
# code is exempt from the prose-pattern scan.
MAINTAINED_VENDORED_PREFIXES = (
    "docs/other/",
    "docs/activeinference_jl/",
    "docs/development/fleet-logs/",
)


def maintained_doc_files() -> list[Path]:
    """The maintained-doc scope for the claim-style checks (prose patterns,
    version claims): repo-root navigation docs + docs/** + src/gnn/** minus
    historical/vendored trees. Broader repo-wide scans (agent dispatch
    reports, private-agent trees) are out of scope — they are not
    maintained documentation."""
    keep: list[Path] = []
    for name in MAINTAINED_ROOT_DOC_FILES:
        p = REPO_ROOT / name
        if p.is_file():
            keep.append(p)
    for base in ("docs", "src/gnn"):
        base_dir = REPO_ROOT / base
        if base_dir.is_dir():
            for p in sorted(base_dir.rglob("*.md")):
                keep.append(p)
    out: list[Path] = []
    for p in keep:
        try:
            rel = p.relative_to(REPO_ROOT)
        except ValueError:
            continue
        if any(part in SKIP_PARTS for part in rel.parts):
            continue
        if any(
            str(rel).startswith(prefix) or str(rel) == prefix.rstrip("/")
            for prefix in MAINTAINED_VENDORED_PREFIXES
        ):
            continue
        if rel.name in MAINTAINED_HISTORICAL_FILES:
            continue
        out.append(p)
    return sorted(set(out))


def audit_prose_src_patterns(files: list[Path]) -> list[tuple[Path, int, str]]:
    """Flag prose references to the removed ``src/<module>/`` layout.

    Two stale spellings are flagged in visible prose (code spans and fenced
    blocks are exempt — they quote rather than assert):

    - ``src/<module>/…`` path spellings (e.g. ``src/gui/AGENTS.md``), which
      must read ``src/gnn/<module>/…`` since v3.3.0. ``src/gnn/`` itself is
      the canonical surface and never matches.
    - ``src.<module>`` import spellings (e.g. ``import src.main``), which
      fail at import time — the canonical package root is ``gnn.``.
    """
    issues: list[tuple[Path, int, str]] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        rel = path.relative_to(REPO_ROOT)
        for lineno, line in enumerate(
            _strip_code_and_links(text).splitlines(), start=1
        ):
            for match in PROSE_SRC_PATH_RE.finditer(line):
                span = match.group(0)
                if span.rstrip("/") + "/" in PROSE_ALLOWED_SRC_PATHS:
                    continue
                issues.append(
                    (
                        rel,
                        lineno,
                        f"prose path '{span}' is not the canonical src/gnn/ surface",
                    )
                )
            for match in PROSE_SRC_IMPORT_RE.finditer(line):
                issues.append(
                    (
                        rel,
                        lineno,
                        f"prose import '{match.group(0)}' is not the canonical gnn.* surface",
                    )
                )
    return issues


def audit_version_claims(files: list[Path]) -> list[tuple[Path, int, str, str]]:
    """Flag version claims that contradict the canonical pyproject version.

    The product version lives in ``pyproject.toml``; docs may reference it
    rather than hard-coding a number. Two claim shapes are enforced, with a
    version-axis split:

    - **Pipeline-version axes** (everywhere in the maintained set):
      ``Pipeline Version: X`` labels and ``Current Version: X`` module
      metadata blocks claim the *pipeline release* and must match
      pyproject.toml. ``[canonical · X]`` STEP_INDEX-style references are
      exempt (they already defer to pyproject).
    - **Bare ``Version: X``** lines: enforced only in ``src/gnn/**`` and the
      named repo-root files, where they can only mean the package version
      (``**Version**: 3.3.0`` page headers). Under ``docs/**`` they are
      treated as per-page revision axes (document revision counters, cited
      paper versions, per-tool metadata) and are exempt; a bare revision
      number there does not claim the pipeline release.

    Historical files (CHANGELOG/VERSION_MAP) and vendored quote trees are
    excluded from the maintained set entirely.
    """
    pyproject = REPO_ROOT / "pyproject.toml"
    version_match = VERSION_RE.search(pyproject.read_text(encoding="utf-8"))
    if not version_match:
        return [(Path("pyproject.toml"), 0, "", "cannot read project version")]
    canonical = version_match.group(1)

    package_label_scope: list[Path] = [REPO_ROOT / "src" / "gnn"]
    package_label_scope.extend(REPO_ROOT / name for name in MAINTAINED_ROOT_DOC_FILES)

    issues: list[tuple[Path, int, str, str]] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(REPO_ROOT)
        path_resolved = path.resolve()
        enforce_bare_version = any(
            path_resolved.is_relative_to(root) for root in package_label_scope
        )
        for lineno, line in enumerate(
            _strip_code_and_links(text).splitlines(), start=1
        ):
            stale_match = STALE_VERSION_CLAIM_RE.search(line)
            if (
                stale_match
                and stale_match.group(1) != canonical
                and not CANONICAL_REF_RE.search(line)
            ):
                is_pipeline_label = (
                    stale_match.group(0)
                    .lstrip("*")
                    .startswith(("pipeline", "Pipeline"))
                )
                if is_pipeline_label or enforce_bare_version:
                    issues.append(
                        (
                            rel,
                            lineno,
                            stale_match.group(1),
                            f"version claim != pyproject {canonical}",
                        )
                    )
            current_match = CURRENT_VERSION_CLAIM_RE.search(line)
            if (
                current_match
                and current_match.group(1) != canonical
                and not CANONICAL_REF_RE.search(line)
                and not PER_MODULE_VERSION_RE.search(line)
            ):
                issues.append(
                    (
                        rel,
                        lineno,
                        current_match.group(1),
                        f"'Current Version' claim != pyproject {canonical} (pipeline-versioned modules only)",
                    )
                )
    return issues


def audit_security_supported_version() -> list[str]:
    """SECURITY.md supported-versions table must carry a row for the version
    declared in pyproject.toml, e.g. a line starting with `| 3.3.0`."""
    issues: list[str] = []
    pyproject = REPO_ROOT / "pyproject.toml"
    security = REPO_ROOT / "SECURITY.md"
    m = VERSION_RE.search(pyproject.read_text(encoding="utf-8", errors="replace"))
    if not m:
        return ['pyproject.toml: no `version = "…"` declaration found']
    version = m.group(1)
    row = f"| {version}"
    if row not in security.read_text(encoding="utf-8", errors="replace"):
        return [f"SECURITY.md: no supported-versions row for current version {version}"]
    return issues


def audit_src_agents_coverage() -> list[Path]:
    """Directories under src/ with at least one .py file and no AGENTS.md."""
    missing: list[Path] = []
    src = REPO_ROOT / "src"
    if not src.is_dir():
        return missing
    for d in sorted(src.rglob("*")):
        if not d.is_dir():
            continue
        if should_skip(d) or d.name == "output":
            continue
        if not any(d.glob("*.py")):
            continue
        if (d / "AGENTS.md").exists():
            continue
        # Ignore __pycache__ dirs (already skipped by name in should_skip - __pycache__ is in SKIP_PARTS)
        if "__pycache__" in d.parts:
            continue
        missing.append(d.relative_to(REPO_ROOT))
    return missing


def _doc_path_is_generated_dump(rel: Path) -> bool:
    """Captured outputs / run artifacts — not maintained doc packages."""
    parts = rel.parts
    if "actinf_jl_src" in parts:
        i = parts.index("actinf_jl_src")
        if len(parts) > i + 1:
            return True
    if "meta-aware-2" in parts and "test_output" in parts:
        return True
    if "pomdp_gridworld_outputs" in parts:
        return True
    if "multiagent_trajectory_planning" in parts and "results" in parts:
        return True
    if "SQLite_exports" in parts:
        return True
    return False


def _doc_dir_is_maintained(d: Path) -> bool:
    """docs/ subtree folder expected to carry AGENTS.md and README.md."""
    try:
        rel = d.relative_to(REPO_ROOT)
    except ValueError:
        return False
    if len(rel.parts) < 2 or rel.parts[0] != "docs":
        return False
    if any(p in DOC_MAINTAINED_SKIP_PARTS for p in rel.parts):
        return False
    if _doc_path_is_generated_dump(rel):
        return False
    parts_set = set(rel.parts)
    if "results" in parts_set and "multiagent_trajectory_planning" in parts_set:
        return False
    if "enhanced_exports" in parts_set or "SQLite_exports" in parts_set:
        return False

    md_files = [x for x in d.iterdir() if x.is_file() and x.suffix == ".md"]
    subdirs = [
        x
        for x in d.iterdir()
        if x.is_dir()
        and x.name not in DOC_MAINTAINED_SKIP_PARTS
        and not any(p in SKIP_PARTS for p in x.parts)
    ]
    if not md_files and not subdirs:
        return False
    has_non_nav_md = any(m.name not in ("AGENTS.md", "README.md") for m in md_files)
    if has_non_nav_md:
        return True
    if subdirs:
        return True
    return len(md_files) >= 2


def audit_doc_maintained_missing_agents() -> list[Path]:
    missing: list[Path] = []
    doc_root = REPO_ROOT / "docs"
    if not doc_root.is_dir():
        return missing
    for d in sorted(doc_root.rglob("*")):
        if not d.is_dir():
            continue
        if should_skip(d):
            continue
        if not _doc_dir_is_maintained(d):
            continue
        if (d / "AGENTS.md").exists():
            continue
        missing.append(d.relative_to(REPO_ROOT))
    return missing


def audit_doc_maintained_missing_readme() -> list[Path]:
    missing: list[Path] = []
    doc_root = REPO_ROOT / "docs"
    if not doc_root.is_dir():
        return missing
    for d in sorted(doc_root.rglob("*")):
        if not d.is_dir():
            continue
        if should_skip(d):
            continue
        if not _doc_dir_is_maintained(d):
            continue
        if (d / "README.md").exists():
            continue
        missing.append(d.relative_to(REPO_ROOT))
    return missing


def _dir_eligible_for_pairing(d: Path) -> bool:
    if d.resolve() == REPO_ROOT.resolve():
        return True
    try:
        rel = d.relative_to(REPO_ROOT)
    except ValueError:
        return False
    if not rel.parts or rel.parts[0] not in ("src", "docs", ".github"):
        return False
    if any(p in PAIRING_SKIP_PARTS for p in rel.parts):
        return False
    if rel.parts[0] == "src" and len(rel.parts) >= 2 and rel.parts[1] == "output":
        return False
    if _doc_path_is_generated_dump(rel):
        return False
    return True


def audit_agents_without_readme() -> list[Path]:
    out: list[Path] = []
    for agents in sorted(REPO_ROOT.rglob("AGENTS.md")):
        if should_skip(agents):
            continue
        parent = agents.parent
        if not _dir_eligible_for_pairing(parent):
            continue
        if not (parent / "README.md").exists():
            out.append(parent.relative_to(REPO_ROOT))
    return sorted(set(out))


def audit_readme_without_agents() -> list[Path]:
    out: list[Path] = []
    for readme in sorted(REPO_ROOT.rglob("README.md")):
        if should_skip(readme):
            continue
        parent = readme.parent
        if not _dir_eligible_for_pairing(parent):
            continue
        if not (parent / "AGENTS.md").exists():
            out.append(parent.relative_to(REPO_ROOT))
    return sorted(set(out))


def audit_doc_agents_structure() -> list[tuple[Path, str]]:
    """
    docs/**/AGENTS.md should include a standard orientation section.
    Accept ## Overview, ## Purpose, or ## Directory Identity (GNN subtree manifests).
    If ## Purpose exists, its body (until the next ## heading) should be substantive.
    """
    issues: list[tuple[Path, str]] = []
    doc_root = REPO_ROOT / "docs"
    if not doc_root.is_dir():
        return issues
    orientation = ("## Overview", "## Purpose", "## Directory Identity")
    for agents in sorted(doc_root.rglob("AGENTS.md")):
        if should_skip(agents):
            continue
        try:
            rel = agents.relative_to(REPO_ROOT)
        except ValueError:
            continue
        if any(p in DOC_MAINTAINED_SKIP_PARTS for p in rel.parts):
            continue
        if _doc_path_is_generated_dump(rel):
            continue
        parts_set = set(rel.parts)
        if "results" in parts_set and "multiagent_trajectory_planning" in parts_set:
            continue
        text = agents.read_text(encoding="utf-8", errors="replace")
        if not any(marker in text for marker in orientation):
            issues.append(
                (
                    rel,
                    f"missing orientation section (need one of: {', '.join(orientation)})",
                )
            )
            continue
        m = re.search(r"## Purpose\s*\n(.*?)(?=\n##[^#]|\Z)", text, re.DOTALL)
        if m:
            body = m.group(1).strip()
            if len(body) < 20:
                issues.append(
                    (
                        rel,
                        f"## Purpose section too short ({len(body)} chars, need >= 20)",
                    )
                )
    return issues


def format_strict_issue_detail(
    *,
    link_issues: list[tuple[Path, int, str, str]],
    anchor_issues: list[tuple[Path, int, str, str]],
    anchor_checked: bool,
    spec_issues: list[tuple[Path, str]],
    coverage: list[Path],
    doc_missing_agents: list[Path],
    doc_missing_readme: list[Path],
    agents_no_readme: list[Path],
    readme_no_agents: list[Path],
    doc_agents_structure: list[tuple[Path, str]],
    security_version_issues: list[str],
    spec_coverage_issues: list[tuple[Path, str]],
    prose_src_issues: list[tuple[Path, int, str]],
    version_claim_issues: list[tuple[Path, int, str, str]],
    engine_count_issues: Sequence[tuple[Path, int, str, str]] = (),
    model_kind_issues: Sequence[tuple[Path, int, str, str]] = (),
    render_doc_issues: Sequence[tuple[Path, int, str, str]] = (),
) -> str:
    """Human-readable listing for terminal fix loops (stderr)."""
    chunks: list[str] = []
    chunks.append("Strict mode: full issue list (fix in source order)\n")

    if link_issues:
        chunks.append(f"## Broken relative links ({len(link_issues)})\n")
        for rel, lineno, href, reason in sorted(
            link_issues, key=lambda x: (str(x[0]), x[1])
        ):
            chunks.append(f"  {rel}:{lineno}  `{href}`  → {reason}\n")

    if anchor_checked and anchor_issues:
        chunks.append(f"## Bad markdown anchors ({len(anchor_issues)})\n")
        for rel, lineno, href, reason in sorted(
            anchor_issues, key=lambda x: (str(x[0]), x[1])
        ):
            chunks.append(f"  {rel}:{lineno}  `{href}`  → {reason}\n")

    if spec_issues:
        chunks.append(f"## AGENTS.md → missing SPEC.md ({len(spec_issues)})\n")
        for rel, msg in sorted(spec_issues, key=lambda x: str(x[0])):
            chunks.append(f"  `{rel}`  → {msg}\n")

    if coverage:
        chunks.append(f"## src/ dirs with .py but no AGENTS.md ({len(coverage)})\n")
        for rel in sorted(coverage, key=str):
            chunks.append(f"  `{rel}`\n")

    if doc_missing_agents:
        chunks.append(
            f"## docs/ maintained folders missing AGENTS.md ({len(doc_missing_agents)})\n"
        )
        for rel in sorted(doc_missing_agents, key=str):
            chunks.append(f"  `{rel}`\n")

    if doc_missing_readme:
        chunks.append(
            f"## docs/ maintained folders missing README.md ({len(doc_missing_readme)})\n"
        )
        for rel in sorted(doc_missing_readme, key=str):
            chunks.append(f"  `{rel}`\n")

    if agents_no_readme:
        chunks.append(
            f"## Directories with AGENTS.md but no README.md ({len(agents_no_readme)})\n"
        )
        for rel in sorted(agents_no_readme, key=str):
            chunks.append(f"  `{rel}`\n")

    if readme_no_agents:
        chunks.append(
            f"## Directories with README.md but no AGENTS.md ({len(readme_no_agents)})\n"
        )
        for rel in sorted(readme_no_agents, key=str):
            chunks.append(f"  `{rel}`\n")

    if doc_agents_structure:
        chunks.append(f"## docs/**/AGENTS.md structure ({len(doc_agents_structure)})\n")
        for rel, msg in sorted(doc_agents_structure, key=lambda x: str(x[0])):
            chunks.append(f"  `{rel}`  → {msg}\n")

    if security_version_issues:
        chunks.append(
            f"## SECURITY.md missing supported-versions row ({len(security_version_issues)})\n"
        )
        for msg in security_version_issues:
            chunks.append(f"  `{msg}`\n")

    if spec_coverage_issues:
        chunks.append(f"## src/ dirs missing SPEC.md ({len(spec_coverage_issues)})\n")
        for rel, msg in sorted(spec_coverage_issues, key=lambda x: str(x[0])):
            chunks.append(f"  `{rel}`  → {msg}\n")

    if prose_src_issues:
        chunks.append(f"## Prose src/<module>/ pattern ({len(prose_src_issues)})\n")
        for rel, lineno, reason in sorted(
            prose_src_issues, key=lambda x: (str(x[0]), x[1])
        ):
            chunks.append(f"  {rel}:{lineno}  → {reason}\n")

    if version_claim_issues:
        chunks.append(f"## Stale version claims ({len(version_claim_issues)})\n")
        for rel, lineno, claimed, reason in sorted(
            version_claim_issues, key=lambda x: (str(x[0]), x[1])
        ):
            chunks.append(f"  {rel}:{lineno}  `{claimed}`  → {reason}\n")

    if engine_count_issues:
        chunks.append(f"## Render-engine count claims ({len(engine_count_issues)})\n")
        for rel, lineno, claimed, reason in sorted(
            engine_count_issues, key=lambda x: (str(x[0]), x[1])
        ):
            chunks.append(f"  {rel}:{lineno}  `{claimed}`  → {reason}\n")

    if model_kind_issues:
        chunks.append(
            f"## Model-kind coverage claims ({len(model_kind_issues)})\n"
        )
        for rel, lineno, claimed, reason in sorted(
            model_kind_issues, key=lambda x: (str(x[0]), x[1])
        ):
            chunks.append(f"  {rel}:{lineno}  `{claimed}`  → {reason}\n")

    if render_doc_issues:
        chunks.append(
            f"## Render-doc frame coverage ({len(render_doc_issues)})\n"
        )
        for rel, lineno, claimed, reason in sorted(
            render_doc_issues, key=lambda x: (str(x[0]), x[1])
        ):
            chunks.append(f"  {rel}:{lineno}  `{claimed}`  → {reason}\n")

    chunks.append("\nTip: full tables also in docs/development/docs_audit_report.md\n")
    return "".join(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Markdown documentation audit for this repository."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any broken links, SPEC gaps, coverage gaps, AGENTS/README pairing, or doc AGENTS structure issues are found.",
    )
    parser.add_argument(
        "--check-anchors",
        action="store_true",
        help="Also verify that #fragments in relative .md links match a heading slug in the target file.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="With --strict: suppress per-issue detail on stderr (summary counts only).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Log extra diagnostics to stderr (e.g. markdown file count).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Run checks without writing docs_audit_report.md.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPO_ROOT / "docs" / "development" / "docs_audit_report.md",
        help="Optional report output path. Ignored when --no-write is set.",
    )
    args = parser.parse_args()

    if not logging.root.handlers:
        logging.basicConfig(
            level=logging.INFO if args.verbose else logging.WARNING,
            format="%(message)s",
            stream=sys.stderr,
            force=True,
        )

    if not REPO_ROOT.joinpath("pyproject.toml").exists():
        print("Run from repo root (pyproject.toml not found).", file=sys.stderr)
        return 1

    md_files = iter_markdown_files()
    if args.verbose:
        logging.info("Markdown files scanned: %d", len(md_files))
    link_issues = audit_broken_links(md_files)
    anchor_issues = audit_bad_markdown_anchors(md_files) if args.check_anchors else []
    spec_issues = audit_agents_spec()
    coverage = audit_src_agents_coverage()
    doc_missing_agents = audit_doc_maintained_missing_agents()
    doc_missing_readme = audit_doc_maintained_missing_readme()
    agents_no_readme = audit_agents_without_readme()
    readme_no_agents = audit_readme_without_agents()
    doc_agents_structure = audit_doc_agents_structure()
    security_version_issues = audit_security_supported_version()
    spec_coverage_issues = audit_src_spec_coverage()
    maintained_files = maintained_doc_files()
    prose_src_issues = audit_prose_src_patterns(maintained_files)
    version_claim_issues = audit_version_claims(maintained_files)
    engine_count_issues = audit_engine_count_claims(engine_count_scan_files())
    model_kind_issues = audit_model_kind_coverage(model_kind_scan_files())
    render_doc_issues = audit_render_doc_frame_coverage()

    report_path = args.report_path
    if not report_path.is_absolute():
        report_path = REPO_ROOT / report_path
    lines = [
        "# Documentation audit report",
        "",
        "Generated by `uv run --extra dev python docs/development/docs_audit.py`. Re-run after doc changes.",
        "",
        "## Broken relative Markdown links",
        "",
    ]
    if not link_issues:
        lines.append("None found (scoped scan).")
    else:
        lines.append("| Source | Line | Href | Issue |")
        lines.append("|--------|------|------|-------|")
        for rel, lineno, href, reason in sorted(
            link_issues, key=lambda x: (str(x[0]), x[1])
        ):
            lines.append(f"| `{rel}` | {lineno} | `{href}` | {reason} |")
    lines.extend(
        [
            "",
            "## Suspicious markdown anchors (optional --check-anchors)",
            "",
        ]
    )
    if not args.check_anchors:
        lines.append(
            "Not run (pass `--check-anchors` to validate `#fragments` against heading slugs)."
        )
    elif not anchor_issues:
        lines.append("None found.")
    else:
        lines.append("| Source | Line | Href | Issue |")
        lines.append("|--------|------|------|-------|")
        for rel, lineno, href, reason in sorted(
            anchor_issues, key=lambda x: (str(x[0]), x[1])
        ):
            lines.append(f"| `{rel}` | {lineno} | `{href}` | {reason} |")
    lines.extend(
        [
            "",
            "## AGENTS.md referencing sibling SPEC.md",
            "",
        ]
    )
    if not spec_issues:
        lines.append("None (all SPEC footers have a file).")
    else:
        for rel, msg in spec_issues:
            lines.append(f"- `{rel}`: {msg}")
    lines.extend(
        [
            "",
            "## src/ directories with .py but no AGENTS.md",
            "",
        ]
    )
    if not coverage:
        lines.append("None.")
    else:
        for rel in coverage:
            lines.append(f"- `{rel}`")
    lines.extend(
        [
            "",
            "## docs/ maintained folders missing AGENTS.md",
            "",
        ]
    )
    if not doc_missing_agents:
        lines.append("None.")
    else:
        for rel in doc_missing_agents:
            lines.append(f"- `{rel}`")
    lines.extend(
        [
            "",
            "## docs/ maintained folders missing README.md",
            "",
        ]
    )
    if not doc_missing_readme:
        lines.append("None.")
    else:
        for rel in doc_missing_readme:
            lines.append(f"- `{rel}`")
    lines.extend(
        [
            "",
            "## Directories with AGENTS.md but no README.md (src, doc, .github, repo root)",
            "",
        ]
    )
    if not agents_no_readme:
        lines.append("None.")
    else:
        for rel in agents_no_readme:
            lines.append(f"- `{rel}`")
    lines.extend(
        [
            "",
            "## Directories with README.md but no AGENTS.md (src, doc, .github, repo root)",
            "",
        ]
    )
    if not readme_no_agents:
        lines.append("None.")
    else:
        for rel in readme_no_agents:
            lines.append(f"- `{rel}`")
    lines.extend(
        [
            "",
            "## SECURITY.md supported-versions rows",
            "",
        ]
    )
    if not security_version_issues:
        lines.append("None.")
    else:
        for msg in security_version_issues:
            lines.append(f"- {msg}")
    lines.extend(
        [
            "",
            "## src/ module directories missing SPEC.md",
            "",
        ]
    )
    if not spec_coverage_issues:
        lines.append("None.")
    else:
        for rel, msg in spec_coverage_issues:
            lines.append(f"- `{rel}`: {msg}")
    lines.extend(
        [
            "",
            "## Prose references to the removed src/<module>/ layout",
            "",
        ]
    )
    if not prose_src_issues:
        lines.append("None found.")
    else:
        lines.append("| Source | Line | Issue |")
        lines.append("|--------|------|-------|")
        for rel, lineno, reason in sorted(
            prose_src_issues, key=lambda x: (str(x[0]), x[1])
        ):
            lines.append(f"| `{rel}` | {lineno} | {reason} |")
    lines.extend(
        [
            "",
            "## Version claims contradicting pyproject.toml",
            "",
        ]
    )
    if not version_claim_issues:
        lines.append("None found.")
    else:
        lines.append("| Source | Line | Claimed | Issue |")
        lines.append("|--------|------|---------|-------|")
        for rel, lineno, claimed, reason in sorted(
            version_claim_issues, key=lambda x: (str(x[0]), x[1])
        ):
            lines.append(f"| `{rel}` | {lineno} | {claimed} | {reason} |")
    lines.extend(
        [
            "",
            "## Render-engine count claims contradicting framework_registry",
            "",
        ]
    )
    if not engine_count_issues:
        lines.append("None found.")
    else:
        lines.append("| Source | Line | Claimed | Issue |")
        lines.append("|--------|------|---------|-------|")
        for rel, lineno, claimed, reason in sorted(
            engine_count_issues, key=lambda x: (str(x[0]), x[1])
        ):
            lines.append(f"| `{rel}` | {lineno} | {claimed} | {reason} |")
    lines.extend(
        [
            "",
            "## Model-kind coverage claims contradicting framework_registry",
            "",
        ]
    )
    if not model_kind_issues:
        lines.append("None found.")
    else:
        lines.append("| Source | Line | Frame | Issue |")
        lines.append("|--------|------|-------|-------|")
        for rel, lineno, frame, reason in sorted(
            model_kind_issues, key=lambda x: (str(x[0]), x[1])
        ):
            lines.append(f"| `{rel}` | {lineno} | {frame} | {reason} |")
    lines.extend(
        [
            "",
            "## Registry frames missing from the maintained render docs",
            "",
        ]
    )
    if not render_doc_issues:
        lines.append("None found.")
    else:
        lines.append("| Source | Line | Frame | Issue |")
        lines.append("|--------|------|-------|-------|")
        for rel, lineno, frame, reason in sorted(
            render_doc_issues, key=lambda x: (str(x[0]), x[1])
        ):
            lines.append(f"| `{rel}` | {lineno} | {frame} | {reason} |")
    lines.extend(
        [
            "",
            "## docs/**/AGENTS.md structure (Overview/Purpose)",
            "",
        ]
    )
    if not doc_agents_structure:
        lines.append("None.")
    else:
        for rel, msg in doc_agents_structure:
            lines.append(f"- `{rel}`: {msg}")

    if args.no_write:
        print("Report not written (--no-write).")
    else:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            display_path = report_path.relative_to(REPO_ROOT)
        except ValueError:
            display_path = report_path
        print(f"Wrote {display_path}")
    print(f"Broken links: {len(link_issues)}")
    print(f"Bad markdown anchors: {len(anchor_issues)}")
    print(f"AGENTS/SPEC gaps: {len(spec_issues)}")
    print(f"src dirs missing AGENTS.md: {len(coverage)}")
    print(f"doc maintained missing AGENTS.md: {len(doc_missing_agents)}")
    print(f"doc maintained missing README.md: {len(doc_missing_readme)}")
    print(f"AGENTS without README: {len(agents_no_readme)}")
    print(f"README without AGENTS: {len(readme_no_agents)}")
    print(f"doc AGENTS structure: {len(doc_agents_structure)}")
    print(f"SECURITY.md version row: {len(security_version_issues)}")
    print(f"src dirs missing SPEC.md: {len(spec_coverage_issues)}")
    print(f"prose src/<module> pattern: {len(prose_src_issues)}")
    print(f"stale version claims: {len(version_claim_issues)}")
    print(f"engine count claims: {len(engine_count_issues)}")
    print(f"model-kind coverage claims: {len(model_kind_issues)}")
    print(f"render-doc frame coverage: {len(render_doc_issues)}")
    total_issues = (
        len(link_issues)
        + len(spec_issues)
        + len(coverage)
        + len(doc_missing_agents)
        + len(doc_missing_readme)
        + len(agents_no_readme)
        + len(readme_no_agents)
        + len(doc_agents_structure)
        + len(security_version_issues)
        + len(spec_coverage_issues)
        + len(prose_src_issues)
        + len(version_claim_issues)
        + len(engine_count_issues)
        + len(model_kind_issues)
        + len(render_doc_issues)
        + (len(anchor_issues) if args.check_anchors else 0)
    )
    if args.strict and total_issues > 0:
        sys.stdout.flush()
        if not args.quiet:
            print(
                format_strict_issue_detail(
                    link_issues=link_issues,
                    anchor_issues=anchor_issues,
                    anchor_checked=args.check_anchors,
                    spec_issues=spec_issues,
                    coverage=coverage,
                    doc_missing_agents=doc_missing_agents,
                    doc_missing_readme=doc_missing_readme,
                    agents_no_readme=agents_no_readme,
                    readme_no_agents=readme_no_agents,
                    doc_agents_structure=doc_agents_structure,
                    security_version_issues=security_version_issues,
                    spec_coverage_issues=spec_coverage_issues,
                    prose_src_issues=prose_src_issues,
                    version_claim_issues=version_claim_issues,
                    engine_count_issues=engine_count_issues,
                    model_kind_issues=model_kind_issues,
                    render_doc_issues=render_doc_issues,
                ),
                file=sys.stderr,
                end="",
            )
        print(f"Strict mode: {total_issues} issue(s); exiting 1.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

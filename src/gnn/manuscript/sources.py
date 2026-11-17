"""Source-surface census for the manuscript token map.

Every quantitative token is derived here from a committed source surface —
``pyproject.toml``, ``src/gnn/STEP_INDEX.md``, the framework registry, the
MCP audit ledger, the model-family manifest, the exemplar corpus under
``input/gnn_files/``, and ``CHANGELOG.md`` — always read through
:class:`~gnn.manuscript.snapshot.RepositorySnapshot` so the numbers describe
one commit. The two generated coverage sentences (``outside_corpus_note``,
``corpus_coverage_notes``) state the relationship between manifest families
and the corpus tree instead of letting hand-typed prose stale against it.
"""

from __future__ import annotations

import ast
import json
import re
import tomllib
from collections.abc import Sequence
from pathlib import Path

from gnn.manuscript.snapshot import RepositorySnapshot


def _line_count(text: str) -> int:
    """Lines in *text*: newline characters plus a final unterminated line."""
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def _read_pyproject_version(snapshot: RepositorySnapshot) -> str:
    text = snapshot.read_text("pyproject.toml")
    version = tomllib.loads(text).get("project", {}).get("version")
    if version:
        return str(version)
    # Fallback: regex the [project] version line.
    match = re.search(r'^\s*version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else "0.0.0"


def _count_python(snapshot: RepositorySnapshot) -> tuple[int, int]:
    """Return ``(file_count, total_lines)`` for package ``.py`` sources.

    The test suite lives outside ``src/`` and has its own tokens
    (``GNN_TEST_FILE_COUNT`` / ``GNN_TEST_FUNCTION_COUNT``), so it is never
    counted here. Both the file set
    and the bytes come from the snapshot's commit, so the pair reproduces from
    that commit alone.
    """
    sources = list(snapshot.glob("src", "*.py"))
    snapshot.prefetch(sources)
    files = 0
    lines = 0
    for py in sources:
        files += 1
        lines += _line_count(snapshot.read_text(py))
    return files, lines


def _count_packages(snapshot: RepositorySnapshot) -> int:
    """Count importable subpackages under ``src/gnn/``."""
    return len(
        {
            rel.parts[2]
            for rel in snapshot.glob("src/gnn", "__init__.py")
            if len(rel.parts) == 4
        }
    )


_TEST_FUNCTION_RE = re.compile(r"^\s*(?:async\s+)?def (test_\w+)", re.MULTILINE)


def _count_test_functions(snapshot: RepositorySnapshot) -> tuple[int, int]:
    """Return ``(test_file_count, test_function_count)`` via static text scan."""
    file_count = 0
    func_count = 0
    pattern = _TEST_FUNCTION_RE
    tests = snapshot.glob("tests", "test_*.py")
    snapshot.prefetch(tests)
    for py in tests:
        file_count += 1
        func_count += len(pattern.findall(snapshot.read_text(py)))
    return file_count, func_count


def _pipeline_steps(snapshot: RepositorySnapshot) -> list[tuple[int, str]]:
    """Return sorted ``(step_number, script_name)`` for ``N_*.py`` step modules.

    Step modules are top-level files directly under ``src/gnn/`` — they are
    siblings of the source packages, never members of one, which is why
    ``GNN_STEP_COUNT`` and ``GNN_SRC_PACKAGE_COUNT`` count disjoint sets.
    """
    steps: list[tuple[int, str]] = []
    for rel in snapshot.glob("src/gnn", "[0-9]*_*.py"):
        if len(rel.parts) != 3:
            continue
        match = re.match(r"(\d+)_", rel.name)
        if match:
            steps.append((int(match.group(1)), rel.name))
    return sorted(steps)


def _step_purposes(snapshot: RepositorySnapshot) -> dict[int, str]:
    """Parse ``src/gnn/STEP_INDEX.md`` master table for per-step purposes."""
    purposes: dict[int, str] = {}
    for line in snapshot.read_text("src/gnn/STEP_INDEX.md").splitlines():
        cells = [c.strip() for c in line.split("|")]
        # Master table rows look like: | 0 | `0_template.py` | template/ | Global | Purpose | ...
        if len(cells) >= 7 and cells[1].isdigit():
            purposes[int(cells[1])] = cells[5]
    return purposes


def _module_literal(text: str, name: str) -> object | None:
    """Return the literal value assigned to *name* in *text*, or ``None``.

    The value is recovered with :mod:`ast` rather than by importing the module,
    so the producer keeps working when the heavy render/pipeline packages'
    dependencies are absent.
    """
    try:
        module = ast.parse(text)
    except SyntaxError:  # pragma: no cover - defensive
        return None
    for node in module.body:
        targets: list[ast.expr]
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.Assign):
            targets = list(node.targets)
        else:
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == name for target in targets
        ):
            continue
        value = node.value
        if value is None:  # pragma: no cover - bare annotation
            continue
        if isinstance(value, ast.Call) and value.args:
            value = value.args[0]
        try:
            literal: object = ast.literal_eval(value)
        except (ValueError, TypeError):  # pragma: no cover - defensive
            return None
        return literal
    return None


def _registry_specs(snapshot: RepositorySnapshot) -> dict[str, dict]:
    """Return ``FRAMEWORK_REGISTRY`` as ``{key: spec}`` from the snapshot.

    This is the single source of truth for every backend capability the
    manuscript reports: ``supports_execution`` (Step-12 executor present) and
    ``supports_continuous`` (linear-Gaussian semantics) are the registry's own
    fields, never restated by hand anywhere downstream.
    """
    text = snapshot.read_text("src/gnn/render/framework_registry.py")
    if not text:
        return {}
    data = _module_literal(text, "FRAMEWORK_REGISTRY")
    if isinstance(data, dict):
        return {str(key): spec for key, spec in data.items() if isinstance(spec, dict)}
    # Fallback: text scan (capability flags unknown, reported as False).
    return {
        block.group("key"): {"name": block.group("name")}
        for block in re.finditer(
            r'"(?P<key>[a-z_]+)"\s*:\s*\{[^{}]*?"name"\s*:\s*"(?P<name>[^"]+)"',
            text,
            re.DOTALL,
        )
    }


def _backends(snapshot: RepositorySnapshot) -> list[tuple[str, str, bool]]:
    """Return ``(key, display_name, supports_execution)`` from the registry.

    ``supports_execution`` is the registry's own field: it separates the render
    targets from the subset that also has a Step-12 executor.
    """
    return [
        (key, str(spec.get("name", key)), bool(spec.get("supports_execution", False)))
        for key, spec in _registry_specs(snapshot).items()
    ]


def _maintained_frameworks(snapshot: RepositorySnapshot) -> tuple[str, ...]:
    """Return ``MAINTAINED_FRAMEWORKS`` from the cross-framework gate.

    This is the set the reliability gate will actually profile:
    ``src/gnn/pipeline/cross_framework_reliability.py`` raises
    ``ValueError("Unprofiled frameworks: ...")`` for anything outside it. It is
    a strict subset of the registry, so no manuscript sentence may use
    ``GNN_BACKEND_COUNT`` to describe what the gate profiles.
    """
    text = snapshot.read_text("src/gnn/pipeline/cross_framework_reliability.py")
    if not text:
        return ()
    value = _module_literal(text, "MAINTAINED_FRAMEWORKS")
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return ()


def _mcp_counts(snapshot: RepositorySnapshot) -> dict[str, int]:
    """MCP tool/module counts.

    ``tools``/``modules`` are read from the project's maintained MCP audit ledger
    (``src/gnn/mcp/audit_report.json``), which is regenerated by
    ``uv run python src/gnn/mcp/validate_tools.py`` by
    actually loading every MCP module and counting registered tools — a count
    that cannot be reproduced by static text scanning. This is a *source ledger*
    (one of the manuscript's allowed evidence types), and it is only as current
    as the committed file: ``tests/mcp/test_mcp_audit.py`` fails when the
    ledger drifts from the live registry. The ``files`` count is recomputed live
    from the snapshot each run.
    """
    counts = {"tools": 0, "modules_total": 0, "modules_loaded": 0, "files": 0}
    raw = snapshot.read_text("src/gnn/mcp/audit_report.json")
    if raw:
        data = json.loads(raw)
        counts["tools"] = int(data.get("tools_total", 0))
        counts["modules_total"] = int(data.get("modules_total", 0))
        counts["modules_loaded"] = int(data.get("modules_loaded", 0))
    counts["files"] = len(snapshot.glob("src/gnn", "mcp.py"))
    return counts


def _families(snapshot: RepositorySnapshot) -> list[dict]:
    raw = snapshot.read_text("input/model_family_manifest.json")
    if not raw:
        return []
    data = json.loads(raw)
    families = data.get("families", [])
    return families if isinstance(families, list) else []


def _count_files(snapshot: RepositorySnapshot, prefix: str, pattern: str) -> int:
    return len(snapshot.glob(prefix, pattern))


# Markdown filenames the pipeline never treats as a model source. Mirrors
# ``gnn.discovery.NON_MODEL_MARKDOWN_FILENAMES`` / ``NON_MODEL_MARKDOWN_SUFFIXES``
# rather than importing them, so the producer stays free of pipeline imports.
# ``test_producer_model_census_matches_pipeline_discovery`` pins the two equal.
_NON_MODEL_MARKDOWN_FILENAMES = frozenset(
    {
        "agents.md",
        "changelog.md",
        "contributing.md",
        "index.md",
        "license.md",
        "readme.md",
    }
)
_NON_MODEL_MARKDOWN_SUFFIXES = (".example.md", ".template.md")
# ``## GNNSection`` must open a line to be the model's header. A doc that merely
# *names* the header inline (`` `## GNNSection` ``) is prose, not a model — which
# is how a fixture README came within one commit of being counted as a model.
_GNN_SECTION_HEADER = re.compile(r"^## GNNSection\b", re.MULTILINE)


def _is_model_markdown(rel: Path) -> bool:
    """Filename half of the model test, mirroring ``gnn.discovery``."""
    name = rel.name.lower()
    if name in _NON_MODEL_MARKDOWN_FILENAMES:
        return False
    return not any(name.endswith(suffix) for suffix in _NON_MODEL_MARKDOWN_SUFFIXES)


def _models_under(snapshot: RepositorySnapshot, prefix: str) -> list[Path]:
    """Markdown files under *prefix* that are GNN model sources."""
    candidates = [md for md in snapshot.glob(prefix, "*.md") if _is_model_markdown(md)]
    snapshot.prefetch(candidates)
    return [
        md for md in candidates if _GNN_SECTION_HEADER.search(snapshot.read_text(md))
    ]


def _example_models(snapshot: RepositorySnapshot) -> list[Path]:
    """Return the GNN example *models* under ``input/gnn_files``.

    A GNN model file is identified by its mandatory ``## GNNSection`` header at
    the start of a line, and by not being one of the README/AGENTS/INDEX
    scaffolds the pipeline itself excludes. Model corpora that live outside
    ``input/gnn_files`` are not in this set; ``GNN_OUTSIDE_CORPUS_NOTE`` reports
    those.
    """
    return _models_under(snapshot, "input/gnn_files")


def _outside_corpus_dirs(snapshot: RepositorySnapshot) -> list[tuple[str, int]]:
    """Model directories directly under ``input/`` other than ``gnn_files``.

    Returns ``(path, model_count)`` pairs, sorted by path, where *model_count*
    is the number of ``## GNNSection``-bearing ``.md`` files the directory
    holds. ``_example_models`` deliberately scans only ``input/gnn_files``, so
    without this the rest of the ``input/`` tree is invisible to every token and
    can only be described by typed prose — which is how a fixture came to carry
    a ``## GNNSection`` while no manifest, count, table or figure knew it
    existed.
    """
    names = sorted(
        {
            rel.parts[1]
            for rel in snapshot.glob("input", "*")
            if len(rel.parts) > 2 and rel.parts[1] != "gnn_files"
        }
    )
    return [
        (f"input/{name}", len(_models_under(snapshot, f"input/{name}")))
        for name in names
    ]


def outside_corpus_note(
    outside_dirs: Sequence[tuple[str, int]],
    family_target_dirs: set[str] | Sequence[str],
    example_count: int,
) -> str:
    """Build the generated sentences about model files outside the corpus tree.

    This is a *sentence*, not a count, for the same reason
    ``corpus_coverage_notes`` is: what goes stale is the relationship between a
    directory, its model files, and the manifest — not any one number. Typed
    prose describing that relationship stayed true only by luck through two
    remediation passes, because every count beside it was computed from
    ``input/gnn_files`` alone and so could not contradict it.

    Flips on all three axes: a directory appearing or disappearing, a model file
    being added to or removed from one, and a manifest family being pointed at
    one.
    """
    registered = {str(d).rstrip("/") for d in family_target_dirs}
    if not outside_dirs:
        return (
            f"Every model file under `input/` lives in that subtree, so the "
            f"{example_count}-file count covers the whole tree."
        )
    fragments: list[str] = []
    for path, count in outside_dirs:
        noun = "model file" if count == 1 else "model files"
        held = f"{count} {noun}" if count else "no model files"
        claim = (
            "is registered as a manifest family target directory"
            if path in registered
            else "is registered by no manifest family"
        )
        fragments.append(f"`{path}/` holds {held} and {claim}")
    listed = "; ".join(fragments)
    total = sum(count for _, count in outside_dirs)
    if total == 0:
        tail = (
            f"No model file lies outside `input/gnn_files`, so the "
            f"{example_count}-file count covers every model in the tree."
        )
    elif total == 1:
        tail = f"That model file is outside the {example_count}-file count above."
    else:
        tail = (
            f"Those {total} model files are outside the "
            f"{example_count}-file count above."
        )
    return f"Outside that subtree, {listed}. {tail}"


def corpus_coverage_notes(
    family_target_dirs: set[str] | Sequence[str],
    unscanned_corpus_dirs: Sequence[str],
    example_count: int,
) -> tuple[str, str]:
    """Build the two generated sentences about family/corpus coverage.

    Returns ``(unscanned_corpus_note, target_dir_coverage_note)``: the first for
    the model-family coverage paragraph, the second for the reproduction
    section's ``--target-dir`` note.

    These are *sentences*, not counts, because the claim that goes stale is the
    relationship, not a number. A commit repointed the ``multiagent`` family's
    ``target_dir`` from ``input/multi_agent_models`` into ``input/gnn_files/``;
    every count beside the three prose sites describing the old layout stayed
    correct, so every count-based check stayed green. Generating the whole clause
    from the manifest removes the typed claim rather than replacing it with a
    newer true value that can stale the same way.
    """
    total = len(set(family_target_dirs))
    outside = list(unscanned_corpus_dirs)
    target_word = "directory" if total == 1 else "directories"
    if not outside:
        return (
            f"Registered but outside the scanned tree: none, because all "
            f"{total} target {target_word} are themselves `input/gnn_files` "
            f"corpus directories.",
            f"All {total} registered family target {target_word} lie inside "
            f"that tree, so a single invocation reaches every registered family.",
        )
    outside_word = "directory" if len(outside) == 1 else "directories"
    outside_verb = "lies" if len(outside) == 1 else "lie"
    outside_needs = "needs" if len(outside) == 1 else "need"
    listed = ", ".join(f"`{d}`" for d in outside)
    return (
        f"Registered but outside the scanned tree: {listed} "
        f"({len(outside)} of the {total} target {target_word}), whose models "
        f"the gates exercise even though they are not among the "
        f"{example_count} models under `input/gnn_files`.",
        f"{len(outside)} registered family target {outside_word} ({listed}) "
        f"{outside_verb} outside that tree and {outside_needs} a separate run.",
    )


_RELEASE_HEADING_RE = re.compile(
    r"^##\s+\[(?P<version>[^\]]+)\]\s*[—-]\s*(?P<date>[0-9-]+)"
)
_RELEASE_CODENAME_RE = re.compile(r"^>\s*\*\*(?P<codename>[^*]+?)\.?\*\*")


def _release_metadata(snapshot: RepositorySnapshot) -> list[tuple[str, str, str]]:
    """Parse ``CHANGELOG.md`` into ``(version, date, codename)`` triples.

    Each release heading (``## [3.2.0] — 2026-09-02``) is followed by a
    block-quoted summary whose leading bold run is the release codename
    (``> **Exemplar Gold Standard.** ...``). Newest release first.
    """
    lines = snapshot.read_text("CHANGELOG.md").splitlines()
    if not lines:
        return []
    heading = _RELEASE_HEADING_RE
    codename = _RELEASE_CODENAME_RE
    releases: list[tuple[str, str, str]] = []
    for index, line in enumerate(lines):
        match = heading.match(line)
        if not match:
            continue
        name = ""
        for follow in lines[index + 1 : index + 6]:
            found = codename.match(follow)
            if found:
                name = found.group("codename").strip()
                break
        releases.append((match.group("version"), match.group("date"), name))
    return releases

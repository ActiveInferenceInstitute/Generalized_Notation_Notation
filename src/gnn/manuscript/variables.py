"""Deterministic manuscript-variable producer for GeneralizedNotationNotation.

This module introspects the *live* GNN repository and emits a flat
``dict[str, str]`` of ``{{UPPERCASE_KEY}}`` token values consumed by the
docxology/template manuscript renderer
(``infrastructure.rendering.manuscript_injection``).

Design contract
---------------
* **Nothing is hard-coded.** Every quantitative token is computed from a source
  surface in the repository (``pyproject.toml``, ``input/model_family_manifest.json``,
  ``src/gnn/render/framework_registry.py``, ``src/gnn/mcp/audit_report.json``,
  ``src/gnn/STEP_INDEX.md``, ``CHANGELOG.md``, direct filesystem counts, and the
  exemplar corpus under ``input/gnn_files/``). If a source surface changes,
  re-running this producer changes the manuscript. The model-kind token set
  treats those exemplars as either continuous linear-Gaussian specs (the
  ``continuous/`` folder, which the corpus reserves for that kind — its specs
  declare the ``F``/``H``/``Q``/``R`` parameterization) or discrete categorical
  specs (every other exemplar folder), matching the binary
  discrete-vs-continuous handling the render path applies to each model.
  ``GNN_FRAMEWORK_TABLE``/``GNN_MODEL_KIND_TABLE`` render from the framework
  registry's own capability flags.
* **Counts describe one commit.** Every source surface is read from the git
  commit named by ``GNN_GIT_COMMIT`` (``HEAD``), not from the working tree: the
  file set comes from ``git ls-tree`` and the bytes come from ``git cat-file``
  (see :class:`RepositorySnapshot`). A tracked-but-modified file therefore
  cannot inflate a published count, a tracked-but-deleted file cannot silently
  vanish from one, and any checkout of that commit reproduces the same numbers.
  When git is unavailable (a source tarball, a vendored copy) the snapshot falls
  back to the working tree and ``GNN_GIT_COMMIT`` reports the ``unknown``
  sentinel: the numbers are checkout-dependent, so provenance is visible in
  the token map itself, and the token gate
  (``scripts/check_manuscript_tokens.py``) and the figure build both FAIL on
  the sentinel — a manuscript whose every count names no commit is not
  publishable.
  ``tests/`` is counted once, by the test tokens, and is excluded from the
  source-file and LOC tokens.
* **Deterministic.** No timestamps, no wall-clock, no randomness. Two runs over an
  unchanged commit produce byte-identical JSON.
* **Dependency-light.** Standard library + ``yaml`` only (already a GNN dependency).
  No import of the heavy pipeline packages, so the producer runs even when optional
  simulation backends are absent.

The thin orchestrator ``scripts/z_generate_manuscript_variables.py`` wires
:func:`generate_variables` to the template's manuscript hydration.

Round-trip helpers: :func:`save_variables` persists a token map,
:func:`load_variables` reads and validates one back, and
:func:`token_checksum` fingerprints a map's canonical JSON so manuscript
tooling can detect token drift without diffing whole documents.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path

try:
    import yaml as _yaml
except ModuleNotFoundError:  # pragma: no cover - yaml is a GNN dependency
    _yaml = None

# Internal layout: snapshot reads in snapshot.py, per-surface census in
# sources.py, table renderers in tables.py. This module keeps the public
# surface: token-map assembly, the producer-owned metadata writers, and the
# JSON round-trip helpers.
from gnn.manuscript.snapshot import RepositorySnapshot
from gnn.manuscript.sources import (
    _backends,
    _count_files,
    _count_packages,
    _count_python,
    _count_test_functions,
    _example_models,
    _families,
    _maintained_frameworks,
    _mcp_counts,
    _models_under,
    _outside_corpus_dirs,
    _pipeline_steps,
    _read_pyproject_version,
    _registry_specs,
    _release_metadata,
    _step_purposes,
    corpus_coverage_notes,
    outside_corpus_note,
)
from gnn.manuscript.tables import (
    _capability_clause,  # noqa: F401 - re-exported for tests/main/test_manuscript_variables.py
    _cross_framework_selection,
    _render_backend_table,
    _render_family_table,
    _render_framework_capability_table,
    _render_model_kind_table,
    _render_step_table,
    select_cross_framework_family,
)

__all__ = [
    "RepositorySnapshot",
    "corpus_coverage_notes",
    "generate_variables",
    "load_variables",
    "save_variables",
    "config_metadata_drift",
    "preamble_metadata_drift",
    "select_cross_framework_family",
    "sync_config_metadata",
    "sync_preamble_metadata",
    "token_checksum",
]


def _load_config(project_root: Path) -> dict:
    config_path = project_root / "manuscript" / "config.yaml"
    if not config_path.is_file() or _yaml is None:
        return {}
    loaded = _yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def generate_variables(
    project_root: Path, *, snapshot: RepositorySnapshot | None = None
) -> dict[str, str]:
    """Compute the manuscript token map by introspecting the live repository.

    Args:
        project_root: Path to the GeneralizedNotationNotation project root.
        snapshot: Optional immutable source snapshot for PR-base audits. Config
            metadata is read from project_root, allowing the same current
            producer to inspect archived base evidence without checking it out.

    Returns:
        Flat ``dict[str, str]`` of ``UPPERCASE_KEY`` -> value. All values are
        strings (the template injector substitutes text). Every count is read
        from the commit reported as ``GNN_GIT_COMMIT``, so the mapping is
        deterministic for that commit regardless of working-tree state. The
        authored manuscript config (``manuscript/config.yaml``) is the one
        working-tree read: it is the input a developer edits and re-renders.

    Raises:
        ValueError: If the cross-framework family declares only frameworks the
            reliability gate refuses to profile.
    """
    project_root = Path(project_root).resolve()
    config = _load_config(project_root)
    paper = config.get("paper", {}) if isinstance(config, dict) else {}
    authors = config.get("authors", []) if isinstance(config, dict) else []
    author = authors[0] if authors else {}
    publication = config.get("publication", {}) if isinstance(config, dict) else {}
    metadata = config.get("metadata", {}) if isinstance(config, dict) else {}
    keywords = config.get("keywords", []) if isinstance(config, dict) else []

    snapshot = snapshot if snapshot is not None else RepositorySnapshot(project_root)
    version = _read_pyproject_version(snapshot)
    steps = _pipeline_steps(snapshot)
    purposes = _step_purposes(snapshot)
    specs = _registry_specs(snapshot)
    backends = _backends(snapshot)
    maintained = _maintained_frameworks(snapshot)
    mcp = _mcp_counts(snapshot)
    families = _families(snapshot)
    py_files, py_loc = _count_python(snapshot)
    package_count = _count_packages(snapshot)
    test_file_count, test_func_count = _count_test_functions(snapshot)

    # --- Derived tables (multi-line tokens) ------------------------------------
    step_table = _render_step_table(steps, purposes)
    family_table = _render_family_table(families, specs)
    backend_table = _render_backend_table(backends)
    framework_table = _render_framework_capability_table(specs)
    model_kind_table = _render_model_kind_table(specs)
    (
        cross_family_name,
        cross_declared_keys,
        cross_profiled_keys,
    ) = _cross_framework_selection(families, backends, maintained)

    family_names = [f.get("name", "?") for f in families]
    backend_names = [name for _, name, _ in backends]
    executable_backends = [name for _, name, executes in backends if executes]
    name_by_key = {key: name for key, name, _ in backends}
    maintained_names = [name_by_key.get(key, key) for key in maintained]
    cross_declared_names = [name_by_key.get(k, k) for k in cross_declared_keys]
    cross_profiled_names = [name_by_key.get(k, k) for k in cross_profiled_keys]

    # Per-step scalars keyed by module stem, so prose never hand-types a step
    # number the registry owns (`Parsing ({{GNN_STEP_GNN}})`).
    step_tokens = {
        f"GNN_STEP_{re.sub(r'^[0-9]+_', '', script.removesuffix('.py')).upper()}": str(
            number
        )
        for number, script in steps
    }

    releases = _release_metadata(snapshot)
    release_by_version = {ver: (date, name) for ver, date, name in releases}
    release_date, release_codename = release_by_version.get(version, ("", ""))
    # The release that introduced the long-running orchestration contracts is the
    # oldest whose CHANGELOG codename names orchestration.
    orchestration_version = next(
        (ver for ver, _, name in reversed(releases) if "orchestration" in name.lower()),
        version,
    )

    example_models = _example_models(snapshot)
    # Continuous exemplars are the model sources under the corpus's
    # continuous/ folder — the folder input/gnn_files/INDEX.md reserves for
    # the linear-Gaussian kind and the split the render path applies
    # (continuous-state iff under continuous/). Discrete is the rest of the
    # corpus, so the two exemplar counts partition GNN_EXAMPLE_COUNT.
    continuous_exemplars = _models_under(snapshot, "input/gnn_files/continuous")
    discrete_exemplars = len(example_models) - len(continuous_exemplars)
    corpus_dirs = sorted(
        {
            rel.parts[2]
            for rel in snapshot.glob("input/gnn_files", "*")
            if len(rel.parts) > 3
        }
    )
    family_target_dirs = {
        str(f.get("target_dir", "")) for f in families if f.get("target_dir")
    }
    # Corpus directories under input/gnn_files/ that no manifest family claims,
    # and family target_dirs that live outside input/gnn_files/. Both sets are
    # emitted so prose can name the gap instead of implying the three sets
    # (families, corpus directories, example models) are coextensive.
    registered_corpus_dirs = {
        Path(d).name for d in family_target_dirs if d.startswith("input/gnn_files/")
    }
    unregistered_corpus_dirs = [
        name for name in corpus_dirs if name not in registered_corpus_dirs
    ]
    unscanned_corpus_dirs = sorted(
        d for d in family_target_dirs if not d.startswith("input/gnn_files/")
    )
    unscanned_corpus_note, target_dir_coverage_note = corpus_coverage_notes(
        family_target_dirs, unscanned_corpus_dirs, len(example_models)
    )
    outside_corpus_dirs = _outside_corpus_dirs(snapshot)
    outside_corpus_models = sum(count for _, count in outside_corpus_dirs)
    figure_count = _count_files(snapshot, "output", "*.png")
    manuscript_figure_count = _count_files(snapshot, "output/figures", "*.png")
    # fleet-logs are historical run records, not maintained documentation; the
    # doc gates exclude them, so the manuscript's count does too.
    doc_file_count = _count_files(snapshot, "docs", "*.md") - _count_files(
        snapshot, "docs/development/fleet-logs", "*.md"
    )

    variables: dict[str, str] = {
        # Identity / config
        "GNN_TITLE": str(paper.get("title", "GeneralizedNotationNotation")),
        "GNN_SUBTITLE": str(paper.get("subtitle", "")),
        "GNN_VERSION": version,
        "GNN_FIRST_AUTHOR": str(author.get("name", "")),
        "GNN_AUTHOR_ORCID": str(author.get("orcid", "")),
        "GNN_AUTHOR_AFFILIATION": str(author.get("affiliation", "")),
        "GNN_AUTHOR_EMAIL": str(author.get("email", "")),
        "GNN_PUBLICATION_YEAR": str(publication.get("year", "")),
        "GNN_REPO_URL": str(publication.get("github_repository", "")),
        "GNN_LICENSE": str(metadata.get("license", "")),
        "GNN_KEYWORDS": ", ".join(str(k) for k in keywords),
        "GNN_GIT_COMMIT": snapshot.commit,
        "GNN_RELEASE_CODENAME": release_codename,
        "GNN_RELEASE_DATE": release_date,
        "GNN_ORCHESTRATION_VERSION": orchestration_version,
        # Pipeline structure
        "GNN_STEP_COUNT": str(len(steps)),
        # The pipeline step modules (the src/gnn/N_*.py orchestrators
        # GNN_STEP_TABLE enumerates) — the modules of the 25-step pipeline;
        # NOT the importable subpackages, which GNN_SRC_PACKAGE_COUNT owns.
        "GNN_MODULE_COUNT": str(len(steps)),
        "GNN_STEP_FIRST": str(steps[0][0]) if steps else "0",
        "GNN_STEP_LAST": str(steps[-1][0]) if steps else "0",
        "GNN_STEP_RANGE": f"{steps[0][0]}–{steps[-1][0]}" if steps else "0",
        "GNN_STEP_TABLE": step_table,
        # Source surface
        "GNN_SRC_PACKAGE_COUNT": str(package_count),
        "GNN_SRC_PY_FILE_COUNT": str(py_files),
        "GNN_SRC_LOC": str(py_loc),
        "GNN_DOC_FILE_COUNT": str(doc_file_count),
        # MCP
        "GNN_MCP_TOOL_COUNT": str(mcp["tools"]),
        # Cross-repo manifest alias for the MCP audit ledger's tool count
        # (same source as GNN_MCP_TOOL_COUNT, which stays canonical here).
        "GNN_TOOL_COUNT": str(mcp["tools"]),
        "GNN_MCP_MODULE_COUNT": str(mcp["modules_total"]),
        "GNN_MCP_MODULE_LOADED": str(mcp["modules_loaded"]),
        "GNN_MCP_FILE_COUNT": str(mcp["files"]),
        # Tests
        "GNN_TEST_FILE_COUNT": str(test_file_count),
        "GNN_TEST_FUNCTION_COUNT": str(test_func_count),
        # Cross-repo manifest alias for the static test-function census
        # (same source as GNN_TEST_FUNCTION_COUNT). Pytest collection is
        # deliberately not used: it is slow and imports the world.
        "GNN_TEST_COUNT": str(test_func_count),
        # Model families / corpora
        "GNN_FAMILY_COUNT": str(len(families)),
        "GNN_FAMILY_LIST": ", ".join(family_names),
        "GNN_FAMILY_TABLE": family_table,
        "GNN_EXAMPLE_COUNT": str(len(example_models)),
        # Model-kind exemplar counts: the corpus partitioned the way the
        # render path classifies it — continuous linear-Gaussian specs under
        # input/gnn_files/continuous/, every other exemplar discrete-state.
        # The two always sum to GNN_EXAMPLE_COUNT.
        "GNN_CONTINUOUS_EXEMPLAR_COUNT": str(len(continuous_exemplars)),
        "GNN_DISCRETE_EXEMPLAR_COUNT": str(discrete_exemplars),
        # Directory census under input/gnn_files/ — NOT the same set as the
        # manifest families (some corpus dirs declare no family; some families
        # point outside input/gnn_files/). GNN_FAMILY_TARGET_DIR_COUNT is the
        # manifest-derived figure and is the one that pairs with
        # GNN_FAMILY_COUNT.
        "GNN_INPUT_FAMILY_DIR_COUNT": str(len(corpus_dirs)),
        "GNN_FAMILY_TARGET_DIR_COUNT": str(len(family_target_dirs)),
        "GNN_UNREGISTERED_CORPUS_DIR_COUNT": str(len(unregistered_corpus_dirs)),
        "GNN_UNREGISTERED_CORPUS_DIRS": ", ".join(
            f"`input/gnn_files/{name}`" for name in unregistered_corpus_dirs
        ),
        "GNN_UNSCANNED_CORPUS_DIR_COUNT": str(len(unscanned_corpus_dirs)),
        "GNN_UNSCANNED_CORPUS_DIRS": ", ".join(f"`{d}`" for d in unscanned_corpus_dirs),
        "GNN_UNSCANNED_CORPUS_NOTE": unscanned_corpus_note,
        # Model files under input/ but outside input/gnn_files. GNN_EXAMPLE_COUNT
        # deliberately excludes them, so without these two tokens nothing the
        # producer emits can see them at all.
        "GNN_OUTSIDE_CORPUS_MODEL_COUNT": str(outside_corpus_models),
        "GNN_OUTSIDE_CORPUS_NOTE": outside_corpus_note(
            outside_corpus_dirs, family_target_dirs, len(example_models)
        ),
        "GNN_TARGET_DIR_COVERAGE_NOTE": target_dir_coverage_note,
        # Backends
        "GNN_BACKEND_COUNT": str(len(backends)),
        "GNN_BACKEND_LIST": ", ".join(backend_names),
        "GNN_BACKEND_TABLE": backend_table,
        # Registry-flag capability matrix (discrete/continuous render and
        # Step-12 executor status per framework) and the model-kind table the
        # generalized manuscript is built around — both derived in
        # _render_framework_capability_table/_render_model_kind_table.
        "GNN_FRAMEWORK_TABLE": framework_table,
        "GNN_MODEL_KIND_TABLE": model_kind_table,
        "GNN_EXECUTABLE_BACKEND_COUNT": str(len(executable_backends)),
        "GNN_EXECUTABLE_BACKEND_LIST": ", ".join(executable_backends),
        # MAINTAINED_FRAMEWORKS in src/gnn/pipeline/cross_framework_reliability.py —
        # the set the reliability gate will profile. A strict subset of the
        # registry: GNN_BACKEND_COUNT is never the right number for a sentence
        # about what the gate does.
        "GNN_MAINTAINED_FRAMEWORK_COUNT": str(len(maintained)),
        "GNN_MAINTAINED_FRAMEWORK_LIST": ", ".join(maintained_names),
        "GNN_CROSS_FRAMEWORK_FAMILY": cross_family_name,
        # GNN_CROSS_FRAMEWORK_BACKENDS is the *profiled* set: the family's
        # declared frameworks intersected with MAINTAINED_FRAMEWORKS. The
        # declared-but-unprofiled remainder (stan today) is exposed separately so
        # a sentence about it cannot silently become a sentence about the gate.
        "GNN_CROSS_FRAMEWORK_BACKENDS": ", ".join(cross_profiled_names),
        "GNN_CROSS_FRAMEWORK_BACKEND_COUNT": str(len(cross_profiled_names)),
        "GNN_CROSS_FRAMEWORK_DECLARED_BACKENDS": ", ".join(cross_declared_names),
        "GNN_CROSS_FRAMEWORK_DECLARED_BACKEND_COUNT": str(len(cross_declared_names)),
        "GNN_CROSS_FRAMEWORK_UNPROFILED_BACKENDS": ", ".join(
            name for name in cross_declared_names if name not in cross_profiled_names
        ),
        # Generated artifacts. GNN_OUTPUT_ARTIFACT_FIGURE_COUNT is a census of
        # every PNG committed under output/ across all pipeline steps and many
        # runs; GNN_MANUSCRIPT_FIGURE_COUNT is only the manuscript's own figures
        # in output/figures/. GNN_OUTPUT_FIGURE_COUNT is the historical name of
        # the census and is retained for sections that still reference it.
        "GNN_OUTPUT_FIGURE_COUNT": str(figure_count),
        "GNN_OUTPUT_ARTIFACT_FIGURE_COUNT": str(figure_count),
        "GNN_MANUSCRIPT_FIGURE_COUNT": str(manuscript_figure_count),
    }
    variables.update(step_tokens)
    # Self-describing size of the map, so a section can state how many tokens the
    # producer emits without hand-typing it. Counted after every other token is
    # in place, and includes itself.
    variables["GNN_TOKEN_COUNT"] = str(len(variables) + 1)
    return variables


# Fields in manuscript/config.yaml the producer owns, and the token that owns
# each. config.yaml is NOT token-substituted (the injector only processes
# manuscript/*.md), so a literal typed here can only be kept true by writing it.
_CONFIG_OWNED_FIELDS = {
    "version": "GNN_VERSION",
    "date": "GNN_RELEASE_DATE",
}


def config_metadata_drift(
    project_root: Path, variables: Mapping[str, str]
) -> list[str]:
    """Return one message per ``config.yaml`` field that disagrees with a token.

    Empty list means the title page states the same version and date the
    repository does.
    """
    config_path = Path(project_root) / "manuscript" / "config.yaml"
    if not config_path.is_file():
        return []
    drift: list[str] = []
    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        for field, token in _CONFIG_OWNED_FIELDS.items():
            expected = variables.get(token, "")
            if not expected or not stripped.startswith(f"{field}:"):
                continue
            actual = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            if actual != expected:
                drift.append(
                    f"config.yaml: {field}: {actual!r} != {token} ({expected!r})"
                )
    return drift


def sync_config_metadata(project_root: Path, variables: Mapping[str, str]) -> list[str]:
    """Rewrite the producer-owned ``manuscript/config.yaml`` fields in place.

    ``version:`` and ``date:`` render onto the PDF title page but sit in a file
    the token injector never touches, so they can only stay true by being
    written from the same source the tokens come from: ``pyproject.toml`` for
    the version and the matching ``CHANGELOG.md`` heading for the date. The
    alternative — typing the correct value once — is what left a 3.2.0 title
    page carrying 3.0.0's release date.

    Returns the list of ``"field: old -> new"`` changes applied (empty when the
    file was already in sync).
    """
    config_path = Path(project_root) / "manuscript" / "config.yaml"
    if not config_path.is_file():
        return []
    lines = config_path.read_text(encoding="utf-8").splitlines(keepends=True)
    changes: list[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        for field, token in _CONFIG_OWNED_FIELDS.items():
            expected = variables.get(token, "")
            if not expected or not stripped.startswith(f"{field}:"):
                continue
            actual = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            if actual == expected:
                continue
            indent = line[: len(line) - len(line.lstrip())]
            newline = "\n" if line.endswith("\n") else ""
            lines[index] = f'{indent}{field}: "{expected}"{newline}'
            changes.append(f"{field}: {actual} -> {expected}")
    if changes:
        config_path.write_text("".join(lines), encoding="utf-8")
    return changes


# Fields in manuscript/preamble.md's LaTeX block the producer owns, and the token
# that owns each. preamble.md IS token-substituted into output/manuscript/, but
# the template's _manuscript_source.py then copies the RAW file over that copy
# (`_shutil.copy2(preamble_src, preamble_dst)`), so a {{TOKEN}} written here
# reaches LaTeX unresolved and hyperref writes the token name into the PDF's
# metadata. Same remedy as config.yaml: the producer writes the value.
_PREAMBLE_OWNED_FIELDS = {
    "pdfsubject": "GNN_SUBTITLE",
    "pdfkeywords": "GNN_KEYWORDS",
}

_PREAMBLE_FIELD_RE = re.compile(
    r"^(?P<indent>\s*)(?P<field>pdf\w+)=\{(?P<value>.*?)\}(?P<tail>[,}]*)\s*$"
)


def _preamble_fields(project_root: Path) -> list[tuple[int, str, str, str]]:
    """Return ``(line_index, field, current_value, whole_line)`` for owned fields."""
    path = Path(project_root) / "manuscript" / "preamble.md"
    if not path.is_file():
        return []
    found = []
    for index, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(keepends=True)
    ):
        match = _PREAMBLE_FIELD_RE.match(line.rstrip("\n"))
        if match and match.group("field") in _PREAMBLE_OWNED_FIELDS:
            found.append((index, match.group("field"), match.group("value"), line))
    return found


def preamble_metadata_drift(
    project_root: Path, variables: Mapping[str, str]
) -> list[str]:
    """Return one message per ``preamble.md`` PDF-metadata field out of sync."""
    drift: list[str] = []
    for _index, field, value, _line in _preamble_fields(project_root):
        expected = variables.get(_PREAMBLE_OWNED_FIELDS[field], "")
        if expected and value != expected:
            token = _PREAMBLE_OWNED_FIELDS[field]
            drift.append(f"preamble.md: {field}: {value!r} != {token} ({expected!r})")
    return drift


def sync_preamble_metadata(
    project_root: Path, variables: Mapping[str, str]
) -> list[str]:
    """Rewrite the producer-owned ``preamble.md`` PDF-metadata values in place.

    Returns the list of ``"field: old -> new"`` changes applied.
    """
    path = Path(project_root) / "manuscript" / "preamble.md"
    fields = _preamble_fields(project_root)
    if not fields:
        return []
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    changes: list[str] = []
    for index, field, value, line in fields:
        expected = variables.get(_PREAMBLE_OWNED_FIELDS[field], "")
        if not expected or value == expected:
            continue
        match = _PREAMBLE_FIELD_RE.match(line.rstrip("\n"))
        assert match is not None
        newline = "\n" if line.endswith("\n") else ""
        lines[index] = (
            f"{match.group('indent')}{field}={{{expected}}}{match.group('tail')}{newline}"
        )
        changes.append(f"{field}: {value} -> {expected}")
    if changes:
        path.write_text("".join(lines), encoding="utf-8")
    return changes


def save_variables(variables: dict[str, str], out_path: Path) -> Path:
    """Write *variables* to *out_path* as sorted, deterministic JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(variables, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out_path


def load_variables(in_path: Path) -> dict[str, str]:
    """Read back a token map written by :func:`save_variables`.

    Args:
        in_path: Path to a ``manuscript_variables.json`` file.

    Returns:
        The validated flat ``{TOKEN: value}`` mapping.

    Raises:
        ValueError: If the file is not a flat JSON object of string values.
    """
    in_path = Path(in_path)
    data = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in data.items()
    ):
        raise ValueError(
            f"{in_path} is not a flat {{TOKEN: value}} JSON map written by save_variables"
        )
    return data


def token_checksum(variables: Mapping[str, str]) -> str:
    """Return the sha256 checksum of a token map's canonical JSON form.

    The canonical form matches :func:`save_variables` output (sorted keys,
    2-space indent, UTF-8) without the trailing newline, so a checksum over
    :func:`load_variables` output equals one over freshly generated variables
    for an unchanged tree.
    """
    canonical = json.dumps(variables, indent=2, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

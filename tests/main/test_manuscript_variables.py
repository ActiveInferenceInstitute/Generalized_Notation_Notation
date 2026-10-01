"""Tests for the deterministic manuscript-variable producer.

Real objects only: every assertion recomputes the expected value from the live repository
and compares it against :func:`gnn.manuscript.variables.generate_variables`, so the
test fails if the producer drifts from the source surfaces it claims to read.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

# Import the producer by its canonical package name; the venv's editable
# install exposes ``gnn``, so no sys.path bootstrap is needed and mypy
# (mypy_path=src) resolves a single module name.
from gnn.manuscript.variables import (
    RepositorySnapshot,
    _capability_clause,
    _registry_specs,
    _render_family_table,
    config_metadata_drift,
    generate_variables,
    preamble_metadata_drift,
    save_variables,
    select_cross_framework_family,
    sync_config_metadata,
    sync_preamble_metadata,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def variables() -> dict[str, str]:
    return generate_variables(_PROJECT_ROOT)


def _registry_flags(snapshot: RepositorySnapshot, flag: str) -> list[str]:
    """Independently parse ``FRAMEWORK_REGISTRY`` for backends whose *flag* is True.

    Parsed here with :mod:`ast` rather than by calling the producer's own
    parser, so an assertion failure means the token really disagrees with the
    registry literal — not that both read the same helper.
    """
    tree = ast.parse(snapshot.read_text("src/gnn/render/framework_registry.py"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = (
            [node.target] if isinstance(node, ast.AnnAssign) else list(node.targets)
        )
        if not any(
            isinstance(target, ast.Name) and target.id == "FRAMEWORK_REGISTRY"
            for target in targets
        ):
            continue
        value = node.value
        assert value is not None, "FRAMEWORK_REGISTRY has no assigned value"
        if isinstance(value, ast.Call) and value.args:
            value = value.args[0]
        registry = ast.literal_eval(value)
        return [key for key, spec in registry.items() if spec.get(flag) is True]
    raise AssertionError("FRAMEWORK_REGISTRY literal not found")


@pytest.fixture(scope="module")
def snapshot() -> RepositorySnapshot:
    """The same commit the producer counts.

    Assertions read source surfaces through this snapshot rather than off disk:
    the producer's contract is that every count describes ``GNN_GIT_COMMIT``, so
    comparing against the working tree would make these tests fail for anyone
    with an uncommitted edit instead of catching real drift.
    """
    return RepositorySnapshot(_PROJECT_ROOT)


def test_all_values_are_strings(variables: dict[str, str]) -> None:
    assert variables, "producer returned an empty token map"
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in variables.items())


def test_token_keys_are_injector_compatible(variables: dict[str, str]) -> None:
    # Template injector regex: {{[A-Z][A-Z0-9_]*}}
    key_re = re.compile(r"^[A-Z][A-Z0-9_]*$")
    bad = [k for k in variables if not key_re.match(k)]
    assert not bad, f"keys not matchable by injector regex: {bad}"


def test_minimum_token_coverage(variables: dict[str, str]) -> None:
    # ISC-3: at least 20 distinct tokens.
    assert len(variables) >= 20, f"only {len(variables)} tokens"


def test_determinism(variables: dict[str, str]) -> None:
    # ISC-4: a second invocation yields an identical map.
    again = generate_variables(_PROJECT_ROOT)
    assert again == variables


def test_version_matches_pyproject(
    variables: dict[str, str], snapshot: RepositorySnapshot
) -> None:
    match = re.search(
        r'^\s*version\s*=\s*"([^"]+)"',
        snapshot.read_text("pyproject.toml"),
        re.MULTILINE,
    )
    assert match is not None
    assert variables["GNN_VERSION"] == match.group(1)


def test_step_count_matches_step_modules(
    variables: dict[str, str], snapshot: RepositorySnapshot
) -> None:
    step_modules = [
        rel for rel in snapshot.glob("src/gnn", "[0-9]*_*.py") if len(rel.parts) == 3
    ]
    assert variables["GNN_STEP_COUNT"] == str(len(step_modules))
    assert int(variables["GNN_STEP_COUNT"]) >= 1


def test_step_modules_and_packages_are_disjoint_sets(
    variables: dict[str, str], snapshot: RepositorySnapshot
) -> None:
    """GNN_STEP_COUNT and GNN_SRC_PACKAGE_COUNT count sets with no overlap.

    Step modules are top-level ``src/gnn/N_*.py`` files; packages are ``src/gnn/<name>/``
    directories. No prose may say the step modules "sit inside" the packages.
    """
    step_modules = [
        rel for rel in snapshot.glob("src/gnn", "[0-9]*_*.py") if len(rel.parts) == 3
    ]
    packages = {
        rel.parts[2]
        for rel in snapshot.glob("src/gnn", "__init__.py")
        if len(rel.parts) == 4
    }
    assert step_modules, "no step modules found"
    assert packages, "no source packages found"
    assert not {rel.name for rel in step_modules} & packages
    assert variables["GNN_SRC_PACKAGE_COUNT"] == str(len(packages))


def test_family_count_matches_manifest(
    variables: dict[str, str], snapshot: RepositorySnapshot
) -> None:
    manifest = json.loads(snapshot.read_text("input/model_family_manifest.json"))
    assert variables["GNN_FAMILY_COUNT"] == str(len(manifest["families"]))


def test_backend_count_matches_registry(
    variables: dict[str, str], snapshot: RepositorySnapshot
) -> None:
    registry = snapshot.read_text("src/gnn/render/framework_registry.py")
    names = re.findall(r'"name"\s*:\s*"([^"]+)"', registry)
    assert variables["GNN_BACKEND_COUNT"] == str(len(names))
    assert int(variables["GNN_BACKEND_COUNT"]) >= 2


def test_executable_backend_count_is_the_execution_subset(
    variables: dict[str, str], snapshot: RepositorySnapshot
) -> None:
    """Backends with ``supports_execution`` are exactly the execution subset.

    ``GNN_BACKEND_COUNT`` is the registry size and must never label a sentence
    or a figure bar about execution. Since bnlearn gained
    ``src/gnn/execute/bnlearn/``, every render-registry backend executes (Lean
    is an execute-side bridge outside the registry), so the subset equals the
    registry — the count must still derive from the registry flags, not a
    constant.
    """
    executable = len(_registry_flags(snapshot, "supports_execution"))
    assert variables["GNN_EXECUTABLE_BACKEND_COUNT"] == str(executable)
    assert int(variables["GNN_EXECUTABLE_BACKEND_COUNT"]) == int(
        variables["GNN_BACKEND_COUNT"]
    )
    assert len(variables["GNN_EXECUTABLE_BACKEND_LIST"].split(", ")) == executable


def test_cross_framework_backends_are_all_profiled_by_the_gate(
    variables: dict[str, str], snapshot: RepositorySnapshot
) -> None:
    """The reported cross-framework engines are ones the gate will actually run.

    ``src/gnn/pipeline/cross_framework_reliability.py`` raises on any framework
    outside ``MAINTAINED_FRAMEWORKS``, so a declared-but-unprofiled framework
    (``stan``) must not be counted among the reference-comparison engines.
    """
    gate = snapshot.read_text("src/gnn/pipeline/cross_framework_reliability.py")
    block = re.search(r"MAINTAINED_FRAMEWORKS = \(([^)]*)\)", gate)
    assert block is not None
    maintained_keys = re.findall(r'"([a-z_]+)"', block.group(1))
    assert variables["GNN_MAINTAINED_FRAMEWORK_COUNT"] == str(len(maintained_keys))

    manifest = json.loads(snapshot.read_text("input/model_family_manifest.json"))
    family = select_cross_framework_family(manifest["families"])
    assert family is not None
    declared = [k.strip() for k in str(family["frameworks"]).split(",") if k.strip()]
    profiled = [k for k in declared if k in maintained_keys]
    assert profiled, "cross-framework family declares nothing the gate profiles"
    assert variables["GNN_CROSS_FRAMEWORK_BACKEND_COUNT"] == str(len(profiled))
    assert variables["GNN_CROSS_FRAMEWORK_DECLARED_BACKEND_COUNT"] == str(len(declared))
    assert len(variables["GNN_CROSS_FRAMEWORK_BACKENDS"].split(", ")) == len(profiled)


def test_manifest_does_not_hand_type_capability_claims() -> None:
    """No manifest description may restate a backend capability in prose.

    The ``discopy``-is-native-on-continuous defect was a free-text capability
    sentence in ``input/model_family_manifest.json`` that contradicted the
    registry's ``supports_continuous`` flag, the acceptance ledger, and
    CLAUDE.md. A family declares which axis it exercises; the split itself is
    generated. Read from the working tree because this is a lint on the
    authored file, not a count.
    """
    manifest = json.loads(
        (_PROJECT_ROOT / "input" / "model_family_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for family in manifest["families"]:
        desc = str(family.get("description", ""))
        assert "Native on" not in desc, (
            f"{family['name']}: capability claim in manifest"
        )
        assert "unsupported" not in desc, (
            f"{family['name']}: capability claim in manifest"
        )
    axes = [f for f in manifest["families"] if f.get("capability_axis")]
    assert axes, "no family declares a capability_axis; the split would vanish"


def test_capability_clause_is_generated_from_the_registry(
    snapshot: RepositorySnapshot,
) -> None:
    """The generated native/unsupported split equals the registry flags."""
    specs = _registry_specs(snapshot)
    native_keys = _registry_flags(snapshot, "supports_continuous")
    clause = _capability_clause("supports_continuous", specs)
    native_part, unsupported_part = clause.split("; ", 1)
    native_names = native_part.removeprefix("Native on ").split(", ")
    unsupported_names = (
        unsupported_part.removeprefix("reported unsupported (not failed) on ")
        .rstrip(".")
        .split(", ")
    )
    assert native_names == [str(specs[k]["name"]) for k in native_keys]
    assert sorted(native_names + unsupported_names) == sorted(
        str(spec["name"]) for spec in specs.values()
    )
    # The registry entry the defect misreported must land on the correct side.
    assert str(specs["discopy"]["name"]) in unsupported_names


def test_family_table_renders_the_split_for_the_declaring_family(
    snapshot: RepositorySnapshot,
) -> None:
    """The rendered family table carries the generated split, not manifest prose.

    Built from the authored manifest and the registry directly, so this holds
    for an edit in progress as well as for the counted commit.
    """
    manifest = json.loads(
        (_PROJECT_ROOT / "input" / "model_family_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    families = manifest["families"]
    declaring = [f for f in families if f.get("capability_axis")]
    assert declaring, "no family declares a capability_axis"
    table = _render_family_table(families, _registry_specs(snapshot))
    for family in declaring:
        row = next(
            line
            for line in table.splitlines()
            if line.startswith(f"| `{family['name']}`")
        )
        assert "Native on" in row
        assert "reported unsupported (not failed) on" in row
    assert "generated from" in table.splitlines()[-1]


def test_mcp_tool_count_matches_audit(
    variables: dict[str, str], snapshot: RepositorySnapshot
) -> None:
    audit = json.loads(snapshot.read_text("src/gnn/mcp/audit_report.json"))
    assert variables["GNN_MCP_TOOL_COUNT"] == str(audit["tools_total"])


def test_module_count_matches_step_modules(
    variables: dict[str, str], snapshot: RepositorySnapshot
) -> None:
    step_modules = [
        rel for rel in snapshot.glob("src/gnn", "[0-9]*_*.py") if len(rel.parts) == 3
    ]
    assert variables["GNN_MODULE_COUNT"] == str(len(step_modules))


def test_tool_count_matches_audit(
    variables: dict[str, str], snapshot: RepositorySnapshot
) -> None:
    audit = json.loads(snapshot.read_text("src/gnn/mcp/audit_report.json"))
    assert variables["GNN_TOOL_COUNT"] == str(audit["tools_total"])


def test_test_count_matches_static_census(
    variables: dict[str, str], snapshot: RepositorySnapshot
) -> None:
    pattern = re.compile(r"^\s*(?:async\s+)?def (test_\w+)", re.MULTILINE)
    test_files = snapshot.glob("tests", "test_*.py")
    snapshot.prefetch(test_files)
    census = sum(len(pattern.findall(snapshot.read_text(rel))) for rel in test_files)
    assert variables["GNN_TEST_COUNT"] == str(census)
    assert int(variables["GNN_TEST_COUNT"]) >= int(variables["GNN_TEST_FILE_COUNT"])


def test_exemplar_counts_partition_the_corpus(
    variables: dict[str, str], snapshot: RepositorySnapshot
) -> None:
    """The kind counts split GNN_EXAMPLE_COUNT on the continuous/ folder."""
    non_model = {"INDEX.md", "AGENTS.md", "README.md"}
    continuous = [
        rel
        for rel in snapshot.glob("input/gnn_files/continuous", "*.md")
        if rel.name not in non_model
        and not rel.name.endswith((".example.md", ".template.md"))
    ]
    assert variables["GNN_CONTINUOUS_EXEMPLAR_COUNT"] == str(len(continuous))
    assert int(variables["GNN_DISCRETE_EXEMPLAR_COUNT"]) + int(
        variables["GNN_CONTINUOUS_EXEMPLAR_COUNT"]
    ) == int(variables["GNN_EXAMPLE_COUNT"])


def test_framework_table_cells_track_registry_flags(
    variables: dict[str, str], snapshot: RepositorySnapshot
) -> None:
    rows = [
        ln
        for ln in variables["GNN_FRAMEWORK_TABLE"].splitlines()
        if ln.startswith("| ")
    ]
    assert rows[0] == (
        "| Framework | Discrete render | Continuous render | Executor status |"
    )
    cells = [[c.strip() for c in ln.split("|")[1:-1]] for ln in rows[1:]]
    specs = _registry_specs(snapshot)
    assert [c[0] for c in cells] == [
        str(spec.get("name", key)) for key, spec in specs.items()
    ]
    by_name = {c[0]: c[1:] for c in cells}
    for key, spec in specs.items():
        name = str(spec.get("name", key))
        discrete, continuous, executor = by_name[name]
        routed = spec.get("pomdp_compatible")
        renders_discrete = routed and not spec.get("continuous_only", False)
        assert discrete == (
            "yes" if renders_discrete else "unsupported" if routed else "no"
        )
        assert continuous == (
            "yes" if spec.get("supports_continuous") else "unsupported"
        )
        assert executor == (
            "executor" if spec.get("supports_execution") else "render-only"
        )
    assert (
        variables["GNN_FRAMEWORK_TABLE"]
        .rstrip()
        .endswith("{#tbl:framework_capability}")
    )


def test_model_kind_table_rows(
    variables: dict[str, str], snapshot: RepositorySnapshot
) -> None:
    rows = [
        ln
        for ln in variables["GNN_MODEL_KIND_TABLE"].splitlines()
        if ln.startswith("| ")
    ]
    assert rows[0] == (
        "| Model kind | Notation block | Exemplar folder | Renderer(s) | Executor(s) |"
    )
    cells = [[c.strip() for c in ln.split("|")[1:-1]] for ln in rows[1:]]
    by_kind = {c[0]: c[1:] for c in cells}
    assert list(by_kind) == [
        "Discrete categorical",
        "Continuous linear-Gaussian",
        "Multi-agent",
        "Recursive",
    ]
    assert by_kind["Multi-agent"][2:] == by_kind["Discrete categorical"][2:]
    assert by_kind["Discrete categorical"][1] == "`input/gnn_files/discrete/`"
    assert by_kind["Continuous linear-Gaussian"][1] == "`input/gnn_files/continuous/`"
    specs = _registry_specs(snapshot)
    continuous_names = ", ".join(
        str(spec.get("name", key))
        for key, spec in specs.items()
        if spec.get("supports_continuous")
    )
    assert by_kind["Continuous linear-Gaussian"][2] == continuous_names
    # A continuous-only backend is routed but never listed as rendering discrete.
    continuous_only = [
        str(spec.get("name", key))
        for key, spec in specs.items()
        if spec.get("continuous_only", False)
    ]
    for name in continuous_only:
        assert name not in by_kind["Discrete categorical"][2]
    assert by_kind["Continuous linear-Gaussian"][3] == continuous_names
    assert by_kind["Multi-agent"][1] == "`input/gnn_files/multiagent/`"
    assert by_kind["Recursive"][1] == "`input/gnn_files/recursive/`"
    assert variables["GNN_MODEL_KIND_TABLE"].rstrip().endswith("{#tbl:model_kinds}")


def test_counts_describe_the_stamped_commit_not_the_working_tree(
    variables: dict[str, str], snapshot: RepositorySnapshot
) -> None:
    """A dirty working tree cannot move a published count.

    The producer's printed contract is that the numbers describe the commit
    ``GNN_GIT_COMMIT`` names. Before this was enforced the PDF shipped 190691
    LOC while the repository's own committed token file said 190943 and neither
    reproduced from the commit the token file stamped.
    """
    # No skip guard: the repo's zero-skip contract (test_zero_skip_contracts.py)
    # forbids one, and a checkout without git metadata cannot honour the
    # producer's reproducibility contract at all — that is a failure, not a skip.
    assert snapshot.from_git, "snapshot is not reading committed blobs"
    assert variables["GNN_GIT_COMMIT"] == snapshot.commit
    assert variables["GNN_GIT_COMMIT"] != "unknown"
    sources = list(snapshot.glob("src", "*.py"))
    snapshot.prefetch(sources)
    total = 0
    for rel in sources:
        text = snapshot.read_text(rel)
        total += text.count("\n") + (0 if not text or text.endswith("\n") else 1)
    assert variables["GNN_SRC_LOC"] == str(total)
    assert variables["GNN_SRC_PY_FILE_COUNT"] == str(len(sources))


def test_tables_are_multiline_markdown(variables: dict[str, str]) -> None:
    for key in (
        "GNN_STEP_TABLE",
        "GNN_FAMILY_TABLE",
        "GNN_BACKEND_TABLE",
        "GNN_FRAMEWORK_TABLE",
        "GNN_MODEL_KIND_TABLE",
    ):
        assert "\n" in variables[key], f"{key} should be a multi-line markdown table"
        assert variables[key].lstrip().startswith("|"), f"{key} should be a pipe table"


def test_license_value_is_resolved(variables: dict[str, str]) -> None:
    # ISC-23 guard: config completion removes the scaffold 'TBD'/empty license.
    assert variables["GNN_LICENSE"] and variables["GNN_LICENSE"] != "TBD"


def test_save_variables_roundtrip(tmp_path: Path, variables: dict[str, str]) -> None:
    out = save_variables(variables, tmp_path / "vars.json")
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert reloaded == variables


def test_preamble_pdf_metadata_is_written_not_typed(variables: dict[str, str]) -> None:
    """preamble.md's PDF Subject/Keywords must equal the tokens that own them.

    preamble.md cannot carry a double-brace token for these: the template
    substitutes the file into ``output/manuscript/`` and then copies the raw
    source back over it (``infrastructure/rendering/_manuscript_source.py``), so
    an unresolved token reaches hyperref and is written into the PDF metadata
    verbatim -- which is what a first attempt produced (``Subject: GNNSUBTITLE``).
    """
    assert preamble_metadata_drift(_PROJECT_ROOT, variables) == []

    preamble = (_PROJECT_ROOT / "manuscript" / "preamble.md").read_text(
        encoding="utf-8"
    )
    assert "{{" not in preamble, "preamble tokens are never substituted in the render"
    assert f"pdfsubject={{{variables['GNN_SUBTITLE']}}}" in preamble
    assert f"pdfkeywords={{{variables['GNN_KEYWORDS']}}}" in preamble


def test_sync_preamble_metadata_repairs_drift(
    tmp_path: Path, variables: dict[str, str]
) -> None:
    """A hand-edited value is rewritten from the token, and drift reports it."""
    manuscript = tmp_path / "manuscript"
    manuscript.mkdir()
    source = (_PROJECT_ROOT / "manuscript" / "preamble.md").read_text(encoding="utf-8")
    stale = source.replace(
        f"pdfsubject={{{variables['GNN_SUBTITLE']}}}", "pdfsubject={stale subtitle}"
    )
    assert stale != source
    (manuscript / "preamble.md").write_text(stale, encoding="utf-8")

    assert preamble_metadata_drift(tmp_path, variables) != []
    changes = sync_preamble_metadata(tmp_path, variables)
    assert any(c.startswith("pdfsubject:") for c in changes)
    assert preamble_metadata_drift(tmp_path, variables) == []
    assert sync_preamble_metadata(tmp_path, variables) == []


def test_config_title_page_metadata_is_written_not_typed(
    variables: dict[str, str],
) -> None:
    """config.yaml's title-page version/date must equal the tokens that own them.

    config.yaml is never token-substituted (the injector only processes
    ``manuscript/*.md``), so the producer must write these fields or the PDF
    title page silently keeps a stale version/date.
    """
    assert config_metadata_drift(_PROJECT_ROOT, variables) == []

    config = (_PROJECT_ROOT / "manuscript" / "config.yaml").read_text(encoding="utf-8")
    assert f'version: "{variables["GNN_VERSION"]}"' in config
    assert f'date: "{variables["GNN_RELEASE_DATE"]}"' in config


def test_sync_config_metadata_repairs_drift(
    tmp_path: Path, variables: dict[str, str]
) -> None:
    """A hand-edited title-page value is rewritten from the token; drift reports it."""
    manuscript = tmp_path / "manuscript"
    manuscript.mkdir()
    source = (_PROJECT_ROOT / "manuscript" / "config.yaml").read_text(encoding="utf-8")
    stale = source.replace(
        f'version: "{variables["GNN_VERSION"]}"', 'version: "stale version"'
    )
    assert stale != source
    (manuscript / "config.yaml").write_text(stale, encoding="utf-8")

    assert config_metadata_drift(tmp_path, variables) != []
    changes = sync_config_metadata(tmp_path, variables)
    assert any(c.startswith("version:") for c in changes)
    assert config_metadata_drift(tmp_path, variables) == []
    assert sync_config_metadata(tmp_path, variables) == []
    assert (manuscript / "config.yaml").read_text(encoding="utf-8") == source

"""Package dependency-direction and import-surface contracts.

Two complementary evidence classes:

* **Static (AST) edges.** Every ``import``/``from ... import`` statement in a
  source package — module scope, function scope and ``TYPE_CHECKING`` blocks
  alike — plus literal ``importlib.import_module("...")``/``__import__("...")``
  targets, resolved to absolute module names (relative imports included).
  This is stricter than import-linter (which skips ``TYPE_CHECKING`` imports
  here), so a contract cannot be dodged by moving an import into a function
  or a string.
* **Fresh-process ``sys.modules``.** Each public entry point is imported in a
  new ``sys.executable -c`` interpreter, because the pytest process has
  already imported most of the tree; only a clean interpreter shows what an
  import actually loads.

Directions pinned here:

* transports: ``gnn.api`` never imports ``gnn.cli`` (both import downward from
  ``gnn.operations``), and ``gnn.operations`` imports neither transport;
* producers: ``gnn.pipeline`` never imports ``gnn.report``, ``gnn.cli`` or
  ``gnn.api`` (renderers live with their pipeline producers; the
  ``gnn.report`` modules are re-export facades);
* execute boundary: ``gnn.pipeline``, ``gnn.setup`` and ``gnn.security``
  never import ``gnn.execute`` (cancellation and the julia probe come from
  standard-library-only leaves that execute re-exports);
* transport boundary for steps: ``gnn.execute``, ``gnn.setup`` and ``gnn.llm``
  never import ``gnn.api`` (the workspace path policy lives in a leaf that
  the API re-exports);
* reciprocal pairs: the set of top-level packages that import each other is
  frozen, so a new two-way dependency fails and a removed one must be
  deleted from the list.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
GNN_ROOT = SRC_ROOT / "gnn"


@dataclass(frozen=True)
class ImportEdge:
    """One static import edge found in a source file."""

    source: str
    target: str
    path: Path
    line: int

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.source} -> {self.target}"


def _module_name(path: Path, src_root: Path) -> str:
    parts = list(path.relative_to(src_root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _package_of(path: Path, module: str) -> str:
    return module if path.name == "__init__.py" else module.rpartition(".")[0]


def _resolve_from(node: ast.ImportFrom, path: Path, module: str) -> list[str]:
    if node.level == 0:
        base = node.module or ""
    else:
        package = _package_of(path, module).split(".")
        anchor = package[: len(package) - (node.level - 1)]
        base = ".".join([*anchor, node.module] if node.module else anchor)
    # ``from pkg import sub`` may name a submodule; report both candidates so
    # a forbidden prefix match on either is caught.
    return [base, *(f"{base}.{alias.name}" for alias in node.names)]


_DYNAMIC_IMPORTERS = {"import_module", "__import__"}


def _dynamic_target(node: ast.Call) -> str | None:
    func = node.func
    name = (
        func.attr
        if isinstance(func, ast.Attribute)
        else func.id
        if isinstance(func, ast.Name)
        else None
    )
    if name not in _DYNAMIC_IMPORTERS or not node.args:
        return None
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def iter_import_edges(
    package_dir: Path, src_root: Path = SRC_ROOT
) -> Iterator[ImportEdge]:
    """Yield every static import edge in ``package_dir`` (recursively)."""
    for path in sorted(package_dir.rglob("*.py")):
        module = _module_name(path, src_root)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                targets = _resolve_from(node, path, module)
            elif isinstance(node, ast.Call):
                dynamic = _dynamic_target(node)
                targets = [dynamic] if dynamic else []
            for target in targets:
                yield ImportEdge(module, target, path, getattr(node, "lineno", 0))


def _is_within(target: str, package: str) -> bool:
    return target == package or target.startswith(f"{package}.")


def forbidden_edges(source_package: str, forbidden: tuple[str, ...]) -> list[str]:
    """Return every edge from ``source_package`` into a forbidden package."""
    package_dir = SRC_ROOT.joinpath(*source_package.split("."))
    assert package_dir.is_dir(), f"missing source package {package_dir}"
    return sorted(
        {
            str(edge)
            for edge in iter_import_edges(package_dir)
            if any(_is_within(edge.target, package) for package in forbidden)
        }
    )


# ── AST scanner self-checks (a vacuous scanner must not pass the contracts) ──


def test_scanner_resolves_absolute_relative_and_dynamic_imports(
    tmp_path: Path,
) -> None:
    fake_src = tmp_path / "src"
    package = fake_src / "gnn" / "pipeline"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("from ..report import facade\n")
    (package / "leaf.py").write_text(
        "import importlib\n"
        "def late():\n"
        "    from gnn.report.pipeline_report import generate_pipeline_report\n"
        "    importlib.import_module('gnn.cli.commands')\n"
        "    from . import sibling\n"
    )
    edges = {
        (edge.source, edge.target)
        for edge in iter_import_edges(package, src_root=fake_src)
    }
    assert ("gnn.pipeline", "gnn.report") in edges
    assert ("gnn.pipeline", "gnn.report.facade") in edges
    assert ("gnn.pipeline.leaf", "gnn.report.pipeline_report") in edges
    assert ("gnn.pipeline.leaf", "gnn.cli.commands") in edges
    assert ("gnn.pipeline.leaf", "gnn.pipeline.sibling") in edges


def test_scanner_sees_known_real_edges() -> None:
    """The report facades import their pipeline producers (downward edge)."""
    edges = {
        (edge.source, edge.target) for edge in iter_import_edges(GNN_ROOT / "report")
    }
    assert (
        "gnn.report.pipeline_report",
        "gnn.pipeline.summary_wiring",
    ) in edges
    assert (
        "gnn.report.semantic_fidelity",
        "gnn.pipeline.semantic_fidelity",
    ) in edges


# ── Static direction contracts ───────────────────────────────────────────────


def test_api_never_imports_cli() -> None:
    """HTTP adapters share operations through ``gnn.operations``, not the CLI."""
    assert forbidden_edges("gnn.api", ("gnn.cli",)) == []


def test_operations_never_import_transport_adapters() -> None:
    """``gnn.operations`` sits below every transport adapter."""
    assert forbidden_edges("gnn.operations", ("gnn.cli", "gnn.api")) == []


def test_pipeline_never_imports_report_cli_or_api() -> None:
    """Pipeline producers own their renderers; reports read pipeline output."""
    assert forbidden_edges("gnn.pipeline", ("gnn.report", "gnn.cli", "gnn.api")) == []


def test_template_mcp_uses_shared_operations_not_cli() -> None:
    """MCP template tools are a third transport over the same operations."""
    assert forbidden_edges("gnn.template", ("gnn.cli", "gnn.api")) == []


@pytest.mark.parametrize("package", ["gnn.pipeline", "gnn.setup", "gnn.security"])
def test_packages_below_execute_never_import_it(package: str) -> None:
    """Shared run vocabulary comes from leaves, not from ``gnn.execute``."""
    assert forbidden_edges(package, ("gnn.execute",)) == []


@pytest.mark.parametrize("package", ["gnn.execute", "gnn.setup", "gnn.llm"])
def test_step_packages_never_import_the_api_transport(package: str) -> None:
    """Step MCP tools take the workspace path policy from its leaf."""
    assert forbidden_edges(package, ("gnn.api",)) == []


def test_shared_run_leaves_import_nothing_from_gnn() -> None:
    """The leaves stay leaves: no ``gnn`` import can re-create the cycle."""
    for leaf in ("cancellation", "julia_probe"):
        path = GNN_ROOT / "utils" / "runtime_safety" / f"{leaf}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = ["gnn"] if node.level else [node.module or ""]
            else:
                continue
            assert all(name.split(".")[0] != "gnn" for name in names), (
                f"{path}:{node.lineno} imports {names}"
            )


def test_workspace_path_policy_stays_inside_runtime_safety() -> None:
    """The path policy may use its own concern package and nothing above it."""
    path = GNN_ROOT / "utils" / "runtime_safety" / "workspace_paths.py"
    edges = {
        edge.target
        for edge in iter_import_edges(path.parent)
        if edge.path == path and edge.target.split(".")[0] == "gnn"
    }
    assert edges
    assert all(_is_within(t, "gnn.utils.runtime_safety") for t in edges), edges


#: Top-level ``gnn`` packages that import each other today, in either scope.
#: The list may only shrink: cut a pair, then delete its row.
KNOWN_RECIPROCAL_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        ("analysis", "execute"),
        ("analysis", "pipeline"),
        ("execute", "utils"),
        ("extract", "render"),
        ("main", "pipeline"),
        ("parsers", "schema_validator"),
        ("parsers", "types"),
        ("parsers", "utils"),
        ("pipeline", "processing"),
        ("pipeline", "utils"),
        ("processing", "utils"),
        ("schema_validator", "testing"),
    }
)


def _top_level_group(module: str) -> str | None:
    parts = module.split(".")
    if parts[0] != "gnn" or len(parts) < 2:
        return None
    return parts[1]


def reciprocal_package_pairs() -> set[tuple[str, str]]:
    """Return every unordered pair of top-level groups with edges both ways."""
    directed: set[tuple[str, str]] = set()
    for edge in iter_import_edges(GNN_ROOT):
        source = _top_level_group(edge.source)
        target = _top_level_group(edge.target)
        if source and target and source != target:
            directed.add((source, target))
    return {
        (source, target)
        for source, target in directed
        if source < target and (target, source) in directed
    }


def test_reciprocal_package_pairs_only_shrink() -> None:
    """No new two-way package dependency, and no stale row for a cut one."""
    actual = reciprocal_package_pairs()
    assert sorted(actual - KNOWN_RECIPROCAL_PAIRS) == [], "new reciprocal pair"
    assert sorted(KNOWN_RECIPROCAL_PAIRS - actual) == [], "delete the cut pair's row"


# ── Compatibility facades keep one implementation ────────────────────────────


def test_cli_facades_reexport_the_shared_operations() -> None:
    import gnn.cli.commands as cli_commands
    import gnn.cli.templates as cli_templates
    import gnn.operations.commands as op_commands
    import gnn.operations.templates as op_templates

    for name in cli_commands.__all__:
        assert getattr(cli_commands, name) is getattr(op_commands, name), name
    for name in cli_templates.__all__:
        assert getattr(cli_templates, name) is getattr(op_templates, name), name
    assert cli_templates._template_record_from_raw is (
        op_templates._template_record_from_raw
    )


def test_operations_facade_resolves_lazily() -> None:
    import gnn.operations as operations
    import gnn.operations.commands as op_commands
    import gnn.operations.templates as op_templates

    assert operations.run_validation_checks is op_commands.run_validation_checks
    assert operations.pull_template is op_templates.pull_template
    assert set(operations.__all__) <= set(dir(operations))
    with pytest.raises(AttributeError):
        _ = operations.no_such_operation


def test_report_facades_reexport_pipeline_renderers() -> None:
    import gnn.pipeline.cross_framework_reliability as pipeline_cfr
    import gnn.pipeline.semantic_fidelity as pipeline_sf
    import gnn.pipeline.summary_wiring as wiring
    import gnn.report.cross_framework_reliability as report_cfr
    import gnn.report.pipeline_report as report_pr
    import gnn.report.semantic_fidelity as report_sf

    assert (
        report_cfr.render_cross_framework_reliability_markdown
        is pipeline_cfr.render_cross_framework_reliability_markdown
    )
    assert (
        report_sf.render_semantic_fidelity_markdown
        is pipeline_sf.render_semantic_fidelity_markdown
    )
    assert report_pr.generate_pipeline_report is wiring.generate_pipeline_report
    assert report_pr.PipelineReportContext is wiring.PipelineReportContext


def test_execute_reexports_the_shared_run_leaves() -> None:
    """Old and new import paths name the same objects."""
    import gnn.execute.julia_setup as julia_setup
    import gnn.execute.subprocess_envelope as envelope
    from gnn.utils.runtime_safety import cancellation, julia_probe

    assert envelope.CancelToken is cancellation.CancelToken
    assert "CancelToken" in envelope.__all__
    assert julia_setup.julia_executable is julia_probe.julia_executable


def test_api_reexports_the_workspace_path_policy() -> None:
    """``gnn.api.path_utils`` names the leaf's objects, not copies."""
    import gnn.api.path_utils as api_paths
    from gnn.utils.runtime_safety import workspace_paths

    assert set(api_paths.__all__) == set(workspace_paths.__all__)
    for name in api_paths.__all__:
        assert getattr(api_paths, name) is getattr(workspace_paths, name), name


# ── Fresh-process import surfaces ────────────────────────────────────────────

#: Packages whose presence in ``sys.modules`` means a light import went heavy.
HEAVY_PREFIXES = (
    "gnn.report",
    "gnn.execute",
    "gnn.cli",
    "gnn.api",
    "jax",
    "matplotlib",
    "pandas",
)

_PROBE = """
import importlib, json, sys
prefixes = tuple(json.loads(sys.argv[1]))
for module in json.loads(sys.argv[2]):
    importlib.import_module(module)
exec(sys.argv[3])
loaded = sorted(
    name for name in sys.modules
    if any(name == p or name.startswith(p + ".") for p in prefixes)
)
print(json.dumps(loaded))
"""


def loaded_after_import(
    modules: list[str],
    prefixes: tuple[str, ...] = HEAVY_PREFIXES,
    then: str = "pass",
) -> list[str]:
    """Import ``modules`` (then run ``then``) in a fresh interpreter."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _PROBE,
            json.dumps(list(prefixes)),
            json.dumps(modules),
            then,
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"fresh import of {modules} failed (rc={result.returncode})\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    loaded: list[str] = json.loads(result.stdout.strip().splitlines()[-1])
    return loaded


@pytest.mark.parametrize(
    "module",
    ["gnn", "gnn.parsers", "gnn.types", "gnn.importers", "gnn.operations"],
)
def test_light_public_surface_stays_unloaded(module: str) -> None:
    """Public light entry points leave transports, reports and heavy deps out."""
    assert loaded_after_import([module]) == []


def test_light_public_surfaces_together_stay_unloaded() -> None:
    modules = ["gnn", "gnn.parsers", "gnn.types", "gnn.importers", "gnn.operations"]
    assert loaded_after_import(modules) == []


def test_shared_operations_do_not_load_transports_when_used() -> None:
    """Listing/showing templates reads package data without importing gnn.cli."""
    loaded = loaded_after_import(
        ["gnn.operations.commands", "gnn.operations.templates"],
        prefixes=("gnn.cli", "gnn.api"),
        then=(
            "from gnn.operations.templates import list_templates, show_template\n"
            "assert len(list_templates()) >= 3\n"
            "show_template('pomdp-gridworld-3x3')\n"
        ),
    )
    assert loaded == []


@pytest.mark.parametrize(
    "module",
    ["gnn.api.parity", "gnn.api.app", "gnn.api.server"],
)
def test_api_surfaces_do_not_load_cli(module: str) -> None:
    assert loaded_after_import([module], prefixes=("gnn.cli",)) == []


@pytest.mark.parametrize(
    "module",
    [
        "gnn.pipeline.summary_wiring",
        "gnn.pipeline.cross_framework_reliability",
        "gnn.pipeline.semantic_fidelity",
    ],
)
def test_pipeline_producers_do_not_load_report(module: str) -> None:
    loaded = loaded_after_import(
        [module], prefixes=("gnn.report", "gnn.cli", "gnn.api")
    )
    assert loaded == []


def test_cancel_token_and_julia_probe_load_without_execute() -> None:
    """Creating a step token or probing julia never loads ``gnn.execute``."""
    loaded = loaded_after_import(
        [
            "gnn.utils.runtime_safety.cancellation",
            "gnn.utils.runtime_safety.julia_probe",
            "gnn.pipeline.step_executor",
        ],
        prefixes=("gnn.execute", "gnn.api", "gnn.cli"),
        then=(
            "from gnn.pipeline.step_executor import _new_cancel_token\n"
            "from gnn.utils.runtime_safety.cancellation import CancelToken\n"
            "token = _new_cancel_token()\n"
            "assert type(token) is CancelToken and not token.cancelled\n"
            "token.cancel('first'); token.cancel('second')\n"
            "assert token.cancelled and token.reason == 'first'\n"
            "from gnn.utils.runtime_safety.julia_probe import julia_executable\n"
            "assert julia_executable() is None or isinstance(julia_executable(), str)\n"
        ),
    )
    assert loaded == []

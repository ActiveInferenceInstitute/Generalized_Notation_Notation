"""Static presence contract for per-module MCP entry points.

Machine-checks the rationale documented in ``src/gnn/doc/mcp.py``: every
importable top-level package under ``src/gnn/`` must either expose an
``mcp.py`` defining ``register_tools`` (the MCP auto-discovery entry point,
see ``gnn.mcp.mcp.MCP.discover_modules``) or be on the explicit
``NO_MCP_MODULES`` allowlist below. The allowlist holds static data/documentation
and internal support packages with deliberately no MCP surface; add a package
there only with an explicit rationale. ``rxinfer_interchange`` contains internal
implementation owners of the supported ``gnn.rxinfer_bridge`` facade; it must
not introduce a second public registry or MCP surface.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.mcp

SRC_GNN = Path(__file__).resolve().parents[2] / "src" / "gnn"

# Importable packages with deliberately no MCP surface: static Markdown
# (documentation, manuscript), generated/checked-in data (grammars, schemas,
# type_systems, types, formal_specs, gnn_examples-less data trees), and
# non-pipeline support code (parsers, testing, rxinfer_interchange).
#
# `extract`, `multimodel`, `processing`, and `schema_validator` previously sat
# here as support-code/data packages; all now expose real MCP surfaces
# (extract_pomdp, generate_dependency_graph, processing.parse_gnn_file,
# schema_validator.validate_comprehensive) and live under auto-discovery
# instead.
NO_MCP_MODULES = frozenset(
    {
        "documentation",
        "formal_specs",
        "grammars",
        "manuscript",
        "parsers",
        "rxinfer_interchange",
        "schema",
        "schemas",
        "testing",
        "type_systems",
        "types",
    }
)


def test_every_top_level_package_has_mcp_entry_point_or_is_allowlisted() -> None:
    """Each src/gnn/<package>/ exposes mcp.py register_tools or is allowlisted."""
    packages = {
        d.name
        for d in SRC_GNN.iterdir()
        if d.is_dir() and not d.name.startswith("_") and (d / "__init__.py").exists()
    }
    offenders = sorted(
        name
        for name in packages - NO_MCP_MODULES
        if not (SRC_GNN / name / "mcp.py").is_file()
        or "def register_tools"
        not in (SRC_GNN / name / "mcp.py").read_text(encoding="utf-8")
    )
    assert not offenders, (
        "Packages without an mcp.py register_tools entry point "
        "(extend NO_MCP_MODULES only with rationale): "
        f"{[str(SRC_GNN / name / 'mcp.py') for name in offenders]}"
    )

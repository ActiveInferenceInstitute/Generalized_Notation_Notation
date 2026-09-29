"""
Test Helpers Module

Provides reusable, typed helpers for test execution:

- ``script_loader``  — load standalone scripts by path (importlib boilerplate)
- ``gnn_samples``    — canonical sample GNN markdown content
- ``mcp_stubs``      — in-memory MCP registry test double (``MCPTools``)
- ``mcp_census``     — exact MCP census pin (``EXPECTED_MCP_TOOLS`` / ``EXPECTED_MCP_MODULES``)
- ``render_recovery``— recovery-friendly bulk render for resilience tests
- ``bar_labels``     — bar value-label pixel-gap checks (import the submodule:
  it pulls in ``matplotlib.pyplot``, so it is not re-exported here)
- path helpers + sample-model loader for ``test_data/``
"""

from pathlib import Path
from typing import Any, Dict

from .gnn_samples import SAMPLE_GNN_CONTENT, write_sample_gnn_markdown
from .mcp_census import CENSUS_SOURCE, EXPECTED_MCP_MODULES, EXPECTED_MCP_TOOLS
from .mcp_stubs import FakeMCPTime, MCPTools
from .render_recovery import render_gnn_files
from .script_loader import load_module_from_path


def get_test_data_dir() -> Path:
    """Get path to test data directory."""
    return Path(__file__).parent.parent / "test_data"


def get_sample_gnn_model() -> Path:
    """Get path to sample GNN model file."""
    return get_test_data_dir() / "sample_gnn_model.md"


def load_sample_gnn_spec() -> Dict[str, Any]:
    """Load and parse the sample GNN specification."""
    sample_file = get_sample_gnn_model()
    if not sample_file.exists():
        return {
            "name": "sample_model",
            "states": ["s1", "s2"],
            "observations": ["o1"],
            "parameters": {},
        }

    # Basic parsing of GNN markdown
    content = sample_file.read_text()
    spec: dict[str, Any] = {"name": "sample_model", "raw_content": content}

    # Extract model name if present
    for line in content.splitlines():
        if line.startswith("## ModelName"):
            # Next non-empty line is the name
            idx = content.find(line) + len(line)
            remaining = content[idx:].strip()
            if remaining:
                spec["name"] = remaining.split("\n")[0].strip()
            break

    return spec


__all__: list[str] = [
    "SAMPLE_GNN_CONTENT",
    "EXPECTED_MCP_TOOLS",
    "EXPECTED_MCP_MODULES",
    "CENSUS_SOURCE",
    "MCPTools",
    "FakeMCPTime",
    "load_module_from_path",
    "write_sample_gnn_markdown",
    "render_gnn_files",
    "get_test_data_dir",
    "get_sample_gnn_model",
    "load_sample_gnn_spec",
]

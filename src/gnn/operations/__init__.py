"""Transport-neutral GNN operations shared by the CLI, HTTP API and MCP.

``gnn.operations`` sits *below* every transport adapter: :mod:`gnn.cli`,
:mod:`gnn.api` and the MCP tool wrappers import downward from here, and this
package never imports any of them. Adapters keep only their own contract
mapping (CLI exit codes, HTTP envelopes and status codes, MCP payloads).

Submodules:

- :mod:`gnn.operations.commands` — validate/parse/extract/preflight/report
  command semantics.
- :mod:`gnn.operations.templates` — the maintained template library
  (list/show/pull with checksum and symlink custody checks).

The package root is a lazy facade (PEP 562): ``import gnn.operations`` loads
no submodule, and each public name resolves its owning submodule on first
attribute access.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    # gnn.operations.commands
    "EXTRACT_DEFECT_EMPTY_SKELETON": "commands",
    "EXTRACT_DEFECT_ERROR_STATUS": "commands",
    "ValidationOutcome": "commands",
    "build_parse_payload": "commands",
    "extract_payload_defect": "commands",
    "preflight_severities": "commands",
    "publish_pipeline_report": "commands",
    "run_validation_checks": "commands",
    # gnn.operations.templates
    "TemplateRecord": "templates",
    "list_templates": "templates",
    "pull_template": "templates",
    "show_template": "templates",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve a public operation from its owning submodule on first use."""
    submodule = _EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{submodule}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})

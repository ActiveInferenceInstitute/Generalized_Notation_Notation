# `gnn/utils/runtime_safety/` — agent contract

Concern-package home (S2-33 split; design: `docs/development/utils_split_design.md`).
The old top-level module paths under `gnn/utils/` were removed with the
SC-38 facade takedown — import from this package.

## Invariants

- `_EXPORT_MAP` keys in `gnn/utils/__init__.py` are frozen (113 keys,
  golden-asserted by `tests/tests/test_infrastructure_exports.py`); this
  package's leaf modules are the *values*.
- No import-time side effects at package level: leaves load on first use
  (`tests/tests/test_light_import.py` is the gate).
- Cross-family imports go through leaf modules, never through any facade.
- Zero behavior changes: this package is a mechanical reorganization; the
  per-module contracts live in each leaf's docstring.

## Shared run leaves

`cancellation.py` (`CancelToken`) and `julia_probe.py` (`julia_executable`)
import only the standard library. `gnn.pipeline`, `gnn.setup` and
`gnn.security` sit below `gnn.execute` and take these names from here, as does
the `gnn.api` transport;
`gnn.execute.subprocess_envelope` and `gnn.execute.julia_setup` re-export the
same objects. Keep both files free of `gnn` imports
(`tests/architecture/test_dependency_direction.py`).

`workspace_paths.py` owns the workspace path policy for the HTTP API and the
MCP tools of `gnn.execute`, `gnn.setup` and `gnn.llm`. It imports only this
package's `filesystem` leaf; `gnn.api.path_utils` re-exports it.

## Tests

`tests/tests/test_infrastructure_exports.py` (frozen surface),
`tests/tests/test_light_import.py` (lazy facade), and the per-family suites
under `tests/`.

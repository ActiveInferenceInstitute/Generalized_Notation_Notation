# Render Processor Package — Agent Scaffolding

## Module Overview

**Purpose**: POMDP-aware rendering of GNN specifications into framework-specific simulation code, with receipt persistence and aggregate summaries.

**Pipeline Step**: Step 11: Code rendering (11_render.py); see [../AGENTS.md](../AGENTS.md) for step-level context.

**Category**: Code Generation / Simulation Framework Integration

**Status**: ✅ Production Ready

**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)

**Last Updated**: 2026-09-28

## File Structure

```
processor/
├── __init__.py   # Facade: module docstring, numpy soft-dependency flag, sys.path bootstrap, registry/naming imports, grouped re-exports from the five leaf modules
├── metadata.py   # Capability metadata: get_module_info info dict, get_available_renderers registry delegate
├── parsing.py    # Spec adaptation and normalization: output stems, node/spec mapping, parse-summary rehydration, vector flattening, POMDP validation, matrix normalization
├── receipts.py   # Receipt persistence: prior-summary loading, file identity digests, atomic JSON writes, same-run receipt replacement, framework selection parsing
├── pipeline.py   # Directory-level orchestration: process_render walker, basic per-file fallback, overview documentation generation
└── rendering.py  # Single-spec dispatch: render_gnn_spec target routing and continuous-model routing via _render_continuous_target
```

No MCP tools are defined in this package; the renderer MCP surface lives in the
parent module ([../AGENTS.md](../AGENTS.md), `src/gnn/render/mcp.py`), which routes
single-framework requests through `render.processor.render_gnn_spec`
(mcp.py:163). Sub-package docs: [README](README.md) (usage overview) and
this AGENTS.md (maintenance contract); the parent [README](../README.md)
covers module-wide usage.

## Module Map

| Module | Owns |
|---|---|
| `__init__.py` | Facade re-export groups (`__init__.py:38-77`), numpy soft-dependency flag (`__init__.py:20-26`), `sys.path` bootstrap (`__init__.py:28-34`) |
| `metadata.py` | Capability metadata: `get_module_info` (metadata.py:14), `get_available_renderers` (metadata.py:48) |
| `parsing.py` | Spec adaptation/normalization: `_safe_output_stem` (parsing.py:25), `_node_to_mapping` (parsing.py:30), `_internal_representation_to_mapping` (parsing.py:46), `_rehydrate_file_backed_parse_summary` (parsing.py:84), `_normalize_initial_vectors` (parsing.py:118), `_render_succeeded` (parsing.py:139), `validate_pomdp_for_rendering` (parsing.py:163), `normalize_matrices` (parsing.py:196) |
| `receipts.py` | Receipt persistence: `_load_prior_render_summary` (receipts.py:23), `_render_file_identity` (receipts.py:44), `_atomic_render_json` (receipts.py:52), `_write_render_receipt` (receipts.py:70), `parse_frameworks_selection` (receipts.py:192) |
| `pipeline.py` | Orchestration: `process_render` (pipeline.py:29), `_process_single_gnn_file_basic` (pipeline.py:323), `_create_overview_documentation` (pipeline.py:401) |
| `rendering.py` | Single-spec dispatch: `_render_continuous_target` (rendering.py:23), `render_gnn_spec` (rendering.py:91) |

## Public Surface (facade contract)

The facade binds every name below at import time; it defines no `__all__`, so
the non-underscore names are the effective public API. Callers should import
from `gnn.render.processor` (or the `gnn.render` re-exports) rather than from
leaf modules — the leaf split is an internal layout detail. Underscore-prefixed
names remain reachable through the facade but are internal helpers consumed by
the parent package and tests.

| Source module | Facade names | Cite |
|---|---|---|
| `gnn.render.framework_registry` | `get_lite_frameworks` (public name); `get_available_renderers` and `get_supported_frameworks` bound only to the private aliases `_registry_get_available_renderers` / `_registry_get_supported_frameworks` | `__init__.py:38-46` |
| `gnn.render.naming` | `safe_output_stem` | `__init__.py:47` |
| `gnn.render.processor.metadata` | `get_available_renderers`, `get_module_info` | `__init__.py:48-51` |
| `gnn.render.processor.parsing` | `_internal_representation_to_mapping`, `_node_to_mapping`, `_normalize_initial_vectors`, `_rehydrate_file_backed_parse_summary`, `_render_succeeded`, `_safe_output_stem`, `normalize_matrices`, `validate_pomdp_for_rendering` | `__init__.py:52-61` |
| `gnn.render.processor.pipeline` | `_create_overview_documentation`, `_process_single_gnn_file_basic`, `process_render` | `__init__.py:62-66` |
| `gnn.render.processor.receipts` | `_atomic_render_json`, `_load_prior_render_summary`, `_render_file_identity`, `_write_render_receipt`, `parse_frameworks_selection` | `__init__.py:67-73` |
| `gnn.render.processor.rendering` | `_render_continuous_target`, `render_gnn_spec` | `__init__.py:74-77` |

Note the two `get_available_renderers` bindings: the registry variant is aliased
to `_registry_get_available_renderers` (`__init__.py:39`), while the public
facade name is the metadata delegate (metadata.py:48). Private-by-convention
state on the facade: `np`, `NUMPY_AVAILABLE`, `logger`, and the `_registry_*`
aliases.

## Provenance

This package was extracted in place from the former monolithic
`src/gnn/render/processor.py`; the facade surface was preserved exactly so
existing import sites continue to work unchanged. Cross-module wiring after the
split:

- `rendering.py:12` imports `gnn.render.processor.parsing` for spec adaptation.
- `pipeline.py:15-24` imports `processor.parsing` and `processor.receipts`.
- `parsing.py:107-108` lazily imports `gnn.extract.pomdp_extractor` and
  `gnn.render.pomdp_processor.pomdp_to_gnn_spec` inside
  `_rehydrate_file_backed_parse_summary`, keeping heavy POMDP extraction off the
  module import path.
- Top-level package: `src/gnn/__init__.py:41` lazily imports
  `get_available_renderers` / `render_gnn_spec` from `.render.processor`, and
  `src/gnn/__init__.py:91-92` maps both names to `render.processor` in the
  lazy-import module map.
- Parent MCP surface: `src/gnn/render/mcp.py:163` documents routing
  single-framework requests through `render.processor.render_gnn_spec`.
- Shared helpers: `src/gnn/render/naming.py:4-5` names `render.processor` (with
  `render.pomdp_processor`) as the consumer of the single-source
  `safe_output_stem`.

## Runtime Bootstrap

- **NumPy soft dependency**: `__init__.py:20-26` (mirrored in
  `parsing.py:16-22`) imports numpy in a `try`/`except ImportError` and sets
  `NUMPY_AVAILABLE`; `np` is `None` (via `cast`) when numpy is missing, so the
  package imports cleanly in minimal environments.
- **Path bootstrap**: `__init__.py:28-34` inserts the resolved `src` directory
  and project root into `sys.path` (idempotently) so cross-module imports work
  when the package is executed outside an installed distribution.
- **Lazy heavy imports**: POMDP extraction and per-framework renderers are
  imported inside functions (parsing.py:107-108, pipeline.py:70-78,
  rendering.py:31) rather than at module top level.

## Testing & Verification

Tests that import this package (verified by import statement):

| Test | Cite |
|---|---|
| `tests/render/test_render_contracts.py` — `parse_frameworks_selection`, `_render_succeeded`, `render_gnn_spec` contracts | test_render_contracts.py:29-32 |
| `tests/render/test_continuous_public_contract.py` — `render_gnn_spec` continuous contract | test_continuous_public_contract.py:13 |
| `tests/render/test_render_cli_targets.py` — CLI target dispatch guard | test_render_cli_targets.py:25 |
| `tests/render/test_composed_model_kinds.py` — `process_render`, `render_gnn_spec` model-kind gating | test_composed_model_kinds.py:28 |
| `tests/gnn/test_pomdp_extractor_continuous.py` — `process_render` on continuous exemplars | test_pomdp_extractor_continuous.py:11 |
| `tests/execute/test_receipt_reliability.py` — `_atomic_render_json` replace semantics | test_receipt_reliability.py:116 |
| `tests/pipeline/test_pipeline_render_execute_analyze.py` — end-to-end render/execute/analyze | test_pipeline_render_execute_analyze.py:33 |

```bash
uv run --extra dev python -m pytest \
  tests/render/test_render_contracts.py \
  tests/render/test_continuous_public_contract.py \
  tests/render/test_render_cli_targets.py \
  tests/render/test_composed_model_kinds.py \
  tests/gnn/test_pomdp_extractor_continuous.py \
  tests/execute/test_receipt_reliability.py \
  tests/pipeline/test_pipeline_render_execute_analyze.py \
  -q
```

Coverage over the broader render suite, mirroring the parent module pattern:

```bash
uv run --extra dev python -m pytest tests/render/test_render*.py \
    --cov=src/gnn/render --cov-report=term-missing
```

---

**Last Updated**: 2026-09-28
**Maintainer**: GNN Pipeline Team
**Status**: ✅ Production Ready
**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern

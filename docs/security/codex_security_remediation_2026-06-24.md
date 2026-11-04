# Codex Security Remediation - 2026-06-24

## Summary

A Codex Security standard scan reviewed the repository at commit
`5e77e657d7c02d39b7879c82a587e4614ece773e` and identified four reportable
findings across MCP execution, MCP LLM file access, generated code emission,
and generated artifact path handling.

All four findings were remediated with repository-local path validation,
manifest-gated execution, generated-code literal escaping, safe output filename
stems, and focused regression tests.

## Findings Closed

| Finding | Severity | Affected surface | Remediation |
| --- | --- | --- | --- |
| `GNN-SEC-001` | High | `execute_gnn_model_mcp` accepted arbitrary caller-supplied Python paths. | The MCP wrapper now requires repository-local GNN source files with `.md`, `.json`, `.yaml`, or `.yml` suffixes before delegating to the executor. Python payloads are rejected before execution. |
| `GNN-SEC-002` | High | `process_execute_mcp` could execute scripts from caller-selected render output trees. | The MCP wrapper now accepts only repository-local Step 11 render output directories containing `render_processing_summary.json`, and calls `process_execute(..., require_render_summary=True)`. The processor refuses to execute discovered scripts when the trusted render summary is absent or invalid for this mode. |
| `GNN-SEC-003` | High | `generate_bnlearn_code` interpolated untrusted model metadata into generated Python source. | The bnlearn generator now emits model metadata through Python string literals, coerces timestep values to positive integer literals, and no longer carries the prior vulnerable template. |
| `GNN-SEC-004` | Medium | LLM MCP tools could read arbitrary local files and send content to configured providers. | LLM MCP entry points now resolve input files, target directories, and optional documentation output paths through repository-local validation before reading content or calling provider-facing code. |

## Behavior Changes

- `process_execute_mcp` now represents the Step 12 execution boundary: callers
  must pass a Step 11 render output directory, not an arbitrary source or script
  directory.
- `execute_gnn_model_mcp`, `execute_pymdp_simulation_mcp`, and LLM MCP tools
  reject files outside the repository root.
- MCP output directories and documentation output files must stay under the
  repository root.
- Generated artifact filenames derive from sanitized stems, while generated code
  can still preserve the original model name as data.

## Changed Surfaces

> **Path note (2026-11-04).** The module paths below are the ones the remediation
> touched at commit `5e77e657`. Three of them have since been split in place and no
> longer resolve as single files:
> `src/gnn/execute/processor.py` → `src/gnn/execute/processor/`,
> `src/gnn/render/processor.py` → `src/gnn/render/processor/`, and
> `src/gnn/render/pomdp_processor.py` → `src/gnn/render/pomdp_processor/`.
> The commands further down are retained exactly as recorded and are a historical
> receipt, not a runnable recipe against current `main`.

- `src/gnn/execute/mcp.py`
- `src/gnn/execute/processor.py`
- `src/gnn/llm/mcp.py`
- `src/gnn/render/generators.py`
- `src/gnn/render/processor.py`
- `src/gnn/render/pomdp_processor.py`

## Regression Coverage

Focused regression tests cover the remediated attack shapes:

- `.py` payloads passed to `execute_gnn_model_mcp` are rejected and not run.
- Render trees without `render_processing_summary.json` are rejected and not run.
- Manifested Step 11 render outputs still execute through `process_execute_mcp`.
- LLM MCP outside-repository inputs and outputs are rejected before provider or
  documentation-generation functions are called.
- Malicious bnlearn model metadata compiles only as string literal data.
- Generated output filenames cannot escape the requested output directory.

## Verification

The remediation was verified with:

```bash
uv run --extra dev python -m pytest \
  tests/execute/test_execute_mcp_wiring.py \
  tests/llm/test_llm_mcp_security.py \
  tests/render/test_render_cli_targets.py \
  tests/render/test_pomdp_renderer_regressions.py -q

uv run --extra dev python -m ruff check \
  src/gnn/execute/mcp.py src/gnn/execute/processor.py src/gnn/llm/mcp.py \
  src/gnn/render/generators.py src/gnn/render/pomdp_processor.py src/gnn/render/processor.py \
  tests/execute/test_execute_mcp_wiring.py \
  tests/llm/test_llm_mcp_security.py \
  tests/render/test_render_cli_targets.py \
  tests/render/test_pomdp_renderer_regressions.py

uv run --extra dev python -m ruff format --check \
  src/gnn/execute/mcp.py src/gnn/execute/processor.py src/gnn/llm/mcp.py \
  src/gnn/render/generators.py src/gnn/render/pomdp_processor.py src/gnn/render/processor.py \
  tests/execute/test_execute_mcp_wiring.py \
  tests/llm/test_llm_mcp_security.py \
  tests/render/test_render_cli_targets.py \
  tests/render/test_pomdp_renderer_regressions.py

git diff --check
```

Observed results:

- Focused pytest: `40 passed`
- Ruff lint: `All checks passed`
- Ruff format check: `10 files already formatted`
- Diff whitespace check: clean

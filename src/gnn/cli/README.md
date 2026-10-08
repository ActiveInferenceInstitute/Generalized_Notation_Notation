# GNN CLI Module

## Overview

Unified command-line interface for the GNN pipeline. Provides subcommands for running, validating, rendering, templating, and managing GNN models.

**Entry point**: `gnn = "gnn.cli:main"` (defined in `pyproject.toml`)

## Subcommands

| Command | Description |
|---------|-------------|
| `gnn run` | Execute the full 25-step pipeline |
| `gnn validate <file>` | Validate a GNN file (sections, state-space, connections, dimensions) |
| `gnn parse <file>` | Parse a GNN file and output JSON/YAML/summary |
| `gnn extract <file>` | Extract the POMDP state space as structured JSON (`--strict`/`--no-strict`, `--compact`, `--json` standard envelope) |
| `gnn render <file>` | Render a GNN file to a specific framework (pymdp, rxinfer, jax, etc.); `--json` emits the standard envelope |
| `gnn report` | Generate pipeline report from existing outputs |
| `gnn reproduce <hash>` | Re-run from a previous run hash (content-addressable) |
| `gnn preflight` | Run environment & config checks |
| `gnn health` | Show renderer generator-module availability and environment preflight status |
| `gnn health --strict` | Exit nonzero when environment preflight reports errors |
| `gnn serve --surface jobs` | Start long-running services; `--surface` selects `runs` (default), `jobs`, `both` (jobs on port+1), or `website` (static server, port 8090) |
| `gnn templates list` | List maintained local GNN templates with checksums |
| `gnn templates show <name>` | Show one maintained template record |
| `gnn models list` | Query the local model registry |
| `gnn pull <name>` | Copy a maintained template into an input directory |
| `gnn lsp` | Launch GNN Language Server (stdio) |
| `gnn watch <dir>` | Monitor directory and live-reparse on file change |
| `gnn graph <file>` | Generate dependency graph from multi-model files |
| `gnn gui` | Run Step 22 GUI processing: headless artifacts or interactive GUI servers (--gui-types, --interactive) |
| `gnn mcp list` | List registered MCP tools (`--json` for the standard envelope) |
| `gnn mcp info <name>` | Show one MCP tool's registry record |

Exit codes follow one contract: `0` is success, `1` is error, and `2` is a
completed command with warnings, validation findings, or degraded readiness.

`gnn run` accepts `--parallel`, `--consolidated-steps` and `--strict` and delegates
to the shared owned invocation. `--only-steps` must contain unique registered
integers; an empty option, unknown steps and duplicates fail before dispatch.
Omit it to select all steps. Required prerequisites are included automatically.
An explicitly empty model selection remains skipped work. Renderer choices come
from the shared live inventory; generated code does not establish native runtime
readiness. Backend options reject unknown keys, coercions and invalid limits.

## Usage

```bash
# Full pipeline
gnn run --target-dir input/gnn_files --only-steps 3 5 11 12 --verbose

# Validate a model
gnn validate input/gnn_files/discrete/actinf_pomdp_agent.md --strict

# Parse to JSON
gnn parse input/gnn_files/discrete/actinf_pomdp_agent.md

# Extract the POMDP state space as JSON (pretty by default)
gnn extract input/gnn_files/discrete/actinf_pomdp_agent.md
gnn extract input/gnn_files/discrete/actinf_pomdp_agent.md --no-strict --compact

# Extract wrapped in the standard CLI JSON envelope
gnn extract input/gnn_files/discrete/actinf_pomdp_agent.md --json

# Check environment
gnn preflight
gnn health
gnn health --strict

# Inspect and dry-run template installation
gnn templates list
gnn templates show pomdp-gridworld-3x3
gnn pull pomdp-gridworld-3x3 --output-dir /tmp/gnn-pull --dry-run

# Start the API: gnn serve [--surface runs|jobs|both|website]; both runs the jobs surface on port+1
gnn serve --surface jobs

# Inspect the MCP tool surface
gnn mcp list
gnn mcp info cli.health
```

## Architecture

The CLI module is a thin dispatcher — each subcommand delegates to the corresponding module's public API:

- `run` → `main.main()`
- `validate` → `gnn.schema` section/state-space/connection/dimension checks plus `validation.validate_content()` shared semantic evidence. JSON includes `data.semantic`. Findings retain exit code 2, or 1 with `--strict`.
- `parse` → `gnn.schema.parse_state_space()` + `gnn.frontmatter.parse_frontmatter()`
- `extract` → `gnn.extract.extract_to_json()` (lazy import; emits a structured error envelope when the extractor is unavailable)
- `render` → `render.processor` (planned full integration)
- `report` → `report.pipeline_report.generate_pipeline_report()`
- `reproduce` → `pipeline.hasher.lookup_run()` + `main.main(override_args=...)`
- `preflight` → `pipeline.preflight.run_preflight()`
- `health` → `render.health.check_renderers()` + `pipeline.preflight.check_environment()`
- `serve` → `api.app.start_server()` (surface `runs`), `api.server.run_server()` (surface `jobs`), or `both` (jobs via `uvicorn.Server` on port+1 while runs blocks on the main thread)
- `templates` / `pull` → `cli.templates` maintained template index, checksum, and copy helpers
- `models` → `model_registry.registry.ModelRegistry`
- `lsp` → `gnn.cli.lsp.start_lsp()`: canonical pygls server (`gnn.lsp`) when pygls is importable, pygls-free JSON-RPC fallback otherwise
- `watch` → `gnn.cli.watcher.GNNWatcher()`
- `graph` → `gnn.dep_graph.render_graph_from_file()`
- `gui` → `gnn.gui.process_gui()` (lazy import; headless artifacts by default, interactive servers with `--interactive`)
- `mcp` → `gnn.mcp.initialize()` plus registry access (`list_available_tools`, `get_tool_info`), lazily imported

## References

- [SPEC.md](SPEC.md) — Module specification
- [AGENTS.md](AGENTS.md) — Agent documentation

# CLI Module — Agent Scaffolding

## Module Overview

**Purpose**: Command-line interface and dispatch functionality for the GNN pipeline.
**Pipeline Step**: Infrastructure module (not a numbered step)
**Category**: Infrastructure / Development Tools
**Status**: ✅ Production Ready
**Version**: [pyproject.toml](../../../pyproject.toml) (canonical)
**Last Updated**: 2026-09-25

The CLI module provides the `gnn` command-line tool — a unified interface to the entire GNN pipeline. Its thin dispatcher routes the registered subcommands to their module APIs. Public exits are `0` for success, `1` for errors, and `2` for completed commands with warnings or degraded readiness.

## Architecture

- **Pattern**: Thin dispatcher (not a pipeline step)
- **Entry point**: `gnn.cli:main` (registered in `pyproject.toml [project.scripts]`)
- **Dependencies**: All pipeline modules (imported lazily per subcommand)

## Capabilities

- **Pipeline execution** via `gnn run` with skip/only-steps, log-format, and skip-llm options
- **Worker limits** via `gnn run --file-workers N` for Step 3/5 CPU file jobs and `--parallel --step-workers N` for concurrent step tiers; values are exact integers from 1 to 256.
- **File validation** via `gnn validate` (section, state-space, connection, dimension checks)
- **JSON/YAML parsing** via `gnn parse` with format and summary modes
- **Code generation** via `gnn render` (PyMDP, RxInfer, ActiveInference.jl, JAX, NumPyro, Stan, PyTorch, DisCoPy, bnlearn)
- **POMDP extraction** via `gnn extract` (structured JSON of the POMDP state space, with graceful degradation when the extractor is unavailable)
- **Run reproduction** via `gnn reproduce` using content-addressable hashing
- **Static complexity bounds** via `gnn complexity <model|dir>` — one `[ESTIMATE]`-labeled BOUNDS row per backend as a terminal table plus the `gnn.complexity_estimate/v1` receipt as stable sorted-key JSON (`--json` standard envelope; `--output PATH` writes the receipt file). No execution, no measurement — estimates are arguments over declared structure, never numbers.
- **Empirical benchmark + calibration** via `gnn benchmark <dir> --frameworks ... --repeats K` — corpus harness through the existing execution envelope (`--frameworks` comma-separated, default `all`; `--repeats` int, default 3; `--json` envelope; `--output-dir` default `output/cross_framework`). Writes `complexity_benchmark.json` (`gnn.complexity_benchmark/v1`, per-run rows with K-rep timing and the environment block) and `complexity_calibration.json` (`gnn.complexity_calibration/v1`, static-vs-measured join on `(source_sha256, framework)` with a factual `calibration_note` per row). Unavailable backends are recorded `available: false`, never skipped silently.
- **Environment checks** via `gnn preflight` and `gnn health`
- **Source-bound numerical witnesses** via `gnn witness SOURCE --results RESULT --program PROGRAM --output RECEIPT`, or `gnn witness --verify RECEIPT`. The restricted source reader, reference calculation and whole-receipt replay live in `gnn.validation`; numerical agreement does not authenticate execution or establish a Lean proof.
- **Live development** via `gnn watch` (file monitoring with 250ms debounce)
- **Dependency graphs** via `gnn graph` (Mermaid/text output)
- **API server** via `gnn serve --surface` (`runs` → `api/app.py`, `jobs` → `api/server.py`, `both` starts the jobs surface on port+1 in a daemon thread, `website` → `gnn/website/serve.py` loopback static server + optional live reload, port 8090)
- **MCP surface inspection** via `gnn mcp list` / `gnn mcp info <tool>` (lazy `gnn.mcp` registry bridge; `--json` emits the standard envelope)
- **LSP server** via `gnn lsp` (canonical `gnn.lsp` pygls server when pygls is importable; `cli/lsp.py` JSON-RPC fallback)

## File Structure

```
cli/
├── __init__.py          # Package surface: dispatch table, main(), re-exports
├── __main__.py          # `python -m gnn.cli` entry point
├── commands.py          # Compatibility facade over gnn.operations.commands
├── lsp.py               # GNN Language Server (stdio)
├── mcp.py               # MCP tool surface for CLI subcommands
├── templates.py         # Compatibility facade over gnn.operations.templates (data stays in template_assets/)
├── parser.py            # build_parser + argparse type validators
├── helpers.py           # Shared envelope/emit/guard/logging helpers + exit codes
├── handlers_pipeline.py # run/validate/parse/extract/render/graph handlers
├── handlers_ops.py      # report/reproduce/preflight/health handlers
├── handlers_service.py  # serve/watch/lsp/gui/mcp handlers
├── handlers_library.py  # templates/models/pull handlers
├── handlers_complexity.py # complexity/benchmark handlers
├── template_index.json  # Externalized template metadata
├── template_assets/     # Packaged GNN template files
├── AGENTS.md            # This file
├── README.md            # Usage guide
├── SPEC.md              # Module specification
└── SKILL.md             # Capability API
```

## References

- [README.md](README.md) — Usage guide with examples
- [SPEC.md](SPEC.md) — Specification

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API

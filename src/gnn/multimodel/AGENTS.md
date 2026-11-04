# Multi-Model Support — Agent Scaffolding

## Module Overview

**Purpose**: Handle GNN files that contain more than one model — split them on
`---` horizontal rules (after stripping optional YAML front-matter), parse each
model block independently, and build/render the inter-model dependency graph
implied by shared variables.

**Pipeline Step**: None — library module consumed by the `gnn graph` subcommand
(`src/gnn/cli/handlers_pipeline.py`), the HTTP API parity endpoint
(`src/gnn/api/parity.py`) and this package's own MCP tool
(`src/gnn/multimodel/mcp.py`). No website module imports it.

**Category**: Infrastructure module

**Status**: Production Ready

**Version**: [pyproject.toml](../../../pyproject.toml) (canonical)

---

## Core Functionality

1. `split_models(content)` — split raw file content into per-model blocks on
   `---` horizontal rules; optional front-matter is removed first via
   `gnn.parsers.frontmatter.has_frontmatter` / `parse_frontmatter`.
2. `parse_multimodel(content, file_path=None)` — parse each block through
   `gnn.schema.parse_state_space` / `parse_connections`, returning per-model
   variable/connection lists plus parse errors and the model index.
3. `build_dependency_graph(models, file_path=None)` — create `ModelNode`s and
   infer `ModelEdge`s between models that share variables.
4. `render_graph_from_file(file_path, output_format)` — read a file, build the
   graph, and render Mermaid (`graph TD`) or a plain-text adjacency list.

## Module Structure

- `src/gnn/multimodel/multimodel.py` — splitting and per-model parsing
- `src/gnn/multimodel/dep_graph.py` — dependency graph dataclasses and rendering
- `src/gnn/multimodel/__init__.py` — curated re-exports (`__all__`)

## MCP Integration

- `mcp.py` registers the `generate_dependency_graph` tool (category
  `multimodel`): the `gnn graph` parity surface — `render_graph_from_file`
  rendered as Mermaid or a text adjacency list, with typed errors for
  missing files and unknown formats.
- Tests: `tests/multimodel/test_multimodel_mcp_tools.py`.

## Dependencies

- `gnn.parsers.frontmatter` — optional front-matter stripping in `split_models`
- `gnn.schema` — per-block parsing inside `parse_multimodel`
- `gnn.multimodel.multimodel` — direct module import from `dep_graph.py`
  (avoids a package-init import cycle)

---

## Testing

```bash
uv run --extra dev python -m pytest tests/gnn/test_gnn_multimodel.py tests/gnn/test_gnn_dep_graph.py tests/multimodel/test_multimodel_mcp_tools.py -q
```

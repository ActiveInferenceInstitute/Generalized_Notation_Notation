# Multi-Model Support

Multi-model GNN file handling: splitting a file that contains several models,
parsing each model independently, and building/rendering the inter-model
dependency graph induced by shared variables.

## Module Structure

| File | Purpose |
|------|---------|
| `multimodel.py` | `split_models` (splits on `---` horizontal rules, stripping optional YAML front-matter via `gnn.parsers.frontmatter`) and `parse_multimodel` (parses each block independently via `gnn.schema`) |
| `dep_graph.py` | `ModelNode` / `ModelEdge` / `DependencyGraph`, `build_dependency_graph` (infers edges from shared variables) and `render_graph_from_file` (Mermaid or plain-text adjacency list) |

## Public API

Re-exported from `gnn.multimodel` (see `__init__.py`):

- `split_models(content) -> List[str]`
- `parse_multimodel(content, file_path=None) -> List[Dict[str, Any]]`
- `build_dependency_graph(models, file_path=None) -> DependencyGraph`
- `render_graph_from_file(file_path, output_format="mermaid") -> str`
- `ModelNode`, `ModelEdge`, `DependencyGraph` (`DependencyGraph.to_mermaid()` / `to_adjacency_list()`)

## Usage

```python
from gnn.multimodel import parse_multimodel, render_graph_from_file

results = parse_multimodel(content, file_path="multi.gnn")
print(render_graph_from_file("multi.gnn", output_format="mermaid"))
```

The `gnn graph` CLI subcommand renders the dependency graph of a multi-model
file through `render_graph_from_file` (`src/gnn/cli/__init__.py`).

## Consumers

- `src/gnn/cli/handlers_pipeline.py` — `graph` subcommand
- `src/gnn/api/parity.py` — HTTP API parity endpoint
- `src/gnn/multimodel/mcp.py` — `generate_dependency_graph` MCP tool

## See Also

- [Parent: gnn/README.md](../README.md)
- [AGENTS.md](AGENTS.md) — Architecture documentation

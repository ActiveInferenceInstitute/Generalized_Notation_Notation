# Multimodel Module Specification

## Overview
Multi-model GNN file handling: splitting a file that contains several models on `---` horizontal rules, parsing each model independently, and building/rendering the inter-model dependency graph induced by shared variables.

## Components
- `multimodel.py` - `split_models(content)` (splits on horizontal rules; strips optional YAML front-matter via `gnn.parsers.frontmatter`) and `parse_multimodel(content, file_path=None)` (parses each block through `gnn.schema.parse_state_space` / `parse_connections`; returns per-model dicts with `variables`, `connections`, `errors`, and counts)
- `dep_graph.py` - `ModelNode` / `ModelEdge` / `DependencyGraph` dataclasses, `build_dependency_graph(models, file_path=None)` (infers edges from shared variables), `render_graph_from_file(file_path, output_format="mermaid")` (Mermaid flowchart or plain-text adjacency list)
- `mcp.py` - Single `generate_dependency_graph` MCP tool (`generate_dependency_graph_mcp`, `register_tools`, category `multimodel`)
- `__init__.py` - Curated public surface (`__all__`, seven names)

## Consumers
- `gnn graph` CLI subcommand renders the dependency graph through `render_graph_from_file` (`src/gnn/cli/handlers_pipeline.py`)
- HTTP API parity endpoint delegates to `render_graph_from_file` (`src/gnn/api/parity.py`)
- MCP tool `generate_dependency_graph` wraps it (`src/gnn/multimodel/mcp.py`)

## Invariants
- `parse_multimodel` does not raise on parse problems: per-model errors are collected into the returned `errors` list.
- A model separator is a `---` horizontal rule on its own line.

## Key Exports
```python
from gnn.multimodel import (
    DependencyGraph,
    ModelEdge,
    ModelNode,
    build_dependency_graph,
    parse_multimodel,
    render_graph_from_file,
    split_models,
)
```

## Receipts
```bash
uv run --extra dev python -m pytest tests/multimodel/test_multimodel_mcp_tools.py \
  tests/gnn/test_gnn_multimodel.py tests/gnn/test_gnn_dep_graph.py -q
```

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification

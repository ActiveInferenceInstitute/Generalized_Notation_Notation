# Type systems — specification

## Role

Reference artifacts for **categorical / type-level** views of GNN-style models (not the main Python runtime):

- `scala.scala`, `categorical.scala` — Scala examples
- `haskell.hs` — Haskell
- `mapping.md` — mapping notes
- `examples/` — smaller examples and local docs

These support documentation and cross-language alignment; the production parser stack remains in **`src/gnn/parsers/`**.

## Requirements

- **Python** >= 3.12 for repository tooling that ships alongside this repo (see `pyproject.toml`).
- Scala / Haskell artifacts assume their respective toolchains when type-checking outside Python.

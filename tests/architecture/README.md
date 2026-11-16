# Architecture Tests

Pytest contracts for package dependency direction and import surfaces across
`src/gnn/`.

- Static AST edges: `gnn.api` never imports `gnn.cli`; `gnn.operations` imports
  neither transport; `gnn.pipeline` never imports `gnn.report`, `gnn.cli` or
  `gnn.api`.
- Fresh-interpreter `sys.modules` probes: `gnn`, `gnn.parsers`, `gnn.types`,
  `gnn.importers` and `gnn.operations` stay light; API apps and pipeline
  producers do not load the packages they must not depend on.

See [interop_api.md](../../docs/development/interop_api.md#shared-operations-and-dependency-direction).

Run:

```bash
uv run --frozen pytest tests/architecture -q
```

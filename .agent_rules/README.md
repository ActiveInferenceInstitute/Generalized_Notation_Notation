# GNN development rules

These guides describe current contracts. Start with the root [AGENTS](../AGENTS.md)
and [SPEC](../SPEC.md), then read the owning module. Version and dependency
constraints are canonical in [pyproject.toml](../pyproject.toml).

| Task | Guide |
| --- | --- |
| Thin scripts and run ownership | [Architecture](architecture.md) |
| Module interfaces | [Module patterns](module_patterns.md) |
| Consumer tests and evidence | [Testing](testing.md) |
| Lint, types and documentation | [Quality](quality.md) |
| Failure, skip and cleanup | [Error handling](error_handling.md) |
| Installation and readiness | [Dependencies](dependencies.md) |
| Backend scientific admission | [Render frameworks](render_frameworks.md) |
| Model sources and extraction | [GNN standards](gnn_standards.md) |
| MCP discovery and calls | [MCP](mcp.md) |
| Measurement and optimization | [Performance](performance.md) |
| Diagnosis and recovery | [Troubleshooting](troubleshooting.md) |

Install one locked development environment before running checks:

```bash
uv sync --frozen --extra dev --python 3.12
uv run --frozen --no-sync gnn --help
```

Use canonical `gnn.*` imports. Numbered scripts delegate to module owners;
step names and prerequisites come from the live registry. Freeze selected
source identities and configuration once per run. An empty model selection
produces no work; directory contents cannot widen it or admit inherited
artifacts as current evidence.

Required failures, exhausted work and failed cleanup prevent success.
Optional unavailability has a distinct diagnosis. Continuing to later steps
does not turn failed visualization or execution into successful work. See
[run ownership](../docs/development/run_ownership_migration.md).

Count-changing source/test and manuscript changes require the complete SC-22
ritual in [AGENTS](../AGENTS.md). Bound source changes require the
[paired-revision procedure](../docs/development/fep_lean_paired_revision.md).
Use [TO-DO verification](../TO-DO.md#verification-and-execution-rules) and the
[workflows](../.github/workflows/README.md) for current gates.

# Operations

Shared command semantics below every transport adapter. The CLI, HTTP API and
MCP template tools call these functions and translate the result into their own
exit code or response envelope.

```
operations/
├── __init__.py   # Lazy facade (PEP 562); importing the package loads nothing else
├── commands.py   # validate / parse / preflight / extract / render / report semantics
├── templates.py  # Template index: list, show, pull (data under src/gnn/cli/)
├── README.md
├── AGENTS.md
└── SPEC.md
```

Dependency direction: `gnn.cli`, `gnn.api` and `gnn.template.mcp` import
`gnn.operations`; `gnn.operations` imports neither transport. See
[interop_api.md](../../../docs/development/interop_api.md#shared-operations-and-dependency-direction).

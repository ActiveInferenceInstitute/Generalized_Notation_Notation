# MCP discovery and execution

MCP exposes GNN through declared tool schemas and real module entrypoints.
Read [MCP AGENTS](../src/gnn/mcp/AGENTS.md), the
[transport guide](../docs/gnn/mcp/README.md) and
[tool reference](../docs/gnn/mcp/tool_reference.md).

```bash
uv run --frozen --no-sync gnn mcp list
uv run --frozen --no-sync gnn mcp info cli.health
uv run --frozen --no-sync python -m pytest tests/mcp/test_mcp_audit.py -q
```

Derive tool inventories and schemas from the live registry. Keep import-time
package isolation, declared option types/precedence and unknown-key behavior.
Execution must honor the requested backend and exact selected sources; listing
old files after a call cannot establish current rendering or execution.

Validate through actual MCP calls and corresponding Python/CLI/REST consumers.
Distinguish invalid arguments, unsupported models, missing dependencies,
backend failures, timeouts and incomplete cleanup. Preserve the owning result
contract; do not declare success merely because a wrapper returned an object.

Functional MCP/capability tests and transport/auth tests have a separate
required [CI lane](../.github/workflows/ci.yml). Registration or a passing tool
count establishes discovery, not native numerical readiness. Optional network
and provider calls require explicitly provisioned acceptance.

## Related contracts

[Public adapter parity and LLM bindings](interfaces.md) owns the cross-cutting guidance.

HTTP follows token/auth, loopback opt-in, pre-auth rate limiting and resource
allowlists in [server_http.py](../src/gnn/mcp/server_http.py). Protocol stdout
contains protocol data; logging and bounded diagnostics use their separate path.

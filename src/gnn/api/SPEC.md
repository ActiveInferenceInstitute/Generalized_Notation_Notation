---
name: api
description: Architectural specification for the API module
---

# API SPEC

## Architecture

The API is a thin FastAPI presentation layer over the canonical pipeline.
Independent run and job factories serve their respective consumers while sharing
CLI-parity routes, admission rules, run identity and subprocess supervision.
In-memory registries are research-service state, not durable scheduling or
tenant isolation. Long-running execution uses asyncio subprocesses with one
shared supervision owner; it does not execute the orchestrator in a worker
thread.

The two applications are `gnn.api.app` (run surface, launched by
`gnn serve --surface runs`) and `gnn.api.server` (job surface,
`python -m gnn.api.server`). Both advertise `MODULE_VERSION`, which re-exports
`gnn.__version__`, and both import their requests and responses from
`gnn.api.models`. Each keeps its own exclusive route table and its own
in-memory store. No cross-release API compatibility contract is declared; the
generated OpenAPI schema and the API contract tests are the authority for the
accepted surface. The
[typed consumer contract](../../../docs/development/interop_api.md) maps models,
inputs/outputs, errors, and MCP interfaces.

The installed `api` extra and its dependency constraints are declared in
[pyproject.toml](../../../pyproject.toml). [README.md](README.md) owns runnable
setup instructions. Each factory's OpenAPI schema owns the exact endpoint and
request inventory.

## Workspace and network admission

- A real source checkout supplies the compatible default workspace. An ordinary
  installed package must receive an operator-configured `GNN_API_ROOT`: an
  absolute, existing directory without symlink or reparse components. Invalid
  roots and caller paths preserve the typed path-validation error contract.
- Request paths stay within the workspace and reject parent-traversal syntax and
  redirecting components. POSIX creation pins directory descriptors; later
  pathname consumers require trusted directory entries and ancestors.
- Workspace selection controls data storage. It cannot substitute workspace code
  for the package-owned pipeline orchestrator.
- Service starters default to loopback and guard non-loopback binding. A
  configured `GNN_API_KEY` requires `X-API-Key` outside the public health/docs
  surface. Direct ASGI launchers must enforce their own bind policy. A shared key
  is service authentication, not per-user filesystem isolation.

## Execution and cleanup invariants

- JSON responses and SSE data have exactly `{status,data,error,meta}`. Validation,
  authentication and unexpected failures retain that envelope; reports are
  native Markdown.
- Each subprocess receives its unique `GNN_RUN_ID`. Summary ingestion checks run
  identity and invocation freshness before admitting evidence. Exit codes 0, 1
  and 2 retain the shared completion, failure and strict-warning policy.
- A cancellation request is not a cleanup certificate. Shared supervision uses
  bounded graceful termination and forced teardown; normal leader exit also
  initiates cleanup of inherited pipes and owned observed processes. Unverified
  cleanup raises the typed process-cleanup failure and retains its receipt rather
  than reporting successful cancellation.
- Cleanup guarantees describe the declared platform boundary. Linux and macOS
  supervise the owned group and observed descendants without a host-wide process
  census. Windows declares `direct_worker_only`; explicit descendant-containment
  or tree-wide accounting requests are refused when unsupported.
- Active-run deletion waits for cancellation to reach terminal state. Timeout,
  ambiguous ownership or unverified cleanup retains the record and returns a
  conflict. Artifact deletion protects workspace roots, ancestors, shared default
  output and redirecting entries; its outcome is reported separately.

The [filesystem and platform boundary contract](../../../docs/security/filesystem_boundaries.md)
is canonical for the adversary model, descriptor-operation limits, bounded
cleanup and native acceptance. These safeguards do not certify arbitrary hostile
producer code or permanent pathname ancestry after concurrent directory rename.

---
## Documentation

- **[README](README.md)**: Launch commands and implemented routes.
- **[AGENTS](AGENTS.md)**: Contributor workflow and module ownership.
- **[SPEC](SPEC.md)**: This specification.
- **[SKILL](SKILL.md)**: Callable discovery.

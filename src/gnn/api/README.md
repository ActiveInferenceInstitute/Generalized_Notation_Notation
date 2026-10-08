# GNN API

The optional FastAPI service exposes GNN validation, pipeline execution and run
evidence over HTTP. Jobs and run records live in memory and are lost on restart;
the service is intended for operator-controlled research workspaces.

## Choose a service surface

`gnn serve --surface runs` serves run submission, status, Markdown reports and
SSE progress. `--surface jobs` serves explicit pipeline jobs and step discovery.
Both share CLI-parity operations, request admission, response envelopes and
subprocess supervision. Open `/docs` or `/openapi.json` on the selected service
for its current routes and request schemas; [models.py](models.py) and
[parity.py](parity.py) own those contracts.

## Start from a checkout

Run these commands from the repository root:

```bash
uv sync --frozen --extra api --python 3.12
uv run --frozen --no-sync --python 3.12 gnn serve --surface runs --host 127.0.0.1 --port 8000
```

To use the job surface, replace `runs` with `jobs`. Its equivalent module entry
point is:

```bash
uv run --frozen --no-sync --python 3.12 python -m gnn.api.server --host 127.0.0.1 --port 8000
```

In another terminal, check readiness and validate a maintained example without
starting pipeline execution or calling model providers:

```bash
curl --fail http://127.0.0.1:8000/api/v1/health
curl --fail -X POST http://127.0.0.1:8000/api/v1/validate \
  -H 'Content-Type: application/json' \
  -d '{"file_path":"input/gnn_files/basics/static_perception.md"}'
```

## Use an installed package

Install the distribution with its `api` extra in your chosen Python environment;
see the [repository installation guide](../../../README.md). An ordinary wheel
installation requires **`GNN_API_ROOT`**: an absolute, existing workspace whose
path components are directories without symlinks or Windows reparse points.
The default root is accepted only when the package belongs to a real source
checkout. An unset or invalid installed workspace produces a path-validation
error rather than treating the Python installation as writable storage.

This portable launcher creates a local workspace and configures it before
starting the run surface:

```python
import os
from pathlib import Path

workspace = Path("gnn-api-workspace").resolve()
workspace.mkdir(parents=True, exist_ok=True)
os.environ["GNN_API_ROOT"] = str(workspace)

from gnn.api.app import start_server

start_server(host="127.0.0.1", port=8000)
```

Place input files inside that workspace and submit workspace-relative paths.
`GNN_API_ROOT` is operator configuration, not a request option. The operator must
control its directory entries and ancestors. It selects data storage; pipeline
code comes from the installed package's orchestrator.

## Authentication and network binding

Loopback research use permits an unset API key. Set **`GNN_API_KEY`** in the
service environment to require its matching **`X-API-Key`** request header.
Health and documentation routes remain public. Keep the value private; avoid
putting it in URLs, committed launchers or shell history.

The CLI and module starters refuse non-loopback binding without authentication.
For a shared service, configure the key first, then start it:

```bash
: "${GNN_API_KEY:?Set a private API key before sharing the service}"
gnn serve --surface jobs --host 0.0.0.0 --port 8000
```

Calling an ASGI server directly bypasses the starter's bind check, so its
operator must enforce the same address and authentication policy. The shared
API key controls service access; it does not provide per-user filesystem or
process isolation. [auth.py](auth.py) owns the public-route and bind rules.

## Execution, cancellation and evidence

Both run and job requests accept `steps`, `parallel` and `consolidated_steps`;
prerequisites are included in the execution plan and reported total. Omitted
`steps` selects all; explicit `[]` is rejected. JSON flags must be booleans
(strings and numeric coercions are rejected), and unknown request keys fail.
The operator sets `GNN_API_ROOT` to an absolute existing workspace for installed
packages; an ordinary installed API without that workspace refuses filesystem
admission. A recognized source checkout retains its checkout-root default.
Pipeline code always comes from the installed package, independently of this
workspace. Output cannot equal or contain the input target. MCP
`gnn_submit_job` accepts the same flags and an optional `output_dir`, and creates
a pending record; callers explicitly start `execute_job_async` when appropriate.
Job/run status exposes the supervisor's optional `process_cleanup` receipt,
including the observed containment boundary and whether cleanup was verified.
Workspace/code preparation errors finish failed instead of leaving a running job.

Migration from 4.0: remove empty execution selections to request all, replace
coerced flags with actual booleans, and remove previously ignored options. Empty
frozen model selections remain valid skipped work. Renderer registration and
code-generation availability do not certify native execution readiness.

Pipeline work runs in asyncio subprocesses under
[process_supervision.py](process_supervision.py). Each receives a unique
`GNN_RUN_ID`; summaries from another run or an older invocation cannot count as
current evidence. Exit code 0 completes, 1 fails, and 2 completes with warnings
unless `strict=true`. JSON responses and SSE data use `{status,data,error,meta}`;
report downloads retain `text/markdown`.

Job cancellation requests termination; poll status for its cleanup receipt.
A cancelled status requires verified cleanup within the reported boundary.
Cleanup that cannot be verified produces a failed status and explicit receipt.
Supervision uses a bounded graceful-stop interval followed by forced teardown;
normal leader exit also triggers cleanup so inherited pipes cannot hold the
request open indefinitely.

Deleting an active run requests cancellation and waits for terminal state.
Timeouts, ambiguous ownership or unverified cleanup return HTTP 409 and retain
the run record. Workspace roots, their ancestors, the shared default output and
redirected artifact paths are protected from removal. Inspect the deletion
response's artifact-removal result and note rather than assuming every accepted
record deletion removed its files.

Linux and macOS supervise the owned process group and observed descendants.
Windows supervision is **`direct_worker_only`**: it can stop the direct worker
but cannot certify descendant cleanup or tree-wide resource accounting. Explicit
requests for unsupported descendant guarantees are refused before execution.
POSIX directory creation and lease operations pin directory descriptors;
Windows filesystem operations require trusted directory entries. Neither scope
is a sandbox for arbitrary hostile producer code.

The [filesystem and platform boundary contract](../../../docs/security/filesystem_boundaries.md)
owns the adversary model, operation-specific guarantees, cleanup bounds and
native acceptance requirements. See [SPEC.md](SPEC.md) for API invariants and
[AGENTS.md](AGENTS.md) for contributor guidance.

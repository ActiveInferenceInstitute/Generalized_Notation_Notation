# GNN API Module — Agent Scaffolding

## Module Overview

**Purpose**: FastAPI-based REST interface for programmatic pipeline invocation, job management, and tool discovery.
**Pipeline Step**: Infrastructure module (not a numbered step)
**Category**: Infrastructure / API
**Status**: Production Ready
**Version**: [pyproject.toml](../../../pyproject.toml) (canonical)
**Last Updated**: 2026-11-11

The `api` module provides two independent local FastAPI applications for the GNN
pipeline: `gnn.api.app` serves runs and evidence, `gnn.api.server` serves jobs and
tool discovery. Neither is a numbered pipeline step. [README.md](README.md) owns
launch commands and implemented routes; [SPEC.md](SPEC.md) owns the contract
boundaries and invariants.

## Architecture

| File | Responsibility |
|------|----------------|
| [app.py](app.py) | Run API started by `gnn serve --surface runs`: `POST /api/v1/run`, hash-addressed runs, Markdown report, SSE stream, health |
| [server.py](server.py) | Job API: `POST /api/v1/process`, UUID jobs, cancellation, pipeline tools |
| [parity.py](parity.py) | Shared CLI-parity routes registered on both factories |
| [models.py](models.py) | Pydantic requests/responses for both surfaces, including `RunRequest`, `RunStatus`, `RunHealthResponse` |
| [responses.py](responses.py) | Canonical `{status,data,error,meta}` envelopes and exception handlers |
| [processor.py](processor.py) | In-memory job and run stores; `create_job`, `get_job`, `cancel_job`, `list_jobs`, `get_pipeline_tools`, `delete_run` |
| [pipeline_runner.py](pipeline_runner.py) | Shared pipeline argv construction, step planning and summary ingestion |
| [process_supervision.py](process_supervision.py) | Bounded owned-process supervision and cleanup receipts |
| [auth.py](auth.py) | Optional `GNN_API_KEY` authentication and non-loopback bind guard |
| [rate_limit.py](rate_limit.py) | Per-client request rate limiting |
| [path_utils.py](path_utils.py) | Symlink-safe repository path resolution |
| [mcp.py](mcp.py) | MCP tool wrappers, executable registration and manifest export |
| [__init__.py](__init__.py) | Availability/metadata; `MODULE_VERSION` re-exports `gnn.__version__` |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/v1/process | Submit pipeline job |
| GET | /api/v1/jobs/{id} | Poll job status |
| DELETE | /api/v1/jobs/{id} | Cancel job |
| GET | /api/v1/jobs | List recent jobs |
| GET | /api/v1/tools | List pipeline steps |
| POST | /api/v1/tools/{step} | Invoke single step |
| GET | /api/v1/health | Health check |

The module exposes two independent FastAPI apps. The table above is the
`gnn.api.server` job/tool surface; `gnn.api.app` (started by `gnn serve`) exposes
`POST /api/v1/run`, `GET /api/v1/runs`, `GET /api/v1/runs/{hash}`,
`GET /api/v1/runs/{hash}/report` (Markdown), `GET /api/v1/runs/{hash}/stream`
(SSE), and `GET /api/v1/health`. Both share the canonical `{status,data,error,meta}` JSON envelope.
ADR 0049
([docs/decisions/0049-dual-async-job-surfaces.md](../../../docs/decisions/0049-dual-async-job-surfaces.md))
records this dual-surface split as deliberate: both factories are canonical
for their consumer class, and coherence is enforced by the shared
`processor` stores plus the per-factory route tables pinned in
`tests/api/test_api_parity.py`.

Beyond those exclusive routes, both factories share the 12-route CLI-parity
surface from `gnn.api.parity.register_parity_routes` (`/api/v1/validate`,
`/parse`, `/extract`, `/render`, `/graph`, `/templates`, `/pull`, `/models`,
`/preflight`, `/report`, and the verify-only `/api/v1/reproduce` — which
resolves and verifies an indexed run and returns the reconstructed
configuration without executing it).

## Installation

The API module requires the optional `api` extra. Run these commands from the
repository root:

```bash
UV_PYTHON=3.12 uv sync --frozen --extra api
UV_PYTHON=3.12 uv run --frozen --no-sync gnn serve --surface runs --host 127.0.0.1 --port 8000
```

Replace `runs` with `jobs` for the job surface. Its equivalent module entry
point is `python -m gnn.api.server --host 127.0.0.1 --port 8000`. Direct ASGI
launchers bypass the starter's bind and authentication checks and must enforce
the same policy themselves.

## Design Decisions

- **Optional API-key authentication** (`api/auth.py`): unauthenticated for
  loopback research use by default; set `GNN_API_KEY` to require a matching
  `X-API-Key` header on every route outside the public health/docs surface
  (constant-time `hmac.compare_digest`). Non-loopback binds are refused unless
  auth is enabled or `GNN_ALLOW_INSECURE_BIND=1` (RED_TEAM V-04).
- **In-memory jobs**: No database required. Jobs lost on restart.
- **AsyncIO execution**: Non-blocking pipeline runs via asyncio subprocess.
- **Background tasks**: FastAPI BackgroundTasks for fire-and-forget job execution.
- **CORS**: Allows localhost origins for browser-based access.
- **Symlink-safe path validation**: `path_utils.resolve_repo_path` rejects any
  symlink or reparse component before resolving (RED_TEAM V-05). POSIX directory
  creation is descriptor-relative; later pathname consumers require trusted
  directory entries. Installed services set the operator-only `GNN_API_ROOT`
  to an absolute, existing, nonredirected scratch workspace. The precise
  guarantees and Windows limits are in
  [filesystem boundaries](../../../docs/security/filesystem_boundaries.md).
- **Sanitized error responses**: job failure tails redact the repository root
  and absolute paths before returning to callers (RED_TEAM V-09).
- **Canonical JSON envelope**: successful and failed JSON responses use exactly
  `{status, data, error, meta}`. Validation, HTTP, authentication, rate-limit,
  and unexpected exceptions pass through the same contract. The Markdown
  report download remains `text/markdown`; SSE event payloads use the envelope.

## Integration with Pipeline

This module is NOT a numbered pipeline step. It is an optional service module
that wraps the pipeline for API access. Import it independently:

```python
from gnn.api.processor import create_job, get_job, cancel_job, list_jobs, get_pipeline_tools
```

## Agent Guidance

When working with this module:
1. The `processor.py` contains job lifecycle logic -- extend it for persistence needs
2. The `models.py` is the API contract -- update schemas and tests together
3. Add new endpoints in `server.py` (job/tool surface) or `app.py` (run/runs surface) following the canonical envelope pattern

Pipeline-submitting requests share strict file/step worker admission through
`gnn.pipeline.admission.validate_worker_count`: `file_workers=1`,
`step_workers=None`, integer bounds 1–256, no Boolean/numeric coercion and null
only for automatic step limits. Preserve those requested values in job/run
records, run identity, responses, MCP schemas and the shared subprocess argv
builder. `parallel` enables concurrent tiers; limits do not establish actual
concurrency or speedup. Step-12 execution worker configuration is not admitted
by these API requests.

## Maintainer rules

1. Keep both factories explicit: their exclusive route tables and response
   models differ despite sharing the `/api/v1` prefix. `gnn serve` dispatches to
   `app.py`; `python -m gnn.api.server` starts `server.py`.
2. Import shared requests/responses from `gnn.api.models` and register the
   shared surface through `gnn.api.parity`; do not fork a model or envelope per
   factory.
3. `RunRequest.config` and `ToolRequest.kwargs` are accepted but must be empty;
   non-empty values are rejected because neither runner forwards them.
4. Both stores are process-local and lost on restart. An MCP submission creates
   a pending job without scheduling execution; a separate HTTP process has a
   different store. Callers start `execute_job_async` explicitly.
5. Preserve repository-local path validation and loopback defaults. A configured
   `GNN_API_KEY` is the only authentication boundary; neither application
   provides per-user filesystem or process isolation.
6. Update generated OpenAPI expectations and the API contract tests together
   when changing routes, models or envelopes. No cross-release HTTP
   compatibility contract is declared.

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API

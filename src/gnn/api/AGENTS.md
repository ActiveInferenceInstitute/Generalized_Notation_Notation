# GNN API Module — Agent Scaffolding

## Module Overview

**Purpose**: FastAPI-based REST interface for programmatic pipeline invocation, job management, and tool discovery.
**Pipeline Step**: Infrastructure module (not a numbered step)
**Category**: Infrastructure / API
**Status**: Production Ready
**Version**: [pyproject.toml](../../../pyproject.toml) (canonical)
**Last Updated**: 2026-09-24

The `api` module provides a FastAPI-based REST interface for the GNN processing pipeline.
It enables programmatic pipeline invocation, job management, and tool discovery without
requiring direct CLI access.

## Architecture

```
src/gnn/api/
  __init__.py    -- Module metadata, availability checks
  app.py         -- FastAPI "run" surface (run/runs/report/stream/health); `gnn serve` entry point
  server.py      -- FastAPI "job/tool" surface (process/jobs/tools/health)
  models.py      -- Pydantic request/response models (API contract)
  responses.py   -- Canonical envelopes and exception handlers
  processor.py   -- In-memory job manager (create_job/get_job/cancel_job/list_jobs/get_pipeline_tools)
  auth.py        -- Optional API-key authentication (GNN_API_KEY)
  path_utils.py  -- Symlink-safe repo path resolution
  rate_limit.py  -- Request rate limiting
  mcp.py         -- MCP tool registration manifest
  AGENTS.md      -- This file
```

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

The API module requires optional dependencies:

```bash
uv sync --extra api
# then:
python -m gnn.api.server
# or:
uvicorn gnn.api.server:app --reload
```

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


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API

# Typed models and API consumer contract

The distribution is `generalized-notation-notation`; [pyproject.toml](../../pyproject.toml)
is authoritative for its version (`5.0.3`), its Python floor (`>=3.11,<3.15`) and its
only console script (`gnn = gnn.cli:main`). `gnn.__version__` mirrors the pyproject
version, and `gnn.api.MODULE_VERSION` re-exports it. Other version strings in the tree
— for example the LSP fallback's `serverInfo` block — are separate identities and do
not identify the distribution release or either HTTP application below.

## Import setup and canonical declarations

The wheel packages `src/gnn` as the top-level `gnn` package
(`[tool.hatch.build.targets.wheel]` in [pyproject.toml](../../pyproject.toml)), so an
installed consumer imports `gnn.*` directly. A source checkout that runs without
installing needs that checkout's `src` root on `PYTHONPATH`; [run.sh](../../run.sh)
exports exactly that for every gate.

From the checkout root after syncing its environment, this import probe uses the
absolute source root and does not rely on the current directory for imports:

```bash
PYTHONPATH="$PWD/src" uv run --frozen --no-sync python -c 'from gnn.types import ParsedGNN; print(ParsedGNN.__module__)'
```

It prints `gnn.types.definitions`, the module that owns the dataclass.

| Declaration | Owner | Consumer note |
| --- | --- | --- |
| `ParsedGNN` | [`gnn/types/definitions.py`](../../src/gnn/types/definitions.py) | Re-exported by `gnn.types`. |
| `ParseResult` (parser result) | [`gnn/parsers/common.py`](../../src/gnn/parsers/common.py) | Returned by the `gnn.parsers` parse entry points. |
| `ParseResult` (typed model) | `gnn.types.definitions` | A separate dataclass; re-exported as `gnn.types.ParseResult`. Do not substitute one for the other. |
| `GNNFormat`, `GNNInternalRepresentation` | `gnn.parsers.common` | Single authoritative definitions, re-exported by `gnn.types`; import either surface, not a third copy. |

The two `ParseResult` declarations are a known duplication. Consumers must import the
one their producer returns; unifying them is open work, not a shipped behavior.

## HTTP applications

`gnn serve --surface` selects the factory. Both factories are in the `api` extra
(`uv sync --extra api`) and both default to `127.0.0.1:8000`
(`gnn.api.DEFAULT_API_HOST` / `DEFAULT_API_PORT`).

| Surface | Factory | What it serves |
| --- | --- | --- |
| `runs` (default) | `gnn.api.app:create_app` via `start_server` | Pipeline-as-a-service runs over the shared pipeline entry points. |
| `jobs` | `gnn.api.server:create_app` via `run_server` | Job submission, status and cancellation. |
| `both` | `runs` on the main thread, `jobs` on port+1 | Both surfaces in one process. |
| `website` | `gnn.website.serve` | Static server over the generated output tree (port 8090). |

Both FastAPI factories call `install_exception_handlers` and `register_parity_routes`
from [`gnn/api/responses.py`](../../src/gnn/api/responses.py), so they share one
envelope and one CLI-parity route set. Their remaining routes differ; the
[API guide](../../src/gnn/api/README.md) is the route owner.

The canonical envelope for every non-streaming endpoint is
`{status, data, error, meta}`: `status` is `success` or `error`, `error` carries
`{code, message, details}`, and `meta` always includes a timezone-aware `timestamp`.
Request-validation failures return HTTP 422 with code `validation_error`, HTTP
exceptions map to stable codes, and unhandled exceptions return HTTP 500 with
`internal_error`. Binding to a non-loopback host requires `GNN_API_KEY`, or an explicit
`GNN_ALLOW_INSECURE_BIND=1`.

## Version limits

- Python `>=3.11,<3.15`, declared once in [pyproject.toml](../../pyproject.toml).
- uv `>=0.13.6,<0.14` ([uv.toml](../../uv.toml) `required-version`); CI and the
  Dockerfile install the exact `0.13.6`.
- The HTTP surfaces are an optional group; install the `api` extra before importing
  them. Importing a class does not establish native backend readiness.

See the [interoperability guide](interop.md) for the cross-axis map and
[benchmark contracts](interop_research.md) for producer entry points and result shapes.

# Typed models and API consumer contract

The distribution is `generalized-notation-notation`; [pyproject.toml](../../pyproject.toml)
is authoritative for its version (`5.1.0`), its Python floor (`>=3.11,<3.15`) and its
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
| `ParseResult` | [`gnn/parsers/common.py`](../../src/gnn/parsers/common.py) | Single authoritative definition, returned by the `gnn.parsers` entry points and re-exported by `gnn.types` and `gnn.types.definitions`; import either surface. |
| `GNNFormat`, `GNNInternalRepresentation` | `gnn.parsers.common` | Single authoritative definitions, re-exported by `gnn.types`; import either surface, not a third copy. |

`ParseResult` has one declaration. `ParseResult()` without a model reports
`success=False`; `ParseResult(model=...)` reports `success=True` unless the caller
passes `success` explicitly. Importing it from `gnn.types` or `gnn.parsers.common`
yields the same class, so neither surface can silently change the contract.

## Validation depth

`GNNParsingSystem.parse_file` and `GNNParsingSystem.parse_string` preserve the
format parser's `ParseResult.success`. With `strict_validation=True`, either
entry point attaches `ParseResult.validation_result` after a successful parse.
Inspect its `is_valid`, `errors`, `warnings` and `metadata["available"]` to
distinguish an invalid model from a check that could not run. These diagnostics
cover the parser's internal model semantics. Normative Markdown section
compliance is a separate [importer check](interop.md#importer-conformance).

For a parsed object, `gnn.schema.check_gnn_object` returns `mode` and `errors`.
`draft-2020-12` runs the full JSON Schema. `required-keys` checks that the root
keys are present when the optional `jsonschema` library is unavailable.
Consumers that require field-type validation should check for the full mode
before accepting an empty error list. `validate_gnn_object` still returns an
error list for callers using the existing API.

`gnn parse --json` and `POST /api/v1/parse` distinguish graph structure
from optional probabilistic parameters. A valid interface declaration can
return variables and connections without `pomdp`. That key requires at
least one parsed parameter with `InitialParameterization` provenance.
A `pomdp` object can still be incomplete. Use `gnn extract --strict` and
check the intended backend before treating a model as executable.

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

Both FastAPI factories call `install_exception_handlers` from
[`gnn/api/responses.py`](../../src/gnn/api/responses.py) and `register_parity_routes`
from [`gnn/api/parity.py`](../../src/gnn/api/parity.py), so they share one
envelope and one CLI-parity route set. Their remaining routes differ; the
[API guide](../../src/gnn/api/README.md) is the route owner.

The canonical envelope for every non-streaming endpoint is
`{status, data, error, meta}`: `status` is `success` or `error`, `error` carries
`{code, message, details}`, and `meta` always includes a timezone-aware `timestamp`.
Request-validation failures return HTTP 422 with code `validation_error`, HTTP
exceptions map to stable codes, and unhandled exceptions return HTTP 500 with
`internal_error`. Binding to a non-loopback host requires `GNN_API_KEY`, or an explicit
`GNN_ALLOW_INSECURE_BIND=1`.

## Shared operations and dependency direction

The CLI, the HTTP parity routes and the template MCP tools are transports over
one implementation in `gnn.operations`, which imports neither transport:

| Operation module | Owns | Transports |
| --- | --- | --- |
| [`gnn.operations.commands`](../../src/gnn/operations/commands.py) | validate check chain, parse payload, extract defect verdicts, preflight severities, atomic `PIPELINE_REPORT.md` publication | `gnn validate/parse/extract/preflight/report`; `POST /api/v1/{validate,parse,extract,preflight,report}` |
| [`gnn.operations.templates`](../../src/gnn/operations/templates.py) | template index, checksums, symlink-refusing atomic pull | `gnn templates`, `gnn pull`; `GET /api/v1/templates[/{name}]`, `POST /api/v1/pull`; template MCP tools |

Each adapter maps the shared outcome onto its own contract only: CLI exit codes
`0`/`2`/`1`; HTTP `200` with `data.exit_code` `0` or `2`, `400` for a would-be
exit 2 under `strict=True` or a `PathValidationError`, `404` for a missing path,
`409` for a template destination collision or symlink. `gnn.cli.commands` and
`gnn.cli.templates` remain re-export facades, and template records keep their
`package://gnn.cli/...` sources because the template data still ships in
`src/gnn/cli`. Import `gnn.operations` in new code; the package root is lazy and
loads no submodule until a name is used.

Dependency direction is pinned by
[`tests/architecture/test_dependency_direction.py`](../../tests/architecture/test_dependency_direction.py):

- `gnn.api` never imports `gnn.cli`; `gnn.operations` imports neither.
- `gnn.pipeline` never imports `gnn.report`, `gnn.cli` or `gnn.api`. Ledger and
  `PIPELINE_REPORT.md` renderers live with their pipeline producers
  (`gnn.pipeline.cross_framework_reliability`, `gnn.pipeline.semantic_fidelity`,
  `gnn.pipeline.summary_wiring`); the same-named `gnn.report` modules re-export
  them with byte-identical Markdown.
- In a fresh interpreter, importing `gnn`, `gnn.parsers`, `gnn.types`,
  `gnn.importers` or `gnn.operations` leaves `gnn.report`, `gnn.execute`,
  `gnn.cli`, `gnn.api`, `jax`, `matplotlib` and `pandas` unloaded, and building
  either FastAPI app leaves `gnn.cli` unloaded.

The static checks resolve relative, function-scope, `TYPE_CHECKING` and literal
`importlib.import_module` imports, so moving an edge into a function or a
string does not hide it.

## Version limits

- Python `>=3.11,<3.15`, declared once in [pyproject.toml](../../pyproject.toml).
- uv `>=0.13.6,<0.14` ([uv.toml](../../uv.toml) `required-version`); CI and the
  Dockerfile install the exact `0.13.6`.
- The HTTP surfaces are an optional group; install the `api` extra before importing
  them. Importing a class does not establish native backend readiness.

See the [interoperability guide](interop.md) for the cross-axis map and
[benchmark contracts](interop_research.md) for producer entry points and result shapes.

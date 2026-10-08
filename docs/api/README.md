# GNN public APIs

Use the installed `gnn.*` package. The package's
[public exports](../../src/gnn/__init__.py), module specifications and
[live step registry](../../src/gnn/pipeline/step_registry.py) define the actual
interfaces. The [API reference](comprehensive_api_reference.md) maps those
owners; runnable examples require their declared dependencies and real inputs.

## Discovery, parsing and scientific extraction

Lightweight discovery and parsing inspect authored files. They are distinct
from complete POMDP extraction, multi-format processing and native inference.

```python
from pathlib import Path
from gnn import GNNFormat, GNNParsingSystem, discover_gnn_files, parse_gnn_file

source = Path("input/gnn_files/discrete/two_state_bistable.md")
paths = discover_gnn_files(source.parent, recursive=True)
assert source in paths
info = parse_gnn_file(source)
assert info["success"] and info["structure_info"]["has_variables"]
parsed = GNNParsingSystem().parse_file(source, format_hint=GNNFormat.MARKDOWN)
assert parsed.success and parsed.model is not None
```

For the scientific extractor and declared syntax validation:

```bash
UV_PYTHON=3.12 uv sync --frozen --extra dev
uv run --frozen --no-sync gnn validate input/gnn_files/discrete/two_state_bistable.md
uv run --frozen --no-sync gnn extract input/gnn_files/discrete/two_state_bistable.md
```

See the [model contract](../../src/gnn/SPEC.md),
[syntax](../gnn/reference/gnn_syntax.md) and
[backend support](../execution/FRAMEWORK_AVAILABILITY.md). A parse success or
compatible tensor shape does not establish a native inference contract.

## Pipeline and rendering

The [pipeline API](../../src/gnn/pipeline/README.md) exposes `run_pipeline`,
`PipelineOrchestrator`, `PipelineConfig` and step execution. Numbered scripts
and the CLI delegate to shared owners. Registered executing steps and backend
options require admission; discovery metadata is a broader inventory.

```bash
uv run --frozen --no-sync python src/gnn/main.py \
  --target-dir input/gnn_files/basics \
  --output-dir /tmp/gnn-api-example --only-steps "3,5" --verbose
```

Read the summary: exit 0 is success, 2 completed work with warnings, and 1
failure. A run freezes its selected sources and resolved configuration. Empty
model selections write no model artifacts; inherited output cannot establish
current completion. See [run ownership](../development/run_ownership_migration.md).

[Rendering](../../src/gnn/render/README.md) and
[execution](../../src/gnn/execute/README.md) are separate. Derive backend
metadata from [frameworks.py](../../src/gnn/frameworks.py) and the
[registry](../../src/gnn/render/framework_registry.py). Experimental THRML and
cpomdp require explicit selection. Unsupported models, missing dependencies,
invalid options, backend failures and failed cleanup retain distinct diagnoses.

## REST and MCP

The local REST service is documented by its
[API owner](../../src/gnn/api/README.md) and generated OpenAPI schema. Provision
`api` explicitly. Authentication, workspace admission, owned process cleanup
and artifact deletion follow their actual runtime contracts. Do not infer
security or platform containment from a successful HTTP response alone.

[MCP documentation](../gnn/mcp/README.md) describes tool registration,
transport and real dispatch. Inspect current tools and their schemas:

```bash
uv run --frozen --no-sync gnn mcp list
uv run --frozen --no-sync gnn mcp info cli.health
```

`gnn health` reports environment diagnoses; the Python
`gnn.execute.collect_doctor_report` API and MCP `get_doctor_report` tool provide
structured backend readiness. Importability and tool registration are separate
from accepted native execution.

## Security

Read the [API authentication and bind owner](../../src/gnn/api/auth.py) and
[filesystem boundaries](../security/filesystem_boundaries.md) before exposing
the service. Installed deployments configure the operator's trusted
`GNN_API_ROOT` explicitly; request paths cannot change it. Output leases are
advisory, and native cleanup receipts declare the actual process boundary.
Unverified cleanup retains run records and artifacts instead of claiming
successful deletion. Preserve provider secrets and use the documented local
service defaults.

## Verification and navigation

Use [tests/AGENTS](../../tests/AGENTS.md), the relevant consumer suites and
[current workflows](../../.github/workflows/README.md). Retain exact-source
outcomes, optional skips, native platform limits and finite visual review scope.
The [roadmap](../../TO-DO.md) defines remaining coverage and scientific work.

- [Documentation hub](../README.md)
- [Pipeline architecture](../../ARCHITECTURE.md)
- [Configuration guide](../configuration/README.md)
- [Security boundaries](../security/README.md)
- [API agent guide](AGENTS.md) and [documentation specification](SPEC.md)

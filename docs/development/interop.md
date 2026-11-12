# GNN interoperability guide

`generalized-notation-notation` is the installed distribution. Its version, Python
floor and console script are declared in [pyproject.toml](../../pyproject.toml)
(currently `5.0.3`, Python `>=3.11,<3.15`, and `gnn = gnn.cli:main`). This guide
describes existing contracts and explicitly identifies surfaces that cannot yet be
consumed through a common interface. Runtime behavior is unchanged by these
declarations.

| Axis | Canonical surface | Consumer guidance |
| --- | --- | --- |
| Typed data and models | [Typed/API contracts](interop_api.md); [`gnn.types`](../../src/gnn/types/); [parser common types](../../src/gnn/parsers/common.py) | Import the declaration for the specific layer you consume. The installed distribution exposes the `gnn` package; a source checkout needs that checkout's `src` root on `PYTHONPATH`. Two `ParseResult` dataclasses exist (`gnn.parsers.common` and `gnn.types.definitions`); do not mix their identities, and treat unification as open work. |
| Benchmarks | [Research contracts](interop_research.md); [scaling harness](../../scripts/experiments/run_pymdp_gnn_scaling_analysis.py); [performance tests](../../tests/pipeline/test_pipeline_performance.py) | Use each producer's exact entry point and result shape. There is no shared benchmark-worker adapter or common result envelope across projects. |
| Documentation | [Root index](../../INDEX.md); [documentation index](../INDEX.md); [guided tour](../START_HERE.md) | Maintained documentation lives under `docs/`. The guides link to the same contract owners and the [canonical backlog](../../TO-DO.md). |
| API surface | [Typed/API contracts](interop_api.md); [CLI](../../src/gnn/cli/README.md); [HTTP applications](../../src/gnn/api/README.md); [MCP](../../src/gnn/mcp/README.md) | `gnn = gnn.cli:main` is the only packaged console script. Select the HTTP surface explicitly: `gnn serve --surface runs` starts `gnn.api.app`, `--surface jobs` starts `gnn.api.server`, and `--surface both` runs the jobs surface on port+1. Both factories install the same `gnn.api.responses` envelopes and report `gnn.__version__` as their version. |

## Ownership boundaries

GNN declares its own typed models, artifacts and identifiers. It does not copy
another project's catalogues, invent identifiers inside them, or declare a runtime
bridge merely because an owner exists elsewhere. The linked guides keep GNN's local
types and artifacts distinct from surfaces owned by other projects.

Integration gaps are tracked as code-grounded backlog rows with acceptance probes in
[TO-DO.md](../../TO-DO.md). There is also no live container deployment, sensor stream
or cluster operator behind the v3 orchestration declarations: the
[v3 contracts](../pipeline/v3_orchestration.md) produce and validate plans and durable
manifests only.

## Verification

Run `./run.sh gates` from the repository root for the local aggregate gate; it mirrors
the CI `quality` job plus the unit-suite selection and coverage floor declared in
[run.sh](../../run.sh). Maintained documentation links and anchors are checked by:

```bash
uv run --frozen --no-sync python docs/development/docs_audit.py --strict --check-anchors --no-write
```

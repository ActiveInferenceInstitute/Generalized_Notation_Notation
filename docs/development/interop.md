# GNN interoperability guide

`generalized-notation-notation` is the installed distribution. Its version, Python
floor and console script are declared in [pyproject.toml](../../pyproject.toml)
(currently `5.0.3`, Python `>=3.11,<3.15`, and `gnn = gnn.cli:main`). This guide
describes existing contracts and explicitly identifies surfaces that cannot yet be
consumed through a common interface. Runtime behavior is unchanged by these
declarations.

| Axis | Canonical surface | Consumer guidance |
| --- | --- | --- |
| Typed data and models | [Typed/API contracts](interop_api.md); [`gnn.types`](../../src/gnn/types/); [parser common types](../../src/gnn/parsers/common.py) | Import the declaration for the specific layer you consume. The installed distribution exposes the `gnn` package; a source checkout needs that checkout's `src` root on `PYTHONPATH`. `ParseResult` has one declaration (`gnn.parsers.common`); `gnn.types` re-exports that same class, so every public import path shares its success semantics and attributes. |
| Benchmarks | [Research contracts](interop_research.md); [scaling harness](../../scripts/experiments/run_pymdp_gnn_scaling_analysis.py); [performance tests](../../tests/pipeline/test_pipeline_performance.py) | Use each producer's exact entry point and result shape. There is no shared benchmark-worker adapter or common result envelope across projects. |
| Documentation | [Root index](../../INDEX.md); [documentation index](../INDEX.md); [guided tour](../START_HERE.md) | Maintained documentation lives under `docs/`. The guides link to the same contract owners and the [canonical backlog](../../TO-DO.md). |
| API surface | [Typed/API contracts](interop_api.md); [CLI](../../src/gnn/cli/README.md); [HTTP applications](../../src/gnn/api/README.md); [MCP](../../src/gnn/mcp/README.md) | `gnn = gnn.cli:main` is the only packaged console script. Select the HTTP surface explicitly: `gnn serve --surface runs` starts `gnn.api.app`, `--surface jobs` starts `gnn.api.server`, and `--surface both` runs the jobs surface on port+1. Both factories install the same `gnn.api.responses` envelopes and report `gnn.__version__` as their version. |

## Importer conformance

A repository that applies GNN records the upstream revision it consumes in a pin
file and keeps `*.gnn.md` specifications. The installed distribution ships the
reader for both, so a consumer checks its own checkout with GNN's own code:

```bash
python -m gnn.importers check --repo . --expected-commit 8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1
```

It normalizes the pin shapes in use (`commit`/`revision`, `tag`/`release`, the
four file locations), reads the specifications the pin declares or discovers
them, and validates each at the compliance level — the normative section
contract, not just a parse. Findings carry stable codes (`pin-missing`,
`pin-commit-mismatch`, `pin-tag-mismatch`, `spec-missing`, `spec-invalid`,
`pin-schema-unversioned`) and `--json` emits the whole report. The canonical pin
schema is `gnn-importer-pin/v1`; pins written before the schema are read and reported, never
rewritten.

The estate-wide view is [the downstream registry](../downstream/README.md) and
its machine-readable mirror
[importers.json](../downstream/importers.json). `scripts/check_importer_registry.py`
keeps those two forms one artifact offline; `--online` re-reads origin and is
report-only.

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

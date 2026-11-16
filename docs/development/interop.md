# GNN interoperability guide

`generalized-notation-notation` is the installed distribution. Its version, Python
floor and console script are declared in [pyproject.toml](../../pyproject.toml)
(currently `5.1.0`, Python `>=3.11,<3.15`, and `gnn = gnn.cli:main`). This guide
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
python -m gnn.importers check --repo . --expected-commit 75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf
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

### Frozen pre-import resources

The standalone [bootstrap](../../interop/gnn_bootstrap.py) and
[v5.1.0 source manifest](../../interop/v5.1.0-source-manifest.json) are
independently qualified, byte-preserved consumer resources, not a replacement
for the current installed `gnn.importers` module. Fetch them from the same
reviewed immutable source commit, not a moving branch or an invented release
asset. This source publication neither changes the v5.1.0 tag/release nor
rebuilds its package.

| Resource | SHA-256 |
| --- | --- |
| `interop/gnn_bootstrap.py` | `f638d4c01fe7de31e43d793e59bf382d0c3d1a82f365c59238334a2b40adfb37` |
| `interop/v5.1.0-source-manifest.json` | `43b33c0b2b40386afa3f306fec4cc13bc1fd7c3b570fbf11d5008bad5c8e6b51` |

The manifest binds all 1,232 package files at
`75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`; do not regenerate it from the
installation being checked or change a consumer pin to match a different
resource. Verify the independently trusted bootstrap digest, then compile and
execute those exact checked bytes before **any** `gnn` import. Bind the frozen
manifest bytes and digest, and keep `qualified_source` around the actual
parser/verdict imports and execution, not just an earlier static check.

The retained qualification covers complete Git-source content and guarded
syntax verdicts. It does not establish installed-wheel identity, dependency
integrity, scientific semantics, publication consent or a native simulation
result. Installed mode still requires its genuine matching Git origin and
complete content; an installed-admission failure does not authorize a source
fallback. Consumers retain their own runtime and acceptance obligations.

Known API limitation of these frozen bytes: when `check_importer()` runs inside
an externally managed `qualified_source()` context, its top-level `qualified`
flag remains false after successful context exit. The shared guard receipt
does finalize after exit. The retained qualification demonstrates direct parser
execution within the guard, not finalized qualification through this nested-report
API. A code repair requires a separately reviewed resource identity; do not
replace these immutable bytes or silently rebind their digest.
These GNN resources retain the repository's
[CC BY-NC-SA 4.0 license](../../LICENSE.md), attribution and noncommercial/
share-alike terms; this publication does not grant a different license.

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

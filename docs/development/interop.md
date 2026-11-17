# GNN interoperability guide

`generalized-notation-notation` is the installed distribution. Its version, Python
floor and console script are declared in [pyproject.toml](../../pyproject.toml)
(currently `5.1.0`, Python `>=3.12,<3.15`, and `gnn = gnn.cli:main`). This guide
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
Pins that omit a repository identity remain supported. A present, non-null
repository declaration that fails normalization is a `pin-unreadable` error,
never an omitted value that can bypass the repository comparison.

The estate-wide view is [the downstream registry](../downstream/README.md) and
its machine-readable mirror
[importers.json](../downstream/importers.json). `scripts/check_importer_registry.py`
keeps those two forms one artifact offline; `--online` re-reads origin and is
report-only.

### Frozen pre-import resources

The standalone [bootstrap](../../interop/gnn_bootstrap.py),
[v5.0.3 source manifest](../../interop/v5.0.3-source-manifest.json) and
[v5.1.0 source manifest](../../interop/v5.1.0-source-manifest.json) are
consumer source-guard resources, not a replacement for the canonical installed
`gnn.importers` module. The manifests retain their originally qualified bytes;
the current bootstrap strictly parses GitHub origins. Use a qualified immutable
source commit, never a moving branch or invented release asset.

| Resource | SHA-256 |
| --- | --- |
| `interop/gnn_bootstrap.py` | `e04e9114c860a622f26ee8d323ac44b77c6484f5a29bdc7654fb298a9b58c3b2` |
| `interop/v5.0.3-source-manifest.json` | `17a35bb2114e76a38119128d290d12377cb641652d70788000482366dca0a14a` |
| `interop/v5.1.0-source-manifest.json` | `43b33c0b2b40386afa3f306fec4cc13bc1fd7c3b570fbf11d5008bad5c8e6b51` |

Bare `org/repo` identities are valid pin metadata, not installed VCS URLs.
Installed origin records must name an exact GitHub HTTP(S), Git SSH or SCP
origin, with the complete repository path and no credentials, port, query,
fragment or extra path. HTTP metadata matching is not transport authentication.
Historical delivery `66d491f86895443cf588011cc6f40c37b8bae8fd` retains the
separate `f638d4c01fe7de31e43d793e59bf382d0c3d1a82f365c59238334a2b40adfb37`
bootstrap and its observed direct-source/genuine-GitHub-URL proof scope; these
immutable bytes and the two source manifests are not rewritten or rebaselined.

The v5.0.3 manifest binds all 1,225 package files at
`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`; the v5.1.0 manifest binds all
1,232 package files at `75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`.
Select the manifest matching the consumer's original pin; in particular, the
v5.1.0 manifest does not admit a v5.0.3 consumer. Do not regenerate either
manifest from the installation being checked or change a pin to fit a resource.
A manifest is produced only by the bootstrap's `source-manifest` command
from an independently qualified exact-revision package: `--package-root`,
its `--commit` and each guarded `--verdict-module`.
Verify the independently trusted bootstrap digest, then compile and
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

### Local registry validation

`scripts/check_importer_registry.py --repo` (with `--name` giving that
checkout's registry row) delegates to the frozen bootstrap
in a fresh target process selected with `--importer-python`. It compares the
checkout with that member row's own pin, not the registry's general upstream
release. Supply `--source-manifest` and its independently trusted
`--source-manifest-sha256`; missing inputs refuse validation rather than using
the registry process's already imported GNN package.

The default `--origin-mode installed` requires genuine installed Git-origin
and complete-content admission. Explicit `--origin-mode source` additionally
requires `--package-root`; that deliberate source selection never serves as
an automatic fallback from failed installed admission. A package-specific
finder selects only `gnn`, without exposing its unadmitted parent directory
to unrelated or deferred standard-library imports. The child checks the
bootstrap digest and executes those exact bytes before importing GNN. A
nonempty successful verdict must carry finalized qualification. Child verdicts
must agree with their error findings; malformed replies and changed source
fail closed.

The canonical pin reader also refuses duplicate identity keys, conflicting
aliases and unsafe or symlinked specification paths. An explicit empty
`spec_files` list remains authoritative; only an absent declaration enables
discovery. Neither these checks nor registry declarations establish scientific
semantics or current hosted success.

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

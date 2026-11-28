# Installed-package and native platform acceptance

The [installed-platform workflow](../../.github/workflows/installed-platforms.yml)
checks ordinary wheel and sdist installations outside the source checkout.
Every event runs Linux/Python 3.12, 3.13, and 3.14, macOS/Python 3.14, and
Windows/Python 3.14.
A supported release must have successful native receipts for its exact source;
a resolvable wheel plan or classifier alone does not establish support.
The ordinary source test suite has a separate Python 3.12–3.14 matrix, and
comprehensive native coverage runs Python 3.12–3.14.

## Version 4.1.0 installed evidence

The exact ordinary wheel/source/lock/environment and native results belong to
this version's [GitHub release](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/releases/tag/v4.1.0). Each lane runs outside the
checkout with complete nonzero selected cases and zero failures/errors/skips.
Full source coverage remains a separate three-Python acceptance record.

| Native lane | Source-bound ordinary-wheel acceptance record |
| --- | --- |
| Linux / Python 3.11 | Python 3.11.16: 8 THRML and 27 boundary cases passed, zero failures/errors/skips; [ordinary installed-wheel job](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/actions/runs/37761366047/job/113258292335) at `6be1506cac5957973dff8edeaf77e84e2e7ceb46` |
| Linux / Python 3.12 | Python 3.12.14: 8 THRML and 27 boundary cases passed, zero failures/errors/skips; [ordinary installed-wheel job](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/actions/runs/37761366047/job/113258292056) at `6be1506cac5957973dff8edeaf77e84e2e7ceb46` |
| Linux / Python 3.14 | Python 3.14.8: 8 THRML and 27 boundary cases passed, zero failures/errors/skips; [ordinary installed-wheel job](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/actions/runs/37761366047/job/113258292476) at `6be1506cac5957973dff8edeaf77e84e2e7ceb46` |
| macOS / Python 3.14 | Python 3.14.7: 8 THRML and 27 boundary cases passed, zero failures/errors/skips; [ordinary installed-wheel job](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/actions/runs/37761366047/job/113258292540) at `6be1506cac5957973dff8edeaf77e84e2e7ceb46` |
| Windows / Python 3.14 | Python 3.14.7: 8 THRML and 15 boundary cases passed, zero failures/errors/skips; [ordinary installed-wheel job](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/actions/runs/37761366047/job/113258292329) at `6be1506cac5957973dff8edeaf77e84e2e7ceb46` |

Core/API/scientific Python 3.14 acceptance does not admit GUI callbacks. Released
Gradio 6.29.1 still calls `asyncio.iscoroutinefunction` in `function_wrapper`
before the later `inspect.iscoroutinefunction` Blocks check. This preserves the strict callback blocker;
it is not an upgrade, warning suppression or new native 6.29.1 acceptance.
The frozen dependency lock remains authoritative. Native Windows acceptance
requires actual payload identity and original cancellation predicates; macOS
diagnostics or a terminal launcher alone do not establish it. The security
owner's [filesystem contract](../security/filesystem_boundaries.md) defines
the precise native guarantee and launcher boundary.

## Installation boundary

Each runner builds a wheel directly and rebuilds it from the sdist using the
hash-pinned [build requirements](../../.github/build-requirements.txt). It
verifies both wheel inventories and `RECORD` entries, compares artifact
metadata, and runs base installs of both distributions with an ordinary
resolver. Linux covers CPython 3.12, 3.13 and 3.14. macOS and Windows retain
their 3.14 lanes. The POSIX lanes also resolve `all` and run the `gui`
wheel smoke test with Gradio and Pillow.

To check built distributions locally, pass the direct wheel, rebuilt wheel,
and source distribution. The receipt records the checked metadata and each
artifact's SHA-256 digest.

```bash
python scripts/check_distribution_artifacts.py \
  --wheel "$DIRECT_WHEEL" --rebuilt-wheel "$REBUILT_WHEEL" \
  --sdist "$SDIST" --receipt "$ARTIFACT_RECEIPT"
```

Each runner exports runtime requirements from `uv.lock` with hashes and
`--frozen`. Core, API and THRML
runtime dependencies must install as ordinary binary wheels. Development test
tools are installed from a separate frozen hashed export. The GNN wheel is
installed without resolving dependencies again, because they have already been admitted
from that lock. No editable install or source `PYTHONPATH` is used. Dependency
installation runs outside the checkout with uv configuration discovery disabled, so unrelated local
resolver settings cannot alter the frozen exported requirements.

Every supported Python (3.12 and newer) shares one JAX/Matplotlib resolution
split. Each receipt records actual installed distribution versions.
The committed lock remains the authority; do not manually substitute a newer
JAX, Matplotlib or upstream THRML checkout to make a candidate pass.

## Base wheel acceptance

Each platform lane also installs the same built wheel into a separate clean
environment. Its hashed export excludes default dependency groups and
optional extras. This environment runs the installed CLI validate and extract commands,
checks the distribution's `gnn` console entry point and exercises an
initialize/shutdown exchange through the LSP fallback. The API command must
diagnose its missing extra and the THRML readiness probe must report that its
module is missing. The acceptance script verifies that the API, THRML, pygls
and development test packages are absent and exercises an invalid model-object
field against the installed schema mode. It records distribution versions
and binds the result to the tested source, wheel, lock and base
export. Each lane retains both its base profile and its extended native
profile evidence.

Both wheels ship the PEP 561 marker `gnn/py.typed`. From the installed base
wheel outside the checkout, strict mypy checks the types of the parser,
shared types, importer pin, and semantic witness APIs. A separate negative
probe must report an argument type error when a caller passes an integer as
the parser path. The base typing environment installs only the locked mypy
dependency closure and checks that FastAPI and pygls are absent.

## Public and numerical acceptance

[`run_installed_platform_acceptance.py`](../../scripts/run_installed_platform_acceptance.py)
runs with isolated Python imports and an external working directory. It checks
scientific imports, the actual PyMDP/JAX stack probe, a native JIT operation,
and a generated JAX simulation with finite normalized posterior arrays.
Installed CLI validation/extraction and template list/show/copy exercise
packaged resources; template CLI commands treat deprecation warnings as errors.
A missing model is a required negative case. Every run writes its JSON witness
to `--receipt PATH`; `--base-wheel` selects the core-wheel profile and
`--omit-api` is a local pre-integration witness that the hosted lane never uses.

Set `GNN_API_ROOT` to an absolute existing trusted data directory **before API
imports**. This operator setting separates installed code from model/output
storage. An installed API without that setting must refuse instead of creating
storage alongside its code. Four real concurrent HTTP model-registry requests must use distinct
scratch snapshots that are removed afterward; an attempted root escape must
return HTTP 400. The operator controls this directory against adversarial parent
replacement. See the [filesystem threat model](../security/README.md)
for exact platform guarantees and limits.

The native THRML test file is copied with its fixtures into an external witness
root and runs against the ordinary installed release wheel. Independent
forward/backward witnesses compare actual sample-derived marginals; source,
seed, managed output and source-composition contracts remain intact. A selected
native test that skips cannot satisfy this lane.

Three concurrent allocating workers check measured RSS, actual cancellation,
absolute timeout, verified cleanup and drained streams. Sampled RSS is a
resident-footprint observation, not an exact peak or memory upper bound.
Native boundary tests separately exercise lease exclusion, real links/junctions,
creation/deletion races where supported, and supported-or-refused descendant
requirements. Windows direct-worker cleanup does not certify descendant
containment or descendant resource accounting. Stronger requests must fail
before a worker starts unless a separately accepted containment implementation
exists. The workflow never borrows POSIX evidence for Windows.

The Windows direct-worker cancellation fixture uses the current Python's native
base interpreter for its standard-library-only child, independently checks the
same Python minor, and binds the actual supervised PID and creation identity to
the child-authored PID. Its original cancellation and cleanup deadlines remain.
A Windows venv `python.exe` can be a redirector that starts a separate payload
process: stopping that supervised redirector does not certify the payload's
termination under `direct_worker_only`. Interpreter launchers, wrappers and their
children remain outside a descendant guarantee. The native unsupported-descendant
prelaunch controls remain required.

## Receipts and limits

Each lane uploads source/wheel/lock/export identities, native JUnit reports,
installed acceptance JSON, CLI logs and tracked-file bookends. Linux/macOS select
the POSIX race tests; Windows explicitly deselects those tests and runs its
native junction, advisory lease and direct-worker cases. All selected cases
must finish without skips, errors or failures.

The private 2026-10-07 candidate run on macOS arm64 installed the unmodified
frozen scientific runtime under Python 3.14.4, imported the real JAX 0.11.2 /
Matplotlib 3.11.2 stack, and completed the native JAX and allocating-worker
witnesses. This is local evidence only. The API root repair and every hosted
candidate lane require acceptance on the final integrated release source.
GUI, public network/provider, audio, GPU and external Julia/Stan/hardware
execution remain explicitly provisioned lanes.

The driver’s `omit_api` setting records a bounded pre-integration local witness. The
required hosted workflow never uses it; that receipt cannot satisfy complete
installed-platform acceptance.

## Wave 1A local wheel and optional runtime profiles (2026-11-15)

Measured on macOS 27.2.1 arm64, CPython 3.14.10, with exact uv 0.13.6.
The candidate branch is `codex/gnn-wave1-packaging-runtime-20261115`, based on
`d8ac475524ee640a88d7d181573642d666b0c78e`. This local receipt does not
replace native Linux/Windows or final integrated-source acceptance.

The wheel is `generalized_notation_notation-5.1.0-py3-none-any.whl`. Core
requirements now include pydantic, jsonschema, rich and packaging. LSP has
its own extra; API declares Starlette; GUI declares websockets and the
optional watchdog consumer. The guarded pkg_resources consumer maps
to setuptools in the research extra. The `all` extra is the exact normalized
union of all named non-dev runtime extras, with matching markers.
`uv.toml` mirrors the resolver constraints in pyproject.toml.

| Local command or witness | Result |
| --- | --- |
| `uv sync --frozen` | Passed |
| `uv lock --check --python 3.11`, `3.12`, `3.13`, `3.14` | All four passed |
| `uv build --wheel --out-dir /tmp/gnn-wave1-packaging/dist` | Passed |
| `python scripts/check_dep_hygiene.py` | Passed; no stale mypy overrides (cap 0) |
| `ruff check .` | Passed |
| `mypy src/gnn` | Passed, 780 source files |
| `lint-imports` | Three contracts kept; zero broken |
| `python -m pytest tests/packaging tests/setup tests/lsp tests/api --basetemp=/tmp/gnn-wave1-packaging/pytest --override-ini addopts= --no-cov -q` | 322 passed in 36.13 seconds |
| `./run.sh lint`, `./run.sh type` | Passed |
| Standalone packaging tests after witness refinements | Three passed in 0.09 seconds |
| Final `pytest tests/setup tests/lsp tests/api tests/docs/test_capability_contracts.py --override-ini addopts= --no-cov -q` | 333 passed in 37.62 seconds |
| Final `check_capability_contracts.py --strict` | Passed |

The broad `ruff format --check .` additionally reaches pre-existing
formatting drift in `bench/mcp_wrapper_metric.py`, outside this lane's
ownership. The repository's maintained formatting scope, exercised by
`run.sh lint`, passes.

Each wheel profile used a fresh venv under `/tmp/gnn-wave1-packaging/`,
outside the checkout, with separate frozen requirements exports and hashed
binary-wheel installations. The GNN wheel was installed with `--no-deps`.
Witnesses ran with `python -I` and checked GNN's installed module location.

- **Base:** `run_installed_platform_acceptance.py --base-wheel` accepted
  `gnn --help`, fallback LSP initialize/shutdown, schema behavior and missing
  API/THRML diagnostics. Public parsers, types, importer and semantic-witness
  modules imported. Direct imports of each guarded API submodule emitted the
  actionable `generalized-notation-notation[api]` diagnostic.
- **LSP:** the actual pygls server initialized and shut down over framed
  stdio. The reader accepts both Content-Length and Content-Type headers.
- **API:** both run and job app factories booted; all shared parity routes
  were present. Both served health and template requests successfully and
  returned the error envelope/404 for a missing parse input. Their configured
  root was an independent scratch directory.
- **Public typing:** ordinary strict downstream mypy reported four
  `import-untyped` errors: the wheel has no `py.typed` marker. No PEP 561
  support is claimed or added. Explicit implementation-type analysis with
  `mypy --strict --no-incremental --follow-untyped-imports --follow-imports=silent`
  passed for `GNNParsingSystem`, `GNNFormat`, `GNNVariable`, `ImporterPin` and
  `build_semantic_witness`. Its base venv contained mypy's locked dependency
  closure and still had neither FastAPI nor pygls installed. This observation
  describes the wheel built for Wave 1A.

The lock resolves opentelemetry-api **1.45.2**, tornado **6.6.1** and
inferactively-pymdp **1.0.5**. It retains Flax **0.12.8/0.12.10**, ngclearn's
Python >=3.12 marker, and soupsieve **2.10/4.0** splits.

Initial offline acquisition failures, the test-reader header failure and the
ordinary untyped-wheel mypy diagnostic remain separate evidence; later runtime
acceptance does not convert those earlier attempts into passing runs.

### Local command receipts and candidate identities

The source checks, wheel-profile collector and final gates are retained at:

- `/tmp/gnn-wave1-packaging/checks-final.json` and command logs.
- `/tmp/gnn-wave1-packaging/profiles-final.json` (base acceptance and initial optional-profile attempt).
- `/tmp/gnn-wave1-packaging/base-acceptance.json`.
- `/tmp/gnn-wave1-packaging/accepted/profiles.json` and individual profile logs
  (runtime witnesses pass; ordinary downstream mypy records the untyped-wheel failure).
- `/tmp/gnn-wave1-packaging/closeout/checks.json` (all checks pass after test relocation).
- `/Volumes/GitVault/Local/execution-launches/gnn-wave1-closeout/4397-fb2ee64b4aee474792e3c70cb99db202/` (supervisor SUCCEEDED, child reaped, no remaining descendants).
- `/tmp/gnn-wave1-packaging/final/checks.json` and logs (opt-in downstream mypy
  and the complete `run.sh gates` attempt).

Command argv select the isolated worktree explicitly. The GitVault supervisor
receipts bind the canonical repository identity/revision separately; they must
not be cited as testing canonical source. Wheel/lock/input digests below bind
the measured candidate payload. Aggregate collector exit zero alone never
certifies nested checks: consult their recorded returncodes.

- wheel SHA-256: `049a0a1ca973743c339c2fdc005adb7446d853668b57f2d3d652f519deb10e2c`.
- pyproject.toml SHA-256: `248a13cab3f1f10146b353076893787133c38feb72d95a99c9de78d782f75a80`.
- uv.lock SHA-256: `723d2b672a88a3e2c7867cb522ed629af1c34b92b96573713977208758d82cc4`.
- uv.toml SHA-256: `8d033c40f34d90a2c652d3f0cdd4eccfee04a7715c92134a9af4fdf16efeaa1b`.

The complete `./run.sh gates` attempt reached **9,094 passed, seven failed,
10 subtests passed**, in 423.88 seconds, with **78.22% coverage** (60% floor).
All seven failures came from the documented test-directory count after
introducing `tests/packaging`. The new witnesses were then relocated into
the already approved `tests/setup` directory; the corrected focused selection
passed all 333 tests, including all seven regressions. Strict capability
contracts and all three relocated installed-profile witnesses also passed.
The full suite was not repeated after relocation. Its later docs, MCP and v3
gates were not reached. The broad gate's collector stdout, nested exit code
and pytest footer are retained; its supervisor terminal receipt is absent,
so no completed supervisor success is inferred from that attempt. The final
closeout run has a complete, passing supervisor receipt.

The initial focused command above is historical: current witnesses are
`tests/setup/test_dependency_declarations.py` and
`tests/setup/qualify_installed_profile.py`. No API CLI/operation call sites
were changed. Scratch and bytecode from the temporary test directory were
preserved outside the checkout.

`./run.sh gates` returned 1; see the retained log for the complete gate boundary and limits. Hosted candidate lanes and release-time paired custody remain unverified by these local runs. Task scratch, venvs and command receipts are retained for review; no worktree retirement or deletion was performed.

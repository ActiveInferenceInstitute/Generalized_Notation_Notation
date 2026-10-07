# Installed-package and native platform acceptance

The [installed-platform workflow](../../.github/workflows/installed-platforms.yml)
accepts an ordinary wheel outside the source checkout. Its candidate matrix is
Linux/Python 3.11, 3.12 and 3.14, macOS/Python 3.14, and Windows/Python 3.14.
A supported release must have successful native receipts for its exact source;
a resolvable wheel plan or classifier alone does not establish support.
The full source test suite remains a separate Python 3.11–3.13 matrix.

## Installation boundary

Each runner builds one wheel from its tested revision and exports runtime
requirements from `uv.lock` with hashes and `--frozen`. Core, API and THRML
runtime dependencies must install as ordinary binary wheels. Development test
tools are installed from a separate frozen hashed export. The GNN wheel is
installed without resolving dependencies again, because they have already been admitted
from that lock. No editable install or source `PYTHONPATH` is used. Dependency
installation runs outside the checkout with uv configuration discovery disabled, so unrelated local
resolver settings cannot alter the frozen exported requirements.

The Python 3.11 lane retains its JAX/Matplotlib split; Python 3.12 and newer
retain theirs. Each receipt records actual installed distribution versions.
The committed lock remains the authority; do not manually substitute a newer
JAX, Matplotlib or upstream THRML checkout to make a candidate pass.

## Public and numerical acceptance

[`run_installed_platform_acceptance.py`](../../scripts/run_installed_platform_acceptance.py)
runs with isolated Python imports and an external working directory. It checks
scientific imports, the actual PyMDP/JAX stack probe, a native JIT operation,
and a generated JAX simulation with finite normalized posterior arrays.
Installed CLI validation/extraction and template list/show/copy exercise
packaged resources; template CLI commands treat deprecation warnings as errors.
A missing model is a required negative case.

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

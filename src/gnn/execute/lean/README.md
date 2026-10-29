# gnn.execute.lean

Lean 4 execution backend for the GNN pipeline: verifies emitted documents
against the `FEP.GnnDocument` typed surface owned by the sibling `fep_formal`
checkout (bridge contract v0.6).

## What this module does

`GNNExecutor` registers `lean` as the ninth execution family. The runner
resolves the canonical fep_formal checkout (`FEP_FORMAL_ROOT` env overrides the
`../fep_formal` default), discovers `*.lean` files and emitted GNN `*.md`
documents under the rendered target directory, and drives
`fep-formal bridge verify-document` for each — writing per-document receipts
into the framework output directory.

## Public API

- `FEP_FORMAL_ROOT_ENV` — environment variable naming the canonical checkout.
- `resolve_fep_formal_root()` — locate + validate the checkout (or `None`).
- `lean_toolchain_available()` — availability probe used by the executor.
- `verify_document(document, receipt=None, *, model, fail_on_warnings,
  gnn_root, timeout)` — verify one document; returns a record with
  `success`, the parsed receipt, or a fail-closed `error`.
- `run_lean_scripts(rendered_simulators_dir, execution_output_dir=None,
  recursive_search=True, verbose=False)` — per-framework runner (same
  signature contract as the other executors); `True` only when every
  document verifies (or there is nothing to verify).

## Scope boundary

`verify-document` proves syntax and `WellFormed` only. It does NOT
auto-prove `DiscreteConforms` / `ContinuousConforms` — that is
research-grade and out of scope (bridge contract v0.6 §13). When the
fep_formal checkout is unavailable the backend is reported skipped, never
silently successful.

Canonical checkout selection uses `FEP_FORMAL_ROOT` before the deprecated
`FEP_LEAN_ROOT` alias, then the `fep_formal` sibling before `fep_lean`.
An invalid explicit override fails closed. `resolve_fep_formal_root` is the
canonical resolver; `resolve_fep_lean_root` remains its released 2.x alias.
The project name, console script and physical `src/<namespace>/cli.py` must
match one exact released contract: `fep-formal` / `fep_formal.cli:main`, or
`fep_lean` / `fep_lean.cli:main`. Importable aliases do not select an origin.
Frozen pair pins and proof receipts keep their original identities; Source
changes require fresh paired qualification.

Bridge dispatch and the toolchain probe read the companion's exact stable uv
pin, `.python-version` and `.venv/pyvenv.cfg`. The environment Python and selected
console must already be present and executable. The command selects the pinned
uv tool in **offline** mode and also runs its child offline, with frozen
project dependencies, no synchronization and Python downloads disabled. A
missing cached tool or unprovisioned environment fails; dispatch does not install
or repair either. Provision this cache/environment separately before selecting
the Lean execution profile.

The single subprocess envelope launches the absolute companion environment
Python. It checks the actual interpreter/version/prefix and the imported CLI's
physical Source path before calling `main`. Source execution deliberately prepends
`<companion>/src` to the child `PYTHONPATH`, preserving any existing path after it.
These Source/metadata checks do not certify an installed wheel. Only controlled
child overrides are recorded; unrelated caller settings are not copied into the
record. Cancellation, timeout and registered process cleanup stay with the
existing subprocess envelope. Path/metadata regression controls do not claim
that an external Lean toolchain was executed.

# AGENTS.md — `gnn.execute.lean`

Lean 4 execution backend: document verification against the `FEP.GnnDocument`
typed surface owned by the sibling `fep_formal` checkout (bridge contract v0.6,
`docs/design/gnn-bridge/bridge-contract.md` in that repo).

## Files

- `lean_runner.py` — `resolve_fep_formal_root` (env `FEP_FORMAL_ROOT` overrides the
  `../fep_formal` default), `lean_toolchain_available`, `verify_document`
  (one document → bridge `verify-document` invocation + parsed receipt),
  `run_lean_scripts` (per-framework runner: discovers `*.lean` and GNN `*.md`
  documents under the rendered target directory and verifies each; returns
  `True` only when every document is well-formed or nothing was found).
- `__init__.py` — curated re-export surface (`__all__`).

## Public API

`FEP_FORMAL_ROOT_ENV`, `resolve_fep_formal_root`, `lean_toolchain_available`,
`verify_document`, `run_lean_scripts`.

## Contract

- The runner mirrors the per-framework runner signature
  (`rendered_simulators_dir`, `execution_output_dir`, `recursive_search`,
  `verbose`) so `gnn.execute.executor`'s `ExecutorFrameworkSpec` registry can
  drive it like any other backend (`lean_executions` result key).
- Unavailable checkout ⇒ skip with an info log and a `False` return from
  `run_lean_scripts`; `verify_document` returns
  `{"success": False, "error": "fep_lean unavailable"}`.
- Verification is fail-closed: warnings fail under `--fail-on-warnings`, and a
  non-zero bridge exit records the CLI error text in the result record.
- Scope boundary: `verify-document` proves syntax + `WellFormed` only. It does
  NOT auto-prove `DiscreteConforms` / `ContinuousConforms` (out of scope, see
  bridge contract v0.6 §13).

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

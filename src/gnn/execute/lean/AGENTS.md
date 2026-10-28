# AGENTS.md — `gnn.execute.lean`

Lean 4 execution backend: document verification against the `FEP.GnnDocument`
typed surface owned by the sibling `fep_formal` checkout (bridge contract v0.5,
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
  bridge contract v0.5 §13).

Canonical checkout selection uses `FEP_FORMAL_ROOT` before the deprecated
`FEP_LEAN_ROOT` alias, then the `fep_formal` sibling before `fep_lean`.
An invalid explicit override fails closed. `resolve_fep_formal_root` is the
canonical resolver. The deprecated `resolve_fep_lean_root` calls that function.
`FEP_FORMAL_ROOT_ENV` names the canonical variable. `fep_bridge_executable`
selects `fep-formal` for a canonical package and `fep-lean` for a legacy-only
checkout. Frozen pair pins and proof receipts keep their original identities;
source changes require fresh paired qualification.

Bridge dispatch and toolchain probes derive the exact uv pin from the FEP
checkout metadata and Python from its .python-version. They invoke a
cache-local pinned uv tool and bind only the child to the FEP .venv,
with frozen dependencies, no synchronization or Python downloads. Missing,
malformed or conflicting runtime metadata fails explicitly; caller secrets
are never copied into diagnostic command/environment records. Cancellation
and timeouts remain owned by the subprocess envelope.

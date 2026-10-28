# gnn.execute.lean

Lean 4 execution backend for the GNN pipeline: verifies emitted documents
against the `FEP.GnnDocument` typed surface owned by the sibling `fep_formal`
checkout (bridge contract v0.5).

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
research-grade and out of scope (bridge contract v0.5 §13). When the
fep_formal checkout is unavailable the backend is reported skipped, never
silently successful.

Canonical checkout selection uses `FEP_FORMAL_ROOT` before the deprecated
`FEP_LEAN_ROOT` alias, then the `fep_formal` sibling before `fep_lean`.
An invalid explicit override fails closed. `resolve_fep_formal_root` is the
canonical resolver; `resolve_fep_lean_root` remains a compatibility alias.
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

# src/ — agent rules

## Overview

`src/` contains one package, `gnn`: the 25-step pipeline, its numbered step
drivers and its support packages. Layout and links: [`README.md`](README.md).

## Purpose

All Python source for GNN lives here. Nothing else belongs at this level.

## Directory Identity

Package root; every module lives under `src/gnn/`.

## Invariants

- `gnn.*` is the only import surface. `gnn/__init__.py` re-exports lazily
  (PEP 562 `__getattr__` over an explicit name map), so `import gnn` executes no
  submodule and pays no heavy module-scope dependency.
- The numbered drivers `gnn/[0-9]*_*.py` stay thin: the hard cap and registered
  ratchet are enforced by
  [`scripts/check_thin_orchestrators.py`](../scripts/check_thin_orchestrators.py)
  with `thin_orchestrator_caps.json`. Domain logic belongs in the module package.
- Every directly runnable `gnn/[0-9]*_*.py` carries the stdlib-shadowing guard
  ahead of its first import
  ([`scripts/check_stdlib_shadow_guard.py`](../scripts/check_stdlib_shadow_guard.py)).
- **FEP custody trap.** Edits to these contract files re-drift the FEP_Formal
  custody pair and must close through fresh bridge/native qualification:
  `pyproject.toml`, `uv.lock`, `src/gnn/main.py`, `docs/gnn/gnn_syntax.md`,
  `src/gnn/pipeline/step_registry.py`, `docs/other/fep_lean/bridge-contract.md`.
  Other files below `src/gnn/` do not re-drift it.
- **Manuscript token trap.** Source changes move the counts the manuscript cites
  (`GNN_SRC_PY_FILE_COUNT`, `GNN_SRC_LOC`) and a count-changing PR owes the SC-22
  custody ritual — see the repository [`AGENTS.md`](../AGENTS.md) and
  [`scripts/check_hydrated_prose.py`](../scripts/check_hydrated_prose.py).

## Commands that must pass

```bash
uv run --extra dev python docs/development/docs_audit.py --strict --check-anchors --no-write
just quality
```

Full pytest selections and profile-specific obligations live in
[`TO-DO.md`](../TO-DO.md).

## Do not

- Do not add logic to a numbered driver, and do not create a second step mapping
  beside `gnn/pipeline/step_registry.py`; the
  [consolidation ADR](../docs/decisions/0001-consolidated-pipeline-execution.md)
  forbids it.
- Do not import optional heavy dependencies at module scope. Resolve them inside
  the call path, as `gnn/__init__.py` and the renderer/executor registries do.
- Do not widen a selector, lower a coverage floor or add a skip to make a gate
  pass. Source, tests and receipts own the numbers.

## Traps seen

- Adding a `--flag` token to prose moves the phantom-doc-flag ratchet
  ([`scripts/check_flag_parity.py`](../scripts/check_flag_parity.py)); mention
  flags only where the CLI owns them.
- A new file under `docs/**` moves `GNN_DOC_FILE_COUNT` and owes the custody
  ritual; source-only PRs pay the source counts instead, not both.

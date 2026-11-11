# docs/decisions/ — Architecture Decision Records

## Purpose

Append-only ADRs. Numbered `NNNN-<kebab-title>.md`. Never edit an
accepted ADR's decision; supersede it with a new record that links
backward. Every ADR carries **Status** (Accepted/Superseded/Deferred),
**Date**, and the issue or proposal that motivated it.

Current records:

- [0001-consolidated-pipeline-execution.md](0001-consolidated-pipeline-execution.md) — opt-in in-process step
  executor behind `--consolidated-steps`; numbered-script contract
  unchanged; first slice covers steps 0/3/5 on flat target dirs.
- [048-composition-seam.md](048-composition-seam.md) — supersedes the imported tool-level
  composition surface (`compose`/`pipe`/`find_tools`/`lift`, never landed
  here) with ADR 0001's registry-driven step executor plus the five-seam
  doctrine (`docs/standards/composability.md`); the tool surface stays
  unadopted absent a consumer-demand receipt (five-seam census /
  COMP-001, COMP-002).
- [0049-dual-async-job-surfaces.md](0049-dual-async-job-surfaces.md) — documents the dual async-job
  surfaces as canonical, each for its consumer class (`gnn.api.app` run UX
  behind `gnn serve`, `gnn.api.server` job/step management API); coherence
  contract is the shared `processor` stores, identical parity route
  registration, and per-factory route tables pinned in
  `tests/api/test_api_parity.py`; motivated by the integration-surface
  census (F3).

New ADRs must be listed here.

# COGANT

**Repository**: [`ActiveInferenceInstitute/COGANT`](https://github.com/ActiveInferenceInstitute/COGANT)
· **Default branch**: `main` · **Verified**: 2026-11-02 from origin `main` with read-only GitHub API calls.

## What it models

COGANT deterministically translates Python, JavaScript and TypeScript codebases
into active inference GNN state-space models, with a reverse synthesizer that
reconstructs runnable code from a GNN bundle. GNN is both the output format and
the validation target.

## GNN specification

- Format documentation: `cogant/docs/theory/gnn_format.md`,
  `cogant/docs/theory/gnn_format_reference.md`, and the concept page
  `cogant/docs/concepts/gnn.md`.
- Emitted models: `cogant/examples/zoo/04_pomdp_minimal/model.gnn.md`,
  `06_hierarchical/model.gnn.md`, `12_full_pomdp/model.gnn.md`.
- Bridge implementation: `cogant/py/cogant/gnn/upstream_bridge/`.

## Pinned GNN revision

`cogant/docs/evaluation/gnn_v5_0_3_compatibility.json` records `locked_version`
5.0.3 with `locked_commit`, `expected_commit` and `origin_head` all
`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`, plus a 20-step upstream run with 20
successes. Two adjacent records are historical or internal and are not the
current upstream pin: `cogant/evaluation/gnn_upstream_audit_surface.json` audits
release 4.1.0 (commit `46459949944e3217f8a6a804d4ff8c52829ef275`), and
`cogant/py/cogant/schema/versions.py` declares an internal schema constant that
is independent of the upstream release pin.

## Conformance and validation evidence

- `.github/workflows/ci.yml` job "Optional upstream GNN smoke (manual)"
  (`workflow_dispatch`) exercises `cogant.gnn.upstream_bridge` and
  `tests/unit/test_gnn_upstream_bridge.py`,
  `tests/unit/test_gnn_upstream_bridge_contract.py`.
- `cogant/docs/cookbook/14_gnn_validation.md` is the validation walkthrough;
  `cogant/docs/evaluation/GNN_VALIDATION_REPORT.md` and
  `cogant/docs/evaluation/GNN_UPSTREAM_AUDIT_SURFACE.md` carry recorded results.

## Verification limits

Not verified: that the manual smoke job has run recently, or that the 5.0.3
compatibility record is reproducible today. The compatibility JSON was read at
`main` on 2026-11-02.

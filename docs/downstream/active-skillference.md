# Active_Skillference

**Repository**: [`ActiveInferenceInstitute/Active_Skillference`](https://github.com/ActiveInferenceInstitute/Active_Skillference)
· **Default branch**: `main` · **Verified**: 2026-11-02 from origin `main` with read-only GitHub API calls; job evidence re-read 2026-11-06 and 2026-11-11.

## What it models

A micro-learning platform for active inference and the free energy principle: a
validated prerequisite graph whose quantitative claims come from tested kernels,
exported to SkillTree. The GNN specification carries the discrete
active inference agent that anchors the course material.

## GNN specification

- `gnn/discrete_active_inference_agent.gnn.md` — the declared discrete agent.

## Pinned GNN revision

`gnn/upstream.json` records tag `v5.0.3` and commit
`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`.

## Conformance and validation evidence

- `tests/test_gnn_conformance.py` checks the declaration against the pinned
  release, with `src/gnn_conformance.py` as the shared implementation and
  `scripts/validate_gnn_spec.py` as the command-line entry point.
- `.github/workflows/gates.yml` job key `gnn-conformance` publishes
  `gnn conformance (pinned)` (re-read 2026-11-06): it validates
  `gnn/discrete_active_inference_agent.gnn.md` with the parser revision pinned
  in `gnn/upstream.json`, and `tests/test_gnn_conformance.py` fails rather than
  skips when the pin is unavailable. The "Tests with the coverage floor" step
  remains the general suite.

## Verification limits

Not verified: that the current `main` run is green. The pin file, specification
and workflow definition were read at `main` on 2026-11-02.

# active_inference_language_model

**Repository**: [`ActiveInferenceInstitute/active_inference_language_model`](https://github.com/ActiveInferenceInstitute/active_inference_language_model)
· **Default branch**: `main` · **Verified**: 2026-11-02 from origin `main` with read-only GitHub API calls.

## What it models

A fully explicit discrete language stack in pymdp, from letters through document
position, with a gate-audit method that binds every reported number to a recorded
run. GNN is the notation for the model ladder.

## GNN specification

- `gnn/models/L1_letter.gnn.md` through `gnn/models/L6_narrative.gnn.md`
  (L1, L1g, L2a, L2b, L3, L4, L5, L5w, L6), with mutation variants under
  `gnn/mutations/`.
- `gnn/SPEC.md` states the AILM-side subset of GNN that `src/ailm/gnn/` reads and
  enforces, and what it deliberately does not check.
- `gnn/README.md` and `gnn/AGENTS.md` route the model ladder.

## Pinned GNN revision

`gnn/upstream.json` pins the upstream repository at tag `v5.0.3`, commit
`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`.

## Conformance and validation evidence

- CI job `gnn-interop` in `.github/workflows/ci.yml` checks out the pinned GNN
  release as `../ailm-gnn-v5.0.3`, asserts its HEAD equals the pin, and runs
  `scripts/verify_gnn_release.py` with `AILM_REQUIRE_GNN_INTEROP=1`, so a missing
  prerequisite fails instead of skipping.
- `src/ailm/gnn/conformance.py` enforces the AILM-side parse contract against a
  live pymdp agent.

## Verification limits

Not verified: that the current `main` run of `gnn-interop` is green, and that the
pin is the latest available GNN release. The pin file and workflow definition
were read at `main` on 2026-11-02.

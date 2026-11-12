# ActInf_RxInfer

**Repository**: [`ActiveInferenceInstitute/ActInf_RxInfer`](https://github.com/ActiveInferenceInstitute/ActInf_RxInfer)
· **Default branch**: `main` · **Verified**: 2026-11-02 from origin `main` with read-only GitHub API calls; job evidence re-read 2026-11-06 and 2026-11-11.

## What it models

A Julia project connecting active inference with RxInfer.jl: a package entry
point plus a worked Beta-Bernoulli message-passing example (coin-bias inference)
that mirrors the discrete generative model from the GNN documentation.

## GNN specification

- `specs/coin_flip.gnn.md` — the declared Beta-Bernoulli model.
- `scripts/validate_gnn_spec.py` and `scripts/gnn_parser_probe.py` implement the
  check the workflow runs.

## Pinned GNN revision

`specs/gnn-pin.json` records tag `v5.0.3` and commit
`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`.

## Conformance and validation evidence

- `.github/workflows/ci.yml` job `gnn-conformance` (display name
  "gnn conformance (pinned)") provisions a Python 3.12 runtime for the
  pinned parser and runs `scripts/validate_gnn_spec.py` against
  `specs/coin_flip.gnn.md`, so the example and the pinned GNN release stay
  aligned. The check name carries no version: the pinned revision is recorded
  in `specs/gnn-pin.json`.
- There is no separate GNN test module; the workflow job is the check.

## Verification limits

Not verified: that the current `main` run is green. The pin file, specification
and workflow definition were read at `main` on 2026-11-02; the check name and
the pin file path were re-read at `main` on 2026-11-05.

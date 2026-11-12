# active_inference_battleship

**Repository**: [`ActiveInferenceInstitute/active_inference_battleship`](https://github.com/ActiveInferenceInstitute/active_inference_battleship)
· **Default branch**: `main` · **Verified**: 2026-11-02 from origin `main` with read-only GitHub API calls; job evidence re-read 2026-11-06 and 2026-11-11.

## What it models

GNN-based active inference for Battleship, with a playable dashboard,
interpretable decisions, a reproducible manuscript and an animated lecture. The
GNN declaration describes a categorical one-shot model under the project's
`binary-static-fleet-v1` contract: observation factor `A[2,M,N²]`, static
transition `B[M,M]`, outcome preference `C[2]`, initial belief `D[M]` and action
prior `E[N²]`.

## GNN specification

- `models/battleship.gnn.md` — variables, dimensions, graph dependencies,
  equations, discrete one-shot time and default parameters.
- `docs/GNN_SPECIFICATION.md` — the stage table (declare, parse, validate,
  compile, execute, inspect) and the contract's limits.

## Pinned GNN revision

`models/gnn.pin.json` records the repository, tag `v5.0.3` and commit
`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`. `model:import` resolves that commit
in the `GNN_ROOT` checkout, materializes the commit's `src/gnn` tree into an
external cache, runs the parser from those exact bytes, and binds per-file
digests; the checkout's own HEAD is not the pin.

## Conformance and validation evidence

- `scripts/parse_gnn.py` imports the pinned upstream `MarkdownGNNParser`.
- `scripts/contract.ts` validates the parser payload (identity, unique
  variables, dimensions, edges, parameters, equations, time semantics).
- Documented commands: `bun run model:check` compares current local source bytes
  with the compiled artifact; `bun run model:import` re-imports after an
  intentional, supported source change.
- `.github/workflows/ci.yml` (workflow name "GNN pin conformance") job
  `model-check` publishes `gnn conformance (pinned)`: it reads
  `models/gnn.pin.json`, resolves the pinned commit and tag, parses with the
  pinned upstream parser and binds per-file digests (re-read 2026-11-06). The
  2026-11-02 statement that no workflows existed is superseded.

## Verification limits

Not verified: that `bun run model:check` passes on current `main`. The project's
own documentation states the check is not proof of the entire upstream pipeline,
all transitive dependencies, or scientific adequacy.

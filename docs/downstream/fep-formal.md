# fep_formal

**Repository**: [`ActiveInferenceInstitute/fep_formal`](https://github.com/ActiveInferenceInstitute/fep_formal)
· **Default branch**: `main` · **Verified**: 2026-11-02 from origin `main` with read-only GitHub API calls.

## What it models

A standalone Lean 4 catalog of formally verified theorems spanning the Free
Energy Principle, active inference, Bayesian mechanics, information geometry and
thermodynamics. Its GNN surface is a bridge that gives Lean denotations to GNN
documents rather than a corpus of generative models.

## GNN specification

- Bridge contract: `docs/design/gnn-bridge/bridge-contract.md`; design notes
  `docs/design/gnn-bridge/direction-1-lean-to-gnn.md` and
  `direction-2-gnn-to-lean.md`.
- Lean modules: `lean/FepSketches/gnn_denotation.lean`,
  `gnn_denotation_continuous.lean`, `gnn_document.lean`,
  `gnn_render_statements.lean`.
- Bridge work packages: `specs/gnn-bridge-p1-finite-spike/`,
  `specs/gnn-bridge-p3-certificates/`, `specs/gnn-bridge-p4b-continuous-emission/`.

## Pinned GNN revision

Content-addressed, not a release tag. `specs/gnn-bridge-w1-bridge-operations/syntax-pin.json`
records SHA-256 hashes for GNN's `docs/gnn/gnn_syntax.md` and
`src/gnn/pipeline/step_registry.py`; `specs/gnn-bridge-w2-source-custody/source-pin.json`
pins the `fep_lean` companion commit `40e30c7ecb6680a9c01d3486fb65140c2aa49410`
plus file hashes. No GNN tag or commit is recorded in these pin files (verified
2026-11-02).

## Conformance and validation evidence

- `.github/workflows/source-pair-qualification.yml` job `qualify-source-pair`
  takes explicit `fep_revision` and `gnn_revision` inputs (40-character lowercase
  hex) and validates them.
- `docs/pin_audit.py`.
- Recorded bridge runs exist under `specs/gnn-bridge-p1-finite-spike/`
  (`gnn_output`, `gnn_output_run1_annotated`, `gnn_output_run2_accepted`) with
  Step 5 type-checker output.
- On the GNN side, the paired custody procedure is
  [../development/fep_lean_paired_revision.md](../development/fep_lean_paired_revision.md)
  and the mirrored contract is [../other/fep_lean/bridge-contract.md](../other/fep_lean/bridge-contract.md).

## Verification limits

Not verified: that the qualification workflow is currently green. Because the
syntax pin is content-addressed, it does not name a GNN release; freshness must
be re-established by hashing the declared files at the intended revision.

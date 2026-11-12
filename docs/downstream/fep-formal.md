# fep_formal

**Repository**: [`ActiveInferenceInstitute/fep_formal`](https://github.com/ActiveInferenceInstitute/fep_formal)
· **Default branch**: `main` · **Verified**: 2026-11-02 from origin `main` with read-only GitHub API calls; job evidence re-read 2026-11-06 and 2026-11-11.

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

Content-addressed, not a release tag, and the source pin names the GNN commit
it was taken from. Two pin files carry the bridge:

- `specs/gnn-bridge-w1-bridge-operations/syntax-pin.json` records SHA-256
  hashes for GNN's `docs/gnn/gnn_syntax.md` and
  `src/gnn/pipeline/step_registry.py`.
- `specs/gnn-bridge-w2-source-custody/source-pin.json` records GNN commit
  `2778e65c2228148417b2cf71832eb86a019785fb` with SHA-256 hashes for the six
  declared contract files (`pyproject.toml`, `uv.lock`, `src/gnn/main.py`,
  `docs/gnn/gnn_syntax.md`, `src/gnn/pipeline/step_registry.py`,
  `docs/other/fep_lean/bridge-contract.md`), and the `fep_lean` companion
  commit `301c2d2405b260ef8b645b0c5a3914fbcdf5c49c` with its own file hashes.
  All six GNN hashes reproduce exactly from that commit.

Per-file binding, measured 2026-11-11 by hashing each declared file at tag
`v5.0.3`, at the pinned `2778e65c` and on GNN `main` at `68d15b2a`:

- `src/gnn/pipeline/step_registry.py`, `src/gnn/main.py` and the mirrored
  `docs/other/fep_lean/bridge-contract.md` have the same bytes at all three
  revisions.
- `docs/gnn/gnn_syntax.md` matches the recorded hash at the pinned commit and
  on `main`, but not at tag `v5.0.3`: the notation contract itself moved after
  the release, and the pin follows it.
- `pyproject.toml` and `uv.lock` match the recorded hashes only at the pinned
  commit; they already differed at tag `v5.0.3` and have moved again on `main`.
  The dependency files bind to the pinned commit rather than to a release tag,
  which is what content-addressing is for.

`2778e65c` is 206 commits after tag `v5.0.3` and is an ancestor of GNN `main`.

## Conformance and validation evidence

- Automatic, on every push and pull request: the `.github/workflows/ci.yml`
  `python` job runs the step "Check the frozen-custody bridge gates", which
  validates committed bridge artifacts
  (`specs/gnn-bridge-q5-artifact-proof/generate_probe.py`,
  `specs/cascadia-geo-act-proof/generate_probe.py`,
  `specs/geo-infer-notation-bridge/check_geo_notation_bridge.py`, each in its
  read-only check mode), and its pytest selection includes
  `tests/test_gnn_bridge_operations.py`, which exercises the bridge's read-only
  checks and their refusal paths.
- A dedicated `.github/workflows/ci.yml` job key `gnn-conformance` publishes the
  `gnn conformance (pinned)` context on every push and pull request
  (re-read 2026-11-11); the `python` job above remains separate bridge evidence.
  The `protect-default-branch` ruleset requires the separate
  `gitvault/fep-gnn-conformance` check, not this context.
- `docs/pin_audit.py` (run in `ci.yml` as "Require the newest stable Lean-Mathlib
  release pair" and in the scheduled `pin-audit` job of `lean-latest.yml`)
  audits the Lean toolchain, Mathlib revision and model pins in project
  markdown. It does not audit the GNN syntax pin and is not bridge evidence.
- Manual, not automatic: the live pair check `fep-formal bridge status`, pointed
  at a GNN checkout, compares the source pin against that checkout
  (`specs/gnn-bridge-w1-bridge-operations/bridge_status.py` is the compatibility
  entry point); the source pin is rewritten only by `fep-formal bridge pin`.
- Dispatch-only: `.github/workflows/source-pair-qualification.yml` job
  `qualify-source-pair` takes explicit `fep_revision` and `gnn_revision` inputs
  (40-character lowercase hex) and validates them. It runs only when dispatched;
  the most recent runs completed 2026-10-28, so the pair is not verified
  continuously.
- The two copies of the bridge contract are not byte-identical: the GNN mirror
  hashes `66ca4b50…` and the canonical `fep_formal` copy `8bfa57cb…`, differing
  only in the two table rows that name the canonical and the mirror copy. The
  bridge's own mirror check (`fep_formal.bridge.operations`) strips those two
  rows before comparing.
- Recorded bridge runs exist under `specs/gnn-bridge-p1-finite-spike/`
  (`gnn_output`, `gnn_output_run1_annotated`, `gnn_output_run2_accepted`) with
  Step 5 type-checker output.
- On the GNN side, the paired custody procedure is
  [../development/fep_lean_paired_revision.md](../development/fep_lean_paired_revision.md)
  and the mirrored contract is [../other/fep_lean/bridge-contract.md](../other/fep_lean/bridge-contract.md).

## Verification limits

Not verified: that the qualification workflow is currently green (its runs are
manual and the most recent are 2026-10-28), and that a GNN checkout matches the
pin today — that comparison needs the manual `fep-formal bridge status` run.
The pins are content-addressed and name a GNN commit rather than a release, so
freshness must be re-established by hashing the declared files at the intended
revision.

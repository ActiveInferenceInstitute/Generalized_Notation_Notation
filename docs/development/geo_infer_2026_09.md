# GEO_INFER contract expansion verification

The paired change adds explicit Gaussian and factored interchange alongside the
strict categorical v1 export. Gaussian models declare discrete F/G/H/Q/R, units
and initial beliefs; factored JSON declares dependency axes and enumerated
policies. The repositories remain independently installable. See the
[Gaussian/categorical contract](../../src/gnn/export/geo_infer_contract.md) and
[factored contract](../../src/gnn/export/geo_infer_factored_contract.md).

Step 7 now consumes original source bytes and per-model physical metadata from
the API or CLI, retains nested identities, rejects output escapes and duplicate
metadata, and reports partial failures. Ordinary five-format defaults are kept.

## Ancestry and integration

The topic starts from local fleet commits `3f2694d3a` and `64d49355a`, based on
remote main `aa20514c77bb2cc4757be645755b3e3755ff8530`. Its PR therefore includes
the earlier module-quality sweep (deduplication, hardening, contract tests and
documentation), in addition to the GEO work. The original concurrently edited
checkout was preserved; integration uses an isolated topic worktree.

## Verification

- Full initial run: 4087 passed, 2 failed, 3 skipped, 2 warnings. The registry
  alias and optional-extra environment failures were reproduced and corrected.
- Targeted baseline corrections: 31 passed, including the numbered CLI, step
  registry, environment check and sync/async provider bridge. The bridge now
  detects a running event loop before allocating a coroutine and never retries
  provider RuntimeError as a loop error.
- Export/utils integration: 330 passed before final additional edge cases.
- Gaussian filtering: analytic unequal dimensions and overflow/axis rejection
  verified independently in GEO Python 3.11 and 3.12.
- Strict mypy and repository-wide Ruff checks run on the integrated tree.
- Paired CI runs in both directions: GEO_INFER pins a reviewed GNN commit, and
  this repository pins GEO_INFER in return (see the paired interchange section
  below). Hosted results belong to the PR checks; local success does not
  establish hosted success.

Fresh independent reviews reproduced and then verified fixes for double
conditioning, covariance-overflow validation, contradictory source axes, output
symlinks, duplicate metadata and factored numerical overflow. All completed
capabilities have explicit bounded semantics; Gaussian control selection and
large unbounded exact policy trees are not claimed.

## GNN-side paired interchange (GNN-04)

This repository now mirrors GEO_INFER's paired-revision mechanism on its side.
No fep_lean file, workflow or pin is involved; the mirror covers only the
GEO_INFER pair:

- [`.github/gnn-pair.json`](../../.github/gnn-pair.json) records the exact
  reviewed GEO_INFER revision. Read the committed pin for the current SHA;
  acceptance receipts name both checkouts and retain per-artifact digests.
  The historical expansion results above do not establish a new run. Note
  the deliberate name reuse: GEO_INFER's own `.github/gnn-pair.json` pins
  this repository, so each side names the other in a same-named file.
- `.github/workflows/geo-infer-interchange.yml` validates the pin (known
  slug, 40-hex revision), checks out GEO_INFER at exactly the pinned SHA into
  `geo-infer/`, syncs independent locked runtimes (matrix Python 3.11/3.12
  for the GEO environment; the GNN environment on 3.11 with the dev and
  geo-infer extras; the same heavy-package exclusions GEO's own CI uses), and
  runs `scripts/run_geo_interchange_checks.py`.
- `scripts/run_geo_interchange_checks.py` refuses a checkout that is not at
  the pinned revision, drives GEO's read-only
  `GEO-INFER-TEST/validate_gnn_interchange.py` — export inside the GNN
  environment, consume and deterministic replay inside the GEO environment,
  covering the categorical gridworld, the seven-cell H3 stay/diffuse model,
  the rectangular Gaussian and the explicit factored fixture — and writes a
  receipt directory: both revisions, the pin, the validator's full JSON
  trace, and `digest-manifest.json` cross-checking the per-artifact digests.
  Exit codes are faithful: 0 green, 1 round-trip or receipt failure, 2 pin or
  revision mismatch with nothing run.

What the hosted job proves: the pinned GEO checkout and this GNN checkout
still agree on the exchange contract at both matrix Pythons, with
deterministic replay certified and per-artifact digests retained. A red run
is the drift signal — one side moved since the pin was recorded — and the
pair must be re-pinned together in the same change set, never by suppressing
the check. What remains manual: local environment sync (the runner requires
an explicitly provided, already-synced GEO checkout and GNN interpreter), the
GEO_INFER side's own paired CI against its pin of this repository, and
hosted confirmation of this new workflow's own first runs. Local success —
including the local run recorded above — does not establish hosted success.

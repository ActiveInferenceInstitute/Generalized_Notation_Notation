# Cross-repo custody: the fep_lean and GEO_INFER pairing contracts (GNN-04)

This is the canonical statement of GNN's cross-repo custody obligations. The
GNN side hosts paired-revision CI for the fep_lean bridge pair, mirroring
the mechanism GEO_INFER already runs for this repository: its
`.github/workflows/gnn-interchange.yml` workflow with pin file
`.github/gnn-pair.json`. The GEO_INFER direction (GEO_INFER pinning GNN) is
covered in the mirror-image section below.

## What is pinned

`.github/fep-lean-pair.json` records the reviewed companion revision — the
fep_lean repository slug and a full 40-hex SHA of fep_lean main HEAD that the
GNN-side contract was checked against. That file is the live source of truth
for the current pin; read it directly. This doc deliberately quotes no SHA,
so it cannot go stale the way a quoted pin does.

In the mirror direction, GEO_INFER pins GNN from its own checkout via its
`.github/gnn-pair.json` (see [GEO_INFER direction](#geo_infer-direction)).

## What runs

`.github/workflows/fep-lean-paired-revision.yml`:

1. Reads and validates the pin (known repository slug, 40-hex revision).
2. Checks out fep_lean at exactly the pinned SHA.
3. Records both checkout revisions plus the pin file as receipts, retained as
   workflow artifacts (`if: always()`, mirroring GEO's receipt semantics).
4. Runs fep_lean's **read-only** bridge surface against this GNN checkout:

   - `uv run fep-lean bridge status --gnn-root <gnn checkout>`
   - `uv run fep-lean bridge emit --check --model finite --gnn-root <gnn>`
   - `uv run fep-lean bridge emit --check --model continuous --gnn-root <gnn>`

   `status` and `emit --check` never write: they compare source bytes and
   emitted-document digests only.

Strictness mirrors GEO: the job is blocking. A red run is the drift signal —
the pair must be re-pinned together, not the check suppressed.

## Who bumps the pin

Pin bumps are paired with fep_lean-side re-pins:

- **GNN side**: changing GNN sources that the bridge's source binding covers
  requires a paired commit on fep_lean (per its FEP-EVIDENCE-CURRENT policy)
  that re-records digests against the new GNN state; the GNN-side PR then
  bumps `.github/fep-lean-pair.json` to the new fep_lean revision in the same
  change set.
- **fep_lean side**: when fep_lean main moves, this repository's pin is
  updated to the new reviewed fep_lean SHA together with any GNN-side source
  changes the new pin requires.

A green run therefore certifies agreement between two specific, named
revisions — never a floating main.

## What the workflow proves and does NOT prove

The workflow is a **drift-detection** surface. It proves that the pinned
fep_lean checkout and this GNN checkout still agree on the bridge's source
binding (owner rosters and content digests), finite/continuous freshness, the
syntax surface, the contract mirror, and the formal projection with no drift.

It does **not** prove any Lean theorem, and no notation mapping claim may
promote it into one: the bridge compares symbols and constructs across the two
repositories; correspondence of notation is a documentation-level statement.
Evidence planes stay distinct — a green paired run says "the two checkouts are
in the state fep_lean reviewed", nothing about the truth of formal claims.

## Closeout ordering (canonical)

Any change touching files the bridge's source binding covers must close out in
this exact order:

1. **All content edits land first** — never after the ritual. Any later
   owner-file edit re-drifts the pair.
2. **Manuscript/token ritual** (when manuscript inputs changed): follow the
   SC-22 ritual in the `scripts/z_generate_manuscript_variables.py` module
   docstring — regenerate the token map → rebuild figures → run the template's
   `stage_03_render` → record the render-custody manifest → commit the
   regenerated `output/` artifacts. `scripts/check_hydrated_prose.py`
   (`local-gates.yml` repo-gates) enforces this at PR time: a regenerated
   token map without the render fails it.
   Run the ritual only at the final content state and **never amend a commit
   after recording**: the recorded manifest's `counts_describe_commit` cites
   the commit it was produced at, and amending afterwards orphans that
   reference (a phantom commit in published provenance) while the next
   hydration computes different counts, re-drifting the chain.
   If the new GNN revision is not remotely reachable, publish the committed
   source/artifact branch and open a draft PR before the companion re-pin.
   Its workflow must fetch an exact commit; an unpublished local SHA cannot
   establish hosted pairing. Keep main publication pending and make no further
   owner-file edits after this source/artifact freeze.
3. **fep_lean bridge re-pin**: bridge pin → emit `--refresh-digests` → emit
   `--check` (finite + continuous) → PR → merge on the fep_lean side.
4. **GNN pin bump as the FINAL commit**: bump `.github/fep-lean-pair.json` to
   the new reviewed fep_lean revision in the final commit, then push that final
   revision. The earlier draft-branch publication only supplies the fetchable
   source revision; merge to GNN main requires the final paired and repository
   gates.

Ledger mentions of this discipline (e.g. the "paired-repin discipline" row in
`TO-DO.md`) defer to this section.

## GEO_INFER direction

GEO_INFER pins GNN from its side via its `.github/gnn-pair.json`, and its
paired interchange CI (`.github/workflows/gnn-interchange.yml` in the
GEO_INFER checkout) checks out GNN at exactly that pinned revision. Bumping
that pin is a reviewed companion-revision decision and follows the same
re-pin discipline in mirror image: GNN content edits land first, then the
GEO_INFER-side PR updates its pin to the new reviewed GNN revision.

## Related

- [GEO_INFER paired interchange (GNN-side docs)](geo_infer_2026_09.md)
- fep_lean bridge contract (GNN-side docs): `docs/other/fep_lean/bridge-contract.md`

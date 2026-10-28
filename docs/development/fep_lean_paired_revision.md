# Cross-repo custody: the FEP_Formal and GEO_INFER pairing contracts (GNN-04)

This is the canonical statement of GNN's cross-repo custody obligations. The
GNN side hosts paired-revision CI for the FEP_Formal bridge pair, mirroring
the mechanism GEO_INFER already runs for this repository: its
`.github/workflows/gnn-interchange.yml` workflow with pin file
`.github/gnn-pair.json`. The GEO_INFER direction (GEO_INFER pinning GNN) is
covered in the mirror-image section below.

## Current-source qualification for GNN 5.0.x

The active paired workflow checks the current GNN candidate against the exact
FEP_Formal revision in `.github/fep-lean-pair.json`. It checks out one GNN tree
and passes that tree and its exact revision to the runtime custody runner.
There is no accepted-scientific-baseline checkout or release-root exception in
the active workflow.

GNN 5.0.x preparation does not itself establish a synchronized pair. Current
pair qualification requires a reviewed companion pin, successful strict bridge
and current-Q7 source checks, fresh separately validated native evidence and
the repository/release gates described below. Until those checks finish, any
current-pair or publication claim remains unqualified. The workflow's source
checks retain `native_claim_ready: false`; numerical witnesses and frontend
displays do not establish native proof or generated-runner execution.

<a id="fixed-baseline-for-gnn-421"></a>

## Historical fixed baseline for GNN 4.2.1 and 4.3.0

GNN 4.2.1 and 4.3.0 used a bounded compatibility policy against the FEP_Formal
revision accepted with GNN 4.2.0. Its exact 4.3.0 policy bytes are preserved in
[`fep_4_3_0_baseline.json`](../releases/fep_4_3_0_baseline.json). This archival
record is not an active workflow input; `.github/fep-release-baseline.json`
is absent in the current release tree.

The historical workflow checked both the accepted GNN scientific baseline and
the current release. The declared delta allowed only canonical package-version
replacements and the reviewed frozen lock. Owner additions, deletions,
scientific changes, unrelated metadata changes and unreviewed locks were
rejected. Historical receipts retained both GNN revisions, companion and owner
hashes, the approved delta and `current_pair_synchronized: false`.

A passing historical fixed-baseline run established release compatibility with
that accepted scientific baseline. It did not establish full synchronization,
a fresh formal proof or current-release Q7 native compilation. Historical
native and publication receipts retain their original source epochs. Runtime
validation and mutation controls for these retained fixtures remain available;
the active current-source workflow cannot use the archived policy to admit a
new release.

## What is pinned

`.github/fep-lean-pair.json` records the reviewed companion revision — the
`ActiveInferenceInstitute/fep_formal` repository and a full 40-hex SHA from its
main branch that the GNN contract was checked against. That file is the live source of truth
for the current pin; read it directly. This doc deliberately quotes no SHA,
so it cannot go stale the way a quoted pin does.

In the mirror direction, GEO_INFER pins GNN from its own checkout via its
`.github/gnn-pair.json` (see [GEO_INFER direction](#geo_infer-direction)).

## What runs

`.github/workflows/fep-lean-paired-revision.yml`:

1. Reads and validates the pin (known repository slug, 40-hex revision).
2. Checks out FEP_Formal at exactly the pinned SHA.
3. Records both checkout revisions plus the pin file as receipts, retained as
   workflow artifacts (`if: always()`, mirroring GEO's receipt semantics).
4. Installs each Python 3.11/3.12/3.14 row once with `uv sync --frozen`, explicit
   `UV_PYTHON` and `UV_PROJECT_ENVIRONMENT`. Every subsequent invocation uses
   the workflow's frozen, no-sync uv command and explicit matrix Python.
   Unqualified `uv run` can select the companion's local `.python-version`
   and replace the installed environment; the matrix label alone is not
   execution evidence.
5. Runs FEP_Formal's **read-only** bridge surface against this GNN checkout:

   - `fep-formal bridge status --gnn-root <gnn checkout>`
   - `fep-formal bridge emit --check --model finite --gnn-root <gnn>`
   - `fep-formal bridge emit --check --model continuous --gnn-root <gnn>`

   `status` and `emit --check` never write: they compare source bytes and
   emitted-document digests only.

The workflow-local [runtime runner](../../.github/fep_paired_runtime.py)
calls the installed canonical `fep_formal.cli.main` in the inspected Python
process, with deprecated-layout compatibility for older pinned checkouts.
It records the requested and actual major/minor, full interpreter version,
executable and prefix, virtual-environment inode, interpreter/link identity,
`pyvenv.cfg`, both source revisions, lock hashes and tracked-file hashes.
The installed baseline must remain identical before and after each command.
All prior successful receipts and bridge-output digests are checked at the
next stage and again at the final stage; missing, changed, failed or
mislabelled evidence fails closed. Imports must come from the pinned FEP
source tree. JSON bridge outputs and additive failure receipts are retained
without replacing earlier evidence. `native_claim_ready` stays `false`.

The [focused controls](../../.github/test_fep_paired_runtime.py) run real
subprocesses, virtual environments and temporary Git repositories, retaining
their fixture paths. They reject wrong interpreters/environments, replacement
or modified environments, source revision/byte drift, changes during execution,
nonzero CLI exits, missing/wrong receipts and changed output bytes. The fixture
CLI checks the custody mechanism; the subsequent hosted stages execute the
actual pinned bridge. Run the controls with Python 3.11, 3.12 or 3.14 and place
`TMPDIR` on the storage volume being qualified.

The Python 3.14 row then runs the pinned Q7 verifier's read-only
`source_snapshot` against the same GNN checkout. It requires all five prior
runtime receipts, checks current render provenance, input/fixture equality,
owner maps, scaffold and generated-manifest freshness, and preserves the same
installed-environment and tracked-source baseline before and after execution.
The source snapshot and failure diagnostics are retained with the pair receipts.
This closes the gap where a source-pin/emission refresh could leave Q7's current
render metadata stale while ordinary bridge status and emission checks passed.
The Q7 parser accepts CPython 3.14; the 3.11/3.12 bridge compatibility rows remain.
This additional gate performs no Lean compilation or generated-runner execution
and keeps `native_claim_ready: false`.

Strictness mirrors GEO: the job is blocking. A red run is the drift signal —
the pair must be re-pinned together, not the check suppressed.

## Who bumps the pin

Pin updates coordinate source pins in FEP_Formal:

- **GNN side**: changing GNN sources that the bridge's source binding covers
  requires a paired commit on FEP_Formal (per its FEP-EVIDENCE-CURRENT policy)
  that re-records digests against the new GNN state; the GNN-side PR then
  bumps `.github/fep-lean-pair.json` to the new FEP_Formal revision in the same
  change set.
- **FEP_Formal side**: when FEP_Formal main moves, this repository's pin is
  updated to the new reviewed FEP_Formal SHA together with any GNN-side source
  changes the new pin requires.

A green run therefore certifies agreement between two specific, named
revisions — never a floating main.

## What the workflow proves and does NOT prove

A passing current-source workflow is a **drift-detection** result. It establishes that the pinned
FEP_Formal checkout and this GNN checkout still agree on the bridge's source
binding (owner rosters and content digests), finite/continuous freshness, the
syntax surface, the contract mirror, and the formal projection with no drift.

It does **not** prove any Lean theorem, and no notation mapping claim may
promote it into one: the bridge compares symbols and constructs across the two
repositories; correspondence of notation is a documentation-level statement.
Evidence planes stay distinct — a green paired run says "the two checkouts are
in the state FEP_Formal reviewed", nothing about the truth of formal claims.

## Closeout ordering (canonical)

Changes touching files the bridge's source binding covers close out in this
order. The [historical fixed-baseline policy](#fixed-baseline-for-gnn-421) does
not exempt a current release from this sequence:

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
3. **FEP_Formal bridge and current-Q7 re-pin**: bridge pin → emit
   `--refresh-digests` → emit `--check` (finite + continuous) → canonical Q7
   `refresh_render.py` against that exact GNN checkout → Q7 probe generation
   and `--check` → fresh current Q7 native compilation and retained-check validation at a new
   receipt path → PR → merge on the FEP_Formal side. Include the current fixture,
   render provenance and generated manifest; ordinary bridge checks alone do
   not validate these derived inputs. Preserve historical native receipts and
   the reviewed runner/scaffold constraints. Do not hand-edit their hashes or
   promote an earlier native receipt to the new source pair.
4. **GNN pin bump as the FINAL commit**: bump `.github/fep-lean-pair.json` to
   the new reviewed FEP_Formal revision in the final commit, then push that final
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
- FEP_Formal bridge contract (GNN-side docs): `docs/other/fep_lean/bridge-contract.md`

# Release and cross-repository custody

Use this guide for manuscript refresh, companion pins and publication.
[Canonical ordering](../docs/development/fep_lean_paired_revision.md) and
[producer docstring](../scripts/z_generate_manuscript_variables.py) own the detailed
procedure. Read them before editing source-bound evidence.

## Content and producers

Finish content edits first. Determine bridge-owner, manuscript-variable, hydration
and figure inputs. Commit content normally. Producers remain reachable in published
history; never amend/orphan them after recording. Keep source/artifact commits
separate. Push a fetchable branch/draft for companion checks. Later owner/input
edits require a new evidence epoch.

## Manuscript ritual when inputs change

1. Regenerate variables/hydration at the committed source.
2. Rebuild manuscript figures.
3. Run the pinned template's full `stage_03_render` in its own environment.
4. Check freshness and record custody.
5. Verify disk evidence and normally commit artifacts.
6. Verify the committed chain and tests/gates.

Record-only/accept-only refresh cannot replace a render. Preserve rejected logs,
output and diagnostics. The verifier compares committed/disk chains differently
before/after artifact commit; follow its owner and retain actual results.

## Companion ordering

For bound changes: freeze GNN source/artifacts, re-pin/verify finite/continuous
companion bindings, merge its accepted PR, then bump GNN's `.github/fep-lean-pair.json`
as the final commit. Exact paired/repository gates precede GNN main merge.
GEO_INFER pins accepted GNN from its side. Preserve the reviewed direction and
compatible relationship without circular pairing. Custody proves named-byte
agreement, not a theorem or native scientific result.

## Publication

Resolve repository/version/tag/metadata/assets from accepted main and its release
procedure. Check existing version/tag/release; report conflicts instead of overwriting.
Require applicable source, native, installed-platform, security, manuscript,
companion, PR/main and tag gates at their exact sources. Pending/missing required
checks keep publication pending. Build ordinary wheel/sdist and source-bound
manuscript/visual/evidence assets.

Verify archives, installed imports/resources outside checkout, metadata, hashes
and receipt bindings. Notes/images describe actual behavior, measured checks and
remaining limits. After publishing, monitor tag CI and download every asset to
verify name/size/hash. Record URL/source/tag/checks/assets. DOI/archive claims need
completed archival evidence; a reservation or credential is insufficient.

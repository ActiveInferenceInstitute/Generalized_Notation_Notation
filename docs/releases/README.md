# GNN release management

Use this page for publication, [TO-DO.md](../../TO-DO.md) for future scope, and
[VERSION_MAP.md](../VERSION_MAP.md) for the compact release history.

## Release Management

Package versions follow semantic versioning: patch releases repair compatible
behavior and documentation, minor releases add compatible capabilities, and major
releases carry breaking changes. A roadmap target does not establish publication.
The accepted Git tag and directly verified public assets define each release.

## Quick Navigation

- [Repeatable release qualification](#repeatable-release-qualification)
- [Canonical manuscript and companion order](../development/fep_lean_paired_revision.md#closeout-ordering-canonical)
- [Current workflow roster](../../.github/workflows/README.md)
- [Exact-version archive and DOI procedure](exact_version_archival.md)
- [Testing](../testing/README.md), [security](../security/README.md), [agent guide](AGENTS.md)

## Repeatable release qualification

The package version is owned by `pyproject.toml` and mirrored by
`gnn.__version__`, the frozen lock and citation metadata. Prepare a release
candidate on an isolated branch; finish content first, then perform the
[manuscript and companion ritual](../development/fep_lean_paired_revision.md#closeout-ordering-canonical).
GNN 5.0.x uses strict current-source pairing with its reviewed FEP_Formal pin.
A candidate version or workflow migration does not establish synchronized-pair
qualification or publication; the actual source, native, manuscript and hosted
gates must pass. The historical 4.2.1/4.3.0
[fixed-baseline policy](../development/fep_lean_paired_revision.md#fixed-baseline-for-gnn-421)
and its exact [4.3.0 policy bytes](fep_4_3_0_baseline.json) remain archival
evidence, without an active release exception. Keep original producer commits
reachable. Never renew receipts by editing hashes.

After provisioning the exact uv version from README, run `just release-preflight`
before publishing a candidate. It installs the frozen dev profile once, then
checks the canonical exit-code documentation contract before source types,
documentation and module inventories, actual flags, thin scripts, release
scheduling controls and every direct entrypoint. It catches deterministic mistakes
before starting expensive hosted lanes; full/native/manuscript/companion qualification remains
required. Existing `just quality` retains its broader local gate roster.

Use the stdlib GitHub CLI collector from the repository root:

```bash
python .github/release_gate.py --revision FULL_COMMIT --ref RELEASE_BRANCH \
  --dispatch-missing --output /absolute/task/work/candidate-observation-1.json
python .github/release_gate.py --revision FULL_COMMIT --ref RELEASE_BRANCH \
  --output /absolute/task/work/candidate-observation-2.json
```

The first command launches only workflows absent from the visible Actions
history for that exact SHA; existing pending or failed work is reported without
redispatch. Use a single dispatch owner and retain its receipt while the provider
registers the runs. Subsequent observations use new output files and omit
`--dispatch-missing`. Exit 1 means incomplete qualification; API/access errors
remain errors. Every latest run/attempt and all its paginated jobs must succeed.
The collector does not equate PR synthetic-merge tests with native artifact
source identity. Inspect each source-bound artifact independently.

After the accepted PR is merged normally, verify main has exactly the qualified
Git tree and preserve its producer ancestry. Create the annotated candidate tag
at that exact main commit. Main and tag qualification may run concurrently;
publication stays blocked until both have terminal successful checks and their
actual artifacts are independently accepted. Never move the tag or count a
main run as tag evidence. All sixteen release workflows run automatically on
version tags. Collect each ref separately with
`--ref main` for main's automatic and explicitly dispatched profiles, and
`--event push --ref vVERSION` for the automatic tag roster. The collector filters
the actual run ref and requested event, even when both point to one commit. Retain failed
attempts; repair functional failures and only retry classified infrastructure
failures with evidence. Do not lower test selections, coverage or warning policy.
External links run at tag qualification with dead links blocking and transient
or blocked responses remaining inconclusive.

A companion's actual main qualification may similarly overlap GNN's final
candidate checks after the companion merges and GNN pins its exact merge.
Both must finish and pass independent acceptance before merging GNN.

Build ordinary wheel and sdist from a clean accepted checkout using the pinned
uv tool. Verify metadata/version, all source/archive members and wheel RECORD;
install outside the checkout and exercise the public CLI, imports and resources.
Publish a GitHub draft containing the wheel, sdist, manuscript, source binding,
verification receipt and SHA256SUMS; verify tag CI and the uploaded bytes before
publishing. Download every public asset afterwards and record parity. GitHub
release publication is the existing distribution channel. PyPI and exact-version
archival require their own destination/owner credentials and verified receipts;
no publication to those services is implied by a GitHub release.

This scheduling change removes manual roster coordination and redundant ready
transitions. It does not claim faster inference or a measured CI speedup.

## Software 4.2.0 release record

The [v4.2.0 release](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/releases/tag/v4.2.0)
owns its exact tag, accepted source, hosted checks and downloaded asset receipts.
Version metadata alone does not establish publication. It includes RxInfer 6.1,
current compatible dependencies, direct-script startup without stdlib module
shadowing, aiohttp/Pillow repairs and the repeatable
qualification procedure below. Python, Gradio/Pillow and scientific backend
exceptions retain the scopes documented in README and CHANGELOG. PyPI and
separate distribution archival require independent publication evidence.

## Historical software 4.1.0 release record

The [v4.1.0 release](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/releases/tag/v4.1.0) is the authoritative record of its
tag/source/checks and ordinary wheel, sdist, manuscript, verification records
and [checksums](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/releases/download/v4.1.0/SHA256SUMS).
Acceptance requires actual terminal gates and directly downloaded asset parity;
version metadata alone does not establish publication.
The manuscript authored epoch and historical release artifacts retain their
original identities. Exact-version archival follows the
[archive procedure](exact_version_archival.md) and its own public DOI/hash evidence.

## GNN 4.0.1 publication (historical)

[v4.0.1](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/releases/tag/v4.0.1) was published on 2026-10-07 from
`17c72cf0f98d7d3bbf0159d1b1cce8c77c4e4daf`. Its
[publication receipt](../development/gnn_4_0_1_post_publication.json) records
normal companion integration, the annotated tag, exact hosted checks, zero open
issues and enabled Dependabot/CodeQL alerts, and ten directly verified assets.
The immutable release PDF retains its authored date; current main publication
metadata is renewed separately. No PyPI upload was performed. The subsequent
[exact 4.0.1 source archive](https://doi.org/10.5281/zenodo.23222085) has
[verified source/manuscript parity](../development/zenodo_4_0_1_exact_source_archive.json);
separately attached distribution-asset archival remains pending under the
[exact-version procedure](exact_version_archival.md).

## GNN 4.0.0 publication (historical)

[v4.0.0](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/releases/tag/v4.0.0)
was published on 2026-10-07 from `1bc3a76eccccb2cc5ce8601770714075b3ec48db`.
The [publication receipt](../development/gnn_4_0_0_post_publication.json) records
normal companion merges, the annotated tag, exact checks and downloaded asset
hashes. The manuscript keeps its 2026-10-02 authored date; no PyPI upload or
version-specific archival DOI is claimed.

![GNN 4.0.0 overview of model notation, numerical execution and the 25-step workflow](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/releases/download/v4.0.0/GNN-4.0.0-release-overview.png)

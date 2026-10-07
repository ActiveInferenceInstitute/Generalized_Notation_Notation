# Exact released-version archival

The project concept DOI is [10.5281/zenodo.7803313](https://doi.org/10.5281/zenodo.7803313).
It identifies the version family and resolves to the latest published record.
A specific software release needs its own version DOI and directly verified
public archive. The [2023 article record 7803328](https://zenodo.org/records/7803328)
is a historical manuscript publication, not a GNN 4.x software version.

## Verified GNN 4.0.1 source archive

The existing GitHub/Zenodo integration published
[record 23222085](https://zenodo.org/records/23222085), version `v4.0.1`, on
2026-10-07. Its exact version DOI is
[10.5281/zenodo.23222085](https://doi.org/10.5281/zenodo.23222085).
The DOI resolves publicly with matching software version metadata and a
supplement relation to the `v4.0.1` GitHub tag.

Directly downloading the public source ZIP established:

- Provider MD5: `aa9637842095c2e5e7a9c8788baeb8cc`; downloaded bytes match.
- ZIP SHA-256: `c2e74f9a5dfe0edf3356922f71eb5517cad33232ecc5752cac7e38aea1b6c227`.
- Source parity: all 3,169 tracked Git blobs match the peeled release commit
  `17c72cf0f98d7d3bbf0159d1b1cce8c77c4e4daf`, with no missing or different blobs.
- Included citation metadata, dependency lock and committed manuscript PDF.
  The archived PDF is byte-identical to the directly downloaded GitHub release PDF.
- Manuscript SHA-256: `56934a5bf5cc8633ec9b18a51d7417520715343f3dd6c0185c2ceedbaca6d056`.

The [archive verification receipt](../development/zenodo_4_0_1_exact_source_archive.json)
records the public checksum, DOI resolution and source parity.

The GitHub integration source ZIP does **not** include the separately attached
GitHub release wheel, sdist or `SHA256SUMS`. Source archival is verified;
complete distribution-asset archival has additional acceptance requirements.
No v4.1.0 DOI has yet been observed or assigned here.

## GNN 4.1.0 archival acceptance

After the release owner accepts the exact source, annotated tag and publication
assets, check the concept DOI for a new record with exact `v4.1.0` metadata.
A successful GitHub release can trigger the already configured integration;
verify that result rather than treating release publication as archive success.
Download the new ZIP from its version-specific record, check the provider
checksum, compare each Git blob to the peeled release commit and match the
manuscript bytes to the accepted release artifact.

For the full archival scope, the provider package also needs the accepted
wheel, sdist, manuscript, `SHA256SUMS` and citation metadata. Preserve the original
filenames and compute SHA-256 from every upload and direct public download.
Metadata must name version `4.1.0`, the exact repository/tag, the existing
credited creators and declared license; preserve the concept/version DOI
relationship. Do not reuse a historical DOI as a current software DOI.

Prepare a manifest containing release commit, annotated/peeled tag IDs, provider
record/version DOI, exact filename, size and SHA-256 for every artifact. The
software wheel must already pass installation outside the checkout and the
source distribution must match the accepted source. Verify public DOI resolution,
version metadata and artifact hashes only after provider publication succeeds.
Update canonical citation/release metadata together, while retaining historical
version DOI references in prior release records.

Adding files or publishing a new provider record requires the archive owner's
connected, authorized provider access. Read-only public record access does not
grant upload authority. Stop for an actual provider authentication or account
ownership blocker; never modify credentials, permissions, legal settings or
payments. A PyPI upload remains a separate publication with an identified
package destination and owner.

# FEP_Formal collaboration documentation

This folder retains its historical `fep_lean` path and linked filenames.
Current public imports use `fep_formal`, commands use `fep-formal`, and public
settings use `FEP_FORMAL_`. Legacy names are deprecated aliases through 2.x.

## Ownership

The canonical bridge contract lives at
`../fep_formal/docs/design/gnn-bridge/bridge-contract.md`. Edit it there first,
then synchronize this folder's [mirror](bridge-contract.md) in the same session.
Normalized bodies must match; only the canonical/mirror location rows differ.
Keep cross-repository paths in code, and local documentation links relative.

## Evidence and maintenance

Historical Q5/Q6/Q7 receipts describe their original source pairs. Current
qualification needs an explicitly named pair and fresh artifact-specific source
and native checks. Syntax, numerical agreement, native proof and generated-runner
execution remain separate evidence planes. Do not promote old receipts.

The [reader entry](README.md), [overview](fep_lean.md), [program](fep_lean_gnn.md)
and [scope](SPEC.md) retain dated program history. The current integration
workflow is owned by the [paired-revision procedure](../../development/fep_lean_paired_revision.md).
After edits, run the strict documentation audit with anchors and no writes:
`uv run --frozen --no-sync python docs/development/docs_audit.py --strict --check-anchors --no-write`.

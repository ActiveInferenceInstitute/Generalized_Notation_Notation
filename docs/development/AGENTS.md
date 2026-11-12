# Development documentation — agent guide

## Overview

Maintainer workflows, repository audit tools and source-bound release evidence live
here. [README.md](README.md) routes human contributors; [SPEC.md](SPEC.md) defines
this directory's documentation contract. This is not a pipeline runtime module.

## Purpose

Keep commands tied to executable owners and scientific acceptance tied to named
source revisions. Use [TO-DO.md](../../TO-DO.md) for current and future scope;
historical receipts and fleet logs retain their original epochs.

## Contents

Use the [developer guide](README.md) for task routes, [documentation review](documentation_review.md)
for prose and signposting, and [paired custody](fep_lean_paired_revision.md) for
manuscript/companion updates. The tree also contains retained JSON receipts and
`fleet-logs/`; a hand-maintained file count is not authoritative.

## Quick Navigation

- **Sibling-project contracts**: [Interoperability](interop.md), [Typed/API contracts](interop_api.md), [Benchmarks and provenance](interop_research.md); open work in [TO-DO.md](../../TO-DO.md)
- [Documentation audit tooling](README.md#documentation-audit-tooling)
- [Contributor handoff](contributor_handoff.md)
- [Roadmap implementation contracts](remaining_roadmap_contracts.md)
- [Testing](../testing/README.md), [release management](../releases/README.md)
- [Main documentation](../README.md), [pipeline owners](../../src/gnn/README.md)

## Verification

Run the frozen checks in [README.md](README.md#documentation-audit-tooling), the
relevant consumer tests, and every applicable source/native/custody gate. Never
refresh a receipt by changing its recorded producer identity.

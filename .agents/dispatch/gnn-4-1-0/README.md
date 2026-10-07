# GNN 4.1.0 integration handoff

The user authorized completing every minor and medium workstream in the
[existing roadmap](../../../TO-DO.md) toward 4.1.0. That file owns scope and
acceptance; this handoff records integration ownership and outstanding evidence.
Major scientific extensions remain outside this release campaign.

Frozen starting revision: `e5461954f45314cfea77b2cf06818e060f68bd0a`.
The integration branch is `codex/gnn-4.1.0`. Implementation workers use separate
worktrees and return local commits; the integrator owns publication and custody.

| Owner | Delivery boundary |
| --- | --- |
| Integrator | M5 global documentation, independent reviews, roadmap closeout, version surfaces, exact-source checks, manuscript/FEP/GEO custody and release |
| contracts410 | M2/M3 public admission, shared run/configuration/outcome contracts; render dispatch without scientific algorithm changes |
| ownership410 | M1/M4 logging and website owners, RxInfer interchange/analysis, bounded authored-parser defects |
| security410 | S5 filesystem operations, leases, owned worker cancellation and precise native containment guarantees |
| science410 | M6 presentation/trace identity and M7 measured production-path bottlenecks |
| coverage410 | S6 reproducible per-environment gap inventory and meaningful consumer/failure tests |
| platform410 | M8 installed-package/native platform acceptance, S3 released upstream verification and E5 archive evidence |

Keep focused tests and exact baselines with each checkpoint. Existing public
exports, scientific/source contracts, required checks and coverage denominators
remain acceptance constraints. Record decisions in [choices.md](choices.md).
The integrator performs independent review before accepting worker commits.

## Dependencies requiring external evidence

- S3: stock released THRML 0.1.4 still fails inactive-padding witnesses on both
  supported JAX splits. Upstream issue closure is not a published compatible
  fix. Retain strict rejection until ordinary released installation passes.
- E5: the public 4.0.1 version archive is
  [Zenodo 23222085](https://zenodo.org/records/23222085), separate from concept
  DOI `10.5281/zenodo.7803313`. Its source archive matches the exact tag but
  does not include attached wheel/sdist/checksum release assets. Verify the
  selected 4.1.0 artifact archive directly before closing this workstream.
- M8: local Python 3.14/macOS evidence does not replace fresh native Linux and
  Windows lane results. Unsupported descendant guarantees must refuse.
- S6: baseline coverage is approximately 70.8% across 80,400 statements.
  Preserve scope and report each environment; the >80% goal needs direct proof.

## Integration and publication order

Freeze accepted content and version metadata, commit the counted source, then
perform the full manuscript variables/figures/template-render/custody ritual
from [root AGENTS](../../../AGENTS.md). Follow the
[paired-revision procedure](../../../docs/development/fep_lean_paired_revision.md)
for FEP/GEO-bound owners. Commit the GNN FEP pin last. No amendment may invalidate
custody. Publish a reviewable frozen branch when companion/native hosted checks
need it, then merge only after required exact-revision gates pass.

Release requires a unique nonconflicting tag, accepted wheel/sdist installation,
public manuscript/checksum artifacts and terminal exact-tag hosted checks.
Remove only accepted completed work from TO-DO; retain externally blocked work
with precise evidence. Never treat a pending check or partial archive as done.

## NextAgentPrompt

Resume the owned `codex/gnn-4.1.0` integration worktree. Read this handoff,
choices.md, existing TO-DO, current Git state and active worker messages before
edits. Preserve uncommitted work and accepted local commits. Collect outstanding
worker deliverables, review their consumer evidence and integrate clean commits
in dependency order. Complete all independent minor/medium scope before any
external blocker decision. Refresh the final source, manuscript, companion,
native-platform, coverage and release evidence after the last content change.
Do not release or claim completion until the exact revision and public artifacts
are verified. Do not broaden into major semantics or unrelated repositories.

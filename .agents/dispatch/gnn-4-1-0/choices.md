# GNN 4.1.0 decision audit

Each checkpoint records its consequential decisions, consumer evidence and
remaining uncertainty. Status describes the decision, not release completion.

| Decision | Audit | Reason and evidence |
| --- | --- | --- |
| Use existing TO-DO as the release scope | Sound; high confidence | User explicitly requested all minor/medium work. A separate plan would duplicate its acceptance criteria; major semantics remain excluded. |
| Preserve public exports/signatures and scientific semantics during ownership extraction | Sound; high confidence | Existing consumer tests and generated-byte parity can demonstrate behavior preservation; file length alone cannot justify a new boundary. |
| Rewrite stale development guidance to cite live owners | Sound; high confidence | Old rules prescribed forced success, unsafe deletion of publication output, unregistered flags and noncanonical imports. Four documentation gates passed. Actual discovery, CLI, MCP and Steps 3/5 examples were exercised; two invalid draft commands were corrected. Health and the pipeline retain qualified warning outcomes. |
| Reject explicit empty executing-step requests while preserving empty model selections | Sound; high confidence | Empty steps currently expand to every step in some entrypoints. Discovery remains permissive; a selected empty model view remains valid no-work. Verify all public consumers and preserve their documented errors. |
| Keep THRML structural-zero rejection | Sound; high confidence | Ordinary released 0.1.4 witnesses still produce NaNs on both supported JAX splits. Upstream closure is insufficient release evidence. |
| State Windows direct-worker containment precisely and refuse unsupported descendant guarantees | Sound; high confidence | A global process census violates existing ownership invariants; post-spawn job assignment leaves a child escape interval. Native tests must establish the stated narrower guarantee. |
| Preserve coverage scope and per-environment reports | Sound; high confidence | The >80% goal requires additional meaningful evidence. Unions across environments, arbitrary exclusions or deleting dormant code solely for coverage would misstate acceptance. |
| Treat the verified Zenodo source archive separately from distribution archival | Sound; high confidence | Exact 4.0.1 source parity is verified, but wheel/sdist/checksum release assets are absent from that archive. E5 remains open until selected distributions have public hash parity. |
| Require an explicit API workspace outside a real checkout | Sound; high confidence | Independent security review found the unset installed default would authorize the Python library directory. Reject that default while preserving checkout use and actual bad-request errors; installed native acceptance must include this negative case. |
| Accept descriptor operations under precise trusted-parent limits | Sound; high confidence | Independent review checked creation, lease and deletion owners and real coordinated replacement tests. Workspace/ancestor deletion and deletion after failed cleanup are refused. The root integrated native suite passed 63 cases; genuine hosted platform evidence remains pending. |
| Retain the unexplained macOS child-watcher failure | Sound; high confidence | A previous run correctly reported failed cleanup. Ten fresh exact-case repeats traced only the asyncio leader reaper and passed; these do not prove the intermittent condition fixed. Preserve the failed log and monitor final native gates. |
| Measure subprocess coverage separately before choosing a floor | Sound; medium confidence | Existing behavior tests invoke Python children absent from core coverage. A provisioned comprehensive report may trace those executions using the same interpreter and source scope; acceptance requires actual per-environment reports and unchanged core evidence. |

Update this ledger after each implementation/review pass. Evidence from an old
head cannot certify a changed source or final release.

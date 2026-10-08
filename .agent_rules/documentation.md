# Documentation and rules maintenance

Use this guide for prose, examples, indexes and rules. [Root AGENTS](../AGENTS.md)
owns README/AGENTS/SPEC conventions; [docs audit](../docs/development/docs_audit.py)
checks links/anchors/structure.

## One owner per fact

Link versions to metadata, dependencies to metadata/lock, inventories to registries,
gates to workflows and future work to [TO-DO.md](../TO-DO.md). Receipts retain their
historical source. Candidates are not published releases. Topic guides state scope,
owners, behavior and verification. Keep [README](README.md) routing synchronized,
stable filenames/fragments and cross-topic links. Avoid second roadmaps or duplicated
inventories and timing targets.

## Examples and links

Use real `gnn.*` imports and installed `uv` commands. Verify flags against parser/help
and examples through the public path. Pipeline directory targets differ from file
APIs. Check relative links/fragments in context, including hidden directories.
Runnable models need complete sections and admitted fixture parameters.

## Maintained audits

After frozen dev setup:

```bash
uv run --frozen --no-sync python docs/development/docs_audit.py --strict --check-anchors --no-write
uv run --frozen --no-sync python scripts/check_repo_terminology.py --strict
uv run --frozen --no-sync python scripts/check_maintained_doc_terms.py --strict
uv run --frozen --no-sync python scripts/check_doc_contracts.py --strict
uv run --frozen --no-sync python scripts/check_gnn_doc_patterns.py --strict
uv run --frozen --no-sync python scripts/check_doc_path_references.py
uv run --frozen --no-sync python scripts/check_flag_parity.py
uv run --frozen --no-sync python scripts/check_capability_contracts.py --strict
```

`--no-write` preserves the report. Run examples and mandatory hosted gates too.
Static link checks alone cannot establish repository acceptance.

## Manuscript and custody

Determine count, hydration, figure and companion-owner impact. Complete
[release.md](release.md)'s ritual after content freeze when inputs change. Never
relabel old producer identities. The PR baseline guard admits only identical
diagnostics with identical supporting bytes; new/worse drift and record-only
refresh remain failures. Main/manual acceptance is strict.

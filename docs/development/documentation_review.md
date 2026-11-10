# Documentation review

Keep the public landing pages easy to use and technical guides consistent with
their executable owners. [Documentation rules](../../.agent_rules/documentation.md)
define the maintenance contract; [TO-DO.md](../../TO-DO.md) owns future work.

## Review the reader's path

Start with the [GitHub landing page](../../.github/README.md),
[root README](../../README.md) and [documentation hub](../README.md). Verify that
a new reader can choose a model, install its runtime, run a documented command
and find the resulting evidence. Use complete committed examples and describe
backend admission and optional dependencies where they affect that task.

Link inventories to registries, versions to metadata and test selections to
workflows. Keep one owner for each fact. Retire obsolete plans, duplicate
instructions and expired snapshots from the current tree; preserve cited
historical evidence through links to its exact Git revision or release receipt.

## Run the maintained checks

From a development checkout:

```bash
uv sync --frozen --extra dev --python 3.14
uv run --frozen --no-sync python docs/development/docs_audit.py --strict --check-anchors --no-write
uv run --frozen --no-sync python scripts/check_doc_contracts.py --strict
uv run --frozen --no-sync python scripts/check_maintained_doc_terms.py --strict
```

The anchor check covers same-page links, cross-file headings, duplicate heading
suffixes, Setext headings and explicit HTML targets. Inline code and fenced
examples are excluded from link extraction, and fences are excluded from
headings. The scan checks multiline Markdown links, relative reference
definitions, URL-encoded paths, and destinations that escape the repository.
Executable examples and external URLs have separate checks.

The [audit implementation](docs_audit.py) defines maintained paths, generated-data
exclusions, pairing and anchor checks. Follow the
[complete verification rules](../../TO-DO.md#verification-and-execution-rules)
for the affected surface. Run examples through their public entrypoints and
inspect rendered pages and images; static links alone do not verify their behavior.

## Check publication impact

Documentation counts, manuscript inputs and figure sources can affect scientific
artifacts. Follow the [custody procedure](../../.agent_rules/release.md) when those
inputs change. Never relabel a historical result as evidence for current source.
Record the final commit, checks, outputs and remaining limits using the
[handoff checklist](contributor_handoff.md).

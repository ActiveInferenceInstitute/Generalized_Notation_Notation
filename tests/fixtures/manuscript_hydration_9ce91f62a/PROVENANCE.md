# manuscript hydration fixture — generated 2026-11-04

Recorded output of the template injector, for pinning the headless mirror in
`src/gnn/manuscript/substitution.py` to a known behaviour.

- **Template revision:** `docxology/template` at
  `9ce91f62af3c6dbd8c031b6f80e2253f6c59f964` — the revision the 2026-11-04 GNN
  custody chain was recorded against.
- **Function:** `infrastructure.rendering.manuscript_injection.write_resolved_manuscript_tree(project_root, variables)`.
- **Input:** `project/manuscript/00_fixture.md`, with `tokens.json` as the
  variable map (one resolvable token, one unknown token that must survive
  verbatim).
- **Recorded output:** `expected/output/manuscript/00_fixture.md`, copied
  verbatim from that function's write at the revision above.

The copy's relative targets are rebased for its location — one level below the
project root — so `](../output/figures/x.png)` becomes `](../figures/x.png)`,
while an unrelated relative link is untouched. The source files keep their own
form.

Regenerate with that template revision on `sys.path`:

```bash
cp -R project /tmp/hydration-project
TEMPLATE_REPO_ROOT=<template-checkout> python - <<'PY'
import json, sys
from pathlib import Path
from infrastructure.rendering.manuscript_injection import write_resolved_manuscript_tree

tokens = json.loads(Path("tokens.json").read_text(encoding="utf-8"))
write_resolved_manuscript_tree(Path("/tmp/hydration-project"), tokens)
PY
```

A template move that changes this copy earns a **new** fixture directory, never
an in-place edit, so this one keeps measuring the behaviour it recorded.

## Placeholder link targets (added 2026-11-04)

`project/` also carries the five files this input's relative links name:
`output/figures/fixture_figure.png`, `output/figures/fixture_inline.png`,
`output/data/fixture_table.json`, `figures/fixture_figure.png` and
`elsewhere/fixture_note.md`. They exist so the repository's own
`docs/development/docs_audit.py --strict` link check resolves the fixture's
links — the `markdown-audit` job runs it on every push, and it caught the four
dangling targets this fixture was committed with.

**No recorded byte changed.** The source, `tokens.json` and the recorded output
are byte-identical to the fixture's first commit; the added files are targets,
read by no assertion, and the fixture still measures the behaviour it recorded.
The alternative — excluding fixture paths from the link audit — was refused:
that is re-scoping a gate to fit a fixture.

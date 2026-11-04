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

# docs/workflows — agent guide

## Overview

Cross-repository workflow contracts. One Markdown page per group, each paired
with a `workflow_groups` block in [../downstream/importers.json](../downstream/importers.json).

- [gnn-fep.md](gnn-fep.md) binds the twenty-two GNN consumers to one pin reader,
  `src/gnn/importers.py`, and records the copies each member still carries.
- The member table sits between `<!-- gnn-fep-members:start -->` and
  `<!-- gnn-fep-members:end -->`; `scripts/check_importer_registry.py` compares
  that table with the JSON block in both directions.

## Editing rules

- Change a member's gate, docs hub, manuscript producer or pin in the contract
  page and the JSON block in the same edit. The gate fails when only one moves.
- Cite a file and line for a copy you claim a member carries. A contract that
  cannot point at the copy cannot ask for its removal.
- Do not record a member's green state. Its own workflow owns that evidence; a
  contract records the mechanism and the migration step.
- The upstream repository is a member for inventory, not an importer of itself.

## Related

- [README.md](README.md) — the page index
- [../downstream/AGENTS.md](../downstream/AGENTS.md) — registry verification rules
- [../AGENTS.md](../AGENTS.md) — documentation tree guidance

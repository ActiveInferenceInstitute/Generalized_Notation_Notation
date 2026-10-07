# gnn.manuscript

Deterministic `{{...}}` token production for the GNN manuscript. The
producer reads the repository snapshot at the current commit and emits
`output/data/manuscript_variables.json`; the render pipeline hydrates the
manuscript from that map.

```python
from gnn.manuscript import generate_variables
```

Private helpers used by the behavior tests live in
`gnn.manuscript.variables`; see `AGENTS.md` for the module contract.

The local gates also verify hydrated prose and require the token map's commit
stamp in both rendered Markdown and TeX. These checks bind evidence; the stamp
alone does not prove the renderer ran or establish scientific correctness.

PR runs may supply `--base-ref TARGET_SHA` to either manuscript gate. Both sides
use the current audit implementation on immutable data at the merge base. Only
diagnostics with identical supporting bytes are inherited with a warning naming
the full base SHA. Changed or new evidence fails. Main and scheduled runs remain
strict: if the guard is red on main, fix main first.

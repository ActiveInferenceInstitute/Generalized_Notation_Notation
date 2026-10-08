# Model source standards

GNN Markdown declares an Active Inference generative model. Use current
[syntax](../docs/gnn/reference/gnn_syntax.md),
[schema](../docs/gnn/reference/gnn_schema.md),
[types](../docs/gnn/reference/gnn_type_system.md) and
[exemplars](../input/gnn_files/INDEX.md). Runnable examples have compatible
source dimensions, values and declared scientific semantics.

```bash
uv run --frozen --no-sync gnn validate input/gnn_files/discrete/two_state_bistable.md
uv run --frozen --no-sync gnn extract input/gnn_files/discrete/two_state_bistable.md
```

Lightweight discovery/parsing is narrower than Step 3 multi-format processing.
Its fields are `sections`, `variables` and `structure_info`; it is not complete
POMDP extraction or an independent semantic witness.

```python
from pathlib import Path
from gnn import discover_gnn_files, parse_gnn_file

sources = discover_gnn_files(Path("input/gnn_files/discrete"), recursive=True)
parsed = parse_gnn_file(Path("input/gnn_files/discrete/two_state_bistable.md"))
assert parsed["success"]
assert parsed["structure_info"]["has_variables"]
```

Finite categorical A/B/C/D[/E] parameters must be finite, dimensionally valid
and have declared probability mass. Canonical B axes are
`[next_state, previous_state, action]`; equal-sized axes still need semantic
orientation. Preserve component axes, source values and factor/agent custody.
Tiny literals must round-trip unchanged.

Linear-Gaussian models retain F/H/Q/R, prior mean/covariance and declared
controls. Covariance determines Gaussian uncertainty; categorical entropy is
not a substitute. Unsupported compositions refuse explicitly. See
[backend scientific admission](render_frameworks.md).

Derive formats from live parsers/exporters. Round trips compare actual model
semantics and source identity, not only file existence or matching shapes.

## Related contracts

[Scientific admission, comparison and proof boundaries](scientific_claims.md) owns the cross-cutting guidance.

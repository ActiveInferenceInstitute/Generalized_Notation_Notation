# RxInfer interchange specification

## Public contract

[`gnn.rxinfer_bridge`](../rxinfer_bridge.py) preserves `GRAPH_SPEC_FORMAT`,
`MARGINALS_FORMAT`, `GraphVariable`, `GraphEdge`, `GraphCPT`, `GraphSpec`,
`load_graphspec`, `load_graphspec_file`, `parse_gnn_subset`,
`render_gnn_subset`, `emit_rxinfer_jl`, `parse_marginals` and `write_marginals`.
Signatures and supported source/JSON/Julia behavior remain unchanged.

## Invariants

- GraphSpec validation refuses unknown keys/endpoints, duplicate edges/CPTs,
  cycles, noncanonical rows, mismatched parent sets/order and invalid probability
  values. Row mass tolerance remains `1e-6`.
- Source authoring consumes the existing strict Bayes-net GNN subset, distinct
  from POMDP tensor blocks and the numbered pipeline workflow.
- Julia emission retains evidence observation interfaces, deterministic names
  and ordering, and existing learning/structured-rule constraints.
- Printed marginal parsing preserves insertion order and rejects duplicate or
  malformed values. Sidecar row-sum tolerance remains `1e-6 + n_states * 5e-7`,
  accounting for the emitter's six-digit rounding.
- `jev_factors` remains a reserved, shape-validated field preserved in JSON and
  ignored for inference. It establishes no additional factor semantics.

## Dependency direction

The public facade imports these owners. `subset.py`, `julia_emitter.py` and
`marginals.py` consume `graphspec.py`; it imports no sibling owner. Each owner
is standard-library-only and performs no native execution during import.

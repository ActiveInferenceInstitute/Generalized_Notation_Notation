# RxInfer Bayes-net interchange owners

Use the released public API from [`gnn.rxinfer_bridge`](../rxinfer_bridge.py).
This internal package separates four responsibilities without changing the
`dafjev.bayesnet/1` data contract or its inference interpretation.

| Owner | Responsibility |
| --- | --- |
| [`graphspec.py`](graphspec.py) | Immutable variables/edges/CPTs, fail-closed validation, canonical parent assignment order and JSON loading |
| [`subset.py`](subset.py) | Strict Bayes-net GNN subset parsing and deterministic source rendering |
| [`julia_emitter.py`](julia_emitter.py) | Deterministic Julia code emission from a validated graph |
| [`marginals.py`](marginals.py) | Printed posterior parsing and the `dafjev.bayesnet-posteriors/1` sidecar |

The implementation uses the standard library only. Generated Julia and JSON
remain byte-compatible with the public facade's prior implementation.
Canonical CPT row order follows `itertools.product`, first parent slowest.
The reserved `jev_factors` field is preserved, with no new inference semantics.

The bridge is separate from the POMDP renderer/strategy system in
[`render/rxinfer`](../render/rxinfer/README.md). Native RxInfer limitations and
the finite verified gap matrix remain documented in the
[bridge examples](../../../examples/rxinfer/README.md).

Run the public consumer cases with:

```bash
uv run --frozen --no-sync python -m pytest tests/gnn/test_rxinfer_bridge.py -q
```

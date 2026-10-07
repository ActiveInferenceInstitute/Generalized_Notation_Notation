# RxInfer interchange agent guidance

Read [SPEC.md](SPEC.md) and [README.md](README.md) before changing these owners.
The supported import surface remains `gnn.rxinfer_bridge`; do not create another
public registry or alter scientific interpretation during maintenance.

`graphspec.py` owns types, ordering and validation. The source parser, Julia
emitter and marginal sidecar owner import it directly and do not import the
facade or each other. Keep the complete Julia emission algorithm together;
template line count alone is not a reason to divide it.

Changes to GraphSpec formats, probability rules, parent order or reserved
`jev_factors` semantics require coordinated daf-jev contract review. Preserve
failure messages and public signatures, exercise authored-source consumers,
and compare generated Julia/source/JSON bytes when extracting responsibilities.
Source changes also follow the repository's FEP/GEO and manuscript custody
procedure; implementation checkpoints are not publication acceptance.

The maintained consumer suite is
[`tests/gnn/test_rxinfer_bridge.py`](../../../tests/gnn/test_rxinfer_bridge.py).

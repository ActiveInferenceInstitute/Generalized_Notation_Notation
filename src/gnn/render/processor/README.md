# `gnn/render/processor/` — README

POMDP-aware rendering of GNN specifications into framework-specific
simulation code, with render receipt persistence and aggregate
`render_processing_summary.json` summaries.

Import from `gnn.render.processor` (or the `gnn.render` re-exports); the
leaf split (`metadata`, `parsing`, `receipts`, `pipeline`, `rendering`) is
an internal layout detail. NumPy is a soft dependency (`NUMPY_AVAILABLE`
flag); heavy POMDP-extraction imports are deferred to call time.

Module map, facade contract, provenance, and the gating tests are in
[`AGENTS.md`](AGENTS.md); the parent module's
[`AGENTS.md`](../AGENTS.md) covers the Step 11 surface.

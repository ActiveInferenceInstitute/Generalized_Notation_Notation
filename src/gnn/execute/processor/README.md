# `gnn/execute/processor/` — README

Step 12 execution processing for rendered simulation scripts: executable
script discovery under the render output, render-summary contract
enforcement, structured subprocess envelopes (security gate, sandbox,
benchmark repeats), and slim aggregate `execution_summary.json` persistence
plus per-script receipts.

Import through the package facade
(`from gnn.execute.processor import ...`); the leaf modules (`envelope`,
`workers`, `summary`, `single`, `pipeline`) are private to the package.
Layout, facade contract, patch-seam rules, and the gating tests are in
[`AGENTS.md`](AGENTS.md); the parent module's
[`AGENTS.md`](../AGENTS.md) covers the Step 12 surface.

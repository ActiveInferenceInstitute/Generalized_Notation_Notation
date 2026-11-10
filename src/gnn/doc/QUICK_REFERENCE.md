# GNN Quick Reference Guide

Run these commands from the repository root after the
[frozen installation](../../../README.md#quick-start).

## Quick Start Commands

```bash
uv run --frozen --no-sync gnn validate input/gnn_files/discrete/two_state_bistable.md --strict
uv run --frozen --no-sync gnn extract input/gnn_files/discrete/two_state_bistable.md --compact

uv run --frozen --no-sync gnn run \
  --target-dir input/gnn_files/basics \
  --output-dir output/quick-reference \
  --only-steps 3 5 6 7 8 \
  --skip-steps 0 1 2
```

Use a file for `gnn validate` and `gnn extract`, and a directory for
`gnn run --target-dir`. Inspect
`output/quick-reference/00_pipeline_summary/pipeline_execution_summary.json`
and the selected step directories. Use a distinct output directory for each
concurrent run.

## GNN Syntax Quick Reference

Start from the [two-state exemplar](../../../input/gnn_files/discrete/two_state_bistable.md)
and follow the [syntax reference](../../../docs/gnn/reference/gnn_syntax.md).
The core section headers are `GNNSection`, `GNNVersionAndFlags`, `ModelName`,
`StateSpaceBlock`, and `Connections`. Declare dimensions and types before
referring to variables in the connection block. `D>s` is directed and
`s-A` is undirected. Model values, time, and ontology annotations add
information used by downstream processing.

For categorical models, `A[o,s]` is the observation likelihood, `B` describes
transitions, and `D` is the initial-state prior. The
[parameterization guide](../../../docs/gnn/reference/gnn_syntax.md)
explains axis conventions and the meaning of `C` and `E`. Other model kinds
use their own declarations and admission rules.

## Configuration and Backends

Use `gnn run --help` for step selection, logging and worker controls.
The [configuration guide](../../../docs/configuration/README.md) describes
`input/config.yaml`. The
[framework availability guide](../../../docs/execution/FRAMEWORK_AVAILABILITY.md)
separates rendering from native execution and lists required runtimes. Check
the relevant model and backend contract before requesting Step 12.

## Troubleshooting

Run `uv run --frozen --no-sync gnn doctor` to inspect runtime capabilities.
Read the status and diagnostics for the current run before changing the
environment or increasing a timeout. See the
[troubleshooting guide](../../../docs/troubleshooting/README.md) for
setup, validation, and execution issues.

## Key Documentation Links

- [Operator quickstart](../../../docs/quickstart.md)
- [Pipeline and step boundaries](../../../docs/pipeline/README.md)
- [API guide](../../../docs/api/README.md)
- [Model examples](../../../input/gnn_files/INDEX.md)
- [Documentation index](../../../docs/INDEX.md)

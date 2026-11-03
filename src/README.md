# src/ — the `gnn` package

This directory holds exactly one package: [`gnn/`](gnn/README.md). There is no
`src.*` import surface — `gnn.*` is canonical, either from an installed wheel or
with `PYTHONPATH=src` in a checkout.

## What is here

- [`gnn/`](gnn/README.md) — package overview: format counts, validation levels
  and the module map.
- [`gnn/STEP_INDEX.md`](gnn/STEP_INDEX.md) — the 25 numbered step drivers
  (`gnn/0_template.py` … `gnn/24_intelligent_analysis.py`); each is a thin
  orchestrator that delegates to its module package.
- [`gnn/main.py`](gnn/main.py) — pipeline driver.
- Support packages beside the steps: `parsers/`, `schema_validator/`, `mcp/`,
  `pipeline/`, `manuscript/`, `utils/`, and one package per remaining step.

## Run it from a checkout

```bash
python src/gnn/main.py --target-dir input/gnn_files --verbose
python src/gnn/3_gnn.py --target-dir input/gnn_files --output-dir output --verbose
```

## Before editing

Read [`AGENTS.md`](AGENTS.md) here, then the guide beside the package you touch:
every `gnn/<package>/` keeps its own README and AGENTS pair. Changes in this
tree move counts the manuscript cites and can re-drift cross-repository custody,
so those rules are not optional.

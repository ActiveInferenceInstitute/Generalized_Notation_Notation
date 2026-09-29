# cpomdp Analysis Sub-module

## Overview

Reads `simulation_results.json` files the cpomdp scripts write under `<model>/cpomdp/simulation_data/` and produces per-model plots and a summary JSON.

## Files

- `analyzer.py`: `CPOMDP_SPEC` (a `FlatPayloadSpec`) and `generate_analysis_from_logs(results_dir, output_dir=None, verbose=False)`, which delegates to `analysis.flat_payload_analyzer`.
- `__init__.py`: re-exports both.

## Integration

- Registered in `analysis/processor.py`'s per-framework analyzer list.
- `cpomdp` is in `framework_common.FRAMEWORK_DIR_NAMES` and `viz_schema.VISUALIZATION_FRAMEWORK_DIRS`, so result paths resolve to the right model name, and in `viz_plots` and the cross-model report's framework order.

## Testing

`tests/analysis/test_cpomdp_analysis.py` checks path inference and that the analyzer writes the trajectory plot, the EFE plot and the summary.

## Documentation

- **[README](README.md)**: usage and outputs

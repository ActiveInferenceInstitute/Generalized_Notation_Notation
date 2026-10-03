# cpomdp Analysis Sub-module

## Overview

Reads `simulation_results.json` files the cpomdp scripts write under `<model>/cpomdp/simulation_data/` and produces per-model plots and a summary JSON.

## Files

- `analyzer.py`: `analyze_payload` and `generate_analysis_from_logs(results_dir, output_dir=None, verbose=False)`, using shared covariance-aware continuous result metrics.
- `__init__.py`: re-exports both.

## Integration

- Registered in `analysis/processor.py`'s per-framework analyzer list.
- `cpomdp` is in `framework_common.FRAMEWORK_DIR_NAMES` and `viz_schema.VISUALIZATION_FRAMEWORK_DIRS`, so result paths resolve to the right model name, and in `viz_plots` and the cross-model report's framework order.

## Testing

`tests/render/test_cpomdp_contracts.py` executes released-wheel Kalman/EFE cases, then verifies readable PNGs and covariance-aware metrics with categorical quantities explicitly unavailable.

## Documentation

- **[README](README.md)**: usage and outputs

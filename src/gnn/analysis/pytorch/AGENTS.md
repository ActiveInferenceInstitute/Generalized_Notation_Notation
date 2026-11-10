# PyTorch Analysis Sub-module

## Overview

Framework-specific analysis module for PyTorch simulation outputs. Reads `simulation_results.json` produced by the PyTorch runner and generates belief trajectory, action distribution, and EFE analysis plots.

## Architecture

```
pytorch/
├── __init__.py            # Package exports
├── analyzer.py            # PyTorch result analysis and visualization
└── torchlens_fidelity.py  # Declared connections vs TorchLens-traced dataflow
```

## Key Functions

- **`generate_analysis_from_logs(results_dir, output_dir=None, verbose=False)`** — Main entry point; reads PyTorch JSON results and produces matplotlib visualizations.
- **Belief trajectory plotting** — Tracks posterior belief evolution over simulation timesteps.
- **Action distribution analysis** — Visualizes policy distributions across actions.
- **`generate_torchlens_fidelity(results_dir, output_dir)`** — Writes the
  structural-fidelity report for every captured run (called by
  `generate_analysis_from_logs`).
- **`torchlens_fidelity.evaluate_capture(summary)`** — Pure classification of
  declared connections against traced site edges; refuses (`unverified`) a
  capture that did not reproduce the run or failed validation.

## Dependencies

- `numpy`, `matplotlib`, `networkx` (required)
- never `torch` or `torchlens`: the analysis reads the JSON capture summary

## Parent Module

See [analysis/AGENTS.md](../AGENTS.md) for the overall analysis architecture.

**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)

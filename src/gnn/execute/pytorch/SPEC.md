# PyTorch Execution — Technical Specification

**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)

## Execution Model

- Subprocess execution via `python -c` or script file
- Pre-flight: syntax check + dependency validation
- Timeout: inherits from Step 12 timeout (default 1800s)

## Input

- PyTorch scripts from `output/11_render_output/pytorch/`

## Output

- `simulation_results.json` — Simulation trajectories and results (its
  `torchlens` field records the optional capture status)
- `torchlens/` — capture manifest, `report.html`, `rollout_rolled.svg`, and
  with `GNN_TORCHLENS_ARTIFACTS=full` `rollout.tlspec` and `ops.csv`; written
  only when `GNN_TORCHLENS_CAPTURE` is set in the child environment
- Execution logs (stdout/stderr)
- Timing data

## Dependencies

- `torch >= 2.13.0` (`torch` extra)
- `torchlens >= 3.0.2, < 4` (optional, `torchlens` extra; imported by the
  child script only when capture is requested)

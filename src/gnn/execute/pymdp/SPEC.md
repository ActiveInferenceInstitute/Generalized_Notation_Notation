# PyMDP Execution — Technical Specification

**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)

## Execution Model

- Primary engine: `run_pymdp_simulation(...)` backed by `inferactively-pymdp>=1.0.0`
- Package detection: validates the JAX-first `pymdp` 1.0.0+ API
- Timeout: inherits from Step 12 timeout (1800s default)

## Architecture

```
pymdp/
├── __init__.py             # Package exports
├── context.py              # Execution context management
├── execute_pymdp.py        # Entry point dispatcher
├── executor.py             # High-level execution coordinator
├── package_detector.py     # PyMDP version detection
├── pymdp_runner.py         # Core simulation runner
├── pymdp_simulation.py     # Full POMDP simulation engine
├── pymdp_utils.py          # Shared utilities
├── simulation.py           # Canonical simulation function
└── validator.py            # Input validation
```

## Input

- PyMDP scripts from `output/11_render_output/<gnn_stem>/pymdp/`
- Step 12 reads the latest `render_processing_summary.json` when present and
  executes only scripts listed as successful for the requested framework.

## Output

- `simulation_results.json` — `pymdp_simulation_v1` beliefs, observations,
  hidden states, actions, EFE/VFE, policy posterior, validation, provenance,
  and runtime metadata
- Execution logs and timing data

The direct flat simulation records `runtime_metadata.native_agent` under
schema `pymdp_native_agent_v1`. The record reports the class and policy
horizon of the agent used for the final update, and the dtype, batch size,
devices and platforms of its likelihood array. Callers can read these
observations from the saved result or the returned payload.

## Dependencies

- `inferactively-pymdp>=1.0.0`

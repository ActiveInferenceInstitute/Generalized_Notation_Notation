# PyTorch Runner

Discovers and executes PyTorch-generated POMDP scripts via subprocess.

## Usage

```python
from gnn.execute.pytorch import run_pytorch_scripts

results = run_pytorch_scripts(
    render_dir="output/11_render_output", output_dir="output/12_execute_output"
)
```

## Features

- Dependency validation (checks `torch` availability)
- Syntax pre-validation before execution
- Log persistence (stdout/stderr capture)
- Wall-clock execution timing

## Optional TorchLens capture

The runner copies the parent environment into each child, so
`GNN_TORCHLENS_CAPTURE=1` (and optionally `GNN_TORCHLENS_ARTIFACTS=full`)
makes each discrete PyTorch script trace its own rollout with TorchLens and
write `torchlens/` beside `simulation_results.json`. TorchLens is imported only
inside that child, never by the runner; install it with
`uv sync --extra torchlens`. See the
[renderer README](../../render/pytorch/README.md#torchlens-capture-opt-in).

## See Also

- [Parent: execute/README.md](../README.md)
- [AGENTS.md](AGENTS.md) — Architecture documentation

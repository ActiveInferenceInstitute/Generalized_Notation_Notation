# PyTorch Render Backend Specification

## Overview

The PyTorch render backend generates **standalone PyTorch POMDP simulation scripts** from parsed GNN specifications.

## Public API

`render.pytorch` must export:

- `render_gnn_to_pytorch(gnn_spec: Dict[str, Any], output_path: Path, options: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, List[str]]`

## Inputs

- `gnn_spec`: parsed dict containing enough information to infer \(A, B, C, D\).
- `output_path`: file path for the generated script.
- `options`: optional dict; `num_timesteps` (at least 1) and `seed`
  (nonnegative 64-bit integer) are read. Step 11 admits exactly these two
  (`gnn.render.admission`).

## Outputs

- exactly one `.py` file at `output_path`.
- the generated script writes `simulation_results.json` under `PYTORCH_OUTPUT_DIR`.
- discrete results keep `schema_version: pytorch_simulation_v1` and add
  `random_seed`, `policy_history`, `habit` and `torchlens` (capture summary;
  `status: disabled` unless `GNN_TORCHLENS_CAPTURE` is set).

## Discrete script contract

- Components are `torch.nn.Module`s at the fixed addresses of
  `sites.SITE_DESCRIPTIONS`; the agent (`agent.*`) and the generative process
  (`process.*`) hold separate buffers.
- One `torch.Generator` seeded from `RANDOM_SEED` drives the samples in the
  order initial state, then per step observation, action, next state.
- Policy: `softmax(log E - G)` with unit precision; a declared `E` is a strict
  probability vector with one entry per action, otherwise uniform.
- `torchlens` is imported only inside `_torchlens_capture`. Capture statuses:
  `disabled`, `unavailable`, `failed`, `captured`; a capture is evidence only
  when `trace_reproduces_simulation` and `validated` are both true.

## Dependencies

- render-time: `numpy`
- generated script run-time: `torch`, `numpy`
- optional capture: `torchlens` (`torchlens` extra), imported by the script only

## Success criteria

Rendering is successful when:

- the script file is written to `output_path`, and
- the function returns `success=True` with `artifact_paths=[str(output_path)]`.


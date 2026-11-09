# PyTorch Analysis

Framework-specific analysis module for PyTorch simulation outputs.

## Usage

```python
from gnn.analysis.pytorch.analyzer import generate_analysis_from_logs

generate_analysis_from_logs(
    results_dir="output/12_execute_output", output_dir="output/16_analysis_output"
)
```

## Outputs

- Belief trajectory plot: `belief_trajectory.png`
- Action distribution histogram: `action_distribution.png`
- EFE history plot: `efe_history.png`
- JSON summary: `pytorch_analysis.json`
- When the run recorded a TorchLens capture: `pytorch_torchlens_fidelity.json`,
  `pytorch_torchlens_fidelity.md` and `pytorch_torchlens_fidelity.png`

## TorchLens structural fidelity

[`torchlens_fidelity.py`](torchlens_fidelity.py) compares the specification's
declared `## Connections` with the dataflow TorchLens traced through the
emitted modules (see the
[renderer README](../../render/pytorch/README.md#module-structure)). It reads
JSON only and never imports torch or torchlens.

Each declared connection is expanded into variable pairs (grouped endpoints
such as `(s, u)` expand to members) and classified:

| Status | Meaning |
| --- | --- |
| `direct` | one traced dataflow hop from a source site to a target site |
| `indirect` | a longer traced path; the shortest site path is reported |
| `reverse_only` | a directed declaration whose only traced path runs backwards |
| `not_observed` | no traced path in either direction |
| `unmapped` | an endpoint has no traced site; the renderer's reason is kept |

Undirected (`-`) declarations accept either direction. The rollout is a closed
perception-action loop, so a path may pass through later timesteps and most
agent variables reach one another indirectly: `direct` is the discriminating
status. The report also lists undeclared one-hop dependencies between mapped
variables and the lesion witnesses recorded by the capture.

A capture is evaluated only when it reproduced the simulation exactly and
passed TorchLens validation; otherwise the report has `status: unverified`
with the reason. Runs without capture (the default) produce no fidelity
files. The MCP tool `evaluate_torchlens_fidelity` returns the same report
read-only for a result file or directory.

## Dependencies

- `numpy`, `matplotlib` (required)
- `networkx` (fidelity figure)
- no torch or torchlens: the analysis reads JSON only

## See Also

- [Parent: analysis/README.md](../README.md)
- [AGENTS.md](AGENTS.md) — Architecture documentation
- [SPEC.md](SPEC.md) — Technical specification

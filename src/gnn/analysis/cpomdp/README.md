# cpomdp Analysis

Analysis of cpomdp simulation results for the analysis step. It binds the shared flat-payload analyzer (`analysis/flat_payload_analyzer.py`), as the NumPyro and PyTorch analyzers do.

## Usage

```python
from gnn.analysis.cpomdp import generate_analysis_from_logs

generate_analysis_from_logs(
    results_dir="output/12_execute_output", output_dir="output/16_analysis_output/cpomdp"
)
```

## Outputs, per model

- `belief_trajectory.png`: posterior mean of each state dimension over time.
- `efe_history.png`: expected free energy of every policy at each step. Passive models score no policies and get none.
- `cpomdp_analysis.json`: framework, model name, state count, validation and the shared metrics.

The shared metrics (belief entropy, confidence) are written for categorical beliefs. On cpomdp's Gaussian posterior means they carry no meaning, the same limitation the NumPyro analyzer has on continuous models.

See [AGENTS.md](AGENTS.md).

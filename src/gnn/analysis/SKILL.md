---
name: gnn-statistical-analysis
description: GNN advanced statistical analysis and result aggregation. Use when performing statistical analysis on simulation results, cross-simulation aggregation, computing information-theoretic metrics, or creating analytical visualizations of pipeline outputs.
---

# GNN Statistical Analysis (Step 16)

## Purpose

Performs advanced statistical analysis on pipeline outputs including simulation results, cross-framework comparison, Active Inference metrics, and comprehensive statistical reporting.

## Key Commands

```bash
# Run analysis
python src/gnn/16_analysis.py --target-dir input/gnn_files --output-dir output --verbose

# As part of pipeline
python src/gnn/main.py --only-steps 16 --verbose
```

```python
from pathlib import Path
from gnn.analysis import (
    process_analysis,
    perform_statistical_analysis,
    calculate_variable_statistics,
    calculate_connection_statistics,
    calculate_complexity_metrics,
    calculate_maintainability_index,
    analyze_framework_outputs,
    generate_framework_comparison_report,
    analyze_simulation_traces,
    analyze_free_energy,
    compute_shannon_entropy,
    compute_kl_divergence,
    compute_variational_free_energy,
    compute_expected_free_energy,
    generate_analysis_summary,
)

# Process analysis step (used by pipeline)
process_analysis(target_dir, output_dir, verbose=True)

# Statistical analysis (takes a file path)
stats = perform_statistical_analysis(Path("models/my_model.md"))

# Complexity metrics (takes a file path)
metrics = calculate_complexity_metrics(Path("models/my_model.md"))

# Framework comparison
report = generate_framework_comparison_report(results)

# Active Inference metrics (numpy arrays)
entropy = compute_shannon_entropy(distribution)
kl_div = compute_kl_divergence(p, q)
vfe = compute_variational_free_energy(observations, beliefs, A_matrix)
efe = compute_expected_free_energy(beliefs, A_matrix, B_matrix)
```

## Key Exports

- `process_analysis` — main pipeline processing function
- `perform_statistical_analysis` — core statistical analysis
- `calculate_complexity_metrics`, `calculate_maintainability_index`, `calculate_technical_debt`
- `analyze_framework_outputs`, `generate_framework_comparison_report`
- `compute_shannon_entropy`, `compute_kl_divergence`, `compute_variational_free_energy`, `compute_expected_free_energy`
## Output

- Analysis reports in `output/16_analysis_output/` (`analysis_results.json`, `analysis_summary.md`, `cross_model_comparison_report.md`)
- Statistical summaries and aggregations
- Comparative visualizations and GridWorld GIFs (disable with `--no-animations`)


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `compute_complexity_metrics`
- `evaluate_torchlens_fidelity`
- `get_analysis_results`
- `list_analysis_tools`
- `process_analysis`

## References

- [AGENTS.md](AGENTS.md) — Module documentation
- [README.md](README.md) — Usage guide
- [SPEC.md](SPEC.md) — Module specification


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API

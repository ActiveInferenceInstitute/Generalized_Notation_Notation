# PyTorch Analysis — Technical Specification

**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)

## Input Format

- `simulation_results.json` from PyTorch execution step (searched recursively under `pytorch/` paths or `pytorch_simulation_results.json`)
- Fields: `beliefs`, `actions`, `observations`, `efe_history`, `validation`, `model_name`
- Optional `torchlens` capture summary (`status`, `trace_reproduces_simulation`,
  `validated`, `declared_connections`, `variable_sites`, `unmapped_variables`,
  `site_edges`, `intervention_witnesses`)

## Output Format

- PNG plots: `belief_trajectory.png`, `action_distribution.png`, `efe_history.png`
- JSON summary: `{model}/pytorch_analysis.json` (framework, metrics, plots_generated)
- Fidelity report (captured runs only): `{model}/pytorch_torchlens_fidelity.json`
  (`schema_version: gnn_torchlens_fidelity_v1`; `status` `evaluated` or
  `unverified`; per-connection `direct` / `indirect` / `reverse_only` /
  `not_observed` / `unmapped`; `structural_agreement` over mapped
  connections; `undeclared_dependencies`), plus `.md` and `.png` views

## Processing Requirements

- Handles variable-length simulation runs
- Graceful degradation when matplotlib unavailable
- Numerical stability for near-zero probability values

## Error Handling

- Missing results file → returns empty analysis with warning
- Malformed JSON → logs parse error, continues with available data
- Capture `unavailable`/`failed`, a trace that does not reproduce the run, or
  failed validation → fidelity `status: unverified` with the reason; no edge
  is evaluated

# RxInfer Analysis — Technical Specification

**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)

## Input

- `simulation_results.json` from RxInfer (Julia) execution step (schema `rxinfer_simulation_v1`)
- Genuine variational message-passing inference results (`@model` + `infer()`, `free_energy = true`)

## Output

- Belief trajectory plots (PNG)
- Message flow analysis (PNG)
- Convergence diagnostics (JSON) — based on the genuine `variational_free_energy` trace (`inference_converged`, `vfe_present`)

## Framework

`result_ingestion.py` owns filesystem discovery and JSON loading.
`metrics.py` owns result normalization and numerical diagnostics, using the
existing shared numerical availability contract. `analyzer.py` orchestrates
analysis and the cohesive categorical plot dispatcher. Existing analyzer
exports and call signatures are preserved; this separation changes no
normalization, convergence, marginalization or plot semantics.

- Julia RxInfer genuine variational message-passing results (`variational_free_energy` populated with real values; previously `Float64[]`)
- Matplotlib visualization

## Error Handling

- Missing Julia results → graceful skip
- Non-convergent inference → diagnostic warning

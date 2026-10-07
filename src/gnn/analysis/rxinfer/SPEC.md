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

Gaussian views label each reported control column by its zero-based identity,
with a legend and distinct marker/line styles. Discrete timestep and inference
iteration axes use integer ticks. An unreported VFE has no measured axis.
Optional result metadata `units: {state: "...", control: "...", vfe: "..."}`
supplies axis units, with `model_parameters.units` as a fallback; undeclared
units are labeled `units unspecified`. Timestep/inference-iteration indices
are explicitly zero-based. VFE uses a declared
`variational_free_energy_convention`, or is labeled `convention unspecified`.
Presentation changes do not alter means, controls or covariance values.

## Error Handling

- Missing Julia results → graceful skip
- Non-convergent inference → diagnostic warning
- Missing execution summary permits standalone result discovery. A present
  summary is authoritative: an explicit empty selection admits no inherited
  result folders; invalid summary bytes/shape refuse analysis with a typed,
  path-specific reason.
- Result JSON must decode to an object. `RxInferResultReadError` preserves the
  original I/O, UTF-8 or JSON exception as its cause. The legacy extraction
  entrypoint still returns its empty/default mapping on a read failure and logs
  the precise file and cause; it does not manufacture successful evidence.

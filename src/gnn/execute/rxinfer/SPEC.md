# RxInfer Execution — Technical Specification

**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)

## Execution Model

- Julia subprocess execution via `julia --startup-file=no --project=src/gnn/execute/rxinfer <script>`
- Pre-flight: Julia + committed RxInfer environment validation (`setup_environment.jl` → `Pkg.activate()` + `Pkg.instantiate()`, no runtime `Pkg.add`)
- Genuine `@model` + `infer()` variational message-passing inference (`free_energy = true`)
- Timeout: inherits from Step 12 timeout (3600s default)

## Input

- `.jl` scripts from `output/11_render_output/<model>/rxinfer/` (genuine `@model pomdp_model` scripts)
- Legacy `*_config.toml` inputs remain accepted by the committed
  `rxinfer_runner.jl` adapter; the pre-execution gate scans that Julia program.

## Output

- `simulation_results.json` — genuine variational message-passing inference results, schema `rxinfer_simulation_v1`
- `variational_free_energy` populated with genuine VFE values (previously `Float64[]`)
- Execution logs (stdout/stderr)
- Convergence diagnostics (`inference_converged`, `vfe_present`)

## Error Handling

- Julia source is read as UTF-8. I/O or decoding failure refuses execution and
  logs the script path and exception type before launching a subprocess.
- Evidence persistence is best effort: expected filesystem, encoding or JSON
  serialization failures log script, destination and typed cause without
  replacing the completed run's boolean verdict.

## Dependencies

- `julia >= 1.10` (`Project.toml` compat); RxInfer 5.5.0 and all deps pinned by the committed `Project.toml` + `Manifest.toml` in this directory

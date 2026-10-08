# ActiveInference.jl Analysis — Technical Specification

**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)

## Input

- `simulation_results.json` from ActiveInference.jl execution step

## Output

- Belief trajectory plots (PNG)
- Action distribution analysis (PNG)
- Free energy plots (PNG)
- Trace reconstruction and model-matrix heatmap PNGs; returns generated file paths (no `analysis_summary.json`)

## Framework

- Julia-based Active Inference results parsed from JSON
- Matplotlib visualization with numpy processing

## Error Handling

- Missing Julia results → warning + empty analysis
- Graceful degradation when matplotlib unavailable
- CSV reconstruction admits a row only after every scalar column parses.
  A malformed row is skipped atomically with file, line and typed-cause context;
  valid rows retain their aligned timestep/observation/action/belief values.

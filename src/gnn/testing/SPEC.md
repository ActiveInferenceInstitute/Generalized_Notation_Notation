# Testing — specification

## Role

- **`round_trip_tester.py`** — `GNNRoundTripTester`; round-trip harness (default **21** format strings in config; see **[../SPEC.md](../SPEC.md)**). Pytest coverage lives under `tests/testing/`.
- **`round_trip_config.py`**, **`round_trip_results.py`**, **`round_trip_comparison.py`**, **`round_trip_report.py`**, **`round_trip_markdown_parser.py`**, **`round_trip_availability.py`**, **`round_trip_strategy.py`** — harness configuration / results / mixins / availability probe / strategy.
- **`performance_benchmarks.py`** — benchmarks helper (exercised by `tests/testing/`).
- **`round_trip_reports/`** — optional output directory for reports.

## Requirements

- **Python** >= 3.11 (see repo `pyproject.toml`).
- The schema validator's round-trip adapter preserves supplied `ParsedGNN`
  parameter names and values, including zero, false and nested containers.
  Each conversion owns independent value copies; it does not normalize,
  flatten or reinterpret the supplied parameters. Existing variable dimensions,
  types, descriptions and model annotations follow the same adapter contract.
- Saved round-trip artifacts use the canonical parser output extensions,
  retaining the valid existing Z `.zed` suffix. Native PKL `.pkl` and binary
  Pickle `.pickle` remain distinct, and each supported target has its own
  artifact rather than overwriting another format's evidence.
- `RoundTripTestStrategy` binds each supplied source to an independent native
  tester and child artifact directory. It preserves the configured format
  selection and shared template, attributes native rows to report provenance,
  and keeps injected per-row source reports compatible. Missing inputs
  retain their actual source path and typed failure rather than testing the
  default reference or inventing successful format evidence.

## Running

```bash
uv run --extra dev python -m pytest tests/testing/ -q
uv run --extra dev python -m pytest tests/gnn/ -q
```

See **[README.md](README.md)** and **[README_round_trip.md](README_round_trip.md)**.

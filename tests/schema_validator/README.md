# tests/schema_validator

Regressions for `src/gnn/schema_validator/`: cross-format consistency,
validation-level resolution, parser failure diagnostics, content-degradation
handling and time-varying dynamics parsing.

```bash
uv run --frozen --no-sync python -m pytest tests/schema_validator/ -q
```

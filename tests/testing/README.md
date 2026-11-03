# tests/testing

Contract and regression coverage for the testing infrastructure itself: the
format round-trip suite (scope declared in `src/gnn/testing/round_trip_config.py`),
its report/comparison/strategy layers, source isolation, the 4.1.0
native-artifact consumers and performance benchmarks.

```bash
uv run --frozen --no-sync python -m pytest tests/testing/ -q
```

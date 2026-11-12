# Specification: Testing Documentation

## Scope
Testing strategy, coverage expectations, and quality gates for GNN.
Complements `tests/` which holds the actual test suite.

## Coverage Gates
- Coverage threshold enforced via `fail_under = 60` in `pyproject.toml`
  `[tool.coverage.report]`. The key must live in the *report* section:
  coverage.py ignores it under `[tool.coverage.run]`, so a placement there
  never failed a build. pytest-cov reads the configured floor on every
  `--cov` run, so the CI test lane and `just test-cov` fail below 60%
  instead of merely reporting the number.
- Real-implementation policy: tests exercise real dependencies. A missing
  surface must fail loudly rather than be skipped or substituted.
- Zero-skip contract: `tests/test_zero_skip_contracts.py` scans every
  `test_*.py` under `tests/` and fails on any `pytest.skip(`,
  `pytest.importorskip(`, `pytest.xfail(`, `@pytest.mark.skip`,
  `@pytest.mark.skipif`, or `@pytest.mark.xfail` token. The sole exemption is
  `DEFAULT_SKIP_ALLOWLIST` in that same module, which lists the files needing
  software outside the Python environment (the Ollama LLM tests, and the
  Julia-dependent cross-framework and RxInfer visualization-log tests).
  Widening that set is a reviewable edit to the contract itself.
- Test naming: `tests/test_{module}_*.py`
- Expected outcome for the default suite is zero failures. Consult the current
  run output for pass counts rather than a number recorded here.

## Reproducibility Requirements
- `PYTHONHASHSEED=0` must be set in the environment for deterministic
  dict iteration across runs. The `ci.yml` test lanes set it at job level, so
  CI runs agree without any per-step export; set the same value locally when
  comparing JSON fingerprints. Without it, JSON output files with unsorted
  keys may differ between pipeline runs even with identical inputs.
- A function-scoped `_auto_seed_rng` autouse fixture (`conftest.py`)
  calls `np.random.seed(0)` before every test, providing a deterministic
  baseline even for unseeded test functions.
- Tests that need their own random data build a local
  `np.random.default_rng(seed)` instead of calling `np.random.seed()`, so the
  fixture above stays the single writer of process-global RNG state.
- Simulation seeds belong in the spec: `run_pymdp_simulation` draws from
  `np.random.default_rng(model_parameters.random_seed)`, never from the
  process-global numpy RNG, so a reproducibility test sets `random_seed`.

## Cross-Framework Reliability Contract
`src/gnn/pipeline/cross_framework_reliability.py` (driven by
`scripts/run_cross_framework_reliability.py`) compares *structural metadata*
between backends a model-family profile declares compatible: recorded
`random_seed` presence, `num_timesteps`, matrix shapes and trace lengths.
Trajectory values are intentionally not compared across backends — the
maintained frameworks compute the same model with different inference
numerics, so cross-backend value equality is not a defined contract.
Per-framework trajectory assertions live in `tests/execute/`.

## Test Categories
| Category | Files | Purpose |
|----------|-------|---------|
| Unit | `test_<module>_overall.py`, `test_<module>_<area>.py` | Module-level behavior |
| Integration | `test_pipeline_*.py` | Cross-module flows |
| Contract | `test_pymdp_contracts.py`, `test_discrete_models_pymdp.py` | Backend contract tests |
| Regression | `test_<phase_or_fix>.py` | One-off regression guards |

## Running
```bash
uv sync --extra dev
uv run --extra dev python -m pytest tests/ -q --tb=no \
  --ignore=tests/llm/test_llm_ollama.py \
  --ignore=tests/llm/test_llm_ollama_integration.py
```

## Status
Maintained. Every contract change (e.g., the Phase 1.1 exit-code-2 widening)
must land with a regression test.

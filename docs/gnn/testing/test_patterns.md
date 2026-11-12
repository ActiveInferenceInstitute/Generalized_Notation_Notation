# GNN Test Patterns and Standards

Real conventions drawn from `tests/conftest.py`, `tests/runner.py`, and test files.

**Last Updated**: 2026-08-07

## Real-Implementation Policy

Every test file follows a strict **real-implementation** policy, described in
`TEST_SUITE_SUMMARY.md`:

- ❌ No standard testing substitution libraries (like patching)
- ❌ No monkeypatching of functions or classes
- ✅ Real code paths executed in every test
- ✅ Real data — representative GNN files from `tests/test_data/`
- ✅ Real dependencies — an unavailable surface fails loudly, it is never substituted
- ✅ File-based assertions on real output artifacts

## The Zero-Skip Contract

The policy above is not enforced by review alone. `tests/test_zero_skip_contracts.py`
enforces it mechanically: it walks every `test_*.py` under `tests/` and fails if the
file's text contains any skip-shaped token —

```text
pytest.skip(       pytest.importorskip(       pytest.xfail(
@pytest.mark.skip  @pytest.mark.skipif        @pytest.mark.xfail
```

The rationale is that a skipped test reports green while verifying nothing, so a missing
dependency silently becomes a coverage hole. The default suite must **fail explicitly**
instead of hiding an unavailable surface.

Exactly one escape hatch exists: `DEFAULT_SKIP_ALLOWLIST`, a set of relative paths in the
same module that are exempt from the scan. It currently covers the Ollama LLM tests and
the Julia-dependent cross-framework and RxInfer visualization-log tests — surfaces that
genuinely require software outside the Python environment. Adding a file to that set is a
deliberate, reviewable edit to the contract, which is the point: the exemption list is
data in version control rather than a decorator scattered through the suite.

```python
# ✅ CORRECT — exercise the real surface and assert on real state
def test_audio_backend():
    from audio import check_audio_backends

    result = check_audio_backends()
    assert isinstance(result, dict)
    assert "available" in result  # may report "unavailable" — that is a valid answer


# ❌ WRONG — substitute the dependency
@patch("audio.backends.soundfile", None)
def test_audio_backend_missing(): ...


# ❌ WRONG — the zero-skip contract fails on this token
def test_audio_backend_optional():
    backends = pytest.importorskip("audio.backends")
```

## Pytest Markers (20 defined in conftest.py)

| Marker | Description |
|--------|-------------|
| `unit` | Unit tests for individual components |
| `integration` | Integration tests for component interactions |
| `performance` | Performance and resource usage tests |
| `slow` | Tests taking significant time |
| `fast` | Quick tests for rapid feedback |
| `destructive` | May modify system state |
| `external` | Requires external dependencies |
| `core` | Core module tests |
| `pipeline` | Pipeline infrastructure tests |
| `recovery` | Pipeline recovery tests |
| `utilities` | Utility function tests |
| `environment` | Environment validation tests |
| `render` | Rendering and code generation tests |
| `export` | Export functionality tests |
| `parsers` | Parser and format tests |
| `main_orchestrator` | Main orchestrator tests |
| `type_checking` | Type checking tests |
| `mcp` | Model Context Protocol tests |
| `sapf` | SAPF audio generation tests |
| `visualization` | Visualization tests |

```bash
# Selective execution by marker
uv run --extra dev python -m pytest tests/ -m fast -q
uv run --extra dev python -m pytest tests/ -m "not slow" -q
uv run --extra dev python -m pytest tests/ -m "integration and not slow" -q
```

## Fixture Convention (conftest.py)

All shared fixtures live in `tests/conftest.py`. Note that the sample-data fixtures
synthesize their content into a temporary directory and tear it down afterwards — they do
**not** hand back a path into the `input/gnn_files/` corpus, so a test may write next to
the fixture file without disturbing the repository.

| Fixture | Scope | Description |
|---------|-------|-------------|
| `tmp_path` | function | Built-in pytest temp directory |
| `_auto_seed_rng` | function (autouse) | Seeds NumPy to 0 before every test, so runs are deterministic |
| `test_config` | session | Session-wide config dict (`test_mode`, `safe_mode`, `temp_dir`, limits) |
| `project_root` / `src_dir` / `test_dir` | session | Repository location anchors |
| `safe_filesystem` | function | Sandboxed filesystem helper with a `create_file()` API |
| `isolated_temp_dir` | function | Standalone temp directory |
| `temp_directories` | function | Named temp directories derived from `tmp_path` |
| `temp_output_dir` | function | Temp directory for output artifacts |
| `sample_gnn_file` | function | Temp `actinf_pomdp_agent.md` written by `_write_sample_gnn_markdown()` |
| `sample_gnn_files` | function | Two temp GNN files (`simple`, `second`) sharing a minimal POMDP schema |
| `test_data_dir` | function | Temp directory holding `samples/actinf_pomdp_agent.md` |
| `sample_gnn_spec` | function | Parsed-spec dictionary for renderer tests |
| `sample_markdown` / `sample_scala` | function | Raw source strings |
| `comprehensive_test_data` | function | Multi-file dataset built inside `isolated_temp_dir` |
| `test_render_module` | function | Real render-module surface exposing `render_gnn_spec()` |
| `test_mcp_tools` | function | Real MCP tool registry exposing `register_tool()` / `execute_tool()` |

## Resource Monitoring

`TestRunner` in `infrastructure/test_runner.py` wraps every test run with `ResourceMonitor`:

```python
# infrastructure/resource_monitor.py (via psutil if available)
class ResourceMonitor:
    def start_monitoring(self): ...  # spins up a thread tracking memory/CPU
    def stop_monitoring(self): ...
    def get_stats(self) -> dict:  # returns {"peak_memory_mb": float, "avg_cpu": float}
        ...
```

Defaults come from `ResourceMonitor.__init__`: `memory_limit_mb=2048` and
`cpu_limit_percent=80`. `TestExecutionConfig` in `infrastructure/test_config.py` carries
the same 2048 MB default and is the knob to change per run.

## Writing Tests

### Parametrize for matrix coverage

```python
@pytest.mark.parametrize("export_format", ["json", "xml", "graphml", "gexf", "pickle"])
def test_export_format(export_format, tmp_path, sample_gnn_file):
    result = process_export(sample_gnn_file, tmp_path, format=export_format)
    assert result["success"] is True, f"process_export failed for {export_format}"
    assert (tmp_path / f"model.{export_format}").exists()
```

### Descriptive assertions with context

```python
# ✅ Good — fails with enough context to diagnose
result = process_validation(gnn_dir, output_dir, logger)
assert result is True, f"process_validation returned {result!r} — check {output_dir}"

# ❌ Bad — silent failure
assert process_validation(gnn_dir, output_dir, logger)
```

### Optional dependency handling

Do not reach for `importorskip` here — the zero-skip contract rejects it. Test the part
of the surface that does not need the optional package. Rendering is pure code
generation, so a PyTorch renderer test never needs `torch` to be installed:

```python
def test_torch_rendering():
    # Rendering emits source text; it does not import torch.
    rendered = render_gnn_to_pytorch(sample_content)
    assert "import torch" in rendered
```

When a test genuinely cannot run without software outside the Python environment (a Julia
toolchain, a running Ollama server), the file belongs in `DEFAULT_SKIP_ALLOWLIST` in
`tests/test_zero_skip_contracts.py` rather than carrying a skip decorator.

## File Naming Conventions

| Pattern | Category |
|---------|----------|
| `test_*_overall.py` | Per-module system tests |
| `test_pipeline_*.py` | Pipeline orchestration |
| `test_execute_pymdp_*.py` | PyMDP simulation |
| `test_mcp_*.py` | MCP layer |
| `test_environment_*.py` | Environment setup |
| `test_*_integration.py` | Cross-module |

## Coverage Configuration

Runtime pytest settings live in `pytest.ini` at the repository root — the only pytest
configuration pytest reads; `pyproject.toml` carries no `[tool.pytest.ini_options]`
block. Markers are registered in `pytest.ini` (single source of truth, covered by
`--strict-markers`):

```ini
# pytest.ini
[pytest]
addopts =
    --strict-markers
    --strict-config
    --tb=short
    --durations=10
    --durations-min=1.0
    --maxfail=10
    -p no:randomly
    -p no:sugar
    -p no:stepwise
```

Coverage settings are in `pyproject.toml`, with the floor in the *report* section —
coverage.py ignores `fail_under` under `[tool.coverage.run]`:

```toml
# pyproject.toml
[tool.coverage.run]
source = ["src/gnn"]
omit = ["*/tests/*", "*/test_*", "*/__pycache__/*", "*/venv/*", "*/.venv/*"]

[tool.coverage.report]
fail_under = 60
```

Generate HTML report:

```bash
uv run --extra dev python -m pytest tests/ \
  --cov=gnn --cov-report=html --cov-report=term-missing -q
# → htmlcov/index.html
```

## See Also

- [testing/README.md](README.md) — test suite overview + category table
- [testing/mcp_audit.md](mcp_audit.md) — MCP audit anatomy
- [`tests/TEST_SUITE_SUMMARY.md`](../../../tests/TEST_SUITE_SUMMARY.md) — canonical reference
- [`tests/conftest.py`](../../../tests/conftest.py) — all fixtures and markers
- [`tests/test_zero_skip_contracts.py`](../../../tests/test_zero_skip_contracts.py) — the zero-skip contract and `DEFAULT_SKIP_ALLOWLIST`

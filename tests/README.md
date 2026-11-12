# Test infrastructure

The [testing guide](../docs/testing/README.md) explains public selections and
runtime prerequisites. This directory owns test infrastructure and consumer
regressions. [pyproject.toml](../pyproject.toml) owns pytest options, markers and
coverage policy; [workflows](../.github/workflows/README.md) own release selections.

## Quick Start

Provision the frozen dev profile before invoking tests:

```bash
uv sync --frozen --extra dev --python 3.14
uv run --frozen --no-sync python -m pytest tests/docs/ -q
uv run --frozen --no-sync python -m pytest tests/gnn/ -q
uv run --frozen --no-sync python -m pytest tests/ --collect-only -q
```

Use the marker expressions in the current workflow for a CI-equivalent run.
Native toolchains, GUI extras and Ollama have separate profiles. A focused test
pass does not replace those profiles or establish repository-wide coverage.

## Architecture

| Owner | Responsibility |
|---|---|
| [runner.py](runner.py) | `run_tests` routing and canonical runner re-export |
| [infrastructure/](infrastructure/) | Pytest subprocess runner, reports and resource monitoring |
| [test_runner_modes.py](test_runner_modes.py) | Fast, comprehensive and reliable execution modes |
| [test_runner_modular.py](test_runner_modular.py) | Category execution and runner factory |
| [categories.py](categories.py) | Live typed category/file routing; `missing_category_files()` detects drift |
| [conftest.py](conftest.py) | Shared pytest configuration and fixtures |
| [helpers/](helpers/) | Reusable model fixtures, import helpers and rendering utilities |

Numbered Step 2 delegates to this infrastructure. For its public function contract,
read [AGENTS.md](AGENTS.md) and [SPEC.md](SPEC.md). Category routes are not a second
pytest collection authority; inspect collection for actual current cases.

## Directory layout

The maintained inventory is 46 maintained first-level subdirectories; 43 contain direct test files.
[scripts/check_capability_contracts.py](../scripts/check_capability_contracts.py)
computes and enforces this description against the actual tree; it is a directory
inventory, not a test pass count.

## Test organization

Place consumer tests under the owning module's directory and name them
`test_*.py`. Use [helpers](helpers/README.md) for complete model fixtures. The live
[category owner](categories.py) defines grouping, rather than a copied directory
inventory or pass-count snapshot. Infrastructure regressions belong under
[tests/](tests/).

## Writing and running tests

Exercise public behavior, explicit failures and relevant negative controls.
Distinguish optional runtime absence from contract rejection and test failure.
Use declared markers only. Keep native numerical witnesses independent of the
implementation under test; fixture success and importability are weaker evidence.

```bash
uv run --frozen --no-sync python src/gnn/2_tests.py --fast-only --verbose
uv run --frozen --no-sync python src/gnn/2_tests.py --comprehensive --verbose
```

The comprehensive path can require provisioned native tools and extras; do not
claim a partial local environment ran every profile. Preserve JUnit, process
outcomes and actual skips. Timing depends on selection and host.

## Environment validation timing

`test_environment_module_performance` runs the real environment validation,
including package inventory in a separate interpreter and the JAX/Optax/Flax/
PyMDP import, JIT compilation and runtime probe. A fixed ten-second limit failed
at **22.4 seconds** in a parallel coverage gate. Three isolated runs on the same
macOS arm64 host measured **3.380, 2.305 and 2.041 seconds** (median **2.305
seconds**), with the JAX subprocess accounting for **3.099, 2.154 and 1.876
seconds** — a single absolute time therefore depends on host load and cold start.

The test interleaves three independent passes of the real prerequisite probes
with three full validations, times both with `time.perf_counter()` and compares
their medians. Full validation must finish within **1.5 times the independent
probe median plus one second**; the 50% allowance and the one-second margin
absorb host variation while still rejecting sustained extra validation work. A
single cold start cannot set either median. Both paths must pass the real
probes, and full validation must report all six diagnostics and the current
Python version. The baseline does not time
`EnvironmentManager.validate_environment`, so repeated work inside that method
cannot enlarge its own allowance. The probes keep their production subprocess
deadlines (package inventory 60 s, JAX stack 120 s); the relative assertion
targets validation overhead rather than changes shared by both probe paths.

Samples, medians and the calculated budget are recorded as pytest properties.
Retain them in a focused JUnit report with:

```bash
uv run --frozen --no-sync python -m pytest \
  tests/infrastructure/test_environment_overall.py::test_environment_module_performance \
  -o junit_family=xunit1 --junitxml=/tmp/gnn-environment-timing.xml -q
```

## Exit-code contract

The pipeline uses **0=success, 1=error, 2=success with warnings/skipped**.
Inspect individual step outcomes and run receipts; warnings do not imply every
optional backend ran, and a collection error cannot count as successful tests.

## Coverage and release acceptance

[Native comprehensive coverage](../docs/development/native_comprehensive_coverage.md)
requires separate strict integer reports on each configured Python version;
[installed platforms](../docs/development/installed_platform_acceptance.md) require
ordinary wheel installation and real native execution outside the checkout.
Keep core coverage, native reports, platform checks and manuscript custody separate.
Never sum overlapping suites or union coverage across environments to meet a floor.

## Troubleshooting

Start with the exact failed command, Python/runtime identity, fixture source and
complete diagnostic. Check optional installation and marker selection before
changing tests. A collection error is a failure, not zero successful cases.
See [troubleshooting](../docs/troubleshooting/README.md) and the current
[CI owners](../.github/workflows/README.md). Do not suppress warnings or lower
coverage/test selections to make a release pass.

## References

- [Repository README](../README.md) and [contribution guide](../CONTRIBUTING.md)
- [Testing guide](../docs/testing/README.md)
- [Agent guide](AGENTS.md), [specification](SPEC.md), [capability guide](SKILL.md)
- [Remaining work](../TO-DO.md)

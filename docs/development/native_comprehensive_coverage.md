# Native comprehensive coverage

The [native coverage observer](../../scripts/run_comprehensive_native_coverage.py)
produces a separate comprehensive report for each supported Python environment.
The existing core CI selector, its report, all coverage source/omit/exclude rules,
and the enforced 60% floor remain unchanged. The roadmap's **>80%** target requires
fresh full reports from Python 3.11, 3.12 and 3.13 before raising a floor.
The [hosted workflow](../../.github/workflows/comprehensive-native-coverage.yml)
uses three independent jobs, checking out each exact candidate revision.

## Run a provisioned environment

Use a clean immutable checkout and an owned virtualenv provisioned from the
frozen lock with `dev` and `api`. The hosted comprehensive jobs additionally
provision the existing `gui`, `thrml` and `cpomdp` extras. Each output directory must be new and outside
the checkout. The script copies the original coverage configuration, adding
parallel data files and their output destination. Child tracing starts before
actual original-source execution; it does not eagerly instrument stdlib-only
children or change their execution deadlines.

```bash
UV_PYTHON=3.12 uv sync --frozen --extra dev --extra api
UV_PYTHON=3.12 uv run --frozen --no-sync python \
  scripts/run_comprehensive_native_coverage.py \
  --checkout "$PWD" --output /tmp/gnn-native-coverage-3.12 \
  --python-version 3.12 --workers 4 --lane-timeout 1800
```

Repeat independently with Python 3.11 and 3.13. `--workers` is an explicit core
worker count; zero keeps core serial. MCP and pipeline run serially. An active
owned environment elsewhere requires `--owned-environment` with its actual
path. The script verifies that environment's purelib before instrumentation.
Keep this environment exclusive to the observation run: another caller using
the same interpreter during its temporary hook could add unrelated execution.
Source, fixtures and configuration must remain unchanged until the terminal
receipt and source bookend have been written.

| Lane | Actual selection |
|---|---|
| Core | `tests -m "not pipeline and not mcp and not ollama and not env_heavy and not toolchain"` |
| MCP | All `tests/mcp` |
| Pipeline | `tests -m "pipeline and not ollama and not env_heavy and (not toolchain or needs_posix)"` |
| Explicit THRML native | All `tests/render/test_thrml_native.py` |
| Explicit Gradio native | All `tests/gui/test_native_gradio_consumers_410.py` |

Ordinary invocations retain the original three selections. The two optional
lanes require explicit requests and the locked native extras; their tests use
actual installed THRML and Gradio objects. The GUI checks build interfaces and
dispatch registered callbacks in process, then reopen saved models. They do
not launch a persistent server or call a provider. CPOMDP's existing native
tests enter the unchanged core selection when that extra is provisioned.
Provisioning extras changes the declared environment, so reports from an
environment without those extras cannot be pooled with this report.

```bash
UV_PYTHON=3.12 uv sync --frozen --extra dev --extra api --extra gui --extra thrml --extra cpomdp
GRADIO_ANALYTICS_ENABLED=False HF_HUB_OFFLINE=1 UV_PYTHON=3.12 \
  uv run --frozen --no-sync python scripts/run_comprehensive_native_coverage.py \
  --checkout "$PWD" --output /tmp/gnn-native-provisioned-3.12 \
  --python-version 3.12 --workers 4 --lane-timeout 1800 \
  --lane core --lane mcp --lane pipeline --lane thrml-native --lane gui-native
```

The declared lanes require local behavior checks, not external provider/model
or training runs. Optional toolchains or live surfaces require their own explicit
provisioning, safety assessment and qualified report before admission. Do not
broaden the selector silently to make a percentage pass.

## Verify instrumentation before a full report

`--pilot` runs five existing controls: the isolated hardware probe, all three
module/distribution/repository entrypoint variants, and a real Step 3 parent/child
artifact equivalence check. It exercises isolated `-I` execution, replaced
`PYTHONPATH`, source/distribution layouts and matching worker collections.

```bash
UV_PYTHON=3.12 uv run --frozen --no-sync python \
  scripts/run_comprehensive_native_coverage.py \
  --checkout "$PWD" --output /tmp/gnn-native-observer-pilot \
  --python-version 3.12 --pilot --workers 2 --lane-timeout 180
```

`--lane mcp` or another single lane is also a partial report. Neither a pilot nor
a partial lane establishes whole-matrix coverage, regardless of its test count.

## Read the evidence

`receipt.json` records the exact Git/source/fixture/configuration manifest,
interpreter and virtualenv, installed distribution versions, lane commands,
selected node identities, overlaps, JUnit failures/skips, raw process data and
coverage totals. Every admitted process must match the native Python minor,
executable, virtualenv, base interpreter and GNN source origin. Installed
distribution names, versions, locations and metadata hashes are inventoried in
the parent before and after each lane. Children record their actual startup and
activation import paths; metadata overlays outside the admitted paths are
explicit and cannot receive original-source credit as the native environment.
Each lane retains its own JUnit, logs, configuration and full-source JSON report.

The observer creates an exclusive temporary `.pth` in the declared virtualenv,
ordered after existing editable entries. It records process startup without
importing GNN, including children that strip `PYTHONPATH` or use `-I`. A public
Python audit hook observes code execution and starts coverage before the first
frame from the original source. This includes imports, `-m`, direct scripts,
`runpy` and spawned Python children. Forked children reset their PID and trace
state; unioning line sets preserves overlap. Stdlib-only children retain an
explicit receipt with no source activation or inferred coverage. Creation,
content hash and restoration are recorded; removal requires the exact owned
bytes. The original package/dependency configuration is unchanged.

An unreceipted data file with **zero executed original-source lines** may be
retained and explicitly qualified outside the union. Any executed original
source without matching native provenance fails integrity. Unreadable raw data
is retained with its path, hash and error, with its source contribution marked
unknown; the report fails closed. Lane commands, terminal status and JUnit
outcomes are saved before raw data is read, so collection or SQLite failures do
not erase the actual test failures. The observer only
combines admitted data and verifies identical per-file statement/exclusion
denominators. Python `-S` can prevent startup tracing; behavior evidence from
such children does not gain unobserved source coverage; observed launch receipts
identify `-S` bypasses. Child source paths resolve to the same absolute original
source scope while the child's working directory remains unchanged. An initial
public coverage save establishes conservative data before source execution;
forced termination can still lose later unsaved lines and never creates inferred
coverage. The observer rejects known source modules loaded before its hook.
Launch classification currently recognizes literal `-S`/`-I` argument tokens;
combined flags, shell wrappers and embedded interpreters are outside that
classification. A startup bypass is unobserved execution, never evidence that
the child executed zero original-source lines.

Each lane and the comprehensive data must equal the exact admitted raw line-set
union. Their reports must preserve the same per-file statement/exclusion census,
and the comprehensive executed statement sets must equal the lane report union.
The observer updates data through the public API without deduplicating by raw
filename hash hints, which can become stale after an early conservative save.

`coverage-comprehensive.json` unions actual line sets within one environment.
Never union across Python versions, operating systems, dependency environments
or changed source; never sum overlapping test counts. Require terminal success,
no integrity errors, a clean source identity and
`complete_declared_core_mcp_pipeline_selection: true` for a full declared report.
An explicit optional run additionally requires the exact `requested_lanes`,
matching completed lane receipts and
`complete_requested_optional_native_selection: true`. All requested lanes
must select real tests and finish without failures, errors or skips. The
original full-source denominator, exclusions and per-lane deadline also apply
to these optional lanes. The existing core 60% gate remains unchanged; the
comprehensive union's 60% floor is checked separately, rather than on each
small optional lane.
The observer's zero exit status certifies the selected run and its receipts;
it does not enforce a percentage threshold. Keep the original core floor check.

For the roadmap target, compare integers: `5 * covered_lines > 4 * num_statements`
for **each** supported environment. The display-rounded percentage is not an
acceptance test. A future enforced floor needs accepted final-source evidence.
The outer deadline bounds the pytest command/process group; it does not certify
cleanup of detached descendant sessions.

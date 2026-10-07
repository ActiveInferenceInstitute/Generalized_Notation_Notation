# Diagnosis and recovery

Start with the exact failing command, environment, selected sources and current
invocation receipt. Preserve failed/unsupported outcomes and successful
siblings. Historical timings or test totals cannot establish health.

```bash
uv sync --frozen --extra dev --python 3.12
uv run --frozen --no-sync gnn --help
uv run --frozen --no-sync gnn doctor
uv run --frozen --no-sync gnn validate input/gnn_files/discrete/two_state_bistable.md
uv run --frozen --no-sync gnn mcp list
```

| Symptom | Diagnosis |
| --- | --- |
| Core import fails | Confirm locked environment/runtime and preserve traceback; required failures are not optional skips. |
| Optional backend unavailable | Read structured readiness, provision its declared extra/toolchain and verify admitted models. |
| Julia unavailable | Use committed projects from [setup](../SETUP_GUIDE.md); distinguish launcher, import and probe failures. |
| Provider unavailable | Verify the selected provider/model/configuration and preserve refusal or unfinished work. |
| Model/artifact counts differ | Inspect frozen selection and current-run indexes; old directories cannot establish new evidence. |
| Timeout/cancellation | Inspect all preparation, execution, stream and cleanup phases against the shared deadline. |
| Tests/coverage differ | Compare environments, markers, JUnit and coverage; investigate failures and qualify optional skips. |
| Token/hydration/pair drift | Complete [custody](../docs/development/fep_lean_paired_revision.md); never suppress an unverified contract. |

Replay into a fresh task-owned output root. Tracked `output/` contains manuscript
and publication evidence; blanket deletion is not recovery. Skipping completed
step numbers is not a resumable run; use the
[durable-run contract](../docs/development/durable-runs.md). Do not disable
plugins or raise undocumented timeout flags to hide failures.

See [operations troubleshooting](../docs/gnn/operations/gnn_troubleshooting.md).

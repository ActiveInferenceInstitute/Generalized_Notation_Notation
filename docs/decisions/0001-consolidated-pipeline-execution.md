# ADR 0001: Consolidated pipeline execution

- **Status:** Accepted (first slice landed)
- **Date:** 2026-09-10
- **Scope item:** SCOPE-2026-09-10 / S2-11 (V4-STAGE)

## Context

The GNN pipeline is a 25-step numbered-script contract: `main.py` selects and
sequences `0_template.py` … `24_intelligent_analysis.py`, and
`execute_pipeline_step` (`src/gnn/main.py`) launches **one subprocess per
step**. Every numbered script is a thin orchestrator that parses CLI arguments
(`gnn.utils.pipeline_orchestration.pipeline_template.create_standardized_pipeline_script`) and calls a
single module function. `src/gnn/pipeline/step_registry.py` is the single
source of truth for that contract: `STEPS` (stem, description,
`module_function`), the derived lookup maps, and
`CONSOLIDATED_STEP_ALIASES` (the renumbering seam).

The subprocess-per-step shape has real costs and real benefits:

- **Costs:** one Python interpreter startup and full `gnn` import per step;
  argument passing across a CLI boundary that each step must re-parse;
  results handed back only as files plus a captured exit code, so parsed
  models are re-derived downstream instead of passed forward in memory.
- **Benefits:** a hard isolation boundary per step (process exit codes, memory
  and timeout accounting, crash containment, stdout/stderr capture), which
  matters for LLM calls, rendering arbitrary generated code, and the
  testing-matrix folder dispatch.

Consolidation must therefore not destroy the numbered-script surface that
tests, docs, MCP metadata, and users rely on.

## Decision

The 2026-10-01 current-run ownership amendment requires a killable subprocess
boundary for top-level runs, including `--consolidated-steps`. Receipts record
the fallback reason. The standalone callback API remains cooperative only.
See [the migration contract](../development/run_ownership_migration.md) for
selection, evidence, deadlines, and compatibility limits.

1. **The 25-step numbered-script contract is preserved as the CLI surface.**
   `step_registry.py` remains the single source for step metadata and wiring;
   numbered scripts keep existing and remain directly runnable.
2. **Execution consolidates through a shared in-process step executor.**
   Numbered scripts delegate to
   `gnn.pipeline.step_executor.execute_step_in_process` instead of one
   subprocess per step. The executor is registry-driven: it imports
   `gnn.<stem>` and calls `StepInfo.module_function`, mirroring the argument
   contract of `create_standardized_pipeline_script`
   (`StepConfiguration`-declared args forwarded as keyword values, standard
   `<stem>_output` directory, shared `coerce_step_exit_code` semantics). It
   adds no second step mapping.
3. **Adoption is additive and opt-in.** A new `--consolidated-steps` flag
   (default off) routes supported steps in the serial path through the
   executor. The subprocess path is untouched and remains the default; it
   remains mandatory for isolation-sensitive steps. `--parallel` runs the
   subprocess path in this slice. Steps that decline consolidated execution
   (not whitelisted, or testing-matrix folder dispatch active) transparently
   fall back to subprocess.
4. **Receipts record which mode ran.** Every recorded step carries an
   `execution_mode` field (`"consolidated"` or `"subprocess"`,
   defaulted by `_record_step_result`), and `arguments.consolidated_steps`
   appears in the pipeline execution summary receipt.

## First slice (V4-STAGE slice 1)

`CONSOLIDATED_IN_PROCESS_STEMS = {"0_template", "3_gnn", "5_type_checker"}` —
the discovery/schema steps — for `--target-dir` inputs. The slice proves the
equivalence contract with `tests/pipeline/test_step_executor.py`: in-process
step 3 produces the same `3_gnn_output` artifact set and model counts as
subprocess mode on `input/gnn_files/basics`, the receipt records the mode, and
unknown steps are refused (`UnsupportedStepError`).

## Consequences

**Positive**

- Fewer interpreter startups per consolidated run; parsed models already live
  in the caller's memory, opening the path to passing them forward between
  consolidated steps without re-reading disk.
- One argument surface (`StepConfiguration`) and one exit-code contract drive
  both modes.
- The whitelist (`CONSOLIDATED_IN_PROCESS_STEMS`) is the single, reviewable
  seam for expanding consolidation.

**Negative / accepted limitations of slice 1**

- No per-step timeout or stdout/stderr capture in consolidated mode; a hung or
  crashing module function runs in-process (exceptions become `exit_code` 1
  receipts).
- Isolation-sensitive steps (LLM analysis, code rendering/execution,
  arbitrary-dependency steps) stay on subprocess mode by design.
- Consolidation and `--parallel` are mutually deferred: the parallel tier
  executor keeps spawning subprocesses.

## Alternatives considered

- **Rewrite `main.py` to always run steps in-process** — rejected: it breaks
  isolation guarantees for steps that execute generated code and LLM calls,
  and couples pipeline survival to single-step memory/timeouts.
- **Ship consolidated execution as a permanent replacement behind a version
  bump** — rejected: no opt-out path, and the subprocess mode is still the
  reference semantics the tests compare against.
- **Extend the numbered scripts to import a shared runner themselves**
  (scripts call in-process when imported) — rejected: changes all 25 scripts'
  behavior and entangles CLI parsing with in-process invocation; the executor
  calling `module_function` directly is smaller and keeps scripts as the CLI
  surface only.

## References

- `src/gnn/pipeline/step_registry.py` (`STEPS`,
  `CONSOLIDATED_STEP_ALIASES`, `CONSOLIDATED_IN_PROCESS_STEMS`)
- `src/gnn/pipeline/step_executor.py`
- `src/gnn/main.py` (`_consolidated_step_selected`,
  `execute_pipeline_step`, `_record_step_result`)
- `tests/pipeline/test_step_executor.py`

## Status log

| Date | Slice | Status | Notes |
|------|-------|--------|-------|
| 2026-09-11 | D: parallel-tier consolidation | Landed | Extracted `_execute_selected_step` in `src/gnn/main.py`; the serial loop and the parallel tier's ThreadPoolExecutor worker submission both route through it, so `--consolidated-steps` now works under `--parallel`. Whitelisted stems run in-process via `execute_step_in_process`, everything else keeps the subprocess path (mixed modes in one run are legal). Receipt normalization, dependency-wave ordering, and `GNN_RUN_ID` scoping unchanged. Restriction recorded: the parallel tier uses threads (shared process), so future whitelist expansion must only admit stems that are safe to run concurrently in one process; the current discovery stems (`0_template`, `3_gnn`, `5_type_checker`) are dependency-ordered and never share a tier. Verified by `tests/pipeline/test_parallel_consolidated.py` (mixed-mode receipts, serial/parallel aggregation parity, in-process execution through the real executor). |
| 2026-09-11 | A+C: in-process timeout/capture + parsed-model carrier | Landed | Slice A (`step_executor.py`): in-process steps run on a worker thread with a wall-clock timeout from `gnn.pipeline.step_timeouts` — the same knob as the subprocess tier (`GNN_STEP_TIMEOUT_{N}`/`GNN_STEP_TIMEOUT_SCALE` honored; per-call `timeout_seconds=` override) — and step `print()`/`sys.stdout`/`sys.stderr` output is tee'd: streamed to console AND captured into the receipt's `stdout`/`stderr` fields (thread-scoped, ref-counted install so concurrent consolidated steps never leave the process streams swapped). Timeout receipts mirror the subprocess tier exactly (exit code -1, FAILED via the same `status_from_exit_code` path, partial captured streams, receipt schema keys unchanged). Honest limit recorded in the receipt's stderr text: in-process steps cannot be force-killed — `future.cancel()` is best-effort, the worker thread keeps running and its further output is discarded from the receipt, and a hung step delays interpreter exit (non-daemon pool thread). Slice C: opt-in `collect_parsed_model=True` on `execute_step_in_process`; after a successful in-process step 3 the executor reads `gnn_processing_results.json` plus every referenced `{model}_parsed.json` once into a per-run in-memory carrier and forwards it as a `parsed_model` kwarg to steps 7/8 — `process_export` and `process_visualization` (threaded through `process_single_gnn_file`, `load_visualization_model`, `generate_combined_visualizations`) consume it, with staleness still derived from on-disk mtimes so `_viz_meta` matches a disk load; absent flag or malformed carrier leaves step behavior byte-for-byte unchanged, and artifacts are identical (pinned: carrier-on export byte-identical to the disk re-read modulo the run timestamp; note step-3 parsed JSON embeds per-node UUIDs/timestamps, so two independent step-3 runs can never byte-match — parity is asserted against one shared step-3 run). Limits: the carrier is per-run in-memory state (refreshed on each collecting step-3 run, dropped on any non-collecting or failed step 3); `main.py` call-site wiring of the flag is deferred — this slice the opt-in lives at the executor API level; capture sees stream writes, not logging handlers holding pre-open stream refs. Tests: `TestInProcessTimeoutAndCapture`, `TestParsedModelCarrier` in `tests/pipeline/test_step_executor.py`. |
| 2026-09-11 | B: whitelist expansion (export/viz/render) | Landed | `CONSOLIDATED_IN_PROCESS_STEMS` grew `+ {"7_export", "8_visualization", "11_render"}`. Per-stem subprocess-vs-executor artifact parity added for all three on `input/gnn_files/basics` (step 7 seeded with in-process step 3 on both roots; steps 8/11 seeded identically on both roots because step 8's artifact set depends on step-3 parse inputs — the JSON-primary path adds `*_ontology_legend.txt` — so parity requires identical seeds, not merely identical modes). Step 11 was investigated as conditional and included: the audit found no subprocess spawn anywhere in the render step path (`render/` spawns live only in READMEs and inside step-12 generated-script templates), no cwd-relative writes in the render path (every write resolves under `output_dir`; generated-script `OUTPUT_ENV`/`GNN_PROJECT_ROOT` defaults execute only in step 12), and the renderer import chain (`render/__init__.py`) is identical in both modes with no heavy third-party imports at module scope. Residual caveat: optional framework-toolchain imports ride in-process with the consolidated run — the same imports the subprocess reference pays, minus process isolation; hang containment is the generic consolidated-mode limitation, now covered by slice A's timeout. `9_advanced_viz` stays out: its D2-diagram path shells out to the optional `d2` CLI, and it remains the parallel-tier tests' non-whitelisted exemplar. Registry remains the single gate source: `main.py` routes only through `_consolidated_step_selected` → `can_execute_in_process`; no second stem list exists. Note for the parallel tier: steps 7/8 render with matplotlib (Agg) — thread-safe in-process, but concurrent consolidated runs of step 8 should expect shared matplotlib state. Verification: `uv run --extra dev python -m pytest tests/pipeline/test_step_executor.py -q` — 20 passed (step-3/7/8/11 parity, gate/whitelist, receipt, refusal, matrix-skip, slice-A timeout/capture, carrier tests). |
| 2026-09-19 | V4-STAGE: D2 CLI shell-out through the shared envelope | Landed | The step-9 D2 compile path (`advanced_visualization/d2_visualizer.py`) routes every `d2` CLI invocation through `gnn.execute.subprocess_envelope.run_subprocess_envelope`, imported lazily at the compile call site (the envelope import pulls the whole `gnn.execute` package init and must not ride on `advanced_visualization`'s import chain). The binary resolves once via `shutil.which` into an absolute argument vector; failures classify through the envelope: `TimeoutExpired` keeps the established "Timeout compiling to {fmt}" warning, non-zero exits report captured stderr, OSError receipts map to never-started. A hung `d2` compile is force-killed at the wall-clock timeout and folded into the diagram-level warnings instead of racing a bare `subprocess.run` kill. Step 9 itself remains out of `CONSOLIDATED_IN_PROCESS_STEMS` (subprocess tier). Standing thread-pool-tier limit re-recorded, unchanged from the slice-B note: steps 7/8 render with matplotlib (Agg) — thread-safe in-process, but concurrent consolidated runs of step 8 should expect shared matplotlib state; no behavioral change attempted for matplotlib. Verification: `uv run --extra dev python -m pytest tests/advanced_visualization/test_d2_envelope_shellout.py tests/visualization/test_d2_visualizer.py -q` — envelope routing, timeout classification, exit-code stderr, no-CLI graceful contract, and a `needs_d2_cli` live-compile test; the timeout test exercises the real envelope kill+drain path. |
| 2026-09-23 | E: BC-13 in-process force-kill (cooperative token) | Landed | The `FutureTimeoutError` path no longer abandons unconditionally. The executor creates a `CancelToken` (the subprocess envelope's type, imported lazily inside `step_executor.py` so the pipeline import chain stays decoupled from the heavy `gnn.execute` package init — the same coupling rule slice D2 set for `advanced_visualization`), injects it as a `cancel_token` kwarg when the resolved step function declares the parameter (same opt-in shape as the parsed-model carrier), and cancels it at the wall-clock deadline. The thread-scoped capture tees enforce the cancellation at the step's console-write safe points: once cancelled, the next `print()`/`sys.stderr.write` from the step thread raises `_StepCancelled` (a `BaseException`, so broad `except Exception` handlers cannot swallow it) and the step unwinds with no module change. After a bounded 2s grace window a thread that reached no safe point is abandoned — and the worker is now a daemon thread (the replaced per-step `ThreadPoolExecutor` threads were non-daemon, so a hung step used to delay interpreter exit indefinitely; the indefinite-slot hazard is gone for every consolidated step, cooperating or not). Receipts: additive `force_killed` key on every consolidated receipt (`False` on clean completion/skip, `True` on any timeout), schema otherwise unchanged and subprocess-tier compatible; the timeout stderr notice honestly distinguishes "stopped via the cooperative cancel token and joined" from "abandoned as a daemon (no cancellation safe point reached)". Timed-out artifacts are marked, never deleted: a `.gnn_step_timed_out` sentinel (deliberately extension-less so no consumer's rglob suffix scan picks it up) in the step's output directory records the step, the deadline, and every file whose mtime postdates the step's start; a later successful run of the same step removes the stale sentinel. Alternative rejected: subprocess workers for the whitelisted steps — true OS-level kill, but it re-adds per-step interpreter spawn, creates a second subprocess seam beside `main.execute_pipeline_step` (the ADR forbids a second step mapping), and kills the slice-C in-memory carrier; cooperative cancellation is the smaller mechanism and matches the documented lean CancelToken holdout (`execute/executor.py:293-294`; threading tokens through step implementations is the same follow-up shape as M-11 — steps that adopt `cancel_token` get prompt termination, while a C-blocked hang still cannot be interrupted mid-flight, recorded as the standing limit). Step-timeout env override parity (`GNN_STEP_TIMEOUT_{N}`/`GNN_STEP_TIMEOUT_SCALE`) is now covered at the executor tier. Verification: `uv run --extra dev python -m pytest tests/pipeline/test_step_executor.py tests/pipeline/test_step_timeouts.py -q` — 37 passed (`TestInProcessForceKill`: cooperative stop, non-cooperating abandon + subsequent step still runs, parallel worker-slot release, artifact marking without deletion, schema additivity over the subprocess tier; `TestInProcessExecutorEnvOverrides`: per-step env override and scale multiplier reach the in-process budget). |

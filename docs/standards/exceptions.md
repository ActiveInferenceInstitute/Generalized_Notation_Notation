# Exception standard: when `except Exception` is acceptable

- **Status:** Normative for new code; the occurrence ratchet is load-bearing
  through the gate listed in Enforcement.
- **Date:** 2026-09-24
- **Evidence basis:** the 2026-09-24 exception census (M-03 re-scope recorded in
  the [repository TO-DO](../../TO-DO.md) M-03 row: 304+ files carry broad
  handlers; spot-checked sites log and emit structured receipts) plus the
  file:line exemplars below, all re-verified in this tree at authoring time.
- **Baseline:** 1148 occurrences under `src/gnn/`
  (counted by
  [`scripts/check_gnn_doc_patterns.py`](../../scripts/check_gnn_doc_patterns.py)
  `EXCEPT_EXCEPTION_BASELINE`). Historical: 1183 at the wave-6 base;
  **tightened to 1182 on 2026-09-25** when the W8-R3 fold narrowed two new
  MCP guards to typed exceptions (`inspect_website` -> `OSError`;
  `get_pipeline_steps` -> `(ImportError, AttributeError)`). 2026-09-25 (W8-CD
  fold): tightened again to 1183 after narrowing two new CLI guards
  (`_cmd_complexity` -> `(ValueError, TypeError, OSError)`;
  `_cmd_benchmark` -> `(OSError, RuntimeError, ValueError)`), then re-raised
  +1 for the one sanctioned new broad site: the benchmark harness's
  accelerator-detect environment probe
  (`src/gnn/analysis/complexity/benchmark.py` — best-effort probe over
  optional third-party stacks; a typed set would crash receipt construction
  on exotic probe failures).
  2026-09-25 (W8-R4 fold): tightened to 1182 after narrowing the
  site-render batch guards to typed sets (render ->
  `(ValueError, TypeError, KeyError, IndexError)`, writes -> `OSError`,
  model-parse skip -> `(ParseError, ValueError, OSError)`).

  2026-10-01: tightened to 1171 after reviewed typed boundaries in the v4
  reliability program. The count excludes tuple handlers and remains an
  occurrence cap rather than a claim of complete exception correctness.

  2026-10-08: the reviewed minor/medium ownership and diagnostic work first
  lowered the measured ceiling to 1151. Subsequent authored-data and native
  consumer repairs reduce the actual AST census to 1148; the gate is tightened
  accordingly. These ceiling differences are not counts of individual repairs.

## Purpose

This standard codifies how the pipeline handles failures it does not type
specifically: the one broad handler form already in wide use —
`except Exception` — is acknowledged rather than rewritten wholesale, new
sites are held to explicit obligations, and the total is prevented from
growing by an occurrence-count ratchet. The census verdict on the long tail:
not a bug farm; a doctrine-and-acknowledgment problem.

## When `except Exception` is acceptable

A broad handler is acceptable at exactly three boundaries:

1. **Batch/per-item loops that must keep a run alive.** One bad item must not
   kill the batch; the failure is recorded and the loop continues.
   (`src/gnn/visualization/core/process.py:150` — collect into
   `processing_errors`, then `log.warning`.)
2. **Outer per-unit boundaries** where the traceback belongs in the log
   because the caller cannot recover it.
   (`src/gnn/visualization/graph/network_visualizations.py:280` —
   `logger.exception`.)
3. **Optional-feature or interop seams** where a failure degrades to a
   documented fallback value.
   (`src/gnn/research/processor.py:515` — `logger.debug` plus the documented
   `None` return.)

Everywhere else: raise, or catch the specific exception types the call can
actually produce. A broad handler inside a leaf function hides programmer
errors (typos, `AttributeError`) from every caller above it.

Never acceptable: a bare `except:` (name `Exception` explicitly; the count
found zero bare handlers in `src/gnn`), or a broad handler that neither logs,
records, re-raises, nor returns a documented sentinel.

Out of this count's scope by definition: tuple handlers that list `Exception`
among other types (2 in `src/gnn` today) and `except BaseException` (7 today).
Both follow the same obligations; the ratchet pins the exact-name form only,
so narrowing a tuple handler does not move the number.

## Mandatory obligations (every acceptable site)

1. **Log through the pipeline logging surface.** Import from
   [`gnn.utils.logging_utils`](../../src/gnn/utils/logging_utils.py)
   (single public entry point over
   `gnn.utils.logging.logging_utils`) — `PipelineLogger.get_logger(...)` for a
   module logger, or the `log_step_*` helpers inside step code.
2. **Emit a structured receipt.** Route the failure into the error surface the
   function already aggregates for callers — an errors list, a summary
   artifact, or the step log — with the failing item named
   (file, step, model, or request). A bare `pass`, `return None` without a
   log line, or exception text thrown away is a doctrine violation.
3. **Keep the traceback when it matters.** Use `logger.exception(...)` (or
   `exc_info=True`) at boundaries where the stack is the diagnosis; a
   one-line message is fine only when the failing input names the cause.
4. **Chain when re-raising.** `raise SpecificError(...) from e` preserves the
   original stack; never re-raise with the context discarded.
5. **State the fallback.** If the handler ends in a sentinel return or default
   value, the function docstring documents that behavior.

## Review checklist (for any new `except Exception`)

- [ ] Which of the three acceptable boundaries is this? (Name it in review.)
- [ ] Is a narrower exception type (or tuple of specific types) possible?
- [ ] Does the handler log with context, or record into a structured error
      surface the caller receives?
- [ ] Is the traceback preserved where the diagnosis needs it?
- [ ] Is every exit path (log / receipt / raise-from / documented sentinel)
      visible in the diff, with none silent?
- [ ] Does the occurrence ratchet stay at or below the baseline after this
      change (`python scripts/check_gnn_doc_patterns.py --strict`)?

## Enforcement

`scripts/check_gnn_doc_patterns.py` carries the occurrence ratchet next to its
footer-staleness probe: `EXCEPT_EXCEPTION_BASELINE` pins the count, one
occurrence = one `ast.ExceptHandler` typed exactly `Exception` in every
`.py` file under `src/gnn/`, and the probe fails on growth under `--strict`
while always printing the current count. It is wired into the same
`doc-patterns` gate invocation (`justfile:152`) inside `just quality`
(`justfile:156`).

Baseline movement: lower it freely as handlers are narrowed or removed; raise
it only in the same change that adds a decision note here, with the census or
consumer reason. The definition — exact-name handlers, parse-failed files
skipped — is pinned in the probe's docstring and must not drift without an
edit to this page.

## Related

- [composability.md](composability.md) — the seam standards this tree's
  structure is held to.
- [ADR index](../decisions/AGENTS.md) — where a baseline-raising decision note
  would land as a decision record.

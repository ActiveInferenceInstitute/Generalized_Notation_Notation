# Composability standard: the five seams

- **Status:** Normative for new code; the structural rules are load-bearing
  through the gates listed in the Enforcement section.
- **Date:** 2026-09-21
- **Decision record:** [ADR 048 — Composition seam](../decisions/048-composition-seam.md)
- **Evidence basis:** the 2026-09-21 five-seam census (scope passes
  CompScopeSeams and CompScopeConsumers), re-verified file-by-file in this
  tree before authoring. Every file:line citation below was re-checked on
  the branch that carries this standard.

## Purpose

This standard codifies how composition actually works in `src/gnn/` — five
de-facto seams the codebase already relies on — so new code follows one
convention instead of drifting into a second one. It also declares the
cross-repo consumer surfaces and keeps the two disciplines separate:
in-repo composition rules and cross-repo custody obligations have different
change protocols.

GNN's canonical step-composition mechanism is the registry-driven
consolidated step executor landed by
[ADR 0001](../decisions/0001-consolidated-pipeline-execution.md):
`execute_step_in_process` (`src/gnn/pipeline/step_executor.py`), the
whitelist `CONSOLIDATED_IN_PROCESS_STEMS`
(`src/gnn/pipeline/step_registry.py:355`), dispatched from
`_execute_selected_step` (`src/gnn/main.py:977`) in both the serial loop
and the parallel tier. A tool-level composition surface
(`compose`/`pipe`/`find_tools`/`lift`) was proposed by the imported
five-seam doctrine; ADR 048 records that it is superseded, not adopted.

## The five seams

### Seam A — orchestrator → module processor

**Canonical pattern.** A numbered script `src/gnn/N_<name>.py` is a thin
orchestrator: the src-layout bootstrap, one import of its implementation
module, and a single `create_standardized_pipeline_script` call
(`src/gnn/utils/pipeline_orchestration/pipeline_template.py`). Step logic
belongs in `src/gnn/<module>/`, never in the numbered file.

**Conformance.** All 25 numbered scripts conform — each carries the
bootstrap, one module import, and the template call (verified 2026-09-21).

**Exception.** COMP-010 (tracked, see the exceptions table): the bootstrap
itself is 25 hand-rolled copies (`0_template.py:38` …
`4_model_registry.py:39`; `2_tests.py:22`, with a second repository-root
copy at `2_tests.py:41`). Direction: hoist into
`create_standardized_pipeline_script`. Growth of the numbered files is
additionally bounded by the thin-orchestrator gate (see Enforcement).

### Seam B — pipeline.config output-dir / registry

**Canonical pattern.** Every consumer resolves a step output directory
through `gnn.pipeline.config.resolve_step_output_dir`
(`src/gnn/pipeline/config.py:205`). The helper documents the single shared
policy: delegate unconditionally, fail loud on an unimportable `gnn`
package, and return the caller-supplied directory unchanged for genuinely
standalone use (`src/gnn/pipeline/config.py:228-233`).

**Conformance.** Conformant since the 2026-09-15 W2-J1 landing: both
wrapper call sites are thin delegates with no policy of their own —
`resolve_execution_dir` (`src/gnn/analysis/framework_common.py:126-136`)
and `resolve_output_root` (`src/gnn/gui/runner.py:81-91`).

No open exceptions.

### Seam C — the gnn.utils PEP 562 facade

**Canonical pattern.** Consumers import facade names from `gnn.utils`; the
facade resolves them at runtime through the `__getattr__` import map
(`src/gnn/utils/__init__.py:279-283`). Concern packages never import the
facade: a leaf module imports its sibling leaf directly
(`from gnn.utils.arguments.arg_parsing import ArgumentParser`, as at
`src/gnn/utils/pipeline_orchestration/pipeline_template.py:136`). Logging
has a single entry: `gnn.utils.logging_utils`.

**Conformance.** Enforced, not aspirational. Three import-linter contracts
(`pyproject.toml:476-534`) — the facade stays lazy (I1/I2), concern
packages never import the facade (I5), and the logging single entry
(SC-38) — were restructured in the wave that landed this standard with
`as_packages = false`, so every source/forbidden pair is a real edge check,
and wired into the justfile quality chain, the CI 3.12 leg, and
local-gates (see Enforcement). Live receipt on this branch: 3 contracts
kept, 0 broken. COMP-007 (a facade import inside the
`pipeline_orchestration` concern, census-flagged against
`pipeline_template.py:135-136` on main) was remediated in the same wave by
the leaf-path import above and is no longer an open exception.

### Seam D — kernel packages

**Canonical pattern.** `gnn.schema`, `gnn.types`, `gnn.frameworks`, and
`gnn.parsers` are kernel packages: they depend on each other, the `gnn`
root, and shared concerns under `gnn.utils` (for example
`gnn.utils.runtime_safety.safe_eval`, used at
`src/gnn/schema/parser.py:261` and
`src/gnn/parsers/markdown_parser_parameter.py:133`). They never import
upward into orchestration or application surfaces (`main`, `cli`, `api`,
`mcp`, `pipeline`, `render`, `gui`, `export`, `extract`).

**Conformance.** Conformant; the upward-edge scan is empty (verified
2026-09-21).

No open exceptions.

### Seam E — registries

**Canonical pattern.** One registry per family, and every list derives from
it: steps from `STEPS` and its derived maps
(`src/gnn/pipeline/step_registry.py:52,233-234`); export formats from
`gnn.export.registry` (`DEFAULT_PIPELINE_FORMATS` at
`src/gnn/export/registry.py:59`, writers via `resolve_format_writer` at
`src/gnn/export/registry.py:128`); render frameworks from
`gnn.render.framework_registry` (`get_available_renderers` at
`src/gnn/render/framework_registry.py:221`; `get_lite_frameworks` at
`src/gnn/render/framework_registry.py:307`, which delegates to
`gnn.frameworks.LITE_FRAMEWORKS`). The consolidated-executor whitelist
(`CONSOLIDATED_IN_PROCESS_STEMS`, `step_registry.py:355`) is the single
reviewable seam for expanding in-process execution — it adds no second step
mapping (ADR 0001, decision 2).

**Conformance.** Conformant.

No open exceptions.

## Tracked exceptions

Every integration that does not fit the five seams is a named, tracked
exception — never hidden, never silently normalized. Tracker refs are the
COMP ids of the 2026-09-21 census.

| Ref | Severity | Site | Issue | Direction |
| --- | --- | --- | --- | --- |
| COMP-003 | HIGH | `src/gnn/sapf/` vs `src/gnn/audio/sapf/` | One module family, two canonical homes: `sapf/__init__.py:5-12` re-exports the audio implementation, while the `sapf/` package also ships its own metadata and an `mcp.py`; tests exercise both paths (`tests/sapf/test_sapf_processor.py:14`, `tests/test_fast_suite.py:221`) | Pick one home; the other stays a pure re-export surface with a golden export test; define metadata once; migrate callers and tests |
| COMP-004 | MEDIUM | `src/gnn/cli/__init__.py:979-980` | Private cross-module import: the reproduce command reaches into `gnn.main` internals (`_resolve_steps_to_execute`) | Promote to a public API with a contract test |
| COMP-005 | MEDIUM | `src/gnn/api/parity.py:569,607,670,751,774` | Parity by mirror: the API re-states `gnn.cli._cmd_*` severity and payload checks in five handlers | Extract the mirrored check bodies into shared pure helpers both surfaces consume |
| COMP-006 | MEDIUM | `src/gnn/utils/__init__.py:280-281` vs `src/gnn/export/core.py:10`, `src/gnn/pipeline/pipeline_step_template.py:46` | Dual canonical path for `get_output_dir_for_script` and `execute_pipeline_step_template` (facade map AND direct `gnn.pipeline` import) | Single path = `gnn.pipeline`; drop the facade re-exports; update the facade golden list (`tests/tests/test_infrastructure_exports.py`) |
| COMP-009 | LOW | `src/gnn/pipeline/pipeline_step_template.py:46-72,561-565` + `src/gnn/pipeline/pipeline_validation.py` | Pre-registry second convention kept inside the composition package (template body plus an 871-line self-validation module; only consumers `tests/pipeline/test_pipeline_infrastructure.py:74-142`) | Retire both; move the guidance to `src/gnn/pipeline/AGENTS.md`; migrate the tests to `step_executor` and `step_registry` |
| COMP-010 | LOW | `0_template.py:38` … `4_model_registry.py:39` (25 files) | 25 hand-rolled src-layout bootstrap copies | Hoist into `create_standardized_pipeline_script` |
| COMP-011 | INFO (folded) | `src/gnn/gui/oxdraw/processor.py:18,87`; `src/gnn/mcp/server_stdio.py:32-33`; `src/gnn/mcp/server_http.py:64-65` | Documented integration notes: the websocket bridge contract note and the direct-script import recovery in the MCP servers | Recorded, not re-reported |

Remediated in the wave that landed this standard (no longer open):
COMP-007 — the `pipeline_template.py:136` facade import was inverted to the
leaf path, and `lint-imports` reports 3 contracts kept, 0 broken on this
branch — and COMP-008 — the import-linter contracts were unwired and are
now wired (see Enforcement).

## Consumer seams: declared cross-repo surfaces

Everything else in `src/gnn/` is private to this repository. Sibling
repositories consume exactly these five declared surfaces:

1. **Export surface** — `gnn.export.geo_infer`,
   `gnn.export.geo_infer_gaussian`, `gnn.export.geo_infer_factored`. The
   artifact contract is `gnn-geo-infer/1` (`CONTRACT_VERSION` at
   `src/gnn/export/geo_infer.py:18`) and every artifact carries
   `provenance.source_sha256` (`src/gnn/export/geo_infer.py:163`); the
   three-variant dispatch lives in `export_to_geo_infer`
   (`src/gnn/export/geo_infer.py:168-213`). Consumer:
   `GEO-INFER-TEST/validate_gnn_interchange.py` in the GEO_INFER checkout,
   driven GNN-side by `scripts/run_geo_interchange_checks.py`.
2. **Runtime-bridge surface** — `gnn.extract.pomdp_extractor`
   (`extract_pomdp_from_file` at
   `src/gnn/extract/pomdp_extractor.py:1628`, content variant at `:1663`)
   and `gnn.schema.parse_state_space` (`src/gnn/schema/parser.py:164`).
   Consumers inject this checkout onto `sys.path` and import exactly these
   symbols (fep_lean's `verify-document`).
3. **Syntax surface** — `docs/gnn/gnn_syntax.md` and
   `src/gnn/pipeline/step_registry.py`. Exactly these two files form the
   hand-maintained syntax pin that the bridge status check compares against
   the live bytes.
4. **Mirror contract** — `docs/other/fep_lean/bridge-contract.md`, the
   mirror of fep_lean's canonical
   `docs/design/gnn-bridge/bridge-contract.md`. The two bodies stay equal
   (only the Canonical/Mirror table rows may differ); substance changes
   bump the contract version and land in both checkouts in the same
   working session.
5. **Pin files** — `.github/fep-lean-pair.json` and
   `.github/gnn-pair.json`. Each side re-pins its own companion; a red
   paired-revision run is the drift signal and is repaired by re-pinning,
   never by suppressing the check.

### Evidence-plane discipline

Each check proves one thing and nothing more:

- `verify-document` proves the notation syntax and well-formedness of a
  GNN document.
- The bridge status proves custody: that the pinned companion revision and
  this checkout agree on source binding digests and the syntax surface.
- The interchange checks prove deterministic replay of exported artifacts
  inside the GEO_INFER environment.
- None of them prove Lean theorems or the truth of any notation-mapping or
  geographic claim. Evidence planes stay distinct; a green paired run says
  "the two checkouts are in the state the companion reviewed" and nothing
  else (canonical wording:
  [docs/development/fep_lean_paired_revision.md](../development/fep_lean_paired_revision.md)).

### Change protocol (announce-and-repin)

Changes to surfaces 1-5 are announce-and-repin events: follow the closeout
ordering in
[docs/development/fep_lean_paired_revision.md](../development/fep_lean_paired_revision.md)
— all content edits land first (the manuscript/token ritual runs at the
final content state when manuscript inputs changed), then the fep_lean-side
bridge re-pin, then the GNN pin bump as the final commit; the GEO_INFER
direction is the mirror image.

Edits outside surfaces 1-5 must not force sibling re-pinning. Census
break-class B1 (the bridge's source binding covering a whole-tree owner glob)
was repaired on the fep_lean side: the GNN owner roster now binds the declared
contract surface only — `pyproject.toml`, `uv.lock`, `src/gnn/main.py`,
`docs/gnn/gnn_syntax.md`, `src/gnn/pipeline/step_registry.py` and the mirrored
`docs/other/fep_lean/bridge-contract.md` — so ordinary `src/gnn/**/*.py`
implementation edits no longer re-drift the pair. Emitted-document freshness is
verified at run time by `emit --check` instead of by pinning the implementation
tree. Widening or narrowing that roster remains an fep_lean-side design change,
never a GNN-side pin edit.

## Enforcement

The structural rules above are load-bearing through these gates, all wired
in this repository (the wiring landed in the wave that carries this
standard):

- `scripts/check_thin_orchestrators.py` — numbered orchestrator line cap
  (150 hard, 133 ratchet file).
- `scripts/check_flag_parity.py` — argparse-registered flags vs
  maintained-doc flag mentions (ratcheted).
- `scripts/check_dep_hygiene.py` — extras and mypy overrides vs actual
  imports.
- `uv run lint-imports` — the three import-linter contracts
  (`pyproject.toml` `[tool.importlinter]`), wired into the justfile quality
  chain, the CI 3.12 leg, and local-gates.
- `scripts/audit_validate_surface.py` — wired into just quality and
  local-gates.

A violating change to any seam fails at least one of these gates in CI.

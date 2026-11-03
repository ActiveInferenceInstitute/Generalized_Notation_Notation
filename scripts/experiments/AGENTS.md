# scripts/experiments — agent rules

## Overview

One-off, offline tooling that sits beside the gates but is never part of them:
`add_module_docstrings.py` (source-mutating codemod),
`run_pymdp_gnn_scaling_analysis.py` (the PyMDP scaling study, with
`pymdp_scaling_config.yaml`) and `verify_pomdp_pipeline.py` (retained offline
verifier).

## Purpose

Keep exploratory runs reproducible without letting them into the pipeline or the
gates. Maintained equivalents live in `scripts/` proper — for the POMDP check
that is `scripts/check_pomdp_gridworld_outputs.py`.

## Directory Identity

Experiments subpackage of `scripts/`; [README.md](README.md) owns the per-script
description.

## Rules

- Nothing here runs in CI. A result worth citing moves into `docs/development/`
  with its receipt, or becomes a gate script under `scripts/` with its contract
  test in `tests/scripts/`.
- Scripts here are not pipeline steps: the 25 numbered drivers stay under
  `src/gnn/`
  ([consolidation ADR](../../docs/decisions/0001-consolidated-pipeline-execution.md)).
- Keep them deterministic and offline, and keep the scaling configuration in
  this directory beside the script that reads it.
- `add_module_docstrings.py` writes to source: run it deliberately, review the
  diff, and never wire it into a gate.

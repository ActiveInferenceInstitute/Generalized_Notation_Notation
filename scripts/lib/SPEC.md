# Specification: scripts/lib/

## Purpose

Shared utility modules for scripts under `scripts/`. `shared.py` provides common functions for repo path resolution, skip-path logic, generated-output detection, and standardized CLI flags (`--strict`). `manuscript_figure_tokens.py` provides the manuscript token-map loader that records figure provenance.

## Dependencies

- **Runtime**: Python 3.12+ standard library (`pathlib`, `argparse`, `typing`)
- **No external dependencies** — zero install footprint

## Exports

| Symbol | Type | Description |
|---|---|---|
| `repo_root()` | `-> Path` | Repository root, derived from file location |
| `should_skip_path(path, root)` | `-> bool` | True when path matches skip/generated-output patterns |
| `is_generated_output(rel)` | `-> bool` | True when relative path is under `_output`/`_outputs` |
| `add_strict_flag(parser)` | `-> None` | Adds `--strict` to an argparse parser |
| `exit_with_findings(count, strict)` | `-> int` | 0 if no findings or non-strict, 1 if strict + findings |
| `load_tokens()` | `-> RecordingTokens` | The manuscript token map, recording every key read; consumed pairs written to `$GNN_FIGURE_TOKEN_PROVENANCE` at exit (`manuscript_figure_tokens.py`) |
| `RecordingTokens` | `dict` subclass | Token map that records reads through `__getitem__`/`get` (`manuscript_figure_tokens.py`) |

## Rules

1. All functions must have type annotations.
2. All functions must have docstrings.
3. No external dependencies beyond Python stdlib.
4. Any script in `scripts/` may import from `scripts.lib.shared` using:
   ```python
   from scripts.lib.shared import repo_root, should_skip_path, exit_with_findings
   ```
5. Every `scripts/manuscript_fig_*.py` generator that needs a value from
   `output/data/manuscript_variables.json` MUST read it through `load_tokens()`.
   Opening the file directly bypasses the provenance record and lets a committed
   PNG print a count the producer has moved past;
   `src/tests/test_manuscript_figure_freshness.py` fails the suite on such a
   generator.
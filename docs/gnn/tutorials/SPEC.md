# Specification: Tutorial Documentation

## Scope
Self-contained walkthroughs for getting started with GNN. Each tutorial
should work end-to-end on a fresh clone + `uv sync`.

## Contents
| File | Level | Goal |
|------|-------|------|
| `quickstart_tutorial.md` | Beginner | First model in 15 minutes |
| `gnn_examples_doc.md` | Beginner–Intermediate | Annotated walkthroughs of sample models |
| `README.md` | Index | Entry point + navigation |

## Tutorial Contract
Every tutorial must:
1. Open with a prerequisites block (Python 3.11+, `uv`, repo clone)
2. Reference real files from `input/gnn_files/` (no invented examples)
3. Include expected output snippets so the reader can verify each step
4. End with a "Next steps" section linking further learning paths

## Versioning
Tutorials pin to documentation bundle v2.0.0; the engine release is canonical
in [pyproject.toml](../../../pyproject.toml). When the CLI changes, tutorials
must be re-run and updated.

## Status
Maintained. Before every release, `quickstart_tutorial.md` must be run
manually against the release candidate and updated if any step's output
changes.

# Manuscript - GeneralizedNotationNotation

This directory contains the canonical manuscript sources for:

**GeneralizedNotationNotation: A Text Language for Active Inference Models**

A standardized text-based language and processing pipeline for Active Inference generative models, transforming specifications into validation, visualization, simulation, and analysis outputs.

## File Inventory

- `config.yaml`
- `preamble.md`
- `references.bib`
- `00_abstract.md`
- `01_introduction.md`
- `02_system_context.md`
- `03_methods.md`
- `04_artifacts_and_evidence.md`
- `05_reproducibility.md`
- `06_limitations_and_next_steps.md`
- `S01_source_surface.md`
- `98_symbols_glossary.md`
- `99_references.md`
- `AGENTS.md`
- `README.md`
- `SYNTAX.md`

## Source Surfaces

| Surface | Role |
|---|---|
| `src/` | Source directory to inspect before turning prose into claims. |
| `input/` | Source directory to inspect before turning prose into claims. |
| `scripts/` | Source directory to inspect before turning prose into claims. |
| `docs/` | Source directory to inspect before turning prose into claims. |
| `output/` | Source directory to inspect before turning prose into claims. |

## Verification

The template's Stage 03 renders the PDF, HTML, and slides. Stage 04 validates
those outputs and writes the publication evidence. Use a disposable template
checkout at the revision declared in
`output/data/manuscript_render_manifest.json`, with this repository as a real
directory at `projects/active/GeneralizedNotationNotation`.

The [run guide](../RUN_GUIDE.md#manuscript-rendering-and-custody) gives the exact
commands, environment settings, and artifacts to retain. Use its verification
path to compare an unchanged baseline with a fresh validated render. Use its
renewal path to record publication evidence after an intentional source or
template update. [Reproducibility](05_reproducibility.md#manuscript-reproducibility)
describes the variable and figure contracts.

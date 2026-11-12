# Manuscript - GeneralizedNotationNotation

This directory contains the template-format manuscript sources for:

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

The authoritative render command is the template's `stage_03_render.py`, with
this repository present as a **real directory** at
`projects/active/GeneralizedNotationNotation`. A `link-projects` symlink is
not sufficient for hydration custody. Keep the reference template read-only;
use a disposable copy when preparing this layout.

From the project checkout:

```bash
uv run --frozen --no-sync python -m scripts.manuscript_build_figures
```

From the prepared template root:

```bash
uv run --frozen python scripts/pipeline/stage_03_render.py \
  --project GeneralizedNotationNotation
```

Then, from the project checkout, verify the fresh render against the existing
chain before recording renewed evidence:

```bash
uv run --frozen --no-sync python scripts/z_verify_fresh_render.py
uv run --frozen --no-sync python scripts/z_record_manuscript_render_manifest.py
uv run --frozen --no-sync python scripts/check_manuscript_tokens.py --strict
uv run --frozen --no-sync python scripts/check_hydrated_prose.py
```

Export `GNN_TEMPLATE_REVISION` from the actual template checkout for the custody
check. A stale-chain verdict requires a deliberate renewal after a successful
render, not suppression. See the [run guide](../docs/RUN_GUIDE.md#manuscript-rendering-and-custody)
for the exported-copy build epoch, measured render and toolchain; see
[reproducibility](05_reproducibility.md#manuscript-reproducibility) for the source,
variable and figure contracts.

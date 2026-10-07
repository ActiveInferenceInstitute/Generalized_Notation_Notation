# Extract Module

Headless extraction of POMDP state spaces from GNN specification files, with a versioned JSON envelope contract suitable for machine consumption by external tools.

## Module Structure

```
src/gnn/extract/
├── __init__.py            # JSON envelope facade: extract_to_json, main (schema 1.0.0)
├── pomdp_extractor.py     # POMDP/state-space extraction (stdlib-only)
├── __main__.py            # python -m gnn.extract entry point
├── README.md              # This documentation
└── AGENTS.md              # Agent scaffolding documentation
```

## Cross-Repo Note

This module is **bridge-pinned**: the fep_lean bridge `verify-document` operation imports `gnn.extract.pomdp_extractor.extract_pomdp_from_file` as its render route for reading emitted GNN documents. The bridge contract (`docs/other/fep_lean/bridge-contract.md`, §13) cites this import path; do not move or rename it without updating the fep_lean checkout and re-pinning source custody.

## Usage

### Headless CLI (documented entry point)

```bash
python -m gnn.extract FILE [--strict|--no-strict] [--compact]
```

- Success: prints the extracted state-space payload JSON, exit code 0.
- Failure: prints the error envelope JSON, exit code 1.

### Programmatic

```python
from gnn.extract import extract_to_json

json_text = extract_to_json("input/gnn_files/example.gnn", strict_validation=True)
```

`extract_to_json(path, *, strict_validation=True, on_error="lenient", compact=False) -> str` always returns a JSON string and never raises:

- **Success**: the JSON object is `POMDPStateSpace.to_dict()` (no `status` key). `compact=True` emits `separators=(",", ":")` with no indentation; the default emits `indent=2`.
- **Failure**: the error envelope `{"status": "error", "error": {"code", "message", "line", "section"}}`. Structured codes come from the extractor (e.g. `GNN-E002` shape contradictions, `GNN-E006` parameter-parse failures); a bare failure uses `GNN-E000`.

## Extraction Details

`pomdp_extractor.py` is stdlib-only at module scope: no GNN pipeline imports, no third-party dependencies. It parses the GNN text directly and produces a `POMDPStateSpace` describing:

- State factors, observation modalities, and control factors with dimensions.
- Matrices A, B, C, D/E with nested shapes and per-matrix provenance (parameter-parse failures are recorded in `matrix_provenance` in every `on_error` mode — never silently dropped).
- Continuous-model support (`x, F, H, Q, R` parameters) with the family (`finite` / `continuous`) reported in the payload.
- Model name and annotation when present.
- Optional `time` and `equations` strings containing the authored section bodies.
  Extraction trims enclosing whitespace and preserves internal text. These
  additive fields retain source metadata; equations are not evaluated as code.

The renderer conversion preserves these fields. Generic JAX uses the Time
declaration to distinguish one-observation static conditioning from dynamic
filtering; passive models emit no selected actions or control objective. JAX
HMM output is filtering, not backward smoothing. The typed-object renderer
adapter preserves structured Time and Equations metadata as well. For passive
two-dimensional B without C, it records the neutral compatibility C vector as
`passive_model_adapter` provenance with the `passive_model_zero_preferences`
adapter note, matching file extraction without changing the authored model.

The `OnErrorMode` literal (`"lenient"`, `"raise"`, `"collect"`) controls how extraction errors surface; `extract_to_json` maps `raise` failures into the error envelope.

## Related Documentation

- [Bridge contract](../../../docs/other/fep_lean/bridge-contract.md) — fep_lean ↔ GNN contract; §13 cites this module as the render route.
- [POMDP extraction overview](../../../docs/gnn/integration/gnn_implementation.md)

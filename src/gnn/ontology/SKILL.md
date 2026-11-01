---
name: gnn-ontology
description: GNN Active Inference ontology processing and validation. Use when working with ActInfOntologyAnnotation sections, mapping GNN variables to ontology terms, validating semantic annotations, or exploring Active Inference concept hierarchies.
---

# GNN Ontology Processing (Step 10)

## Purpose

Processes Active Inference ontology annotations in GNN models. Maps GNN variables to standardized Active Inference concepts and validates semantic consistency.

## Key Commands

```bash
# Run ontology processing
python src/gnn/10_ontology.py --target-dir input/gnn_files --output-dir output --verbose

# As part of pipeline
python src/gnn/main.py --only-steps 10 --verbose
```

## Ontology Term Mapping

| GNN Annotation | Ontology Concept | Description |
| ---------------- | ------------------ | ------------- |
| `LikelihoodMatrix` | A matrix | Observation model |
| `TransitionMatrix` | B matrix | State dynamics |
| `HiddenState` | s vector | Latent state beliefs |
| `Observation` | o vector | Sensory observations |
| `Policy` | π | Action sequences |
| `Preference` | C matrix | Preferred observations |

## API

```python
from gnn.ontology import (
    process_ontology,
    parse_gnn_ontology_section,
    process_gnn_ontology,
    analyze_ontology_content,
    load_defined_ontology_terms,
    validate_annotations,
    suggest_terms,
    summarise_coverage,
    build_ontology_terms,
    generate_ontology_report_for_file,
    parse_annotation,
    ParsedAnnotation,
    OntologyTermIndex,
    OntologyProcessor,
    OntologyValidator,
)

# Process ontology section from GNN content
parsed = parse_gnn_ontology_section(gnn_content)

# Load defined ontology terms
terms = load_defined_ontology_terms()

# Validate annotations against defined terms (case-insensitive)
result = validate_annotations(annotations_list, terms)

# Single pure entry point: parse + load + validate in one call
analysis = analyze_ontology_content(gnn_content, terms)

# Nearest-term suggestions for unknown annotations
suggestions = suggest_terms(["x=HidenState"], terms)

# Compact coverage line for reports/LLM prompts
summary = summarise_coverage(result)

# Build a custom vocabulary in memory (no JSON file)
custom = build_ontology_terms(["Foo", "Bar"], descriptions={"Foo": "a foo"})

# Use OntologyProcessor class
processor = OntologyProcessor()
result = processor.process_ontology(data)

# Use OntologyValidator class
validator = OntologyValidator()
is_valid = validator.validate_ontology(content)
```

## Key Exports

- `process_ontology` — main pipeline processing function
- `parse_gnn_ontology_section` — extract ontology from GNN content
- `analyze_ontology_content` — pure parse + load + validate in one call
- `validate_annotations` — validate against known terms (case-insensitive)
- `suggest_terms` — nearest-ontology-term suggestions for unknown annotations
- `summarise_coverage` — human-readable coverage line for report/LLM consumers
- `build_ontology_terms` — build a vocabulary dict in memory
- `parse_annotation` → `ParsedAnnotation` — typed 3-tuple (key, value, comment)
- `OntologyProcessor` — class with `process_ontology()`, `validate_terms()`
- `OntologyValidator` — class with `validate_ontology()`, `check_consistency()`
- `OntologyTermIndex` — prebuilt case-insensitive vocabulary index with lookup, known_terms, validate, suggest and contains checks
- `resolve_declared_label` / `pinned_declared_index` / `verify_ontology_pin` — resolve declared labels to release-v5 `term-*` identifiers through the pinned vocabulary bridge (verify the copy against its pin with `just ontology-pin`)

## Output

- Ontology mappings in `output/10_ontology_output/`
- Validation reports for annotation consistency


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `extract_ontology_annotations`
- `list_standard_ontology_terms`
- `process_ontology`
- `validate_ontology_terms`

## References

- [AGENTS.md](AGENTS.md) — Module documentation
- [README.md](README.md) — Usage guide
- [SPEC.md](SPEC.md) — Module specification


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API

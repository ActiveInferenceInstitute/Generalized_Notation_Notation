# Ontology Module - Agent Scaffolding

## Module Overview

**Purpose**: Active Inference Ontology processing, validation, and term mapping for GNN models

**Pipeline Step**: Step 10: Ontology processing (10_ontology.py)

**Category**: Semantic Validation / Ontology Management

**Status**: Production Ready

**Version**: re-exports canonical package `__version__` (SC-16)

**Last Updated**: 2026-11-01

---

## Core Functionality

### Primary Responsibilities
1. Extract ontology terms from GNN models
2. Validate against Active Inference Ontology
3. Map GNN components to ontological concepts
4. Generate ontology compliance reports
5. Identify semantic inconsistencies

### Key Capabilities
- Term extraction from GNN specifications
- Ontology compliance validation
- Semantic relationship mapping
- Hierarchical concept analysis
- Cross-reference validation

---

## API Reference

### Public Functions

#### `process_ontology(target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs) -> bool`
**Description**: Main ontology processing function called by orchestrator (10_ontology.py). Processes GNN files for ontology validation and mapping.

**Parameters**:
- `target_dir` (Path): Directory containing GNN files to process
- `output_dir` (Path): Output directory for ontology results
- `verbose` (bool): Enable verbose logging (default: False)
- `strict_validation` (bool, optional): Require all terms to be in ontology (default: False; read from `**kwargs`)
- `recursive` (bool, optional): Process directories recursively (default: True; read from `**kwargs`)
- `ontology_terms_file` (Path, optional): Path to ontology terms JSON file (default: `src/gnn/ontology/act_inf_ontology_terms.json`; read from `**kwargs`)
- `**kwargs`: Additional processing options (other keys are accepted but not consumed)

**Returns**: `bool` - True if processing succeeded, False otherwise

```python
from gnn.ontology import process_ontology
from pathlib import Path

success = process_ontology(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/10_ontology_output"),
    ontology_terms_file=Path("src/gnn/ontology/act_inf_ontology_terms.json"),
    strict_validation=True,
)
```
#### `parse_gnn_ontology_section(content: str) -> Dict[str, Any]`
**Description**: Extract the ontology annotation section (e.g. `## ActInfOntologyAnnotation`, also accepted as the spaced `## ActInf Ontology Annotation` or the `## Ontology` alias) from GNN Markdown content.

**Parameters**:
- `content` (str): Raw GNN Markdown content

**Returns**: `Dict[str, Any]` - Parsed ontology content (annotations + any extracted fields)

#### `load_defined_ontology_terms(ontology_terms_file: Path | None = None, *, search_paths: Sequence[Path] | None = None) -> Dict[str, Any]`
**Description**: Load the ontology term dictionary. Precedence: explicit `ontology_terms_file` (authoritative, fails closed) → caller-supplied `search_paths` (warn-and-continue on each miss) → built-in module-relative paths → built-in default term set. The `search_paths` keyword is the dependency-injection hook for tests and alternate installs.

**Returns**: `Dict[str, Any]` - Term definitions (descriptions; ontology identifiers come from the pinned vocabulary bridge, never from the declared `uri` metadata)

#### `build_ontology_terms(terms: List[str], *, descriptions: Dict[str, str] | None = None, uris: Dict[str, str] | None = None) -> Dict[str, Any]`
**Description**: Build a normalized ontology-terms dictionary in memory (the shape `load_defined_ontology_terms` returns and `validate_annotations` consumes), without writing a JSON file. Rejects empty names, exact duplicates, and case-folded duplicates (e.g. `["A", "a"]`) so the built vocabulary upholds the case-insensitive lookup invariant. Complement of `_normalise_ontology_terms`.

**Returns**: `Dict[str, Any]` - Term-name -> `{"description", "uri"?}` mapping.

#### `validate_annotations(annotations: List[str], ontology_terms: Dict[str, Any] | None = None, *, ontology_bridge: Dict[str, Any] | None = None) -> Dict[str, Any]`
**Description**: Validate ontology annotations (e.g. `A=LikelihoodMatrix`) against the known ontology term set, then resolve each matched label through the pinned vocabulary bridge.

**Parameters**:
- `annotations` (List[str]): Annotation strings
- `ontology_terms` (Dict[str, Any] | None): Optional explicit ontology term set (default: module’s loaded terms)
- `ontology_bridge` (Dict[str, Any] | None): Optional bridge-document override (tests, alternate pins); omitted, the pinned copy is loaded

**Returns**: `Dict[str, Any]` - Validation details: `valid_annotations`, `invalid_annotations`, `matched_terms` (each carrying `ontology_id`, `ontology_status`, `ontology_term`, `ontology_reason` next to `annotation`/`term_name`/`description`), `suggestions`, `coverage_score`, `invalid_details` and `ontology_identifier_counts` (`resolved`, `unresolved`, `not_declared`, `bridge_unavailable`). Declared `obo:` values are never emitted as identifiers.

#### `process_gnn_ontology(gnn_file: str, ontology_terms: Dict[str, Any] | None = None) -> Dict[str, Any]`
**Description**: Process ontology annotations for a single GNN file path (reads file, parses ontology section, validates annotations). Delegates the parse→load→validate pipeline to `analyze_ontology_content`.

#### `analyze_ontology_content(content: str, ontology_terms: Dict[str, Any] | None = None) -> Dict[str, Any]`
**Description**: Single pure entry point that parses GNN content and validates its annotations, returning `{"ontology_data", "validation_result", "ontology_terms"}`. Shared by `process_gnn_ontology` (file input) and `OntologyProcessor.process_ontology` (content input) so the pipeline exists in exactly one place.

#### `suggest_terms(annotations: List[str], ontology_terms: Dict[str, Any] | None = None, *, max_distance: int = 3) -> List[Dict[str, Any]]`
**Description**: Return nearest-ontology-term suggestions for unknown annotations. Each result is `{"annotation", "suggested_term", "description", "distance"}` ranked closest-first (distance `0` = substring match). Reuses the same case-folded substring + Levenshtein heuristic as `validate_annotations`, exposed for LLM/report consumers.

#### `summarise_coverage(validation_result: Dict[str, Any]) -> str`
**Description**: Render a `validate_annotations` result as a compact human-readable coverage line (e.g. `"3/4 annotations valid (coverage 75.0%); 1 suggestion"`). Pure helper for report/LLM consumers.


#### `parse_annotation(annotation: str) -> ParsedAnnotation`
**Description**: Parse a `KEY=VALUE` annotation into a `ParsedAnnotation` NamedTuple (`key`, `value`, `comment`), still a 3-tuple so positional unpacking keeps working. `key`/`comment` are `None` when absent.

#### `generate_ontology_report_for_file(gnn_file: Path, output_dir: Path, *, ontology_terms: Dict[str, Any] | None = None) -> Dict[str, Any]`
**Description**: Generate and write a per-file ontology report JSON for a single GNN file.

#### `validate_ontology_terms(terms: List[str] | str | None = None) -> bool`
**Description**: Convenience validator for terms/annotations (returns boolean validity; used by integrations/tests).

#### Vocabulary bridge (`gnn.ontology.bridge`)
**Description**: Resolves declared vocabulary labels to ontology identifiers through the pinned Active Inference Ontology vocabulary bridge. `load_ontology_bridge`/`load_ontology_pin` read the copy (`active_inference_ontology_terms.json`, schema `active-inference-ontology/gnn-terms/v1`, release v5, 64 declared labels) and its pin (`ontology-pin.json`); `verify_ontology_pin` checks copy SHA-256, schema, embedded source/export digests, counts and the resolved/unresolved identifier invariants; `ontology_bridge_status` returns the compact record stored in run artifacts; `pinned_declared_index` (cached) returns the case-folded declared map; `resolve_declared_label(label, declared)` returns `DeclaredResolution(status, identifier, ontology_term, reason)` with statuses `resolved`, `unresolved` (`not in ontology release v5`), `not_declared` or `bridge_unavailable`. The declared `obo:ACTO_*`/`obo:TEMP_*` values are metadata only — no `ACTO` ontology is registered in the OBO Foundry, Bioregistry or OLS4 and `http://purl.obolibrary.org/obo/ACTO_000001` returns HTTP 404 — so they are never emitted as identifiers. Gate: `scripts/check_ontology_pin.py` (`just ontology-pin`).

### Public Classes

#### `OntologyProcessor`
**Description**: Convenience wrapper exposing `process_ontology()` and `validate_terms()` around the functional API.

#### `OntologyValidator`
**Description**: Validator exposing `validate_ontology()` and `check_consistency()` for quick boolean checks.

#### `OntologyTermIndex`
**Description**: Prebuilt case-insensitive index over an ontology vocabulary for batch callers. Construct once via `OntologyTermIndex(terms)`, `.from_file(path)`, or `.from_names(...)`; then use `lookup(value)` (O(1) case-insensitive membership), `known_terms()`, `validate(annotations)`, `suggest(annotations)`, `len()`, and `in` checks. Immutable after construction; delegates to the module-level pure functions so behaviour stays in lock-step with the functional API.

---

## Ontology Structure

### Active Inference Ontology Terms
- **Core Concepts**: Belief, Preference, Action, Observation
- **Mathematical**: Matrix, Vector, Probability, Distribution
- **Process**: Inference, Learning, Planning, Control
- **Architecture**: Agent, Environment, Model, Policy

---

## Configuration

### Configuration Options

Consumed `**kwargs` on `process_ontology()`:
- `ontology_terms_file` (Path): Path to ontology terms JSON file (default: `src/gnn/ontology/act_inf_ontology_terms.json`)
- `strict_validation` (bool): Require all terms to be in ontology (default: `False`)
- `recursive` (bool): Process directories recursively (default: `True`)

No other configuration keys are read; term suggestions come from Levenshtein-distance matching in `processor.py`.

---

## Dependencies

### Required Dependencies
- `json` - Ontology file loading
- `pathlib` - File operations

### Internal Dependencies
- `gnn.utils.pipeline_orchestration.pipeline_template` - Logging utilities
- `pipeline.config` - Configuration management
- `ontology.processor` - Core processing logic

---

## Usage Examples

### Basic Usage
```python
from gnn.ontology import process_ontology

success = process_ontology(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/10_ontology_output"),
    ontology_terms_file=Path("src/gnn/ontology/act_inf_ontology_terms.json"),
)
```

### Single-file processing
```python
from gnn.ontology import process_gnn_ontology

result = process_gnn_ontology("input/gnn_files/discrete/simple_mdp.md")
print(result["success"], result.get("validation_result", {}))
```

---

## Output Specification

### Output Products
- `ontology_results.json` - Aggregate summary across processed files (carries the `ontology_bridge` verification record)
- `*_ontology_report.json` - Per-model ontology validation reports (including `ontology_identifier_counts`)

### Output Directory Structure
```
output/10_ontology_output/
├── ontology_results.json
├── actinf_pomdp_agent_ontology_report.json
├── deep_planning_horizon_ontology_report.json
└── ... (one per processed model)
```

---

## Validation Rules

### Compliance Checks
1. All GNN variables map to ontology terms
2. Connections align with ontological relationships
3. POMDP components match Active Inference structure
4. Terminology is consistent with ontology

### Warning Conditions
- Unrecognized terms (potential new concepts)
- Ambiguous mappings (multiple possible interpretations)
- Missing required ontological components

---

## Performance Characteristics

### Expected Performance
Lightweight JSON-based validation; runtime is dominated by file I/O.

---

## Error Handling

### Graceful Degradation
- **Ontology File Missing**: Use default ontology, log warning
- **Invalid Ontology Format**: Parse what's possible, log error
- **Invalid GNN Model**: Skip model, log error, continue with others
- **Term Validation Failure**: Log warning, continue processing

### Error Categories
1. **File I/O Errors**: Cannot read ontology file (recovery: use default ontology)
2. **Validation Errors**: Invalid ontology structure (recovery: skip validation)
3. **Term Extraction Errors**: Cannot extract terms from model (recovery: skip model)
4. **Mapping Errors**: Cannot generate term mappings (recovery: partial mapping)

### Error Recovery
- **Default Ontology**: Use built-in ontology if file unavailable
- **Partial Validation**: Validate what's possible, report failures
- **Resource Cleanup**: Proper cleanup of ontology resources on errors

---

## Integration Points

### Pipeline Integration
- **Input**: Re-parses GNN `.md` files directly from the target directory (no Step 3 artifacts consumed)
- **Output**: Writes `10_ontology_output/ontology_results.json` and per-file ontology reports
- **Dependencies**: None on prior steps' outputs

### Module Dependencies
- **gnn/**: Reads raw GNN file content for term extraction
- **llm/**: Step 13 optionally injects `ontology_results.json` into its prompts
- **report/website**: Include `10_ontology_output/` in their uniform output-dir scans

### External Integration
- **Ontology File**: JSON-based ontology term definitions
- **Active Inference Standards**: Validates against Active Inference ontology

### Data Flow
```
input/gnn_files (re-parsed by 10_ontology.py)
  ↓
10_ontology.py (Ontology processing)
  ↓
  ├→ 13_llm.py (optional ontology injection)
  └→ output/10_ontology_output/ (Ontology results; scanned by Steps 20 and 23)
```

---

## Testing

### Test Files
- `tests/ontology/test_ontology_overall.py` — module-level behaviour
- `tests/ontology/test_ontology_annotations.py` — annotation parsing/validation
- `tests/ontology/test_ontology_public_api.py` — public export surface
- `tests/ontology/test_ontology_composability.py` — composability helpers, vocabulary dedup invariant, MCP real-vocabulary behaviour

### Test Coverage
Measure on demand:

```bash
uv run --extra dev python -m pytest tests/ontology/ \
    --cov=src/gnn/ontology --cov-report=term-missing
```

### Key Test Scenarios
1. Ontology term extraction
2. Ontology compliance validation
3. Semantic relationship mapping
4. Cross-reference validation

---

## MCP Integration

### Tools Registered
Registered by `src/gnn/ontology/mcp.py` `register_tools(mcp_instance)` (4 tools):
- `process_ontology` - Run ontology processing on a directory
- `validate_ontology_terms` - Validate a term or list of terms against the ontology
- `list_standard_ontology_terms` - List the standard ontology terms (derived from the real vocabulary via `load_defined_ontology_terms()`, so it cannot drift from what `validate_annotations` accepts)

### MCP File Location
- `src/gnn/ontology/mcp.py` - MCP tool registrations

---

## Troubleshooting

### Common Issues

#### Issue 1: Ontology validation fails
**Symptom**: Validation reports errors even for valid terms  
**Cause**: Ontology terms file missing or outdated  
**Solution**: 
- Check that `src/gnn/ontology/act_inf_ontology_terms.json` exists
- Verify ontology terms file format is valid JSON
- Update ontology terms file if needed
- Use `--verbose` flag for detailed validation messages

#### Issue 2: Terms not found in ontology
**Symptom**: Valid Active Inference terms reported as invalid  
**Cause**: Ontology terms file incomplete or term naming mismatch  
**Solution**:
- Check term spelling and case sensitivity
- Verify ontology terms file includes all required terms
- Use `--strict-validation=False` for lenient validation
- Review ontology terms file structure

#### Issue 3: Semantic mapping incomplete
**Symptom**: Mapping generation produces incomplete results  
**Cause**: Missing ontology relationships or incomplete GNN model  
**Solution**:
- Ensure GNN model has complete ontology annotations
- Verify ontology terms file includes relationship data
- Check that GNN processing (step 3) completed successfully

---

## Version History

### Current Version: [pyproject.toml](../../../pyproject.toml) (canonical)

**Features**:
- Ontology term extraction
- Ontology compliance validation
- Semantic relationship mapping
- Hierarchical concept analysis
- Cross-reference validation

**Known Issues**:
- None currently

### Roadmap
- **Next Version**: Enhanced semantic relationship detection
- **Future**: Automated ontology term suggestion

---

## References

### Related Documentation
- [Pipeline Overview](../../../README.md)
- [Architecture Guide](../../../ARCHITECTURE.md)
- [Active Inference Ontology](https://activeinference.org)
- [GNN Ontology Guide](../../../docs/gnn/advanced/gnn_ontology.md)

### External Resources
- [Active Inference Institute](https://activeinference.institute/)
- [Active Inference Ontology Documentation](../../../docs/gnn/advanced/ontology_system.md)

---

**Last Updated**: 2026-11-01
**Maintainer**: GNN Pipeline Team
**Status**: Production Ready
**Version**: [pyproject.toml](../../../pyproject.toml) (canonical)



---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API

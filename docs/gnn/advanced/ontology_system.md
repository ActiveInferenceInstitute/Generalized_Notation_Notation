# GNN Ontology System Documentation

**Version**: v3.2.0 Engine (Bundle v2.0.0)  
**Last Updated**: 2026-11-01  
**Status**: Maintained
**Scope**: Ontology annotations and processing. See [framework implementations](../implementations/README.md) for current backend coverage.

This document describes the ontology system used in conjunction with GNN files, focusing on the Active Inference Ontology.

For implementation details and usage, see:

- **[src/gnn/ontology/AGENTS.md](../../../src/gnn/ontology/AGENTS.md)**: Ontology processing module documentation
- **[src/gnn/AGENTS.md](../../../src/gnn/AGENTS.md)**: Complete pipeline module registry
- **[src/gnn/main.py](../../../src/gnn/main.py)**: Pipeline orchestrator (Step 10 handles ontology processing)

## 1. Ontology Terms Definition File

The core of the ontology system is a JSON file that defines recognized ontological terms.

- **Location:** `src/gnn/ontology/act_inf_ontology_terms.json`
- **Structure:** The file is a JSON object where each key is an ontological term (e.g., `HiddenState`, `TransitionMatrix`). The value for each term is another JSON object containing:
  - `description`: A human-readable explanation of the term.
  - `uri`: A **declared** value from the GNN vocabulary (`obo:ACTO_*` / `obo:TEMP_*`). These are declared metadata, not resolvable identifiers: no `ACTO` ontology is registered in the OBO Foundry, Bioregistry or OLS4, and `http://purl.obolibrary.org/obo/ACTO_000001` returns HTTP 404. Resolvable identifiers come from the pinned vocabulary bridge below.

**Example Entry from `act_inf_ontology_terms.json`:**

```json
{
    "HiddenState": {
        "description": "A state of the environment or agent that is not directly observable.",
        "uri": "obo:ACTO_000001" 
    },
    // ... other terms
}
```

### Resolvable identifiers: the pinned vocabulary bridge

Annotation identifiers come from the Active Inference Ontology's generated
vocabulary bridge, not from the declared `uri` values above:

- **Copy:** `src/gnn/ontology/active_inference_ontology_terms.json` — schema `active-inference-ontology/gnn-terms/v1`, release v5 (2023-05-25), 429 terms, 64 declared labels.
- **Pin:** `src/gnn/ontology/ontology-pin.json` — source repository, commit, release, path, the copy SHA-256, and the bridge's embedded source/export digests and counts.
- **Resolution:** `validate_annotations` matches each annotation value against the vocabulary file and then resolves that label through the bridge. Resolved labels (12 of the 64 today) report their stable `term-*` identifier; labels with no counterpart in release v5 (52 today) stay unresolved and are reported as `not in ontology release v5` instead of being guessed.
- **Freshness gate:** `uv run --frozen --no-sync python scripts/check_ontology_pin.py` re-verifies the copy against the pin and fails when they disagree.

## 2. Usage in GNN Files

Ontological terms are associated with variables defined in a GNN model using a dedicated section.

- **Section Header:** `## ActInfOntologyAnnotation`
- **Syntax:** Within this section, each line maps a variable name from the GNN model's `StateSpaceBlock` to an ontological term defined in `act_inf_ontology_terms.json`.
  - Format: `VariableName=OntologyTerm`
  - One mapping per line.

**Example `ActInfOntologyAnnotation` section in a GNN file:**

```markdown
## ActInfOntologyAnnotation
A=RecognitionMatrix
B=TransitionMatrix
D=Prior
s=HiddenState
o=Observation
```

In this example, `A`, `B`, `D`, `s`, and `o` are variables defined elsewhere in the GNN file (typically in the `StateSpaceBlock`). `RecognitionMatrix`, `TransitionMatrix`, etc., are terms expected to be present as keys in `act_inf_ontology_terms.json`.

## 3. Validation Process

The GNN processing pipeline includes Step 10 (`src/gnn/10_ontology.py`) that validates these annotations.

For detailed implementation, see **[src/gnn/ontology/AGENTS.md](../../../src/gnn/ontology/AGENTS.md)**.

- **Parsing:** The `ActInfOntologyAnnotation` section is parsed to extract all `VariableName=OntologyTerm` mappings.
- **Validation:** Each `OntologyTerm` used in the GNN file is checked for its existence as a key in the loaded `act_inf_ontology_terms.json` file.
  - If the term is found, the annotation is considered **valid**.
  - If the term is not found, the annotation is considered **invalid**.
- **Identifier resolution:** Every matched term is resolved through the pinned vocabulary bridge, which records `ontology_id` (the release-v5 `term-*` identifier, or `null`), `ontology_status` (`resolved`, `unresolved`, `not_declared` or `bridge_unavailable`), `ontology_term` and `ontology_reason` per term. Each run also carries an `ontology_bridge` verification record and an `ontology_identifier_counts` summary, so a run whose pinned copy no longer matches the pin is visible in its own artifacts.
- **Reporting:** A run writes `output/10_ontology_output/ontology_results.json` plus a `*_ontology_report.json` per processed model, summarizing the validation results, including:
  - A list of all processed GNN files.
  - For each file, the parsed annotations and a summary of valid/invalid terms.
  - Overall statistics on the number of annotations found, passed, and failed.

This validation helps ensure that GNN models use consistent and recognized terminology from the specified ontology, aiding in model comparison, interoperability, and semantic clarity.

**Run ontology validation:**

```bash
# Validate ontology annotations (Step 10)
uv run python src/gnn/10_ontology.py --target-dir input/gnn_files --verbose

# Or as part of the full pipeline
uv run python src/gnn/main.py --only-steps "3,10" --target-dir input/gnn_files
```

## 4. MCP Integration

The ontology processing logic, including loading term definitions, parsing GNN files, and validating annotations, is typically exposed via MCP tools defined in `src/gnn/ontology/mcp.py`. This allows other parts of the system or external tools to leverage the ontology functionalities.

Key functions involved (often wrapped as MCP tools):

- `load_defined_ontology_terms`: Loads the `.json` file.
- `parse_gnn_ontology_section`: Extracts annotations from GNN file content.
- `validate_annotations`: Performs the validation against defined terms.
- `generate_ontology_report_for_file`: Creates the per-file report section.

"""
MCP integration for the ontology module.

Exposes GNN ontology tools: term validation, ontology mapping,
annotation extraction, and ontology report generation through MCP.
"""

import logging
import re
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)

_HEADING_WHITESPACE = re.compile(r"\s+")

from gnn.utils.mcp.dispatch import run_pipeline_step_mcp

from . import load_defined_ontology_terms, process_ontology, validate_ontology_terms


def process_ontology_mcp(
    target_directory: str, output_directory: str, verbose: bool = False
) -> Dict[str, Any]:
    """
    Run ontology processing on a directory of GNN files.

    Maps GNN variables to Active Inference Ontology (ActInfO) terms,
    validates annotations, and produces an ontology mapping report.

    Args:
        target_directory: Directory containing GNN files
        output_directory: Directory to save ontology results
        verbose: Enable verbose logging

    Returns:
        Dictionary with success status and mapping summary.
    """
    return run_pipeline_step_mcp(
        process_ontology,
        wrapper_name="process_ontology_mcp",
        logger=logger,
        target_directory=target_directory,
        output_directory=output_directory,
        verbose=verbose,
        label="Ontology processing",
    )


def validate_ontology_terms_mcp(terms: Union[str, List[str]]) -> Dict[str, Any]:
    """
    Validate one or more ontology term names against the Active Inference Ontology.

    Args:
        terms: A single term string or list of term strings to validate

    Returns:
        Dictionary with validation results for each term.
    """
    try:
        if isinstance(terms, str):
            terms_list = [t.strip() for t in terms.split(",") if t.strip()]
        else:
            terms_list = list(terms)

        is_valid = validate_ontology_terms(terms_list)
        return {
            "success": True,
            "terms": terms_list,
            "is_valid": is_valid,
            "message": f"{len(terms_list)} term(s) {'all valid' if is_valid else 'contain invalid entries'}",
        }
    except Exception as e:
        logger.error(f"validate_ontology_terms_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def extract_ontology_annotations_mcp(gnn_content: str) -> Dict[str, Any]:
    """Extract ``ActInfOntologyAnnotation`` entries from GNN model content.

    Parses the ``## ActInfOntologyAnnotation`` section (the spaced
    ``## ActInf Ontology Annotation`` spelling is equivalent) and returns all
    variable-to-term mappings found, partitioned by whether the mapped
    term exists in the Active Inference ontology. Validation uses the real
    vocabulary (:func:`load_defined_ontology_terms`) matched
    case-insensitively, so the MCP wrapper stays in sync with
    ``validate_annotations`` instead of drifting against a hard-coded subset.

    Args:
        gnn_content: GNN model content as a string.

    Returns:
        ``{"success", "annotations", "validated_mappings", "unknown_terms",
        "total_annotations", "valid_count"}`` on success.
    """
    try:
        lines = gnn_content.splitlines()
        in_annotation = False
        annotations: Dict[str, str] = {}

        for line in lines:
            stripped = line.strip()
            # Headings compare with whitespace collapsed, so the spaced spelling
            # "## ActInf Ontology Annotation" opens the same block as the
            # canonical heading instead of yielding silently empty bindings.
            # Whitespace is the only accepted variance: case folding and the
            # "## Ontology" alias of parse_gnn_ontology_section stay deferred.
            if (
                stripped.startswith("## ")
                and _HEADING_WHITESPACE.sub("", stripped[3:])
                == "ActInfOntologyAnnotation"
            ):
                in_annotation = True
                continue
            if in_annotation:
                if stripped.startswith("##"):
                    break
                if "=" in stripped and not stripped.startswith("#"):
                    parts = stripped.split("=", 1)
                    var_name = parts[0].strip()
                    term = parts[1].strip()
                    if var_name:
                        annotations[var_name] = term

        known = {name.casefold(): name for name in load_defined_ontology_terms()}
        validated: Dict[str, str] = {
            k: v for k, v in annotations.items() if v.casefold() in known
        }
        unknown: Dict[str, str] = {
            k: v for k, v in annotations.items() if v.casefold() not in known
        }

        return {
            "success": True,
            "annotations": annotations,
            "validated_mappings": validated,
            "unknown_terms": unknown,
            "total_annotations": len(annotations),
            "valid_count": len(validated),
        }
    except Exception as e:
        logger.error("extract_ontology_annotations_mcp error: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}


def list_standard_ontology_terms_mcp() -> Dict[str, Any]:
    """Return the canonical Active Inference Ontology (ActInfO) term list.

    Derived from the real vocabulary (:func:`load_defined_ontology_terms`)
    rather than a hand-maintained subset, so this tool cannot drift from the
    terms ``validate_annotations`` and ``extract_ontology_annotations_mcp``
    actually accept. Each entry maps the canonical term name to its
    vocabulary description.

    Returns:
        ``{"success": True, "terms": {name: description}, "count": N}``.
    """
    vocabulary = load_defined_ontology_terms()
    terms: Dict[str, str] = {
        name: str(entry.get("description", ""))
        if isinstance(entry, dict)
        else str(entry)
        for name, entry in vocabulary.items()
    }
    return {"success": True, "terms": terms, "count": len(terms)}


# ── MCP Registration ────────────────────────────────────────────────────────


def register_tools(mcp_instance: Any) -> None:
    """Register ontology tools with the MCP server."""

    mcp_instance.register_tool(
        "process_ontology",
        process_ontology_mcp,
        {
            "type": "object",
            "properties": {
                "target_directory": {
                    "type": "string",
                    "description": "Directory with GNN files",
                },
                "output_directory": {
                    "type": "string",
                    "description": "Directory for ontology results",
                },
                "verbose": {"type": "boolean", "default": False},
            },
            "required": ["target_directory", "output_directory"],
        },
        "Map GNN variables to Active Inference Ontology terms and produce an ontology report.",
        module=__package__,
        category="ontology",
    )

    mcp_instance.register_tool(
        "validate_ontology_terms",
        validate_ontology_terms_mcp,
        {
            "type": "object",
            "properties": {
                "terms": {
                    "oneOf": [
                        {"type": "string", "description": "Comma-separated term names"},
                        {"type": "array", "items": {"type": "string"}},
                    ]
                },
            },
            "required": ["terms"],
        },
        "Validate ontology term names against the Active Inference Ontology.",
        module=__package__,
        category="ontology",
    )

    mcp_instance.register_tool(
        "extract_ontology_annotations",
        extract_ontology_annotations_mcp,
        {
            "type": "object",
            "properties": {
                "gnn_content": {
                    "type": "string",
                    "description": "GNN model content as a string",
                },
            },
            "required": ["gnn_content"],
        },
        "Extract ActInfOntologyAnnotation variable-to-term mappings from GNN model content.",
        module=__package__,
        category="ontology",
    )

    mcp_instance.register_tool(
        "list_standard_ontology_terms",
        list_standard_ontology_terms_mcp,
        {},
        "Return the canonical list of Active Inference Ontology (ActInfO) terms and descriptions.",
        module=__package__,
        category="ontology",
    )

    logger.info("ontology module MCP tools registered (4 tools).")

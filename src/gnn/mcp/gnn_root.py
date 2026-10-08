"""
MCP (Model Context Protocol) integration for GNN comprehensive validation and processing.

This module exposes GNN documentation files, schema definitions, validation capabilities,
round-trip testing, cross-format validation, and enhanced processing through the Model Context Protocol.

Enhanced Features:
- Complete format ecosystem support (23 formats)
- Multi-level validation (Basic, Standard, Strict, Research, Round-trip)
- Cross-format consistency validation
- Round-trip semantic preservation testing
- Enhanced processing with performance metrics
- Binary file validation for pickle/binary formats
"""

import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union, cast

import yaml

logger = logging.getLogger(__name__)

# Import enhanced GNN capabilities
try:
    from gnn.schema_validator import (
        validate_cross_format_consistency,
        validate_schema_consistency,
    )

    from ..schema_validator import (
        GNNParser,
        GNNValidator,
        ValidationLevel,
    )
    from .processors import (
        check_cross_format_consistency,
        process_gnn_folder,
        run_gnn_round_trip_tests,
    )

    ENHANCED_CAPABILITIES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Enhanced GNN capabilities not available: {e}")
    ENHANCED_CAPABILITIES_AVAILABLE = False


# MCP Tools for GNN Documentation and Schema Module


def get_gnn_documentation(
    doc_name: Literal[
        "file_structure", "punctuation", "schema_json", "schema_yaml", "grammar"
    ],
) -> Dict[str, Any]:
    """
    Retrieve content of a GNN documentation or schema file.

    Args:
        doc_name: Name of the GNN document to retrieve.
                  Allowed values: "file_structure", "punctuation", "schema_json", "schema_yaml", "grammar".

    Returns:
        Dictionary containing document content or an error.
    """

    base_path = Path(__file__).parent

    file_map: dict[str, Any] = {
        "file_structure": "documentation/file_structure.md",
        "punctuation": "documentation/punctuation.md",
        "schema_json": "schemas/json.json",
        "schema_yaml": "schemas/yaml.yaml",
        "grammar": "../grammars/ebnf.ebnf",
    }

    if doc_name not in file_map:
        error_msg = (
            f"Invalid document name: {doc_name}. Allowed: {', '.join(file_map.keys())}."
        )
        logger.error(f"get_gnn_documentation: {error_msg}")
        return {"success": False, "error": error_msg}

    file_to_read = base_path / file_map[doc_name]

    if not file_to_read.exists():
        error_msg = f"Documentation file not found: {file_to_read.name} (expected at {file_to_read.resolve()})"
        logger.error(f"get_gnn_documentation: {error_msg}")
        return {"success": False, "error": error_msg}

    try:
        content = file_to_read.read_text()
        return {"success": True, "doc_name": doc_name, "content": content}
    except Exception as e:
        error_msg = f"Error reading {file_to_read.name}: {str(e)}"
        logger.error(f"get_gnn_documentation: {error_msg}", exc_info=True)
        return {"success": False, "error": error_msg}


def validate_gnn_content(
    content: str,
    validation_level: Union[ValidationLevel, str] = ValidationLevel.STANDARD,
    enable_round_trip: bool = False,
    format_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Enhanced validation of GNN content with comprehensive testing capabilities.

    Args:
        content: GNN file content as string
        validation_level: Validation strictness level (ValidationLevel enum or
            equivalent string value, e.g. "basic", "standard", "strict")
        enable_round_trip: Whether to enable round-trip testing
        format_hint: Optional format hint for content detection

    Returns:
        Dictionary containing enhanced validation results
    """
    if not ENHANCED_CAPABILITIES_AVAILABLE:
        return {
            "success": False,
            "error": "Enhanced validation capabilities not available",
        }

    try:
        if isinstance(validation_level, ValidationLevel):
            val_level = validation_level
        else:
            try:
                val_level = ValidationLevel(validation_level)
            except ValueError:
                val_level = ValidationLevel.STANDARD

        # Create temporary file for validation
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            temp_file = Path(f.name)

        try:
            # Initialize enhanced validator
            validator = GNNValidator(
                validation_level=val_level, enable_round_trip_testing=enable_round_trip
            )

            # Perform validation
            result = validator.validate_file(temp_file, val_level)

            # Convert result to JSON-serializable format
            return {
                "success": True,
                "is_valid": result.is_valid,
                "validation_level": result.validation_level.value,
                "format_tested": result.format_tested,
                "errors": result.errors,
                "warnings": result.warnings,
                "suggestions": result.suggestions,
                "metadata": result.metadata,
                "semantic_checksum": result.semantic_checksum,
                "cross_format_consistent": result.cross_format_consistent,
                "round_trip_success_rate": result.get_round_trip_success_rate(),
                "performance_metrics": result.performance_metrics,
            }

        finally:
            # Clean up temporary file
            temp_file.unlink(missing_ok=True)

    except Exception as e:
        logger.error(f"validate_gnn_content: {str(e)}", exc_info=True)
        return {"success": False, "error": f"Validation error: {str(e)}"}


def parse_gnn_content(
    content: str, format_hint: str = "markdown", enhanced_validation: bool = True
) -> Dict[str, Any]:
    """
    Parse GNN content with enhanced multi-format support.

    Args:
        content: GNN file content as string
        format_hint: Format hint for parsing (markdown, json, xml, yaml, etc.)
        enhanced_validation: Whether to use enhanced parsing capabilities

    Returns:
        Dictionary containing parsed GNN model structure
    """
    if not ENHANCED_CAPABILITIES_AVAILABLE:
        return {
            "success": False,
            "error": "Enhanced parsing capabilities not available",
        }

    try:
        # Initialize enhanced parser
        parser = GNNParser(enhanced_validation=enhanced_validation)

        # Parse content
        parsed_gnn = parser.parse_content(content, format_hint=format_hint)

        # Convert to JSON-serializable format
        variables_dict: dict[Any, Any] = {}
        for var_name, var in parsed_gnn.variables.items():
            variables_dict[var_name] = {
                "name": var.name,
                "dimensions": var.dimensions,
                "data_type": var.data_type,
                "description": var.description,
                "ontology_mapping": var.ontology_mapping,
                "line_number": var.line_number,
            }

        connections_list: list[Any] = []
        for conn in parsed_gnn.connections:
            connections_list.append(
                {
                    "source": conn.source,
                    "target": conn.target,
                    "connection_type": conn.connection_type,
                    "symbol": conn.symbol,
                    "description": conn.description,
                    "line_number": conn.line_number,
                }
            )

        return {
            "success": True,
            "gnn_section": parsed_gnn.gnn_section,
            "version": parsed_gnn.version,
            "model_name": parsed_gnn.model_name,
            "model_annotation": parsed_gnn.model_annotation,
            "variables": variables_dict,
            "connections": connections_list,
            "parameters": parsed_gnn.parameters,
            "equations": parsed_gnn.equations,
            "time_config": parsed_gnn.time_config,
            "ontology_mappings": parsed_gnn.ontology_mappings,
            "model_parameters": parsed_gnn.model_parameters,
            "footer": parsed_gnn.footer,
            "signature": parsed_gnn.signature,
            "metadata": parsed_gnn.metadata,
            "source_format": parsed_gnn.source_format,
            "semantic_checksum": parsed_gnn.semantic_checksum,
        }

    except Exception as e:
        logger.error(f"parse_gnn_content: {str(e)}", exc_info=True)
        return {"success": False, "error": f"Parsing error: {str(e)}"}


def validate_cross_format_consistency_content(
    content: str, source_format: str = "markdown", enable_round_trip: bool = False
) -> Dict[str, Any]:
    """
    Validate cross-format consistency for GNN content.

    Args:
        content: GNN file content as string
        source_format: Source format of the content
        enable_round_trip: Whether to enable round-trip testing

    Returns:
        Dictionary containing cross-format validation results
    """
    if not ENHANCED_CAPABILITIES_AVAILABLE:
        return {
            "success": False,
            "error": "Cross-format validation capabilities not available",
        }

    try:
        # Perform cross-format validation
        result = validate_cross_format_consistency(content, enable_round_trip)

        # Convert result to JSON-serializable format
        format_results_dict: dict[Any, Any] = {}
        for format_name, validation_result in result.format_results.items():
            format_results_dict[format_name] = {
                "is_valid": validation_result.is_valid,
                "errors": validation_result.errors,
                "warnings": validation_result.warnings,
                "suggestions": validation_result.suggestions,
                "validation_level": validation_result.validation_level.value,
                "format_tested": validation_result.format_tested,
            }

        return {
            "success": True,
            "is_consistent": result.is_consistent,
            "consistency_rate": result.get_consistency_rate(),
            "schema_formats": result.schema_formats,
            "inconsistencies": result.inconsistencies,
            "warnings": result.warnings,
            "format_results": format_results_dict,
            "metadata": result.metadata,
            "semantic_checksums": result.semantic_checksums,
            "performance_metrics": result.performance_metrics,
            "round_trip_compatibility": result.round_trip_compatibility,
            "format_specific_issues": result.format_specific_issues,
        }

    except Exception as e:
        logger.error(
            f"validate_cross_format_consistency_content: {str(e)}", exc_info=True
        )
        return {"success": False, "error": f"Cross-format validation error: {str(e)}"}


def validate_schema_definitions_consistency() -> Dict[str, Any]:
    """
    Validate consistency between GNN schema definition files.

    Returns:
        Dictionary containing schema consistency validation results
    """
    if not ENHANCED_CAPABILITIES_AVAILABLE:
        return {
            "success": False,
            "error": "Schema validation capabilities not available",
        }

    try:
        # Perform schema consistency validation
        result = validate_schema_consistency()

        # Convert result to JSON-serializable format
        return {
            "success": True,
            "is_consistent": result.is_consistent,
            "consistency_rate": result.get_consistency_rate(),
            "schema_formats": result.schema_formats,
            "inconsistencies": result.inconsistencies,
            "warnings": result.warnings,
            "metadata": result.metadata,
        }

    except Exception as e:
        logger.error(
            f"validate_schema_definitions_consistency: {str(e)}", exc_info=True
        )
        return {
            "success": False,
            "error": f"Schema consistency validation error: {str(e)}",
        }


def process_gnn_directory(
    target_dir: str,
    output_dir: str,
    recursive: bool = False,
    verbose: bool = False,
    validation_level: str = "standard",
    enable_round_trip: bool = False,
) -> Dict[str, Any]:
    """
    Process a directory of GNN files with enhanced validation and testing.

    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Output directory for results
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        validation_level: Validation level (basic, standard, strict, research, round_trip)
        enable_round_trip: Whether to enable round-trip testing

    Returns:
        Dictionary containing processing results
    """
    if not ENHANCED_CAPABILITIES_AVAILABLE:
        return {
            "success": False,
            "error": "Enhanced processing capabilities not available",
        }

    try:
        import logging

        # Create a logger for this operation
        logger_instance = logging.getLogger(f"gnn_mcp_processing_{int(time.time())}")
        logger_instance.setLevel(logging.INFO)

        # Convert paths
        target_path = Path(target_dir)
        output_path = Path(output_dir)

        # Process the directory
        success = process_gnn_folder(
            target_dir=target_path,
            output_dir=output_path,
            logger=logger_instance,
            recursive=recursive,
            verbose=verbose,
            validation_level=validation_level,
            enable_round_trip=enable_round_trip,
        )

        return {
            "success": success,
            "target_directory": str(target_path),
            "output_directory": str(output_path),
            "recursive": recursive,
            "validation_level": validation_level,
            "round_trip_enabled": enable_round_trip,
            "message": "Processing completed successfully"
            if success
            else "Processing completed with issues",
        }

    except Exception as e:
        logger.error(f"process_gnn_directory: {str(e)}", exc_info=True)
        return {"success": False, "error": f"Directory processing error: {str(e)}"}


def run_round_trip_tests(
    target_dir: str,
    output_dir: str,
    reference_file: Optional[str] = None,
    test_subset: Optional[List[str]] = None,
    enable_parallel: bool = False,
) -> Dict[str, Any]:
    """
    Run comprehensive round-trip tests on GNN files.

    Args:
        target_dir: Directory containing GNN files
        output_dir: Output directory for test results
        reference_file: Optional specific reference file
        test_subset: Optional list of formats to test
        enable_parallel: Whether to enable parallel testing

    Returns:
        Dictionary containing round-trip test results
    """
    if not ENHANCED_CAPABILITIES_AVAILABLE:
        return {
            "success": False,
            "error": "Round-trip testing capabilities not available",
        }

    try:
        import logging

        # Create a logger for this operation
        logger_instance = logging.getLogger(f"gnn_mcp_roundtrip_{int(time.time())}")
        logger_instance.setLevel(logging.INFO)

        # Convert paths
        target_path = Path(target_dir)
        output_path = Path(output_dir)

        # Run round-trip tests
        success = run_gnn_round_trip_tests(
            target_dir=target_path,
            output_dir=output_path,
            logger=logger_instance,
            reference_file=reference_file,
            test_subset=test_subset,
            enable_parallel=enable_parallel,
        )

        return {
            "success": success,
            "target_directory": str(target_path),
            "output_directory": str(output_path),
            "reference_file": reference_file,
            "test_subset": test_subset,
            "parallel_enabled": enable_parallel,
            "message": "Round-trip tests completed successfully"
            if success
            else "Round-trip tests completed with issues",
        }

    except Exception as e:
        logger.error(f"run_round_trip_tests: {str(e)}", exc_info=True)
        return {"success": False, "error": f"Round-trip testing error: {str(e)}"}


def validate_directory_cross_format_consistency(
    target_dir: str,
    output_dir: str,
    files_to_test: Optional[List[str]] = None,
    include_binary: bool = False,
) -> Dict[str, Any]:
    """
    Validate cross-format consistency for files in a directory.

    Args:
        target_dir: Directory containing GNN files to test
        output_dir: Output directory for validation results
        files_to_test: Optional list of specific files to test
        include_binary: Whether to include binary formats in validation

    Returns:
        Dictionary containing cross-format validation results
    """
    if not ENHANCED_CAPABILITIES_AVAILABLE:
        return {
            "success": False,
            "error": "Cross-format validation capabilities not available",
        }

    try:
        import logging

        # Create a logger for this operation
        logger_instance = logging.getLogger(f"gnn_mcp_crossformat_{int(time.time())}")
        logger_instance.setLevel(logging.INFO)

        # Convert paths
        target_path = Path(target_dir)
        output_path = Path(output_dir)

        # Validate cross-format consistency
        success = check_cross_format_consistency(
            target_dir=target_path,
            output_dir=output_path,
            logger=logger_instance,
            files_to_test=files_to_test,
            include_binary=include_binary,
        )

        return {
            "success": success,
            "target_directory": str(target_path),
            "output_directory": str(output_path),
            "files_tested": files_to_test,
            "binary_included": include_binary,
            "message": "Cross-format validation completed successfully"
            if success
            else "Cross-format validation completed with issues",
        }

    except Exception as e:
        logger.error(
            f"validate_directory_cross_format_consistency: {str(e)}", exc_info=True
        )
        return {
            "success": False,
            "error": f"Directory cross-format validation error: {str(e)}",
        }


def get_gnn_schema_info() -> Dict[str, Any]:
    """
    Get comprehensive information about the GNN schema.

    Returns:
        Dictionary containing schema metadata and structure information
    """
    try:
        base_path = Path(__file__).parent

        # Load schema info from YAML if available
        yaml_path = base_path / "schemas/yaml.yaml"
        if yaml_path.exists():
            with open(yaml_path, "r") as f:
                schema_data = yaml.safe_load(f)

            return {
                "success": True,
                "schema_info": schema_data.get("schema_info", {}),
                "required_sections": schema_data.get("required_sections", []),
                "optional_sections": schema_data.get("optional_sections", []),
                "syntax_rules": schema_data.get("syntax_rules", {}),
                "active_inference_patterns": schema_data.get("active_inference", {}),
                "validation_levels": schema_data.get("validation_levels", {}),
                "enhanced_capabilities_available": ENHANCED_CAPABILITIES_AVAILABLE,
            }
        else:
            return {
                "success": False,
                "error": "Schema YAML file not found",
                "enhanced_capabilities_available": ENHANCED_CAPABILITIES_AVAILABLE,
            }

    except Exception as e:
        logger.error(f"get_gnn_schema_info: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"Error loading schema info: {str(e)}",
            "enhanced_capabilities_available": ENHANCED_CAPABILITIES_AVAILABLE,
        }


def get_gnn_module_info() -> Dict[str, Any]:
    """
    Get comprehensive information about the GNN module capabilities.

    Returns:
        Dictionary containing module metadata and feature information
    """
    try:
        from gnn import get_module_info

        module_info = get_module_info()
        module_info.update(
            {
                "success": True,
                "enhanced_capabilities_available": ENHANCED_CAPABILITIES_AVAILABLE,
                "mcp_integration_version": "2.0.0",
                "supported_operations": [
                    "validate_gnn_content",
                    "parse_gnn_content",
                    "validate_cross_format_consistency_content",
                    "validate_schema_definitions_consistency",
                    "process_gnn_directory",
                    "run_round_trip_tests",
                    "validate_directory_cross_format_consistency",
                    "get_gnn_documentation",
                    "get_gnn_schema_info",
                ],
            }
        )

        return module_info

    except Exception as e:
        logger.error(f"get_gnn_module_info: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"Error loading module info: {str(e)}",
            "enhanced_capabilities_available": ENHANCED_CAPABILITIES_AVAILABLE,
        }


# Resource retrievers
def _retrieve_gnn_doc_resource(uri: str) -> Dict[str, Any]:
    """
    Retrieve GNN documentation resource by URI.
    Example URI: gnn://documentation/file_structure
    """
    if not uri.startswith("gnn://documentation/"):
        error_msg = f"Invalid URI format for GNN documentation: {uri}"
        logger.error(f"_retrieve_gnn_doc_resource: {error_msg}")
        raise ValueError(error_msg)

    doc_name_part = uri.replace("gnn://documentation/", "")

    allowed_docs: list[Any] = [
        "file_structure",
        "punctuation",
        "schema_json",
        "schema_yaml",
        "grammar",
    ]
    if doc_name_part not in allowed_docs:
        error_msg = f"Invalid document name in URI: {doc_name_part}"
        logger.error(f"_retrieve_gnn_doc_resource: {error_msg}")
        raise ValueError(error_msg)

    doc_name_literal = cast(
        Literal[
            "file_structure", "punctuation", "schema_json", "schema_yaml", "grammar"
        ],
        doc_name_part,
    )

    result = get_gnn_documentation(doc_name=doc_name_literal)
    if not result["success"]:
        error_msg = f"Failed to retrieve document {doc_name_part}: {result.get('error', 'Unknown error')}"
        logger.error(f"_retrieve_gnn_doc_resource: {error_msg}")
        raise ValueError(error_msg)
    return result


# MCP Registration Function
def register_tools(mcp_instance: Any) -> Any:
    """Register comprehensive GNN tools and resources with the MCP."""

    # Documentation tools
    mcp_instance.register_tool(
        "get_gnn_documentation",
        get_gnn_documentation,
        {
            "doc_name": {
                "type": "string",
                "description": "Name of the GNN document (e.g., 'file_structure', 'punctuation', 'schema_json', 'schema_yaml', 'grammar')",
                "enum": [
                    "file_structure",
                    "punctuation",
                    "schema_json",
                    "schema_yaml",
                    "grammar",
                ],
            }
        },
        "Retrieve the content of a GNN documentation file, schema definition, or grammar specification.",
    )

    # Enhanced validation tools
    mcp_instance.register_tool(
        "validate_gnn_content",
        validate_gnn_content,
        {
            "content": {
                "type": "string",
                "description": "GNN file content to validate",
            },
            "validation_level": {
                "type": "string",
                "description": "Validation strictness level",
                "enum": ["basic", "standard", "strict", "research", "round_trip"],
                "default": "standard",
            },
            "enable_round_trip": {
                "type": "boolean",
                "description": "Whether to enable round-trip testing",
                "default": False,
            },
            "format_hint": {
                "type": "string",
                "description": "Optional format hint for content detection",
                "enum": ["markdown", "json", "xml", "yaml", "binary"],
            },
        },
        "Enhanced validation of GNN file content with comprehensive testing capabilities.",
    )

    # Parsing tools
    mcp_instance.register_tool(
        "parse_gnn_content",
        parse_gnn_content,
        {
            "content": {"type": "string", "description": "GNN file content to parse"},
            "format_hint": {
                "type": "string",
                "description": "Format hint for parsing",
                "enum": ["markdown", "json", "xml", "yaml", "binary"],
                "default": "markdown",
            },
            "enhanced_validation": {
                "type": "boolean",
                "description": "Whether to use enhanced parsing capabilities",
                "default": True,
            },
        },
        "Parse GNN content with enhanced multi-format support and return structured model representation.",
    )

    # Cross-format validation tools
    mcp_instance.register_tool(
        "validate_cross_format_consistency_content",
        validate_cross_format_consistency_content,
        {
            "content": {
                "type": "string",
                "description": "GNN file content to validate for cross-format consistency",
            },
            "source_format": {
                "type": "string",
                "description": "Source format of the content",
                "enum": ["markdown", "json", "xml", "yaml", "binary"],
                "default": "markdown",
            },
            "enable_round_trip": {
                "type": "boolean",
                "description": "Whether to enable round-trip testing",
                "default": False,
            },
        },
        "Validate cross-format consistency for GNN content across all supported formats.",
    )

    # Schema validation tools
    mcp_instance.register_tool(
        "validate_schema_definitions_consistency",
        validate_schema_definitions_consistency,
        {},
        "Validate consistency between GNN schema definition files across different formats.",
    )

    # Directory processing tools
    mcp_instance.register_tool(
        "process_gnn_directory",
        process_gnn_directory,
        {
            "target_dir": {
                "type": "string",
                "description": "Directory containing GNN files to process",
            },
            "output_dir": {
                "type": "string",
                "description": "Output directory for results",
            },
            "recursive": {
                "type": "boolean",
                "description": "Whether to process files recursively",
                "default": False,
            },
            "verbose": {
                "type": "boolean",
                "description": "Whether to enable verbose logging",
                "default": False,
            },
            "validation_level": {
                "type": "string",
                "description": "Validation level",
                "enum": ["basic", "standard", "strict", "research", "round_trip"],
                "default": "standard",
            },
            "enable_round_trip": {
                "type": "boolean",
                "description": "Whether to enable round-trip testing",
                "default": False,
            },
        },
        "Process a directory of GNN files with enhanced validation and testing capabilities.",
    )

    # Round-trip testing tools
    mcp_instance.register_tool(
        "run_round_trip_tests",
        run_round_trip_tests,
        {
            "target_dir": {
                "type": "string",
                "description": "Directory containing GNN files",
            },
            "output_dir": {
                "type": "string",
                "description": "Output directory for test results",
            },
            "reference_file": {
                "type": "string",
                "description": "Optional specific reference file to use for testing",
            },
            "test_subset": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of formats to test (e.g., ['json', 'xml', 'yaml'])",
            },
            "enable_parallel": {
                "type": "boolean",
                "description": "Whether to enable parallel testing",
                "default": False,
            },
        },
        "Run comprehensive round-trip tests on GNN files across all supported formats.",
    )

    # Directory cross-format validation tools
    mcp_instance.register_tool(
        "validate_directory_cross_format_consistency",
        validate_directory_cross_format_consistency,
        {
            "target_dir": {
                "type": "string",
                "description": "Directory containing GNN files to test",
            },
            "output_dir": {
                "type": "string",
                "description": "Output directory for validation results",
            },
            "files_to_test": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of specific files to test",
            },
            "include_binary": {
                "type": "boolean",
                "description": "Whether to include binary formats in validation",
                "default": False,
            },
        },
        "Validate cross-format consistency for files in a directory with comprehensive analysis.",
    )

    # Information tools
    mcp_instance.register_tool(
        "get_gnn_schema_info",
        get_gnn_schema_info,
        {},
        "Get comprehensive information about the GNN schema structure and validation rules.",
    )

    mcp_instance.register_tool(
        "get_gnn_module_info",
        get_gnn_module_info,
        {},
        "Get comprehensive information about the GNN module capabilities and features.",
    )

    # Resources
    mcp_instance.register_resource(
        "gnn://documentation/{doc_name}",
        _retrieve_gnn_doc_resource,
        "Access GNN core documentation files like syntax and file structure definitions.",
    )

    logger.info("Enhanced GNN MCP tools and resources registered successfully.")
    logger.info(f"Enhanced capabilities available: {ENHANCED_CAPABILITIES_AVAILABLE}")

#!/usr/bin/env python3
"""
Model comparison mixin for the GNN round-trip test suite.

Extracted from the former ``testing.test_round_trip`` monolith.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any, List, Optional

from gnn.parsers.common import GNNFormat, GNNInternalRepresentation

from .round_trip_availability import CROSS_FORMAT_AVAILABLE
from .round_trip_results import ComprehensiveTestReport, RoundTripResult


class RoundTripComparisonMixin:
    """Verbatim methods moved from ``GNNRoundTripTester``."""

    if TYPE_CHECKING:
        from pathlib import Path

        from gnn.parsers import GNNParsingSystem
        from gnn.schema_validator import CrossFormatValidator

        supported_formats: Any
        reference_file: Any
        parsing_system: Optional[GNNParsingSystem]
        cross_validator: Optional[CrossFormatValidator]
        temp_dir: Path
        logger: logging.Logger

    def _compare_models(
        self,
        original: GNNInternalRepresentation,
        converted: GNNInternalRepresentation,
        result: RoundTripResult,
    ) -> Any:
        """Compare two models for semantic equivalence."""

        # Compare basic metadata
        if original.model_name != converted.model_name:
            result.add_difference(
                f"Model name mismatch: '{original.model_name}' vs '{converted.model_name}'"
            )

        if original.annotation != converted.annotation:
            result.add_difference("Annotation mismatch")

        # Compare variables
        self._compare_variables(original.variables, converted.variables, result)

        # Compare connections
        self._compare_connections(original.connections, converted.connections, result)

        # Compare parameters
        self._compare_parameters(original.parameters, converted.parameters, result)

        # Compare equations
        self._compare_equations(original.equations, converted.equations, result)

        # Compare time specification
        self._compare_time_specification(
            original.time_specification, converted.time_specification, result
        )

        # Compare ontology mappings
        self._compare_ontology_mappings(
            original.ontology_mappings, converted.ontology_mappings, result
        )

    def _compare_variables(
        self, orig_vars: List, conv_vars: List, result: RoundTripResult
    ) -> Any:
        """Compare variable lists."""
        orig_dict = {var.name: var for var in orig_vars}
        conv_dict = {var.name: var for var in conv_vars}

        # Check for missing variables
        missing_in_converted = set(orig_dict) - set(conv_dict)
        extra_in_converted = set(conv_dict) - set(orig_dict)

        for var_name in missing_in_converted:
            result.add_difference(f"Variable missing in converted: {var_name}")

        for var_name in extra_in_converted:
            result.add_difference(f"Extra variable in converted: {var_name}")

        # Compare common variables
        for var_name in set(orig_dict) & set(conv_dict):
            orig_var = orig_dict[var_name]
            conv_var = conv_dict[var_name]

            if hasattr(orig_var, "var_type") and hasattr(conv_var, "var_type"):
                # Compare using .value attribute to handle different object types
                orig_type = (
                    orig_var.var_type.value
                    if hasattr(orig_var.var_type, "value")
                    else str(orig_var.var_type)
                )
                conv_type = (
                    conv_var.var_type.value
                    if hasattr(conv_var.var_type, "value")
                    else str(conv_var.var_type)
                )
                if orig_type != conv_type:
                    result.add_difference(
                        f"Variable {var_name} type mismatch: {orig_type} vs {conv_type}"
                    )

            if hasattr(orig_var, "data_type") and hasattr(conv_var, "data_type"):
                # Compare using .value attribute to handle different object types
                orig_dtype = (
                    orig_var.data_type.value
                    if hasattr(orig_var.data_type, "value")
                    else str(orig_var.data_type)
                )
                conv_dtype = (
                    conv_var.data_type.value
                    if hasattr(conv_var.data_type, "value")
                    else str(conv_var.data_type)
                )
                if orig_dtype != conv_dtype:
                    result.add_difference(
                        f"Variable {var_name} data type mismatch: {orig_dtype} vs {conv_dtype}"
                    )

            if hasattr(orig_var, "dimensions") and hasattr(conv_var, "dimensions"):
                if orig_var.dimensions != conv_var.dimensions:
                    result.add_difference(
                        f"Variable {var_name} dimensions mismatch: {orig_var.dimensions} vs {conv_var.dimensions}"
                    )

    def _compare_connections(
        self, orig_conns: List, conv_conns: List, result: RoundTripResult
    ) -> Any:
        """Compare connection lists."""
        if len(orig_conns) != len(conv_conns):
            result.add_difference(
                f"Connection count mismatch: {len(orig_conns)} vs {len(conv_conns)}"
            )

        # Compare connections by content (simplified)
        orig_conn_strs: set[Any] = set()
        conv_conn_strs: set[Any] = set()

        for conn in orig_conns:
            if (
                hasattr(conn, "source_variables")
                and hasattr(conn, "target_variables")
                and hasattr(conn, "connection_type")
            ):
                # Handle different object types for connection_type
                conn_type = (
                    conn.connection_type.value
                    if hasattr(conn.connection_type, "value")
                    else str(conn.connection_type)
                )
                conn_str = f"{','.join(conn.source_variables)}--{conn_type}-->{','.join(conn.target_variables)}"
                orig_conn_strs.add(conn_str)

        for conn in conv_conns:
            if (
                hasattr(conn, "source_variables")
                and hasattr(conn, "target_variables")
                and hasattr(conn, "connection_type")
            ):
                # Handle different object types for connection_type
                conn_type = (
                    conn.connection_type.value
                    if hasattr(conn.connection_type, "value")
                    else str(conn.connection_type)
                )
                conn_str = f"{','.join(conn.source_variables)}--{conn_type}-->{','.join(conn.target_variables)}"
                conv_conn_strs.add(conn_str)

        missing_conns = orig_conn_strs - conv_conn_strs
        extra_conns = conv_conn_strs - orig_conn_strs

        for conn in missing_conns:
            result.add_difference(f"Missing connection: {conn}")

        for conn in extra_conns:
            result.add_difference(f"Extra connection: {conn}")

    def _compare_parameters(
        self, orig_params: List, conv_params: List, result: RoundTripResult
    ) -> Any:
        """Compare parameter lists."""
        orig_dict = {param.name: param for param in orig_params}
        conv_dict = {param.name: param for param in conv_params}

        missing_params = set(orig_dict) - set(conv_dict)
        extra_params = set(conv_dict) - set(orig_dict)

        for param_name in missing_params:
            result.add_difference(f"Missing parameter: {param_name}")

        for param_name in extra_params:
            result.add_difference(f"Extra parameter: {param_name}")

        # Compare parameter values (simplified - could be more sophisticated)
        for param_name in set(orig_dict) & set(conv_dict):
            orig_val = orig_dict[param_name].value
            conv_val = conv_dict[param_name].value

            if str(orig_val) != str(conv_val):  # Simple string comparison
                result.add_difference(
                    f"Parameter {param_name} value mismatch: {orig_val} vs {conv_val}"
                )

    def _compare_equations(
        self, orig_eqs: List, conv_eqs: List, result: RoundTripResult
    ) -> Any:
        """Compare equation lists."""
        if len(orig_eqs) != len(conv_eqs):
            result.add_difference(
                f"Equation count mismatch: {len(orig_eqs)} vs {len(conv_eqs)}"
            )

    def _compare_time_specification(
        self, orig_time: Any, conv_time: Any, result: RoundTripResult
    ) -> Any:
        """Compare time specifications."""
        if (orig_time is None) != (conv_time is None):
            result.add_difference("Time specification presence mismatch")
        elif orig_time and conv_time:
            if hasattr(orig_time, "time_type") and hasattr(conv_time, "time_type"):
                if orig_time.time_type != conv_time.time_type:
                    result.add_difference(
                        f"Time type mismatch: {orig_time.time_type} vs {conv_time.time_type}"
                    )

    def _compare_ontology_mappings(
        self, orig_mappings: List, conv_mappings: List, result: RoundTripResult
    ) -> Any:
        """Compare ontology mappings."""
        orig_dict = {
            mapping.variable_name: mapping.ontology_term for mapping in orig_mappings
        }
        conv_dict = {
            mapping.variable_name: mapping.ontology_term for mapping in conv_mappings
        }

        if orig_dict != conv_dict:
            result.add_difference("Ontology mappings mismatch")

    def _test_cross_format_consistency(
        self,
        reference_model: GNNInternalRepresentation,
        report: ComprehensiveTestReport,
    ) -> Any:
        """Test cross-format consistency validation."""
        try:
            # Convert to multiple formats and test consistency
            format_contents: dict[Any, Any] = {}

            print("   ➤ Generating content for all formats...")
            for fmt in self.supported_formats:
                if fmt == GNNFormat.MARKDOWN:
                    # Read original content
                    format_contents[fmt] = self.reference_file.read_text()
                    print(
                        f"      ✓ {fmt.value}: read original ({len(format_contents[fmt])} chars)"
                    )
                else:
                    try:
                        if self.parsing_system:
                            format_contents[fmt] = self.parsing_system.serialize(
                                reference_model, fmt
                            )
                        else:
                            format_contents[fmt] = None

                        if format_contents[fmt]:
                            print(
                                f"      ✓ {fmt.value}: serialized ({len(format_contents[fmt])} chars)"
                            )
                        else:
                            print(f"      ❌ {fmt.value}: empty content")
                    except Exception as e:
                        print(f"      ❌ {fmt.value}: serialization failed - {e}")
                        report.critical_errors.append(
                            f"Failed to serialize to {fmt.value}: {e}"
                        )

            # Test cross-format validation if available
            if CROSS_FORMAT_AVAILABLE and self.cross_validator:
                print("   ➤ Validating cross-format consistency...")
                consistent_formats = 0
                total_formats = 0

                for fmt, content in format_contents.items():
                    if content:
                        total_formats += 1
                        try:
                            cross_result = (
                                self.cross_validator.validate_cross_format_consistency(
                                    content
                                )
                            )
                            if cross_result.is_consistent:
                                print(f"      ✓ {fmt.value}: consistent")
                                consistent_formats += 1
                            else:
                                print(f"      ❌ {fmt.value}: inconsistent")
                                for inconsistency in cross_result.inconsistencies:
                                    print(f"         • {inconsistency}")
                                report.critical_errors.extend(
                                    cross_result.inconsistencies
                                )
                        except Exception as e:
                            print(f"      ❌ {fmt.value}: validation error - {e}")
                            report.critical_errors.append(
                                f"Cross-format validation failed for {fmt.value}: {e}"
                            )

                if total_formats > 0:
                    consistency_rate = (consistent_formats / total_formats) * 100
                    print(
                        f"      📊 Consistency rate: {consistent_formats}/{total_formats} ({consistency_rate:.1f}%)"
                    )
            else:
                print("   ➤ Cross-format validation skipped (module not available)")

        except Exception as e:
            print(f"   ❌ Cross-format consistency test failed: {e}")
            report.critical_errors.append(f"Cross-format consistency test failed: {e}")

    def _compute_model_checksum(self, model: GNNInternalRepresentation) -> str:
        """Compute a semantic checksum for a model."""
        # Create a normalized representation for checksumming
        checksum_data: dict[str, Any] = {
            "model_name": model.model_name,
            "variables": sorted(
                [
                    {
                        "name": var.name,
                        "type": var.var_type.value
                        if hasattr(var, "var_type")
                        else "unknown",
                        "dimensions": var.dimensions
                        if hasattr(var, "dimensions")
                        else [],
                        "data_type": var.data_type.value
                        if hasattr(var, "data_type")
                        else "unknown",
                    }
                    for var in model.variables
                ],
                key=lambda x: x["name"],
            ),
            "connections": sorted(
                [
                    {
                        "sources": sorted(conn.source_variables)
                        if hasattr(conn, "source_variables")
                        else [],
                        "targets": sorted(conn.target_variables)
                        if hasattr(conn, "target_variables")
                        else [],
                        "type": conn.connection_type.value
                        if hasattr(conn, "connection_type")
                        else "unknown",
                    }
                    for conn in model.connections
                ],
                key=lambda x: str(x),
            ),
            "parameters": sorted(
                [
                    {"name": param.name, "value": str(param.value)}
                    for param in model.parameters
                ],
                key=lambda x: x["name"],
            ),
        }

        checksum_str = json.dumps(checksum_data, sort_keys=True)
        return hashlib.md5(checksum_str.encode(), usedforsecurity=False).hexdigest()

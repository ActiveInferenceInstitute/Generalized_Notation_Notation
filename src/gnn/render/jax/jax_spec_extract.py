#!/usr/bin/env python3
"""
Spec extraction and matrix repair helpers for GNN Step 11 JAX rendering.

Extracted from ``render.jax.jax_renderer``.
"""

import logging
import re
from copy import deepcopy
from typing import (
    Any,
    Dict,
    cast,
)

import numpy as np

from gnn.render.pomdp_contract import (
    build_canonical_pomdp_spec,
    normalise_matrix_columns,
    normalise_vector,
)

logger = logging.getLogger(__name__)


# --- Internal code generation helpers ---


def _require_braced_assignments(text: str) -> str:
    """Do not treat a present but malformed table as an omitted parameter."""
    uncommented = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    for name in ("A", "B", "C", "D"):
        if re.search(rf"\b{name}\s*=", uncommented) and not re.search(
            rf"\b{name}\s*=\s*\{{[^}}]+\}}", uncommented, re.DOTALL
        ):
            raise ValueError(
                f"Invalid authored {name} parameter: expected braced literal"
            )
    return uncommented


def _parse_gnn_matrix_string(matrix_str: str, *, strict: bool = False) -> np.ndarray:
    """Parse GNN matrix string format to numpy array."""
    try:
        # Remove comments and clean up
        lines = matrix_str.split("\n")
        cleaned_lines: list[Any] = []
        for line in lines:
            if "#" in line:
                line = line.split("#")[0]
            line = line.strip()
            if line:
                cleaned_lines.append(line)

        # Reconstruct the matrix string
        matrix_str = " ".join(cleaned_lines)

        def numeric_fields(text: str) -> list[float]:
            fields = text.split(",")
            if strict:
                # A single terminal comma is valid tuple/set syntax. An
                # interior or leading empty field denotes a missing value.
                if len(fields) > 1 and not fields[-1].strip():
                    fields.pop()
                if any(not field.strip() for field in fields):
                    raise ValueError("Authored matrix contains an empty numeric field")
            return [float(field.strip()) for field in fields if field.strip()]

        # Handle A matrix format: { (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0) }
        if matrix_str.startswith("{") and matrix_str.endswith("}"):
            inner = matrix_str[1:-1].strip()

            # Split by commas, but be careful with nested tuples
            rows: list[Any] = []
            current_row = ""
            paren_count = 0

            for char in inner:
                if char == "(":
                    paren_count += 1
                elif char == ")":
                    paren_count -= 1

                if char == "," and paren_count == 0:
                    # End of a row
                    if strict and not current_row.strip():
                        raise ValueError(
                            "Authored matrix contains an empty row or value"
                        )
                    if current_row.strip():
                        rows.append(current_row.strip())
                    current_row = ""
                else:
                    current_row += char

            # Add the last row
            if current_row.strip():
                rows.append(current_row.strip())

            # Parse each row
            matrix: list[Any] = []
            for row in rows:
                row = row.strip()
                try:
                    if row.startswith("((") and row.endswith("))"):
                        # A tensor row contains tuples, one per previous state.
                        inner_row = row[2:-2]
                        nested_tuples = re.split(r"\)\s*,\s*\(", inner_row)
                        nested_row_values: list[list[float]] = []
                        for nested_tuple in nested_tuples:
                            nested_tuple = nested_tuple.strip("()")
                            tuple_values = numeric_fields(nested_tuple)
                            nested_row_values.append(tuple_values)
                        matrix.append(nested_row_values)
                    elif row.startswith("(") and row.endswith(")"):
                        # Parse tuple row
                        inner_row = row[1:-1]
                        row_values = numeric_fields(inner_row)
                        matrix.append(row_values)
                    else:
                        # Try to parse as simple values
                        row_values = numeric_fields(row)
                        if row_values:
                            matrix.append(row_values)
                except Exception as e:
                    if strict:
                        raise ValueError(
                            f"Invalid authored matrix row {row!r}: {e}"
                        ) from e
                    logger.warning(f"Failed to parse row '{row}': {e}")
                    # Add a default row to maintain matrix structure
                    if matrix:
                        # Use the same length as previous rows
                        matrix.append([0.0] * len(matrix[0]))
                    else:
                        matrix.append([1.0])

            if not matrix:
                if strict:
                    raise ValueError("Authored matrix literal must not be empty")
                return np.array([[1.0]])

            # Ensure all rows have the same length
            max_len = max(len(row) for row in matrix)
            if strict and any(len(row) != max_len for row in matrix):
                raise ValueError("Authored matrix literal must not contain ragged rows")
            for _, row in enumerate(matrix):
                while len(row) < max_len:
                    row.append(0.0)

            return np.array(matrix)

        if strict:
            raise ValueError("Authored matrix literal requires enclosing braces")
        return np.array([[1.0]])  # Default recovery

    except Exception as e:
        if strict:
            raise ValueError(f"Invalid authored matrix literal: {e}") from e
        logger.warning(f"Failed to parse matrix string: {e}")
        return np.array([[1.0]])  # Default recovery


def _extract_gnn_matrices(gnn_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Extract A, B, C, D matrices from GNN specification."""
    matrices: dict[Any, Any] = {}

    # --- Primary: Handle POMDP processor format ---
    if "model_parameters" in gnn_spec:
        logger.info("Extracting matrices from POMDP processor format")
        init_candidate = gnn_spec.get("initialparameterization", {})
        if isinstance(init_candidate, dict) and all(
            key in init_candidate for key in ("A", "B", "C", "D")
        ):
            # The shared contract consumes nested numeric sequences. Adapt
            # ndarray containers locally, without changing caller arrays or
            # weakening shared backend admission. Explicit validation errors
            # must propagate; they cannot authorize replacement defaults.
            candidate = deepcopy(gnn_spec)
            for name in ("A", "B", "C", "D", "E"):
                value = candidate["initialparameterization"].get(name)
                if np.iscomplexobj(value):
                    raise ValueError(f"{name} must contain real values")
                if isinstance(value, np.ndarray):
                    candidate["initialparameterization"][name] = value.tolist()
            canonical = build_canonical_pomdp_spec(candidate)
            logger.info("Canonicalized POMDP matrices for JAX renderer")
            return {
                name: np.asarray(
                    canonical["initialparameterization"][name], dtype=float
                )
                for name in ("A", "B", "C", "D")
            }

        model_params = gnn_spec["model_parameters"]
        n_states = model_params.get("num_hidden_states", 3)
        n_obs = model_params.get("num_obs", 3)
        n_actions = model_params.get("num_actions", 3)

        logger.info(
            f"Extracted variable dimensions: {{n_states: {n_states}, n_obs: {n_obs}, n_actions: {n_actions}}}"
        )

        # Create default matrices based on dimensions
        default_matrices: dict[str, Any] = {
            "A": np.eye(n_obs, n_states),  # Identity-like likelihood matrix
            "B": np.stack(
                [np.eye(n_states) for _ in range(n_actions)], axis=2
            ),  # Identity transitions for each action
            "C": np.zeros(n_obs),  # Zero preferences
            "D": np.ones(n_states) / n_states,  # Uniform prior
        }
        logger.info(
            f"Created default matrices: A={default_matrices['A'].shape}, B={default_matrices['B'].shape}, C={default_matrices['C'].shape}, D={default_matrices['D'].shape}"
        )

        # Initialize matrices with defaults
        matrices.update(default_matrices)

        # Extract actual parameter values from initialparameterization
        init_params = gnn_spec.get("initialparameterization", {})

        # Override dimensions from B matrix if available (consistent with other renderers)
        if "B" in init_params:
            B_matrix = init_params["B"]
            if isinstance(B_matrix, (list, np.ndarray)) and len(B_matrix) > 0:
                B_array = np.asarray(B_matrix, dtype=float)
                n_actions_from_b = B_array.shape[2] if B_array.ndim >= 3 else 1
                n_actions = n_actions_from_b
                logger.info(f"Corrected n_actions from B matrix: {n_actions}")

            # Also check for explicit dimensions in model_params that might override
            if "B" in gnn_spec.get("model_params", {}):
                B_spec = gnn_spec["model_params"]["B"]
                if "shape" in B_spec:
                    shape_parts = B_spec["shape"].strip("()").split(",")
                    if len(shape_parts) >= 3:
                        try:
                            n_actions_from_spec = int(shape_parts[2])
                            if n_actions_from_spec > 1:
                                n_actions = n_actions_from_spec
                                logger.info(
                                    f"Corrected n_actions from B matrix specification: {n_actions}"
                                )
                        except (ValueError, IndexError):
                            logger.debug(
                                "Could not extract n_actions from B matrix spec shape_parts"
                            )
        if init_params:
            logger.info(
                "Found initialparameterization, extracting actual matrix values"
            )

            # Extract A matrix
            if "A" in init_params:
                try:
                    A_data = init_params["A"]
                    # Handle both list and tuple (POMDP extractor returns tuples)
                    if isinstance(A_data, (list, tuple)):
                        A_matrix = np.array(A_data)
                        if A_matrix.ndim == 2:
                            matrices["A"] = A_matrix
                            logger.info(
                                f"Successfully extracted A matrix: shape {A_matrix.shape}"
                            )
                except Exception as e:
                    logger.warning(f"Failed to extract A matrix: {e}")

            # Extract B matrix
            if "B" in init_params:
                try:
                    B_data = init_params["B"]
                    # Handle both list and tuple (POMDP extractor returns tuples)
                    if isinstance(B_data, (list, tuple)):
                        B_matrix = np.array(B_data)
                    elif isinstance(B_data, np.ndarray):
                        B_matrix = B_data
                    else:
                        B_matrix = np.array([])
                    if B_matrix.ndim == 2:
                        matrices["B"] = B_matrix[:, :, np.newaxis]
                        logger.info(
                            f"Successfully extracted passive B matrix: shape {matrices['B'].shape}"
                        )
                    elif B_matrix.ndim == 3:
                        matrices["B"] = B_matrix
                        logger.info(
                            f"Successfully extracted B matrix: shape {B_matrix.shape}"
                        )
                except Exception as e:
                    logger.warning(f"Failed to extract B matrix: {e}")

            # Extract C vector
            if "C" in init_params:
                try:
                    C_data = init_params["C"]
                    if isinstance(C_data, list):
                        C_vector = np.array(C_data)
                        if C_vector.ndim == 1:
                            matrices["C"] = C_vector
                            logger.info(
                                f"Successfully extracted C vector: shape {C_vector.shape}"
                            )
                except Exception as e:
                    logger.warning(f"Failed to extract C vector: {e}")

            # Extract D vector
            if "D" in init_params:
                try:
                    D_data = init_params["D"]
                    if isinstance(D_data, list):
                        D_vector = np.array(D_data)
                        if D_vector.ndim == 1:
                            matrices["D"] = D_vector
                            logger.info(
                                f"Successfully extracted D vector: shape {D_vector.shape}"
                            )
                except Exception as e:
                    logger.warning(f"Failed to extract D vector: {e}")

    # --- Recovery: Handle the JSON export format from GNN processing pipeline ---
    elif "statespaceblock" in gnn_spec:
        logger.info("Extracting matrices from GNN JSON export structure")

        # Extract variable dimensions from statespaceblock
        var_dims: dict[Any, Any] = {}
        for var_data in gnn_spec.get("statespaceblock", []):
            var_name = var_data.get("id", "")
            dimensions_str = var_data.get("dimensions", "")
            # Parse dimensions like "3,3,type=float" -> [3, 3]
            if dimensions_str:
                dims_parts = dimensions_str.split(",")
                dims: list[Any] = []
                for part in dims_parts:
                    part = part.strip()
                    if part.startswith("type="):
                        break
                    try:
                        dims.append(int(part))
                    except ValueError:
                        logger.debug("Skipping non-integer dimension token: %s", part)
                        continue
                var_dims[var_name] = dims

        # Create default matrices based on dimensions
        default_matrices = {}
        if "A" in var_dims:
            dims = var_dims["A"]
            if len(dims) >= 2:
                default_matrices["A"] = np.eye(dims[0], dims[1])  # Identity matrix
                logger.info(f"Created default A matrix with dimensions {dims}")

        if "B" in var_dims:
            dims = var_dims["B"]
            if len(dims) >= 3:
                # Create identity-like transition matrix
                default_matrices["B"] = np.eye(dims[0], dims[1])[:, :, np.newaxis]
                default_matrices["B"] = np.repeat(
                    default_matrices["B"], dims[2], axis=2
                )
                logger.info(f"Created default B matrix with dimensions {dims}")

        if "C" in var_dims:
            dims = var_dims["C"]
            if len(dims) >= 1:
                default_matrices["C"] = np.zeros(dims[0])  # Zero preferences
                logger.info(f"Created default C vector with dimensions {dims}")

        if "D" in var_dims:
            dims = var_dims["D"]
            if len(dims) >= 1:
                default_matrices["D"] = np.ones(dims[0]) / dims[0]  # Uniform prior
                logger.info(f"Created default D vector with dimensions {dims}")

        # Initialize matrices with defaults
        matrices.update(default_matrices)

        # Extract actual parameter values from InitialParameterization
        initial_params = gnn_spec.get("raw_sections", {}).get(
            "InitialParameterization", ""
        )
        if initial_params:
            initial_params = _require_braced_assignments(initial_params)
            # Parse A matrix
            a_match = re.search(r"\bA\s*=\s*\{([^}]+)\}", initial_params, re.DOTALL)
            if a_match:
                try:
                    a_str = a_match.group(1).strip()
                    parsed_matrix = _parse_gnn_matrix_string(
                        f"{{{a_str}}}", strict=True
                    )
                    if parsed_matrix.size:
                        matrices["A"] = parsed_matrix
                        logger.info(
                            f"Successfully parsed A matrix from InitialParameterization: shape {parsed_matrix.shape}"
                        )
                    elif "A" in default_matrices:
                        improved_matrix = _create_improved_default_matrix(
                            "A", default_matrices["A"], a_str
                        )
                        matrices["A"] = improved_matrix
                        logger.info(
                            f"Used improved default A matrix: shape {improved_matrix.shape}"
                        )
                except Exception as e:
                    raise ValueError(f"Invalid authored A parameter: {e}") from e

            # Parse B matrix
            b_match = re.search(r"\bB\s*=\s*\{([^}]+)\}", initial_params, re.DOTALL)
            if b_match:
                try:
                    b_str = b_match.group(1).strip()
                    parsed_matrix = _parse_gnn_matrix_string(
                        f"{{{b_str}}}", strict=True
                    )
                    if parsed_matrix.size:
                        matrices["B"] = parsed_matrix
                        logger.info(
                            f"Successfully parsed B matrix from InitialParameterization: shape {parsed_matrix.shape}"
                        )
                    elif "B" in default_matrices:
                        improved_matrix = _create_improved_default_matrix(
                            "B", default_matrices["B"], b_str
                        )
                        matrices["B"] = improved_matrix
                        logger.info(
                            f"Used improved default B matrix: shape {improved_matrix.shape}"
                        )
                except Exception as e:
                    raise ValueError(f"Invalid authored B parameter: {e}") from e

            # Parse C vector
            c_match = re.search(r"\bC\s*=\s*\{([^}]+)\}", initial_params, re.DOTALL)
            if c_match:
                try:
                    c_str = c_match.group(1).strip()
                    parsed_vector = _parse_gnn_matrix_string(
                        f"{{{c_str}}}", strict=True
                    )
                    if parsed_vector.size:
                        matrices["C"] = parsed_vector.flatten()
                        logger.info(
                            f"Successfully parsed C vector from InitialParameterization: shape {parsed_vector.flatten().shape}"
                        )
                    elif "C" in default_matrices:
                        improved_vector = _create_improved_default_matrix(
                            "C", default_matrices["C"], c_str
                        )
                        matrices["C"] = improved_vector
                        logger.info(
                            f"Used improved default C vector: shape {improved_vector.shape}"
                        )
                except Exception as e:
                    raise ValueError(f"Invalid authored C parameter: {e}") from e

            # Parse D vector
            d_match = re.search(r"\bD\s*=\s*\{([^}]+)\}", initial_params, re.DOTALL)
            if d_match:
                try:
                    d_str = d_match.group(1).strip()
                    parsed_vector = _parse_gnn_matrix_string(
                        f"{{{d_str}}}", strict=True
                    )
                    if parsed_vector.size:
                        matrices["D"] = parsed_vector.flatten()
                        logger.info(
                            f"Successfully parsed D vector from InitialParameterization: shape {parsed_vector.flatten().shape}"
                        )
                    elif "D" in default_matrices:
                        improved_vector = _create_improved_default_matrix(
                            "D", default_matrices["D"], d_str
                        )
                        matrices["D"] = improved_vector
                        logger.info(
                            f"Used improved default D vector: shape {improved_vector.shape}"
                        )
                except Exception as e:
                    raise ValueError(f"Invalid authored D parameter: {e}") from e

    # Handle parsed GNN data structure (older format)
    elif "variables" in gnn_spec:
        logger.info("Extracting matrices from parsed GNN data structure")

        # Extract variable dimensions from the variables list
        var_dims = {}
        for _, var_data in enumerate(gnn_spec.get("variables", [])):
            var_name = var_data.get("id", "")  # Use 'id' instead of 'name'
            dimensions_str = var_data.get(
                "dimensions", ""
            )  # This is a string like "3,3,type=float"

            # Parse dimensions string like "3,3,type=float" -> [3, 3]
            if var_name and dimensions_str:
                dims = []
                for part in dimensions_str.split(","):
                    part = part.strip()
                    if part.startswith("type="):
                        break
                    try:
                        dims.append(int(part))
                    except ValueError:
                        logger.debug("Skipping non-integer dimension token: %s", part)
                        continue

                if dims:  # Only add if we successfully parsed dimensions
                    var_dims[var_name] = dims
                    logger.info(f"Found variable '{var_name}' with dimensions {dims}")
                else:
                    logger.warning(
                        f"Could not parse dimensions for variable '{var_name}': '{dimensions_str}'"
                    )

        logger.info(f"Extracted variable dimensions: {var_dims}")

        # Create default matrices based on dimensions
        default_matrices = {}
        if "A" in var_dims:
            dims = var_dims["A"]
            if len(dims) >= 2:
                default_matrices["A"] = np.eye(dims[0], dims[1])  # Identity matrix
                logger.info(
                    f"Created default A matrix with dimensions {dims} -> shape {default_matrices['A'].shape}"
                )

        if "B" in var_dims:
            dims = var_dims["B"]
            if len(dims) >= 3:
                # Create identity-like transition matrix
                default_matrices["B"] = np.eye(dims[0], dims[1])[:, :, np.newaxis]
                default_matrices["B"] = np.repeat(
                    default_matrices["B"], dims[2], axis=2
                )
                logger.info(
                    f"Created default B matrix with dimensions {dims} -> shape {default_matrices['B'].shape}"
                )

        if "C" in var_dims:
            dims = var_dims["C"]
            if len(dims) >= 1:
                default_matrices["C"] = np.zeros(dims[0])  # Zero preferences
                logger.info(
                    f"Created default C vector with dimensions {dims} -> shape {default_matrices['C'].shape}"
                )

        if "D" in var_dims:
            dims = var_dims["D"]
            if len(dims) >= 1:
                default_matrices["D"] = np.ones(dims[0]) / dims[0]  # Uniform prior
                logger.info(
                    f"Created default D vector with dimensions {dims} -> shape {default_matrices['D'].shape}"
                )

        # Initialize matrices with defaults (will be overwritten if parameter parsing succeeds)
        matrices.update(default_matrices)
        logger.info(
            f"Initialized matrices with shapes: A={matrices.get('A', 'None')}, B={matrices.get('B', 'None')}, C={matrices.get('C', 'None')}, D={matrices.get('D', 'None')}"
        )

        # Extract actual parameter values from InitialParameterization if available
        initial_params = gnn_spec.get("InitialParameterization", "")
        if initial_params:
            logger.info(
                "Found InitialParameterization section, attempting to parse matrix values"
            )
            initial_params = _require_braced_assignments(initial_params)

            # Parse A matrix
            a_match = re.search(r"\bA\s*=\s*\{([^}]+)\}", initial_params, re.DOTALL)
            if a_match:
                try:
                    a_str = a_match.group(1).strip()
                    parsed_matrix = _parse_gnn_matrix_string(
                        f"{{{a_str}}}", strict=True
                    )
                    if parsed_matrix.size:
                        matrices["A"] = parsed_matrix
                        logger.info(
                            f"Successfully parsed A matrix from InitialParameterization: shape {parsed_matrix.shape}"
                        )
                    elif "A" in default_matrices:
                        # Use default matrix with correct dimensions
                        matrices["A"] = default_matrices["A"]
                        logger.info(
                            f"Used default A matrix: shape {default_matrices['A'].shape}"
                        )
                except Exception as e:
                    raise ValueError(f"Invalid authored A parameter: {e}") from e

            # Parse B matrix
            b_match = re.search(r"\bB\s*=\s*\{([^}]+)\}", initial_params, re.DOTALL)
            if b_match:
                try:
                    b_str = b_match.group(1).strip()
                    parsed_matrix = _parse_gnn_matrix_string(
                        f"{{{b_str}}}", strict=True
                    )
                    if parsed_matrix.size:
                        matrices["B"] = parsed_matrix
                        logger.info(
                            f"Successfully parsed B matrix from InitialParameterization: shape {parsed_matrix.shape}"
                        )
                    elif "B" in default_matrices:
                        matrices["B"] = default_matrices["B"]
                        logger.info(
                            f"Used default B matrix: shape {default_matrices['B'].shape}"
                        )
                except Exception as e:
                    raise ValueError(f"Invalid authored B parameter: {e}") from e

            # Parse C vector
            c_match = re.search(r"\bC\s*=\s*\{([^}]+)\}", initial_params, re.DOTALL)
            if c_match:
                try:
                    c_str = c_match.group(1).strip()
                    parsed_vector = _parse_gnn_matrix_string(
                        f"{{{c_str}}}", strict=True
                    )
                    if parsed_vector.size:
                        matrices["C"] = parsed_vector.flatten()
                        logger.info(
                            f"Successfully parsed C vector from InitialParameterization: shape {parsed_vector.flatten().shape}"
                        )
                    elif "C" in default_matrices:
                        matrices["C"] = default_matrices["C"]
                        logger.info(
                            f"Used default C vector: shape {default_matrices['C'].shape}"
                        )
                except Exception as e:
                    raise ValueError(f"Invalid authored C parameter: {e}") from e

            # Parse D vector
            d_match = re.search(r"\bD\s*=\s*\{([^}]+)\}", initial_params, re.DOTALL)
            if d_match:
                try:
                    d_str = d_match.group(1).strip()
                    parsed_vector = _parse_gnn_matrix_string(
                        f"{{{d_str}}}", strict=True
                    )
                    if parsed_vector.size:
                        matrices["D"] = parsed_vector.flatten()
                        logger.info(
                            f"Successfully parsed D vector from InitialParameterization: shape {parsed_vector.flatten().shape}"
                        )
                    elif "D" in default_matrices:
                        matrices["D"] = default_matrices["D"]
                        logger.info(
                            f"Used default D vector: shape {default_matrices['D'].shape}"
                        )
                except Exception as e:
                    raise ValueError(f"Invalid authored D parameter: {e}") from e

        # Extract actual parameter values if available (older parameters section)
        for param_data in gnn_spec.get("parameters", []):
            param_name = param_data.get("name", "")
            param_value = param_data.get("value")

            if param_name in ["A", "B", "C", "D"] and param_value is not None:
                try:
                    if isinstance(param_value, str):
                        # Parse GNN matrix string format
                        parsed_matrix = _parse_gnn_matrix_string(
                            param_value, strict=True
                        )

                        # Enhanced dimension inference and validation
                        if parsed_matrix.size:
                            # Parsing succeeded with meaningful dimensions
                            matrices[param_name] = parsed_matrix
                            logger.info(
                                f"Successfully parsed {param_name} matrix from string: shape {parsed_matrix.shape}"
                            )
                        elif param_name in default_matrices:
                            # Parsing failed but we have default dimensions - use improved matrix
                            improved_matrix = _create_improved_default_matrix(
                                param_name, default_matrices[param_name], param_value
                            )
                            matrices[param_name] = improved_matrix
                            logger.info(
                                f"Used improved default {param_name} matrix based on context: shape {improved_matrix.shape}"
                            )
                        else:
                            # Parsing failed and no defaults - try dimension inference from context
                            inferred_matrix = _infer_matrix_from_context(
                                param_name, param_value, var_dims
                            )
                            matrices[param_name] = inferred_matrix
                            logger.info(
                                f"Inferred {param_name} matrix from context: shape {inferred_matrix.shape}"
                            )

                    elif isinstance(param_value, (list, tuple)):
                        matrices[param_name] = np.array(param_value)
                        logger.info(f"Converted {param_name} list/tuple to array")
                    elif isinstance(param_value, set):
                        # Convert set to list then to array
                        matrices[param_name] = np.array(list(param_value))
                        logger.info(f"Converted {param_name} set to array")
                    else:
                        matrices[param_name] = param_value
                        logger.info(f"Used {param_name} parameter value directly")
                except Exception as e:
                    raise ValueError(
                        f"Invalid authored {param_name} parameter: {e}"
                    ) from e

    else:
        # Handle older raw text format
        logger.info("Extracting matrices from raw text format")

        # Extract InitialParameterization section
        init_params = gnn_spec.get("InitialParameterization", "")
        if not init_params:
            logger.warning("No InitialParameterization found in GNN spec")
            return matrices

        # Parse A matrix (observation model)
        a_match = re.search(r"A\s*=\s*\[(.*?)\]", init_params, re.DOTALL)
        if a_match:
            try:
                a_str = a_match.group(1).strip()
                a_matrix = _parse_matrix_string(a_str)
                matrices["A"] = a_matrix
            except Exception as e:
                logger.error(f"Failed to parse A matrix: {e}")

        # Parse B matrix (transition model)
        b_match = re.search(r"B\s*=\s*\[(.*?)\]", init_params, re.DOTALL)
        if b_match:
            try:
                b_str = b_match.group(1).strip()
                b_matrix = _parse_matrix_string(b_str)
                matrices["B"] = b_matrix
            except Exception as e:
                logger.error(f"Failed to parse B matrix: {e}")

        # Parse C vector (preferences)
        c_match = re.search(r"C\s*=\s*\[(.*?)\]", init_params, re.DOTALL)
        if c_match:
            try:
                c_str = c_match.group(1).strip()
                c_vector = _parse_vector_string(c_str)
                matrices["C"] = c_vector
            except Exception as e:
                logger.error(f"Failed to parse C vector: {e}")

        # Parse D vector (priors)
        d_match = re.search(r"D\s*=\s*\[(.*?)\]", init_params, re.DOTALL)
        if d_match:
            try:
                d_str = d_match.group(1).strip()
                d_vector = _parse_vector_string(d_str)
                matrices["D"] = d_vector
            except Exception as e:
                logger.error(f"Failed to parse D vector: {e}")

    return matrices


def _validated_jax_matrices(gnn_spec: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Extract a coherent dense POMDP matrix set or fail closed.

    Generated code must reflect the supplied model. Historically incomplete
    inputs fell through to unrelated 2-state defaults and, on any exception,
    a recovery script that did no inference. Validate the contract before
    interpolation so a successful render always contains the requested model.
    """
    extracted = _extract_gnn_matrices(gnn_spec)
    missing = [name for name in ("A", "B", "C", "D") if name not in extracted]
    if missing:
        raise ValueError(
            "JAX rendering requires canonical A/B/C/D parameters; missing "
            + ", ".join(missing)
        )

    if any(np.iscomplexobj(extracted[name]) for name in ("A", "B", "C", "D")):
        raise ValueError("JAX matrices must contain only real values")

    matrices = {
        name: np.asarray(extracted[name], dtype=np.float64)
        for name in ("A", "B", "C", "D")
    }
    matrices["C"] = matrices["C"].reshape(-1)
    matrices["D"] = matrices["D"].reshape(-1)
    a_matrix = matrices["A"]
    b_matrix = matrices["B"]
    c_vector = matrices["C"]
    d_vector = matrices["D"]

    if a_matrix.ndim != 2:
        raise ValueError(f"A must be 2-D [observation, state], got {a_matrix.shape}")
    if b_matrix.ndim != 3:
        raise ValueError(
            f"B must be 3-D [next_state, previous_state, action], got {b_matrix.shape}"
        )
    num_observations, num_states = a_matrix.shape
    if b_matrix.shape[:2] != (num_states, num_states):
        raise ValueError(
            "B state axes must match A state count: "
            f"A={a_matrix.shape}, B={b_matrix.shape}"
        )
    if c_vector.shape != (num_observations,):
        raise ValueError(
            f"C length must match A observations: A={a_matrix.shape}, C={c_vector.shape}"
        )
    if d_vector.shape != (num_states,):
        raise ValueError(
            f"D length must match A states: A={a_matrix.shape}, D={d_vector.shape}"
        )
    if b_matrix.shape[2] <= 0:
        raise ValueError("B must declare at least one action")
    if any(not np.all(np.isfinite(value)) for value in matrices.values()):
        raise ValueError("JAX matrices must contain only finite values")

    # existing variables/JSON exports carry explicit dimensions independently
    # of their values. A successful render must honor both declarations.
    declarations = gnn_spec.get("statespaceblock", gnn_spec.get("variables", []))
    for declaration in declarations:
        name = declaration.get("id", "")
        if name not in matrices:
            continue
        dimensions = []
        for token in declaration.get("dimensions", "").split(","):
            token = token.strip()
            if token.startswith("type="):
                break
            try:
                dimensions.append(int(token))
            except ValueError:
                continue
        declared_source_shape = matrices[name].shape
        initial = gnn_spec.get("initialparameterization", {})
        if (
            name == "B"
            and "model_parameters" in gnn_spec
            and isinstance(initial, dict)
            and all(key in initial for key in ("A", "B", "C", "D"))
        ):
            # Declarations describe the authored tensor's axes. Canonical
            # admission may transpose action-major B or add a passive action
            # axis; neither operation rewrites the source declaration.
            declared_source_shape = np.asarray(initial["B"]).shape
        if dimensions and tuple(dimensions) != declared_source_shape:
            raise ValueError(
                f"{name} source shape {declared_source_shape} does not match declared dimensions {dimensions}"
            )

    # C is a real-valued payoff vector, including negative preferences.
    # A/B/D are probabilities, so only declared rounding drift may be
    # normalized under the same contract used by other dense backends.
    matrices["A"] = np.asarray(normalise_matrix_columns(a_matrix, name="A"))
    matrices["B"] = np.stack(
        [
            normalise_matrix_columns(b_matrix[:, :, action], name="B")
            for action in range(b_matrix.shape[2])
        ],
        axis=2,
    )
    matrices["D"] = np.asarray(normalise_vector(d_vector, name="D"))
    return matrices


def _jax_model_name(gnn_spec: Dict[str, Any], fallback: str) -> str:
    """Return a safe Python-facing model label from canonical or earlier keys."""
    value = (
        gnn_spec.get("model_name")
        or gnn_spec.get("ModelName")
        or gnn_spec.get("name")
        or fallback
    )
    return str(value).replace(" ", "_")


def _create_improved_default_matrix(
    param_name: str, default_matrix: np.ndarray, param_value: str
) -> np.ndarray:
    """Create an improved default matrix based on context clues from the failed parsing."""
    # Try to extract numerical values from the failed parsing string
    import re

    # Look for numerical patterns in the string
    numbers = re.findall(r"-?\d+\.?\d*", param_value)

    if numbers and len(numbers) > 1:
        # We found numbers, try to use them to improve the default matrix
        try:
            float_numbers = [float(n) for n in numbers]

            if param_name == "A":
                # For A matrix, try to create observation model with extracted values
                shape = default_matrix.shape
                new_matrix = np.zeros(shape)
                for i, val in enumerate(float_numbers[: int(np.prod(shape))]):
                    row = i // shape[1]
                    col = i % shape[1]
                    if row < shape[0] and col < shape[1]:
                        new_matrix[row, col] = val

                # Normalize rows to make it a proper probability matrix
                row_sums = new_matrix.sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums > 0, row_sums, 1.0)
                new_matrix = new_matrix / row_sums
                return new_matrix

            elif param_name == "B":
                # For B matrix, create transition model with extracted values
                shape = default_matrix.shape
                new_matrix = np.zeros(shape)
                for i, val in enumerate(float_numbers[: int(np.prod(shape))]):
                    # Map linear index to 3D coordinates
                    idx_2d = i % (shape[0] * shape[1])
                    action = i // (shape[0] * shape[1])
                    row = idx_2d // shape[1]
                    col = idx_2d % shape[1]
                    if action < shape[2] and row < shape[0] and col < shape[1]:
                        new_matrix[row, col, action] = val

                # Normalize each action matrix
                for a in range(shape[2]):
                    action_matrix = new_matrix[:, :, a]
                    row_sums = action_matrix.sum(axis=1, keepdims=True)
                    row_sums = np.where(row_sums > 0, row_sums, 1.0)
                    new_matrix[:, :, a] = action_matrix / row_sums
                return new_matrix

            elif param_name in ["C", "D", "E"]:
                # For vectors, use extracted values directly
                shape = default_matrix.shape
                new_vector = np.zeros(shape)
                for i, val in enumerate(float_numbers[: shape[0]]):
                    new_vector[i] = val

                # Normalize if it's a probability vector (D, E)
                if param_name in ["D", "E"]:
                    vector_sum = new_vector.sum()
                    if vector_sum > 0:
                        new_vector = new_vector / vector_sum
                    else:
                        new_vector = np.ones(shape) / shape[0]

                return new_vector
        except (ValueError, TypeError, IndexError):
            logger.debug(
                "Could not improve matrix from parameterization for %s, using default",
                param_name,
            )

    # If we can't improve it, return the default
    return default_matrix


def _infer_matrix_from_context(
    param_name: str, param_value: str, var_dims: dict
) -> np.ndarray:
    """Infer matrix dimensions and create appropriate matrix when no defaults are available."""

    # Try to infer dimensions from variable information
    if param_name in var_dims:
        dims = var_dims[param_name]
    else:
        # Use standard POMDP defaults
        dims = {
            "A": [2, 2],  # 2 observations x 2 states
            "B": [2, 2, 2],  # 2 states x 2 states x 2 actions
            "C": [2],  # 2 observations
            "D": [2],  # 2 states
            "E": [2],  # 2 actions
        }.get(param_name, [2])

    # Create appropriate matrix based on parameter type
    if param_name == "A":
        # Observation model - create informative but not deterministic
        shape = tuple(dims) if len(dims) >= 2 else (2, 2)
        matrix = np.eye(min(shape)) + 0.1 * np.random.rand(*shape)
        # Normalize rows
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
        return cast(np.ndarray, matrix)

    elif param_name == "B":
        # Transition model - create identity-like transitions with some noise
        shape = tuple(dims) if len(dims) >= 3 else (2, 2, 2)
        matrix = np.zeros(shape)
        for a in range(shape[2]):
            action_matrix = np.eye(shape[0], shape[1]) + 0.1 * np.random.rand(
                shape[0], shape[1]
            )
            matrix[:, :, a] = action_matrix / action_matrix.sum(axis=1, keepdims=True)
        return cast(np.ndarray, matrix)

    elif param_name == "C":
        # Preferences - slight preference for later observations
        vector_len = int(dims[0]) if dims else 2
        vector = np.linspace(0.1, 1.0, vector_len)
        return cast(np.ndarray, vector)

    elif param_name == "D":
        # Prior - uniform
        vector_len = int(dims[0]) if dims else 2
        return cast(np.ndarray, np.ones(vector_len) / vector_len)

    elif param_name == "E":
        # Action prior - uniform
        vector_len = int(dims[0]) if dims else 2
        return cast(np.ndarray, np.ones(vector_len) / vector_len)

    else:
        # Generic recovery
        vector_len = int(dims[0]) if dims else 2
        return cast(np.ndarray, np.ones(vector_len) / vector_len)


def _create_fallback_matrix(param_name: str, var_dims: dict) -> np.ndarray:
    """Create a recovery matrix when all else fails."""
    return _infer_matrix_from_context(param_name, "", var_dims)


def _parse_matrix_string(matrix_str: str) -> np.ndarray:
    """Parse matrix string to numpy array."""
    # Remove extra whitespace and newlines
    matrix_str = re.sub(r"\s+", " ", matrix_str.strip())

    # Split by rows and parse each row
    rows: list[Any] = []
    for row_str in matrix_str.split(";"):
        row_str = row_str.strip()
        if row_str:
            # Parse row as list of floats
            row_values: list[Any] = []
            for val_str in row_str.split(","):
                val_str = val_str.strip()
                if val_str:
                    try:
                        row_values.append(float(val_str))
                    except ValueError:
                        logger.warning(f"Could not parse value: {val_str}")
                        row_values.append(0.0)
            rows.append(row_values)

    if not rows:
        return np.array([[1.0]])

    # Ensure all rows have same length
    max_len = max(len(row) for row in rows)
    for _, row in enumerate(rows):
        while len(row) < max_len:
            row.append(0.0)

    return np.array(rows)


def _parse_vector_string(vector_str: str) -> np.ndarray:
    """Parse vector string to numpy array."""
    # Remove extra whitespace
    vector_str = re.sub(r"\s+", " ", vector_str.strip())

    # Parse as list of floats
    values: list[Any] = []
    for val_str in vector_str.split(","):
        val_str = val_str.strip()
        if val_str:
            try:
                values.append(float(val_str))
            except ValueError:
                logger.warning(f"Could not parse value: {val_str}")
                values.append(0.0)

    if not values:
        return np.array([1.0])

    return np.array(values)

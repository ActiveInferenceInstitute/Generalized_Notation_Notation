"""Numeric extraction from GNN parameter payloads."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from ..compat.viz_compat import np

NUMPY_AVAILABLE = np is not None


def is_transition_tensor_name(name: str) -> bool:
    """Canonical GNN B and explicitly indexed B factors carry transition axes."""
    return name == "B" or (name.startswith("B_f") and name[3:].isdigit())


def convert_to_matrix(value: Any, name: str = "") -> Optional[Any]:
    """Convert nested lists / tuples to a numpy array, or None."""
    if not NUMPY_AVAILABLE or np is None:
        return None
    if value is None:
        return None
    try:
        if isinstance(value, (list, tuple)):
            if len(value) > 0 and isinstance(value[0], (list, tuple)):
                matrix = np.array(value, dtype=float)
                if matrix.size > 0:
                    return matrix
            matrix = np.array(value, dtype=float)
            if matrix.size > 0:
                return matrix
        matrix = np.array(value, dtype=float)
        if matrix.size > 0:
            return matrix
    except (ValueError, TypeError):
        return None
    return None


def extract_matrix_data_from_parameters(
    parameters: Union[List[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """Map parameter names to numpy arrays (same rules as MatrixVisualizer)."""
    matrices: Dict[str, Any] = {}
    if not parameters:
        return matrices
    if isinstance(parameters, dict):
        for param_name, param_value in parameters.items():
            matrix = convert_to_matrix(param_value, param_name)
            if matrix is not None:
                matrices[param_name] = matrix
        return matrices
    for param in parameters:
        param_obj: object = param
        if not isinstance(param_obj, dict):
            continue
        param_name = param_obj.get("name", "")
        if not param_name:
            continue
        param_value = param_obj.get("value")
        if param_value is None:
            continue
        matrix = convert_to_matrix(param_value, param_name)
        if matrix is not None:
            matrices[param_name] = matrix
    return matrices


def collect_visualization_matrices(parsed_data: Dict[str, Any]) -> Dict[str, Any]:
    """Collect numeric matrices for rendering from a parsed GNN model dict.

    Resolution order mirrors ``process_single_gnn_file``: parameters →
    variables (re-using :func:`extract_matrix_data_from_parameters`) →
    raw ``matrices`` entries (converted via :func:`convert_to_matrix`). The
    first non-empty stage wins; later stages are skipped once a matrix is
    found, exactly as the inline loop did.

    Returns an ordered ``{name: np.ndarray}`` mapping (possibly empty).
    """
    parameters = parsed_data.get("parameters") or []
    matrices = extract_matrix_data_from_parameters(parameters)
    if not matrices:
        matrices = extract_matrix_data_from_parameters(
            parsed_data.get("variables") or []
        )
    if not matrices:
        for m_info in parsed_data.get("matrices") or []:
            if not isinstance(m_info, dict) or "data" not in m_info:
                continue
            m_name = m_info.get("name", f"matrix_{len(matrices)}")
            m_data = convert_to_matrix(m_info.get("data"), m_name)
            if m_data is not None:
                matrices[m_name] = m_data
    return matrices

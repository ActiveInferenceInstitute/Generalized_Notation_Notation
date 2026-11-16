"""GNN schema — lightweight parsing and validation for GNN documents.

Thin re-export surface; the implementation lives in
:mod:`gnn.schema.parser`. This module stays import-light (stdlib only at
module scope) so headless consumers (lsp, cli watcher, extract) can import it
without dragging pipeline weight.
"""

from gnn.schema.parser import (
    GNN_MODEL_SCHEMA,
    REQUIRED_SECTIONS,
    GNNConnectionEdge,
    GNNParseError,
    GNNVariable,
    check_gnn_object,
    parse_connections,
    parse_state_space,
    validate_gnn_object,
    validate_matrix_dimensions,
    validate_required_sections,
)
from gnn.schema.types import GNNObjectValidation

__all__ = [
    "GNN_MODEL_SCHEMA",
    "GNNConnectionEdge",
    "GNNObjectValidation",
    "GNNParseError",
    "GNNVariable",
    "REQUIRED_SECTIONS",
    "check_gnn_object",
    "parse_connections",
    "parse_state_space",
    "validate_gnn_object",
    "validate_matrix_dimensions",
    "validate_required_sections",
]

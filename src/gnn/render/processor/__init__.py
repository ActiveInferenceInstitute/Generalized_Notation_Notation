#!/usr/bin/env python3
"""
Enhanced Render processor module for GNN code generation with POMDP-aware processing.

This module provides comprehensive rendering capabilities that:
1. Extract POMDP state spaces from GNN specifications
2. Modularly inject them into framework-specific renderers
3. Create implementation-specific output subfolders
4. Provide structured documentation and results
"""

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from gnn.types import GNNInternalRepresentation

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    np = cast(Any, None)
    NUMPY_AVAILABLE = False

# Ensure src and project root are on the path for cross-module imports
_src_path = Path(__file__).parent.parent.parent
_project_root = _src_path.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logger = logging.getLogger(__name__)

from gnn.render.framework_registry import (
    get_available_renderers as _registry_get_available_renderers,
)
from gnn.render.framework_registry import (
    get_lite_frameworks,
)
from gnn.render.framework_registry import (
    get_supported_frameworks as _registry_get_supported_frameworks,
)
from gnn.render.naming import safe_output_stem

from gnn.render.processor.metadata import (
    get_available_renderers,
    get_module_info,
)
from gnn.render.processor.parsing import (
    _internal_representation_to_mapping,
    _node_to_mapping,
    _normalize_initial_vectors,
    _rehydrate_file_backed_parse_summary,
    _render_succeeded,
    _safe_output_stem,
    normalize_matrices,
    validate_pomdp_for_rendering,
)
from gnn.render.processor.pipeline import (
    _create_overview_documentation,
    _process_single_gnn_file_basic,
    process_render,
)
from gnn.render.processor.rendering import (
    _render_continuous_target,
    render_gnn_spec,
)
from gnn.render.processor.receipts import (
    _atomic_render_json,
    _load_prior_render_summary,
    _render_file_identity,
    _write_render_receipt,
    parse_frameworks_selection,
)

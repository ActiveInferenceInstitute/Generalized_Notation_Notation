"""Matrix string parsing and batch matrix viz entrypoint (keeps analysis/ free of visualization)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

try:
    import numpy as np

    _NP_AVAILABLE = True
except ImportError:
    np = cast(Any, None)
    _NP_AVAILABLE = False


def parse_matrix_data(matrix_str: str) -> Optional[Any]:
    """Extract a flat finite numeric array for advisory visualization, or None."""
    try:
        if not isinstance(matrix_str, str) or not _NP_AVAILABLE or np is None:
            return None
        if re.search(r"\b(?:nan|inf(?:inity)?)\b", matrix_str, re.IGNORECASE):
            return None
        numbers = re.findall(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", matrix_str
        )
        if not numbers:
            return None
        values = np.array([float(number) for number in numbers])
        return values if np.isfinite(values).all() else None
    except Exception:
        return None


def generate_matrix_visualizations(
    parsed_data: Dict[str, Any], output_dir: Path, model_name: str
) -> List[str]:
    """Run matrix analysis + stats plots from parsed GNN data."""
    try:
        from .visualizer import (
            generate_matrix_visualizations as _run,
        )

        return _run(parsed_data, output_dir, model_name)
    except Exception:
        return []

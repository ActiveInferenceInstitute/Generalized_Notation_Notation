#!/usr/bin/env python3
"""Cross-mixin typing contract for the POMDP render processor package.

Mechanical support module for the M-01 band split, following the established
``gnn.extract.pomdp_support`` precedent: the composed class
``POMDPRenderProcessor`` assembles seven topic mixins, and the moved bodies
reference shared state and sibling-mixin methods that mypy cannot infer from
composition alone. Declarations here are annotation-only under
``TYPE_CHECKING`` so every mixin type-checks standalone; the real values and
implementations are created by the owning mixins (``_flow.py`` state init;
method bodies on their owning modules). Runtime behavior is unchanged: this
module defines no runtime members.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    import logging

    from gnn.extract.pomdp_extractor import POMDPStateSpace


class _POMDPProcessorSupportMixin:
    """Declares the shared cross-mixin surface of ``POMDPRenderProcessor``."""

    if TYPE_CHECKING:
        # Shared state, created by ``_ProcessFlowMixin.__init__``.
        logger: "logging.Logger"
        framework_configs: Dict[str, Any]

        # Sibling-mixin methods referenced across composition boundaries.
        def _validate_pomdp_framework_compatibility(
            self, pomdp_space: "POMDPStateSpace", framework: str
        ) -> Dict[str, Any]: ...

        def _time_indexed_transition_key(
            self, matrices: Dict[str, Any]
        ) -> Optional[str]: ...

        def _validate_state_spaces_in_spec(
            self, gnn_spec: Dict[str, Any], framework: str
        ) -> Dict[str, Any]: ...

        def _validate_state_spaces_in_script(
            self, script_path: Path, gnn_spec: Dict[str, Any]
        ) -> Dict[str, Any]: ...

        def _build_canonical_initialparameterization(
            self,
            pomdp_space: "POMDPStateSpace",
        ) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, Any]]: ...

        def _pomdp_to_gnn_spec(
            self, pomdp_space: "POMDPStateSpace", **kwargs: Any
        ) -> Dict[str, Any]: ...

        def _create_framework_documentation(
            self,
            framework: str,
            pomdp_space: "POMDPStateSpace",
            output_dir: Path,
            render_result: Dict[str, Any],
        ) -> None: ...

        def _call_framework_renderer(
            self,
            framework: str,
            gnn_spec: Dict[str, Any],
            output_dir: Path,
            **kwargs: Any,
        ) -> Dict[str, Any]: ...

#!/usr/bin/env python3
"""
File-level DisCoPy translation orchestrators for GNN Step 11.

Extracted from ``render.discopy.translator``.
"""

import logging
from pathlib import Path
from typing import (
    Any,
    Optional,
)

import numpy

from .bootstrap import (
    JAX_AVAILABLE,
    TENSOR_COMPONENTS_AVAILABLE,
    Diagram,
    create_discopy_error_report,
    discopy_backend,
    jax,
    jnp,
)
from .diagram_builders import (
    gnn_connections_to_discopy_diagram,
    gnn_statespace_to_discopy_dims_map,
)
from .gnn_parsing import parse_gnn_content
from .matrix_builders import gnn_connections_to_discopy_matrix_diagram

logger = logging.getLogger(__name__)


def gnn_file_to_discopy_diagram(
    gnn_file_path: Path, verbose: bool = False
) -> Optional[Diagram]:
    """
    Orchestrates the conversion of a GNN file to a DisCoPy diagram (tensor.Diagram).
    Reads the file, parses content, converts state space and connections.
    """
    # Set logger level for this module based on verbose flag from the caller
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.info(f"Attempting to convert GNN file to DisCoPy diagram: {gnn_file_path}")
    if not gnn_file_path.exists():
        logger.error(f"GNN file not found: {gnn_file_path}")
        return None

    # Check if DisCoPy components are available
    if not TENSOR_COMPONENTS_AVAILABLE:
        create_discopy_error_report(gnn_file_path, "unavailable")
        logger.error("DisCoPy components are not available. Cannot create diagrams.")
        logger.info("Run generate_setup_report() for installation instructions")
        return None

    try:
        with open(gnn_file_path, "r", encoding="utf-8") as f:
            content = f.read()

        parsed_gnn = parse_gnn_content(content)
        if not parsed_gnn:
            logger.error(
                f"Failed to parse GNN content from {gnn_file_path}. No sections found."
            )
            return None

        discopy_dims_map = gnn_statespace_to_discopy_dims_map(parsed_gnn)
        if not discopy_dims_map:
            logger.warning(
                f"No DisCoPy Dims generated from StateSpaceBlock in {gnn_file_path}."
            )
            # Proceeding as some diagrams might not need explicit dims (e.g. only names)

        diagram = gnn_connections_to_discopy_diagram(parsed_gnn, discopy_dims_map)

        if diagram:
            logger.info(
                f"Successfully created DisCoPy diagram from GNN file: {gnn_file_path}"
            )
            logger.debug(
                f"Diagram structure: dom={diagram.dom}, cod={diagram.cod}, #boxes={len(diagram.boxes) if hasattr(diagram, 'boxes') else 'N/A'}"
            )
        else:
            logger.warning(
                f"Could not create a DisCoPy diagram from GNN file: {gnn_file_path}. Check Connections section."
            )

        return diagram

    except Exception as e:
        logger.error(
            f"Error converting GNN file {gnn_file_path} to DisCoPy diagram: {e}",
            exc_info=True,
        )
        return None


def gnn_file_to_discopy_matrix_diagram(
    gnn_file_path: Path, verbose: bool = False, jax_seed: int = 0
) -> Optional[Diagram]:
    """
    Experimental compatibility utility for JAX-backed matrix construction.

    Numerical evaluation is unsupported: ordinary real rows can be interpreted
    as complex pairs, and tensor/matrix constructor semantics are unresolved.
    Dependency availability does not establish model validity. See the sibling
    README support boundary; prefer the supported package generator or abstract
    tensor diagram translator for their respective documented contracts.
    """
    if not JAX_AVAILABLE:  # Check the overall JAX_AVAILABLE flag
        create_discopy_error_report(gnn_file_path, "jax_unavailable")
        logger.error(
            "JAX or essential DisCoPy components for matrix operations are not available. Cannot create a JAX-backed MatrixDiagram."
        )
        logger.info("Run generate_setup_report() for installation instructions")
        return None

    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.info(
        f"Attempting to convert GNN file to DisCoPy MatrixDiagram: {gnn_file_path}"
    )
    if not gnn_file_path.exists():
        logger.error(f"GNN file not found: {gnn_file_path}")
        return None

    try:
        with open(gnn_file_path, "r", encoding="utf-8") as f:
            content = f.read()

        parsed_gnn = parse_gnn_content(content)
        if not parsed_gnn:
            logger.error(
                f"Failed to parse GNN content from {gnn_file_path}. No sections found."
            )
            return None

        # Set JAX backend for DisCoPy matrix operations
        with discopy_backend("jax"):
            discopy_dims_map = gnn_statespace_to_discopy_dims_map(parsed_gnn)
            if not discopy_dims_map:
                logger.warning(
                    f"No DisCoPy Dims generated from StateSpaceBlock in {gnn_file_path}."
                )

            tensor_definitions = parsed_gnn.get("TensorDefinitions", {})
            if not tensor_definitions:
                logger.warning(
                    f"No 'TensorDefinitions' section found in {gnn_file_path}. Boxes may not be initialized."
                )

            # PRNG key provider for random initializations - only if JAX is available
            key_provider = None
            if JAX_AVAILABLE and jax:
                jax_random_module_local = getattr(jax, "random", None)
                if (
                    jax_random_module_local
                    and hasattr(jax_random_module_local, "PRNGKey")
                    and hasattr(jax_random_module_local, "fold_in")
                ):
                    base_key = jax_random_module_local.PRNGKey(jax_seed)

                    # Capture jax_random_module_local in the closure
                    def _key_provider_impl(
                        name_suffix: str, _jrm: Any = jax_random_module_local
                    ) -> Any:
                        """Handle key provider impl for internal callers."""
                        hashed_suffix = hash(name_suffix) & ((1 << 32) - 1)
                        return _jrm.fold_in(
                            base_key, hashed_suffix
                        )  # Use captured _jrm

                    key_provider = _key_provider_impl
                elif jax_random_module_local:
                    logger.warning(
                        "jax.random module is available, but PRNGKey or fold_in attribute is missing. Cannot create PRNG key provider."
                    )
                else:
                    logger.warning(
                        "jax.random module is not available within JAX. Cannot create PRNG key provider."
                    )
            else:
                logger.info(
                    "JAX is not available, PRNG key provider will not be created."
                )

            diagram = gnn_connections_to_discopy_matrix_diagram(
                parsed_gnn,
                discopy_dims_map,
                tensor_definitions,
                key_provider,  # Pass the potentially None key_provider
                default_dtype_str=getattr(jnp, "float32", "float32")
                if JAX_AVAILABLE and jnp
                else "float32",  # Pass jnp.dtype or recovery
            )

        if diagram:
            logger.info(
                f"Successfully created DisCoPy MatrixDiagram from GNN file: {gnn_file_path}"
            )
            logger.debug(
                f"MatrixDiagram: dom={diagram.dom}, cod={diagram.cod}, #boxes={len(diagram.boxes) if hasattr(diagram, 'boxes') else 'N/A'}"
            )
            if (
                hasattr(diagram, "boxes")
                and diagram.boxes
                and hasattr(diagram.boxes[0], "data")
                and diagram.boxes[0].data is not None
            ):
                first_box_data = diagram.boxes[0].data
                # Check if it's a JAX array (if jnp is not None and it's an instance)
                if JAX_AVAILABLE and jnp and isinstance(first_box_data, jnp.ndarray):
                    logger.info(f"  First box data (JAX array): {first_box_data}")
                elif isinstance(
                    first_box_data, numpy.ndarray
                ):  # Check for numpy array if JAX not used or as recovery
                    logger.info(f"  First box data (NumPy array): {first_box_data}")
                else:
                    logger.info(f"  First box data type: {type(first_box_data)}")

                # Evaluation test
                if discopy_backend is not None:
                    backend_context = discopy_backend("jax")
                    if backend_context:
                        with backend_context:
                            eval_result = diagram.eval()
                            logger.info(
                                f"MatrixDiagram evaluation result (JAX backend): {eval_result}"
                            )
                            if hasattr(eval_result, "array"):
                                logger.info(
                                    f"  Evaluation result array: {eval_result.array}"
                                )
                    else:
                        logger.warning(
                            "Could not obtain JAX backend context for evaluation."
                        )
                else:
                    logger.warning(
                        "DisCoPy JAX backend is not available. Skipping evaluation test."
                    )
                # Log diagram properties safely
                if diagram is not None:
                    logger.info(
                        f"MatrixDiagram created: dom={diagram.dom}, cod={diagram.cod}, boxes: {len(diagram.boxes) if hasattr(diagram, 'boxes') else 'N/A'}"
                    )
                    if (
                        hasattr(diagram, "boxes")
                        and diagram.boxes
                        and hasattr(diagram.boxes[0], "data")
                        and diagram.boxes[0].data is not None
                    ):
                        first_box_data = diagram.boxes[0].data
                        # Check if it's a JAX array (if jnp is not None and it's an instance)
                        if (
                            JAX_AVAILABLE
                            and jnp
                            and isinstance(first_box_data, jnp.ndarray)
                        ):
                            logger.info(
                                f"  First box data (JAX array): {first_box_data}"
                            )
                        elif isinstance(
                            first_box_data, numpy.ndarray
                        ):  # Check for numpy array if JAX not used or as recovery
                            logger.info(
                                f"  First box data (NumPy array): {first_box_data}"
                            )
                        else:
                            logger.info(
                                f"  First box data type: {type(first_box_data)}"
                            )
                    else:
                        logger.info(
                            "MatrixDiagram has no boxes or first box has no data."
                        )
                else:
                    logger.info("MatrixDiagram creation failed.")
            else:
                logger.info("MatrixDiagram has no boxes or first box has no data.")
        else:
            logger.error(f"Overall MatrixDiagram creation failed for {gnn_file_path}.")

        return diagram

    except Exception as e:
        logger.error(
            f"Error converting GNN file {gnn_file_path} to DisCoPy MatrixDiagram: {e}",
            exc_info=True,
        )
        return None

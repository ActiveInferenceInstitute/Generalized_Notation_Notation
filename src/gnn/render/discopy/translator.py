"""
GNN to DisCoPy Diagram Translator

This module provides functions to parse GNN files (in their string representation)
and convert them into DisCoPy diagrams.

Mechanical split facade: implementations live in ``bootstrap``,
``gnn_parsing``, ``diagram_builders``, ``matrix_builders``,
``file_translation``, and ``code_templates`` sibling modules; every
previously public and private name is re-exported here so consumer
import paths are unchanged. The standalone __main__ smoke harness is
retained as a diagnostic. The file-level JAX-backed MatrixDiagram utility
is experimental and unsupported for numerical evaluation; see the sibling
README support boundary. Abstract tensor diagrams and the supported package
generator have separate contracts.
"""

import logging
from pathlib import Path

import numpy

from .bootstrap import (
    DISCOPY_MATRIX_MODULE_AVAILABLE,
    JAX_AVAILABLE,
    JAX_CORE_AVAILABLE,
    TENSOR_COMPONENTS_AVAILABLE,
    TY_AVAILABLE,
    Box,
    Cap,
    Cup,
    Diagram,
    Dim,
    DisCoPySetupError,
    Functor,
    Id,
    Matrix,
    Spider,
    Swap,
    Ty,
    Word,
    _discopy_initialized,
    check_discopy_availability,
    create_discopy_error_report,
    discopy_backend,
    generate_setup_report,
    initialize_discopy_components,
    jax,
    jnp,
)
from .code_templates import (
    gnn_spec_to_discopy_code,
    gnn_spec_to_discopy_jax_code,
)
from .diagram_builders import (
    gnn_connections_to_discopy_diagram,
    gnn_statespace_to_discopy_dims_map,
)
from .file_translation import (
    gnn_file_to_discopy_diagram,
    gnn_file_to_discopy_matrix_diagram,
)
from .gnn_parsing import (
    _convert_json_to_complex_array,
    _parse_dims_str,
    parse_gnn_content,
)
from .matrix_builders import (
    gnn_connections_to_discopy_matrix_diagram,
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Example usage for standalone testing of this translator module
    # This requires a test GNN file to be present at the specified path.

    # Configure basic logging for standalone testing
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create a test GNN file for testing
    test_gnn_file_content = """
## ModelName
Test DisCoPy Model

## StateSpaceBlock
# Variable definitions
A[2]
B[3]
C[2]
D # A simple variable

## Connections
# Model connections
A > B
B > C
# D > A # Example of a connection that might cause issues with simple linear assumption if not handled
"""
    test_gnn_path = Path("__test_discopy_gnn.md")
    with open(test_gnn_path, "w", encoding="utf-8") as f_test:
        f_test.write(test_gnn_file_content)

    logger.info(f"--- Running Translator Standalone Test with {test_gnn_path} ---")

    # Test parsing
    parsed_data = parse_gnn_content(test_gnn_file_content)
    logger.info(f"Parsed GNN content: {parsed_data}")

    if parsed_data:
        # Test Dim map creation (replaces type conversion)
        dims_map_test = gnn_statespace_to_discopy_dims_map(parsed_data)
        logger.info(f"Generated DisCoPy Dims map: {dims_map_test}")

        # Test diagram creation
        diagram_test = gnn_connections_to_discopy_diagram(parsed_data, dims_map_test)
        if diagram_test:
            logger.info(f"Generated DisCoPy diagram: {diagram_test}")
            logger.info(f"  Diagram DOM: {diagram_test.dom}, COD: {diagram_test.cod}")
            logger.info(f"  Diagram Boxes: {diagram_test.boxes}")

            # Try to draw if matplotlib is available
            try:
                output_image_path = Path("__test_discopy_diagram.png")
                diagram_test.draw(
                    path=str(output_image_path), show_types=True, figsize=(8, 4)
                )
                logger.info(f"Diagram drawn to {output_image_path}")
            except ImportError:
                logger.warning("matplotlib not found, skipping diagram drawing.")
            except Exception as e_draw:
                logger.error(f"Error drawing diagram: {e_draw}")

        else:
            logger.warning("Diagram creation failed in standalone test.")
    else:
        logger.error("Parsing failed in standalone test.")

    # Test the main orchestrator function
    logger.info(f"--- Testing gnn_file_to_discopy_diagram on {test_gnn_path} ---")
    overall_diagram = gnn_file_to_discopy_diagram(test_gnn_path, verbose=True)
    if overall_diagram:
        # Log diagram properties safely
        if overall_diagram is not None:
            logger.info(
                f"Overall diagram created successfully: {overall_diagram}. Dom: {overall_diagram.dom}, Cod: {overall_diagram.cod}, Boxes: {len(overall_diagram.boxes) if hasattr(overall_diagram, 'boxes') else 'N/A'}"
            )
            if (
                TENSOR_COMPONENTS_AVAILABLE
                and TY_AVAILABLE
                and hasattr(overall_diagram, "draw")
            ):  # Draw if real components are available
                try:
                    draw_path = Path("__test_discopy_diagram.png")
                    overall_diagram.draw(
                        path=str(draw_path), show_types=True, figsize=(8, 4)
                    )
                    logger.info(f"Diagram drawn to {draw_path}")
                except Exception as e_draw:
                    logger.error(f"Error drawing diagram: {e_draw}")
        else:
            logger.info("Overall diagram creation failed.")
    else:
        logger.error(f"Overall diagram creation failed for {test_gnn_path}.")

    # Clean up test file
    test_gnn_path.unlink(missing_ok=True)
    Path("__test_discopy_diagram.png").unlink(missing_ok=True)

    # Example for MatrixDiagram (if JAX is available and GNN file is adapted)
    if JAX_AVAILABLE:
        test_gnn_matrix_content = """
## ModelName
Test DisCoPy MatrixDiagram Model

## StateSpaceBlock
A[2]
B[2]
C[2]

## TensorDefinitions
# BoxName | DomSpec (ignored) | CodSpec (ignored) | Initializer
A_to_B    | 2                 | 2                 | [[1.0, 0.0], [0.0, 1.0]]
B_to_C    | 2                 | 2                 | "random_normal:bc_key" 
# C_to_A | 2                 | 2                 | "load:./test_tensor_data.npy" # Needs test_tensor_data.npy

## Connections
A > B
B > C
# C > A # Cycle
"""
        test_matrix_gnn_path = Path("__test_discopy_matrix_gnn.md")
        # numpy.save("__test_tensor_data.npy", numpy.array([[0.5,0.5],[0.5,0.5]])) # If using load

        with open(test_matrix_gnn_path, "w", encoding="utf-8") as f_test_matrix:
            f_test_matrix.write(test_gnn_matrix_content)

        logger.info(
            f"--- Testing gnn_file_to_discopy_matrix_diagram on {test_matrix_gnn_path} ---"
        )
        if (
            JAX_AVAILABLE
            and DISCOPY_MATRIX_MODULE_AVAILABLE
            and discopy_backend is not None
        ):  # Ensure backend is available for the context manager
            matrix_diagram = gnn_file_to_discopy_matrix_diagram(
                test_matrix_gnn_path, verbose=True, jax_seed=42
            )
            if matrix_diagram:
                logger.info(
                    f"MatrixDiagram created: dom={matrix_diagram.dom}, cod={matrix_diagram.cod}, boxes: {len(matrix_diagram.boxes) if hasattr(matrix_diagram, 'boxes') else 'N/A'}"
                )
                if (
                    hasattr(matrix_diagram, "boxes")
                    and matrix_diagram.boxes
                    and hasattr(matrix_diagram.boxes[0], "data")
                    and matrix_diagram.boxes[0].data is not None
                ):
                    first_box_data = matrix_diagram.boxes[0].data
                    # Check if it's a JAX array (if jnp is not None and it's an instance)
                    if (
                        JAX_AVAILABLE
                        and jnp
                        and isinstance(first_box_data, jnp.ndarray)
                    ):
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
                            eval_result = matrix_diagram.eval()
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
                # Log matrix_diagram properties safely
                if matrix_diagram is not None:
                    logger.info(
                        f"MatrixDiagram created: dom={matrix_diagram.dom}, cod={matrix_diagram.cod}, boxes: {len(matrix_diagram.boxes) if hasattr(matrix_diagram, 'boxes') else 'N/A'}"
                    )
                    if (
                        hasattr(matrix_diagram, "boxes")
                        and matrix_diagram.boxes
                        and hasattr(matrix_diagram.boxes[0], "data")
                        and matrix_diagram.boxes[0].data is not None
                    ):
                        first_box_data = matrix_diagram.boxes[0].data
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
                    logger.info(f"MatrixDiagram type: {type(matrix_diagram)}")
            else:
                logger.error(
                    f"Overall MatrixDiagram creation failed for {test_matrix_gnn_path}."
                )

        test_matrix_gnn_path.unlink(missing_ok=True)
        # Path("__test_tensor_data.npy").unlink(missing_ok=True) # If using load

    logger.info("--- Standalone Translator Test Finished ---")

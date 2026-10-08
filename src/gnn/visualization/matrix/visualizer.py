#!/usr/bin/env python3
"""
Matrix Visualization Module for GNN Processing Pipeline

This module provides matrix visualization capabilities for GNN models,
including heatmaps, statistics, and analysis of model parameters.
Specialized support for 3D tensors like POMDP transition matrices.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from ..compat.viz_compat import MATPLOTLIB_AVAILABLE, get_sns, np, plt
from ..plotting.utils import contrasting_text_color, safe_tight_layout, save_figure
from .extract import (
    convert_to_matrix,
    is_transition_tensor_name,
)
from .extract import (
    extract_matrix_data_from_parameters as extract_matrices_from_parameter_list,
)
from .threejs_template import _threejs_tensor_html

logger = logging.getLogger(__name__)

NUMPY_AVAILABLE = np is not None
SEABORN_AVAILABLE = importlib.util.find_spec("seaborn") is not None


# Maximum figure dimension (inches) to prevent RendererAgg pixel overflow.
# At 300 DPI, 200 inches = 60,000 pixels — well within safe 32-bit limits.
_MAX_FIGURE_DIMENSION = 200


def _safe_figsize(width: float, height: float) -> Tuple[float, float]:
    """Clamp figure dimensions to prevent matplotlib RendererAgg overflow.

    Large models can produce data-dependent figsize values that exceed
    the renderer's pixel buffer capacity (e.g. 5 * 10,000 actions).
    This helper caps both dimensions to _MAX_FIGURE_DIMENSION inches.

    Args:
        width: Desired figure width in inches.
        height: Desired figure height in inches.

    Returns:
        Tuple of (clamped_width, clamped_height).
    """
    clamped_w = min(max(width, 1), _MAX_FIGURE_DIMENSION)
    clamped_h = min(max(height, 1), _MAX_FIGURE_DIMENSION)
    if clamped_w != width or clamped_h != height:
        logger.debug(
            "Figure size clamped from (%.1f, %.1f) to (%.1f, %.1f) to prevent renderer overflow",
            width,
            height,
            clamped_w,
            clamped_h,
        )
    return (clamped_w, clamped_h)


class MatrixVisualizer:
    """
    Handles matrix visualization for GNN models.

    This class provides methods to extract matrix data from GNN parameters
    and generate various visualizations including heatmaps, statistics,
    and combined overviews. Specialized support for 3D tensors like
    POMDP transition matrices (B matrix).
    """

    # Maximum matrix cell count for which text annotations are shown
    _ANNOTATION_CELL_LIMIT = 25

    def export_matrix_to_csv(
        self, matrix: np.ndarray, matrix_name: str, output_path: Path
    ) -> bool:
        """
        Export matrix data to CSV format for accessibility.

        Args:
            matrix: Numpy array to export
            matrix_name: Name of the matrix
            output_path: Output path for CSV file

        Returns:
            True if successful
        """
        try:
            csv_path = output_path.with_suffix(".csv")

            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)

                # Write header with matrix info
                writer.writerow([f"Matrix: {matrix_name}"])
                writer.writerow([f"Shape: {matrix.shape}"])
                writer.writerow([f"Data type: {matrix.dtype}"])
                writer.writerow([])  # Empty row

                # Write matrix data
                if matrix.ndim == 1:
                    # Vector - write as single row
                    writer.writerow([f"Vector element {i}" for i in range(len(matrix))])
                    writer.writerow(matrix.tolist())
                elif matrix.ndim == 2:
                    # Matrix - write with row/column headers
                    writer.writerow([f"Col {j}" for j in range(matrix.shape[1])])
                    for i, row in enumerate(matrix):
                        writer.writerow([f"Row {i}"] + row.tolist())
                elif matrix.ndim == 3:
                    transition = is_transition_tensor_name(matrix_name)
                    for index in range(matrix.shape[2]):
                        plane = (
                            f"Action slice {index}"
                            if transition
                            else f"Axis 2 slice {index}"
                        )
                        writer.writerow([plane])
                        writer.writerow(
                            ["Next \\ Previous" if transition else "Axis 0 \\ Axis 1"]
                            + [
                                f"{'Previous' if transition else 'Axis 1'} {j}"
                                for j in range(matrix.shape[1])
                            ]
                        )
                        for next_state in range(matrix.shape[0]):
                            writer.writerow(
                                [f"{'Next' if transition else 'Axis 0'} {next_state}"]
                                + matrix[next_state, :, index].tolist()
                            )
                        writer.writerow([])

            return True

        except Exception as e:
            logger.error(f"Failed to export matrix to CSV: {e}")
            return False

    def extract_matrix_data_from_parameters(
        self, parameters: Union[List[Dict], Dict]
    ) -> Dict[str, np.ndarray]:
        """Extract matrix data from parameters section."""
        return extract_matrices_from_parameter_list(parameters)

    def _convert_to_matrix(self, value: Any, name: str = "") -> Optional[np.ndarray]:
        """Convert to matrix."""
        return convert_to_matrix(value, name)

    def extract_from_parsed_gnn(
        self, parsed_data: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Extract matrix data from parsed GNN structure.

        Checks multiple locations:
        - parameters field (primary location)
        - InitialParameterization section
        - matrices field
        - variables with matrix values

        Args:
            parsed_data: Parsed GNN data dictionary

        Returns:
            Dictionary mapping matrix names to numpy arrays
        """
        matrices: dict[Any, Any] = {}

        # Primary: Extract from parameters field
        parameters = parsed_data.get("parameters", [])
        if parameters:
            param_matrices = self.extract_matrix_data_from_parameters(parameters)
            matrices.update(param_matrices)

        # Secondary: Check InitialParameterization section
        initial_params = parsed_data.get("InitialParameterization", {})
        if isinstance(initial_params, dict):
            for param_name, param_value in initial_params.items():
                matrix = self._convert_to_matrix(param_value, param_name)
                if matrix is not None:
                    matrices[param_name] = matrix

        # Tertiary: Check matrices field
        matrices_list = parsed_data.get("matrices", [])
        if matrices_list:
            for m_info in matrices_list:
                if isinstance(m_info, dict):
                    m_name = m_info.get("name", f"matrix_{len(matrices)}")
                    m_data = m_info.get("data")
                    if m_data is not None:
                        matrix = self._convert_to_matrix(m_data, m_name)
                        if matrix is not None:
                            matrices[m_name] = matrix

        return matrices

    def generate_matrix_heatmap(
        self,
        matrix_name: str,
        matrix: np.ndarray,
        output_path: Path,
        title: Optional[str] = None,
        cmap: str = "viridis",
        *,
        units: str | None = None,
        row_label: str | None = None,
        column_label: str | None = None,
    ) -> bool:
        """
        Generate a heatmap visualization for a matrix.

        Args:
            matrix_name: Name of the matrix
            matrix: Numpy array representing the matrix
            output_path: Output file path
            title: Optional title for the plot
            cmap: Colormap to use

        Returns:
            True if successful, False otherwise
        """
        try:
            # Reshape 0D scalar to 2D 1x1, or 1D vector to 2D row
            if matrix.ndim == 0:
                matrix = matrix.reshape(1, 1)
            elif matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)

            plt.figure(figsize=(10, 8))

            # Create heatmap
            im = plt.imshow(matrix, cmap=cmap, aspect="auto")

            # Add colorbar
            cbar = plt.colorbar(im)
            unit_label = (
                units.strip()
                if isinstance(units, str) and units.strip()
                else "units unspecified"
            )
            cbar.set_label(f"Value ({unit_label})", rotation=270, labelpad=15)

            # Add title
            if title is None:
                title = f"Matrix {matrix_name}"
            plt.title(title, fontsize=16, fontweight="bold")

            # Add axis labels
            plt.xlabel(column_label or "Column Index (0-based)")
            plt.ylabel(row_label or "Row Index (0-based)")

            # Add text annotations for matrix values (skip for large matrices)
            total_cells = matrix.shape[0] * matrix.shape[1]
            if total_cells <= self._ANNOTATION_CELL_LIMIT:
                for i in range(matrix.shape[0]):
                    for j in range(matrix.shape[1]):
                        value = float(matrix[i, j])
                        plt.text(
                            j,
                            i,
                            f"{value:.3f}",
                            ha="center",
                            va="center",
                            color=contrasting_text_color(im.cmap(im.norm(value))),
                            fontsize=8,
                            fontweight="bold",
                        )

            # Set axis ticks
            plt.xticks(range(matrix.shape[1]))
            plt.yticks(range(matrix.shape[0]))

            safe_tight_layout()
            # Ensure parent directory exists
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.debug("mkdir for %s: %s", output_path.parent, e)
            save_figure(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            # Export matrix data to CSV for accessibility
            self.export_matrix_to_csv(matrix, matrix_name, output_path)

            return True

        except Exception as e:
            logger.error(f"Error generating matrix heatmap for {matrix_name}: {e}")
            plt.close()
            return False

    def create_heatmap(self, matrix: List[List[float]] | np.ndarray) -> bool:
        """Create heatmap."""
        try:
            arr = np.array(matrix, dtype=float)
            # Save to project output/test_artifacts to avoid polluting repo root
            base_dir = Path.cwd() / "output" / "2_tests_output"
            base_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = base_dir / "matrix_heatmap.png"
            return self.generate_matrix_heatmap("matrix", arr, tmp_path)
        except Exception:
            return False

    def generate_3d_tensor_visualization(
        self,
        tensor_name: str,
        tensor: np.ndarray,
        output_path: Path,
        title: Optional[str] = None,
        tensor_type: str | None = None,
    ) -> bool:
        """
        Generate specialized visualization for 3D tensors like POMDP transition matrices.

        Args:
            tensor_name: Name of the tensor (e.g., 'B')
            tensor: 3D numpy array
            output_path: Output file path
            title: Optional title for the plot
            tensor_type: Explicit role; omitted role is transition only for canonical
                B/B_fN names, otherwise generic with one shared value scale.

        Returns:
            True if successful, False otherwise
        """
        try:
            if tensor.ndim != 3:
                logger.warning(
                    f"Tensor {tensor_name} is not 3D (shape: {tensor.shape})"
                )
                return False

            # Get dimensions
            dim1, dim2, dim3 = tensor.shape
            if tensor_type is None:
                tensor_type = (
                    "transition"
                    if is_transition_tensor_name(tensor_name)
                    else "generic"
                )

            # Create figure with subplots for each slice (clamped to prevent overflow)
            fig = plt.figure(figsize=_safe_figsize(5 * dim3, 8))

            # Create subplot grid
            gs = fig.add_gridspec(2, dim3, height_ratios=[3, 1], hspace=0.3, wspace=0.3)

            # Generate titles based on tensor type
            if tensor_type == "transition":
                slice_titles = [f"Action {i}" for i in range(dim3)]
                xlabel = "Previous State (0-based)"
                ylabel = "Next State (0-based)"
                main_title = f"POMDP Transition Matrix {tensor_name} (P(s'|s,u))"
            else:
                slice_titles = [f"Axis 2 slice {i} (0-based)" for i in range(dim3)]
                xlabel = "Axis 1 index (0-based)"
                ylabel = "Axis 0 index (0-based)"
                main_title = f"3D Tensor {tensor_name}"

            # Plot each slice as a heatmap
            for i in range(dim3):
                ax = fig.add_subplot(gs[0, i])

                # Extract slice
                slice_data = tensor[:, :, i]

                # Create heatmap
                probability_range = (
                    {"vmin": 0, "vmax": 1}
                    if tensor_type == "transition"
                    else {"vmin": float(np.min(tensor)), "vmax": float(np.max(tensor))}
                )
                im = ax.imshow(
                    slice_data, cmap="Blues", aspect="auto", **probability_range
                )

                # Add text annotations for small matrices
                if (
                    slice_data.size <= self._ANNOTATION_CELL_LIMIT
                ):  # Only add text for reasonably sized matrices
                    for row in range(slice_data.shape[0]):
                        for col in range(slice_data.shape[1]):
                            value = float(slice_data[row, col])
                            ax.text(
                                col,
                                row,
                                f"{value:.2f}",
                                ha="center",
                                va="center",
                                color=contrasting_text_color(im.cmap(im.norm(value))),
                                fontsize=10,
                                fontweight="bold",
                            )

                # Set labels
                ax.set_title(slice_titles[i], fontweight="bold", fontsize=12)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)

                # Set ticks
                ax.set_xticks(range(slice_data.shape[1]))
                ax.set_yticks(range(slice_data.shape[0]))

                # Add colorbar for first slice only
                if i == 0:
                    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                    cbar.set_label(
                        "Transition Probability"
                        if tensor_type == "transition"
                        else "Value (units unspecified)",
                        rotation=270,
                        labelpad=15,
                    )

            # Add summary statistics below
            ax_summary = fig.add_subplot(gs[1, :])
            ax_summary.axis("off")

            # Calculate and display statistics
            stats_text = self._generate_tensor_statistics(
                tensor, tensor_name, tensor_type
            )
            ax_summary.text(
                0.05,
                0.5,
                stats_text,
                transform=ax_summary.transAxes,
                fontsize=10,
                verticalalignment="center",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "lightgray",
                    "alpha": 0.8,
                },
            )

            # Set main title
            fig.suptitle(main_title, fontsize=16, fontweight="bold", y=0.95)

            safe_tight_layout()
            save_figure(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            return True

        except Exception as e:
            logger.error(
                f"Error generating 3D tensor visualization for {tensor_name}: {e}"
            )
            plt.close()
            return False

    def generate_threejs_tensor_explorer(
        self,
        tensor_name: str,
        tensor: np.ndarray,
        output_path: Path,
        title: Optional[str] = None,
    ) -> bool:
        """Generate an interactive Three.js/WebGL explorer for a 3D tensor."""
        try:
            if tensor.ndim != 3:
                logger.warning(
                    "Tensor %s is not 3D (shape: %s)", tensor_name, tensor.shape
                )
                return False
            output_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "name": tensor_name,
                "shape": list(tensor.shape),
                "values": tensor.tolist(),
            }
            fallback_path = output_path.with_suffix(".json")
            fallback_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            page_title = title or f"Three.js Tensor Explorer: {tensor_name}"
            html = _threejs_tensor_html(page_title, payload, fallback_path.name)
            output_path.write_text(html, encoding="utf-8")
            return True
        except Exception as e:
            logger.error(
                "Error generating Three.js tensor explorer for %s: %s", tensor_name, e
            )
            return False

    def _generate_tensor_statistics(
        self, tensor: np.ndarray, tensor_name: str, tensor_type: str
    ) -> str:
        """
        Generate statistical summary for a 3D tensor.

        Args:
            tensor: 3D numpy array
            tensor_name: Name of the tensor
            tensor_type: Type of tensor

        Returns:
            Formatted statistics string
        """
        dim1, dim2, dim3 = tensor.shape

        # Basic statistics
        mean_val = np.mean(tensor)
        std_val = np.std(tensor)
        min_val = np.min(tensor)
        max_val = np.max(tensor)

        # Transition-specific statistics
        if tensor_type == "transition":
            # POMDP transition tensors use (next_state, previous_state, action).
            # Each previous-state/action column must sum over next_state.
            column_sums = np.sum(tensor, axis=0)
            valid_transitions = (
                np.isfinite(tensor).all()
                and np.all(tensor >= 0)
                and np.all(tensor <= 1)
                and np.allclose(column_sums, 1.0, atol=1e-6)
            )

            # Calculate entropy of transitions
            if valid_transitions:
                # The exact limit 0*log(0)=0 preserves deterministic entropy.
                log_probs = np.log(np.where(tensor > 0, tensor, 1.0))
                entropy = -np.sum(tensor * log_probs, axis=0) + 0.0
                entropy_label = f"{np.mean(entropy):.3f} nats"
            else:
                entropy_label = "unavailable: not normalized nonnegative probabilities"

            stats = f"""Tensor {tensor_name} Statistics:
Shape: {dim1}×{dim2}×{dim3} (Next×Previous×Actions)
Mean: {mean_val:.3f}, Std: {std_val:.3f}
Range: [{min_val:.3f}, {max_val:.3f}]
Valid Transition Matrices: {"✓" if valid_transitions else "✗"}
Mean Transition Entropy: {entropy_label}"""
        else:
            stats = f"""Tensor {tensor_name} Statistics:
Shape: {dim1}×{dim2}×{dim3}
Mean: {mean_val:.3f}, Std: {std_val:.3f}
Range: [{min_val:.3f}, {max_val:.3f}]"""

        return stats

    def generate_pomdp_transition_analysis(
        self, tensor: np.ndarray, output_path: Path
    ) -> bool:
        """Render canonical next-state/previous-state/action tensor diagnostics."""
        from .transition_analysis import generate_pomdp_transition_analysis

        return generate_pomdp_transition_analysis(tensor, output_path, logger=logger)

    def generate_matrix_analysis(
        self,
        parameters: List[Dict] | List[List[float]],
        output_path: Path | None = None,
    ) -> bool:
        """
        Generate comprehensive matrix analysis from parameters.

        Args:
            parameters: List of parameter dictionaries from GNN
            output_path: Path where to save the analysis image

        Returns:
            bool: True if analysis was generated successfully
        """
        sns = get_sns()
        if output_path is None:
            base_dir = Path.cwd() / "output" / "2_tests_output"
            base_dir.mkdir(parents=True, exist_ok=True)
            output_path = base_dir / "matrix_analysis.png"

        # Convenience: if called with raw matrix-like and no output_path
        if (
            isinstance(parameters, list)
            and parameters
            and isinstance(parameters[0], list)
        ):
            try:
                arr = np.array(parameters, dtype=float)
                return self.generate_matrix_heatmap("matrix", arr, output_path)
            except Exception:
                return False

        if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
            # Create a simple text report instead
            try:
                parameter_records = cast(List[Dict], parameters)
                with open(output_path.with_suffix(".txt"), "w") as f:
                    f.write("Matrix Analysis Report\n")
                    f.write("=====================\n\n")
                    f.write("Dependencies Status:\n")
                    f.write(f"- Matplotlib: {MATPLOTLIB_AVAILABLE}\n")
                    f.write(f"- NumPy: {NUMPY_AVAILABLE}\n")
                    f.write(f"- Seaborn: {SEABORN_AVAILABLE}\n\n")
                    f.write(f"Parameters found: {len(parameter_records)}\n")
                    for i, param in enumerate(parameter_records[:10]):  # Show first 10
                        f.write(
                            f"  {i + 1}. {param.get('name', 'unnamed')}: {param.get('type', 'unknown')}\n"
                        )
                    if len(parameter_records) > 10:
                        f.write(f"  ... and {len(parameter_records) - 10} more\n")
                return True
            except Exception:
                return False

        try:
            # Extract matrix data from parameters
            matrices = (
                self.extract_matrix_data_from_parameters(
                    cast(list[dict[Any, Any]], parameters)
                )
                if isinstance(parameters, list)
                and parameters
                and isinstance(parameters[0], dict)
                else {}
            )

            if not matrices:
                return False

            # Create figure with subplots
            n_matrices = len(matrices)
            if n_matrices == 0:
                return False

            # Calculate grid dimensions
            cols = min(3, n_matrices)
            rows = (n_matrices + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=_safe_figsize(15, 5 * rows))
            if n_matrices == 1:
                axes = [axes]
            elif rows == 1:
                axes = [axes] if not hasattr(axes, "__len__") else axes
            else:
                axes = axes.flatten()

            # Generate visualizations for each matrix
            for i, (name, matrix) in enumerate(matrices.items()):
                if i >= len(axes):
                    break

                ax = axes[i]

                # Handle different matrix shapes
                if matrix.ndim == 1:
                    # Vector - plot as bar chart
                    ax.bar(range(len(matrix)), matrix)
                    ax.set_title(f"{name} (Vector)")
                elif matrix.ndim == 2:
                    # Matrix - plot as heatmap
                    if SEABORN_AVAILABLE:
                        sns.heatmap(
                            matrix,
                            ax=ax,
                            cmap="viridis",
                            annot=True if matrix.size <= 100 else False,
                        )
                    else:
                        im = ax.imshow(matrix, cmap="viridis", aspect="auto")
                        plt.colorbar(im, ax=ax)
                    ax.set_title(f"{name} (Matrix {matrix.shape})")
                elif matrix.ndim == 3:
                    # Only the canonical named GNN transition tensor declares
                    # an action axis. Other tensors retain their axis-0 slice.
                    is_transition = is_transition_tensor_name(name)
                    first_plane = matrix[:, :, 0] if is_transition else matrix[0]
                    if SEABORN_AVAILABLE:
                        sns.heatmap(
                            first_plane,
                            ax=ax,
                            cmap="viridis",
                            annot=True if first_plane.size <= 100 else False,
                        )
                    else:
                        im = ax.imshow(first_plane, cmap="viridis", aspect="auto")
                        plt.colorbar(im, ax=ax)
                    plane_label = "action 0" if is_transition else "axis 0 slice 0"
                    ax.set_title(f"{name} (3D Tensor {matrix.shape}, {plane_label})")

                # Add statistics text
                stats_text = f"Mean: {np.mean(matrix):.3f}\nStd: {np.std(matrix):.3f}"
                ax.text(
                    0.02,
                    0.98,
                    stats_text,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
                )

            # Hide unused subplots
            for i in range(n_matrices, len(axes)):
                axes[i].set_visible(False)

            safe_tight_layout()
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.debug("mkdir for %s: %s", output_path.parent, e)
            save_figure(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            # Export CSV data for each matrix
            csv_exports: list[Any] = []
            for index, (name, matrix) in enumerate(matrices.items()):
                # Index disambiguates labels which sanitize to the same filename.
                label = "".join(
                    c if c.isascii() and (c.isalnum() or c in "_-") else "_"
                    for c in name
                )[:80]
                stem = output_path.stem
                if len(stem.encode("utf-8")) > 100:
                    stem = (
                        stem.encode("utf-8")[:100].decode("utf-8", errors="ignore")
                        + "_"
                        + hashlib.sha256(output_path.stem.encode()).hexdigest()[:8]
                    )
                csv_path = output_path.with_name(
                    f"{stem}_matrix_{index}_{label or 'unnamed'}.csv"
                )
                csv_success = self.export_matrix_to_csv(matrix, name, csv_path)
                if csv_success:
                    csv_exports.append(csv_path)

            return True

        except Exception as e:
            # Recovery: create error report
            try:
                with open(output_path.with_suffix(".txt"), "w") as f:
                    f.write("Matrix Analysis Failed\n")
                    f.write(f"Error: {str(e)}\n")
                    f.write(f"Parameters: {len(parameters)} found\n")
                return True
            except Exception:
                return False

    def generate_combined_matrix_overview(
        self, matrices: Dict[str, np.ndarray], output_path: Path
    ) -> bool:
        """
        Generate a combined overview of all matrices.

        Args:
            matrices: Dictionary of matrix name to numpy array mappings
            output_path: Output file path

        Returns:
            True if successful, False otherwise
        """
        try:
            n_matrices = len(matrices)
            if n_matrices == 0:
                return True

            # Separate 2D and 3D matrices
            matrices_2d = {
                name: matrix for name, matrix in matrices.items() if matrix.ndim == 2
            }
            matrices_3d = {
                name: matrix for name, matrix in matrices.items() if matrix.ndim == 3
            }

            # Calculate layout
            total_plots = len(matrices_2d) + len(matrices_3d)
            if total_plots == 0:
                return True

            cols = min(3, total_plots)
            rows = (total_plots + cols - 1) // cols

            fig, axes = plt.subplots(
                rows, cols, figsize=_safe_figsize(5 * cols, 4 * rows)
            )
            if total_plots == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes.reshape(1, -1)

            # Flatten axes for easier indexing
            axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

            plot_idx = 0

            # Plot 2D matrices
            for matrix_name, matrix in matrices_2d.items():
                if plot_idx >= len(axes_flat):
                    break

                ax = axes_flat[plot_idx]

                # Create heatmap
                image = ax.imshow(matrix, cmap="viridis", aspect="auto")

                # Add title
                ax.set_title(f"Matrix {matrix_name}", fontweight="bold")

                # Add text annotations for small matrices
                if (
                    matrix.size <= self._ANNOTATION_CELL_LIMIT
                ):  # Only add text for reasonably sized matrices
                    for row in range(matrix.shape[0]):
                        for col in range(matrix.shape[1]):
                            value = float(matrix[row, col])
                            ax.text(
                                col,
                                row,
                                f"{value:.2f}",
                                ha="center",
                                va="center",
                                color=contrasting_text_color(
                                    image.cmap(image.norm(value))
                                ),
                                fontsize=8,
                            )

                # Set axis labels
                ax.set_xlabel("Column")
                ax.set_ylabel("Row")

                plot_idx += 1

            # Plot 3D matrices (show first slice)
            for matrix_name, matrix in matrices_3d.items():
                if plot_idx >= len(axes_flat):
                    break

                ax = axes_flat[plot_idx]

                # Show first slice of 3D tensor
                slice_data = matrix[:, :, 0]
                image = ax.imshow(slice_data, cmap="Blues", aspect="auto")

                # Add title
                ax.set_title(f"Tensor {matrix_name} (Slice 0)", fontweight="bold")

                # Add text annotations for small matrices
                if slice_data.size <= self._ANNOTATION_CELL_LIMIT:
                    for row in range(slice_data.shape[0]):
                        for col in range(slice_data.shape[1]):
                            value = float(slice_data[row, col])
                            ax.text(
                                col,
                                row,
                                f"{value:.2f}",
                                ha="center",
                                va="center",
                                color=contrasting_text_color(
                                    image.cmap(image.norm(value))
                                ),
                                fontsize=8,
                            )

                # Set axis labels
                ax.set_xlabel("Column")
                ax.set_ylabel("Row")

                plot_idx += 1

            # Hide unused subplots
            for i in range(plot_idx, len(axes_flat)):
                axes_flat[i].set_visible(False)

            safe_tight_layout()
            save_figure(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            return True

        except Exception as e:
            logger.error(f"Error generating combined matrix overview: {e}")
            plt.close()
            return False

    def generate_matrix_statistics(
        self, parameters: List[Dict], output_path: Path
    ) -> bool:
        """
        Generate statistics about matrices in the model.

        Args:
            parameters: List of parameter dictionaries
            output_path: Output file path

        Returns:
            True if successful, False otherwise
        """
        try:
            matrices = self.extract_matrix_data_from_parameters(parameters)

            if not matrices:
                # Render an explicit empty-data panel when no matrices are present.
                plt.figure(figsize=(10, 6))
                plt.text(
                    0.5,
                    0.5,
                    "No matrix data found",
                    ha="center",
                    va="center",
                    transform=plt.gca().transAxes,
                    fontsize=16,
                    fontweight="bold",
                )
                plt.title("Matrix Statistics", fontsize=16, fontweight="bold")
                save_figure(output_path, dpi=300, bbox_inches="tight")
                plt.close()
                return True

            # Calculate statistics for each matrix
            matrix_stats: dict[Any, Any] = {}
            for matrix_name, matrix in matrices.items():
                matrix_stats[matrix_name] = {
                    "shape": matrix.shape,
                    "size": matrix.size,
                    "mean": np.mean(matrix),
                    "std": np.std(matrix),
                    "min": np.min(matrix),
                    "max": np.max(matrix),
                    "sum": np.sum(matrix),
                    "dimensions": matrix.ndim,
                }

                # Special statistics for 3D tensors
                if matrix.ndim == 3:
                    # Calculate entropy for transition matrices
                    epsilon = 1e-10
                    log_probs = np.log(matrix + epsilon)
                    entropy = -np.sum(matrix * log_probs, axis=1)
                    matrix_stats[matrix_name]["mean_entropy"] = np.mean(entropy)
                    matrix_stats[matrix_name]["max_entropy"] = np.max(entropy)

                    # Calculate determinism (max probability per row)
                    max_probs = np.max(matrix, axis=1)
                    matrix_stats[matrix_name]["mean_determinism"] = np.mean(max_probs)
                    matrix_stats[matrix_name]["min_determinism"] = np.min(max_probs)

            # Create statistics visualization
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

            # Matrix sizes
            names = list(matrix_stats.keys())
            sizes = [stats["size"] for stats in matrix_stats.values()]
            ax1.bar(names, sizes, color="skyblue", alpha=0.7)
            ax1.set_title("Matrix Sizes", fontweight="bold")
            ax1.set_ylabel("Number of Elements")

            # Matrix means
            means = [stats["mean"] for stats in matrix_stats.values()]
            ax2.bar(names, means, color="lightcoral", alpha=0.7)
            ax2.set_title("Matrix Means", fontweight="bold")
            ax2.set_ylabel("Mean Value")

            # Matrix ranges (min to max)
            mins = [stats["min"] for stats in matrix_stats.values()]
            maxs = [stats["max"] for stats in matrix_stats.values()]
            ax3.bar(names, maxs, color="lightgreen", alpha=0.7, label="Max")
            ax3.bar(names, mins, color="lightyellow", alpha=0.7, label="Min")
            ax3.set_title("Matrix Value Ranges", fontweight="bold")
            ax3.set_ylabel("Value")
            ax3.legend()

            # Matrix shapes and dimensions
            shapes = [str(stats["shape"]) for stats in matrix_stats.values()]
            dimensions = [stats["dimensions"] for stats in matrix_stats.values()]

            # Color by dimension
            colors = [
                "lightsteelblue" if dim == 2 else "gold" if dim == 3 else "lightpink"
                for dim in dimensions
            ]

            ax4.bar(names, [1] * len(names), color=colors, alpha=0.7)
            ax4.set_title("Matrix Shapes and Dimensions", fontweight="bold")
            ax4.set_ylabel("Count")

            # Add shape labels
            for i, shape in enumerate(shapes):
                ax4.text(i, 0.5, shape, ha="center", va="center", fontweight="bold")

            safe_tight_layout()
            save_figure(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            return True

        except Exception as e:
            logger.error(f"Error generating matrix statistics: {e}")
            plt.close()
            return False

    def visualize_directory(self, input_dir: Path, output_dir: Path) -> List[str]:
        """
        Visualize matrices from all GNN files in a directory.

        Args:
            input_dir: Input directory containing GNN files
            output_dir: Output directory for visualizations

        Returns:
            List of generated visualization file paths
        """
        generated_files: list[Any] = []

        try:
            # Create output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)

            # Find all GNN files in the directory
            gnn_files = list(input_dir.glob("*.md"))

            for gnn_file in gnn_files:
                try:
                    # Create subdirectory for this file's visualizations
                    file_output_dir = output_dir / gnn_file.stem
                    file_output_dir.mkdir(parents=True, exist_ok=True)

                    # Read and parse the GNN file
                    with open(gnn_file, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Parse GNN content to extract parameters
                    parsed_data = self._parse_gnn_content_for_parameters(content)

                    if parsed_data.get("parameters"):
                        # Generate matrix visualizations for this file
                        file_visualizations = self.generate_matrix_analysis(
                            parsed_data["parameters"],
                            file_output_dir / "matrix_analysis.png",
                        )

                        if file_visualizations:
                            generated_files.append(
                                str(file_output_dir / "matrix_analysis.png")
                            )

                        # Generate matrix statistics
                        stats_visualizations = self.generate_matrix_statistics(
                            parsed_data["parameters"],
                            file_output_dir / "matrix_statistics.png",
                        )

                        if stats_visualizations:
                            generated_files.append(
                                str(file_output_dir / "matrix_statistics.png")
                            )

                except Exception as e:
                    logger.error(f"Error processing {gnn_file}: {e}")
                    continue

            return generated_files

        except Exception as e:
            logger.error(f"Error processing directory {input_dir}: {e}")
            return generated_files

    def _parse_gnn_content_for_parameters(self, content: str) -> Dict[str, Any]:
        """
        Parse GNN content to extract parameters for matrix visualization.

        Args:
            content: GNN file content

        Returns:
            Dictionary with parsed parameters
        """
        import re

        parsed_data: dict[str, Any] = {"parameters": []}

        # Extract initial parameterization section
        init_match = re.search(
            r"## InitialParameterization\n(.*?)(?=\n##|\Z)", content, re.DOTALL
        )
        if init_match:
            init_content = init_match.group(1)

            # Parse matrix definitions
            matrix_pattern = r"([A-Z])\s*=\s*\{([^}]+)\}"
            for match in re.finditer(matrix_pattern, init_content):
                matrix_name = match.group(1)
                matrix_data = match.group(2)

                try:
                    # Convert matrix data to list format
                    matrix_list = self._parse_matrix_string(matrix_data)
                    parsed_data["parameters"].append(
                        {"name": matrix_name, "value": matrix_list}
                    )
                except Exception as e:
                    logger.debug(
                        f"Skipping matrix {matrix_name} due to parse error: {e}"
                    )
                    continue

        return parsed_data

    def generate_single_matrix_heatmap(
        self, matrix_data: np.ndarray, matrix_name: str, output_path: Path
    ) -> bool:
        """
        Generate a single matrix heatmap with enhanced styling.

        Args:
            matrix_data: Matrix data as numpy array
            matrix_name: Name of the matrix for labeling
            output_path: Path to save the visualization

        Returns:
            True if successful, False otherwise
        """
        if not self._check_dependencies():
            return False

        try:
            plt.figure(figsize=(10, 8))

            # Sample large matrices for better visualization
            display_matrix = self._sample_matrix_for_display(matrix_data)

            sns = get_sns()
            if SEABORN_AVAILABLE and display_matrix.size <= 100:
                sns.heatmap(
                    display_matrix,
                    annot=True,
                    cmap="viridis",
                    fmt=".2f",
                    cbar_kws={"shrink": 0.8},
                    square=True,
                )
            else:
                im = plt.imshow(display_matrix, cmap="viridis", aspect="auto")
                plt.colorbar(im, fraction=0.046, pad=0.04, shrink=0.8)

            plt.title(f"Matrix: {matrix_name}", fontsize=14, fontweight="bold")
            plt.xlabel("Columns", fontsize=12)
            plt.ylabel("Rows", fontsize=12)
            safe_tight_layout()
            save_figure(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            return True

        except Exception as e:
            logger.error(
                f"Error generating single matrix heatmap for {matrix_name}: {e}"
            )
            return False

    def generate_matrix_correlation_plot(
        self, matrix_data: np.ndarray, matrix_name: str, output_path: Path
    ) -> bool:
        """
        Generate correlation matrix visualization.

        Args:
            matrix_data: Matrix data as numpy array
            matrix_name: Name of the matrix for labeling
            output_path: Path to save the visualization

        Returns:
            True if successful, False otherwise
        """
        if not self._check_dependencies():
            return False

        try:
            # Calculate correlation matrix (suppress warnings from constant columns)
            if matrix_data.shape[0] > 1 and matrix_data.shape[1] > 1:
                with np.errstate(divide="ignore", invalid="ignore"):
                    corr_matrix = np.corrcoef(
                        matrix_data.T
                    )  # Transpose for proper correlation
                corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            else:
                corr_matrix = matrix_data

            plt.figure(figsize=(10, 8))

            sns = get_sns()
            if SEABORN_AVAILABLE and corr_matrix.size <= 100:
                sns.heatmap(
                    corr_matrix,
                    annot=True,
                    cmap="coolwarm",
                    fmt=".2f",
                    cbar_kws={"shrink": 0.8},
                    square=True,
                    center=0,
                )
            else:
                im = plt.imshow(corr_matrix, cmap="coolwarm", aspect="auto")
                plt.colorbar(im, fraction=0.046, pad=0.04, shrink=0.8)

            plt.title(
                f"Correlation Matrix: {matrix_name}", fontsize=14, fontweight="bold"
            )
            plt.xlabel("Variables", fontsize=12)
            plt.ylabel("Variables", fontsize=12)
            safe_tight_layout()
            save_figure(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            return True

        except Exception as e:
            logger.error(f"Error generating correlation plot for {matrix_name}: {e}")
            return False

    def generate_matrix_histogram(
        self, matrix_data: np.ndarray, matrix_name: str, output_path: Path
    ) -> bool:
        """
        Generate histogram of matrix values.

        Args:
            matrix_data: Matrix data as numpy array
            matrix_name: Name of the matrix for labeling
            output_path: Path to save the visualization

        Returns:
            True if successful, False otherwise
        """
        if not self._check_dependencies():
            return False

        try:
            # Flatten matrix for histogram
            flat_data = matrix_data.flatten()

            plt.figure(figsize=(10, 6))
            plt.hist(flat_data, bins=50, alpha=0.7, edgecolor="black", density=True)
            plt.title(
                f"Value Distribution: {matrix_name}", fontsize=14, fontweight="bold"
            )
            plt.xlabel("Value", fontsize=12)
            plt.ylabel("Density", fontsize=12)
            plt.grid(True, alpha=0.3)
            safe_tight_layout()
            save_figure(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            return True

        except Exception as e:
            logger.error(f"Error generating histogram for {matrix_name}: {e}")
            return False

    def generate_matrix_composed_view(
        self, matrices: Dict[str, np.ndarray], output_path: Path
    ) -> bool:
        """
        Generate a composed view of multiple matrices.

        Args:
            matrices: Dictionary of matrix names to matrix data
            output_path: Path to save the visualization

        Returns:
            True if successful, False otherwise
        """
        if not self._check_dependencies():
            return False

        try:
            num_matrices = len(matrices)
            if num_matrices == 0:
                return False

            # Determine grid layout
            cols = min(3, num_matrices)
            rows = (num_matrices + cols - 1) // cols

            fig, axes = plt.subplots(
                rows, cols, figsize=_safe_figsize(5 * cols, 4 * rows)
            )

            if rows == 1 and cols == 1:
                axes = [axes]
            elif rows == 1 or cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()

            for i, (name, matrix) in enumerate(matrices.items()):
                if i >= len(axes):
                    break

                ax = axes[i]

                # Sample large matrices
                display_matrix = self._sample_matrix_for_display(matrix)

                sns = get_sns()
                if SEABORN_AVAILABLE and display_matrix.size <= 100:
                    sns.heatmap(
                        display_matrix, ax=ax, cmap="viridis", cbar=False, square=True
                    )
                else:
                    ax.imshow(display_matrix, cmap="viridis", aspect="auto")

                ax.set_title(
                    f"{name}\n({matrix.shape[0]}×{matrix.shape[1]})", fontsize=10
                )

            # Hide unused subplots
            for i in range(num_matrices, len(axes)):
                axes[i].set_visible(False)

            plt.suptitle("Matrix Overview", fontsize=16, fontweight="bold")
            safe_tight_layout()
            save_figure(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            return True

        except Exception as e:
            logger.error(f"Error generating composed matrix view: {e}")
            return False

    def _sample_matrix_for_display(
        self, matrix: np.ndarray, max_size: int = 20
    ) -> np.ndarray:
        """
        Sample a matrix for display purposes to avoid performance issues with large matrices.

        Args:
            matrix: Input matrix
            max_size: Maximum dimension size for display

        Returns:
            Sampled matrix suitable for visualization
        """
        if matrix.size <= max_size * max_size:
            return matrix

        # Sample rows and columns
        if len(matrix.shape) == 2:
            row_step = max(1, matrix.shape[0] // max_size)
            col_step = max(1, matrix.shape[1] // max_size)
            return matrix[::row_step, ::col_step]
        elif len(matrix.shape) == 3:
            # For 3D tensors, sample each dimension
            steps = [max(1, dim // max_size) for dim in matrix.shape]
            return matrix[:: steps[0], :: steps[1], :: steps[2]]

        return matrix

    def _check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        return MATPLOTLIB_AVAILABLE and NUMPY_AVAILABLE

    def _parse_matrix_string(self, matrix_str: str) -> List[List[float]]:
        """
        Parse matrix string into list format.

        Args:
            matrix_str: Matrix data as string

        Returns:
            List representation of matrix
        """
        import re

        from gnn.utils.runtime_safety.safe_eval import (
            MATRIX_MAX_LEN,
            safe_literal_eval,
        )

        # Remove extra whitespace and newlines
        matrix_str = re.sub(r"\s+", " ", matrix_str.strip())

        # Parse nested tuples
        matrix_str = matrix_str.replace("(", "[").replace(")", "]")

        # Convert to Python list structure
        matrix_str = matrix_str.replace("[", "[").replace("]", "]")

        # Evaluate as Python expression
        matrix_data = safe_literal_eval(matrix_str, max_len=MATRIX_MAX_LEN)

        return cast("list[list[float]]", matrix_data)


def generate_matrix_visualizations(
    parsed_data: Dict[str, Any], output_dir: Path, model_name: str
) -> List[str]:
    """
    Generate matrix visualizations for a parsed GNN model.

    Args:
        parsed_data: Parsed GNN model data
        output_dir: Output directory for visualizations
        model_name: Name of the model

    Returns:
        List of generated visualization file paths
    """
    visualizer = MatrixVisualizer()
    generated_files: list[Any] = []

    # Create model-specific output directory
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Extract parameters from parsed data
    parameters = parsed_data.get("parameters", [])

    # Generate matrix analysis
    matrix_analysis_path = model_output_dir / "matrix_analysis.png"
    if visualizer.generate_matrix_analysis(parameters, matrix_analysis_path):
        generated_files.append(str(matrix_analysis_path))

    # Generate matrix statistics
    matrix_stats_path = model_output_dir / "matrix_statistics.png"
    if visualizer.generate_matrix_statistics(parameters, matrix_stats_path):
        generated_files.append(str(matrix_stats_path))

    # Generate specialized POMDP transition analysis if B matrix is present
    matrices = visualizer.extract_matrix_data_from_parameters(parameters)
    if "B" in matrices and matrices["B"].ndim == 3:
        pomdp_analysis_path = model_output_dir / "pomdp_transition_analysis.png"
        if visualizer.generate_pomdp_transition_analysis(
            matrices["B"], pomdp_analysis_path
        ):
            generated_files.append(str(pomdp_analysis_path))

    return generated_files


def process_matrix_visualization(
    parameters: List[Dict], output_path: Path, **kwargs: Any
) -> bool:
    """
    Process matrix visualization using the MatrixVisualizer class.

    This function provides a standalone interface for matrix visualization
    that can be called from other modules.

    Args:
        parameters: List of parameter dictionaries from GNN
        output_path: Path where to save the visualization
        **kwargs: Additional keyword arguments

    Returns:
        bool: True if visualization was successful
    """
    try:
        visualizer = MatrixVisualizer()
        return visualizer.generate_matrix_analysis(parameters, output_path)
    except Exception:
        return False

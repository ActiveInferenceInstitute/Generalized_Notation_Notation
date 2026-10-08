"""POMDP transition tensor diagnostics and figure assembly.

The public MatrixVisualizer method delegates here; axes remain canonical
(next_state, previous_state, action), including stochasticity and entropy checks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..compat.viz_compat import np, plt
from ..plotting.utils import contrasting_text_color, save_figure


def generate_pomdp_transition_analysis(
    tensor: np.ndarray, output_path: Path, *, logger: logging.Logger
) -> bool:
    """
    Generate specialized analysis for POMDP transition matrices.

    Args:
        tensor: 3D numpy array representing transition matrix
        output_path: Output file path

    Returns:
        True if successful, False otherwise
    """
    try:
        if tensor.ndim != 3:
            logger.warning(f"Expected 3D tensor, got shape: {tensor.shape}")
            return False

        dim1, dim2, dim3 = tensor.shape

        # Create comprehensive analysis figure with safe dimensions
        try:
            # Ensure sane figure dimensions and clear any stale figures
            plt.close("all")

            # Use very conservative figure size to avoid dimension overflow
            # Keep it small to prevent pixel calculation overflow
            safe_figsize = (12, 9)  # Safe, tested dimensions
            fig = plt.figure(figsize=safe_figsize, clear=True)

            # Set DPI early and ensure it's safe
            safe_dpi = 96  # Use a very safe, low DPI to prevent overflow
            fig.set_dpi(safe_dpi)
        except Exception:
            logger.exception(
                "Could not create POMDP transition analysis figure for %s "
                "(tensor shape %s)",
                output_path,
                tensor.shape,
            )
            plt.close("all")
            return False

        # Use a simpler approach without gridspec

        # 1. Main transition matrices (top row). The 3x3 grid has three top-row
        # cells; further actions would overlap the rows below (or raise past 9).
        shown_actions = min(dim3, 3)
        if dim3 > shown_actions:
            logger.info(
                f"POMDP transition analysis: showing {shown_actions} of "
                f"{dim3} action matrices"
            )
        for i in range(shown_actions):
            ax = fig.add_subplot(3, 3, i + 1)

            slice_data = tensor[:, :, i]
            im = ax.imshow(slice_data, cmap="Blues", aspect="auto", vmin=0, vmax=1)

            # Add text annotations
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

            ax.set_title(f"Action {i} Transition Matrix", fontweight="bold")
            ax.set_xlabel("Previous State (0-based)")
            ax.set_ylabel("Next State (0-based)")
            ax.set_xticks(range(dim2))
            ax.set_yticks(range(dim1))

            if i == 0:
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label("P(s'|s,u)", rotation=270, labelpad=15)

        # 2. Transition entropy analysis (middle row)
        ax_entropy = fig.add_subplot(3, 3, 4)

        # Calculate entropy for each action
        # 0·log(0) := 0 exactly; an epsilon inside the log makes deterministic
        # (p=1) transitions come out as tiny negative entropies.
        log_probs = np.log(np.where(tensor > 0, tensor, 1.0))
        entropy = -np.sum(
            tensor * log_probs, axis=0
        )  # Entropy per previous-state/action pair
        mean_entropy_per_action = np.mean(entropy, axis=0)  # Average entropy per action

        actions = range(dim3)
        entropy_colors = (["skyblue", "lightcoral", "lightgreen"] * ((dim3 // 3) + 1))[
            :dim3
        ]
        bars = ax_entropy.bar(
            actions, mean_entropy_per_action, color=entropy_colors, alpha=0.7
        )

        # Add value labels on bars
        for bar, value in zip(bars, mean_entropy_per_action):
            # Offset in points, not data units, so a near-zero y-range cannot
            # push the label (and the tight bbox) billions of pixels away.
            ax_entropy.annotate(
                f"{value:.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax_entropy.set_title("Transition Entropy by Action", fontweight="bold")
        ax_entropy.set_xlabel("Action")
        ax_entropy.set_ylabel("Mean Entropy (nats)")
        ax_entropy.set_xticks(actions)
        ax_entropy.margins(y=0.18)
        ax_entropy.grid(True, alpha=0.3)

        # 3. Determinism analysis (bottom left)
        ax_determinism = fig.add_subplot(3, 3, 7)

        # Calculate determinism (max next-state probability per previous-state/action).
        max_probs = np.max(tensor, axis=0)
        mean_determinism_per_action = np.mean(
            max_probs, axis=0
        )  # Average determinism per action

        determinism_colors = (["gold", "orange", "red"] * ((dim3 // 3) + 1))[:dim3]
        bars = ax_determinism.bar(
            actions,
            mean_determinism_per_action,
            color=determinism_colors,
            alpha=0.7,
        )

        for bar, value in zip(bars, mean_determinism_per_action):
            # Offset in points, not data units, so a near-zero y-range cannot
            # push the label (and the tight bbox) billions of pixels away.
            ax_determinism.annotate(
                f"{value:.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax_determinism.set_title("Transition Determinism by Action", fontweight="bold")
        ax_determinism.set_xlabel("Action")
        ax_determinism.set_ylabel("Mean Max Probability")
        ax_determinism.set_xticks(actions)
        ax_determinism.margins(y=0.18)
        ax_determinism.grid(True, alpha=0.3)

        # 4. State reachability (bottom middle)
        ax_reachability = fig.add_subplot(3, 3, 8)

        # Calculate reachability (how many states can be reached from each state)
        reachability = np.sum(tensor > 0.01, axis=0)  # Count reachable next states
        mean_reachability_per_action = np.mean(reachability, axis=0)

        reachability_colors = (
            ["lightblue", "lightgreen", "lightyellow"] * ((dim3 // 3) + 1)
        )[:dim3]
        bars = ax_reachability.bar(
            actions,
            mean_reachability_per_action,
            color=reachability_colors,
            alpha=0.7,
        )

        for bar, value in zip(bars, mean_reachability_per_action):
            # Offset in points, not data units, so a near-zero y-range cannot
            # push the label (and the tight bbox) billions of pixels away.
            ax_reachability.annotate(
                f"{value:.1f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax_reachability.set_title("State Reachability by Action", fontweight="bold")
        ax_reachability.set_xlabel("Action")
        ax_reachability.set_ylabel("Mean Reachable States")
        ax_reachability.set_xticks(actions)
        ax_reachability.margins(y=0.18)
        ax_reachability.grid(True, alpha=0.3)

        # 5. Matrix validation (bottom right)
        ax_validation = fig.add_subplot(3, 3, 9)
        ax_validation.axis("off")

        # Validation checks
        column_sums = np.sum(tensor, axis=0)
        valid_transitions = np.allclose(column_sums, 1.0, atol=1e-6)
        max_deviation = np.max(np.abs(column_sums - 1.0))

        validation_text = f"""POMDP Transition Matrix Validation:
✓ Shape: {dim1}×{dim2}×{dim3}
✓ Valid Transition Matrices: {"Yes" if valid_transitions else "No"}
✓ Max Column Sum Deviation: {max_deviation:.6f}
✓ Probability Range: [{np.min(tensor):.3f}, {np.max(tensor):.3f}]
✓ Mean Entropy: {np.mean(entropy):.3f} nats
✓ Mean Determinism: {np.mean(max_probs):.3f}"""

        ax_validation.text(
            0.05,
            0.5,
            validation_text,
            transform=ax_validation.transAxes,
            fontsize=10,
            verticalalignment="center",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "lightgreen",
                "alpha": 0.8,
            },
        )

        # Set main title
        fig.suptitle(
            "POMDP Transition Matrix Analysis",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # Adjust layout manually instead of using tight_layout
        fig.subplots_adjust(
            top=0.87, bottom=0.08, left=0.08, right=0.95, hspace=0.55, wspace=0.3
        )

        save_attempts: list[tuple[tuple[int, int] | None, dict[str, Any]]] = [
            (None, {"dpi": 96, "bbox_inches": "tight"}),
            ((8, 6), {"dpi": 72}),
            ((6, 4), {}),
        ]
        last_error: Exception | None = None
        for size, kwargs in save_attempts:
            try:
                if size:
                    fig.set_size_inches(*size)
                save_figure(output_path, **kwargs)
                break
            # TypeError: an oversized Agg canvas raises it from the
            # RendererAgg constructor, so it must reach the smaller fallbacks.
            except (RuntimeError, OSError, ValueError, TypeError) as save_error:
                last_error = save_error
                continue
        else:
            logger.error(
                f"Error saving POMDP transition analysis to {output_path}: {last_error}"
            )
            plt.close()
            return False
        plt.close()
        return True

    except Exception as e:
        logger.error(f"Error generating POMDP transition analysis: {e}")
        plt.close()
        return False

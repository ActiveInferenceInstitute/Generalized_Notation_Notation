#!/usr/bin/env python3
"""
Type Checker Visualization Module for GNN Pipeline
Generates visual abstracts, mosaics, and distributions of typing statuses across the repository models.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, cast

logger = logging.getLogger(__name__)


class _PieContainer(Protocol):
    """Runtime shape of Matplotlib 3.11's pie result, independent of type definitions."""

    wedges: Any
    texts: Any


def _pie_parts(result: object) -> tuple[Any, Any, Any]:
    """Adapt both supported Matplotlib API shapes at one typed boundary."""
    if hasattr(result, "wedges"):
        container = cast(_PieContainer, result)
        text_groups = list(container.texts)
        return (
            container.wedges,
            text_groups[0] if text_groups else [],
            text_groups[1] if len(text_groups) > 1 else [],
        )
    parts = cast(tuple[Any, ...], result)
    return parts[0], parts[1], parts[2] if len(parts) > 2 else []


MATPLOTLIB_AVAILABLE = False
try:
    from gnn.utils.system_env.matplotlib_setup import apply_env_backend_if_set

    apply_env_backend_if_set()
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    MATPLOTLIB_AVAILABLE = True
except ImportError as exc:
    logger.debug("Matplotlib unavailable for type-checker visualizations: %s", exc)


def attach_metadata_footer(fig: Any, framework: str = "gnn_type_checker") -> None:
    """Attach standard metadata footer to a matplotlib figure."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        meta_str = f"Generated: {timestamp} | Component: {framework}"
        fig.text(
            0.99,
            0.01,
            meta_str,
            ha="right",
            va="bottom",
            fontsize=8,
            color="gray",
            alpha=0.7,
        )
    except Exception as e:
        logger.debug(f"Failed to attach metadata footer: {e}")


def generate_type_validity_mosaic(
    results: Dict[str, Any], output_dir: Path
) -> Optional[Path]:
    """Generate a grid heatmap / mosaic showing pass/fail status of all models."""
    if not MATPLOTLIB_AVAILABLE:
        return None

    try:
        validations = results.get("validation_results", [])
        if not validations:
            return None

        file_names = [v.get("file_name", "Unknown") for v in validations]

        # Calculate a nice grid dimension (e.g. roughly square)
        n = len(file_names)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols)) if cols > 0 else 0

        # Status: 0 for valid, 1 for warning (warnings > 0, errors == 0), 2 for invalid (errors > 0)
        status_grid = np.zeros((rows, cols))
        status_grid[:] = np.nan  # Fill unused cells with NaN

        for i, val in enumerate(validations):
            r = i // cols
            c = i % cols
            if val.get("valid", False) and not val.get("errors"):
                if val.get("warnings"):
                    status_grid[r, c] = 1  # Warning
                else:
                    status_grid[r, c] = 0  # Perfect
            else:
                status_grid[r, c] = 2  # Invalid

        fig, ax = plt.subplots(figsize=(max(6, cols * 1.5), max(4, rows * 1.5)))
        cmap = matplotlib.colormaps["RdYlGn_r"].resampled(
            3
        )  # Red(2), Yellow(1), Green(0)
        cmap.set_bad(color="white")

        ax.imshow(status_grid, cmap=cmap, vmin=0, vmax=2)

        # Grid lines and labels
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Attempt to label blocks if not too many
        if n <= 100:
            for i, name in enumerate(file_names):
                r = i // cols
                c = i % cols
                short_name = name[:10] + ".." if len(name) > 12 else name
                ax.text(
                    c,
                    r,
                    short_name,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                    alpha=0.8,
                    rotation=15,
                )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Model Type Validity Mosaic (N={n})", pad=20, size=14)

        # Custom legend
        from matplotlib.patches import Patch

        legend_elements: list[Any] = [
            Patch(facecolor=cmap(0.0), label="Valid"),
            Patch(facecolor=cmap(0.5), label="Warnings"),
            Patch(facecolor=cmap(1.0), label="Invalid"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.05, 1))

        attach_metadata_footer(fig)

        output_path = output_dir / "type_validity_mosaic.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return output_path

    except Exception as e:
        logger.error(f"Failed to generate type validity mosaic: {e}")
        return None


def generate_issue_distribution_chart(
    results: Dict[str, Any], output_dir: Path
) -> Optional[Path]:
    """Generate categorized bar chart comparing volume of issues per model."""
    if not MATPLOTLIB_AVAILABLE:
        return None

    try:
        validations = results.get("validation_results", [])
        if not validations:
            return None

        # Filter out models with no issues to keep chart clean if N is large
        models_with_issues = [
            v
            for v in validations
            if (len(v.get("warnings", [])) > 0 or len(v.get("errors", [])) > 0)
        ]
        if not models_with_issues:
            # If all are perfectly clean, chart the top 5 sizes just to show perfection
            models_with_issues = validations[:5]

        names = [v.get("file_name", "Unknown")[:15] for v in models_with_issues]
        warnings = [len(v.get("warnings", [])) for v in models_with_issues]
        errors = [len(v.get("errors", [])) for v in models_with_issues]

        x = np.arange(len(names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.5), 6))
        ax.bar(
            x - width / 2,
            warnings,
            width,
            label="Warnings / Unknown Types",
            color="gold",
            alpha=0.8,
        )
        ax.bar(
            x + width / 2,
            errors,
            width,
            label="Errors / Consistency Faults",
            color="crimson",
            alpha=0.8,
        )

        ax.set_ylabel("Issue Count")
        ax.set_title("Type Issue Distribution Across Models")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.legend()

        ax.grid(axis="y", linestyle="--", alpha=0.7)
        attach_metadata_footer(fig)

        output_path = output_dir / "type_issue_distribution.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return output_path

    except Exception as e:
        logger.error(f"Failed to generate issue distribution chart: {e}")
        return None


def generate_dimension_compatibility_radar(
    results: Dict[str, Any], output_dir: Path
) -> Optional[Path]:
    """Generate abstract mapping of dimensional compatibility issues."""
    if not MATPLOTLIB_AVAILABLE:
        return None

    try:
        validations = results.get("validation_results", [])
        if not validations:
            return None

        compat_scores: list[Any] = []
        names: list[Any] = []
        for v in validations:
            dc = v.get("dimension_compatibility", {})
            total_checks = len(dc.get("variables_checked", []))
            issues = len(dc.get("issues", []))

            names.append(v.get("file_name", "Unknown")[:15])
            # Construct a naive "compatibility score": 100 if perfect, drops per issue scaling on variables
            if total_checks == 0:
                score = 100.0
            else:
                score = max(0.0, 100.0 - (issues / float(total_checks)) * 100.0)
            compat_scores.append(score)

        if not names:
            return None

        # Horizontal bar chart abstract mapping
        fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.3)))
        y_pos = np.arange(len(names))

        colors = [
            "mediumseagreen" if score == 100 else "tomato" for score in compat_scores
        ]
        ax.barh(y_pos, compat_scores, align="center", color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()  # top-down
        ax.set_xlabel("Compatibility Score (%)")
        ax.set_title("Dimensional Constraint Compatibility Abstract")
        ax.set_xlim(0, 105)
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        # Mark averages
        avg_score = np.mean(compat_scores)
        ax.axvline(
            avg_score,
            color="black",
            linestyle="-.",
            alpha=0.5,
            label=f"Mean: {avg_score:.1f}%",
        )
        ax.legend()

        attach_metadata_footer(fig)

        output_path = output_dir / "dimension_compatibility_abstract.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return output_path

    except Exception as e:
        logger.error(f"Failed to generate dimensional compatibility abstract: {e}")
        return None


def generate_type_category_pie_chart(
    results: Dict[str, Any], output_dir: Path
) -> Optional[Path]:
    """Generate a global pie chart detailing the frequency of specific data types."""
    if not MATPLOTLIB_AVAILABLE:
        return None

    try:
        analyses = results.get("type_analysis", [])
        if not analyses:
            return None

        # Aggregate global type distribution
        global_distribution: dict[Any, Any] = {}
        for analysis in analyses:
            dist = analysis.get("type_distribution", {})
            for t, count in dist.items():
                # Avoid massive long text fragments which are false positives getting through
                if len(t) <= 20:
                    global_distribution[t] = global_distribution.get(t, 0) + count

        if not global_distribution:
            return None

        # Filter out anything below 2% to 'Other' to keep pie chart neat
        total_items = sum(global_distribution.values())
        threshold = total_items * 0.02

        filtered_dist: dict[Any, Any] = {}
        other_count = 0
        for t, count in global_distribution.items():
            if count >= threshold:
                filtered_dist[t] = count
            else:
                other_count += count

        if other_count > 0:
            filtered_dist["Other (<2%)"] = other_count

        labels = list(filtered_dist.keys())
        sizes = list(filtered_dist.values())

        fig, ax = plt.subplots(figsize=(8, 8))
        # Use a diverse colormap
        cmap = matplotlib.colormaps["tab20"].resampled(len(labels))
        colors = [tuple(color) for color in cmap(np.linspace(0, 1, len(labels)))]

        # Plot pie
        pie_result = ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            colors=colors,
            startangle=140,
            textprops=dict(color="w"),
        )
        # matplotlib >= 3.11 returns a PieContainer (no len()); older versions
        # return a (wedges, texts[, autotexts]) tuple.
        wedges, texts, autotexts = _pie_parts(pie_result)

        ax.legend(
            wedges,
            labels,
            title="Primary Types",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
        )
        plt.setp(autotexts, size=8, weight="bold", color="darkblue")
        plt.setp(texts, size=10, color="dimgray")

        ax.set_title("Global Type Distribution Abstract", pad=20, size=14)
        attach_metadata_footer(fig)

        output_path = output_dir / "type_category_distribution.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return output_path

    except Exception as e:
        logger.error(f"Failed to generate type category pie chart: {e}")
        return None


def generate_model_cards(results: Dict[str, Any], output_dir: Path) -> List[Path]:
    """Generate individual high-fidelity 'baseball cards' for each model containing complexity and type metrics."""
    if not MATPLOTLIB_AVAILABLE:
        return []

    cards_generated: list[Any] = []
    cards_dir = output_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)

    try:
        from matplotlib.patches import Rectangle

        validations = results.get("validation_results", [])
        for v in validations:
            name = v.get("file_name", "Unknown")
            short_name = name[:20] + ".." if len(name) > 22 else name
            is_valid = v.get("valid", False)
            errors = v.get("errors", [])
            warnings = v.get("warnings", [])

            # Resource extractions
            resources = v.get("resource_estimation", {})
            complexity = resources.get("complexity_tier", "Unknown")
            comp_score = resources.get("complexity_score", 0)
            mem_bytes = resources.get("estimated_memory_bytes", 0)

            # Format memory beautifully
            if mem_bytes < 1024 * 10:
                mem_str = f"{mem_bytes} Bytes"
            elif mem_bytes < 1024 * 1024:
                mem_str = f"{(mem_bytes / 1024):.1f} KB"
            else:
                mem_str = f"{(mem_bytes / (1024 * 1024)):.2f} MB"

            params = resources.get("total_parameters", 0)
            flops = resources.get("flops_estimate", 0)

            # Dark Theme Configuration
            fig, ax = plt.subplots(figsize=(5.5, 8.25))  # Trading card ratio

            # Colors
            bg_color = "#1E1E24"  # Deep carbon gray
            panel_color = "#292A33"  # Slightly lighter inset panel
            text_color = "#E0E1E6"  # Off-white for text
            subtext_color = "#9496A1"  # Cool gray for labels

            # Neon status accents
            if is_valid and not warnings:
                border_color = "#00F0FF"  # Cyan neon
                status_text = "VALID"
            elif is_valid:
                border_color = "#FFD700"  # Neon Gold
                status_text = "WARNINGS"
            else:
                border_color = "#FF3366"  # Neon Hot Pink/Red
                status_text = "INVALID"

            # Draw core border and body
            card_out = Rectangle(
                (0, 0), 1, 1, facecolor=border_color, edgecolor="none", zorder=0
            )
            card_in = Rectangle(
                (0.02, 0.015),
                0.96,
                0.97,
                facecolor=bg_color,
                edgecolor="none",
                zorder=1,
            )
            ax.add_patch(card_out)
            ax.add_patch(card_in)

            # Header Title Box (sleek full width top)
            header_rect = Rectangle(
                (0.02, 0.88), 0.96, 0.105, facecolor=panel_color, zorder=2
            )
            ax.add_patch(header_rect)

            # Model Name Text
            ax.text(
                0.5,
                0.93,
                short_name.upper(),
                ha="center",
                va="center",
                color=text_color,
                fontweight="black",
                fontsize=15,
                family="sans-serif",
                zorder=3,
            )

            # Abstract Center Graphic Box (simulating card art)
            art_rect = Rectangle(
                (0.06, 0.52),
                0.88,
                0.33,
                facecolor=panel_color,
                edgecolor=border_color,
                linewidth=1.5,
                zorder=2,
            )
            ax.add_patch(art_rect)

            ax.text(
                0.5,
                0.72,
                "COMPLEXITY CLASS",
                ha="center",
                va="center",
                color=subtext_color,
                fontsize=10,
                zorder=3,
                fontweight="bold",
            )
            ax.text(
                0.5,
                0.63,
                f"{complexity.upper()}",
                ha="center",
                va="center",
                color=border_color,
                fontweight="black",
                fontsize=26,
                zorder=3,
            )
            ax.text(
                0.5,
                0.55,
                f"Structural Score: {comp_score:.1f}",
                ha="center",
                va="center",
                color=subtext_color,
                fontsize=9,
                zorder=3,
                style="italic",
            )

            # Separator Line
            ax.plot(
                [0.1, 0.9], [0.47, 0.47], color=subtext_color, lw=1, alpha=0.3, zorder=2
            )

            # Left Data Column (Statistics)
            x_left = 0.1
            ax.text(
                x_left,
                0.42,
                "STATISTICS",
                fontweight="bold",
                fontsize=9,
                color=border_color,
            )

            ax.text(x_left, 0.36, "Memory Footprint", fontsize=8, color=subtext_color)
            ax.text(
                x_left, 0.33, mem_str, fontsize=11, color=text_color, fontweight="bold"
            )

            ax.text(x_left, 0.28, "Total Parameters", fontsize=8, color=subtext_color)
            ax.text(
                x_left,
                0.25,
                f"{params:,}",
                fontsize=11,
                color=text_color,
                fontweight="bold",
            )

            ax.text(x_left, 0.20, "Complexity (FLOPS)", fontsize=8, color=subtext_color)
            ax.text(
                x_left,
                0.17,
                f"{flops:,.1f}",
                fontsize=11,
                color=text_color,
                fontweight="bold",
            )

            # Right Data Column (Attributes & Validation)
            x_right = 0.55
            ax.text(
                x_right,
                0.42,
                "VALIDATION",
                fontweight="bold",
                fontsize=9,
                color=border_color,
            )

            ax.text(x_right, 0.36, "Global Status", fontsize=8, color=subtext_color)
            ax.text(
                x_right,
                0.33,
                status_text,
                fontsize=11,
                color=border_color,
                fontweight="bold",
            )

            # Display real warning context if available
            ax.text(x_right, 0.28, "Active Warnings", fontsize=8, color=subtext_color)
            warn_str = "None"
            if len(warnings) > 0:
                first_w = str(warnings[0])
                warn_str = first_w[:22] + ".." if len(first_w) > 22 else first_w
                warn_str = f"[{len(warnings)}] {warn_str}"
            ax.text(
                x_right,
                0.25,
                warn_str,
                fontsize=10,
                color=text_color,
                fontweight="bold",
            )

            # Display real error context if available
            ax.text(x_right, 0.20, "Critical Errors", fontsize=8, color=subtext_color)
            err_str = "None"
            if len(errors) > 0:
                first_e = str(errors[0])
                err_str = first_e[:22] + ".." if len(first_e) > 22 else first_e
                err_str = f"[{len(errors)}] {err_str}"
            ax.text(
                x_right, 0.17, err_str, fontsize=10, color=text_color, fontweight="bold"
            )

            # Flavor text / Types Footer
            traits = set([t.get("type", "Unknown") for t in v.get("type_issues", [])])
            trait_str = (
                ", ".join(list(traits)[:3])
                if traits
                else "Float matrices, Discrete variables"
            )

            footer_rect = Rectangle(
                (0.02, 0.015), 0.96, 0.10, facecolor=panel_color, zorder=2
            )
            ax.add_patch(footer_rect)
            ax.text(
                0.06,
                0.08,
                "PRIMARY TRAITS",
                fontweight="bold",
                fontsize=7,
                color=subtext_color,
            )
            ax.text(
                0.06,
                0.045,
                trait_str.upper(),
                fontsize=9,
                color=border_color,
                style="italic",
                fontweight="bold",
            )

            # Global style configs
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

            card_path = cards_dir / f"{name}_card.png"
            fig.savefig(card_path, dpi=250, bbox_inches="tight", facecolor=bg_color)
            plt.close(fig)
            cards_generated.append(card_path)

        return cards_generated

    except Exception as e:
        logger.error(f"Failed to generate baseball cards: {e}")
        return []


def generate_all_visualizations(results: Dict[str, Any], output_dir: Path) -> List[str]:
    """Generate all visualizations for type checker results and return markdown embedding strings."""
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    embeddings: list[Any] = []

    mosaic_path = generate_type_validity_mosaic(results, viz_dir)
    if mosaic_path and mosaic_path.exists():
        embeddings.append(f"![Type Validity Mosaic](visualizations/{mosaic_path.name})")

    issue_path = generate_issue_distribution_chart(results, viz_dir)
    if issue_path and issue_path.exists():
        embeddings.append(f"![Issue Distribution](visualizations/{issue_path.name})")

    dim_path = generate_dimension_compatibility_radar(results, viz_dir)
    if dim_path and dim_path.exists():
        embeddings.append(
            f"![Dimension Compatibility Abstract](visualizations/{dim_path.name})"
        )

    pie_path = generate_type_category_pie_chart(results, viz_dir)
    if pie_path and pie_path.exists():
        embeddings.append(
            f"![Global Type Category Distribution](visualizations/{pie_path.name})"
        )

    cards = generate_model_cards(results, viz_dir)
    if cards:
        # Load up to 2 preview cards, state rest are in dir
        embeddings.append("\n### Model Baseball Cards Preview")
        for i, card in enumerate(cards[:2]):
            embeddings.append(
                f"![Model Card Preview](visualizations/cards/{card.name})"
            )
        if len(cards) > 2:
            embeddings.append(
                f"\n*(Remaining {len(cards) - 2} Model Cards are located in `visualizations/cards/`)*"
            )

    return embeddings

"""Shared matplotlib save helpers for visualization (no imports from core.process)."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

plt: Any
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except (ImportError, RecursionError):
    plt = cast(Any, None)
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)


def contrasting_text_color(rgba: Any) -> str:
    """Choose black or white from the rendered cell's relative luminance.

    Matrix values alone do not locate a color in a normalized colormap. Using
    the actual color also works for signed and rescaled matrix ranges.
    """
    channels = [
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in rgba[:3]
    ]
    luminance = sum(
        weight * value for weight, value in zip((0.2126, 0.7152, 0.0722), channels)
    )
    black_contrast = (luminance + 0.05) / 0.05
    white_contrast = 1.05 / (luminance + 0.05)
    return "black" if black_contrast >= white_contrast else "white"


def save_figure(plot_path: Path, **savefig_kwargs: Any) -> None:
    """Save unchanged pixels with faster PNG encoding and caller-owned options.

    PNG compression changes encoded bytes and disk size, not the figure's
    resolution, artists, metadata or decoded pixels. Other formats retain
    Matplotlib's defaults. Errors propagate to the caller's existing boundary.
    """
    if plt is None:
        raise RuntimeError("matplotlib is unavailable")
    image_format = str(
        savefig_kwargs.get("format") or Path(plot_path).suffix.lstrip(".")
    ).lower()
    if image_format == "png":
        savefig_kwargs["pil_kwargs"] = {
            "compress_level": 3,
            **(savefig_kwargs.get("pil_kwargs") or {}),
        }
    plt.savefig(plot_path, **savefig_kwargs)


def save_plot_safely(plot_path: Path, dpi: int = 300, **savefig_kwargs: Any) -> bool:
    """Save current figure with DPI fallbacks."""
    if plt is None:
        return False

    def _safe_dpi_value(dpi_input: Any) -> int:
        """Handle safe dpi value for internal callers."""
        try:
            dpi_val = int(dpi_input) if isinstance(dpi_input, (int, float)) else 150
            return max(50, min(dpi_val, 600))
        except (ValueError, TypeError, OverflowError):
            return 150

    safe_dpi = _safe_dpi_value(dpi)
    try:
        save_figure(plot_path, dpi=safe_dpi, **savefig_kwargs)
        logger.debug("Saved plot with DPI %s", safe_dpi)
        return True
    except Exception as e:
        logger.debug("Error saving with DPI %s: %s", safe_dpi, e)
        try:
            fallback_dpi = _safe_dpi_value(matplotlib.rcParams.get("savefig.dpi", 100))
            save_figure(plot_path, dpi=fallback_dpi, **savefig_kwargs)
            return True
        except Exception as e2:
            logger.debug("Recovery DPI failed: %s", e2)
            try:
                save_figure(plot_path, **savefig_kwargs)
                return True
            except Exception as e3:
                logger.error("Failed to save plot %s: %s", plot_path, e3)
                return False


def safe_tight_layout() -> None:
    """Apply tight_layout with warning suppression."""
    if plt is None:
        return
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, message=".*[Tt]ight.?layout.*"
            )
            plt.tight_layout()
    except (ValueError, RuntimeError):
        logger.debug("tight_layout skipped (non-critical)")

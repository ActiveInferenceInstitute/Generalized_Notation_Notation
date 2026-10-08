# Visualization Plotting

Shared Matplotlib utilities used across all visualization sub-packages.

## Utilities in `utils.py`

- `plt` — Matplotlib pyplot when available
- `save_figure(path, **savefig_kwargs)` — Caller-authoritative Matplotlib save with PNG compression level 3 beneath explicit `pil_kwargs`; other formats retain their defaults
- `save_plot_safely()` — Safe file writer with error recovery (requested DPI, rcParams DPI, then default save)
- `contrasting_text_color(rgba)` — Black/white text contrast from the actual normalized cell color
- `safe_tight_layout()` — Tight layout with fallback for headless rendering
- `MATPLOTLIB_AVAILABLE` — Boolean flag for dependency checking

## Dependencies

- `matplotlib` (optional; plotting calls report unavailable state when absent)

PNG compression affects encoded bytes and disk size while preserving decoded pixels, metadata, resolution and figure artists. Direct `save_figure` leaves the caller's DPI untouched; `save_plot_safely` retains its existing DPI coercion/range and recovery behavior. The separate analysis save helper retains its existing bounded 72-DPI recovery. A caller can explicitly request `pil_kwargs={"compress_level": 6}`. The measured level-3 tradeoff and finite production-path coverage are documented in [analysis/PERFORMANCE.md](../../analysis/PERFORMANCE.md).

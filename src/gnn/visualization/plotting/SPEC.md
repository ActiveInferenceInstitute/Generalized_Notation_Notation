# Plotting Utilities — Technical Specification

**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)

## Style Configuration

The figure owner chooses fonts, figure size, axes, labels and background.
`save_plot_safely()` defaults to 300 DPI, bounds numeric requests to 50–600,
and uses 150 for invalid DPI values. Its recovery attempts preserve an explicit
success/failure result; inspect that result before publishing an artifact.

## Color Palette

Based on matplotlib's `tab10` with custom harmonization for up to 20 series.

## Save Formats

The Python caller supplies the output filename and Matplotlib save keywords,
including format and bounding-box controls. Native visualization producers
choose their PNG paths. There is no separate SVG command-line option.

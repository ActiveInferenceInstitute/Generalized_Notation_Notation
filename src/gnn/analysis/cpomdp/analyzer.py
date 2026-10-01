"""cpomdp analyzer: binds the shared flat-payload analyzer to cpomdp results.

cpomdp writes the continuous result schema plus ``efe_history`` (one row of
expected free energy per step, one column per policy). The shared analyzer
plots the posterior-mean trajectory and every policy's G over time, and writes
``cpomdp_analysis.json`` per model.
"""

from pathlib import Path

from gnn.analysis.flat_payload_analyzer import FlatPayloadSpec
from gnn.analysis.flat_payload_analyzer import (
    generate_analysis_from_logs as _generate_from_spec,
)

CPOMDP_SPEC = FlatPayloadSpec(
    framework="cpomdp",
    file_patterns=(
        "**/cpomdp/**/simulation_results.json",
        "**/cpomdp_simulation_results.json",
    ),
    analysis_filename="cpomdp_analysis.json",
    title_prefix="cpomdp",
    bar_color="#4a3aa7",
    log_label="cpomdp",
)


def generate_analysis_from_logs(
    results_dir: Path,
    output_dir: Path | None = None,
    verbose: bool = False,
) -> list[str]:
    """Generate analysis from cpomdp simulation results.

    Args:
        results_dir: Root to search for results, e.g. ``12_execute_output``.
        output_dir: Directory for analysis artifacts. Defaults to ``results_dir``.
        verbose: Enable verbose logging.

    Returns:
        Paths of the generated ``cpomdp_analysis.json`` files.
    """
    return _generate_from_spec(CPOMDP_SPEC, results_dir, output_dir, verbose)


__all__ = ["CPOMDP_SPEC", "generate_analysis_from_logs"]

#!/usr/bin/env python3
"""
Step 12: Execute Processing (Thin Orchestrator)

This step orchestrates execute processing for GNN models.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/gnn/execute/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the execute module.

Pipeline Flow:
    main.py → 12_execute.py (this script) → execute/ (modular implementation)

How to run:
  uv run python src/gnn/12_execute.py --target-dir input/gnn_files --output-dir output --verbose
  uv run python src/gnn/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Execute processing results in the specified output directory
  - Comprehensive execute reports and summaries
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that execute dependencies are installed
  - Check that src/gnn/execute/ contains execute modules
  - Check that the output directory is writable
  - Verify execute configuration and requirements
"""

import argparse
from pathlib import Path
from typing import cast

from gnn.execute import process_execute
from gnn.utils.pipeline_orchestration.pipeline_template import (
    create_standardized_pipeline_script,
)

run_script = create_standardized_pipeline_script(
    "12_execute.py",
    process_execute,
    "Execute processing for GNN simulations",
    additional_arguments={
        "render_output_dir": {
            "flag": "--render-output-dir",
            "type": Path,
            "default": None,
            "help": "Explicit path to the 11_render_output directory to execute (avoids filesystem heuristics)",
        },
        "frameworks": {
            "flag": "--frameworks",
            "type": str,
            "default": "all",
            "help": "Frameworks to execute (all, lite, or comma-separated list: pymdp,rxinfer,activeinference_jl,jax,discopy,pytorch,numpyro,stan,bnlearn,cpomdp,lean)",
        },
        "timeout": {
            "flag": "--timeout",
            "type": int,
            "default": 3600,
            "help": "Maximum execution time in seconds for each subprocess (default: 3600s = 1 hour)",
        },
        "distributed": {
            "flag": "--distributed",
            "action": "store_true",
            "help": "Run scripts and model parameter sweeps in parallel across a Ray/Dask cluster",
        },
        "execution_workers": {
            "flag": "--execution-workers",
            "type": int,
            "default": 1,
            "help": "Number of local or distributed workers for rendered script execution (default: 1)",
        },
        "backend": {
            "flag": "--backend",
            "type": str,
            "choices": ["ray", "dask"],
            "default": "ray",
            "help": "Backend to use for distributed execution (default is ray)",
        },
        "execution_benchmark_repeats": {
            "flag": "--execution-benchmark-repeats",
            "type": int,
            "default": 1,
            "help": "Sequential benchmark repeats per script; reports median duration when >1",
        },
        "execution_summary_detail": {
            "flag": "--execution-summary-detail",
            "action": argparse.BooleanOptionalAction,
            "default": False,
            "help": (
                "Also write summaries/execution_summary_detail.json with full per-script "
                "payloads (aggregate execution_summary.json stays slim)"
            ),
        },
    },
    default_target_dir="output/11_render_output",
    default_recursive=True,
)


def main() -> int:
    """Main entry point for the execute step."""
    return cast("int", run_script())


if __name__ == "__main__":
    raise SystemExit(main())

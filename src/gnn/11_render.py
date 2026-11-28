#!/usr/bin/env python3
"""
Step 11: Render Processing (Thin Orchestrator)

This step orchestrates render processing for GNN models.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/gnn/render/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the render module.

Pipeline Flow:
    main.py → 11_render.py (this script) → render/ (modular implementation)

How to run:
  uv run python src/gnn/11_render.py --target-dir input/gnn_files --output-dir output --verbose
  uv run python src/gnn/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Render processing results in the specified output directory
  - Comprehensive render reports and summaries
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that render dependencies are installed
  - Check that src/gnn/render/ contains render modules
  - Check that the output directory is writable
  - Verify render configuration and requirements
"""

import sys

if globals().get("__package__") is None and not sys.flags.safe_path:
    sys.path.pop(0)  # Keep adjacent gnn/types from shadowing stdlib types.

from typing import cast

from gnn.render import process_render
from gnn.utils.pipeline_orchestration.pipeline_template import (
    create_standardized_pipeline_script,
)

run_script = create_standardized_pipeline_script(
    "11_render.py",
    process_render,
    "Render processing for GNN specifications",
    additional_arguments={
        "timesteps": {
            "flag": "--timesteps",
            "type": int,
            "default": None,
            "help": "Number of timesteps for generated Active Inference simulations",
        },
        "simulation_params": {
            "flag": "--simulation-params",
            "type": str,
            "default": "{}",
            "help": "JSON object with THRML overrides. Requires --frameworks thrml.",
        },
        "frameworks": {
            "flag": "--frameworks",
            "type": str,
            "default": "all",
            "help": "Frameworks to render (all, lite, or comma-separated list)",
        },
        "strict_framework_success": {
            "flag": "--strict-framework-success",
            "action": "store_true",
            "help": "Fail the render step when any requested framework render fails",
        },
    },
    default_target_dir="input/gnn_files",
    default_recursive=True,
)


def main() -> int:
    """Main entry point for the render step."""
    return cast("int", run_script())


if __name__ == "__main__":
    raise SystemExit(main())

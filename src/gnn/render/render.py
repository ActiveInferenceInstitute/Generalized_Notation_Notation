"""
Main rendering module for GNN specifications.

This module coordinates the rendering of GNN specifications to various
target platforms, including RxInfer.jl and PyMDP.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from .processor import render_gnn_spec

RENDER_CLI_TARGETS = [
    "pymdp",
    "rxinfer",
    "activeinference_jl",
    "discopy",
    "discopy_combined",
    "bnlearn",
    "jax",
    "jax_pomdp",
    "pytorch",
    "numpyro",
    "stan",
    "cpomdp",
]


def main(cli_args: Any = None) -> Any:
    """Command-line entry point for the renderer."""
    parser = argparse.ArgumentParser(
        description="Render GNN specifications to various target platforms"
    )
    parser.add_argument("gnn_file", help="Path to the GNN specification file")
    parser.add_argument("output_dir", help="Output directory for rendered files")
    parser.add_argument(
        "target",
        choices=RENDER_CLI_TARGETS,
        default="pymdp",
        help="Target platform",
    )
    parser.add_argument(
        "--output_filename", help="Base filename for the output (without extension)"
    )
    parser.add_argument(
        "--debug", "--verbose", action="store_true", help="Enable debug logging"
    )

    args = parser.parse_args(cli_args)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    gnn_file_path = Path(args.gnn_file)
    if not gnn_file_path.exists():
        logger.error(f"GNN file not found: {gnn_file_path}")
        return 1

    try:
        from gnn import parse_gnn_file

        gnn_spec = parse_gnn_file(gnn_file_path)
        logger.info(f"Successfully parsed GNN file using parser: {gnn_file_path}")
    except (ImportError, ModuleNotFoundError) as e:
        logger.error(
            f"GNN parser unavailable — cannot render without canonical parser: {e}"
        )
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_filename:
        base_filename = args.output_filename
    else:
        base_filename = gnn_spec.get("name", gnn_file_path.stem)

    success, message, artifacts = render_gnn_spec(
        gnn_spec, args.target, output_dir, {"output_filename": base_filename}
    )

    if success:
        print(f"Successfully rendered to {args.target}: {message}")
        print(f"Output artifacts: {', '.join(artifacts)}")
        return 0
    else:
        print(f"Error rendering to {args.target}: {message}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

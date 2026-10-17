#!/usr/bin/env python3
"""Explicit local-only corpus driver; requires a provisioned Ollama model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from gnn.llm import process_llm


def main() -> int:
    """Parse a Mac-sized profile and delegate all processing to Step 13."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, default=Path("input/gnn_files"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/local_llm"))
    parser.add_argument("--model", default="gemma3:12b")
    parser.add_argument("--context-tokens", type=int, default=131072)
    parser.add_argument("--source-chunk-bytes", type=int, default=65536)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--prompt-timeout", type=float, default=180)
    parser.add_argument("--response-words", type=int, default=160)
    parser.add_argument("--no-batch-prompts", action="store_true")
    parser.add_argument("--llm-timeout", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not args.target_dir.is_dir():
        parser.error("--target-dir must be an existing directory")
    success = process_llm(
        args.target_dir,
        args.output_dir,
        provider="ollama",
        model=args.model,
        llm_config={"ollama_options": {"num_ctx": args.context_tokens, "seed": 42}},
        max_tokens=args.max_tokens,
        source_chunk_bytes=args.source_chunk_bytes,
        max_prompt_timeout=args.prompt_timeout,
        llm_timeout=args.llm_timeout,
        resume=args.resume,
        batch_prompts=not args.no_batch_prompts,
        response_words=args.response_words,
    )
    logging.info("Local corpus receipt: %s", args.output_dir / "llm_results.json")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

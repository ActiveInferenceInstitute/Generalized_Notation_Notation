#!/usr/bin/env python3
"""Worker dispatch for Step 12 script execution (coercion + process pools)."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from gnn.execute.processor.envelope import (
    _make_local_worker_pool_failure_result,
)
from gnn.execute.processor.single import execute_single_script

def _coerce_execution_workers(value: Any) -> int:
    """Normalize the configured local/distributed worker count."""
    try:
        workers = int(value)
    except (TypeError, ValueError):
        workers = 1
    return max(1, workers)


def _coerce_dispatch_retries(value: Any) -> int:
    """Normalize the distributed task retry limit."""
    try:
        retries = int(value)
    except (TypeError, ValueError):
        retries = 3
    return max(0, retries)


def _execute_script_worker(
    bundle: Tuple[Dict[str, Any], Path, bool, int, int],
) -> Dict[str, Any]:
    """Process-pool entry point for a single rendered script."""
    script_info, results_dir, verbose, timeout, repeats = bundle
    worker_logger = logging.getLogger("execute.worker")
    worker_logger.setLevel(logging.INFO)
    result = execute_single_script(
        script_info,
        results_dir,
        verbose,
        worker_logger,
        timeout,
        execution_benchmark_repeats=repeats,
    )
    result.setdefault("skipped", False)
    return result


def _run_scripts_with_local_workers(
    executable_scripts: List[Dict[str, Any]],
    results_dir: Path,
    verbose: bool,
    logger: logging.Logger,
    timeout: int,
    execution_workers: int,
    execution_benchmark_repeats: int,
) -> List[Dict[str, Any]]:
    """Execute rendered scripts locally, using multiple processes when requested."""
    repeats = max(1, int(execution_benchmark_repeats))
    if execution_workers <= 1 or len(executable_scripts) <= 1:
        details: list[Any] = []
        for script_info in executable_scripts:
            exec_result = execute_single_script(
                script_info,
                results_dir,
                verbose,
                logger,
                timeout,
                execution_benchmark_repeats=repeats,
            )
            exec_result.setdefault("skipped", False)
            details.append(exec_result)
        return details

    bounded_workers = min(execution_workers, len(executable_scripts))
    logger.info(
        "Dispatching %s executable scripts with %s local workers",
        len(executable_scripts),
        bounded_workers,
    )
    bundles = [
        (info, results_dir, verbose, timeout, repeats) for info in executable_scripts
    ]
    try:
        from gnn.execute import processor as _processor_facade

        with _processor_facade.ProcessPoolExecutor(
            max_workers=bounded_workers
        ) as pool:
            return list(pool.map(_execute_script_worker, bundles))
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Local worker pool failed while executing %d scripts: %s",
            len(executable_scripts),
            exc,
        )
        return [
            _make_local_worker_pool_failure_result(script_info, exc)
            for script_info in executable_scripts
        ]


"""Run synchronous LLM entrypoints from ordinary or already-running loops."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Coroutine, TypeVar

_T = TypeVar("_T")


def run_async(coroutine: Coroutine[Any, Any, _T]) -> _T:
    """Await exactly once on an owned loop, preserving provider deadlines."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coroutine).result()

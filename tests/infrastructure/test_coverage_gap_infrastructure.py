#!/usr/bin/env python3
"""
Infrastructure Coverage Gap Tests

Addresses modules that historically had 0% coverage. The timeout_manager
gap remains covered here; the visualization_optimizer gap module was
removed as dead code (2026-09-21), so only timeout_manager coverage
remains. The ``utils/recovery.py`` fallback was removed in
Phase 6 as dead code — ``setup_step_logging`` is covered in place via
``utils/logging/logging_utils``. ``utils/simulation_utils.py`` was removed as
dead code (R4 residue); its only importer was this file.
"""

import asyncio
import warnings
from typing import Any

from gnn.utils.runtime_safety.timeout_manager import (
    LLMTimeoutManager,
    ProcessTimeoutManager,
    TimeoutConfig,
    TimeoutManager,
    TimeoutStrategy,
)


# 1. Tests for utils/timeout_manager.py
class TestTimeoutManager:
    def test_sync_timeout_success(self) -> Any:
        manager = TimeoutManager()
        config = TimeoutConfig(base_timeout=1.0, max_retries=0)

        def fast_func(x: Any) -> Any:
            return x * 2

        with manager.sync_timeout("test_sync", config, fast_func, 5) as result:
            assert result.success is True
            assert result.result == 10

    def test_async_timeout_success(self) -> Any:
        """Exercise async_timeout without pytest-asyncio (core env may omit dev extras)."""

        async def _run() -> None:
            manager = TimeoutManager()
            config = TimeoutConfig(base_timeout=1.0, max_retries=0)

            async def fast_async(x: int) -> int:
                return x * 2

            async with manager.async_timeout(
                "test_async", config, fast_async, 5
            ) as result:
                assert result.success is True
                assert result.result == 10

        asyncio.run(_run())

    def test_async_graceful_fallback_awaits_coroutine_without_deprecation(
        self,
    ) -> Any:
        """Coroutine fallbacks are detected via ``inspect`` and awaited.

        ``asyncio.iscoroutinefunction`` is deprecated on Python 3.14; the
        degradation path must not emit that DeprecationWarning.
        """

        async def _run() -> Any:
            async def failing_op() -> int:
                raise RuntimeError("primary failed")

            async def async_fallback() -> str:
                return "async-recovered"

            manager = TimeoutManager()
            config = TimeoutConfig(
                base_timeout=1.0,
                max_retries=0,
                strategy=TimeoutStrategy.GRACEFUL_DEGRADATION,
                graceful_fallback=async_fallback,
                log_retries=False,
            )
            async with manager.async_timeout(
                "graceful_async", config, failing_op
            ) as result:
                return result

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            result = asyncio.run(_run())
        assert result.success is True
        assert result.used_fallback is True
        assert result.result == "async-recovered"
        assert result.error is None

    def test_async_graceful_fallback_calls_sync_function(self) -> Any:
        async def _run() -> Any:
            async def failing_op() -> int:
                raise RuntimeError("primary failed")

            manager = TimeoutManager()
            config = TimeoutConfig(
                base_timeout=1.0,
                max_retries=0,
                strategy=TimeoutStrategy.GRACEFUL_DEGRADATION,
                graceful_fallback=lambda: "sync-recovered",
                log_retries=False,
            )
            async with manager.async_timeout(
                "graceful_sync", config, failing_op
            ) as result:
                return result

        result = asyncio.run(_run())
        assert result.success is True
        assert result.used_fallback is True
        assert result.result == "sync-recovered"

    def test_llm_timeout_manager(self) -> Any:
        manager = LLMTimeoutManager()
        assert manager.default_config.base_timeout == 60.0

    def test_process_timeout_manager(self) -> Any:
        manager = ProcessTimeoutManager()
        assert manager.default_config.base_timeout == 120.0

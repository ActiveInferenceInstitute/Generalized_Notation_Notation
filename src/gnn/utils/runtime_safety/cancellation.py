"""Cooperative cancellation vocabulary shared by every run surface.

:class:`CancelToken` is the one cancellation type in the repository: the
subprocess envelope polls it, the pipeline step executor injects one per
step, and the API cancels a run through the token stored on its record.

It lives in this standard-library-only leaf so that a package can create or
accept a token without importing ``gnn.execute``. The execute package
re-exports the same class object from
``gnn.execute.subprocess_envelope``, so existing imports keep working and
``isinstance`` checks agree across both paths.
"""

from __future__ import annotations

import threading
from typing import Optional


class CancelToken:
    """Thread-safe cooperative cancellation flag for in-flight subprocess runs.

    Pass a token to ``gnn.execute.subprocess_envelope.run_subprocess_envelope``
    (or the executor helpers built on it) and call :meth:`cancel` from any
    thread: the in-flight run's poll loop observes the flag within one poll
    slice (<=0.25s), then runs bounded cleanup and reports
    ``error_type="Cancelled"`` without raising.
    A cleanup failure takes precedence and retains cancellation evidence.
    ``cancel()`` is idempotent; the first ``reason`` wins.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason: Optional[str] = None

    def cancel(self, reason: Optional[str] = None) -> None:
        """Request cancellation (idempotent); the first ``reason`` wins."""
        if not self._event.is_set():
            self._reason = reason
        self._event.set()

    @property
    def cancelled(self) -> bool:
        """True once :meth:`cancel` has been called from any thread."""
        return self._event.is_set()

    @property
    def reason(self) -> Optional[str]:
        """Reason recorded by the first :meth:`cancel` call (may be None)."""
        return self._reason


__all__ = ["CancelToken"]

"""The shared ``julia`` executable probe.

Availability checks for setup, security scanning, integration diagnostics and
execution resolve the interpreter through :func:`julia_executable`. The probe
lives in this standard-library-only leaf so that ``gnn.setup``,
``gnn.security`` and ``gnn.integration`` can ask whether Julia is present
without importing ``gnn.execute``. ``gnn.execute.julia_setup`` re-exports the
same function object and owns everything that actually runs Julia (version
checks, project setup, the headless environment).
"""

from __future__ import annotations

import shutil
from typing import Optional


def julia_executable() -> Optional[str]:
    """Return the path to the ``julia`` executable, or ``None`` if not on PATH."""
    return shutil.which("julia")


__all__ = ["julia_executable"]

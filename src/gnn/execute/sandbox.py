"""Opt-in sandbox for rendered-script execution (Step 12).

GNN renders untrusted text specifications into executable Python/Julia scripts.
By default those scripts run **unsandboxed** with the operator's privileges; the
documented contract is trusted, local, single-user input (see
``SECURITY.md`` and ``RED_TEAM_REVIEW.md``). This module adds a safe, opt-in
wrapper that runs rendered scripts under an available sandbox/namespace tool and
degrades **loudly** (never silently) when no such tool is present.

Modes (selected via ``GNN_SANDBOX`` or an explicit argument):

- ``off`` (default): behave exactly as before — no wrapping. Preserves the
  existing trusted-local contract.
- ``prefer``: wrap when a sandbox binary is found; otherwise run unsandboxed
  and emit a warning.
- ``require``: wrap when a sandbox binary is found; otherwise **refuse to run**
  the script and return an error envelope.

Explicit settings outside these modes are rejected before execution.

Supported sandbox backends, in preference order (first found wins):

- ``firejail``: ``firejail --noprofile --net=none --private --caps.drop=all``
- ``bwrap``  (bubblewrap): ``bwrap --unshare-all --die-with-parent --ro-bind / / ...``
- ``nsjail``: ``nsjail -Mo --disable-proc --net none``

The wrapper is intentionally conservative: it only changes the *command prefix*,
leaving the executor's own timeout/cwd/env handling untouched.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Valid ``GNN_SANDBOX`` modes.
SANDBOX_MODES = ("off", "prefer", "require")
_DEFAULT_MODE = "off"


@dataclass(frozen=True)
class SandboxSpec:
    """One recognized sandbox backend and the prefix it contributes."""

    binary: str
    prefix: tuple[str, ...]

    def available(self) -> bool:
        """Whether the backend binary is on PATH."""
        return shutil.which(self.binary) is not None


@dataclass(frozen=True)
class SandboxAdmission:
    """The validated mode and backend selected for a launch."""

    mode: str
    backend: Optional[SandboxSpec]


class SandboxAdmissionError(ValueError):
    """An invalid policy or an unavailable required backend."""

    def __init__(self, mode: str, error_type: str, message: str) -> None:
        super().__init__(message)
        self.mode = mode
        self.error_type = error_type


#: Preference-ordered backends. ``bwrap`` builds a read-only bind of the root
#: filesystem so a fixed ``/`` exists inside the namespace; ``--dev /dev`` and
#: ``--proc /proc`` keep standard runtime mounts available without network.
_SANDBOX_BACKENDS: tuple[SandboxSpec, ...] = (
    SandboxSpec(
        "firejail",
        ("firejail", "--noprofile", "--net=none", "--private", "--caps.drop=all"),
    ),
    SandboxSpec(
        "bwrap",
        (
            "bwrap",
            "--unshare-all",
            "--die-with-parent",
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
        ),
    ),
    SandboxSpec(
        "nsjail",
        ("nsjail", "-Mo", "--disable-proc", "--net", "none"),
    ),
)


def _resolve_mode(mode: Optional[str]) -> str:
    """Resolve the effective sandbox mode from an explicit arg or the env var."""
    if mode is None:
        mode = os.environ.get("GNN_SANDBOX", _DEFAULT_MODE)
    mode = mode.strip().lower()
    if mode not in SANDBOX_MODES:
        raise SandboxAdmissionError(
            "invalid",
            "InvalidSandboxMode",
            "GNN_SANDBOX must be one of off, prefer, or require",
        )
    return mode


def detect_sandbox() -> Optional[SandboxSpec]:
    """Return the first available sandbox backend, or ``None`` if none exist."""
    for spec in _SANDBOX_BACKENDS:
        if spec.available():
            return spec
    return None


def wrap_command(command: List[str], spec: SandboxSpec) -> List[str]:
    """Return ``command`` prefixed with the sandbox backend's isolation flags."""
    return list(spec.prefix) + list(command)


def resolve_sandbox(mode: Optional[str] = None) -> SandboxAdmission:
    """Validate the requested sandbox and select an available backend."""
    effective = _resolve_mode(mode)
    spec = detect_sandbox() if effective != "off" else None
    if effective == "require" and spec is None:
        raise SandboxAdmissionError(
            effective,
            "SandboxUnavailable",
            "GNN_SANDBOX=require but no sandbox backend "
            "(firejail/bwrap/nsjail) is installed",
        )
    return SandboxAdmission(effective, spec)


def run_sandboxed(
    command: List[str],
    *,
    timeout: Optional[int] = None,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    capture_output: bool = True,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Run ``command`` under a sandbox according to ``mode``.

    Returns a uniform envelope regardless of sandbox availability, so callers
    never have to branch on the failure mode:

    - ``sandboxed`` (bool): whether the command was actually wrapped.
    - ``sandbox`` (str|None): backend binary name, or ``None``.
    - ``mode`` (str): the effective mode used.
    - ``blocked`` (bool): True when policy admission refused execution.
    - ``success`` (bool), ``return_code`` (int), ``stdout``/``stderr`` (str).
    """
    from gnn.execute.subprocess_envelope import NEVER_STARTED, run_subprocess_envelope

    envelope: Dict[str, Any] = {
        "sandboxed": False,
        "sandbox": None,
        "blocked": False,
        "success": False,
        "return_code": NEVER_STARTED,
        "stdout": "",
        "stderr": "",
    }
    try:
        admission = resolve_sandbox(mode)
    except SandboxAdmissionError as exc:
        envelope.update(
            blocked=True, mode=exc.mode, error=str(exc), error_type=exc.error_type
        )
        logger.error("%s", exc)
        return envelope

    # Canonical subprocess envelope (MAJ-10): one structured outcome for
    # timeout / OSError / non-zero exit; the sandbox-specific timeout message
    # shape is preserved on top of the shared envelope.
    outcome = run_subprocess_envelope(
        command,
        timeout=timeout,
        env=env,
        cwd=cwd,
        capture_output=capture_output,
        sandbox=admission,
    )
    envelope["mode"] = admission.mode
    envelope["sandboxed"] = outcome["sandboxed"]
    if admission.backend is not None:
        envelope["sandbox"] = admission.backend.binary
    envelope["return_code"] = outcome["return_code"]
    envelope["success"] = outcome["success"]
    envelope["stdout"] = outcome["stdout"]
    envelope["stderr"] = outcome["stderr"]
    if outcome.get("error") is not None:
        if outcome.get("error_type") == "TimeoutExpired":
            envelope["error"] = f"Sandboxed execution timed out after {timeout}s"
        else:
            envelope["error"] = outcome["error"]
        envelope["error_type"] = outcome.get("error_type")

    return envelope


# Convenience re-export so callers can reason about the module's surface without
# importing private helpers.
__all__ = [
    "SANDBOX_MODES",
    "SandboxSpec",
    "detect_sandbox",
    "wrap_command",
    "run_sandboxed",
]

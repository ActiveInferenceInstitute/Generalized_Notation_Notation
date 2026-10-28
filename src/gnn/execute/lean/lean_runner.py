"""Lean verification runner: drives ``fep-formal bridge verify-document``.

Resolves the sibling ``fep_formal`` checkout (env ``FEP_FORMAL_ROOT`` overrides
the ``../fep_formal`` default; legacy root names remain supported), discovers emitted Lean/GNN documents under the
rendered target directory, and verifies each through the FEP_Formal bridge
``verify-document`` operation (bridge contract v0.6, Direction 2 S7:
well-formedness against the ``FEP.GnnDocument`` typed AST).
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Optional, Union

from gnn.execute.subprocess_envelope import (
    NEVER_STARTED,
    CancelToken,
    run_subprocess_envelope,
)

logger = logging.getLogger(__name__)

#: Canonical checkout override; FEP_LEAN_ROOT remains a deprecated alias.
FEP_FORMAL_ROOT_ENV: Final = "FEP_FORMAL_ROOT"
FEP_LEAN_ROOT_ENV: Final = "FEP_LEAN_ROOT"

#: Default ceiling for one verify-document invocation (Lean + Mathlib).
_VERIFY_TIMEOUT_SECONDS = 1800


def resolve_fep_formal_root() -> Path | None:
    """Resolve the canonical checkout, retaining legacy environment/layout support.

    An explicit canonical override wins over the deprecated override, even
    when invalid. Without overrides, prefer the canonical sibling checkout.
    """
    env = os.environ.get(FEP_FORMAL_ROOT_ENV)
    if env is None:
        env = os.environ.get(FEP_LEAN_ROOT_ENV)
    if env is not None and not env.strip():
        return None
    parent = Path(__file__).resolve().parents[4].parent
    candidates = (
        (Path(env),)
        if env is not None
        else (parent / "fep_formal", parent / "fep_lean")
    )
    for candidate in candidates:
        root = candidate.resolve()
        if (root / "pyproject.toml").is_file() and any(
            (root / "src" / package).is_dir() for package in ("fep_formal", "fep_lean")
        ):
            return root
    return None


def resolve_fep_lean_root() -> Path | None:
    """Deprecated compatibility alias for :func:`resolve_fep_formal_root`."""
    return resolve_fep_formal_root()


def fep_bridge_executable(root: Path) -> str:
    """Select the canonical CLI, retaining support for legacy-only checkouts."""
    return "fep-formal" if (root / "src" / "fep_formal").is_dir() else "fep-lean"


class FepRuntimeConfigurationError(ValueError):
    """The companion's declared runtime cannot be safely selected."""


@dataclass(frozen=True, slots=True)
class FepRuntimeInvocation:
    """Command and controlled child overrides; never contains caller secrets."""

    command: tuple[str, ...]
    environment: Mapping[str, str]


def fep_runtime_invocation(
    root: Path, arguments: Sequence[str]
) -> FepRuntimeInvocation:
    """Bind the exact companion uv pin, declared Python and installed environment."""
    root = root.resolve()
    try:
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        standalone = root / "uv.toml"
        uv_config = (
            tomllib.loads(standalone.read_text(encoding="utf-8"))
            if standalone.is_file()
            else None
        )
        python = (root / ".python-version").read_text(encoding="utf-8").strip()
    except (OSError, tomllib.TOMLDecodeError, UnicodeError):
        raise FepRuntimeConfigurationError(
            "FEP runtime metadata is missing or malformed"
        ) from None
    tool = project.get("tool", {})
    project_uv = tool.get("uv", {}) if isinstance(tool, dict) else None
    if not isinstance(project_uv, dict):
        raise FepRuntimeConfigurationError("FEP uv metadata must be a table")
    config = uv_config if uv_config is not None else project_uv
    pin = config.get("required-version")
    if (
        not isinstance(pin, str)
        or re.fullmatch(r"==[0-9]+\.[0-9]+\.[0-9]+", pin) is None
    ):
        raise FepRuntimeConfigurationError(
            "FEP requires an exact stable uv version pin"
        )
    if uv_config is not None and project_uv.get("required-version", pin) != pin:
        raise FepRuntimeConfigurationError("FEP uv metadata contains conflicting pins")
    if re.fullmatch(r"3\.[0-9]+(?:\.[0-9]+)?", python) is None:
        raise FepRuntimeConfigurationError("FEP requires one declared CPython version")
    environment = root / ".venv"
    try:
        venv_metadata = (environment / "pyvenv.cfg").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise FepRuntimeConfigurationError(
            "FEP project environment is not installed"
        ) from None
    installed_versions = re.findall(
        r"^(?:version_info|version)\s*=\s*(3\.[0-9]+(?:\.[0-9]+)?)\s*$",
        venv_metadata,
        re.MULTILINE,
    )
    if not installed_versions or any(
        version.split(".")[: len(python.split("."))] != python.split(".")
        for version in installed_versions
    ):
        raise FepRuntimeConfigurationError(
            "FEP installed environment does not match declared Python"
        )
    return FepRuntimeInvocation(
        command=(
            "uv",
            "tool",
            "run",
            "--from",
            f"uv{pin}",
            "uv",
            "run",
            "--project",
            str(root),
            "--frozen",
            "--no-sync",
            "--python",
            python,
            *arguments,
        ),
        environment=MappingProxyType(
            {
                "UV_PROJECT_ENVIRONMENT": str(environment),
                "VIRTUAL_ENV": str(environment),
                "UV_PYTHON": python,
                "UV_PYTHON_DOWNLOADS": "never",
            }
        ),
    )


def lean_toolchain_available() -> bool:
    """True when a FEP_Formal checkout is resolvable for verification."""
    return resolve_fep_formal_root() is not None


def verify_document(
    document: Union[str, Path],
    receipt: Union[str, Path] | None = None,
    *,
    model: str = "finite",
    fail_on_warnings: bool = True,
    gnn_root: Union[str, Path] | None = None,
    timeout: int = _VERIFY_TIMEOUT_SECONDS,
    cancel_token: CancelToken | None = None,
) -> dict[str, Any]:
    """Verify one emitted document via ``fep-formal bridge verify-document``.

    Returns a record with ``success`` plus either the parsed receipt or an
    ``error`` message captured from the bridge CLI (fail-closed). A fired
    ``cancel_token`` cancels the dispatch cooperatively inside the
    subprocess envelope (pre-spawn or mid-flight).
    """
    root = resolve_fep_formal_root()
    if root is None:
        logger.info(
            "ℹ️ FEP_Formal not available - skipping Lean verification (set %s)",
            FEP_FORMAL_ROOT_ENV,
        )
        return {"success": False, "error": "fep_lean unavailable"}

    document_path = Path(document).resolve()
    gnn_root_path = (
        Path(gnn_root).resolve() if gnn_root else Path(__file__).resolve().parents[4]
    )
    if receipt is not None:
        receipt_path = Path(receipt).resolve()
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        _temp_dir = None
    else:
        # Scoped temp dir (cleaned up in the finally below); the previous
        # mkdtemp leaked one directory per bare verify_document call that
        # did not supply a receipt path.
        _temp_dir = tempfile.TemporaryDirectory(prefix="gnn-lean-verify-")
        receipt_path = Path(_temp_dir.name) / f"{document_path.stem}-receipt.json"

    try:
        return _verify_document_impl(
            document_path,
            gnn_root_path,
            receipt_path,
            model,
            fail_on_warnings,
            timeout,
            cancel_token=cancel_token,
        )
    finally:
        if _temp_dir is not None:
            _temp_dir.cleanup()


def _verify_document_impl(
    document_path: Path,
    gnn_root_path: Path,
    receipt_path: Path,
    model: str,
    fail_on_warnings: bool,
    timeout: int,
    cancel_token: CancelToken | None,
) -> dict[str, Any]:
    root = Path(_lean_cwd(gnn_root_path))
    arguments = [
        fep_bridge_executable(root),
        "bridge",
        "verify-document",
        "--gnn-root",
        str(gnn_root_path),
        "--document",
        str(document_path),
        "--model",
        model,
        "--receipt",
        str(receipt_path),
    ]
    if fail_on_warnings:
        arguments.append("--fail-on-warnings")
    try:
        invocation = fep_runtime_invocation(root, arguments)
    except FepRuntimeConfigurationError as error:
        return {
            "success": False,
            "document": str(document_path),
            "error": str(error),
            "error_type": type(error).__name__,
            "return_code": NEVER_STARTED,
            "returncode": NEVER_STARTED,
        }
    command = list(invocation.command)

    record: dict[str, Any] = {
        "success": False,
        "document": str(document_path),
        "command": command,
    }
    envelope = run_subprocess_envelope(
        command,
        timeout=timeout,
        cwd=str(root),
        env=dict(invocation.environment),
        cancel_token=cancel_token,
    )
    # Canonical ``return_code`` key (MAJ-10); ``returncode`` kept for
    # existing consumers of the lean record.
    record["return_code"] = envelope["return_code"]
    record["returncode"] = envelope["return_code"]
    if envelope.get("cancelled"):
        # Cooperative cancel (pre-spawn or mid-flight): surface it so
        # callers can distinguish cancellation from a bridge failure.
        record["cancelled"] = True
        record["error_type"] = envelope.get("error_type", "Cancelled")
    if envelope["success"]:
        record["success"] = True
        if receipt_path.is_file():
            try:
                record["receipt"] = json.loads(receipt_path.read_text())
            except json.JSONDecodeError:
                record["receipt_error"] = "unparseable receipt JSON"
        return record

    if envelope["return_code"] == NEVER_STARTED:
        # Timeout or invocation failure (OSError) — fail closed with the cause.
        record["error"] = (
            f"verify-document invocation failed: {envelope.get('error', '')}"
        )
        return record

    record["error"] = (
        envelope["stderr"] or envelope["stdout"] or "verify-document failed"
    ).strip()[-2000:]
    return record


def _lean_cwd(gnn_root_path: Path) -> str:
    """Resolve the FEP_Formal cwd for the subprocess envelope."""
    root = resolve_fep_formal_root()
    return str(root) if root is not None else str(gnn_root_path.parent)


def run_lean_scripts(
    rendered_simulators_dir: Union[str, Path],
    execution_output_dir: Union[str, Path] | None = None,
    recursive_search: bool = True,
    verbose: bool = False,
    timeout: Optional[int] = None,
    *,
    cancel_token: CancelToken | None = None,
) -> bool:
    """Verify every emitted Lean/GNN document under the target directory.

    Mirrors the per-framework runner contract: returns ``True`` when every
    discovered document verifies (or when there is nothing to verify),
    ``False`` when the FEP_Formal checkout is unavailable or any document
    fails. Per-document receipts are written under ``execution_output_dir``.
    When ``timeout`` is given it overrides the per-document ceiling;
    ``None`` keeps ``verify_document``'s default. ``cancel_token`` threads
    into every per-document call; the envelope owns the cooperative checks
    (pre-spawn and mid-flight).
    """
    if resolve_fep_formal_root() is None:
        logger.info(
            "ℹ️ FEP_Formal not available - skipping Lean verification (set %s)",
            FEP_FORMAL_ROOT_ENV,
        )
        return False

    target = Path(rendered_simulators_dir)
    glob = target.rglob if recursive_search else target.glob
    documents = sorted(set(glob("*.lean")))
    documents += sorted(path for path in glob("*.md") if _is_gnn_document(path))

    output_dir = Path(execution_output_dir) if execution_output_dir else target / "lean"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not documents:
        logger.info("ℹ️ No Lean/GNN documents found to verify")
        return True

    all_ok = True
    for document in documents:
        receipt_path = output_dir / f"{document.stem}-receipt.json"
        if timeout is not None:
            record = verify_document(
                document, receipt_path, timeout=timeout, cancel_token=cancel_token
            )
        else:
            record = verify_document(document, receipt_path, cancel_token=cancel_token)
        ok = bool(record.get("success"))
        all_ok = all_ok and ok
        status_icon = "✅" if ok else "❌"
        message = record.get("error", "well-formed")
        if ok or verbose:
            logger.info(f"{status_icon} Lean verification {document.name}: {message}")
        else:
            logger.warning(
                f"{status_icon} Lean verification {document.name}: {message}"
            )
    return all_ok


def _is_gnn_document(path: Path) -> bool:
    """A GNN ``.md`` document carries the required ``GNNSection`` header."""
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:4096]
    except OSError:
        return False
    return "## GNNSection" in head

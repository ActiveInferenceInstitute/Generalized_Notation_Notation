"""Lean verification runner: drives ``fep-formal bridge verify-document``.

Resolves ``FEP_FORMAL_ROOT`` (then the released ``FEP_LEAN_ROOT`` name) or the
canonical ``../fep_formal`` sibling before ``../fep_lean``. It discovers emitted
Lean/GNN documents and verifies each through the FEP_Formal bridge
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

#: Canonical checkout override; FEP_LEAN_ROOT is retained throughout FEP 2.x.
FEP_FORMAL_ROOT_ENV: Final = "FEP_FORMAL_ROOT"
FEP_LEAN_ROOT_ENV: Final = "FEP_LEAN_ROOT"
_FEP_CLI_CONTRACTS: Final = {
    "fep_lean": ("fep_lean", "fep-lean", "fep_lean.cli:main"),
    "fep-formal": ("fep_formal", "fep-formal", "fep_formal.cli:main"),
}

#: Default ceiling for one verify-document invocation (Lean + Mathlib).
_VERIFY_TIMEOUT_SECONDS = 1800


class FepRuntimeConfigurationError(ValueError):
    """The companion's declared source/runtime cannot be safely selected."""


def _fep_cli_contract(
    root: Path, metadata: Mapping[str, Any] | None = None
) -> tuple[str, str]:
    """Select one physical CLI using the two released project identities."""
    try:
        data = (
            metadata
            if metadata is not None
            else tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        )
        project = data.get("project")
        if not isinstance(project, dict):
            raise FepRuntimeConfigurationError("Unsupported FEP project identity")
        name = project.get("name")
        if not isinstance(name, str) or name not in _FEP_CLI_CONTRACTS:
            raise FepRuntimeConfigurationError("Unsupported FEP project identity")
        namespace, executable, target = _FEP_CLI_CONTRACTS[name]
        scripts = project.get("scripts")
        if not isinstance(scripts, dict) or scripts.get(executable) != target:
            raise FepRuntimeConfigurationError(
                "FEP CLI entry point differs from project identity"
            )
        cli = root / "src" / namespace / "cli.py"
        if not cli.is_file() or cli.resolve() != cli:
            raise FepRuntimeConfigurationError("FEP physical CLI source is unavailable")
    except (OSError, tomllib.TOMLDecodeError, UnicodeError):
        raise FepRuntimeConfigurationError(
            "FEP project metadata is missing or malformed"
        ) from None
    return namespace, executable


def resolve_fep_formal_root() -> Path | None:
    """Resolve the canonical checkout and the older released root name.

    An explicit canonical override wins over the older override, even
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
        try:
            _fep_cli_contract(root)
        except FepRuntimeConfigurationError:
            continue
        else:
            return root
    return None


def resolve_fep_lean_root() -> Path | None:
    """Released 2.x alias for :func:`resolve_fep_formal_root`."""
    return resolve_fep_formal_root()


def fep_bridge_executable(root: Path) -> str:
    """Select the CLI declared by the exact released project/script contract."""
    return _fep_cli_contract(root.resolve())[1]


@dataclass(frozen=True, slots=True)
class FepRuntimeInvocation:
    """Only explicit child overrides; unrelated caller settings omitted."""

    command: tuple[str, ...]
    environment: Mapping[str, str]


def fep_runtime_invocation(
    root: Path, arguments: Sequence[str]
) -> FepRuntimeInvocation:
    """Plan offline source execution in an already provisioned companion runtime.

    This reads path/metadata prerequisites; it does not certify an installed
    wheel or invoke a tool. The selected interpreter/source origin is checked
    inside the existing subprocess envelope before the CLI main function runs.
    """
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
    namespace, executable = _fep_cli_contract(root, project)
    if not arguments or arguments[0] != executable:
        raise FepRuntimeConfigurationError(
            "FEP invocation does not select its declared CLI"
        )
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
    bin_dir = environment / ("Scripts" if os.name == "nt" else "bin")
    interpreter = bin_dir / ("python.exe" if os.name == "nt" else "python")
    console = bin_dir / (executable + (".exe" if os.name == "nt" else ""))
    if (
        not interpreter.is_file()
        or not console.is_file()
        or console.resolve().parent != bin_dir.resolve()
        or (
            os.name != "nt"
            and not all(os.access(p, os.X_OK) for p in (interpreter, console))
        )
    ):
        raise FepRuntimeConfigurationError(
            "FEP environment Python or declared console is not provisioned"
        )
    # A single owned child checks the actual interpreter and Source import;
    # there is no extra origin/version probe outside the existing envelope.
    bootstrap = (
        "import importlib,sys\nfrom pathlib import Path\n"
        f"expected={python!r}.split('.')\n"
        "actual=[str(v) for v in sys.version_info[:len(expected)]]\n"
        f"if (actual!=expected or Path(sys.prefix).resolve()!=Path({str(environment)!r}).resolve() "
        f"or Path(sys.executable).absolute()!=Path({str(interpreter)!r}).absolute()):\n"
        " raise RuntimeError('FEP interpreter differs from the declared environment')\n"
        f"cli=importlib.import_module({namespace + '.cli'!r})\n"
        f"if Path(cli.__file__).resolve()!=Path({str(root / 'src' / namespace / 'cli.py')!r}):\n"
        " raise RuntimeError('FEP imported CLI source differs')\n"
        f"sys.argv=[{str(console)!r},*sys.argv[1:]]\n"
        "raise SystemExit(cli.main())"
    )
    source_path = str(root / "src")
    inherited_pythonpath = os.environ.get("PYTHONPATH")
    if inherited_pythonpath:
        source_path += os.pathsep + inherited_pythonpath
    return FepRuntimeInvocation(
        command=(
            "uv",
            "tool",
            "run",
            "--offline",
            "--from",
            f"uv{pin}",
            "uv",
            "run",
            "--offline",
            "--project",
            str(root),
            "--frozen",
            "--no-sync",
            "--python",
            str(interpreter),
            str(interpreter),
            "-B",
            "-c",
            bootstrap,
            *arguments[1:],
        ),
        environment=MappingProxyType(
            {
                "UV_PROJECT_ENVIRONMENT": str(environment),
                "VIRTUAL_ENV": str(environment),
                "UV_PYTHON": str(interpreter),
                "UV_PYTHON_DOWNLOADS": "never",
                "UV_OFFLINE": "1",
                "PYTHONPATH": source_path,
                "PYTHONDONTWRITEBYTECODE": "1",
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
    try:
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

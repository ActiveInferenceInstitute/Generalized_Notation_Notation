"""Dynamic availability probes behind the ``needs_*`` pytest markers.

The zero-skip contract (``tests/test_zero_skip_contracts.py``) bans in-file
``skipif``/``importorskip`` toolchain gates. External-toolchain requirements
are instead declared statically as registered markers (``needs_julia``,
``needs_lean``, ...) and resolved dynamically here: ``tests/conftest.py``
maps every ``needs_*`` marker to the probe below and applies a skip marker
when the toolchain is unavailable.

Probe results are cached per process (``functools.lru_cache``), so expensive
probes (Julia environment imports, the Lean bridge check) run at most once
per pytest worker — and only for tests that survived marker deselection, so
the default suite (``-m "not ... not toolchain"``) never pays for them.
"""

from __future__ import annotations

import functools
import importlib.util
import logging
import os
import shutil
import subprocess  # nosec B404
from collections.abc import Callable


def _module_available(name: str) -> bool:
    """True when *name* is importable in the current interpreter."""
    return importlib.util.find_spec(name) is not None


@functools.lru_cache(maxsize=1)
def julia_binary_ready() -> bool:
    """Julia interpreter is on PATH (parse-only gates)."""
    return shutil.which("julia") is not None


@functools.lru_cache(maxsize=1)
def julia_env_ready() -> bool:
    """Julia is on PATH and the committed framework project environments load.

    Julia on PATH alone is not enough: the committed RxInfer /
    ActiveInference project environments must be instantiated (CI images
    ship julia without RxInfer/ActiveInference.jl).
    """
    if not julia_binary_ready():
        return False
    from gnn.execute.julia_env import check_julia_dependencies

    return bool(
        check_julia_dependencies(
            False, logging.getLogger(__name__), ["rxinfer", "activeinference_jl"]
        )
    )


@functools.lru_cache(maxsize=1)
def lean_ready() -> bool:
    """fep_lean checkout resolves, ``lake`` is on PATH, bridge has verify-document."""
    if shutil.which("lake") is None:
        return False
    from gnn.execute.lean.lean_runner import resolve_fep_lean_root

    root = resolve_fep_lean_root()
    if root is None:
        return False
    try:
        completed = subprocess.run(  # nosec B603 B607
            ["uv", "run", "fep-lean", "bridge", "--help"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "verify-document" in completed.stdout


@functools.lru_cache(maxsize=1)
def cmdstan_ready() -> bool:
    """cmdstanpy is importable and a compiled CmdStan installation resolves."""
    if not _module_available("cmdstanpy"):
        return False
    try:
        import cmdstanpy

        cmdstanpy.cmdstan_path()
    except Exception:  # noqa: BLE001 - any failure means CmdStan is unusable
        return False
    return True


@functools.lru_cache(maxsize=1)
def pkl_ready() -> bool:
    """The ``pkl`` CLI is on PATH (native-eval parser path)."""
    return shutil.which("pkl") is not None


@functools.lru_cache(maxsize=1)
def torch_ready() -> bool:
    """PyTorch is importable (``uv sync --extra torch``)."""
    return _module_available("torch")


@functools.lru_cache(maxsize=1)
def sklearn_ready() -> bool:
    """scikit-learn is importable (``uv sync --extra ml-ai``)."""
    return _module_available("sklearn")


@functools.lru_cache(maxsize=1)
def distributed_ready() -> bool:
    """dask.distributed is importable (``uv sync --extra scaling``)."""
    return _module_available("distributed")


def ray_ready() -> bool:
    """Ray is importable for explicit scaling-extra integration checks."""
    return _module_available("ray")


@functools.lru_cache(maxsize=1)
def cpomdp_ready() -> bool:
    """The selected interpreter imports the cpomdp runtime under a deadline."""
    from gnn.utils.runtime_safety.framework_availability import check_framework

    return check_framework("cpomdp").available


@functools.lru_cache(maxsize=1)
def thrml_ready() -> bool:
    """Released THRML and its required API load under a supervised ceiling."""
    from gnn.utils.runtime_safety.framework_availability import check_framework

    return check_framework("thrml").available


@functools.lru_cache(maxsize=1)
def d2_ready() -> bool:
    """The D2 visualizer module is importable."""
    return _module_available("gnn.advanced_visualization.d2_visualizer")


@functools.lru_cache(maxsize=1)
def d2_cli_ready() -> bool:
    """The ``d2`` system binary is on PATH."""
    return shutil.which("d2") is not None


@functools.lru_cache(maxsize=1)
def pymdp_stack_ready() -> bool:
    """JAX + inferactively-pymdp dev stack passes the integrity probe."""
    from gnn.utils.runtime_safety.jax_stack_validation import jax_pymdp_stack_ok

    return jax_pymdp_stack_ok()


@functools.lru_cache(maxsize=1)
def ollama_ready() -> bool:
    """A local Ollama daemon answers model-list requests.

    Prefers the Python client; falls back to the ``ollama`` CLI when the
    package is not installed. Mirrors the availability check previously
    inlined in ``tests/llm/test_llm_ollama.py``.
    """
    try:
        import ollama

        try:
            ollama.list()
            return True
        except Exception:  # noqa: BLE001 - service not running
            return False
    except ImportError:
        pass
    if shutil.which("ollama") is None:
        return False
    try:
        result = subprocess.run(  # nosec B603 B607
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


@functools.lru_cache(maxsize=1)
def nonroot_posix_ready() -> bool:
    """Running on POSIX as a non-root user (permission-probe requirement)."""
    return os.name == "posix" and (not hasattr(os, "geteuid") or os.geteuid() != 0)


@functools.lru_cache(maxsize=1)
def posix_ready() -> bool:
    """Running on a POSIX platform (process-group kill requirement)."""
    return os.name == "posix"


# Registered ``needs_*`` marker -> (availability probe, skip reason).
# Every key MUST be registered in pytest.ini ``markers``; the pairing is
# pinned by ``tests/test_zero_skip_contracts.py``.
# Ordered so that concrete markers precede the markers they imply; consumed
# by ``tests/conftest.py`` when auto-tagging ``toolchain`` for deselection.
TOOLCHAIN_MARKERS: dict[str, tuple[Callable[[], bool], str]] = {
    "needs_julia": (julia_binary_ready, "Julia binary is not on PATH"),
    "needs_julia_env": (
        julia_env_ready,
        "julia or the committed RxInfer/ActiveInference Julia project "
        "environments are unavailable",
    ),
    "needs_lean": (
        lean_ready,
        "fep_lean toolchain or bridge verify-document op unavailable",
    ),
    "needs_cmdstan": (
        cmdstan_ready,
        "cmdstanpy/CmdStan toolchain is not installed (uv sync --extra stan)",
    ),
    "needs_pkl": (pkl_ready, "pkl CLI not on PATH; native eval path not exercised"),
    "needs_torch": (torch_ready, "torch is not installed (uv sync --extra torch)"),
    "needs_sklearn": (
        sklearn_ready,
        "scikit-learn is not installed (uv sync --extra ml-ai)",
    ),
    "needs_distributed": (
        distributed_ready,
        "dask.distributed is not installed (uv sync --extra scaling)",
    ),
    "needs_cpomdp": (cpomdp_ready, "cpomdp is not installed (uv sync --extra cpomdp)"),
    "needs_thrml": (
        thrml_ready,
        "THRML is not installed or its API is incompatible (uv sync --extra thrml)",
    ),
    "needs_ray": (ray_ready, "Ray is not installed (uv sync --extra scaling)"),
    "needs_d2": (d2_ready, "D2 visualizer module is not importable"),
    "needs_d2_cli": (d2_cli_ready, "D2 CLI not available"),
    "needs_pymdp": (
        pymdp_stack_ready,
        "JAX + inferactively-pymdp>=1.0 required (uv sync --extra dev)",
    ),
    "needs_ollama": (ollama_ready, "Ollama not available locally"),
    "needs_nonroot": (
        nonroot_posix_ready,
        "permission-based probe tests need a non-root POSIX user",
    ),
    "needs_posix": (posix_ready, "POSIX platform required (process-group kill)"),
}

# A marker that implies a weaker sibling gets both markers auto-applied, so
# ``-m`` deselection and skip resolution always see the full requirement set.
MARKER_IMPLICATIONS: dict[str, str] = {
    "needs_julia_env": "needs_julia",
    "needs_d2_cli": "needs_d2",
}

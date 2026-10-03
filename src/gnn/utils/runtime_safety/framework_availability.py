"""Bounded, interpreter-specific framework readiness and honest diagnostics."""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    from gnn.execute.subprocess_envelope import CancelToken

FRAMEWORK_IMPORT_CHECK: Dict[str, Tuple[str, str]] = {
    "jax": ("jax", "uv sync"),
    "numpyro": ("numpyro", "uv sync"),
    "pytorch": ("torch", "uv sync --extra torch"),
    "stan": ("cmdstanpy", "uv sync --extra stan"),
    "discopy": ("discopy", "uv sync"),
    "bnlearn": ("bnlearn", "uv sync --extra bnlearn"),
    "pymdp": ("pymdp", "uv sync"),
    "ngclearn": ("ngclearn", "uv sync --extra ngclearn"),
    "cpomdp": ("cpomdp", "uv sync --extra cpomdp"),
    "thrml": ("thrml", "uv sync --extra thrml"),
}
FRAMEWORK_PROBE_STATEMENT = {"stan": "import cmdstanpy; cmdstanpy.cmdstan_path()"}
FRAMEWORK_REQUIRED_VERSIONS = {"cpomdp": "0.4.4", "thrml": "0.1.4"}
FRAMEWORK_JULIA_PACKAGES = {
    "rxinfer": (
        "RxInfer",
        "JSON",
        "Distributions",
        "StatsBase",
        "Plots",
        "GnnRxInferModels",
    ),
    "activeinference_jl": ("ActiveInference", "JSON", "Distributions", "StatsBase"),
}
DEFAULT_PROBE_TIMEOUT_SECONDS = 30.0
_CONTEXT_PROBE_CACHE: dict[tuple[str, str, str], FrameworkStatus] = {}


@dataclass(frozen=True)
class FrameworkStatus:
    """Availability and its cause; uncertainty never implies package absence."""

    name: str
    available: bool
    missing_module: Optional[str] = None
    install_hint: Optional[str] = None
    reason_code: Optional[str] = None
    reason: Optional[str] = None
    execution_error_type: Optional[str] = None
    cleanup_verified: Optional[bool] = None
    streams_drained: Optional[bool] = None
    runtime_version: Optional[str] = None
    backend_version: Optional[str] = None


# A fixed program; framework/module identifiers are passed as JSON data in argv.
# A marker separates the answer from libraries that print while importing.
_PROBE_MARKER = "GNN_FRAMEWORK_STATUS:"
_PROBE_CODE = """
import importlib, importlib.util, importlib.metadata, json, sys
framework, module_name, required_version = json.loads(sys.argv[1])
answer = {"available": False}
if framework == "ngclearn" and sys.version_info < (3, 12):
    answer.update(reason_code="unsupported_python", reason="ngc-learn requires Python >= 3.12; target interpreter is " + sys.version.split()[0])
elif framework == "cpomdp" and sys.version_info < (3, 11):
    answer.update(reason_code="unsupported_python", reason="cpomdp requires Python >= 3.11; target interpreter is " + sys.version.split()[0])
elif framework == "thrml" and sys.version_info < (3, 10):
    answer.update(reason_code="unsupported_python", reason="THRML requires Python >= 3.10; target interpreter is " + sys.version.split()[0])
else:
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            answer.update(reason_code="missing_module", reason="Dependency not installed: " + module_name, missing_module=module_name)
        else:
            module = importlib.import_module(module_name)
            if framework == "ngclearn":
                for dependency in ("jax", "ngcsimlib", "numpy"):
                    if importlib.util.find_spec(dependency) is None:
                        answer.update(reason_code="missing_module", reason="Dependency not installed: " + dependency, missing_module=dependency)
                        break
                    importlib.import_module(dependency)
            if framework == "thrml":
                installed = importlib.metadata.version(module_name)
                if installed != required_version:
                    answer.update(reason_code="unsupported_version", reason="Required " + module_name + "==" + required_version + "; target interpreter has " + installed)
                for dependency in ("jax", "equinox", "jaxtyping"):
                    if answer.get("reason_code"):
                        break
                    if importlib.util.find_spec(dependency) is None:
                        answer.update(reason_code="missing_module", reason="Dependency not installed: " + dependency, missing_module=dependency)
                        break
                    importlib.import_module(dependency)
                if not answer.get("reason_code"):
                    required_api = ("Block", "BlockGibbsSpec", "FactorSamplingProgram", "SamplingSchedule", "CategoricalNode", "sample_states")
                    if any(not hasattr(module, name) for name in required_api) or any(not hasattr(module.models, name) for name in ("CategoricalEBMFactor", "CategoricalGibbsConditional")):
                        answer.update(reason_code="probe_failed", reason="THRML released categorical API is incomplete")
                    else:
                        jax = importlib.import_module("jax")
                        if not jax.devices():
                            answer.update(reason_code="probe_failed", reason="THRML JAX runtime has no devices")
                        answer["runtime_version"] = jax.__version__
            if answer.get("reason_code"):
                pass
            elif required_version:
                installed_version = importlib.metadata.version(module_name)
                if installed_version != required_version:
                    answer.update(reason_code="unsupported_version", reason="Required " + module_name + "==" + required_version + "; target interpreter has " + installed_version)
                else:
                    answer["available"] = True
                    if framework == "thrml":
                        answer["backend_version"] = installed_version
            elif framework == "stan":
                try:
                    module.cmdstan_path()
                except ValueError:
                    answer.update(reason_code="missing_toolchain", reason="CmdStan toolchain not installed; run cmdstanpy.install_cmdstan()")
                else:
                    answer["available"] = True
            else:
                answer["available"] = True
    except Exception as error:
        if framework == "thrml" and isinstance(error, ModuleNotFoundError) and error.name in {"jax", "jaxlib", "equinox", "jaxtyping", "thrml"}:
            answer.update(reason_code="missing_module", reason="Dependency not installed: " + error.name, missing_module=error.name)
        else:
            answer.update(reason_code="probe_failed", reason="Framework import/toolchain probe failed: " + type(error).__name__)
print("GNN_FRAMEWORK_STATUS:" + json.dumps(answer))
"""

# No Pkg operations: resolve and load only the committed project's packages.
_JULIA_PROBE_CODE = r"""
function json_string(value)
    return "\"" * replace(string(value), "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n", "\r" => "\\r", "\t" => "\\t") * "\""
end
function emit_status(available; code="", reason="", missing="", backend="")
    print("GNN_FRAMEWORK_STATUS:{\"available\":", available,
          ",\"runtime_version\":", json_string(VERSION))
    if !isempty(code)
        print(",\"reason_code\":", json_string(code), ",\"reason\":", json_string(reason))
    end
    if !isempty(missing)
        print(",\"missing_module\":", json_string(missing))
    end
    if !isempty(backend)
        print(",\"backend_version\":", json_string(backend))
    end
    println("}")
end
if VERSION < v"1.10"
    emit_status(false; code="unsupported_version", reason="Committed Julia projects require Julia >= 1.10")
    exit(0)
end
try
    for package in ARGS
        if isnothing(Base.find_package(package))
            emit_status(false; code="missing_module", reason="Dependency not installed in committed Julia project: " * package, missing=package)
            exit(0)
        end
    end
    for package in ARGS
        try
            Base.require(Main, Symbol(package))
        catch error
            emit_status(false; code="probe_failed", reason="Julia package load failed: " * package * " (" * string(nameof(typeof(error))) * ")")
            exit(0)
        end
    end
    backend_module = Base.require(Main, Symbol(first(ARGS)))
    emit_status(true; backend=string(Base.pkgversion(backend_module)))
catch error
    emit_status(false; code="probe_failed", reason="Julia project/package probe failed: " * string(nameof(typeof(error))))
end
"""


def _run_probe_envelope(
    command: list[str],
    *,
    timeout: float,
    env: dict[str, str] | None = None,
    deadline_monotonic: float | None = None,
    cancel_token: CancelToken | None = None,
) -> dict[str, Any]:
    """Lazy supervised child boundary; avoid importing execution at module load."""
    from gnn.execute.subprocess_envelope import run_subprocess_envelope

    options: dict[str, Any] = {
        "timeout": timeout,
        "sandbox": False,
        "env": env,
        "deadline_monotonic": deadline_monotonic,
    }
    if cancel_token is not None:
        options["cancel_token"] = cancel_token
    return run_subprocess_envelope(command, **options)


def check_framework(
    framework: str,
    executor: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    *,
    timeout: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
    deadline_monotonic: float | None = None,
    cancel_token: CancelToken | None = None,
) -> FrameworkStatus:
    """Probe the requested interpreter under a finite deadline, without installs.

    Known Julia backends resolve and load their committed project dependencies.
    Unrelated unknown frameworks retain the historical available verdict.
    """
    is_julia = framework in FRAMEWORK_JULIA_PACKAGES
    if framework not in FRAMEWORK_IMPORT_CHECK and not is_julia:
        return FrameworkStatus(name=framework, available=True)
    if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Framework probe timeout must be positive and finite")
    if cancel_token is not None and cancel_token.cancelled:
        return FrameworkStatus(
            framework,
            False,
            reason_code="probe_cancelled",
            reason=cancel_token.reason or "Framework probe cancelled",
            execution_error_type="Cancelled",
        )
    from gnn.execute.preconditions import current_script_deadline

    scoped_deadline = current_script_deadline()
    if scoped_deadline is not None:
        # Validate caller-provided values before intersecting either ceiling.
        if deadline_monotonic is not None and (
            isinstance(deadline_monotonic, bool)
            or not isinstance(deadline_monotonic, (int, float))
            or not math.isfinite(deadline_monotonic)
        ):
            raise ValueError("Framework probe deadline must be finite and numeric")
        deadline_monotonic = (
            scoped_deadline
            if deadline_monotonic is None
            else min(deadline_monotonic, scoped_deadline)
        )
    if deadline_monotonic is not None:
        if (
            isinstance(deadline_monotonic, bool)
            or not isinstance(deadline_monotonic, (int, float))
            or not math.isfinite(deadline_monotonic)
        ):
            raise ValueError("Framework probe deadline must be finite and numeric")
        timeout = min(timeout, deadline_monotonic - time.monotonic())
        if timeout <= 0:
            return FrameworkStatus(
                framework,
                False,
                reason_code="probe_timeout",
                reason="Script deadline exhausted before framework probe",
                execution_error_type="TimeoutExpired",
            )
    if is_julia:
        from gnn.execute.julia_env import (
            _julia_project_for_framework,
            julia_subprocess_env,
        )
        from gnn.execute.julia_setup import julia_executable

        interpreter = (
            julia_executable() if executor is None or executor == "julia" else executor
        )
        project = _julia_project_for_framework(framework)
        if interpreter is None:
            return FrameworkStatus(
                framework,
                False,
                reason_code="executor_unavailable",
                reason="Julia executable not found on PATH",
            )
        if (
            project is None
            or not (project / "Project.toml").is_file()
            or not (project / "Manifest.toml").is_file()
        ):
            return FrameworkStatus(
                framework,
                False,
                reason_code="missing_toolchain",
                reason="Committed Julia project or manifest is missing",
            )
        hint = f"julia --startup-file=no --project={project} {project / 'setup_environment.jl'}"
        command = [
            interpreter,
            "--startup-file=no",
            "--history-file=no",
            f"--project={project}",
            "-e",
            _JULIA_PROBE_CODE,
            *FRAMEWORK_JULIA_PACKAGES[framework],
        ]
        probe_env = julia_subprocess_env(
            {
                "JULIA_PROJECT": str(project),
                "JULIA_PKG_OFFLINE": "true",
                "JULIA_PKG_PRECOMPILE_AUTO": "0",
            }
        )
    else:
        module_name, hint = FRAMEWORK_IMPORT_CHECK[framework]
        interpreter = executor or sys.executable
        command = [
            interpreter,
            "-c",
            _PROBE_CODE,
            json.dumps(
                [framework, module_name, FRAMEWORK_REQUIRED_VERSIONS.get(framework)]
            ),
        ]
        probe_env = None
    from gnn.pipeline.run_context import current_run_context

    context = current_run_context()
    cache_key = None
    if context is not None:
        cache_key = (context.run_id, framework, interpreter)
        timeout = context.bounded_timeout(timeout)
        if timeout <= 0:
            return FrameworkStatus(
                framework,
                False,
                reason_code="probe_timeout",
                reason="Pipeline deadline exhausted before framework probe",
            )
        if cache_key in _CONTEXT_PROBE_CACHE:
            return _CONTEXT_PROBE_CACHE[cache_key]
        for stale in [key for key in _CONTEXT_PROBE_CACHE if key[0] != context.run_id]:
            del _CONTEXT_PROBE_CACHE[stale]
    probe_options: dict[str, Any] = {"timeout": timeout, "env": probe_env}
    if deadline_monotonic is not None:
        probe_options["deadline_monotonic"] = deadline_monotonic
    if cancel_token is not None:
        probe_options["cancel_token"] = cancel_token
    envelope = _run_probe_envelope(command, **probe_options)
    envelope_error = envelope.get("error_type")
    if envelope_error == "Cancelled":
        return FrameworkStatus(
            framework,
            False,
            reason_code="probe_cancelled",
            reason="Framework probe cancelled; package absence was not established",
            execution_error_type="Cancelled",
            cleanup_verified=envelope.get("cleanup_verified"),
            streams_drained=envelope.get("streams_drained"),
        )
    failed_status = None
    if envelope_error == "ProcessCleanupFailure":
        failed_status = FrameworkStatus(
            framework,
            False,
            reason_code="probe_failed",
            reason="Framework probe cleanup/pipe drain could not be verified; package absence was not established",
            execution_error_type=envelope.get("execution_error_type"),
            cleanup_verified=envelope.get("cleanup_verified"),
            streams_drained=envelope.get("streams_drained"),
        )
    elif envelope_error == "TimeoutExpired":
        failed_status = FrameworkStatus(
            framework,
            False,
            reason_code="probe_timeout",
            reason=f"Framework probe timed out after {timeout:g}s; package absence was not established",
            execution_error_type="TimeoutExpired",
            cleanup_verified=envelope.get("cleanup_verified"),
            streams_drained=envelope.get("streams_drained"),
        )
    elif envelope_error in {"FileNotFoundError", "PermissionError", "OSError"}:
        failed_status = FrameworkStatus(
            framework,
            False,
            reason_code="executor_unavailable",
            reason="Framework probe interpreter could not be started",
            execution_error_type=envelope_error,
        )
    if failed_status is not None:
        if cache_key is not None:
            _CONTEXT_PROBE_CACHE[cache_key] = failed_status
        return failed_status
    answer: dict[str, Any] = {}
    if envelope.get("success") and envelope.get("return_code") == 0:
        for line in reversed(envelope.get("stdout", "").splitlines()):
            if line.startswith(_PROBE_MARKER):
                try:
                    answer = json.loads(line[len(_PROBE_MARKER) :])
                except json.JSONDecodeError:
                    pass
                break
    if not isinstance(answer, dict) or not isinstance(answer.get("available"), bool):
        answer = {
            "available": False,
            "reason_code": "probe_failed",
            "reason": f"Framework probe did not return a valid status (exit {envelope.get('return_code')})",
        }
    available = answer["available"]
    reason_code = answer.get("reason_code")
    install_hint = (
        None
        if available
        or reason_code in {"probe_failed", "probe_timeout", "executor_unavailable"}
        else hint
    )
    if reason_code == "missing_toolchain" and not is_julia:
        install_hint = (
            'uv run python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"'
        )
    if reason_code == "unsupported_python":
        install_hint = (
            "Use Python >= 3.11, then uv sync --extra cpomdp"
            if framework == "cpomdp"
            else "Use Python >= 3.10, then uv sync --extra thrml"
            if framework == "thrml"
            else "Use Python >= 3.12, then uv sync --extra ngclearn"
        )
    status = FrameworkStatus(
        framework,
        available,
        answer.get("missing_module"),
        install_hint,
        reason_code,
        answer.get("reason"),
        execution_error_type=envelope.get("execution_error_type") or envelope_error,
        cleanup_verified=envelope.get("cleanup_verified"),
        streams_drained=envelope.get("streams_drained"),
        runtime_version=answer.get("runtime_version"),
        backend_version=answer.get("backend_version"),
    )
    if cache_key is not None:
        _CONTEXT_PROBE_CACHE[cache_key] = status
    if logger is not None and not available:
        logger.debug("Framework %s: %s", framework, status.reason)
    return status


def is_framework_available(
    framework: str,
    executor: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Compatibility wrapper around the same bounded structured probe."""
    return check_framework(framework, executor=executor, logger=logger).available


__all__ = [
    "FRAMEWORK_IMPORT_CHECK",
    "FRAMEWORK_JULIA_PACKAGES",
    "FRAMEWORK_PROBE_STATEMENT",
    "FrameworkStatus",
    "check_framework",
    "is_framework_available",
]

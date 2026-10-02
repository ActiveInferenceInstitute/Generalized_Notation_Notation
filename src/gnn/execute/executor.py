"""
GNN Executor Module

This module provides the main execution functionality for GNN models,
including script execution, simulation management, and result collection.
"""

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union

if TYPE_CHECKING:
    from gnn.utils.runtime_safety.framework_availability import FrameworkStatus

from .preconditions import execution_precondition as _execution_precondition
from .result_cache import ExecutionResultCache, cache_key_for_script
from .security_gate import check_script_allowed
from .subprocess_envelope import CancelToken, run_subprocess_envelope
from .types import _EXECUTABLE_SUFFIXES, DEFAULT_RUNNER_TIMEOUT_SECONDS


# Lazy runner registry: each backend's runner imports only when a spec is
# resolved or a runner is invoked, keeping module import free of heavy
# optional dependencies (jax, torch, discopy, ...).
@dataclass(frozen=True)
class _RunnerState:
    """Availability verdict plus runner callable for one backend."""

    available: bool
    runner: Any
    diagnosis: "FrameworkStatus | None" = None


def _load_pymdp() -> tuple[bool, Any]:
    from .pymdp.pymdp_runner import run_pymdp_scripts

    return True, run_pymdp_scripts


def _load_rxinfer() -> tuple[bool, Any]:
    from .rxinfer.rxinfer_runner import run_rxinfer_scripts

    return True, run_rxinfer_scripts


def _load_discopy() -> tuple[bool, Any]:
    from .discopy.discopy_executor import run_discopy_analysis

    return True, run_discopy_analysis


def _load_activeinference() -> tuple[bool, Any]:
    from .activeinference_jl.activeinference_runner import run_activeinference_analysis

    return True, run_activeinference_analysis


def _load_jax() -> tuple[bool, Any]:
    from .jax.jax_runner import run_jax_scripts

    return True, run_jax_scripts


def _load_numpyro() -> tuple[bool, Any]:
    from .numpyro.numpyro_runner import run_numpyro_scripts

    return True, run_numpyro_scripts


def _load_pytorch() -> tuple[bool, Any]:
    from .pytorch.pytorch_runner import run_pytorch_scripts

    return True, run_pytorch_scripts


def _load_ngclearn() -> tuple[bool, Any]:
    from .ngclearn.ngclearn_runner import run_ngclearn_scripts

    return True, run_ngclearn_scripts


def _load_thrml() -> tuple[bool, Any]:
    from .thrml.thrml_runner import run_thrml_records

    return True, run_thrml_records


def _load_lean() -> tuple[bool, Any]:
    from .lean.lean_runner import lean_toolchain_available, run_lean_scripts

    return lean_toolchain_available(), run_lean_scripts


def _load_stan() -> tuple[bool, Any]:
    from .stan.stan_runner import run_stan_scripts

    return True, run_stan_scripts


def _load_bnlearn() -> tuple[bool, Any]:
    from .bnlearn.bnlearn_runner import run_bnlearn_scripts

    return True, run_bnlearn_scripts


_RUNNER_LOADERS: dict[str, Callable[[], tuple[bool, Any]]] = {
    "pymdp": _load_pymdp,
    "rxinfer": _load_rxinfer,
    "discopy": _load_discopy,
    "activeinference_jl": _load_activeinference,
    "jax": _load_jax,
    "numpyro": _load_numpyro,
    "pytorch": _load_pytorch,
    "lean": _load_lean,
    "ngclearn": _load_ngclearn,
    "thrml": _load_thrml,
    "stan": _load_stan,
    "bnlearn": _load_bnlearn,
}


def _runner_state(framework_dir_key: str) -> _RunnerState:
    """Resolve live readiness; invocation-scoped probes own any verdict cache."""
    loader = _RUNNER_LOADERS.get(framework_dir_key)
    if loader is None:
        return _RunnerState(False, None)
    from gnn.utils.runtime_safety.framework_availability import (
        FRAMEWORK_IMPORT_CHECK,
        FRAMEWORK_JULIA_PACKAGES,
        FrameworkStatus,
        check_framework,
    )

    diagnosis = None
    if (
        framework_dir_key in FRAMEWORK_IMPORT_CHECK
        or framework_dir_key in FRAMEWORK_JULIA_PACKAGES
    ):
        # Readiness imports run in a supervised child before even the runner
        # module loads. A missing, failed or hanging optional dependency must
        # never reach an unbounded parent import through a runner's imports.
        diagnosis = check_framework(framework_dir_key)
        if not diagnosis.available:
            return _RunnerState(False, None, diagnosis)
    try:
        available, runner = loader()
    except ImportError as error:
        return _RunnerState(
            False,
            None,
            FrameworkStatus(
                framework_dir_key,
                False,
                reason_code="runner_import_failed",
                reason=f"Framework runner could not import: {error}",
                execution_error_type="ImportError",
            ),
        )
    if not available:
        diagnosis = FrameworkStatus(
            framework_dir_key,
            False,
            reason_code="missing_toolchain",
            reason="Framework runner prerequisites are unavailable",
        )
    return _RunnerState(available, runner, diagnosis)


from gnn.utils.logging_utils import (
    log_step_error as log_step_error,
)
from gnn.utils.logging_utils import (
    log_step_start as log_step_start,
)
from gnn.utils.logging_utils import (
    log_step_success as log_step_success,
)
from gnn.utils.logging_utils import (
    log_step_warning as log_step_warning,
)

from .executor_report import (
    _append_framework_result as _append_framework_result,
)
from .executor_report import (
    _check_julia_availability as _check_julia_availability,
)
from .executor_report import (
    _check_python_dependencies as _check_python_dependencies,
)
from .executor_report import (
    _count_framework_execution_records as _count_framework_execution_records,
)
from .executor_report import (
    _create_framework_dirs as _create_framework_dirs,
)
from .executor_report import (
    _execute_configured_frameworks as _execute_configured_frameworks,
)
from .executor_report import (
    _execute_framework_spec as _execute_framework_spec,
)
from .executor_report import (
    _initialize_execution_results as _initialize_execution_results,
)
from .executor_report import (
    _log_execution_outcome as _log_execution_outcome,
)
from .executor_report import (
    _validate_pymdp_script_syntax as _validate_pymdp_script_syntax,
)
from .executor_report import (
    _write_execution_artifacts as _write_execution_artifacts,
)
from .executor_report import (
    _write_execution_report as _write_execution_report,
)
from .executor_report import (
    _write_framework_report_section as _write_framework_report_section,
)
from .executor_report import (
    execute_rendered_simulators as execute_rendered_simulators,
)
from .executor_specs import (
    FRAMEWORK_DIR_NAMES as FRAMEWORK_DIR_NAMES,
)
from .executor_specs import (
    ExecutorFrameworkSpec as ExecutorFrameworkSpec,
)
from .executor_specs import (
    _framework_specs as _framework_specs,
)
from .executor_specs import (
    _run_stan_registry as _run_stan_registry,
)

logger = logging.getLogger(__name__)

# Shared execution result cache. Off by default (GNN_EXEC_CACHE opt-in);
# GNNExecutor and execute_script_safely fall back to it when no explicit
# cache instance is passed.
_EXECUTION_RESULT_CACHE = ExecutionResultCache()


# Provide a simple hardware detection function used in tests for patching
class _HardwareContainmentFailure(RuntimeError):
    """Hardware readiness cannot conceal an unverified child cleanup."""

    def __init__(self, envelope: Dict[str, Any]) -> None:
        super().__init__("Hardware probe cleanup or pipe drain could not be verified")
        self.envelope = envelope


def get_available_hardware(
    *,
    deadline_monotonic: Optional[float] = None,
    cancel_token: Optional[CancelToken] = None,
) -> list[str]:
    """Inspect actual JAX devices in a bounded child; CPU is a best-effort fallback."""
    envelope = run_subprocess_envelope(
        [
            sys.executable,
            "-c",
            "import json,jax; print('GNN_DEVICES:' + json.dumps(sorted({device.platform for device in jax.devices()})))",
        ],
        timeout=5,
        sandbox=False,
        deadline_monotonic=deadline_monotonic,
        cancel_token=cancel_token,
    )
    if (
        envelope.get("cleanup_verified") is False
        or envelope.get("streams_drained") is False
        or envelope.get("error_type") == "ProcessCleanupFailure"
    ):
        raise _HardwareContainmentFailure(envelope)
    if envelope.get("success"):
        for line in reversed(envelope.get("stdout", "").splitlines()):
            if line.startswith("GNN_DEVICES:"):
                try:
                    devices = json.loads(line[len("GNN_DEVICES:") :])
                except json.JSONDecodeError:
                    break
                if (
                    isinstance(devices, list)
                    and devices
                    and all(isinstance(device, str) for device in devices)
                ):
                    return devices
    logger.debug("Device probe did not establish hardware; using CPU fallback")
    return ["cpu"]


class GNNExecutor:
    """
    Main executor for GNN model simulations and scripts.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        cache: Optional[ExecutionResultCache] = None,
    ) -> None:
        """
        Initialize the GNN executor.

        Args:
            output_dir: Directory for execution outputs
            cache: Execution result cache for subprocess dispatches
                (None → shared module-level cache instance)
        """
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Default to a subdirectory within the project root
            self.output_dir = (
                Path(__file__).parent.parent.parent / "output" / "12_execute_output"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.execution_log: list[dict[str, Any]] = []
        self._cache = cache if cache is not None else _EXECUTION_RESULT_CACHE

    def execute_gnn_model(
        self,
        model_path: str,
        execution_type: str = "pymdp",
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        cancel_token: Optional[CancelToken] = None,
    ) -> Dict[str, Any]:
        """
        Execute a GNN model with the specified execution type.

        Args:
            model_path: Path to the GNN model or rendered script
            execution_type: Type of execution (pymdp, rxinfer, discopy, etc.)
            options: Additional execution options
            cancel_token: Optional cooperative cancellation token threaded
                into every subprocess dispatch, lean included — the token
                fires pre-spawn and mid-flight via the shared envelope; the
                fep-lean bridge process itself has no in-process
                cooperative-cancel protocol yet (held fep-side substance)

        Returns:
            Dictionary with execution results
        """
        precondition = _execution_precondition(timeout, cancel_token)
        if precondition is not None:
            return precondition
        from .preconditions import script_execution_scope

        automatic_timeout = {
            "pymdp": DEFAULT_RUNNER_TIMEOUT_SECONDS,
            "activeinference_jl": 600,
            "stan": 1800,
            "bnlearn": 1800,
            "lean": 1800,
        }.get(execution_type, 300)
        invocation_timeout = timeout if timeout is not None else automatic_timeout
        # Lean's established options timeout remains an explicit caller override.
        if execution_type == "lean" and (options or {}).get("timeout") is not None:
            invocation_timeout = (options or {})["timeout"]
        precondition = _execution_precondition(invocation_timeout, cancel_token)
        if precondition is not None:
            return precondition
        start_time = time.monotonic()
        with script_execution_scope(invocation_timeout):
            try:
                # SC-1: pre-execution security gate — same shared helper as the
                # Step 12 processor path. GNNExecutor runs rendered scripts; the
                # MCP tools (execute_gnn_model_mcp → execute_simulation_from_gnn)
                # reach execution only through this dispatch, so one gate here
                # covers every script GNNExecutor is about to run.
                if Path(model_path).suffix.lower() in _EXECUTABLE_SUFFIXES:
                    gate_verdict = check_script_allowed(Path(model_path))
                elif (
                    execution_type == "lean"
                    and Path(model_path).suffix.lower() == ".md"
                ):
                    # Lean dispatch executes .md documents through the fep-lean
                    # bridge (lean_runner.verify_document), so they are gate-
                    # checked: the shared helper scans their fenced code blocks
                    # with the same rendered-script verdict machinery.
                    gate_verdict = check_script_allowed(Path(model_path))
                else:
                    # Model sources that this dispatch only parses as data (e.g.
                    # .md under pymdp/jax) are not executed here; the scanner only
                    # accepts executable scripts, so skip the gate.
                    gate_verdict = {"ok": True, "overridden": False, "blocked": []}
                if gate_verdict["overridden"]:
                    logger.warning(
                        "GNN_ALLOW_UNSAFE_EXEC set: pre-execution security gate "
                        "bypassed for %s (trusted-local use only)",
                        model_path,
                    )
                if not gate_verdict["ok"]:
                    return {
                        "success": False,
                        "error": (
                            f"Pre-execution security gate blocked {model_path}: "
                            f"{gate_verdict['reason']}"
                        ),
                        "error_type": gate_verdict.get(
                            "error_type", "SecurityGateBlocked"
                        ),
                        "security_findings": gate_verdict["blocked"],
                        "execution_type": execution_type,
                        "model_path": model_path,
                    }

                if execution_type == "pymdp":
                    result = self._execute_pymdp_script(
                        model_path, options, timeout=timeout, cancel_token=cancel_token
                    )
                elif execution_type == "rxinfer":
                    result = self._execute_rxinfer_config(
                        model_path, options, timeout=timeout, cancel_token=cancel_token
                    )
                elif execution_type == "discopy":
                    result = self._execute_discopy_diagram(
                        model_path, options, timeout=timeout, cancel_token=cancel_token
                    )
                elif execution_type == "jax":
                    result = self._execute_jax_script(
                        model_path, options, timeout=timeout, cancel_token=cancel_token
                    )
                elif execution_type == "lean":
                    # Cancellable GNN-side: the token rides into the fep-lean
                    # bridge dispatch through the shared envelope (pre-spawn and
                    # mid-flight cooperative checks). The bridge process itself
                    # has no in-process cooperative protocol yet — that is the
                    # held fep-side substance.
                    result = self._execute_lean_verification(
                        model_path, options, timeout=timeout, cancel_token=cancel_token
                    )
                elif execution_type == "activeinference_jl":
                    result = self._execute_activeinference_script(
                        model_path, options, timeout=timeout, cancel_token=cancel_token
                    )
                elif execution_type == "numpyro":
                    result = self._execute_numpyro_script(
                        model_path, options, timeout=timeout, cancel_token=cancel_token
                    )
                elif execution_type == "pytorch":
                    result = self._execute_pytorch_script(
                        model_path, options, timeout=timeout, cancel_token=cancel_token
                    )
                elif execution_type == "ngclearn":
                    result = self._execute_ngclearn_script(
                        model_path, options, timeout=timeout, cancel_token=cancel_token
                    )
                elif execution_type == "stan":
                    result = self._execute_stan_script(
                        model_path, options, timeout=timeout, cancel_token=cancel_token
                    )
                elif execution_type == "bnlearn":
                    result = self._execute_bnlearn_script(
                        model_path, options, timeout=timeout, cancel_token=cancel_token
                    )
                elif execution_type == "thrml":
                    from .thrml.thrml_runner import execute_thrml_script

                    result = execute_thrml_script(
                        model_path,
                        output_dir=(options or {}).get("output_dir"),
                        timeout=timeout if timeout is not None else 300,
                        cancel_token=cancel_token,
                    )
                elif execution_type == "cpomdp":
                    from .preconditions import script_readiness

                    local_timeout = timeout if timeout is not None else 300
                    deadline, readiness_error = script_readiness(
                        "cpomdp", local_timeout
                    )
                    precondition = _execution_precondition(
                        local_timeout, cancel_token, deadline_monotonic=deadline
                    )
                    if precondition is not None:
                        result = precondition
                    elif readiness_error is not None:
                        result = readiness_error
                    else:
                        output = Path(
                            (options or {}).get("output_dir") or Path(model_path).parent
                        ).resolve()
                        output.mkdir(parents=True, exist_ok=True)
                        result = run_subprocess_envelope(
                            [sys.executable, str(Path(model_path).resolve())],
                            env={"CPOMDP_OUTPUT_DIR": str(output)},
                            cwd=str(output),
                            timeout=local_timeout,
                            deadline_monotonic=deadline,
                            cancel_token=cancel_token,
                        )
                else:
                    result = {
                        "success": False,
                        "error": f"Unsupported execution type: {execution_type}",
                    }

                result["execution_type"] = execution_type
                result["model_path"] = model_path
                # Optional diagnostics share the invocation ceiling and containment.
                diagnostic_precondition = _execution_precondition(5, cancel_token)
                if (
                    result.get("success")
                    and not result.get("cache_hit")
                    and diagnostic_precondition is not None
                ):
                    result.setdefault("execution_device", "unknown")
                    result["execution_device_note"] = (
                        "Hardware diagnostic omitted: "
                        + str(diagnostic_precondition.get("error"))
                    )
                else:
                    try:
                        devices = (
                            get_available_hardware()
                            if result.get("success")
                            and not result.get("cache_hit")
                            and cancel_token is None
                            else get_available_hardware(cancel_token=cancel_token)
                            if result.get("success") and not result.get("cache_hit")
                            else ["cpu"]
                        )
                        result.setdefault(
                            "execution_device", devices[0] if devices else "cpu"
                        )
                    except _HardwareContainmentFailure as error:
                        result["execution_result_success"] = bool(result.get("success"))
                        result.update(
                            success=False,
                            status="failed",
                            error_type="ProcessCleanupFailure",
                            error=str(error),
                            execution_device="unknown",
                        )
                        result["hardware_probe"] = {
                            key: error.envelope.get(key)
                            for key in (
                                "error_type",
                                "execution_error_type",
                                "containment",
                                "cleanup_verified",
                                "streams_drained",
                                "duration_seconds",
                            )
                        }
                        for key in ("cleanup_verified", "streams_drained"):
                            result["execution_" + key] = result.get(key)
                            result[key] = error.envelope.get(key)
                    except Exception as error:
                        logger.debug(
                            "Device discovery failed; falling back to cpu: %s", error
                        )
                        result.setdefault("execution_device", "cpu")
                        result["execution_device_fallback"] = (
                            f"cpu: device discovery failed ({error})"
                        )

                result["execution_time"] = time.monotonic() - start_time
                # Log execution
                self.execution_log.append(result)

                return result

            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "execution_type": execution_type,
                    "model_path": model_path,
                }

    def run_simulation(self, simulation_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a simulation based on configuration.

        Args:
            simulation_config: Configuration dictionary for the simulation

        Returns:
            Dictionary with simulation results
        """
        try:
            model_path = simulation_config.get("model_path")
            execution_type = simulation_config.get("execution_type", "pymdp")
            options = simulation_config.get("options", {})

            if not model_path:
                return {
                    "success": False,
                    "error": "No model path specified in simulation config",
                }

            return self.execute_gnn_model(
                model_path,
                execution_type,
                options,
                timeout=simulation_config.get("timeout"),
            )

        except Exception as e:
            return {"success": False, "error": str(e), "error_type": type(e).__name__}

    def generate_execution_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate an execution report from the execution log.

        Args:
            output_file: Path for the output report file

        Returns:
            Path to the generated report
        """
        if not output_file:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"execution_report_{timestamp}.json"
        else:
            output_path = Path(output_file)

        report_data: dict[str, Any] = {
            "execution_summary": {
                "total_executions": len(self.execution_log),
                "successful_executions": sum(
                    1 for r in self.execution_log if r.get("success", False)
                ),
                "failed_executions": sum(
                    1 for r in self.execution_log if not r.get("success", False)
                ),
                "total_execution_time": sum(
                    r.get("execution_time", 0) for r in self.execution_log
                ),
            },
            "execution_details": self.execution_log,
        }

        try:
            with open(output_path, "w") as f:
                json.dump(report_data, f, indent=2)
        except OSError as e:
            raise RuntimeError(
                f"Failed to write execution report to {output_path}: {e}"
            ) from e

        return str(output_path)

    def _execute_lean_verification(
        self,
        model_path: str,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        *,
        cancel_token: Optional[CancelToken] = None,
    ) -> Dict[str, Any]:
        """Verify one document via the fep_lean bridge (contract v0.6).

        ``cancel_token`` forwards into the envelope-backed dispatch; a fired
        token cancels the run pre-spawn or mid-flight GNN-side (the fep-lean
        bridge process itself has no in-process cooperative protocol).
        """
        precondition = _execution_precondition(timeout, cancel_token)
        if precondition is not None:
            return precondition
        state = _runner_state("lean")
        if not state.available or state.runner is None:
            return {"success": False, "error": "fep_lean unavailable"}

        from .lean.lean_runner import verify_document

        opts = options or {}
        return verify_document(
            model_path,
            opts.get("receipt"),
            model=opts.get("model", "finite"),
            fail_on_warnings=bool(opts.get("fail_on_warnings", True)),
            timeout=opts.get("timeout", timeout if timeout is not None else 1800),
            cancel_token=cancel_token,
        )

    def _dispatch_cache_lookup(
        self, interpreter: str, content_path: Union[str, Path]
    ) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Consult the execution cache before one subprocess dispatch.

        The key covers the exact argv ``[interpreter, content_path]``: the
        interpreter and the hashed content of the script/config file, with
        no cwd/env overrides and captured output (the dispatch defaults).

        Returns:
            (cache_key, cached_envelope); cache_key is None when the cache
            is disabled, cached_envelope is None on miss.
        """
        from gnn.pipeline.run_context import current_run_context

        if not self._cache.enabled or current_run_context() is not None:
            return None, None
        cache_key = cache_key_for_script(content_path, interpreter=interpreter)
        return cache_key, self._cache.lookup(cache_key)

    def _store_dispatch_envelope(
        self, cache_key: Optional[str], envelope: Dict[str, Any]
    ) -> None:
        """Store a successful dispatch envelope under its cache key."""
        if cache_key is not None and envelope.get("success"):
            self._cache.store(cache_key, envelope)

    def _execute_pymdp_script(
        self,
        script_path: str,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        *,
        cancel_token: Optional[CancelToken] = None,
    ) -> Dict[str, Any]:
        """Execute a PyMDP script with graceful recovery for tests."""
        precondition = _execution_precondition(timeout, cancel_token)
        if precondition is not None:
            return precondition
        script = Path(script_path)
        if script.suffix.lower() not in {".py"}:
            return {
                "success": False,
                "status": "skipped",
                "skipped": True,
                "error_type": "RenderedScriptRequired",
                "stdout": f"Input {script.name} requires rendering before simulation.",
                "stderr": "",
                "return_code": -1,
            }
        cache_key, cached = self._dispatch_cache_lookup(sys.executable, script_path)
        if cached is not None:
            cached["cache_hit"] = True
            return cached
        envelope = run_subprocess_envelope(
            [sys.executable, script_path],
            timeout=timeout if timeout is not None else DEFAULT_RUNNER_TIMEOUT_SECONDS,
            cancel_token=cancel_token,
        )
        self._store_dispatch_envelope(cache_key, envelope)
        return envelope

    def _execute_rxinfer_config(
        self,
        config_path: str,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        *,
        cancel_token: Optional[CancelToken] = None,
    ) -> Dict[str, Any]:
        """Execute an RxInfer.jl script or TOML config via the committed runner.

        Both .jl and .toml inputs route through ``execute_rxinfer_script``
        (the committed RxInfer.jl project environment); this dispatch
        synthesizes its envelope from the runner's verdict and the evidence
        sidecars it persists beside the script.
        """
        precondition = _execution_precondition(timeout, cancel_token)
        if precondition is not None:
            return precondition
        cache_key, cached = self._dispatch_cache_lookup("julia", config_path)
        if cached is not None:
            cached["cache_hit"] = True
            return cached
        from .rxinfer.rxinfer_runner import execute_rxinfer_script

        script = Path(config_path)
        success = execute_rxinfer_script(
            script,
            timeout=timeout if timeout is not None else 300,
            cancel_token=cancel_token,
        )
        envelope = _synthesize_rxinfer_envelope(script, success)
        self._store_dispatch_envelope(cache_key, envelope)
        return envelope

    def _execute_discopy_diagram(
        self,
        diagram_path: str,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        *,
        cancel_token: Optional[CancelToken] = None,
    ) -> Dict[str, Any]:
        """Execute a DisCoPy diagram."""
        precondition = _execution_precondition(timeout, cancel_token)
        if precondition is not None:
            return precondition
        cache_key, cached = self._dispatch_cache_lookup(sys.executable, diagram_path)
        if cached is not None:
            cached["cache_hit"] = True
            return cached
        envelope = run_subprocess_envelope(
            [sys.executable, diagram_path],
            timeout=timeout if timeout is not None else 300,
            cancel_token=cancel_token,
        )
        self._store_dispatch_envelope(cache_key, envelope)
        return envelope

    def _execute_jax_script(
        self,
        script_path: str,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        *,
        cancel_token: Optional[CancelToken] = None,
    ) -> Dict[str, Any]:
        """Execute a JAX script."""
        precondition = _execution_precondition(timeout, cancel_token)
        if precondition is not None:
            return precondition
        cache_key, cached = self._dispatch_cache_lookup(sys.executable, script_path)
        if cached is not None:
            cached["cache_hit"] = True
            return cached
        envelope = run_subprocess_envelope(
            [sys.executable, script_path],
            timeout=timeout if timeout is not None else 300,
            cancel_token=cancel_token,
        )
        self._store_dispatch_envelope(cache_key, envelope)
        return envelope

    def _execute_python_script(
        self,
        script_path: str,
        timeout: Optional[int] = None,
        *,
        cancel_token: Optional[CancelToken] = None,
    ) -> Dict[str, Any]:
        """Run one rendered Python framework script under ``sys.executable``.

        Shared by the numpyro, pytorch, and ngclearn dispatches; mirrors the
        JAX envelope contract.
        """
        precondition = _execution_precondition(timeout, cancel_token)
        if precondition is not None:
            return precondition
        cache_key, cached = self._dispatch_cache_lookup(sys.executable, script_path)
        if cached is not None:
            cached["cache_hit"] = True
            return cached
        envelope = run_subprocess_envelope(
            [sys.executable, script_path],
            timeout=timeout if timeout is not None else 300,
            cancel_token=cancel_token,
        )
        self._store_dispatch_envelope(cache_key, envelope)
        return envelope

    def _execute_numpyro_script(
        self,
        script_path: str,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        *,
        cancel_token: Optional[CancelToken] = None,
    ) -> Dict[str, Any]:
        """Execute a NumPyro script."""
        precondition = _execution_precondition(timeout, cancel_token)
        if precondition is not None:
            return precondition
        return self._execute_python_script(
            script_path, timeout, cancel_token=cancel_token
        )

    def _execute_pytorch_script(
        self,
        script_path: str,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        *,
        cancel_token: Optional[CancelToken] = None,
    ) -> Dict[str, Any]:
        """Execute a PyTorch script."""
        precondition = _execution_precondition(timeout, cancel_token)
        if precondition is not None:
            return precondition
        return self._execute_python_script(
            script_path, timeout, cancel_token=cancel_token
        )

    def _execute_ngclearn_script(
        self,
        script_path: str,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        *,
        cancel_token: Optional[CancelToken] = None,
    ) -> Dict[str, Any]:
        """Execute an ngc-learn script."""
        precondition = _execution_precondition(timeout, cancel_token)
        if precondition is not None:
            return precondition
        return self._execute_python_script(
            script_path, timeout, cancel_token=cancel_token
        )

    def _execute_activeinference_script(
        self,
        script_path: str,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        *,
        cancel_token: Optional[CancelToken] = None,
    ) -> Dict[str, Any]:
        """Execute a rendered ActiveInference.jl script."""
        precondition = _execution_precondition(timeout, cancel_token)
        if precondition is not None:
            return precondition
        script = Path(script_path)
        project_dir = Path(__file__).parent / "activeinference_jl"
        cache_key, cached = self._dispatch_cache_lookup("julia", script_path)
        if cached is not None:
            cached["cache_hit"] = True
            return cached
        from .julia_env import julia_subprocess_env

        env = julia_subprocess_env()
        env.setdefault("JULIA_PROJECT", str(project_dir))
        envelope = run_subprocess_envelope(
            ["julia", f"--project={project_dir}", str(script)],  # nosec B607 B603
            timeout=timeout if timeout is not None else 600,
            env=env,
            cwd=str(script.parent),
            cancel_token=cancel_token,
        )
        self._store_dispatch_envelope(cache_key, envelope)
        return envelope

    def _execute_stan_script(
        self,
        script_path: str,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        *,
        cancel_token: Optional[CancelToken] = None,
    ) -> Dict[str, Any]:
        """Execute a rendered Stan driver; skip cleanly without cmdstanpy."""
        precondition = _execution_precondition(timeout, cancel_token)
        if precondition is not None:
            return precondition
        from .stan.stan_runner import execute_stan_script

        opts = options or {}
        output_dir = opts.get("output_dir") or Path(script_path).parent
        return execute_stan_script(
            script_path,
            output_dir,
            timeout=timeout if timeout is not None else 1800,
            cancel_token=cancel_token,
        )

    def _execute_bnlearn_script(
        self,
        script_path: str | None = None,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        *,
        cancel_token: Optional[CancelToken] = None,
    ) -> Dict[str, Any]:
        """Run the maintained language-aware bnlearn backend with skip receipts."""
        precondition = _execution_precondition(timeout, cancel_token)
        if precondition is not None:
            return precondition
        if script_path is None:
            return {
                "success": False,
                "skipped": True,
                "status": "skipped",
                "error": "Supply a rendered bnlearn script; output uses BNLEARN_OUTPUT_DIR",
            }
        from .bnlearn.bnlearn_runner import execute_bnlearn_script

        source = Path(script_path).resolve()
        output = Path((options or {}).get("output_dir") or source.parent).resolve()
        return execute_bnlearn_script(
            source,
            output,
            timeout=timeout if timeout is not None else 1800,
            cancel_token=cancel_token,
        )

    def execute_simulation_from_gnn(
        self,
        gnn_file: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute a simulation from a GNN file path."""
        gnn_path = Path(gnn_file) if not isinstance(gnn_file, Path) else gnn_file
        out_dir = Path(output_dir) if output_dir is not None else self.output_dir
        sim_cfg: dict[str, Any] = {
            "model_path": str(gnn_path),
            "execution_type": "pymdp",
            "options": {"output_dir": str(out_dir)},
        }
        if timeout is not None:
            sim_cfg["timeout"] = timeout
        return self.run_simulation(sim_cfg)


def execute_gnn_model(
    model_path: str,
    execution_type: Union[str, Path] = "pymdp",
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to execute a GNN model.

    Args:
        model_path: Path to the GNN model or rendered script
        execution_type: Type of execution
        options: Additional execution options

    Returns:
        Dictionary with execution results
    """
    normalized_execution_type: str = "pymdp"
    normalized_options = options

    exec_type_obj: object = execution_type
    if isinstance(exec_type_obj, Path):
        normalized_options = dict(options or {})
        normalized_options.setdefault("output_dir", str(exec_type_obj))
    elif isinstance(exec_type_obj, str):
        normalized_execution_type = exec_type_obj
    else:
        normalized_options = dict(options or {})
        normalized_options.setdefault("output_dir", str(exec_type_obj))

    executor = GNNExecutor()
    result = executor.execute_gnn_model(
        model_path, normalized_execution_type, normalized_options
    )
    result.setdefault("status", "SUCCESS" if result.get("success") else "FAILED")
    return result


def run_simulation(simulation_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to run a simulation.

    Args:
        simulation_config: Configuration dictionary for the simulation

    Returns:
        Dictionary with simulation results
    """
    executor = GNNExecutor()
    return executor.run_simulation(simulation_config)


def generate_execution_report(
    execution_log: List[Dict[str, Any]], output_file: Optional[str] = None
) -> str:
    """
    Convenience function to generate an execution report.

    Args:
        execution_log: List of execution results
        output_file: Path for the output report file

    Returns:
        Path to the generated report
    """
    executor = GNNExecutor()
    executor.execution_log = execution_log
    return executor.generate_execution_report(output_file)


def _read_text_sidecar(path: Path) -> str:
    """Read a persisted sidecar file, or return an empty string if absent."""
    try:
        return path.read_text()
    except OSError:
        return ""


def _synthesize_rxinfer_envelope(script_path: Path, success: bool) -> Dict[str, Any]:
    """Build a dispatch envelope from ``execute_rxinfer_script``'s verdict.

    The committed runner persists ``{stem}_stdout.txt`` / ``{stem}_stderr.txt``
    and an ``{stem}_execution_log.json`` beside the script; stdout/stderr are
    read back from those sidecars and the elapsed time from the log when
    present.
    """
    stem = script_path.stem
    envelope: Dict[str, Any] = {
        "success": bool(success),
        "return_code": 0 if success else 1,
        "stdout": _read_text_sidecar(script_path.parent / f"{stem}_stdout.txt"),
        "stderr": _read_text_sidecar(script_path.parent / f"{stem}_stderr.txt"),
    }
    try:
        log = json.loads(
            _read_text_sidecar(script_path.parent / f"{stem}_execution_log.json")
        )
        envelope["elapsed_seconds"] = float(log["elapsed_seconds"])
    except (ValueError, KeyError, TypeError):
        envelope["elapsed_seconds"] = None
        envelope["elapsed_seconds_note"] = (
            "rxinfer execution log sidecar missing or unparseable"
        )
    return envelope


def list_frameworks() -> list[dict[str, Any]]:
    """Introspect the executor framework registry.

    Returns one record per registered backend with its key, the
    ``*_executions`` result key, and whether the backend's runner is currently
    importable. Useful for CLI/MCP diagnostics and tests that want to assert
    the registry shape without importing the private ``_framework_specs``
    helper.
    """
    from dataclasses import asdict

    return [
        {
            "framework": spec.framework_dir_key,
            "result_key": spec.result_key,
            "available": bool(spec.available),
            "operation": spec.operation_name,
            "readiness": asdict(spec.diagnosis)
            if spec.diagnosis is not None
            else {
                "framework": spec.framework_dir_key,
                "available": spec.available,
                "reason_code": "ready" if spec.available else "runner_unavailable",
            },
        }
        for spec in _framework_specs()
    ]


def execute_script_safely(
    script_path: Union[str, Path],
    timeout: float | None = 3600,
    capture_output: bool = True,
    cwd: Optional[Union[str, Path]] = None,
    env: Optional[Dict[str, str]] = None,
    args: Optional[Sequence[str]] = None,
    cancel_token: Optional[CancelToken] = None,
    cache: Optional[ExecutionResultCache] = None,
    deadline_monotonic: Optional[float] = None,
) -> Dict[str, Any]:
    """Execute a Python script via ``subprocess.run`` with a structured envelope.

    Returns a uniform dict regardless of the failure mode so callers never have
    to distinguish between a missing file, a dependency error, a timeout, and a
    non-zero exit code.

    Args:
        script_path:    Path to the ``.py`` script to execute.
        timeout:        Wall-clock timeout in seconds (default ``3600``).
        capture_output: If True, capture stdout/stderr; otherwise stream to the
                        parent process.
        cwd:            Working directory for the subprocess.
        env:            Environment variables override (merged into ``os.environ``).
        args:           Extra command-line arguments passed to the script.
        cancel_token:   Optional cooperative cancellation token; when cancelled
                        mid-flight the run is group-killed and reported as
                        ``error_type="Cancelled"``.
        deadline_monotonic: Absolute monotonic deadline including readiness and cleanup.
        cache:          Execution result cache; None falls back to the shared
                        module instance (active only when ``GNN_EXEC_CACHE`` is
                        truthy or the instance was built with ``enabled=True``).
                        Successful envelopes are cached; a cache hit short-
                        circuits before the spawn and is flagged ``cache_hit``.

    Returns:
        Dict with keys:
            - ``success`` (bool): True iff the script exited with return code 0.
            - ``script_path`` (str): Resolved script path.
            - ``return_code`` (int): Subprocess exit code (``-1`` if not started).
            - ``stdout`` (str): Captured stdout (empty if ``capture_output`` is False).
            - ``stderr`` (str): Captured stderr (empty if ``capture_output`` is False).
            - ``duration_seconds`` (float): Wall-clock execution time.
            - ``error`` (str, optional): Populated on failure.
            - ``error_type`` (str, optional): Exception class name on failure
              (``"SecurityGateBlocked"`` when the pre-exec gate denies the
              script; ``"SandboxUnavailable"`` when ``GNN_SANDBOX=require``
              found no backend).
            - ``sandbox_mode``/``sandboxed``: GNN_SANDBOX mode and whether the
              command was actually wrapped (set by the shared envelope).
            - ``cache_hit`` (bool, optional): True when the envelope was served
              from the execution cache instead of a fresh spawn.
            - ``cancelled`` (bool): True when the run was cancelled via the
              cancel token (additive envelope key, False on every other path).
    """
    precondition = _execution_precondition(
        timeout, cancel_token, deadline_monotonic=deadline_monotonic
    )
    if precondition is not None:
        precondition["script_path"] = str(script_path)
        return precondition
    script = Path(script_path)
    if not script.exists():
        return {
            "success": False,
            "script_path": str(script),
            "return_code": -1,
            "stdout": "",
            "stderr": "",
            "duration_seconds": 0.0,
            "error": f"Script not found: {script}",
            "error_type": "FileNotFoundError",
        }
    if script.suffix.lower() != ".py":
        return {
            "success": False,
            "script_path": str(script),
            "return_code": -1,
            "stdout": "",
            "stderr": "",
            "duration_seconds": 0.0,
            "error": (
                f"execute_script_safely only runs Python scripts; got suffix "
                f"{script.suffix!r}"
            ),
            "error_type": "ValueError",
        }

    # SEC-R2: the registry runners (pymdp/jax/numpyro/pytorch/discopy) funnel
    # every rendered-script execution through this helper, so the
    # pre-execution security gate applies here before anything runs.
    gate_verdict = check_script_allowed(script)
    if gate_verdict["overridden"]:
        logger.warning(
            "GNN_ALLOW_UNSAFE_EXEC set: pre-execution security gate "
            "bypassed for %s (trusted-local use only)",
            script,
        )
    if not gate_verdict["ok"]:
        logger.error(
            "Pre-execution security gate blocked %s: %s",
            script,
            gate_verdict["reason"],
        )
        return {
            "success": False,
            "script_path": str(script),
            "return_code": -1,
            "stdout": "",
            "stderr": "",
            "duration_seconds": 0.0,
            "error": (
                f"Pre-execution security gate blocked {script}: "
                f"{gate_verdict['reason']}"
            ),
            "error_type": gate_verdict.get("error_type", "SecurityGateBlocked"),
            "security_findings": gate_verdict["blocked"],
        }

    # Sandboxing (SEC-R2, Step-12 GNN_SANDBOX semantics) lives in the shared
    # envelope: sandbox=True applies the prefix, emits the
    # sandbox_disabled/active receipts, and refuses on require-without-backend.
    script_args = list(args) if args is not None else []
    effective_cache = cache if cache is not None else _EXECUTION_RESULT_CACHE
    cache_key: Optional[str] = None
    from gnn.pipeline.run_context import current_run_context

    if effective_cache.enabled and current_run_context() is None:
        # Cache consult happens AFTER the security gate, so gate-blocked
        # scripts can never be served from (or written into) the cache.
        cache_key = cache_key_for_script(
            script,
            interpreter=sys.executable,
            args=script_args,
            cwd=str(cwd) if cwd is not None else None,
            env_overrides=env,
            capture_output=capture_output,
        )
        cached_envelope = effective_cache.lookup(cache_key)
        if cached_envelope is not None:
            cached_envelope["cache_hit"] = True
            return cached_envelope

    envelope = run_subprocess_envelope(
        [sys.executable, str(script), *script_args],
        timeout=timeout,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        capture_output=capture_output,
        cancel_token=cancel_token,
        deadline_monotonic=deadline_monotonic,
    )
    envelope["script_path"] = str(script)
    # Failures, timeouts, and cancels are never stored.
    if cache_key is not None and envelope["success"]:
        effective_cache.store(cache_key, envelope)
    return envelope


def clear_execution_cache() -> int:
    """Invalidate the shared execution result cache; returns entries removed."""
    return _EXECUTION_RESULT_CACHE.invalidate()

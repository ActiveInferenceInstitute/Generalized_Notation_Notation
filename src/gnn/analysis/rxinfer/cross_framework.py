#!/usr/bin/env python3
"""Cross-framework comparison for GNN simulation results (roadmap A6).

Renders one GNN file to six backends — RxInfer.jl, PyMDP,
ActiveInference.jl, JAX, PyTorch, and NumPyro — from a single parsed spec,
executes each, and produces a self-contained HTML page with a metrics
table and an animated belief-trajectory comparison overlaying
every framework that produced results.

Failure is never silent: each framework yields a :class:`FrameworkRun` whose
``status`` and ``detail`` record exactly why it did not succeed, and those
reasons are surfaced in the HTML. Missing Python dependencies (jax, torch,
numpyro) are probed before rendering and surface as ``unavailable`` receipts,
so a backend is skipped rather than executed into a crash. Only narrow,
expected exceptions are caught (missing optional backend, subprocess timeout,
malformed results JSON); anything else propagates.
"""

from __future__ import annotations

import html
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from gnn.utils.julia_runtime import FrameworkRuntime, default_julia_runtime
from gnn.utils.runtime_safety.framework_availability import (
    FRAMEWORK_IMPORT_CHECK,
    is_framework_available,
)

logger = logging.getLogger(__name__)

SRC_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = SRC_ROOT

RXINFER_JULIA_PROJECT = SRC_ROOT / "src" / "gnn" / "execute" / "rxinfer"
ACTIVEINFERENCE_JULIA_PROJECT = (
    SRC_ROOT / "src" / "gnn" / "execute" / "activeinference_jl"
)

FRAMEWORKS = (
    "rxinfer",
    "pymdp",
    "activeinference_jl",
    "jax",
    "pytorch",
    "numpyro",
)

FRAMEWORK_COLORS = {
    "rxinfer": "#1f77b4",
    "pymdp": "#d62728",
    "activeinference_jl": "#2ca02c",
    "jax": "#9467bd",
    "pytorch": "#ff7f0e",
    "numpyro": "#8c564b",
}

JULIA_TIMEOUT_SECONDS = 1200
PYTHON_TIMEOUT_SECONDS = 600
STDERR_HEAD_LINES = 8
STDERR_TAIL_LINES = 12
STDERR_FLAGGED_LINES = 3
_ERROR_MARKERS = ("ERROR", "Error:", "Traceback", "Exception")


RunStatus = Literal[
    "success",
    "validation_failed",
    "render_failed",
    "execution_failed",
    "unavailable",
    "invalid_results",
]


@dataclass(frozen=True)
class FrameworkRun:
    """Outcome of rendering and executing one framework for one GNN model."""

    framework: str
    status: RunStatus
    detail: str
    results: dict[str, Any] | None = None


def _stderr_excerpt(stderr: str) -> str:
    """Return a readable excerpt of stderr that always keeps the diagnostic.

    Neither end of the stream reliably holds the cause: Julia floods the head
    with precompile progress and the tail with stack frames, leaving the one
    ``ERROR:`` line stranded in the middle. Keep both ends plus any error
    lines from the elided region.
    """
    lines = [line for line in stderr.splitlines() if line.strip()]
    if len(lines) <= STDERR_HEAD_LINES + STDERR_TAIL_LINES:
        return "\n".join(lines)

    middle = lines[STDERR_HEAD_LINES : len(lines) - STDERR_TAIL_LINES]
    flagged = [
        line for line in middle if any(marker in line for marker in _ERROR_MARKERS)
    ][:STDERR_FLAGGED_LINES]
    return "\n".join(
        lines[:STDERR_HEAD_LINES]
        + [f"... {len(middle) - len(flagged)} lines elided ..."]
        + flagged
        + lines[-STDERR_TAIL_LINES:]
    )


def _read_results(results_path: Path, framework: str) -> dict[str, Any] | FrameworkRun:
    """Parse a results JSON file, returning a failed run on malformed payloads."""
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        detail = f"{results_path.name} is not valid JSON: {exc}"
        logger.error("%s: %s", framework, detail)
        return FrameworkRun(framework, "invalid_results", detail)
    if not isinstance(payload, dict):
        detail = f"{results_path.name} holds {type(payload).__name__}, expected object"
        logger.error("%s: %s", framework, detail)
        return FrameworkRun(framework, "invalid_results", detail)
    return payload


def _classify_exit(
    framework: str,
    returncode: int,
    stderr: str,
    results_path: Path,
) -> FrameworkRun:
    """Map a subprocess exit code plus results presence onto a FrameworkRun.

    Exit 0 with results is the only clean success. Exit 1 with results is a
    completed run whose own validation failed — the numbers are still worth
    comparing, so they are kept and flagged. Everything else is an execution
    failure and the stderr tail is logged at error level.
    """
    results_present = results_path.exists()

    if returncode == 0 and results_present:
        payload = _read_results(results_path, framework)
        if isinstance(payload, FrameworkRun):
            return payload
        return FrameworkRun(framework, "success", "completed with exit code 0", payload)

    if returncode == 1 and results_present:
        payload = _read_results(results_path, framework)
        if isinstance(payload, FrameworkRun):
            return payload
        detail = "run completed but reported failed validation (exit code 1)"
        logger.warning("%s: %s", framework, detail)
        return FrameworkRun(framework, "validation_failed", detail, payload)

    if returncode == 0:
        detail = f"exit code 0 but {results_path.name} was not written"
        logger.error("%s: %s", framework, detail)
        return FrameworkRun(framework, "execution_failed", detail)

    detail = f"exit code {returncode}" + (
        " with results present" if results_present else f", no {results_path.name}"
    )
    logger.error(
        "%s execution failed (%s)\nstderr:\n%s",
        framework,
        detail,
        _stderr_excerpt(stderr),
    )
    return FrameworkRun(framework, "execution_failed", detail)


def _run_subprocess(
    framework: str,
    command: list[str],
    cwd: Path,
    timeout: int,
    results_path: Path,
    script_path: Path,
    env: dict[str, str] | None = None,
    runtime: FrameworkRuntime | None = None,
) -> FrameworkRun:
    """Execute a rendered script and classify the outcome.

    Every rendered script passes the shared pre-execution security gate
    first (same helper as the executor and Step-12 paths; fail closed when
    the scanner cannot be imported). The gate and the subprocess envelope
    arrive through :class:`FrameworkRuntime`; ``None`` resolves the stock
    runtime from ``gnn.execute`` — this analysis module must not depend on
    the executor/runner stack at import time.

    The execution itself runs through the shared subprocess envelope, the
    same structured contract as every per-framework runner: timeout, exit
    code, and spawn failures all arrive in one outcome mapping instead of
    per-site exception handling.
    """
    if runtime is None:
        runtime = default_julia_runtime()
    try:
        check_script_allowed = runtime.load_security_gate()
    except ImportError as exc:
        detail = f"pre-execution security gate unavailable ({exc}); refusing to execute"
        logger.error("%s: %s", framework, detail)
        return FrameworkRun(framework, "execution_failed", detail)

    gate_verdict = check_script_allowed(script_path)
    if gate_verdict["overridden"]:
        logger.warning(
            "GNN_ALLOW_UNSAFE_EXEC set: pre-execution security gate "
            "bypassed for %s (trusted-local use only)",
            script_path,
        )
    if not gate_verdict["ok"]:
        detail = (
            f"Pre-execution security gate blocked {script_path.name}: "
            f"{gate_verdict['reason']}"
        )
        logger.error("%s: %s", framework, detail)
        return FrameworkRun(framework, "execution_failed", detail)

    # SANCTIONED DELTA (envelope contract): a spawn failure (OSError,
    # SandboxUnavailable refusal) previously propagated out of this
    # function and crashed the whole comparison; the envelope returns
    # ``return_code == runtime.never_started`` for it and we classify it as
    # a per-framework execution failure instead.
    envelope = runtime.run_envelope(command, timeout=timeout, cwd=str(cwd), env=env)

    if envelope.get("error_type") == "TimeoutExpired":
        detail = f"execution exceeded {timeout}s timeout"
        logger.error("%s: %s", framework, detail)
        return FrameworkRun(framework, "execution_failed", detail)
    if envelope["return_code"] == runtime.never_started:
        detail = f"execution failed to start: {envelope.get('error', '')}"
        logger.error("%s: %s", framework, detail)
        return FrameworkRun(framework, "execution_failed", detail)

    return _classify_exit(
        framework, envelope["return_code"], envelope["stderr"], results_path
    )


def _render(
    framework: str,
    renderer: Any,
    spec: dict[str, Any],
    script_path: Path,
) -> FrameworkRun | None:
    """Render a spec to a script, returning a failed run when rendering fails."""
    success, message, _warnings = renderer(spec, script_path)
    if not success:
        logger.error("%s render failed: %s", framework, message)
        return FrameworkRun(framework, "render_failed", message)
    if not script_path.exists():
        detail = f"renderer reported success but {script_path.name} is missing"
        logger.error("%s: %s", framework, detail)
        return FrameworkRun(framework, "render_failed", detail)
    return None


def _require_julia(framework: str, runtime: FrameworkRuntime) -> str | FrameworkRun:
    """Resolve the julia executable, or report the framework as unavailable."""
    julia = runtime.julia_executable()
    if julia is None:
        detail = "julia is not on PATH"
        logger.warning("%s unavailable: %s", framework, detail)
        return FrameworkRun(framework, "unavailable", detail)
    return julia


def _execute_python_lane(
    framework: str,
    render_fn: Any,
    spec: dict[str, Any],
    fw_dir: Path,
    timeout: Optional[int],
    runtime: FrameworkRuntime,
    output_env_var: str,
    extra_env: dict[str, str] | None = None,
    *,
    probe_availability: bool = True,
) -> FrameworkRun:
    """Shared body for the Python backends: probe, render, redirect, execute.

    The rendered script runs under ``sys.executable`` with the framework's
    output environment variable pointed at ``fw_dir`` so results land beside
    the rendered script (the same contract the Step-12 executor sets in
    ``src/gnn/execute/processor/``). When ``probe_availability`` is set, a
    missing dependency yields an ``unavailable`` receipt before any rendering
    happens — a skip, never an execution failure.
    """
    if probe_availability and not is_framework_available(framework):
        module_name, hint = FRAMEWORK_IMPORT_CHECK.get(
            framework, (framework, "uv sync")
        )
        detail = f"{module_name} not installed ({hint})"
        logger.warning("%s unavailable: %s", framework, detail)
        return FrameworkRun(framework, "unavailable", detail)

    script_path = fw_dir / f"model_{framework}.py"
    failure = _render(framework, render_fn, spec, script_path)
    if failure is not None:
        return failure

    env = os.environ.copy()
    env[output_env_var] = str(fw_dir)
    if extra_env:
        env.update(extra_env)
    env["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(PROJECT_ROOT), str(SRC_ROOT), env.get("PYTHONPATH", ""))
        if part
    )

    return _run_subprocess(
        framework,
        [sys.executable, str(script_path)],
        cwd=fw_dir,
        timeout=PYTHON_TIMEOUT_SECONDS if timeout is None else timeout,
        results_path=fw_dir / "simulation_results.json",
        script_path=script_path,
        env=env,
        runtime=runtime,
    )


def _execute_rxinfer(
    spec: dict[str, Any],
    fw_dir: Path,
    timeout: Optional[int] = None,
    runtime: FrameworkRuntime | None = None,
) -> FrameworkRun:
    """Render and run the RxInfer.jl backend inside its committed Julia project."""
    framework = "rxinfer"
    if runtime is None:
        runtime = default_julia_runtime()
    try:
        from gnn.render.rxinfer.rxinfer_renderer import render_gnn_to_rxinfer
    except ImportError as exc:
        detail = f"render.rxinfer is not importable: {exc}"
        logger.warning("%s unavailable: %s", framework, detail)
        return FrameworkRun(framework, "unavailable", detail)

    julia = _require_julia(framework, runtime)
    if isinstance(julia, FrameworkRun):
        return julia

    script_path = fw_dir / "model_rxinfer.jl"
    failure = _render(framework, render_gnn_to_rxinfer, spec, script_path)
    if failure is not None:
        return failure

    return _run_subprocess(
        framework,
        [
            julia,
            "--startup-file=no",
            f"--project={RXINFER_JULIA_PROJECT}",
            str(script_path),
        ],
        cwd=fw_dir,
        timeout=JULIA_TIMEOUT_SECONDS if timeout is None else timeout,
        results_path=fw_dir / "simulation_results.json",
        script_path=script_path,
        env=runtime.julia_subprocess_env(),
        runtime=runtime,
    )


def _execute_pymdp(
    spec: dict[str, Any],
    fw_dir: Path,
    timeout: Optional[int] = None,
    runtime: FrameworkRuntime | None = None,
) -> FrameworkRun:
    """Render and run the PyMDP backend with results redirected into ``fw_dir``."""
    framework = "pymdp"
    if runtime is None:
        runtime = default_julia_runtime()
    try:
        from gnn.render.pymdp.pymdp_renderer import render_gnn_to_pymdp
    except ImportError as exc:
        detail = f"render.pymdp is not importable: {exc}"
        logger.warning("%s unavailable: %s", framework, detail)
        return FrameworkRun(framework, "unavailable", detail)

    # The generated runner resolves the checkout from GNN_PROJECT_ROOT and
    # honours PYMDP_OUTPUT_DIR, so results land in fw_dir instead of
    # output/pymdp_simulations/<model>/ under the CWD. Both variables mirror
    # what the Step-12 executor sets (src/gnn/execute/processor/single.py).
    return _execute_python_lane(
        framework,
        render_gnn_to_pymdp,
        spec,
        fw_dir,
        timeout,
        runtime,
        "PYMDP_OUTPUT_DIR",
        extra_env={"GNN_PROJECT_ROOT": str(PROJECT_ROOT)},
        probe_availability=False,
    )


def _execute_activeinference_jl(
    spec: dict[str, Any],
    fw_dir: Path,
    timeout: Optional[int] = None,
    runtime: FrameworkRuntime | None = None,
) -> FrameworkRun:
    """Render and run the ActiveInference.jl backend in its Julia project."""
    framework = "activeinference_jl"
    if runtime is None:
        runtime = default_julia_runtime()
    try:
        from gnn.render.activeinference_jl.activeinference_renderer import (
            render_gnn_to_activeinference_jl,
        )
    except ImportError as exc:
        detail = f"render.activeinference_jl is not importable: {exc}"
        logger.warning("%s unavailable: %s", framework, detail)
        return FrameworkRun(framework, "unavailable", detail)

    julia = _require_julia(framework, runtime)
    if isinstance(julia, FrameworkRun):
        return julia

    script_path = fw_dir / "model_activeinference.jl"
    failure = _render(framework, render_gnn_to_activeinference_jl, spec, script_path)
    if failure is not None:
        return failure

    return _run_subprocess(
        framework,
        [
            julia,
            "--startup-file=no",
            f"--project={ACTIVEINFERENCE_JULIA_PROJECT}",
            str(script_path),
        ],
        cwd=fw_dir,
        timeout=JULIA_TIMEOUT_SECONDS if timeout is None else timeout,
        results_path=fw_dir / "simulation_results.json",
        script_path=script_path,
        env=runtime.julia_subprocess_env(),
        runtime=runtime,
    )


def _execute_jax(
    spec: dict[str, Any],
    fw_dir: Path,
    timeout: Optional[int] = None,
    runtime: FrameworkRuntime | None = None,
) -> FrameworkRun:
    """Render and run the JAX backend with results redirected into ``fw_dir``."""
    if runtime is None:
        runtime = default_julia_runtime()
    try:
        from gnn.render.jax.jax_renderer import render_gnn_to_jax
    except ImportError as exc:
        detail = f"render.jax is not importable: {exc}"
        logger.warning("%s unavailable: %s", "jax", detail)
        return FrameworkRun("jax", "unavailable", detail)

    return _execute_python_lane(
        "jax", render_gnn_to_jax, spec, fw_dir, timeout, runtime, "GNN_OUTPUT_DIR"
    )


def _execute_pytorch(
    spec: dict[str, Any],
    fw_dir: Path,
    timeout: Optional[int] = None,
    runtime: FrameworkRuntime | None = None,
) -> FrameworkRun:
    """Render and run the PyTorch backend with results redirected into ``fw_dir``."""
    if runtime is None:
        runtime = default_julia_runtime()
    try:
        from gnn.render.pytorch.pytorch_renderer import render_gnn_to_pytorch
    except ImportError as exc:
        detail = f"render.pytorch is not importable: {exc}"
        logger.warning("%s unavailable: %s", "pytorch", detail)
        return FrameworkRun("pytorch", "unavailable", detail)

    return _execute_python_lane(
        "pytorch",
        render_gnn_to_pytorch,
        spec,
        fw_dir,
        timeout,
        runtime,
        "PYTORCH_OUTPUT_DIR",
    )


def _execute_numpyro(
    spec: dict[str, Any],
    fw_dir: Path,
    timeout: Optional[int] = None,
    runtime: FrameworkRuntime | None = None,
) -> FrameworkRun:
    """Render and run the NumPyro backend with results redirected into ``fw_dir``."""
    if runtime is None:
        runtime = default_julia_runtime()
    try:
        from gnn.render.numpyro.numpyro_renderer import render_gnn_to_numpyro
    except ImportError as exc:
        detail = f"render.numpyro is not importable: {exc}"
        logger.warning("%s unavailable: %s", "numpyro", detail)
        return FrameworkRun("numpyro", "unavailable", detail)

    return _execute_python_lane(
        "numpyro",
        render_gnn_to_numpyro,
        spec,
        fw_dir,
        timeout,
        runtime,
        "NUMPYRO_OUTPUT_DIR",
    )


_EXECUTORS = {
    "rxinfer": _execute_rxinfer,
    "pymdp": _execute_pymdp,
    "activeinference_jl": _execute_activeinference_jl,
    "jax": _execute_jax,
    "pytorch": _execute_pytorch,
    "numpyro": _execute_numpyro,
}


def _execute_framework(
    spec: dict[str, Any],
    framework: str,
    fw_dir: Path,
    timeout: Optional[int] = None,
    runtime: FrameworkRuntime | None = None,
) -> FrameworkRun:
    """Render and execute one framework from an already-parsed GNN spec."""
    if framework not in _EXECUTORS:
        raise ValueError(f"Unknown framework: {framework}")
    fw_dir.mkdir(parents=True, exist_ok=True)
    return _EXECUTORS[framework](spec, fw_dir, timeout, runtime)


def _belief_rows(results: dict[str, Any]) -> list[list[float]]:
    """Extract belief trajectories as ``[timestep][state]`` float rows."""
    beliefs = results.get("beliefs")
    if beliefs is None:
        by_factor = results.get("beliefs_by_factor")
        if isinstance(by_factor, dict) and by_factor:
            beliefs = next(iter(by_factor.values()))
    if not isinstance(beliefs, list):
        return []
    rows: list[list[float]] = []
    for row in beliefs:
        if isinstance(row, list) and row:
            rows.append([float(value) for value in row])
    return rows


def _metric_cell(run: FrameworkRun, extractor: Any) -> str:
    """Render one metrics-table cell for a framework, or an em dash if absent."""
    if run.results is None:
        return "—"
    return html.escape(str(extractor(run.results)))


def _chart_payload(runs: list[FrameworkRun]) -> dict[str, Any]:
    """Build the embedded JSON payload driving the belief-trajectory chart."""
    series = []
    for run in runs:
        if run.results is None:
            continue
        rows = _belief_rows(run.results)
        if not rows:
            continue
        series.append(
            {
                "framework": run.framework,
                "color": FRAMEWORK_COLORS.get(run.framework, "#666666"),
                "beliefs": rows,
            }
        )
    num_states = max((len(s["beliefs"][0]) for s in series), default=0)
    num_steps = max((len(s["beliefs"]) for s in series), default=0)
    return {"series": series, "num_states": num_states, "num_steps": num_steps}


_CHART_SCRIPT = """
const payload = JSON.parse(document.getElementById("cf-chart-data").textContent);
const canvas = document.getElementById("cf-canvas");
const slider = document.getElementById("cf-slider");
const playBtn = document.getElementById("cf-play");
const readout = document.getElementById("cf-step-readout");

if (canvas && payload.num_steps > 0) {
  const ctx = canvas.getContext("2d");
  const numStates = payload.num_states;
  const numSteps = payload.num_steps;
  const panelH = 130;
  const padL = 46, padR = 14, padT = 22, padB = 26;
  canvas.width = 900;
  canvas.height = numStates * panelH;
  slider.max = String(numSteps - 1);

  let step = 0;
  let timer = null;

  function xAt(t) {
    const span = canvas.width - padL - padR;
    return padL + (numSteps === 1 ? span / 2 : (t / (numSteps - 1)) * span);
  }

  function yAt(panel, value) {
    const top = panel * panelH + padT;
    const h = panelH - padT - padB;
    return top + (1 - value) * h;
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.font = "12px system-ui, sans-serif";
    for (let s = 0; s < numStates; s++) {
      const top = s * panelH + padT;
      const h = panelH - padT - padB;
      ctx.strokeStyle = "#e2e2e2";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padL, top);
      ctx.lineTo(padL, top + h);
      ctx.lineTo(canvas.width - padR, top + h);
      ctx.stroke();
      ctx.fillStyle = "#666";
      ctx.fillText("state " + s, padL + 4, top - 6);
      ctx.fillText("1.0", 12, top + 10);
      ctx.fillText("0.0", 12, top + h);

      payload.series.forEach(function (entry) {
        const rows = entry.beliefs;
        ctx.strokeStyle = entry.color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        let started = false;
        for (let t = 0; t < rows.length; t++) {
          const row = rows[t];
          if (s >= row.length) { continue; }
          const x = xAt(t), y = yAt(s, row[s]);
          if (!started) { ctx.moveTo(x, y); started = true; } else { ctx.lineTo(x, y); }
        }
        ctx.stroke();

        const idx = Math.min(step, rows.length - 1);
        const row = rows[idx];
        if (row && s < row.length) {
          ctx.fillStyle = entry.color;
          ctx.beginPath();
          ctx.arc(xAt(idx), yAt(s, row[s]), 4, 0, Math.PI * 2);
          ctx.fill();
        }
      });

      ctx.strokeStyle = "#999";
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(xAt(step), top);
      ctx.lineTo(xAt(step), top + h);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    readout.textContent = "t = " + step + " / " + (numSteps - 1);
    slider.value = String(step);
  }

  function stop() {
    if (timer !== null) { clearInterval(timer); timer = null; }
    playBtn.textContent = "Play";
  }

  playBtn.addEventListener("click", function () {
    if (timer !== null) { stop(); return; }
    playBtn.textContent = "Pause";
    timer = setInterval(function () {
      step = (step + 1) % numSteps;
      draw();
    }, 400);
  });

  slider.addEventListener("input", function () {
    stop();
    step = Number(slider.value);
    draw();
  });

  draw();
}
"""

_STYLE = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: #f5f5f5; color: #222; }
.container { max-width: 1000px; margin: 0 auto; padding: 20px; }
h1 { text-align: center; margin: 20px 0; font-size: 1.5em; }
h2 { margin: 24px 0 10px; font-size: 1.1em; }
table { width: 100%; border-collapse: collapse; background: #fff; }
th, td { padding: 8px; text-align: center; border: 1px solid #ddd; font-size: 0.9em; }
th { background: #f0f0f0; }
td.reason { text-align: left; font-size: 0.8em; color: #555; }
.status-success { color: #1a7f37; font-weight: bold; }
.status-validation_failed { color: #9a6700; font-weight: bold; }
.status-render_failed,
.status-execution_failed,
.status-invalid_results { color: #b42318; font-weight: bold; }
.status-unavailable { color: #666; font-weight: bold; }
.chart { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 15px; }
.controls { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
.controls button { padding: 5px 14px; cursor: pointer; }
.controls input[type=range] { flex: 1; }
.legend span { display: inline-block; margin-right: 14px; font-size: 0.85em; }
.swatch { display: inline-block; width: 12px; height: 12px; margin-right: 5px;
  vertical-align: middle; border-radius: 2px; }
canvas { width: 100%; height: auto; }
.empty { color: #888; padding: 30px; text-align: center; }
.footer { text-align: center; color: #888; margin-top: 26px; font-size: 0.8em; }
"""

_METRICS: tuple[tuple[str, Any], ...] = (
    ("Framework", lambda r: r.get("framework", "N/A")),
    ("States", lambda r: (r.get("model_parameters") or {}).get("num_states", "N/A")),
    ("Timesteps", lambda r: r.get("num_timesteps", "N/A")),
    ("VFE trace length", lambda r: len(r.get("variational_free_energy") or [])),
    ("Belief timesteps", lambda r: len(_belief_rows(r))),
    (
        "Belief accuracy",
        lambda r: (r.get("validation") or {}).get("belief_accuracy", "N/A"),
    ),
    (
        "Inference converged",
        lambda r: (r.get("validation") or {}).get("inference_converged", "N/A"),
    ),
    ("All valid", lambda r: (r.get("validation") or {}).get("all_valid", "N/A")),
)


def render_comparison_html(
    model_name: str, runs: list[FrameworkRun], output_path: Path
) -> str:
    """Render the cross-framework comparison page. Pure apart from the file write.

    Args:
        model_name: Model identifier shown in the page title.
        runs: One run per framework, in display order.
        output_path: Destination HTML file.

    Returns:
        The path written, as a string.
    """
    safe_model = html.escape(model_name)
    payload = _chart_payload(runs)
    payload_json = json.dumps(payload).replace("</", "<\\/")
    succeeded = [run for run in runs if run.status == "success"]

    parts: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="UTF-8">',
        f"<title>Cross-Framework Comparison — {safe_model}</title>",
        f"<style>{_STYLE}</style>",
        "</head>",
        "<body>",
        '<div class="container">',
        f"<h1>Cross-Framework Comparison — {safe_model}</h1>",
        "<h2>Metrics</h2>",
        "<table>",
        "<tr><th>Metric</th>"
        + "".join(f"<th>{html.escape(run.framework)}</th>" for run in runs)
        + "</tr>",
        "<tr><td><b>Status</b></td>"
        + "".join(
            f'<td class="status-{html.escape(run.status)}">'
            f"{html.escape(run.status)}</td>"
            for run in runs
        )
        + "</tr>",
        "<tr><td><b>Reason</b></td>"
        + "".join(f'<td class="reason">{html.escape(run.detail)}</td>' for run in runs)
        + "</tr>",
    ]

    for label, extractor in _METRICS:
        parts.append(
            f"<tr><td><b>{label}</b></td>"
            + "".join(f"<td>{_metric_cell(run, extractor)}</td>" for run in runs)
            + "</tr>"
        )
    parts.append("</table>")

    parts.append("<h2>Belief trajectories</h2>")
    parts.append('<div class="chart">')
    parts.append(
        '<script id="cf-chart-data" type="application/json">'
        + payload_json
        + "</script>"
    )
    if payload["series"]:
        parts.append(
            '<div class="legend">'
            + "".join(
                f'<span><span class="swatch" style="background:{entry["color"]}"></span>'
                f"{html.escape(str(entry['framework']))}</span>"
                for entry in payload["series"]
            )
            + "</div>"
        )
        parts.append(
            '<div class="controls">'
            '<button id="cf-play" type="button">Play</button>'
            '<input id="cf-slider" type="range" min="0" max="0" value="0">'
            '<span id="cf-step-readout">t = 0</span>'
            "</div>"
        )
        parts.append('<canvas id="cf-canvas"></canvas>')
    else:
        parts.append(
            '<div class="empty">No framework produced belief trajectories.</div>'
        )
    parts.append("</div>")

    parts.append(
        f'<div class="footer">Model {safe_model} — '
        f"{len(succeeded)}/{len(runs)} frameworks succeeded.</div>"
    )
    parts.append("</div>")
    parts.append(f"<script>{_CHART_SCRIPT}</script>")
    parts.append("</body>")
    parts.append("</html>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")
    logger.info("Wrote cross-framework comparison: %s", output_path)
    return str(output_path)


def compare_with_status(
    gnn_file: Path,
    output_dir: Path,
    timeout: Optional[int] = None,
    runtime: FrameworkRuntime | None = None,
) -> tuple[str, list[FrameworkRun]]:
    """Render, execute, and compare one GNN model, exposing per-framework outcomes.

    Args:
        gnn_file: Path to the GNN specification file.
        output_dir: Directory receiving per-framework artifacts and the HTML.
        timeout: Optional per-framework execution timeout in seconds; each
            backend falls back to its module default (JULIA_TIMEOUT_SECONDS
            for the Julia lanes, PYTHON_TIMEOUT_SECONDS for the Python
            lanes) when omitted.
        runtime: Optional subprocess runtime override (tests inject a
            recording runtime here).

    Returns:
        Tuple of the generated comparison HTML path (string) and one
        :class:`FrameworkRun` per registered framework, in display order.

    Raises:
        FileNotFoundError: The GNN file does not exist.
        ValueError: The GNN file yielded no POMDP state space.
    """
    gnn_file = Path(gnn_file)
    if not gnn_file.is_file():
        raise FileNotFoundError(f"GNN file not found: {gnn_file}")

    if runtime is None:
        runtime = default_julia_runtime()

    from gnn.extract.pomdp_extractor import extract_pomdp_from_file
    from gnn.render.pomdp_processor import pomdp_to_gnn_spec

    pomdp_space = extract_pomdp_from_file(gnn_file, strict_validation=True)
    if pomdp_space is None:
        raise ValueError(f"No POMDP state space could be extracted from {gnn_file}")
    spec = pomdp_to_gnn_spec(pomdp_space)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        _execute_framework(spec, framework, output_dir / framework, timeout, runtime)
        for framework in FRAMEWORKS
    ]

    for run in runs:
        if run.status != "success":
            logger.warning(
                "%s did not succeed for %s: %s (%s)",
                run.framework,
                gnn_file.name,
                run.status,
                run.detail,
            )

    html_path = render_comparison_html(
        gnn_file.stem, runs, output_dir / f"{gnn_file.stem}_comparison.html"
    )
    return html_path, runs


def run_cross_framework_comparison(
    gnn_file: Path,
    output_dir: Path,
    timeout: Optional[int] = None,
    runtime: FrameworkRuntime | None = None,
) -> str:
    """Render, execute, and compare one GNN model across all six backends.

    Discards the per-framework outcome records; use
    :func:`compare_with_status` when callers need the run statuses (the
    execute-module MCP tool does).

    Returns:
        Path to the generated comparison HTML, as a string.

    Raises:
        FileNotFoundError: The GNN file does not exist.
        ValueError: The GNN file yielded no POMDP state space.
    """
    html_path, _runs = compare_with_status(gnn_file, output_dir, timeout, runtime)
    return html_path


__all__ = [
    "FrameworkRun",
    "FrameworkRuntime",
    "compare_with_status",
    "render_comparison_html",
    "run_cross_framework_comparison",
]

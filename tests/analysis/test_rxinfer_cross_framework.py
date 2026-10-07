"""Tests for the cross-framework comparison module (roadmap A6).

The HTML renderer is a pure function over ``FrameworkRun`` records, so the
bulk of this file runs unconditionally with no Julia and no subprocesses,
asserting on real data structures. Exit-code classification is exercised
against real files on disk. Only the end-to-end run against a live Julia
toolchain is gated.

Public functions:
- test_julia_project_paths_resolve
- test_every_renderer_accepts_the_shared_spec
- test_html_reports_status_and_reasons_for_every_framework
- test_html_embeds_belief_payload_for_successful_frameworks
- test_html_renders_when_every_framework_failed
- test_chart_payload_reads_beliefs_by_factor
- test_clean_exit_with_results_is_success
- test_exit_one_with_results_is_validation_failed
- test_nonzero_exit_without_results_is_execution_failed
- test_clean_exit_without_results_is_execution_failed
- test_malformed_results_json_is_invalid_results
- test_stderr_excerpt_keeps_the_leading_diagnostic
- test_stderr_excerpt_surfaces_an_error_buried_in_precompile_noise
- test_missing_gnn_file_raises
- test_live_cross_framework_comparison
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pytest

import gnn.analysis.rxinfer.cross_framework as cross_framework
from gnn.analysis.rxinfer.cross_framework import (
    ACTIVEINFERENCE_JULIA_PROJECT,
    FRAMEWORKS,
    RXINFER_JULIA_PROJECT,
    FrameworkRun,
    FrameworkRuntime,
    _chart_payload,
    _classify_exit,
    _run_subprocess,
    _stderr_excerpt,
    render_comparison_html,
    run_cross_framework_comparison,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIMPLE_MDP = PROJECT_ROOT / "input" / "gnn_files" / "discrete" / "simple_mdp.md"


def _results(beliefs: list[list[float]], *, all_valid: bool = True) -> dict:
    """Build a minimal results payload in the shape the frameworks emit."""
    return {
        "framework": "canned",
        "num_timesteps": len(beliefs),
        "beliefs": beliefs,
        "variational_free_energy": [3.0, 2.5, 2.1],
        "model_parameters": {"num_states": len(beliefs[0]) if beliefs else 0},
        "validation": {
            "all_valid": all_valid,
            "belief_accuracy": 0.75,
            "inference_converged": True,
        },
    }


def _canned_runs() -> list[FrameworkRun]:
    """Two frameworks with results, one unavailable."""
    return [
        FrameworkRun(
            framework="rxinfer",
            status="success",
            detail="completed with exit code 0",
            results=_results([[0.7, 0.3], [0.6, 0.4], [0.9, 0.1]]),
        ),
        FrameworkRun(
            framework="pymdp",
            status="validation_failed",
            detail="run completed but reported failed validation (exit code 1)",
            results=_results([[0.5, 0.5], [0.55, 0.45], [0.8, 0.2]], all_valid=False),
        ),
        FrameworkRun(
            framework="activeinference_jl",
            status="unavailable",
            detail="julia is not on PATH",
        ),
    ]


def _payload_from_html(html_text: str) -> dict[str, Any]:
    """Extract the embedded chart JSON from the rendered page."""
    marker = '<script id="cf-chart-data" type="application/json">'
    start = html_text.index(marker) + len(marker)
    end = html_text.index("</script>", start)
    payload = json.loads(html_text[start:end].replace("<\\/", "</"))
    assert isinstance(payload, dict)
    return payload


# --- path derivation (CF-7) ---------------------------------------------------


def test_julia_project_paths_resolve() -> None:
    """Both Julia project dirs are CWD-independent and carry a Project.toml."""
    assert RXINFER_JULIA_PROJECT == PROJECT_ROOT / "src" / "gnn" / "execute" / "rxinfer"
    assert (
        ACTIVEINFERENCE_JULIA_PROJECT
        == PROJECT_ROOT / "src" / "gnn" / "execute" / "activeinference_jl"
    )
    assert (RXINFER_JULIA_PROJECT / "Project.toml").is_file()
    assert (ACTIVEINFERENCE_JULIA_PROJECT / "Project.toml").is_file()


# --- renderer input contract (CF-1) -------------------------------------------


def test_every_renderer_accepts_the_shared_spec(tmp_path: Path) -> None:
    """All six renderers take the one parsed spec dict, not the GNN file path.

    Guards the defect where PyMDP and ActiveInference.jl were handed a Path
    and therefore always failed. No Julia and no subprocess: rendering only.
    """
    from gnn.extract.pomdp_extractor import extract_pomdp_from_file
    from gnn.render.activeinference_jl.activeinference_renderer import (
        render_gnn_to_activeinference_jl,
    )
    from gnn.render.jax.jax_renderer import render_gnn_to_jax
    from gnn.render.numpyro.numpyro_renderer import render_gnn_to_numpyro
    from gnn.render.pomdp_processor import pomdp_to_gnn_spec
    from gnn.render.pymdp.pymdp_renderer import render_gnn_to_pymdp
    from gnn.render.pytorch.pytorch_renderer import render_gnn_to_pytorch
    from gnn.render.rxinfer.rxinfer_renderer import render_gnn_to_rxinfer

    assert SIMPLE_MDP.is_file(), f"missing exemplar: {SIMPLE_MDP}"
    pomdp_space = extract_pomdp_from_file(SIMPLE_MDP, strict_validation=True)
    assert pomdp_space is not None
    spec = pomdp_to_gnn_spec(pomdp_space)
    assert isinstance(spec, dict)

    renderers = {
        "rxinfer": (render_gnn_to_rxinfer, "model_rxinfer.jl"),
        "pymdp": (render_gnn_to_pymdp, "model_pymdp.py"),
        "activeinference_jl": (
            render_gnn_to_activeinference_jl,
            "model_activeinference.jl",
        ),
        "jax": (render_gnn_to_jax, "model_jax.py"),
        "pytorch": (render_gnn_to_pytorch, "model_pytorch.py"),
        "numpyro": (render_gnn_to_numpyro, "model_numpyro.py"),
    }
    for framework, (renderer, filename) in renderers.items():
        script_path = tmp_path / filename
        success, message, _warnings = renderer(spec, script_path)
        assert success, f"{framework} render failed: {message}"
        assert script_path.is_file()
        assert script_path.stat().st_size > 0

    # The PyMDP runner must honour PYMDP_OUTPUT_DIR so results land in the
    # per-framework directory rather than output/pymdp_simulations/ under CWD.
    assert "PYMDP_OUTPUT_DIR" in (tmp_path / "model_pymdp.py").read_text(
        encoding="utf-8"
    )


# --- pure HTML rendering ------------------------------------------------------


def test_html_reports_status_and_reasons_for_every_framework(tmp_path: Path) -> None:
    """Every framework contributes a status cell and a human-readable reason."""
    runs = _canned_runs()
    out = Path(render_comparison_html("simple_mdp", runs, tmp_path / "cmp.html"))
    assert out.is_file()
    text = out.read_text(encoding="utf-8")

    for run in runs:
        assert run.framework in text
        assert run.status in text
        assert run.detail in text

    assert "julia is not on PATH" in text
    assert "1/3 frameworks succeeded" in text


def test_html_embeds_belief_payload_for_successful_frameworks(tmp_path: Path) -> None:
    """The chart payload carries belief trajectories for every framework with results."""
    runs = _canned_runs()
    out = Path(render_comparison_html("simple_mdp", runs, tmp_path / "cmp.html"))
    text = out.read_text(encoding="utf-8")

    payload = _payload_from_html(text)
    frameworks = [entry["framework"] for entry in payload["series"]]
    assert frameworks == ["rxinfer", "pymdp"]
    assert payload["num_states"] == 2
    assert payload["num_steps"] == 3
    assert payload["series"][0]["beliefs"] == [[0.7, 0.3], [0.6, 0.4], [0.9, 0.1]]
    assert len({entry["color"] for entry in payload["series"]}) == 2

    # Self-contained scrubber controls, no external dependencies.
    assert 'id="cf-play"' in text
    assert 'id="cf-slider"' in text
    assert 'id="cf-canvas"' in text
    assert "http://" not in text and "https://" not in text


def test_html_renders_when_every_framework_failed(tmp_path: Path) -> None:
    """An all-failed comparison still renders, with each failure reason shown."""
    runs = [
        FrameworkRun("rxinfer", "render_failed", "renderer rejected the spec"),
        FrameworkRun("pymdp", "execution_failed", "exit code 2, no results"),
        FrameworkRun("activeinference_jl", "invalid_results", "not valid JSON"),
    ]
    out = Path(render_comparison_html("broken_model", runs, tmp_path / "cmp.html"))
    text = out.read_text(encoding="utf-8")

    assert "0/3 frameworks succeeded" in text
    assert "renderer rejected the spec" in text
    assert "exit code 2, no results" in text
    assert "not valid JSON" in text
    assert "No framework produced belief trajectories." in text
    assert _payload_from_html(text)["series"] == []


def test_chart_payload_reads_beliefs_by_factor() -> None:
    """Factored RxInfer payloads expose their first factor's beliefs."""
    run = FrameworkRun(
        framework="rxinfer",
        status="success",
        detail="ok",
        results={"beliefs_by_factor": {"factor_0": [[0.2, 0.8], [0.4, 0.6]]}},
    )
    payload = _chart_payload([run])
    assert payload["series"][0]["beliefs"] == [[0.2, 0.8], [0.4, 0.6]]
    assert payload["num_steps"] == 2


# --- exit-code classification (CF-3) ------------------------------------------


def test_clean_exit_with_results_is_success(tmp_path: Path) -> None:
    """Exit 0 plus a results file is the only clean success."""
    results_path = tmp_path / "simulation_results.json"
    results_path.write_text(json.dumps(_results([[1.0, 0.0]])), encoding="utf-8")

    run = _classify_exit("rxinfer", 0, "", results_path)
    assert run.status == "success"
    assert run.results is not None
    assert run.results["beliefs"] == [[1.0, 0.0]]


def test_exit_one_with_results_is_validation_failed(tmp_path: Path) -> None:
    """Exit 1 with results keeps the payload but flags failed validation."""
    results_path = tmp_path / "simulation_results.json"
    results_path.write_text(
        json.dumps(_results([[0.5, 0.5]], all_valid=False)), encoding="utf-8"
    )

    run = _classify_exit("pymdp", 1, "", results_path)
    assert run.status == "validation_failed"
    assert run.results is not None
    assert "validation" in run.detail


def test_nonzero_exit_without_results_is_execution_failed(tmp_path: Path) -> None:
    """A crashed run with no results is an execution failure with no payload."""
    run = _classify_exit(
        "activeinference_jl", 2, "ERROR: LoadError\nstack frame", tmp_path / "none.json"
    )
    assert run.status == "execution_failed"
    assert run.results is None
    assert "exit code 2" in run.detail


def test_clean_exit_without_results_is_execution_failed(tmp_path: Path) -> None:
    """Exit 0 that writes nothing violates the results contract."""
    run = _classify_exit("rxinfer", 0, "", tmp_path / "simulation_results.json")
    assert run.status == "execution_failed"
    assert "not written" in run.detail


def test_malformed_results_json_is_invalid_results(tmp_path: Path) -> None:
    """Unparseable results are reported, never swallowed."""
    results_path = tmp_path / "simulation_results.json"
    results_path.write_text("{not json", encoding="utf-8")

    run = _classify_exit("rxinfer", 0, "", results_path)
    assert run.status == "invalid_results"
    assert run.results is None


def test_stderr_excerpt_keeps_the_leading_diagnostic() -> None:
    """Long Julia stack traces keep their first line, where the cause is printed."""
    stderr = "\n".join(
        ["ERROR: LoadError: ArgumentError: Package ActiveInference is not installed"]
        + [f"stack frame {i}" for i in range(60)]
    )
    excerpt = _stderr_excerpt(stderr)

    assert "Package ActiveInference is not installed" in excerpt
    assert "stack frame 59" in excerpt
    assert "lines elided" in excerpt
    assert len(excerpt.splitlines()) < 30


def test_stderr_excerpt_surfaces_an_error_buried_in_precompile_noise() -> None:
    """Julia precompile chatter must not push the real cause out of the excerpt.

    Reproduces the observed shape: dozens of "precompile OK" lines, the actual
    error in the middle, then a long stack trace.
    """
    stderr = "\n".join(
        [f"  precompile OK: config {i}" for i in range(40)]
        + ["ERROR: LoadError: Half-edge has been found: z_37"]
        + [f"  [{i}] stack frame" for i in range(40)]
    )
    excerpt = _stderr_excerpt(stderr)

    assert "Half-edge has been found: z_37" in excerpt
    assert "lines elided" in excerpt
    assert len(excerpt.splitlines()) <= (8 + 1 + 3 + 12)


# --- entry point contract -----------------------------------------------------


def test_missing_gnn_file_raises(tmp_path: Path) -> None:
    """A missing GNN file is a loud FileNotFoundError, not an empty string."""
    with pytest.raises(FileNotFoundError):
        run_cross_framework_comparison(tmp_path / "nope.md", tmp_path / "out")


def test_injected_runtime_drives_the_comparison_without_julia(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller-supplied runtime drives the full comparison without julia.

    Pins the inversion end-to-end: the injected runtime's julia probe
    reports both julia frameworks unavailable, the dependency probe reports
    the three probed Python lanes unavailable, pymdp is classified through
    the recorded fake envelope, and no julia environment is ever built.
    """
    recorded: list[list[str]] = []

    def fake_envelope(command: list[str], **_: Any) -> dict[str, Any]:
        recorded.append(list(command))
        return {
            "success": False,
            "return_code": 2,
            "stdout": "",
            "stderr": "boom",
            "error_type": None,
        }

    def no_julia_env() -> dict[str, str]:
        raise AssertionError("julia env must not be built while julia is unavailable")

    runtime = FrameworkRuntime(
        julia_executable=lambda: None,
        julia_subprocess_env=no_julia_env,
        run_envelope=fake_envelope,
        load_security_gate=lambda: lambda _script: {"ok": True, "overridden": False},
    )

    # Deterministic regardless of the ambient environment: the dependency
    # probe reports every probed Python lane missing, so those backends are
    # skip receipts and never reach the subprocess envelope.
    monkeypatch.setattr(cross_framework, "is_framework_available", lambda fw: False)

    html_path = Path(
        run_cross_framework_comparison(SIMPLE_MDP, tmp_path / "out", runtime=runtime)
    )
    text = html_path.read_text(encoding="utf-8")

    assert "0/6 frameworks succeeded" in text
    reasons = re.findall(r'<td class="reason">(.*?)</td>', text)
    assert reasons.count("julia is not on PATH") == 2
    assert reasons.count("jax not installed (uv sync)") == 1
    assert reasons.count("torch not installed (uv sync --extra torch)") == 1
    assert reasons.count("numpyro not installed (uv sync)") == 1
    assert "exit code 2" in text
    assert recorded == [
        [sys.executable, str(tmp_path / "out" / "pymdp" / "model_pymdp.py")]
    ]


# --- live end-to-end run ------------------------------------------------------
# Requires the committed RxInfer/ActiveInference Julia project environments
# (``needs_julia_env`` marker; see tests/helpers/toolchain_probes.py). Julia's
# first-run precompile may take additional time, hence the explicit timeout.


@pytest.mark.needs_julia_env
@pytest.mark.timeout(900)
def test_live_cross_framework_comparison(tmp_path: Path) -> None:
    """The full pipeline renders, runs, and compares the simple_mdp exemplar."""
    assert SIMPLE_MDP.is_file(), f"missing exemplar: {SIMPLE_MDP}"

    html_path = Path(run_cross_framework_comparison(SIMPLE_MDP, tmp_path / "cmp"))
    assert html_path.is_file()
    text = html_path.read_text(encoding="utf-8")

    for framework in FRAMEWORKS:
        assert framework in text

    reasons = re.findall(r'<td class="reason">(.*?)</td>', text)
    payload = _payload_from_html(text)
    frameworks_with_beliefs = {entry["framework"] for entry in payload["series"]}
    assert "rxinfer" in frameworks_with_beliefs, (
        "RxInfer produced no belief trajectory. Per-framework reasons: "
        + "; ".join(f"{fw}={reason}" for fw, reason in zip(FRAMEWORKS, reasons))
    )

    assert payload["num_steps"] > 0


# --- pre-execution security gate (SEC-R1) --------------------------------------

_UNSAFE_RENDERED = """import pathlib
pathlib.Path("marker.txt").write_text("ran")
import subprocess
subprocess.run("ls", shell=True)
"""


def test_unsafe_rendered_script_is_gate_blocked(tmp_path: Path) -> None:
    """A rendered script with a dangerous pattern is refused before any run."""
    script = tmp_path / "model_pymdp.py"
    script.write_text(_UNSAFE_RENDERED, encoding="utf-8")

    run = _run_subprocess(
        "pymdp",
        [sys.executable, str(script)],
        tmp_path,
        60,
        tmp_path / "simulation_results.json",
        script_path=script,
    )

    assert run.status == "execution_failed"
    assert "Pre-execution security gate blocked" in run.detail
    assert run.results is None
    # The command would have created the marker; its absence proves the
    # subprocess never started.
    assert not (tmp_path / "marker.txt").exists()


def test_security_scanner_unavailable_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sabotaged security import hard-blocks instead of skipping the scan."""
    script = tmp_path / "model_pymdp.py"
    script.write_text(_UNSAFE_RENDERED, encoding="utf-8")
    monkeypatch.setitem(sys.modules, "gnn.security.processor", None)

    run = _run_subprocess(
        "pymdp",
        [sys.executable, str(script)],
        tmp_path,
        60,
        tmp_path / "simulation_results.json",
        script_path=script,
    )

    assert run.status == "execution_failed"
    assert "unavailable" in run.detail
    assert not (tmp_path / "marker.txt").exists()


def test_safe_rendered_script_is_executed_not_blocked(tmp_path: Path) -> None:
    """A clean script passes the gate and is classified like any other run."""
    script = tmp_path / "model_pymdp.py"
    script.write_text("print('safe')\n", encoding="utf-8")

    run = _run_subprocess(
        "pymdp",
        [sys.executable, str(script)],
        tmp_path,
        60,
        tmp_path / "simulation_results.json",
        script_path=script,
    )

    # Exit 0 with no results violates the results contract — which proves the
    # script actually ran (the gate did not block it).
    assert run.status == "execution_failed"
    assert "not written" in run.detail


# --- shared subprocess envelope migration (MAJ-07) ------------------------------


def test_timeout_maps_to_execution_failed_with_exact_detail(tmp_path: Path) -> None:
    """A run exceeding its wall-clock budget is execution_failed, never a raise."""
    script = tmp_path / "model_pymdp.py"
    script.write_text("import time\ntime.sleep(3)\n", encoding="utf-8")

    run = _run_subprocess(
        "pymdp",
        [sys.executable, str(script)],
        tmp_path,
        1,
        tmp_path / "simulation_results.json",
        script_path=script,
    )

    assert run.status == "execution_failed"
    assert run.detail == "execution exceeded 1s timeout"
    assert run.results is None


def test_failed_to_start_maps_to_execution_failed_not_crash(tmp_path: Path) -> None:
    """A command vector whose binary does not exist is a per-framework failure.

    Pins the sanctioned envelope delta: the spawn failure (OSError /
    SandboxUnavailable) that used to propagate and crash the comparison now
    arrives as ``return_code == NEVER_STARTED`` and is classified like any
    other execution failure.
    """
    script = tmp_path / "model_pymdp.py"
    script.write_text("print('never runs')\n", encoding="utf-8")

    run = _run_subprocess(
        "pymdp",
        [str(tmp_path / "definitely-not-a-real-binary"), str(script)],
        tmp_path,
        60,
        tmp_path / "simulation_results.json",
        script_path=script,
    )

    assert run.status == "execution_failed"
    assert run.detail.startswith("execution failed to start:")
    assert run.results is None

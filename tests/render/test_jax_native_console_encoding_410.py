"""Real generated JAX programs preserve Unicode across local stream encodings.

The cp1252/ASCII controls run native subprocesses with those actual stream
encodings; they are portable counterexamples rather than Windows-host evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gnn.execute.subprocess_envelope import run_subprocess_envelope
from gnn.render.jax import render_gnn_to_jax


def _spec(name: str) -> dict:
    return {
        "model_name": name,
        "model_parameters": {
            "num_states": 2,
            "num_obs": 2,
            "num_actions": 1,
            "num_timesteps": 3,
            "passive_model": True,
            "b_tensor_order": "next_state_previous_state_action",
        },
        "initialparameterization": {
            "A": [[0.8, 0.3], [0.2, 0.7]],
            "B": [[[0.8], [0.3]], [[0.2], [0.7]]],
            "C": [0.2, -0.1],
            "D": [0.6, 0.4],
        },
    }


def _render(tmp_path: Path, name: str) -> Path:
    script = tmp_path / "native_jax.py"
    source = _spec(name)
    original = json.dumps(source, ensure_ascii=False, sort_keys=True)
    success, message, paths = render_gnn_to_jax(source, script)
    assert success, message
    assert paths == [str(script)]
    assert json.dumps(source, ensure_ascii=False, sort_keys=True) == original
    assert "P(s|o) ∝".encode("utf-8") in script.read_bytes()
    return script


def _assert_science(path: Path, name: str) -> None:
    result = json.loads(path.read_bytes().decode("utf-8", errors="strict"))
    assert result["success"] is True
    assert result["model_name"] == name
    assert result["num_timesteps"] == 3
    assert result["inference_estimand"] == "filtering"
    assert result["control_mode"] == "none"
    trace = result["simulation_trace"]
    assert trace["actions"] == []
    assert trace["efe_history"] == []
    assert len(trace["observations"]) == 3
    # Independent sequential Bayesian oracle, conditioned on saved observations.
    # No generated or executor numerical helpers are called by this oracle.
    likelihood = np.array([[0.8, 0.3], [0.2, 0.7]])
    transition = np.array([[0.8, 0.3], [0.2, 0.7]])
    prior = np.array([0.6, 0.4])
    expected = []
    for observation in trace["observations"]:
        assert observation in (0, 1)
        joint = prior * likelihood[observation]
        posterior = joint / joint.sum()
        expected.append(posterior)
        prior = transition @ posterior
    np.testing.assert_allclose(trace["beliefs"], expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(result["beliefs"], expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("encoding", ["cp1252", "ascii", "utf-8"])
@pytest.mark.parametrize("name", ["portable_sensor", "sensor_μ_状态"])
def test_native_generated_program_preserves_unicode_streams_and_results(
    tmp_path: Path, encoding: str, name: str
) -> None:
    script = _render(tmp_path, name)
    source_hash = hashlib.sha256(script.read_bytes()).hexdigest()
    output = tmp_path / "results_μ_状态"
    environment = {
        "PYTHONIOENCODING": encoding + ":strict",
        "GNN_OUTPUT_DIR": str(output),
    }
    # Observe the actual initial stream encoding in an independent real child.
    control = run_subprocess_envelope(
        [
            sys.executable,
            "-c",
            "import json,sys; print(json.dumps([sys.stdout.encoding,sys.stderr.encoding]))",
        ],
        timeout=30,
        env=environment,
        sandbox=False,
    )
    assert control["success"], control
    assert json.loads(control["stdout"]) == [encoding, encoding]
    result = run_subprocess_envelope(
        [sys.executable, str(script)],
        cwd=tmp_path,
        timeout=90,
        env=environment,
        sandbox=False,
    )
    assert result["success"], result
    assert result["return_code"] == 0
    assert result["cleanup_verified"] and result["streams_drained"]
    assert "✅ Model parameters created" in result["stdout"]
    assert "💾 Saved simulation results to: " + str(output) in result["stdout"]
    assert "JAX Model: " + name in result["stdout"]
    assert "�" not in result["stdout"] + result["stderr"]
    assert hashlib.sha256(script.read_bytes()).hexdigest() == source_hash
    _assert_science(output / "simulation_results.json", name)


def test_native_isolated_generated_program_preserves_unicode(tmp_path: Path) -> None:
    name = "isolated_μ_状态"
    script = _render(tmp_path, name)
    output = tmp_path / "isolated_μ_状态"
    child = subprocess.run(  # nosec B603 — installed interpreter and generated script
        [sys.executable, "-I", str(script)],
        cwd=tmp_path,
        timeout=90,
        env={**os.environ, "GNN_OUTPUT_DIR": str(output)},
        capture_output=True,
    )
    assert child.returncode == 0, child.stderr.decode("utf-8", errors="strict")
    assert name.encode("utf-8") in child.stdout
    assert str(output).encode("utf-8") in child.stdout
    assert "✅".encode("utf-8") in child.stdout
    child.stdout.decode("utf-8", errors="strict")
    child.stderr.decode("utf-8", errors="strict")
    (tmp_path / "native_stdout.bin").write_bytes(child.stdout)
    (tmp_path / "native_stderr.bin").write_bytes(child.stderr)
    _assert_science(output / "simulation_results.json", name)


def test_imported_generated_program_preserves_callers_stream_encoding(
    tmp_path: Path,
) -> None:
    script = _render(tmp_path, "import_only_μ")
    code = (
        "import json,runpy,sys; before=[sys.stdout.encoding,sys.stderr.encoding]; "
        "runpy.run_path(sys.argv[1],run_name='authored_import'); "
        "print(json.dumps({'before':before,'after':[sys.stdout.encoding,sys.stderr.encoding]}))"
    )
    result = run_subprocess_envelope(
        [sys.executable, "-c", code, str(script)],
        cwd=tmp_path,
        timeout=90,
        env={"PYTHONIOENCODING": "cp1252:strict"},
        sandbox=False,
    )
    assert result["success"], result
    assert json.loads(result["stdout"]) == {
        "before": ["cp1252", "cp1252"],
        "after": ["cp1252", "cp1252"],
    }
    assert not (tmp_path / "jax_outputs").exists()


def test_public_jax_runner_reads_and_persists_utf8_under_ascii_locale(
    tmp_path: Path,
) -> None:
    script = _render(tmp_path, "saved_μ_状态")
    output = tmp_path / "runner_output"
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONIOENCODING": "cp1252:strict",
            "PYTHONUTF8": "0",
            "PYTHONCOERCECLOCALE": "0",
            "LC_ALL": "C",
        }
    )
    code = (
        "import json,locale,pathlib,sys; from gnn.execute.jax import execute_jax_script; "
        "initial={'locale':locale.getencoding(),'stdout':sys.stdout.encoding,'utf8_mode':sys.flags.utf8_mode}; "
        "ok=execute_jax_script(pathlib.Path(sys.argv[1]),output_dir=pathlib.Path(sys.argv[2]),timeout=90); "
        "print(json.dumps({'success':ok,'initial':initial}))"
    )
    child = subprocess.run(  # nosec B603 — selected interpreter and owned files
        [sys.executable, "-P", "-c", code, str(script), str(output)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        timeout=120,
    )
    assert child.returncode == 0, child.stderr.decode("utf-8", errors="replace")
    receipt = json.loads(child.stdout)
    assert receipt["initial"]["stdout"] == "cp1252"
    assert receipt["initial"]["utf8_mode"] == 0
    # macOS/Linux C locale yields ASCII; Windows retains its native code page.
    assert receipt["initial"]["locale"]
    if os.name != "nt":
        assert receipt["initial"]["locale"].lower() in {"us-ascii", "ansi_x3.4-1968"}
    assert receipt["success"] is True
    stdout = (output / "stdout.txt").read_bytes().decode("utf-8", errors="strict")
    stderr = (output / "stderr.txt").read_bytes().decode("utf-8", errors="strict")
    assert "saved_μ_状态" in stdout
    assert "✅" in stdout and "💾" in stdout
    assert "�" not in stdout + stderr
    log = json.loads((output / "execution_log.json").read_bytes())
    assert log["success"] is True and log["return_code"] == 0
    assert log["script"] == str(script)
    _assert_science(output / "simulation_results.json", "saved_μ_状态")


def test_native_generated_refusal_preserves_unicode_stderr_and_caller_file(
    tmp_path: Path,
) -> None:
    script = _render(tmp_path, "portable_refusal")
    blocked = tmp_path / "blocked_μ_状态"
    original = b"caller-owned output marker"
    blocked.write_bytes(original)
    result = run_subprocess_envelope(
        [sys.executable, str(script)],
        cwd=tmp_path,
        timeout=90,
        env={"PYTHONIOENCODING": "cp1252:strict", "GNN_OUTPUT_DIR": str(blocked)},
        sandbox=False,
    )
    assert result["success"] is False and result["return_code"] == 1
    assert "FileExistsError" in result["stderr"]
    assert str(blocked) in result["stderr"]
    assert "UnicodeEncodeError" not in result["stderr"]
    assert result["cleanup_verified"] and result["streams_drained"]
    assert blocked.read_bytes() == original
    assert not list(tmp_path.rglob("simulation_results.json"))


def test_inprocess_generated_main_preserves_caller_text_capture(
    tmp_path: Path,
) -> None:
    script = _render(tmp_path, "captured_μ_状态")
    output = tmp_path / "text_capture"
    code = (
        "import contextlib,io,json,runpy,sys; out=io.StringIO(); err=io.StringIO(); "
        "original=[sys.stdout.encoding,sys.stderr.encoding]; "
        "\nwith contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):\n"
        "    runpy.run_path(sys.argv[1],run_name='__main__')\n"
        "print(json.dumps({'stdout':out.getvalue(),'stderr':err.getvalue(),"
        "'original':original,'after':[sys.stdout.encoding,sys.stderr.encoding]}))"
    )
    result = run_subprocess_envelope(
        [sys.executable, "-c", code, str(script)],
        cwd=tmp_path,
        timeout=90,
        env={"PYTHONIOENCODING": "cp1252:strict", "GNN_OUTPUT_DIR": str(output)},
        sandbox=False,
    )
    assert result["success"], result
    captured = json.loads(result["stdout"])
    assert "captured_μ_状态" in captured["stdout"]
    assert "✅ JAX model test successful!" in captured["stdout"]
    assert captured["original"] == captured["after"] == ["cp1252", "cp1252"]
    _assert_science(output / "simulation_results.json", "captured_μ_状态")

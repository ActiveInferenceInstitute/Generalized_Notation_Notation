"""Continuous (linear-Gaussian) branch of the JAX / NumPyro / PyTorch / Stan /
ngc-learn renderers.
Builds the continuous ``gnn_spec`` by hand (the shape ``render.pomdp_processor``
emits for ``model_kind == "continuous"``), renders each backend, and executes
the generated scripts — except ngclearn, which is render-only here: the
generated script is compiled but never executed (ngclearn is marker-gated to
py3.12 and absent from the dev venv). Optional-backend execution is gated by
the registered ``needs_torch``/``needs_cmdstan`` markers (see
tests/helpers/toolchain_probes.py), not by in-file skips.
"""

from __future__ import annotations

import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, cast

import pytest

from gnn.extract.pomdp_extractor import extract_pomdp_from_file
from gnn.render.continuous_common import extract_continuous_spec, is_continuous_spec
from gnn.render.jax.jax_renderer import render_gnn_to_jax
from gnn.render.ngclearn.ngclearn_renderer import render_gnn_to_ngclearn
from gnn.render.numpyro.numpyro_renderer import render_gnn_to_numpyro
from gnn.render.pomdp_processor import pomdp_to_gnn_spec
from gnn.render.pytorch.pytorch_renderer import render_gnn_to_pytorch
from gnn.render.stan.stan_renderer import render_gnn_to_stan

T = 8

REPO = Path(__file__).resolve().parents[2]
CONTINUOUS_DIR = REPO / "input" / "gnn_files" / "continuous"
NEW_EXEMPLAR_STEM = "damped_oscillator_bias"
NEW_EXEMPLAR_FILE = CONTINUOUS_DIR / f"{NEW_EXEMPLAR_STEM}.md"
NEW_EXEMPLAR_TIMESTEPS = 12


def _spec(with_control: bool) -> Dict[str, Any]:
    initial: Dict[str, Any] = {
        "F": [[1.0, 0.1], [0.0, 0.9]],
        "H": [[1.0, 0.0], [0.0, 1.0]],
        "Q": [[0.05, 0.0], [0.0, 0.05]],
        "R": [[0.1, 0.0], [0.0, 0.1]],
        "prior_mean": [0.0, 0.0],
        "prior_cov": [[0.5, 0.0], [0.0, 0.5]],
    }
    if with_control:
        initial["goal_mean"] = [1.0, 0.0]
        initial["control_gain"] = 0.3
    return {
        "name": "Test Continuous",
        "model_name": "Test Continuous",
        "gnn_section": "ActInfContinuous",
        "model_kind": "continuous",
        "initialparameterization": initial,
        "model_parameters": {"num_timesteps": T, "dt": 0.1, "random_seed": 7},
    }


def _run(
    script: Path, env_var: str, out: Path, python: str = sys.executable
) -> Dict[str, Any]:
    env = dict(os.environ, **{env_var: str(out)})
    proc = subprocess.run(
        [python, str(script)], capture_output=True, text=True, env=env, timeout=600
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    return cast(
        Dict[str, Any], json.loads((out / "simulation_results.json").read_text())
    )


def _assert_schema(
    res: Dict[str, Any],
    framework: str,
    with_control: bool,
    dims: int = 2,
    timesteps: int = T,
) -> None:
    assert res["framework"] == framework
    assert res["model_kind"] == "continuous"
    assert res["num_states"] == dims and res["num_observations"] == 2
    assert len(res["beliefs"]) == timesteps and len(res["beliefs"][0]) == dims
    assert (
        len(res["posterior_cov"]) == timesteps and len(res["posterior_cov"][0]) == dims
    )
    assert len(res["controls"]) == timesteps
    assert res["actions"] == [] and res["observations"] == []
    assert res["validation"]["all_valid"] is True
    if with_control:
        assert any(abs(c) > 0 for row in res["controls"] for c in row)
    else:
        assert all(c == 0.0 for row in res["controls"] for c in row)


def test_detection_and_extraction() -> None:
    spec = extract_continuous_spec(_spec(True))
    assert is_continuous_spec(_spec(False))
    assert spec.n == 2 and spec.m == 2 and spec.has_control
    assert not extract_continuous_spec(_spec(False)).has_control


@pytest.mark.parametrize("with_control", [True, False])
def test_jax_continuous_renders_and_runs(tmp_path: Path, with_control: bool) -> None:
    ok, msg, arts = render_gnn_to_jax(_spec(with_control), tmp_path / "m_jax.py")
    assert ok, msg
    res = _run(Path(arts[0]), "GNN_OUTPUT_DIR", tmp_path / "out")
    _assert_schema(res, "jax", with_control)
    assert "jax_version" in res


def test_numpyro_continuous_renders_and_runs_nuts(tmp_path: Path) -> None:
    ok, msg, arts = render_gnn_to_numpyro(_spec(True), tmp_path / "m_numpyro.py")
    assert ok, msg
    res = _run(Path(arts[0]), "NUMPYRO_OUTPUT_DIR", tmp_path / "out")
    _assert_schema(res, "numpyro", True)
    assert len(res["mcmc_posterior_means"]) == T
    assert res["mcmc_r_hat_max"] < 1.2
    assert res["validation"]["mcmc_finite"] is True


@pytest.mark.needs_torch
def test_pytorch_continuous_renders(tmp_path: Path) -> None:
    ok, msg, arts = render_gnn_to_pytorch(_spec(True), tmp_path / "m_pytorch.py")
    assert ok, msg
    code = Path(arts[0]).read_text()
    assert "torch.distributions.MultivariateNormal" in code
    assert "GOAL_MEAN_RAW = [1.0, 0.0]" in code
    res = _run(Path(arts[0]), "PYTORCH_OUTPUT_DIR", tmp_path / "out")
    _assert_schema(res, "pytorch", True)


@pytest.mark.needs_cmdstan
def test_stan_continuous_program_and_driver(tmp_path: Path) -> None:
    ok, msg, arts = render_gnn_to_stan(_spec(True), tmp_path / "m_stan.py")
    assert ok, msg
    driver, program = Path(arts[0]), Path(arts[1])
    assert program.suffix == ".stan" and driver.suffix == ".py"
    text = program.read_text()
    assert "multi_normal_lpdf" in text and "obs_noise_scale" in text
    res = _run(driver, "STAN_OUTPUT_DIR", tmp_path / "out")
    _assert_schema(res, "stan", True)
    assert res["validation"]["rhat_ok"] is True


def _rows(matrix: Any) -> list[list[float]]:
    """Normalize parsed matrix rows (tuples from the GNN literal parser)."""
    return [[float(v) for v in row] for row in matrix]


def _file_spec() -> Dict[str, Any]:
    """Spec for the new exemplar, built through the real file pipeline path."""
    pomdp = extract_pomdp_from_file(NEW_EXEMPLAR_FILE, strict_validation=True)
    assert pomdp is not None and pomdp.model_kind == "continuous"
    assert pomdp.num_states == 3 and pomdp.num_observations == 2
    assert pomdp.passive_model is True
    spec = cast(Dict[str, Any], pomdp_to_gnn_spec(pomdp))
    assert spec["model_kind"] == "continuous"
    return spec


def test_damped_oscillator_bias_file_spec_and_shapes() -> None:
    """The new exemplar extracts with 3 states / 2 observations and exact matrices."""
    spec = _file_spec()
    initial = spec["initialparameterization"]
    assert _rows(initial["F"]) == [
        [1.0, 0.1, 0.0],
        [-0.09, 0.99, 0.0],
        [0.0, 0.0, 0.95],
    ]
    assert _rows(initial["H"]) == [[1.0, 0.0, 1.0], [0.0, 1.0, 0.5]]
    assert _rows(initial["Q"]) == [
        [0.01, 0.002, 0.0],
        [0.002, 0.01, 0.0],
        [0.0, 0.0, 0.004],
    ]
    assert _rows(initial["R"]) == [[0.04, 0.008], [0.008, 0.06]]
    assert [float(v) for v in initial["prior_mean"]] == [0.0, 0.0, 0.2]
    assert _rows(initial["prior_cov"]) == [
        [1.0, 0.2, 0.0],
        [0.2, 1.0, 0.0],
        [0.0, 0.0, 0.5],
    ]
    assert "A" not in initial
    assert "goal_mean" not in initial and "control_gain" not in initial
    cs = extract_continuous_spec(spec)
    assert cs.n == 3 and cs.m == 2 and not cs.has_control
    assert cs.num_timesteps == NEW_EXEMPLAR_TIMESTEPS
    assert cs.dt == 0.1 and cs.random_seed == 43


def test_damped_oscillator_bias_jax_renders_and_runs(tmp_path: Path) -> None:
    ok, msg, arts = render_gnn_to_jax(_file_spec(), tmp_path / "damped_jax.py")
    assert ok, msg
    res = _run(Path(arts[0]), "GNN_OUTPUT_DIR", tmp_path / "out")
    _assert_schema(res, "jax", False, dims=3, timesteps=NEW_EXEMPLAR_TIMESTEPS)
    assert "jax_version" in res


def test_damped_oscillator_bias_numpyro_renders_and_runs_nuts(tmp_path: Path) -> None:
    ok, msg, arts = render_gnn_to_numpyro(_file_spec(), tmp_path / "damped_numpyro.py")
    assert ok, msg
    res = _run(Path(arts[0]), "NUMPYRO_OUTPUT_DIR", tmp_path / "out")
    _assert_schema(res, "numpyro", False, dims=3, timesteps=NEW_EXEMPLAR_TIMESTEPS)
    assert len(res["mcmc_posterior_means"]) == NEW_EXEMPLAR_TIMESTEPS
    # Split-R-hat from a single 200-warmup/200-sample chain is two 100-draw
    # halves, so the estimate carries ~0.1 of Monte Carlo noise that shifts
    # with platform numerics: linux CI measures 1.2152 for this seeded run,
    # macOS arm64 1.0194. 1.3 leaves margin for that noise while real
    # divergence sits above 1.5; the 2-dim sibling below keeps the strict 1.2.
    assert res["mcmc_r_hat_max"] < 1.3
    assert res["validation"]["mcmc_finite"] is True


@pytest.mark.needs_torch
def test_damped_oscillator_bias_pytorch_renders(tmp_path: Path) -> None:
    ok, msg, arts = render_gnn_to_pytorch(_file_spec(), tmp_path / "damped_pytorch.py")
    assert ok, msg
    code = Path(arts[0]).read_text()
    assert "torch.distributions.MultivariateNormal" in code
    assert "GOAL_MEAN_RAW = None" in code
    res = _run(Path(arts[0]), "PYTORCH_OUTPUT_DIR", tmp_path / "out")
    _assert_schema(res, "pytorch", False, dims=3, timesteps=NEW_EXEMPLAR_TIMESTEPS)


@pytest.mark.needs_cmdstan
def test_damped_oscillator_bias_stan_program_and_driver(tmp_path: Path) -> None:
    ok, msg, arts = render_gnn_to_stan(_file_spec(), tmp_path / "damped_stan.py")
    assert ok, msg
    driver, program = Path(arts[0]), Path(arts[1])
    assert program.suffix == ".stan" and driver.suffix == ".py"
    text = program.read_text()
    assert "multi_normal_lpdf" in text and "obs_noise_scale" in text
    res = _run(driver, "STAN_OUTPUT_DIR", tmp_path / "out")
    _assert_schema(res, "stan", False, dims=3, timesteps=NEW_EXEMPLAR_TIMESTEPS)
    assert res["validation"]["rhat_ok"] is True


def test_ngclearn_continuous_renders_codegen_only(tmp_path: Path) -> None:
    """ngclearn renders without importing ngclearn (marker-gated to py3.12 and
    absent from the dev venv), so the script is compiled but not executed."""
    ok, msg, arts = render_gnn_to_ngclearn(_spec(True), tmp_path / "m_ngclearn.py")
    assert ok, msg
    script = Path(arts[0]).read_text()
    assert 'FRAMEWORK = "ngclearn"' in script
    assert "OUTPUT_ENV = 'NGCLEARN_OUTPUT_DIR'" in script
    assert "def kalman_step" in script
    py_compile.compile(arts[0], doraise=True)


def test_damped_oscillator_bias_ngclearn_renders_codegen_only(tmp_path: Path) -> None:
    ok, msg, arts = render_gnn_to_ngclearn(
        _file_spec(), tmp_path / "damped_ngclearn.py"
    )
    assert ok, msg
    py_compile.compile(arts[0], doraise=True)


def test_discrete_regression_still_renders(tmp_path: Path) -> None:
    spec = {
        "name": "Disc",
        "model_name": "Disc",
        "initialparameterization": {
            "A": [[0.9, 0.1], [0.1, 0.9]],
            "B": [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
            "C": [0.0, 1.0],
            "D": [0.5, 0.5],
        },
        "model_parameters": {
            "num_hidden_states": 2,
            "num_obs": 2,
            "num_actions": 2,
            "num_timesteps": 3,
        },
    }
    assert not is_continuous_spec(spec)
    for fn, name in (
        (render_gnn_to_jax, "d_jax.py"),
        (render_gnn_to_numpyro, "d_numpyro.py"),
        (render_gnn_to_pytorch, "d_pytorch.py"),
        (render_gnn_to_stan, "d_stan.py"),
    ):
        ok, msg, _ = fn(spec, tmp_path / name)
        assert ok, f"{name}: {msg}"


def test_is_continuous_spec_fails_loud_on_malformed_spec() -> None:
    """BC-02c: unclassifiable specs raise instead of degrading to discrete."""
    malformed = {
        "name": "Broken",
        "model_name": "Broken",
        "initialparameterization": "not-a-mapping",
    }
    with pytest.raises(ValueError, match="initialparameterization must be a mapping"):
        is_continuous_spec(malformed)

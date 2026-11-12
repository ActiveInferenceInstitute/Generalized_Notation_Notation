#!/usr/bin/env python3
"""
PyMDP contract tests for render/execute/analysis integration.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from gnn.analysis.framework_extractors import extract_pymdp_data
from gnn.execute.pymdp.simulation import run_pymdp_simulation
from gnn.render.processor import render_gnn_spec

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTINF_POMDP_PATH = (
    REPO_ROOT / "input" / "gnn_files" / "discrete" / "actinf_pomdp_agent.md"
)


def _pymdp_importable() -> bool:
    try:
        from pymdp.agent import Agent  # noqa: F401

        return True
    except ImportError:
        return False


def _actinf_gnn_spec(timesteps: int = 12) -> dict:
    """Same POMDP → spec path as Step 11 (POMDPRenderProcessor)."""
    import logging

    from gnn.extract.pomdp_extractor import extract_pomdp_from_file
    from gnn.render.pomdp_processor import POMDPRenderProcessor
    from gnn.render.processor import normalize_matrices

    pomdp = extract_pomdp_from_file(ACTINF_POMDP_PATH, strict_validation=True)
    assert pomdp is not None, (
        "extract_pomdp_from_file returned None for actinf_pomdp_agent.md"
    )
    pomdp = normalize_matrices(pomdp, logging.getLogger("test_pymdp_contracts"))
    proc = POMDPRenderProcessor(ACTINF_POMDP_PATH.parent)
    return proc._pomdp_to_gnn_spec(pomdp, timesteps=timesteps)


def _minimal_gnn_spec() -> dict:
    return {
        "name": "contract_test_model",
        "model_name": "contract_test_model",
        "initialparameterization": {
            "A": [[0.9, 0.1], [0.1, 0.9]],
            # (action, prev, next) format accepted by the canonical transpose logic
            "B": [[[0.8, 0.2], [0.2, 0.8]]],
            "C": [0.0, 1.0],
            "D": [0.5, 0.5],
        },
        "model_parameters": {"num_timesteps": 8},
    }


@pytest.mark.integration
def test_render_execute_contract_pymdp(tmp_path: Path) -> None:
    """
    Render should succeed and execution should produce schema-valid outputs.
    """
    gnn_spec = _minimal_gnn_spec()
    render_dir = tmp_path / "render"

    render_ok, msg, _artifacts = render_gnn_spec(gnn_spec, "pymdp", render_dir)
    assert render_ok, f"render failed: {msg}"

    # Execute using the local PyMDP execution pathway.
    exec_dir = tmp_path / "execute"
    exec_ok, results = run_pymdp_simulation(gnn_spec, exec_dir)
    if not exec_ok:
        raise AssertionError(
            f"PyMDP unavailable for runtime contract check: {results.get('error')}"
        )

    assert results.get("success") is True
    assert results.get("framework") == "PyMDP"
    assert results.get("schema_version") == "pymdp_simulation_v1"
    assert isinstance(results.get("observations"), list)
    assert isinstance(results.get("actions"), list)
    assert isinstance(results.get("beliefs"), list)
    assert len(results["observations"]) == gnn_spec["model_parameters"]["num_timesteps"]


@pytest.mark.integration
def test_extract_pymdp_data_from_real_execution_payload(tmp_path: Path) -> None:
    """
    Extractor should populate standard fields from a real simulation_results file.
    """
    gnn_spec = _minimal_gnn_spec()
    run_dir = tmp_path / "run"
    ok, results = run_pymdp_simulation(gnn_spec, run_dir)
    if not ok:
        raise AssertionError(
            f"PyMDP unavailable for extractor contract check: {results.get('error')}"
        )

    impl_dir = tmp_path / "impl"
    sim_data_dir = impl_dir / "simulation_data"
    sim_data_dir.mkdir(parents=True, exist_ok=True)
    payload_file = sim_data_dir / "contract_test_model_simulation_results.json"
    payload_file.write_text(json.dumps(results), encoding="utf-8")

    extracted = extract_pymdp_data(
        {
            "framework": "pymdp",
            "implementation_directory": str(impl_dir),
            "simulation_data": {},
        }
    )

    for key in [
        "free_energy",
        "beliefs",
        "observations",
        "actions",
        "states",
        "traces",
    ]:
        assert key in extracted
    assert len(extracted["beliefs"]) > 0
    assert len(extracted["observations"]) > 0


@pytest.mark.integration
def test_pymdp_seeded_reproducibility_contract(tmp_path: Path) -> None:
    """
    Repeated runs of one spec reproduce identical trajectories.

    The runner draws from ``np.random.default_rng(model_parameters.random_seed)``,
    never from the process-global numpy RNG, so the seed is part of the spec.
    """
    gnn_spec = _minimal_gnn_spec()
    gnn_spec["model_parameters"]["random_seed"] = 123

    ok1, res1 = run_pymdp_simulation(gnn_spec, tmp_path / "r1")
    if not ok1:
        raise AssertionError(
            f"PyMDP unavailable for reproducibility check: {res1.get('error')}"
        )

    ok2, res2 = run_pymdp_simulation(gnn_spec, tmp_path / "r2")
    assert ok2, "second PyMDP simulation run should succeed"

    assert res1["observations"] == res2["observations"]
    assert res1["actions"] == res2["actions"]


@pytest.mark.integration
def test_actinf_pomdp_golden_pymdp_simulation(tmp_path: Path) -> None:
    """
    Canonical ActInfPOMDP fixture: parsed matrices must run under run_pymdp_simulation
    without NaNs and with internal validation flags true (B-axis / D rounding robustness).
    """
    if not ACTINF_POMDP_PATH.is_file():
        raise AssertionError(f"Fixture missing: {ACTINF_POMDP_PATH}")
    if not _pymdp_importable():
        raise AssertionError("PyMDP (inferactively-pymdp) not installed")

    spec = _actinf_gnn_spec(timesteps=12)
    # Deterministic rollout from the spec's seed, not the process-global RNG.
    spec.setdefault("model_parameters", {})["random_seed"] = 42
    ok, results = run_pymdp_simulation(spec, tmp_path / "actinf_run")
    assert ok, f"golden ActInf POMDP simulation failed: {results.get('error')}"
    assert results.get("framework") == "PyMDP"
    val = results.get("validation") or {}
    assert val.get("beliefs_sum_to_one") is True
    assert val.get("all_beliefs_valid") is True
    assert val.get("actions_in_range") is True
    assert len(results.get("observations", [])) == 12
    assert not any(
        np.isnan(np.array(b, dtype=float)).any()
        for b in results.get("beliefs", [])
        if b
    )


@pytest.mark.integration
@pytest.mark.slow
def test_actinf_pomdp_render_execute_analyze_e2e(tmp_path: Path) -> None:
    """
    Render pymdp script → execute via Step-12 subprocess (GNN_PROJECT_ROOT) → collect
    simulation_results.json → analysis step sees PyMDP outputs.
    """
    if not ACTINF_POMDP_PATH.is_file():
        raise AssertionError(f"Fixture missing: {ACTINF_POMDP_PATH}")
    if not _pymdp_importable():
        raise AssertionError("PyMDP (inferactively-pymdp) not installed")

    from gnn.analysis.processor import process_analysis
    from gnn.execute.processor import process_execute
    from gnn.render.processor import process_render

    in_dir = tmp_path / "input" / "gnn_files"
    in_dir.mkdir(parents=True)
    shutil.copy(ACTINF_POMDP_PATH, in_dir / "actinf_pomdp_agent.md")

    base = tmp_path / "output"
    render_out = base / "11_render_output"
    exec_out = base / "12_execute_output"
    analysis_out = base / "16_analysis_output"

    render_ok = process_render(
        in_dir,
        render_out,
        verbose=False,
        frameworks=["pymdp"],
        strict_validation=True,
    )
    assert render_ok, "process_render should succeed for ActInf POMDP (pymdp only)"

    pymdp_scripts = list(render_out.rglob("*pymdp.py"))
    assert pymdp_scripts, f"expected *pymdp.py under {render_out}"

    exec_ok = process_execute(
        in_dir,
        exec_out,
        verbose=False,
        frameworks="pymdp",
        render_output_dir=render_out,
        timeout=180,
    )
    assert exec_ok, "process_execute should complete without fatal errors"

    collected = list(exec_out.glob("**/pymdp/simulation_data/*simulation_results.json"))
    assert collected, (
        f"no simulation_results.json under {exec_out}/**/pymdp/simulation_data/"
    )

    payload = json.loads(collected[0].read_text(encoding="utf-8"))
    assert payload.get("framework") == "PyMDP"
    assert payload.get("schema_version") == "pymdp_simulation_v1"
    beliefs = payload.get("beliefs_by_factor", {}).get("joint_state")
    obs = payload.get("observations_by_modality", {}).get("joint_observation")
    assert beliefs and obs, "payload should include beliefs and observations"

    analysis_ok = process_analysis(in_dir, analysis_out, verbose=False)
    assert analysis_ok, "analysis step should succeed"

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return

    pymdp_pngs = list((analysis_out / "pymdp").rglob("*.png"))
    assert pymdp_pngs, "with matplotlib, analysis should emit at least one PyMDP png"

"""Declared Gaussian agents retain identity, dimensions, controls and seeds."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from gnn.analysis.result_adapter import continuous_result_metrics, result_views
from gnn.execute.security_gate import check_script_allowed
from gnn.extract.pomdp_extractor import extract_pomdp_from_file
from gnn.render.multi_agent_continuous import extract_multi_agent_continuous_specs
from gnn.render.pomdp_contract import ModelKind, detect_model_kinds
from gnn.render.pomdp_processor import pomdp_to_gnn_spec
from gnn.render.processor import render_gnn_spec

SOURCE = Path("input/gnn_files/continuous/independent_gaussian_agents.md")


def _spec() -> dict:
    space = extract_pomdp_from_file(SOURCE, strict_validation=True)
    assert space is not None
    return pomdp_to_gnn_spec(space)


def test_asymmetric_source_retains_all_agent_blocks() -> None:
    spec = _spec()
    assert detect_model_kinds(spec) == frozenset(
        {ModelKind.MULTI_AGENT, ModelKind.CONTINUOUS}
    )
    agents = extract_multi_agent_continuous_specs(spec)
    assert [(agent.n, agent.m, agent.random_seed) for agent in agents.values()] == [
        (1, 1, 17),
        (2, 1, 93),
    ]
    assert agents["agent2"].H.tolist() == [[1.0, -0.4]]


@pytest.mark.parametrize(
    "failure", ["coupling", "missing", "extra", "partial_control", "nonfinite"]
)
def test_incomplete_or_ambiguous_agents_are_rejected(failure: str) -> None:
    spec = deepcopy(_spec())
    if failure == "coupling":
        spec["model_parameters"].pop("agent_coupling")
    elif failure == "missing":
        spec["initialparameterization"].pop("R_agent2")
    elif failure == "extra":
        spec["initialparameterization"]["F_agent3"] = [[1.0]]
    elif failure == "partial_control":
        spec["initialparameterization"].pop("control_gain_agent2")
    else:
        spec["initialparameterization"]["Q_agent2"][0][0] = float("nan")
    with pytest.raises(ValueError):
        extract_multi_agent_continuous_specs(spec)


def test_shared_position_composition_is_honestly_unsupported(tmp_path: Path) -> None:
    source = Path("input/gnn_files/continuous/multi_agent_lgssm.md")
    spec = pomdp_to_gnn_spec(extract_pomdp_from_file(source, strict_validation=True))
    for backend in ("jax", "rxinfer"):
        success, message, _ = render_gnn_spec(spec, backend, tmp_path / backend)
        assert not success and "agent_coupling: independent" in message


@pytest.mark.parametrize(
    "backend", ["jax", pytest.param("rxinfer", marks=pytest.mark.needs_julia_env)]
)
def test_native_agents_execute_with_source_bound_controls(
    tmp_path: Path, backend: str
) -> None:
    spec = _spec()
    success, message, artifacts = render_gnn_spec(spec, backend, tmp_path)
    assert success, message
    script = Path(artifacts[0])
    assert check_script_allowed(script)["ok"]
    command = (
        [sys.executable, str(script)]
        if backend == "jax"
        else [
            "julia",
            "--startup-file=no",
            f"--project={Path.cwd() / 'src/gnn/execute/rxinfer'}",
            str(script),
        ]
    )
    child = subprocess.run(
        command,
        cwd=tmp_path,
        env={**os.environ, "GNN_OUTPUT_DIR": str(tmp_path), "GKSwstype": "100"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert child.returncode == 0, child.stderr[-2000:]
    data = json.loads((tmp_path / "simulation_results.json").read_text())
    assert data["validation"]["all_valid"]
    views = result_views(data)
    assert list(views) == ["agent1", "agent2"]
    for name, agent in extract_multi_agent_continuous_specs(spec).items():
        view = views[name]
        assert continuous_result_metrics(view)["num_states"] == agent.n
        # RxInfer reports smoothed beliefs but controls were computed online
        # during data collection. JAX beliefs are the same online estimand.
        if backend == "jax":
            np.testing.assert_allclose(
                view["controls"],
                agent.control_gain * (agent.goal_mean - np.asarray(view["beliefs"])),
            )
        assert np.asarray(view["controls"]).shape == (4, agent.n)


@pytest.mark.parametrize("invalid_seed", [True, float("nan"), -1, 0.5])
def test_invalid_global_seed_cannot_be_hidden_by_agent_seed_overrides(
    invalid_seed: object,
) -> None:
    spec = _spec()
    spec["model_parameters"]["random_seed"] = invalid_seed
    with pytest.raises(ValueError):
        extract_multi_agent_continuous_specs(spec)


def test_untrusted_display_name_is_data_in_both_native_programs(tmp_path: Path) -> None:
    import ast
    import base64
    import re

    from gnn.render.multi_agent_continuous import generate_multi_agent_continuous_script

    spec = _spec()
    spec["model_name"] = '"""\nraise RuntimeError("injected")\n$(error("injected"))'
    python = generate_multi_agent_continuous_script(spec, "jax")
    ast.parse(python)
    julia = generate_multi_agent_continuous_script(spec, "rxinfer")
    programs = json.loads(
        re.search(r"^const PROGRAMS = (.+)$", julia, re.MULTILINE).group(1)
    )
    for encoded in programs:
        code = base64.b64decode(encoded).decode()
        assert r"\$(error" in code
        assert '\nraise RuntimeError("injected")\n' not in code

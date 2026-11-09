"""Traceable PyTorch rendering and opt-in TorchLens capture.

Static tests (no torch) pin the emitted module structure, the habit/seed
contracts and the GNN-variable-to-site mapping. ``needs_torch`` tests execute
the emitted script; ``needs_torchlens`` tests run the real TorchLens capture
and the Step 16 fidelity report end to end.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS = REPO_ROOT / "input" / "gnn_files"


def _spec(**overrides: Any) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "modelName": "pytorch_torchlens_smoke",
        "model_parameters": {
            "num_hidden_states": 2,
            "num_obs": 2,
            "num_timesteps": 6,
            "b_tensor_order": "next_state_previous_state_action",
        },
        "initialparameterization": {
            "A": [[0.85, 0.2], [0.15, 0.8]],
            "B": [
                [[0.9, 0.2], [0.3, 0.1]],
                [[0.1, 0.8], [0.7, 0.9]],
            ],
            "C": [0.0, 1.5],
            "D": [0.6, 0.4],
        },
        "connections": [
            {"source": "D", "relation": ">", "target": "s"},
            {"source": "s", "relation": "-", "target": "A"},
            {"source": "A", "relation": "-", "target": "o"},
            {"source": "(s, u)", "relation": ">", "target": "s_prime"},
            {"source": "C", "relation": ">", "target": "G"},
            {"source": "E", "relation": ">", "target": "π"},
            {"source": "G", "relation": ">", "target": "π"},
            {"source": "π", "relation": ">", "target": "u"},
            {"source": "F", "relation": "-", "target": "s"},
        ],
    }
    spec["model_parameters"]["state_factors"] = [
        {"name": "s", "role": "factor"},
        {"name": "s_prime", "role": "bookkeeping"},
    ]
    spec["model_parameters"]["observation_modalities"] = [
        {"name": "o", "role": "factor"}
    ]
    spec["model_parameters"]["control_factors"] = [
        {"name": "π", "role": "bookkeeping"},
        {"name": "u", "role": "factor"},
    ]
    for key, value in overrides.items():
        if key == "E":
            spec["initialparameterization"]["E"] = value
        else:
            spec[key] = value
    return spec


def _render(
    tmp_path: Path, spec: Dict[str, Any], options: Optional[dict] = None
) -> Path:
    from gnn.render.pytorch import render_gnn_to_pytorch

    script = tmp_path / "model_pytorch.py"
    ok, message, artifacts = render_gnn_to_pytorch(spec, script, options)
    assert ok, message
    assert artifacts == [str(script)]
    return script


def _run(
    script: Path, output: Path, extra_env: Optional[Dict[str, str]] = None
) -> subprocess.CompletedProcess:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("GNN_TORCHLENS_CAPTURE", "GNN_TORCHLENS_ARTIFACTS")
    }
    env["PYTORCH_OUTPUT_DIR"] = str(output)
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=script.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


# --- Static emission contracts -------------------------------------------------


def test_emitted_script_defines_gnn_named_modules(tmp_path: Path) -> None:
    script = _render(tmp_path, _spec())
    source = script.read_text(encoding="utf-8")
    tree = ast.parse(source)
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert {
        "Transition",
        "StateInference",
        "Ambiguity",
        "Risk",
        "ExpectedFreeEnergy",
        "PolicySelection",
        "ActionSampling",
        "ActiveInferenceAgent",
        "GenerativeProcess",
        "ActiveInferenceRollout",
    } <= classes
    for buffer in ("A", "B", "C", "D", "E"):
        assert f'self.register_buffer("{buffer}", {buffer})' in source
    assert "self.process = GenerativeProcess(A.clone(), B.clone(), D.clone())" in source
    assert "torch.Generator().manual_seed(int(seed))" in source
    assert "T = 6" in source
    assert "RANDOM_SEED = 42" in source


def test_torchlens_is_never_imported_at_module_scope(tmp_path: Path) -> None:
    tree = ast.parse(_render(tmp_path, _spec()).read_text(encoding="utf-8"))
    module_imports = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            module_imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            module_imports.add(node.module)
        elif isinstance(node, ast.Try):
            for inner in node.body:
                if isinstance(inner, ast.Import):
                    module_imports.update(alias.name for alias in inner.names)
    assert "torch" in module_imports
    assert not any(name.startswith("torchlens") for name in module_imports)
    capture = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_torchlens_capture"
    )
    nested = [
        alias.name
        for node in ast.walk(capture)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert nested == ["torchlens"]


def test_declared_habit_is_emitted_and_uniform_is_the_default(tmp_path: Path) -> None:
    declared = _render(tmp_path, _spec(E=[0.25, 0.75])).read_text(encoding="utf-8")
    assert "HABIT_SOURCE = 'declared'" in declared
    assert "E = torch.tensor([0.25, 0.75], dtype=torch.float64)" in declared
    default = _render(tmp_path, _spec()).read_text(encoding="utf-8")
    assert "HABIT_SOURCE = 'uniform_default'" in default
    assert "E = torch.tensor([0.5, 0.5], dtype=torch.float64)" in default


@pytest.mark.parametrize(
    "habit,needle",
    [
        ([0.2, 0.2], "probability mass must be one"),
        ([1.2, -0.2], "finite nonnegative"),
        ([0.2, 0.3, 0.5], "E has 3 entries but B declares 2 actions"),
    ],
)
def test_invalid_habit_is_refused(tmp_path: Path, habit: list, needle: str) -> None:
    from gnn.render.pytorch import render_gnn_to_pytorch

    ok, message, artifacts = render_gnn_to_pytorch(_spec(E=habit), tmp_path / "bad.py")
    assert not ok and artifacts == []
    assert needle in message
    assert not (tmp_path / "bad.py").exists()


def test_random_seed_precedence_and_validation(tmp_path: Path) -> None:
    spec = _spec()
    spec["model_parameters"]["random_seed"] = 7
    assert "RANDOM_SEED = 7" in _render(tmp_path, spec).read_text(encoding="utf-8")
    assert "RANDOM_SEED = 11" in _render(tmp_path, spec, {"seed": 11}).read_text(
        encoding="utf-8"
    )
    from gnn.render.pytorch import render_gnn_to_pytorch

    for bad in (-1, 1.5, True, 2**64):
        ok, message, _ = render_gnn_to_pytorch(
            _spec(), tmp_path / "bad_seed.py", {"seed": bad}
        )
        assert not ok and "random_seed" in message


def test_horizon_below_one_is_refused(tmp_path: Path) -> None:
    from gnn.render.pytorch import render_gnn_to_pytorch

    ok, message, _ = render_gnn_to_pytorch(
        _spec(), tmp_path / "zero.py", {"num_timesteps": 0}
    )
    assert not ok and "num_timesteps must be at least 1" in message


def test_emitted_sites_and_connections_match_the_specification(tmp_path: Path) -> None:
    namespace = _emitted_constants(_render(tmp_path, _spec()))
    assert namespace["GNN_DECLARED_CONNECTIONS"][3] == {
        "source": "(s, u)",
        "relation": ">",
        "target": "s_prime",
    }
    assert namespace["GNN_VARIABLE_SITES"]["s"] == ["agent.state_inference"]
    assert namespace["GNN_VARIABLE_SITES"]["s_prime"] == ["agent.transition"]
    assert namespace["GNN_VARIABLE_SITES"]["o"] == ["process.observe"]
    assert namespace["GNN_VARIABLE_SITES"]["π"] == ["agent.policy"]
    assert namespace["GNN_VARIABLE_SITES"]["u"] == ["agent.action"]
    assert namespace["GNN_VARIABLE_SITES"]["A"] == ["agent.A", "process.A"]
    assert "F" in namespace["GNN_UNMAPPED_VARIABLES"]


# --- Variable-to-site mapping --------------------------------------------------


def test_connection_endpoints_expand_groups() -> None:
    from gnn.render.pytorch.sites import connection_endpoints

    assert connection_endpoints("(s_f0, u , )") == ["s_f0", "u"]
    assert connection_endpoints(" s ") == ["s"]
    assert connection_endpoints("") == []


def test_multiple_factors_are_reported_unmapped_not_guessed() -> None:
    from gnn.render.pytorch.sites import gnn_variable_sites

    spec = {
        "model_parameters": {
            "state_factors": [
                {"name": "s_f0", "role": "factor"},
                {"name": "s_f1", "role": "factor"},
            ]
        },
        "connections": [
            {"source": "s_f0", "relation": "-", "target": "A"},
            {"source": "s_f1", "relation": "-", "target": "A"},
            {"source": "mystery", "relation": ">", "target": "G_epi"},
        ],
    }
    sites, unmapped = gnn_variable_sites(spec)
    assert sites == {"A": ["agent.A", "process.A"], "G_epi": ["agent.efe.ambiguity"]}
    assert "2 hidden-state factor variables are flattened" in unmapped["s_f0"]
    assert "s_f1" in unmapped and "mystery" in unmapped


def _emitted_constants(script: Path) -> Dict[str, Any]:
    constants: Dict[str, Any] = {}
    for node in ast.parse(script.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.startswith("GNN_") or name in ("HABIT_SOURCE", "RANDOM_SEED"):
                constants[name] = ast.literal_eval(node.value)
    return constants


def _render_corpus(tmp_path: Path, relative: str) -> Dict[str, Any]:
    """Render one corpus model through the Step 11 entry point."""
    from gnn import parse_gnn_file
    from gnn.render.processor import render_gnn_spec

    ok, message, paths = render_gnn_spec(
        parse_gnn_file(CORPUS / relative), "pytorch", tmp_path
    )
    assert ok, message
    script = next(Path(p) for p in paths if p.endswith("_pytorch.py"))
    return _emitted_constants(script)


def test_corpus_actinf_agent_maps_every_connection_variable(tmp_path: Path) -> None:
    constants = _render_corpus(tmp_path, "discrete/actinf_pomdp_agent.md")
    sites = constants["GNN_VARIABLE_SITES"]
    assert constants["GNN_UNMAPPED_VARIABLES"] == {}
    assert constants["HABIT_SOURCE"] == "declared"
    assert sites["s"] == ["agent.state_inference"]
    assert sites["s_prime"] == ["agent.transition"]
    assert sites["o"] == ["process.observe"]
    assert sites["G"] == ["agent.efe"]
    assert sites["π"] == ["agent.policy"]
    assert sites["u"] == ["agent.action"]
    assert sites["E"] == ["agent.E"]
    assert {"source": "E", "relation": ">", "target": "π"} in constants[
        "GNN_DECLARED_CONNECTIONS"
    ]


def test_corpus_dynamic_perception_reports_free_energy_unmapped(tmp_path: Path) -> None:
    constants = _render_corpus(tmp_path, "basics/dynamic_perception.md")
    assert constants["GNN_UNMAPPED_VARIABLES"]["F"].startswith(
        "variational free energy is not computed"
    )
    assert constants["GNN_VARIABLE_SITES"]["s_t"] == ["agent.state_inference"]
    assert constants["HABIT_SOURCE"] == "uniform_default"


# --- Executed script (torch) ---------------------------------------------------


def _policy_witness(result: Dict[str, Any], habit: np.ndarray) -> np.ndarray:
    efe = np.asarray(result["efe_history"], dtype=float)
    logits = np.log(habit + 1e-16)[None, :] - efe
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


@pytest.mark.needs_torch
def test_rollout_is_seeded_reproducible_and_capture_is_off_by_default(
    tmp_path: Path,
) -> None:
    script = _render(tmp_path, _spec(E=[0.3, 0.7]))
    first = _run(script, tmp_path / "first")
    second = _run(script, tmp_path / "second")
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    a = json.loads((tmp_path / "first" / "simulation_results.json").read_text())
    b = json.loads((tmp_path / "second" / "simulation_results.json").read_text())
    for key in ("beliefs", "actions", "observations", "efe_history", "policy_history"):
        assert a[key] == b[key], key
    assert a["torchlens"]["status"] == "disabled"
    assert not (tmp_path / "first" / "torchlens").exists()
    assert a["habit"] == {"source": "declared", "E": [0.3, 0.7]}
    assert a["random_seed"] == 42
    np.testing.assert_allclose(
        np.asarray(a["policy_history"]),
        _policy_witness(a, np.array([0.3, 0.7])),
        atol=1e-12,
    )


@pytest.mark.needs_torch
def test_missing_torchlens_is_recorded_unavailable_without_failing(
    tmp_path: Path,
) -> None:
    blocker = tmp_path / "blocked" / "torchlens"
    blocker.mkdir(parents=True)
    (blocker / "__init__.py").write_text(
        "raise ImportError('torchlens blocked by test')\n", encoding="utf-8"
    )
    script = _render(tmp_path, _spec())
    run = _run(
        script,
        tmp_path / "out",
        {
            "GNN_TORCHLENS_CAPTURE": "1",
            "PYTHONPATH": os.pathsep.join(
                [str(tmp_path / "blocked"), os.environ.get("PYTHONPATH", "")]
            ),
        },
    )
    assert run.returncode == 0, run.stderr
    result = json.loads((tmp_path / "out" / "simulation_results.json").read_text())
    assert result["validation"]["all_valid"]
    assert result["torchlens"]["status"] == "unavailable"
    assert "uv sync --extra torchlens" in result["torchlens"]["reason"]


@pytest.mark.needs_torch
def test_invalid_artifact_mode_fails_capture_but_not_the_simulation(
    tmp_path: Path,
) -> None:
    script = _render(tmp_path, _spec())
    run = _run(
        script,
        tmp_path / "out",
        {"GNN_TORCHLENS_CAPTURE": "yes", "GNN_TORCHLENS_ARTIFACTS": "everything"},
    )
    assert run.returncode == 0, run.stderr
    result = json.loads((tmp_path / "out" / "simulation_results.json").read_text())
    assert result["torchlens"]["status"] == "failed"
    assert "GNN_TORCHLENS_ARTIFACTS" in result["torchlens"]["reason"]


# --- Real TorchLens capture and Step 16 fidelity -------------------------------


@pytest.mark.needs_torchlens
def test_capture_reproduces_the_run_and_feeds_the_fidelity_report(
    tmp_path: Path,
) -> None:
    script = _render(tmp_path, _spec(E=[0.3, 0.7]))
    plain = _run(script, tmp_path / "plain")
    assert plain.returncode == 0, plain.stderr
    execute_root = tmp_path / "12_execute_output"
    captured_dir = execute_root / "smoke" / "pytorch" / "simulation_data"
    captured = _run(
        script,
        captured_dir,
        {"GNN_TORCHLENS_CAPTURE": "1", "GNN_TORCHLENS_ARTIFACTS": "full"},
    )
    assert captured.returncode == 0, captured.stderr
    plain_result = json.loads(
        (tmp_path / "plain" / "simulation_results.json").read_text()
    )
    result = json.loads((captured_dir / "simulation_results.json").read_text())
    for key in ("beliefs", "actions", "observations", "efe_history", "policy_history"):
        assert result[key] == plain_result[key], key

    summary = result["torchlens"]
    assert summary["status"] == "captured", summary
    assert summary["trace_reproduces_simulation"] is True
    assert summary["agent_step_validation"] == {"first_step": True, "last_step": True}
    assert summary["validated"] is True
    assert summary["num_ops"] > 0
    edges = {(source, target) for source, target, _ in summary["site_edges"]}
    assert ("agent.E", "agent.policy") in edges
    assert ("agent.efe", "agent.policy") in edges
    assert ("process.observe", "agent.state_inference") in edges
    assert ("agent.action", "agent.transition") in edges
    # The rolled SVG needs the Graphviz ``dot`` binary; a host without it
    # must record that one failure instead of dropping the other artifacts.
    allowed_errors = set() if shutil.which("dot") else {"graph_rolled_svg"}
    assert set(summary["artifact_errors"]) <= allowed_errors, summary["artifact_errors"]
    assert {"report_html", "tlspec", "ops_csv", "manifest"} <= set(summary["artifacts"])
    for relative in summary["artifacts"].values():
        assert (captured_dir / relative).exists(), relative
    witness = summary["intervention_witnesses"]
    assert witness["timestep"] == 5
    assert set(witness["lesions"]) == {"no_ambiguity", "no_risk"}
    assert np.isclose(sum(witness["policy"]), 1.0)

    from gnn.analysis.pytorch.analyzer import generate_analysis_from_logs

    analysis_dir = tmp_path / "16_analysis_output" / "pytorch"
    written = generate_analysis_from_logs(execute_root, analysis_dir)
    fidelity_path = analysis_dir / "smoke" / "pytorch_torchlens_fidelity.json"
    assert str(fidelity_path) in written
    report = json.loads(fidelity_path.read_text())
    assert report["status"] == "evaluated"
    statuses = {
        (edge["source"], edge["relation"], edge["target"]): edge["status"]
        for edge in report["declared_edges"]
    }
    assert statuses[("D", ">", "s")] == "direct"
    assert statuses[("E", ">", "π")] == "direct"
    assert statuses[("G", ">", "π")] == "direct"
    assert statuses[("π", ">", "u")] == "direct"
    assert statuses[("(s, u)", ">", "s_prime")] == "direct"
    assert statuses[("F", "-", "s")] == "unmapped"
    assert report["structural_agreement"] == 1.0
    assert (analysis_dir / "smoke" / "pytorch_torchlens_fidelity.md").exists()

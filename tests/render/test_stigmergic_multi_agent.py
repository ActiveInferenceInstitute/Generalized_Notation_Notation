"""Tests for native stigmergic multi-agent compilation (roadmap MAJ-03).

Pins the native multi-agent rendering contract introduced for the swarm
exemplar:

- ``render.multi_agent_common`` detects complete per-agent matrix groups
  (``A_agentN``/``B_agentN``/``C_agentN``/``D_agentN``) and the shared
  environmental affordance (``env_signal`` + ``signal_decay``).
- The RxInfer.jl strategy emits one genuine ``pomdp_model`` inference per
  agent (no joint state-space expansion), reconstructs a shared ``env_signal``
  trace (deposit at MAP position, decay per timestep), and — when the spec
  declares an env-conditioned observation likelihood (MAJ-03) — infers the
  local signal level as a latent from observations and conditions action
  selection on it (signal-seeking).
- The ActiveInference.jl renderer emits the equivalent per-agent simulation
  with the same post-hoc trace and the same MAJ-03 env-conditioned latent
  signal inference / action-conditioning when declared.

Pure-Python structure tests run unconditionally; Julia parse and execution
tests are gated by the ``needs_julia``/``needs_julia_env`` markers
(tests/helpers/toolchain_probes.py), like the other live-backend gates.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from gnn.extract.pomdp_extractor import extract_pomdp_from_file
from gnn.render.activeinference_jl.activeinference_renderer import (
    render_gnn_to_activeinference_jl,
)
from gnn.render.multi_agent_common import (
    detect_agent_groups,
    detect_env_conditioned,
    detect_env_coupling,
    has_env_conditioned_action_selection,
    has_native_multi_agent_structure,
)
from gnn.render.pomdp_contract import build_canonical_pomdp_spec
from gnn.render.pomdp_processor import pomdp_to_gnn_spec
from gnn.render.rxinfer.rxinfer_renderer import render_gnn_to_rxinfer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GNN_FILES = PROJECT_ROOT / "input" / "gnn_files"
SWARM_FILE = GNN_FILES / "multiagent" / "stigmergic_swarm.md"
COORDINATION_FILE = GNN_FILES / "multiagent" / "multi_agent_coordination.md"
GRIDWORLD_FILE = GNN_FILES / "pomdp_gridworld" / "pomdp_gridworld_3x3.md"
RXINFER_JULIA_PROJECT = str(PROJECT_ROOT / "src" / "gnn" / "execute" / "rxinfer")
ACTINF_JULIA_PROJECT = str(
    PROJECT_ROOT / "src" / "gnn" / "execute" / "activeinference_jl"
)


def _canonical_spec(gnn_file: Path) -> dict:
    pomdp = extract_pomdp_from_file(gnn_file, strict_validation=True)
    assert pomdp is not None, f"extraction failed for {gnn_file}"
    return build_canonical_pomdp_spec(pomdp_to_gnn_spec(pomdp))


def _render_rxinfer(gnn_file: Path, tmp_path: Path) -> Path:
    script = tmp_path / f"{gnn_file.stem}_rxinfer.jl"
    ok, message, _warnings = render_gnn_to_rxinfer(_canonical_spec(gnn_file), script)
    assert ok, f"rxinfer render failed: {message}"
    return script


def _render_activeinference_jl(gnn_file: Path, tmp_path: Path) -> Path:
    script = tmp_path / f"{gnn_file.stem}_aijl.jl"
    ok, message, _artifacts = render_gnn_to_activeinference_jl(
        _canonical_spec(gnn_file), script
    )
    assert ok, f"activeinference_jl render failed: {message}"
    return script


def _run_julia(
    script: Path, project: str, workdir: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # nosec B603 B607
        ["julia", f"--project={project}", "--startup-file=no", str(script)],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=300,
    )


class TestMultiAgentDetection:
    """Shared detection layer over the parsed GNN spec."""

    def test_swarm_declares_three_complete_agent_groups(self) -> None:
        groups = detect_agent_groups(_canonical_spec(SWARM_FILE))
        assert sorted(groups) == ["agent1", "agent2", "agent3"]
        for group in groups.values():
            assert set(group) == {"A", "B", "C", "D"}

    def test_coordination_declares_two_complete_agent_groups(self) -> None:
        groups = detect_agent_groups(_canonical_spec(COORDINATION_FILE))
        assert sorted(groups) == ["agent1", "agent2"]
        for group in groups.values():
            assert set(group) == {"A", "B", "C", "D"}

    def test_flat_model_has_no_agent_groups(self) -> None:
        assert detect_agent_groups(_canonical_spec(GRIDWORLD_FILE)) == {}

    def test_swarm_declares_environment_coupling(self) -> None:
        env = detect_env_coupling(_canonical_spec(SWARM_FILE))
        assert env is not None
        assert env["variable"] == "env_signal"
        assert len(env["initial"]) == 9
        assert env["decay"] == pytest.approx(0.9)

    def test_coordination_has_no_environment_coupling(self) -> None:
        assert detect_env_coupling(_canonical_spec(COORDINATION_FILE)) is None

    def test_swarm_declares_env_conditioned_likelihood(self) -> None:
        env_cond = detect_env_conditioned(_canonical_spec(SWARM_FILE))
        assert env_cond is not None
        assert len(env_cond["obs_likelihood"]) == 4  # empty/low/high/goal
        assert len(env_cond["obs_likelihood"][0]) == 3  # none/low/high
        assert len(env_cond["signal_prior"]) == 3
        assert env_cond["seek"] == pytest.approx(2.0)

    def test_swarm_conditions_action_selection_on_env(self) -> None:
        assert has_env_conditioned_action_selection(_canonical_spec(SWARM_FILE))

    def test_coordination_is_not_env_conditioned(self) -> None:
        assert detect_env_conditioned(_canonical_spec(COORDINATION_FILE)) is None
        assert not has_env_conditioned_action_selection(
            _canonical_spec(COORDINATION_FILE)
        )

    def test_native_structure_threshold(self) -> None:
        assert has_native_multi_agent_structure(_canonical_spec(SWARM_FILE))
        assert has_native_multi_agent_structure(_canonical_spec(COORDINATION_FILE))
        assert not has_native_multi_agent_structure(_canonical_spec(GRIDWORLD_FILE))


def _assert_swarm_native_script(text: str) -> None:
    """Structure shared by both backends' native multi-agent scripts.

    Agent count is per-model (swarm=3, coordination=2) and asserted at the
    call site.
    """
    assert 'const MODEL_KIND = "multi_agent"' in text
    assert "const AGENT_AS" in text
    assert "function simulate_agent" in text
    assert "function compute_env_trace" in text
    assert "env_signal_trace" in text
    # No joint state-space expansion in the executed model.
    assert "const NUM_STATES = 729" not in text


def _assert_env_conditioned_script(text: str) -> None:
    """MAJ-03: env-conditioned latent signal inference + action conditioning."""
    assert "const ENV_DECAY = 0.9" in text
    assert "const ENV_ACTION_CONDITIONED = true" in text
    assert "const ENV_OBS_LIKELIHOOD" in text
    assert "const ENV_SIGNAL_PRIOR = [0.7" in text
    assert "function update_signal_belief" in text
    assert "function signal_seeking_preference" in text
    assert '"mode" => "env_conditioned_signal_selection"' in text
    assert '"latent_inference" => ENV_ACTION_CONDITIONED' in text
    assert '"action_selection_conditioned" => ENV_ACTION_CONDITIONED' in text


def _assert_unconditioned_script(text: str) -> None:
    """Coordination scripts keep post-hoc trace handling, no latent inference."""
    assert "const ENV_ACTION_CONDITIONED = false" in text
    assert '"mode" => "post_hoc_deposit_decay_trace"' in text
    assert '"latent_inference" => ENV_ACTION_CONDITIONED' in text
    assert '"action_selection_conditioned" => ENV_ACTION_CONDITIONED' in text


def _assert_flat_script(text: str, *, assert_model_kind: bool = True) -> None:
    """A flat model must not pick up multi-agent scaffolding."""
    assert "simulate_agent" not in text
    assert "AGENT_AS" not in text
    if assert_model_kind:
        assert 'const MODEL_KIND = "multi_agent"' not in text


class TestRxInferStigmergicScript:
    """The rxinfer native multi-agent script structure."""

    def test_swarm_renders_native_per_agent_script(self, tmp_path: Path) -> None:
        text = _render_rxinfer(SWARM_FILE, tmp_path).read_text(encoding="utf-8")
        _assert_swarm_native_script(text)
        assert "const NUM_AGENTS = 3" in text
        assert 'const AGENTS = ["agent1", "agent2", "agent3"]' in text
        assert "const AGENT_BS" in text
        assert (
            "pomdp_model(A=A, B=B, D=D, u=model_actions, T=TIME_STEPS)" in text
        )  # one genuine inference per agent

    def test_swarm_script_embeds_declared_env_coupling(self, tmp_path: Path) -> None:
        text = _render_rxinfer(SWARM_FILE, tmp_path).read_text(encoding="utf-8")
        assert "env_signal" in text
        assert "const ENV_INITIAL = [0.0" in text
        _assert_env_conditioned_script(text)
        assert "const SIGNAL_SEEK = 2.0" in text

    def test_coordination_script_stays_unconditioned(self, tmp_path: Path) -> None:
        text = _render_rxinfer(COORDINATION_FILE, tmp_path).read_text(encoding="utf-8")
        _assert_unconditioned_script(text)

    def test_coordination_renders_native_without_env(self, tmp_path: Path) -> None:
        text = _render_rxinfer(COORDINATION_FILE, tmp_path).read_text(encoding="utf-8")
        _assert_swarm_native_script(text)
        assert "const NUM_AGENTS = 2" in text
        assert "const NUM_STATES = 16" not in text

    def test_flat_model_renders_through_flat_strategy(self, tmp_path: Path) -> None:
        """A flat model must not pick up multi-agent scaffolding."""
        text = _render_rxinfer(GRIDWORLD_FILE, tmp_path).read_text(encoding="utf-8")
        _assert_flat_script(text)


class TestActiveInferenceJlStigmergicScript:
    """The activeinference_jl native multi-agent script structure."""

    def test_swarm_renders_native_per_agent_script(self, tmp_path: Path) -> None:
        text = _render_activeinference_jl(SWARM_FILE, tmp_path).read_text(
            encoding="utf-8"
        )
        _assert_swarm_native_script(text)
        assert "const NUM_AGENTS = 3" in text
        _assert_env_conditioned_script(text)

    def test_coordination_renders_native_without_env(self, tmp_path: Path) -> None:
        text = _render_activeinference_jl(COORDINATION_FILE, tmp_path).read_text(
            encoding="utf-8"
        )
        _assert_swarm_native_script(text)
        assert "const NUM_AGENTS = 2" in text
        _assert_unconditioned_script(text)

    def test_flat_model_renders_through_flat_path(self, tmp_path: Path) -> None:
        text = _render_activeinference_jl(GRIDWORLD_FILE, tmp_path).read_text(
            encoding="utf-8"
        )
        _assert_flat_script(text)


@pytest.mark.needs_julia
class TestJuliaParse:
    """Both generated scripts must be accepted by the Julia parser.

    The probe is the shared ``render.emitted_artifact_checks`` one: a bare
    ``Meta.parseall`` probe reports success for unbalanced delimiters, a stray
    ``end`` and truncated expressions, so it gated nothing.
    """

    @staticmethod
    def _assert_parses(script: Path) -> None:
        from gnn.render.emitted_artifact_checks import julia_syntax_findings

        findings, available = julia_syntax_findings(script.read_text(encoding="utf-8"))
        assert available, "julia must be on PATH for this case"
        assert findings == [], f"Julia rejected {script.name}: {findings}"

    @pytest.mark.parametrize(
        ("renderer", "suffix"),
        [
            (_render_rxinfer, "_rxinfer.jl"),
            (_render_activeinference_jl, "_aijl.jl"),
        ],
    )
    def test_swarm_script_parses(
        self,
        renderer: Callable[[Path, Path], Path],
        suffix: str,
        tmp_path: Path,
    ) -> None:
        self._assert_parses(renderer(SWARM_FILE, tmp_path))

    @pytest.mark.parametrize(
        "renderer",
        [_render_rxinfer, _render_activeinference_jl],
    )
    def test_coordination_script_parses(
        self,
        renderer: Callable[[Path, Path], Path],
        tmp_path: Path,
    ) -> None:
        self._assert_parses(renderer(COORDINATION_FILE, tmp_path))


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.needs_julia_env
class TestJuliaExecution:
    """Live execution of the stigmergic swarm scripts (both backends)."""

    def _assert_swarm_results(self, workdir: Path) -> dict:
        results_path = workdir / "simulation_results.json"
        assert results_path.exists(), "simulation_results.json was not produced"
        results: dict = json.loads(results_path.read_text(encoding="utf-8"))
        assert results["model_kind"] == "multi_agent"
        assert results["num_agents"] == 3
        assert sorted(results["agents"]) == ["agent1", "agent2", "agent3"]
        assert results["validation"]["all_valid"] is True
        assert all(
            per_agent["all_valid"] is True
            for per_agent in results["validation"]["per_agent"].values()
        )
        assert len(results["env_signal_trace"]) == results["num_timesteps"] + 1
        assert len(results["env_signal_trace"][0]) == 9
        assert results["env_coupling"] == {
            "variable": "env_signal",
            "initial": [0.0] * 9,
            "decay": 0.9,
            "mode": "env_conditioned_signal_selection",
            "latent_inference": True,
            "action_selection_conditioned": True,
            "signal_prior": [0.7, 0.2, 0.1],
            "signal_seek": 2.0,
        }
        for agent in results["agents"]:
            assert len(results["actions_by_agent"][agent]) == results["num_timesteps"]
            assert len(results["beliefs_by_agent"][agent]) == results["num_timesteps"]
            # MAJ-03: each agent maintains a latent signal belief over time.
            sig = results["env_signal_belief_by_agent"][agent]
            assert len(sig) == results["num_timesteps"]
            assert all(abs(sum(b) - 1.0) < 1e-6 for b in sig)  # normalized
        # The executed model never expands the joint state space.
        assert results["model_parameters"]["per_agent_state_sizes"] == [9, 9, 9]
        assert results["model_parameters"]["joint_state_space_size"] == 729
        return results

    def test_rxinfer_swarm_executes(self, tmp_path: Path) -> None:
        script = _render_rxinfer(SWARM_FILE, tmp_path)
        result = _run_julia(script, RXINFER_JULIA_PROJECT, tmp_path)
        assert result.returncode == 0, (
            f"rxinfer stigmergic execution failed:\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        results = self._assert_swarm_results(tmp_path)
        env_trace = results["env_signal_trace"]
        assert env_trace[0] == [0.0] * 9  # declared all-zero initialisation
        assert any(sum(step) > 0 for step in env_trace[1:])  # deposits accumulate

    def test_activeinference_jl_swarm_executes(self, tmp_path: Path) -> None:
        script = _render_activeinference_jl(SWARM_FILE, tmp_path)
        result = _run_julia(script, ACTINF_JULIA_PROJECT, tmp_path)
        assert result.returncode == 0, (
            f"activeinference_jl stigmergic execution failed:\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        self._assert_swarm_results(tmp_path)

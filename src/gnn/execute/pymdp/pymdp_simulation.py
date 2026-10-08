#!/usr/bin/env python3
"""
PyMDPSimulation — GNN-driven wrapper around pymdp 1.0.0 (JAX-first).

This class builds a real pymdp 1.0.0 ``Agent`` from a GNN configuration
dictionary and runs an active-inference rollout. It preserves the public
surface used by the pipeline tests:

    sim = PyMDPSimulation(gnn_config={...}, output_dir=Path(...))
    sim.num_states, sim.num_actions, sim.num_observations  # ints
    sim.state_names, sim.action_names, sim.observation_names  # list[str]
    sim.agent            # pymdp.agent.Agent instance (batch_size=1)
    sim.A, sim.B, sim.C, sim.D  # length-1 lists of numpy arrays
    sim.A[0].shape == (num_obs, num_states)
    sim.B[0].shape == (num_states, num_states, num_actions)
    sim.run_simulation(num_timesteps=T) -> {
        'observations': List[int], 'actions': List[int],
        'beliefs': List[List[float]], 'performance': {...},
        'trace': List[dict], 'success': True
    }

Architecture note: this lives in the EXECUTE step (12). It runs simulations
and writes raw data only. Visualisation belongs to the ANALYSIS step (16),
which reads ``simulation_results.json`` via ``src/analysis/pymdp``.
"""

from __future__ import annotations

import logging
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from .pymdp_utils import (
    convert_numpy_for_json,
    format_duration,
    safe_json_dump,
    safe_pickle_dump,
)

# pymdp 1.0.0 emits a stream of deprecation warnings at import time. Silence
# them only for this import (not process-wide) so user code keeps its own
# warning filters intact.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from .simulation import (
        _build_pymdp_agent,
        _canonicalise_A,
        _canonicalise_B,
        _canonicalise_C,
        _canonicalise_D,
        _canonicalise_E,
        _normalise_prob_vector,
        _require_pymdp_1,
    )

# The convention constant lives in .simulation (single source; this module
# imports from .simulation below, so the dependency direction is acyclic).
from .simulation import (  # noqa: E402
    EFE_CONVENTION_PYMDP,
    EFE_CONVENTION_PYMDP_SUMMARY,
)


class PyMDPSimulation:
    """
    GNN-configured active inference simulation using real pymdp 1.0.0.

    This class calls the real JAX-backed ``pymdp.agent.Agent``. If pymdp
    1.0.0 is not installed, construction raises ``ImportError``.
    """

    _DEFAULT_POLICY_LEN: int = 1  # pymdp 1.0.0 default; tunable via gnn_config

    def __init__(
        self,
        gnn_config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        allow_demo_spec: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        gnn_config
            Parsed GNN POMDP configuration. Recognised keys:

            * ``states`` / ``actions`` / ``observations`` — either a list
              of names or an integer count
            * ``model_name`` — display name
            * ``parameters`` — ``num_timesteps``, ``learning_rate``,
              ``gamma``, ``alpha``, ``policy_len``, ``preferences``,
              ``prior_beliefs``, ``transition_structure``, ``random_seed``
            * ``initialparameterization`` — required GNN-style A/B/C/D/E
              matrices

        output_dir
            Directory for any future ``_save_results`` call.
        allow_demo_spec
            Enables the named demo constructor path. Public pipeline execution
            should pass explicit matrices instead.
        """
        self.gnn_config = gnn_config or {}
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.allow_demo_spec = allow_demo_spec
        self.logger = logging.getLogger(__name__)

        # Validate pymdp 1.0.0 up-front (fails fast with a clear message).
        _require_pymdp_1()

        self._initialize_parameters()

        # Model fields filled by create_pymdp_model(...)
        self.agent: Any = None
        self.A: List[np.ndarray] = []
        self.B: List[np.ndarray] = []
        self.C: List[np.ndarray] = []
        self.D: List[np.ndarray] = []
        self.E: Optional[np.ndarray] = None
        self.A_np: np.ndarray = np.zeros((0, 0), dtype=np.float64)
        self.B_np: np.ndarray = np.zeros((0, 0, 0), dtype=np.float64)
        self.C_np: np.ndarray = np.zeros(0, dtype=np.float64)
        self.D_np: np.ndarray = np.zeros(0, dtype=np.float64)
        self.model_matrices: Dict[str, np.ndarray] = {}
        self.simulation_trace: List[Dict[str, Any]] = []
        self.results: Dict[str, Any] = {}

        init_params = self.gnn_config.get(
            "initialparameterization"
        ) or self.gnn_config.get("initial_parameterization")
        if init_params:
            self._build_model_from_initial_parameterization(init_params)
        elif self.allow_demo_spec:
            self._build_model_from_demo_spec()
        else:
            raise ValueError(
                "PyMDPSimulation requires explicit initialparameterization matrices; "
                "use create_demo_pymdp_simulation() for the demo model."
            )

    # ------------------------------------------------------------------
    # Parameter extraction
    # ------------------------------------------------------------------
    def _initialize_parameters(self) -> None:
        """Handle initialize parameters for internal callers."""
        cfg = self.gnn_config

        def _resolve(name: str, default_prefix: str, default_count: int) -> List[str]:
            """Handle resolve for internal callers."""
            raw = cfg.get(name, [])
            if isinstance(raw, int):
                return [f"{default_prefix}_{i}" for i in range(raw)]
            if isinstance(raw, list) and raw:
                return list(raw)
            return [f"{default_prefix}_{i}" for i in range(default_count)]

        if cfg:
            self.states = _resolve("states", "state", 4)
            self.actions = _resolve("actions", "action", 5)
            self.observations = _resolve("observations", "obs", 4)
            self.model_name = cfg.get("model_name", "GNN_POMDP")
        else:
            self.states = [f"location_{i}" for i in range(4)]
            self.actions = ["move_up", "move_down", "move_left", "move_right", "stay"]
            self.observations = [f"obs_location_{i}" for i in range(4)]
            self.model_name = "Demo_Gridworld"

        params = cfg.get("parameters", {}) or {}
        self.num_timesteps = int(params.get("num_timesteps", 20))
        self.learning_rate = float(params.get("learning_rate", 0.5))
        self.alpha = float(params.get("alpha", 16.0))
        self.gamma = float(params.get("gamma", 16.0))
        self.policy_len = int(params.get("policy_len", self._DEFAULT_POLICY_LEN))
        self.random_seed = int(params.get("random_seed", 0))

        # Friendly aliases
        self.state_names = list(self.states)
        self.action_names = list(self.actions)
        self.observation_names = list(self.observations)

        self.num_states = len(self.state_names)
        self.num_actions = len(self.action_names)
        self.num_observations = len(self.observation_names)

        self.logger.info(
            "Initialized: %dS, %dA, %dO (model=%s)",
            self.num_states,
            self.num_actions,
            self.num_observations,
            self.model_name,
        )

    # ------------------------------------------------------------------
    # Matrix construction
    # ------------------------------------------------------------------
    def _demo_observation_model(self) -> np.ndarray:
        """Noisy identity likelihood, column-normalised."""
        No = self.num_observations
        Ns = self.num_states
        base = np.eye(No, Ns) * 0.9
        if No > 1:
            noise = 0.1 / (No - 1)
            for i in range(No):
                for j in range(Ns):
                    if i != j:
                        base[i, j] = noise
        return _canonicalise_A(base, (No, Ns))

    def _demo_transition_model(self) -> np.ndarray:
        """Gridworld or GNN-described transitions in (next, prev, action) shape."""
        Ns = self.num_states
        Na = max(self.num_actions, 1)
        B = np.zeros((Ns, Ns, Na), dtype=np.float64)

        transition_structure = self.gnn_config.get("transition_structure") or (
            self.gnn_config.get("parameters", {}) or {}
        ).get("transition_structure")

        if transition_structure:
            for action_idx, action_name in enumerate(self.actions):
                action_trans = transition_structure.get(action_name)
                if not action_trans:
                    continue
                for from_state, to_states in action_trans.items():
                    if from_state not in self.states:
                        continue
                    f_idx = self.states.index(from_state)
                    for to_state, prob in to_states.items():
                        if to_state not in self.states:
                            continue
                        t_idx = self.states.index(to_state)
                        B[t_idx, f_idx, action_idx] = float(prob)
        else:
            for action_idx, action_name in enumerate(self.actions):
                for s in range(Ns):
                    if action_name == "stay":
                        B[s, s, action_idx] = 1.0
                    elif action_name == "move_up":
                        B[max(0, s - 2), s, action_idx] = 1.0
                    elif action_name == "move_down":
                        B[min(Ns - 1, s + 2), s, action_idx] = 1.0
                    elif action_name == "move_left":
                        B[s - 1 if s % 2 == 1 else s, s, action_idx] = 1.0
                    elif action_name == "move_right":
                        B[s + 1 if s % 2 == 0 and s + 1 < Ns else s, s, action_idx] = (
                            1.0
                        )
                    else:
                        B[s, s, action_idx] = 1.0

        return _canonicalise_B(B, Ns, Na)

    def _demo_preference_model(self) -> np.ndarray:
        """Handle demo preference model for internal callers."""
        params = self.gnn_config.get("parameters", {}) or {}
        prefs = params.get("preferences")
        if isinstance(prefs, (list, tuple, np.ndarray)):
            return _canonicalise_C(prefs, self.num_observations)
        # Named dict: prefer the 'goal_reward' → final observation
        C = np.zeros(self.num_observations, dtype=np.float64)
        if self.num_observations > 0:
            if isinstance(prefs, dict) and "goal_reward" in prefs:
                C[-1] = float(prefs["goal_reward"])
            else:
                C[-1] = 2.0
        return C

    def _demo_prior_beliefs(self) -> np.ndarray:
        """Handle demo prior beliefs for internal callers."""
        params = self.gnn_config.get("parameters", {}) or {}
        prior = params.get("prior_beliefs")
        if isinstance(prior, (list, tuple, np.ndarray)):
            return _canonicalise_D(prior, self.num_states)
        D = np.ones(self.num_states, dtype=np.float64) / max(self.num_states, 1)
        # slight mass on starting state
        if self.num_states > 1:
            D[0] = D[0] + 0.1
            D = _normalise_prob_vector(D)
        return D

    def _build_model_from_demo_spec(self) -> None:
        """Build model from demo spec."""
        A_np = self._demo_observation_model()
        B_np = self._demo_transition_model()
        C_np = self._demo_preference_model()
        D_np = self._demo_prior_beliefs()
        E_np = None  # habit vector optional; learned later if requested

        self._install_matrices(A_np, B_np, C_np, D_np, E_np)
        self._instantiate_agent()

    def _build_model_from_initial_parameterization(
        self, init_params: Dict[str, Any]
    ) -> None:
        """Build model from initial parameterization."""
        fallback_shape = (self.num_observations, self.num_states)
        A_np = _canonicalise_A(init_params.get("A"), fallback_shape)

        # Update dimensions if GNN matrices disagree with the name counts
        if A_np.shape != (self.num_observations, self.num_states):
            self.num_observations, self.num_states = A_np.shape
            self.observations = [f"obs_{i}" for i in range(self.num_observations)]
            self.states = [f"state_{i}" for i in range(self.num_states)]
            self.observation_names = list(self.observations)
            self.state_names = list(self.states)

        B_np = _canonicalise_B(
            init_params.get("B"), self.num_states, max(self.num_actions, 1)
        )
        self.num_actions = int(B_np.shape[2])
        if len(self.actions) != self.num_actions:
            self.actions = [f"action_{i}" for i in range(self.num_actions)]
            self.action_names = list(self.actions)

        C_np = _canonicalise_C(init_params.get("C"), self.num_observations)
        D_np = _canonicalise_D(init_params.get("D"), self.num_states)
        E_np = _canonicalise_E(init_params.get("E"), expected_policies=self.num_actions)

        self._install_matrices(A_np, B_np, C_np, D_np, E_np)
        self._instantiate_agent()

    def _install_matrices(
        self,
        A_np: np.ndarray,
        B_np: np.ndarray,
        C_np: np.ndarray,
        D_np: np.ndarray,
        E_np: Optional[np.ndarray],
    ) -> None:
        """Handle install matrices for internal callers."""
        self.A_np = A_np
        self.B_np = B_np
        self.C_np = C_np
        self.D_np = D_np
        self.E = E_np
        self.A = [A_np]
        self.B = [B_np]
        self.C = [C_np]
        self.D = [D_np]
        self.model_matrices = {
            "A": A_np,
            "B": B_np,
            "C": C_np,
            "D": D_np,
            "E": E_np if E_np is not None else np.zeros(0),
        }

    def _instantiate_agent(self) -> None:
        """Handle instantiate agent for internal callers."""
        self.agent = _build_pymdp_agent(
            A_np=self.A_np,
            B_np=self.B_np,
            C_np=self.C_np,
            D_np=self.D_np,
            E_np=self.E,
            batch_size=1,
            policy_len=self.policy_len,
            gamma=self.gamma,
            alpha=self.alpha,
        )
        self.logger.info(
            "pymdp 1.0.0 Agent built: %dS, %dA, %dO, policy_len=%d",
            self.num_states,
            self.num_actions,
            self.num_observations,
            self.policy_len,
        )

    # ------------------------------------------------------------------
    # Model construction entry points
    # ------------------------------------------------------------------
    def create_pymdp_model(self) -> Tuple[Any, Dict[str, np.ndarray]]:
        """Re-build the model from explicit GNN matrices and return ``(agent, matrices)``."""
        init_params = self.gnn_config.get(
            "initialparameterization"
        ) or self.gnn_config.get("initial_parameterization")
        if not init_params:
            raise ValueError(
                "create_pymdp_model requires initialparameterization matrices"
            )
        self._build_model_from_initial_parameterization(init_params)
        return self.agent, self.model_matrices

    def create_pymdp_model_from_gnn(self) -> Tuple[Any, Dict[str, np.ndarray]]:
        """Re-build the model from ``self.gnn_config['initialparameterization']``."""
        init_params = self.gnn_config.get(
            "initialparameterization"
        ) or self.gnn_config.get("initial_parameterization")
        if init_params:
            self._build_model_from_initial_parameterization(init_params)
        else:
            raise ValueError(
                "create_pymdp_model_from_gnn requires initialparameterization matrices"
            )
        return self.agent, self.model_matrices

    def configure_from_gnn(self, gnn_spec: Dict[str, Any]) -> None:
        """
        Accept an upstream ``gnn_spec`` with ``initialparameterization`` keys
        and rebuild the agent.
        """
        init = gnn_spec.get("initialparameterization") or gnn_spec.get(
            "initial_parameterization"
        )
        if init:
            self._build_model_from_initial_parameterization(init)
        else:
            raise ValueError(
                "configure_from_gnn requires initialparameterization matrices"
            )

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    def _sample_observation(
        self,
        current_state: int,
        np_rng: np.random.Generator,
        stream_obs: Optional[int] = None,
    ) -> int:
        """Sample observation or consume from incoming observation stream seam."""
        if stream_obs is not None:
            return int(stream_obs)
        probs = _normalise_prob_vector(self.A_np[:, current_state])
        return int(np_rng.choice(self.num_observations, p=probs))

    def _run_simulation_step(
        self,
        t: int,
        current_state: int,
        empirical_prior: Any,
        np_rng: np.random.Generator,
        jax_key: Any,
        stream_obs: Optional[int] = None,
    ) -> Tuple[int, Any, Any, Dict[str, Any]]:
        """Run simulation step."""
        import jax.numpy as jnp
        import jax.random as jr

        obs_idx = self._sample_observation(current_state, np_rng, stream_obs=stream_obs)
        obs_jax: list[Any] = [jnp.array([obs_idx], dtype=jnp.int32)]

        qs, info = self.agent.infer_states(
            obs_jax,
            empirical_prior=empirical_prior,
            return_info=True,
        )
        q_pi, neg_efe = self.agent.infer_policies(qs)
        jax_key, subkey = jr.split(jax_key)
        action_keys = jr.split(subkey, 2)  # batch_size=1
        action = self.agent.sample_action(q_pi, rng_key=action_keys[1:])
        action_idx = int(np.asarray(action)[0, 0])

        belief_vec = np.asarray(qs[0][0, -1], dtype=np.float64).flatten()
        efe_vec = np.asarray(neg_efe[0], dtype=np.float64).flatten()
        try:
            vfe = float(np.asarray(info["vfe"]).mean())
        except Exception as e:  # noqa: BLE001
            vfe = None
            self.logger.warning(
                "VFE extraction failed at timestep %d; recorded as unavailable "
                "(null in trace, excluded from vfe_history): %s",
                t,
                e,
            )

        next_probs = _normalise_prob_vector(self.B_np[:, current_state, action_idx])
        next_state = int(np_rng.choice(self.num_states, p=next_probs))

        new_prior = self.agent.update_empirical_prior(action, qs)

        step_data: dict[str, Any] = {
            "timestep": t,
            "current_state": int(current_state),
            "observation": int(obs_idx),
            "action": int(action_idx),
            "next_state": int(next_state),
            "beliefs": belief_vec,
            "policy_probs": np.asarray(q_pi[0], dtype=np.float64).flatten(),
            "expected_free_energy": efe_vec,
            "variational_free_energy": vfe,
        }
        return next_state, new_prior, jax_key, step_data

    def run_simulation(
        self,
        output_dir: Optional[Path] = None,
        num_timesteps: Optional[int] = None,
        observation_stream: Optional[Iterable[int]] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """Run simulation with optional streaming observation ingestion."""
        if num_timesteps is not None:
            try:
                self.num_timesteps = int(num_timesteps)
            except (ValueError, TypeError) as e:
                self.logger.debug("num_timesteps cast failed, keeping default: %s", e)

        if self.agent is None:
            return {
                "success": False,
                "error": "No agent available — model construction failed",
            }

        import jax.random as jr

        start_time = time.time()
        self.logger.info("Starting pymdp 1.0.0 rollout: %s", self.model_name)

        np_rng = np.random.default_rng(self.random_seed)
        jax_key = jr.PRNGKey(self.random_seed)

        current_state = int(np_rng.choice(self.num_states, p=self.D_np))
        self.simulation_trace = []
        empirical_prior = self.agent.D

        stream_iter = (
            iter(observation_stream) if observation_stream is not None else None
        )

        try:
            for t in range(self.num_timesteps):
                stream_obs = (
                    next(stream_iter, None) if stream_iter is not None else None
                )
                current_state, empirical_prior, jax_key, step_data = (
                    self._run_simulation_step(
                        t,
                        current_state,
                        empirical_prior,
                        np_rng,
                        jax_key,
                        stream_obs=stream_obs,
                    )
                )
                self.simulation_trace.append(step_data)

                if t % 5 == 0:
                    self.logger.info(
                        "t=%d state=%d obs=%d action=%d",
                        t,
                        step_data["current_state"],
                        step_data["observation"],
                        step_data["action"],
                    )
        except Exception as e:  # noqa: BLE001
            self.logger.exception("Simulation error at timestep %d: %s", t, e)
            return {"success": False, "error": str(e), "traceback_at": t}

        duration = time.time() - start_time
        performance = self._analyze_results(duration)

        observations = [int(step["observation"]) for step in self.simulation_trace]
        actions = [int(step["action"]) for step in self.simulation_trace]
        beliefs = [
            np.asarray(step["beliefs"], dtype=np.float64).flatten().tolist()
            for step in self.simulation_trace
        ]
        true_states = (
            [int(self.simulation_trace[0]["current_state"])]
            + [int(step["next_state"]) for step in self.simulation_trace]
            if self.simulation_trace
            else []
        )
        efe_history = [
            np.asarray(step["expected_free_energy"], dtype=np.float64)
            .flatten()
            .tolist()
            for step in self.simulation_trace
        ]
        vfe_history = [
            float(step["variational_free_energy"])
            for step in self.simulation_trace
            if step["variational_free_energy"] is not None
        ]
        vfe_unavailable = [
            int(step["timestep"])
            for step in self.simulation_trace
            if step["variational_free_energy"] is None
        ]
        policy_posterior = [
            np.asarray(step["policy_probs"], dtype=np.float64).flatten().tolist()
            for step in self.simulation_trace
        ]

        # Hardware / accelerator metadata
        accelerator_type = "cpu"
        try:
            if shutil.which("nvidia-smi") is not None:
                accelerator_type = "cuda"
            elif sys.platform == "darwin":
                accelerator_type = "mps"
        except Exception as e:
            self.logger.debug(
                "Accelerator detection failed; falling back to cpu: %s", e
            )
            accelerator_type = "cpu"

        results_out: Dict[str, Any] = {
            "schema_version": "pymdp_simulation_v1",
            "framework": "PyMDP",
            "model_name": self.model_name,
            "vfe_unavailable_timesteps": vfe_unavailable,
            "execution_metadata": {
                "accelerator_type": accelerator_type,
                "vectorization": "jax" if "jax" in sys.modules else "numpy",
                "duration_seconds": float(duration),
            },
            "observations": observations,
            "actions": actions,
            "beliefs": beliefs,
            "true_states": true_states,
            "observations_by_modality": {"joint_observation": observations},
            "actions_by_control_factor": {"joint_action": actions},
            "beliefs_by_factor": {"joint_state": beliefs},
            "hidden_states_by_factor": {"joint_state": true_states},
            "expected_free_energy": efe_history,
            "expected_free_energy_convention": EFE_CONVENTION_PYMDP,
            "expected_free_energy_convention_summary": EFE_CONVENTION_PYMDP_SUMMARY,
            "variational_free_energy": vfe_history,
            "policy_posterior": policy_posterior,
            "simulation_trace": {
                "observations": observations,
                "actions": actions,
                "beliefs": beliefs,
                "true_states": true_states,
                "efe_history": efe_history,
                "vfe_history": vfe_history,
                "policy_posterior": policy_posterior,
                "belief_confidence": [float(max(b)) if b else 0.0 for b in beliefs],
            },
            "model_parameters": {
                "A_shape": list(self.A_np.shape),
                "B_shape": list(self.B_np.shape),
                "C_shape": list(self.C_np.shape),
                "D_shape": list(self.D_np.shape),
                "num_states": int(self.num_states),
                "num_observations": int(self.num_observations),
                "num_actions": int(self.num_actions),
                "policy_len": int(self.policy_len),
                "gamma": float(self.gamma),
                "alpha": float(self.alpha),
            },
            "metrics": {
                "expected_free_energy": efe_history,
                "variational_free_energy": vfe_history,
                "policy_posterior": policy_posterior,
                "belief_confidence": [float(max(b)) if b else 0.0 for b in beliefs],
                "cumulative_preference": [
                    float(self.C_np[obs]) for obs in observations
                ],
            },
            "performance": performance,
            "trace": self.simulation_trace,
            "success": True,
        }

        if output_dir is not None or self.output_dir is not None:
            results_dir = (
                Path(output_dir) if output_dir is not None else self.output_dir
            )
            if results_dir is not None:
                self._save_results(results_dir)

        self.logger.info("Simulation completed in %s", format_duration(duration))
        self.results = results_out
        return results_out

    def _analyze_results(self, duration: float) -> Dict[str, Any]:
        """Analyze results."""
        if not self.simulation_trace:
            return {}

        total = len(self.simulation_trace)
        final_state = self.simulation_trace[-1]["next_state"]
        visited = {step["current_state"] for step in self.simulation_trace}

        belief_entropies: List[float] = []
        action_counts = np.zeros(max(self.num_actions, 1), dtype=np.float64)
        for step in self.simulation_trace:
            beliefs = np.asarray(step["beliefs"], dtype=np.float64)
            entropy = float(-np.sum(beliefs * np.log(beliefs + 1e-16)))
            belief_entropies.append(entropy)
            a = int(step["action"])
            if 0 <= a < self.num_actions:
                action_counts[a] += 1

        return {
            "model_name": self.model_name,
            "total_timesteps": total,
            "duration_seconds": duration,
            "final_state": final_state,
            "states_visited": len(visited),
            "unique_states_ratio": len(visited) / max(self.num_states, 1),
            "mean_belief_entropy": float(np.mean(belief_entropies)),
            "action_distribution": (action_counts / max(total, 1)).tolist(),
            "most_used_action": int(np.argmax(action_counts))
            if action_counts.any()
            else 0,
            "exploration_efficiency": len(visited) / max(total, 1),
            "gnn_config_used": bool(self.gnn_config),
            "configuration": {
                "num_states": self.num_states,
                "num_actions": self.num_actions,
                "num_observations": self.num_observations,
                "learning_rate": self.learning_rate,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "policy_len": self.policy_len,
            },
        }

    def _save_results(self, output_dir: Path) -> None:
        """Save results."""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            results_file = output_dir / f"pymdp_results_{self.model_name}.json"
            safe_json_dump(self.results, results_file)

            trace_file = output_dir / f"pymdp_trace_{self.model_name}.json"
            cleaned_trace = [
                convert_numpy_for_json(step) for step in self.simulation_trace
            ]
            safe_json_dump(cleaned_trace, trace_file)

            matrices_file = output_dir / f"pymdp_matrices_{self.model_name}.pkl"
            safe_pickle_dump(self.model_matrices, matrices_file)

            self.logger.info(
                "Results saved to %s (visualisation by analysis step)", output_dir
            )
        except Exception as e:  # noqa: BLE001
            self.logger.error("Error saving results: %s", e)

    def get_summary(self) -> Dict[str, Any]:
        """Return summary."""
        if not self.results:
            return {"status": "no_results"}
        perf = self.results.get("performance", {})
        return {
            "model_name": perf.get("model_name", self.model_name),
            "timesteps": perf.get("total_timesteps", 0),
            "duration": perf.get("duration_seconds", 0.0),
            "final_state": perf.get("final_state", -1),
            "states_explored": perf.get("states_visited", 0),
            "gnn_configured": perf.get("gnn_config_used", bool(self.gnn_config)),
            "pymdp_version_ge_1_0_0": True,
            "success": True,
        }


def create_pymdp_simulation_from_gnn(gnn_config: Dict[str, Any]) -> PyMDPSimulation:
    """Create a PyMDP simulation from parsed GNN configuration."""
    return PyMDPSimulation(gnn_config=gnn_config)


def create_demo_pymdp_simulation(
    output_dir: Optional[Union[str, Path]] = None,
) -> PyMDPSimulation:
    """Create the named demo PyMDP simulation used by examples and tests."""
    return PyMDPSimulation(gnn_config={}, output_dir=output_dir, allow_demo_spec=True)


def run_pymdp_simulation_from_gnn(
    gnn_config: Dict[str, Any], output_dir: Path
) -> Dict[str, Any]:
    """End-to-end helper: build + run + persist."""
    sim = create_pymdp_simulation_from_gnn(gnn_config)
    return sim.run_simulation(output_dir=output_dir)

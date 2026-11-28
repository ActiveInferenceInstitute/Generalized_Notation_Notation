#!/usr/bin/env python3
"""
PyMDP 1.0.0 simulation for GNN POMDP specifications.

pymdp 1.0.0 (https://github.com/infer-actively/pymdp) is a JAX-first rewrite.
The Agent accepts batched ``list[jax.Array]`` models and the public loop is:

    qs, info       = agent.infer_states(obs, empirical_prior=prior, return_info=True)
    q_pi, neg_efe  = agent.infer_policies(qs)
    action         = agent.sample_action(q_pi, rng_key=keys)
    prior          = agent.update_empirical_prior(action, qs)

This module converts a GNN specification (whose matrices are plain nested
lists / numpy arrays) into the pymdp 1.0.0 batched list-of-arrays format,
runs a rollout, and writes a ``simulation_results.json`` file that the
analysis step (``src/analysis/pymdp/``) consumes.

Architecture note:
    This module is part of EXECUTE (step 12). It ONLY runs simulations and
    logs raw data. ALL visualisations belong to ANALYSIS (step 16).
    Flow: Render (prepare scripts) → Execute (run + log raw) → Analysis (plot).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from gnn.extract.pomdp_axes import declared_b_order
from gnn.utils.runtime_safety.resource_manager import get_current_memory_usage

logger = logging.getLogger(__name__)

EFE_CONVENTION_PYMDP = (
    "pymdp 1.0.0 compute_neg_efe_policy sign convention: the historical"
    " emitted expected_free_energy field stores neg_efe unchanged, with"
    " neg_efe = -EFE = expected utility + hidden-state information gain"
    " under the producer's enabled terms (utility and state information"
    " gain enabled, parameter information gain disabled). Expected utility"
    " is the linear payoff sum sum_o q(o) C[o] over predicted observations"
    " — not a KL against C. The policy posterior is proportional to"
    " E(pi) exp(gamma * neg_efe). These reported neg_efe values are not"
    " directly comparable to risk+ambiguity EFE without sign and convention"
    " mapping. No energy unit is declared by this receipt."
)
EFE_CONVENTION_PYMDP_SUMMARY = (
    "Reported neg_efe = -EFE = expected utility + hidden-state information"
    " gain (producer-enabled terms)."
)


# ---------------------------------------------------------------------------
# GNN matrix normalisation helpers (pure numpy; framework-agnostic)
# ---------------------------------------------------------------------------


def _normalise_prob_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a computed distribution without fabricating missing mass."""
    arr = np.asarray(v, dtype=np.float64).flatten()
    total = float(arr.sum())
    if not np.isfinite(total) or total <= 0 or np.any(arr < 0):
        raise ValueError("PyMDP computed an invalid probability distribution")
    return arr / total


def _canonicalise_B(
    B_data: Any,
    num_states: int,
    num_actions: int | None,
    b_tensor_order: str = "",
    b_provenance: Any = None,
) -> np.ndarray:
    """Project source transitions to the same probability axes as Step 11."""
    from gnn.render.pomdp_contract import canonicalise_b_matrix

    if B_data is None:
        raise ValueError("B matrix is required for PyMDP execution")
    canonical, _ = canonicalise_b_matrix(
        B_data,
        num_states=num_states,
        num_actions=num_actions,
        model_parameters={"b_tensor_order": b_tensor_order},
        b_provenance=b_provenance,
        strict_types=True,
    )
    return np.asarray(canonical, dtype=np.float64)


def _build_nonstationary_b_steps(
    b_tensor: np.ndarray,
    *,
    key: str,
    schedule: list[int] | None,
    num_timesteps: int,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Resolve per-step transition tensors from a nonstationary declaration.

    - ``B_t`` (time-indexed): step ``t`` uses slice ``min(t, T-1)``; beyond
      the declared span the last phase is held (documented hold-last
      semantics, recorded as ``schedule_truncated``).
    - ``B_regime`` (regime-switched): step ``t`` uses the regime named by
      ``b_regime_schedule[min(t, len(schedule) - 1)]``; the schedule
      likewise holds its last entry beyond the declared span.
    """
    horizon = max(1, num_timesteps)
    if schedule is not None:
        steps = [b_tensor[schedule[min(t, len(schedule) - 1)]] for t in range(horizon)]
        meta = {
            "kind": "regime_switched",
            "transition_key": key,
            "declared_span": int(b_tensor.shape[0]),
            "horizon": horizon,
            "schedule": schedule,
            "schedule_truncated": horizon > len(schedule),
        }
        return steps, meta
    steps = [b_tensor[min(t, b_tensor.shape[0] - 1)] for t in range(horizon)]
    meta = {
        "kind": "time_varying",
        "transition_key": key,
        "declared_span": int(b_tensor.shape[0]),
        "horizon": horizon,
        "schedule": None,
        "schedule_truncated": horizon > b_tensor.shape[0],
    }
    return steps, meta


def _rebuild_nonstationary_agent(
    *,
    A_np: np.ndarray,
    B_step: np.ndarray,
    C_np: np.ndarray,
    empirical_prior: Any,
    E_np: Optional[np.ndarray],
    batch_size: int,
    policy_len: int,
    gamma: float,
    alpha: float,
) -> Any:
    """Rebuild the pymdp Agent for one scheduled transition tensor.

    Per-step rebuild semantics: a nonstationary rollout constructs a fresh
    Agent each timestep with the scheduled B so pymdp's internal state is
    always consistent with the active transition. The running empirical
    prior (the previous ``update_empirical_prior`` output) carries over as
    D; batch rows share the prior, matching the broadcast construction of
    the static route.
    """
    prior_np = np.asarray(empirical_prior, dtype=np.float64)
    if prior_np.ndim > 1:
        prior_np = prior_np[0]
    D_np = _normalise_prob_vector(prior_np.reshape(-1))
    return _build_pymdp_agent(
        A_np=A_np,
        B_np=B_step,
        C_np=C_np,
        D_np=D_np,
        E_np=E_np,
        batch_size=batch_size,
        policy_len=policy_len,
        gamma=gamma,
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# pymdp 1.0.0 (JAX-first) import + Agent construction
# ---------------------------------------------------------------------------


def _require_pymdp_1() -> Any:
    """
    Import pymdp 1.0.0 (JAX-first). We probe for the new surface explicitly so
    we fail fast with an actionable error if an old 0.x wheel is installed.
    """
    try:
        import jax.numpy as jnp  # noqa: F401
        import jax.random as jr  # noqa: F401
        from pymdp import utils as pymdp_utils  # noqa: F401
        from pymdp.agent import Agent  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "pymdp 1.0.0 (JAX-first) is required. Install with:\n"
            "    uv pip install 'inferactively-pymdp>=1.0.5'\n"
            f"(original error: {e})"
        ) from e

    # pymdp 1.0.0 has Agent.update_empirical_prior; 0.x does not.
    if not hasattr(Agent, "update_empirical_prior"):
        raise ImportError(
            "Detected unsupported pymdp (<1.0.0). This module requires pymdp 1.0.0. "
            "Upgrade with: uv pip install --upgrade 'inferactively-pymdp>=1.0.5'"
        )
    return Agent, pymdp_utils, jnp, jr


def _to_jax_batched(mat_np: np.ndarray, batch_size: int) -> Any:
    """Add a leading batch dim and convert to jnp float32, pymdp 1.0.0 convention."""
    import jax.numpy as jnp

    arr = jnp.asarray(mat_np, dtype=jnp.float32)
    if batch_size == 1:
        return arr[None, ...]
    return jnp.broadcast_to(arr[None, ...], (batch_size, *arr.shape))


def _build_pymdp_agent(
    *,
    A_np: np.ndarray,
    B_np: np.ndarray,
    C_np: np.ndarray,
    D_np: np.ndarray,
    E_np: Optional[np.ndarray],
    batch_size: int = 1,
    policy_len: int = 1,
    gamma: float = 16.0,
    alpha: float = 16.0,
) -> Any:
    """
    Build a pymdp 1.0.0 ``Agent`` from canonical GNN numpy matrices.

    Parameters
    ----------
    A_np : (num_obs, num_states) column-normalised
    B_np : (num_states, num_states, num_actions) column-normalised per action
    C_np : (num_obs,)
    D_np : (num_states,)
    E_np : (num_policies,) or None
    """
    Agent, _, _, _ = _require_pymdp_1()

    num_actions = int(B_np.shape[-1])

    A_list: list[Any] = [_to_jax_batched(A_np, batch_size)]
    B_list: list[Any] = [_to_jax_batched(B_np, batch_size)]
    C_list: list[Any] = [_to_jax_batched(C_np, batch_size)]
    D_list: list[Any] = [_to_jax_batched(D_np, batch_size)]

    agent_kwargs: Dict[str, Any] = dict(
        A=A_list,
        B=B_list,
        C=C_list,
        D=D_list,
        num_controls=[num_actions],
        policy_len=policy_len,
        gamma=gamma,
        alpha=alpha,
        batch_size=batch_size,
    )

    # pymdp 1.0.0 asserts ``num_controls[fi] > 1`` for every factor listed in
    # ``control_fac_idx``. A pure HMM (num_actions == 1) must therefore omit
    # control_fac_idx entirely; pymdp defaults to an empty / passive control set.
    if num_actions > 1:
        agent_kwargs["control_fac_idx"] = [0]

    if E_np is not None:
        # E is a plain Array of shape (batch, num_policies) — not a list.
        agent_kwargs["E"] = _to_jax_batched(E_np, batch_size)

    agent = Agent(**agent_kwargs)
    return agent


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def pymdp_kind_refusal(gnn_spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Classify a GNN spec against PyMDP's categorical-only backend contract.

    PyMDP renders only discrete POMDPs with categorical A/B/C/D[/E]
    matrices. Specs that declare the linear-Gaussian (F/H/Q/R) family —
    alone or composed with other families — must be refused here with an
    explicit structured receipt instead of failing later inside the matrix
    normalisation path (or silently after a partial rollout).

    Detection mirrors the render-side doctrine in
    ``gnn.render.pomdp_contract``: kind sets are composed (every declared
    family is reported), and the refusal reason for a composed spec comes
    from ``unsupported_composition_reason`` so receipts share the single
    stable ``unsupported-composition:`` grep anchor.

    Parameters
    ----------
    gnn_spec : dict
        Either an in-memory spec (``initialparameterization`` /
        ``initial_parameterization`` present) or a lightweight parse
        receipt carrying ``file_path`` for late extraction.

    Returns
    -------
    dict or None
        ``None`` when the spec is not refusable (discrete/other
        singletons, or not classifiable at all — the engine's existing
        matrix checks fail loud downstream). Otherwise a receipt:

        - Continuous family composed with others →
          ``{"success": False, "unsupported": True, "status": "unsupported",
          "reason": unsupported_composition_reason(kinds),
          "model_kinds": sorted kind names}``.
        - Singleton continuous → same shape with a
          ``continuous-spec:`` reason.
        - Extraction failure on a ``file_path`` receipt →
          ``{"success": False, "unsupported": False, "status": "failed",
          "reason": "pymdp-kind-gate: extraction failed (<codes>)"}``.

    Notes
    -----
    Imports are lazy to keep module import light. Only the
    extraction-failure path yields a structured ``failed`` receipt;
    unexpected detection errors (e.g. ``ValueError`` from
    ``detect_model_kinds`` on a malformed mapping) propagate, matching
    the render side's fail-loud doctrine.
    """
    from gnn.render.pomdp_contract import (
        ModelKind,
        detect_model_kinds,
        detect_pomdp_space_model_kinds,
        pymdp_nonstationary_transition,
        unsupported_composition_reason,
    )

    initial = gnn_spec.get("initialparameterization") or gnn_spec.get(
        "initial_parameterization"
    )
    if isinstance(initial, dict) and initial:
        kinds = detect_model_kinds(gnn_spec)
        declared_keys = [
            str(key)
            for key in {
                **initial,
                **((gnn_spec.get("structured_pomdp") or {}).get("matrices") or {}),
            }
        ]
    else:
        file_path = gnn_spec.get("file_path")
        if not file_path:
            return None
        from gnn.extract.pomdp_extractor import extract_pomdp_from_file

        space, errors = extract_pomdp_from_file(
            file_path, strict_validation=False, on_error="collect"
        )
        if space is None:
            codes = "; ".join(f"{e.code}: {e.message}" for e in errors) or "unknown"
            return {
                "success": False,
                "unsupported": False,
                "status": "failed",
                "reason": f"pymdp-kind-gate: extraction failed ({codes})",
            }
        kinds = detect_pomdp_space_model_kinds(space)
        declared_keys = [
            str(key)
            for key in {
                **(getattr(space, "initial_parameterization", None) or {}),
                **(getattr(space, "matrices", None) or {}),
            }
        ]

    if ModelKind.CONTINUOUS in kinds and len(kinds) > 1:
        return {
            "success": False,
            "unsupported": True,
            "status": "unsupported",
            "reason": unsupported_composition_reason(kinds),
            "model_kinds": sorted(k.value for k in kinds),
        }
    if kinds == frozenset({ModelKind.CONTINUOUS}):
        return {
            "success": False,
            "unsupported": True,
            "status": "unsupported",
            "reason": (
                "continuous-spec: PyMDP requires categorical A/B/C/D[/E] "
                "matrices; continuous linear-Gaussian models render to a "
                "native continuous backend instead"
            ),
            "model_kinds": sorted(k.value for k in kinds),
        }
    if ModelKind.NONSTATIONARY in kinds:
        try:
            transition = pymdp_nonstationary_transition(declared_keys)
            if isinstance(initial, dict) and initial and transition not in initial:
                raise ValueError(
                    f"unsupported-nonstationary: {transition} is absent from executable parameters"
                )
        except ValueError as error:
            return {
                "success": False,
                "unsupported": True,
                "status": "unsupported",
                "reason": str(error),
                "model_kinds": sorted(k.value for k in kinds),
            }
    return None


def run_pymdp_simulation(
    gnn_spec: Dict[str, Any],
    output_dir: Path,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Run a discrete POMDP active inference rollout with real pymdp 1.0.0.

    Parameters
    ----------
    gnn_spec : dict
        Parsed GNN specification. Matrices live under
        ``gnn_spec["initialparameterization"]`` with keys A, B, C, D, E.
        Scalar runtime knobs live under ``gnn_spec["model_parameters"]``
        (``num_timesteps``, ``random_seed``, ``batch_size``, ``policy_len``,
        ``gamma``, ``alpha``).
    output_dir : Path
        Directory for ``simulation_results.json``. Created if missing.

    Returns
    -------
    (success, results) : (bool, dict)
        On success, ``results`` matches the schema consumed by
        ``src/analysis/pymdp/framework_extractors``:
        keys include ``observations``, ``actions``, ``beliefs``,
        ``true_states``, ``simulation_trace``, ``validation``, ``metrics``,
        ``model_parameters``, ``framework == "PyMDP"``.
        plus the in-process memory trio ``memory_usage_mb`` /
        ``peak_memory_mb`` / ``memory_delta_mb`` (self-RSS via psutil, same
        semantics as the pipeline step executor).
    """
    # Categorical-backend doctrine: PyMDP renders only discrete POMDPs, so
    # refuse continuous/composed specs here — before importing pymdp — with
    # an explicit receipt instead of a downstream matrix failure.
    gate = pymdp_kind_refusal(gnn_spec)
    if gate is not None:
        return False, gate

    try:
        _, _, jnp, jr = _require_pymdp_1()
    except ImportError as e:
        logger.error(str(e))
        return False, {
            "success": False,
            "error": str(e),
            "suggestion": "Install with: uv pip install 'inferactively-pymdp>=1.0.5'",
        }

    try:
        import importlib.metadata as _ilm

        pymdp_version = _ilm.version("inferactively-pymdp")
    except Exception:  # pragma: no cover - metadata missing is non-fatal
        pymdp_version = "unknown"

    logger.info("Starting pymdp %s rollout (JAX backend)", pymdp_version)

    init_params = gnn_spec.get("initialparameterization") or gnn_spec.get(
        "initial_parameterization", {}
    )
    model_params = gnn_spec.get("model_parameters", {}) or {}
    from gnn.render.pomdp_contract import (
        ModelKind,
        canonicalise_pymdp_likelihood_prior,
        canonicalise_pymdp_phase_transitions,
        canonicalise_pymdp_policy_prior,
        detect_model_kinds,
        pymdp_nonstationary_transition,
        pymdp_policy_horizon,
    )

    required_matrices = ["A", "B", "C", "D"]
    nonstationary_key = (
        pymdp_nonstationary_transition(init_params)
        if ModelKind.NONSTATIONARY in detect_model_kinds(gnn_spec)
        else None
    )
    missing_matrices = [
        name
        for name in required_matrices
        if name not in init_params
        and not (name == "B" and nonstationary_key is not None)
    ]
    if missing_matrices:
        return False, {
            "success": False,
            "error": f"Missing required PyMDP matrices: {missing_matrices}",
            "schema_version": "pymdp_simulation_v1",
        }

    A_np, C_np, D_np = canonicalise_pymdp_likelihood_prior(init_params, model_params)
    num_obs, num_states = A_np.shape

    b_raw = init_params.get("B")
    provided_actions = model_params.get("num_actions")
    num_actions = int(provided_actions) if provided_actions is not None else None
    provenance = gnn_spec.get("matrix_provenance") or {}

    b_steps: Optional[List[np.ndarray]] = None
    nonstationary_meta: Optional[Dict[str, Any]] = None
    num_timesteps = int(model_params.get("num_timesteps", 20))
    if nonstationary_key is not None:
        b_tensor, schedule = canonicalise_pymdp_phase_transitions(
            init_params[nonstationary_key],
            key=nonstationary_key,
            num_states=num_states,
            num_actions=num_actions,
            parameters=model_params,
            provenance=provenance.get(nonstationary_key),
        )
        b_steps, nonstationary_meta = _build_nonstationary_b_steps(
            b_tensor,
            key=nonstationary_key,
            schedule=schedule,
            num_timesteps=num_timesteps,
        )
        # The scheduled slices are already in canonical stored order.
        B_np = b_steps[0]
    else:
        B_np = _canonicalise_B(
            b_raw,
            num_states,
            num_actions,
            declared_b_order(model_params),
            b_provenance=provenance.get("B"),
        )
    # Ensure num_actions reflects the canonicalised tensor.
    num_actions = int(B_np.shape[2])
    policy_len = pymdp_policy_horizon(model_params)
    E_np = canonicalise_pymdp_policy_prior(
        init_params.get("E"), num_actions=num_actions, policy_len=policy_len
    )
    batch_size = int(model_params.get("batch_size", 1))
    gamma = float(model_params.get("gamma", 16.0))
    alpha = float(model_params.get("alpha", 16.0))
    seed = int(model_params.get("random_seed", 0))

    logger.info(
        "Dimensions: No=%d, Ns=%d, Nu=%d | T=%d, batch=%d, policy_len=%d",
        num_obs,
        num_states,
        num_actions,
        num_timesteps,
        batch_size,
        policy_len,
    )

    agent = _build_pymdp_agent(
        A_np=A_np,
        B_np=B_np,
        C_np=C_np,
        D_np=D_np,
        E_np=E_np,
        batch_size=batch_size,
        policy_len=policy_len,
        gamma=gamma,
        alpha=alpha,
    )
    logger.info("pymdp 1.0.0 Agent built (batch_size=%d)", batch_size)

    # RNG keys: numpy for the "environment" draws, jax for pymdp sampling.
    np_rng = np.random.default_rng(seed)
    jax_key = jr.PRNGKey(seed)

    # Initial true state
    true_state = int(np_rng.choice(num_states, p=D_np))
    true_states: List[int] = [true_state]
    observations: List[int] = []
    actions: List[int] = []
    beliefs: List[List[float]] = []
    efe_history: List[List[float]] = []
    vfe_history: List[Optional[float]] = []
    vfe_unavailable: List[int] = []
    policy_posterior_history: List[List[float]] = []

    empirical_prior = agent.D

    # In-process resource measurement (pipeline step_executor naming trio):
    # self-RSS sampled before/after the rollout; psutil, best-effort.
    rollout_start_memory_mb = get_current_memory_usage()

    for t in range(num_timesteps):
        if b_steps is not None:
            # Per-step rebuild semantics: a fresh pymdp Agent is constructed
            # for the scheduled transition tensor so pymdp's internal state
            # is always consistent with the active B; the running empirical
            # prior carries over as D (batch rows share the prior, matching
            # the broadcast construction used for the static route).
            step_b = b_steps[min(t, len(b_steps) - 1)]
            agent = _rebuild_nonstationary_agent(
                A_np=A_np,
                B_step=step_b,
                C_np=C_np,
                empirical_prior=empirical_prior,
                E_np=E_np,
                batch_size=batch_size,
                policy_len=policy_len,
                gamma=gamma,
                alpha=alpha,
            )
            B_np = step_b
        # Environment: sample observation from A given true state
        obs_probs = A_np[:, true_state]
        obs_idx = int(np_rng.choice(num_obs, p=_normalise_prob_vector(obs_probs)))
        observations.append(obs_idx)

        obs_jax: list[Any] = [jnp.array([obs_idx], dtype=jnp.int32)]

        # Agent inference
        qs, info = agent.infer_states(
            obs_jax,
            empirical_prior=empirical_prior,
            return_info=True,
        )

        # qs[f] shape: (batch, time, num_states[f]) — take most recent
        belief_vec = np.asarray(qs[0][0, -1], dtype=np.float64).flatten()
        beliefs.append(belief_vec.tolist())

        try:
            vfe_history.append(float(np.asarray(info["vfe"]).mean()))
        except Exception as e:  # noqa: BLE001 - informational only
            vfe_history.append(None)
            vfe_unavailable.append(t)
            logger.warning(
                "VFE extraction failed at timestep %d; recorded as unavailable: %s",
                t,
                e,
            )

        q_pi, neg_efe = agent.infer_policies(qs)
        # q_pi / neg_efe shape: (batch, num_policies)
        policy_posterior_history.append(
            np.asarray(q_pi[0], dtype=np.float64).flatten().tolist()
        )
        efe_history.append(np.asarray(neg_efe[0], dtype=np.float64).flatten().tolist())

        jax_key, subkey = jr.split(jax_key)
        action_keys = jr.split(subkey, batch_size + 1)
        action = agent.sample_action(q_pi, rng_key=action_keys[1:])
        # action shape: (batch, num_factors) — single control factor
        action_idx = int(np.asarray(action)[0, 0])
        actions.append(action_idx)

        # Environment: sample next true state from B
        next_probs = _normalise_prob_vector(B_np[:, true_state, action_idx])
        true_state = int(np_rng.choice(num_states, p=next_probs))
        true_states.append(true_state)

        # Update empirical prior for next step (pymdp 1.0.0 canonical helper)
        empirical_prior = agent.update_empirical_prior(action, qs)

        logger.info(
            "t=%02d obs=%d belief=%s action=%d next_state=%d",
            t,
            obs_idx,
            np.round(belief_vec, 3).tolist(),
            action_idx,
            true_state,
        )

    rollout_end_memory_mb = get_current_memory_usage()
    # ---------------------------------------------------------------------
    # Assemble results (pymdp_simulation_v1 schema consumed by analysis).
    # ---------------------------------------------------------------------
    model_name = gnn_spec.get("model_name") or gnn_spec.get("name") or "pymdp_model"
    agent_type = type(agent)
    likelihood = agent.A[0]
    likelihood_devices = likelihood.devices()

    results: Dict[str, Any] = {
        "schema_version": "pymdp_simulation_v1",
        "success": True,
        "framework": "PyMDP",
        "pymdp_version": pymdp_version,
        "vfe_unavailable_timesteps": vfe_unavailable,
        "backend": "jax",
        "model_name": model_name,
        "num_timesteps": num_timesteps,
        "observations_by_modality": {"joint_observation": observations},
        "hidden_states_by_factor": {"joint_state": true_states},
        "actions_by_control_factor": {"joint_action": actions},
        "beliefs_by_factor": {"joint_state": beliefs},
        "expected_free_energy": efe_history,
        "expected_free_energy_convention": EFE_CONVENTION_PYMDP,
        "expected_free_energy_convention_summary": EFE_CONVENTION_PYMDP_SUMMARY,
        "variational_free_energy": vfe_history,
        "policy_posterior": policy_posterior_history,
        "simulation_trace": {
            "observations": observations,
            "true_states": true_states,
            "beliefs": beliefs,
            "actions": actions,
            "efe_history": efe_history,
            "vfe_history": vfe_history,
            "policy_posterior": policy_posterior_history,
            "belief_confidence": [float(max(b)) if b else 0.0 for b in beliefs],
        },
        "observations": observations,
        "true_states": true_states,
        "beliefs": beliefs,
        "actions": actions,
        "model_parameters": {
            "A_shape": list(A_np.shape),
            "B_shape": list(B_np.shape),
            "C_shape": list(C_np.shape),
            "D_shape": list(D_np.shape),
            "num_states": int(num_states),
            "num_observations": int(num_obs),
            "num_actions": int(num_actions),
            "batch_size": batch_size,
            "policy_len": policy_len,
            "gamma": gamma,
            "alpha": alpha,
        },
        "matrix_provenance": gnn_spec.get("matrix_provenance", {}),
        "memory_usage_mb": rollout_end_memory_mb,
        "peak_memory_mb": max(rollout_start_memory_mb, rollout_end_memory_mb),
        "memory_delta_mb": rollout_end_memory_mb - rollout_start_memory_mb,
        "runtime_metadata": {
            "output_dir": str(output_dir),
            "random_seed": seed,
            "schema_version": "pymdp_simulation_v1",
            "native_agent": {
                "schema_version": "pymdp_native_agent_v1",
                "agent": f"{agent_type.__module__}.{agent_type.__qualname__}",
                "dtype": str(likelihood.dtype),
                "devices": sorted(str(device) for device in likelihood_devices),
                "platforms": sorted({device.platform for device in likelihood_devices}),
                "batch_size": int(likelihood.shape[0]),
                "policy_horizon": int(agent.policy_len),
            },
        },
        "metrics": {
            "expected_free_energy": efe_history,
            "variational_free_energy": vfe_history,
            "policy_posterior": policy_posterior_history,
            "belief_confidence": [float(max(b)) if b else 0.0 for b in beliefs],
            "cumulative_preference": [float(C_np[obs]) for obs in observations],
        },
        "validation": {
            "all_beliefs_valid": all(0.0 <= v <= 1.0 for b in beliefs for v in b),
            "beliefs_sum_to_one": all(abs(sum(b) - 1.0) < 1e-2 for b in beliefs if b),
            "actions_in_range": all(0 <= a < num_actions for a in actions),
            "pymdp_version_ge_1_0_0": _is_version_ge(pymdp_version, (1, 0, 0)),
        },
    }
    results["validation"]["all_valid"] = (
        results["validation"]["all_beliefs_valid"]
        and results["validation"]["beliefs_sum_to_one"]
        and results["validation"]["actions_in_range"]
        and results["validation"]["pymdp_version_ge_1_0_0"]
    )

    if nonstationary_meta is not None:
        distinct_transitions = {
            tuple(np.round(step, 9).ravel().tolist()) for step in b_steps or []
        }
        results["nonstationary"] = {
            **nonstationary_meta,
            "distinct_transitions": len(distinct_transitions),
        }
        results["validation"]["nonstationary_schedule_applied"] = True

    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "simulation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Saved simulation results to %s", results_file)
    return True, results


def _is_version_ge(ver: str, target: Sequence[int]) -> bool:
    """Return whether version ge."""
    try:
        parts = tuple(int(p) for p in ver.split(".")[: len(target)] if p.isdigit())
    except Exception:  # noqa: BLE001
        return False
    return parts >= tuple(target)
